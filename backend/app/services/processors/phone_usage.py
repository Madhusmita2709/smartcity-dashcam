from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import json

import cv2
from ultralytics import YOLO
from backend.app.core.config import get_settings
from backend.app.services.storage import upload_file, download_file
from backend.app.models.video import VideoRoute 
from backend.app.db.database import SessionLocal

settings = get_settings()


class PhoneUsageDetector:
    def __init__(self):
        print("[PHONE DETECTOR INIT]", flush=True)
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models and tracking parameters bound statefully per session pass
        self.base_model: Optional[YOLO] = None
        self.detect_model: Optional[YOLO] = None
        self.tracker_module: str = "bytetrack.yaml"
        
        # Stateful tracking caches running across the stream pass
        self.pending_uploads: List[dict] = []
        self.violation_records: List[dict] = []
        self.already_saved_violations: Set[int] = set()
        
        self.video_id: Optional[str] = None
        self.violation_dir: Optional[Path] = None
        self.device: str = "cpu"
        self.half_precision: bool = False

    def setup_session(
        self, 
        output_dir: Path, 
        video_id: str, 
        vehicle_model: str = "yolov8n.pt",
        phone_model: str = "cell_phone(best (9).pt",
        tracker_module: str = "bytetrack.yaml"
    ) -> None:
        """
        Loads configuration settings and triggers upfront custom model weights mapping once per run,
        pulling down missing asset files statefully from the system S3 model storage layer.
        """
        import torch
        torch.set_num_threads(1)
        models_bucket = getattr(settings, "minio_models_bucket", "models")
        self.video_id = video_id
        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear persistent memory caches securely between consecutive feeds
        self.pending_uploads.clear()
        self.violation_records.clear()
        self.already_saved_violations.clear()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half_precision = True if self.device == 'cuda' else False
        self.tracker_module = tracker_module

        # Format exact baseline check storage paths boundaries
        base_model_path = Path(vehicle_model) if Path(vehicle_model).is_absolute() else self.models_dir / vehicle_model
        custom_model_path = Path(phone_model) if Path(phone_model).is_absolute() else self.models_dir / phone_model

        # --- MINIO AUTOMATED WEIGHTS SYNCHRONIZATION ---
        try:
            if not base_model_path.exists():
                print(f"[PHONE] Worker node missing baseline track weights. Syncing {vehicle_model} from MinIO...", flush=True)
                download_file(models_bucket, vehicle_model, base_model_path)
            
            if not custom_model_path.exists():
                print(f"[PHONE] Worker node missing specific detector layers. Syncing {phone_model} from MinIO...", flush=True)
                download_file(models_bucket, phone_model, custom_model_path)
        except Exception as sync_err:
            print(f"⚠️ [PHONE WARNING] Cloud model synchronization step asset check failed: {sync_err}", flush=True)

        try:
            self.base_model = YOLO(str(base_model_path))
            self.detect_model = YOLO(str(custom_model_path))
        except Exception as e:
            raise RuntimeError(f"Unable to initialize PhoneUsageDetector weights layer mapping: {e}")

    def process_frame(self, context: Dict[str, Any]) -> cv2.Mat:
        """
        Accepts the global FrameContext payload on sampled intervals.
        Mutates context["frame"] canvas properties directly with evaluation tracks.
        """
        if self.base_model is None or self.detect_model is None or self.violation_dir is None or self.video_id is None:
            raise RuntimeError("PhoneUsageDetector workspace uninitialized. Call setup_session() first.")

        frame = context["frame"]
        timestamp = context["timestamp_seconds"]
        frame_index = context["frame_index"]
        gps = context.get("gps")
        
        height, width = frame.shape[:2]
        original_frame = frame.copy()
        any_violation_found = False

        # --- STEP 1: TRACK MOTORCYCLES (Class 3) ---
        bike_results = self.base_model.track(
            frame,
            classes=[3],  
            conf=0.35,        
            imgsz=640,      
            verbose=False,
            persist=True,    
            tracker=self.tracker_module
        )
        
        # --- STEP 2: DETECT ALL PEOPLE (Class 0) ---
        person_results = self.base_model(
            frame,
            classes=[0],
            conf=0.30,
            imgsz=640,
            verbose=False
        )
        
        current_frame_people = []
        if len(person_results) > 0 and len(person_results[0].boxes) > 0:
            current_frame_people = person_results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Process tracked motorcycles
        for result in bike_results:
            if len(result.boxes) == 0 or result.boxes.id is None:
                continue
                
            bike_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            bike_track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for b_box, associated_bike_id in zip(bike_boxes, bike_track_ids):
                bx1, by1, bx2, by2 = b_box
                bike_h = by2 - by1
                
                # Maximum intersection area selection to map the foreground rider perfectly
                rider_box = None
                max_overlap_area = -1
                
                for p_box in current_frame_people:
                    px1, py1, px2, py2 = p_box
                    
                    # Intersection calculations
                    ix1 = max(bx1, px1)
                    iy1 = max(by1 - int(bike_h * 0.45), py1)  # Extended window to securely capture head/ear zone
                    ix2 = min(bx2, px2)
                    iy2 = min(by2, py2)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        if overlap_area > max_overlap_area:
                            max_overlap_area = overlap_area
                            rider_box = p_box
                            
                if rider_box is None:
                    continue
                    
                # --- STEP 3: HIGH-RES CROP ISOLATION ---
                rx1, ry1, rx2, ry2 = rider_box
                pad_x1, pad_y1 = max(0, rx1 - 10), max(0, ry1 - 10)
                pad_x2, pad_y2 = min(width, rx2 + 10), min(height, ry2 + 10)
                
                rider_crop = original_frame[pad_y1:pad_y2, pad_x1:pad_x2]
                if rider_crop.size == 0:
                    continue
                    
                crop_h, crop_w = pad_y2 - pad_y1, pad_x2 - pad_x1
                
                # --- STEP 4: RESOLUTION FIX ON CUSTOM PHONE DETECTION ---
                phone_results = self.detect_model(rider_crop, conf=0.45, imgsz=640, verbose=False)[0]
                
                rider_violation = False
                max_phone_conf = 0.0
                valid_phone_center = None
                
                for p_box_det in phone_results.boxes:
                    if int(p_box_det.cls[0]) == 0:  # Class 0: Cell Phone
                        x1_loc, y1_loc, x2_loc, y2_loc = p_box_det.xyxy[0].tolist()
                        phone_h = y2_loc - y1_loc
                        
                        # CRITICAL SPATIAL GUARD: Foreground vs Background filter
                        if phone_h < (crop_h * 0.08) and crop_h > 250:
                            continue
                            
                        # Torso boundary check (Phone upper 70% me hi hona chahiye)
                        if y1_loc > (crop_h * 0.70):
                            continue
                            
                        cx = ((x1_loc + x2_loc) / 2) + pad_x1
                        cy = ((y1_loc + y2_loc) / 2) + pad_y1
                        
                        valid_phone_center = [cx, cy]
                        max_phone_conf = max(max_phone_conf, float(p_box_det.conf[0]))
                        rider_violation = True
                        break
                        
                # --- STEP 5: VISUAL OVERLAY ASSEMBLY ---
                if rider_violation:
                    any_violation_found = True
                    
                    label = f"ALERT: PHONE USAGE [BIKE ID {associated_bike_id}] [{int(max_phone_conf * 100)}%]"
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (rx1, ry1 - text_size[1] - 10), (rx1 + text_size[0], ry1), (0, 0, 255), -1)
                    cv2.putText(frame, label, (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    if valid_phone_center is not None:
                        cv2.circle(frame, (int(valid_phone_center[0]), int(valid_phone_center[1])), 10, (255, 0, 0), -1)
                        
                    if associated_bike_id not in self.already_saved_violations:
                        image_name = f"{self.video_id}_{int(timestamp)}s_bike{associated_bike_id}_PU.jpg"
                        temp_path = self.violation_dir / image_name
                        cv2.imwrite(str(temp_path), frame)
                        
                        self.pending_uploads.append({
                            "temp_path": temp_path,
                            "object_key": f"videos/{self.video_id}/violations/{image_name}",
                            "timestamp": timestamp,
                            "id": associated_bike_id,
                            "conf": max_phone_conf
                        })
                        self.already_saved_violations.add(associated_bike_id)
        
        if any_violation_found:
            cv2.putText(frame, "ALERT: Rider phone violation validated!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "System Scan Status: Road Normal.", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
        return frame

    def finish(self) -> dict:
        """
        Uploads photos to clean MinIO and creates companion JSON maps after loop completion.
        """
        print(f"[PHONE] Uploading {len(self.pending_uploads)} saved violation photos to storage...", flush=True)
        
        db_session = SessionLocal()
        try:
            for item in self.pending_uploads:
                try:
                    local_file_path = Path(item["temp_path"])
                    if not local_file_path.exists():
                        continue

                    uploaded = upload_file(
                        settings.minio_images_bucket,
                        item["object_key"],
                        local_file_path,  
                        "image/jpeg"
                    )
                    
                    current_timestamp = item["timestamp"]
                    
                    # Extract coordinates from route logs for database reference
                    lat, lon = None, None
                    try:
                        route_record = db_session.query(VideoRoute).filter(
                            VideoRoute.video_id == self.video_id,
                            VideoRoute.timestamp_seconds <= current_timestamp
                        ).order_by(VideoRoute.timestamp_seconds.desc()).first()
                        
                        if route_record:
                            lat, lon = route_record.latitude, route_record.longitude
                    except Exception as db_err:
                        print(f"⚠️ [Phone Processor] Database lookup error: {db_err}")

                    coords_display = f"Latitude: {lat}, Longitude: {lon}" if (lat and lon) else "DISABLED"

                    # Explicit database NULL configuration applied to plate_number fields
                    record_data = {
                        "timestamp_seconds": current_timestamp,
                        "bike_track_id": str(item["id"]),
                        "plate_number": None,  
                        "violation_type": "phone_usage",
                        "coordinates": coords_display,
                        "confidence": item["conf"],
                        "image_url": uploaded.object_url if uploaded else f"http://127.0.0.1:9000/{settings.minio_images_bucket}/{item['object_key']}"
                    }

                    self.violation_records.append(record_data)

                    # Companion JSON metadata profile construction step
                    json_filename = local_file_path.name.replace(".jpg", ".json")
                    local_json_path = local_file_path.parent / json_filename
                    
                    with open(local_json_path, "w", encoding="utf-8") as json_file:
                        json.dump(record_data, json_file, indent=2)
                    
                    json_object_key = item["object_key"].replace(".jpg", ".json")
                    
                    upload_file(
                        settings.minio_images_bucket,
                        json_object_key,
                        local_json_path,
                        "application/json"
                    )
                    
                    if local_json_path.exists():
                        local_json_path.unlink()
                    
                except Exception as e:
                    print(f"Network error processing phone cloud upload {item['object_key']}: {e}")
                finally:
                    if 'local_file_path' in locals() and local_file_path.exists():
                        local_file_path.unlink(missing_ok=True)
        finally:
            db_session.close()
            
        print("[PHONE USAGE DETECTOR FINISHED]", flush=True)
        return {
            "status": "completed",
            "violations": self.violation_records
        }