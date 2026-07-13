from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import cv2
from ultralytics import YOLO
from backend.app.core.config import get_settings
from backend.app.services.storage import upload_file, download_file  # Hooked into storage downloader engine
from backend.app.models.video import VideoRoute 
from backend.app.db.database import SessionLocal

settings = get_settings()


class HelmetDetector:
    def __init__(self):
        print("[HELMET INIT]", flush=True)
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models and parameters initialized dynamically per execution session
        self.base_model: Optional[YOLO] = None
        self.custom_model: Optional[YOLO] = None
        self.tracker_module: str = "botsort.yaml"  
        
        # Stateful tracking caches running across the stream pass
        self.pending_uploads: List[dict] = []
        self.violation_records: List[dict] = []
        self.tracked_violations: dict = {}
        
        self.video_id: Optional[str] = None
        self.violation_dir: Optional[Path] = None
        self.check_no_helmet: bool = True
        self.device: str = "cpu"
        self.half_precision: bool = False

    def setup_session(
        self, 
        output_dir: Path, 
        video_id: str, 
        pipeline_config: Optional[dict] = None,
        vehicle_model: str = "yolov8n.pt",
        helmet_model: str = "cnn_helmet_detection(best).pt",
        tracker_module: str = "botsort.yaml"  
    ) -> None:
        """
        Loads configuration settings and triggers upfront custom model weights mapping once per run,
        pulling down missing asset files statefully from the system S3 model storage layer.
        """
        import torch
        torch.set_num_threads(4) 
        models_bucket = getattr(settings, "minio_models_bucket", "models")
        self.video_id = video_id
        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_uploads.clear()
        self.violation_records.clear()
        self.tracked_violations.clear()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half_precision = True if self.device == 'cuda' else False

        if pipeline_config is None:
            pipeline_config = {
                "face_blur": {"enabled": True, "method": "gaussian", "intensity": 25},
                "list_violations": ["no_helmet"]
            }
        
        self.check_no_helmet = "no_helmet" in pipeline_config.get("list_violations", ["no_helmet"])
        self.tracker_module = tracker_module

        # Determine structural validation check paths targets
        base_model_path = Path(vehicle_model) if Path(vehicle_model).is_absolute() else self.models_dir / vehicle_model
        custom_model_path = Path(helmet_model) if Path(helmet_model).is_absolute() else self.models_dir / helmet_model

        # --- REVOLUTIONARY MINIO WEIGHTS FALLBACK SYNCHRONIZATION ---
        try:
            if not base_model_path.exists():
                print(f"[HELMET] Downstream worker cluster missing baseline weight node. Syncing {vehicle_model} out of MinIO bucket store...", flush=True)
                download_file(models_bucket, vehicle_model, base_model_path)
            
            if not custom_model_path.exists():
                print(f"[HELMET] Downstream worker cluster missing custom classifier layer. Syncing {helmet_model} out of MinIO bucket store...", flush=True)
                download_file(models_bucket, helmet_model, custom_model_path)
        except Exception as sync_err:
            raise RuntimeError(f"Unable to download required model(s) from MinIO: {sync_err}")

        try:
            print(f"[HELMET] Activating base model track weights: {base_model_path}", flush=True)
            self.base_model = YOLO(str(base_model_path))
            print(f"[HELMET] Activating custom classifier weights: {custom_model_path}", flush=True)
            self.custom_model = YOLO(str(custom_model_path))
        except Exception as e:
            raise RuntimeError(f"Unable to initialize HelmetDetector weights mapping: {e}")
        
    def process_frame(self, context: Dict[str, Any]) -> cv2.Mat:
        """
        Accepts the global FrameContext payload on sampled intervals.
        Mutates context["frame"] canvas properties directly with evaluation tracks.
        """
        if self.base_model is None or self.custom_model is None or self.violation_dir is None or self.video_id is None:
            raise RuntimeError("HelmetDetector workspace uninitialized. Call setup_session() first.")

        frame = context["frame"]
        timestamp = context["timestamp_seconds"]
        frame_index = context["frame_index"]
        gps = context.get("gps") 
        
        height, width = frame.shape[:2]
        original_frame = frame.copy()

        combined_results = self.base_model.track(
            frame,
            classes=[0, 3],  
            conf=0.15,        
            imgsz=640,       
            half=self.half_precision,
            verbose=False,
            device=self.device,
            persist=True,     
            tracker=self.tracker_module 
        )

        current_frame_bikes = []
        current_frame_people = []

        if len(combined_results) > 0 and combined_results[0].boxes is not None:
            result = combined_results[0]
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [None] * len(boxes)

            for idx, (box, cls, t_id) in enumerate(zip(boxes, clss, track_ids)):
                final_id = t_id if t_id is not None else f"untracked_{frame_index}_{idx}"
                if cls == 3:
                    current_frame_bikes.append({"box": box, "id": final_id})
                elif cls == 0:
                    current_frame_people.append({"box": box, "id": final_id})

        for bike in current_frame_bikes:
            associated_id = bike["id"]
            bx1, by1, bx2, by2 = bike["box"]
            bike_h = by2 - by1
            bike_w = bx2 - bx1

            if not self.check_no_helmet:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bike: {associated_id}", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue

            rider_patches_to_check = []
            has_overlapping_person = False
            expanded_by1 = max(0, by1 - int(bike_h * 0.60))

            for person in current_frame_people:
                px1, py1, px2, py2 = person["box"]
                
                ix1 = max(bx1, px1)
                iy1 = max(expanded_by1, py1)
                ix2 = min(bx2, px2)
                iy2 = min(by2, py2)
                
                if (ix2 > ix1) and (iy2 > iy1): 
                    has_overlapping_person = True
                    rh = py2 - py1
                    hy1 = max(0, py1 - int(rh * 0.25))
                    hy2 = min(height, py1 + int(rh * 0.35))
                    hx1 = max(0, px1 - int((px2 - px1) * 0.15))
                    hx2 = min(width, px2 + int((px2 - px1) * 0.15))

                    patch = original_frame[hy1:hy2, hx1:hx2]
                    if patch.size > 0:
                        rider_patches_to_check.append(patch)

            if not has_overlapping_person:
                f_hy1 = max(0, by1 - int(bike_h * 0.50))
                f_hy2 = min(height, by1 + int(bike_h * 0.35))
                f_hx1 = max(0, bx1 - int(bike_w * 0.15))
                f_hx2 = min(width, bx2 + int(bike_w * 0.15))
                
                fallback_patch = original_frame[f_hy1:f_hy2, f_hx1:f_hx2]
                if fallback_patch.size > 0:
                    rider_patches_to_check.append(fallback_patch)

            bike_has_violation = False
            highest_violation_conf = 0.0
            target_conf = 0.40 if bike_h < 100 else 0.55

            for rider_patch in rider_patches_to_check:
                custom_results = self.custom_model(
                    rider_patch, 
                    conf=target_conf, 
                    imgsz=320, 
                    verbose=False, 
                    device=self.device
                )
                custom_dets = custom_results[0].boxes
                if custom_dets is not None and len(custom_dets) > 0:
                    for c_box in custom_dets:
                        if int(c_box.cls[0]) == 0: 
                            bike_has_violation = True
                            conf_val = float(c_box.conf[0])
                            if conf_val > highest_violation_conf:
                                highest_violation_conf = conf_val

            if bike_has_violation:
                if associated_id not in self.tracked_violations:
                    self.tracked_violations[associated_id] = highest_violation_conf
                    should_save_snapshot = True if "untracked" not in str(associated_id) else False
                elif highest_violation_conf > self.tracked_violations[associated_id]:
                    self.tracked_violations[associated_id] = highest_violation_conf
                    should_save_snapshot = True if "untracked" not in str(associated_id) else False
                else:
                    should_save_snapshot = False

                display_conf = self.tracked_violations[associated_id]
                render_y1 = max(0, by1 - int(bike_h * 0.1))
                label = f"ALERT: NO HELMET [ID {associated_id}] [{int(display_conf * 100)}%]"
                cv2.rectangle(frame, (bx1, render_y1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(frame, label, (bx1, render_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if should_save_snapshot:
                    image_name = f"{self.video_id}_{int(timestamp)}s_bike{associated_id}_nH.jpg"
                    temp_path = self.violation_dir / image_name
                    cv2.imwrite(str(temp_path), frame)

                    self.pending_uploads.append({
                        "temp_path": temp_path,
                        "object_key": f"videos/{self.video_id}/violations/{image_name}",
                        "timestamp": timestamp,
                        "id": associated_id,
                        "conf": display_conf
                    })
            
            elif associated_id in self.tracked_violations:
                display_conf = self.tracked_violations[associated_id]
                render_y1 = max(0, by1 - int(bike_h * 0.1))
                label = f"ALERT: NO HELMET [ID {associated_id}] [{int(display_conf * 100)}%]"
                cv2.rectangle(frame, (bx1, render_y1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(frame, label, (bx1, render_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            else:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bike: {associated_id}", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def finish(self) -> dict:
        """
        Uploads photos to clean MinIO and creates companion JSON maps after loop completion.
        """
        print(f"[HELMET] Uploading {len(self.pending_uploads)} saved violation photos to storage...", flush=True)
        
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
                        print(f"⚠️ [Helmet Processor] Database lookup error: {db_err}")

                    coords_display = f"Latitude: {lat}, Longitude: {lon}" if (lat and lon) else "DISABLED"

                    # Fix #1: plate_number falls back to clean explicit database NULL state value
                    record_data = {
                        "timestamp_seconds": current_timestamp,
                        "bike_track_id": str(item["id"]),
                        "plate_number": None,  
                        "violation_type": "no_helmet",
                        "coordinates": coords_display,
                        "confidence": item["conf"],
                        "image_url": uploaded.object_url if uploaded else f"http://127.0.0.1:9000/{settings.minio_images_bucket}/{item['object_key']}"
                    }

                    self.violation_records.append(record_data)

                    # Dynamic companion metadata file output tracking map pass
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
                    print(f"Network error processing helmet cloud upload {item['object_key']}: {e}")
                finally:
                    if 'local_file_path' in locals() and local_file_path.exists():
                        local_file_path.unlink(missing_ok=True)
        finally:
            db_session.close()
        print("[HELMET DETECTOR FINISHED]", flush=True)

        return {
            "status": "completed",
            "violations": self.violation_records  
        }