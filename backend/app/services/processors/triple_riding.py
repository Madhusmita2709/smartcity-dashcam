# Standard library imports
from collections import deque
from pathlib import Path

# Third-party imports
import cv2
from ultralytics import YOLO

# Local imports
from backend.app.core.config import get_settings
from backend.app.services.processors.plate_reader import PlateReader
from backend.app.services.storage import upload_file

settings = get_settings()


class TripleRidingDetector:
    def __init__(self):
        # Dynamically calculate the models directory based on file location
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.default_model_path = self.models_dir / "triple_riding.pt"
        
        self.plate_reader = PlateReader()
        self.violation_cooldown = {}

    def scale_bbox(self, box, sx, sy):
        x1, y1, x2, y2 = box
        return [
            int(x1 * sx),
            int(y1 * sy),
            int(x2 * sx),
            int(y2 * sy)
        ]

    def generate_track_key(self, mx1, my1, mx2, my2):
        center_x = (mx1 + mx2) // 2
        center_y = (my1 + my2) // 2
        return (center_x // 120, center_y // 120)

    def run(self, input_path, output_dir, video_id, vehicle_model="yolov8n.pt", person_model="triple_riding.pt"):
        """
        Runs the Triple Riding detection workflow.
        
        NOTE: Triple Riding uses a single unified YOLO weights file that handles both
        motorcycles (Class 0) and persons (Class 1) together. The 'vehicle_model' parameter
        is accepted here solely to maintain interface consistency across the pipeline execution engine.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_dir / "triple_riding_output.mp4"
        violation_dir = output_dir / "violations"
        violation_dir.mkdir(exist_ok=True)

        # FIXED: Pure Path manipulation for future-proof robust tracking resolution
        target_model = Path(person_model)
        if not target_model.is_absolute() and not target_model.exists():
            local_check = self.models_dir / person_model
            target_model = local_check if local_check.exists() else self.default_model_path

        # Convert back to plain string specifically for YOLO loader contract compliance
        target_model_str = str(target_model)

        print(f"[TRIPLE RIDING] Initializing dynamic model: {target_model_str}", flush=True)
        
        # Guarded model initialization to prevent unhandled stack trace crashes
        try:
            triple_model = YOLO(target_model_str)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load Triple Riding execution weights layer from '{target_model_str}': {str(error)}"
            )
            
        class_names = triple_model.names
        print(f"[MODEL CLASSES] {class_names}", flush=True)

        cap = cv2.VideoCapture(str(input_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        violation_records = []
        frame_index = 0
        frame_buffer = deque(maxlen=int(fps * 5))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[VIDEO END]")
                break

            frame_buffer.append((frame_index, frame.copy()))
            frame_index += 1
            original_frame = frame.copy()

            # Execute model inference
            results = triple_model(frame, conf=0.25, imgsz=1280)
            
            motorcycles = []
            persons = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == 0:   # Motorcycle
                        motorcycles.append([x1, y1, x2, y2, conf])
                    elif cls == 1:   # Person
                        persons.append([x1, y1, x2, y2, conf])

            for moto in motorcycles:
                mx1, my1, mx2, my2, moto_conf = moto
                rider_count = 0
                conf_scores = []

                for person in persons:
                    px1, py1, px2, py2, pconf = person
                    foot_x = (px1 + px2) // 2
                    foot_y = py2
                    if (mx1 - 40 <= foot_x <= mx2 + 40 and my1 - 30 <= foot_y <= my2 + 30):
                         rider_count += 1
                         conf_scores.append(pconf)

                if rider_count >= 3:
                    avg_person_conf = (sum(conf_scores) / len(conf_scores) if conf_scores else 0)
                    final_conf = (avg_person_conf + moto_conf) / 2
                    
                    cv2.imwrite(str(violation_dir / f"debug_{frame_index}.jpg"), original_frame)
                    
                    for idx, prev_frame in frame_buffer:
                        prev_name = f"{video_id}_buffer_{frame_index}_{idx}.jpg"
                        prev_path = violation_dir / prev_name
                        cv2.imwrite(str(prev_path), prev_frame)

                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    track_key = self.generate_track_key(mx1, my1, mx2, my2)
                    candidate = "UNKNOWN"

                    plate_results = self.plate_reader.read_plate(original_frame, track_key)
                    if plate_results:
                        candidate = plate_results[0].get("plate", "UNKNOWN")

                    plate_number = candidate if (candidate and candidate.strip() and candidate != "UNKNOWN") else "UNKNOWN"

                    if plate_number == "UNKNOWN":
                        buffer_candidates = []
                        for _, prev_frame in list(frame_buffer)[::-5]:
                            prev_results = self.plate_reader.read_plate(prev_frame, track_key)
                            if prev_results:
                                candidate = prev_results[0].get("plate", "UNKNOWN")
                                if candidate and candidate != "UNKNOWN":
                                     buffer_candidates.append(candidate)
                        if buffer_candidates:
                            plate_number = self.plate_reader.vote_plate(buffer_candidates)

                    label = f"TRIPLE {final_conf:.2f} | {plate_number}"

                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (mx1, my1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    last_saved = self.violation_cooldown.get(track_key, -999)
                    if (timestamp - last_saved) >= 2:
                        image_name = f"{video_id}_{int(timestamp)}s_{plate_number}.jpg"
                        temp_path = violation_dir / image_name
                        cv2.imwrite(str(temp_path), frame)

                        object_key = f"videos/{video_id}/violations/{image_name}"
                        uploaded = upload_file(
                            settings.minio_images_bucket,
                            object_key,
                            temp_path,
                            "image/jpeg"
                        )

                        violation_records.append({
                            "timestamp_seconds": timestamp,
                            "plate_number": plate_number,
                            "violation_type": "triple_riding",
                            "confidence": final_conf,
                            "image_url": uploaded.object_url
                        })

                        self.violation_cooldown[track_key] = timestamp
                        temp_path.unlink(missing_ok=True)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return output_video, {
            "status": "completed",
            "violations": violation_records
        }