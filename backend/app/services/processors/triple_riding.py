# Standard library imports
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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
        
        # Heavy component ()
        self.violation_cooldown: Dict[Tuple[int, int], float] = {}
        
        # Session state placeholders
        self.model: Optional[YOLO] = None
        self.violation_records: List[Dict[str, Any]] = []
        self.frame_buffer: deque = deque(maxlen=125)  # Fallback size, re-adjusted in setup_session
        self.video_id: Optional[str] = None
        self.violation_dir: Optional[Path] = None
        self.plate_reader = PlateReader()

    def setup_session(self, output_dir: Path, video_id: str,vehicle_model: str = "yolov8n.pt", person_model: str = "triple_riding.pt", fps: float = 25.0) -> None:
        """
        Initializes persistent structural session assets and loads model weights once
        per video processing track execution loop. Safely flushes historical residual cache tracking maps.
        """
        self.video_id = video_id
        self.violation_records.clear()
        self.violation_cooldown.clear()

        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)

        # Re-initialize the sliding frame queue length based on active video FPS tracking metrics
        self.frame_buffer = deque(maxlen=int(max(1.0, fps) * 5))

        # Compile and check dynamic neural network asset directory locations safely
        target_model = Path(person_model)
        if not target_model.is_absolute() and not target_model.exists():
            local_check = self.models_dir / person_model
            target_model = local_check if local_check.exists() else self.default_model_path

        target_model_str = str(target_model)
        print(f"[TRIPLE RIDING] Initializing dynamic model session: {target_model_str}", flush=True)
        
        try:
            self.model = YOLO(target_model_str)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load Triple Riding execution weights layer from '{target_model_str}': {str(error)}"
            )

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

    def process_frame(self, context: Dict[str, Any]):
        """
        Executes localized violation tracking loops over incoming generator contexts.
        Mutates the incoming frame reference context parameters directly.
        """
        # Guard clause: protect array evaluations against missing session parameters
        if self.model is None or self.violation_dir is None or self.video_id is None:
            raise RuntimeError("Detector session variables not initialized. Call setup_session() first.")

        # Unpack normalized context objects straight out of the stream node
        frame = context["frame"]
        frame_index = context["frame_index"]
        timestamp = context["timestamp_seconds"]
        gps = context.get("gps")

        # Cache target image vectors sequentially inside historical matrix sequence arrays
        self.frame_buffer.append((frame_index, frame.copy()))
        original_frame = frame.copy()

        # Execute object classification sweeps using the structural model configuration state
        results = self.model(frame, conf=0.25, imgsz=1280)
        
        motorcycles = []
        persons = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 0:    # Motorcycle
                    motorcycles.append([x1, y1, x2, y2, conf])
                elif cls == 1:  # Person
                    persons.append([x1, y1, x2, y2, conf])

        # Mathematical verification sweeps scanning intersection grids
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

            # Violation triggered path
            if rider_count >= 3:
                avg_person_conf = (sum(conf_scores) / len(conf_scores) if conf_scores else 0)
                final_conf = (avg_person_conf + moto_conf) / 2
                
                # Draw local trace asset parameters down into standard disk layers
                cv2.imwrite(str(self.violation_dir / f"debug_{frame_index}.jpg"), original_frame)
                
                for idx, prev_frame in self.frame_buffer:
                    prev_name = f"{self.video_id}_buffer_{frame_index}_{idx}.jpg"
                    prev_path = self.violation_dir / prev_name
                    cv2.imwrite(str(prev_path), prev_frame)

                track_key = self.generate_track_key(mx1, my1, mx2, my2)
                candidate = "UNKNOWN"

                plate_results = self.plate_reader.read_plate(original_frame, track_key)
                if plate_results:
                    candidate = plate_results[0].get("plate", "UNKNOWN")

                plate_number = candidate if candidate and candidate.strip() and candidate != "UNKNOWN" else "UNKNOWN"

                if plate_number == "UNKNOWN":
                    buffer_candidates = []
                    for _, prev_frame in list(self.frame_buffer)[::-5]:
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

                last_saved = self.violation_cooldown.get(track_key, -999.0)
                if (timestamp - last_saved) >= 2.0:
                    image_name = f"{self.video_id}_{int(timestamp)}s_{plate_number}.jpg"
                    temp_path = self.violation_dir / image_name
                    cv2.imwrite(str(temp_path), frame)

                    object_key = f"videos/{self.video_id}/violations/{image_name}"
                    uploaded = upload_file(
                        settings.minio_images_bucket,
                        object_key,
                        temp_path,
                        "image/jpeg"
                    )

                    self.violation_records.append({
                        "timestamp_seconds": timestamp,
                        "plate_number": plate_number,
                        "violation_type": "triple_riding",
                        "confidence": final_conf,
                        "image_url": uploaded.object_url
                    })

                    self.violation_cooldown[track_key] = timestamp
                    temp_path.unlink(missing_ok=True)


    def finish(self) -> Dict[str, Any]:
        """
        Compiles the historical execution payload telemetry dictionary back to orchestrator.
        """
        return {
            "status": "completed",
            "violations": self.violation_records
        }