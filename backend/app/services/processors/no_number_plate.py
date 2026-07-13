from pathlib import Path
import re
from typing import Dict, Any, List, Tuple, Optional

import cv2
from ultralytics import YOLO

from backend.app.core.config import get_settings
from backend.app.services.processors.plate_quality import PlateQualityAnalyzer
from backend.app.services.processors.plate_reader import PlateReader
from backend.app.services.storage import upload_file

settings = get_settings()


class NoNumberPlateDetector:
    def __init__(self):
        self.detector = YOLO("backend/app/services/models/license_plate.pt")
        self.analyzer = PlateQualityAnalyzer()
        self.ocr = PlateReader()
        
        # State tracking caches persistent across the video stream run
        self.violation_records: List[Dict[str, Any]] = []
        self.violation_cooldown: Dict[Tuple[int, int, int, int], float] = {}
        
        # Pipeline configuration variables initialized during compilation setup
        self.violation_dir: Optional[Path] = None
        self.video_id: Optional[str] = None

    def setup_session(self, output_dir: Path, video_id: str) -> None:
        """
        Initializes persistent structural requirements before a processing stream execution.
        Clears residual tracking parameters safely.
        """
        self.violation_records.clear()
        self.violation_cooldown.clear()
        
        self.video_id = video_id
        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_plate(self, text: str) -> bool:
        if text is None:
            return False

        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        if len(text) < 6:
            return False

        banned = ["FRONT", "KMH", "DDP", "PRO", "PIX", "PM", "AM"]
        for word in banned:
            if word in text:
                return False

        return True

    def process_frame(self, context: Dict[str, Any]) -> cv2.Mat:
        """
        Runs execution inference over a pre-filtered context snapshot.
        Mutates the target frame directly by drawing analytical tracking visuals.
        """
        # Unpack pure frame parameters from the generator payload
        frame = context["frame"]
        timestamp = context["timestamp_seconds"]
        
        # Guard clause: ensure setup was performed
        if self.violation_dir is None or self.video_id is None:
            raise RuntimeError("Detector session variables not initialized. Call setup_session() first.")

        original_frame = frame.copy()
        results = self.detector(original_frame, conf=0.35)[0]

        # Plate bounding box processing loop
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            crop = original_frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Run analytical features
            quality = self.analyzer.analyze(crop)
            plate_text = self.ocr.run_ocr(crop) or ""
            plate_text = plate_text.strip()

            if not self.is_valid_plate(plate_text):
                plate_text = "UNKNOWN"

            is_violation = (quality["status"] == "BLANK") or (
                quality["status"] == "UNREADABLE" and plate_text == "UNKNOWN"
            )

            # Spatial mapping calculations: tracking grid size of 120 pixels
            track_key = (x1 // 120, y1 // 120, x2 // 120, y2 // 120)

            if is_violation:
                color = (0, 0, 255)  # Red for violations
                last_saved = self.violation_cooldown.get(track_key, -999.0)

                # Cooldown validation checks
                if (timestamp - last_saved) >= 2.0:
                    plate_name = plate_text
                    image_name = f"{self.video_id}_{int(timestamp)}s_{plate_name}.jpg"
                    temp_path = self.violation_dir / image_name

                    # Save and upload asset snapshot arrays
                    cv2.imwrite(str(temp_path), frame)
                    object_key = f"videos/{self.video_id}/violations/{image_name}"

                    uploaded = upload_file(
                        settings.minio_images_bucket,
                        object_key,
                        temp_path,
                        "image/jpeg"
                    )

                    # Store decoupled record map entry
                    self.violation_records.append({
                        "timestamp_seconds": timestamp,
                        "plate_number": plate_name,
                        "violation_type": "no_number_plate",
                        "confidence": 1.0,
                        "image_url": uploaded.object_url
                    })

                    print(f"[NO NUMBER PLATE] Status={quality['status']}")
                    self.violation_cooldown[track_key] = timestamp
                    temp_path.unlink(missing_ok=True)
            else:
                if quality["status"] == "BLUR":
                    color = (255, 0, 0)
                elif quality["status"] == "TOO_SMALL":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                print(f"[OCR] {plate_text}")
                print(
                    f"[QUALITY] {quality['status']} | "
                    f"Blur={quality['blur']:.1f} | "
                    f"Contrast={quality['contrast']:.1f}"
                )

            # Mutate frame annotations locally
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{quality['status']} | {plate_text if plate_text else 'UNKNOWN'}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
        return frame

    def finish(self) -> Dict[str, Any]:
        """
        Compiles structural session reports after processing stream ends.
        """
        return {
            "status": "completed",
            "violations": self.violation_records
        }