from pathlib import Path
import re
import cv2
from ultralytics import YOLO

from backend.app.core.config import get_settings
from backend.app.services.processors.plate_quality import PlateQualityAnalyzer
from backend.app.services.processors.plate_reader import PlateReader
from backend.app.services.storage import upload_file

settings = get_settings()


class NoNumberPlateDetector:

    def __init__(self):

        self.detector = YOLO(
            "backend/app/services/models/license_plate.pt"
        )

        self.analyzer = PlateQualityAnalyzer()

        self.ocr = PlateReader()

        self.violation_cooldown = {}
    
    def is_valid_plate(self, text):

        if text is None:
            return False

        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        if len(text) < 6:
            return False

        banned = [
            "FRONT",
            "KMH",
            "DDP",
            "PRO",
            "PIX",
            "PM",
            "AM"
        ]

        for word in banned:
            if word in text:
                return False

        return True

    def run(self, input_path, output_dir, video_id):

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_dir / "no_number_plate_output.mp4"

        violation_dir = output_dir / "violations"

        violation_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
        )

        violation_records = []

        # Frame loop
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            original_frame = frame.copy()

            results = self.detector(
                original_frame,
                conf=0.35
            )[0]
            # Plate loop
            for box in results.boxes.xyxy.cpu().numpy():

                x1, y1, x2, y2 = map(int, box[:4])

                crop = original_frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue
                # Run analyzer
                quality = self.analyzer.analyze(crop)
                plate_text = "UNKNOWN"

                plate_text = self.ocr.run_ocr(crop)
                if plate_text is None:
                    plate_text = ""

                plate_text = plate_text.strip()

                if not self.is_valid_plate(plate_text):
                    plate_text = "UNKNOWN"

                is_violation = (quality["status"] == "BLANK") or (quality["status"] == "UNREADABLE" and plate_text == "UNKNOWN")

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # floor division by a grid size of 120 pixels
                track_key = (
                x1 // 120,
                y1 // 120,
                x2 // 120,
                y2 // 120
                )
                if is_violation:

                    color = (0,0,255)

                    last_saved = self.violation_cooldown.get(track_key, -999)

                    if (timestamp - last_saved) >= 2:

                        plate_name = plate_text

                        image_name = f"{video_id}_{int(timestamp)}s_{plate_name}.jpg"

                        temp_path = violation_dir / image_name

                        # Save temporary image
                        cv2.imwrite(str(temp_path), frame)

                        # Upload to MinIO
                        object_key = f"videos/{video_id}/violations/{image_name}"

                        uploaded = upload_file(
                        settings.minio_images_bucket,
                        object_key,
                        temp_path,
                        "image/jpeg"
                        )

                        # Save violation record
                        violation_records.append({
                        "timestamp_seconds": timestamp,
                        "plate_number": plate_name,
                        "violation_type": "no_number_plate",
                        "confidence": 1.0,
                        "image_url": uploaded.object_url
                        })

                        print(
                        f"[NO NUMBER PLATE] "
                        f"Status={quality['status']}"
                        )

                        self.violation_cooldown[track_key] = timestamp

                        # Delete temporary image
                        temp_path.unlink(missing_ok=True)

                else:

                    if quality["status"] == "BLUR":
                        color = (255, 0, 0)

                    elif quality["status"] == "TOO_SMALL":
                        color = (0, 255, 255)

                    else:
                        color = (0, 255, 0)
                print(
                f"[OCR] {plate_text}"
                )

                print(
                f"[QUALITY] "
                f"{quality['status']} | "
                f"Blur={quality['blur']:.1f} | "
                f"Contrast={quality['contrast']:.1f} | "
                f"Edges={quality['edge_density']:.3f} | "
                f"Contours={quality['contours']}"
                )
                

                cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                color,
                2
                )

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
            #Save frame
            out.write(frame)
        cap.release()

        out.release()

        cv2.destroyAllWindows()

        return output_video, {
            "status":"completed",
            "violations": violation_records
            }
if __name__ == "__main__":

    detector = NoNumberPlateDetector()

    detector.run(
        input_path=r"C:\videoset1_videos_part1\20221010122208_0060speed2.mp4",
        output_dir=r"C:\temp\nnp_output",
        video_id="test_video"
    )