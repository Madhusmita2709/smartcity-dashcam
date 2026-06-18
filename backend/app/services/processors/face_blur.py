from pathlib import Path
import cv2
from ultralytics import YOLO

from backend.app.schemas.config import FaceBlurConfig


class FaceBlurProcessor:
    def __init__(self) -> None:
        # Load the YOLOv8 face model. 
        # 'yolov8n-face.pt' is a popular community face model, or you can use a generic weights file.
        # It will automatically download on the first run if not present locally.
        self.model = YOLO("C:/Users/madhu/Downloads/smartcity-dashcam-main/smartcity-dashcam-main/backend/app/services/processors/yolov8n-face.pt")

    def run(self, source: Path, config: FaceBlurConfig, output_dir: Path) -> tuple[Path, dict]:
        print("[FACE BLUR STARTED]", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{source.stem}_faces{source.suffix}"

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        processed_frames = 0
        faces_detected = 0
        kernel_size = max(3, config.intensity | 1)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # YOLOv8 expects BGR frames natively, so no grayscale conversion needed!
            # verbose=False keeps your terminal clean from frame-by-frame log printouts
            results = self.model(frame, verbose=False)
            
            # Extract bounding boxes from the first result object
            boxes = results[0].boxes

            for box in boxes:
                # Convert tensor coordinates to integers (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Double-check bounding box boundaries stay within frame dimensions
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                # Define Region of Interest (ROI) using coordinates
                roi = frame[y1:y2, x1:x2]
                
                if config.method == "gaussian":
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                elif config.method == "pixelate":
                    scale = max(1, config.intensity // 4)
                    small = cv2.resize(
                        roi,
                        (max(1, w // scale), max(1, h // scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame[y1:y2, x1:x2] = cv2.resize(
                        small,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    frame[y1:y2, x1:x2] = (0, 0, 0)
                    
                faces_detected += 1

            writer.write(frame)
            processed_frames += 1

        capture.release()
        writer.release()

        print(
            f"[FACE BLUR FINISHED] "
            f"frames={processed_frames} "
            f"faces={faces_detected}",
            flush=True
        )
        return target, {
            "status": "completed",
            "frames_processed": processed_frames,
            "faces_detected": faces_detected,
            "method": config.method,
            "intensity": config.intensity,
            "path": str(target),
        }