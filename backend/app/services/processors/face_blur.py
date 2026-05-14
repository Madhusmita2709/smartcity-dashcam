from pathlib import Path

import cv2

from backend.app.schemas.config import FaceBlurConfig


class FaceBlurProcessor:
    def __init__(self) -> None:
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def run(self, source: Path, config: FaceBlurConfig, output_dir: Path) -> tuple[Path, dict]:
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi = frame[y : y + h, x : x + w]
                if config.method == "gaussian":
                    frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                elif config.method == "pixelate":
                    scale = max(1, config.intensity // 4)
                    small = cv2.resize(
                        roi,
                        (max(1, w // scale), max(1, h // scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame[y : y + h, x : x + w] = cv2.resize(
                        small,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    frame[y : y + h, x : x + w] = (0, 0, 0)
                faces_detected += 1

            writer.write(frame)
            processed_frames += 1

        capture.release()
        writer.release()

        return target, {
            "status": "completed",
            "frames_processed": processed_frames,
            "faces_detected": faces_detected,
            "method": config.method,
            "intensity": config.intensity,
            "path": str(target),
        }
