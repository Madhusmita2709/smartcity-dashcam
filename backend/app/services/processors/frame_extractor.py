from pathlib import Path

import cv2

from backend.app.schemas.config import FrameExtractionConfig


class FrameExtractionProcessor:
    def run(self, source: Path, config: FrameExtractionConfig, output_dir: Path) -> tuple[list[dict], dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        frame_index = 0
        previous_gray = None

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            save_frame = False
            if config.method == "interval":
                step = max(1, int(config.value))
                save_frame = frame_index % step == 0
            elif config.method == "fps":
                interval = max(1, int(round(fps / config.value)))
                save_frame = frame_index % interval == 0
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if previous_gray is None:
                    save_frame = True
                else:
                    delta = cv2.absdiff(previous_gray, gray)
                    save_frame = float(delta.mean()) >= config.motion_threshold
                previous_gray = gray

            if save_frame:
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                        "path": str(frame_path),
                    }
                )

            frame_index += 1

        capture.release()
        return frames, {
            "status": "completed",
            "method": config.method,
            "value": config.value,
            "motion_threshold": config.motion_threshold,
            "frames_extracted": len(frames),
        }
