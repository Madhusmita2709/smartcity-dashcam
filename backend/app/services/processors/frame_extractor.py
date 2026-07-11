from pathlib import Path
import re

import cv2
import easyocr

from backend.app.schemas.config import FrameExtractionConfig

class FrameExtractionProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.coord_pattern = re.compile(r"[Nn]\s*(\d+\.\d+).*?[Ee]\s*(\d+\.\d+)")

    def run(self, source: Path, config: FrameExtractionConfig, output_dir: Path) -> tuple[list[dict], dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        gps_timeline = []
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
                timestamp_seconds = frame_index / fps
                frames.append(
                    {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "path": str(frame_path),
                    }
                )
                h, w, _ = frame.shape

                crop_top = int(h * 0.88)
                crop_bottom = int(h * 0.98)
                crop_left = int(w * 0.15)
                crop_right = int(w * 0.65)

                cropped_zone = frame[crop_top:crop_bottom, crop_left:crop_right]

                gray_zone = cv2.cvtColor(cropped_zone, cv2.COLOR_BGR2GRAY)

                try:
                    ocr_results = self.reader.readtext(gray_zone, detail=0)
                    combined_text = " ".join(ocr_results)
                    match = self.coord_pattern.search(combined_text)
                except Exception:
                    match = None

                if match:
                    lat = float(match.group(1))
                    lon = float(match.group(2))

                    is_duplicate = False

                    if gps_timeline:
                        last = gps_timeline[-1]

                        if (last["latitude"] == lat and last["longitude"] == lon):
                            is_duplicate = True

                    if not is_duplicate:
                        gps_timeline.append({
                            "timestamp": round(timestamp_seconds, 2),
                            "frame_index": frame_index,
                            "latitude": lat,
                            "longitude": lon
                        })
        

            frame_index += 1

        capture.release()
        return frames, {
            "status": "completed",
            "method": config.method,
            "value": config.value,
            "motion_threshold": config.motion_threshold,
            "frames_extracted": len(frames),
            "gps_timeline": gps_timeline
        }
