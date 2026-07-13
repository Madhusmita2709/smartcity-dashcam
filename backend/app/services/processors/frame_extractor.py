from pathlib import Path
import re
from typing import Generator, Dict, Any, Optional, List, Tuple

import cv2
import easyocr

from backend.app.schemas.config import FrameExtractionConfig


class FrameStreamProcessor:
    def __init__(self):
        # Heavy model initialization stays here
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.coord_pattern = re.compile(r"[Nn]\s*(\d+\.\d+).*?[Ee]\s*(\d+\.\d+)")
        
        # Telemetry storage
        self.frames_saved: List[Dict[str, Any]] = []
        self.gps_timeline: List[Dict[str, Any]] = []
        self.total_frames = 0

    def run(
        self, 
        source: Path, 
        config: FrameExtractionConfig, 
        output_dir: Path
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Sequentially reads a video stream, applies spatial/temporal frame filtering, 
        performs GPS OCR, and yields a lightweight, decoupled FrameContext dict.
        """
        # Clear state cache to support safe multi-video execution loops
        self.frames_saved.clear()
        self.gps_timeline.clear()
        self.total_frames = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_index = 0
        previous_gray = None

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # --- 1. SAMPLING FILTER MATRIX ---
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

                if not save_frame:
                    frame_index += 1
                    continue

                # --- 2. STORAGE AND TELEMETRY MANAGEMENT ---
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                timestamp_seconds = frame_index / fps

                self.frames_saved.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "path": str(frame_path)
                })

                # --- 3. ROBUST GPS OCR SUB-SYSTEM ---
                h, w, _ = frame.shape
                crop_top = int(h * 0.88)
                crop_bottom = int(h * 0.98)
                crop_left = int(w * 0.15)
                crop_right = int(w * 0.65)

                cropped_zone = frame[crop_top:crop_bottom, crop_left:crop_right]
                gray_zone = cv2.cvtColor(cropped_zone, cv2.COLOR_BGR2GRAY)

                gps_data = None
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

                    if self.gps_timeline:
                        last = self.gps_timeline[-1]
                        if last["latitude"] == lat and last["longitude"] == lon:
                            is_duplicate = True

                    if not is_duplicate:
                        gps_data = {
                            "timestamp": round(timestamp_seconds, 2),
                            "frame_index": frame_index,
                            "latitude": lat,
                            "longitude": lon
                        }
                        self.gps_timeline.append(gps_data)

                # --- 4. YIELD CLEAN FRAME CONTEXT ---
                print(f"[STREAM] Yielding frame {frame_index}")
                yield {
                    "frame": frame,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "gps": gps_data,
                }
                
                frame_index += 1

        finally:
            capture.release()

    def get_summary(self, config: FrameExtractionConfig) -> Tuple[List[dict], dict]:
        """Exposes detailed performance and extraction tracking maps."""
        return self.frames_saved, {
            "status": "completed",
            "method": config.method,
            "value": config.value,
            "motion_threshold": config.motion_threshold,
            "total_frames": self.total_frames,
            "frames_analyzed": len(self.frames_saved),
            "frames_skipped": self.total_frames - len(self.frames_saved),
            "gps_timeline": self.gps_timeline
        }