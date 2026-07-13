from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import json

import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.core.config import get_settings
from backend.app.services.processors.dashboard_speed_reader import DashboardSpeedReader
from backend.app.services.processors.plate_reader import PlateReader
from backend.app.services.storage import upload_file

settings = get_settings()


class VehicleSpeedEstimator:

    def __init__(self):
        print("[VEHICLE SPEED INIT]", flush=True)
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.default_model = self.models_dir / "yolov8n.pt"
        
        # Heavy static sub-processors initialized once per application lifecycle
        self.speed_reader = DashboardSpeedReader()
        self.plate_reader = PlateReader()

        # Session properties and weights mapping configs
        self.model: Optional[YOLO] = None
        self.config: Optional[dict] = None
        self.H: Optional[np.ndarray] = None
        self.tracker: str = "bytetrack.yaml"
        self.conf_threshold: float = 0.15
        self.imgsz: int = 1280
        self.meters_per_pixel: float = 0.0
        self.fps: float = 25.0
        
        # Persistent state tracking variables per video sequence execution pass
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.overspeed_ids: Set[int] = set()
        self.violation_records: List[dict] = []
        self.frame_count: int = 0
        self.ego_speed: float = 0.0
        self.speed_limit: float = 40.0  # km/h
        
        self.violation_dir: Optional[Path] = None
        self.video_id: Optional[str] = None

    def scale_bbox(self, box, sx, sy):
        x1, y1, x2, y2 = box
        return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

    def generate_track_key(self, mx1, my1, mx2, my2):
        center_x = (mx1 + mx2) // 2
        center_y = (my1 + my2) // 2
        return (center_x // 120, center_y // 120)

    def load_config(self, config_path: Path) -> dict:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    print(f"[CONFIG] Loading overspeed configuration from {config_path}", flush=True)
                    return json.load(f)
            except Exception as e:
                print(f"[CONFIG] Error reading config file: {e}. Falling back to default settings.", flush=True)
        else:
            print("[CONFIG] config.json not found for speed estimation. Using core factory defaults.", flush=True)
            
        return {
            "ipm": {
                "src": [[920, 700], [1100, 700], [350, 1440], [900, 1440]],
                "dst": [[100, 0], [200, 0], [100, 1000], [200, 1000]]
            },
            "confidence_threshold": 0.15,
            "imgsz": 1280
        }

    def to_bev(self, x: float, y: float) -> np.ndarray:
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped[0][0]
        
    def get_lane_width_pixels(self, y_img: float) -> float:
        ipm = self.config["ipm"]
        src = np.float32(ipm["src"])

        left_top = src[0]
        right_top = src[1]
        left_bottom = src[2]
        right_bottom = src[3]

        ratio = (y_img - left_top[1]) / (left_bottom[1] - left_top[1])
        ratio = np.clip(ratio, 0.0, 1.0)

        left_x = left_top[0] + ratio * (left_bottom[0] - left_top[0])
        right_x = right_top[0] + ratio * (right_bottom[0] - right_top[0])

        p1 = self.to_bev(left_x, y_img)
        p2 = self.to_bev(right_x, y_img)

        return float(np.linalg.norm(p2 - p1))

    def setup_session(
        self, 
        output_dir: Path, 
        video_id: str, 
        config_path: Path,  # Fix: Decoupled dependency calibration path injection
        vehicle_model: str = "yolov8n.pt", 
        tracker_module: str = "bytetrack",
        fps: float = 25.0
    ) -> None:
        """
        Loads path calibration logs and prepares custom neural weights networks once, 
        while safely clearing past state properties between consecutive videos.
        """
        self.video_id = video_id
        self.fps = fps if fps > 0 else 25.0

        # Fix #9: Isolating image violation logs to a clean, unique violations partition directory
        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize loop state memory tracking caches safely
        self.track_history.clear()
        self.overspeed_ids.clear()
        self.violation_records.clear()
        self.frame_count = 0
        self.ego_speed = 0.0
        self.speed_limit = 40.0  # km/h

        # Bind workspace spatial calibration properties
        self.config = self.load_config(config_path)

        ipm = self.config["ipm"]
        src = np.float32(ipm["src"])
        dst = np.float32(ipm["dst"])
        self.H = cv2.getPerspectiveTransform(src, dst)
        
        # Dynamic object model selection verification loop tracks
        target_model = Path(vehicle_model)
        if not target_model.is_absolute() and not target_model.exists():
            local = self.models_dir / vehicle_model
            target_model = local if local.exists() else self.default_model

        target_model_str = str(target_model)
        try:
            self.model = YOLO(target_model_str)
        except Exception as e:
            raise RuntimeError(f"Unable to load Overspeed tracking weights {target_model_str}: {e}")

        # Standardize configuration tracker filename paths references
        self.tracker = tracker_module
        if not self.tracker.endswith(".yaml"):
            self.tracker += ".yaml"

        self.conf_threshold = self.config.get("confidence_threshold", 0.15)
        self.imgsz = self.config.get("imgsz", 1280)
        
        # Precompute target baseline meters-per-pixel pixel resolution calculations
        left_top_cal = np.array([[[1083, 800]]], dtype=np.float32)
        right_top_cal = np.array([[[1404, 800]]], dtype=np.float32)

        left_bev = cv2.perspectiveTransform(left_top_cal, self.H)[0][0]
        right_bev = cv2.perspectiveTransform(right_top_cal, self.H)[0][0]

        lane_width_pixels = np.linalg.norm(right_bev - left_bev)
        self.meters_per_pixel = 3.5 / lane_width_pixels if lane_width_pixels > 0 else 0.0

        print(f"[OVERSPEED] Session configured. Measured Mpp = {self.meters_per_pixel:.6f} | Tracker = {self.tracker}", flush=True)

    def process_frame(self, context: Dict[str, Any]) -> cv2.Mat:
        """
        Processes real-time tracking velocity estimates on incoming frame snapshots.
        Mutates context["frame"] canvas properties directly with vehicle velocity visuals.
        """
        if self.model is None or self.violation_dir is None or self.video_id is None:
            raise RuntimeError("Overspeed tracking variables uninitialized. Call setup_session() first.")

        # Fixed: Real-time loop engine metrics tracking incremented safely
        self.frame_count += 1

        # Unpack pure contextual carrier coordinates straight from the pipeline engine
        frame = context["frame"]
        timestamp = context["timestamp_seconds"]
        frame_index = context["frame_index"]
        gps = context.get("gps")
        original_frame = frame.copy()

        # Execute OCR speed readout tracking safely inside uniform sampling step segments
        if self.frame_count % 10 == 0:
            read_speed = self.speed_reader.get_speed(frame)
            if read_speed is not None:
                self.ego_speed = float(read_speed)

        if self.ego_speed is None:
            self.ego_speed = 0.0

        # Execute fast persist tracking inferences over the streaming loop sequence
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False
        )

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if box.id is None:
                    continue

                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                if class_name not in ["car", "truck", "bus", "motorcycle"]:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = y2

                bev = self.to_bev(cx, cy)
                bx = int(bev[0])
                by = int(bev[1])

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append((bev[0], bev[1], timestamp))
                history = self.track_history[track_id]
                old_index = None

                for i in range(len(history) - 1):
                    if timestamp - history[i][2] >= 1.0:
                        old_index = i
                        break

                if old_index is not None:
                    old_x, old_y, old_t = history[old_index]
                    new_x, new_y, new_t = history[-1]
                    dt = new_t - old_t
                    
                    if dt > 0:
                        dx = new_x - old_x
                        dy = new_y - old_y
                        
                        if abs(dx) < 2 and abs(dy) < 2:
                            continue
                            
                        distance_pixels = np.sqrt(dx * dx + dy * dy)
                        lane_width_pixels = self.get_lane_width_pixels(cy)
                        if lane_width_pixels < 10:
                            continue

                        current_mpp = 3.5 / lane_width_pixels
                        distance_meters = distance_pixels * current_mpp
                        speed_mps = distance_meters / dt
                        speed_kmh = speed_mps * 3.6
                        absolute_speed = speed_kmh + self.ego_speed
                    
                        # Draw velocity tracking text vectors down into the canvas reference array
                        is_overspeeding = absolute_speed > self.speed_limit
                        text_color = (0, 0, 255) if is_overspeeding else (0, 255, 0)
                        
                        cv2.putText(
                            frame,
                            f"{absolute_speed:.1f} km/h",
                            (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            text_color,
                            2
                        )
                        
                        cv2.putText(
                            frame,
                            f"{speed_kmh:.1f}",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

                        # Trigger violation pipeline uploads
                        if is_overspeeding and track_id not in self.overspeed_ids:
                            self.overspeed_ids.add(track_id)

                            track_key = self.generate_track_key(x1, y1, x2, y2)
                            plate_number = "UNKNOWN"
                            plate_results = self.plate_reader.read_plate(original_frame, track_key)

                            if plate_results:
                                candidate = plate_results[0].get("plate", "UNKNOWN")
                                if candidate and candidate != "UNKNOWN":
                                    plate_number = candidate

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

                            # Note: Absolute velocity logged inside the confidence key per project payload specifications
                            self.violation_records.append({
                                "timestamp_seconds": timestamp,
                                "plate_number": plate_number,
                                "violation_type": "overspeed",
                                "confidence": absolute_speed,
                                "image_url": uploaded.object_url
                            })

                            print(f"[OVERSPEED TRIGGERED] ID={track_id} Absolute Velocity={absolute_speed:.1f} km/h")
                            temp_path.unlink(missing_ok=True)

                if len(self.track_history[track_id]) > 120:
                    self.track_history[track_id].pop(0)

                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
        return frame

    def finish(self) -> Dict[str, Any]:
        print("[VEHICLE SPEED ESTIMATOR FINISHED]", flush=True)
        return {
            "status": "completed",
            "violations": self.violation_records
        }