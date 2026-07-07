from pathlib import Path
import cv2
import numpy as np
import json
from ultralytics import YOLO
from backend.app.services.processors.dashboard_speed_reader import DashboardSpeedReader
from backend.app.core.config import get_settings
from backend.app.services.storage import upload_file

settings = get_settings()
from backend.app.services.processors.plate_reader import PlateReader

class VehicleSpeedEstimator:

    def __init__(self):
        print("[VEHICLE SPEED INIT]", flush=True)
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.default_model = self.models_dir / "yolov8n.pt"
        self.model = None
        self.track_history = {}
        self.H = None
        self.config = None
        self.speed_reader = DashboardSpeedReader()
        self.plate_reader = PlateReader()
    
    def scale_bbox(self, box, sx, sy):
        x1, y1, x2, y2 = box
        return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]

    def generate_track_key(self, mx1, my1, mx2, my2):
        center_x = (mx1 + mx2) // 2
        center_y = (my1 + my2) // 2
        return (center_x // 120, center_y // 120)

    def load_config(self, config_path):
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

    def to_bev(self, x, y):
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped[0][0]
    
    def get_lane_width_pixels(self, y_img):
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

        return np.linalg.norm(p2 - p1)

    def run(self, input_video, output_dir, video_id, config_path, vehicle_model="yolov8n.pt", tracker_module="bytetrack"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clear track history mapping to completely isolate multi-video runtime state calls
        self.track_history = {}

        output_video = output_dir / "overspeed_output.mp4"
        violation_dir = output_dir / "violations"
        violation_dir.mkdir(exist_ok=True)
        violation_records = []

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        print(f"[CONFIG PATH] {config_path}")
        print(f"[CONFIG EXISTS] {config_path.exists()}")
        self.config = self.load_config(config_path)

        ipm = self.config["ipm"]
        src = np.float32(ipm["src"])
        dst = np.float32(ipm["dst"])
        self.H = cv2.getPerspectiveTransform(src, dst)
        
        print(f"[IPM] SRC:\n{src}")
        print(f"[IPM] DST:\n{dst}")
        print(f"[IPM] Transform Matrix H:\n{self.H}")

        # Dynamic model validation lookup
        target_model = Path(vehicle_model)
        if not target_model.is_absolute() and not target_model.exists():
            local = self.models_dir / vehicle_model
            if local.exists():
                target_model = local
            else:
                print(f"[OVERSPEED] Model '{vehicle_model}' not found. Using '{self.default_model.name}' instead.", flush=True)
                target_model = self.default_model

        target_model_str = str(target_model)
        try:
            self.model = YOLO(target_model_str)
        except Exception as e:
            raise RuntimeError(f"Unable to load Overspeed calculation weight assets {target_model_str}: {e}")

        tracker = tracker_module
        if not tracker.endswith(".yaml"):
            tracker += ".yaml"

        conf = self.config.get("confidence_threshold", 0.15)
        imgsz = self.config.get("imgsz", 1280)
        
        # Measure lane width in BEV using calibration points
        left_top = np.array([[[1083, 800]]], dtype=np.float32)
        right_top = np.array([[[1404, 800]]], dtype=np.float32)

        left_bev = cv2.perspectiveTransform(left_top, self.H)[0][0]
        right_bev = cv2.perspectiveTransform(right_top, self.H)[0][0]

        lane_width_pixels = np.linalg.norm(right_bev - left_bev)
        print(f"Measured BEV lane width = {lane_width_pixels:.2f} pixels")

        self.meters_per_pixel = 3.5 / lane_width_pixels
        print(f"Meters per pixel = {self.meters_per_pixel:.6f}")

        frame_count = 0
        ego_speed = 0
        speed_limit = 40    # km/h
        
        # Headless execution fallback guard layer wrapper
        show_gui = False
        try:
            cv2.namedWindow("Vehicle Speed", cv2.WINDOW_NORMAL)
            show_gui = True
        except Exception:
            print("[GUI] Running headlessly inside computing runtime layer. Local canvas displays bypassed.", flush=True)

        overspeed_ids = set()
        while cap.isOpened():
            bev_canvas = np.zeros((1000, 600, 3), dtype=np.uint8)
            ret, frame = cap.read()
            if not ret:
                break
            original_frame = frame.copy()
            frame_count += 1

            if frame_count % 10 == 0:
                s = self.speed_reader.get_speed(frame)
                if s is not None:
                    ego_speed = s

            if ego_speed is None:
                ego_speed = 0

            results = self.model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=conf,
                imgsz=imgsz,
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

                    if 0 <= bx < 600 and 0 <= by < 1000:
                        cv2.circle(bev_canvas, (bx, by), 5, (0,255,0), -1)
                        cv2.putText(
                            bev_canvas,
                            str(track_id),
                            (bx+5, by),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255,255,255),
                            1
                        )
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []

                    self.track_history[track_id].append(
                        (bev[0], bev[1], cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                    )
                    history = self.track_history[track_id]
                    current_time = history[-1][2]
                    old_index = None

                    for i in range(len(history) - 1):
                        if current_time - history[i][2] >= 1.0:
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
                            print(f"ID={track_id} dx={dx:.2f} dy={dy:.2f} dist={distance_pixels:.2f} dt={dt:.2f}")

                            lane_width_pixels = self.get_lane_width_pixels(cy)
                            if lane_width_pixels < 10:
                                continue

                            meters_per_pixel = 3.5 / lane_width_pixels
                            distance_meters = distance_pixels * meters_per_pixel
                            speed_mps = distance_meters / dt
                            speed_kmh = speed_mps * 3.6
                            absolute_speed = speed_kmh + ego_speed
                        
                            print(f"ID={track_id} | Ego={ego_speed:.1f} | Relative={speed_kmh:.1f} | Vehicle={absolute_speed:.1f} km/h")
        
                            cv2.putText(
                                frame,
                                f"{absolute_speed:.1f} km/h",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255) if absolute_speed > speed_limit else (0, 255, 0),
                                2
                            )
                            if absolute_speed > speed_limit and track_id not in overspeed_ids:
                                overspeed_ids.add(track_id)
                                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
	 
                                track_key = self.generate_track_key(x1, y1, x2, y2)
                                plate_number = "UNKNOWN"
                                plate_results = self.plate_reader.read_plate(original_frame, track_key)
                                print("Plate OCR:", plate_results)

                                if plate_results:
                                    candidate = plate_results[0].get("plate", "UNKNOWN")
                                    if candidate and candidate != "UNKNOWN":
                                        plate_number = candidate

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

                                # NOTE: Stores measured velocity inside confidence schema payload field directly
                                violation_records.append({
                                    "timestamp_seconds": timestamp,
                                    "plate_number": plate_number,
                                    "violation_type": "overspeed",
                                    "confidence": absolute_speed,
                                    "image_url": uploaded.object_url
                                })

                                print(f"[OVERSPEED] ID={track_id} Speed={absolute_speed:.1f}")
                                temp_path.unlink(missing_ok=True)
                                
                            cv2.putText(
                                frame,
                                f"{speed_kmh:.1f}",
                                (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2
                            )

                    if len(self.track_history[track_id]) > 120:
                        self.track_history[track_id].pop(0)

                    cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )
            out.write(frame)
            
            if show_gui:
                cv2.imshow("Vehicle Speed", cv2.resize(frame, (1280, 720)))
                if cv2.waitKey(1) == ord("q"):
                    break
            
        cap.release()
        out.release()
        if show_gui:
            cv2.destroyAllWindows()
            
        return output_video, {
            "status": "completed",
            "violations": violation_records
        }