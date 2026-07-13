from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
import json

import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.core.config import get_settings
from backend.app.services.processors.plate_reader import PlateReader
from backend.app.services.storage import upload_file

settings = get_settings()


class WrongWayDetector:
    def __init__(self):
        print("[WRONG WAY INIT]", flush=True)
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.default_model = self.models_dir / "yolov8n.pt"
        self.plate_reader = PlateReader()
        
        # Session state models and spatial configuration mappings
        self.model: Optional[YOLO] = None
        self.config: dict = {}
        self.polygons: Dict[str, np.ndarray] = {}
        self.H: Optional[np.ndarray] = None
        self.tracker: str = "bytetrack.yaml"
        self.fps: float = 25.0  # Kept purely as a legacy fallback parameter

        # Stateful metrics running across frame stream loops
        self.track_history: dict = {}
        self.wrong_way_frames: dict = {}
        self.flagged_wrong_way: Set[int] = set()
        self.saved_violations: Set[int] = set()
        self.violations_log: List[dict] = []
        
        # Lucas-Kanade optical flow settings
        self.lk_params = dict(
            winSize=(15, 15), 
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.current_v_ego: float = 300.0
        self.frame_count: int = 0
        
        self.violation_dir: Optional[Path] = None
        self.video_id: Optional[str] = None

    def load_config(self, config_path: Path) -> dict:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    print(f"[CONFIG] Loading calibration file from {config_path}", flush=True)
                    return json.load(f)
            except Exception as e:
                print(f"[CONFIG] Error reading config.json: {e}. Applying default values.", flush=True)
        else:
            print(f"[CONFIG] config.json target not found at {config_path}. Applying hardcoded fallback defaults.", flush=True)
            
        return {
            "polygons": {
                "SHOULDER": [[100, 1440], [850, 700], [920, 700], [350, 1440]],
                "LANE_1": [[350, 1440], [920, 700], [1100, 700], [900, 1440]],
                "LANE_2": [[900, 1440], [1100, 700], [1280, 700], [1650, 1440]],
                "DIVIDER": [[1650, 1440], [1280, 700], [1320, 700], [1750, 1440]],
                "OPPOSITE": [[1750, 1440], [1320, 700], [2500, 700], [2560, 1440]]
            },
            "monitored_regions": ["LANE_1", "LANE_2"],
            "speed_thresholds": {
                "stationary_max_speed": 80.0,
                "wrong_way_min_speed": 30.0
            },
            "confidence_threshold": 0.15,
            "imgsz": 1280,
            "draw_lanes": False,
            "speed_window_size": 5
        }

    def setup_session(
        self, 
        output_dir: Path, 
        video_id: str, 
        config_path: Path, 
        vehicle_model: str = "yolov8n.pt", 
        tracker_module: str = "bytetrack"
    ) -> None:
        """
        Loads configurations and configures dynamic custom models, while resetting
        internal data maps to prevent cross-contamination across multiple videos.
        """
        self.video_id = video_id

        # Isolated tracking images storage path partition folder
        self.violation_dir = Path(output_dir) / "violations"
        self.violation_dir.mkdir(parents=True, exist_ok=True)

        # Flush tracking metrics completely between video segments
        self.track_history.clear()
        self.wrong_way_frames.clear()
        self.flagged_wrong_way.clear()
        self.saved_violations.clear()
        self.violations_log.clear()
        self.prev_gray = None
        self.prev_pts = None
        self.current_v_ego = 300.0
        self.frame_count = 0

        # Load specific lane calibration outputs
        self.config = self.load_config(config_path)
        
        # Load bounding lane coordinates
        self.polygons = {}
        for name, pts in self.config.get("polygons", {}).items():
            self.polygons[name] = np.array(pts, np.int32)

        # Precompute Perspective Transform Homography parameter mappings
        ipm_config = self.config.get("ipm", {})
        src = np.float32(ipm_config.get("src", [[920, 700], [1100, 700], [350, 1440], [900, 1440]]))
        dst = np.float32(ipm_config.get("dst", [[100, 0], [200, 0], [100, 1000], [200, 1000]]))
        self.H = cv2.getPerspectiveTransform(src, dst)

        # Dynamic vehicle weights resolution 
        target_model = Path(vehicle_model)
        if not target_model.is_absolute() and not target_model.exists():
            local = self.models_dir / vehicle_model
            target_model = local if local.exists() else self.default_model

        target_model_str = str(target_model)
        try:
            print(f"######## USING MODEL ######## {target_model_str}", flush=True)
            self.model = YOLO(target_model_str)
        except Exception as e:
            raise RuntimeError(f"Unable to load Wrong Way processing model {target_model_str}: {e}")

        # Configure customized track settings properties
        self.tracker = tracker_module
        if not self.tracker.endswith(".yaml"):
            self.tracker += ".yaml"

        print(f"[WRONG WAY] Processing workspace setup complete. Tracker: {self.tracker}", flush=True)

    def to_bev(self, x: float, y: float) -> Tuple[float, float]:
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped[0][0][0], warped[0][0][1]

    def get_region(self, cx: int, cy: int) -> str:
        for name, pts in self.polygons.items():
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                return name
        return "UNKNOWN"

    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def process_frame(self, context: Dict[str, Any]) -> cv2.Mat:
        """
        Evaluates kinematic spatial movements on the pre-filtered contextual frames.
        Mutates the incoming visual frame reference context layers directly.
        """
        if self.model is None or self.violation_dir is None or self.video_id is None:
            raise RuntimeError("WrongWayDetector runtime state uninitialized. Call setup_session() first.")

        # Fixed: Running frame execution tracking incremented safely
        self.frame_count += 1

        # Unpack pure frame parameters from the strict FrameContext bounds
        frame = context["frame"]
        timestamp = context["timestamp_seconds"]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgsz = self.config.get("imgsz", 1280)
        conf = self.config.get("confidence_threshold", 0.15)
        
        # Execute non-blocking custom persist tracking on the live canvas pipeline
        results = self.model.track(
            frame, persist=True, tracker=self.tracker, conf=conf, imgsz=imgsz, verbose=False
        )
        
        vehicle_boxes = []
        frame_vehicles = []
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                if class_name not in ["car", "truck", "motorcycle", "bus"]:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                
                vehicle_boxes.append((x1, y1, x2, y2))
                confidence = float(box.conf[0])
                frame_vehicles.append({
                    "id": track_id,
                    "class": class_name,
                    "conf": confidence,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, (y1 + y2) // 2),
                    "bottom_center": (cx, y2)
                })
                
        # Calculate Ego-Motion matrices via Lucas-Kanade Optical Flow (Fixed: Retained absolute 25.0 design parameter boundary)
        if self.prev_gray is not None:
            shoulder_mask = np.zeros_like(self.prev_gray)
            if "SHOULDER" in self.polygons:
                cv2.fillPoly(shoulder_mask, [self.polygons["SHOULDER"]], 255)
                
            for x1, y1, x2, y2 in vehicle_boxes:
                cv2.rectangle(shoulder_mask, (x1, y1), (x2, y2), 0, -1)
                
            if self.prev_pts is None or len(self.prev_pts) < 15:
                self.prev_pts = cv2.goodFeaturesToTrack(
                    self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=shoulder_mask
                )
                
            if self.prev_pts is not None and len(self.prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_pts, None, **self.lk_params
                )
                good_prev = self.prev_pts[status == 1]
                good_next = next_pts[status == 1]
                
                displacements = []
                for p_prev, p_next in zip(good_prev, good_next):
                    bx_prev, by_prev = self.to_bev(p_prev[0], p_prev[1])
                    bx_next, by_next = self.to_bev(p_next[0], p_next[1])
                    displacements.append(by_next - by_prev)
                    
                if displacements:
                    self.current_v_ego = np.median(displacements) * self.fps
                    
                self.prev_pts = good_next.reshape(-1, 1, 2)
            else:
                self.prev_pts = None
        
        # Track Propagation evaluation structures
        monitored_regions = self.config.get("monitored_regions", ["LANE_1", "LANE_2"])
        speed_thresholds = self.config.get("speed_thresholds", {})
        wrong_way_min_speed = speed_thresholds.get("wrong_way_min_speed", 30.0)
        speed_window = self.config.get("speed_window_size", 5)
        
        for veh in frame_vehicles:
            tid = veh["id"]
            if tid in self.flagged_wrong_way:
                continue
            for other_veh in frame_vehicles:
                other_tid = other_veh["id"]
                if other_tid == tid or other_tid not in self.flagged_wrong_way:
                    continue
                if self.calculate_iou(veh["box"], other_veh["box"]) > 0.25:
                    self.flagged_wrong_way.add(tid)
                    self.wrong_way_frames[tid] = 10
                    break
        
        # Speed mapping and directional processing operations
        for veh in frame_vehicles:
            tid = veh["id"]
            class_name = veh["class"]
            confidence = veh["conf"]
            x1, y1, x2, y2 = veh["box"]
            bc_x, bc_y = veh["bottom_center"]
            
            region = self.get_region(bc_x, bc_y)
            if region not in monitored_regions:
                continue
            bev_x, bev_y = self.to_bev(bc_x, bc_y)
            
            if tid not in self.track_history:
                self.track_history[tid] = {"bev_positions": [], "img_positions": [], "timestamps": []}
                
            self.track_history[tid]["bev_positions"].append((bev_x, bev_y))
            self.track_history[tid]["img_positions"].append((bc_x, bc_y))
            self.track_history[tid]["timestamps"].append(timestamp)
            
            if len(self.track_history[tid]["bev_positions"]) > 30:
                self.track_history[tid]["bev_positions"].pop(0)
                self.track_history[tid]["img_positions"].pop(0)
                self.track_history[tid]["timestamps"].pop(0)
                
            history = self.track_history[tid]
            
            if tid in self.flagged_wrong_way:
                img_hist = history["img_positions"]
                for i in range(1, len(img_hist)):
                    cv2.line(frame, img_hist[i-1], img_hist[i], (0, 0, 255), 2)
                
            is_wrong_way = False
            
            if tid in self.flagged_wrong_way:
                is_wrong_way = True
            elif len(history["bev_positions"]) >= speed_window:
                bev_start = history["bev_positions"][-speed_window]
                bev_end = history["bev_positions"][-1]
                img_start = history["img_positions"][-speed_window]
                img_end = history["img_positions"][-1]
                t_start = history["timestamps"][-speed_window]
                t_end = history["timestamps"][-1]
                
                dt = t_end - t_start
                if dt > 0:
                    movement_pixels = np.sqrt((img_end[0] - img_start[0])**2 + (img_end[1] - img_start[1])**2)
                    if movement_pixels >= 60:
                        v_rel = (bev_end[1] - bev_start[1]) / dt
                        v_abs = self.current_v_ego - v_rel
                        dy_speed = (img_end[1] - img_start[1]) / dt
                        
                        if abs(v_abs) >= 30:
                            if v_abs < -wrong_way_min_speed and dy_speed > 35.0:
                                is_wrong_way = True
                                
            if len(self.track_history[tid]["img_positions"]) < 8:
                continue
                
            if region in monitored_regions and is_wrong_way:
                self.wrong_way_frames[tid] = self.wrong_way_frames.get(tid, 0) + 1
            else:
                if tid not in self.flagged_wrong_way:
                    self.wrong_way_frames[tid] = max(0, self.wrong_way_frames.get(tid, 0) - 1)
            
            # Violation overlay processing bounds
            if self.wrong_way_frames.get(tid, 0) >= 3 or tid in self.flagged_wrong_way:
                self.flagged_wrong_way.add(tid)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame, f"WRONG WAY {class_name.upper()} {confidence:.2f} ID:{tid}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                
                if tid not in self.saved_violations:
                    plate_number = "UNKNOWN"
                    try:
                        vehicle_crop = frame[y1:y2, x1:x2]
                        plate_results = self.plate_reader.read_plate(vehicle_crop, tid)
                        if plate_results:
                            candidate = plate_results[0].get("plate", "UNKNOWN")
                            if candidate and candidate.strip():
                                plate_number = candidate
                    except Exception as e:
                        print(f"[OCR ERROR] {e}", flush=True)

                    image_name = f"{self.video_id}_{int(timestamp)}s_wrong_way_{plate_number}_{tid}.jpg"
                    violation_path = self.violation_dir / image_name
                    cv2.imwrite(str(violation_path), frame)

                    object_key = f"videos/{self.video_id}/violations/{image_name}"
                    uploaded = upload_file(settings.minio_images_bucket, object_key, violation_path, "image/jpeg")

                    self.violations_log.append({
                        "track_id": tid,
                        "class": class_name,
                        "plate_number": plate_number,
                        "timestamp_seconds": timestamp,
                        "violation_type": "wrong_way",
                        "confidence": confidence,
                        "image_url": uploaded.object_url
                    })

                    self.saved_violations.add(tid)
                    violation_path.unlink(missing_ok=True)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{class_name} ID:{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        
        # Draw static lanes boundary coordinates overlays
        if self.config.get("draw_lanes", False):
            for name, pts in self.polygons.items():
                color = (0, 255, 255)
                if name in monitored_regions:
                    color = (0, 255, 0)
                elif name == "OPPOSITE":
                    color = (0, 0, 255)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                
        self.prev_gray = gray.copy()
        return frame

    def finish(self) -> Dict[str, Any]:
        print("[WRONG WAY DETECTOR FINISHED]", flush=True)
        return {
            "status": "completed", 
            "frame_count": self.frame_count, 
            "violations": self.violations_log
        }