import math
from pathlib import Path
from backend.app.core.config import get_settings
from backend.app.services.storage import upload_file

settings = get_settings()
from backend.app.services.processors.plate_reader import PlateReader
import cv2
from ultralytics import YOLO
import numpy as np
import json
import os
import math

class LaneChangeDetector:
    # Initializes detector, loads configuration, instantiates YOLO, and sets up Homography matrix for BEV.
    def __init__(self):
        print("[LANE CHANGE INIT]", flush=True)
        self.model = YOLO("yolov8n.pt")
        # Load config if available, otherwise use defaults
        #self.config = self.load_config()
        #self.model = YOLO(self.config.get("yolo_model", "yolov8n.pt"))
        self.plate_reader = PlateReader()
        self.flagged_lane_change = set()
        self.center_history = {}
        
        # Load polygons from config
        #self.polygons = {}
        #for name, pts in self.config.get("polygons", {}).items():
            #self.polygons[name] = np.array(pts, np.int32)
            
        # Homography setup for Bird's-Eye View (BEV)
        #ipm_config = self.config.get("ipm", {})
        #src = np.float32(ipm_config.get("src", [[920, 700], [1100, 700], [350, 1440], [900, 1440]]))
        #dst = np.float32(ipm_config.get("dst", [[100, 0], [200, 0], [100, 1000], [200, 1000]]))
        #self.H = cv2.getPerspectiveTransform(src, dst)
        
    # Loads configuration parameters from config.json with a hardcoded fallback config.
    def load_config(self, config_path):
        #config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    print(f"[CONFIG] Loading configuration from {config_path}", flush=True)
                    return json.load(f)
            except Exception as e:
                print(f"[CONFIG] Error loading config.json: {e}. Using defaults.", flush=True)
        else:
            print("[CONFIG] config.json not found. Using defaults.", flush=True)
            
        # Default fallback config
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
            "time_range": {
                "start_msec": 0,
                "end_msec": 59000
            },
            "tracker": "botsort.yaml",
            "confidence_threshold": 0.15,
            "yolo_model": "yolov8n.pt",
            "imgsz": 1280,
            "draw_lanes": False,
            "speed_window_size": 5
        }
        
    # Projects perspective coordinates (x, y) to Bird's-Eye View (BEV) coordinates using the Homography matrix.
    def to_bev(self, x, y):
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped[0][0][0], warped[0][0][1]
        
    # Determines which lane polygon (e.g., LANE_1, LANE_2) contains the target point (cx, cy).
    def get_region(self, cx, cy):
        for name, pts in self.polygons.items():
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                return name
        return "UNKNOWN"
        
    # Calculates the Intersection-over-Union (IoU) between two bounding boxes (used to propagate sticky wrong-way flags).
    def calculate_iou(self, box1, box2):
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
        
        return inter_area / union_area if union_area > 0 else 0
        
    # Processes input video frame-by-frame: tracks vehicles, calculates ego-motion, computes vehicle speed, and flags wrong-way violators.
    def run(self, input_path, output_dir, video_id):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "config.json"

        self.config = self.load_config(config_path)
        # Load polygons from config
        self.polygons = {}
        for name, pts in self.config.get("polygons", {}).items():
            self.polygons[name] = np.array(pts, np.int32)

        # Homography setup for Bird's-Eye View (BEV)
        ipm_config = self.config.get("ipm", {})

        src = np.float32(ipm_config.get("src",[[920, 700], [1100, 700], [350, 1440], [900, 1440]]))

        dst = np.float32(ipm_config.get("dst",[[100, 0], [200, 0], [100, 1000], [200, 1000]]))

        self.H = cv2.getPerspectiveTransform(src, dst)

        print(f"[CONFIG] Loaded: {config_path}", flush=True)
        print("[POLYGONS]", self.polygons.keys(), flush=True)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {input_path}", flush=True)
            return input_path, {"status": "failed", "error": "file_not_found"}
            
        time_range = self.config.get("time_range", {})
        start_msec = time_range.get("start_msec", 0)
        end_msec = time_range.get("end_msec", None)
        #end_msec = None
        
        if start_msec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"WIDTH={width}, HEIGHT={height}, FPS={fps}", flush=True)
        
        output_video = output_dir / f"{video_id}_processed.mp4"
        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        print(f"[VIDEO] Saving to {output_video}", flush=True)
        
        # State tracking structures
        lane_history = {}
        flagged_lane_change = set()
        
        
        # Lucas-Kanade optical flow parameters
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        prev_gray = None
        prev_pts = None
        
        show_gui = False
        try:
            cv2.namedWindow("Wrong Way Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Wrong Way Debug", 1280, 720)
            show_gui = True
        except Exception:
            print("[GUI] Running headlessly, GUI window disabled.", flush=True)
            
        frame_count = 0
        speed_window = self.config.get("speed_window_size", 5)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(f"[TIME] {current_time/1000:.2f}s", flush=True)
            
            if end_msec is not None and current_time > end_msec:
                print(f"[STOP] Reached end timestamp {end_msec/1000:.2f}s", flush=True)
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect and track vehicles using YOLO tracker (uses BoT-SORT to prevent ID switching).
            imgsz = self.config.get("imgsz", 1280)
            conf = self.config.get("confidence_threshold", 0.15)
            tracker = self.config.get("tracker", "botsort.yaml")
            
            results = self.model.track(frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz, verbose=False)
            
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
                    cy = (y1 + y2) // 2
                    
                    vehicle_boxes.append((x1, y1, x2, y2))
                    confidence = float(box.conf[0])
                    frame_vehicles.append({
                        "id": track_id,
                        "class": class_name,
                        "conf": confidence,
                        "box": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "bottom_center": (cx, y2)
                    })
                
            # 2. Dynamic Ego-Motion Estimation: Calculates forward camera speed using optical flow of static features on the shoulder (excluding moving vehicle boxes).
            if prev_gray is not None:
                # Create mask for shoulder region, blacking out moving vehicles
                shoulder_mask = np.zeros_like(prev_gray)
                if "SHOULDER" in self.polygons:
                    cv2.fillPoly(shoulder_mask, [self.polygons["SHOULDER"]], 255)
                    
                for x1, y1, x2, y2 in vehicle_boxes:
                    cv2.rectangle(shoulder_mask, (x1, y1), (x2, y2), 0, -1)
                    
                if prev_pts is None or len(prev_pts) < 15:
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=shoulder_mask)
                    
                if prev_pts is not None and len(prev_pts) > 0:
                    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
                    good_prev = prev_pts[status == 1]
                    good_next = next_pts[status == 1]
                    
                    displacements = []
                    for p_prev, p_next in zip(good_prev, good_next):
                        bx_prev, by_prev = self.to_bev(p_prev[0], p_prev[1])
                        bx_next, by_next = self.to_bev(p_next[0], p_next[1])
                        dy = by_next - by_prev
                        displacements.append(dy)
                        
                    if displacements:
                        median_dy = np.median(displacements)
                        
                    prev_pts = good_next.reshape(-1, 1, 2)
                else:
                    prev_pts = None
            
            
            # 4. Track processing and velocity estimation: Computes absolute speed relative to the road using a rolling-window of the last 5 frames.
            for veh in frame_vehicles:
                tid = veh["id"]
                class_name = veh["class"]
                confidence = veh["conf"]
                x1, y1, x2, y2 = veh["box"]
                cx, cy = veh["center"]
                bc_x, bc_y = veh["bottom_center"]
                
                region = self.get_region(bc_x, bc_y)
                print(f"[LANE] ID:{tid} REGION:{region}",flush=True)
                # ---------- REGION HISTORY ----------
                if tid not in lane_history:
                    lane_history[tid] = []

                lane_history[tid].append(region)

                if len(lane_history[tid]) > 10:
                    lane_history[tid].pop(0)
                history = lane_history[tid]
                # ---------- CENTER HISTORY ----------
                if tid not in self.center_history:
                    self.center_history[tid] = []

                self.center_history[tid].append((bc_x, bc_y))

                if len(self.center_history[tid]) > 10:
                    self.center_history[tid].pop(0)
                # ---------- LANE CHANGE CHECK ----------
                if len(history) >= 10:
                    print(
                        f"[HISTORY] ID:{tid} "
                        f"START:{history[0]} "
                        f"END:{history[-1]}",
                        flush=True
                    )
                    old_lane = history[0]
                    new_lane = history[-1]
                    # movement filter
                    start_x, start_y = self.center_history[tid][0]
                    end_x, end_y = self.center_history[tid][-1]
                    movement = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
                    print(f"[MOVE] ID:{tid} MOVEMENT:{movement:.2f}", flush=True)

                    if movement < 5:
                        continue  # Skip if the vehicle hasn't moved enough to consider a lane change

                    if (old_lane != new_lane and old_lane in ["LANE_1", "LANE_2"] and new_lane in ["LANE_1", "LANE_2"]):

                        if tid not in self.flagged_lane_change:

                            self.flagged_lane_change.add(tid)

                            print(
                                f"[LANE CHANGE] ID:{tid} "
                                f"{old_lane} -> {new_lane}",
                                flush=True
                            )

                # 5. Violation flagging: Marks vehicle as a wrong-way violator if wrong-way counter exceeds threshold or if previously flagged.
                if tid in self.flagged_lane_change:

                    cv2.rectangle(frame,(x1, y1),(x2, y2),(255, 0, 0),3)

                    cv2.putText(frame,f"LANE CHANGE ID:{tid}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2)

                else:

                    cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

                    cv2.putText(frame,f"{class_name} ID:{tid}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
                
            # 6. Visualization overlay: Draws the configured lane polygons on the frame if configured in config.json.
            monitored_regions = self.config.get("monitored_regions",["LANE_1", "LANE_2"])
            if self.config.get("draw_lanes", False):
                for name, pts in self.polygons.items():
                    color = (0, 255, 255)
                    if name in monitored_regions:
                        color = (0, 255, 0)
                    elif name == "OPPOSITE":
                        color = (0, 0, 255)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                
            out.write(frame)
            
            if show_gui:
                display_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("Wrong Way Debug", display_frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                    
            prev_gray = gray.copy()
            frame_count += 1
            
        cap.release()
        out.release()
        if show_gui:
            cv2.destroyAllWindows()
            
        print("[LANE CHANGE DETECTOR FINISHED]", flush=True)
        return output_video, {
            "status": "completed",
            "frame_count": frame_count,
            #"violations": violations_log
        }

if __name__ == "__main__":
    detector = LaneChangeDetector()
    video_path = r"C:\videoset1_videos_part1\lane_change\20211125134600_0060.mp4"
    output_dir = r"C:\videoset1_videos_part1\lane_change\output"
    
    # Run the detector
    detector.run(video_path, output_dir, "lane_change_test")