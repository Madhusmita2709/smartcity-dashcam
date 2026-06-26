from pathlib import Path
from collections import deque
import cv2
from ultralytics import YOLO
import numpy as np
import json
import os
import csv

class ProductionMovingDetector:
    def __init__(self, log_filename="telemetry_log.csv"):
        print("[PRODUCTION MOVING DETECTOR INIT]", flush=True)
        self.track_history = {}
        self.config = self.load_config()
        print("[CONFIG POLYGONS]",self.config.get("polygons", {}).keys(),flush=True)
        self.model = YOLO(self.config.get("yolo_model", "yolov8n.pt"))
        
        # Homography Setup 
        # Load calibrated IPM points from wrong_way config
        scale_x = 1280 / 2560.0
        scale_y = 720 / 1440.0
        src = np.array(self.config["ipm"]["src"],dtype=np.float32)
        src[:, 0] *= scale_x
        src[:, 1] *= scale_y
        self.src = src
        self.dst = np.float32([[100, 0],[400, 0],[100, 1000],[400, 1000]])
        print("[IPM SRC]")
        print(self.src)
        self.H = cv2.getPerspectiveTransform(self.src,self.dst)
        
        # Sparse Optical Flow Parameters
        self.feature_params = dict(maxCorners=150, qualityLevel=0.01, minDistance=4, blockSize=3)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.old_gray = None
        self.p0 = None  # Persistent keypoints array
        self.ego_buffer = deque(maxlen=30)
        
        # Setup Telemetry CSV Exporter
        self.log_filename = log_filename
        self.init_csv_log()

    def load_config(self):
        config_path = r"media\295\wrong_way\config.json"
        with open(config_path, "r") as f:
            return json.load(f)

    def init_csv_log(self):
        """Creates or clears the telemetry file and writes header columns."""
        with open(self.log_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "EgoScore", "EgoDX_BEV", "EgoDY_BEV"])
        print(f"[LOGGING] Telemetry initialized: {self.log_filename}", flush=True)

    def log_telemetry(self, frame_no, score, dx, dy):
        """Appends raw ego data vectors to the baseline CSV sheet for future scaling analysis."""
        with open(self.log_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([int(frame_no), round(score, 4), round(dx, 4), round(dy, 4)])

    def to_bev(self, x, y):
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped[0][0][0], warped[0][0][1]

    def to_bev_array(self, pts_array):
        if pts_array is None or len(pts_array) == 0:
            return np.empty((0, 2), dtype=np.float32)
        reshaped = pts_array.reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(reshaped, self.H)
        return warped.reshape(-1, 2)
    
    def build_lane_mask(self, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        polygons = self.config["polygons"]
        scale_x = w / 2560.0
        scale_y = h / 1440.0
        for lane_name in ["LANE_1", "LANE_2"]:
            pts = np.array(polygons[lane_name],dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask,[pts],255)

        return mask

    def run_video(self, video_path):
        print("[RUN_VIDEO STARTED]")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, first_frame = cap.read()
        if not ret: return
        first_frame = cv2.resize(first_frame, (1280, 720))
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        #h, w = self.old_gray.shape
        #self.base_roi_mask = np.zeros_like(self.old_gray)
        
        # Horizon Cutoff: Limit features below 65% road height
        #roi_corners = np.array([[(int(w*0.15), int(h*0.50)),(int(w*0.85), int(h*0.50)),(int(w*0.98), h),(int(w*0.02), h)]], dtype=np.int32)
        #cv2.fillPoly(self.base_roi_mask, roi_corners, 255)
        self.base_roi_mask = self.build_lane_mask(self.old_gray.shape)
        print("[MASK PIXELS]",cv2.countNonZero(self.base_roi_mask),flush=True)
        cv2.imshow("Road Mask",self.base_roi_mask)
        #pts = np.array(self.config["polygons"]["LANE_1"])
        # #print("[LANE_1 MAX Y]",pts[:,1].max(),flush=True)
    
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("[VIDEO ENDED]",flush=True)
                break
            
            frame = cv2.resize(frame, (1280, 720))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
            dt = 1.0 / video_fps 

            results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=0.15, verbose=False)
            dynamic_road_mask = self.base_roi_mask.copy()
            
            active_boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.id is None: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    active_boxes.append((int(box.id[0]), x1, y1, x2, y2))
                    cv2.rectangle(dynamic_road_mask, (x1, y1), (x2, y2), 0, -1)
                cv2.imshow("Road Mask", dynamic_road_mask)
                print(
                        f"[MASK] NONZERO={cv2.countNonZero(dynamic_road_mask)}",
                        flush=True
                    )
    

            # Persistent Keypoint Generation Block
            if self.p0 is None or len(self.p0) < 20:
                self.p0 = cv2.goodFeaturesToTrack(self.old_gray,mask=dynamic_road_mask,**self.feature_params)

            if self.p0 is None:
                print("[FLOW] P0=None", flush=True)
            else:
                print(f"[FLOW] P0={len(self.p0)}", flush=True)
            ego_dx_bev, ego_dy_bev = 0.0, 0.0
            ego_motion_score = 0.0

            if self.p0 is not None and len(self.p0) >= 5:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
                if p1 is None:
                    print("[FLOW] P1=None", flush=True)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = self.p0[st == 1]
                    print(f"[FLOW] GOOD={len(good_new)}",flush=True)
                    for pt in good_old:
                        x, y = map(int, pt.ravel())
                        cv2.circle(frame, (x, y), 3, (0,0,255), -1)
                    img_disp = good_new - good_old

                    print(
                        f"[IMG] DX={np.mean(img_disp[:,0]):.4f} "
                        f"DY={np.mean(img_disp[:,1]):.4f}",
                        flush=True
                    )
                    
                    if len(good_new) >= 5:
                        bev_old = self.to_bev_array(good_old)
                        bev_new = self.to_bev_array(good_new)
                        displacements = bev_new - bev_old
                        print(
                            f"[BEV] DX={np.mean(displacements[:,0]):.4f} "
                            f"DY={np.mean(displacements[:,1]):.4f}",
                            flush=True
                        )
                        print(
                            f"[STARTS] "
                            f"mean_dx={np.mean(displacements[:,0]):.4f} "
                            f"median_dx={np.median(displacements[:,0]):.4f} "
                            f"mean_dy={np.mean(displacements[:,1]):.4f} "
                            f"median_dy={np.median(displacements[:,1]):.4f}",
                            flush=True
                        )
                        ego_dx_bev = np.mean(displacements[:, 0])
                        ego_dy_bev = np.mean(displacements[:, 1])
                        ego_motion_score = np.sqrt(ego_dx_bev**2 + ego_dy_bev**2) / dt
                        self.ego_buffer.append(ego_motion_score)
                        ego_motion_score = np.mean(self.ego_buffer)
                    
                        print(
                            f"[TEST_MEAN] DX={ego_dx_bev:.4f} "
                            f"DY={ego_dy_bev:.4f} "
                            f"SCORE={ego_motion_score:.2f}",
                            flush=True
                        )
                        print(f"[EGO] SCORE={ego_motion_score:.2f}",flush=True)
                        print(
                            f"[EGO] DX={ego_dx_bev:.2f} "
                            f"DY={ego_dy_bev:.2f} "
                            f"SCORE={ego_motion_score:.2f}",
                            flush=True
                        )
                        # Calibration log
                        print(f"[CALIB] ego={ego_motion_score:.2f}",flush=True)
                        
                        # VALIDATION 1: Draw Tracking Motion Vector Tails on Main Live Frame
                        for old_pt, new_pt in zip(good_old, good_new):
                            ox, oy = map(int, old_pt.ravel())
                            nx, ny = map(int, new_pt.ravel())
                            cv2.line(frame, (ox, oy), (nx, ny), (0, 255, 255), 1)
                            cv2.circle(frame, (nx, ny), 2, (0, 255, 0), -1)
                        
                        self.p0 = good_new.reshape(-1, 1, 2)
                    else:
                        self.p0 = None
                else:
                    self.p0 = None

            # Stream data out to active spreadsheet loop row
            self.log_telemetry(frame_no, ego_motion_score, ego_dx_bev, ego_dy_bev)

            # Initialize clear Bird's-Eye View Grid Canvas
            bev_canvas = np.zeros((1000, 400, 3), dtype=np.uint8)

            for track_id, x1, y1, x2, y2 in active_boxes:
                bc_x = (x1 + x2) // 2
                bc_y = y2 
                bev_x, bev_y = self.to_bev(bc_x, bc_y)

                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((frame_no, bev_x, bev_y))
                
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                history = self.track_history[track_id]
                actual_speed_score = 0.0

                # Temporal Lookback window calculation 
                if len(history) >= 10:
                    f1, bx1, by1 = history[0]
                    f2, bx2, by2 = history[-1]
                    frames_elapsed = f2 - f1
                    
                    if frames_elapsed > 0:
                        total_dt = frames_elapsed * dt
                        target_rel_dx = (bx2 - bx1) / total_dt
                        target_rel_dy = (by2 - by1) / total_dt

                        actual_dx_bev = (ego_dx_bev / dt) + target_rel_dx
                        actual_dy_bev = (ego_dy_bev / dt) + target_rel_dy
                        actual_speed_score = np.sqrt(actual_dx_bev**2 + actual_dy_bev**2)

                # VALIDATION 2: Render Tail Trajectory Paths onto the BEV Canvas
                for i in range(1, len(history)):
                    _, x_prev, y_prev = history[i-1]
                    _, x_curr, y_curr = history[i]
                    cv2.line(bev_canvas, (int(x_prev), int(y_prev)), (int(x_curr), int(y_curr)), (255, 255, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} Comb_Score:{int(actual_speed_score)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
                
                if 0 <= bev_x < 400 and 0 <= bev_y < 1000:
                    cv2.circle(bev_canvas, (int(bev_x), int(bev_y)), 5, (0, 255, 0), -1)

            # Diagnostics telemetry readout onto viewing frames
            cv2.putText(frame,f"RAW EGO BEV SCORE: {ego_motion_score:.1f} units/sec",(30, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2)
            
            
            cv2.imshow("Moving Frame Tracking Output", frame)
            cv2.imshow("Ego Adjusted BEV Grid", bev_canvas)
            
            self.old_gray = frame_gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ProductionMovingDetector()
    video_path = r"C:\videoset1_videos_part1\20220824155045_0060speed_highway.mp4"
    detector.run_video(video_path)