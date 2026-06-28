from pathlib import Path
import cv2
import numpy as np
import json
from ultralytics import YOLO

METERS_PER_PIXEL = 0.00875

class EgoSpeedReader:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        print("[EGO SPEED INIT]", flush=True)
        self.frame_count = 0
        self.prev_gray = None
        self.prev_pts = None

        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )
        config_path = Path(r"C:\Users\madhu\Videos\config.json")
        config = self.load_config(config_path)
        ipm = config["ipm"]

        src = np.float32(ipm["src"])
        dst = np.float32(ipm["dst"])

        self.H = cv2.getPerspectiveTransform(src, dst)

    def load_config(self, config_path):

        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)

        raise FileNotFoundError(
            f"Config not found : {config_path}"
        )
        
    def run(self, input_video):

        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"FPS = {fps}")

        while True:

            ret, frame = cap.read()
            self.frame_count += 1
            
            if not ret:
                break
            # Convert frame to Bird's Eye View
            bev_frame = cv2.warpPerspective(frame,self.H,(600, 1000))

            h, w = frame.shape[:2]

            # Use only road area
            #roi = frame[int(h * 0.45):, :]

            gray = cv2.cvtColor(bev_frame,cv2.COLOR_BGR2GRAY)
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            results = self.model(frame, verbose=False)
            for r in results:

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    cls = int(box.cls[0])
                    name = self.model.names[cls]

                    if name not in ["car", "truck", "bus", "motorcycle"]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    p1 = cv2.perspectiveTransform(
                    np.array([[[x1, y1]]], dtype=np.float32),
                    self.H
                    )[0][0]

                    p2 = cv2.perspectiveTransform(
                    np.array([[[x2, y2]]], dtype=np.float32),
                    self.H
                    )[0][0]

                    cv2.rectangle(
                    mask,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    0,
                    -1
                    )
            cv2.imshow("Mask", mask)
            if self.prev_gray is None:

                self.prev_gray = gray

                self.prev_pts = cv2.goodFeaturesToTrack(
                    gray,
                    mask=mask,
                    **self.feature_params
                )

                continue

            if self.prev_pts is None or len(self.prev_pts) < 50 or self.frame_count % 15 == 0:

                self.prev_pts = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    mask=mask,
                    **self.feature_params
                )

            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_pts,
                None,
                **self.lk_params
            )

            if next_pts is not None:

                good_new = next_pts[status == 1]
                good_old = self.prev_pts[status == 1]
                print("Tracked points =", len(good_new))

                dx_list = []
                dy_list = []

                for new, old in zip(good_new, good_old):

                    a, b = new.ravel()
                    c, d = old.ravel()

                    dx = a - c
                    dy = b - d

                    motion = np.sqrt(dx*dx + dy*dy)
                    print(f"Motion = {motion:.2f}")
                    color = (0,255,0)
                    if motion > 5:
                        color = (0,0,255)

                    cv2.circle(
                    bev_frame,
                    (int(a), int(b)),
                    2,
                    color,
                    -1
                    )

                    cv2.line(
                    bev_frame,
                    (int(c), int(d)),
                    (int(a), int(b)),
                    color,
                    1
                    )

                    if motion < 15:      # ignore moving vehicles
                        dx_list.append(dx)
                        dy_list.append(dy)

                print("Number of tracked points:", len(dx_list))

                if len(dx_list):
                    print(
                    "DX min/max:",
                    np.min(dx_list),
                    np.max(dx_list)
                    )
                    print(
                    "DY min/max:",
                    np.min(dy_list),
                    np.max(dy_list)
                    )
                    
                    avg_dx = np.mean(dx_list)
                    avg_dy = np.mean(dy_list)

                    flow_pixels = np.sqrt(avg_dx**2 + avg_dy**2)
                    flow_pixels_per_second = flow_pixels * fps
                    ego_speed_mps = flow_pixels_per_second * METERS_PER_PIXEL
                    ego_speed_kmh = ego_speed_mps * 3.6
                    print(
                    f"AVG_DX={avg_dx:.2f}  "
                    f"AVG_DY={avg_dy:.2f}  "
                    f"FLOW={flow_pixels:.2f}px/frame  "
                    f"EGO={ego_speed_kmh:.2f} km/h"
                    )
                    
                    cv2.putText(
                        bev_frame,
                        f"DX:{avg_dx:.2f} DY:{avg_dy:.2f}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,255),
                        2
                    )

                self.prev_gray = gray.copy()
                if self.frame_count % 15 == 0:
                # Every 15 frames detect fresh road features
                    self.prev_pts = cv2.goodFeaturesToTrack(
                        gray,
                        mask=mask,
                        **self.feature_params
                    )
                else:
                # Continue tracking existing points
                    self.prev_pts = good_new.reshape(-1,1,2)
            
            cv2.imshow("BEV Optical Flow",cv2.resize(bev_frame, (500, 800)))
            #cv2.imshow("ROI", bev_frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = EgoSpeedReader()

    detector.run(
    input_video=r"C:\videoset1_videos_part1\20220824155045_0060speed_highway.mp4"
    )