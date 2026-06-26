import argparse
import json
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("backend/app/services/models/yolov8n.pt")
#we must flatten the road.
SRC = np.float32([
    [1010, 520],   # Top-left  (left lane line)
    [1510, 520],   # Top-right (right lane line)
    [650, 1430],   # Bottom-left
    [1880, 1430]   # Bottom-right
])

DST = np.float32([
    [0,0],
    [400,0],
    [0,1000],
    [400,1000]
])

H = cv2.getPerspectiveTransform(SRC, DST)
# This module reads ego motion from a video using dense optical flow.
# The configured polygons define regions of interest (ROI) where motion
# is measured. The average vertical flow (dy) inside those polygons is
# computed and displayed on the video output.
# Temporary local debug paths. Uncomment and set the video path for
# quick testing when you do not want to pass --video explicitly.
# TEMP_VIDEO_PATH = Path(r"C:\videoset1_videos_part1\lane_change\20211125134600_0060.mp4")
TEMP_VIDEO_PATH: Path | None = Path(r"C:\videoset1_videos_part1\20220824155045_0060speed_highway.mp4")

# The config file is loaded automatically from this module directory.
DEFAULT_CONFIG_PATH = Path( r"C:\Users\madhu\Videos\config.json" )
# ============================================
# Ego speed calibration
# ============================================
LANE_WIDTH_METERS = 3.5
CALIBRATION_FACTOR = 1.0    # Change after calibration

def load_config(config_path: Path) -> dict:
    # Load the JSON configuration from the given path.
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

    
def build_flow_mask(frame_shape: tuple[int, int], polygons: dict) -> np.ndarray:
    # Create a binary mask where the polygon ROIs are white.
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for pts in polygons.values():
        poly = np.array(pts, dtype=np.int32)
        if poly.size > 0:
            cv2.fillPoly(mask, [poly], 255)
    print("mask shape:", mask.shape)
    print("mask min:", mask.min())
    print("mask max:", mask.max())
    print("mask nonzero:", np.count_nonzero(mask))
    print("frame_shape:", frame_shape)

    for name, pts in polygons.items():
        print(name, pts)
    return mask


def compute_average_motion(flow, mask, threshold=0.5):

    magnitude = np.linalg.norm(flow, axis=2)

    masked_mag = magnitude[mask > 0]

    if masked_mag.size == 0:
        return 0.0

    # Ignore tiny motions
    masked_mag = masked_mag[masked_mag > threshold]

    if masked_mag.size == 0:
        return 0.0

    # -------- Robust Statistics --------

    median = np.median(masked_mag)

    mad = np.median(np.abs(masked_mag - median))

    if mad > 0:

        filtered = masked_mag[
            np.abs(masked_mag - median) < 3 * mad
        ]

    else:

        filtered = masked_mag

    if filtered.size == 0:
        return 0.0

    return float(np.mean(filtered))

def run_video(video_path: Path, config_path: Path, show_video: bool = True) -> None:
    # Open the video and compute optical flow between consecutive frames.
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    config = load_config(config_path)
    print(config_path)
    print(config.keys())
    lane_width_pixels = config["lane_width_pixels"]
    print(type(lane_width_pixels))
    print(lane_width_pixels)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS =", fps)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        raise RuntimeError("Unable to read the first frame from the video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    bev_prev = cv2.warpPerspective(prev_gray,H,(400,1000))
    flow_mask = build_flow_mask(prev_gray.shape, config.get("polygons", {}))
    cv2.imshow("Flow Mask", flow_mask)
    print("flow_mask:",np.count_nonzero(flow_mask))

    #road_only_mask = flow_mask.copy()
    print("Mask pixels:", np.count_nonzero(flow_mask))
    #motion_history = []
    #speed_history = deque(maxlen=10)
    #display_speed = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
         # DEBUG SRC POINTS
        debug = frame.copy()

        for i, p in enumerate(SRC.astype(int)):
            x, y = p

            print(f"SRC[{i}] = ({x},{y})")

            cv2.circle(
                    debug,
                    (x, min(y, frame.shape[0]-10)),
                    15,
                    (0,0,255),
                    -1
                )

            cv2.putText(
                    debug,
                    str(i),
                    (x, min(y, frame.shape[0]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                 )
            cv2.polylines(
                    debug,
                    [SRC.astype(np.int32)],
                    True,
                    (0,255,0),
                    3
                )

            cv2.putText(
                debug,
                str(i),
                tuple(p),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )
        cv2.imshow("SRC Points", debug)

        results = model.predict(frame,conf=0.3,verbose=False)[0]
        print("detections:", len(results.boxes))
        #add vehicle mask
        vehicle_mask = np.zeros(flow_mask.shape,dtype=np.uint8)

        VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle"}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            print("Detected:", label)
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            padding = 15

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)

            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            cv2.rectangle(vehicle_mask,(x1, y1),(x2, y2),255,-1)
            print("vehicle_mask:", np.count_nonzero(vehicle_mask))
            cv2.imshow("Vehicle Mask", vehicle_mask)

        road_only_mask = cv2.bitwise_and(flow_mask,cv2.bitwise_not(vehicle_mask))

        print("road_only_mask:", np.count_nonzero(road_only_mask))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bev_gray = cv2.warpPerspective(gray,H,(400,1000))
        print("BEV shape:", bev_gray.shape)
        print("BEV nonzero:", np.count_nonzero(bev_gray))

        ys, xs = np.where(bev_gray > 0)

        print("xmin =", xs.min(), "xmax =", xs.max())
        print("ymin =", ys.min(), "ymax =", ys.max())
        cv2.imshow("BEV Gray", bev_gray)

        flow = cv2.calcOpticalFlowFarneback(
            bev_prev,
            bev_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        print("flow shape:", flow.shape)
        print("mask shape:", road_only_mask.shape)
        #avg_motion = compute_average_motion(flow, road_only_mask)
        bev_mask = cv2.warpPerspective(road_only_mask,H,(400,1000))

        dx = flow[:, :, 0]
        dy = flow[:, :, 1]
        road = bev_mask > 0
        dy_values = dy[road]
        # reject tiny motion
        dy_values = dy_values[np.abs(dy_values) > 1.0]
        # Reject crazy vectors
        dy_values = dy_values[np.abs(dy_values) < 12]
        # Reject statistical outliers
        q1 = np.percentile(dy_values,25)
        q3 = np.percentile(dy_values,75)
        iqr = q3-q1

        dy_values = dy_values[
            (dy_values>=q1-1.5*iqr) &
            (dy_values<=q3+1.5*iqr)
        ]
        if len(dy_values) == 0:
            continue
        median_dy = np.percentile(np.abs(dy_values), 80)
        # Convert
        #meters = median_dy * (3.5 / 100)
        meters_per_pixel = LANE_WIDTH_METERS / 100.0

        meters = median_dy * meters_per_pixel

        raw_speed = meters * fps * 3.6

        speed = raw_speed * CALIBRATION_FACTOR
        print(
            f"dy_count={len(dy_values)}  "
            f"median_dy={median_dy:.3f}  "
            f"speed={speed:.2f} km/h"
        )

        #avg_motion = compute_average_motion(flow,bev_mask)
        cv2.imshow("BEV Mask", bev_mask)

        #motion_history.append(avg_motion)

        #if len(motion_history) > 10:
            #motion_history.pop(0)

        #smooth_motion = np.mean(motion_history)
        """
        step = 40   
        h, w = flow.shape[:2]

        motions = []

        sample_half_width = 35      # pixels on each side of road center

        top_margin = int(h * 0.35)       # ignore top 20%
        bottom_margin = int(h * 0.80)    # ignore bottom 10%

        dxs = []
        dys = []

        for y in range(top_margin, bottom_margin, step):

            # Convert BEV row (0-999) back to original image Y (700-1400)
            orig_y = 700 + (y / 1000.0) * (1400 - 700)

            nearest_y = min(
                lane_width_pixels.keys(),
                key=lambda k: abs(int(k) - orig_y)
            )

            lane_width_px = lane_width_pixels[nearest_y]

            #meters_per_pixel = 3.5 / lane_width_px
            meters_per_pixel = 3.5 / 100.0

            road_pixels = np.where(bev_mask[y] > 0)[0]

            if len(road_pixels) < 20:
                continue

            left = road_pixels[0]
            right = road_pixels[-1]

            center = (left + right) // 2

            x_start = max(0, center - sample_half_width)
            x_end   = min(w, center + sample_half_width)

            for x in range(x_start, x_end, step):
                if bev_mask[y, x] == 0:
                    continue
                dx, dy = flow[y, x]
                dxs.append(dx)
                dys.append(dy)
                #print(
                    #f"dx median={np.median(dxs):.2f}, "
                    #f"dy median={np.median(dys):.2f}"
                #)

                #mag = np.sqrt(dx * dx + dy * dy)
                mag = abs(dy)
                
               # print(
                    #f"dx={dx:.2f} "
                   # f"dy={dy:.2f} "
                   # f"mag={mag:.2f}"
                #)
                if mag < 1.0:
                    continue

                # Reject impossible optical flow vectors
                if mag > 12:
                    continue

                meters = mag * meters_per_pixel
                motions.append(meters)

                cv2.arrowedLine(
                    frame,
                    (x, y),
                    (int(x + dx * 5), int(y + dy * 5)),
                    (0, 0, 255),
                    1,
                    tipLength=0.3,
                )
                print("motions =", len(motions))
        if len(motions) > 5:

            motion_array = np.array(motions)

            q1 = np.percentile(motion_array, 25)
            q3 = np.percentile(motion_array, 75)

            iqr = q3 - q1
            
            motion_array = motion_array[
                (motion_array >= q1 - 1.5 * iqr) &
                (motion_array <= q3 + 1.5 * iqr)
            ]
            print("motion_array =", len(motion_array))

            motion = float(np.median(motion_array))
            speed_kmh = 0.0
            if len(motion_array) >= 3:
                motion = float(np.median(motion_array))
                speed_kmh = motion * meters_per_pixel * fps * 3.6
                speed_history.append(speed_kmh)
                display_speed = np.mean(speed_history)
                print(
                f"y={orig_y:.0f}  "
                f"lane={lane_width_px:.1f}px  "
                f"m/px={meters_per_pixel:.5f}  "
                f"speed={speed_kmh:.1f}"
            )

            else:
                motion = 0.0
                speed_kmh = 0.0
                display_speed = 0.0
        else:
            motion = 0.0
            speed_kmh = 0.0
            display_speed = 0.0
        """
        #Road masks are easier to see if converted to color:
        road_mask_vis = cv2.cvtColor( road_only_mask,cv2.COLOR_GRAY2BGR )

        cv2.imshow("Road Mask",road_mask_vis)

        for pts in config.get("polygons", {}).values():
            poly = np.array(pts, dtype=np.int32)
            if poly.size == 0:
                continue
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        cv2.circle(frame, (831,700), 10, (0,0,255), -1)      # Top Left
        cv2.circle(frame, (1205,700), 10, (0,255,0), -1)     # Top Right

        cv2.circle(frame, (394,1400), 10, (255,0,0), -1)     # Bottom Left
        cv2.circle(frame, (1300,1400), 10, (255,255,0), -1)  # Bottom Right

        cv2.putText(
            frame,
            f"Speed = {speed:.1f} km/h",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        """
        print(
            f"MOTION={avg_motion:.2f} "
            f"SMOOTH={smooth_motion:.2f}"
        )
        """
        if show_video:
            frame = cv2.resize(frame,None,fx=0.7,fy=0.7)
            cv2.imshow("Optical Flow Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_gray = gray
        bev_prev = bev_gray
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    # Parse command line arguments for video and config paths.
    default_config = DEFAULT_CONFIG_PATH
    parser = argparse.ArgumentParser(
        description="Compute ego motion from video using configured polygon ROIs."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=TEMP_VIDEO_PATH or Path.cwd() / "input.mp4",
        help="Path to the video file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the polygon configuration JSON file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_video(args.video, args.config, show_video=not args.no_display)
    except Exception as error:
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()