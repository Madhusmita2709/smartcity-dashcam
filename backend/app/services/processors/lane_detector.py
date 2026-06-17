import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO

# Main calibration function: auto-detects lane markings, fits curves, and saves polygons/IPM to config.json.
def detect_lanes_and_save_config(video_path, output_config_path, debug_image_path):
    print(f"[LANE DETECTOR] Loading video: {video_path}", flush=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", flush=True)
        return False
        
    # Read a frame at 5 seconds (125 frames at 25fps) to find lanes when road is clear
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Failed to read frame", flush=True)
        return False
        
    h, w = frame.shape[:2]
    
    # Run YOLO object detector to locate vehicles so we can mask them out (prevents fitting lanes on vehicle edges).
    print("[LANE DETECTOR] Detecting and masking out vehicles...", flush=True)
    model = YOLO("yolov8n.pt")
    results = model(frame, verbose=False)
    
    # 1. Image preprocessing: Grayscale, Gaussian blur, and Canny edge detection to isolate lane lines.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 2. Mask the road Region of Interest (ROI) to ignore background clutter like sky, trees, and side barriers.
    roi_mask = np.zeros_like(edges)
    poly_pts = np.array([[200, h], [800, 700], [1800, 700], [w - 200, h]], np.int32)
    cv2.fillPoly(roi_mask, [poly_pts], 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    
    # 3. Mask out vehicle boxes to remove fake lane lines from vehicle silhouettes.
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                if class_name in ["car", "truck", "bus", "motorcycle"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad = 15
                    px1 = max(0, x1 - pad)
                    py1 = max(0, y1 - pad)
                    px2 = min(w, x2 + pad)
                    py2 = min(h, y2 + pad)
                    cv2.rectangle(masked_edges, (px1, py1), (px2, py2), 0, -1)
    
    # 4. Detect straight lane markings using Probabilistic Hough Transform on the edge map.
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=60,
        maxLineGap=40
    )
    
    if lines is None:
        print("[ERROR] No lines detected in the ROI", flush=True)
        return False
        
    # 5. Cluster line segments into Left, Middle, and Right lane boundaries using K-means on their x-intercepts at y=1000.
    x_intercepts = []
    line_points = []
    target_y = 1000
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y2 == y1:
            continue
        slope = (x2 - x1) / (y2 - y1)
        if abs(slope) > 2.0:
            continue
            
        x_int = x1 + (target_y - y1) * slope
        if 400 < x_int < 1800:
            x_intercepts.append(x_int)
            line_points.append(((x1, y1), (x2, y2), x_int))
            
    if len(x_intercepts) < 10:
        print("[ERROR] Too few valid lane lines found", flush=True)
        return False
        
    x_intercepts = np.array(x_intercepts, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(x_intercepts, 3, None, criteria, 10, flags)
    
    sorted_idx = np.argsort(centers.flatten())
    cluster_mapping = {sorted_idx[0]: "L1", sorted_idx[1]: "L2", sorted_idx[2]: "L3"}
    
    points_by_cluster = {"L1": [], "L2": [], "L3": []}
    for i, label in enumerate(labels.flatten()):
        cluster_name = cluster_mapping[label]
        pt1, pt2, _ = line_points[i]
        points_by_cluster[cluster_name].append(pt1)
        points_by_cluster[cluster_name].append(pt2)
        
    # 6. Fit quadratic curves to each group. Falls back to a linear fit if the curve is too sharp (prevents crazy bends at the horizon).
    fitted_curves = {}
    for cluster_name, pts in points_by_cluster.items():
        if len(pts) < 4:
            print(f"[WARNING] Not enough points to fit {cluster_name}", flush=True)
            return False
        pts_arr = np.array(pts)
        X = pts_arr[:, 0]
        Y = pts_arr[:, 1]
        
        poly_coeffs = np.polyfit(Y, X, 2)
        a, b, c = poly_coeffs
        
        if abs(a) > 0.0003:
            print(f"[LANE DETECTOR] Curvature of {cluster_name} too high ({a:.6f}). Falling back to linear fit.", flush=True)
            m, c_lin = np.polyfit(Y, X, 1)
            poly_coeffs = [0.0, m, c_lin]
            
        fitted_curves[cluster_name] = poly_coeffs
        
    # Verify L1 (left), L2 (middle), and L3 (right) curves are fit
    if "L1" not in fitted_curves or "L2" not in fitted_curves or "L3" not in fitted_curves:
        print("[ERROR] Failed to fit all three lane lines", flush=True)
        return False
        
    print("[LANE DETECTOR] Successfully fit L1, L2, and L3 curves", flush=True)
    
    # 7. Generate lane polygons at Y-intervals to define lanes: SHOULDER, LANE_1, LANE_2, DIVIDER, OPPOSITE.
    y_top = 700
    y_bottom = 1440
    y_steps = np.arange(y_top, y_bottom + 10, 50)
    if y_steps[-1] > y_bottom:
        y_steps[-1] = y_bottom
        
    # Helper: computes curve x-coordinate at a given y-coordinate.
    def get_x_curve(coeffs, y):
        a, b, c = coeffs
        return int(a * (float(y)**2) + b * float(y) + c)
        
    pts_L1 = [(int(get_x_curve(fitted_curves["L1"], y)), int(y)) for y in y_steps]
    pts_L2 = [(int(get_x_curve(fitted_curves["L2"], y)), int(y)) for y in y_steps]
    pts_L3 = [(int(get_x_curve(fitted_curves["L3"], y)), int(y)) for y in y_steps]
    
    pts_L0 = []
    pts_L4 = []
    for idx, y in enumerate(y_steps):
        factor = (float(y) - y_top) / (y_bottom - y_top)
        offset_L0 = int(70 + 280 * factor)
        offset_L4 = int(120 + 680 * factor)
        
        pts_L0.append((int(pts_L1[idx][0] - offset_L0), int(y)))
        pts_L4.append((int(pts_L3[idx][0] + offset_L4), int(y)))
        
    pts_L3_div = []
    for idx, y in enumerate(y_steps):
        factor = (float(y) - y_top) / (y_bottom - y_top)
        offset_divider = int(15 + 85 * factor)
        pts_L3_div.append((int(pts_L3[idx][0] + offset_divider), int(y)))
        
    # Helper: builds a polygon boundary from left and right line points.
    def make_poly(left_pts, right_pts):
        poly = []
        for pt in left_pts:
            poly.append([int(pt[0]), int(pt[1])])
        for pt in reversed(right_pts):
            poly.append([int(pt[0]), int(pt[1])])
        return poly
        
    polygons = {
        "SHOULDER": make_poly(pts_L0, pts_L1),
        "LANE_1": make_poly(pts_L1, pts_L2),
        "LANE_2": make_poly(pts_L2, pts_L3),
        "DIVIDER": make_poly(pts_L3, pts_L3_div),
        "OPPOSITE": make_poly(pts_L3_div, pts_L4)
    }
    
    # 8. Save the lane polygons and Inverse Perspective Mapping (IPM) parameters to config.json.
    config_data = {}
    if Path(output_config_path).exists():
        try:
            with open(output_config_path, "r") as f:
                config_data = json.load(f)
        except Exception:
            pass
            
    config_data["polygons"] = polygons
    config_data["ipm"] = {
        "src": [
            [int(pts_L1[0][0]), int(y_top)],
            [int(pts_L2[0][0]), int(y_top)],
            [int(pts_L1[-1][0]), int(y_bottom)],
            [int(pts_L2[-1][0]), int(y_bottom)]
        ],
        "dst": [
            [100, 0],
            [200, 0],
            [100, 1000],
            [200, 1000]
        ]
    }
    
    if "monitored_regions" not in config_data:
        config_data["monitored_regions"] = ["LANE_1", "LANE_2"]
    if "speed_thresholds" not in config_data:
        config_data["speed_thresholds"] = {"stationary_max_speed": 80.0, "wrong_way_min_speed": 30.0}
    if "time_range" not in config_data:
        config_data["time_range"] = {"start_msec": 24000, "end_msec": 30000}
    if "tracker" not in config_data:
        config_data["tracker"] = "botsort.yaml"
    if "confidence_threshold" not in config_data:
        config_data["confidence_threshold"] = 0.15
    if "yolo_model" not in config_data:
        config_data["yolo_model"] = "yolov8n.pt"
    if "imgsz" not in config_data:
        config_data["imgsz"] = 1280
    if "draw_lanes" not in config_data:
        config_data["draw_lanes"] = False
    if "speed_window_size" not in config_data:
        config_data["speed_window_size"] = 5
        
    with open(output_config_path, "w") as f:
        json.dump(config_data, f, indent=2)
        
    print(f"[LANE DETECTOR] Config updated successfully: {output_config_path}", flush=True)
    
    # 9. Draw and save debug lane visualization for verification of the lane boundaries.
    debug_colors = {
        "SHOULDER": (0, 255, 255),  # Yellow
        "LANE_1": (0, 255, 0),      # Green
        "LANE_2": (255, 0, 0),      # Blue
        "DIVIDER": (128, 128, 128),  # Gray
        "OPPOSITE": (0, 0, 255)     # Red
    }
    
    for name, pts in polygons.items():
        pts_np = np.array(pts, np.int32)
        cv2.polylines(frame, [pts_np], isClosed=True, color=debug_colors[name], thickness=3)
        bottom_pt = pts_np[np.argmax(pts_np[:, 1])]
        cv2.putText(frame, name, (bottom_pt[0] - 50, bottom_pt[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, debug_colors[name], 3)
        
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imwrite(str(debug_image_path), frame_resized)
    print(f"[LANE DETECTOR] Debug lane visualization saved to {debug_image_path}", flush=True)
    return True

if __name__ == "__main__":
    video_path = r"c:\Users\DELL\Desktop\Code More\wrong-way-detection\wrong_way-video.mp4"
    config_path = r"c:\Users\DELL\Desktop\Code More\wrong-way-detection\config.json"
    debug_path = r"c:\Users\DELL\Desktop\Code More\wrong-way-detection\detected_lanes.jpg"
    
    detect_lanes_and_save_config(video_path, config_path, debug_path)
