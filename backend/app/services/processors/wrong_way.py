from pathlib import Path
import cv2
from shapely.geometry import box
from ultralytics import YOLO

class WrongWayDetector:

    def __init__(self):
        print("[WRONG WAY INIT]", flush=True)
        self.model = YOLO("yolov8n.pt")

    def run(self, input_path, output_dir, video_id):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(input_path))
        previous_positions = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame,persist=True,tracker="bytetrack.yaml")
            #print(f"[VEHICLE] {class_name}")
            for r in results:
                for box in r.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    # Only vehicles
                    if class_name not in ["car","motorcycle","bus","truck"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    vehicle_key = track_id
                    if vehicle_key in previous_positions:
                        prev_x, prev_y = previous_positions[vehicle_key]
                        dx = center_x - prev_x
                        print(f"[TRACK] ID={track_id} "f"{class_name} "f"PrevX={prev_x} "f"CurrX={center_x} "f"DX={dx}",flush=True)
                    previous_positions[vehicle_key] = (center_x, center_y)
        cap.release()
        print("[WRONG WAY DETECTOR FINISHED]", flush=True)
        return input_path, {
            "status": "completed",
            "violations": []
        }