from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np

class WrongWayDetector:

    def __init__(self):
        print("[WRONG WAY INIT]", flush=True)
        self.model = YOLO("yolov8n.pt")
    def get_zone(self,cx):

        if cx < 250:
            return "shoulder"

        elif cx < 520:
            return "lane1"

        elif cx < 700:
            return "lane2"

        elif cx < 820:
            return "divider"

        else:
            return "other_side"
            
    def run(self, input_path, output_dir, video_id):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(input_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = output_dir / f"{video_id}_processed.mp4"

        out = cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width, height))

        print(f"[VIDEO] Saving to {output_video}", flush=True)
        ROAD_POLYGON = np.array([
    [180, 478],   # bottom left
    [700, 478],   # bottom right
    [520, 240],   # top right
    [320, 240]    # top left
], np.int32)
        previous_positions = {}
        wrong_way_count = {}
        flow_dx = []
        flow_dy = []
        saved_violations = set()
        track_history = {}
        
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame,persist=True,tracker="bytetrack.yaml",conf=0.25)
            #print(f"[VEHICLE] {class_name}")
            for r in results:
                for box in r.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    print(
                        f"[TRACK_ID] {track_id} {class_name}",
                        flush=True
                    )
                    
                    # Only vehicles
                    if class_name not in ["car","motorcycle","bus","truck"]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if track_id not in track_history:
                        track_history[track_id] = []

                    track_history[track_id].append((center_x, center_y))

                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                    history = track_history[track_id]

                    print(
                        f"[ANGLE START] "
                        f"ID={track_id} "
                        f"HISTORY={len(history)}",
                        flush=True
                    )

                    if len(history) >= 5:

                        start_x, start_y = history[0]
                        end_x, end_y = history[-1]

                        traj_dx = end_x - start_x
                        traj_dy = end_y - start_y

                        vehicle_angle = np.degrees(np.arctan2(traj_dy, traj_dx))

                        if len(flow_dx) < 5:
                            continue
                        flow_angle = np.degrees(np.arctan2(avg_dy, avg_dx))

                        angle_diff = abs(vehicle_angle - flow_angle)

                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        print(
                            f"[SEEN] "
                            f"ID={track_id} "
                            f"CLASS={class_name}",
                            flush=True
                        )
                        print(
                            f"[ANGLE CHECK] "
                            f"ID={track_id} "
                            f"VEHICLE={vehicle_angle:.2f} "
                            f"FLOW={flow_angle:.2f} "
                            f"DIFF={angle_diff:.2f}",
                            flush=True
                        )
                        if angle_diff > 150:
                            wrong_way_count[track_id] = wrong_way_count.get(track_id, 0) + 1
                        else:
                            wrong_way_count[track_id] = 0
                        print(
                            f"[ANGLE CHECK] "
                            f"ID={track_id} "
                            f"VEHICLE={vehicle_angle:.2f} "
                            f"FLOW={flow_angle:.2f} "
                            f"DIFF={angle_diff:.2f} "
                            f"COUNT={wrong_way_count[track_id]}",
                            flush=True
                        )

                    print(
                        f"[CLASS] "
                        f"ID={track_id} "
                        f"CLASS={class_name}"
                        f"CX={center_x} "
                        f"CY={center_y}",
                        flush=True
                    )
                    inside = cv2.pointPolygonTest(ROAD_POLYGON,(center_x, center_y),False)

                    if inside < 0:
                        continue
                    print(
                        f"[ROI] "
                        f"ID={track_id} "
                        f"CLASS={class_name} "
                        f"CX={center_x} "
                        f"CY={center_y}",
                        flush=True
                    )
                    zone = self.get_zone(center_x)
                
                    # Ignore divider and opposite carriageway
                    if zone in ["shoulder","divider", "other_side"]:
                        print(f"[SKIP_ZONE] ID={track_id} ZONE={zone}", flush=True)
                        continue

                    cv2.putText(frame,zone,(x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 0),2)
            
                    # Draw bounding box
                    cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)

                    # Draw Track ID
                    # ROI
                    cv2.polylines(frame,[ROAD_POLYGON],True,(0,255,255),2)

                    # Vehicle box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame,f"{class_name} ID:{track_id}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
                    # Zone
                    cv2.putText(frame,zone,(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

                    vehicle_key = track_id
                    if vehicle_key in previous_positions:
                        print(
                            f"ID={track_id} "
                            f"ZONE={zone} "
                            f"CX={center_x} "
                            f"CY={center_y}",
                            flush=True
                        )
                
                        prev_x, prev_y = previous_positions[vehicle_key]
                        dx = center_x - prev_x
                        dy = center_y - prev_y
                        if abs(dx) > 50:
                            print(f"[SKIP_DX] ID={track_id} DX={dx}", flush=True)
                            continue
                        if abs(dx) < 5 and abs(dy) < 5:
                            continue
                        if class_name == "motorcycle":
                            print(
                                f"[MOTORCYCLE] "
                                f"ID={track_id} "
                                 f"PREV=({prev_x},{prev_y}) "
                                f"CURR=({center_x},{center_y}) "
                                f"DX={dx} "
                                f"DY={dy}",
                                flush=True
                            )
                        movement = abs(dx) + abs(dy)

                        if movement < 8:
                            print(f"[SKIP_MOVE] ID={track_id} MOVE={movement}", flush=True)
                            continue
                        flow_dx.append(dx)
                        flow_dy.append(dy)
                        if len(flow_dx) > 100:
                            flow_dx.pop(0)
                            flow_dy.pop(0)

                        avg_dx = sum(flow_dx) / len(flow_dx)
                        avg_dy = sum(flow_dy) / len(flow_dy)
                        print(f"[FLOW] AVG_DX={avg_dx:.2f} "f"AVG_DY={avg_dy:.2f}",flush=True)

                        if len(flow_dx) < 20:
                            continue
                    
                        flow_vector = np.array([avg_dx, avg_dy])
                        history = track_history[track_id]

                        start_x, start_y = history[0]
                        end_x, end_y = history[-1]

                        traj_dx = end_x - start_x
                        traj_dy = end_y - start_y

                        print(
                            f"[CHECK] ID={track_id} "
                            f"AVG_DY={avg_dy:.2f} "
                            f"DX={dx} "
                            f"DY={dy} "
                            f"COUNT={wrong_way_count.get(track_id,0)}",
                            flush=True
                        )
                        if wrong_way_count.get(track_id, 0) >= 5:

                            if track_id not in saved_violations:
                                violation_path = output_dir / f"violation_{track_id}.jpg"
                                cv2.imwrite(str(violation_path), frame)
                                print(f"[VIOLATION SAVED] {violation_path}", flush=True)
                                saved_violations.add(track_id)

                            cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 0, 255),3)

                            cv2.putText(frame,f"WRONG WAY ID:{track_id}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)
                            

                        #print(f"[FLOW] AVG_DX={avg_dx:.2f} AVG_DY={avg_dy:.2f}",flush=True)
                        #print(f"ID={track_id} "f"ZONE={zone} "f"DX={dx} "f"DY={dy} "f"MOVE={movement}")
                        #if dx < -8 and abs(dy) < 15:
                            #wrong_way_count[track_id] = wrong_way_count.get(track_id, 0) + 1
                        #else:
                            #wrong_way_count[track_id] = 0

                        print(
                            f"[TRACK] ID={track_id} {class_name} Prev=({prev_x},{prev_y}) "
                            f"Curr=({center_x},{center_y}) DX={dx} DY={dy} ",
                            flush=True,
                        )
                    
                        #if wrong_way_count[track_id] >= 10:

                            #print(f"[WRONG WAY DETECTED] ID={track_id}",flush=True)
                            #cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 0, 255),3)
                            #cv2.putText(frame,"WRONG WAY",(x1, y1 - 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
                            #cv2.imshow("Wrong Way Debug", frame)
                            #cv2.waitKey(0)
                    previous_positions[vehicle_key] = (center_x, center_y)
            cv2.polylines(frame,[ROAD_POLYGON],True,(0,255,255),2)
            out.write(frame)
            cv2.imshow("Wrong Way Debug", frame)

            key = cv2.waitKey(30) 

            if key == ord("p"):
                cv2.waitKey(0)

            if key == ord("q"):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[WRONG WAY DETECTOR FINISHED]", flush=True)
        return input_path, {
            "status": "completed",
            "violations": []
        }