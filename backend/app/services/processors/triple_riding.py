from ultralytics import YOLO
import cv2
from pathlib import Path
import easyocr


class TripleRidingDetector:

    def __init__(self):

        self.model = YOLO(
            "backend/app/services/processors/yolov8n.pt"
        )

        self.PERSON_CLASS = 0
        self.MOTORCYCLE_CLASS = 3
        self.reader = easyocr.Reader(['en'])

    def extract_number_plate(self, frame, moto_box):

        x1, y1, x2, y2 = map(int, moto_box)

        # Better number plate region

        plate_region = frame[
                max(0, int(y2 - 60)):min(frame.shape[0], int(y2 + 40)),
            max(0, x1):min(frame.shape[1], x2)
            ]
        if plate_region.size == 0:
            return None

        gray = cv2.cvtColor(
        plate_region,
        cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
        )

        results = self.reader.readtext(gray)

        for result in results:

            text = result[1]

            # Remove spaces
            text = text.replace(" ", "")

            # Basic filtering
            if len(text) >= 6:
                return text

        return None

    def run(self, input_path, output_dir):

        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_dir / "triple_riding_output.mp4"

        cap = cv2.VideoCapture(str(input_path))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model(
                frame,
                conf=0.45
            )

            persons = []
            motorcycles = []

            for r in results:

                for box in r.boxes:

                    cls = int(box.cls[0])

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]   
                    )

                    confidence = float(box.conf[0])

                    if cls == self.PERSON_CLASS:

                        persons.append(
                            [x1, y1, x2, y2,confidence]
                        )

                    elif cls == self.MOTORCYCLE_CLASS:

                        motorcycles.append(
                            [x1, y1, x2, y2,confidence]
                        )

            for moto in motorcycles:

                mx1, my1, mx2, my2, moto_conf = moto

                rider_count = 0
                
                confidence_scores = []

                for person in persons:

                    px1, py1, px2, py2, person_conf = person

                    foot_x = (px1 + px2) // 2
                    foot_y = py2

                    if (
                        mx1 - 40 <= foot_x <= mx2 + 40
                        and
                        my1 - 30 <= foot_y <= my2 + 30
                    ):

                        rider_count += 1
                        confidence_scores.append(person_conf)

                color = (0, 255, 0)

                label = f"Riders: {rider_count}"

                if rider_count >= 3:

                    color = (0, 0, 255)

                    avg_conf = (
                        sum(confidence_scores) / len(confidence_scores)
                        if confidence_scores else 0
                    )

                    final_conf = (avg_conf + moto_conf) / 2

                    label = f"TRIPLE RIDING {final_conf:.2f}"

                    plate_number = self.extract_number_plate(frame, [mx1, my1, mx2, my2])

                    if plate_number:
                        print(f"TRIPLE RIDING DETECTED -> Plate Number: {plate_number}")
                    else:
                        print("TRIPLE RIDING DETECTED -> Plate Number Not Found")

                cv2.rectangle(
                    frame,
                    (mx1, my1),
                    (mx2, my2),
                    color,
                    3
                )

                cv2.putText(
                    frame,
                    label,
                    (mx1, my1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            out.write(frame)

        cap.release()
        out.release()

        return output_video, {
            "status": "completed",
            "message": "Triple riding detection completed"
        }