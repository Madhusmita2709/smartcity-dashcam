from ultralytics import YOLO
import cv2
from pathlib import Path

from backend.app.services.storage import upload_file
from backend.app.core.config import get_settings
from backend.app.services.processors.plate_reader import PlateReader

settings = get_settings()


class TripleRidingDetector:

    def __init__(self):

        self.model = YOLO(
            "backend/app/services/models/triple_riding.pt"
        )

        self.PERSON_CLASS = 1
        self.MOTORCYCLE_CLASS = 0

        self.plate_reader = PlateReader()

        # cooldown
        self.violation_cooldown = {}

        # stable memory
        self.last_valid_plate = {}

    def generate_track_key(
        self,
        mx1,
        my1,
        mx2,
        my2
    ):

        # BIGGER GRID
        center_x = (mx1 + mx2) // 2
        center_y = (my1 + my2) // 2

        return (
            center_x // 120,
            center_y // 120
        )

    def run(
        self,
        input_path,
        output_dir,
        video_id
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_video = (
            output_dir /
            "triple_riding_output.mp4"
        )

        violation_dir = (
            output_dir /
            "violations"
        )

        violation_dir.mkdir(
            exist_ok=True
        )

        cap = cv2.VideoCapture(
            str(input_path)
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        out = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        violation_records = []

        while cap.isOpened():

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

                    cls = int(
                        box.cls[0]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    conf = float(
                        box.conf[0]
                    )

                    if cls == self.PERSON_CLASS:

                        persons.append(
                            [x1, y1, x2, y2, conf]
                        )

                    elif cls == self.MOTORCYCLE_CLASS:

                        motorcycles.append(
                            [x1, y1, x2, y2, conf]
                        )

            for moto in motorcycles:

                mx1, my1, mx2, my2, moto_conf = moto

                rider_count = 0
                conf_scores = []

                for person in persons:

                    px1, py1, px2, py2, person_conf = person

                    foot_x = (
                        px1 + px2
                    ) // 2

                    foot_y = py2

                    if (
                        mx1 - 40 <= foot_x <= mx2 + 40
                        and
                        my1 - 30 <= foot_y <= my2 + 30
                    ):

                        rider_count += 1

                        conf_scores.append(
                            person_conf
                        )

                if rider_count >= 3:

                    avg_conf = (
                        sum(conf_scores)
                        / len(conf_scores)
                        if conf_scores else 0
                    )

                    final_conf = (
                        avg_conf + moto_conf
                    ) / 2

                    timestamp = (
                        cap.get(
                            cv2.CAP_PROP_POS_MSEC
                        ) / 1000
                    )

                    # STABLE TRACK KEY
                    track_key = (
                        self.generate_track_key(
                            mx1,
                            my1,
                            mx2,
                            my2
                        )
                    )

                    # INIT MEMORY
                    if (
                        track_key
                        not in
                        self.last_valid_plate
                    ):

                        self.last_valid_plate[
                            track_key
                        ] = "UNKNOWN"

                    plate_number = (
                        self.last_valid_plate[
                            track_key
                        ]
                    )

                    # OCR
                    plate_results = (
                        self.plate_reader.read_plate(
                            frame,
                            track_key
                        )
                    )

                    if plate_results:

                        detected_plate = (
                            plate_results[0]
                            .get(
                                "plate",
                                "UNKNOWN"
                            )
                        )

                        # SAVE ONLY VALID
                        if (
                            detected_plate != "UNKNOWN"
                            and
                            len(detected_plate) >= 6
                        ):

                            self.last_valid_plate[
                                track_key
                            ] = detected_plate

                            plate_number = (
                                detected_plate
                            )

                        else:

                            # REUSE OLD PLATE
                            plate_number = (
                                self.last_valid_plate[
                                    track_key
                                ]
                            )

                    print(
                        f"[PLATE] "
                        f"{track_key} -> "
                        f"{plate_number}",
                        flush=True
                    )

                    # STILL UNKNOWN?
                    if (
                        plate_number == "UNKNOWN"
                    ):
                        continue

                    last_saved = (
                        self.violation_cooldown.get(
                            track_key,
                            -999
                        )
                    )

                    if (
                        timestamp -
                        last_saved
                    ) >= 2:

                        image_name = (
                            f"{video_id}_"
                            f"{int(timestamp)}s_"
                            f"{plate_number}.jpg"
                        )

                        temp_path = (
                            violation_dir /
                            image_name
                        )

                        cv2.imwrite(
                            str(temp_path),
                            frame
                        )

                        object_key = (
                            f"videos/"
                            f"{video_id}/"
                            f"violations/"
                            f"{image_name}"
                        )

                        uploaded = upload_file(
                            settings.minio_images_bucket,
                            object_key,
                            temp_path,
                            "image/jpeg"
                        )

                        violation_records.append(
                            {
                                "timestamp_seconds":
                                timestamp,

                                "plate_number":
                                plate_number,

                                "violation_type":
                                "triple_riding",

                                "confidence":
                                final_conf,

                                "image_url":
                                uploaded.object_url
                            }
                        )

                        self.violation_cooldown[
                            track_key
                        ] = timestamp

                        temp_path.unlink(
                            missing_ok=True
                        )

                        print(
                            f"TRIPLE RIDING | "
                            f"Plate: {plate_number} | "
                            f"Time: {timestamp:.2f}s",
                            flush=True
                        )

                    cv2.rectangle(
                        frame,
                        (mx1, my1),
                        (mx2, my2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"TRIPLE {final_conf:.2f}",
                        (mx1, my1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            out.write(frame)

        cap.release()
        out.release()

        cv2.destroyAllWindows()

        return output_video, {
            "status": "completed",
            "violations": violation_records
        }