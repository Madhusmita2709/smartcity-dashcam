# Standard library imports
import re
from collections import Counter
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


class PlateReader:
    def __init__(self):
        self.detector = YOLO("backend/app/services/models/license_plate.pt")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        # Stable memories for tracking plates across frames
        self.track_memory = {}
        self.global_plate_memory = []

    def clean_text(self, txt):
        txt = re.sub(r"[^A-Z0-9]", "", txt.upper())
        return txt

    def vote_plate(self, samples):
        valid = []
        for s in samples:
            s = self.clean_text(s)
            if len(s) >= 6:
                valid.append(s)

        if not valid:
            return "UNKNOWN"

        return Counter(valid).most_common(1)[0][0]

    def run_ocr(self, img):
        try:
            res = self.ocr.ocr(img, cls=True)
            txt = ""
            if res and res[0]:
                for line in res[0]:
                    txt += line[1][0]
            return self.clean_text(txt)
        except Exception as e:
            print(f"[OCR ERROR] {e}", flush=True)
            return ""

    def read_plate(self, frame, track_key=None):
        results = self.detector(frame, conf=0.35)[0]
        print(f"[PLATE DETECTIONS] {len(results.boxes)}", flush=True)

        final_results = []

        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # DEBUG
            debug_dir = Path("media/debug_plates")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"raw_{x1}_{y1}.jpg"), crop)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # DESKEW
            coords = cv2.findNonZero(gray)
            if coords is not None:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                if angle < -45:
                    angle = 90 + angle

                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # UPSCALE
            up = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(up)

            # SHARPEN
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(clahe, -1, kernel)

            # SPLIT
            h = sharp.shape[0]
            split = int(h * 0.60)
            top = sharp[:split, :]
            bottom = sharp[split:, :]

            cv2.imwrite(str(debug_dir / f"top_{x1}_{y1}.jpg"), top)
            cv2.imwrite(str(debug_dir / f"bottom_{x1}_{y1}.jpg"), bottom)

            # OCR
            raw_txt = self.run_ocr(crop)
            full_txt = self.run_ocr(sharp)
            top_txt = self.run_ocr(top)
            bottom_txt = self.run_ocr(bottom)
            split_txt = top_txt + bottom_txt

            print(f"[OCR CANDIDATES] RAW={raw_txt} FULL={full_txt} SPLIT={split_txt}", flush=True)

            # COLLECT
            candidates = []
            for txt in [raw_txt, full_txt, split_txt]:
                txt = self.clean_text(txt)
                if len(txt) >= 6:
                    candidates.append(txt)

            # BEST OCR
            plate = max(candidates, key=len) if candidates else "UNKNOWN"
            print(f"[OCR PICKED] {track_key} -> {plate}", flush=True)

            # MEMORY
            if track_key:
                if track_key not in self.track_memory:
                    self.track_memory[track_key] = []

                if plate != "UNKNOWN" and len(plate) >= 6:
                    self.track_memory[track_key].append(plate)
                    self.track_memory[track_key] = self.track_memory[track_key][-10:]
                    self.global_plate_memory.append(plate)
                    self.global_plate_memory = self.global_plate_memory[-30:]

                track_vote = self.vote_plate(self.track_memory[track_key])
                global_vote = self.vote_plate(self.global_plate_memory)

                if track_vote != "UNKNOWN":
                    voted_plate = track_vote
                elif global_vote != "UNKNOWN":
                    voted_plate = global_vote
                else:
                    voted_plate = plate
            else:
                voted_plate = plate

            print(f"[FINAL PLATE VOTED] {track_key} -> {voted_plate}", flush=True)

            final_results.append({
                "plate": voted_plate if voted_plate else "UNKNOWN",
                "bbox": [x1, y1, x2, y2]
            })

        return final_results