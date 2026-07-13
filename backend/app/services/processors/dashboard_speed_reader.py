import cv2
import easyocr
import re
from typing import Optional, List


class DashboardSpeedReader:

    def __init__(self):
        # Heavy OCR engine initialization stays here once
        self.reader = easyocr.Reader(['en'], gpu=False)

    def get_speed(self, frame: cv2.Mat) -> Optional[float]:
        """
        Extracts numeric speed readings dynamically from a percentage-based ROI boundary.
        """
        h, w = frame.shape[:2]

        # --- DYNAMIC PERCENTAGE-BASED ROI ---
        # Maps matching proportions perfectly whether image is 720p, 1080p, or 1440p
        crop_top = int(h * 0.88)
        crop_bottom = int(h * 1.0)
        crop_left = 0
        crop_right = int(w * 0.45)

        roi = frame[crop_top:crop_bottom, crop_left:crop_right]

        # Safety Check Guard: Bypasses calculation immediately if slice arrays evaluate empty
        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"[SPEED READER WARNING] Bypassed invalid slice frame target boundaries: shape={frame.shape}")
            return None

        # Process matrix maps under valid dimension arrays
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Scaling interpolation to expand characters safely for EasyOCR engine processing
        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )
        
        result = self.reader.readtext(
            gray,
            allowlist="0123456789.",
            detail=0,
            paragraph=False
        )

        speed = None
        for text in result:
            # Enforce validation constraints matching layout decimal rules (e.g., 45.3)
            if "." in text:
                try:
                    value = float(text)
                    if 1.0 <= value <= 180.0:
                        speed = value
                        break
                except ValueError:
                    pass

        return speed
"""   # standalone testing code 
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

if __name__ == "__main__":

    cap = cv2.VideoCapture(
        r"C:\videoset1_videos_part1\20220824155045_0060speed_highway.mp4"
    )

    reader = DashboardSpeedReader()
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(frame.shape)
        #cv2.imshow("Raw Frame", frame)
        speed = reader.get_speed(frame)

        print("OCR Speed =", speed)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
"""