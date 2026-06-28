import cv2
import easyocr
import re


class DashboardSpeedReader:

    def __init__(self):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    def get_speed(self, frame):

        h, w = frame.shape[:2]
        #print("Frame Size:", h, w)

        #print("ROI:",int(h * 0.90),int(h * 0.98),int(w * 0.28),int(w * 0.48))


        # Bottom-left ROI
        roi = frame[
        1270:1440,
        0:900
        ]
        
        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )
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

        #cv2.imshow("Speed ROI", gray)
        speed = None
        #print("RAW OCR:", result)
        speed = None

        for text in result:

            #print("OCR:", text)

            # We only want decimal speeds like 45.3
            if "." in text:
                try:
                    value = float(text)

                    if 1 <= value <= 180:
                        speed = value
                        break

                except:
                    pass

        #cv2.imshow("Speed ROI", gray)
        #cv2.imshow("Frame", frame)
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