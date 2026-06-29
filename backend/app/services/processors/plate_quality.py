import cv2
import numpy as np


class PlateQualityAnalyzer:

    def __init__(self):
        # Initial thresholds (we'll tune later)
        self.blur_threshold = 80
        self.contrast_threshold = 20
        self.edge_threshold = 0.02
        self.min_width = 60
        self.min_height = 25

    def analyze(self, crop):

        if crop is None or crop.size == 0:
            return self._result("INVALID", 0, 0, 0, 0)

        h, w = crop.shape[:2]

        # Very small crop -> impossible to judge
        if h < self.min_height or w < self.min_width:
            return self._result("TOO_SMALL", 0, 0, 0, 0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        blur = self.blur_score(gray)
        contrast = self.contrast(gray)
        edge_density = self.edge_density(gray)
        contours = self.character_contours(gray)

        status = self.classify(
            blur,
            contrast,
            edge_density,
            contours
        )

        return self._result(
            status,
            blur,
            contrast,
            edge_density,
            contours
        )

    def blur_score(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def contrast(self, gray):
        return float(gray.std())

    def edge_density(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        return np.count_nonzero(edges) / edges.size

    def character_contours(self, gray):

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid = 0

        for c in contours:

            area = cv2.contourArea(c)

            if 20 < area < 1000:
                valid += 1

        return valid

    def classify(
        self,
        blur,
        contrast,
        edge_density,
        contours
    ):

        if blur < self.blur_threshold:
            return "BLUR"

        if (
            contrast < self.contrast_threshold
            and edge_density < self.edge_threshold
        ):
            return "BLANK"

        if contours < 5:
            return "UNREADABLE"

        return "READABLE"

    def _result(
        self,
        status,
        blur,
        contrast,
        edge_density,
        contours
    ):
        return {
            "status": status,
            "blur": blur,
            "contrast": contrast,
            "edge_density": edge_density,
            "contours": contours
        }