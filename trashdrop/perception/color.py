"""Colour-keyed detector. Simulation only -- not a perception result.

Each stand-in object is spawned in the colour of its own category and then
segmented back out by exactly that colour, so this module is told the answer it
returns. It exists for one reason: to drive the motion stack end to end before
a trained classifier exists, and to keep a regression test that can fail if the
kinematics break.

Nothing here transfers to real trash. The real path is
:class:`~trashdrop.perception.background.BackgroundDetector` plus a crop
classifier.
"""

from __future__ import annotations

import numpy as np

from ..station import CAMERA, PICK_ZONE
from . import Detection
from .calibration import PinholeTopDown, PlaneCalibration

# Hue/saturation/value windows matching the spawn colours in station.py.
CLASS_HSV = {
    "bio": ((40, 120, 60), (80, 255, 255)),
    "paper": ((100, 120, 60), (130, 255, 255)),
    "plastic": ((20, 120, 60), (35, 255, 255)),
    "metal": ((0, 120, 60), (10, 255, 255)),
}

MIN_CONTOUR_AREA_PX = 300


class ColorDetector:
    """Segment the synthetic objects by their spawn colour."""

    def __init__(self, calibration: PlaneCalibration | None = None) -> None:
        self.calibration = calibration or PinholeTopDown(
            height=CAMERA.height,
            center_x=CAMERA.center_x,
            center_y=CAMERA.center_y,
            fovy_degrees=CAMERA.fovy_degrees,
            width=CAMERA.width,
            height_px=CAMERA.height_px,
        )
        self.overlay: np.ndarray | None = None

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        metres_per_pixel = getattr(self.calibration, "metres_per_pixel", None)
        found: list[Detection] = []

        for category, (low, high) in CLASS_HSV.items():
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
                    continue
                (cu, cv_), (w, h), angle = cv2.minAreaRect(contour)
                x, y = self.calibration.pixel_to_world(cu, cv_)
                # Anything outside the pick zone is a bin, not an item.
                if not PICK_ZONE.contains(x, y, margin=0.005):
                    continue
                # Close the jaws across the short side.
                yaw = np.deg2rad(angle if w < h else angle + 90.0)
                short_px, long_px = (w, h) if w < h else (h, w)
                scale = metres_per_pixel or 0.0
                found.append(
                    Detection(
                        category=category,
                        x=x,
                        y=y,
                        yaw=float((yaw + np.pi) % (2 * np.pi) - np.pi),
                        confidence=1.0,
                        width=float(short_px * scale),
                        length=float(long_px * scale),
                        pixel=(int(cu), int(cv_)),
                        box=np.intp(cv2.boxPoints(((cu, cv_), (w, h), angle))),
                    )
                )
                cv2.drawContours(overlay, [found[-1].box], 0, (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"{category} {x:+.3f},{y:+.3f}",
                    (int(cu) - 60, int(cv_) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        self.overlay = overlay
        return found
