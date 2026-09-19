"""Class-agnostic detector for the real cell: what changed against the table.

The camera is fixed and the surface is plain, so "is something lying there" is
answered by comparing against a reference photo of the empty table. This needs
no training data at all, which is the whole point -- all of the labelling
effort then goes into the crop classifier, which needs far less of it.

Two practical notes:

* Keep one reference frame per lighting condition and re-shoot it whenever the
  lights change. A stale background is the main failure mode.
* Shadows read as change. ``shadow_value_ratio`` suppresses regions that only
  got darker without changing colour, which is what a shadow does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..station import MIXED_CATEGORY, PICK_ZONE
from . import Detection
from .calibration import PlaneCalibration


class BackgroundDetector:
    """Find items by differencing against an empty-table reference.

    ``classifier`` names the material in each crop. Without one every item is
    reported as mixed with zero confidence, which is the correct behaviour
    before a model is trained: the cell still picks and still sorts, it just
    sorts everything into the mixed bin.
    """

    def __init__(
        self,
        background: np.ndarray,
        calibration: PlaneCalibration,
        classifier=None,
        *,
        diff_threshold: int = 28,
        min_area_px: int = 300,
        shadow_value_ratio: float = 0.65,
        crop_padding_px: int = 8,
    ) -> None:
        if background is None:
            raise ValueError("A background reference frame is required")
        self.background = background
        self.calibration = calibration
        self.classifier = classifier
        self.diff_threshold = diff_threshold
        self.min_area_px = min_area_px
        self.shadow_value_ratio = shadow_value_ratio
        self.crop_padding_px = crop_padding_px
        self.overlay: np.ndarray | None = None

    @classmethod
    def from_file(cls, path: Path, calibration: PlaneCalibration, classifier=None, **kwargs):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read background frame: {path}")
        return cls(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), calibration, classifier, **kwargs)

    # --- internals ---------------------------------------------------------

    def _foreground_mask(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if rgb.shape != self.background.shape:
            raise ValueError(
                f"Frame {rgb.shape} does not match the background "
                f"{self.background.shape}; re-shoot the reference at the same "
                "resolution as the live feed"
            )
        blur_live = cv2.GaussianBlur(rgb, (5, 5), 0)
        blur_reference = cv2.GaussianBlur(self.background, (5, 5), 0)
        difference = cv2.absdiff(blur_live, blur_reference).max(axis=2)
        mask = (difference > self.diff_threshold).astype(np.uint8) * 255

        # Drop shadow: darker than the reference but the same hue.
        live_v = cv2.cvtColor(blur_live, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
        reference_v = cv2.cvtColor(blur_reference, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
        ratio = np.divide(live_v, np.maximum(reference_v, 1.0))
        shadow = (ratio > self.shadow_value_ratio) & (ratio < 1.0)
        hue_live = cv2.cvtColor(blur_live, cv2.COLOR_RGB2HSV)[:, :, 0].astype(np.int16)
        hue_reference = cv2.cvtColor(blur_reference, cv2.COLOR_RGB2HSV)[:, :, 0].astype(np.int16)
        same_hue = np.abs(hue_live - hue_reference) < 8
        mask[shadow & same_hue] = 0

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _crop(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        pad = self.crop_padding_px
        x0 = max(int(box[:, 0].min()) - pad, 0)
        x1 = min(int(box[:, 0].max()) + pad, rgb.shape[1])
        y0 = max(int(box[:, 1].min()) - pad, 0)
        y1 = min(int(box[:, 1].max()) + pad, rgb.shape[0])
        return rgb[y0:y1, x0:x1]

    # --- public ------------------------------------------------------------

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        import cv2

        mask = self._foreground_mask(rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        metres_per_pixel = self._metres_per_pixel()
        found: list[Detection] = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area_px:
                continue
            (cu, cv_), (w, h), angle = cv2.minAreaRect(contour)
            x, y = self.calibration.pixel_to_world(cu, cv_)
            if not PICK_ZONE.contains(x, y, margin=0.005):
                continue

            box = np.intp(cv2.boxPoints(((cu, cv_), (w, h), angle)))
            crop = self._crop(rgb, box)
            if self.classifier is not None and crop.size:
                category, confidence = self.classifier.classify(crop)
            else:
                category, confidence = MIXED_CATEGORY, 0.0

            yaw = np.deg2rad(angle if w < h else angle + 90.0)
            short_px, long_px = (w, h) if w < h else (h, w)
            item_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(item_mask, [contour], -1, 255, -1)

            found.append(
                Detection(
                    category=category,
                    x=x,
                    y=y,
                    yaw=float((yaw + np.pi) % (2 * np.pi) - np.pi),
                    confidence=float(confidence),
                    width=float(short_px * metres_per_pixel),
                    length=float(long_px * metres_per_pixel),
                    pixel=(int(cu), int(cv_)),
                    box=box,
                    mask=item_mask,
                )
            )
            cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"{category} {confidence:.2f}",
                (int(cu) - 50, int(cv_) - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        self.overlay = overlay
        return found

    def _metres_per_pixel(self) -> float:
        """Local scale, so a detection can report a width in metres.

        A homography is not a uniform scale, so this measures it where it
        matters: across the middle of the pick zone.
        """

        direct = getattr(self.calibration, "metres_per_pixel", None)
        if direct is not None:
            return float(direct)
        min_x, max_x, _, _ = PICK_ZONE.bounds
        u0, v0 = self.calibration.world_to_pixel(min_x, PICK_ZONE.center_y)
        u1, v1 = self.calibration.world_to_pixel(max_x, PICK_ZONE.center_y)
        pixels = float(np.hypot(u1 - u0, v1 - v0))
        return (max_x - min_x) / pixels if pixels > 1e-6 else 0.0
