"""Take a frame by itself once the hand is out and the item has settled.

Shooting a dataset by hand has one annoying loop: place the item, walk back to
the keyboard, press a key, walk back to the item. Worse, the frame is often
taken with a hand still in shot, and autolabel has to throw it away.

The shutter watches the live feed and fires when three things are true at
once: nothing has moved for a moment, exactly one item is in view, and that
item is not in the same pose as the last frame taken. The operator just keeps
repositioning the item and stepping back.

Everything runs on a downscaled copy of the frame, so it keeps up with a 1080p
stream on a laptop. Motion is measured after removing the frame's mean, so an
auto-exposure step is not mistaken for someone touching the item.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..perception.photometric import compensated_reference
from ..perception.regions import find_item_region

# What the operator sees for each reason an item region was not found.
REASON_TEXT = {
    "nothing_changed": ("empty table - drop an item in the zone", "idle"),
    "touches_frame_edge": ("hand or item at the frame edge - step back", "problem"),
    "more_than_one_object": ("more than one object in view", "problem"),
    "region_too_large": ("huge change - did the light change? re-shoot background (b)", "problem"),
}


@dataclass
class ShutterView:
    """What the shutter decided about one frame, and what to show for it."""

    capture: bool
    status: str
    tone: str  # idle | wait | ready | problem
    box: tuple[int, int, int, int] | None = None  # full-resolution pixels


class AutoShutter:
    """Decide, frame by frame, whether this is a frame worth keeping."""

    def __init__(
        self,
        background: np.ndarray | None = None,
        *,
        analysis_width: int = 320,
        settle_seconds: float = 0.6,
        motion_threshold: float = 2.0,
        diff_threshold: int = 28,
        min_interval_seconds: float = 0.8,
        duplicate_iou: float = 0.85,
        duplicate_appearance: float = 10.0,
        clock=time.monotonic,
    ) -> None:
        self.analysis_width = analysis_width
        self.settle_seconds = settle_seconds
        self.motion_threshold = motion_threshold
        self.diff_threshold = diff_threshold
        self.min_interval_seconds = min_interval_seconds
        self.duplicate_iou = duplicate_iou
        self.duplicate_appearance = duplicate_appearance
        self.clock = clock

        self._background_shape: tuple[int, ...] | None = None
        self._reference_small: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._stable_since: float | None = None
        self._last_capture_time = -1e9
        self._last_mask: np.ndarray | None = None
        self._last_grey: np.ndarray | None = None
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        if background is not None:
            self.set_background(background)

    # --- state -------------------------------------------------------------

    def _small(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2

        height, width = frame.shape[:2]
        scale = self.analysis_width / width
        size = (self.analysis_width, max(1, int(round(height * scale))))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale

    def set_background(self, background: np.ndarray | None) -> None:
        """Adopt a new empty-table reference (or drop it with ``None``)."""

        import cv2

        if background is None:
            self._background_shape = None
            self._reference_small = None
            return
        small, _ = self._small(background)
        self._background_shape = background.shape
        self._reference_small = cv2.GaussianBlur(small, (5, 5), 0)
        self.forget_last()

    def forget_last(self) -> None:
        """Start a new object: the next settled pose is always worth taking."""

        self._last_mask = None
        self._last_grey = None
        self._pending = None

    def mark_captured(self) -> None:
        """Record that the frame just approved was actually saved."""

        if self._pending is not None:
            self._last_mask, self._last_grey = self._pending
            self._pending = None
        self._last_capture_time = self.clock()

    # --- the decision ------------------------------------------------------

    def update(self, frame: np.ndarray) -> ShutterView:
        import cv2

        small, scale = self._small(frame)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grey -= grey.mean()
        now = self.clock()

        if self._prev is None or self._prev.shape != grey.shape:
            self._prev = grey
            self._stable_since = None
            return ShutterView(False, "starting", "wait")
        motion = float(np.abs(grey - self._prev).mean())
        self._prev = grey
        if motion > self.motion_threshold:
            self._stable_since = None
            return ShutterView(False, "moving", "wait")
        if self._stable_since is None:
            self._stable_since = now

        if self._reference_small is None:
            return ShutterView(False, "no background yet: clear the table and press b", "problem")
        if self._background_shape != frame.shape:
            return ShutterView(
                False, "background was shot at another resolution: press b", "problem"
            )

        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        corrected, fit = compensated_reference(blurred, self._reference_small)
        if not fit.trusted:
            return ShutterView(
                False, "view changed too much: camera moved? re-shoot background (b)", "problem"
            )
        difference = cv2.absdiff(blurred, corrected).max(axis=2)
        mask = (difference > self.diff_threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        region, reason = find_item_region(mask, edge_margin_px=2)
        if region is None:
            text, tone = REASON_TEXT.get(reason, (reason, "problem"))
            return ShutterView(False, text, tone)

        box = tuple(int(round(value / scale)) for value in region.box)
        if now - self._stable_since < self.settle_seconds:
            return ShutterView(False, "hold still...", "wait", box)
        if now - self._last_capture_time < self.min_interval_seconds:
            return ShutterView(False, "hold still...", "wait", box)

        item_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(item_mask, [region.hull], -1, 255, -1)
        item_mask = item_mask > 0

        if self._last_mask is not None and self._last_mask.shape == item_mask.shape:
            union = np.logical_or(item_mask, self._last_mask)
            overlap = float(np.logical_and(item_mask, self._last_mask).sum()) / max(
                1.0, float(union.sum())
            )
            now_pixels = grey[union]
            then_pixels = self._last_grey[union]
            # Compare shapes of the brightness pattern, not absolute levels, so
            # a camera that re-exposed between the two does not count as a
            # new pose.
            appearance = float(
                np.abs((now_pixels - now_pixels.mean()) - (then_pixels - then_pixels.mean())).mean()
            )
            if overlap > self.duplicate_iou and appearance < self.duplicate_appearance:
                return ShutterView(False, "got it - move the item to a new pose", "ready", box)

        self._pending = (item_mask, grey.copy())
        return ShutterView(True, "snap", "ready", box)
