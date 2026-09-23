"""A physical pick area calibrated into camera pixels with the printed sheet."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_ZONE_CONFIG = Path("capture_zone.toml")
DEFAULT_CALIBRATION = Path("camera_zone.json")


@dataclass(frozen=True)
class DetectionZone:
    frame_width: int
    frame_height: int
    x: int
    y: int
    width: int
    height: int
    corners: tuple[tuple[float, float], ...] | None = None

    def matches(self, frame) -> bool:
        return frame.shape[1] == self.frame_width and frame.shape[0] == self.frame_height

    def crop(self, frame):
        if not self.matches(frame):
            raise ValueError("camera resolution changed since the detection zone was selected")
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]

    def analysis_mask(self):
        """Polygon of valid pixels inside the enclosing camera-space crop."""

        import cv2

        mask = np.full((self.height, self.width), 255, np.uint8)
        if self.corners is not None:
            mask[:] = 0
            points = np.rint(np.asarray(self.corners) - (self.x, self.y)).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

    def preview_corners(self, preview_shape: tuple[int, ...]):
        height, width = preview_shape[:2]
        points = self.corners or (
            (self.x, self.y), (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height), (self.x, self.y + self.height),
        )
        return np.rint(np.asarray(points) * (width / self.frame_width, height / self.frame_height)).astype(np.int32)


def zone_size(path: Path = DEFAULT_ZONE_CONFIG) -> tuple[float, float]:
    values = tomllib.loads(Path(path).read_text(encoding="utf-8"))["zone"]
    width, height = float(values["width_cm"]), float(values["height_cm"])
    if values.get("center") != "camera" or not (0 < width <= 100 and 0 < height <= 100):
        raise ValueError(f"Invalid physical zone in {path}")
    return width, height


def calibrate_zone(frame, width_cm: float, height_cm: float) -> DetectionZone:
    """Project a camera-centred metric rectangle through the sheet homography."""

    from ..camera.markers import MARKER_WORLD
    from ..perception.calibration import HomographyCalibration, detect_aruco_corners

    ids = [0, 1, 2, 3]
    image_points = detect_aruco_corners(frame, ids)
    if image_points is None:
        raise ValueError("All four ArUco markers (0-3) must be visible for zone calibration")
    calibration = HomographyCalibration(image_points, [MARKER_WORLD[i] for i in ids])
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = calibration.pixel_to_world(frame_width / 2, frame_height / 2)
    half_w, half_h = width_cm / 200, height_cm / 200
    corners = tuple(calibration.world_to_pixel(x, y) for x, y in (
        (center_x - half_w, center_y + half_h),
        (center_x + half_w, center_y + half_h),
        (center_x + half_w, center_y - half_h),
        (center_x - half_w, center_y - half_h),
    ))
    points = np.asarray(corners)
    if not np.isfinite(points).all():
        raise ValueError("Marker calibration produced non-finite zone corners")
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    if x0 < 0 or y0 < 0 or x1 > frame_width or y1 > frame_height:
        raise ValueError(f"{width_cm:g} x {height_cm:g} cm zone does not fit in the camera image; widen the view")
    if x1 - x0 < 40 or y1 - y0 < 40:
        raise ValueError("Calibrated zone is too small in pixels")
    return DetectionZone(frame_width, frame_height, int(x0), int(y0),
                         int(x1 - x0), int(y1 - y0), corners)


def load_zone(path: Path) -> DetectionZone | None:
    if not path.is_file():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    zone = DetectionZone(**values)
    if (
        min(zone.frame_width, zone.frame_height, zone.width, zone.height) <= 0
        or zone.x < 0
        or zone.y < 0
        or zone.x + zone.width > zone.frame_width
        or zone.y + zone.height > zone.frame_height
    ):
        raise ValueError(f"Invalid detection zone in {path}")
    return zone


def save_zone(path: Path, zone: DetectionZone) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(zone), indent=2) + "\n", encoding="utf-8")
