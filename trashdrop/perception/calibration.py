"""Pixel to table-frame conversion.

Everything in the cell sits on one flat surface at a known height, which is
what makes this cheap: a plane-to-plane homography is enough and no depth is
needed. That matters twice over here -- the camera may end up being an Android
phone rather than a depth camera, and stereo depth fails on exactly the items
this cell handles most (clear PET bottles).

Two implementations of the same interface:

* :class:`PinholeTopDown` -- the simulator's ideal camera, where the intrinsics
  are known exactly because we wrote them.
* :class:`HomographyCalibration` -- the real one. Four markers at known table
  coordinates give a 3x3 homography, so nothing about lens, focal length,
  height or tilt has to be known in advance. Re-run it in seconds if the
  camera is bumped, and re-run it whenever the camera moves at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class PlaneCalibration(Protocol):
    """Maps image pixels to metres in the shared table frame."""

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]: ...

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class PinholeTopDown:
    """Ideal camera looking straight down at a plane. Simulation only."""

    height: float
    center_x: float
    center_y: float
    fovy_degrees: float
    width: int
    height_px: int

    @property
    def metres_per_pixel(self) -> float:
        return 2.0 * self.height * np.tan(np.deg2rad(self.fovy_degrees) / 2) / self.height_px

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        scale = self.metres_per_pixel
        x = self.center_x + (u - self.width / 2 + 0.5) * scale
        # Image rows increase downward; the table's +y points away from it.
        y = self.center_y - (v - self.height_px / 2 + 0.5) * scale
        return float(x), float(y)

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        scale = self.metres_per_pixel
        u = (x - self.center_x) / scale + self.width / 2 - 0.5
        v = (self.center_y - y) / scale + self.height_px / 2 - 0.5
        return float(u), float(v)


class HomographyCalibration:
    """Plane homography from four reference points. Use this on hardware.

    ``image_points`` are pixel coordinates of four markers, ``world_points``
    their table coordinates in metres, in the same order. ArUco markers taped
    at the pick-zone corners are the intended source, but any four known,
    non-collinear points work -- corners of a printed sheet included.
    """

    def __init__(
        self,
        image_points: np.ndarray | list,
        world_points: np.ndarray | list,
    ) -> None:
        import cv2

        image = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        world = np.asarray(world_points, dtype=np.float32).reshape(-1, 2)
        if image.shape[0] < 4 or image.shape != world.shape:
            raise ValueError("Need at least four matching image/world points")

        self.image_points = image
        self.world_points = world
        matrix, _ = cv2.findHomography(image, world, method=0)
        if matrix is None:
            raise ValueError(
                "Homography could not be computed; the four points are "
                "probably collinear or duplicated"
            )
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)

    @staticmethod
    def _apply(matrix: np.ndarray, a: float, b: float) -> tuple[float, float]:
        vector = matrix @ np.array([a, b, 1.0])
        if abs(vector[2]) < 1e-12:
            raise ValueError("Degenerate homography result")
        return float(vector[0] / vector[2]), float(vector[1] / vector[2])

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        return self._apply(self.matrix, u, v)

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return self._apply(self.inverse, x, y)

    def residuals_mm(self) -> list[float]:
        """Reprojection error at each reference point, in millimetres.

        Check this after every calibration. Anything above a few millimetres
        means a marker was misidentified or the surface is not flat, and the
        arm will miss grasps by that much.
        """

        errors = []
        for (u, v), (x, y) in zip(self.image_points, self.world_points):
            px, py = self.pixel_to_world(float(u), float(v))
            errors.append(float(np.hypot(px - x, py - y) * 1000.0))
        return errors

    # --- persistence -------------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "image_points": self.image_points.tolist(),
                    "world_points": self.world_points.tolist(),
                    "residuals_mm": self.residuals_mm(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "HomographyCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["image_points"], payload["world_points"])


def detect_aruco_corners(image, marker_ids: list[int], dictionary: str = "DICT_4X4_50"):
    """Find the centres of four ArUco markers, in the order of ``marker_ids``.

    Returns ``None`` when any requested marker is missing, so a caller can ask
    the operator to fix the lighting or the marker placement rather than
    calibrating against whatever happened to be visible.
    """

    import cv2

    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary)
    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(dict_id), aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None
    found = {int(marker_id): corner.reshape(4, 2) for marker_id, corner in zip(ids.flatten(), corners)}
    if not set(marker_ids).issubset(found):
        return None
    return np.array([found[marker_id].mean(axis=0) for marker_id in marker_ids], dtype=np.float32)
