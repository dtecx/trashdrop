"""Where each arm stands relative to the marker sheet.

The overhead camera finds the sheet's four markers and so can say where an
item is on the sheet. Each arm learns where the sheet is by touching the
marker centres with the tip of its fixed finger: the kinematics say where the
fingertip was in the arm's own frame, the sheet says where the markers are.
A rotation and a shift in the table plane, fitted to those pairs, turns
sheet coordinates into the arm's.

Everything here is centimetres and degrees, and plain numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Placement:
    """arm_xy = R(yaw) @ sheet_xy + (x, y), in cm; ``table_z`` is the sheet's height in the arm's frame."""

    x: float
    y: float
    yaw: float
    table_z: float = 0.0

    def to_arm(self, sheet_xy) -> np.ndarray:
        angle = np.radians(self.yaw)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return rotation @ np.asarray(sheet_xy, dtype=float) + np.array([self.x, self.y])

    def direction_to_arm(self, sheet_degrees: float) -> float:
        """A direction on the sheet, as an angle in the arm's frame."""

        return (sheet_degrees + self.yaw + 180.0) % 360.0 - 180.0


def fit_placement(sheet_points, arm_points, table_z: float = 0.0) -> tuple[Placement, list[float]]:
    """Best rotation and shift from sheet to arm (no scale); residuals in cm per point."""

    sheet = np.asarray(sheet_points, dtype=float)
    arm = np.asarray(arm_points, dtype=float)
    if len(sheet) < 2 or sheet.shape != arm.shape:
        raise ValueError("need at least two matching points")
    sheet_mean, arm_mean = sheet.mean(axis=0), arm.mean(axis=0)
    a, b = sheet - sheet_mean, arm - arm_mean
    # 2-D Kabsch: the angle that best turns the sheet's spread into the arm's.
    angle = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = arm_mean - rotation @ sheet_mean
    placement = Placement(float(shift[0]), float(shift[1]), float(np.degrees(angle)), table_z)
    residuals = [float(np.linalg.norm(placement.to_arm(s) - p)) for s, p in zip(sheet, arm)]
    return placement, residuals
