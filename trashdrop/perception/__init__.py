"""Perception: find items on the table and say what they are made of.

The pipeline is deliberately two-stage:

    frame -> [1] class-agnostic detector -> crop -> [2] crop classifier

Stage 1 answers "something is lying there", which on a fixed camera over a
plain surface needs no training at all -- background subtraction does it. Stage
2 answers "what is it", and needs only a few hundred crops per class because it
never has to learn localisation. The split also makes the honest answer cheap:
a low classifier score becomes a mixed-bin routing instead of a guess.

``ColorDetector`` short-circuits both stages and exists only so the motion
stack can be exercised without a trained model. It reads the colour that the
simulator itself assigned, so it proves nothing about perception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """One item found on the table, in the shared frame.

    ``width`` is the short side of the item's minimum-area box in metres --
    the dimension the jaws have to close across, so it decides graspability.
    """

    category: str
    x: float
    y: float
    yaw: float
    confidence: float = 1.0
    width: float = 0.0
    length: float = 0.0
    pixel: tuple[int, int] = (0, 0)
    box: np.ndarray | None = field(default=None, repr=False)
    mask: np.ndarray | None = field(default=None, repr=False)


class Detector(Protocol):
    """Anything that turns an RGB frame into table-frame detections."""

    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class CropClassifier(Protocol):
    """Anything that names the material in a cropped image of one item."""

    def classify(self, crop: np.ndarray) -> tuple[str, float]: ...


from .background import BackgroundDetector  # noqa: E402
from .calibration import (  # noqa: E402
    HomographyCalibration,
    PinholeTopDown,
    PlaneCalibration,
    detect_aruco_corners,
)
from .classifier import ConstantClassifier, OnnxCropClassifier  # noqa: E402
from .color import ColorDetector  # noqa: E402

__all__ = [
    "BackgroundDetector",
    "ColorDetector",
    "ConstantClassifier",
    "CropClassifier",
    "Detection",
    "Detector",
    "HomographyCalibration",
    "OnnxCropClassifier",
    "PinholeTopDown",
    "PlaneCalibration",
    "detect_aruco_corners",
]
