"""Safe Spectacles input preview for a centrally mounted reBot B601-RS.

The coordinates here are displacements in the Lens world, not robot poses.
No B601 motor driver is wired until its zero, base transform, limits and
clearance from the two SO-101 arms have been measured on the actual rig.
"""

from __future__ import annotations

import math

import numpy as np

from .spectacles import facing_frame


def _hand_frame(hand: dict) -> list[list[float]] | None:
    """A right-handed frame from the wrist and knuckles, if tracking is sound."""

    try:
        points = {key: [float(value) for value in hand[key]]
                  for key in ("wrist", "indexKnuckle", "middleKnuckle", "pinkyKnuckle")}
    except (KeyError, TypeError, ValueError):
        return None
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point)
           for point in points.values()):
        return None

    def unit(vector: list[float]) -> list[float] | None:
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length > 0.01 else None

    across = unit([a - b for a, b in zip(points["indexKnuckle"], points["pinkyKnuckle"])])
    along = unit([a - b for a, b in zip(points["middleKnuckle"], points["wrist"])])
    if across is None or along is None:
        return None
    normal = unit([across[1] * along[2] - across[2] * along[1],
                   across[2] * along[0] - across[0] * along[2],
                   across[0] * along[1] - across[1] * along[0]])
    if normal is None:
        return None
    along = [normal[1] * across[2] - normal[2] * across[1],
             normal[2] * across[0] - normal[0] * across[2],
             normal[0] * across[1] - normal[1] * across[0]]
    return [across, along, normal]


def _relative_rpy(anchor: list[list[float]], current: list[list[float]]) -> list[float]:
    """Hand-frame roll, pitch and yaw in degrees, relative to a clutch start."""

    rotation = [[sum(anchor[i][k] * current[j][k] for k in range(3))
                 for j in range(3)] for i in range(3)]
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    roll = math.atan2(rotation[2][1], rotation[2][2])
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    return [round(math.degrees(angle), 1) for angle in (roll, pitch, yaw)]


class B601Preview:
    """Clutched one-hand translation preview; never sends a motor command."""

    def __init__(self, *, hand: str = "right", scale: float = 1.0) -> None:
        if hand not in ("left", "right"):
            raise ValueError("hand must be left or right")
        if not 0 < scale <= 3:
            raise ValueError("scale must be above 0 and at most 3")
        self.hand = hand
        self.scale = scale
        self.offset = [0.0, 0.0, 0.0]
        self._anchor: tuple[list[float], list[float]] | None = None
        self._rotation_anchor: list[list[float]] | None = None
        self.rotation_deg = [0.0, 0.0, 0.0]
        self.state = "waiting for the hand"

    def update(self, packet: dict | None, age: float) -> dict:
        """Return an honest preview guide from a fresh tracked hand packet."""

        hand = packet.get(self.hand) if isinstance(packet, dict) and age <= 0.5 else None
        if not isinstance(hand, dict) or not hand.get("tracked"):
            self._anchor = self._rotation_anchor = None
            self.state = "hand lost: held"
            return self.guide()
        try:
            thumb = [float(value) for value in hand["thumb"]]
            index = [float(value) for value in hand["index"]]
            if (len(thumb) != 3 or len(index) != 3 or
                    not all(math.isfinite(value) for value in thumb + index)):
                raise ValueError("invalid hand point")
        except (KeyError, TypeError, ValueError):
            self._anchor = self._rotation_anchor = None
            self.state = "hand tracking incomplete: held"
            return self.guide()
        point = [(a + b) / 2 for a, b in zip(thumb, index)]
        gap = math.dist(thumb, index)
        pinched = hand.get("pinch")
        dragging = pinched if isinstance(pinched, bool) else gap < 2.5
        if not dragging:
            self._anchor = self._rotation_anchor = None
            self.state = "free: pinch to preview movement"
        else:
            if self._anchor is None:
                self._anchor = point, self.offset.copy()
                self._rotation_anchor = _hand_frame(hand)
            start, origin = self._anchor
            self.offset = [round(origin[i] + self.scale * (point[i] - start[i]), 1) for i in range(3)]
            frame = _hand_frame(hand)
            if self._rotation_anchor is not None and frame is not None:
                self.rotation_deg = _relative_rpy(self._rotation_anchor, frame)
            self.state = "dragging preview only"
        return self.guide()

    def guide(self) -> dict:
        return {"mode": "moving" if self._anchor else "holding", "state": self.state,
                "offset": self.offset.copy(), "rotation": self.rotation_deg.copy(), "blocked": []}


class B601HandMotion:
    """A clutched six-axis hand displacement, expressed in the robot base frame.

    This class does not solve IK or command hardware. One unpinched hand packet
    is required before the first clutch, so releasing a UIKit button cannot
    become an arm movement. Missing tracking immediately releases the clutch.
    """

    def __init__(self, scale: float = 0.5) -> None:
        self.scale = scale
        self.ready = False
        self.anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.state = "release pinch to arm"

    def update(self, packet: dict | None, age: float) -> tuple[np.ndarray, np.ndarray] | None:
        hand = packet.get("right") if isinstance(packet, dict) and age <= 0.3 else None
        head = packet.get("head") if isinstance(packet, dict) else None
        if not isinstance(hand, dict) or not hand.get("tracked") or not isinstance(head, dict):
            self.anchor = None
            self.ready = False
            self.state = "hand lost: holding"
            return None
        try:
            thumb = np.asarray(hand["thumb"], dtype=float)
            index = np.asarray(hand["index"], dtype=float)
            look = np.asarray(head["look"], dtype=float)
            eye = np.asarray(head["p"], dtype=float)
            if any(value.shape != (3,) or not np.isfinite(value).all()
                   for value in (thumb, index, look, eye)):
                raise ValueError("invalid tracking vector")
        except (KeyError, TypeError, ValueError):
            self.anchor = None
            self.ready = False
            self.state = "tracking incomplete: holding"
            return None
        point = (thumb + index) / 2
        frame = _hand_frame(hand)
        if frame is None:
            self.anchor = None
            self.ready = False
            self.state = "knuckles missing: holding"
            return None
        frame = np.asarray(frame, dtype=float)
        pinch = hand.get("pinch")
        pinched = pinch if isinstance(pinch, bool) else float(np.linalg.norm(thumb - index)) < 2.5
        if not pinched:
            self.anchor = None
            self.ready = True
            self.state = "free: pinch to move"
            return None
        if not self.ready:
            self.state = "release pinch to arm"
            return None
        if self.anchor is None:
            wearer = facing_frame(look, eye, point)
            # The robot's +X points forward, -Y to the wearer's right, +Z up.
            mapping = np.stack((wearer[0], -wearer[1], wearer[2]))
            self.anchor = point, frame, mapping
            self.state = "clutch set: move slowly"
            return np.zeros(3), np.eye(3)
        start, first_frame, mapping = self.anchor
        delta_m = mapping @ (point - start) * (self.scale / 100)
        rotation_world = frame.T @ first_frame
        rotation_robot = mapping @ rotation_world @ mapping.T
        self.state = "following right hand"
        return delta_m, rotation_robot
