"""Spectacles hand input for the centrally mounted reBot B601-RS.

The coordinates here are displacements in the Lens world, not robot poses.
The motor driver owns the physical limits and the separate LIVE gate.
"""

from __future__ import annotations

import math
import time

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
    """Separate translation and orientation clutches in the robot base frame.

    Index pinch translates from the palm centre, rather than moving fingertips.
    Thumb-middle touch rotates the tool without translating it. Coupling both
    to an index pinch made normal wrist motion exceed the orientation limit and
    silently blocked lifting. Missing tracking releases either clutch.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.ready = False
        self.anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.gesture = "free"
        self.anchor_generation = 0
        self._middle_since: float | None = None
        self.state = "release pinch to arm"

    def update(self, packet: dict | None, age: float, *, now: float | None = None
               ) -> tuple[np.ndarray, np.ndarray] | None:
        now = time.monotonic() if now is None else now
        hand = packet.get("right") if isinstance(packet, dict) and age <= 0.3 else None
        head = packet.get("head") if isinstance(packet, dict) else None
        if not isinstance(hand, dict) or not hand.get("tracked") or not isinstance(head, dict):
            self.anchor = None
            self.ready = False
            self.gesture = "free"
            self._middle_since = None
            self.state = "hand lost: holding"
            return None
        try:
            thumb = np.asarray(hand["thumb"], dtype=float)
            index = np.asarray(hand["index"], dtype=float)
            middle = np.asarray(hand["middleTip"], dtype=float)
            wrist = np.asarray(hand["wrist"], dtype=float)
            knuckle = np.asarray(hand["middleKnuckle"], dtype=float)
            look = np.asarray(head["look"], dtype=float)
            eye = np.asarray(head["p"], dtype=float)
            if any(value.shape != (3,) or not np.isfinite(value).all()
                   for value in (thumb, index, middle, wrist, knuckle, look, eye)):
                raise ValueError("invalid tracking vector")
        except (KeyError, TypeError, ValueError):
            self.anchor = None
            self.ready = False
            self.gesture = "free"
            self._middle_since = None
            self.state = "tracking incomplete: holding"
            return None

        point = (wrist + knuckle) / 2
        index_gap = float(np.linalg.norm(thumb - index))
        middle_gap = float(np.linalg.norm(thumb - middle))
        detected = hand.get("pinch")
        pinched = detected if isinstance(detected, bool) else index_gap < 2.5
        middle_touch = not pinched and index_gap > 4.0 and middle_gap < 2.5
        if middle_touch:
            if self._middle_since is None:
                self._middle_since = now
        else:
            self._middle_since = None
        turning = (self.gesture == "turn" and not pinched and index_gap > 3.5 and middle_gap < 3.5)
        turning = turning or (middle_touch and self._middle_since is not None and
                              now - self._middle_since >= 0.15)
        gesture = "drag" if pinched else "turn" if turning else "free"
        if gesture == "free":
            self.anchor = None
            self.gesture = "free"
            self.ready = True
            self.state = "hold thumb-middle to turn" if middle_touch else "free: index pinch to move"
            return None
        if not self.ready:
            self.state = "release pinch to arm"
            return None
        if gesture != self.gesture and self.anchor is not None:
            self.anchor = None
            self.ready = False
            self.gesture = "free"
            self.state = "release between gestures"
            return None
        frame = _hand_frame(hand) if gesture == "turn" else None
        if gesture == "turn" and frame is None:
            self.anchor = None
            self.ready = False
            self.state = "knuckles missing: holding"
            return None
        orientation = np.asarray(frame, dtype=float) if frame is not None else np.eye(3)
        if self.anchor is None:
            wearer = facing_frame(look, eye, point)
            # The robot's +X points forward, -Y to the wearer's right, +Z up.
            mapping = np.stack((wearer[0], -wearer[1], wearer[2]))
            self.anchor = point, orientation, mapping
            self.anchor_generation += 1
            self.gesture = gesture
            self.state = "drag: move palm up/forward/sideways" if gesture == "drag" else "turn: rotate hand"
            return np.zeros(3), np.eye(3)
        start, first_frame, mapping = self.anchor
        if gesture == "drag":
            self.state = "drag: move palm up/forward/sideways"
            return mapping @ (point - start) * (self.scale / 100), np.eye(3)
        rotation_world = orientation.T @ first_frame
        self.state = "turn: rotate hand; position held"
        return np.zeros(3), mapping @ rotation_world @ mapping.T
