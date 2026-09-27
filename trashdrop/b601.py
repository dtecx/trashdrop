"""Spectacles hand input for the centrally mounted reBot B601-RS.

The coordinates here are displacements in the Lens world, not robot poses.
The motor driver owns the physical limits and the separate LIVE gate.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .spectacles import TURN_DEG_PER_CM, facing_frame

JAW_PER_CM = 0.1  # thumb-pinky touch: the jaw's opening changes this share of its travel a sideways centimetre
TILT_DEG_PER_CM = 5.0  # thumb-ring touch: the jaw tilts this much a sideways centimetre (right: down)
# Which finger on the thumb (the index well apart) makes which sideways gesture.
SIDEWAYS = {"middle": "turn", "ring": "tilt", "pinky": "grip"}


def jaw_frame(heading: float, pitch: float) -> np.ndarray:
    """The tool frame with the jaw pointing ``pitch`` degrees below the horizontal (0 level, -90 straight
    down) towards ``heading`` degrees (0 forward, +x of the base; positive to the left), its fingers
    level.

    The jaw points along the gripper_end frame's +x: the gripper's own mass lies behind it, along -x
    (URDF); heading 0 and pitch 0 is the URDF zero pose's tool frame. Straight down, the heading is
    the jaw's turn about the vertical.
    """

    psi, phi = math.radians(heading), math.radians(pitch)
    approach = np.array([math.cos(phi) * math.cos(psi), math.cos(phi) * math.sin(psi), math.sin(phi)])
    across = np.array([-math.sin(psi), math.cos(psi), 0.0])
    return np.column_stack((approach, across, np.cross(approach, across)))


def heading_and_pitch(rotation) -> tuple[float, float]:
    """The heading and pitch (degrees) of jaw_frame nearest this tool frame."""

    rotation = np.asarray(rotation, dtype=float)
    approach, across = rotation[:, 0], rotation[:, 1]
    return (math.degrees(math.atan2(-across[0], across[1])),
            math.degrees(math.asin(min(max(approach[2], -1.0), 1.0))))


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
    """Separate translation and orientation clutches in the robot base frame, as the SO-101 pinch mode.

    Index pinch translates from the palm centre, rather than moving fingertips.
    Thumb-middle touch turns the tool about the vertical without translating
    it, TURN_DEG_PER_CM for each centimetre the hand goes sideways (right:
    clockwise from above), as the SO-101 wrist roll turns. It used to follow
    the whole hand's orientation from the knuckles, which jitters. Coupling
    both to an index pinch made normal wrist motion exceed the orientation
    limit and silently blocked lifting. Thumb-pinky touch sets the jaw the same
    way, JAW_PER_CM of its travel a centimetre (right closes, left opens),
    into ``jaw`` (0 closed, 1 open); it used to toggle open and shut. Missing
    tracking releases any clutch. Either hand drives: between gestures,
    whichever starts one (it was the right hand only). Thumb-ring touch tilts
    the jaw the same way into ``tilt``, TILT_DEG_PER_CM a centimetre, right
    down, left up (the user). The jaw's pitch had followed the hand's own tilt
    while dragging: a pinching hand points a little up, which asked for a level
    jaw the folded arm could not give, and the arm leant back and rose.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.ready = False
        self.anchor: tuple[np.ndarray, np.ndarray] | None = None  # palm point, wearer-to-base mapping
        self.gesture = "free"
        self.anchor_generation = 0
        self.side = "right"  # the hand that drives: between gestures, whichever starts one
        self._touch: tuple[str, float] | None = None  # the finger on the thumb, and since when
        self.jaw = 0.0  # how far open the jaw is to be: 0 closed, 1 open
        self._jaw_start = 0.0
        self.tilt = 0.0  # during the tilt gesture: how far further down the jaw is to point, degrees
        self.state = "release pinch to arm"

    @staticmethod
    def _read(hand) -> dict | None:
        """A tracked, sound hand's palm centre and finger gaps (cm); None for anything less."""

        if not isinstance(hand, dict) or not hand.get("tracked"):
            return None
        try:
            points = {name: np.asarray(hand[name], dtype=float)
                      for name in ("thumb", "index", "middleTip", "wrist", "middleKnuckle")}
        except (KeyError, TypeError, ValueError):
            return None
        if any(point.shape != (3,) or not np.isfinite(point).all() for point in points.values()):
            return None
        thumb = points["thumb"]
        try:
            pinky = np.asarray(hand["pinkyTip"], dtype=float)
            pinky_gap = float(np.linalg.norm(thumb - pinky)) if pinky.shape == (3,) else math.inf
        except (KeyError, TypeError, ValueError):
            pinky_gap = math.inf
        index_gap = float(np.linalg.norm(thumb - points["index"]))
        middle_gap = float(np.linalg.norm(thumb - points["middleTip"]))
        try:
            ring = np.asarray(hand["ringTip"], dtype=float)
            ring_gap = float(np.linalg.norm(thumb - ring)) if ring.shape == (3,) else math.inf
        except (KeyError, TypeError, ValueError):
            ring_gap = math.inf
        detected = hand.get("pinch")
        pinched = detected if isinstance(detected, bool) else index_gap < 2.5
        return {"point": (points["wrist"] + points["middleKnuckle"]) / 2, "pinched": pinched,
                "index_gap": index_gap, "middle_gap": middle_gap,
                "ring_gap": ring_gap if math.isfinite(ring_gap) else math.inf,
                "pinky_gap": pinky_gap if math.isfinite(pinky_gap) else math.inf}

    @staticmethod
    def _touching(read: dict) -> str | None:
        """The finger (middle, ring or pinky) on the thumb, clearly nearer it than the next; the index
        well apart and no pinch. Ring and pinky tips lie close together: nearest is not enough."""

        if read["pinched"] or read["index_gap"] <= 4.0:
            return None
        (nearest, finger), (second, _) = sorted((read[f"{name}_gap"], name) for name in SIDEWAYS)[:2]
        return finger if nearest < 2.5 and second - nearest > 0.8 else None

    def update(self, packet: dict | None, age: float, *, now: float | None = None
               ) -> tuple[np.ndarray, np.ndarray] | None:
        now = time.monotonic() if now is None else now
        fresh = isinstance(packet, dict) and age <= 0.3
        head = packet.get("head") if isinstance(packet, dict) else None
        if fresh and self.anchor is None:  # between gestures either hand may take over
            other = "left" if self.side == "right" else "right"
            mine, theirs = self._read(packet.get(self.side)), self._read(packet.get(other))
            busy = lambda read: read is not None and (read["pinched"] or self._touching(read) is not None)
            if theirs is not None and (mine is None or (busy(theirs) and not busy(mine))):
                self.side, self._touch = other, None
        read = self._read(packet.get(self.side)) if fresh else None
        try:
            look = np.asarray(head["look"], dtype=float)
            eye = np.asarray(head["p"], dtype=float)
            if look.shape != (3,) or eye.shape != (3,) or not np.isfinite(np.concatenate((look, eye))).all():
                raise ValueError("invalid head")
        except (KeyError, TypeError, ValueError):
            read = None
        if read is None:
            self.anchor = None
            self.ready = False
            self.gesture = "free"
            self._touch = None
            self.state = "hand lost: holding"
            return None

        point, pinched = read["point"], read["pinched"]
        finger = self._touching(read)
        if finger is None or self._touch is None or self._touch[0] != finger:
            self._touch = (finger, now) if finger is not None else None
        held = self._touch is not None and now - self._touch[1] >= 0.15
        own = {gesture: name for name, gesture in SIDEWAYS.items()}.get(self.gesture)
        going_on = (own is not None and not pinched and read["index_gap"] > 3.5 and read[f"{own}_gap"] < 3.5)
        gesture = ("drag" if pinched else self.gesture if going_on else
                   SIDEWAYS[self._touch[0]] if held else "free")
        if gesture == "free":
            self.anchor = None
            self.gesture = "free"
            self.ready = True
            self.state = ({"middle": "hold thumb-middle to turn", "ring": "hold thumb-ring to tilt",
                           "pinky": "hold thumb-pinky to set the jaw"}[finger] if finger is not None
                          else "free: index pinch to move")
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
        if self.anchor is None:
            wearer = facing_frame(look, eye, point)
            # The robot's +X points forward, -Y to the wearer's right, +Z up.
            mapping = np.stack((wearer[0], -wearer[1], wearer[2]))
            self.anchor = point, mapping
            self.anchor_generation += 1
            self.gesture = gesture
            self._jaw_start = self.jaw
            self.tilt = 0.0
            self.state = {"drag": "drag: move palm up/forward/sideways", "turn": "turn: move the hand sideways",
                          "tilt": "tilt: move the hand sideways, right points the jaw down",
                          "grip": "jaw: move the hand sideways, right closes"}[gesture]
            return np.zeros(3), np.eye(3)
        start, mapping = self.anchor
        moved = mapping @ (point - start)  # cm: forward, left, up
        if gesture == "drag":
            self.state = "drag: move palm up/forward/sideways"
            return moved * (self.scale / 100), np.eye(3)
        if gesture == "tilt":  # the arm holds its place while the jaw tilts
            self.tilt = -TILT_DEG_PER_CM * moved[1]  # moved[1] is to the left
            self.state = f"tilt: {abs(self.tilt):.0f}° {'down' if self.tilt >= 0 else 'up'}; right points down"
            return np.zeros(3), np.eye(3)
        if gesture == "grip":  # the arm holds while the jaw is set
            self.jaw = min(max(self._jaw_start + JAW_PER_CM * moved[1], 0.0), 1.0)
            self.state = f"jaw: {self.jaw * 100:.0f}% open; right closes, left opens"
            return np.zeros(3), np.eye(3)
        angle = math.radians(TURN_DEG_PER_CM) * moved[1]  # left: anticlockwise from above
        cos, sin = math.cos(angle), math.sin(angle)
        self.state = f"turn: {abs(math.degrees(angle)):.0f}° {'cw' if angle < 0 else 'ccw'}; position held"
        return np.zeros(3), np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
