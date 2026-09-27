"""Safe Spectacles input preview for a centrally mounted reBot B601-DM.

The coordinates here are displacements in the Lens world, not robot poses.
No B601 motor driver is wired until its zero, base transform, limits and
clearance from the two SO-101 arms have been measured on the actual rig.
"""

from __future__ import annotations

import math


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
        self.state = "waiting for the hand"

    def update(self, packet: dict | None, age: float) -> dict:
        """Return an honest preview guide from a fresh tracked hand packet."""

        hand = packet.get(self.hand) if isinstance(packet, dict) and age <= 0.5 else None
        if not isinstance(hand, dict) or not hand.get("tracked"):
            self._anchor = None
            self.state = "hand lost: held"
            return self.guide()
        try:
            thumb = [float(value) for value in hand["thumb"]]
            index = [float(value) for value in hand["index"]]
            if (len(thumb) != 3 or len(index) != 3 or
                    not all(math.isfinite(value) for value in thumb + index)):
                raise ValueError("invalid hand point")
        except (KeyError, TypeError, ValueError):
            self._anchor = None
            self.state = "hand tracking incomplete: held"
            return self.guide()
        point = [(a + b) / 2 for a, b in zip(thumb, index)]
        gap = math.dist(thumb, index)
        pinched = hand.get("pinch")
        dragging = pinched if isinstance(pinched, bool) else gap < 2.5
        if not dragging:
            self._anchor = None
            self.state = "free: pinch to preview movement"
        else:
            if self._anchor is None:
                self._anchor = point, self.offset.copy()
            start, origin = self._anchor
            self.offset = [round(origin[i] + self.scale * (point[i] - start[i]), 1) for i in range(3)]
            self.state = "dragging preview only"
        return self.guide()

    def guide(self) -> dict:
        return {"mode": "moving" if self._anchor else "holding", "state": self.state,
                "offset": self.offset.copy(), "blocked": []}

