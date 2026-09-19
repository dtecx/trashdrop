"""Planar frame transforms between the shared table frame and each arm base.

Every arm in this cell is mounted flat on the table, so the only difference
between the shared frame and an arm's own frame is a translation plus a yaw.
Keeping that conversion in one place is what lets the inverse kinematics stay
identical to the single-arm baseline: the solver always receives a target
expressed the way the arm's own base sees it.

No MuJoCo import here on purpose -- a hardware adapter needs these transforms
just as much as the simulator does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(value: float) -> float:
    """Fold an angle into [-pi, pi).

    Note the half-open end: an input of exactly pi comes back as -pi. Both name
    the same heading, and this matches the convention the single-arm baseline
    used, so wrist-roll limits behave identically.
    """

    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    """A planar frame: origin in the shared table frame plus a yaw."""

    x: float
    y: float
    yaw: float = 0.0

    @classmethod
    def from_degrees(cls, x: float, y: float, yaw_degrees: float) -> "Pose2D":
        return cls(x=x, y=y, yaw=math.radians(yaw_degrees))

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        """Express a shared-frame point in this frame."""

        dx, dy = x - self.x, y - self.y
        cos, sin = math.cos(-self.yaw), math.sin(-self.yaw)
        return cos * dx - sin * dy, sin * dx + cos * dy

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        """Express a point of this frame in the shared frame."""

        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + cos * x - sin * y, self.y + sin * x + cos * y

    def yaw_to_local(self, yaw: float) -> float:
        """Convert a heading in the shared frame to this frame."""

        return wrap_angle(yaw - self.yaw)

    def yaw_to_world(self, yaw: float) -> float:
        return wrap_angle(yaw + self.yaw)
