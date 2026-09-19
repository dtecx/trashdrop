"""Hardware-neutral station dimensions and named workcell zones.

All positions are metres in the shared MuJoCo/world frame. The arm bases sit
behind the table and face negative Y. Keep these values independent from a
future camera or robot SDK so measured calibration can replace them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArmMount:
    """A named arm base and the zone it owns by default."""

    name: str
    x: float
    y: float
    yaw_degrees: float
    min_x: float
    max_x: float

    @property
    def prefix(self) -> str:
        return f"{self.name}_"


@dataclass(frozen=True)
class RecyclingBin:
    """A target bin centre used by the planner and the visual scene."""

    category: str
    x: float
    y: float
    rgba: str


LEFT_ARM = ArmMount(
    name="left",
    x=-0.19,
    y=0.13,
    yaw_degrees=0.0,
    min_x=-0.42,
    max_x=-0.025,
)
RIGHT_ARM = ArmMount(
    name="right",
    x=0.19,
    y=0.13,
    yaw_degrees=0.0,
    min_x=0.025,
    max_x=0.42,
)
ARMS = (LEFT_ARM, RIGHT_ARM)

# The 5 cm centre strip is deliberately not owned by either arm. Jobs detected
# there are assigned by load, then must be executed through a future explicit
# hand-off/clearance policy instead of both arms reaching into it at once.
HANDOFF_HALF_WIDTH = 0.025

BINS = {
    "bio": RecyclingBin("bio", -0.33, -0.31, "0.15 0.65 0.20 1"),
    "paper": RecyclingBin("paper", -0.11, -0.31, "0.20 0.35 0.85 1"),
    "plastic": RecyclingBin("plastic", 0.11, -0.31, "0.95 0.80 0.10 1"),
    "metal": RecyclingBin("metal", 0.33, -0.31, "0.85 0.15 0.15 1"),
}

SAMPLE_ITEMS = (
    ("plastic_bottle", "plastic", 0.17, -0.13),
    ("banana_peel", "bio", -0.14, -0.15),
    ("paper_card", "paper", -0.08, -0.24),
    ("can", "metal", 0.13, -0.23),
    ("handoff_cup", "plastic", 0.0, -0.18),
)


def repository_root() -> Path:
    """Return the project root without relying on the current directory."""

    return Path(__file__).resolve().parents[1]


def find_arm_model(root: Path | None = None) -> Path:
    """Find a local sparse menagerie checkout in its preferred location.

    The legacy location keeps this workspace usable before the optional model
    bootstrap is run. Both locations are intentionally Git-ignored.
    """

    root = root or repository_root()
    candidates = (
        root / "vendor/mujoco_menagerie/trs_so_arm100",
        root / "mujoco_menagerie/trs_so_arm100",
    )
    for candidate in candidates:
        if (candidate / "so_arm100.xml").is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "SO-ARM model not found. Run ./scripts/bootstrap_model.sh, then retry.\n"
        f"Searched:\n  {searched}"
    )
