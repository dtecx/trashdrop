"""Hardware-neutral cell geometry: arm mounts, bins, pick zone and camera.

Layout is "facing each other": two SO-101 bases on opposite sides of a shared
pick zone, 43 cm apart. The consequence that drives the rest of the code is
that *both arms reach the whole pick zone*, so an item is assigned to an arm by
its material, never by where it happens to lie. There is no hand-off.

The reach envelope encoded here was measured on the SO-ARM100 model with the
gripper held vertical: nothing above z = 0.12 m is reachable, and the usable
patch in front of a base is roughly 20 x 20 cm. Run ``trashdrop probe`` after
changing any number in this file -- it re-solves IK to every bin and pick-zone
corner and tells you what stopped being reachable.

All distances are metres in the shared table frame, which is also the MuJoCo
world frame. Its origin is the FRONT arm's base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geometry import Pose2D

# --- arms ------------------------------------------------------------------

# 43 cm between bases. Each arm's own frame then sees the shared pick zone at
# local y in [-0.14, -0.29], which is inside the verified envelope for both.
BASE_SEPARATION = 0.43


@dataclass(frozen=True)
class ArmMount:
    """A named arm base in the shared frame, plus its MJCF namespace."""

    name: str
    pose: Pose2D

    @property
    def prefix(self) -> str:
        # Trailing slash matches what MjSpec.attach produces for a prefix.
        return f"{self.name}/"


FRONT_ARM = ArmMount(name="front", pose=Pose2D.from_degrees(0.0, 0.0, 0.0))
BACK_ARM = ArmMount(name="back", pose=Pose2D.from_degrees(0.0, -BASE_SEPARATION, 180.0))
ARMS: tuple[ArmMount, ...] = (FRONT_ARM, BACK_ARM)
ARMS_BY_NAME = {mount.name: mount for mount in ARMS}

# --- waste categories ------------------------------------------------------

# rgba is only used to draw bins and the synthetic stand-in objects. It is NOT
# a classification signal: the sim-only colour detector is the one place that
# treats colour as meaning, and it exists to exercise the motion stack before
# a trained classifier is available. See trashdrop/perception/.
CATEGORY_COLORS = {
    "bio": "0.15 0.65 0.20 1",
    "paper": "0.20 0.35 0.85 1",
    "plastic": "0.95 0.80 0.10 1",
    "metal": "0.85 0.15 0.15 1",
    "mixed": "0.45 0.45 0.48 1",
}
SORT_CATEGORIES: tuple[str, ...] = ("bio", "paper", "plastic", "metal")
# Everything the cell will not commit to a material bin: an unrecognised item,
# a low-confidence classification, or one the arm physically cannot handle.
MIXED_CATEGORY = "mixed"
ALL_CATEGORIES: tuple[str, ...] = SORT_CATEGORIES + (MIXED_CATEGORY,)

# Which arm owns which material. Both arms reach every item, so this split is
# purely about keeping each transfer short and each bin on its owner's side.
CATEGORY_OWNER = {
    "bio": "front",
    "paper": "front",
    "plastic": "back",
    "metal": "back",
}

# --- bins ------------------------------------------------------------------

# Bin centres in the OWNING arm's local frame. Local y = -0.12 keeps them
# between the base and the pick zone's near edge (-0.14), and |x| >= 0.20 keeps
# them clear of the zone's x span of +/-0.10. Both were verified by probe.
_BIN_LOCAL = {
    "bio": (0.20, -0.12),
    "paper": (-0.20, -0.12),
    "plastic": (0.20, -0.12),
    "metal": (-0.20, -0.12),
}
# One shared mixed bin on the side of the table, positioned so BOTH arms reach
# it -- whichever arm holds an item it cannot place should not need a hand-off.
# ``trashdrop probe`` checks this for both arms.
_MIXED_WORLD = (0.28, -0.215)

BIN_HALF_WIDTH = 0.045
BIN_WALL = 0.004
BIN_WALL_HEIGHT = 0.025
# Scoring tolerance: an object counts as binned if it lands within this of the
# bin centre in x and y. Slightly smaller than the bin's inner half-width so a
# result on the rim does not count as success.
BIN_TOLERANCE = 0.06


@dataclass(frozen=True)
class RecyclingBin:
    """One bin: which arm owns it, what it takes, and where it sits."""

    key: str
    category: str
    arm: str
    x: float
    y: float
    rgba: str


def _build_bins() -> dict[str, RecyclingBin]:
    bins: dict[str, RecyclingBin] = {}
    for category, local in _BIN_LOCAL.items():
        arm = ARMS_BY_NAME[CATEGORY_OWNER[category]]
        x, y = arm.pose.to_world(*local)
        bins[category] = RecyclingBin(
            key=category,
            category=category,
            arm=arm.name,
            x=x,
            y=y,
            rgba=CATEGORY_COLORS[category],
        )
    x, y = _MIXED_WORLD
    bins[MIXED_CATEGORY] = RecyclingBin(
        key=MIXED_CATEGORY,
        category=MIXED_CATEGORY,
        arm="any",
        x=x,
        y=y,
        rgba=CATEGORY_COLORS[MIXED_CATEGORY],
    )
    return bins


BINS: dict[str, RecyclingBin] = _build_bins()


def bin_for(category: str, arm: str | None = None) -> RecyclingBin:
    """Resolve a material to its bin. ``arm`` is unused but kept explicit."""

    if category in BINS:
        return BINS[category]
    raise ValueError(f"No bin for category {category!r}")


def owner_of(category: str, arm_hint: str | None = None) -> str:
    """Which arm handles this material; mixed goes to whoever is free."""

    if category in CATEGORY_OWNER:
        return CATEGORY_OWNER[category]
    if category == MIXED_CATEGORY:
        return arm_hint if arm_hint in ARMS_BY_NAME else ARMS[0].name
    raise ValueError(f"Unknown category {category!r}")


# --- what the arm can physically handle ------------------------------------

# SO-101 payload falls off sharply at extension, and the jaws open a limited
# amount. An item outside these limits is routed to the mixed bin rather than
# attempted: a dropped item mid-transfer is the one failure the demo cannot
# recover from. Measure MAX_GRASP_WIDTH on the real gripper before the event.
MAX_PAYLOAD_KG = 0.25
MAX_GRASP_WIDTH = 0.035
# An empty 0.5 L aluminium can is ~15 g and ~66 mm across: light enough, but it
# must be grasped across the body, not the diameter. A 0.5 L glass bottle is
# ~300-400 g empty and is deliberately out of scope -- it goes to mixed.
GLASS_IS_OUT_OF_SCOPE = True


def is_graspable(width_m: float, mass_kg: float | None = None) -> bool:
    """Whether the gripper should attempt an item at all."""

    if width_m > MAX_GRASP_WIDTH:
        return False
    if mass_kg is not None and mass_kg > MAX_PAYLOAD_KG:
        return False
    return True


# --- pick zone and table ---------------------------------------------------


@dataclass(frozen=True)
class PickZone:
    """The shared area where delivered items land, in the shared frame."""

    center_x: float
    center_y: float
    half_x: float
    half_y: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, max_x, min_y, max_y)."""

        return (
            self.center_x - self.half_x,
            self.center_x + self.half_x,
            self.center_y - self.half_y,
            self.center_y + self.half_y,
        )

    def contains(self, x: float, y: float, margin: float = 0.01) -> bool:
        min_x, max_x, min_y, max_y = self.bounds
        return (min_x - margin) < x < (max_x + margin) and (min_y - margin) < y < (max_y + margin)

    def corners(self) -> tuple[tuple[float, float], ...]:
        min_x, max_x, min_y, max_y = self.bounds
        return ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))


# Centred exactly between the two bases so the geometry is symmetric.
PICK_ZONE = PickZone(
    center_x=0.0,
    center_y=-BASE_SEPARATION / 2,
    half_x=0.10,
    half_y=0.075,
)

SURFACE_THICKNESS = 0.004
OBJECT_HALF_SIZE = 0.014
# Height of an object's centre when it rests on the work surface.
GRASP_Z = SURFACE_THICKNESS + OBJECT_HALF_SIZE
# Travel height. Above this the gripper cannot be held vertical (measured).
SAFE_Z = 0.10
MAX_VERTICAL_Z = 0.12

# --- overhead camera -------------------------------------------------------


@dataclass(frozen=True)
class TopDownCamera:
    """The single overhead camera shared by both arms."""

    height: float
    center_x: float
    center_y: float
    fovy_degrees: float
    width: int
    height_px: int


CAMERA = TopDownCamera(
    height=0.60,
    center_x=PICK_ZONE.center_x,
    center_y=PICK_ZONE.center_y,
    fovy_degrees=45.0,
    width=640,
    height_px=480,
)

# --- joints ----------------------------------------------------------------

JOINTS: tuple[str, ...] = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
IK_JOINTS: tuple[str, ...] = JOINTS[:4]
JAW_OPEN, JAW_CLOSED = 1.5, 0.05
# Grasp point expressed in the Fixed_Jaw body frame.
TCP_LOCAL = (0.005, -0.085, 0.0)
# Stow: the arm folded straight up over its own base. Every link stays within
# ~8 cm of the base, which does three things at once -- it clears the overhead
# camera's view of the pick zone, it keeps the arm out of the other arm's
# working volume, and it is a safe state to start and finish in.
#
# The single-arm baseline used a pose that swung the arm forward and aside
# instead. That is wrong here: with two bases 43 cm apart, both arms reach
# ~24 cm forward, so two "swung aside" arms meet over the middle of the table,
# occlude half the items and physically jam against each other.
STOW_POSE = (0.0, -2.0, 0.5, 0.19, 0.0, JAW_OPEN)
# Transit pose for the arm that is working. Safe only while the other arm is
# stowed, which the executor guarantees by running one arm at a time.
HOME_POSE = (0.0, -1.57, 1.57, 1.57, -1.57, JAW_OPEN)
# Maximum horizontal distance any link reaches from its base while stowed.
STOW_RADIUS = 0.08


# --- model location --------------------------------------------------------


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_arm_model(root: Path | None = None) -> Path:
    """Locate the sparse mujoco_menagerie checkout."""

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
