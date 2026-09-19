"""TrashDrop: a two-arm SO-101 cell that sorts household waste.

Import layers, lightest first:

    station / geometry / planning   pure Python, no heavy dependencies
    dataset / perception            needs opencv
    scene_builder / control /
    simulator / probe               needs mujoco

Nothing in the first layer imports the others, so a laptop that only shoots
dataset photos never needs MuJoCo installed.
"""

from .planning import ArmAssignment, DetectedItem, TwoArmDispatcher
from .station import ALL_CATEGORIES, ARMS, BINS, PICK_ZONE, SORT_CATEGORIES

__all__ = [
    "ALL_CATEGORIES",
    "ARMS",
    "ArmAssignment",
    "BINS",
    "DetectedItem",
    "PICK_ZONE",
    "SORT_CATEGORIES",
    "TwoArmDispatcher",
]
