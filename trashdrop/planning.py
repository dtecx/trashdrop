"""Deterministic work assignment for two SO-101 arms.

This module intentionally has no MuJoCo, camera, or robot-SDK imports. A
detector can produce :class:`DetectedItem` records and a hardware adapter can
consume the resulting :class:`ArmAssignment` records without importing the
virtual workcell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .station import ARMS, HANDOFF_HALF_WIDTH, SORT_CATEGORIES, ArmMount, bin_for


@dataclass(frozen=True)
class DetectedItem:
    """One classified object in the common calibrated table frame."""

    item_id: str
    category: str
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ArmAssignment:
    """A single-owner job with the target bin already resolved."""

    arm: str
    item: DetectedItem
    bin_category: str
    bin_x: float
    bin_y: float
    requires_handoff_clearance: bool


class TwoArmDispatcher:
    """Assign items to exactly one arm while keeping zone rules visible.

    Items in an arm's outer zone always belong to that arm. Items within the
    centre strip are load-balanced deterministically and marked so a motion
    executor can reserve the shared area before moving. This avoids the unsafe
    implication that two independent controllers may enter a common zone.
    """

    def __init__(self, arms: tuple[ArmMount, ArmMount] = ARMS) -> None:
        self.left, self.right = arms

    def dispatch(self, items: Iterable[DetectedItem]) -> list[ArmAssignment]:
        """Return stable assignments; reject unknown waste categories early."""

        loads = {self.left.name: 0, self.right.name: 0}
        assignments: list[ArmAssignment] = []
        for item in items:
            if item.category not in SORT_CATEGORIES:
                raise ValueError(
                    f"{item.item_id!r} has unknown category {item.category!r}; "
                    "do not route it to a bin"
                )
            arm, shared = self._select_arm(item, loads)
            bin_spec = bin_for(arm.name, item.category)
            assignments.append(
                ArmAssignment(
                    arm=arm.name,
                    item=item,
                    bin_category=bin_spec.category,
                    bin_x=bin_spec.x,
                    bin_y=bin_spec.y,
                    requires_handoff_clearance=shared,
                )
            )
            loads[arm.name] += 1
        return assignments

    def _select_arm(
        self, item: DetectedItem, loads: dict[str, int]
    ) -> tuple[ArmMount, bool]:
        if item.x < -HANDOFF_HALF_WIDTH:
            return self.left, False
        if item.x > HANDOFF_HALF_WIDTH:
            return self.right, False

        # Tie-break on name makes repeated runs and test expectations stable.
        selected = min((self.left, self.right), key=lambda arm: (loads[arm.name], arm.name))
        return selected, True
