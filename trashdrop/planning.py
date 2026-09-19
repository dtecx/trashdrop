"""Work assignment for two SO-101 arms sharing one pick zone.

No MuJoCo, camera or robot-SDK imports: a detector produces
:class:`DetectedItem` records and a hardware adapter consumes
:class:`ArmAssignment` records without either importing the simulator.

The "facing each other" layout makes this simpler than a split-table one.
Both arms reach every point of the pick zone, so an item goes to the arm that
*owns its material*, and the awkward case -- an item only one arm can reach --
does not exist. What the layout costs instead is exclusivity: two arms sharing
one volume must never enter it at once, so every assignment is marked as
needing the zone and the executor serialises them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .station import (
    ALL_CATEGORIES,
    ARMS,
    PICK_ZONE,
    MIXED_CATEGORY,
    bin_for,
    owner_of,
)


@dataclass(frozen=True)
class DetectedItem:
    """One classified object in the shared table frame.

    ``x``/``y`` are metres, ``yaw`` is the grasp heading in radians, and
    ``confidence`` is the classifier's own score. A low score is not silently
    routed to a material bin -- see :meth:`TwoArmDispatcher.dispatch`.
    """

    item_id: str
    category: str
    x: float
    y: float
    yaw: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class ArmAssignment:
    """A single-owner job with the target bin already resolved."""

    arm: str
    item: DetectedItem
    bin_key: str
    bin_category: str
    bin_x: float
    bin_y: float
    requires_shared_zone: bool
    rerouted: bool = False


class TwoArmDispatcher:
    """Assign items by material, keeping the shared-zone rule explicit.

    ``confidence_floor`` is the promise in the pitch that the cell flags what
    it cannot handle rather than guessing: anything below it is rerouted to the
    shared mixed bin instead of a material bin.
    """

    def __init__(self, confidence_floor: float = 0.55) -> None:
        self.confidence_floor = confidence_floor

    def dispatch(self, items: Iterable[DetectedItem]) -> list[ArmAssignment]:
        mixed_loads = {mount.name: 0 for mount in ARMS}
        assignments: list[ArmAssignment] = []

        for item in items:
            if item.category not in ALL_CATEGORIES:
                raise ValueError(
                    f"{item.item_id!r} has unknown category {item.category!r}; "
                    "do not route it to a bin"
                )

            rerouted = (
                item.category != MIXED_CATEGORY
                and item.confidence < self.confidence_floor
            )
            category = MIXED_CATEGORY if rerouted else item.category

            if category == MIXED_CATEGORY:
                # One shared mixed bin that both arms reach, so balance the
                # work rather than the bin: whichever arm has done less.
                arm = min(ARMS, key=lambda m: (mixed_loads[m.name], m.name)).name
                mixed_loads[arm] += 1
            else:
                arm = owner_of(category)

            spec = bin_for(category, arm)
            assignments.append(
                ArmAssignment(
                    arm=arm,
                    item=item,
                    bin_key=spec.key,
                    bin_category=spec.category,
                    bin_x=spec.x,
                    bin_y=spec.y,
                    # Both arms share the pick zone, so any item lying in it
                    # requires exclusive access for the duration of the pick.
                    requires_shared_zone=PICK_ZONE.contains(item.x, item.y),
                    rerouted=rerouted,
                )
            )
        return assignments

    def order(
        self, assignments: list[ArmAssignment], last: str | None = None
    ) -> list[ArmAssignment]:
        """Interleave arms where possible.

        Execution is serialised anyway, so this changes nothing about safety.
        It alternates which arm acts next when both have work, which keeps the
        cell visibly two-armed instead of draining one arm's queue first.
        Pass ``last`` -- the arm that acted in the previous cycle -- to keep
        alternating across re-detections, not just within one batch.
        """

        queues: dict[str, list[ArmAssignment]] = {m.name: [] for m in ARMS}
        for assignment in assignments:
            queues[assignment.arm].append(assignment)

        ordered: list[ArmAssignment] = []
        while any(queues.values()):
            candidates = [name for name, queue in queues.items() if queue]
            pick = next((name for name in candidates if name != last), candidates[0])
            ordered.append(queues[pick].pop(0))
            last = pick
        return ordered
