"""Dispatch rules: by material, never a silent guess, alternating arms."""

from __future__ import annotations

import unittest

from trashdrop.planning import DetectedItem, TwoArmDispatcher
from trashdrop.station import CATEGORY_OWNER, MIXED_CATEGORY, PICK_ZONE


def item(category: str, confidence: float = 1.0, x: float = 0.0, y: float = -0.215):
    return DetectedItem(
        item_id=f"{category}-{confidence}", category=category, x=x, y=y, confidence=confidence
    )


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = TwoArmDispatcher(confidence_floor=0.55)

    def test_material_decides_the_arm_not_position(self) -> None:
        left_edge, right_edge = PICK_ZONE.bounds[0], PICK_ZONE.bounds[1]
        for x in (left_edge, right_edge):
            assignment = self.dispatcher.dispatch([item("bio", x=x)])[0]
            self.assertEqual(assignment.arm, CATEGORY_OWNER["bio"])

    def test_low_confidence_is_rerouted_to_mixed(self) -> None:
        assignment = self.dispatcher.dispatch([item("metal", confidence=0.4)])[0]
        self.assertTrue(assignment.rerouted)
        self.assertEqual(assignment.bin_category, MIXED_CATEGORY)

    def test_confident_items_reach_their_material_bin(self) -> None:
        assignment = self.dispatcher.dispatch([item("metal", confidence=0.9)])[0]
        self.assertFalse(assignment.rerouted)
        self.assertEqual(assignment.bin_category, "metal")

    def test_unknown_category_raises_rather_than_guessing(self) -> None:
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch([item("batteries")])

    def test_items_in_the_zone_need_exclusive_access(self) -> None:
        inside = self.dispatcher.dispatch([item("bio", x=0.0, y=PICK_ZONE.center_y)])[0]
        self.assertTrue(inside.requires_shared_zone)
        outside = self.dispatcher.dispatch([item("bio", x=0.0, y=-0.9)])[0]
        self.assertFalse(outside.requires_shared_zone)

    def test_mixed_work_is_balanced_between_arms(self) -> None:
        assignments = self.dispatcher.dispatch([item(MIXED_CATEGORY) for _ in range(4)])
        used = [a.arm for a in assignments]
        self.assertEqual(len(set(used)), 2, "mixed items all went to one arm")


class OrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = TwoArmDispatcher()

    def test_arms_alternate_when_both_have_work(self) -> None:
        items = [item("bio"), item("paper"), item("plastic"), item("metal")]
        ordered = self.dispatcher.order(self.dispatcher.dispatch(items))
        arms = [a.arm for a in ordered]
        self.assertNotEqual(arms[0], arms[1])
        self.assertNotEqual(arms[1], arms[2])

    def test_last_arm_carries_alternation_across_batches(self) -> None:
        items = [item("bio"), item("plastic")]
        ordered = self.dispatcher.order(self.dispatcher.dispatch(items), last="front")
        self.assertEqual(ordered[0].arm, "back")

    def test_order_keeps_every_assignment(self) -> None:
        items = [item("bio"), item("paper"), item("plastic")]
        assignments = self.dispatcher.dispatch(items)
        self.assertEqual(len(self.dispatcher.order(assignments)), len(assignments))


if __name__ == "__main__":
    unittest.main()
