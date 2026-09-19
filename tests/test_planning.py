from __future__ import annotations

import unittest

from trashdrop.planning import DetectedItem, TwoArmDispatcher


class TwoArmDispatcherTests(unittest.TestCase):
    def test_outer_zones_have_single_obvious_owner(self) -> None:
        jobs = [
            DetectedItem("paper", "paper", -0.15, -0.12),
            DetectedItem("can", "metal", 0.15, -0.12),
        ]

        assignments = TwoArmDispatcher().dispatch(jobs)

        self.assertEqual([assignment.arm for assignment in assignments], ["left", "right"])
        self.assertFalse(any(assignment.requires_handoff_clearance for assignment in assignments))

    def test_shared_strip_is_single_owned_and_flagged(self) -> None:
        assignments = TwoArmDispatcher().dispatch(
            [
                DetectedItem("cup_1", "plastic", 0.0, -0.12),
                DetectedItem("cup_2", "plastic", 0.01, -0.18),
            ]
        )

        self.assertEqual([assignment.arm for assignment in assignments], ["left", "right"])
        self.assertTrue(all(assignment.requires_handoff_clearance for assignment in assignments))

    def test_unknown_class_is_never_routed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown category"):
            TwoArmDispatcher().dispatch([DetectedItem("mystery", "glass", 0.1, -0.1)])


if __name__ == "__main__":
    unittest.main()
