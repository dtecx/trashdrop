"""End-to-end regression: two arms sort four items, scored from physics.

Slow (a few seconds) but it is the test that matters -- it is the only one
that would catch a broken IK seed, a stow pose that occludes the camera, or a
release that drops items outside the bin.
"""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.scene_builder import DEFAULT_OBJECTS  # noqa: E402
from trashdrop.simulator import SortingCell  # noqa: E402


class SortingRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = SortingCell(record=False).run(verbose=False)

    def test_every_item_reaches_its_own_bin(self) -> None:
        self.assertEqual(
            self.report.placed,
            len(DEFAULT_OBJECTS),
            f"misplaced: {self.report.misplaced}",
        )

    def test_nothing_was_dropped(self) -> None:
        self.assertEqual(self.report.missed, 0)

    def test_both_arms_did_work(self) -> None:
        arms = {cycle.arm for cycle in self.report.cycles}
        self.assertEqual(arms, {"front", "back"})

    def test_tool_arrives_close_to_each_bin(self) -> None:
        for cycle in self.report.cycles:
            self.assertLess(cycle.tcp_error_mm, 15.0, f"{cycle.item_id}: {cycle.tcp_error_mm} mm")

    def test_run_is_reported_as_successful(self) -> None:
        self.assertTrue(self.report.success)


if __name__ == "__main__":
    unittest.main()
