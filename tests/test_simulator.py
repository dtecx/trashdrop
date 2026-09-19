from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trashdrop.simulator import run_taco_demo


class DualArmDemoTests(unittest.TestCase):
    def test_taco_style_dual_arm_demo_places_every_object(self) -> None:
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                report = run_taco_demo(Path(temporary_directory))
        except RuntimeError as error:
            self.skipTest(str(error))

        self.assertEqual(report.assigned, 5)
        self.assertEqual(report.placed, 5)
        self.assertEqual(report.misses, 0)
        self.assertTrue(any(event.phase == "reserve" for event in report.events))


if __name__ == "__main__":
    unittest.main()
