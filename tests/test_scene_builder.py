from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trashdrop.scene_builder import build_station
from trashdrop.station import find_arm_model


class SceneBuilderTests(unittest.TestCase):
    def test_scene_contains_two_namespaced_arms(self) -> None:
        try:
            model_directory = find_arm_model()
        except FileNotFoundError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temp_directory:
            scene = build_station(Path(temp_directory) / "station.xml", model_directory)
            xml = scene.read_text(encoding="utf-8")

        self.assertIn('name="left_Rotation"', xml)
        self.assertIn('name="right_Rotation"', xml)
        self.assertIn('name="topdown"', xml)
        self.assertIn('name="bin_plastic_floor"', xml)


if __name__ == "__main__":
    unittest.main()
