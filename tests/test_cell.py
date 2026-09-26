"""The cell as the web page drives it, on a synthetic camera and pretend arms.

The camera sees the table at 1 mm per pixel, the calibration sheet's frame in
its middle; the arms stand where the venue's did. What is pinned down: a look
finds the item, asks what it is and gives it to the arm on its side; an unsure
item goes nowhere; a pick counts only once the zone is empty again; auto mode
keeps sorting items as they are tossed in, until STOP; and the page's JSON
and its settings stay sane.
"""

from __future__ import annotations

import json
import threading
import time
import unittest

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the dataset extra")
pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.cell import Camera, Cell, SimArm  # noqa: E402
from trashdrop.kinematics import Kinematics  # noqa: E402
from trashdrop.perception.calibration import HomographyCalibration  # noqa: E402
from trashdrop.placement import Placement  # noqa: E402
from trashdrop.rig import ArmDevices, Rig  # noqa: E402
from trashdrop.sorter import RELEASE_OPEN  # noqa: E402
from trashdrop.web.server import _plain  # noqa: E402

LEFT = Placement(x=24.65, y=-19.64, yaw=-92.31, table_z=-2.42)
RIGHT = Placement(x=17.97, y=20.73, yaw=-84.12, table_z=-0.47)
NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0,
           "gripper": 0.0}


def camera() -> HomographyCalibration:
    corners = [(-10, 7.5), (10, 7.5), (10, -7.5), (-10, -7.5)]
    return HomographyCalibration([(960 + x * 10, 540 - y * 10) for x, y in corners],
                                 [(x / 100, y / 100) for x, y in corners])


def table() -> np.ndarray:
    frame = np.full((1080, 1920, 3), (215, 220, 218), np.uint8)
    return np.clip(frame + np.random.default_rng(0).normal(0, 2, frame.shape), 0, 255).astype(np.uint8)


def with_item(at_cm=(-5.0, -3.0)) -> np.ndarray:
    frame = table()
    box = cv2.boxPoints(((960 + at_cm[0] * 10, 540 - at_cm[1] * 10), (30.0, 120.0), 20.0))
    cv2.fillPoly(frame, [np.rint(box).astype(np.int32)], (60, 90, 170))
    return frame


class Classifier:
    def __init__(self, probabilities) -> None:
        self.answer = probabilities

    def probabilities(self, crop):
        return dict(self.answer)

    def close(self) -> None:
        pass


class Scene:
    """What the synthetic camera shows; an arm opening its jaw over its side takes the item away."""

    def __init__(self) -> None:
        self.frame = table()

    def read(self):
        time.sleep(0.01)
        return self.frame


class CatchingArm(SimArm):
    def __init__(self, name: str, scene: Scene) -> None:
        super().__init__(name, 45.0, move_s=0.0)
        self.scene, self.moves = scene, []

    def move(self, targets, *, speed=None):
        self.moves.append(dict(targets))
        if targets.get("gripper") == RELEASE_OPEN:
            self.scene.frame = table()  # dropped on its side: the zone is empty
        return super().move(targets, speed=speed)


def make_cell(probabilities=None) -> tuple[Cell, Scene]:
    scene = Scene()
    rig = Rig()
    rig.arms["left"] = ArmDevices(bus="L", sheet=(LEFT.x, LEFT.y, LEFT.yaw, LEFT.table_z))
    rig.arms["right"] = ArmDevices(bus="R", sheet=(RIGHT.x, RIGHT.y, RIGHT.yaw, RIGHT.table_z))
    kinematics = Kinematics()
    cell = Cell(
        rig=rig, homography=camera(), placements={"left": LEFT, "right": RIGHT},
        arms={name: CatchingArm(name, scene) for name in ("left", "right")},
        poses={"left": {"neutral": NEUTRAL}, "right": {"neutral": NEUTRAL}},
        kinematics={"left": kinematics, "right": kinematics},
        classifier=Classifier(probabilities or {"plastic": 0.95, "paper": 0.03, "metal": 0.01, "other": 0.01}),
        camera=Camera(scene.read), limits={"left": {}, "right": {}}, log=lambda *_: None,
    )
    cell.start()
    cell.photograph_empty()
    return cell, scene


class CellTests(unittest.TestCase):
    def tearDown(self) -> None:
        self.cell.close()

    def test_a_look_finds_the_item_and_gives_it_to_the_arm_on_its_side(self) -> None:
        self.cell, scene = make_cell()
        scene.frame = with_item()
        look = self.cell.look()
        self.assertEqual(look.code, "ok", look.message)
        self.assertEqual((look.side, look.arm), ("left", "left"), "plastic goes left")
        self.assertGreater(len(look.outline), 3)
        self.assertIsNotNone(look.fixed)

    def test_an_unsure_item_goes_nowhere(self) -> None:
        self.cell, scene = make_cell({"plastic": 0.55, "paper": 0.45})
        scene.frame = with_item()
        look = self.cell.look()
        self.assertEqual(look.code, "unsure")
        self.assertIsNone(look.arm)

    def test_an_empty_zone_says_so_in_the_pages_words(self) -> None:
        self.cell, _ = make_cell()
        self.assertEqual(self.cell.look().message, "the zone is empty")

    def test_a_pick_counts_once_the_zone_is_empty_again(self) -> None:
        self.cell, scene = make_cell({"paper": 0.97, "plastic": 0.03})
        scene.frame = with_item((5.0, -3.0))
        self.assertTrue(self.cell.pick())
        self.assertTrue(self.cell.arms["right"].moves, "paper: the right arm")
        self.assertFalse(self.cell.arms["left"].moves)
        self.assertTrue(any("done" in line for line in self.cell.log_lines))

    def test_auto_sorts_items_as_they_are_tossed_in_until_stop(self) -> None:
        self.cell, scene = make_cell()
        self.assertIsNone(self.cell.begin("auto"))
        seen = 0
        for _ in range(2):  # two items, one after the other
            scene.frame = with_item()
            deadline = time.monotonic() + 10
            done = False
            while time.monotonic() < deadline and not done:
                lines = list(self.cell.log_lines)
                done = any("done: the zone is empty again" in line for line in lines[seen:])
                time.sleep(0.05)
            self.assertTrue(done, "the item tossed in was sorted")
            seen = len(list(self.cell.log_lines))
        self.assertTrue(self.cell.auto)
        self.cell.stop()
        deadline = time.monotonic() + 5
        while self.cell.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNone(self.cell.busy)
        self.assertFalse(self.cell.auto)

    def test_stop_says_what_it_stopped(self) -> None:
        self.cell, _ = make_cell()
        self.assertIn("nothing was moving", self.cell.stop())
        release = threading.Event()
        self.cell.look = lambda: release.wait(5)
        self.cell.begin("look")
        self.assertIn("look stopped", self.cell.stop())
        release.set()

    def test_one_action_at_a_time(self) -> None:
        self.cell, _ = make_cell()
        release = threading.Event()
        self.cell.look = lambda: release.wait(5)  # a look that takes its time
        self.assertIsNone(self.cell.begin("look"))
        self.assertIn("busy", self.cell.begin("pick"))
        release.set()

    def test_the_state_is_json_and_the_settings_are_checked(self) -> None:
        self.cell, scene = make_cell()
        scene.frame = with_item()
        self.cell.look()
        state = json.loads(json.dumps(self.cell.state(), default=_plain))
        self.assertEqual(state["scene"]["size"], [1920, 1080])
        self.assertEqual(len(state["scene"]["zone"]), 4)
        self.assertEqual(state["last"]["arm"], "left")
        self.cell.set_options({"fingertips_cm": "0.8", "material": "paper", "only_arm": ""})
        self.assertEqual((self.cell.options.fingertips_cm, self.cell.options.material, self.cell.options.only_arm),
                         (0.8, "paper", None))
        with self.assertRaises(ValueError):
            self.cell.set_options({"material": "glass"})
        with self.assertRaises(ValueError):
            self.cell.set_speeds("left", 500.0, 20.0, save=False)
        self.cell.set_speeds("left", 60.0, 30.0, save=False)
        self.assertEqual(self.cell.arms["left"].max_speed, 60.0)


if __name__ == "__main__":
    unittest.main()
