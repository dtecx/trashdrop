"""The assembled cell compiles and the second arm is actually turned around."""

from __future__ import annotations

import unittest

import pytest

mujoco = pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.scene_builder import build_model  # noqa: E402
from trashdrop.station import ARMS, BINS, JOINTS, PICK_ZONE  # noqa: E402


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_model()

    def test_both_arms_are_present_and_namespaced(self) -> None:
        actuators = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        }
        for mount in ARMS:
            for joint in JOINTS:
                self.assertIn(f"{mount.prefix}{joint}", actuators)

    def test_every_bin_exists_in_the_scene(self) -> None:
        for key in BINS:
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bin_{key}"), 0
            )

    def test_arms_face_each_other(self) -> None:
        """MJCF angles default to degrees; a radian yaw silently becomes ~3 deg.

        With the rotation lost, the back arm's tool sits roughly a full base
        separation away from where it belongs, so check the actual geometry
        rather than only that the model compiled.
        """

        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        for mount in ARMS:
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{mount.prefix}Fixed_Jaw"
            )
            tool_y = float(data.xpos[body][1])
            # Each arm must extend from its base toward the middle of the cell.
            # If the attach frame lost its rotation, the back arm extends away
            # from the table instead, which is what this catches.
            reach = tool_y - mount.pose.y
            toward_centre = PICK_ZONE.center_y - mount.pose.y
            self.assertGreater(
                reach * toward_centre,
                0.0,
                f"{mount.name} arm reaches away from the pick zone (tool y={tool_y:.3f})",
            )

    def test_cameras_exist(self) -> None:
        for camera in ("topdown", "operator"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera), 0
            )


if __name__ == "__main__":
    unittest.main()
