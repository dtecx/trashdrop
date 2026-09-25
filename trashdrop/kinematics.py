"""Where an SO-101's gripper is for given joint angles, and the reverse.

The model is TheRobotStudio's official SO-101 cut down to its kinematic chain
(trashdrop/models/so101_kinematics.xml). Its joint angles are LeRobot's
calibrated degrees -- the upright pose (0, 0, -90, 0, 0) stands it straight up,
as it does the real arms -- so what this module solves goes to Arm.move()
unchanged.

Positions are metres in the arm's own frame: x forward (where it reaches at
shoulder_pan 0), y to the arm's left, z up from the bottom of its base. The
point is the TCP: the inner face of the FIXED finger, 7 mm above its tip,
which is where a grasp plan puts it (see perception/grasp.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).resolve().parent / "models" / "so101_kinematics.xml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"
NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0}
# A radian of pointing error weighs as much as this many metres of position error.
ORIENTATION_WEIGHT = 0.05
REACHED_M = 0.003  # position error that still counts as reaching the point
POINTING_TOLERANCE = 0.05  # |pointing error vector|, about 3 degrees
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Solution:
    degrees: dict[str, float]
    position_error_m: float
    pointing_error: float

    @property
    def reachable(self) -> bool:
        return self.position_error_m <= REACHED_M and self.pointing_error <= POINTING_TOLERANCE


class Kinematics:
    def __init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
        self.data = mujoco.MjData(self.model)
        ids = {joint: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in ARM_JOINTS}
        self._qpos = {joint: self.model.jnt_qposadr[i] for joint, i in ids.items()}
        self.model_limits = {joint: tuple(np.degrees(self.model.jnt_range[i])) for joint, i in ids.items()}
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        jaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        hinge = self.model.jnt_axis[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]

        # Which way the fingers point, and which way the jaw closes, in the
        # TCP site's own frame -- read off the upright pose, where the fingers
        # point straight up. The jaw closes square to both the fingers and its
        # own hinge, towards the moving jaw.
        self._forward(NEUTRAL)
        rotation = self.data.site_xmat[self._site].reshape(3, 3)
        approach = np.array([0.0, 0.0, 1.0])
        hinge_world = self.data.xmat[jaw].reshape(3, 3) @ hinge
        across = np.cross(hinge_world, approach)
        across /= np.linalg.norm(across)
        if across @ (self.data.xpos[jaw] - self.data.site_xpos[self._site]) < 0:
            across = -across
        self._approach_local = rotation.T @ approach
        self._across_local = rotation.T @ across

    # --- forward ---------------------------------------------------------------

    def _forward(self, degrees: dict[str, float]) -> None:
        self.data.qpos[:] = 0.0
        for joint, value in degrees.items():
            if joint in self._qpos:
                self.data.qpos[self._qpos[joint]] = np.radians(value)
        self._mujoco.mj_kinematics(self.model, self.data)

    def tcp(self, degrees: dict[str, float]) -> np.ndarray:
        """TCP position, metres, in the arm's frame."""

        self._forward(degrees)
        return self.data.site_xpos[self._site].copy()

    def pointing(self, degrees: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(where the fingers point, which way the jaw closes), unit vectors."""

        self._forward(degrees)
        rotation = self.data.site_xmat[self._site].reshape(3, 3)
        return rotation @ self._approach_local, rotation @ self._across_local

    # --- inverse ---------------------------------------------------------------

    def solve(self, target, *, yaw_deg: float | None = None, start: dict[str, float] | None = None,
              limits: dict[str, tuple[float, float]] | None = None) -> Solution:
        """Joint degrees putting the TCP at ``target`` with the fingers pointing down.

        ``yaw_deg`` turns the jaw so it closes along that direction in the
        arm's xy plane (the grasp plan's ``across``); without it the wrist
        roll stays where ``start`` has it. ``limits`` are the real joints'
        limits in degrees; the model's own are used otherwise.
        """

        target = np.asarray(target, dtype=float)
        # The tighter of the model's design range and the real joint's limits.
        limits = {
            joint: (
                max(self.model_limits[joint][0], (limits or {}).get(joint, self.model_limits[joint])[0]),
                min(self.model_limits[joint][1], (limits or {}).get(joint, self.model_limits[joint])[1]),
            )
            for joint in ARM_JOINTS
        }
        variables = list(ARM_JOINTS) if yaw_deg is not None else list(ARM_JOINTS[:4])
        wanted_across = (
            np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg)), 0.0]) if yaw_deg is not None else None
        )

        def residual(degrees: dict[str, float]) -> np.ndarray:
            position = self.tcp(degrees)
            approach, across = self.pointing(degrees)
            parts = [position - target, ORIENTATION_WEIGHT * (approach - DOWN)]
            if wanted_across is not None:
                parts.append(ORIENTATION_WEIGHT * (across - wanted_across))
            return np.concatenate(parts)

        seeds = [dict(NEUTRAL, **(start or {})), dict(NEUTRAL), dict(NEUTRAL, shoulder_lift=30.0, elbow_flex=-30.0, wrist_flex=70.0)]
        best = None
        for seed in seeds:
            degrees = {joint: float(np.clip(seed.get(joint, 0.0), *limits[joint])) for joint in ARM_JOINTS}
            for _ in range(150):
                r = residual(degrees)
                jacobian = np.zeros((len(r), len(variables)))
                for column, joint in enumerate(variables):
                    nudged = dict(degrees)
                    nudged[joint] += 0.05
                    jacobian[:, column] = (residual(nudged) - r) / np.radians(0.05)
                damping = 0.01
                step = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(len(r)), r)
                step = np.clip(step, -0.15, 0.15)  # radians per iteration
                for column, joint in enumerate(variables):
                    degrees[joint] = float(np.clip(degrees[joint] + np.degrees(step[column]), *limits[joint]))
                if np.abs(step).max() < 1e-5:
                    break
            position_error = float(np.linalg.norm(self.tcp(degrees) - target))
            approach, across = self.pointing(degrees)
            pointing_error = float(np.linalg.norm(approach - DOWN))
            if wanted_across is not None:
                pointing_error = max(pointing_error, float(np.linalg.norm(across - wanted_across)))
            candidate = Solution({j: round(v, 2) for j, v in degrees.items()}, position_error, pointing_error)
            score = position_error + ORIENTATION_WEIGHT * pointing_error
            if best is None or score < best[0]:
                best = (score, candidate)
            if candidate.reachable:
                break
        return best[1]
