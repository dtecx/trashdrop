"""Per-arm Cartesian control: the seam between planning and a robot.

``ArmController.send`` is the only method that talks to an actuator. Swapping
the simulator for hardware means implementing that one call against
``lerobot``'s ``SO101Follower.send_action`` and keeping everything else.

The solver is damped least squares over four joints, with a secondary
objective that keeps the jaws pointing down. Its constants are the ones the
single-arm baseline was verified with; the only thing added here is the frame
conversion, because a second arm has a different base position and heading.
Jacobians and position error stay in the shared frame -- only the quantities
that are meaningful relative to a base (the shoulder heading, and therefore
the wrist roll that cancels it) are converted.
"""

from __future__ import annotations

import numpy as np

from .geometry import wrap_angle
from .station import IK_JOINTS, JOINTS, TCP_LOCAL, ArmMount

# Damped least squares: 200 iterations at damping 0.12, per-step joint deltas
# clipped to 0.1 rad. Raising the damping stalls the solver near the reach
# limit; lowering it makes the elbow snap between configurations.
IK_ITERATIONS = 200
IK_DAMPING = 0.12
IK_MAX_STEP = 0.1
# Weight on the keep-the-jaws-vertical objective relative to position error.
IK_ORIENTATION_WEIGHT = 0.25
WRIST_ROLL_LIMIT = 2.7


class ArmController:
    """Cartesian control for one namespaced arm in a shared MuJoCo model."""

    def __init__(self, mujoco, model, data, mount: ArmMount) -> None:
        self._mj = mujoco
        self.model = model
        self.data = data
        self.mount = mount
        self.name = mount.name
        prefix = mount.prefix

        self.jaw_body = self._body_id(f"{prefix}Fixed_Jaw")
        self.qadr = {j: model.jnt_qposadr[self._joint_id(f"{prefix}{j}")] for j in JOINTS}
        self.dofadr = [model.jnt_dofadr[self._joint_id(f"{prefix}{j}")] for j in IK_JOINTS]
        self.limits = {j: tuple(model.jnt_range[self._joint_id(f"{prefix}{j}")]) for j in JOINTS}
        self.actuator = {
            j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{j}")
            for j in JOINTS
        }
        # A scratch MjData so IK iterations never disturb the live simulation.
        self.ik_data = mujoco.MjData(model)
        self.tcp_local = np.asarray(TCP_LOCAL, dtype=float)

        # Grasp bookkeeping, filled by the cell when this arm picks something.
        self.held: dict | None = None
        self.jaw = 0.0

    # --- lookup helpers ----------------------------------------------------

    def _body_id(self, name: str) -> int:
        index = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise KeyError(f"No body named {name!r} in the compiled model")
        return index

    def _joint_id(self, name: str) -> int:
        index = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise KeyError(f"No joint named {name!r} in the compiled model")
        return index

    # --- state -------------------------------------------------------------

    def tcp(self, data=None) -> np.ndarray:
        """Grasp point in the shared frame."""

        d = self.data if data is None else data
        rot = d.xmat[self.jaw_body].reshape(3, 3)
        return d.xpos[self.jaw_body] + rot @ self.tcp_local

    def joint_positions(self) -> dict[str, float]:
        return {j: float(self.data.qpos[self.qadr[j]]) for j in JOINTS}

    # --- kinematics --------------------------------------------------------

    def solve_ik(
        self,
        target: np.ndarray,
        yaw: float = 0.0,
        keep_vertical: bool = True,
    ) -> tuple[dict[str, float], float]:
        """Solve for a shared-frame target; return joint targets and residual.

        ``yaw`` is the desired grasp heading in the shared frame.
        """

        mj = self._mj
        ik = self.ik_data
        ik.qpos[:] = self.data.qpos
        ik.qvel[:] = 0

        # The shoulder is what aims this arm, so the finger heading is whatever
        # the wrist has to add on top of it. Both are base-relative quantities,
        # so both are computed in the arm's own frame.
        local_x, local_y = self.mount.pose.to_local(float(target[0]), float(target[1]))
        base_yaw = np.arctan2(-local_x, -local_y)
        local_yaw = self.mount.pose.yaw_to_local(yaw)
        ik.qpos[self.qadr["Wrist_Roll"]] = float(
            np.clip(wrap_angle(local_yaw - base_yaw), -WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT)
        )

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        weight = IK_ORIENTATION_WEIGHT if keep_vertical else 0.0
        down = np.array([0.0, 0.0, -1.0])

        for _ in range(IK_ITERATIONS):
            mj.mj_kinematics(self.model, ik)
            mj.mj_comPos(self.model, ik)
            rot = ik.xmat[self.jaw_body].reshape(3, 3)
            approach = rot @ np.array([0.0, -1.0, 0.0])  # jaws point along local -Y
            err_p = target - self.tcp(ik)
            err_r = np.cross(approach, down)
            if np.linalg.norm(err_p) < 1e-4 and (
                not keep_vertical or np.linalg.norm(err_r) < 1e-3
            ):
                break
            mj.mj_jac(self.model, ik, jacp, jacr, self.tcp(ik), self.jaw_body)
            jacobian = np.vstack([jacp[:, self.dofadr], weight * jacr[:, self.dofadr]])
            error = np.concatenate([err_p, weight * err_r])
            delta = np.linalg.solve(
                jacobian.T @ jacobian + IK_DAMPING**2 * np.eye(len(self.dofadr)),
                jacobian.T @ error,
            )
            delta = np.clip(delta, -IK_MAX_STEP, IK_MAX_STEP)
            for index, joint in enumerate(IK_JOINTS):
                low, high = self.limits[joint]
                address = self.qadr[joint]
                ik.qpos[address] = np.clip(ik.qpos[address] + delta[index], low, high)

        pose = {j: float(ik.qpos[self.qadr[j]]) for j in (*IK_JOINTS, "Wrist_Roll")}
        return pose, float(np.linalg.norm(target - self.tcp(ik)))

    def reach_error(self, x: float, y: float, z: float, keep_vertical: bool = True) -> float:
        """Residual IK error at one point, for reachability probes."""

        _, error = self.solve_ik(np.array([x, y, z]), 0.0, keep_vertical)
        return error

    # --- the hardware seam -------------------------------------------------

    def send(self, targets: dict[str, float]) -> None:
        """Command joint positions.

        On hardware this becomes a single ``send_action`` call with the same
        joint names; nothing above this method knows which it is talking to.
        """

        for joint, value in targets.items():
            self.data.ctrl[self.actuator[joint]] = value

    def hold(self) -> None:
        """Re-issue the current command so an idle arm does not drift."""

        self.send({j: float(self.data.ctrl[self.actuator[j]]) for j in JOINTS})
