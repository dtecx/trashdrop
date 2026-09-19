"""Headless two-arm MuJoCo execution smoke test for the TACO station design.

This is intentionally a kinematic pick/place demonstrator: it validates named
actuators, reachability, per-arm bin ownership, a shared-zone reservation, and
where every object finishes. It is not a substitute for physical safety logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .planning import ArmAssignment, DetectedItem, TwoArmDispatcher
from .scene_builder import build_station
from .station import ARMS, SAMPLE_ITEMS, ArmMount, bin_for, repository_root


JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
IK_JOINTS = JOINTS[:4]
HOME = (0.0, -1.57, 1.57, 1.57, -1.57, 1.5)
JAW_OPEN, JAW_CLOSED = 1.5, 0.05
TCP_LOCAL = np.array([0.005, -0.085, 0.0])
SAFE_Z, GRASP_Z, DROP_Z = 0.095, 0.024, 0.070


@dataclass(frozen=True)
class DemoEvent:
    phase: str
    arms: tuple[str, ...]
    items: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class DemoReport:
    scene: str
    trace: str
    assigned: int
    placed: int
    misses: int
    events: tuple[DemoEvent, ...]


class _Arm:
    """Namespaced SO-101 control plus the minimum grasp state for the demo."""

    def __init__(self, mujoco, model, data, mount: ArmMount) -> None:
        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.mount = mount
        self.prefix = mount.prefix
        self.jaw_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}Fixed_Jaw"
        )
        self.qadr = {
            joint: model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}{joint}")
            ]
            for joint in JOINTS
        }
        self.dofadr = [
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}{joint}")
            ]
            for joint in IK_JOINTS
        ]
        self.actuator = {
            joint: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}{joint}"
            )
            for joint in JOINTS
        }
        self.ik_data = mujoco.MjData(model)
        self.held: tuple[int, list[int]] | None = None

    def tcp(self, data=None) -> np.ndarray:
        data = data if data is not None else self.data
        rotation = data.xmat[self.jaw_body].reshape(3, 3)
        return data.xpos[self.jaw_body] + rotation @ TCP_LOCAL

    def pose(self) -> dict[str, float]:
        return {joint: float(self.data.qpos[self.qadr[joint]]) for joint in JOINTS}

    def command(self, pose: dict[str, float]) -> None:
        for joint, value in pose.items():
            self.data.ctrl[self.actuator[joint]] = value

    def solve_ik(self, target: np.ndarray, yaw: float = 0.0) -> tuple[dict[str, float], float]:
        """Damped least-squares IK in the mounted arm's shared world frame."""

        mujoco = self._mujoco
        ik = self.ik_data
        ik.qpos[:] = self.data.qpos
        ik.qvel[:] = 0
        relative = target[:2] - np.array([self.mount.x, self.mount.y])
        base_yaw = np.arctan2(-relative[0], -relative[1])
        ik.qpos[self.qadr["Wrist_Roll"]] = np.clip(_wrap(yaw - base_yaw), -2.7, 2.7)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(240):
            mujoco.mj_kinematics(self.model, ik)
            mujoco.mj_comPos(self.model, ik)
            rotation = ik.xmat[self.jaw_body].reshape(3, 3)
            approach = rotation @ np.array([0.0, -1.0, 0.0])
            position_error = target - self.tcp(ik)
            orientation_error = np.cross(approach, np.array([0.0, 0.0, -1.0]))
            if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(orientation_error) < 1e-3:
                break
            mujoco.mj_jac(self.model, ik, jacp, jacr, self.tcp(ik), self.jaw_body)
            jacobian = np.vstack((jacp[:, self.dofadr], 0.25 * jacr[:, self.dofadr]))
            error = np.concatenate((position_error, 0.25 * orientation_error))
            delta = np.linalg.solve(
                jacobian.T @ jacobian + 0.12**2 * np.eye(len(self.dofadr)), jacobian.T @ error
            )
            for index, joint in enumerate(IK_JOINTS):
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.prefix}{joint}"
                )
                low, high = self.model.jnt_range[joint_id]
                address = self.qadr[joint]
                ik.qpos[address] = np.clip(ik.qpos[address] + np.clip(delta[index], -0.1, 0.1), low, high)
        pose = {joint: float(ik.qpos[self.qadr[joint]]) for joint in IK_JOINTS + ("Wrist_Roll",)}
        return pose, float(np.linalg.norm(target - self.tcp(ik)))

    def attach(self, body_id: int) -> bool:
        distance = float(np.linalg.norm(self.data.xpos[body_id] - self.tcp()))
        if distance > 0.052:
            return False
        joint_id = self.model.body_jntadr[body_id]
        geoms = [index for index in range(self.model.ngeom) if self.model.geom_bodyid[index] == body_id]
        for geom in geoms:
            self.model.geom_contype[geom] = 0
            self.model.geom_conaffinity[geom] = 0
        self.held = (joint_id, geoms)
        return True

    def carry(self) -> None:
        if self.held is None:
            return
        joint_id, _ = self.held
        address = self.model.jnt_qposadr[joint_id]
        self.data.qpos[address : address + 3] = self.tcp()
        self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof = self.model.jnt_dofadr[joint_id]
        self.data.qvel[dof : dof + 6] = 0

    def release(self, target_xy: tuple[float, float]) -> None:
        if self.held is None:
            return
        joint_id, geoms = self.held
        address = self.model.jnt_qposadr[joint_id]
        self.data.qpos[address : address + 3] = (target_xy[0], target_xy[1], 0.026)
        self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        dof = self.model.jnt_dofadr[joint_id]
        self.data.qvel[dof : dof + 6] = 0
        for geom in geoms:
            self.model.geom_contype[geom] = 1
            self.model.geom_conaffinity[geom] = 1
        self.held = None


def _wrap(value: float) -> float:
    return (value + np.pi) % (2 * np.pi) - np.pi


class DualArmTacoDemo:
    """Execute the built-in TACO-style jobs with simultaneous safe-side moves."""

    def __init__(self, scene_path: Path) -> None:
        try:
            import mujoco
        except ImportError as error:  # pragma: no cover - environment guidance
            raise RuntimeError("MuJoCo is optional. Run: uv sync --extra simulation") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.arms = {mount.name: _Arm(mujoco, self.model, self.data, mount) for mount in ARMS}
        for arm in self.arms.values():
            for joint, value in zip(JOINTS, HOME, strict=True):
                self.data.qpos[arm.qadr[joint]] = value
                self.data.ctrl[arm.actuator[joint]] = value
        mujoco.mj_forward(self.model, self.data)
        self.events: list[DemoEvent] = []

    def _step(self, count: int = 1) -> None:
        for _ in range(count):
            for arm in self.arms.values():
                arm.carry()
            self.mujoco.mj_step(self.model, self.data)

    def _move(self, targets: dict[str, tuple[float, float, float]], jaw: float, duration: float) -> bool:
        poses: dict[str, dict[str, float]] = {}
        for name, target in targets.items():
            pose, error = self.arms[name].solve_ik(np.array(target))
            if error > 0.010:
                self.events.append(DemoEvent("unreachable", (name,), (), f"IK residual {error * 1000:.1f} mm"))
                return False
            pose["Jaw"] = jaw
            poses[name] = pose
        starts = {name: self.arms[name].pose() for name in targets}
        steps = max(1, int(duration / self.model.opt.timestep))
        for index in range(steps):
            progress = (index + 1) / steps
            progress = 3 * progress**2 - 2 * progress**3
            for name, pose in poses.items():
                self.arms[name].command(
                    {joint: starts[name][joint] + progress * (pose[joint] - starts[name][joint]) for joint in JOINTS}
                )
            self._step()
        for _ in range(400):
            for name, pose in poses.items():
                self.arms[name].command(pose)
            self._step()
        return True

    def _body_for(self, item_id: str) -> int:
        return self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"item_{item_id}")

    def _execute_group(self, group: list[ArmAssignment]) -> int:
        names = tuple(assignment.arm for assignment in group)
        items = tuple(assignment.item.item_id for assignment in group)
        pick_above = {assignment.arm: (assignment.item.x, assignment.item.y, SAFE_Z) for assignment in group}
        pick_down = {assignment.arm: (assignment.item.x, assignment.item.y, GRASP_Z) for assignment in group}
        if not self._move(pick_above, JAW_OPEN, 0.65) or not self._move(pick_down, JAW_OPEN, 0.45):
            return 0
        self._move(pick_down, JAW_CLOSED, 0.18)
        attached: list[ArmAssignment] = []
        for assignment in group:
            if self.arms[assignment.arm].attach(self._body_for(assignment.item.item_id)):
                attached.append(assignment)
            else:
                distance = np.linalg.norm(
                    self.data.xpos[self._body_for(assignment.item.item_id)] - self.arms[assignment.arm].tcp()
                )
                self.events.append(
                    DemoEvent(
                        "miss",
                        (assignment.arm,),
                        (assignment.item.item_id,),
                        f"TCP {distance * 1000:.1f} mm from object",
                    )
                )
        if not attached:
            return 0
        names = tuple(assignment.arm for assignment in attached)
        items = tuple(assignment.item.item_id for assignment in attached)
        self.events.append(DemoEvent("pick", names, items, "grasp verified"))
        pick_lift = {assignment.arm: (assignment.item.x, assignment.item.y, SAFE_Z) for assignment in attached}
        if not self._move(pick_lift, JAW_CLOSED, 0.45):
            return 0
        bin_positions = {
            assignment.arm: bin_for(assignment.arm, assignment.bin_category) for assignment in attached
        }
        bin_above = {
            arm: (bin_spec.x, bin_spec.y, SAFE_Z) for arm, bin_spec in bin_positions.items()
        }
        bin_down = {
            arm: (bin_spec.x, bin_spec.y, DROP_Z) for arm, bin_spec in bin_positions.items()
        }
        if not self._move(bin_above, JAW_CLOSED, 0.75) or not self._move(bin_down, JAW_CLOSED, 0.40):
            return 0
        self._move(bin_down, JAW_OPEN, 0.18)
        for assignment in attached:
            target = bin_for(assignment.arm, assignment.bin_category)
            self.arms[assignment.arm].release((target.x, target.y))
        self._step(120)
        self.events.append(DemoEvent("place", names, items, "released into local bins"))
        self._move(bin_above, JAW_OPEN, 0.35)
        return len(attached)

    def run(self) -> tuple[list[ArmAssignment], int]:
        detections = [
            DetectedItem(item_id=item_id, category=category, x=x, y=y)
            for item_id, category, x, y in SAMPLE_ITEMS
        ]
        assignments = TwoArmDispatcher().dispatch(detections)
        pending = list(assignments)
        while any(not job.requires_handoff_clearance for job in pending):
            group: list[ArmAssignment] = []
            for arm in ("left", "right"):
                job = next(
                    (candidate for candidate in pending if candidate.arm == arm and not candidate.requires_handoff_clearance),
                    None,
                )
                if job is not None:
                    pending.remove(job)
                    group.append(job)
            self._execute_group(group)
        for job in pending:
            self.events.append(DemoEvent("reserve", (job.arm,), (job.item.item_id,), "exclusive shared-strip reservation"))
            self._execute_group([job])

        self._move({name: (mount.x, mount.y - 0.16, SAFE_Z) for name, mount in ((arm.name, arm) for arm in ARMS)}, JAW_OPEN, 0.5)
        placed = 0
        for assignment in assignments:
            body = self._body_for(assignment.item.item_id)
            position = self.data.xpos[body]
            target = bin_for(assignment.arm, assignment.bin_category)
            if abs(position[0] - target.x) < 0.065 and abs(position[1] - target.y) < 0.065:
                placed += 1
        return assignments, placed


def run_taco_demo(output_directory: Path | None = None) -> DemoReport:
    """Run the headless two-arm task and save a concise trace for the visual demo."""

    output_directory = (output_directory or repository_root() / "build").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    scene = build_station(output_directory / "dual_arm_taco_station.xml")
    demo = DualArmTacoDemo(scene)
    assignments, placed = demo.run()
    trace = output_directory / "dual_arm_taco_trace.json"
    trace.write_text(
        json.dumps(
            {
                "assigned": len(assignments),
                "placed": placed,
                "misses": len(assignments) - placed,
                "assignments": [
                    {
                        "item": assignment.item.item_id,
                        "category": assignment.bin_category,
                        "arm": assignment.arm,
                        "shared": assignment.requires_handoff_clearance,
                    }
                    for assignment in assignments
                ],
                "events": [asdict(event) for event in demo.events],
            },
            indent=2,
        )
    )
    return DemoReport(
        scene=str(scene),
        trace=str(trace),
        assigned=len(assignments),
        placed=placed,
        misses=len(assignments) - placed,
        events=tuple(demo.events),
    )
