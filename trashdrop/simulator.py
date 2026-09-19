"""Two-arm sorting cell in MuJoCo.

What this does and does not prove:

* It proves the *motion* stack -- reachability of every bin and every point of
  the pick zone from the owning arm, the transfer paths, the shared-zone
  serialisation, and the scoring.
* It does not prove *grasping*. The grasp is kinematic: once the jaws close on
  an item its contacts are disabled and it tracks the tool centre point.
  Frictional grasping in simulation is not predictive of a real gripper on a
  crushed can, and pretending otherwise would hide the one thing that has to
  be tested on hardware.
* It does not prove *perception*. See trashdrop/perception/color.py.

The scoring is deliberately unforgiving: an item is released from where the
tool actually is, physics decides where it lands, and success is measured
from the resulting position. A release that teleports the item into the bin
would make the number meaningless.

Constants here were measured on this model and carry over to the real arm:
the position servos sag a few millimetres under load, so the grasp aims low;
long transfers interpolate in joint space because a Cartesian path lets the
solver flip to a mirrored elbow halfway; and the jaws open before the item is
handed back to physics, or the contact solver resolves the overlap explosively.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .control import ArmController
from .planning import ArmAssignment, DetectedItem, TwoArmDispatcher
from .scene_builder import DEFAULT_OBJECTS, SceneObject, build_model
from .station import (
    ARMS,
    BIN_TOLERANCE,
    CAMERA,
    GRASP_Z,
    HOME_POSE,
    JAW_CLOSED,
    JAW_OPEN,
    JOINTS,
    MIXED_CATEGORY,
    SAFE_Z,
    STOW_POSE,
    bin_for,
    is_graspable,
)

# Aim this far below the item's centre: the position servos sag under load.
GRASP_SAG_COMPENSATION = 0.004
# How far below the tool the item is let go, so it clears the jaw blades.
RELEASE_CLEARANCE = 0.045
# Height above a bin at which the item is dropped.
BIN_DROP_Z = 0.075
# Radius within which the jaws are considered to have caught the item.
GRASP_CAPTURE_RADIUS = 0.045


@dataclass
class CycleReport:
    """What happened to one item."""

    item_id: str
    category: str
    arm: str
    bin_key: str
    detected_at: tuple[float, float]
    grasped: bool
    tcp_error_mm: float = 0.0
    note: str = ""


@dataclass
class RunReport:
    """Outcome of a full sorting run, scored from final item positions."""

    detected: int
    picked: int
    missed: int
    placed: int
    total_objects: int
    cycles: list[CycleReport] = field(default_factory=list)
    misplaced: list[dict] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.placed == self.total_objects and self.missed == 0

    def to_json(self, path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


class SortingCell:
    """The simulated cell: two arms, one camera, one shared pick zone."""

    def __init__(
        self,
        objects: tuple[SceneObject, ...] = DEFAULT_OBJECTS,
        *,
        detector=None,
        record: bool = True,
        viewer: bool = False,
    ) -> None:
        import mujoco

        self.mj = mujoco
        self.objects = objects
        self.model = build_model(objects)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.arms = {
            mount.name: ArmController(mujoco, self.model, self.data, mount) for mount in ARMS
        }
        # Start stowed, not at HOME: two arms at HOME have their jaws 4 cm
        # apart over the middle of the table and jam before anything runs.
        for arm in self.arms.values():
            self._apply_pose(arm, STOW_POSE, command_only=False)
        mujoco.mj_forward(self.model, self.data)

        if detector is None:
            from .perception.color import ColorDetector

            detector = ColorDetector()
        self.detector = detector

        self.viewer = None
        self._viewer_requested = viewer
        self.frames: list[np.ndarray] | None = [] if (record and not viewer) else None
        self.renderer = (
            mujoco.Renderer(self.model, CAMERA.height_px, CAMERA.width)
            if self.frames is not None
            else None
        )
        self.camera_renderer = mujoco.Renderer(self.model, CAMERA.height_px, CAMERA.width)
        self.overlay: np.ndarray | None = None

    # --- low level ---------------------------------------------------------

    def _apply_pose(self, arm: ArmController, pose, command_only: bool = True) -> None:
        for joint, value in zip(JOINTS, pose):
            self.data.ctrl[arm.actuator[joint]] = value
            if not command_only:
                self.data.qpos[arm.qadr[joint]] = value
        arm.jaw = pose[5]

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            for arm in self.arms.values():
                self._carry(arm)
            self.mj.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            if self.frames is not None and self.data.time % 0.04 < self.model.opt.timestep:
                self._capture()

    def _carry(self, arm: ArmController) -> None:
        """A held item tracks the tool, keeping the orientation it was caught in."""

        if arm.held is None:
            return
        address = self.model.jnt_qposadr[arm.held["joint"]]
        dof = self.model.jnt_dofadr[arm.held["joint"]]
        self.data.qpos[address : address + 3] = arm.tcp()
        self.data.qpos[address + 3 : address + 7] = arm.held["quat"]
        self.data.qvel[dof : dof + 6] = 0

    def _capture(self) -> None:
        import cv2

        self.renderer.update_scene(self.data, camera="operator")
        scene = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        panel = self.overlay if self.overlay is not None else scene
        if panel.shape != scene.shape:
            panel = cv2.resize(panel, (scene.shape[1], scene.shape[0]))
        self.frames.append(np.hstack([scene, panel]))

    # --- perception --------------------------------------------------------

    def look(self) -> list:
        """Move both arms clear of the camera, then detect."""

        for arm in self.arms.values():
            self._apply_pose(arm, STOW_POSE)
        self.step(350)
        self.camera_renderer.update_scene(self.data, camera="topdown")
        frame = self.camera_renderer.render()
        detections = self.detector.detect(frame)
        self.overlay = getattr(self.detector, "overlay", None)
        return detections

    # --- motion ------------------------------------------------------------

    def goto_pose(self, arm_name: str, pose, settle: int = 450) -> None:
        self._apply_pose(self.arms[arm_name], pose)
        self.step(settle)

    def set_jaw(self, arm_name: str, value: float, duration: float = 0.35) -> None:
        arm = self.arms[arm_name]
        arm.jaw = value
        arm.send({"Jaw": value})
        self.step(int(duration / self.model.opt.timestep))

    def move_to(
        self,
        arm_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration: float = 0.9,
        keep_vertical: bool = True,
    ) -> float:
        """Straight-line Cartesian move. Use for the short approach and lift."""

        arm = self.arms[arm_name]
        start = arm.tcp().copy()
        target = np.array([x, y, z])
        steps = int(duration / self.model.opt.timestep)
        for index in range(steps):
            alpha = (index + 1) / steps
            alpha = 3 * alpha**2 - 2 * alpha**3
            pose, _ = arm.solve_ik(start + alpha * (target - start), yaw, keep_vertical)
            arm.send({**pose, "Jaw": arm.jaw})
            self.step()
        pose, _ = arm.solve_ik(target, yaw, keep_vertical)
        for _ in range(600):
            arm.send({**pose, "Jaw": arm.jaw})
            self.step()
            if np.linalg.norm(arm.tcp() - target) < 0.004:
                break
        return float(np.linalg.norm(arm.tcp() - target))

    def move_joints(
        self,
        arm_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        duration: float = 1.0,
        keep_vertical: bool = False,
    ) -> float:
        """Solve once, then ramp joint targets. Use for long transfers."""

        arm = self.arms[arm_name]
        target = np.array([x, y, z])
        pose, _ = arm.solve_ik(target, yaw, keep_vertical)
        start = {joint: float(self.data.qpos[arm.qadr[joint]]) for joint in pose}
        steps = int(duration / self.model.opt.timestep)
        for index in range(steps):
            alpha = (index + 1) / steps
            alpha = 3 * alpha**2 - 2 * alpha**3
            arm.send({j: start[j] + alpha * (pose[j] - start[j]) for j in pose})
            arm.send({"Jaw": arm.jaw})
            self.step()
        for _ in range(400):
            arm.send({**pose, "Jaw": arm.jaw})
            self.step()
            if np.linalg.norm(arm.tcp() - target) < 0.006:
                break
        return float(np.linalg.norm(arm.tcp() - target))

    # --- grasp -------------------------------------------------------------

    def _held_bodies(self) -> set[int]:
        return {arm.held["body"] for arm in self.arms.values() if arm.held is not None}

    def attach_nearest(self, arm_name: str) -> bool:
        """Catch the closest free item, if the jaws actually reached it."""

        arm = self.arms[arm_name]
        tcp = arm.tcp()
        taken = self._held_bodies()
        best, best_distance = None, 1e9
        for index in range(len(self.objects)):
            body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"item_{index}")
            if body in taken:
                continue
            distance = float(np.linalg.norm(self.data.xpos[body] - tcp))
            if distance < best_distance:
                best, best_distance = body, distance
        if best is None or best_distance > GRASP_CAPTURE_RADIUS:
            return False

        joint = self.model.body_jntadr[best]
        address = self.model.jnt_qposadr[joint]
        geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == best]
        # The item is driven kinematically from here, so leaving it collidable
        # makes it fight the gripper geometry.
        for geom in geoms:
            self.model.geom_contype[geom] = 0
            self.model.geom_conaffinity[geom] = 0
        arm.held = {
            "body": best,
            "joint": joint,
            "geoms": geoms,
            "quat": self.data.qpos[address + 3 : address + 7].copy(),
        }
        return True

    def release(self, arm_name: str) -> None:
        """Hand the item back to physics, clear of the fingers.

        It is dropped from where the tool actually is -- not moved to the bin.
        Whether it lands in the bin is then a physical result, which is the
        only thing worth scoring.
        """

        arm = self.arms[arm_name]
        if arm.held is None:
            return
        address = self.model.jnt_qposadr[arm.held["joint"]]
        dof = self.model.jnt_dofadr[arm.held["joint"]]
        self.data.qpos[address : address + 3] = arm.tcp() - np.array([0.0, 0.0, RELEASE_CLEARANCE])
        self.data.qvel[dof : dof + 6] = 0
        for geom in arm.held["geoms"]:
            self.model.geom_contype[geom] = 1
            self.model.geom_conaffinity[geom] = 1
        arm.held = None

    # --- one item ----------------------------------------------------------

    def execute(self, assignment: ArmAssignment, detection) -> CycleReport:
        """Run one full pick-and-place. Only this arm moves."""

        arm_name = assignment.arm
        report = CycleReport(
            item_id=assignment.item.item_id,
            category=assignment.item.category,
            arm=arm_name,
            bin_key=assignment.bin_key,
            detected_at=(assignment.item.x, assignment.item.y),
            grasped=False,
        )

        x, y, yaw = assignment.item.x, assignment.item.y, assignment.item.yaw
        # Pass through home so the Cartesian approach never sweeps sideways
        # across the pick zone and scatters the remaining items.
        self.goto_pose(arm_name, HOME_POSE, settle=300)
        self.move_to(arm_name, x, y, SAFE_Z, yaw, duration=1.1)
        self.move_to(arm_name, x, y, GRASP_Z - GRASP_SAG_COMPENSATION, yaw, duration=0.7)
        self.set_jaw(arm_name, JAW_CLOSED)

        if not self.attach_nearest(arm_name):
            report.note = "grasp missed"
            self.set_jaw(arm_name, JAW_OPEN)
            self.goto_pose(arm_name, STOW_POSE, settle=250)
            return report
        report.grasped = True

        self.move_to(arm_name, x, y, SAFE_Z, yaw, duration=0.6)
        # The gripper need not stay vertical while carrying, which widens reach.
        self.move_joints(arm_name, assignment.bin_x, assignment.bin_y, SAFE_Z, duration=1.4)
        error = self.move_joints(
            arm_name, assignment.bin_x, assignment.bin_y, BIN_DROP_Z, duration=0.5
        )
        report.tcp_error_mm = error * 1000.0

        # Open before handing back to physics, or the solver resolves the
        # overlap between the item and the closed fingers explosively.
        self.set_jaw(arm_name, JAW_OPEN)
        self.release(arm_name)
        self.step(250)
        # Lift straight out, otherwise the arm sweeps the item back out.
        self.move_joints(
            arm_name, assignment.bin_x, assignment.bin_y, SAFE_Z + 0.04, duration=0.5
        )
        self.goto_pose(arm_name, STOW_POSE, settle=250)
        return report

    # --- the run -----------------------------------------------------------

    def run(self, max_cycles: int | None = None, verbose: bool = True) -> RunReport:
        if self._viewer_requested and self.viewer is None:
            from mujoco import viewer as mj_viewer

            self.viewer = mj_viewer.launch_passive(self.model, self.data)

        dispatcher = TwoArmDispatcher()
        limit = max_cycles if max_cycles is not None else len(self.objects) + 3
        cycles: list[CycleReport] = []
        picked = missed = 0
        detected_total = 0
        last_arm: str | None = None

        for cycle in range(limit):
            detections = self.look()
            if verbose:
                print(f"cycle {cycle}: {len(detections)} item(s) in the pick zone")
            if not detections:
                break
            detected_total = max(detected_total, len(detections))

            items = []
            for index, detection in enumerate(detections):
                category = detection.category
                note = ""
                if not is_graspable(detection.width):
                    category = MIXED_CATEGORY
                    note = "too wide for the jaws"
                items.append(
                    (
                        DetectedItem(
                            item_id=f"c{cycle}_{index}",
                            category=category,
                            x=detection.x,
                            y=detection.y,
                            yaw=detection.yaw,
                            confidence=detection.confidence,
                        ),
                        detection,
                        note,
                    )
                )

            # order() alternates arms so the cell stays visibly two-armed;
            # only the first job runs, because both arms share this volume and
            # execution is serialised on purpose.
            assignments = dispatcher.order(
                dispatcher.dispatch([i for i, _, _ in items]), last=last_arm
            )
            assignment = assignments[0]
            by_id = {item.item_id: (detection, note) for item, detection, note in items}
            detection, note = by_id[assignment.item.item_id]

            if verbose:
                flag = " [rerouted]" if assignment.rerouted else ""
                print(
                    f"  {assignment.item.category:8s} at "
                    f"({assignment.item.x:+.3f}, {assignment.item.y:+.3f}) "
                    f"-> {assignment.arm:5s} -> bin {assignment.bin_key}"
                    f"{flag}{' ' + note if note else ''}"
                )

            report = self.execute(assignment, detection)
            report.note = report.note or note
            last_arm = assignment.arm
            cycles.append(report)
            if report.grasped:
                picked += 1
            else:
                missed += 1
            if verbose and report.grasped:
                print(f"    above bin, tcp error {report.tcp_error_mm:.0f} mm")
            elif verbose:
                print("    grasp missed")

        for name in self.arms:
            self.goto_pose(name, STOW_POSE, settle=200)

        placed, misplaced = self._score()
        result = RunReport(
            detected=detected_total,
            picked=picked,
            missed=missed,
            placed=placed,
            total_objects=len(self.objects),
            cycles=cycles,
            misplaced=misplaced,
        )
        if verbose:
            print(
                f"sorted correctly: {placed}/{len(self.objects)}  "
                f"(picks {picked}, misses {missed})"
            )
        return result

    def _score(self) -> tuple[int, list[dict]]:
        """Score from where the items physically ended up."""

        placed = 0
        misplaced: list[dict] = []
        for index, item in enumerate(self.objects):
            body = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, f"item_{index}")
            position = self.data.xpos[body]
            target = bin_for(item.category)
            if (
                abs(position[0] - target.x) < BIN_TOLERANCE
                and abs(position[1] - target.y) < BIN_TOLERANCE
            ):
                placed += 1
            else:
                misplaced.append(
                    {
                        "item_id": item.item_id,
                        "category": item.category,
                        "expected_bin": [round(target.x, 3), round(target.y, 3)],
                        "ended_at": [round(float(v), 3) for v in position],
                    }
                )
        return placed, misplaced

    # --- output ------------------------------------------------------------

    def write_video(self, path: Path, fps: int = 25) -> Path | None:
        import cv2

        if not self.frames:
            return None
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            return None
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        return path

    def write_overlay(self, path: Path) -> Path | None:
        import cv2

        if self.overlay is None:
            return None
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.overlay)
        return path
