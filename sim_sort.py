"""
Virtual trash sorting with an SO-ARM100/101 arm in MuJoCo.

Pipeline (same structure as the physical setup):
    top-down camera -> segmentation -> classification -> pixel-to-world
    -> inverse kinematics -> pick -> place into the bin of that class

Only two functions are simulation-specific and must be swapped for hardware:
    segment_objects() / classify_crop()  -> YOLO or a crop classifier
    Arm.send()                           -> lerobot SO101Follower.send_action()

Usage:
    uv run python sim_sort.py              # headless, writes sort_demo.mp4
    uv run mjpython sim_sort.py --viewer   # live window (macOS requires mjpython)
    uv run python sim_sort.py --probe      # reachability map of the workspace
    uv run python sim_sort.py --trace      # print object poses after every cycle

Measured on this model (SO-ARM100 from mujoco_menagerie, kinematically the
SO-101): with the gripper held vertical nothing above z = 0.12 m is reachable,
the position servos sag about 11 mm under load at full extension, and transfers
must be planned in joint space -- interpolating in Cartesian space lets the
solver flip to a mirrored elbow configuration halfway through the path.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

MENAGERIE = Path(__file__).parent / "mujoco_menagerie" / "trs_so_arm100"
# The generated scene is written next to the arm model so that the relative
# <include> and meshdir paths inside so_arm100.xml keep resolving.

# Waste classes. Colour stands in for the output of a real classifier.
# rgba is used to spawn the object, hsv_lo/hsv_hi to segment it back out.
CLASSES = {
    "bio":     dict(rgba="0.15 0.65 0.20 1", hsv_lo=(40, 120, 60),  hsv_hi=(80, 255, 255)),
    "paper":   dict(rgba="0.20 0.35 0.85 1", hsv_lo=(100, 120, 60), hsv_hi=(130, 255, 255)),
    "plastic": dict(rgba="0.95 0.80 0.10 1", hsv_lo=(20, 120, 60),  hsv_hi=(35, 255, 255)),
    "metal":   dict(rgba="0.85 0.15 0.15 1", hsv_lo=(0, 120, 60),   hsv_hi=(10, 255, 255)),
}

# Drop position above each bin, in metres, in the arm base frame.
BINS = {
    "bio":     (0.20, -0.12),
    "paper":   (0.20, -0.24),
    "plastic": (-0.20, -0.12),
    "metal":   (-0.20, -0.24),
}

# Objects spawned on the work surface: (class, x, y, yaw_deg)
SCENE_OBJECTS = [
    ("plastic", 0.01, -0.27, 20.0),
    ("bio", -0.06, -0.20, 0.0),
    ("paper", 0.06, -0.22, 60.0),
    ("metal", -0.02, -0.16, -30.0),
]

CAM_HEIGHT = 0.60           # top-down camera height above the work surface
CAM_CENTER = (0.0, -0.215)   # what the camera is centred on
CAM_FOVY = 45.0
IMG_W, IMG_H = 640, 480

BOARD_Z = 0.004             # work surface thickness (the "tablecloth")
CUBE = 0.014                # object half-size
GRASP_Z = BOARD_Z + CUBE    # height of the object centre
SAFE_Z = 0.10               # travel height
IK_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch"]
TCP_LOCAL = np.array([0.005, -0.085, 0.0])   # grasp point in the Fixed_Jaw frame
JAW_OPEN, JAW_CLOSED = 1.5, 0.05
RELEASE_CLEARANCE = 0.045    # how far below the TCP the object is let go
HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57, JAW_OPEN])
# Arm folded and swung aside so it does not occlude the work surface in the photo.
OBSERVE = np.array([1.55, -3.0, 2.9, 1.0, 0.0, JAW_OPEN])


def build_scene() -> Path:
    """Write the scene MJCF next to the arm model so meshdir/include resolve."""
    objs = []
    for i, (cls, x, y, yaw) in enumerate(SCENE_OBJECTS):
        objs.append(
            f'<body name="obj{i}" pos="{x} {y} {GRASP_Z}" euler="0 0 {np.deg2rad(yaw)}">'
            f'  <freejoint name="obj{i}_free"/>'
            f'  <geom type="box" size="{CUBE} {CUBE * 0.6} {CUBE}" rgba="{CLASSES[cls]["rgba"]}"'
            f'        mass="0.02" friction="1.5 0.02 0.001"/>'
            f"</body>"
        )
    bins = []
    for cls, (x, y) in BINS.items():
        c = CLASSES[cls]["rgba"][:-1]
        w, wall, t = 0.045, 0.004, 0.025
        bins.append(
            f'<geom name="bin_{cls}" type="box" pos="{x} {y} 0.004" size="{w} {w} 0.004"'
            f'      rgba="{c}0.9"/>'
            f'<geom type="box" pos="{x + w} {y} {t / 2}" size="{wall} {w} {t}" rgba="{c}0.5"/>'
            f'<geom type="box" pos="{x - w} {y} {t / 2}" size="{wall} {w} {t}" rgba="{c}0.5"/>'
            f'<geom type="box" pos="{x} {y + w} {t / 2}" size="{w} {wall} {t}" rgba="{c}0.5"/>'
            f'<geom type="box" pos="{x} {y - w} {t / 2}" size="{w} {wall} {t}" rgba="{c}0.5"/>'
        )
    xml = f"""<mujoco model="so_arm100 sorting">
  <include file="so_arm100.xml"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <global azimuth="130" elevation="-30"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.28 0.32" rgb2="0.05 0.05 0.07"
             width="512" height="3072"/>
    <material name="floor_mat" rgba="0.18 0.18 0.20 1"/>
    <material name="board_mat" rgba="0.92 0.92 0.92 1"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 1.0" dir="-0.2 0.2 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>
    <geom name="board" type="box" pos="{CAM_CENTER[0]} {CAM_CENTER[1]} {BOARD_Z / 2}"
          size="0.10 0.075 {BOARD_Z / 2}" material="board_mat"/>
    <camera name="topdown" pos="{CAM_CENTER[0]} {CAM_CENTER[1]} {CAM_HEIGHT + BOARD_Z}"
            xyaxes="1 0 0 0 1 0" fovy="{CAM_FOVY}"/>
    <camera name="scene" pos="0.45 -0.55 0.40" xyaxes="0.75 0.66 0 -0.30 0.34 0.89"/>
    {"".join(bins)}
    {"".join(objs)}
  </worldbody>
</mujoco>
"""
    path = MENAGERIE / "_sorting_scene.xml"
    path.write_text(xml)
    return path


class Arm:
    """Cartesian control for the arm. Replace send() to drive real hardware."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.jaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in IK_JOINTS + ["Wrist_Roll", "Jaw"]}
        self.dofadr = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                       for n in IK_JOINTS]
        self.act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in IK_JOINTS + ["Wrist_Roll", "Jaw"]}
        self.ik_data = mujoco.MjData(model)

    def tcp(self, data=None):
        d = data if data is not None else self.d
        rot = d.xmat[self.jaw_body].reshape(3, 3)
        return d.xpos[self.jaw_body] + rot @ TCP_LOCAL

    def solve_ik(self, target, yaw, keep_vertical=True, iters=200, damping=0.12):
        """Damped least squares on 4 joints: position + keep the jaws pointing down."""
        d = self.ik_data
        d.qpos[:] = self.d.qpos
        d.qvel[:] = 0
        # Base rotation is what aims the arm, so the finger yaw is handled by Wrist_Roll.
        base_yaw = np.arctan2(-target[0], -target[1])
        d.qpos[self.qadr["Wrist_Roll"]] = np.clip(_wrap(yaw - base_yaw), -2.7, 2.7)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            rot = d.xmat[self.jaw_body].reshape(3, 3)
            approach = rot @ np.array([0.0, -1.0, 0.0])      # jaws point along local -Y
            err_p = target - self.tcp(d)
            w = 0.25 if keep_vertical else 0.0
            err_r = np.cross(approach, np.array([0.0, 0.0, -1.0]))
            if np.linalg.norm(err_p) < 1e-4 and (not keep_vertical
                                                 or np.linalg.norm(err_r) < 1e-3):
                break
            mujoco.mj_jac(self.m, d, jacp, jacr, self.tcp(d), self.jaw_body)
            J = np.vstack([jacp[:, self.dofadr], w * jacr[:, self.dofadr]])
            err = np.concatenate([err_p, w * err_r])
            dq = np.linalg.solve(J.T @ J + damping**2 * np.eye(len(self.dofadr)), J.T @ err)
            dq = np.clip(dq, -0.1, 0.1)
            for k, name in enumerate(IK_JOINTS):
                lo, hi = self.m.jnt_range[mujoco.mj_name2id(
                    self.m, mujoco.mjtObj.mjOBJ_JOINT, name)]
                a = self.qadr[name]
                d.qpos[a] = np.clip(d.qpos[a] + dq[k], lo, hi)
        return {n: float(d.qpos[self.qadr[n]]) for n in IK_JOINTS + ["Wrist_Roll"]}, \
               float(np.linalg.norm(target - self.tcp(d)))

    def send(self, targets: dict):
        for name, value in targets.items():
            self.d.ctrl[self.act[name]] = value


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Sim:
    def __init__(self, scene_path, record=True):
        self.m = mujoco.MjModel.from_xml_path(str(scene_path))
        self.d = mujoco.MjData(self.m)
        # mj_resetData keeps the spawn pose of the free-joint objects; the arm
        # keyframe only covers the first 6 qpos, so apply it by hand.
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:6] = HOME
        self.d.ctrl[:6] = HOME
        mujoco.mj_forward(self.m, self.d)
        self.arm = Arm(self.m, self.d)
        self.held = None
        self.jaw = JAW_OPEN
        self.frames = [] if record else None
        self.renderer = mujoco.Renderer(self.m, IMG_H, IMG_W) if record else None
        self.cam_renderer = mujoco.Renderer(self.m, IMG_H, IMG_W)
        self.overlay = None
        self.viewer = None

    # ---------- vision ----------
    def grab_topdown(self):
        self.cam_renderer.update_scene(self.d, camera="topdown")
        return self.cam_renderer.render()

    def pixel_to_world(self, u, v):
        """Top-down pinhole camera over a flat plane. On hardware this becomes
        cv2.getPerspectiveTransform() over four ArUco markers at known points."""
        scale = 2.0 * CAM_HEIGHT * np.tan(np.deg2rad(CAM_FOVY) / 2) / IMG_H
        x = CAM_CENTER[0] + (u - IMG_W / 2 + 0.5) * scale
        y = CAM_CENTER[1] - (v - IMG_H / 2 + 0.5) * scale
        return float(x), float(y)

    def detect(self):
        """Background is uniform, so contours are enough. Replace with YOLO."""
        rgb = self.grab_topdown()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        found = []
        for cls, cfg in CLASSES.items():
            mask = cv2.inRange(hsv, np.array(cfg["hsv_lo"]), np.array(cfg["hsv_hi"]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 300:
                    continue
                (cu, cv_), (w, h), angle = cv2.minAreaRect(c)
                x, y = self.pixel_to_world(cu, cv_)
                if not (-0.11 < x < 0.11 and -0.30 < y < -0.13):
                    continue  # outside the work surface: that is a bin, not an object
                # Grasp across the short side of the object.
                yaw = np.deg2rad(angle if w < h else angle + 90.0)
                found.append(dict(cls=cls, x=x, y=y, yaw=_wrap(yaw), px=(int(cu), int(cv_))))
                box = np.intp(cv2.boxPoints(((cu, cv_), (w, h), angle)))
                cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
                cv2.putText(vis, f"{cls} {x:+.3f},{y:+.3f}", (int(cu) - 60, int(cv_) - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        self.overlay = vis
        return found

    # ---------- motion ----------
    def step(self, n=1):
        for _ in range(n):
            if self.held is not None:
                self._carry()
            mujoco.mj_step(self.m, self.d)
            if self.viewer is not None:
                self.viewer.sync()
            if self.frames is not None and self.d.time % 0.04 < self.m.opt.timestep:
                self._capture()

    def _capture(self):
        self.renderer.update_scene(self.d, camera="scene")
        scene = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        panel = self.overlay if self.overlay is not None else scene
        self.frames.append(np.hstack([scene, panel]))

    def _carry(self):
        """Kinematic grasp: the held object tracks the TCP. Friction grasping in
        simulation is unreliable; standard benchmarks do the same."""
        adr = self.m.jnt_qposadr[self.held["joint"]]
        self.d.qpos[adr:adr + 3] = self.arm.tcp()
        self.d.qpos[adr + 3:adr + 7] = self.held["quat"]
        self.d.qvel[self.m.jnt_dofadr[self.held["joint"]]:
                    self.m.jnt_dofadr[self.held["joint"]] + 6] = 0

    def move_to(self, x, y, z, yaw, duration=0.9, keep_vertical=True):
        start = self.arm.tcp().copy()
        target = np.array([x, y, z])
        steps = int(duration / self.m.opt.timestep)
        for i in range(steps):
            a = (i + 1) / steps
            a = 3 * a**2 - 2 * a**3
            q, _ = self.arm.solve_ik(start + a * (target - start), yaw, keep_vertical)
            self.arm.send({**q, "Jaw": self.jaw})
            self.step()
        # Hold the final command until the servos actually settle there.
        q, _ = self.arm.solve_ik(target, yaw, keep_vertical)
        for _ in range(600):
            self.arm.send({**q, "Jaw": self.jaw})
            self.step()
            if np.linalg.norm(self.arm.tcp() - target) < 0.004:
                break
        return float(np.linalg.norm(self.arm.tcp() - target))

    def move_joints(self, x, y, z, yaw, duration=1.0, keep_vertical=False):
        """Joint-space move: solve IK once, then ramp the servo targets. Used for
        long transfers, where interpolating in Cartesian space can make the
        solver jump to a mirrored elbow configuration mid-path."""
        q, _ = self.arm.solve_ik(np.array([x, y, z]), yaw, keep_vertical)
        start = {n: float(self.d.qpos[self.arm.qadr[n]]) for n in q}
        steps = int(duration / self.m.opt.timestep)
        for i in range(steps):
            a = (i + 1) / steps
            a = 3 * a**2 - 2 * a**3
            self.arm.send({n: start[n] + a * (q[n] - start[n]) for n in q})
            self.arm.send({"Jaw": self.jaw})
            self.step()
        for _ in range(400):
            self.arm.send({**q, "Jaw": self.jaw})
            self.step()
            if np.linalg.norm(self.arm.tcp() - np.array([x, y, z])) < 0.006:
                break
        return float(np.linalg.norm(self.arm.tcp() - np.array([x, y, z])))

    def set_jaw(self, value, duration=0.35):
        self.jaw = value
        self.arm.send({"Jaw": value})
        self.step(int(duration / self.m.opt.timestep))

    def attach_nearest(self):
        tcp = self.arm.tcp()
        best, best_d = None, 1e9
        for i in range(len(SCENE_OBJECTS)):
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
            dist = np.linalg.norm(self.d.xpos[bid] - tcp)
            if dist < best_d:
                best, best_d = bid, dist
        if best_d > 0.045:
            return False
        jid = self.m.body_jntadr[best]
        adr = self.m.jnt_qposadr[jid]
        geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == best]
        # Disable contact for the held object: it is driven kinematically, so
        # leaving it collidable makes it fight the gripper geometry.
        for g in geoms:
            self.m.geom_contype[g] = 0
            self.m.geom_conaffinity[g] = 0
        self.held = dict(body=best, joint=jid, geoms=geoms,
                         quat=self.d.qpos[adr + 3:adr + 7].copy())
        return True

    def release(self):
        """Hand the object back to the physics engine, clear of the fingers.
        Dropping it exactly at the TCP leaves it overlapping the jaw blades, and
        it then rides up with the arm or gets flicked away when contact resumes."""
        if self.held is None:
            return
        adr = self.m.jnt_qposadr[self.held["joint"]]
        dof = self.m.jnt_dofadr[self.held["joint"]]
        self.d.qpos[adr:adr + 3] = self.arm.tcp() - np.array([0.0, 0.0, RELEASE_CLEARANCE])
        self.d.qvel[dof:dof + 6] = 0
        for g in self.held["geoms"]:
            self.m.geom_contype[g] = 1
            self.m.geom_conaffinity[g] = 1
        self.held = None

    def goto_pose(self, pose, settle=450):
        for i, name in enumerate(["Rotation", "Pitch", "Elbow",
                                  "Wrist_Pitch", "Wrist_Roll", "Jaw"]):
            self.d.ctrl[self.arm.act[name]] = pose[i]
        self.jaw = pose[5]
        self.step(settle)


def probe(sim, zs=(0.02, 0.08, 0.12), keep_vertical=True):
    """Where can the arm actually put the gripper? Use this to place the bins and
    the pick area before touching hardware. A dot means under 5 mm of error."""
    print("ik error in mm, '.' = reachable, gripper vertical:", keep_vertical)
    xs = np.arange(-0.28, 0.29, 0.04)
    for z in zs:
        print(f"z = {z:.2f}        x: " + " ".join(f"{x:+.2f}" for x in xs))
        for y in np.arange(-0.10, -0.33, -0.04):
            row = ""
            for x in xs:
                _, e = sim.arm.solve_ik(np.array([x, y, z]), 0.0, keep_vertical)
                row += "   .  " if e < 0.005 else f" {min(999, int(e * 1000)):4d} "
            print(f"  y = {y:+.2f}  {row}")


def main():
    global MENAGERIE
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="live window (mjpython on macOS)")
    ap.add_argument("--probe", action="store_true", help="print a reachability map and exit")
    ap.add_argument("--trace", action="store_true", help="print object poses each cycle")
    ap.add_argument("--menagerie", default=str(MENAGERIE), help="path to trs_so_arm100")
    ap.add_argument("--out", default="sort_demo.mp4")
    args = ap.parse_args()
    MENAGERIE = Path(args.menagerie)

    if not (MENAGERIE / "so_arm100.xml").exists():
        sys.exit(f"Arm model not found at {MENAGERIE}. Clone mujoco_menagerie first.")

    if args.probe:
        probe(Sim(build_scene(), record=False))
        return

    sim = Sim(build_scene(), record=not args.viewer)
    if args.viewer:
        from mujoco import viewer as mj_viewer
        sim.viewer = mj_viewer.launch_passive(sim.m, sim.d)

    sim.goto_pose(OBSERVE)
    handled, misses = 0, 0
    for cycle in range(len(SCENE_OBJECTS) + 3):
        sim.goto_pose(OBSERVE, settle=350)
        remaining = sim.detect()
        print(f"cycle {cycle}: {len(remaining)} object(s) on the surface")
        if not remaining:
            break
        det = remaining[0]
        bx, by = BINS[det["cls"]]
        print(f"  {det['cls']:8s} at ({det['x']:+.3f}, {det['y']:+.3f}) "
              f"yaw {np.rad2deg(det['yaw']):+.0f} deg -> bin ({bx:+.2f}, {by:+.2f})")
        # Pass through the home pose so the Cartesian path never sweeps
        # sideways across the work surface and scatters the other objects.
        sim.goto_pose(HOME, settle=300)
        sim.move_to(det["x"], det["y"], SAFE_Z, det["yaw"], duration=1.1)
        # Aim slightly low: position servos sag a few mm under load, on the real
        # arm as well. The grasp is verified afterwards, not by the IK residual.
        sim.move_to(det["x"], det["y"], GRASP_Z - 0.004, det["yaw"], duration=0.7)
        sim.set_jaw(JAW_CLOSED)
        if not sim.attach_nearest():
            print("    grasp missed")
            misses += 1
            continue
        sim.move_to(det["x"], det["y"], SAFE_Z, det["yaw"], duration=0.6)
        # The gripper need not stay vertical while carrying, which widens the reach.
        sim.move_joints(bx, by, SAFE_Z, 0.0, duration=1.4)
        err = sim.move_joints(bx, by, 0.075, 0.0, duration=0.5)
        print(f"    above bin, tcp error {err * 1000:.0f} mm")
        # Open the jaw before handing the object back to the physics engine:
        # re-enabling contact while it is still between closed fingers makes the
        # solver resolve the penetration explosively and fire it across the table.
        sim.set_jaw(JAW_OPEN)
        sim.release()
        sim.step(250)
        # Lift straight out of the tray, otherwise the arm sweeps the object out
        # of the bin on its way back to the observation pose.
        sim.move_joints(bx, by, SAFE_Z + 0.04, 0.0, duration=0.5)
        handled += 1
        if args.trace:
            for i in range(len(SCENE_OBJECTS)):
                b = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
                print("      obj%d %s" % (i, np.round(sim.d.xpos[b], 3)))
    sim.goto_pose(HOME)

    # Score by where the objects actually ended up, not by what the arm believes.
    placed = 0
    for i, (cls, *_rest) in enumerate(SCENE_OBJECTS):
        bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
        pos = sim.d.xpos[bid]
        bx, by = BINS[cls]
        if abs(pos[0] - bx) < 0.06 and abs(pos[1] - by) < 0.06:
            placed += 1
        else:
            print(f"  obj{i} ({cls}) ended at {np.round(pos, 3)}")
    print(f"sorted correctly: {placed}/{len(SCENE_OBJECTS)}  (picks {handled}, misses {misses})")

    if sim.frames:
        cv2.imwrite("topdown_detection.png", sim.overlay)
        h, w = sim.frames[0].shape[:2]
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        if writer.isOpened():
            for f in sim.frames:
                writer.write(f)
            writer.release()
            print(f"wrote {args.out} ({len(sim.frames)} frames) and topdown_detection.png")
        else:
            Path("frames").mkdir(exist_ok=True)
            for i, f in enumerate(sim.frames):
                cv2.imwrite(f"frames/{i:04d}.png", f)
            print(f"video encoder unavailable, wrote {len(sim.frames)} frames to frames/")


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    main()
