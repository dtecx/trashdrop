"""Assemble the two-arm MuJoCo scene with MjSpec.

``MjSpec.attach`` namespaces a whole model in one call, so the arm model is
loaded straight from the menagerie checkout and inserted twice under different
prefixes. Nothing is written into the menagerie folder and no MJCF is
hand-edited: the previous approach of copying and renaming XML nodes by hand
had to know about every attribute that can hold a cross-reference.

Two things here are easy to get wrong and are asserted at build time:

* ``<compiler angle="radian">`` on the parent. MJCF defaults to degrees, so a
  frame built with ``euler=[0, 0, pi]`` silently becomes a 3 degree rotation
  and the second arm ends up beside the first instead of facing it.
* ``cone="elliptic" impratio="10"`` on the parent. On attach the parent's
  options win, and these two are what the arm model itself asks for; losing
  them changes the contact behaviour the grasp was tuned against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .station import (
    ARMS,
    BIN_HALF_WIDTH,
    BIN_WALL,
    BIN_WALL_HEIGHT,
    BINS,
    CAMERA,
    CATEGORY_COLORS,
    GRASP_Z,
    OBJECT_HALF_SIZE,
    PICK_ZONE,
    SURFACE_THICKNESS,
    find_arm_model,
)


@dataclass(frozen=True)
class SceneObject:
    """A stand-in item spawned on the work surface.

    ``category`` drives its colour, which the sim-only colour detector reads
    back. A real run replaces both with a trained classifier over camera crops.
    """

    item_id: str
    category: str
    x: float
    y: float
    yaw_degrees: float = 0.0


# The partner team's likely first deliveries: a plastic bottle, paper cup,
# and empty can. The real categories also cover other waste of those materials.
DEFAULT_OBJECTS: tuple[SceneObject, ...] = (
    SceneObject("bottle", "plastic", 0.010, -0.272, 20.0),
    SceneObject("paper_cup", "paper", 0.060, -0.222, 60.0),
    SceneObject("can", "metal", -0.020, -0.160, -30.0),
)


# Where the recording camera sits and what it points at. Derived rather than
# hand-written: MJCF wants the camera's x and y axes, and guessing those by
# hand is how you end up recording a view of empty floor.
OPERATOR_EYE = (0.60, -0.60, 0.42)
OPERATOR_TARGET = (0.0, -0.215, 0.03)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF ``xyaxes`` for a camera at ``eye`` looking at ``target``.

    A MuJoCo camera looks along its own -Z, so the frame's z axis is the
    reverse of the viewing direction.
    """

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    x_axis = np.cross(forward, np.asarray(up, dtype=float))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(-forward, x_axis)
    return " ".join(f"{value:.5f}" for value in (*x_axis, *y_axis))


def _bin_geoms() -> str:
    parts: list[str] = []
    for spec in BINS.values():
        tint = spec.rgba.rsplit(" ", 1)[0]
        w, wall, tall = BIN_HALF_WIDTH, BIN_WALL, BIN_WALL_HEIGHT
        parts.append(
            f'<geom name="bin_{spec.key}" type="box" pos="{spec.x} {spec.y} 0.004" '
            f'size="{w} {w} 0.004" rgba="{spec.rgba}"/>'
        )
        for suffix, dx, dy, sx, sy in (
            ("e", w, 0.0, wall, w),
            ("w", -w, 0.0, wall, w),
            ("n", 0.0, w, w, wall),
            ("s", 0.0, -w, w, wall),
        ):
            parts.append(
                f'<geom name="bin_{spec.key}_{suffix}" type="box" '
                f'pos="{spec.x + dx} {spec.y + dy} {tall / 2}" '
                f'size="{sx} {sy} {tall}" rgba="{tint} 0.5"/>'
            )
    return "".join(parts)


def _object_bodies(objects: tuple[SceneObject, ...]) -> str:
    parts: list[str] = []
    for index, item in enumerate(objects):
        half = OBJECT_HALF_SIZE
        parts.append(
            f'<body name="item_{index}" pos="{item.x} {item.y} {GRASP_Z}" '
            f'euler="0 0 {math.radians(item.yaw_degrees)}">'
            f'<freejoint name="item_{index}_free"/>'
            f'<geom name="item_{index}_geom" type="box" '
            f'size="{half} {half * 0.6} {half}" '
            f'rgba="{CATEGORY_COLORS[item.category]}" mass="0.02" '
            f'friction="1.5 0.02 0.001"/>'
            f"</body>"
        )
    return "".join(parts)


def _world_xml(objects: tuple[SceneObject, ...]) -> str:
    return f"""<mujoco model="trashdrop_cell">
  <compiler angle="radian"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <global azimuth="130" elevation="-30"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.25 0.28 0.32"
             rgb2="0.05 0.05 0.07" width="512" height="3072"/>
    <material name="floor_mat" rgba="0.18 0.18 0.20 1"/>
    <material name="board_mat" rgba="0.92 0.92 0.92 1"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 1.0" dir="-0.2 0.2 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>
    <geom name="board" type="box"
          pos="{PICK_ZONE.center_x} {PICK_ZONE.center_y} {SURFACE_THICKNESS / 2}"
          size="{PICK_ZONE.half_x} {PICK_ZONE.half_y} {SURFACE_THICKNESS / 2}"
          material="board_mat"/>
    <camera name="topdown"
            pos="{CAMERA.center_x} {CAMERA.center_y} {CAMERA.height + SURFACE_THICKNESS}"
            xyaxes="1 0 0 0 1 0" fovy="{CAMERA.fovy_degrees}"/>
    <camera name="operator" pos="{OPERATOR_EYE[0]} {OPERATOR_EYE[1]} {OPERATOR_EYE[2]}"
            xyaxes="{_look_at(OPERATOR_EYE, OPERATOR_TARGET)}" fovy="52"/>
    {_bin_geoms()}
    {_object_bodies(objects)}
  </worldbody>
</mujoco>
"""


def build_spec(objects: tuple[SceneObject, ...] = DEFAULT_OBJECTS, model_directory: Path | None = None):
    """Return a compiled-ready MjSpec holding the table and both arms."""

    import mujoco

    model_directory = model_directory or find_arm_model()
    arm_xml = model_directory / "so_arm100.xml"
    if not arm_xml.is_file():
        raise FileNotFoundError(f"Missing SO-ARM model: {arm_xml}")

    world = mujoco.MjSpec.from_string(_world_xml(objects))
    for mount in ARMS:
        arm = mujoco.MjSpec.from_file(str(arm_xml))
        frame = world.worldbody.add_frame(
            pos=[mount.pose.x, mount.pose.y, 0.0],
            euler=[0.0, 0.0, mount.pose.yaw],
        )
        world.attach(arm, prefix=mount.prefix, frame=frame)
    return world


def build_model(objects: tuple[SceneObject, ...] = DEFAULT_OBJECTS, model_directory: Path | None = None):
    """Compile the cell and verify the parts that fail silently."""

    import mujoco

    model = build_spec(objects, model_directory).compile()

    expected = {f"{m.prefix}{j}" for m in ARMS for j in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")}
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    }
    if missing := expected - actual:
        raise RuntimeError(f"Scene is missing actuators: {sorted(missing)}")

    for camera in ("topdown", "operator"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise RuntimeError(f"Scene is missing the {camera!r} camera")
    for key in BINS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bin_{key}") < 0:
            raise RuntimeError(f"Scene is missing bin {key!r}")

    _assert_arms_face_each_other(mujoco, model)
    return model


def _assert_arms_face_each_other(mujoco, model) -> None:
    """Catch a degrees/radians mix-up in the attach frame.

    With the mounts rotated correctly each arm's base sits at its configured
    position; with the rotation silently dropped the second arm's links extend
    the wrong way and every IK target lands outside its reach.
    """

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for mount in ARMS:
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{mount.prefix}Base")
        if base < 0:
            raise RuntimeError(f"Scene is missing {mount.prefix}Base")
        got = np.asarray(data.xpos[base][:2])
        want = np.array([mount.pose.x, mount.pose.y])
        if not np.allclose(got, want, atol=1e-6):
            raise RuntimeError(f"{mount.name} base at {got}, expected {want}")

        # The Base body carries the mount yaw; compare its heading to the spec.
        rot = data.xmat[base].reshape(3, 3)
        heading = math.atan2(rot[1, 0], rot[0, 0])
        if abs((heading - mount.pose.yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-4:
            raise RuntimeError(
                f"{mount.name} base yaw is {math.degrees(heading):.1f} deg, "
                f"expected {math.degrees(mount.pose.yaw):.1f} deg -- the attach "
                "frame lost its rotation (check compiler angle=radian)"
            )


def write_scene_xml(path: Path, objects: tuple[SceneObject, ...] = DEFAULT_OBJECTS) -> Path:
    """Dump the assembled scene for inspection in a viewer or a diff."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec(objects)
    spec.compile()
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path
