"""Build one MuJoCo MJCF station containing two namespaced SO-ARM models."""

from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from .station import ARM_BINS, ARMS, SAMPLE_ITEMS, ArmMount, bin_for, find_arm_model, repository_root


_REFERENCE_ATTRIBUTES = {
    "body1",
    "body2",
    "childclass",
    "class",
    "joint",
    "material",
    "mesh",
}


def _symbol_map(source: ET.Element, prefix: str) -> dict[str, str]:
    """Map all named model symbols and default classes into one arm namespace."""

    symbols: set[str] = set()
    for element in source.iter():
        if name := element.get("name"):
            symbols.add(name)
        if default_class := element.get("class"):
            symbols.add(default_class)
    return {symbol: f"{prefix}{symbol}" for symbol in symbols}


def _namespaced_copy(source: ET.Element, prefix: str) -> ET.Element:
    """Copy one Menagerie arm and namespace names plus all local references."""

    symbols = _symbol_map(source, prefix)
    clone = copy.deepcopy(source)
    for element in clone.iter():
        for attribute, value in tuple(element.attrib.items()):
            if attribute == "name" or attribute in _REFERENCE_ATTRIBUTES:
                element.set(attribute, symbols.get(value, value))
    return clone


def _append_open_bin(worldbody: ET.Element, arm: str, category: str) -> None:
    """Append a shallow visible collection bin at the configured position."""

    spec = bin_for(arm, category)
    width, wall, height = 0.052, 0.004, 0.035
    r, g, b, _a = spec.rgba.split()
    faint = f"{r} {g} {b} 0.35"
    ET.SubElement(
        worldbody,
        "geom",
        name=f"bin_{arm}_{category}_floor",
        type="box",
        pos=f"{spec.x} {spec.y} 0.004",
        size=f"{width} {width} 0.004",
        rgba=spec.rgba,
    )
    for suffix, dx, dy, sx, sy in (
        ("east", width, 0.0, wall, width),
        ("west", -width, 0.0, wall, width),
        ("north", 0.0, width, width, wall),
        ("south", 0.0, -width, width, wall),
    ):
        ET.SubElement(
            worldbody,
            "geom",
            name=f"bin_{arm}_{category}_{suffix}",
            type="box",
            pos=f"{spec.x + dx} {spec.y + dy} {height / 2}",
            size=f"{sx} {sy} {height}",
            rgba=faint,
        )


def _append_station(worldbody: ET.Element) -> None:
    """Add a shared table, zones, bins, cameras, and coloured demo objects."""

    ET.SubElement(worldbody, "light", name="key", pos="0 -0.1 0.85", directional="true")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="0 0 0.05", rgba="0.06 0.07 0.09 1")
    ET.SubElement(
        worldbody,
        "geom",
        name="workbench",
        type="box",
        pos="0 -0.14 -0.025",
        size="0.46 0.37 0.025",
        rgba="0.82 0.84 0.87 1",
    )
    # Transparent visual markers make arm ownership obvious in a viewer.
    ET.SubElement(
        worldbody,
        "geom",
        name="left_zone",
        type="box",
        pos="-0.18 -0.15 0.003",
        size="0.18 0.13 0.002",
        rgba="0.20 0.45 0.95 0.12",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="right_zone",
        type="box",
        pos="0.18 -0.15 0.003",
        size="0.18 0.13 0.002",
        rgba="0.95 0.35 0.20 0.12",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="handoff_zone",
        type="box",
        pos="0 -0.02 0.004",
        size="0.025 0.06 0.003",
        rgba="0.95 0.85 0.20 0.25",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="topdown",
        pos="0 -0.13 0.82",
        xyaxes="1 0 0 0 1 0",
        fovy="46",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="operator",
        pos="0.72 -0.72 0.5",
        xyaxes="0.72 0.69 0 -0.29 0.30 0.91",
        fovy="55",
    )
    for arm, bins in ARM_BINS.items():
        for category in bins:
            _append_open_bin(worldbody, arm, category)
    for index, (item_id, category, x, y) in enumerate(SAMPLE_ITEMS):
        body = ET.SubElement(worldbody, "body", name=f"item_{item_id}", pos=f"{x} {y} 0.02")
        ET.SubElement(body, "freejoint", name=f"item_{index}_free")
        ET.SubElement(
            body,
            "geom",
            name=f"item_{item_id}_geom",
            type="box",
            size="0.018 0.012 0.018",
            mass="0.02",
            friction="1.5 0.02 0.001",
            rgba=bin_for("left" if x < 0 else "right", category).rgba,
        )


def _append_arm(scene: ET.Element, source: ET.Element, mount: ArmMount) -> None:
    """Merge one copied arm's assets, defaults, kinematics and actuators."""

    arm = _namespaced_copy(source, mount.prefix)
    assets = arm.find("asset")
    defaults = arm.find("default")
    arm_worldbody = arm.find("worldbody")
    actuators = arm.find("actuator")
    contacts = arm.find("contact")
    if any(section is None for section in (assets, defaults, arm_worldbody, actuators, contacts)):
        raise ValueError("SO-ARM model is missing a required MJCF section")

    for child in assets:
        scene.find("asset").append(copy.deepcopy(child))
    for child in defaults:
        scene.find("default").append(copy.deepcopy(child))

    source_body = next(iter(arm_worldbody), None)
    if source_body is None:
        raise ValueError("SO-ARM model has no base body")
    source_body.set("pos", f"{mount.x} {mount.y} 0")
    source_body.set("euler", f"0 0 {math.radians(mount.yaw_degrees)}")
    scene.find("worldbody").append(source_body)

    for actuator in actuators:
        scene.find("actuator").append(copy.deepcopy(actuator))
    for exclusion in contacts:
        scene.find("contact").append(copy.deepcopy(exclusion))


def build_station(
    output_path: Path | None = None,
    model_directory: Path | None = None,
) -> Path:
    """Generate a dual-arm scene and return its absolute output path.

    The source model is read, never modified. Generated MJCF lives under
    ``build/`` so it is safe to discard and rebuild after layout changes.
    """

    root = repository_root()
    model_directory = model_directory or find_arm_model(root)
    source_path = model_directory / "so_arm100.xml"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing SO-ARM model: {source_path}")
    source = ET.parse(source_path).getroot()

    output_path = output_path or root / "build/dual_arm_station.xml"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meshdir = Path(os.path.relpath(model_directory / "assets", output_path.parent)).as_posix()

    scene = ET.Element("mujoco", model="trashdrop_dual_so101_station")
    ET.SubElement(scene, "compiler", angle="radian", meshdir=meshdir)
    ET.SubElement(scene, "option", timestep="0.002", cone="elliptic", impratio="10")
    ET.SubElement(scene, "asset")
    ET.SubElement(scene, "default")
    ET.SubElement(scene, "worldbody")
    ET.SubElement(scene, "actuator")
    ET.SubElement(scene, "contact")

    for mount in ARMS:
        _append_arm(scene, source, mount)
    _append_station(scene.find("worldbody"))

    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def validate_station(scene_path: Path) -> dict[str, int]:
    """Load the generated scene with MuJoCo and verify its core interface."""

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - user-environment guidance
        raise RuntimeError(
            "MuJoCo is optional. Run: uv sync --extra simulation"
        ) from error

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    joints = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
    expected_actuators = {f"{arm.prefix}{joint}" for arm in ARMS for joint in joints}
    available_actuators = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    }
    missing = expected_actuators - available_actuators
    if missing:
        raise RuntimeError(f"Scene is missing actuators: {sorted(missing)}")
    for camera in ("topdown", "operator"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise RuntimeError(f"Scene is missing {camera!r} camera")
    for arm, bins in ARM_BINS.items():
        for category in bins:
            name = f"bin_{arm}_{category}_floor"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
                raise RuntimeError(f"Scene is missing {name!r}")
    return {"joints": model.njnt, "actuators": model.nu, "bodies": model.nbody}
