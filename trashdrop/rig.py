"""Which USB device is which part of the cell.

rig.toml names every device by something that survives moving a cable to
another socket: a servo adapter's serial number, a camera's USB id. Serial
port names and OpenCV indices are looked up from those every time and never
written down, because they change.

Which arm is left and which is right cannot be read off a USB device, so
``trashdrop rig identify`` asks a person to move each joint of each arm by
hand in turn and watches which adapter, and which motor on it, answers. That
also proves every motor carries the ID LeRobot's setup gives its joint. Left
and right are the arms' own: stand behind them, looking where they reach.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .station import repository_root

RIG_FILE = repository_root() / "rig.toml"
OVERHEAD_DEFAULT = "046d:08e5"  # Logitech C920
ARM_NAMES = ("left", "right")
# What is written on each arm, for people; the code never relies on it.
DEFAULT_LABELS = {"left": "F01", "right": "F02"}
# Joint speed of every move the tools make, degrees per second: slow enough
# to reach the power switch before anything is hit.
DEFAULT_MAX_SPEED = 45.0
# The last descent is slower than travel so a calibration error is easier to
# stop before the fixed fingertip reaches the table.
DEFAULT_DESCENT_SPEED = 20.0
# A deliberate push by hand, well above servo read noise: ~13 degrees.
MOVE_TICKS = 150
# The joints whose angles are kept with each touch, in this order.
TOUCH_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass
class ArmDevices:
    bus: str | None = None  # serial number of the arm's servo adapter
    camera: str | None = None  # wrist camera, USB vendor:product
    label: str = ""  # what is written on the arm
    max_speed: float = DEFAULT_MAX_SPEED  # degrees per second
    descent_speed: float = DEFAULT_DESCENT_SPEED  # final approach, degrees per second
    # Where the arm stands relative to the calibration frame, and the table's
    # plane in its own frame: (x cm, y cm, yaw deg, table_z cm, dz/dx, dz/dy),
    # from `trashdrop rig touch`. None until it is touched.
    sheet: tuple[float, ...] | None = None
    # Tape corners this arm touched: name -> (x, y, z) cm in its own frame.
    touches: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # The joint degrees each touch was made with, so that the touches can be
    # worked out again when the kinematics are corrected.
    touch_poses: dict[str, dict[str, float]] = field(default_factory=dict)
    # model wrist roll = LeRobot wrist roll + this, degrees (from `trashdrop rig roll`).
    wrist_roll_offset: float = 0.0


@dataclass(frozen=True)
class PickZone:
    """Where items are put down and looked for: a rectangle on the table, in the
    calibration sheet's frame (cm). By default the sheet's own footprint: lay the
    sheet where the zone should be, calibrate, take it away."""

    x: float = 0.0
    y: float = 0.0
    width: float = 30.0
    height: float = 21.0
    # The taped quadrilateral's real corners, when the zone comes from tape.
    polygon: tuple[tuple[float, float], ...] | None = None

    def corners(self, inset: float = 0.0) -> list[tuple[float, float]]:
        """Far left, far right, near right, near left -- far is away from the arms."""

        if self.polygon:
            return [tuple(point) for point in self.polygon]
        hw, hh = self.width / 2 - inset, self.height / 2 - inset
        return [(self.x - hw, self.y + hh), (self.x + hw, self.y + hh),
                (self.x + hw, self.y - hh), (self.x - hw, self.y - hh)]


@dataclass
class Rig:
    overhead: str = OVERHEAD_DEFAULT
    pick_zone: PickZone = field(default_factory=PickZone)
    # Where the camera sees the tape corners: name -> (u, v) pixels.
    tape_pixels: dict[str, tuple[float, float]] = field(default_factory=dict)
    arms: dict[str, ArmDevices] = field(
        default_factory=lambda: {name: ArmDevices(label=DEFAULT_LABELS[name]) for name in ARM_NAMES}
    )

    def cameras(self) -> dict[str, str]:
        """Role -> USB id for every camera the rig names."""

        roles = {"overhead": self.overhead}
        roles.update({f"{name} wrist": arm.camera for name, arm in self.arms.items() if arm.camera})
        return roles


def load_rig(path: Path = RIG_FILE) -> Rig:
    if not path.is_file():
        return Rig()
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    zone = payload.get("pick_zone", {})
    polygon = tuple(tuple(float(v) for v in point) for point in zone["corners"]) if "corners" in zone else None
    rig = Rig(
        overhead=payload.get("overhead", {}).get("camera", OVERHEAD_DEFAULT),
        pick_zone=PickZone(**{key: float(value) for key, value in zone.items() if key in ("x", "y", "width", "height")},
                           polygon=polygon),
        tape_pixels={name: tuple(float(v) for v in point) for name, point in payload.get("tape", {}).items()},
    )
    for name in ARM_NAMES:
        section = payload.get(name, {})
        descent_speed = float(section.get("descent_speed", DEFAULT_DESCENT_SPEED))
        if descent_speed <= 0:
            raise ValueError(f"{name} descent_speed must be positive")
        rig.arms[name] = ArmDevices(
            bus=section.get("bus"),
            camera=section.get("camera"),
            label=section.get("label", DEFAULT_LABELS[name]),
            max_speed=float(section.get("max_speed", DEFAULT_MAX_SPEED)),
            descent_speed=descent_speed,
            sheet=(
                tuple(float(section["sheet"].get(key, 0.0)) for key in ("x", "y", "yaw", "table_z", "dz_dx", "dz_dy"))
                if "sheet" in section else None
            ),
            touches={name: tuple(float(v) for v in point) for name, point in section.get("touches", {}).items()},
            touch_poses={name: dict(zip(TOUCH_JOINTS, (float(v) for v in angles)))
                         for name, angles in section.get("touch_poses", {}).items()},
            wrist_roll_offset=float(section.get("wrist_roll_offset", 0.0)),
        )
    return rig


def render(rig: Rig) -> str:
    lines = [
        "# Which USB device is which part of the cell. Devices are named by their",
        "# serial number or USB id, never by /dev name or camera index: those change",
        "# when a cable moves to another socket. Written by `trashdrop rig identify`;",
        "# `trashdrop rig check` verifies everything is plugged in and answering.",
        "",
        "[overhead]",
        f'camera = "{rig.overhead}"',
        "",
        "# Where items are put down and looked for, in cm in the calibration sheet's frame",
        "# (x to the right of the printed page, y towards its top). By default the sheet's own",
        "# footprint: lay the sheet where the zone should be, calibrate, take it away.",
        "[pick_zone]",
        f"x = {rig.pick_zone.x:g}",
        f"y = {rig.pick_zone.y:g}",
        f"width = {rig.pick_zone.width:g}",
        f"height = {rig.pick_zone.height:g}",
    ]
    if rig.pick_zone.polygon:
        corners = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in rig.pick_zone.polygon)
        lines.append(f"corners = [{corners}]  # the taped zone: far left, far right, near right, near left")
    if rig.tape_pixels:
        lines += ["", "# Where the overhead camera sees the tape corners, pixels (from `camera tape`).", "[tape]"]
        lines += [f"{name} = [{u:.1f}, {v:.1f}]" for name, (u, v) in rig.tape_pixels.items()]
    for name, arm in rig.arms.items():
        lines += ["", f"[{name}]"]
        lines.append(f'label = "{arm.label}"          # written on the arm')
        lines.append(f'bus = "{arm.bus}"       # servo adapter serial number' if arm.bus else "# bus = unknown")
        lines.append(f'camera = "{arm.camera}"      # wrist camera' if arm.camera else "# camera = unknown")
        lines.append(f"max_speed = {arm.max_speed:g}             # degrees per second, every move")
        lines.append(f"descent_speed = {arm.descent_speed:g}         # degrees per second, final approach to an item")
        if arm.wrist_roll_offset:
            lines.append(f"wrist_roll_offset = {arm.wrist_roll_offset:.1f}  # model roll = LeRobot roll + this "
                         "(from `rig roll`)")
        if arm.touches:
            points = ", ".join(f"{name} = [{x:.2f}, {y:.2f}, {z:.2f}]" for name, (x, y, z) in arm.touches.items())
            lines.append(f"touches = {{ {points} }}  # tape corners touched, cm in the arm's frame")
        if arm.touch_poses:
            poses = ", ".join(
                f"{name} = [{', '.join(f'{pose[joint]:.1f}' for joint in TOUCH_JOINTS)}]"
                for name, pose in arm.touch_poses.items()
            )
            lines.append(f"touch_poses = {{ {poses} }}  # joint degrees of each touch: pan, lift, elbow, "
                         "wrist flex, wrist roll")
        if arm.sheet:
            x, y, yaw, table_z, dz_dx, dz_dy = (tuple(arm.sheet) + (0.0, 0.0))[:6]
            lines.append(
                f"sheet = {{ x = {x:.2f}, y = {y:.2f}, yaw = {yaw:.2f}, table_z = {table_z:.2f}, "
                f"dz_dx = {dz_dx:.4f}, dz_dy = {dz_dy:.4f} }}  # from `rig touch`: cm, deg, table plane"
            )
    return "\n".join(lines) + "\n"


def save_rig(rig: Rig, path: Path = RIG_FILE) -> Path:
    path.write_text(render(rig), encoding="utf-8")
    return path


def overhead_usb_id(path: Path = RIG_FILE) -> str | None:
    """The overhead camera's USB id, if a rig.toml names one."""

    return load_rig(path).overhead if path.is_file() else None


# --- identify: which adapter and which motor is which joint of which arm -------

# What to tell a person to do with each joint, in the order the wizard asks.
JOINT_HINTS = {
    "shoulder_pan": "turn the whole arm left or right at its base",
    "shoulder_lift": "raise or lower the upper arm at the shoulder",
    "elbow_flex": "bend the elbow",
    "wrist_flex": "tilt the wrist up or down",
    "wrist_roll": "twist the wrist",
    "gripper": "open or close the jaw",
}
STILL_TICKS = 15  # below this between polls, a joint counts as still


def first_moved(buses: dict, seconds: float, *, ignore=frozenset(), poll: float = 0.05,
                clock=time.monotonic, sleep=time.sleep) -> tuple[str, str] | None:
    """(bus key, joint) of the first joint pushed MOVE_TICKS from where it started.

    ``buses`` maps a key to anything with ``positions()``. Moving one joint by
    hand nudges its neighbours a little, so of the joints past the threshold
    the one that moved furthest is reported. Joints in ``ignore`` -- (key,
    joint) pairs already identified -- are never reported: a person often
    keeps turning the last joint while reading the next question.
    """

    start = {key: bus.positions() for key, bus in buses.items()}
    deadline = clock() + seconds
    while clock() < deadline:
        moved = []
        for key, bus in buses.items():
            for joint, value in bus.positions().items():
                if (key, joint) in ignore:
                    continue
                shift = abs(value - start[key][joint])
                if shift >= MOVE_TICKS:
                    moved.append((shift, key, joint))
        if moved:
            _, key, joint = max(moved)
            return key, joint
        sleep(poll)
    return None


def wait_until_still(buses: dict, *, calm: float = 1.0, limit: float = 10.0, poll: float = 0.1,
                     clock=time.monotonic, sleep=time.sleep) -> bool:
    """Wait until no joint has moved for ``calm`` seconds, so the next question starts clean."""

    last = {key: bus.positions() for key, bus in buses.items()}
    quiet_since = clock()
    deadline = clock() + limit
    while clock() < deadline:
        sleep(poll)
        now = {key: bus.positions() for key, bus in buses.items()}
        if any(abs(now[key][joint] - last[key][joint]) > STILL_TICKS for key in now for joint in now[key]):
            quiet_since = clock()
        last = now
        if clock() - quiet_since >= calm:
            return True
    return False
