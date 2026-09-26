"""Command line for the TrashDrop cell.

Everything that needs MuJoCo imports it lazily inside its handler, so the
dataset and planning commands work in an environment without the simulator --
which is what the team members who are only shooting photos will have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --- simulation ------------------------------------------------------------


def _cmd_sim(args: argparse.Namespace) -> int:
    from .simulator import SortingCell

    cell = SortingCell(record=not args.viewer and not args.no_video, viewer=args.viewer)
    report = cell.run()

    if not args.viewer and not args.no_video:
        video = cell.write_video(args.out / "sort_demo.mp4")
        overlay = cell.write_overlay(args.out / "topdown_detection.png")
        if video:
            print(f"wrote {video}")
        if overlay:
            print(f"wrote {overlay}")
    path = report.to_json(args.out / "run_report.json")
    print(f"wrote {path}")
    return 0 if report.success else 1


def _cmd_probe(args: argparse.Namespace) -> int:
    from .probe import check_layout, reachability_map
    from .simulator import SortingCell

    cell = SortingCell(record=False)
    ok = check_layout(cell)
    if args.map:
        reachability_map(cell, z=args.z, keep_vertical=not args.free)
    return 0 if ok else 1


def _cmd_scene(args: argparse.Namespace) -> int:
    from .scene_builder import write_scene_xml

    path = write_scene_xml(args.out / "cell.xml")
    print(f"wrote {path}")
    return 0


def _cmd_plan(_args: argparse.Namespace) -> int:
    from .planning import DetectedItem, TwoArmDispatcher
    from .scene_builder import DEFAULT_OBJECTS

    items = [
        DetectedItem(item_id=o.item_id, category=o.category, x=o.x, y=o.y)
        for o in DEFAULT_OBJECTS
    ]
    dispatcher = TwoArmDispatcher()
    for assignment in dispatcher.order(dispatcher.dispatch(items)):
        flag = " [rerouted: low confidence]" if assignment.rerouted else ""
        zone = " [needs shared zone]" if assignment.requires_shared_zone else ""
        print(
            f"{assignment.item.item_id:10} {assignment.item.category:8} -> "
            f"{assignment.arm:5} -> {assignment.bin_key:8} "
            f"({assignment.bin_x:+.2f}, {assignment.bin_y:+.2f}){flag}{zone}"
        )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api import DEFAULT_PORT, SortingLog, mock_sorter, serve

    log = SortingLog(snapshot=args.out / "intake_log.json")
    sorter = mock_sorter(log) if args.mock else None
    if not args.mock:
        print(
            "note: no cell is wired in yet, so deliveries will sit in 'queued'.\n"
            "      Use --mock to serve plausible results while the arms do not exist."
        )
    serve(host=args.host, port=args.port, log=log, sorter=sorter,
          mode="mock" if args.mock else "live")
    return 0


# --- dataset ---------------------------------------------------------------


def _source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _webcam():
    """The UVC webcam, or None where it cannot be reached (other OS, a phone)."""

    try:
        from .camera import UvcCamera
        from .rig import overhead_usb_id

        return UvcCamera.find(overhead_usb_id())
    except Exception:
        return None


def _resolve_camera(value: str, width: int, height: int) -> str | int:
    """Turn --camera into a stream source; "auto" finds the webcam's index.

    An afternoon of tuning once ran on the laptop's built-in camera because
    the webcam was index 0, not 1. "auto" asks the webcam itself: it zooms for
    a moment, and the stream that zooms with it is the webcam.
    """

    if value != "auto":
        return _source(value)
    camera = _webcam()
    if camera is None:
        print("camera: cannot reach the webcam's controls to identify it; using index 0")
        return 0
    from .camera.identify import find_stream_index
    from .dataset.capture import open_camera

    print("finding the webcam's video stream (it zooms in and out for a moment)...")
    return find_stream_index(camera, lambda index: open_camera(index, width, height))


def _require_same_camera(capture, source, camera, *, refuse: bool) -> None:
    """Check the stream really is the webcam whose settings we change."""

    if camera is None or not isinstance(source, int):
        return  # a phone URL, or no controls to wiggle
    from .camera.identify import stream_follows_camera

    verdict, detail = stream_follows_camera(capture, camera)
    if verdict is False:
        message = (
            f"--camera {source} is NOT the webcam {camera.usb_id}: {detail}.\n"
            "Leave --camera out and the webcam is found automatically."
        )
        if refuse:
            capture.release()
            raise RuntimeError(message)
        print(f"WARNING: {message}")
    elif verdict is None:
        print(f"camera: could not confirm the stream is the webcam ({detail})")


def _cmd_cameras(args: argparse.Namespace) -> int:
    from .dataset.cameras import device_names, list_cameras

    names = device_names()
    if names:
        print("devices the OS knows about:")
        for name in names:
            print(f"  - {name}")
        print()
    found = list_cameras(args.max_index, args.width, args.height, args.out / "cameras")
    if not found:
        print(
            "no camera opened. On macOS the first run only ASKS for permission -- "
            "click Allow, then run this again."
        )
        return 1
    for info in found:
        print(
            f"  index {info.index}: {info.resolution[0]}x{info.resolution[1]} "
            f"({info.backend})  snapshot -> {info.snapshot}"
        )
    camera = _webcam()
    if camera is not None:
        from .camera.identify import find_stream_index
        from .dataset.capture import open_camera

        try:
            index = find_stream_index(
                camera, lambda i: open_camera(i, args.width, args.height), args.max_index, log=lambda *_: None
            )
            print(f"\nthe webcam {camera.usb_id} is index {index}. Commands find it on their own;")
            print(f"pass --camera {index} only to force it.")
            return 0
        except RuntimeError as error:
            print(f"\n{error}")
    print("\nopen the snapshots: the webcam is the one looking down at the table.")
    return 0


def _servo_buses() -> dict[str, str]:
    """Serial number -> port for every adapter with a full SO-101 behind it."""

    from .servo import ServoBus, list_buses

    found = {}
    for serial, port in list_buses():
        try:
            with ServoBus(port) as bus:
                if not bus.missing():
                    found[serial] = port
        except Exception:
            continue  # not a servo adapter, or busy
    return found


def _print_bus(serial: str, port: str) -> bool:
    from .servo import ServoBus

    with ServoBus(port) as bus:
        missing = bus.missing()
        if missing:
            print(f"    NOT ANSWERING: {', '.join(missing)}")
            return False
        for motor in bus.state():
            print(
                f"    {motor.motor} {motor.name:13s} position {motor.position:4d}  "
                f"limits {motor.min_limit:4d}..{motor.max_limit:<4d}  torque {'ON ' if motor.torque else 'off'}  "
                f"{motor.voltage:4.1f} V  {motor.temperature} C"
            )
    return True


def _cmd_rig_check(args: argparse.Namespace) -> int:
    from .camera import UvcCamera
    from .camera.identify import find_stream_index
    from .dataset.capture import open_camera
    from .rig import RIG_FILE, load_rig
    from .servo import list_buses

    rig = load_rig()
    ok = True
    print(f"rig: {RIG_FILE if RIG_FILE.is_file() else 'no rig.toml yet -- run: uv run trashdrop rig identify'}\n")

    adapters = dict(list_buses())
    named = {arm.bus for arm in rig.arms.values() if arm.bus}
    for name, arm in rig.arms.items():
        if not arm.bus:
            print(f"{name} arm: not identified yet")
            ok = False
        elif arm.bus not in adapters:
            print(f"{name} arm: adapter {arm.bus} is NOT plugged in")
            ok = False
        else:
            print(f"{name} arm: adapter {arm.bus} at {adapters[arm.bus]}")
            ok &= _print_bus(arm.bus, adapters[arm.bus])
    for serial, port in _servo_buses().items():
        if serial not in named:
            print(f"arm not in rig.toml: adapter {serial} at {port} -- run: uv run trashdrop rig identify")
            ok = False

    print()
    snapshots = args.out / "rig"
    roles = list(rig.cameras().items())
    if sys.platform == "darwin":
        from .camera.iokit import list_devices

        named_cameras = {usb_id for _, usb_id in roles}
        for vendor, product in list_devices():
            usb_id = f"{vendor:04x}:{product:04x}"
            if usb_id not in named_cameras:
                roles.append(("unassigned", usb_id))
    for role, usb_id in roles:
        try:
            camera = UvcCamera.find(usb_id)
        except Exception as error:
            print(f"{role} camera {usb_id}: NOT FOUND ({error})")
            ok = False
            continue
        if args.no_video:
            print(f"{role} camera {usb_id}: on USB")
            continue
        try:
            from .camera.tune import sharpness

            index = find_stream_index(camera, lambda index: open_camera(index, 640, 480), log=lambda *_: None)
            capture = open_camera(index, 640, 480)
            for _ in range(10):  # let exposure settle
                ok_frame, frame = capture.read()
            capture.release()
            snapshots.mkdir(parents=True, exist_ok=True)
            name = role if role != "unassigned" else f"unassigned_{usb_id}"
            target = snapshots / f"{name.replace(' ', '_').replace(':', '-')}.jpg"
            if ok_frame:
                import cv2

                cv2.imwrite(str(target), frame)
            detail = f"sharpness {sharpness(frame):.0f}  " if ok_frame else ""
            print(f"{role} camera {usb_id}: stream index {index}  {detail}snapshot -> {target}")
        except Exception as error:
            print(f"{role} camera {usb_id}: on USB, but no video: {error}")
            ok = False
    print("\nRIG OK" if ok else "\nRIG NOT READY")
    return 0 if ok else 1


def _cmd_rig_identify(args: argparse.Namespace) -> int:
    """Move each joint of each arm by hand when asked; learn which adapter and motor it is."""

    import time

    from .rig import JOINT_HINTS, RIG_FILE, first_moved, load_rig, save_rig, wait_until_still
    from .servo import MOTORS, ServoBus

    rig = load_rig()
    buses = _servo_buses()
    if len(buses) != 2:
        print(f"expected two arms with six answering servos each, found {len(buses)}: {buses or 'none'}")
        return 1
    opened = {serial: ServoBus(port) for serial, port in buses.items()}
    try:
        stiff = [serial for serial, bus in opened.items() if any(m.torque for m in bus.state())]
        if stiff:
            print("an arm is holding its pose (torque on). It goes limp in 3 s -- hold it if it is in the air")
            time.sleep(3)
            for serial in stiff:
                for motor in MOTORS.values():
                    opened[serial].write(motor, "torque_enable", 0)

        joints = ["gripper"] if args.quick else list(JOINT_HINTS)
        found: dict[tuple[str, str], tuple[str, str]] = {}
        print("Move ONLY the joint asked for, about 20 degrees, then let go. Ctrl+C stops.")
        for side in ("left", "right"):
            label = rig.arms[side].label
            print(f"\n{side.upper()} arm ({label}) -- left and right as seen from behind the arms")
            for joint in joints:
                wait_until_still(opened)
                print(f"  {joint}: {JOINT_HINTS[joint]}...", end="", flush=True)
                # Joints already found are ignored, so finishing the last move is harmless.
                moved = first_moved(opened, args.seconds, ignore=set(found.values()))
                if moved is None:
                    print(" nothing moved; skipped")
                    continue
                serial, moved_joint = moved
                verdict = "ok" if moved_joint == joint else f"WRONG: that was motor {MOTORS[moved_joint]}, {moved_joint}"
                print(f" adapter {serial}, motor {MOTORS[moved_joint]} -- {verdict}. Got it, let go.")
                found[(side, joint)] = (serial, moved_joint)
    except KeyboardInterrupt:
        print("\nstopped; rig.toml not changed")
        return 130
    finally:
        for bus in opened.values():
            bus.close()

    problems = []
    sides = {}
    for side in ("left", "right"):
        serials = {serial for (s, _), (serial, _) in found.items() if s == side}
        if len(serials) != 1:
            problems.append(f"the {side} arm's joints turned up on {len(serials)} adapters: {sorted(serials)}")
        else:
            sides[side] = serials.pop()
    if len(sides) == 2 and sides["left"] == sides["right"]:
        problems.append("both arms' joints turned up on the same adapter -- was the same arm moved twice?")
    wrong = [f"{side} {joint} is motor {MOTORS[got]}" for (side, joint), (_, got) in found.items() if got != joint]
    if wrong:
        problems.append("motor IDs do not follow LeRobot's setup (" + "; ".join(wrong) + ")."
                        " Fix with lerobot-setup-motors before driving the arm")
    if problems:
        print("\nrig.toml NOT changed:\n  - " + "\n  - ".join(problems))
        return 1
    for side, serial in sides.items():
        rig.arms[side].bus = serial
    save_rig(rig)
    print(f"\nall answered as expected. Wrote {RIG_FILE}:")
    for side in ("left", "right"):
        devices = rig.arms[side]
        print(f"  {side:5s} ({devices.label}): adapter {devices.bus}, wrist camera {devices.camera or 'not set'}")
    return 0


def _apply_tape(rig) -> None:
    """Solve the tape calibration with what is known so far; save what it determines."""

    from .rig import PickZone, save_rig
    from .station import repository_root
    from .tape import LABELS, solve

    _touches_from_poses(rig)
    solution = solve({arm: devices.touches for arm, devices in rig.arms.items() if devices.touches},
                     rig.tape_pixels or None)
    for arm, placement in solution.placements.items():
        rig.arms[arm].sheet = (placement.x, placement.y, placement.yaw, placement.table_z,
                               placement.dz_dx, placement.dz_dy)
        misses = ", ".join(f"{LABELS[name]} {value:.1f} cm" for name, value in solution.residuals[arm].items())
        print(f"  {arm} arm fits its touches to within: {misses}")
    if solution.corners:
        rig.pick_zone = PickZone(polygon=tuple(tuple(float(v) for v in solution.corners[name])
                                               for name in ("far_left", "far_right", "near_right", "near_left")))
    if solution.homography is not None:
        solution.homography.save(repository_root() / "camera_sheet.json")
    save_rig(rig)
    if solution.missing:
        print("  still needed: " + "; ".join(solution.missing))
    else:
        print("  calibrated: camera, zone and every touched arm. Next: uv run trashdrop pick --dry-run")


def _touches_from_poses(rig) -> None:
    """Work each touched corner out again from its joint angles, with the arm's current wrist roll offset."""

    from .kinematics import Kinematics

    for devices in rig.arms.values():
        if devices.touch_poses:
            kinematics = Kinematics(devices.wrist_roll_offset)
            for name, pose in devices.touch_poses.items():
                devices.touches[name] = tuple(float(v) for v in kinematics.fingertip(pose) * 100)


def _cmd_rig_touch(args: argparse.Namespace) -> int:
    """Touch the sheet's markers with the fixed fingertip; learn where the arm stands."""

    import time

    import numpy as np

    from .camera.markers import MARKER_SHEET_CM
    from .kinematics import Kinematics
    from .placement import Placement, fit_placement, fit_table
    from .rig import TOUCH_JOINTS, load_rig, save_rig

    rig = load_rig()
    arm = _open_arm(args.arm)
    kinematics = Kinematics(rig.arms[arm.name].wrist_roll_offset)
    if args.tape:
        from .tape import CORNERS, LABELS, ON_CAMERA

        names = tuple(f"{LABELS[key]} ({ON_CAMERA[key]} in the camera picture)" for key in CORNERS)
        keys = dict(zip(names, CORNERS))
        points = {}
        what = "INNER corner of the taped zone"
    else:
        names = ("marker 0 (top left of the page)", "marker 1 (top right)", "marker 2 (bottom right)",
                 "marker 3 (bottom left)")
        points = dict(zip(names, (MARKER_SHEET_CM[i] for i in (0, 1, 2, 3))))
        what = "marker's centre"
    touched = []
    try:
        if arm.torque_is_on():
            print(f"{arm.name}: going limp in 3 s -- hold it")
            time.sleep(3)
            arm.torque_off()
        print(
            f"For each point: put the tip of the FIXED finger (the one that does not move when the\n"
            f"gripper opens) on the {what}, hold the arm there and press Enter.\n"
            "s skips a point the arm cannot reach (three are enough), Ctrl+C stops."
        )
        for name in names:
            if input(f"  {name}: ").strip().lower() == "s":
                continue
            pose = arm.pose()
            tip = kinematics.fingertip(pose) * 100
            touched.append((name, tip, {joint: pose[joint] for joint in TOUCH_JOINTS}))
            print(f"      fingertip at x {tip[0]:.1f}  y {tip[1]:.1f}  z {tip[2]:.1f} cm")
    except KeyboardInterrupt:
        print("\nstopped; rig.toml not changed")
        return 130
    finally:
        arm.bus.close()
    if len(touched) < 3:
        print("at least three points are needed; rig.toml not changed")
        return 1
    if args.tape:
        rig.arms[arm.name].touches = {keys[name]: tuple(float(v) for v in tip) for name, tip, _ in touched}
        rig.arms[arm.name].touch_poses = {keys[name]: pose for name, _, pose in touched}
        print(f"{arm.name}: {len(touched)} corners recorded")
        _apply_tape(rig)
        return 0

    placement, residuals = fit_placement([points[name] for name, _, _ in touched], [tip[:2] for _, tip, _ in touched])
    table_z, dz_dx, dz_dy = fit_table([tip for _, tip, _ in touched])
    placement = Placement(placement.x, placement.y, placement.yaw, table_z, dz_dx, dz_dy)
    print()
    for (name, _, _), residual in zip(touched, residuals):
        print(f"  {name}: the fit misses it by {residual:.1f} cm")
    if max(residuals) > 1.5:
        print("  more than 1.5 cm off: a fingertip was not on its point, the zone's size in rig.toml is wrong,\n"
              "  or a joint's calibration is off. Touch again.")
    rig.arms[arm.name].sheet = (placement.x, placement.y, placement.yaw, placement.table_z,
                                placement.dz_dx, placement.dz_dy)
    save_rig(rig)
    print(
        f"{arm.name}: the sheet's centre is {placement.x:.1f} cm forward, {placement.y:.1f} cm left of the arm, "
        f"turned {placement.yaw:.0f} deg. Saved to rig.toml."
    )
    return 0


def _cmd_rig_roll(args: argparse.Namespace) -> int:
    """Find where this arm's wrist roll really has its zero (see trashdrop/wrist.py)."""

    from .arm import load_poses
    from .kinematics import Kinematics
    from .placement import Placement
    from .rig import load_rig, save_rig
    from .wrist import HOVER_CM, KEYS, NUDGE_DEG, hover_pose, settled_offset, turned_by, wrap

    rig = load_rig()
    arm = _open_arm(args.arm)
    devices = rig.arms[arm.name]
    if not devices.sheet:
        arm.bus.close()
        print(f"{arm.name}: the arm has not touched the tape yet: uv run trashdrop rig touch {arm.name} --tape")
        return 1
    placement = Placement(*devices.sheet)
    neutral = load_poses().get(arm.name, {}).get("neutral")
    before = offset = devices.wrist_roll_offset
    try:
        input(f"{arm.name}: it will hold its fingers down {HOVER_CM:g} cm over the middle of the pick zone and open "
              "the jaw.\nClear the zone, keep a hand near Ctrl+C and press Enter...")
        if not arm.torque_is_on():
            arm.torque_on()
        if neutral:
            arm.move(neutral)
        while True:
            kinematics = Kinematics(offset)
            pose = hover_pose(kinematics, placement, arm.limits_degrees())
            if pose is None:
                print(f"{arm.name}: the middle of the pick zone is out of its reach with this wrist; nothing saved")
                return 1
            arm.move(pose)
            arm.move({"gripper": 50.0})
            answer = input(
                "Which tape edge did the MOVING jaw (the one that just opened) go towards?\n"
                "  f = far (top of the camera picture), n = near (the arms), l = left, r = right: "
            ).strip().lower()[:1]
            if answer not in KEYS:
                continue
            delta = turned_by(kinematics, pose, placement, KEYS[answer])
            if delta == 0.0:
                break
            offset = wrap(offset + delta)
            print(f"  the wrist is turned {delta:+.0f} deg from what the model thought; trying again with that")
        roll = pose["wrist_roll"]
        while True:
            answer = input(
                "Look from above: the fixed and the moving fingertip should line up straight towards the far edge,\n"
                f"  square to the near tape. Enter = they do; a / d = turn the wrist {NUDGE_DEG:g} deg one way / the other: "
            ).strip().lower()[:1]
            if answer == "":
                break
            if answer in ("a", "d"):
                roll += NUDGE_DEG if answer == "a" else -NUDGE_DEG
                arm.move({"wrist_roll": roll})
        offset = round(settled_offset(kinematics, pose, placement, roll - pose["wrist_roll"]), 1)
        if neutral:
            arm.move(neutral)
    except KeyboardInterrupt:
        print("\nstopped; the arm holds where it is; rig.toml not changed")
        return 130
    finally:
        arm.bus.close()

    devices.wrist_roll_offset = offset
    print(f"{arm.name}: model wrist roll = LeRobot roll {offset:+.1f} deg (was {before:+.1f}). Saved to rig.toml.")
    if devices.touches and all(name in devices.touch_poses for name in devices.touches):
        _apply_tape(rig)  # the touches worked out again with the new offset
        return 0
    save_rig(rig)
    if devices.touches and abs(wrap(offset - before)) >= 5.0:
        # Worked out with the old roll: the fingertip is off the roll axis.
        print(f"  its tape corners were touched with the old wrist roll and are now off by up to 1.6 cm.\n"
              f"  Touch them again: uv run trashdrop rig touch {arm.name} --tape")
    return 0


def _click_corners(frame, labels: tuple[str, ...]):
    """Let a person click points on a frame, in order. Returns full-resolution points, or None."""

    import cv2
    import numpy as np

    scale = min(1.0, 1400 / frame.shape[1])
    shown = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points.append((x / scale, y / scale))

    window = "trashdrop: click the corners"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            view = shown.copy()
            for index, (u, v) in enumerate(points):
                cv2.circle(view, (int(u * scale), int(v * scale)), 8, (0, 0, 255), -1)
                cv2.putText(view, labels[index], (int(u * scale) + 10, int(v * scale) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(points) < len(labels):
                prompt = f"click the {labels[len(points)]}   (r = start over, Esc = cancel)"
            else:
                prompt = "Enter = save   r = start over   Esc = cancel"
            cv2.rectangle(view, (0, 0), (view.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(view, prompt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == len(labels):
                return np.array(points, dtype=np.float32)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def _cmd_camera_tape(args: argparse.Namespace) -> int:
    """Click the taped zone's inner corners in the overhead camera's picture."""

    import cv2
    import numpy as np

    from .dataset.capture import open_camera
    from .rig import load_rig

    rig = load_rig()
    capture = open_camera(_resolve_camera(args.camera, 1920, 1080), 1920, 1080)
    try:
        for _ in range(30):  # let exposure settle
            ok, frame = capture.read()
    finally:
        capture.release()
    from .tape import CORNERS, ON_CAMERA

    names = CORNERS
    labels = tuple(f"{ON_CAMERA[key]} inner corner" for key in CORNERS)
    print("a window opens: click the taped zone's INNER corners -- top left, top right, bottom right,\n"
          "bottom left, as the picture shows them")
    corners = _click_corners(frame, labels)
    if corners is None:
        print("cancelled; nothing saved")
        return 1
    rig.tape_pixels = {name: (float(u), float(v)) for name, (u, v) in zip(names, corners)}
    view = frame.copy()
    cv2.polylines(view, [np.rint(corners).astype(np.int32)], True, (60, 220, 60), 3)
    for (u, v), label in zip(corners, labels):
        cv2.circle(view, (int(u), int(v)), 10, (0, 0, 255), -1)
        cv2.putText(view, label, (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "camera_tape.jpg"), view)
    print("corners saved (out/camera_tape.jpg shows them)")
    _apply_tape(rig)
    return 0


def _cmd_camera_sheet(args: argparse.Namespace) -> int:
    """Where the marker sheet is in the overhead camera: pixels -> sheet cm."""

    import cv2
    import numpy as np

    from .camera.markers import MARKER_SHEET_CM
    from .dataset.capture import open_camera
    from .perception.calibration import HomographyCalibration, detect_aruco_corners
    from .station import repository_root

    source = _resolve_camera(args.camera, args.width, args.height)
    capture = open_camera(source, args.width, args.height)
    try:
        for _ in range(30):  # let exposure settle
            ok, frame = capture.read()
    finally:
        capture.release()
    ids = [0, 1, 2, 3]
    points = detect_aruco_corners(frame, ids)
    if points is None:
        print("not all four markers are visible to the overhead camera; nothing saved")
        return 1
    sheet = [np.array(MARKER_SHEET_CM[marker_id]) / 100.0 for marker_id in ids]
    calibration = HomographyCalibration(points, sheet)
    path = calibration.save(repository_root() / "camera_sheet.json")
    for (u, v), marker_id in zip(points, ids):
        cv2.circle(frame, (int(u), int(v)), 12, (0, 0, 255), 3)
        cv2.putText(frame, str(marker_id), (int(u) + 15, int(v) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    snapshot = args.out / "camera_sheet.jpg"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(snapshot), frame)
    residuals = ", ".join(f"{value:.1f}" for value in calibration.residuals_mm())
    print(f"found all four markers; pixels -> sheet cm saved to {path} (residuals {residuals} mm)")
    print(f"snapshot with the markers circled: {snapshot}")
    return 0


def _cmd_pick(args: argparse.Namespace) -> int:
    """Camera finds an item, an arm takes it and drops it on its own side."""

    import cv2
    import numpy as np

    from .arm import connect, load_poses, move_together, resolve_arm
    from .dataset.capture import open_camera
    from .kinematics import Kinematics
    from .perception.calibration import HomographyCalibration
    from .placement import Placement
    from .rig import load_rig
    from .sorter import (
        FINGERTIPS_CM,
        MIN_CONFIDENCE,
        PICK_TRIES,
        SIDE_OF,
        draw_detection,
        draw_plan,
        draw_zone,
        execute_pick,
        find_item,
        item_crop,
        plan_pick,
        reach_mask,
        refine_item,
        side_for,
        zone_mask,
    )
    from .station import FLAT_PINCH_WIDTH, repository_root

    rig = load_rig()
    placements = {name: Placement(*devices.sheet) for name, devices in rig.arms.items() if devices.sheet}
    if args.arm:
        only = resolve_arm(args.arm, rig)
        placements = {name: placement for name, placement in placements.items() if name == only}
    if not placements:
        print("no arm has touched the tape yet: run `uv run trashdrop rig touch left --tape` first")
        return 1
    for name in placements:
        devices = rig.arms[name]
        # Touches without joint angles were worked out with no offset; the
        # fingertip is 8 mm off the roll axis, so they are off by this much.
        stale_cm = 2 * 0.8 * abs(np.sin(np.radians(devices.wrist_roll_offset) / 2))
        if stale_cm > 0.3 and any(corner not in devices.touch_poses for corner in devices.touches):
            print(f"WARNING: the {name} arm touched its corners before its wrist zero was measured, so its\n"
                  f"  grasps land up to {stale_cm:.1f} cm off. Touch them again: "
                  f"uv run trashdrop rig touch {name} --tape")
    sheet_file = repository_root() / "camera_sheet.json"
    if not sheet_file.is_file():
        print("the camera has not found the sheet yet: run `uv run trashdrop camera sheet` first")
        return 1
    homography = HomographyCalibration.load(sheet_file)
    poses = load_poses()
    kinematics = {name: Kinematics(rig.arms[name].wrist_roll_offset) for name in placements}
    classifier = None
    if not args.any_arm and not args.material:
        from .perception.classifier import ClipMaterialClassifier

        try:
            classifier = ClipMaterialClassifier()
        except (FileNotFoundError, RuntimeError) as error:
            print(f"no material classifier ({error}).\n  Every item goes to the nearer arm; --any-arm hides this.")
    arms = {name: connect(name, rig) for name in placements}
    capture = None
    try:
        limits = {name: arm.limits_degrees() for name, arm in arms.items()}
        capture = open_camera(_resolve_camera(args.camera, 1920, 1080), 1920, 1080)

        def grab():
            frame = None
            for _ in range(8):  # drop what was buffered while the arm moved
                ok, latest = capture.read()
                frame = latest if ok else frame
            return frame

        input(f"{', '.join(arms)} go to neutral (straight up), together. Keep clear and press Enter...")
        for arm in arms.values():
            if not arm.torque_is_on():
                arm.torque_on()
        move_together([(arm, poses[name]["neutral"]) for name, arm in arms.items()])
        args.out.mkdir(parents=True, exist_ok=True)

        def photograph_empty_table():
            input("Clear the pick zone (inside the tape square; the rest of the table can stay as it is),\n"
                  "then press Enter to photograph it empty...")
            empty = grab()
            small_shape = (round(empty.shape[0] * 320 / empty.shape[1]), 320)
            scale = empty.shape[1] / 320
            zone = zone_mask(small_shape, scale, homography, rig.pick_zone)
            searched = zone & reach_mask(small_shape, scale, homography, placements)
            cv2.imwrite(str(args.out / "pick_background.jpg"), empty)
            cv2.imwrite(str(args.out / "pick_zone.jpg"), draw_zone(empty, zone, searched))
            unreachable = 1.0 - (searched > 0).sum() / max((zone > 0).sum(), 1)
            if unreachable > 0.05:
                print(f"  {unreachable:.0%} of the pick zone is out of every calibrated arm's reach "
                      "(shaded in out/pick_zone.jpg); items there will be refused")
            if len(placements) > 1:
                shared = zone & reach_mask(small_shape, scale, homography, placements, require_all=True)
                one_arm_only = 1.0 - (shared > 0).sum() / max((zone > 0).sum(), 1)
                if one_arm_only > 0:
                    print(f"  {one_arm_only:.0%} of the pick zone is outside at least one arm's approximate reach; "
                          "an item assigned to that arm may be refused")
            return empty, searched

        background, valid = photograph_empty_table()
        print("ready. The camera looks only inside the pick zone (green in out/pick_zone.jpg).")

        while True:
            answer = input("\nPut an item within reach, then Enter (b = photograph the empty table again, q = quit): ")
            if answer.strip().lower() == "q":
                break
            if answer.strip().lower() == "b":
                background, valid = photograph_empty_table()
                continue
            frame = grab()
            detection = find_item(frame, background, valid)
            cv2.imwrite(str(args.out / "pick_frame.jpg"), frame)
            cv2.imwrite(str(args.out / "pick_seen.jpg"), draw_detection(frame, detection, valid))
            if detection.item is None:
                print(f"  {detection.reason}  (out/pick_seen.jpg shows what changed)")
                continue
            item, item_scale = refine_item(frame, background, detection, valid)
            side = None
            if args.material:
                side = SIDE_OF[args.material]
                print(f"  told it is {args.material}: it goes {side}")
            elif classifier is not None:
                crop = item_crop(frame, item, item_scale)
                cv2.imwrite(str(args.out / "pick_crop.jpg"), crop)
                probabilities = classifier.probabilities(crop)
                side, sure = side_for(probabilities, MIN_CONFIDENCE if args.min_confidence is None else args.min_confidence)
                ranked = ", ".join(f"{name} {p:.2f}" for name, p in sorted(probabilities.items(), key=lambda kv: -kv[1]))
                if side is None:
                    print(f"  not sure what it is ({ranked}): leave it for a person (out/pick_crop.jpg)")
                    continue
                print(f"  it goes {side} ({ranked})")
            candidates = placements if side is None else {name: p for name, p in placements.items() if name == side}
            if not candidates:
                print(f"  it goes {side}, but the {side} arm is not in use: leave it, or run without --arm")
                continue
            plan, reason = plan_pick(item, item_scale, homography, candidates, kinematics, limits,
                                     fingertips_cm=FINGERTIPS_CM if args.fingertips is None else args.fingertips)
            cv2.imwrite(str(args.out / "pick_plan.jpg"), draw_plan(frame, item, item_scale, plan))
            if plan is None:
                if side is not None:
                    if "wider than" in reason:
                        print(f"  the {side} arm cannot take it ({reason}): reorient it for a narrower grip "
                              "or leave it for a person")
                    else:
                        print(f"  the {side} arm cannot take it ({reason}): move it towards the {side} arm")
                else:
                    print(f"  cannot pick it: {reason}")
                continue
            others = ", ".join(f"the {name} {cm:.0f} cm" for name, cm in plan.distances_cm.items() if name != plan.arm)
            print(
                f"  the {plan.arm} arm takes it ({plan.distances_cm[plan.arm]:.0f} cm from its base"
                f"{'; ' + others if others else ''}): {plan.grasp_plan.width_m * 100:.1f} cm across, jaws open "
                f"{plan.open_percent:.0f} %, fixed finger to ({plan.target_cm[0]:.1f}, {plan.target_cm[1]:.1f}) cm "
                f"in its frame, fingers leaning {plan.lean_deg:.0f} deg (red dot in out/pick_plan.jpg)"
            )
            if reason:
                print(f"  (the nearer arm could not: {reason})")
            if plan.grasp_plan.width_m > FLAT_PINCH_WIDTH:
                print(f"  careful: wider than {FLAT_PINCH_WIDTH * 100:.1f} cm. Opened that far the moving jaw rides "
                      "high and comes down\n  on top of anything low (a pack lying flat): expect a miss unless "
                      "the item is tall.")
            if input("  Enter = go, s = skip: ").strip().lower() == "s":
                continue
            name = plan.arm
            execute_pick(arms[name], plan, poses[name]["neutral"], kinematics=kinematics[name], dry_run=args.dry_run,
                         descent_speed=rig.arms[name].descent_speed)
            # Caught or not, the overhead camera says: the arm is back in neutral,
            # as it was when the empty zone was photographed.
            tries = 1
            while not args.dry_run:
                after_frame = grab()
                after = find_item(after_frame, background, valid)
                cv2.imwrite(str(args.out / "pick_after.jpg"), draw_detection(after_frame, after, valid))
                if after.code == "nothing_changed":
                    print("  done: the zone is empty again")
                    break
                if after.item is None or tries == PICK_TRIES:
                    print(f"  it is still in the zone after {tries} {'try' if tries == 1 else 'tries'}"
                          f"{'' if after.item is not None else ' (' + after.reason + ')'}: "
                          "leave it for a person (out/pick_after.jpg)")
                    break
                item, item_scale = refine_item(after_frame, background, after, valid)
                plan, reason = plan_pick(item, item_scale, homography, {name: placements[name]}, kinematics, limits,
                                         fingertips_cm=FINGERTIPS_CM if args.fingertips is None else args.fingertips)
                if plan is None:
                    print(f"  it is still in the zone, and the {name} arm cannot take it now: {reason}")
                    break
                tries += 1
                print(f"  it is still in the zone -- it slipped out, or was never caught. Try {tries} of {PICK_TRIES}")
                execute_pick(arms[name], plan, poses[name]["neutral"], kinematics=kinematics[name],
                             descent_speed=rig.arms[name].descent_speed)
    except KeyboardInterrupt:
        print("\nstopped; the arms hold where they are")
        return 130
    finally:
        if capture is not None:
            capture.release()
        for arm in arms.values():
            arm.bus.close()
        if classifier is not None:
            classifier.close()
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    """The cell's page: the overhead stream with what it sees drawn over it, and its controls."""

    import webbrowser

    from .cell import Cell
    from .station import repository_root
    from .web.server import make_server

    if args.demo:
        out = repository_root() / "out"
        cell = Cell.demo(empty=out / "pick_background.jpg", item=out / "pick_frame.jpg")
    else:
        cell = Cell.open(camera=args.camera, only_arm=args.arm)
    try:
        server = make_server(cell, args.host, args.port)
    except OSError as error:
        cell.close()
        print(f"cannot listen on {args.host}:{args.port} ({error}); try --port 8001")
        return 1
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{server.server_address[1]}"
    try:
        cell.start()
        print(f"open the page in a browser: {url}\n  Ctrl+C here stops it; the arms hold where they are.")
        if args.host != "127.0.0.1":
            print("  listening beyond this machine: anyone who can open the page can move the arms")
        if args.open:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cell.stop()
        server.server_close()
        cell.close()
    return 0


def _open_arm(name: str):
    from .arm import connect

    return connect(name)


def _cmd_arm_status(args: argparse.Namespace) -> int:
    from .arm import connect
    from .rig import load_rig

    rig = load_rig()
    names = [args.arm] if args.arm else [name for name, devices in rig.arms.items() if devices.bus]
    for name in names:
        arm = connect(name, rig)
        try:
            state = {motor.name: motor for motor in arm.bus.state()}
            devices = rig.arms[arm.name]
            print(f"{arm.name} arm ({devices.label}), adapter {devices.bus}, max {devices.max_speed:g} deg/s")
            for joint, value in arm.pose().items():
                motor = state[joint]
                unit = "% open" if joint == "gripper" else "deg"
                print(
                    f"    {joint:13s} {value:7.1f} {unit:6s} tick {motor.position:4d} "
                    f"(limits {motor.min_limit}..{motor.max_limit})  torque {'ON ' if motor.torque else 'off'}  "
                    f"{motor.voltage:4.1f} V  {motor.temperature} C"
                )
        finally:
            arm.bus.close()
    return 0


def _move_and_report(arm, targets: dict[str, float], speed: float | None) -> int:
    if not arm.torque_is_on():
        print(f"{arm.name}: torque on, holding where it is")
        arm.torque_on()
    clamped = {joint: arm.from_ticks(joint, arm.to_ticks(joint, value)) for joint, value in targets.items()}
    plan = ", ".join(f"{joint} -> {value:.1f}" for joint, value in clamped.items())
    print(f"{arm.name}: moving {plan} at up to {min(speed or arm.max_speed, arm.max_speed):g} deg/s. Ctrl+C stops it.")
    for joint in targets:
        if abs(clamped[joint] - targets[joint]) > 0.5:
            print(f"    {joint}: {targets[joint]:.1f} is past the joint's limit; going to {clamped[joint]:.1f}")
    try:
        reached = arm.move(targets, speed=speed)
    except KeyboardInterrupt:
        print("\nstopped; holding where it is")
        return 130
    report = ", ".join(f"{joint} {reached[joint]:.1f}" for joint in targets)
    print(f"{arm.name}: reached {report}. Holding; `uv run trashdrop arm relax {arm.name}` lets it go limp.")
    return 0


def _cmd_arm_go(args: argparse.Namespace) -> int:
    from .arm import load_poses

    arm = _open_arm(args.arm)
    try:
        poses = load_poses().get(arm.name, {})
        if args.pose not in poses:
            print(f"no pose {args.pose!r} for the {arm.name} arm in poses.toml; it has: {', '.join(poses) or 'none'}")
            return 1
        return _move_and_report(arm, poses[args.pose], args.speed)
    finally:
        arm.bus.close()


def _cmd_arm_jog(args: argparse.Namespace) -> int:
    arm = _open_arm(args.arm)
    try:
        current = arm.pose()
        if args.joint not in current:
            print(f"no joint {args.joint!r}; joints: {', '.join(current)}")
            return 1
        return _move_and_report(arm, {args.joint: current[args.joint] + args.degrees}, args.speed)
    finally:
        arm.bus.close()


def _cmd_arm_gripper(args: argparse.Namespace) -> int:
    value = {"open": 100.0, "close": 0.0}.get(args.value)
    if value is None:
        value = float(args.value)
    arm = _open_arm(args.arm)
    try:
        return _move_and_report(arm, {"gripper": value}, args.speed)
    finally:
        arm.bus.close()


def _cmd_arm_test(args: argparse.Namespace) -> int:
    from .arm import nudge_joints, save_pose

    degrees = min(args.degrees, 15.0)
    arm = _open_arm(args.arm)
    holding = False
    try:
        print(
            f"{arm.name}: small test moves, one joint at a time: +-{degrees:g} deg (gripper opens 15 %),\n"
            f"at {args.speed:g} deg/s, back to where it started after each. Ctrl+C stops the arm; it keeps holding."
        )
        holding = arm.torque_is_on()
        if not holding:
            input("Hold the arm clear of the table (or leave it where every joint can move freely), "
                  "then press Enter: torque goes on...")
            arm.torque_on()
            holding = True
            input("It holds itself now. Let go, keep a hand near the power switch, press Enter...")
        save_pose(arm.name, "before_test", arm.pose())
        moved = nudge_joints(arm, degrees, args.speed)
    except KeyboardInterrupt:
        print("\nstopped" + (f"; holding where it is. Back to the start: uv run trashdrop arm go {arm.name} before_test"
                              if holding else "; torque was never switched on"))
        return 130
    finally:
        arm.bus.close()
    print(
        f"done: {', '.join(moved) or 'nothing'} moved. Holding where it started; "
        f"`uv run trashdrop arm relax {arm.name}` lets it go limp."
    )
    return 0


def _cmd_arm_pick(args: argparse.Namespace) -> int:
    from .arm import load_poses, pick_and_drop

    arm = _open_arm(args.arm)
    try:
        poses = load_poses().get(arm.name, {})
        if not arm.torque_is_on():
            print(f"{arm.name}: torque on, holding where it is")
            arm.torque_on()
        carried = pick_and_drop(arm, poses, speed=args.speed, check=not args.no_check)
    except ValueError as error:
        print(f"{error}\nTeach them: `uv run trashdrop arm relax {args.arm}`, pose the arm by hand, then "
              f"`uv run trashdrop arm save {args.arm} above` (and grab, drop; neutral you have).")
        return 1
    except KeyboardInterrupt:
        print("\nstopped; holding where it is")
        return 130
    finally:
        arm.bus.close()
    return 0 if carried else 2


def _cmd_arm_where(args: argparse.Namespace) -> int:
    from .kinematics import Kinematics

    from .rig import load_rig

    arm = _open_arm(args.arm)
    try:
        pose = arm.pose()
    finally:
        arm.bus.close()
    kinematics = Kinematics(load_rig().arms[arm.name].wrist_roll_offset)
    x, y, z = kinematics.tcp(pose) * 100
    fingers, _ = kinematics.pointing(pose)
    print(f"{arm.name}: TCP at x {x:.1f}  y {y:.1f}  z {z:.1f} cm   (the arm's own frame: x forward, y left, z up)")
    down = "  -- straight down" if fingers[2] < -0.98 else ""
    print(f"    fingers point ({fingers[0]:+.2f}, {fingers[1]:+.2f}, {fingers[2]:+.2f}){down}")
    return 0


def _cmd_arm_reach(args: argparse.Namespace) -> int:
    import numpy as np

    from .kinematics import Kinematics

    if args.z < 3.0 and not args.low:
        print("below 3 cm the fingertips are within 2 cm of the table; add --low if that is meant")
        return 1
    from .rig import load_rig

    arm = _open_arm(args.arm)
    try:
        kinematics = Kinematics(load_rig().arms[arm.name].wrist_roll_offset)
        solution = kinematics.solve(
            np.array([args.x, args.y, args.z]) / 100, yaw_deg=args.yaw, start=arm.pose(), limits=arm.limits_degrees()
        )
        if not solution.reachable:
            print(
                f"{arm.name}: cannot put the TCP at ({args.x:g}, {args.y:g}, {args.z:g}) cm with the fingers straight down; "
                f"the closest is {solution.position_error_m * 100:.1f} cm off. Fingers-down reach is about 10-28 cm "
                "in front of the base, below about 8 cm."
            )
            return 1
        print(f"{arm.name}: joints for that point: " + ", ".join(f"{j} {v:.1f}" for j, v in solution.degrees.items()))
        return _move_and_report(arm, solution.degrees, args.speed)
    finally:
        arm.bus.close()


def _cmd_arm_save(args: argparse.Namespace) -> int:
    from .arm import POSES_FILE, save_pose

    arm = _open_arm(args.arm)
    try:
        values = arm.pose()
    finally:
        arm.bus.close()
    save_pose(arm.name, args.pose, values)
    print(f"saved {arm.name}.{args.pose} to {POSES_FILE}: " + ", ".join(f"{j} {v:.1f}" for j, v in values.items()))
    return 0


def _cmd_arm_hold(args: argparse.Namespace) -> int:
    arm = _open_arm(args.arm)
    try:
        arm.torque_on()
    finally:
        arm.bus.close()
    print(f"{arm.name}: torque on, holding where it is")
    return 0


def _cmd_arm_relax(args: argparse.Namespace) -> int:
    import time

    arm = _open_arm(args.arm)
    try:
        if not args.now:
            print(f"{arm.name}: going limp in 3 s -- if it is up in the air, hold it now")
            time.sleep(3)
        arm.torque_off()
    finally:
        arm.bus.close()
    print(f"{arm.name}: limp")
    return 0


def _cmd_camera_sharpness(args: argparse.Namespace) -> int:
    """A live sharpness number while a lens is turned by hand."""

    import time

    from .camera import UvcCamera
    from .camera.identify import find_stream_index
    from .camera.tune import sharpness
    from .dataset.capture import open_camera
    from .rig import overhead_usb_id

    camera = UvcCamera.find(args.usb_id or overhead_usb_id())
    print(f"finding the video of {camera.usb_id} (its picture flickers for a moment)...")
    index = find_stream_index(camera, lambda index: open_camera(index, 640, 480), log=lambda *_: None)
    capture = open_camera(index, 640, 480)
    print("point it at the marker sheet's star from the distance it will work at, then turn the lens")
    print("slowly. The number peaks when the picture is sharpest. Ctrl+C to stop.\n")
    best = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            value = sharpness(frame)
            best = max(best, value)
            bar = "#" * int(40 * value / best) if best else ""
            print(f"\r  sharpness {value:7.0f}   best so far {best:7.0f}   {bar:<40s}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        capture.release()
    return 0


def _cmd_camera_show(args: argparse.Namespace) -> int:
    from .camera import UvcCamera
    from .rig import overhead_usb_id

    camera = UvcCamera.find(args.usb_id or overhead_usb_id())
    print(f"{camera.describe()}\n")
    print(f"  {'control':24s} {'now':>7} {'min':>7} {'max':>7} {'step':>5} {'default':>8}")
    for name, known in camera.ranges().items():
        print(
            f"  {name:24s} {camera.get(name):>7} {known.minimum:>7} {known.maximum:>7} "
            f"{known.step:>5} {known.default:>8}"
        )
    return 0


def _cmd_camera_apply(args: argparse.Namespace) -> int:
    from .camera import UvcCamera, apply, config_path, format_applied, load

    path = args.config or config_path()
    settings = load(path)
    camera = UvcCamera.find(args.usb_id or (settings.device if ":" in settings.device else None))
    results = apply(camera, settings)
    print(f"applied {path} to {camera.describe()}\n{format_applied(results)}")
    return 0 if all(result.ok for result in results) else 1


def _cmd_camera_tune(args: argparse.Namespace) -> int:
    import time

    from .camera import CameraSettings, UvcCamera, config_path, focus_chart, save, tune
    from .dataset.capture import open_camera
    from .dataset.zone import DEFAULT_CALIBRATION, DEFAULT_ZONE_CONFIG, calibrate_zone, save_zone, zone_size

    from .rig import overhead_usb_id

    # Talk to the camera first: if control is refused, say so before any
    # video is opened.
    camera = UvcCamera.find(args.usb_id or overhead_usb_id())
    ranges = camera.ranges()
    print(f"{camera.describe()}: {len(ranges)} adjustable controls")

    source = _resolve_camera(args.camera, args.width, args.height)
    capture = open_camera(source, args.width, args.height)
    _require_same_camera(capture, source, camera, refuse=True)
    zone = None
    zone_error = None
    try:
        print("warming up...")
        until = time.time() + 2.0
        while time.time() < until:
            capture.read()
        result = tune(capture, camera, power_line_frequency=1 if args.mains == 50 else 2)
        try:
            width_cm, height_cm = zone_size(DEFAULT_ZONE_CONFIG)
            ok, zone_frame = capture.read()
            if not ok or zone_frame is None:
                raise ValueError("Camera stopped delivering frames before zone calibration")
            zone = calibrate_zone(zone_frame, width_cm, height_cm)
        except (ValueError, TypeError, KeyError, FileNotFoundError) as error:
            zone_error = str(error)
    finally:
        capture.release()

    settings = CameraSettings(
        backend="uvc",
        device=camera.usb_id,
        controls=result.controls,
        tuned_at=time.strftime("%Y-%m-%d %H:%M"),
        note=args.note,
    )
    path = save(settings, args.config or config_path(), ranges)
    if zone is not None:
        save_zone(DEFAULT_CALIBRATION, zone)
        print(f"calibrated {width_cm:g} x {height_cm:g} cm zone -> {DEFAULT_CALIBRATION}")
    elif zone_error:
        if DEFAULT_CALIBRATION.is_file():
            stale = DEFAULT_CALIBRATION.with_name(f"camera_zone.stale-{int(time.time())}.json")
            DEFAULT_CALIBRATION.rename(stale)
            print(f"previous calibration moved to {stale}")
        print(f"ZONE NOT CALIBRATED: {zone_error}; leave all four markers in view and tune again")
    print("\nfocus sweep (sharpness of the star at each lens position):")
    print(focus_chart(result.focus_curve, chosen=result.controls.get("focus")))
    if result.autofocus_guess is not None:
        print(f"autofocus suggested {result.autofocus_guess}; chose {result.controls.get('focus')}")
    for note in result.notes:
        print(f"NOTE: {note}")
    print(f"\nwrote {path}. It is applied now; check with: uv run trashdrop camcheck")
    return 0


def _cmd_camera_zone(args: argparse.Namespace) -> int:
    """Recalibrate the pick area without sweeping focus or changing controls."""

    from .dataset.capture import open_camera
    from .dataset.zone import DEFAULT_CALIBRATION, DEFAULT_ZONE_CONFIG, calibrate_zone, save_zone, zone_size

    source = _resolve_camera(args.camera, args.width, args.height)
    capture = open_camera(source, args.width, args.height)
    try:
        for _ in range(20):
            ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped delivering frames")
        width_cm, height_cm = zone_size(args.config)
        zone = calibrate_zone(frame, width_cm, height_cm)
    finally:
        capture.release()
    save_zone(args.output, zone)
    print(f"calibrated {width_cm:g} x {height_cm:g} cm camera-centred zone -> {args.output}")
    print(f"image box: x={zone.x} y={zone.y} width={zone.width} height={zone.height}")
    return 0


def _cmd_camera_probe(args: argparse.Namespace) -> int:
    from .camera.uvc import probe_access
    from .rig import overhead_usb_id

    verdict = probe_access(args.usb_id or overhead_usb_id())
    if verdict == "ok":
        print()
        print("VERDICT: camera settings can be read and written. Next:")
        print("    uv run trashdrop camera tune --camera auto")
        return 0
    print()
    print("VERDICT: blocked -- unplug and replug the camera and try again; if it persists,")
    print("see the fallback in docs/CAMERA.md.")
    return 1


def _cmd_camera_markers(args: argparse.Namespace) -> int:
    from .camera.markers import write_sheet

    pdf, png = write_sheet(args.out)
    print(f"wrote {pdf}\n      {png}")
    print("print the PDF at ACTUAL SIZE (100 %, no 'fit to page'); the scale bar must measure 100 mm")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    from .dataset.capture import CaptureConfig, run_capture

    source = _resolve_camera(args.camera, args.width, args.height)
    config = CaptureConfig(
        session=args.session,
        root=args.data,
        source=source,
        width=args.width,
        height=args.height,
        lighting=args.lighting,
    )
    run_capture(config, category=args.category, object_id=args.object_id)
    return 0


def _cmd_camcheck(args: argparse.Namespace) -> int:
    from .dataset.camcheck import run_camcheck

    report = run_camcheck(
        _resolve_camera(args.camera, args.width, args.height),
        idle_seconds=args.seconds,
        interactive=not args.quick,
        width=args.width,
        height=args.height,
    )
    return 0 if report.usable else 1


def _cmd_autolabel(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .dataset.autolabel import autolabel_session

    report = autolabel_session(
        args.session, root=args.data, threshold=args.threshold, holdout_fraction=args.holdout
    )
    print(json.dumps(asdict(report), indent=2))
    if report.labelled == 0:
        print("\nNothing was labelled. Check that the background matches the lighting label.")
        return 1
    thin = [c for c, n in report.per_category_objects.items() if n < 10]
    if thin:
        print(
            f"\nWarning: fewer than 10 distinct objects in {', '.join(thin)}. "
            "Variety of objects matters more than number of frames."
        )
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from .dataset.review import contact_sheets, drop_objects, reject_frames

    if args.reject_frame:
        added = reject_frames(args.session, args.reject_frame, root=args.data)
        print(f"excluded {added} frame(s) from future autolabel runs")
        return 0
    if args.drop:
        removed = drop_objects(args.session, args.drop, root=args.data)
        print(f"removed {removed} crops for {', '.join(args.drop)}")
        return 0
    for path in contact_sheets(args.session, root=args.data):
        print(f"wrote {path}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from .dataset.manifest import read_manifest, summarise

    rows = read_manifest(args.data / "raw" / args.session / "manifest.csv")
    print(json.dumps(summarise(rows), indent=2))
    return 0


def _cmd_dataset_index(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .dataset.trashnet import write_manifest

    report = write_manifest(args.dataset_root, args.manifest)
    print(json.dumps(asdict(report), indent=2))
    return 0


def _cmd_taco_index(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .dataset.taco import write_taco_manifest

    report = write_taco_manifest(args.annotations, args.images, args.manifest)
    print(json.dumps(asdict(report), indent=2))
    return 0


# --- wiring ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from .station import SORT_CATEGORIES

    parser = argparse.ArgumentParser(
        prog="trashdrop", description="Two-arm SO-101 trash sorting cell"
    )
    parser.add_argument("--out", type=Path, default=Path("out"), help="artefact directory")
    parser.add_argument("--data", type=Path, default=Path("data"), help="dataset directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("sim", help="run the two-arm sorting demo")
    sim.add_argument("--viewer", action="store_true", help="live window (mjpython on macOS)")
    sim.add_argument("--no-video", action="store_true", help="skip recording")
    sim.set_defaults(func=_cmd_sim)

    probe = sub.add_parser("probe", help="check the layout is reachable")
    probe.add_argument("--map", action="store_true", help="also print a reachability grid")
    probe.add_argument("--z", type=float, default=0.018, help="height for the grid")
    probe.add_argument("--free", action="store_true", help="do not hold the gripper vertical")
    probe.set_defaults(func=_cmd_probe)

    scene = sub.add_parser("scene", help="dump the assembled MJCF")
    scene.set_defaults(func=_cmd_scene)

    plan = sub.add_parser("plan", help="show assignment without running physics")
    plan.set_defaults(func=_cmd_plan)

    serve_parser = sub.add_parser("serve", help="open intake API for other teams' robots")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8742)
    serve_parser.add_argument(
        "--mock", action="store_true", help="sort on a timer so others can integrate early"
    )
    serve_parser.set_defaults(func=_cmd_serve)

    cameras = sub.add_parser("cameras", help="which camera index is which? saves a snapshot of each")
    cameras.add_argument("--max-index", type=int, default=5)
    cameras.add_argument("--width", type=int, default=1920)
    cameras.add_argument("--height", type=int, default=1080)
    cameras.set_defaults(func=_cmd_cameras)

    rig = sub.add_parser("rig", help="arms, wrist cameras and the overhead camera: which is which, all answering?")
    rig_sub = rig.add_subparsers(dest="rig_command", required=True)
    rig_check = rig_sub.add_parser("check", help="every device in rig.toml plugged in and answering; a snapshot per camera")
    rig_check.add_argument("--no-video", action="store_true", help="skip opening the cameras' video")
    rig_check.set_defaults(func=_cmd_rig_check)
    rig_identify = rig_sub.add_parser(
        "identify", help="move each joint of each arm by hand when asked; learns which adapter is which arm"
    )
    rig_identify.add_argument("--seconds", type=float, default=30.0, help="time allowed for each joint")
    rig_identify.add_argument("--quick", action="store_true", help="only the gripper of each arm")
    rig_identify.set_defaults(func=_cmd_rig_identify)
    rig_touch = rig_sub.add_parser("touch", help="touch reference points with a fingertip; where the arm stands")
    rig_touch.add_argument("arm")
    rig_touch.add_argument("--tape", action="store_true", help="the tape square's inner corners, not the sheet's markers")
    rig_touch.set_defaults(func=_cmd_rig_touch)
    rig_roll = rig_sub.add_parser("roll", help="find where the arm's wrist roll really has its zero")
    rig_roll.add_argument("arm")
    rig_roll.set_defaults(func=_cmd_rig_roll)

    pick = sub.add_parser("pick", help="the overhead camera finds an item; an arm picks it and drops it aside")
    pick.add_argument("--camera", default="auto", help="stream index; auto finds the webcam")
    pick.add_argument("--dry-run", action="store_true", help="only hover over the item, never grasp")
    pick.add_argument("--arm", default=None,
                      help="use only this arm: left, right (or its label); by default the arm on the side the "
                           "item's material goes to takes it")
    pick.add_argument("--material", choices=("plastic", "metal", "paper"), default=None,
                      help="say what the item is instead of asking the classifier (plastic and metal go left)")
    pick.add_argument("--any-arm", action="store_true",
                      help="do not sort: the nearer arm takes every item and drops it on its own side")
    pick.add_argument("--min-confidence", type=float, default=None,
                      help="how sure the classifier must be before an item is sorted (default 0.8)")
    pick.add_argument("--fingertips", type=float, default=None,
                      help="how far above the table the fixed fingertip comes down, cm (default 0.5, at least 0.2)")
    pick.set_defaults(func=_cmd_pick)

    web = sub.add_parser("web", help="the cell in a browser: live stream with overlays, pick, auto sort, settings")
    web.add_argument("--host", default="127.0.0.1",
                     help="0.0.0.0 lets other machines open the page -- and move the arms")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--camera", default="auto", help="stream index; auto finds the webcam")
    web.add_argument("--arm", default=None, help="use only this arm: left, right (or its label)")
    web.add_argument("--demo", action="store_true", help="no camera or arms: out/'s saved pictures and pretend arms")
    web.add_argument("--open", action="store_true", help="open the page in the default browser")
    web.set_defaults(func=_cmd_web)

    arm = sub.add_parser("arm", help="the real arms: status, named poses, slow moves (left / right / F01 / F02)")
    arm_sub = arm.add_subparsers(dest="arm_command", required=True)
    arm_status = arm_sub.add_parser("status", help="every joint in degrees, torque, temperature; moves nothing")
    arm_status.add_argument("--arm", default=None)
    arm_status.set_defaults(func=_cmd_arm_status)
    arm_save = arm_sub.add_parser("save", help="record the arm's current pose into poses.toml")
    arm_save.add_argument("arm")
    arm_save.add_argument("pose")
    arm_save.set_defaults(func=_cmd_arm_save)
    arm_go = arm_sub.add_parser("go", help="move slowly to a pose from poses.toml")
    arm_go.add_argument("arm")
    arm_go.add_argument("pose")
    arm_go.add_argument("--speed", type=float, default=None, help="deg/s, capped by max_speed in rig.toml")
    arm_go.set_defaults(func=_cmd_arm_go)
    arm_jog = arm_sub.add_parser("jog", help="move one joint by some degrees")
    arm_jog.add_argument("arm")
    arm_jog.add_argument("joint")
    arm_jog.add_argument("degrees", type=float)
    arm_jog.add_argument("--speed", type=float, default=None)
    arm_jog.set_defaults(func=_cmd_arm_jog)
    arm_gripper = arm_sub.add_parser("gripper", help="open, close, or a percent open")
    arm_gripper.add_argument("arm")
    arm_gripper.add_argument("value", help="open, close, or 0..100")
    arm_gripper.add_argument("--speed", type=float, default=None)
    arm_gripper.set_defaults(func=_cmd_arm_gripper)
    arm_test = arm_sub.add_parser("test", help="small back-and-forth moves of each joint, asking before each")
    arm_test.add_argument("arm")
    arm_test.add_argument("--degrees", type=float, default=6.0, help="per joint, at most 15")
    arm_test.add_argument("--speed", type=float, default=15.0, help="deg/s")
    arm_test.set_defaults(func=_cmd_arm_test)
    arm_pick = arm_sub.add_parser("pick", help="play above -> grab -> close -> drop -> neutral from poses.toml")
    arm_pick.add_argument("arm")
    arm_pick.add_argument("--speed", type=float, default=None, help="deg/s, capped by max_speed in rig.toml")
    arm_pick.add_argument("--no-check", action="store_true", help="carry on even if the jaws closed on nothing")
    arm_pick.set_defaults(func=_cmd_arm_pick)
    arm_where = arm_sub.add_parser("where", help="where the gripper is now, cm in the arm's frame; moves nothing")
    arm_where.add_argument("arm")
    arm_where.set_defaults(func=_cmd_arm_where)
    arm_reach = arm_sub.add_parser("reach", help="put the gripper at x y z cm (arm's frame), fingers down")
    arm_reach.add_argument("arm")
    arm_reach.add_argument("x", type=float, help="cm forward of the base")
    arm_reach.add_argument("y", type=float, help="cm to the arm's left")
    arm_reach.add_argument("z", type=float, help="cm above the bottom of the base")
    arm_reach.add_argument("--yaw", type=float, default=None, help="deg: the direction the jaw closes along")
    arm_reach.add_argument("--speed", type=float, default=None)
    arm_reach.add_argument("--low", action="store_true", help="allow a TCP below 3 cm")
    arm_reach.set_defaults(func=_cmd_arm_reach)
    arm_hold = arm_sub.add_parser("hold", help="torque on where the arm is; moves nothing")
    arm_hold.add_argument("arm")
    arm_hold.set_defaults(func=_cmd_arm_hold)
    arm_relax = arm_sub.add_parser("relax", help="torque off: the arm goes limp (hold it if it is in the air)")
    arm_relax.add_argument("arm")
    arm_relax.add_argument("--now", action="store_true", help="skip the 3 s warning")
    arm_relax.set_defaults(func=_cmd_arm_relax)

    camera = sub.add_parser("camera", help="lock focus/exposure/white balance via camera.toml")
    camera_sub = camera.add_subparsers(dest="camera_command", required=True)

    sharp = camera_sub.add_parser("sharpness", help="live sharpness number while turning a lens by hand")
    sharp.add_argument("--usb-id", default=None, help="vvvv:pppp; default: the overhead camera")
    sharp.set_defaults(func=_cmd_camera_sharpness)

    show = camera_sub.add_parser("show", help="every control the webcam offers, with current values")
    show.add_argument("--usb-id", default=None, help="vvvv:pppp, if several webcams are plugged in")
    show.set_defaults(func=_cmd_camera_show)

    apply_parser = camera_sub.add_parser("apply", help="push camera.toml to the webcam")
    apply_parser.add_argument("--config", type=Path, default=None)
    apply_parser.add_argument("--usb-id", default=None)
    apply_parser.set_defaults(func=_cmd_camera_apply)

    tape_parser = camera_sub.add_parser("tape", help="click the taped zone's inner corners in the camera picture")
    tape_parser.add_argument("--camera", default="auto", help="stream index; auto finds the webcam")
    tape_parser.set_defaults(func=_cmd_camera_tape)

    sheet_parser = camera_sub.add_parser("sheet", help="find the marker sheet in the overhead camera; pixels -> cm")
    sheet_parser.add_argument("--camera", default="auto", help="stream index; auto finds the webcam")
    sheet_parser.add_argument("--width", type=int, default=1920)
    sheet_parser.add_argument("--height", type=int, default=1080)
    sheet_parser.set_defaults(func=_cmd_camera_sheet)

    tune_parser = camera_sub.add_parser("tune", help="find settings for this rig and write camera.toml")
    tune_parser.add_argument("--camera", default="auto", help="stream index; auto finds the webcam")
    tune_parser.add_argument("--width", type=int, default=1920)
    tune_parser.add_argument("--height", type=int, default=1080)
    tune_parser.add_argument("--mains", type=int, choices=(50, 60), default=50, help="Hz; Europe is 50")
    tune_parser.add_argument("--note", default="", help="e.g. 'home rig, 70 cm'")
    tune_parser.add_argument("--config", type=Path, default=None)
    tune_parser.add_argument("--usb-id", default=None)
    tune_parser.set_defaults(func=_cmd_camera_tune)

    zone_parser = camera_sub.add_parser("zone", help="calibrate the capture zone from the ArUco sheet")
    zone_parser.add_argument("--camera", default="auto")
    zone_parser.add_argument("--width", type=int, default=1920)
    zone_parser.add_argument("--height", type=int, default=1080)
    zone_parser.add_argument("--config", type=Path, default=Path("capture_zone.toml"))
    zone_parser.add_argument("--output", type=Path, default=Path("camera_zone.json"))
    zone_parser.set_defaults(func=_cmd_camera_zone)

    probe = camera_sub.add_parser("probe", help="which route to the camera's settings does this Mac allow?")
    probe.add_argument("--usb-id", default=None)
    probe.set_defaults(func=_cmd_camera_probe)

    markers = camera_sub.add_parser("markers", help="printable A4: focus target + pick-zone ArUco markers")
    markers.set_defaults(func=_cmd_camera_markers)

    capture = sub.add_parser("capture", help="shoot a dataset session on the rig")
    capture.add_argument("--session", required=True, help="e.g. 2026-09-23-home")
    capture.add_argument("--category", default="plastic", choices=SORT_CATEGORIES,
                         help="starting material; keys 1-3 switch")
    capture.add_argument("--object-id", default=None, help="default: next free <material>_NN")
    capture.add_argument("--camera", default="auto", help="auto finds the webcam; or an index, or a phone URL")
    capture.add_argument("--width", type=int, default=1920)
    capture.add_argument("--height", type=int, default=1080)
    capture.add_argument("--burst", type=int, default=1, help=argparse.SUPPRESS)
    capture.add_argument("--lighting", default="default")
    capture.add_argument("--no-auto", action="store_true", help=argparse.SUPPRESS)
    capture.set_defaults(func=_cmd_capture)

    camcheck = sub.add_parser("camcheck", help="is this camera good enough to shoot through?")
    camcheck.add_argument("--camera", default="auto", help="auto finds the webcam; or an index, or a URL")
    camcheck.add_argument("--seconds", type=float, default=8.0, help="idle measurement window")
    camcheck.add_argument("--quick", action="store_true", help="skip the dark-item step")
    camcheck.add_argument("--width", type=int, default=1920)
    camcheck.add_argument("--height", type=int, default=1080)
    camcheck.set_defaults(func=_cmd_camcheck)

    autolabel = sub.add_parser("autolabel", help="derive masks and crops from a session")
    autolabel.add_argument("--session", required=True)
    autolabel.add_argument("--threshold", type=int, default=28)
    autolabel.add_argument("--holdout", type=float, default=0.25)
    autolabel.set_defaults(func=_cmd_autolabel)

    review = sub.add_parser("review", help="contact sheets, or reject bad captures")
    review.add_argument("--session", required=True)
    review_actions = review.add_mutually_exclusive_group()
    review_actions.add_argument("--drop", nargs="*", help="object ids to remove")
    review_actions.add_argument("--reject-frame", nargs="+", help="CATEGORY/OBJECT/FILE to permanently exclude from autolabel")
    review.set_defaults(func=_cmd_review)

    summary = sub.add_parser("summary", help="counts for a capture session")
    summary.add_argument("--session", required=True)
    summary.set_defaults(func=_cmd_summary)

    trashnet = sub.add_parser("dataset-index", help="index a TrashNet-style folder set")
    trashnet.add_argument("dataset_root", type=Path)
    trashnet.add_argument("--manifest", type=Path, default=Path("out/trashnet_manifest.jsonl"))
    trashnet.set_defaults(func=_cmd_dataset_index)

    taco = sub.add_parser("taco-index", help="index TACO COCO annotations")
    taco.add_argument("--annotations", type=Path, required=True)
    taco.add_argument("--images", type=Path, required=True)
    taco.add_argument("--manifest", type=Path, default=Path("out/taco_manifest.jsonl"))
    taco.set_defaults(func=_cmd_taco_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
