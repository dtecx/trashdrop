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
