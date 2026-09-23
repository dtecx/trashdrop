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
    print()
    print("open the snapshots: the webcam is the one looking down at the table.")
    print("then use it with --camera <index>")
    return 0


def _cmd_camera_show(args: argparse.Namespace) -> int:
    from .camera import UvcCamera

    camera = UvcCamera.find(args.usb_id)
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

    # Talk to the camera first: if this needs sudo, say so before anything
    # else happens.
    camera = UvcCamera.find(args.usb_id)
    ranges = camera.ranges()
    print(f"{camera.describe()}: {len(ranges)} adjustable controls")

    capture = open_camera(_source(args.camera), args.width, args.height)
    try:
        print("warming up...")
        until = time.time() + 2.0
        while time.time() < until:
            capture.read()
        result = tune(capture, camera, power_line_frequency=1 if args.mains == 50 else 2)
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
    print("\nfocus sweep (sharpness of the target at each lens position):")
    print(focus_chart(result.focus_curve))
    for note in result.notes:
        print(f"note: {note}")
    print(f"\nwrote {path}. It is applied now; check with: uv run trashdrop camcheck --camera {args.camera}")
    return 0


def _cmd_camera_markers(args: argparse.Namespace) -> int:
    from .camera.markers import write_sheet

    pdf, png = write_sheet(args.out)
    print(f"wrote {pdf}\n      {png}")
    print("print the PDF at ACTUAL SIZE (100 %, no 'fit to page'); the scale bar must measure 100 mm")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    from .dataset.capture import CaptureConfig, run_capture

    config = CaptureConfig(
        session=args.session,
        root=args.data,
        source=_source(args.camera),
        width=args.width,
        height=args.height,
        burst=args.burst,
        lighting=args.lighting,
        auto=not args.no_auto,
    )
    run_capture(config, category=args.category, object_id=args.object_id)
    return 0


def _cmd_camcheck(args: argparse.Namespace) -> int:
    from .dataset.camcheck import run_camcheck

    report = run_camcheck(
        _source(args.camera),
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
    from .dataset.review import contact_sheets, drop_objects

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

    camera = sub.add_parser("camera", help="lock focus/exposure/white balance via camera.toml")
    camera_sub = camera.add_subparsers(dest="camera_command", required=True)

    show = camera_sub.add_parser("show", help="every control the webcam offers, with current values")
    show.add_argument("--usb-id", default=None, help="vvvv:pppp, if several webcams are plugged in")
    show.set_defaults(func=_cmd_camera_show)

    apply_parser = camera_sub.add_parser("apply", help="push camera.toml to the webcam")
    apply_parser.add_argument("--config", type=Path, default=None)
    apply_parser.add_argument("--usb-id", default=None)
    apply_parser.set_defaults(func=_cmd_camera_apply)

    tune_parser = camera_sub.add_parser("tune", help="find settings for this rig and write camera.toml")
    tune_parser.add_argument("--camera", default="0", help="OpenCV index of the same webcam")
    tune_parser.add_argument("--width", type=int, default=1920)
    tune_parser.add_argument("--height", type=int, default=1080)
    tune_parser.add_argument("--mains", type=int, choices=(50, 60), default=50, help="Hz; Europe is 50")
    tune_parser.add_argument("--note", default="", help="e.g. 'home rig, 70 cm'")
    tune_parser.add_argument("--config", type=Path, default=None)
    tune_parser.add_argument("--usb-id", default=None)
    tune_parser.set_defaults(func=_cmd_camera_tune)

    markers = camera_sub.add_parser("markers", help="printable A4: focus target + pick-zone ArUco markers")
    markers.set_defaults(func=_cmd_camera_markers)

    capture = sub.add_parser("capture", help="shoot a dataset session on the rig")
    capture.add_argument("--session", required=True, help="e.g. 2026-09-23-home")
    capture.add_argument("--category", default="plastic", help="starting class; keys 1-5 switch")
    capture.add_argument("--object-id", default=None, help="default: next free <class>_NN")
    capture.add_argument("--camera", default="0", help="device index or stream URL")
    capture.add_argument("--width", type=int, default=1920)
    capture.add_argument("--height", type=int, default=1080)
    capture.add_argument("--burst", type=int, default=3, help="frames per SPACE press")
    capture.add_argument("--lighting", default="default")
    capture.add_argument("--no-auto", action="store_true", help="start with the auto-shutter off")
    capture.set_defaults(func=_cmd_capture)

    camcheck = sub.add_parser("camcheck", help="is this camera good enough to shoot through?")
    camcheck.add_argument("--camera", default="0", help="device index or stream URL")
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

    review = sub.add_parser("review", help="contact sheets, or drop bad objects")
    review.add_argument("--session", required=True)
    review.add_argument("--drop", nargs="*", help="object ids to remove")
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
