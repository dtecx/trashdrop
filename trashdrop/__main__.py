"""Command-line entry points for the local TrashDrop virtual workcell."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import write_manifest
from .planning import DetectedItem, TwoArmDispatcher
from .scene_builder import build_station, validate_station
from .station import SAMPLE_ITEMS
from .taco import write_taco_manifest


def _sample_detections() -> list[DetectedItem]:
    return [
        DetectedItem(item_id=item_id, category=category, x=x, y=y)
        for item_id, category, x, y in SAMPLE_ITEMS
    ]


def _print_plan() -> None:
    for assignment in TwoArmDispatcher().dispatch(_sample_detections()):
        shared = " reserve shared strip" if assignment.requires_handoff_clearance else ""
        print(
            f"{assignment.item.item_id:16} -> {assignment.arm:5} "
            f"-> {assignment.bin_category:7} "
            f"({assignment.bin_x:+.2f}, {assignment.bin_y:+.2f}){shared}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TrashDrop dual SO-101 virtual workcell")
    parser.add_argument(
        "command",
        choices=("build", "validate", "viewer", "plan", "demo", "dataset-index", "taco-index"),
        nargs="?",
        default="validate",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        help="TrashNet-style class-folder dataset root (dataset-index only)",
    )
    parser.add_argument("--out", type=Path, help="generated MJCF location")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/trashnet_manifest.jsonl"),
        help="JSONL manifest location (dataset-index only)",
    )
    parser.add_argument("--annotations", type=Path, help="TACO COCO annotations JSON (taco-index only)")
    args = parser.parse_args()

    if args.command == "plan":
        _print_plan()
        return
    if args.command == "dataset-index":
        if args.dataset_root is None:
            parser.error("dataset-index requires DATASET_ROOT")
        report = write_manifest(args.dataset_root, args.manifest)
        print(f"indexed {report.images} images -> {report.manifest}")
        print(f"source classes: {report.source_counts}")
        print(f"station routes: {report.station_counts}; reject: {report.reject_images}")
        return
    if args.command == "taco-index":
        if args.dataset_root is None or args.annotations is None:
            parser.error("taco-index requires IMAGE_ROOT and --annotations ANNOTATIONS_JSON")
        manifest = args.manifest
        if manifest == Path("build/trashnet_manifest.jsonl"):
            manifest = Path("build/taco_object_manifest.jsonl")
        report = write_taco_manifest(args.annotations, args.dataset_root, manifest)
        print(f"indexed {report.objects} TACO objects -> {report.manifest}")
        print(f"station routes: {report.route_counts}; missing images: {report.missing_images}")
        return
    if args.command == "demo":
        from .simulator import run_taco_demo

        report = run_taco_demo()
        print(f"two-arm TACO demo: {report.placed}/{report.assigned} placed; misses {report.misses}")
        print(f"scene: {report.scene}")
        print(f"trace: {report.trace}")
        return

    scene = build_station(output_path=args.out)
    print(f"generated {scene}")
    if args.command == "build":
        return

    result = validate_station(scene)
    print(
        "validated dual-arm scene: "
        f"{result['joints']} joints, {result['actuators']} actuators, {result['bodies']} bodies"
    )
    if args.command == "viewer":
        import mujoco
        from mujoco import viewer

        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
