"""Command-line entry points for the local TrashDrop virtual workcell."""

from __future__ import annotations

import argparse
from pathlib import Path

from .planning import DetectedItem, TwoArmDispatcher
from .scene_builder import build_station, validate_station
from .station import SAMPLE_ITEMS


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
        choices=("build", "validate", "viewer", "plan"),
        nargs="?",
        default="validate",
    )
    parser.add_argument("--out", type=Path, help="generated MJCF location")
    args = parser.parse_args()

    if args.command == "plan":
        _print_plan()
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
