"""Reachability probing: where can each arm actually put the gripper?

Run this before drilling anything into the real table, and again after any
change to the numbers in ``station.py``. It answers the only question that
decides the layout: for a candidate bin or pick-zone position, does the IK
solver converge from the arm that has to reach it?

The map is printed in the shared table frame, so the two arms' maps can be
read against each other -- a cell that both arms cover is reachable by both.
"""

from __future__ import annotations

import numpy as np

from .station import (
    ARMS,
    BINS,
    GRASP_Z,
    MAX_VERTICAL_Z,
    PICK_ZONE,
    SAFE_Z,
)

# Under this residual the pose is considered achievable.
REACH_TOLERANCE = 0.005


def reachability_map(cell, z: float = GRASP_Z, keep_vertical: bool = True) -> None:
    """Print one grid per arm over the whole table, in the shared frame."""

    xs = np.arange(-0.32, 0.33, 0.04)
    ys = np.arange(-0.02, -0.45, -0.04)
    print(
        f"\nz = {z:.3f} m, gripper vertical: {keep_vertical}   "
        f"('.' = reachable, else IK residual in mm)"
    )
    for mount in ARMS:
        arm = cell.arms[mount.name]
        print(f"\n  arm {mount.name}  (base at {mount.pose.x:+.2f}, {mount.pose.y:+.2f})")
        print("        x:" + "".join(f"{x:+7.2f}" for x in xs))
        for y in ys:
            row = ""
            for x in xs:
                error = arm.reach_error(float(x), float(y), z, keep_vertical)
                row += "      ." if error < REACH_TOLERANCE else f"{min(9999, int(error * 1000)):7d}"
            print(f"  y ={y:+.2f} {row}")


def check_layout(cell, verbose: bool = True) -> bool:
    """Verify every bin and every pick-zone corner is reachable by its arm.

    Returns True when the layout is usable. This is the check to run in CI and
    after re-measuring the real table -- it turns a silent geometry mistake
    into a failure with the offending position named.
    """

    ok = True

    if verbose:
        print("bins (gripper free, at drop height):")
    for key, spec in BINS.items():
        owners = [spec.arm] if spec.arm != "any" else [m.name for m in ARMS]
        for owner in owners:
            error = cell.arms[owner].reach_error(spec.x, spec.y, SAFE_Z, keep_vertical=False)
            good = error < REACH_TOLERANCE
            ok &= good
            if verbose:
                mark = "ok " if good else "MISS"
                print(
                    f"  {mark} {key:8s} ({spec.x:+.3f}, {spec.y:+.3f}) "
                    f"from {owner:5s}  residual {error * 1000:6.1f} mm"
                )

    if verbose:
        print("\npick zone (gripper vertical, at grasp height) -- both arms must reach:")
    points = list(PICK_ZONE.corners()) + [(PICK_ZONE.center_x, PICK_ZONE.center_y)]
    for x, y in points:
        for mount in ARMS:
            error = cell.arms[mount.name].reach_error(x, y, GRASP_Z, keep_vertical=True)
            good = error < REACH_TOLERANCE
            ok &= good
            if verbose:
                mark = "ok " if good else "MISS"
                print(
                    f"  {mark} ({x:+.3f}, {y:+.3f}) from {mount.name:5s}  "
                    f"residual {error * 1000:6.1f} mm"
                )

    if verbose:
        print(
            f"\ntravel height {SAFE_Z:.3f} m, vertical-gripper ceiling "
            f"{MAX_VERTICAL_Z:.3f} m"
        )
        print("LAYOUT OK" if ok else "LAYOUT HAS UNREACHABLE POSITIONS")
    return ok
