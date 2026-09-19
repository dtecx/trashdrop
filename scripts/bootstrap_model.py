#!/usr/bin/env python3
"""Fetch the SO-ARM100 model from mujoco_menagerie, sparsely.

The full repository is several hundred megabytes and we need one folder, so
this does a blobless sparse checkout (~7 MB). Written in Python rather than
shell because most of the team is on Windows.

The SO-ARM100 is kinematically the SO-101, which is why it is the model used.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
FOLDER = "trs_so_arm100"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "mujoco_menagerie"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    marker = DESTINATION / FOLDER / "so_arm100.xml"
    if marker.is_file():
        print(f"already present: {marker.relative_to(ROOT)}")
        return 0

    if shutil.which("git") is None:
        print("error: git is not on PATH", file=sys.stderr)
        return 1

    if DESTINATION.exists():
        print(f"error: {DESTINATION} exists but has no model; remove it and retry", file=sys.stderr)
        return 1

    try:
        run(
            [
                "git", "clone", "--depth", "1",
                "--filter=blob:none", "--sparse",
                REPOSITORY, str(DESTINATION),
            ]
        )
        run(["git", "sparse-checkout", "set", FOLDER], cwd=DESTINATION)
    except subprocess.CalledProcessError as error:
        print(f"error: git failed with status {error.returncode}", file=sys.stderr)
        return error.returncode

    if not marker.is_file():
        print(f"error: clone finished but {marker} is missing", file=sys.stderr)
        return 1

    size = sum(f.stat().st_size for f in DESTINATION.rglob("*") if f.is_file())
    print(f"fetched {marker.relative_to(ROOT)} ({size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
