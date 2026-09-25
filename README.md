# TrashDrop

Two SO-101 arms facing each other across a shared pick zone, sorting plastic,
paper, and metal waste into three material bins with a shared `mixed` fallback.
Built for Alien Bazaar 2026, where another team's robot delivers the trash.

This repository holds the virtual model of the cell and the tooling to build a
real dataset. Everything stays inside this folder: dependencies live in a
project-local `.venv` managed by `uv`, nothing is installed system-wide, and
deleting the folder removes all of it.

```
uv run trashdrop sim
...
cycle 0: 3 item(s) in the pick zone
  paper    at (+0.062, -0.222) -> front -> bin paper
    above bin, tcp error 2 mm
cycle 1: 2 item(s) in the pick zone
  plastic  at (+0.010, -0.274) -> back  -> bin plastic
...
sorted correctly: 3/3  (picks 3, misses 0)
```

## Quick start

```bash
python scripts/bootstrap_model.py        # SO-ARM100 model, ~7 MB sparse checkout
uv sync --extra simulation --group dev

uv run trashdrop probe                   # is every bin and pick-zone corner reachable?
uv run trashdrop sim                     # full two-arm sort, ~7 s, writes out/
uv run python -m pytest tests/ -q        # full test suite
```

The real arms and cameras (`rig.toml` says which USB device is which):

```bash
uv sync --inexact --extra rig --extra arm
uv run trashdrop rig identify            # move the front arm by hand when asked
uv run trashdrop rig check               # every arm and camera answering, a snapshot each
```

Live viewer (macOS needs `mjpython`, which the `mujoco` wheel installs):

```bash
uv run mjpython -m trashdrop sim --viewer
```

Only shooting dataset photos? You do not need MuJoCo:

```bash
uv sync --extra dataset
uv run trashdrop capture --session 2026-09-20-kitchen --category plastic --object-id bottle_01
uv run trashdrop autolabel --session 2026-09-20-kitchen
uv run trashdrop review --session 2026-09-20-kitchen
```

## What is here

| | |
|---|---|
| `trashdrop/station.py` | All cell geometry as data. Change numbers here, then run `probe` |
| `trashdrop/planning.py` | Assignment by material; anything uncertain goes to `mixed` |
| `trashdrop/control.py` | Per-arm IK. `send()` is the hardware seam |
| `trashdrop/perception/` | Class-agnostic detector + crop classifier; ArUco homography |
| `trashdrop/dataset/` | Capture on the rig, autolabel, review; TACO and TrashNet indexers |
| `trashdrop/simulator.py` | The cell: stepping, grasp, scoring |
| `trashdrop/api.py` | Open intake API other teams' robots call. Standard library only |

## Reading order

- [AGENTS.md](AGENTS.md) — architecture, module layers, and the invariants that
  must not break. Start here before changing code.
- [docs/API.md](docs/API.md) — the open intake API. Hand this to any team
  whose robot wants to deliver trash to us.
- [docs/DATASET.md](docs/DATASET.md) — how to shoot the dataset so that nobody
  draws a bounding box.
- [docs/CAMERA.md](docs/CAMERA.md) — locking focus, exposure and white balance
  in `camera.toml`, and why that needs sudo on macOS.
- [docs/APPROACH.md](docs/APPROACH.md) — why classic CV + IK rather than a
  learned policy, and where a policy would still help.
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — the real table, ArUco, and using
  a phone as the camera.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layout diagram, data flow,
  motion sequence, frames.

## What the simulation proves

Reachability, transfers, the layout, the serialisation between two arms sharing
one volume, and the scoring — items are released from where the tool actually
is and physics decides where they land.

It does **not** prove grasping: the grasp is kinematic, so nothing here
predicts whether a real gripper holds a crushed can. It does not prove
perception either — `perception/color.py` reads back the colour the simulator
itself assigned, and exists only to exercise the motion stack. Real numbers for
both have to come from hardware and from our own crops.

## Not installed by default

No ROS, no training stack, no `lerobot` unless asked for with
`uv sync --extra teleop`. Train models elsewhere, export to ONNX, and load them
through `perception/classifier.py`.
