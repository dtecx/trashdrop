# Working agreement for agents

Read this before changing anything. It is the shared contract for any coding
agent in this repository; `CLAUDE.md` adds only Claude Code specifics.

## What this is

A cell that sorts household waste with **two SO-101 arms**, built for the Alien
Bazaar 2026 hackathon. Another team's robot delivers trash into a shared pick
zone; these two arms sort it into five bins.

The repository contains the *virtual* model of that cell plus the tooling to
build a real dataset. Hardware is not connected yet.

### Facts that shape every decision

| | |
|---|---|
| Hackathon | Alien Bazaar, Warsaw, 2026-09-25 to 09-27; this is preparation |
| Judged on | creativity, **proactive collaboration with other teams**, and effort during the event |
| Hardware | issued at the venue. Nothing can be bring-upped in advance |
| Success criterion the team set | **zero sorting errors**, not speed |
| Bins | `bio`, `paper`, `plastic`, `metal`, `mixed` |
| Compute on site | one MacBook, arms plugged into it |
| Camera | **not decided** — possibly an Android phone as a webcam |
| Team | one person on macOS, everyone else on **Windows** |
| Leader arm | at least one available, so teleop recording is possible |
| Trash | food packaging, drink cans, bottles — unknown exact mix |

## Layout: the arms face each other

Two bases 43 cm apart on opposite sides of a shared 20 x 15 cm pick zone.

The single most important consequence: **both arms reach every point of the
pick zone**, so an item is assigned to an arm by *material*, never by position,
and there is no hand-off problem. What it costs instead is exclusivity — two
arms sharing one volume must never enter it at once, so execution is
serialised, one arm at a time. Do not "optimise" that into concurrent motion
without a real collision checker.

## Module layers

Import direction is strictly downward. Respect it: a teammate who is only
shooting dataset photos installs `--extra dataset` and must never be forced to
install MuJoCo.

```
station.py  geometry.py  planning.py        pure Python, zero dependencies
        |
dataset/  perception/                       needs opencv
        |
scene_builder.py  control.py                needs mujoco
simulator.py  probe.py
```

- `station.py` — all cell geometry as data: mounts, bins, pick zone, camera,
  poses, payload limits. **Change numbers here, not in the modules that read
  them**, then run `uv run trashdrop probe`.
- `geometry.py` — `Pose2D` frame transforms. The only place that knows how a
  rotated arm base maps to the shared table frame.
- `planning.py` — assignment by material, low-confidence rerouting. No MuJoCo,
  no camera. A hardware executor consumes its output unchanged.
- `control.py` — per-arm IK and `send()`. **`ArmController.send` is the
  hardware seam**: implementing it against `lerobot`'s `SO101Follower` is what
  makes the rest of the stack drive real arms.
- `perception/` — two-stage: class-agnostic detector, then crop classifier.
  `photometric.py` cancels exposure drift before any background subtraction;
  without it a camera that re-exposes when an item arrives makes the whole
  frame read as foreground.
- `dataset/` — capture on the rig, autolabel, review; plus indexers for the
  public datasets.
- `simulator.py` — the cell, stepping, grasp, scoring.
- `api.py` — the open intake API. Standard library only, on purpose: no team
  should install anything to talk to us, and it must not break on venue wifi.
  Collaboration is a judging criterion, so treat this as product surface, not
  scaffolding.

## Invariants that must not break

These each cost real debugging time to find. Tests guard them; if a test in
this list fails, fix the code, not the test.

1. **`<compiler angle="radian"/>` in the assembled world MJCF.** MJCF defaults
   to degrees. Without it, `euler=[0,0,pi]` becomes a 3-degree rotation and the
   second arm quietly ends up beside the first instead of facing it. Guarded by
   `test_scene_builder.py::test_arms_face_each_other`.

2. **`cone="elliptic" impratio="10"` on the parent spec.** On `MjSpec.attach`
   the parent's options win, and these are what the arm model asks for.

3. **The stow pose, not the baseline's observe pose.** `STOW_POSE` folds each
   arm vertically over its own base. The single-arm baseline swung the arm
   forward and aside, which with two facing arms puts both tools over the
   middle of the table: they occlude half the items from the camera and
   physically jam against each other.

4. **Scoring is measured from physics.** `release()` drops the item from where
   the tool actually is and lets it fall. Never move an item to the bin
   coordinates on release — that turns the score into a constant. Guarded by
   `test_simulator.py`.

5. **Dataset splits are by `object_id`, never by frame.** Frames in one burst
   are near-duplicates; splitting inside a burst inflates accuracy badly.

6. **`perception/color.py` is simulation-only and proves nothing.** It reads
   back the colour the simulator itself assigned. Never cite its results as
   perception performance, and never extend it toward real trash.

7. **Never require the camera's exposure to be locked.** It frequently cannot
   be: OpenCV on macOS goes through AVFoundation, which ignores the exposure
   property on most cameras. Compensate instead — `perception/photometric.py`
   fits the drift on the pixels that did not change. Guarded by
   `test_photometric.py`, which asserts that a naive difference floods and the
   compensated one does not.

8. **Anything uncertain goes to `mixed`.** Low classifier confidence, unknown
   category, too wide for the jaws, too heavy. The pitch promises the cell
   flags what it cannot handle instead of guessing; that promise lives in
   `TwoArmDispatcher.dispatch` and `station.is_graspable`.

## Commands

```bash
./scripts/bootstrap_model.sh          # or: python scripts/bootstrap_model.py
uv sync --extra simulation --group dev
uv run trashdrop probe                # layout reachability -- run after ANY geometry edit
uv run trashdrop sim                  # full sort, writes out/
uv run trashdrop serve --mock         # intake API for other teams
uv run trashdrop camcheck             # is this camera worth shooting through?
uv run mjpython -m trashdrop sim --viewer   # live window (macOS needs mjpython)
uv run python -m pytest tests/ -q
```

Dataset work needs only `uv sync --extra dataset`:

```bash
uv run trashdrop capture --session 2026-09-20-kitchen --category plastic --object-id bottle_01
uv run trashdrop autolabel --session 2026-09-20-kitchen
uv run trashdrop review --session 2026-09-20-kitchen
```

## Conventions

- **All code, comments, docstrings and commit messages in English.** The team
  speaks Russian; the repository does not.
- Comments explain *why*, especially for a constant that came from measurement.
  A number with no explanation is a number nobody dares change.
- Never commit: `mujoco_menagerie/`, `.venv/`, `out/`, `data/raw/`,
  `data/bg/`, `data/crops/`, model weights.
- Windows matters. Prefer Python scripts over shell scripts, and `pathlib`
  over string paths.
- Do not add a training stack (torch, ultralytics) to this project's
  dependencies. Train elsewhere, export to ONNX, load it with
  `perception/classifier.py`.

## Before saying something works

Run `uv run trashdrop probe` and `uv run python -m pytest tests/ -q`. For a
change touching motion or geometry, also run `uv run trashdrop sim` and quote
the actual `sorted correctly: N/4` line. The simulation is fast (about 7
seconds); there is no excuse for reporting a motion change unverified.
