# Why classic CV + IK, and where a learned policy fits

## The decision

**Scripted motion with damped-least-squares IK is the primary path.** A learned
policy (ACT / diffusion policy via LeRobot) is an optional extra, installed
only on purpose, and is a stretch goal for specific hard items.

The deciding factor is the team's own success criterion: **zero sorting
errors**.

An ACT-style policy learns from roughly 50–100 teleoperated demonstrations per
task variation, and generalises poorly to objects it has not seen. On the day,
another team's robot delivers trash collected from a 350 m² house — the exact
mix is unknown in advance. A policy trained over six days on our own items will
meet something unfamiliar and fail, and it will fail *silently*, producing a
confident wrong motion.

The scripted path fails differently, and better:

- Its failures are legible — IK residual, classifier confidence, item width.
- It can decline. Low confidence, unknown class, too wide for the jaws, too
  heavy → the `mixed` bin. The pitch promises the cell flags what it cannot
  handle instead of guessing; that is one branch in scripted code and an open
  research problem in a policy.
- It adapts to a re-measured table by editing numbers in `station.py`.
- It needs no GPU. Training ACT on a MacBook is slow.

## Where a policy genuinely wins

Grasping irregular, deformable items. A fixed top-down pinch is weakest exactly
where trash is hardest — a crushed can, a flattened carton, a bottle on its
side. This is real and worth respecting.

Two things capture most of that benefit without the risk:

1. **Grasp pose from the segmentation mask.** The minimum-area rectangle gives
   a principal axis; close the jaws across the short side and refuse the item
   when that side exceeds `MAX_GRASP_WIDTH`. Pure classical CV, already wired
   into `Detection.width` and `station.is_graspable`, and it handles irregular
   shapes far better than a fixed heading.
2. **Record teleop demonstrations anyway.** The leader arm exists. Recording
   costs an evening, builds dataset value, and keeps the policy option open if
   the scripted grasp stalls on a specific item class on the day.

## If the policy path is taken

```bash
uv sync --extra teleop
```

Deliberately opt-in: LeRobot pulls a large dependency tree, and nothing else in
this repository needs it. Keep the scripted path working as the demo fallback —
do not replace it.

The integration point is already there. `ArmController.send` is the only method
that talks to an actuator; a policy drives the same interface.

## What the simulation does and does not prove

Worth stating plainly, because it is easy to overclaim:

| | |
|---|---|
| Reachability, transfers, layout, serialisation | **Proved** — that is what `probe` and `sim` measure |
| Scoring | **Honest** — items are dropped from the tool and physics decides |
| Grasping | **Not proved.** The grasp is kinematic: contacts are disabled and the item tracks the tool. Frictional grasping in simulation does not predict a real gripper on a crushed can |
| Perception | **Not proved at all.** `perception/color.py` reads back the colour the simulator assigned. It is a wiring test |

So `sorted correctly: 3/3` is a statement about motion. Perception performance
can only come from real crops, and grasp reliability only from hardware.

## Sequence for the remaining days

1. **Dataset first.** It has the longest lead time and nothing else is blocked
   by code. See [DATASET.md](DATASET.md).
2. Train a crop classifier outside this repository; export ONNX; load it with
   `perception/classifier.py`. Wire `BackgroundDetector` with it and check it
   against a held-out session.
3. Calibrate on the real table as soon as the arms are mounted — four ArUco
   markers, `HomographyCalibration`, residuals under a few millimetres. See
   [CALIBRATION.md](CALIBRATION.md).
4. Re-measure the layout, update `station.py`, run `trashdrop probe`.
5. Implement `send()` against the real arms.
6. Only then, if time remains, record teleop demonstrations.
