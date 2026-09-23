# Architecture

## The cell

```
                              +y
        [paper]
       (-0.20,-0.12)
                    +-------------------+
                    |   FRONT  (0, 0)   |   yaw 0
                    +-------------------+
   +--------------------------------------------------+
   | ArUco                                      ArUco |
   |          SHARED PICK ZONE  0.20 x 0.15 m         |   <- delivery robot
   |            centred at (0, -0.215)                |      approaches along x
   | ArUco                                      ArUco |
   +--------------------------------------------------+        [mixed]
                    +-------------------+            (+0.28,-0.215), either arm
                    | BACK  (0, -0.43)  |   yaw 180
                    +-------------------+
       (-0.20,-0.31)                          (+0.20,-0.31)
        [plastic]                                  [metal]
```

Overhead camera at 0.60 m above the pick zone centre, 45° vertical FOV.

**Bases 43 cm apart, facing each other.** Each arm's own frame then sees the
pick zone at local y in [-0.14, -0.29], inside the envelope the single-arm
baseline was verified against. Both arms reach every point of the zone and
every material bin and the shared `mixed` bin, which `trashdrop probe` asserts.

Two consequences follow from this, and most of the design is downstream of them:

- **Assignment is by material, not by position.** Since either arm can reach
  any item, an item goes to whichever arm owns its bin. There is no hand-off
  and no item that only one arm can serve.
- **Execution is serialised.** Two arms sharing one volume must never enter it
  at once. One arm works while the other is stowed. Concurrent motion would
  need a real collision checker; the `requires_shared_zone` flag on every
  assignment is where that would hook in.

## Data flow

```
overhead frame
   |
   v
[1] detector  ---- class-agnostic: background subtraction against an empty-
   |                table reference. No training data needed.
   |  crop
   v
[2] classifier --- material + confidence. A few hundred crops per class.
   |
   v
Detection(category, x, y, yaw, confidence, width)   -- table frame, metres
   |
   v
TwoArmDispatcher  -- by material; low confidence / too wide / too heavy -> mixed
   |
   v
ArmAssignment(arm, bin, requires_shared_zone)
   |
   v
SortingCell.execute   -- IK, approach, grasp, transfer, release
   |
   v
ArmController.send    <-- THE HARDWARE SEAM
```

Splitting detection from classification is the central choice. Stage 1 needs
zero labels on a fixed camera over a plain surface; stage 2 needs only crops,
because it never has to learn localisation. It also makes "I don't know" cheap:
a low score routes to `mixed` instead of guessing, which is what the pitch
promises.

## Swapping in hardware

Three replacements, nothing else:

| Simulation | Hardware |
|---|---|
| `PinholeTopDown` | `HomographyCalibration` from four ArUco markers |
| `ColorDetector` | `BackgroundDetector` + `OnnxCropClassifier` |
| `ArmController.send` writing `data.ctrl` | `lerobot` `SO101Follower.send_action` |

Everything above `send` — planning, bin assignment, the motion sequence, the
graspability gate — is already hardware-neutral and untouched by the swap.

## Motion sequence

Per item, and every step of it is there for a measured reason:

```
both arms -> STOW          fold vertically; clears the camera and the other arm
detect
acting arm -> HOME         transit pose; safe only because the other arm is stowed
move_to   (x, y, SAFE_Z)   Cartesian approach from above
move_to   (x, y, GRASP_Z - 4 mm)   aim low: position servos sag under load
close jaw, attach
move_to   (x, y, SAFE_Z)   lift straight up
move_joints -> above bin   JOINT space: a Cartesian path lets the solver flip
move_joints -> drop height   to a mirrored elbow halfway through
open jaw, release          jaws open BEFORE contacts are re-enabled
move_joints -> lift out    or the arm sweeps the item back out of the bin
-> STOW
```

Scoring reads the items' final positions. Release drops from where the tool
actually is; nothing is moved into a bin by fiat.

## Frames

One shared table frame, origin at the front arm's base, metres. `Pose2D` in
`geometry.py` converts into and out of each arm's own frame.

Only two quantities are base-relative and therefore converted: the shoulder
heading, and the wrist roll that cancels it. Jacobians and position error stay
in the shared frame, because that is the frame MuJoCo reports them in. Getting
this split right is what let the verified single-arm IK carry over unchanged to
a rotated second arm.
