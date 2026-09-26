# Grasping

The simulator's grasp is kinematic: the item is attached to the jaw when it
closes nearby. It proves the arm gets there, not that the gripper holds
anything. What follows is what is known about the real SO-101 jaw, how a pick
is meant to work, and what has to be measured on day one.

## The jaw, from CAD

Measured on the finger surfaces of the official SO-101 MuJoCo model
(TheRobotStudio/SO-ARM100, `Simulation/SO101`); the SO-ARM100 model in this
repository agrees within a few millimetres. The numbers live in
`trashdrop/station.py`.

| item width | 27 mm (bottle neck) | 32 mm | 46 mm | 60 mm | 66 mm (0.5 L can) | 74 mm |
|---|---|---|---|---|---|---|
| angle between the fingers | −5° | 0° | ~5° | ~15° | ~28° | ~38° |
| friction needed to hold | – | – | 0.05 | 0.13 | 0.25 | 0.34 |

- The moving finger swings on a hinge, so the fingers are parallel at one
  width only. Anything wider sits in a V that squeezes it out of the jaw.
  Bare PLA on aluminium is about 0.3, so a can is marginal without grip tape.
- **The jaw is not symmetric.** One finger belongs to the wrist and never
  moves; the TCP sits on its inner face. The fixed finger has to come down
  *beside* the item and the moving finger sweeps the item onto it. Aiming the
  TCP at the item's centre lands the fixed finger on top of it.
- With the jaw wide open the moving fingertip is several centimetres higher
  than the fixed one and swings ~10 cm out to the side. Open only as far as
  the item needs.

## One pick

1. The detector's mask goes to `perception.grasp.plan_grasp`, which returns a
   **pinch** (the narrowest place with parallel sides, preferring the middle:
   a bottle by its neck or cap, a can across its body) or, for an item wider
   than the jaw everywhere, an **edge** plan (fixed finger beside a long
   edge, moving finger landing on the item -- only a sheet can be taken like
   that).
2. Open the jaw to `plan.opening_m`, not fully.
3. Lower the fixed finger to `plan.fixed_finger(...)`, fingertips about
   13 mm above the table (`sorter.GRASP_HEIGHT_CM`), which covers the middle
   of a 3 cm handle and of a can or bottle lying down. "The table" is the
   plane through the corners that arm touched: the venue's arms read the flat
   table up to 3 cm lower at full reach than beside their bases, so one
   height for the whole zone either hits the table near the base or closes
   above the item far out. The pick prints where the fingertips really
   ended up (raising the descent by the sag seen at the hover was tried:
   the grasp barely sags, and it left them 5 mm high). The jaw is
   turned to close across the item with each arm's own wrist roll zero
   (`rig roll`): at the venue the left arm's was a quarter turn from the
   model's, and its first grasp closed along a screwdriver, beside it.
4. Close with a torque limit and read the gripper position. Closed all the way
   means nothing is between the fingers: a miss.
5. Lift 2 cm and look again. If the item is still in the pick zone, try the
   next candidate. After two misses, hand it to the other arm, whose gripper
   may suit it better; that arm cannot reach this material's bin, so the item
   goes to mixed. Failing that, leave it for a human. Nothing is carried over
   a material bin unless the checks passed.

Things the planner cannot know from one overhead camera:

- **Flat or not.** A flattened carton gets a pinch plan that cannot work. For
  paper, try the pinch; if it misses, try the edge.
- **Standing items.** A standing can shows its lid, its side and its shadow,
  so its silhouette is not its footprint, and its top is above the height the
  vertical gripper reaches (`MAX_VERTICAL_Z`). Knock it over first.
- **Clear plastic.** Where the mask has a gap, the jaw is never placed there;
  a very fragmented item is planned on its convex hull.

## Day one

- Measure the real opening at five gripper commands with a ruler and compare
  with the table above.
- Put grip tape (3M gripping tape, or any anti-slip tape) on both fingers'
  inner faces.
- With the leader arm, pick each kind of item by hand: can lying down, bottle
  by the cap, crumpled paper, flat cardboard, paper cup, standing can. What
  fails by hand will fail autonomously.
- Find the fingertip height above the table that clears it without missing
  thin items.

## Two different grippers

The 3D printers on site make a second end effector possible. Material
ownership already splits the work by shape: the front arm takes paper
(sheets, cardboard, crumpled paper, cups), the back arm takes plastic and
metal (bottles, cans). Candidates:

- **Grip tape on the stock jaw** -- both arms, whatever else happens.
- **Compliant TPU jaw** (official, for SO-101): `Optional/Compliant_Gripper`
  in TheRobotStudio/SO-ARM100, or the newer TPU finger on a PLA base
  (`hardware/SO101_soft_fin.stl` in Vector-Wangel/XLeRobot, two extra M3
  screws). Conforms to crushed and irregular items. Needs TPU 95A and a
  printer that handles it.
- **Parallel gripper** (roboninecom/SO-ARM100-101-Parallel-Gripper): 84 mm
  stroke with parallel fingers, so no V at all. Needs two MF106ZZ bearings
  and two 6 mm tubes, 125 mm long, bought in advance.
- **A thin fingernail on the fixed finger** of the paper arm: a ~1 mm
  chamfered plate that sits flat on the table, so the moving finger can push
  a sheet onto it.
- **Suction** is what industrial waste sorters use, and it handles flat items
  and can lids. It needs a pump, a valve and a driver; worth it only if
  someone brings the parts.

Whatever is fitted, give each arm its own `max_width_m` when calling
`plan_grasp`.
