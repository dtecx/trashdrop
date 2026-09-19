# Hardware hand-off checklist

Before connecting either SO-101, replace the simulated constants with measured
values. This keeps the simulation useful without letting it become a source of
false confidence.

1. Establish a shared world frame on the tabletop, then measure both arm-base
   transforms into that frame.
2. Calibrate the top-down camera to the same frame using fixed visual markers.
   Record reprojection and pick-position error at the left zone, shared strip,
   and right zone.
3. Measure each arm's reachable, collision-free envelope at the planned pick
   and bin heights. Keep bins and human-access areas outside those envelopes.
4. Implement an explicit reservation protocol for the yellow shared strip.
   Never have two independent motion loops infer that the other arm is clear.
5. Start at reduced speed with empty grippers, independent e-stops, a physical
   barrier for observers, and a reject bin for low-confidence classifications.

The `plan` command is safe to use without MuJoCo. `validate` only confirms
that the virtual MJCF is structurally loadable; it is not a hardware approval.
