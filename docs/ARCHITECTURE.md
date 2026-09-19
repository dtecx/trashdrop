# Dual-arm station architecture

The virtual workcell uses one shared world frame and two namespaced copies of
the MuJoCo Menagerie SO-ARM100 model. SO-ARM100 is the current kinematic
stand-in for the SO-101 hardware; it is useful for reachability and sequencing
work, but it must not be treated as a torque, timing, or safety model of the
physical installation.

```text
top-down camera / detector
            |
     DetectedItem[]  (shared calibrated table frame)
            |
     TwoArmDispatcher
       |          |
 left queue    right queue
       |          |
 per-arm controller / future SO-101 adapter
       \          /
   shared-strip reservation and hardware safety supervisor
```

`trashdrop.scene_builder` does a structural MJCF composition step rather than
copying model files: every body, joint, actuator, mesh, material and default
class gets `left_` or `right_` namespace. That is what makes both arms load
together in one MuJoCo model without duplicate-name errors.

The dispatcher has three rules:

1. An object outside the centre strip belongs to the arm owning that side.
2. An object in the 5 cm centre strip is assigned to one arm only, balanced by
   queue length, and marked `requires_handoff_clearance`.
3. An unknown category fails closed: it produces no bin assignment.

The third rule is intentional for the hackathon. A physical executor must also
have independent e-stop, speed, joint-limit, collision, and human-presence
controls; none of those are delegated to simulation scheduling.
