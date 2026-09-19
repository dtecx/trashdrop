# TrashDrop

TrashDrop is a local-first virtual workcell for sorting household waste with
two SO-101 arms. It separates the station model, scheduling logic, and future
hardware adapters so the hackathon prototype can grow without installing a
large robotics stack.

The initial scope is deliberately narrow: a MuJoCo scene, a deterministic
dispatcher, reachability-oriented station geometry, and a retained single-arm
vision/IK baseline. It does **not** install ROS, LeRobot, a training stack, or
a web dashboard.

## Quick start

The default project has no runtime dependencies. Install the simulator only
when you need it; `uv` keeps everything in this folder's `.venv`.

```bash
uv sync --extra simulation
./scripts/bootstrap_model.sh
uv run --extra simulation python -m trashdrop validate
```

That command generates `build/dual_arm_station.xml`, loads it in MuJoCo, and
checks that both independently named arms, all 12 actuators, cameras, bins,
and sample objects are present. To inspect the intended division of work
without importing MuJoCo:

```bash
uv run python -m trashdrop plan
```

For a live MuJoCo window on macOS:

```bash
uv run --extra simulation mjpython -m trashdrop viewer
```

The supplied `sim_sort.py` remains available as a one-arm vision/IK baseline.
It has a separate optional group because it additionally needs OpenCV:

```bash
uv sync --extra vision-baseline
uv run --extra vision-baseline python sim_sort.py
```

## Layout

```text
trashdrop/                Python package: station builder and dispatcher
  scene_builder.py         Namespaces the SO-ARM100 model twice into one MJCF
  planning.py              Deterministic, zone-safe two-arm job assignment
  station.py               Layout constants and hardware-neutral station data
  __main__.py              build / validate / viewer / plan commands
scripts/bootstrap_model.sh Sparse model checkout; no global install
tests/                     Fast dispatcher tests and optional scene checks
docs/                      Architecture, calibration and reference notes
sim_sort.py                Preserved single-arm vision + IK baseline
reference/                 Original supplied baseline copy
```

`mujoco_menagerie/` is supported as a legacy local model location because this
workspace already has one. New clones place the same sparse checkout under
`vendor/mujoco_menagerie/`; both locations are ignored by the project Git
repository. The source model is Apache-2.0 licensed by Google DeepMind.

## Design boundary for hardware

The simulation builds confidence in **layout and coordination**, not physical
safety certification. A future SO-101 adapter should consume
`ArmAssignment` values from `trashdrop.planning`, use per-arm calibration, and
enforce hardware e-stops, joint limits, speed limits, and a shared exclusion
zone independently of this demo.

The supplied `docs/SETUP.md` is retained as a single-arm setup reference. It
is not the specification for this dual-arm workspace.

## Dataset intake

For the common TrashNet/"TrashDataset" class-folder layout, use the built-in
manifest generator. It never downloads, copies, or adds images to Git:

```bash
uv run python -m trashdrop dataset-index /path/to/dataset-resized
```

See [dataset integration notes](docs/DATASET.md) for the mapping and the
important limitation: TrashNet does not include bio-waste or multi-object
detection labels.
