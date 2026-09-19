# CLAUDE.md

**Read [AGENTS.md](AGENTS.md) first** — it holds the architecture, the module
layers, and the seven invariants that must not break. This file adds only what
is specific to working here with Claude Code.

## Quick orientation

Two SO-101 arms facing each other across a shared pick zone, sorting waste into
five bins (`bio`, `paper`, `plastic`, `metal`, `mixed`). Hackathon is around
2026-09-25; the team's stated success criterion is **zero sorting errors**.
This repository is the virtual model plus dataset tooling. No hardware yet.

## Running things

Everything goes through `uv`; there is no global install and nothing should
ever be added to system Python.

```bash
uv run trashdrop probe            # layout reachability check, ~2 s
uv run trashdrop sim              # full two-arm sort, ~7 s, writes out/
uv run python -m pytest tests/ -q # 50 tests, ~7 s
```

The full simulation takes seven seconds. **Run it** before reporting anything
about motion, geometry or grasping, and quote the real
`sorted correctly: N/4` line rather than describing the change.

`uv run trashdrop sim --viewer` needs `mjpython` on macOS:
`uv run mjpython -m trashdrop sim --viewer`. Plain `python` fails with a main
thread error. Headless runs need no such thing, and `MUJOCO_GL` must not be set
on macOS.

## Where to change what

Geometry — mounts, bins, pick zone, poses, payload limits — lives only in
`trashdrop/station.py`. Edit the numbers there and then run
`uv run trashdrop probe`; it re-solves IK to every bin and pick-zone corner and
names whatever stopped being reachable. Do not scatter constants into the
modules that consume them.

## Things that have already gone wrong here

- **Two sessions editing this folder at once.** In the session that built this,
  a parallel agent wrote a whole package into the same directory and the two
  overwrote each other's files. Before starting, run `git status` and
  `git log --oneline -5`; if there is unexpected work, stop and ask rather than
  overwriting it.
- **A degrees/radians mix-up** put the second arm beside the first instead of
  facing it, and everything still compiled and ran. See invariant 1.
- **A self-fulfilling score.** An earlier implementation teleported each item
  into its bin on release and then reported 5/5. If a metric cannot fail, it is
  not a metric. See invariant 4.

## Reporting results

The colour detector in `perception/color.py` is told the answer by
construction. When the simulation sorts 4/4, that is a result about *motion*,
not about perception, and saying otherwise overstates what exists. The grasp is
kinematic too, so nothing here predicts whether the real gripper holds a
crushed can.

## Language

The team speaks Russian; chat in Russian if the user does. **All code,
comments, docstrings and commit messages stay in English.**
