> **Historical.** This is the original single-arm brief the project grew
> from, kept for provenance. Several of its constraints no longer hold:
> the cell now has two arms, and `lerobot` is available behind
> `uv sync --extra teleop` (see [APPROACH.md](APPROACH.md)). The verified
> script it describes is preserved unchanged at `reference/sim_sort.py`.
> For the current setup, read the README and [AGENTS.md](../AGENTS.md).

# Task: local SO-101 trash-sorting sandbox

Set up a self-contained MuJoCo sandbox in this folder so `sim_sort.py` runs. The script
simulates an SO-101 arm sorting coloured objects into four bins using a top-down camera,
OpenCV segmentation, a pixel-to-world mapping and a damped-least-squares IK solver.

`sim_sort.py` is already written and verified. Do not rewrite its logic. Your job is the
environment plus the robot model, then a verification run.

## Hard constraints

- Nothing installed globally. No Homebrew packages, no `pip install` into system Python,
  no conda, no ROS.
- All Python dependencies live in a project-local `.venv` managed by `uv`.
- Deleting this folder must remove everything except the `uv` binary itself.
- Do not install `lerobot`, `placo`, `pinocchio` or ROS. They are not needed at this
  stage and `placo` drags in a heavy toolchain.
- Do not install ffmpeg. The `opencv-python` wheels already bundle an encoder.
- Use whatever Python the machine has, as long as it is 3.10 or newer. `mujoco` ships
  macOS arm64 wheels up to cp315, `numpy` up to cp314, and `opencv-python` is an abi3
  wheel, so 3.14 is fine. If any wheel turns out to be missing, do not downgrade the
  system Python — run `uv python pin 3.13`, which fetches a private interpreter into
  the project only.

## Steps

1. **uv.** If `uv --version` fails, install it with the standalone installer
   (`curl -LsSf https://astral.sh/uv/install.sh | sh`), which drops a single binary in
   `~/.local/bin`. Do not install it through Homebrew.

2. **Project.** In this folder: `uv init --bare` if there is no `pyproject.toml` yet.
   Then `uv add mujoco opencv-python numpy`. Nothing else.

3. **Robot model.** The arm is `trs_so_arm100` from Google DeepMind's `mujoco_menagerie`
   (SO-ARM100, kinematically the same arm as the SO-101). The full repository is several
   hundred megabytes, so take only that folder:

   ```
   git clone --depth 1 --filter=blob:none --sparse \
     https://github.com/google-deepmind/mujoco_menagerie.git
   cd mujoco_menagerie && git sparse-checkout set trs_so_arm100
   ```

   Verify `mujoco_menagerie/trs_so_arm100/so_arm100.xml` and `assets/*.stl` exist. If the
   git version is too old for `--sparse`, fall back to a normal `--depth 1` clone and
   delete the other robot folders.

4. **Verify.** Run `uv run python sim_sort.py`. Expected on stdout:

   - five lines reading `cycle N: K object(s) on the surface`, counting down 4, 3, 2, 1, 0
   - each cycle naming a class, its measured world coordinates and the target bin
   - a final `sorted correctly: 4/4  (picks 4, misses 0)`
   - `sort_demo.mp4` and `topdown_detection.png` written to the folder

   Anything less than 4/4 is a failure. Report the actual output, do not paper over it.

5. **Report.** Print the tree of what was created, the total size of `.venv` plus
   `mujoco_menagerie`, and the exact command for the live viewer:
   `uv run mjpython sim_sort.py --viewer`.

## Notes that will save you time

- On macOS the interactive viewer only works through `mjpython`, a launcher that the
  `mujoco` wheel installs into the venv. Plain `python sim_sort.py --viewer` fails with a
  message about the main thread. Headless runs need no such thing.
- Do not set `MUJOCO_GL` on macOS. The script sets it to `egl` on Linux only; on macOS
  the default backend is correct and forcing a value breaks rendering.
- The script writes a generated scene file to
  `mujoco_menagerie/trs_so_arm100/_sorting_scene.xml`. That is deliberate: MuJoCo
  resolves `<include>` and `meshdir` relative to the top-level model file, so the scene
  has to sit next to the arm model. Leave it there and do not commit it.
- `sim_sort.py --probe` prints a reachability map instead of running the demo. Useful
  later for deciding where the bins and the pick area go on the real table.
- `sim_sort.py --trace` prints every object's pose after each cycle. Use it if a pick or
  a drop misbehaves.

## Out of scope

Do not add a training pipeline, a dataset downloader, YOLO, a web UI or a Docker file.
Do not touch anything outside this folder.
