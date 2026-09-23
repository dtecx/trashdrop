# Locking the camera: camera.toml

`trashdrop camcheck` on the C920 said two things that matter:

- **`controllable: nothing`** — on macOS OpenCV reaches the camera through
  AVFoundation, which accepts every property and then ignores it. Focus,
  exposure and white balance cannot be set from OpenCV at all.
- **focus is hunting** — the C920's autofocus is contrast-driven, and a plain
  board gives it nothing to lock onto. The "view moved 3 px" reported alongside
  it is most likely the same thing: when the lens moves, the image scale changes
  slightly ("focus breathing"), which reads as drift. Lock the focus first, then
  re-check before blaming the mount.

So the settings go to the camera directly over USB, in UVC — the standard every
webcam speaks — and live in `camera.toml` at the project root, a file written
for people to read and edit.

## Why `sudo`

macOS lets any program *read* a webcam's USB descriptors, but only root may
*send it settings*, because macOS's own driver holds the device. Measured on
this project's MacBook: every control request as a normal user comes back
`Access denied`. The commands below only talk to the camera. Run them with
`.venv/bin/python -B` rather than `uv run`: `-B` stops Python leaving root-owned
cache files in the project, and files the tune writes are handed back to you.

## Setup, once

```bash
uv sync --extra rig                       # dataset tools + camera control
uv run trashdrop camera markers           # -> out/trashdrop_markers_A4.pdf
```

Print the PDF at **actual size / 100 %** — no "fit to page". Check the scale bar
with a ruler: it must be exactly 100 mm. The sheet is two things at once: the
Siemens star in the middle is the focus target for the tune, and the four
markers sit exactly on the corners of the 20 x 15 cm pick zone, so at the venue
the same sheet is the calibration.

## Tune

Lay the sheet flat in the middle of the board, top edge toward where the front
arm will be. Room lights as they will be during the shoot.

```bash
sudo .venv/bin/python -B -m trashdrop camera tune --camera 1 --note "home rig, 70 cm"
```

About 20 seconds. It lets auto-exposure and auto white balance settle on the
scene and freezes what they chose; then it turns autofocus **off**, sweeps the
lens across its whole range, and keeps the position where the star is sharpest.
It prints the sweep as a bar chart — you want one clear peak — and writes
`camera.toml`. The previous file is kept as a `.bak`.

If it says **no clear focus peak**, the sheet is not in view or not flat.

## Check it held

```bash
uv run trashdrop camcheck --camera 1
```

Expect `focus swing` near zero now. This is also the answer to an open
question: whether macOS resets the settings when a program opens the camera.

- **Focus swing near zero** — the settings survive. From now on: plug the camera
  in, run `camera apply` once with sudo, then shoot as normal.
- **Still hunting** — opening the camera resets them. Then run the capture
  itself under sudo, which pushes `camera.toml` right after opening the stream
  and hands the frames back to you afterwards:

  ```bash
  sudo .venv/bin/python -B -m trashdrop capture --session 2026-09-23-home --camera 1
  ```

## Every time the camera is plugged in

The camera forgets its settings when it loses power.

```bash
sudo .venv/bin/python -B -m trashdrop camera apply
```

## Editing by hand

Open `camera.toml`, change a number, run `camera apply`. Every value has a
comment saying what it is and the range the camera reported. Unknown names are
refused, so a typo cannot quietly leave autofocus on. Values outside the
camera's range are clamped and snapped to its step, and `apply` says so.

`camera show` (with sudo) lists every control the camera offers with its current
value — useful when something looks off.

## On site

The first thing after mounting the camera: lay the sheet in the pick zone and
tune again. Different height, light and table all change the answer. Commit the
new `camera.toml` so the venue settings are recorded next to the home ones.

## If sudo is not an option

Keep the marker sheet (or any printed page) in view at the edge of the board:
autofocus hunts because a plain board gives it nothing to hold, and texture in
view calms it considerably. Exposure is less of a worry — on this C920 it moved
by under one level when an item appeared, and background subtraction
compensates for exposure drift anyway.
