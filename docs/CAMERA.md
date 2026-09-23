# Locking the camera: camera.toml

`trashdrop camcheck` on the C920 said two things that matter:

- **OpenCV `controllable: nothing`** — on macOS OpenCV reaches the camera
  through AVFoundation, which accepts every property and then ignores it.
  Focus, exposure and white balance cannot be set from OpenCV at all. That line
  in camcheck stays "nothing" on a Mac, and that is expected.
- **focus is hunting** — the C920's autofocus is contrast-driven, and a plain
  board gives it nothing to lock onto. The "view moved 3 px" reported alongside
  it is most likely the same thing: when the lens moves the image scale shifts
  slightly ("focus breathing"), which reads as drift.

So settings go to the camera directly in UVC — the USB standard every webcam
speaks — and live in `camera.toml` at the project root, written for people to
read and edit.

## How it reaches the camera on macOS

Through a small IOKit helper (`trashdrop/camera/uvc_iokit.c`), compiled
automatically the first time it is needed. It needs the Xcode Command Line
Tools and nothing else — no Python packages, no Homebrew, **no sudo**.

Verified on this project's MacBook with the C920, as a normal user: all
fourteen controls read, writes accepted and read back from a separate process.

For the record, the route that does *not* work: libusb. It opens the device in
"seize" mode, and current macOS refuses that even to root because its own
camera driver is attached. That driver only claims the video *interfaces*,
not the device, so a plain IOKit open succeeds. The approach is the one
[camtint](https://github.com/bornaware/camtint) documents working on macOS 26.

## Setup, once

```bash
uv sync --extra rig
uv run trashdrop camera probe             # "read/write: ok" -- control works
uv run trashdrop camera markers           # -> out/trashdrop_markers_A4.pdf
```

Print the PDF at **actual size / 100 %** — no "fit to page". Check the scale bar
with a ruler: it must be exactly 100 mm. The Siemens star in the middle is the
focus target; the four markers sit exactly on the corners of the 20 x 15 cm
pick zone, so at the venue the same sheet is the calibration.

## Tune

**Light first.** On the first reading the camera was running at its longest
exposure for 30 fps (33 ms) with gain 159 of 255 — it was starved of light and
amplifying noise. More light on the board before tuning means a cleaner freeze.

Then lay the sheet flat in the middle of the board, lights as they will be
during the shoot:

```bash
uv run trashdrop camera tune --camera 1 --note "home rig, 70 cm"
```

About 20 seconds. It sets mains to 50 Hz (the camera shipped at 60 Hz, which
flickers under European LED light), lets auto-exposure and auto white balance
settle and freezes what they chose, then turns autofocus **off**, sweeps the
lens across its whole range and keeps the position where the star is sharpest.
It prints the sweep as a bar chart — you want one clear peak — and writes
`camera.toml`. The previous file is kept as a `.bak`.

**No clear focus peak** means the sheet is not in view or not flat.

## Check it

```bash
uv run trashdrop camcheck --camera 1
```

`focus swing` should now be near zero, and the first line should read
`camera: camera.toml applied`.

## Day to day

Nothing to do. `capture` and `camcheck` push `camera.toml` every time they open
the camera, so a replugged camera — which forgets its settings when it loses
power — is set again automatically. To push it by hand:

```bash
uv run trashdrop camera apply
```

## Editing by hand

Open `camera.toml`, change a number, run `camera apply`. Every value carries a
comment saying what it is and the range the camera reported. Unknown names are
refused, so a typo cannot quietly leave autofocus on. Values outside the range
are clamped and snapped to the camera's step (a C920 moves focus in steps of
5), and `apply` says so. `camera show` lists every control with its current
value.

## On site

The first thing after mounting the camera: lay the sheet in the pick zone and
tune again. Different height, light and table all change the answer. Commit the
new `camera.toml` so the venue settings are recorded next to the home ones.

## If camera control ever fails

`camera probe` says so. Unplug and replug the camera first. If it is still
blocked, the pipeline still works — keep a printed page in view at the edge of
the board so autofocus has texture to hold, and rely on the exposure
compensation in background subtraction.
