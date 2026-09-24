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

## What macOS lets us set, and what it does not

Tested on the C920 by writing a value over USB while QuickTime streamed and
reading it back every half second:

| | who controls it on macOS | what the tune does |
|---|---|---|
| **focus** | us -- held | fixed in `camera.toml` |
| **white balance** | us -- held | measured on the sheet's paper, fixed |
| **exposure and gain** | **macOS** -- rewritten within 0.5 s | left automatic |

macOS runs its own auto-exposure for UVC cameras while a stream is open: it
keeps the camera in manual mode and writes exposure and gain itself. Values
we never wrote kept appearing (77 at gain 255, 312 at gain 222), and fighting
it produced every strange reading -- a tune that walked exposure down to 0.3 ms
while the picture stayed at median 142, a white QuickTime preview, "brightness
wandering" in camcheck.

It is not a problem worth solving. On a still scene macOS's exposure is stable
-- nothing moved in eight seconds of reading -- and the small step when an item
arrives is compensated by background subtraction. The real problem was
autofocus hunting over a plain board, and focus is ours.

A side effect: since exposure is automatic everywhere, **QuickTime shows what
the pipeline sees** and is a fine preview for framing and focus.

## Which camera is which

Commands find the webcam's video stream on their own (`--camera auto`, the
default): the webcam zooms for a moment, and the stream that zooms with it is
the one. An explicit `--camera N` is checked the same way and refused if it
does not follow the webcam.

This exists because it went wrong: on the MacBook the webcam was index 0 and
the built-in camera index 1, and a whole afternoon of tuning ran on the
laptop's camera while the settings went to the webcam. `uv run trashdrop
cameras` prints which index the webcam is and saves a snapshot of each.

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

**Light first.** Even with exposure left to macOS, more light means lower
gain and less noise on every crop. A desk lamp aimed at the board is enough.

Then lay the sheet flat in view, lights as they will be during the shoot:

```bash
uv run trashdrop camera tune --note "home rig, 70 cm"
```

About 30 seconds. It sets mains to 50 Hz (the camera shipped at 60 Hz, which
flickers under European LED light); lets autofocus find a first guess on the
star; finds the sheet by its markers; turns autofocus **off** and sweeps the
lens, judging sharpness on the star only; and bisects the colour temperature
until the paper is neutral. It prints the sweep as a bar chart — one hump is
what you want — and writes `camera.toml`, keeping the previous one as `.bak`.

With all four markers visible, tuning also reads the dimensions in
`capture_zone.toml` (45 x 45 cm on the home rig) and writes `camera_zone.json`.
The physical centre of this zone maps to the camera image centre. If camera
settings are already good, recalibrate only the zone with:

```bash
uv run trashdrop camera zone --camera 0
```

The 10 cm ruler verifies the sheet was printed at actual size; the four marker
centres supply the horizontal and vertical measurements. All four must be
visible. Capture copies the polygon into each new session and refuses to save
if the camera resolution no longer matches it.

**"marker sheet NOT found"** means the sheet is not wholly in view: check in
QuickTime that all four markers are in the picture.

## Check it

```bash
uv run trashdrop camcheck
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
