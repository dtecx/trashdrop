# TrashDrop handoff for Claude — 2026-09-24

This records the preparation work across September 23–24 and the state after
the latest bottle and can capture. Read `AGENTS.md` first; it is the repository
contract. Read `CLAUDE.md` for Claude-specific notes. The repository has a large
uncommitted working tree containing the changes described below. Preserve it.

## Goal and scope

- Another team's robot delivers waste into a shared pick area. Two SO-101 arms
  sort by material into `plastic`, `paper`, and `metal`; uncertain or unsupported
  items go to the shared `mixed` fallback. Bottles, cups, and cans are examples,
  not the boundaries of the material classes.
- The team prioritizes zero wrong-bin decisions over throughput. The arms and
  real sorting hardware are not connected yet. Simulation cannot establish
  real perception or grasp reliability.
- The robot's physical pick zone in `trashdrop/station.py` is **20 × 15 cm**.
  The current **home data-capture zone** in `capture_zone.toml` is **45 × 45 cm**.
  These serve different purposes; do not silently equate them. Current
  `camera_zone.json` maps that capture zone to a 1920 × 1080 camera frame.

## What was built and fixed during this work session

### Camera and calibration

- Camera settings on macOS go through the UVC IOKit helper in
  `trashdrop/camera/uvc_iokit.c`, controlled by `camera.toml`. This avoids
  OpenCV/AVFoundation properties that report success but do not apply settings,
  and avoids libusb seizing the device. It works without root on the C920.
- The C920 was identified as OpenCV index 0 on this Mac. Camera commands can
  also auto-identify it. Current home settings: fixed focus 25, fixed white
  balance 4100 K, 50 Hz power frequency, minimum digital zoom, automatic
  exposure. macOS rewrites manual exposure/gain during streaming; foreground
  extraction compensates the resulting exposure drift instead.
- The printed ArUco sheet has four markers and a 10 cm ruler. The ruler must
  measure 100 mm on paper. `uv run trashdrop camera zone --camera 0` measures
  pixels per centimetre and centres the requested rectangle on the camera
  image. Re-run after changing `capture_zone.toml`, camera position, zoom, or
  resolution. `camera_zone.json` is a measured pixel result, not a live formula.
- Each new capture session snapshots the zone into
  `data/raw/<session>/zone.json`. Do not mix frames shot under different camera
  geometry in one nonempty session.

### Capture workflow

- `trashdrop capture` now has a readable full-screen UI, a cyan capture-zone
  outline, a live item box, and five recent photo thumbnails. It saves one
  image only when asked: `Q` or Space = shoot, `W` = undo the latest capture,
  `E` = next physical item, `1/2/3` = plastic/paper/metal, `R` = empty-table
  background, `T` = lighting tag, `F` = fullscreen, Esc = exit. A `mixed`
  capture key is not present because `mixed` is currently a routing fallback.
- Sessions can be resumed without overwriting earlier frames; frame numbering
  and the manifest continue. Undo removes the manifest entry, moves the raw
  frame to `_undone`, and removes derived labels/crops. Capturing a replacement
  does not require hand-editing object folders.
- New background photos are versioned when earlier frames use the old one.
  Existing frames keep their original reference. A class label is selected at
  capture time; `--category paper` merely sets the starting class in the UI.
- The UI may show a partial/incorrect live box. Do not rely on frame count
  alone: inspect the contact sheets and the full-size box overlays before
  training. See `docs/DATASET.md` for the capture and review commands.

### Class-agnostic detection and autolabel

- Live preview and offline autolabel use the same analysis width (320 px) and
  the same detector path. It compensates camera response on unchanged table
  pixels, subtracts the empty reference, suppresses broad cast shadows,
  recovers novel object edges, groups nearby fragments, and boxes their union.
  This is meant to handle crumpled paper, reflective cans, and clear packaging
  without frame-specific coordinate tweaks.
- A second substantial disconnected region is flagged as multiple objects.
  A box touching the calibrated zone boundary is flagged/rejected because
  pixels beyond that boundary cannot be recovered. The cyan rectangle marks
  the area used for detection; outside clutter is ignored.
- The detector still has a physical limit: nearly invisible clear plastic on
  a pale table may expose only a base, cap, or label. `plastic_03` (clear cup)
  had under-sized boxes in all seven captures; those seven frames were marked
  rejected, with original images retained. For a reshoot, change the light or
  use a matte contrasting surface, shoot a fresh `R` background, and shoot
  other materials on the same surface to avoid background/class leakage.
- An earlier stale background caused the first six `metal_01` frames to fail.
  Background versioning now prevents overwriting references used by prior
  captures. Two older paper frames touch the zone edge. Do not silently remove
  raw frames or relabel them as valid.

### Three-bin integration and public baseline

- The planner, station configuration, API, capture UI, and dataset classes now
  reflect three material destinations plus `mixed`. `mixed` catches uncertain
  material or unsupported size/mass. Two arms share the same pick volume and
  must use it serially; assignment is by material, not image position.
- Public-dataset exploration included a very small 18-image pilot against
  TrashNet and RealWaste. That pilot is superseded by the full `test_1` audit
  below. No production classifier was trained or deployed; any RealWaste
  numbers are an external-domain baseline only.

## Current local dataset: `test_1`

The session has **422 manifest frames from 18 physical object IDs**. The
latest capture added **191 frames**: `plastic_04` 40, `plastic_05` 48,
`plastic_06` 63, and `metal_08` 40. `uv run trashdrop autolabel --session
test_1` produced **407 labels** and rejected **15** frames: 7 manual rejects
(`plastic_03`) and 8 frames touching the capture boundary. Accepted crops:
metal 78, paper 81, plastic 248. Only **17 physical objects** have usable
labels because `plastic_03` is fully rejected. All 191 latest frames were
accepted.

Full-size box overlays for all latest frames were visually reviewed in:

- `out/test_1_plastic_04_boxes.jpg`
- `out/test_1_plastic_05_boxes.jpg`
- `out/test_1_plastic_06_boxes.jpg`
- `out/test_1_metal_08_boxes.jpg`

The new boxes generally enclose the whole item across pose changes; no
obvious systematic clipping like the clear cup was seen. This is a visual
review, not a guarantee that every pixel-level mask is perfect. Autolabel's
contact sheets are in `data/review/test_1/`. Raw captures, backgrounds, crops,
and `out/` are local/ignored assets and must not be committed.

## RealWaste check after the latest capture

The one-off script `out/realwaste_session_eval.py` uses an ImageNet-pretrained
MobileNetV3-small as a frozen feature extractor, then a balanced logistic
regression head trained on 4,752 RealWaste images. It evaluates the 407 local
autolabel crops. Two variants were tried: the native nine RealWaste classes
mapped to our four routes, and a direct four-route head. It uses host Python
with Torch/scikit-learn; it is intentionally outside the project dependencies.

| Variant | Correct material on all 407 local frames | New 191 frames |
|---|---:|---:|
| Nine RealWaste classes mapped to routes | 185/407 (45.5%) | 71/191 (37.2%) |
| Direct four-route head | 221/407 (54.3%) | 85/191 (44.5%) |

For the direct four-route head, the newly captured objects scored:

| Physical object | Correct / frames | Predictions as plastic / paper / metal / mixed |
|---|---:|---:|
| `plastic_04` | 4/40 | 4 / 18 / 6 / 12 |
| `plastic_05` | 25/48 | 25 / 8 / 3 / 12 |
| `plastic_06` | 34/63 | 34 / 26 / 0 / 3 |
| `metal_08` | 22/40 | 8 / 8 / 22 / 2 |

Across all accepted local frames, the direct head was correct on metal 56/78,
paper 57/81, and plastic 108/248. The held-out *public image* split gave
83.7% route accuracy, but it is not an object-held-out or local-domain score.
The local frames contain many near-duplicate poses from just 17 objects.

The direct head's maximum class probabilities are **uncalibrated**. Sending
low-score predictions to `mixed` still does not satisfy zero wrong-bin
sorting: at thresholds 0.5 / 0.7 / 0.9, the (correct material / wrong material
bin / mixed) counts were 162/72/173, 90/19/298, and 35/2/370 respectively.
This RealWaste-only model must not control the real arms. The result is evidence
of a substantial camera/background/domain gap, especially for plastics.

Detailed per-frame routes and scores: `out/test_1_realwaste.csv`; aggregate
summary: `out/test_1_realwaste.json`. Reproduce with:

```bash
uv run trashdrop autolabel --session test_1
uv run trashdrop review --session test_1
python3 out/realwaste_session_eval.py --session test_1
```

## Practical next steps for Claude

1. Run `git status --short` and preserve the uncommitted work. Read
   `AGENTS.md`, `CLAUDE.md`, `docs/DATASET.md`, and this handoff. Some prose in
   `docs/DATASET.md` and `docs/CAMERA.md` still calls the current capture zone
   20 × 15 cm; the live config is 45 × 45 cm. `CLAUDE.md` also has stale test
   count/simulation score examples. Correct these if editing documentation.
2. Prioritize more **distinct physical objects**, especially plastic and
   paper, over more views of the same bottle. Shoot across lighting/background
   setups with fresh empty references. Use the same setup across classes so
   the model cannot infer material from the background.
3. Develop and evaluate a local classifier using RealWaste plus our reviewed
   crops. Split local train/validation/test by `object_id` (never by frame),
   and reserve whole unseen objects for a final check. Calibrate the `mixed`
   decision on those unseen objects with **wrong-bin count** as the primary
   metric. Export a proven model to ONNX for `perception/classifier.py`; do not
   add Torch or Ultralytics to the project runtime dependencies.
4. Reshoot/test `plastic_03` on a contrasting, matte capture surface with
   improved side lighting. Keep its raw rejected captures excluded from
   training until full-object boxes are visually confirmed.
5. Before reporting code as working, run `uv run trashdrop probe` and
   `uv run python -m pytest tests/ -q`; for motion/geometry changes also run
   `uv run trashdrop sim` and quote the actual score. Hardware integration and
   real grasp performance remain untested.

No production code was changed specifically during the latest 191-frame
RealWaste evaluation; it updated local labels/review images and the ignored
`out/` reports. Nothing was committed in that evaluation step.

## Verification at handoff

- `git diff --check`: clean.
- `uv run trashdrop probe`: `LAYOUT OK`.
- `uv run python -m pytest tests/ -q`: **185 passed, 2 subtests passed**.

The first sandboxed verification attempt could not open a local HTTP socket or
the macOS CoreGraphics context required by MuJoCo. The same commands passed
when run with those host capabilities. The tests do not validate the RealWaste
classifier's usefulness on new physical objects; the object-held-out evaluation
described above is still needed.
