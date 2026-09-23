# Shooting the dataset

The trick that removes annotation from the critical path: **shoot one item at a
time against a reference photo of the empty table.** Background subtraction
then recovers the mask and box for free, and the class comes from the folder
name. Nobody draws a single box.

Budget: five people, one evening. Target 300 crops per class, 150 is the
working minimum.

## Before the first photo

0. **Grant camera permission, the day before.** On macOS the first call to the
   camera fails with `not authorized to capture video` until the terminal is
   allowed in System Settings → Privacy & Security → Camera. Find that out now,
   not on the morning of the shoot.

1. **Check the camera is worth shooting through.** Sixty seconds:

   ```bash
   uv run trashdrop camcheck
   ```

   It measures the four things that break background subtraction — exposure
   wandering on a still scene, focus hunting, the view drifting, and which
   camera properties can actually be set — and prints a verdict with the fix
   for anything it fails. Run it again whenever the camera or the lighting
   moves.

   One warning worth knowing in advance: if it reports **the view moved** while
   nothing was happening, suspect "auto-framing" or "smart zoom" before you
   suspect the clamp. Those features digitally pan the image to follow a
   subject, which voids the calibration and is fixed in the camera's software,
   not on the table.

2. **Fix the camera and do not touch it again.** Tape it down. If it moves, the
   calibration is void and every frame shot before the move belongs to a
   different geometry. Re-shoot the homography if it happens.
3. **Calibrate the 20 x 15 cm pick area.** Print the sheet from
   `uv run trashdrop camera markers` at actual size. Verify its ruler measures
   10 cm. Place the sheet flat with all four ArUco markers in view, then run
   `uv run trashdrop camera tune` (settings and zone) or
   `uv run trashdrop camera zone` (zone only). `capture_zone.toml` holds the
   physical dimensions and `camera_zone.json` holds the measured camera pixels.
   The projected polygon is centred on the **camera image centre**, including
   when the printed sheet is slightly off-centre. Remove the sheet before shooting.
   A new session copies the calibration to `data/raw/<session>/zone.json` so
   autolabel uses precisely the same area. Recalibrate if the camera moves.
   Reopening an empty session refreshes its copy after recalibration. Once the
   session contains photos, start a new session for a changed zone so old and
   new photos are not labelled against different areas.
4. **Shoot the empty table** — press `R` in the capture window. One reference
   per lighting condition. A stale background is the main failure mode of the
   whole pipeline.
5. Shoot at **the same resolution and field of view** that inference will use.
   Start a new session if the camera or the pick area moves.

## The loop

```bash
uv sync --extra rig
uv run trashdrop camcheck                     # 60 s verdict; webcam is found automatically
uv run trashdrop capture --session 2026-09-23-home
```

Run these from your own Terminal, not through an agent: the capture window needs
the keyboard, and macOS grants camera access per application.

The window starts full-screen. The large cyan polygon is the measured pick area;
everything outside it is dimmed and ignored by autolabel. The bottom filmstrip
shows the last five captures, cropped to that area, with labels. The full
camera frames are saved to disk. The green box covers the detected item; a red
box means it touches the zone edge and should be moved inward.

1. Clear the pick area and press **`R`** to store the empty background.
2. Press **`1`–`3`** to select the material: 1 plastic, 2 paper, 3 metal.
   Bottles, cups, and cans are examples; capture other waste of the same
   materials too. Changing class starts a fresh object ID. `mixed` is the
   fallback route for uncertain or unsupported items, not a capture key.
3. Place one item in the cyan area and press **`Q`** or **Space**. Each press
   saves exactly one frame; nothing is saved automatically. Rotate or move the
   item, then press again for a useful new pose.
4. Press **`W`** if the last frame was wrong. It disappears from the manifest
   and is moved to `_undone`, and any derived crop or label is removed.
5. Press **`E`** for the next physical item of the same class. Use a digit to
   switch class. Press **`T`** to cycle through `default`, `daylight`, and
   `side_light`; after changing the real lighting, clear the zone and press
   **`R`** again. `R` creates a new numbered background when photos already
   use the previous one, so an older reference is never overwritten. Press
   **`F`** to toggle fullscreen and **Esc** to finish.

The class and object ID appear at the top of the window. Check them before
shooting each new item. About 12 varied poses per physical item is useful;
pressing repeatedly without moving it only creates near-duplicates.

The box and autolabel compare each frame to its empty-table reference. They
remove smooth cast shadows, then recover new object contours so a pale paper
fold, metal can side, or clear bottle wall is included. A hard contact shadow
can still remain beside a dark item. Use diffused light or a second lamp if it
dominates the crop. After changing the lighting, press **`R`** with the zone
empty; `T` can name a distinct lighting setup if useful. Each photo keeps the
reference version that was active when it was saved. Review crops before
training.

The same detector runs in the live preview and autolabel at a fixed analysis
width. It first compensates camera exposure using unchanged table pixels, then
finds reliable foreground and removes smooth cast shadows. New edges relative
to the empty table restore pale paper, reflective metal, and transparent
plastic outlines that the strong foreground threshold misses. Nearby pieces
are joined into one item, and the box encloses the joined outline. A box
touching the calibrated zone is shown red and rejected by autolabel: pixels
outside the zone cannot be reconstructed. There are no coordinates tuned to a
particular session or object ID.

## What actually matters

**Transparent objects need a visible outline.** A bottle's cap, label and
highlights often provide fragments that can be grouped into one box. An almost
colourless cup on a pale table may leave only its reflective base: no
background-subtraction threshold can recover an invisible wall reliably. If
the green box covers only the base, do not train on that crop. Change the light
or use a matte contrasting pick surface, clear the zone and press **`R`** to
save a new reference. The camera and zone can stay fixed. Shoot other material
classes on the same surface too, so the classifier does not learn a shortcut
from background colour to class. Review the new crops before training.

**Variety of objects beats number of frames.** Fifteen to twenty-five *different
physical items* per class is worth far more than two hundred photos of one
bottle. The split holds out whole objects, so a class with four objects cannot
produce a meaningful test number no matter how many frames it has.

Shoot each item in several states — this is exactly what the pitch promises to
handle and what public datasets do not contain:

- intact / crushed / partly crushed
- cap on / cap off, label on / label torn off
- dry / wet / greasy
- empty / with residue

Lighting: at least three conditions, with a background reference for each.
Overhead room light, room light plus daylight from a window, and one harsh
raking light (a phone torch at a low angle) to generate the shadows that will
appear on the day.

Also shoot:

- **Multi-object frames** with overlap — 100 to 200. Autolabel refuses these on
  purpose; they are for a detector later and need manual boxes.
- **Negatives** — empty table, hands, the gripper, the other team's robot,
  shadows, the bins themselves.
- **A held-out test set** on a different day, with items nobody shot for
  training. Without it the accuracy number means nothing.

## Then

```bash
uv run trashdrop autolabel --session 2026-09-20-kitchen
uv run trashdrop review   --session 2026-09-20-kitchen   # contact sheets
uv run trashdrop review   --session 2026-09-20-kitchen --reject-frame paper/paper_01/frame_0000.jpg
uv run trashdrop review   --session 2026-09-20-kitchen --drop bad_object_03
```

`--reject-frame` records an accidental or wrongly classified capture in
`data/raw/<session>/rejected_frames.txt`. The raw photo stays untouched, but
its crop and label are removed and future `autolabel` runs skip it. Use this
when one bad frame appears in an otherwise useful object burst.

`autolabel` refuses frames it cannot label confidently — more than one large
region, a region touching the frame edge, or nothing changed. Read the
`reasons` block it prints, and the `exposure_drifted_frames` count next to it. A high `touches_frame_edge` count usually means the
items are landing too close to the edge of the view; `more_than_one_object`
usually means a hand stayed in shot.

### About auto-exposure

You do not have to defeat it. Put a dark item on a light table and most cameras
brighten the whole frame; a naive difference then marks everything as
foreground and the item disappears into it. Locking exposure is the textbook
answer and often impossible — OpenCV on macOS talks to AVFoundation, which
ignores the exposure property on most cameras.

So the reference is pushed through the camera's current response before
differencing, and the item survives regardless. `exposure_drifted_frames` in
the autolabel report counts how often that correction was needed: a high number
is not a failure, it just means re-shooting the background more often would
give cleaner masks.

The one case it cannot save is an item covering most of the view — there is
then too little table left to measure the drift against. Those frames are
refused as `exposure_fit_failed`.

Then look at the contact sheets. A handful of wrong crops in a class of two
hundred is enough to confuse a small classifier, and one bad burst is one
`--drop` away.

## Layout on disk

```
data/
  raw/<session>/<class>/<object_id>/frame_0000.jpg   originals, never modified
  bg/<session>/<lighting>.jpg                        empty-table references
  crops/<session>/<class>/<object_id>/...            what you train on
  labels/<session>.jsonl                             box, mask, split, lighting
  raw/<session>/manifest.csv                         one row per frame
```

Nothing under `data/` is committed — the frames are large and belong in shared
storage. The layout above is what the tooling creates for you.

## Public datasets

Use them to pretrain and to sanity-check a class, never as a substitute. Every
one was shot through a different camera at a different angle; the domain gap
eats most of the benefit. Realistically they get you to roughly 60–70 % on our
table, and a few hundred of our own crops take it past 90 %.

| Dataset | Size (approx.) | Why | Role here |
|---|---|---|---|
| [TACO](http://tacodataset.org/) | ~1.5k images, 60 categories, COCO + masks | Real litter: crushed, dirty, transparent | Pretrain a detector |
| [TrashNet](https://github.com/garythung/trashnet) | ~2.5k images, 6 classes | One item on a plain background — the same geometry as our crops | Pretrain the crop classifier |
| Drinking Waste Classification (Kaggle) | ~4.8k images, 4 classes, YOLO boxes | Cans and bottles dominate hackathon trash | Bulk data for two classes |
| [ZeroWaste](http://ai.bu.edu/zerowaste/) | ~4.5k frames, segmentation | Conveyor clutter and occlusion | Only if a detector for piles is needed |

Skip TrashCan (underwater) and UAVVaste (aerial) — wrong domain entirely.

Index them in place; nothing is copied or modified:

```bash
uv run trashdrop dataset-index /path/to/trashnet/dataset-resized
uv run trashdrop taco-index --annotations /path/to/annotations.json --images /path/to/images
```

Both map their own taxonomy onto three material bins plus `mixed` and **fail
closed to `mixed`**. Organic waste is outside the three target materials, and
TrashNet's `glass` goes to `mixed` because a glass bottle is heavier than this
arm should lift.

Check the licence of each before publishing our own dataset.
