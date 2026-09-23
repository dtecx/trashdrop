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
3. **Shoot the empty table** — press `b` in the capture window. One reference
   per lighting condition. A stale background is the main failure mode of the
   whole pipeline.
4. Shoot at **the same resolution and field of view** that inference will use.

## The loop

```bash
uv sync --extra dataset
uv run trashdrop cameras                       # which index is the webcam?
uv run trashdrop camcheck --camera 1           # 60 s verdict
uv run trashdrop capture --session 2026-09-23-home --camera 1
```

Run these from your own Terminal, not through an agent: the capture window needs
the keyboard, and macOS grants camera access per application.

1. **Clear the table and press `b`** — the empty-table reference.
2. **Press `1`–`5` for the class**: 1 bio, 2 paper, 3 plastic, 4 metal,
   5 mixed. The object gets the next free id (`plastic_01`, `plastic_02`, ...)
   so nobody types names.
3. **Drop the item in the zone and step back.** The auto-shutter takes the
   frame by itself once your hand is out and nothing has moved for about half
   a second — the preview flashes white.
4. **Reposition** — rotate, flip, crush a bit — **step back**, wait for the
   flash. About **12 poses per item**.
5. **`n` for the next item** of the same class, or a digit for another class.

The status line says why it is not shooting: `moving`, `hand or item at the
frame edge`, `more than one object`, `empty table`. A box around the item turns
green when it is ready. `SPACE` still shoots a burst of three by hand, and `a`
turns the auto-shutter off.

One frame per pose, deliberately. Frames of a pose that has not changed are
near-duplicates and add nothing; the shutter refuses them and asks for a new
pose instead.

## What actually matters

**Transparent bottles work.** Background subtraction only sees the cap, the
label and a few highlights of a clear bottle, as separate blobs. Those are
grouped back into one item by proximity, so do not avoid clear PET — it is
exactly what the pitch promises to handle.

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
uv run trashdrop review   --session 2026-09-20-kitchen --drop bad_object_03
```

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

Both map their own taxonomy onto our five bins and **fail closed to `mixed`**.
Two mappings are deliberate rather than lazy: TACO's food *containers* are not
bio-waste, and TrashNet's `glass` goes to `mixed` because a glass bottle is
heavier than this arm should lift. Note also that TrashNet has no bio class at
all — that one can only come from our own captures.

Check the licence of each before publishing our own dataset.
