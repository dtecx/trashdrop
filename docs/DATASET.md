# Shooting the dataset

The trick that removes annotation from the critical path: **shoot one item at a
time against a reference photo of the empty table.** Background subtraction
then recovers the mask and box for free, and the class comes from the folder
name. Nobody draws a single box.

Budget: five people, one evening. Target 300 crops per class, 150 is the
working minimum.

## Before the first photo

1. **Fix the camera and do not touch it again.** Tape it down. If it moves, the
   calibration is void and every frame shot before the move belongs to a
   different geometry. Re-shoot the homography if it happens.
2. **Shoot the empty table** — press `b` in the capture window. One reference
   per lighting condition. A stale background is the main failure mode of the
   whole pipeline.
3. Shoot at **the same resolution and field of view** that inference will use.

## The loop

```bash
uv sync --extra dataset
uv run trashdrop capture --session 2026-09-20-kitchen \
    --category plastic --object-id cola_bottle_01
```

Keys: `SPACE` burst · `b` background · `n` next object · `c` class ·
`l` lighting · `q` quit.

For each physical item: **drop it** on the table — do not place it neatly —
then take a burst while nudging it between frames. About 25 frames per pose,
two or three poses per item.

## What actually matters

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
`reasons` block it prints. A high `touches_frame_edge` count usually means the
items are landing too close to the edge of the view; `more_than_one_object`
usually means a hand stayed in shot.

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
