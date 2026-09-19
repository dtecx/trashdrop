# Integrating TrashNet safely

This project treats a data source and a waste-routing policy as different
things. The commonly referenced TrashNet dataset is a small **classification**
dataset: it has six folders (`cardboard`, `glass`, `metal`, `paper`, `plastic`,
and `trash`) and images of a single item on a controlled background. It does
not provide object locations for a cluttered workcell and does not have a bio
class. It is therefore a good starting point for crop classification, but not
a complete perception solution for TrashDrop.

## Keep the data outside Git

Download or unzip the dataset wherever you keep large data, for example:

```text
/Volumes/SSD/datasets/trashnet/dataset-resized/
  cardboard/  glass/  metal/  paper/  plastic/  trash/
```

Build an immutable manifest without copying the images or installing ML tools:

```bash
uv run python -m trashdrop dataset-index \
  /Volumes/SSD/datasets/trashnet/dataset-resized \
  --manifest build/trashnet_manifest.jsonl
```

The command checks every expected class folder is present, assigns a stable
70/15/15 train/validation/test split from the filename hash, and records the
absolute image path, source class, station route, and split in JSONL.

## Routing contract

| TrashNet label | TrashDrop result |
| --- | --- |
| `cardboard`, `paper` | paper bin |
| `metal` | metal bin |
| `plastic` | plastic bin |
| `glass`, `trash` | reject lane / human decision |
| bio-waste | **not covered by TrashNet** |

Do not silently map `glass` or generic `trash` to a recycling bin. Add a
physical reject tray and train an additional bio/glass dataset before enabling
those routes.

## Recommended hackathon perception path

1. Use the OAK-D to locate an item on the table and crop it. TrashNet alone is
   unsuitable for this multi-object detection step.
2. Train or fine-tune a small image classifier outside this core repository,
   using the manifest's `train` and `val` rows. Export a portable runtime model
   such as ONNX or TFLite to an ignored `models/` directory.
3. At runtime, require a calibrated crop, a confidence threshold, and the
   mapping above. Anything low-confidence, glass, generic trash, or bio goes
   to the reject path—not an arm/bin command.
4. Log the image, prediction, confidence, chosen route, and final human
   correction. Those locally collected workcell images are more valuable than
   further tuning on TrashNet's clean-background images.

When you share the exact dataset link or local folder format, its classes and
annotations can be validated before training is added.
