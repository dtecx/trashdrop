# Calibrating the real table

Everything sits on one flat surface at a known height, so a plane-to-plane
homography is all that is needed. **No depth camera is required**, which is
fortunate twice over: the camera may end up being an Android phone, and stereo
depth fails on exactly the items this cell handles most — clear PET bottles.

## What to do

1. Mount the camera overhead, roughly 55–65 cm above the table, pointing
   straight down. **Tape it.** Once calibrated it must not move.
2. Print four ArUco markers (`DICT_4X4_50`, ids 0–3) and tape them at the four
   corners of the pick zone. Measure their centres in the table frame — the
   frame's origin is the front arm's base, `+y` points from the front arm
   toward the back arm's side being negative, matching `station.py`.
3. Grab a frame and build the homography:

```python
import cv2
from trashdrop.perception.calibration import HomographyCalibration, detect_aruco_corners
from trashdrop.station import PICK_ZONE

frame = cv2.imread("calib.jpg")
image_points = detect_aruco_corners(frame, marker_ids=[0, 1, 2, 3])
assert image_points is not None, "a marker was not found; fix lighting or placement"

calibration = HomographyCalibration(image_points, PICK_ZONE.corners())
print(calibration.residuals_mm())      # expect a few mm at most
calibration.save("data/calibration.json")
```

4. **Read the residuals.** Anything above a few millimetres means a marker was
   misidentified or the surface is not flat, and the arm will miss grasps by
   that amount. Do not proceed past a bad number.

Any four known, non-collinear points work — the corners of a printed sheet are
a fine fallback if the markers do not detect.

## Checking it against the arms

Camera calibration says where an item is in the *table* frame. It does not say
whether the arms agree with that frame. Verify the two separately:

1. Put an item at a known table coordinate and confirm the detector reports it.
2. Command the arm to that same coordinate and see whether the tool lands on
   the item. A constant offset means the arm base position in `station.py` is
   wrong; re-measure the base and update it, then run `trashdrop probe`.

## After any change to the table

Re-run the probe. It re-solves IK to every bin and pick-zone corner and names
what stopped being reachable:

```bash
uv run trashdrop probe --map
```

Use the map to place the bins before drilling anything. The measured envelope
is tight: nothing above z = 0.12 m is reachable with the gripper held vertical,
and the usable patch in front of a base is about 20 x 20 cm.

## If the camera is a phone

Install any "IP webcam" app and pass the stream URL:

```bash
uv run trashdrop capture --camera http://192.168.1.5:8080/video ...
```

Two cautions. Disable any auto-rotation, auto-exposure lock changes and
"beautify" processing — the homography assumes a fixed geometry and the
background subtraction assumes stable exposure. And keep the phone on mains
power; the stream drops when it sleeps.
