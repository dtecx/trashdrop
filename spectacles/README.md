# Spectacles drive the arms

Each hand drives the arm on its side. The glasses run a Lens (the Lens
Studio project in `spectacles/spectacles/`) that sends the hands to the Mac
over a WebSocket; the Mac (`trashdrop/spectacles.py`) drives the arms. The
details are in that module's docstring. Two ways to drive them:

**`--mode pinch`** (the default), as VR teleoperation does it with a grip
button:

1. **Grab and drag.** Thumb and index together grab the jaw: while you hold
   the pinch and move your hand, the jaw follows it -- as far as it goes
   (`--scale 0.5`: half as far). Let go: the arm stays where it is. Pinch
   again anywhere and go on from there.
2. **Grab and twist.** Pinch and, without moving your hand, twist it like a
   screwdriver: the jaw turns the same way -- clockwise as you see the back
   of your hand is clockwise from above, as the overhead camera shows it.
   The first 10 degrees do nothing; the glasses say how far it has turned
   ("turned 30° cw", or "at the wrist's limit"). Let go, turn your hand
   back, pinch and twist again to turn further. A pinch either drags or
   twists, whichever your hand does first (moves 1.5 cm or turns 10
   degrees), never both, so dragging does not turn the jaw and twisting does
   not move it; a pinch that began as a drag but turns past 25 degrees
   within 4 cm was a twist after all, and the jaw goes back.
3. **Open and close.** Touch thumb and pinky: a closed jaw opens, an open
   one closes. Once per touch; never while pinching.
4. No calibration: forward is where you look when you pinch.

**`--mode joystick`**:

1. **Calibrate.** Both hands up in the air (not on the table), roughly level
   (within 15 cm) and still (within 5 cm) for 5 s: the glasses count down.
   Where the hands rest is their neutral, and "right" is the way from the
   left hand to the right one -- so it does not matter where you looked.
2. **Steer.** Within 3 cm of its neutral, on each axis, a hand moves
   nothing. Past that the jaw goes the same way: the hand a little lower,
   the jaw goes down; lower and to the right, down and to the right. The
   farther past, the faster, 6 cm/s at most. Back to neutral, the arm stops.
   The glasses say which way they read each hand: "moving right, down".
3. **Turn the jaw with a fist.** Close the hand, turn the fist, open it:
   the wrist roll turned as far as the fist did and stays there. While the
   fist is closed nothing else moves; close it again to turn further.
   Closing the hand does not close the jaw: a pinch reaches the jaw 0.4 s
   late, and what thumb and index did on the way into a fist is dropped.
4. **Grip with a pinch.** After a fist, or a hand out of sight, the jaw
   waits until thumb and index agree with how open it is ("jaw waits for a
   pinch"), so opening the hand does not drop what the jaw holds.

The camera looks straight down, so it cannot show how high a jaw is: under
the status the glasses show each fixed finger's height above the table
("left tip 4 cm"), and the text turns red while an arm is down at the table.

Every session's hand data goes to `out/spectacles/session-*.jsonl`
(`--no-record`: not): what the tracking really saw, to tune against.

The arms keep 10 cm between them -- from foot to jaw, both being placed on
the sheet -- so either can reach into the middle, or past it, while the
other is elsewhere. When an arm stops although its hand says go, the glasses
say why: "moving left: at the other arm", "moving down: at the table",
"... at full reach", "... at a joint limit". Driving one arm alone (`--arm`),
it keeps to its own side, 14 cm past its base's line at most: the other arm
is not being watched.

A hand out of sight holds its arm. Back within half a second it carries on;
later, its arm waits until the hand is back at its neutral, and the glasses
say which way ("back to the middle: up 6 cm"). Both hands out of sight for
3 s: calibrate again -- also the way to recalibrate on purpose.

## The Mac

For the unified jury demo, run `uv run trashdrop web --dry-run --open` and
select **Enter Spectacles UI** on the page. Entering only opens the glasses
interface; it does not move the arms. Raise a hand near your face and look at
its wrist briefly to open the menu beside it. The two large controls are
**MANUAL** (follow the wearer's tracked hands) and **AUTO** (sort without
hand control); the active mode is highlighted. **NEUTRAL** returns both arms
to their saved upright pose, and the small **EXIT UI** returns to the web
controls. The menu holds the arms while it is open, so tapping it cannot
become a robot-hand gesture. Look away and it closes after a short pause.
Manual mode uses
the hand-control scheme selected on the page before entering. In auto mode,
the movable video Frame shows the same sorting information as the web page,
including the selected item's planned path to its drop, drawn over the
overhead camera; in manual mode it shows the clear camera feed. Compact status
tags follow the wearer's hands during manual control.
The modes share the web process and its already-open arm buses.

    uv run python -m trashdrop.spectacles --dry-run   # first: no arms, just the link and where they would go
    uv run python -m trashdrop.spectacles             # then the arms (stop `trashdrop web` first: one program a bus)

It prints the address for the Lens. Easiest is the glasses' USB-C cable to
the Mac: they answer adb (`adb devices` lists `Snap_matador`), so the Mac
forwards the glasses' port 8765 to its own (`adb reverse tcp:8765 tcp:8765`,
done for you when they are plugged in at start) and the Lens connects to
`ws://127.0.0.1:8765` -- no Wi-Fi involved. Plug them in again, or restart
them, and start the Mac side again to renew it. Needs adb:
`brew install android-platform-tools`.

Over Wi-Fi instead, the Lens connects to `ws://<the Mac's address>:8765`,
also printed. macOS may ask whether Python may accept incoming connections:
allow it. The glasses and the Mac must be on the same network, and one that
keeps its devices apart (many venue networks do) stops them seeing each
other.

Options: `--hold 5` (seconds to calibrate), `--dead-zone 3` (cm on each
axis), `--gain 1.5` (cm/s per cm past the dead zone), `--top-speed 6` (cm/s,
the fastest the jaw moves), `--speed 120` (deg/s, the most any joint turns),
`--facing them` (standing in front of the arms, facing them), `--arm left`
(one arm only: calibrating then wants that hand alone).

Start with `--dry-run --video` to see in the Mac's log where each jaw would
go, its roll, and each hand's curl (about 2 open, below 1.1 a fist, above
1.35 open again) while you try it. This does not connect to the servo buses.
Keep clear of the arms when they move to READY and while tracking hands.

### Overhead camera in the glasses

    uv run python -m trashdrop.spectacles --dry-run --video --camera auto

`--video` opens the identified overhead USB webcam without starting `trashdrop
web` or claiming the arm buses in a dry run. The camera is selected through the
same `--camera auto` resolver used by the rest of TrashDrop: it briefly zooms
the configured webcam to find its OpenCV stream. If that identification is
unavailable, the bridge refuses to guess an index. An explicit camera index or
URL can be passed with `--camera` when needed.

The bridge forwards port 8766 over the Spectacles USB cable alongside the
hand-control port 8765. `WebcamView` connects to `ws://127.0.0.1:8766`; for
Wi-Fi, set its Url to the video address printed by the Mac. It displays the
newest JPEG frame at up to 30 fps (`--video-fps`; `--video-width` 640 and
`--video-quality` 60 set its size) in `Webcam Window`, a Camera Object child
centred below `Arm Status`. The scene's window is 45 x 34 cm at 60 cm from the
viewer. Its `Preview Too` input is off by default, as on `HandStream`.

The bridge trims only the top of the 800 x 600 camera image, keeping both sides
and arms visible. `rig.toml`'s clicked tape corners and `camera_zone.json`'s
reference image size place the inner square at the vertical centre of the
remaining picture. The JPEG is then resized to 640 pixels wide. If the overhead
camera moves, recalibrate the tape corners before trusting this framing or the
sorter.

Every 10 s the bridge prints the camera's own frame rate, the rate it sends
and the size of a frame: in dim light the webcam slows itself down, and more
light is the fix. `WebcamView` prints displayed fps and an estimated
capture-to-display delay in the Lens Studio device log. The estimate synchronizes the Mac and Spectacles
clocks with WebSocket pings; it can vary with USB and render scheduling. No
frames queue on the Mac or in the Lens decoder: a slow viewer skips old frames.

## The Lens

Built: `spectacles/spectacles/`, a Lens Studio 5.15.4 project from the
Spectacles template. It has to be 5.15.x, not the newest: Lens Studio 5.24
builds for Snap's newer Specs and signs in with a Specs developer account,
and its release notes send Spectacles (2024) owners to 5.15.x
(ar.snap.com/download/v5-15-4). The project's one script is
`Assets/Scripts/HandStream.ts`, on the `HandStream` object; `Arm Status`, a
text above the middle of the view, shows what the Mac says the arms are
doing. Project Settings has Experimental API on, as a plain `ws://` needs:
such a Lens runs on your own glasses but cannot be published, which is fine
here.

To put it on the glasses:

1. Lens Studio signed in (My Lenses > Login) with the Snapchat account the
   glasses are paired with. Signed out, the glasses never connect to it, and
   sending ends in "No connected clients". The browser may call Lens Studio
   back more than once, so its log can show `invalid_grant` next to
   `Authorized`; `Authorized` is what counts.
2. Spectacles app on the phone, once: Developer Settings > Lens Development >
   Enable Wired Connectivity.
3. The glasses on the USB-C cable: Lens Studio's Logger says
   `Spectacles connected to USB`.
4. Open `spectacles.esproj` and press Preview Lens. Lens Studio may say it
   cannot read the glasses' version (it wants 5.64.396 or later); the Lens
   is sent and starts anyway.

The `HandStream` object's Url is `ws://127.0.0.1:8765`, the Mac over the
cable; over Wi-Fi, set it to the address the Mac prints. In the Lens, look at
your hands: the text says `connected`, then what each arm is doing. Hands
out of view hold the arms; Ctrl+C on the Mac holds them too.

Lens Studio's own preview runs the Lens on the Mac, where that same address
reaches the Mac side, and its hands -- none, as a rule -- would mix with the
glasses' and keep the arms from following. So it does not connect unless
Preview Too is ticked on `HandStream`. Ticked, with the preview's device set
to Spectacles, it tries the whole link without the glasses: start the Mac
side with `--dry-run`, refresh the preview, and the simulated hands show up
in what the Mac prints. Untick it before the glasses drive the arms.
