# Spectacles drive the arms

Each hand moves the arm on its side; thumb and index apart open its jaw,
together close it. The glasses run a Lens (the Lens Studio project in
`spectacles/spectacles/`) that sends the hands to the Mac over a WebSocket;
the Mac (`trashdrop/spectacles.py`) has the arms follow them. How a hand maps
to its arm is in that module's docstring.

To control the wrist, hold a hand in view, then **raise or lower its fingers**
to tilt that arm's jaw (`wrist_flex`). **Rotate the palm around the direction
the fingers point** to turn the jaw (`wrist_roll`), as if turning a screwdriver.
The two motions work together, independently for the left and right hands.
The pose when each hand appears is its zero: you can start comfortably without
the jaw jumping. Move the hand out of view and bring it back to reset that
reference. Pinch only changes the jaw opening; the palm knuckles drive wrist
orientation even while pinched. Wrist joint limits and `--speed` still apply.

## The Mac

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

Options: `--scale 1.5` (arm cm per hand cm), `--speed 120` (deg/s, the most
any joint turns), `--facing them` (standing in front of the arms, facing
them), `--arm left` (one arm only).

Start with `--dry-run --video --camera auto` to see both wrist angles in the
Mac's log while turning your hands. This does not connect to the servo buses.
For a first arm trial, `--speed 60` makes every joint slower. Keep clear of the
arms when they move to READY and while tracking hands.

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
newest JPEG frame at up to 10 fps in `Webcam Window`, a Camera Object child
centred below `Arm Status`. The scene's window is 45 x 34 cm at 60 cm from the
viewer. Its `Preview Too` input is off by default, as on `HandStream`.

The bridge trims only the top of the 800 x 600 camera image, keeping both sides
and arms visible. `rig.toml`'s clicked tape corners and `camera_zone.json`'s
reference image size place the inner square at the vertical centre of the
remaining picture. The JPEG is then resized to 640 pixels wide. If the overhead
camera moves, recalibrate the tape corners before trusting this framing or the
sorter.

`WebcamView` prints displayed fps and an estimated capture-to-display delay in
the Lens Studio device log. The estimate synchronizes the Mac and Spectacles
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
