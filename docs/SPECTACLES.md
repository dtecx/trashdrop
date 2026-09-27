# Spectacles teleoperation: state, facts, and what is next

Written 2026-09-27 ~03:00 as a handoff between agents. Read `AGENTS.md`
first; this file holds everything specific to driving the arms through Snap
Spectacles. The user-facing how-to is `spectacles/README.md`; the control
logic is documented in `trashdrop/spectacles.py`'s module docstring.

## The pivot

On 2026-09-27 the team made **teleoperating the two SO-101 arms through Snap
Spectacles the main goal** of the hackathon project. The trash-sorting code
stays and must keep working: after any change, run the full test suite and
`uv run trashdrop sim` (expect `sorted correctly: 3/3`; that line is about
motion, since the simulated colour detector is told the answer). Everything
will be shown to the jury, so an intuitive interface matters most.

## How to run it

```bash
uv run trashdrop web --dry-run --open                    # unified jury UI; Spectacles sockets and video included
uv run trashdrop web --open                              # real cell; confirm the manual-mode move in the page
uv run python -m trashdrop.spectacles --dry-run --video   # no arms: the link, the video, where the arms would go
uv run python -m trashdrop.spectacles --video             # the arms (stop `trashdrop web` first: one program a bus)
uv run python -m trashdrop.spectacles --mode joystick     # the other control scheme
uv run python -m trashdrop.spectacles --scale 0.5         # pinch mode, half the travel: fine work
```

Run it from the user's own terminal: this agent's shell has no macOS camera
permission (see "Machine facts"). Then press **Preview Lens** in Lens Studio
5.15.4 with the project `spectacles/spectacles/` open: that is the only way
to send the Lens to the glasses (see "Lens Studio facts").

## The pieces

| Piece | What it is |
|---|---|
| `trashdrop/spectacles.py` | The bridge: WebSocket servers (hands on 8765, video on 8766), `adb reverse` for both, calibration, `Follower` (joystick), `PinchFollower` (pinch), IK steps, limits, clearance between arms, JSON report to the glasses, session recording, video pacing |
| `trashdrop/web/manual.py` | The unified page's owner of both Spectacles sockets, overhead stream and exclusive manual session. It reuses the cell's already-open camera and arm objects, so no bus is opened twice |
| `spectacles/spectacles/` | Lens Studio **5.15.4** project (committed). Scripts in `Assets/Scripts/` |
| `.../Assets/Scripts/HandStream.ts` | Sends both hands ~30/s and handles requested optical snapshots / spectator frames by encoding the colour camera (started by the first request, not at launch) and Lens render target. `previewToo` is off, so the LS preview does not connect |
| `.../Assets/Scripts/SpectaclesUI.ts` | Movable video Frame, compact hand tags, a palm-summoned UIKit menu (manual hand control, auto sort, neutral and exit), and a hint card whenever nothing else would show (no Mac yet, or the glasses UI off) |
| `.../Assets/Scripts/WebcamView.ts` | Video WebSocket 8766 -> `Base64.decodeTextureAsync` -> a movable/resizable UIKit Frame. Sorting overlays are burned into the camera stream by `trashdrop/web/overlay.py`; manual mode keeps the camera unobscured |
| `tests/test_spectacles.py` | Socket, snapshots, spectator stream, video pacing, wearer frame, calibration, joystick, pinch, fist/jaw and two-arm-clearance tests |
| `trashdrop/kinematics.py` `links()` | The arm's centre line (foot, lift, elbow, wrist, TCP) for keeping the arms apart |
| `trashdrop/placement.py` `to_sheet()` | Arm frame -> sheet frame (inverse of `to_arm`) |
| `out/spectacles/session-*.jsonl` | Every session's messages from the glasses, one JSON line each with the bridge's time `at` (gitignored) |
| `spectables/spectables-integration/` | The **obsolete** Lens Studio 5.24 project, committed by Codex in `2371440`. Safe to delete (not done) |

## Control, as it stands

### `--mode pinch` (default, `PinchFollower`)

The VR-teleoperation pattern (grip button as a clutch; as the SO-101 Quest
kits and OPEN TEACH do it), with pinches for buttons.

- **Thumb + index pinch = drag.** The jaw goes to where it was at the pinch
  plus `--scale` x the pinch point's travel (thumb/index midpoint, 1-euro
  filtered; forward/right/up from the gaze at the pinch, level). **The wrist
  roll never changes in a drag**, however the hand turns.
- **Thumb + middle touch = turn** (since 2026-09-27 ~09:10, at the user's
  request: twisting the hand was the hardest thing to do). Thumb and middle
  tip < 2.5 cm (part > 3.5), the thumb-index gap > 4 cm, no index pinch,
  held 0.2 s. Then the hand's sideways travel (the wearer's right, from the
  gaze at the touch) turns the jaw about where it points, 5 deg/cm, right =
  clockwise from above (as the overhead camera shows the jaw); the jaw
  stays where it is. `roll_sense()` derives the joint's sign from the model
  (on these arms +wrist_roll turns the downward jaw anticlockwise from
  above). Thumb and middle meet by themselves during index pinches (7 % of
  pinched frames under 2.5 cm) and in relaxed hands (3 %), hence the index
  and dwell conditions: on 43 minutes of recordings without the gesture,
  one false start. Turning into the wrist's limit shows `turning: at the
  wrist's limit`. The earlier screwdriver-twist rules (`TWIST_DEG`,
  `LATE_TWIST`, the drag/twist split) are gone; their history is below.
- **Thumb + pinky touch = toggle the jaw** open (60 %) / closed; once per
  touch (re-arms when > 5.5 cm apart), never while pinching. OPEN TEACH uses
  the same gesture.
- Pinch detection: the Lens's `pinch` (SIK) when present, else thumb-index
  distance with hysteresis `GRAB_CM` = (2.5, 4.0).
- Status per arm: `free`, `pinched` (undecided), `dragging`, `turning`,
  `no hand`, plus `: at <reason>`, `, turned N° cw|ccw`, `, jaw closed`.

### `--mode joystick` (`Follower`)

Calibrate (both hands up, level within 15 cm, still within 5 cm, 5 s; right =
from the left hand to the right one), then each hand is a rate joystick:
3 cm dead zone per axis, 1.5 cm/s per cm past it, 6 cm/s top, target <= 2 cm
ahead of the jaw. A fist (curl < 1.1, open > 1.35) freezes motion and turns
the wrist roll; the jaw follows the pinch 0.4 s late (so closing into a fist
does not grip) and waits for the pinch to agree after a fist or a long loss.
Hand lost > 0.5 s: its arm waits until the hand is back at its neutral; all
hands lost 3 s: recalibrate. The bridge also sends each hand's box (the dead
zone around the neutral, in the glasses' world) in the report; the Lens does
not draw it yet.

### Safety and limits (both modes)

- Target within reach (12-38 cm from the pan axis), 3-35 cm above the table
  (the TCP; the fingertip is 0.7 cm further), jaw pointing down.
- **The two arms keep `CLEARANCE_CM` = 10 cm between their centre lines** on
  the sheet; a step that brings them nearer is not taken. Driving one arm
  alone (`--arm`) or without sheet placements, each keeps to its own side,
  `SIDE_CM` = 14 cm past its base line (that static rule used to leave the
  middle of the sheet to neither arm -- the "it stops at some boundary" bug).
- Joints no faster than `--speed` (120 deg/s), jaw 150 %/s.
- The status tells why an arm stops: `at the table`, `at the top`, `at full
  reach`, `at the base`, `at the other arm`, `at the other arm's side`, `at a
  joint limit`.

### The report to the glasses (bridge -> Lens, JSON text)

```json
{"status": "left: free | right: dragging: at the table (dry run)",
 "manual": true, "auto": false, "busy": "spectacles", "presentation": true,
 "hands": {"right": {"mode": "moving", "blocked": ["down"], "tip": 3}}}
```

Per hand (keyed by the *hand*, which with `--facing them` drives the other
arm): `mode` (holding/moving/turning/lost/centring/calibrating), `blocked`
(directions in the wearer's words), `tip` (fingertip cm above the table);
joystick mode adds `centre`, `axes`, `half` (the dead-zone box). The Lens
falls back to showing plain text from an older bridge.
The unified web bridge adds `manual`, `auto`, `busy`, `controlError`,
`emptyPhotographed` and `presentation` even when no hand follower is
running, so the controls show the actual cell state. Lens commands are
`manual` and `auto` with an `enabled` boolean, `neutral`, `empty`
(explicitly photograph the cleared zone), and `presentation` with
`enabled: false`.

### Video

`--video` opens the overhead USB webcam (046d:08e5) via `--camera auto`
(the webcam is found by zooming it, never by a guessed index), encodes JPEG
(640 px, quality 60) and pushes the newest frame only (no queue) at up to
`--video-fps` 30. It used to be capped at 10 fps on the Mac (the glasses
showed 10 fps, 0-94 ms capture-to-display). Every 10 s the bridge prints the
camera's own rate, the rate sent and the frame size; in dim light the webcam
slows itself down.

## Where the numbers came from (recordings in `out/spectacles/`)

- `session-20260927-014728.jsonl` (joystick trial): the right arm pinned at
  y = 14.0 (the old side rule); idle hands drifting down pinned arms on the
  table floor.
- `session-20260927-023531.jsonl` (first pinch trial, 70 pinches):
  thumb-index pinches < 2.5 cm, open hands 10-15 cm; thumb-pinky never under
  6 cm (fists included) -> toggle thresholds; the wearer's twists were
  screwdriver rolls of 30-99 deg with < 17 deg about the vertical; the pinch
  point *and* the wrist strayed 3-8 cm while twisting (so the drag/twist
  split, not a different reference point). Replayed, 12 deg / 1.5 cm sorts 40
  of 41 drags and 13 of 19 twists (the wearer not yet knowing the rule).
- `session-20260927-025229` and `-030424` (pinch mode, drag/twist split):
  "the left arm behaves strangely when turning, the right is fine". The
  left hand's twists mostly started as drags (8 of 20 recognised over three
  sessions, right hand 19 of 24): the pinch swings with the twist and the
  left hand also shifts. The wrist as the reference point did not help (the
  whole hand moves). 10 deg plus the late switch: 15 of 20 left twists,
  drags unchanged. Also: the **left arm's wrist roll range is lopsided** --
  `wrist_roll_offset` -80 in `rig.toml`, so the model allows LeRobot roll
  -77..243 and from the ready pose (0) only 77 deg clockwise (the right arm:
  -162..158). The real limits come from the servos' EEPROM
  (`Arm.limits_degrees()`) and were not read; check them if the left jaw
  still stops turning early.

To replay a session through the real controller, feed its lines to a
`Follower`/`PinchFollower` with `dt` from successive `at` values and the
rig's models (`Kinematics(rig.arms[n].wrist_roll_offset)`,
`Placement(*rig.arms[n].sheet)`, `ready_pose(...)`), passing the other
arm's `centre_line()` as `others` and the message's `head`. That is how the
findings above were made; no replay tool is in the repo yet (worth adding).

- `session-20260927-090127` (manual on, off, on again): after re-enabling,
  both arms showed `stopped` a second later and never answered.
  `ManualBridge.stop()` queued a `{"command": "stop"}` and then set the
  shutdown flag; a loop ending mid-tick never took it, so the next session's
  `follow()` took it first. Fixed: `stop()` queues nothing, `start()` and
  `follow()` drop commands left from before (regression test
  `test_a_stop_left_from_a_session_before_does_not_stop_the_next`).

## Lens Studio facts and gotchas

- **Use Lens Studio 5.15.4**: installed as `/Applications/Lens Studio
  5.14.4.app` (the name is wrong; it is 5.15.4). `/Applications/Lens
  Studio.app` is **5.24**, for Snap's newer Specs, wants a SPECS account, and
  its projects do not run on the team's Spectacles (2024, adb model
  `Snap_matador`).
- Sign in: My Lenses > Login with the Snapchat account paired to the
  glasses. The browser calls back several times, so the log shows
  `invalid_grant` beside `Authorized`; `Authorized` is what counts.
- Project Settings has **Experimental API on**: plain `ws://` needs it (the
  Lens runs on the developer's glasses, cannot be published).
- **Sending the Lens to the glasses cannot be automated**: no keyboard
  shortcut (official list), no MCP tool, `File > Send to Spectacles` is only
  a submenu ("Open Spectacles Monitor"), and Lens Studio's window exposes no
  accessibility elements (its menu bar does). Only a real mouse click at a
  learned position would work -- not built. It warns it "cannot read the
  glasses' version (wants 5.64.396)"; the Lens is sent anyway.
- MCP server `lens-studio`: `http://localhost:50049/mcp` (same port as 5.24,
  new token; configured for Claude in `~/.claude.json` and for Codex in
  `~/.codex/config.toml`). No save, push, screenshot or `ExecuteEditorCode`
  tools in 5.15.4. `SetLensStudioProperty` silently drops object values
  (vec3/rect/transform): set scalar sub-paths (`localTransform.position.z`,
  `worldSpaceRect.left`). Asset files can be written straight into `Assets/`
  (`.ts`, or `- !<InternetModule/<uuid>>` + `  PackagePath: ""`) and Lens
  Studio imports them. `CompileWithLogsTool` may time out: look for
  `TypeScript compilation succeeded` in the log instead.
- Log: `~/Library/Preferences/Snap/Lens Studio/logs/LensStudioLog-*.txt`.
  The glasses' `print` output lands there too (`handleLogMessage`), as do
  login, device connection and push events.
- Lens Studio 5.15.4 can die with a native `SIGSEGV` and no preceding script
  exception if a UIKit hierarchy is disabled from the trigger callback of a
  button that is still hovered. Presentation exit is therefore applied one
  second after trigger-up, and re-centering is deferred to the next hand
  update. Repeated status text is de-duplicated and video statistics are
  printed every 10 seconds to keep Push to Device logging light.
- The LS preview has no hand tracking. Sockets it opens with `previewToo` on
  survive a preview reset: restart the bridge after such a test.
- `CameraModule.createCameraRequest()` cannot run from a script's `onAwake`:
  the real Spectacles throw `Unable to access camera` and the rest of that
  callback (including the hand socket) never starts. It ran from an
  `OnStartEvent` until 2026-09-27; now the first snapshot request starts the
  camera, not the launch (see "Crash at launch" below). Keep camera failure
  non-fatal so controls still connect.
- **"Crash at launch" (2026-09-27 10:12).** The user relaunched the Lens four
  times in a minute and called it a crash. It was the same build that had run
  fine at 09:02 (6 895 515 bytes sent each time). What the log and the session
  recording show:
  (1) `trashdrop web` had just been restarted, and a restart turns the glasses
  UI off until the page's **Enter Spectacles UI**. With it off, the Lens
  hides the video a second after it connects; the menu does not open, and
  `Arm Status` is disabled in the scene. So the wearer saw the video flash
  and then nothing at all. Three of the four Lenses were alive the whole time
  (their hands kept arriving until the next push replaced them).
  (2) The second one died right after starting: `connecting to` was its last
  print, it never connected, and there was no script exception. That is a
  native kill. At that moment the only thing the Lens had asked of the system
  was the colour camera, which it then requested at every launch.
  Fixes: a card in front of the wearer whenever nothing else is shown
  ("CONNECTING TO THE MAC", "NO ANSWER FROM THE MAC", "GLASSES UI IS OFF / ON
  THE WEB PAGE: ENTER SPECTACLES UI"); the camera only on a snapshot request;
  no JPEG decoding while the video is hidden; and a pending snapshot is
  dropped when the glasses disconnect, so a capture that kills the Lens is
  not asked of every relaunch. Lens Studio logs nothing when a Lens on the
  glasses dies. Look for its prints stopping, and in
  `out/spectacles/session-*.jsonl` (the `t` field restarts with each Lens) for
  its hands stopping. About 7 messages a second instead of 30 means the
  glasses were throttling the Lens: they were put down, or taken off.
- **"Jerky" arms (2026-09-27 11:14): the hands arrive in bunches.** The Lens
  sends one message every 33 ms (its `t`, steady). They reach the Mac (`at` in
  the recording) three at once every ~100 ms, in 15% of the gaps while
  pinching, 7% with hands in view, 1% with none. This was so in every session
  that day (5-24% while pinching). Steered by the newest message, the jaw
  lunged and waited ten times a second. Replayed through the real 50 Hz loop:
  jaw speed as a 10 Hz sawtooth, p95 acceleration 777 cm/s2, against 611 with
  the same messages arriving evenly.
  Likely cause: the Mac answered only when its report changed. While dragging,
  answers came and went. The glasses' socket (Snap offers no TCP_NODELAY)
  waits for the acknowledgement of what it sent, and an unanswered message's
  acknowledgement was delayed. The Mac now answers every message and sets
  TCP_NODELAY; the next recording shows whether the bunches are gone.
  In any case `follow()` now steers by `Hands.playout()`. The hands are played
  back by the Lens's clock, interpolated, and held behind by what the bunching
  needs: 33 ms when messages come on time, about 100 ms with bunches, 150 ms
  at most. Replayed, p95 acceleration 333 cm/s2, with no sawtooth.
  To check a session: `at` gaps against `t` gaps in `out/spectacles/session-*.jsonl`.
- The active display Render Target reports 1392 x 1590 on the real glasses even
  with asset `ResolutionScale` set to 0.3. Reading it back kills the Lens inside
  `Base64.encodeTextureAsync`, before either callback. The 2 fps spectator
  stream therefore follows the raw-camera path from Snap's live-streaming
  example: a 360 px `CameraModule` texture, low JPEG quality, five seconds of
  startup time, and readback only from `CameraTextureProvider.onNewFrame`.
  This is the real optical camera without the Lens overlay. Explicit snapshots
  retain the separate high-quality render-target path for later investigation.
- `adb` works (`adb reverse` for the sockets) but the glasses' shell is
  closed: `adb exec-out screencap` answers `error: closed`.
- The Spectacles UI Kit (`Packages/SpectaclesUIKit.lspkg`: Frame, Capsule/
  Rectangle/Round buttons, Slider, Switch, Toggle, ScrollWindow, GridLayout,
  TextInputField) and SIK (ContainerFrame, PinchButton, ToggleButton, ...)
  are installed in the project: the building blocks for a proper UI.
- The project's `.gitattributes` routes `Assets/**` and `Packages/**` to Git
  LFS, but git-lfs is not installed and the files were committed as plain
  blobs; teammates with git-lfs may see "should have been pointers"
  warnings. Removing those two lines is the likely fix.

## Machine facts (the user's Mac)

- The bridge and `trashdrop web` both open the arms' serial buses: **one
  program a bus**. A combined demo must run everything in one process.
- This agent's shell has no camera permission and no screen-recording
  permission (`screencapture` fails); `osascript` launched through `uv run
  python` does have accessibility access.
- A bridge started in the background here ignores SIGINT; stop it by PID
  with SIGTERM.
- Glasses on USB: `adb devices` lists `Snap_matador`; the bridge sets up
  `adb reverse tcp:8765` / `8766` itself when they are plugged in at start.

## Team and hackathon facts

- Alien Bazaar 2026, Warsaw, 25-27 September; ~20 teams; judged on
  creativity, proactive collaboration with other teams, effort during the
  event, and the final presentation. All hardware is issued at the venue.
- The user writes Russian; every artefact in the repo is English.
- The user may run several agents (Claude, Codex) in this folder at once:
  run `git status` and `git log --oneline -5` before writing, and commit
  early.
- Observation not yet reconciled: `AGENTS.md` describes the arms facing each
  other 43 cm apart, but the venue rig's sheet placements (`rig.toml`) put
  the bases side by side, about 38 cm apart, both facing the sheet's +y, the
  sheet between and ahead of them.

## Next steps (in order)

### Central reBot B601-DM (2026-09-27)

The team expects a third, 6-DoF-plus-gripper B601-DM at the centre of the
table. The two SO-101 arms must be put in their saved neutral poses before
the B601 is commissioned. `trashdrop web --dry-run` now offers **center B601
· preview** and the palm menu offers **B601 PREVIEW**. This reserves the
manual session exclusively, uses the right hand's thumb/index pinch as a
clutch, shows Lens-world displacement, and sends **no** motor commands to
the B601 or the SO-101 arms. The B601 entry is refused when dry run is off;
do not present the preview as working robot teleoperation. The menu still
suppresses hand tracking while a button or video Frame is manipulated.

Before enabling real B601 motion, obtain the actual adapter identity,
confirm the exact DM model and 24 V power arrangement, perform Seeed's
motor-ID/zero checks, measure the central base transform and the clearance
to *both* SO-101 neutral poses, and test tiny single-joint moves with a
physical stop ready. Seeed's official SDK uses Pinocchio and MotorBridge
and requires Python >=3.10,<3.12; its full control tutorial recommends a
physical Ubuntu system, while MotorBridge's setup page says its gateway can
run on macOS. This repository uses Python 3.13, so a separate 3.11 process
or a verified compatible driver will be needed rather than importing the
SDK into `trashdrop web` as it stands. The two SO-101 buses must remain
owned by the web process.

Mac feasibility check (read-only, 2026-09-27): the official SDK cloned into a
separate Python 3.11 uv environment and MotorBridge imported on this Mac.
The SDK's lockfile selected `cmeel-tinyxml2` 10, although its Pinocchio/URDF
wheel links `libtinyxml2.11.dylib`; upgrading that package to 11 in the
temporary environment made the 8-DoF URDF load. This proves imports, **not**
USB-CAN or actuation. More importantly, SDK `RebotArm._make_controller()`
selects the DM serial transport only for a `/dev/tty...` prefix (Mac ports
are usually `/dev/cu...`), and the example `RebotArmEndPose.start()` leaves
its first joint target at zero. Neither path is safe to use unchanged on
the live arm. The eventual adapter must explicitly use
`Controller.from_dm_serial(channel, 921600)`, read and validate all seven
joint positions, seed every target to those measured positions **before**
enabling, and never run `safe_home()` automatically on stop. The rig-specific
base transform, workspace and collision clearance remain to be measured.

- https://wiki.seeedstudio.com/rebot_b601_dm_getting_started/
- https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/
- https://github.com/Seeed-Projects/reBotArm_control_py

1. **Snapshots from the glasses: built in `afa41ce`.** A trigger at
   `out/spectacles/snap` requests the Lens render target and colour camera;
   the bridge validates both JPEGs and writes them plus their additive
   composite to `out/spectacles/snaps/`. Still needs a real-glasses capture
   after the next Preview Lens push.
2. **Spatial UIKit interface: built after `bc097ef`; device QA pending.**
   The video stays in a movable, resizable Frame. Compact status tags follow
   the wearer's tracked hands during manual control. **The menu comes to a
   palm held to the glasses** (SIK `isFacingCamera()`, sent as `palm`), not
   pinching, 20-75 cm away, for 0.6 s. It opens 45 cm ahead, just below the
   line of sight and a little towards the other hand, stays fixed in space,
   and closes after a mode choice (then the palm must go down before it
   reopens), or 1 s after nothing uses it: not in its first 2.5 s, and not
   while the palm is up, an index fingertip is within 18 cm of it, or a pinch
   holds. Buttons are 7 x 4 cm, 2 cm apart; the menu is 16 x 18 cm, about 20
   degrees. Until 2026-09-27 ~11:00 it opened at the wrist and was 20 x 22 cm:
   a comfortably bent arm (palm 30-40 cm away) put it nearer than 35 cm and
   past the edges of the glasses' view, and the session recording showed the
   wearer opening it at 45-58 cm, almost every time ("I have to hold my arm out very
   far"). Until 2026-09-27 ~08:45 it came to a *look* at a raised wrist
   in a 30-degree cone for 0.45 s -- where the eyes are while driving -- so it
   kept opening and holding the arms mid-drive ("manual control sometimes
   stops working"), and its buttons were 0.3-0.7 cm apart ("too close").
   While it is open, hand packets are marked untracked so menu pinches hold
   the robot instead of dragging it. Grabbing, resizing or pressing the video
   Frame also holds both arms; release both pinches before hand control resumes.
   **MANUAL** starts or stops following the
   wearer's tracked hands; **AUTO** starts or stops ordinary sorting; **NEUTRAL**
   stops the active mode and returns both arms to their saved neutral poses;
   **EXIT UI** returns to the web controls. Starting a mode first releases the
   previous owner of the arms. If no empty-zone reference exists, the menu
   offers **EMPTY ZONE**: clear the taped zone, capture its background, then
   tap **AUTO** again. It never photographs automatically with an item present.
   Active modes are highlighted; neutral is prominent and exit is small.
   The next real optical snapshots decide final
   type and angular sizes. In auto mode the video also draws a planned path
   arrow from the selected grasp to the chosen drop; this is not live motion
   tracking.
3. **Unified jury page: implemented locally after `bc097ef`, real-device QA
   pending.** `trashdrop web` owns the existing Cell and the Spectacles
   bridge together. It has auto sort/pick, manual start/stop/mode, arm state,
   snapshots and the complete calibration checklist. **Enter Spectacles UI**
   hides every ordinary web control and shows only the stable overhead camera;
   it does not start manual mode or move the arms. **EXIT UI** in the glasses
   stops manual mode and restores the page, while auto sorting can continue.
   The video Frame shows the camera with zone, arm, drop, item, grasp and class
   overlays outside manual mode. The raw
   optical spectator stream works, but is deliberately parked while the Lens
   UI is tuned. The overhead Frame now contains the whole cropped width instead
   of cropping its left edge. Next: push Preview Lens, test this round trip and
   tune it from real snapshots.
4. **Recalibrate the left arm** (the team took it apart and may have put a
   horn or the gripper back at another angle). Evidence, read with
   `uv run trashdrop arm status` on 2026-09-27 03:30: `rig.toml`'s left
   `touch_poses` reach `shoulder_pan` 105.5 deg, but the servo's limits now
   are 836..2794 ticks, +-84 deg -- the pan's zero or range changed after
   the tape was touched (the right arm agrees: touches to -97, limits
   +-112). Its `wrist_roll_offset` (-80; the right arm's is 5) may be stale
   too. In order, arm clear, bridge and `trashdrop web` stopped: `arm zero
   left` if the gripper sits turned on the roll shaft; `rig touch left
   --tape` (refits the placement, absorbing a pan offset; large residuals
   mean a lift/elbow/flex zero is off: redo the servo calibration first);
   `rig roll left`.
   **Done 2026-09-27 ~03:45** (commit after `ead8c3c`): new touches (fit
   0.6-1.4 cm; they moved 1.4-3.4 cm from the stale ones), wrist roll zero
   -81.6 (was -80: the gripper had not moved on its shaft). Still open: the
   left `shoulder_pan` servo limits are +-84 deg, yet the tape's near-left
   corner needed pan 97.9 deg (moved there by hand, limp): under its own
   power the left arm cannot reach that corner. Widen the pan's EEPROM
   limits (the servo calibration, moving the pan through its whole physical
   range). Also odd: `rig touch` printed the same residuals for the right
   arm as for the left.
5. Cleanups: delete `spectables/`; fix the Lens project's `.gitattributes`;
   add a session replay tool; reconcile the layout description in
   `AGENTS.md` with `rig.toml`.
