"""The sorting cell as one object: camera, arms, calibration, classifier, and what it last saw.

`trashdrop pick` asks its questions at the keyboard. `trashdrop web` drives
the same steps from a browser, and for that the cell has to run on its own:
the camera read continuously by a thread, so the page's stream and a look at
the zone share it; one action at a time on a worker thread, because the arms'
buses are not safe to share; a stop that holds the arms from any thread
(Arm.stop); and everything the page draws kept as plain data -- the zone, the
item's outline, where each finger goes -- rather than burnt into frames.

The steps themselves are sorter.py's, in pick's order: photograph the empty
zone; look (find the item, classify it, plan with the arm on its material's
side); pick (go, then look again: an item still in the zone is tried again,
PICK_TRIES in all).

`Cell.demo()` stands in saved pictures for the camera and simulated arms for
the real ones, so the page can be worked on without the cell.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .sorter import (
    FINGERTIPS_CM,
    MIN_CONFIDENCE,
    PICK_TRIES,
    SIDE_OF,
    base_on_sheet,
    drop_pose,
    execute_pick,
    find_item,
    item_crop,
    plan_pick,
    reach_mask,
    refine_item,
    side_for,
    zone_mask,
)

ANALYSIS_WIDTH = 320
SETTLE_AFTER_MOTION_S = 0.4  # frames older than this since the arms stopped still show them moving
# Auto mode, for a person tossing items in while it works: it looks this
# often at a zone with nothing to take; an item is taken only if a second
# look this much later finds it where the first did (a thrown item rolls, a
# hand leaves); and a place it failed PICK_TRIES times at is left alone
# until what is there changes.
AUTO_IDLE_S = 1.0
AUTO_STEADY_S = 0.6
AUTO_SAME_PLACE_PX = 30  # about 2 cm at the venue
DEMO_TOSS_S = 6.0  # the demo tosses its item back in this long after it was taken
STARTING = {
    "empty": "empty zone: photographing it",
    "look": "look: finding the item",
    "pick": "pick: finding the item, then taking it",
    "neutral": "neutral: standing the arms up",
    "auto": "auto sort: starting",
    "relax": "relax: torque off",
}
# The detector speaks to the keyboard ("press b"); the page has buttons.
PAGE_WORDS = {
    "nothing_changed": "the zone is empty",
    "untrusted": "the picture changed too much to compare (light? camera moved?): press Empty zone again",
    "touches_frame_edge": "the item touches the edge of the searched area: move it towards the middle",
    "more_than_one_object": "two separate things changed: one item at a time, and hands out of the zone",
}


@dataclass
class Options:
    """How the next pick is made; the page changes these while the cell runs."""

    fingertips_cm: float = FINGERTIPS_CM
    min_confidence: float = MIN_CONFIDENCE
    material: str | None = None  # say what it is instead of asking the classifier
    any_arm: bool = False  # do not sort: the nearer arm takes it
    only_arm: str | None = None  # "left" or "right": use that arm alone
    dry_run: bool = False  # hover over the item, never grasp


@dataclass
class Look:
    """What one look at the zone found, as the page draws it (full-resolution pixels)."""

    at: float
    message: str
    code: str = "ok"
    outline: list[list[int]] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    side: str | None = None
    sure: float = 0.0
    arm: str | None = None
    fixed: list[float] | None = None
    moving: list[float] | None = None
    width_cm: float | None = None
    open_percent: float | None = None
    lean_deg: float | None = None
    distances_cm: dict[str, float] = field(default_factory=dict)


class Camera:
    """Reads frames on a thread; keeps the newest, and when it came."""

    def __init__(self, read, *, name: str = "camera") -> None:
        self._read = read
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._at = 0.0
        self._times: deque[float] = deque(maxlen=30)
        self._running = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def close(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            frame = self._read()
            if frame is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            with self._lock:
                self._frame, self._at = frame, now
                self._times.append(now)

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._at

    def fresh(self, after: float, timeout: float = 3.0) -> np.ndarray:
        """The first frame taken after ``after`` (monotonic seconds)."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, at = self.latest()
            if frame is not None and at > after:
                return frame.copy()
            time.sleep(0.02)
        raise RuntimeError("the camera stopped sending pictures")

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)


class Cell:
    def __init__(self, *, rig, homography, placements, arms, poses, kinematics, classifier, camera: Camera,
                 limits, log=print, save_rig=None) -> None:
        self.rig, self.homography, self.placements = rig, homography, placements
        self.arms, self.poses, self.kinematics = arms, poses, kinematics
        self.classifier, self.camera, self.limits = classifier, camera, limits
        self._print, self._save_rig = log, save_rig
        self.options = Options()
        self.log_lines: deque[str] = deque(maxlen=300)
        self.stop_event = threading.Event()
        for arm in arms.values():
            arm.stop = self.stop_event
        self.busy: str | None = None
        self.auto = False
        self.background: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        self.last: Look | None = None
        self._last_plan = None
        self._item = None
        self.searched_outline: list[list[list[int]]] = []
        self.arm_poses: dict[str, dict[str, float]] = {}
        self.moved_at = 0.0
        self.version = 0  # bumped whenever what the page draws changes
        self._lock = threading.Lock()
        self._frame_used: np.ndarray | None = None
        self._stale_limits: set[str] = set()  # arms whose servo speed limit waits for the bus to be free
        # Drawn over the stream and fixed while the cell runs. Worked out here,
        # once: Kinematics is not thread-safe, and the page asks from its own
        # threads while the worker plans with it.
        self._bases_px, self._drops_px = {}, {}
        for name, placement in placements.items():
            self._bases_px[name] = list(homography.world_to_pixel(*(base_on_sheet(placement) / 100)))
            tcp = kinematics[name].tcp(drop_pose(name)) * 100
            angle = np.radians(placement.yaw)
            back = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            table = back @ (tcp[:2] - [placement.x, placement.y])
            self._drops_px[name] = list(homography.world_to_pixel(table[0] / 100, table[1] / 100))
        self._zone_px = [list(homography.world_to_pixel(x / 100, y / 100)) for x, y in rig.pick_zone.corners()]

    # --- setting up --------------------------------------------------------------

    @classmethod
    def open(cls, *, camera: str = "auto", only_arm: str | None = None, log=print) -> Cell:
        """The real cell: rig.toml, the calibration files, the arms and the overhead webcam."""

        from .__main__ import _resolve_camera
        from .arm import connect, load_poses, resolve_arm
        from .dataset.capture import open_camera
        from .kinematics import Kinematics
        from .perception.calibration import HomographyCalibration
        from .placement import Placement
        from .rig import load_rig, save_rig
        from .station import repository_root

        rig = load_rig()
        placements = {name: Placement(*devices.sheet) for name, devices in rig.arms.items() if devices.sheet}
        if only_arm:
            wanted = resolve_arm(only_arm, rig)
            placements = {name: p for name, p in placements.items() if name == wanted}
        if not placements:
            raise RuntimeError("no arm has touched the tape: uv run trashdrop rig touch left --tape")
        sheet = repository_root() / "camera_sheet.json"
        if not sheet.is_file():
            raise RuntimeError("the camera has not been calibrated: uv run trashdrop camera tape")
        classifier = None
        try:
            from .perception.classifier import ClipMaterialClassifier

            classifier = ClipMaterialClassifier()
        except (FileNotFoundError, RuntimeError) as error:
            log(f"no material classifier ({error}): the nearer arm takes every item")
        arms = {name: connect(name, rig) for name in placements}
        capture = open_camera(_resolve_camera(camera, 1920, 1080), 1920, 1080)

        def read():
            ok, frame = capture.read()
            return frame if ok else None

        cell = cls(
            rig=rig, homography=HomographyCalibration.load(sheet), placements=placements, arms=arms,
            poses=load_poses(), kinematics={name: Kinematics(rig.arms[name].wrist_roll_offset) for name in placements},
            classifier=classifier, camera=Camera(read), limits={name: arm.limits_degrees() for name, arm in arms.items()},
            log=log, save_rig=save_rig,
        )
        cell._release = capture.release
        return cell

    @classmethod
    def demo(cls, *, empty: Path, item: Path, log=print) -> Cell:
        """The page without the cell: two saved pictures and arms that only pretend to move."""

        import cv2

        from .arm import load_poses
        from .kinematics import Kinematics
        from .perception.calibration import HomographyCalibration
        from .placement import Placement
        from .rig import load_rig
        from .station import repository_root

        pictures = {"empty": cv2.imread(str(empty)), "item": cv2.imread(str(item))}
        if any(picture is None for picture in pictures.values()):
            raise RuntimeError(f"the demo needs {empty} and {item} (a pick run leaves them in out/)")
        rig = load_rig()
        placements = {name: Placement(*devices.sheet) for name, devices in rig.arms.items() if devices.sheet}
        scene = {"now": "empty", "back_at": None}

        def read():
            time.sleep(1 / 15)
            if scene["back_at"] is not None and time.monotonic() > scene["back_at"]:
                scene["now"], scene["back_at"] = "item", None  # someone tosses another one in
            return pictures[scene["now"]].copy()

        classifier = None
        try:
            from .perception.classifier import ClipMaterialClassifier

            classifier = ClipMaterialClassifier()
        except (FileNotFoundError, RuntimeError) as error:
            log(f"no material classifier ({error})")
        arms = {name: SimArm(name, rig.arms[name].max_speed) for name in placements}
        cell = cls(
            rig=rig, homography=HomographyCalibration.load(repository_root() / "camera_sheet.json"),
            placements=placements, arms=arms, poses=load_poses(),
            kinematics={name: Kinematics(rig.arms[name].wrist_roll_offset) for name in placements},
            classifier=classifier, camera=Camera(read), limits={name: {} for name in placements}, log=log,
        )
        cell._scene = scene
        return cell

    def start(self) -> None:
        self.camera.start()
        self.camera.fresh(0.0, timeout=10.0)
        self.log(f"camera running; arms: {', '.join(self.arms)}; classifier: {'yes' if self.classifier else 'no'}")

    def close(self) -> None:
        self.auto = False
        self.stop_event.set()
        self.camera.close()
        for arm in self.arms.values():
            arm.bus.close()
        if getattr(self, "_release", None):
            self._release()
        if self.classifier is not None:
            self.classifier.close()

    # --- running one action at a time ---------------------------------------------

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        self.log_lines.append(line)
        self._print(line)

    def begin(self, action: str) -> str | None:
        """Start an action on the worker thread; the reason it cannot start, if it cannot."""

        actions = {"empty": self.photograph_empty, "look": self.look, "pick": self.pick,
                   "neutral": self.neutral, "auto": self.run_auto, "relax": self.relax}
        if action not in actions:
            return f"no action {action!r}"
        if action not in ("empty", "neutral", "relax") and self.background is None:
            return "photograph the empty zone first"
        with self._lock:
            if self.busy:
                return f"busy: {self.busy}"
            self.busy = action
            self.stop_event.clear()
        self.log(STARTING[action])
        threading.Thread(target=self._run, args=(actions[action],), name=action, daemon=True).start()
        return None

    def _run(self, action) -> None:
        try:
            action()
        except KeyboardInterrupt:  # arm.Stopped: the page's stop button
            self.log("stopped: the arms hold where they are")
        except Exception as error:  # a page must say what went wrong, not die
            self.log(f"{self.busy} failed: {error}")
        finally:
            self.auto = False
            self._refresh_poses()
            with self._lock:
                self.busy = None
            self.version += 1

    def stop(self) -> str:
        """Stop what is running; the arms keep holding where they are. What happened, in words."""

        running = self.busy
        self.auto = False
        self.stop_event.set()
        message = (f"STOP: {running} stopped, the arms hold where they are" if running
                   else "STOP: nothing was moving (Relax arms lets them go limp)")
        self.log(message)
        return message

    # --- the steps --------------------------------------------------------------

    def photograph_empty(self) -> None:
        if getattr(self, "_scene", None):
            self._scene["now"] = "empty"
        frame = self.camera.fresh(max(self.moved_at + SETTLE_AFTER_MOTION_S, time.monotonic()))
        small_shape = (round(frame.shape[0] * ANALYSIS_WIDTH / frame.shape[1]), ANALYSIS_WIDTH)
        scale = frame.shape[1] / ANALYSIS_WIDTH
        zone = zone_mask(small_shape, scale, self.homography, self.rig.pick_zone)
        self.valid = zone & reach_mask(small_shape, scale, self.homography, self.placements)
        self.background = frame
        self.searched_outline = _outlines(self.valid > 0, scale)
        self.last = None
        self.version += 1
        self.log("empty zone photographed: put an item in it")
        if getattr(self, "_scene", None):
            self._scene["now"] = "item"

    def look(self, *, quiet: bool = False) -> Look:
        """Find the item, say what it is and which arm takes it, and plan the grasp. Moves nothing."""

        frame = self.camera.fresh(max(self.moved_at + SETTLE_AFTER_MOTION_S, time.monotonic()))
        self._frame_used = frame
        detection = find_item(frame, self.background, self.valid)
        look = Look(time.time(), PAGE_WORDS.get(detection.code, detection.reason), detection.code)
        self._last_plan, self._item = None, None
        self._quiet = quiet
        if detection.item is None:
            return self._seen(look)
        item, scale = refine_item(frame, self.background, detection, self.valid)
        outlines = _outlines(item, scale)
        look.outline = outlines[0] if outlines else []
        side = None
        options = self.options
        if options.material:
            side = SIDE_OF[options.material]
            look.probabilities = {options.material: 1.0}
        elif self.classifier is not None and not options.any_arm:
            look.probabilities = self.classifier.probabilities(item_crop(frame, item, scale))
            side, look.sure = side_for(look.probabilities, options.min_confidence)
            if side is None:
                look.message = "not sure what it is: leave it for a person"
                look.code = "unsure"
                return self._seen(look)
        look.side = side
        candidates = {name: p for name, p in self.placements.items()
                      if (side is None or name == side) and (options.only_arm in (None, name))}
        if not candidates:
            look.message = f"it goes {side}, but the {side} arm is not in use"
            look.code = "no_arm"
            return self._seen(look)
        plan, reason = plan_pick(item, scale, self.homography, candidates, self.kinematics, self.limits,
                                 fingertips_cm=options.fingertips_cm)
        if plan is None:
            look.message = f"cannot take it: {reason}"
            look.code = "no_plan"
            return self._seen(look)
        look.arm, look.fixed, look.moving = plan.arm, list(plan.pixel), list(plan.moving_pixel or plan.pixel)
        look.width_cm, look.open_percent = plan.grasp_plan.width_m * 100, plan.open_percent
        look.lean_deg, look.distances_cm = plan.lean_deg, dict(plan.distances_cm)
        look.message = f"the {plan.arm} arm takes it" + (f" ({reason})" if reason else "")
        self._last_plan, self._item = plan, (item, scale)
        return self._seen(look)

    def _seen(self, look: Look) -> Look:
        self.last = look
        self.version += 1
        if not getattr(self, "_quiet", False):
            self.log(look.message)
        return look

    def pick(self) -> bool:
        """Look, take the item to its side, and check the zone is empty; True if it is."""

        look = self.look()
        if self._last_plan is None:
            return False
        return self._pick_planned(look)

    def _pick_planned(self, look: Look, tries: int = PICK_TRIES) -> bool:
        """Carry out the last look's plan, then look again: True once the zone is empty.

        What is still there is looked at afresh -- classified, and given to the
        arm on its side -- since it may be something tossed in meanwhile rather
        than the item that slipped.
        """

        for attempt in range(1, tries + 1):
            self._go(self._last_plan)
            if self.options.dry_run:
                return False
            after = self.look()
            if after.code == "nothing_changed":
                self.background = self._frame_used  # empty now: keeps up with the light
                self.log("done: the zone is empty again")
                return True
            if self._last_plan is None or attempt == tries:
                if tries > 1:
                    self.log(f"still in the zone after {attempt} {'try' if attempt == 1 else 'tries'}: "
                             "leave it for a person")
                return False
            self.log(f"still in the zone: try {attempt + 1} of {tries}")
        return False

    def _go(self, plan) -> None:
        name = plan.arm
        arm = self.arms[name]
        if name in self._stale_limits and hasattr(arm, "limit_speed"):
            arm.limit_speed()
            self._stale_limits.discard(name)
        descent = getattr(self.rig.arms[name], "descent_speed", None)
        extra = {"descent_speed": descent} if descent else {}
        execute_pick(arm, plan, self.poses[name]["neutral"], kinematics=self.kinematics[name],
                     dry_run=self.options.dry_run, log=self.log, **extra)
        self.moved_at = time.monotonic()
        if getattr(self, "_scene", None) and not self.options.dry_run:
            self._scene["now"] = "empty"  # the demo's item is always caught...
            self._scene["back_at"] = time.monotonic() + DEMO_TOSS_S  # ...and another tossed in soon

    def run_auto(self) -> None:
        """Sort whatever lands in the zone, one item after another, until STOP.

        Nothing but STOP ends it: a failure is logged and it carries on.
        """

        self.auto = True
        self.log("auto: sorting whatever lands in the zone -- STOP ends it")
        said, failures = None, {}
        while self.auto and not self.stop_event.is_set():
            try:
                first = self.look(quiet=True)
                if self._last_plan is None:
                    if first.message != said:
                        self.log(first.message)
                        said = first.message
                    if first.code == "nothing_changed":
                        failures.clear()
                    self._wait(AUTO_IDLE_S)
                    continue
                self._wait(AUTO_STEADY_S)
                look = self.look(quiet=True)
                if self._last_plan is None or not _same_place(first, look):
                    continue  # still rolling, or a hand in the zone
                place = (look.arm, round(look.fixed[0] / AUTO_SAME_PLACE_PX), round(look.fixed[1] / AUTO_SAME_PLACE_PX))
                if failures.get(place, 0) >= PICK_TRIES:
                    message = "could not take the item there: take it away or move it"
                    if message != said:
                        self.log(message)
                        said = message
                    self._wait(AUTO_IDLE_S)
                    continue
                said = None
                self.log(look.message)
                if self._pick_planned(look, tries=1):
                    failures.clear()
                else:
                    failures[place] = failures.get(place, 0) + 1
            except KeyboardInterrupt:
                raise
            except Exception as error:  # non-stop: say it, and carry on
                self.log(f"auto: {error} -- carrying on")
                self._wait(2.0)
        self.log("auto: off")

    def _wait(self, seconds: float) -> None:
        """Sleep, but not through a STOP."""

        self.stop_event.wait(seconds)

    def relax(self) -> None:
        """Torque off: the arms go limp and whatever holds them up falls. A person must hold them."""

        for name, arm in self.arms.items():
            if hasattr(arm, "torque_off"):
                arm.torque_off()
        self.log("arms limp: Neutral stands them up again")

    def neutral(self) -> None:
        from .arm import Arm, move_together

        for arm in self.arms.values():
            if not arm.torque_is_on():
                arm.torque_on()
        moves = [(arm, self.poses[name]["neutral"]) for name, arm in self.arms.items()]
        if all(isinstance(arm, Arm) for arm, _ in moves):
            move_together(moves)
        else:
            for arm, pose in moves:  # the demo's pretend arms
                arm.move(pose)
        self.moved_at = time.monotonic()
        self.log("arms in neutral")

    # --- settings ---------------------------------------------------------------

    def set_options(self, changes: dict) -> None:
        known = asdict(self.options)
        for key, value in changes.items():
            if key not in known:
                raise ValueError(f"no option {key!r}")
            if key in ("fingertips_cm", "min_confidence"):
                value = float(value)
            if key == "material" and value not in (None, "", *SIDE_OF):
                raise ValueError(f"material must be one of {', '.join(SIDE_OF)}")
            if key == "only_arm" and value not in (None, "", *self.arms):
                raise ValueError(f"only_arm must be one of {', '.join(self.arms)}")
            if key in ("material", "only_arm"):
                value = value or None
            elif key in ("any_arm", "dry_run"):
                value = bool(value)
            setattr(self.options, key, value)
        self.version += 1

    def set_speeds(self, arm: str, max_speed: float, descent_speed: float, *, save: bool = True) -> str:
        """Change an arm's speeds at once, running or not: the next move uses them. What was set, in words."""

        if arm not in self.arms:
            raise ValueError(f"no arm {arm!r}")
        if not 5.0 <= max_speed <= 120.0:
            raise ValueError("travel speed: 5-120 deg/s")
        descent_speed = min(max(descent_speed, 2.0), max_speed)  # never faster than travel
        devices = self.rig.arms[arm]
        devices.max_speed = float(max_speed)
        if hasattr(devices, "descent_speed"):
            devices.descent_speed = float(descent_speed)
        self.arms[arm].max_speed = float(max_speed)
        if self.busy is None and hasattr(self.arms[arm], "limit_speed"):
            self.arms[arm].limit_speed()
        else:
            self._stale_limits.add(arm)  # the worker has the bus: written before its next pick
        if save and self._save_rig is not None:
            self._save_rig(self.rig)
        self.version += 1
        message = f"{arm} arm: {max_speed:g} deg/s, last approach {descent_speed:g} deg/s" + (
            ", saved to rig.toml" if save and self._save_rig is not None else "")
        self.log(message)
        return message

    # --- what the page shows ------------------------------------------------------

    def _refresh_poses(self) -> None:
        for name, arm in self.arms.items():
            try:
                self.arm_poses[name] = arm.pose()
            except Exception:  # a bus that stopped answering must not take the page down
                self.arm_poses.pop(name, None)

    def scene(self) -> dict:
        """What is drawn over the stream and does not change between looks, full-resolution pixels."""

        frame, _ = self.camera.latest()
        size = [frame.shape[1], frame.shape[0]] if frame is not None else [1920, 1080]
        return {"size": size, "zone": self._zone_px, "searched": self.searched_outline, "bases": self._bases_px,
                "drops": self._drops_px, "sides": dict(SIDE_OF)}

    def state(self) -> dict:
        arms = {}
        for name, arm in self.arms.items():
            devices = self.rig.arms[name]
            arms[name] = {
                "max_speed": arm.max_speed,
                "descent_speed": getattr(devices, "descent_speed", None),
                "wrist_roll_offset": devices.wrist_roll_offset,
                "pose": self.arm_poses.get(name),
            }
        return {
            "version": self.version,
            "busy": self.busy,
            "auto": self.auto,
            "empty_photographed": self.background is not None,
            "camera_fps": round(self.camera.fps, 1),
            "classifier": self.classifier is not None,
            "arms": arms,
            "options": asdict(self.options),
            "scene": self.scene(),
            "last": asdict(self.last) if self.last else None,
            "log": list(self.log_lines)[-120:],
        }


def _same_place(first: Look, second: Look) -> bool:
    """Two looks planned the same grasp, near enough: the item lies still."""

    if first.fixed is None or second.fixed is None or first.arm != second.arm:
        return False
    return float(np.hypot(first.fixed[0] - second.fixed[0], first.fixed[1] - second.fixed[1])) <= AUTO_SAME_PLACE_PX


def _outlines(mask: np.ndarray, scale: float) -> list[list[list[int]]]:
    """Outer outlines of a mask, as full-resolution pixel polygons, largest first."""

    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return [(np.asarray(contour).reshape(-1, 2) * scale + scale / 2).round().astype(int).tolist()
            for contour in contours if len(contour) >= 3]


class SimArm:
    """An arm that only pretends: every move takes a moment and lands exactly."""

    def __init__(self, name: str, max_speed: float, *, move_s: float = 0.3) -> None:
        self.name, self.max_speed, self.move_s = name, max_speed, move_s
        self.goal = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0,
                     "wrist_roll": 0.0, "gripper": 0.0}
        self.stop: threading.Event | None = None
        self.bus = type("Bus", (), {"close": lambda self: None})()

    def torque_is_on(self) -> bool:
        return True

    def torque_on(self) -> None:
        pass

    def limit_speed(self) -> None:
        pass

    def limits_degrees(self) -> dict:
        return {}

    def move(self, targets: dict[str, float], *, speed: float | None = None) -> dict[str, float]:
        from .arm import Stopped

        for _ in range(6):
            if self.stop is not None and self.stop.is_set():
                raise Stopped
            time.sleep(self.move_s / 6)
        self.goal.update(targets)
        return self.pose()

    def pose(self) -> dict[str, float]:
        return dict(self.goal)
