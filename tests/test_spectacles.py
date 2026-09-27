"""Spectacles drive the arms: the socket, the video, the wearer's frame, and each hand steering its arm.

What is pinned down: the WebSocket speaks RFC 6455 to a raw client -- the
handshake, masked text of any length, ping, close -- and answers with what
the arms do; the video sends only the newest frame and keeps a camera's own
rate; forward is where the wearer looks, whichever way round the camera's
axis comes; calibration wants both hands level and still; and on the model
at the venue's placement a hand is a joystick -- nothing moves within the
dead zone, past it the jaw goes the hand's way no faster than the top speed
and stops when the hand comes back, a fist turns the wrist and leaves it
turned, the jaw waits for the pinch after a fist, a hand out of sight holds
its arm and after a while must come back to its neutral, the jaw keeps above
the table and off the other arm's side, and no joint turns faster than
asked; and every tick writes both arms, calibrating first and again once
every hand has long been gone.
"""

from __future__ import annotations

import importlib.util
import base64
import io
import json
import re
import socket
import struct
import threading
import time
import unittest

import numpy as np

from tests.test_arm import FakeBus, FakeClock
from trashdrop.spectacles import (
    ARM_JOINTS,
    FIST,
    GRACE_S,
    CLEARANCE_CM,
    HAND_FOR,
    HEIGHT_CM,
    LEAD_CM,
    OPEN,
    RECALIBRATE_S,
    SIDE_CM,
    STALE_S,
    Calibration,
    Hands,
    JAW_DELAY_S,
    VideoFrames,
    accept_key,
    angle_delta,
    body_frame,
    curl,
    frame,
    facing_frame,
    gripper_for,
    hand_angles,
    make_server,
    make_video_server,
    heading,
    lines_apart,
    past_dead_zone,
    read_frame,
    segment_gap,
    way_back,
)

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": -90.0, "wrist_flex": 0.0, "wrist_roll": 0.0,
           "gripper": 0.0}
HEAD = {"p": [0.0, 0.0, 0.0], "look": [0.0, 0.0, -1.0]}  # the glasses' world: y up, looking along -z


def client_frame(payload: bytes, opcode: int = 1, mask: bytes = b"\x0f\x1e\x2d\x3c") -> bytes:
    size = len(payload)
    head = bytes([0x80 | opcode])
    head += bytes([0x80 | size]) if size < 126 else bytes([0x80 | 126]) + struct.pack(">H", size)
    return head + mask + bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))


def read_server_frame(connection: socket.socket) -> tuple[int, bytes]:
    head = connection.recv(2)
    size = head[1] & 0x7F
    if size == 126:
        size = struct.unpack(">H", connection.recv(2))[0]
    elif size == 127:
        size = struct.unpack(">Q", connection.recv(8))[0]
    data = b""
    while len(data) < size:
        data += connection.recv(size - len(data))
    return head[0] & 0x0F, data


def hand_at(point, gap: float = 6.0) -> dict:
    """A tracked hand whose thumb and index tips sit ``gap`` cm apart, across, around ``point``."""

    point = np.asarray(point, dtype=float)
    return {"tracked": True, "wrist": list(point + [0.0, -3.0, 8.0]),
            "thumb": list(point - [gap / 2, 0.0, 0.0]), "index": list(point + [gap / 2, 0.0, 0.0])}


def oriented_hand(point, *, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                  mirror: bool = False, gap: float = 6.0, fist: bool = False) -> dict:
    """Stationary fingertips with a palm that tilts and rotates around them; middle, ring and pinky
    straight out along the palm, or folded back into a fist."""

    hand = hand_at(point, gap)
    wrist = np.asarray(hand["wrist"])
    pitch, roll, yaw = np.radians((pitch, roll, yaw))
    fingers = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    across = right * np.cos(roll) + np.cross(fingers, right) * np.sin(roll)
    if mirror:  # The other hand's index-to-pinky direction points the other way.
        across = -across
    middle = wrist + 8 * fingers
    reach = 6.0 if fist else 17.0  # fingertips to the wrist, the palm being 8
    hand.update(middleKnuckle=list(middle), indexKnuckle=list(middle + 2 * across),
                pinkyKnuckle=list(middle - 2 * across),
                middleTip=list(wrist + reach * fingers), ringTip=list(wrist + reach * fingers - 1.0 * across),
                pinkyTip=list(wrist + reach * fingers - 2.0 * across))
    return hand


class SocketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hands = Hands()
        self.server = make_server(self.hands, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.connection = socket.create_connection(self.server.server_address, timeout=5)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()

    def test_the_rfc_example_key(self) -> None:
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_a_lens_connects_sends_hands_and_hears_what_the_arms_do(self) -> None:
        self.connection.sendall(b"GET / HTTP/1.1\r\nHost: mac\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n")
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += self.connection.recv(1024)
        self.assertIn(b" 101 ", reply)
        self.assertIn(b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", reply)

        self.connection.sendall(client_frame(json.dumps({"head": HEAD}).encode()))
        self.assertEqual(read_server_frame(self.connection), (1, b"waiting for the arms"))
        self.assertEqual(self.hands.latest()[0], {"head": HEAD})

        self.hands.status = "left arm: following"
        long = {"head": HEAD, "left": hand_at([0, -20, -35]), "padding": "x" * 300}  # past 125 bytes
        self.connection.sendall(client_frame(json.dumps(long).encode()))
        self.assertEqual(read_server_frame(self.connection), (1, b"left arm: following"))
        self.assertEqual(self.hands.latest()[0]["left"]["tracked"], True)

        self.connection.sendall(client_frame(b"still there?", opcode=9))
        self.assertEqual(read_server_frame(self.connection), (10, b"still there?"))
        self.connection.sendall(client_frame(b"", opcode=8))
        self.assertEqual(read_server_frame(self.connection)[0], 8)

    def test_a_browser_is_told_what_this_is(self) -> None:
        self.connection.sendall(b"GET / HTTP/1.1\r\nHost: mac\r\n\r\n")
        self.assertIn(b"Spectacles", self.connection.recv(1024))

    def test_large_server_frame_uses_the_64_bit_length(self) -> None:
        payload = b"x" * 100_000
        final, opcode, received = read_frame(io.BytesIO(frame(payload)))
        self.assertEqual((final, opcode, received), (True, 1, payload))


class VideoSocketTests(unittest.TestCase):
    def test_video_sends_the_newest_jpeg_with_capture_time(self) -> None:
        video = VideoFrames()
        server = make_video_server(video, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with socket.create_connection(server.server_address, timeout=5) as connection:
                connection.sendall(b"GET / HTTP/1.1\r\nHost: mac\r\nUpgrade: websocket\r\n"
                                   b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n")
                reply = b""
                while b"\r\n\r\n" not in reply:
                    reply += connection.recv(1024)
                self.assertIn(b" 101 ", reply)
                video.publish(b"old", 1000)
                video.publish(b"\xff\xd8" + b"x" * 70_000, 2000)
                opcode, raw = read_server_frame(connection)
                packet = json.loads(raw)
                self.assertEqual(opcode, 1)
                self.assertEqual(packet["capturedMs"], 2000)
                self.assertEqual(packet["seq"], 2)
                self.assertEqual(base64.b64decode(packet["jpeg"]), b"\xff\xd8" + b"x" * 70_000)
                connection.sendall(client_frame(json.dumps({"pingMs": 1234.5}).encode()))
                pong = json.loads(read_server_frame(connection)[1])
                self.assertEqual(pong["pongMs"], 1234.5)
                self.assertLess(abs(pong["serverMs"] - time.time() * 1000), 5000)
        finally:
            video.close()
            server.shutdown()
            server.server_close()


class WearerTests(unittest.TestCase):
    def test_forward_is_where_the_wearer_looks_level_and_right_is_to_its_right(self) -> None:
        forward, right, up = facing_frame([0.0, -0.3, -1.0], [0, 0, 0], [0, -20, -35])
        np.testing.assert_allclose(forward, [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(right, [1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(up, [0, 1, 0])

    def test_the_hands_settle_which_way_round_the_cameras_axis_is(self) -> None:
        forward, right, _ = facing_frame([0.0, 0.0, 1.0], [0, 0, 0], [0, -20, -35])
        np.testing.assert_allclose(forward, [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(right, [1, 0, 0], atol=1e-9)

    def test_thumb_and_index_open_and_close_the_jaw(self) -> None:
        self.assertEqual(gripper_for(1.0), 0.0)
        self.assertEqual(gripper_for(12.0), OPEN)
        self.assertGreater(gripper_for(6.0), 0.0)
        self.assertLess(gripper_for(6.0), OPEN)

    def test_in_front_of_the_arms_each_hand_drives_the_arm_on_its_side(self) -> None:
        self.assertEqual(HAND_FOR["same"], {"left": "left", "right": "right"})
        self.assertEqual(HAND_FOR["them"], {"left": "right", "right": "left"})

    def test_knuckles_measure_roll_and_pitch_independently(self) -> None:
        wearer = facing_frame(HEAD["look"], HEAD["p"], [0, -20, -35])
        measured = hand_angles(oriented_hand([0, -20, -35], roll=25, pitch=-18), wearer)
        np.testing.assert_allclose(measured, [25, -18], atol=1e-8)
        yawed = hand_angles(oriented_hand([0, -20, -35], yaw=60), wearer)
        np.testing.assert_allclose(yawed, [0, 0], atol=1e-8)
        mirrored_start = hand_angles(oriented_hand([0, -20, -35], mirror=True), wearer)[0]
        mirrored_turn = hand_angles(oriented_hand([0, -20, -35], roll=25, mirror=True), wearer)[0]
        self.assertAlmostEqual(angle_delta(mirrored_turn, mirrored_start), 25)
        self.assertIsNone(hand_angles(hand_at([0, -20, -35]), wearer))




class VideoPacingTests(unittest.TestCase):
    def read_with(self, period: float, reads: int) -> VideoFrames:
        clock = FakeClock()
        video = VideoFrames(fps=30, clock=clock, wall_clock=lambda: 0.0)
        jitter = (0.0, -0.004, 0.003, -0.002, 0.004)  # a real camera's frames come a few ms early or late

        class Camera:
            count = 0

            def read(self):
                self.count += 1
                clock.now += period + jitter[self.count % len(jitter)]
                if self.count >= reads:
                    video.running = False
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        video.capture = Camera()
        video._read()
        return video

    def test_a_camera_at_the_frame_rate_loses_no_frame_to_its_jitter(self) -> None:
        self.assertEqual(self.read_with(1 / 30, 60).sent_count, 60)

    def test_a_faster_camera_is_thinned_to_the_frame_rate(self) -> None:
        self.assertAlmostEqual(self.read_with(1 / 60, 120).sent_count, 60, delta=2)

    def test_the_report_says_the_rates_and_starts_counting_afresh(self) -> None:
        video = self.read_with(1 / 30, 30)
        camera, sent = (float(value) for value in re.findall(r"([\d.]+) fps", video.report()))
        self.assertAlmostEqual(camera, 30.0, delta=0.5)
        self.assertAlmostEqual(sent, 30.0, delta=0.5)
        self.assertEqual((video.read_count, video.sent_count), (0, 0))


class HandShapeTests(unittest.TestCase):
    def test_curl_tells_a_fist_from_an_open_hand_and_a_pinch(self) -> None:
        self.assertGreater(curl(oriented_hand([0, -20, -35])), FIST[1])
        self.assertGreater(curl(oriented_hand([0, -20, -35], gap=1.0)), FIST[1], "a pinch is no fist")
        self.assertLess(curl(oriented_hand([0, -20, -35], fist=True)), FIST[0])
        self.assertIsNone(curl(hand_at([0, -20, -35])), "an older Lens sends no fingertips")

    def test_the_dead_zone_takes_its_width_off_every_axis(self) -> None:
        np.testing.assert_allclose(past_dead_zone(np.array([2.0, -5.0, 7.5]), 3.0), [0.0, -2.0, 4.5])

    def test_the_way_back_names_the_farthest_axis(self) -> None:
        self.assertEqual(way_back(np.array([1.0, 0.5, -6.2])), "up 6 cm")
        self.assertEqual(way_back(np.array([4.0, -1.0, 2.0])), "closer 4 cm")
        self.assertEqual(way_back(np.array([0.0, -5.0, 1.0])), "right 5 cm")


class CalibrationTests(unittest.TestCase):
    LEVEL = {"left": [-15.0, -20.0, -35.0], "right": [15.0, -22.0, -35.0]}

    def ticks_until_done(self, calibration: Calibration, wrists, limit: int = 400) -> tuple[int, dict | None]:
        for tick in range(1, limit + 1):
            neutral = calibration.update(wrists, 0.02)
            if neutral is not None:
                return tick, neutral
        return limit, None

    def test_both_hands_held_level_and_still_set_where_they_rest(self) -> None:
        calibration = Calibration(("left", "right"), hold_s=1.0)
        calibration.update(self.LEVEL, 0.02)
        self.assertEqual(calibration.status, "calibrate: hold still 1")
        ticks, neutral = self.ticks_until_done(calibration, self.LEVEL)
        self.assertAlmostEqual(ticks + 1, 50, delta=1)
        np.testing.assert_allclose(neutral["left"], self.LEVEL["left"])
        np.testing.assert_allclose(neutral["right"], self.LEVEL["right"])

    def test_a_hand_that_moves_starts_the_count_again(self) -> None:
        calibration = Calibration(("left", "right"), hold_s=1.0)
        for _ in range(40):
            calibration.update(self.LEVEL, 0.02)
        moved = dict(self.LEVEL, right=[15.0, -22.0, -27.0])  # 8 cm nearer
        ticks, _ = self.ticks_until_done(calibration, moved)
        self.assertAlmostEqual(ticks, 50, delta=1)

    def test_hands_at_different_heights_are_not_level(self) -> None:
        calibration = Calibration(("left", "right"), hold_s=0.2)
        uneven = dict(self.LEVEL, right=[15.0, 0.0, -35.0])
        self.assertEqual(self.ticks_until_done(calibration, uneven, limit=50), (50, None))
        self.assertEqual(calibration.status, "calibrate: right hand 20 cm higher, level them")

    def test_a_hand_out_of_sight_is_asked_for(self) -> None:
        calibration = Calibration(("left", "right"), hold_s=0.2)
        self.assertIsNone(calibration.update(dict(self.LEVEL, left=None), 0.02))
        self.assertEqual(calibration.status, "calibrate: both hands up in the air, level, still")
        one = Calibration(("right",), hold_s=0.2)
        one.update({"right": None}, 0.02)
        self.assertIn("the right hand", one.status)
        self.assertIsNotNone(self.ticks_until_done(one, {"right": [15.0, -22.0, -35.0]})[1])


REST = [0.0, -20.0, -35.0]  # thumb and index at rest; the wrist is 3 cm lower and 8 cm nearer


def resting(offset=(0.0, 0.0, 0.0), **shape) -> dict:
    """The hand ``offset`` cm (x right, y up, z back) from where it rested at calibration."""

    return oriented_hand(np.add(REST, offset), **shape)


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class FollowerTests(unittest.TestCase):
    """The venue's left arm, on the model, calibrated with the hand at REST."""

    @classmethod
    def setUpClass(cls) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.placement import Placement
        from trashdrop.spectacles import ready_pose

        cls.kinematics = Kinematics(-80.0)
        cls.placement = Placement(11.34, -22.40, -95.14, -0.36, -0.0318, 0.0617)
        cls.limits = cls.kinematics.own_limits()
        cls.ready = ready_pose(cls.kinematics, cls.placement, NEUTRAL, cls.limits)

    def follower(self, name: str = "left", head=HEAD, mirror: bool = False, **options):
        from trashdrop.spectacles import Follower

        if name == "left":
            follower = Follower(name, self.kinematics, self.placement, self.ready, limits=self.limits, **options)
        else:
            from trashdrop.kinematics import Kinematics
            from trashdrop.placement import Placement
            from trashdrop.spectacles import ready_pose

            model = Kinematics(5.0)  # the right arm's measured wrist zero in rig.toml
            placement = Placement(10.98, 19.12, -79.47, -1.02, -0.0315, -0.0252)
            limits = model.own_limits()
            follower = Follower(name, model, placement, ready_pose(model, placement, NEUTRAL, limits),
                                limits=limits, **options)
        wrist = np.asarray(resting(mirror=mirror)["wrist"])
        follower.engage(wrist, facing_frame(head["look"], head["p"], wrist))
        return follower

    def run_for(self, follower, hand, seconds: float = 1.0) -> None:
        for _ in range(round(seconds / 0.02)):
            follower.update(hand, 0.02)

    def test_the_ready_pose_is_fingers_down_over_the_table(self) -> None:
        fingers, _ = self.kinematics.pointing(self.ready)
        self.assertLess(fingers[2], -0.7)
        tcp = self.kinematics.tcp(self.ready) * 100
        self.assertGreater(tcp[2], self.placement.table_height(*tcp[:2]) + 5.0)

    def test_within_the_dead_zone_nothing_moves(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, resting((1.5, -2.5, 2.0)), seconds=2.0)
        np.testing.assert_allclose(follower.tcp_cm(), start, atol=0.05)
        self.assertEqual(follower.state, "holding")

    def test_a_hand_held_lower_drives_the_jaw_down_and_back_at_rest_it_stops(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, resting((0.0, -6.0, 0.0)))  # 3 cm past the dead zone: 4.5 cm/s
        self.assertEqual(follower.state, "moving down")
        moved = follower.tcp_cm() - start
        self.assertAlmostEqual(moved[2], -4.5, delta=1.0)
        np.testing.assert_allclose(moved[:2], [0.0, 0.0], atol=0.5)
        self.run_for(follower, resting(), seconds=0.5)
        stopped = follower.tcp_cm()
        self.assertLess(np.linalg.norm(stopped - (start + moved)), LEAD_CM + 0.5)
        self.run_for(follower, resting(), seconds=1.0)
        np.testing.assert_allclose(follower.tcp_cm(), stopped, atol=0.1)
        self.assertEqual(follower.state, "holding")

    def test_down_and_to_the_right_moves_the_jaw_down_and_to_the_right(self) -> None:
        for facing, right_sign in (("same", -1.0), ("them", 1.0)):
            with self.subTest(facing=facing):
                follower = self.follower(facing=facing)
                start = follower.tcp_cm()
                self.run_for(follower, resting((6.0, -6.0, 0.0)), seconds=0.8)
                self.assertEqual(follower.state, "moving right, down", "what the glasses show")
                moved = follower.tcp_cm() - start
                self.assertGreater(moved[1] * right_sign, 1.5, "to the wearer's right")
                self.assertLess(moved[2], -1.5, "down")

    def test_the_camera_axis_the_other_way_round_changes_nothing(self) -> None:
        follower = self.follower(head={"p": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]})
        start = follower.tcp_cm()
        self.run_for(follower, resting((0.0, 0.0, -7.0)))  # 7 cm ahead
        self.assertGreater(follower.tcp_cm()[0] - start[0], 2.0, "still forward")

    def test_the_jaw_moves_no_faster_than_the_top_speed(self) -> None:
        follower = self.follower(top_speed=4.0)
        start = follower.tcp_cm()
        self.run_for(follower, resting((0.0, 0.0, -30.0)))  # far ahead
        self.assertLess(np.linalg.norm(follower.tcp_cm() - start), 4.0 + 0.3)

    def test_a_target_never_runs_far_ahead_of_the_jaw(self) -> None:
        follower = self.follower(speed=5.0, top_speed=20.0)  # joints far slower than the target
        self.run_for(follower, resting((0.0, 0.0, -30.0)))
        self.assertLessEqual(np.linalg.norm(follower.target - follower.tcp_cm()), LEAD_CM + 1e-6)

    def test_a_fist_turns_the_wrist_and_leaves_it_turned(self) -> None:
        for name in ("left", "right"):
            with self.subTest(name=name):
                mirror = name == "right"
                follower = self.follower(name=name, mirror=mirror)
                start = follower.q["wrist_roll"]
                self.run_for(follower, resting(fist=True, mirror=mirror), seconds=0.2)
                self.run_for(follower, resting(fist=True, roll=30, mirror=mirror))
                self.assertEqual(follower.state, "turning the jaw")
                self.assertAlmostEqual(follower.q["wrist_roll"] - start, 30, delta=1.5)
                self.run_for(follower, resting(roll=0, mirror=mirror))  # open, and the hand turned back
                self.assertAlmostEqual(follower.q["wrist_roll"] - start, 30, delta=1.5)
                self.run_for(follower, resting(fist=True, mirror=mirror), seconds=0.2)
                self.run_for(follower, resting(fist=True, roll=-20, mirror=mirror))
                self.assertAlmostEqual(follower.q["wrist_roll"] - start, 10, delta=1.5, msg="turn by turn")

    def test_large_wrist_turns_stop_at_the_calibrated_limits(self) -> None:
        follower = self.follower()
        low, high = follower.limits["wrist_roll"]
        for roll in (170, -170):
            self.run_for(follower, resting(fist=True), seconds=0.2)
            self.run_for(follower, resting(fist=True, roll=roll), seconds=2.0)
            self.run_for(follower, resting(), seconds=0.2)
            self.assertGreaterEqual(follower.q["wrist_roll"], low)
            self.assertLessEqual(follower.q["wrist_roll"], high)

    def test_while_the_fist_is_closed_nothing_else_moves(self) -> None:
        follower = self.follower()
        self.run_for(follower, resting(gap=12.0), seconds=0.5)
        start, jaw = follower.tcp_cm(), follower.q["gripper"]
        self.run_for(follower, resting((0.0, -6.0, 0.0), fist=True, gap=1.0))
        np.testing.assert_allclose(follower.tcp_cm(), start, atol=0.3)
        self.assertEqual(follower.q["gripper"], jaw)

    def test_after_a_fist_the_jaw_waits_for_the_pinch(self) -> None:
        follower = self.follower()
        self.run_for(follower, resting(gap=1.0), seconds=0.5)
        self.assertAlmostEqual(follower.q["gripper"], 0.0)  # holding something
        self.run_for(follower, resting(fist=True, gap=1.0), seconds=0.3)
        self.run_for(follower, resting(gap=12.0))  # the hand opens after the fist
        self.assertAlmostEqual(follower.q["gripper"], 0.0, msg="what it holds stays held")
        self.assertIn("jaw waits for a pinch", follower.state)
        self.run_for(follower, resting(gap=1.0), seconds=0.2)  # thumb and index agree: the jaw is theirs again
        self.run_for(follower, resting(gap=12.0))
        self.assertAlmostEqual(follower.q["gripper"], OPEN)

    def test_the_jaw_opens_and_closes_with_thumb_and_index(self) -> None:
        follower = self.follower()
        self.run_for(follower, resting(gap=12.0))
        self.assertAlmostEqual(follower.q["gripper"], OPEN)
        self.run_for(follower, resting(gap=1.0))
        self.assertAlmostEqual(follower.q["gripper"], 0.0)

    def test_a_hand_out_of_sight_for_a_moment_holds_its_arm_and_carries_on(self) -> None:
        follower = self.follower()
        self.run_for(follower, resting((0.0, -6.0, 0.0)), seconds=0.5)
        self.run_for(follower, resting(), seconds=0.5)  # settle: nothing still on its way
        held = dict(follower.q)
        self.run_for(follower, None, seconds=GRACE_S - 0.1)
        self.assertEqual(follower.q, held)
        self.assertEqual(follower.state, "no hand")
        before = follower.tcp_cm()
        self.run_for(follower, resting((0.0, -6.0, 0.0)), seconds=0.5)
        self.assertLess(follower.tcp_cm()[2], before[2] - 1.0, "no need to come back to the middle")

    def test_after_a_while_out_of_sight_the_arm_waits_for_the_hand_at_its_neutral(self) -> None:
        follower = self.follower()
        self.run_for(follower, None, seconds=GRACE_S + 0.5)
        held = dict(follower.q)
        self.run_for(follower, resting((0.0, -6.0, 0.0)))
        self.assertEqual(follower.q, held)
        self.assertEqual(follower.state, "back to the middle: up 6 cm")
        self.run_for(follower, resting(), seconds=0.2)
        self.assertTrue(follower.state.startswith("holding"))
        before = follower.tcp_cm()
        self.run_for(follower, resting((0.0, -6.0, 0.0)), seconds=0.5)
        self.assertLess(follower.tcp_cm()[2], before[2] - 1.0)

    def test_never_into_the_table_nor_onto_the_other_arms_side(self) -> None:
        follower = self.follower(top_speed=20.0)
        self.run_for(follower, resting((40.0, -40.0, 0.0)), seconds=5.0)  # far right and far down
        x, y, z = follower.target
        self.assertGreaterEqual(y, -SIDE_CM - 1e-9, "the right arm is that way")
        self.assertGreaterEqual(z, self.placement.table_height(x, y) + HEIGHT_CM[0] - 1e-9)

    def test_no_joint_turns_faster_than_asked(self) -> None:
        follower = self.follower(speed=60.0, top_speed=20.0)
        self.run_for(follower, resting((10.0, -10.0, -10.0)), seconds=0.3)
        before = dict(follower.q)
        follower.update(resting((10.0, -10.0, -10.0)), 0.02)
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
            self.assertLessEqual(abs(follower.q[joint] - before[joint]), 60.0 * 0.02 + 1e-9)

    def test_a_calibrating_arm_holds(self) -> None:
        from trashdrop.spectacles import Follower

        follower = Follower("left", self.kinematics, self.placement, self.ready, limits=self.limits)
        self.run_for(follower, resting((0.0, -10.0, 0.0)))
        self.assertEqual(follower.q, self.ready)
        self.assertEqual(follower.state, "calibrating")


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class FollowTests(unittest.TestCase):
    def arms_and_followers(self, clock):
        from trashdrop.arm import Arm
        from trashdrop.kinematics import Kinematics
        from trashdrop.spectacles import Follower

        buses = {name: FakeBus() for name in ("left", "right")}
        arms = {name: Arm(name, bus, 45.0, clock=clock, sleep=clock.sleep) for name, bus in buses.items()}
        followers = {name: Follower(name, Kinematics(), None, dict(NEUTRAL, shoulder_lift=30.0, wrist_flex=60.0))
                     for name in arms}
        return buses, arms, followers

    def test_the_hands_calibrate_then_every_tick_writes_both_arms_until_they_go_stale(self) -> None:
        from trashdrop.spectacles import follow

        clock = FakeClock(stop_after=40)  # 0.8 s of ticks, then Ctrl+C
        buses, arms, followers = self.arms_and_followers(clock)
        hands = Hands(clock=clock)
        hands.put({"head": HEAD, "left": hand_at([-15, -20, -35]), "right": hand_at([15, -20, -35])})
        states = []
        with self.assertRaises(KeyboardInterrupt):
            follow(followers, hands, arms, hold_s=0.2, log=states.append, clock=clock, sleep=clock.sleep)
        self.assertEqual(len(buses["left"].goals_streamed()), len(buses["right"].goals_streamed()))
        self.assertGreaterEqual(len(buses["left"].goals_streamed()), 39)
        states = [state.removesuffix("; the glasses are not connected") for state in states]
        self.assertEqual(states[0], "calibrate: hold still 1")
        self.assertIn("left: holding | right: holding", states)
        self.assertEqual(states[-1], "left: no hand | right: no hand", f"stale after {STALE_S} s")

    def test_every_hand_gone_long_enough_means_calibrating_again(self) -> None:
        from trashdrop.spectacles import follow

        clock = FakeClock(stop_after=round((0.3 + RECALIBRATE_S) / 0.02) + 10)
        _, _, followers = self.arms_and_followers(clock)
        hands = Hands(clock=clock)
        hands.put({"head": HEAD, "left": hand_at([-15, -20, -35]), "right": hand_at([15, -20, -35])})
        states = []
        with self.assertRaises(KeyboardInterrupt):
            follow(followers, hands, None, hold_s=0.2, log=states.append, clock=clock, sleep=clock.sleep)
        statuses = [state.removesuffix("; the glasses are not connected") for state in states
                    if not state.startswith("  ")]
        self.assertIn("left: holding | right: holding (dry run)", statuses)
        self.assertTrue(statuses[-1].startswith("calibrate: both hands up"), statuses[-1])
        self.assertTrue(all(follower.neutral is None for follower in followers.values()))


def blend(start: dict, end: dict, share: float) -> dict:
    """A hand ``share`` of the way from ``start`` to ``end``, every keypoint in a straight line."""

    return {key: (list(np.add(np.multiply(start[key], 1 - share), np.multiply(end[key], share)))
                  if isinstance(start[key], list) else end[key]) for key in end}


class BodyFrameTests(unittest.TestCase):
    LEFT, RIGHT = [-15.0, -20.0, -35.0], [15.0, -22.0, -35.0]

    def test_right_runs_from_the_left_hand_to_the_right_wherever_the_wearer_looked(self) -> None:
        for look in ([0.0, 0.0, -1.0], [0.87, -0.3, -0.5], [-0.87, 0.0, -0.5]):  # ahead, 60 deg right, 60 deg left
            with self.subTest(look=look):
                forward, right, up = body_frame(self.LEFT, self.RIGHT, look, [0.0, 0.0, 0.0])
                np.testing.assert_allclose(right, [1.0, 0.0, 0.0], atol=1e-9)
                np.testing.assert_allclose(forward, [0.0, 0.0, -1.0], atol=1e-9)
                np.testing.assert_allclose(up, [0.0, 1.0, 0.0])

    def test_hands_crossed_or_together_leave_it_to_the_gaze(self) -> None:
        _, crossed, _ = body_frame(self.RIGHT, self.LEFT, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(crossed, [1.0, 0.0, 0.0], atol=1e-9)
        together = body_frame([-2.0, -20.0, -35.0], [2.0, -20.0, -35.0], [0.87, 0.0, -0.5], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(together, facing_frame([0.87, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, -20.0, -35.0]))

    def test_the_heading_names_each_axis_past_the_dead_zone(self) -> None:
        self.assertEqual(heading(np.array([0.5, 6.0, -5.0]), 3.0), "right, down")
        self.assertEqual(heading(np.array([-4.0, -3.5, 1.0]), 3.0), "back, left")
        self.assertEqual(heading(np.array([1.0, 2.0, -2.9]), 3.0), "")


class RecordTests(unittest.TestCase):
    def test_every_message_is_kept_with_the_bridges_time(self) -> None:
        clock = FakeClock()
        record = io.StringIO()
        hands = Hands(clock=clock, record=record)
        hands.put({"t": 1.5, "left": {"tracked": False}})
        clock.now = 0.25
        hands.put({"t": 1.6, "right": {"tracked": False}})
        kept = record.getvalue()
        hands.stop_recording()
        self.assertTrue(record.closed)
        hands.put({"t": 1.7})  # nothing more is written: the closed record would refuse it
        lines = [json.loads(line) for line in kept.splitlines()]
        self.assertEqual([line["at"] for line in lines], [0.0, 0.25])
        self.assertEqual(lines[1]["right"], {"tracked": False})


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class FistAndJawTests(unittest.TestCase):
    """Thumb and index meet on the way into a fist: the jaw must not take that for a pinch."""

    @classmethod
    def setUpClass(cls) -> None:
        FollowerTests.setUpClass.__func__(cls)

    follower = FollowerTests.follower
    run_for = FollowerTests.run_for

    def test_closing_the_hand_into_a_fist_leaves_the_jaw_as_it_was(self) -> None:
        follower = self.follower()
        open_hand, fist = resting(gap=12.0), resting(fist=True, gap=1.0)
        self.run_for(follower, open_hand)
        self.assertAlmostEqual(follower.q["gripper"], OPEN)
        closing_s = JAW_DELAY_S - 0.1  # thumb and index meet a moment before the fist is told
        steps = round(closing_s / 0.02)
        for step in range(steps):
            share = (step + 1) / steps
            hand = blend(open_hand, fist, share)
            if share < 1.0:  # still folding: middle, ring and pinky not in yet
                hand.update({key: open_hand[key] for key in ("middleTip", "ringTip", "pinkyTip")})
            follower.update(hand, 0.02)
        self.run_for(follower, resting(fist=True, gap=1.0, roll=30))
        self.assertEqual(follower.state, "turning the jaw")
        self.assertAlmostEqual(follower.q["gripper"], OPEN, delta=1.0, msg="the fist gripped nothing")
        self.run_for(follower, open_hand)
        self.assertAlmostEqual(follower.q["gripper"], OPEN, delta=1.0)
        self.assertEqual(follower.state, "holding", "open again, the jaw is the pinch's at once")

    def test_a_pinch_reaches_the_jaw_after_the_delay(self) -> None:
        follower = self.follower()
        self.run_for(follower, resting(gap=12.0))
        self.run_for(follower, resting(gap=1.0), seconds=JAW_DELAY_S - 0.06)
        self.assertAlmostEqual(follower.q["gripper"], OPEN, msg="not yet")
        self.run_for(follower, resting(gap=1.0), seconds=0.6)
        self.assertAlmostEqual(follower.q["gripper"], 0.0)


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class CalibratedFrameTests(unittest.TestCase):
    def test_after_calibrating_with_the_head_turned_right_is_still_the_hands_right(self) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.spectacles import Follower, follow

        clock = FakeClock(stop_after=60)
        followers = {name: Follower(name, Kinematics(), None, dict(NEUTRAL, shoulder_lift=30.0, wrist_flex=60.0))
                     for name in ("left", "right")}
        hands = Hands(clock=clock)
        turned = {"p": [0.0, 0.0, 0.0], "look": [0.87, 0.0, -0.5]}  # looking 60 degrees to the right
        left, right = [-15.0, -20.0, -35.0], [15.0, -20.0, -35.0]
        ticks = [0]

        def message() -> dict:
            moved = np.add(right, [6.0, 0.0, 0.0]) if ticks[0] > 15 else right  # calibrated, then 6 cm right
            return {"head": turned, "left": hand_at(left), "right": hand_at(moved)}

        def sleep(seconds: float) -> None:
            clock.sleep(seconds)
            ticks[0] += 1
            hands.put(message())

        hands.put(message())
        start = {}
        with self.assertRaises(KeyboardInterrupt):
            follow(followers, hands, None, hold_s=0.2, log=lambda line: start.setdefault(
                "tcp", followers["right"].tcp_cm()) if "holding" in line else None, clock=clock, sleep=sleep)
        moved = followers["right"].tcp_cm() - start["tcp"]
        self.assertLess(moved[1], -1.0, "to the arm's right")
        self.assertLess(abs(moved[0]), 0.5, "not forward nor back")


class GapTests(unittest.TestCase):
    def test_how_close_two_segments_come(self) -> None:
        self.assertAlmostEqual(segment_gap([0, 0, 0], [10, 0, 0], [5, 3, 0], [5, 3, 8]), 3.0)  # across, apart
        self.assertAlmostEqual(segment_gap([0, 0, 0], [10, 0, 0], [0, 4, 0], [10, 4, 0]), 4.0)  # side by side
        self.assertAlmostEqual(segment_gap([0, 0, 0], [10, 0, 0], [13, 4, 0], [20, 4, 0]), 5.0)  # end to end
        self.assertAlmostEqual(segment_gap([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0]), 5.0)  # two points
        self.assertAlmostEqual(segment_gap([0, 0, 0], [10, 0, 0], [5, -5, 0], [5, 5, 0]), 0.0)  # crossing

    def test_how_close_two_lines_come(self) -> None:
        one = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
        other = np.array([[30.0, 0.0, 0.0], [30.0, 0.0, 10.0], [16.0, 0.0, 10.0]])
        self.assertAlmostEqual(lines_apart(one, other), 6.0)


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class TwoArmTests(unittest.TestCase):
    """Both of the venue's arms, placed on the sheet: each may reach into the middle, never into the other."""

    @classmethod
    def setUpClass(cls) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.placement import Placement
        from trashdrop.spectacles import ready_pose

        cls.models = {"left": Kinematics(-80.0), "right": Kinematics(5.0)}
        cls.placements = {"left": Placement(11.34, -22.40, -95.14, -0.36, -0.0318, 0.0617),
                          "right": Placement(10.98, 19.12, -79.47, -1.02, -0.0315, -0.0252)}
        cls.limits = {name: model.own_limits() for name, model in cls.models.items()}
        cls.ready = {name: ready_pose(model, cls.placements[name], NEUTRAL, cls.limits[name])
                     for name, model in cls.models.items()}

    def pair(self):
        from trashdrop.spectacles import Follower

        followers = {name: Follower(name, self.models[name], self.placements[name], self.ready[name],
                                    limits=self.limits[name], top_speed=20.0) for name in ("left", "right")}
        for name, follower in followers.items():
            wrist = np.asarray(resting()["wrist"])
            follower.engage(wrist, facing_frame(HEAD["look"], HEAD["p"], wrist))
        return followers

    def steer(self, followers, hands: dict, seconds: float) -> None:
        for _ in range(round(seconds / 0.02)):
            lines = {name: follower.centre_line() for name, follower in followers.items()}
            for name, follower in followers.items():
                others = [line for other, line in lines.items() if other != name]
                follower.update(hands[name], 0.02, others=others)
                lines[name] = follower.centre_line()

    def test_an_arm_reaches_past_its_old_side_while_the_other_is_away(self) -> None:
        followers = self.pair()
        self.steer(followers, {"left": resting(), "right": resting((-9.0, 0.0, 0.0))}, seconds=6.0)
        self.assertGreater(followers["right"].tcp_cm()[1], SIDE_CM + 5.0, "well into the middle")

    def test_the_arms_centre_lines_never_come_nearer_than_the_clearance(self) -> None:
        followers = self.pair()
        hands = {"left": resting((9.0, 0.0, 0.0)), "right": resting((-9.0, 0.0, 0.0))}  # towards each other
        for _ in range(8):
            self.steer(followers, hands, seconds=1.0)
            gap = lines_apart(followers["left"].centre_line(), followers["right"].centre_line())
            self.assertGreaterEqual(gap, CLEARANCE_CM - 0.05)
        self.assertEqual(followers["right"].state, "moving left: at the other arm")
        self.assertEqual(followers["left"].state, "moving right: at the other arm")

    def test_away_from_the_other_arm_it_moves_again_at_once(self) -> None:
        followers = self.pair()
        self.steer(followers, {"left": resting((9.0, 0.0, 0.0)), "right": resting((-9.0, 0.0, 0.0))}, seconds=8.0)
        before = followers["right"].tcp_cm()
        self.steer(followers, {"left": resting(), "right": resting((9.0, 0.0, 0.0))}, seconds=0.5)
        self.assertLess(followers["right"].tcp_cm()[1], before[1] - 1.0)

    def test_the_glasses_say_what_stopped_an_arm(self) -> None:
        followers = self.pair()
        self.steer(followers, {"left": resting((0.0, -12.0, 0.0)), "right": resting()}, seconds=4.0)
        self.assertEqual(followers["left"].state, "moving down: at the table")

    def test_driving_one_arm_alone_it_keeps_to_its_own_side(self) -> None:
        from trashdrop.spectacles import Follower

        follower = Follower("right", self.models["right"], self.placements["right"], self.ready["right"],
                            limits=self.limits["right"], top_speed=20.0)
        wrist = np.asarray(resting()["wrist"])
        follower.engage(wrist, facing_frame(HEAD["look"], HEAD["p"], wrist))
        for _ in range(300):
            follower.update(resting((-9.0, 0.0, 0.0)), 0.02)
        self.assertLessEqual(follower.tcp_cm()[1], SIDE_CM + 0.5)
        self.assertEqual(follower.state, "moving left: at the other arm's side")


def pinching(offset=(0.0, 0.0, 0.0), *, pinched: bool = True, roll: float = 0.0, pinky: bool = False,
             told=None) -> dict:
    """The hand ``offset`` cm from REST, thumb and index together or apart; the pinky on the thumb, or not;
    and the glasses' own pinch detection, when ``told``."""

    hand = oriented_hand(np.add(REST, offset), roll=roll, gap=1.0 if pinched else 12.0)
    if pinky:
        hand["pinkyTip"] = list(np.add(hand["thumb"], [0.5, -1.0, 0.5]))
    if told is not None:
        hand["pinch"] = told
    return hand


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class PinchTests(unittest.TestCase):
    """--mode pinch on the venue's left arm: thumb and index grab the jaw, thumb and pinky open or close it."""

    @classmethod
    def setUpClass(cls) -> None:
        FollowerTests.setUpClass.__func__(cls)

    def follower(self, **options):
        from trashdrop.spectacles import PinchFollower

        return PinchFollower("left", self.kinematics, self.placement, self.ready, limits=self.limits, **options)

    def run_for(self, follower, hand, seconds: float = 1.0) -> None:
        for _ in range(round(seconds / 0.02)):
            follower.update(hand, 0.02, head=HEAD)

    def glide(self, follower, start, end, seconds: float = 1.0, **shape) -> None:
        """The hand from ``start`` to ``end`` (offsets from REST) at an even pace, then a moment still."""

        steps = round(seconds / 0.02)
        for step in range(steps):
            follower.update(pinching(np.add(start, np.subtract(end, start) * (step + 1) / steps), **shape), 0.02,
                            head=HEAD)
        self.run_for(follower, pinching(end, **shape), seconds=1.0)

    def test_an_open_hand_moves_nothing(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, pinching(pinched=False))
        self.glide(follower, (0.0, 0.0, 0.0), (6.0, -6.0, 0.0), pinched=False)
        np.testing.assert_allclose(follower.tcp_cm(), start, atol=1e-9)
        self.assertEqual(follower.state, "free, jaw closed")

    def test_a_pinch_grabs_the_jaw_and_drags_it_as_far_as_the_hand_goes(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -5.0, 0.0))
        self.assertTrue(follower.state.startswith("dragging"))
        moved = follower.tcp_cm() - start
        np.testing.assert_allclose(moved, [0.0, 0.0, -5.0], atol=0.8)

    def test_right_and_forward_are_as_the_wearer_looked(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (5.0, 0.0, -5.0))  # 5 cm right, 5 cm ahead
        np.testing.assert_allclose(follower.tcp_cm() - start, [5.0, -5.0, 0.0], atol=0.8)

    def test_letting_go_leaves_the_arm_and_the_next_pinch_goes_on_from_there(self) -> None:
        follower = self.follower()
        start = follower.tcp_cm()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -4.0, 0.0))
        self.run_for(follower, pinching((0.0, -4.0, 0.0), pinched=False), seconds=0.3)
        let_go = follower.tcp_cm()
        self.glide(follower, (0.0, -4.0, 0.0), (0.0, 0.0, 0.0), pinched=False)  # the free hand goes back up
        np.testing.assert_allclose(follower.tcp_cm(), let_go, atol=1e-9)
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -2.0, 0.0))
        self.assertAlmostEqual(follower.tcp_cm()[2] - start[2], -6.0, delta=1.0)

    def test_the_scale_shrinks_the_drag(self) -> None:
        follower = self.follower(scale=0.5)
        start = follower.tcp_cm()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, 0.0, -8.0))
        self.assertAlmostEqual(follower.tcp_cm()[0] - start[0], 4.0, delta=0.8)

    def jaw_turn(self, follower, before: dict[str, float]) -> float:
        """How far the jaw has turned since ``before`` about where it points, degrees by the right-hand rule."""

        pointing, across = self.kinematics.pointing({joint: before[joint] for joint in ARM_JOINTS})
        _, turned = self.kinematics.pointing({joint: follower.q[joint] for joint in ARM_JOINTS})
        return float(np.degrees(np.arctan2(np.cross(across, turned) @ pointing, across @ turned)))

    def test_a_twist_in_place_turns_the_jaw_the_same_way(self) -> None:
        follower = self.follower()
        before = dict(follower.q)
        self.run_for(follower, pinching(), seconds=0.3)
        self.run_for(follower, pinching(roll=6.0))
        self.assertEqual(follower.state, "pinched, jaw closed", "not clear yet what this pinch does")
        self.assertAlmostEqual(follower.q["wrist_roll"], before["wrist_roll"], delta=0.01)
        # 30 degrees by the right-hand rule about the fingers: clockwise, seen from behind the hand
        self.run_for(follower, pinching(roll=30.0))
        self.assertAlmostEqual(self.jaw_turn(follower, before), 18.0, delta=2.0)  # clockwise from above
        self.assertEqual(follower.state, "turning, turned 18° cw, jaw closed")
        self.run_for(follower, pinching(roll=30.0, pinched=False), seconds=0.3)
        self.run_for(follower, pinching(roll=0.0, pinched=False))
        self.assertAlmostEqual(self.jaw_turn(follower, before), 18.0, delta=2.0, msg="let go, it stays")
        self.run_for(follower, pinching(), seconds=0.3)  # pinch again to turn further
        self.run_for(follower, pinching(roll=-20.0))
        self.assertAlmostEqual(self.jaw_turn(follower, before), 10.0, delta=2.0)

    def test_while_turning_the_jaw_stays_put(self) -> None:
        follower = self.follower()
        start, before = follower.tcp_cm(), dict(follower.q)
        self.run_for(follower, pinching(), seconds=0.3)
        self.run_for(follower, pinching(roll=20.0), seconds=0.5)
        swung = pinching(roll=40.0)  # the pinch swings round the forearm as the hand turns on
        swung["thumb"] = list(np.add(swung["thumb"], [4.0, -3.0, 0.0]))
        swung["index"] = list(np.add(swung["index"], [4.0, -3.0, 0.0]))
        self.run_for(follower, swung)
        np.testing.assert_allclose(follower.tcp_cm(), start, atol=0.05)
        self.assertAlmostEqual(self.jaw_turn(follower, before), 28.0, delta=2.0)

    def test_while_dragging_the_jaw_does_not_turn(self) -> None:
        follower = self.follower()
        before = dict(follower.q)
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -4.0, 0.0))
        self.glide(follower, (0.0, -4.0, 0.0), (0.0, -4.0, 0.0), roll=30.0)
        self.assertAlmostEqual(follower.q["wrist_roll"], before["wrist_roll"], delta=0.01)
        self.assertTrue(follower.state.startswith("dragging"))

    def test_thumb_and_pinky_toggle_the_jaw_once_a_touch(self) -> None:
        follower = self.follower()
        self.assertAlmostEqual(follower.q["gripper"], 0.0)
        self.run_for(follower, pinching(pinched=False, pinky=True))  # held a whole second: one toggle
        self.assertAlmostEqual(follower.q["gripper"], OPEN)
        self.run_for(follower, pinching(pinched=False), seconds=0.3)
        self.run_for(follower, pinching(pinched=False, pinky=True))
        self.assertAlmostEqual(follower.q["gripper"], 0.0)
        self.assertEqual(follower.state, "free, jaw closed")

    def test_no_toggle_while_dragging(self) -> None:
        follower = self.follower()
        self.run_for(follower, pinching(pinky=True))
        self.assertAlmostEqual(follower.q["gripper"], 0.0)

    def test_the_glasses_own_pinch_detection_wins(self) -> None:
        follower = self.follower()
        self.run_for(follower, pinching(pinched=False, told=True), seconds=0.3)
        self.assertTrue(follower.state.startswith("pinched"))
        self.run_for(follower, pinching(pinched=True, told=False), seconds=0.3)
        self.assertTrue(follower.state.startswith("free"))

    def test_down_to_the_table_the_glasses_are_told(self) -> None:
        follower = self.follower()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -20.0, 0.0))
        self.assertEqual(follower.state, "dragging: at the table, jaw closed")
        self.assertEqual(follower.guide()["blocked"], ["down"])

    def test_out_of_sight_it_holds(self) -> None:
        follower = self.follower()
        self.run_for(follower, pinching(), seconds=0.3)
        self.glide(follower, (0.0, 0.0, 0.0), (0.0, -3.0, 0.0), seconds=0.5)
        held = dict(follower.q)
        self.run_for(follower, None, seconds=0.5)
        self.assertEqual(follower.q, held)
        self.assertEqual(follower.state, "no hand, jaw closed")

    def test_no_calibration_first(self) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.spectacles import PinchFollower, follow

        clock = FakeClock(stop_after=10)
        followers = {name: PinchFollower(name, Kinematics(), None, dict(NEUTRAL, shoulder_lift=30.0, wrist_flex=60.0))
                     for name in ("left", "right")}
        hands = Hands(clock=clock)
        hands.put({"head": HEAD, "left": hand_at([-15, -20, -35], gap=12.0), "right": hand_at([15, -20, -35], gap=12.0)})
        states = []
        with self.assertRaises(KeyboardInterrupt):
            follow(followers, hands, None, log=states.append, clock=clock, sleep=clock.sleep)
        self.assertTrue(states[0].startswith("left: free, jaw closed | right: free, jaw closed (dry run)"), states[0])


if __name__ == "__main__":
    unittest.main()
