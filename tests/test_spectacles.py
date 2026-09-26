"""Spectacles drive the arms: the socket, the wearer's frame, and each arm following its hand.

What is pinned down: the WebSocket speaks RFC 6455 to a raw client -- the
handshake, masked text of any length, ping, close -- and answers with what
the arms do; forward is where the wearer looks, whichever way round the
camera's axis comes; on the model at the venue's placement an arm anchors
where it is when a hand appears (no jump), follows the hand forward, right
and up, holds when it is lost, keeps above the table and off the other arm's
side, never turns a joint faster than asked, and opens its jaw as thumb and
index part; and every tick writes both arms, until the hands go stale.
"""

from __future__ import annotations

import importlib.util
import base64
import io
import json
import socket
import struct
import threading
import time
import unittest

import numpy as np

from tests.test_arm import FakeBus, FakeClock
from trashdrop.spectacles import (
    HAND_FOR,
    HEIGHT_CM,
    OPEN,
    SIDE_CM,
    STALE_S,
    Hands,
    VideoFrames,
    accept_key,
    angle_delta,
    frame,
    facing_frame,
    gripper_for,
    hand_angles,
    make_server,
    make_video_server,
    read_frame,
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
                  mirror: bool = False) -> dict:
    """Stationary fingertips with a palm that tilts and rotates around them."""

    hand = hand_at(point)
    wrist = np.asarray(hand["wrist"])
    pitch, roll, yaw = np.radians((pitch, roll, yaw))
    fingers = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    across = right * np.cos(roll) + np.cross(fingers, right) * np.sin(roll)
    if mirror:  # The other hand's index-to-pinky direction points the other way.
        across = -across
    middle = wrist + 8 * fingers
    hand.update(middleKnuckle=list(middle), indexKnuckle=list(middle + 2 * across),
                pinkyKnuckle=list(middle - 2 * across))
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


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class FollowerTests(unittest.TestCase):
    """The venue's left arm, on the model."""

    @classmethod
    def setUpClass(cls) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.placement import Placement
        from trashdrop.spectacles import ready_pose

        cls.kinematics = Kinematics(-80.0)
        cls.placement = Placement(11.34, -22.40, -95.14, -0.36, -0.0318, 0.0617)
        cls.limits = cls.kinematics.own_limits()
        cls.ready = ready_pose(cls.kinematics, cls.placement, NEUTRAL, cls.limits)

    def follower(self, name: str = "left", **options):
        from trashdrop.spectacles import Follower

        return Follower(name, self.kinematics, self.placement, self.ready, limits=self.limits, **options)

    def run_for(self, follower, hand, seconds: float = 2.0, head=HEAD) -> None:
        for _ in range(int(seconds / 0.02)):
            follower.update(hand, head, 0.02)

    def test_the_ready_pose_is_fingers_down_over_the_table(self) -> None:
        fingers, _ = self.kinematics.pointing(self.ready)
        self.assertLess(fingers[2], -0.7)
        tcp = self.kinematics.tcp(self.ready) * 100
        self.assertGreater(tcp[2], self.placement.table_height(*tcp[:2]) + 5.0)

    def test_a_hand_seen_anchors_its_arm_where_it_is(self) -> None:
        follower = self.follower()
        before = follower.tcp_cm()
        follower.update(hand_at([0, -20, -35]), HEAD, 0.02)
        np.testing.assert_allclose(follower.target, before, atol=0.01)
        self.assertEqual(follower.state, "following")

    def test_the_arm_follows_the_hand_forward_right_and_up(self) -> None:
        for facing, forward_sign, right_sign in (("same", 1.0, -1.0), ("them", -1.0, 1.0)):
            with self.subTest(facing=facing):
                follower = self.follower(facing=facing)
                follower.update(hand_at([0, -20, -35]), HEAD, 0.02)
                start = follower.tcp_cm()
                self.run_for(follower, hand_at([3, -17, -39]))  # 4 cm ahead, 3 right, 3 up
                moved = follower.tcp_cm() - start
                np.testing.assert_allclose(moved, [4 * forward_sign, 3 * right_sign, 3], atol=0.6)

    def test_the_camera_axis_the_other_way_round_changes_nothing(self) -> None:
        follower = self.follower()
        head = {"p": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]}
        follower.update(hand_at([0, -20, -35]), head, 0.02)
        start = follower.tcp_cm()
        self.run_for(follower, hand_at([0, -20, -39]), head=head)
        self.assertGreater(follower.tcp_cm()[0] - start[0], 3.4, "still forward")

    def test_a_lost_hand_holds_and_is_anchored_afresh(self) -> None:
        follower = self.follower()
        follower.update(hand_at([0, -20, -35]), HEAD, 0.02)
        self.run_for(follower, hand_at([0, -20, -38]), seconds=1.0)
        held = dict(follower.q)
        self.run_for(follower, None, seconds=0.5)
        self.assertEqual(follower.q, held)
        self.assertEqual(follower.state, "no hand")
        where = follower.tcp_cm()
        follower.update(hand_at([10, -5, -30]), HEAD, 0.02)  # back in view somewhere else
        np.testing.assert_allclose(follower.target, where, atol=0.01)

    def test_never_into_the_table_nor_onto_the_other_arms_side(self) -> None:
        follower = self.follower()
        follower.update(hand_at([0, -20, -35]), HEAD, 0.02)
        follower.update(hand_at([40, -60, -35]), HEAD, 0.02)  # far right and far down
        x, y, z = follower.target
        self.assertGreaterEqual(y, -SIDE_CM - 1e-9, "the right arm is that way")
        self.assertGreaterEqual(z, self.placement.table_height(x, y) + HEIGHT_CM[0] - 1e-9)

    def test_no_joint_turns_faster_than_asked(self) -> None:
        follower = self.follower(speed=60.0)
        follower.update(hand_at([0, -20, -35]), HEAD, 0.02)
        before = dict(follower.q)
        follower.update(hand_at([0, 10, -60]), HEAD, 0.02)
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
            self.assertLessEqual(abs(follower.q[joint] - before[joint]), 60.0 * 0.02 + 1e-9)
        self.assertEqual(follower.q["wrist_roll"], before["wrist_roll"], "the roll stays")

    def test_palm_roll_and_finger_tilt_turn_both_wrist_joints(self) -> None:
        from trashdrop.kinematics import Kinematics
        from trashdrop.placement import Placement
        from trashdrop.spectacles import Follower, ready_pose

        for name in ("left", "right"):
            with self.subTest(name=name):
                if name == "left":
                    follower = self.follower(name=name)
                else:
                    model = Kinematics(5.0)  # right arm's measured wrist zero in rig.toml
                    placement = Placement(10.98, 19.12, -79.47, -1.02, -0.0315, -0.0252)
                    limits = model.own_limits()
                    ready = ready_pose(model, placement, NEUTRAL, limits)
                    follower = Follower(name, model, placement, ready, limits=limits)
                neutral = oriented_hand([0, -20, -35], mirror=name == "right")
                follower.update(neutral, HEAD, 0.02)
                start = dict(follower.q)
                tcp = follower.tcp_cm()
                tilted = oriented_hand([0, -20, -35], roll=25, pitch=15, mirror=name == "right")
                self.run_for(follower, tilted)
                self.assertAlmostEqual(follower.q["wrist_roll"] - start["wrist_roll"], 25, delta=1)
                self.assertAlmostEqual(follower.q["wrist_flex"] - start["wrist_flex"], -15, delta=1)
                np.testing.assert_allclose(follower.tcp_cm(), tcp, atol=1.0)

    def test_large_wrist_turns_stop_at_both_calibrated_limits(self) -> None:
        follower = self.follower()
        follower.update(oriented_hand([0, -20, -35]), HEAD, 0.02)
        for roll, pitch in ((170, -80), (-170, 80)):
            self.run_for(follower, oriented_hand([0, -20, -35], roll=roll, pitch=pitch))
            for joint in ("wrist_flex", "wrist_roll"):
                low, high = follower.limits[joint]
                self.assertGreaterEqual(follower.q[joint], low)
                self.assertLessEqual(follower.q[joint], high)

    def test_wrist_tracks_at_speed_then_holds_and_reanchors_after_occlusion(self) -> None:
        follower = self.follower(speed=60.0)
        follower.update(oriented_hand([0, -20, -35]), HEAD, 0.02)
        before = dict(follower.q)
        turned = oriented_hand([0, -20, -35], roll=35, pitch=-25)
        follower.update(turned, HEAD, 0.02)
        for joint in ("wrist_flex", "wrist_roll"):
            self.assertLessEqual(abs(follower.q[joint] - before[joint]), 1.2 + 1e-9)
        self.run_for(follower, turned)
        held = dict(follower.q)
        follower.update(None, HEAD, 0.02)
        self.assertEqual(follower.q, held)
        follower.update(oriented_hand([0, -20, -35], roll=-60, pitch=30), HEAD, 0.02)
        self.assertAlmostEqual(follower.q["wrist_roll"], held["wrist_roll"], delta=0.01)
        self.assertAlmostEqual(follower.q["wrist_flex"], held["wrist_flex"], delta=0.01)

    def test_the_jaw_opens_and_closes_with_thumb_and_index(self) -> None:
        follower = self.follower()
        self.run_for(follower, hand_at([0, -20, -35], gap=12.0), seconds=1.0)
        self.assertAlmostEqual(follower.q["gripper"], OPEN)
        self.run_for(follower, hand_at([0, -20, -35], gap=1.0), seconds=1.0)
        self.assertAlmostEqual(follower.q["gripper"], 0.0)


@unittest.skipUnless(HAS_MUJOCO, "needs the simulation extra")
class FollowTests(unittest.TestCase):
    def test_every_tick_writes_both_arms_until_the_hands_go_stale(self) -> None:
        from trashdrop.arm import Arm
        from trashdrop.kinematics import Kinematics
        from trashdrop.spectacles import Follower, follow

        clock = FakeClock(stop_after=40)  # 0.8 s of ticks, then Ctrl+C
        buses = {name: FakeBus() for name in ("left", "right")}
        arms = {name: Arm(name, bus, 45.0, clock=clock, sleep=clock.sleep) for name, bus in buses.items()}
        followers = {name: Follower(name, Kinematics(), None, dict(NEUTRAL, shoulder_lift=30.0, wrist_flex=60.0))
                     for name in arms}
        hands = Hands(clock=clock)
        hands.put({"head": HEAD, "left": hand_at([0, -20, -35]), "right": hand_at([20, -20, -35])})
        states = []
        with self.assertRaises(KeyboardInterrupt):
            follow(followers, hands, arms, log=states.append, clock=clock, sleep=clock.sleep)
        self.assertEqual(len(buses["left"].goals_streamed()), len(buses["right"].goals_streamed()))
        self.assertGreaterEqual(len(buses["left"].goals_streamed()), 39)
        self.assertIn("following", states[0])
        self.assertIn("no hand", states[-1], f"stale after {STALE_S} s")


if __name__ == "__main__":
    unittest.main()
