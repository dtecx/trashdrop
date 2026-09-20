"""The intake API, exercised over real HTTP.

Other teams integrate against this, so the tests drive an actual server on an
ephemeral port rather than calling handlers directly -- a routing or header
mistake should fail here, not at the venue.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from trashdrop.api import SortedItem, SortingLog, _Handler, capabilities
from trashdrop.station import ALL_CATEGORIES, MIXED_CATEGORY


class ApiTestCase(unittest.TestCase):
    """Boots one server for the class and tears it down afterwards."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.log = SortingLog()
        handler = type(
            "TestHandler",
            (_Handler,),
            {"log": cls.log, "sorter": None, "mode": "test"},
        )
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=5) as response:
            return response.status, response.read(), dict(response.headers)

    def post(self, path: str, payload):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base}{path}", data=body, headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())


class RoutingTests(ApiTestCase):
    def test_health(self) -> None:
        status, body, _ = self.get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])

    def test_index_is_human_readable(self) -> None:
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        # The page has to carry a working example, not just endpoint names:
        # another team should be able to copy one line and get a delivery id.
        self.assertIn("curl -X POST", body.decode())
        self.assertIn("/deliveries", body.decode())

    def test_cors_is_open(self) -> None:
        _, _, headers = self.get("/health")
        self.assertEqual(headers.get("access-control-allow-origin"), "*")

    def test_unknown_route_points_at_the_index(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/nope")
        self.assertEqual(caught.exception.code, 404)
        self.assertIn("see /", json.loads(caught.exception.read())["error"])

    def test_trailing_slash_is_accepted(self) -> None:
        self.assertEqual(self.get("/health/")[0], 200)


class CapabilitiesTests(ApiTestCase):
    def test_declares_every_bin_and_the_fallback(self) -> None:
        payload = capabilities()
        self.assertEqual(set(payload["sorts_into"]), set(ALL_CATEGORIES))
        self.assertEqual(payload["fallback_bin"], MIXED_CATEGORY)

    def test_drop_zone_is_a_real_rectangle(self) -> None:
        zone = capabilities()["drop_zone"]["bounds"]
        self.assertLess(zone["min_x"], zone["max_x"])
        self.assertLess(zone["min_y"], zone["max_y"])

    def test_served_over_http(self) -> None:
        status, body, _ = self.get("/capabilities")
        self.assertEqual(status, 200)
        self.assertIn("drop_zone", json.loads(body))


class DeliveryTests(ApiTestCase):
    def test_announce_returns_a_pollable_id(self) -> None:
        status, payload = self.post("/deliveries", {"robot": "cleaner-1", "items": 2})
        self.assertEqual(status, 202)
        self.assertIn("delivery_id", payload)

        status, body, _ = self.get(payload["poll"])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["robot"], "cleaner-1")

    def test_robot_name_is_required(self) -> None:
        # Without it we cannot tell two teams' deliveries apart.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/deliveries", {"items": 1})
        self.assertEqual(caught.exception.code, 400)

    def test_item_count_is_validated(self) -> None:
        for payload in ({"robot": "r", "items": 0}, {"robot": "r", "items": 10_000}):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post("/deliveries", payload)
            self.assertEqual(caught.exception.code, 400)

    def test_non_numeric_item_count_is_refused(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/deliveries", {"robot": "r", "items": "lots"})
        self.assertEqual(caught.exception.code, 400)

    def test_unknown_delivery_is_404(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/deliveries/doesnotexist")
        self.assertEqual(caught.exception.code, 404)


class StatsTests(ApiTestCase):
    def test_totals_follow_recorded_items(self) -> None:
        _, payload = self.post("/deliveries", {"robot": "stats-bot", "items": 1})
        before = json.loads(self.get("/stats")[1])["items_sorted"]
        self.log.record(
            payload["delivery_id"],
            SortedItem(
                item_id="x",
                category="metal",
                bin="metal",
                confidence=0.9,
                arm="back",
                sorted_at="now",
            ),
        )
        after = json.loads(self.get("/stats")[1])
        self.assertEqual(after["items_sorted"], before + 1)
        self.assertGreaterEqual(after["per_bin"]["metal"], 1)


class LogTests(unittest.TestCase):
    def test_totals_start_at_zero_for_every_bin(self) -> None:
        totals = SortingLog().totals()
        self.assertEqual(set(totals["per_bin"]), set(ALL_CATEGORIES))
        self.assertEqual(totals["items_sorted"], 0)

    def test_state_transitions_are_recorded(self) -> None:
        log = SortingLog()
        delivery = log.announce("r", 1)
        self.assertEqual(delivery.state, "queued")
        log.set_state(delivery.delivery_id, "done")
        self.assertEqual(log.get(delivery.delivery_id).state, "done")

    def test_recording_against_an_unknown_delivery_is_ignored(self) -> None:
        log = SortingLog()
        log.set_state("nope", "done")  # must not raise
        self.assertIsNone(log.get("nope"))


if __name__ == "__main__":
    unittest.main()
