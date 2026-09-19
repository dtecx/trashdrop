"""Guard for the layout: every bin and the whole pick zone stays reachable.

This is the test to watch after editing station.py. It fails with the exact
position that stopped being reachable rather than as a mysterious sorting miss.
"""

from __future__ import annotations

import unittest

import pytest

pytest.importorskip("mujoco", reason="needs the simulation extra")

from trashdrop.probe import check_layout  # noqa: E402
from trashdrop.simulator import SortingCell  # noqa: E402


class LayoutTests(unittest.TestCase):
    def test_layout_is_fully_reachable(self) -> None:
        cell = SortingCell(record=False)
        self.assertTrue(check_layout(cell, verbose=False), "run: uv run trashdrop probe")


if __name__ == "__main__":
    unittest.main()
