from __future__ import annotations

from transcript_search_v2 import server


def test_clamp_limit_clamps_to_bounds():
    assert server._clamp_limit(20) == 20
    assert server._clamp_limit(0) == 1
    assert server._clamp_limit(-5) == 1
    assert server._clamp_limit(1_000_000) == 500


def test_clamp_offset_rejects_negative():
    assert server._clamp_offset(0) == 0
    assert server._clamp_offset(50) == 50
    assert server._clamp_offset(-10) == 0


def test_clamp_window_clamps_before_after():
    assert server._clamp_window(5) == 5
    assert server._clamp_window(-5) == 0
    assert server._clamp_window(1_000) == 100
