from datetime import UTC, datetime

from transcript_search_v2 import rerank


def _result(chunk_id, score, signal, timestamp):
    return {"chunk_id": chunk_id, "score": score, "signal": signal, "timestamp": timestamp}


def test_rerank_promotes_high_signal_recent_over_low_signal_old_at_equal_rrf_score():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    results = [
        _result("old-low", 0.05, "low", "2025-01-01T00:00:00.000Z"),
        _result("new-high", 0.05, "high", "2026-07-31T00:00:00.000Z"),
    ]
    out = rerank.rerank(results, limit=2, now=now)
    assert [r["chunk_id"] for r in out] == ["new-high", "old-low"]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


def test_rerank_truncates_to_limit():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    results = [
        _result(f"c{i}", 1.0 / (i + 1), "high", "2026-08-01T00:00:00.000Z") for i in range(10)
    ]
    out = rerank.rerank(results, limit=3, now=now)
    assert len(out) == 3
    assert [r["chunk_id"] for r in out] == ["c0", "c1", "c2"]


def test_rerank_handles_unparseable_timestamp_gracefully():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    results = [_result("c1", 0.05, "high", "not-a-timestamp")]
    out = rerank.rerank(results, limit=1, now=now)
    assert out[0]["rerank_score"] > 0


def test_recency_multiplier_halves_at_half_life():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    half_life_ago = "2026-07-02T00:00:00.000Z"  # 30 days before now
    mult = rerank._recency_multiplier(half_life_ago, now, half_life_days=30.0)
    assert abs(mult - 0.5) < 0.01
