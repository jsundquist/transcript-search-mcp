"""Heuristic reranking pass for hybrid_search, beyond plain RRF.

RRF fuses keyword/semantic rank alone -- it has no notion that a routine
tool_result chunk from six months ago is usually less useful for personal
recall than a high-signal conversational chunk from yesterday, even at the
same fused rank. This blends the RRF score with a signal weight and an
exponential recency decay, then re-sorts and truncates. It's a tunable
heuristic, not a learned model -- swap for a cross-encoder later if ranking
quality demands it; this needs no new dependency and stays fast.
"""

from __future__ import annotations

from datetime import datetime, timezone

SIGNAL_WEIGHT = {"high": 1.0, "medium": 0.85, "low": 0.6}
DEFAULT_HALF_LIFE_DAYS = 30.0


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _recency_multiplier(timestamp: str, now: datetime, half_life_days: float) -> float:
    ts = _parse_ts(timestamp)
    if ts is None:
        return 1.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400)
    return 0.5 ** (age_days / half_life_days)


def rerank(
    results: list[dict],
    *,
    limit: int,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    now: datetime | None = None,
) -> list[dict]:
    """Re-score `results` (each needs "score", "signal", "timestamp") by
    RRF-score * signal-weight * recency-decay, re-sort, and truncate to
    `limit`. Adds a "rerank_score" key; the original "score" is untouched."""
    now = now or datetime.now(timezone.utc)
    rescored = []
    for r in results:
        base_score = r.get("score") or 0.0
        signal_mult = SIGNAL_WEIGHT.get(r.get("signal"), 1.0)
        recency_mult = _recency_multiplier(r["timestamp"], now, half_life_days)
        out = dict(r)
        out["rerank_score"] = base_score * signal_mult * recency_mult
        rescored.append(out)
    rescored.sort(key=lambda r: -r["rerank_score"])
    return rescored[:limit]
