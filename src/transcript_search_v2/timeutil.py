"""Local-timezone date_from/date_to params <-> stored UTC timestamp strings.

Transcript timestamps are stored as UTC (matching the source data), but a
user typing date_from="2026-07-23" means that day in their own timezone, not
UTC -- converting at the query boundary avoids silent off-by-a-day bugs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from transcript_search_v2.config import LOCAL_TZ


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_local(date_str: str, *, end_of_day: bool) -> datetime:
    date_only = len(date_str) <= 10  # "YYYY-MM-DD"
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is not None:
        return dt
    if date_only and end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    return dt.replace(tzinfo=LOCAL_TZ)


def local_range_to_utc(date_from: str | None, date_to: str | None) -> tuple[str | None, str | None]:
    utc_from = _to_utc_iso(_parse_local(date_from, end_of_day=False)) if date_from else None
    utc_to = _to_utc_iso(_parse_local(date_to, end_of_day=True)) if date_to else None
    return utc_from, utc_to
