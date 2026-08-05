"""Shared SQL-fragment helpers so filter/cost logic isn't hand-rolled per tool.

project_date_clause is the single place that builds a project-substring +
date-range WHERE fragment; escape_like is what makes that substring match
safe against a caller-supplied value containing literal LIKE wildcards.
"""

from __future__ import annotations

_LIKE_ESCAPE_CHAR = "\\"


def escape_like(value: str) -> str:
    """Escape LIKE metacharacters (`%`, `_`, and the escape char itself) so
    `value` matches only literally when embedded in a `LIKE ... ESCAPE '\\'`
    pattern -- without this, a project name containing `%`/`_` silently
    broadens the match to unintended rows."""
    return (
        value.replace(_LIKE_ESCAPE_CHAR, _LIKE_ESCAPE_CHAR * 2)
        .replace("%", f"{_LIKE_ESCAPE_CHAR}%")
        .replace("_", f"{_LIKE_ESCAPE_CHAR}_")
    )


def project_date_clause(
    project: str | None,
    utc_from: str | None,
    utc_to: str | None,
    params: list,
    *,
    project_col: str,
    ts_col: str,
) -> list[str]:
    """Returns a list of `"col OP ?"` clauses (no leading AND/WHERE -- the
    caller joins them however its query needs) for the project/date-range
    filter repeated across search/temporal/analytics/admin, appending bound
    values to `params` in the same order."""
    clauses: list[str] = []
    if project:
        clauses.append(f"{project_col} LIKE ? ESCAPE '{_LIKE_ESCAPE_CHAR}'")
        params.append(f"%{escape_like(project)}%")
    if utc_from:
        clauses.append(f"{ts_col} >= ?")
        params.append(utc_from)
    if utc_to:
        clauses.append(f"{ts_col} <= ?")
        params.append(utc_to)
    return clauses


# Shared LLM-cost SQL fragment -- duplicated verbatim across health.py's
# _cost_since/sessions_today and analytics.get_usage before this existed.
COST_EXPR_SQL = """(
    lc.input_tokens * mp.input_cost_per_token
    + lc.output_tokens * mp.output_cost_per_token
    + lc.cache_write_tokens * COALESCE(mp.cache_write_cost_per_token, 0)
    + lc.cache_read_tokens * COALESCE(mp.cache_read_cost_per_token, 0)
)"""
