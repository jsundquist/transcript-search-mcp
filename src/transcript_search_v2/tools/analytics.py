"""get_usage MCP tool: LLM cost and token usage breakdown.

cost_usd is computed via a LEFT JOIN against model_pricing rather than stored
on llm_calls -- see db.py's llm_calls DDL comment. A model with no pricing
row still returns full token counts with cost_usd = None, it never blocks
the query.
"""

from __future__ import annotations

import sqlite3

from transcript_search_v2 import db
from transcript_search_v2.timeutil import local_range_to_utc

_GROUP_BY_COLUMNS = {
    "model": "lc.model",
    "session": "lc.session_id",
    "day": "date(lc.timestamp)",
}


def get_usage(
    group_by: str = "model",
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Break down LLM token usage and cost.

    group_by: "model" (default), "session", or "day".
    """
    if group_by not in _GROUP_BY_COLUMNS:
        raise ValueError(f"unknown group_by: {group_by!r}. Use: model, session, day")

    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        utc_from, utc_to = local_range_to_utc(date_from, date_to)
        clauses: list[str] = []
        params: list = []
        if project:
            clauses.append("lc.project_cwd LIKE ?")
            params.append(f"%{project}%")
        if utc_from:
            clauses.append("lc.timestamp >= ?")
            params.append(utc_from)
        if utc_to:
            clauses.append("lc.timestamp <= ?")
            params.append(utc_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        group_col = _GROUP_BY_COLUMNS[group_by]
        sql = f"""
            SELECT
                {group_col} AS group_key,
                count(*) AS calls,
                sum(lc.input_tokens) AS input_tokens,
                sum(lc.output_tokens) AS output_tokens,
                sum(lc.cache_write_tokens) AS cache_write_tokens,
                sum(lc.cache_read_tokens) AS cache_read_tokens,
                sum(
                    lc.input_tokens * mp.input_cost_per_token
                    + lc.output_tokens * mp.output_cost_per_token
                    + lc.cache_write_tokens * COALESCE(mp.cache_write_cost_per_token, 0)
                    + lc.cache_read_tokens * COALESCE(mp.cache_read_cost_per_token, 0)
                ) AS cost_usd,
                min(lc.timestamp) AS first_call,
                max(lc.timestamp) AS last_call
            FROM llm_calls lc
            LEFT JOIN model_pricing mp ON lc.model = mp.model
            {where}
            GROUP BY group_col
            ORDER BY cost_usd DESC NULLS LAST
            LIMIT ?
        """.replace("GROUP BY group_col", f"GROUP BY {group_col}")
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d[group_by] = d.pop("group_key")
            results.append(d)
        return results
    finally:
        if own_conn:
            conn.close()
