from __future__ import annotations

from transcript_search_v2 import ingest
from transcript_search_v2.tools import analytics

from ..conftest import copy_fixture


def test_split_response_produces_exactly_one_llm_call(conn, tmp_env):
    """split_response.jsonl has two assistant lines sharing one response's
    usage stats -- only the final one (with service_tier) should land a row,
    otherwise cost/token totals would double-count."""
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl")
    ingest.backfill(conn)

    rows = conn.execute(
        "SELECT message_uuid, input_tokens, output_tokens FROM llm_calls "
        "WHERE session_id = 'sp-session'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["message_uuid"] == "sp-a2"
    assert rows[0]["input_tokens"] == 20
    assert rows[0]["output_tokens"] == 15


def test_get_usage_grouped_by_model_computes_cost_from_seeded_pricing(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl")
    ingest.backfill(conn)

    conn.execute("UPDATE llm_calls SET model = 'claude-sonnet-5' WHERE session_id = 'sp-session'")
    conn.commit()

    rows = analytics.get_usage(group_by="model", conn=conn)
    assert len(rows) == 1
    assert rows[0]["model"] == "claude-sonnet-5"
    assert rows[0]["calls"] == 1
    assert rows[0]["input_tokens"] == 20
    assert rows[0]["cost_usd"] is not None
    assert rows[0]["cost_usd"] > 0


def test_get_usage_unpriced_model_returns_null_cost_not_error(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl")
    ingest.backfill(conn)

    conn.execute("UPDATE llm_calls SET model = 'some-future-model' WHERE session_id = 'sp-session'")
    conn.commit()

    rows = analytics.get_usage(group_by="model", conn=conn)
    assert len(rows) == 1
    assert rows[0]["model"] == "some-future-model"
    assert rows[0]["cost_usd"] is None


def test_get_usage_project_filter(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl", project_dir="-proj-a")
    ingest.backfill(conn)

    matched = analytics.get_usage(group_by="session", project="fixture-proj", conn=conn)
    assert len(matched) == 1

    unmatched = analytics.get_usage(group_by="session", project="nonexistent-project", conn=conn)
    assert unmatched == []
