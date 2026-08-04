from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from transcript_search_v2 import db, embed, ingest
from transcript_search_v2.tools import admin

from ..conftest import copy_fixture


def _make_old(conn, session_id: str, days_ago: int) -> str:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"
    conn.execute("UPDATE chunks SET timestamp = ? WHERE session_id = ?", (old_ts, session_id))
    conn.execute("UPDATE llm_calls SET timestamp = ? WHERE session_id = ?", (old_ts, session_id))
    conn.commit()
    return old_ts


def test_prune_dry_run_reports_without_deleting(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _make_old(conn, "ne-session", days_ago=100)

    before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    result = admin.prune(older_than_days=30, dry_run=True, conn=conn)
    after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    assert result["dry_run"] is True
    assert result["chunks_to_delete"] == before
    assert after == before  # nothing actually deleted


def test_prune_deletes_only_matching_rows_and_their_fts_vec_rows(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl", project_dir="-old-proj")
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl", project_dir="-new-proj")
    ingest.backfill(conn)
    _make_old(conn, "ne-session", days_ago=100)

    pending = db.pending_embeddings(conn, limit=1000)
    vectors = embed.embed_texts([t for _, t in pending])
    for (rowid, _), vector in zip(pending, vectors):
        db.upsert_embedding(conn, rowid, vector)
    conn.commit()

    old_chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'ne-session'"
    ).fetchone()[0]
    new_chunk_count_before = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tr-session'"
    ).fetchone()[0]
    fts_before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    vec_before = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]

    result = admin.prune(older_than_days=30, dry_run=False, conn=conn)

    assert result["dry_run"] is False
    assert result["chunks_deleted"] == old_chunk_count

    remaining_old = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'ne-session'"
    ).fetchone()[0]
    remaining_new = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tr-session'"
    ).fetchone()[0]
    assert remaining_old == 0
    assert remaining_new == new_chunk_count_before  # untouched

    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == fts_before - old_chunk_count
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == vec_before - old_chunk_count

    # files/sessions bookkeeping is untouched by design
    assert conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'ne-session'"
    ).fetchone()[0] == 1


def test_prune_project_filter_scopes_deletion(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl")
    ingest.backfill(conn)
    _make_old(conn, "ne-session", days_ago=100)
    _make_old(conn, "tr-session", days_ago=100)
    conn.execute("UPDATE chunks SET project_cwd = '/tmp/project-alpha' WHERE session_id = 'ne-session'")
    conn.execute("UPDATE chunks SET project_cwd = '/tmp/project-beta' WHERE session_id = 'tr-session'")
    conn.commit()

    result = admin.prune(older_than_days=30, project="alpha", dry_run=True, conn=conn)
    alpha_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'ne-session'"
    ).fetchone()[0]
    assert result["chunks_to_delete"] == alpha_count

    result_none = admin.prune(older_than_days=30, project="nonexistent-project", dry_run=True, conn=conn)
    assert result_none["chunks_to_delete"] == 0


def test_prune_rejects_non_positive_older_than_days(conn, tmp_env):
    for bad_value in (0, -1, -100):
        with pytest.raises(ValueError):
            admin.prune(older_than_days=bad_value, dry_run=True, conn=conn)


def test_prune_writes_run_history_row_on_delete(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _make_old(conn, "ne-session", days_ago=100)

    runs_before = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    admin.prune(older_than_days=30, dry_run=False, conn=conn)
    runs_after = conn.execute(
        "SELECT COUNT(*) FROM run_history WHERE trigger LIKE 'prune(%'"
    ).fetchone()[0]

    assert runs_after == 1
    row = conn.execute(
        "SELECT finished_at, error FROM run_history WHERE trigger LIKE 'prune(%'"
    ).fetchone()
    assert row["finished_at"] is not None
    assert row["error"] is None


def test_prune_dry_run_does_not_write_run_history(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _make_old(conn, "ne-session", days_ago=100)

    admin.prune(older_than_days=30, dry_run=True, conn=conn)
    count = conn.execute(
        "SELECT COUNT(*) FROM run_history WHERE trigger LIKE 'prune(%'"
    ).fetchone()[0]
    assert count == 0


def test_prune_deletes_llm_calls_older_than_cutoff(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl")
    ingest.backfill(conn)
    _make_old(conn, "sp-session", days_ago=100)

    before = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert before > 0
    result = admin.prune(older_than_days=30, dry_run=False, conn=conn)
    assert result["llm_calls_deleted"] == before
    after = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    assert after == 0
