from __future__ import annotations

from transcript_search_v2 import db, ingest

from ..conftest import copy_fixture


def test_backfill_ingests_all_fixtures(conn, tmp_env):
    for name in [
        "normal_exchange.jsonl",
        "tool_use_result.jsonl",
        "thinking_block.jsonl",
        "skipped_type.jsonl",
        "split_response.jsonl",
        "sidechain.jsonl",
    ]:
        copy_fixture(tmp_env["watch_root"], name)

    result = ingest.backfill(conn)
    assert result["files_discovered"] == 6
    assert result["files_touched"] == 6
    assert result["chunks_added"] > 0

    sessions = {r["session_id"] for r in conn.execute("SELECT session_id FROM sessions")}
    assert sessions == {
        "ne-session", "tr-session", "th-session", "sk-session", "sp-session", "sc-session",
    }


def test_skipped_type_lines_are_not_indexed(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "skipped_type.jsonl")
    ingest.backfill(conn)
    rows = conn.execute("SELECT text FROM chunks WHERE session_id = 'sk-session'").fetchall()
    assert len(rows) == 1
    assert rows[0]["text"] == "hello after the noise line"


def test_malformed_trailing_line_does_not_abort_file(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "malformed_trailing.jsonl")
    result = ingest.backfill(conn)
    assert result["chunks_added"] == 1
    row = conn.execute("SELECT text FROM chunks WHERE session_id = 'mf-session'").fetchone()
    assert row["text"] == "good line before garbage"


def test_tool_result_resolves_name_from_prior_tool_use(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl")
    ingest.backfill(conn)
    row = conn.execute(
        "SELECT tool_name FROM chunks WHERE block_type = 'tool_result' AND session_id = 'tr-session'"
    ).fetchone()
    assert row["tool_name"] == "Bash"


def test_split_response_lines_each_produce_independent_chunks(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "split_response.jsonl")
    ingest.backfill(conn)
    rows = conn.execute(
        "SELECT message_uuid, block_type FROM chunks "
        "WHERE session_id = 'sp-session' AND role = 'assistant' ORDER BY rowid"
    ).fetchall()
    assert [dict(r) for r in rows] == [
        {"message_uuid": "sp-a1", "block_type": "thinking"},
        {"message_uuid": "sp-a2", "block_type": "text"},
    ]


def test_second_backfill_is_noop(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    result2 = ingest.backfill(conn)
    assert result2["files_touched"] == 0
    assert result2["chunks_added"] == 0


def test_incremental_reingest_only_adds_new_lines(conn, tmp_env):
    path = copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    before_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    state_before = db.get_file_state(conn, str(path))

    with open(path, "a") as f:
        f.write(
            '{"type":"user","uuid":"ne-u2","parentUuid":"ne-a1",'
            '"timestamp":"2026-07-20T10:01:00.000Z","sessionId":"ne-session",'
            '"cwd":"/tmp/fixture-proj","gitBranch":"main","isSidechain":false,'
            '"message":{"role":"user","content":"a follow-up question"}}\n'
        )

    result = ingest.backfill(conn)
    assert result["chunks_added"] == 1
    after_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert after_count == before_count + 1
    state_after = db.get_file_state(conn, str(path))
    assert state_after.bytes_read > state_before.bytes_read


def test_idempotent_replay_does_not_duplicate(conn, tmp_env):
    path = copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    count_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    fts_before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    # simulate re-processing the same lines (e.g. after a bytes_read correction).
    # last_seen_mtime/size must also be perturbed, otherwise ingest_file's
    # early "nothing changed" stat check would skip re-reading entirely.
    conn.execute(
        "UPDATE files SET bytes_read = 0, lines_read = 0, last_seen_mtime = 0 WHERE path = ?",
        (str(path),),
    )
    conn.commit()
    ingest.ingest_file(conn, path)

    count_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    fts_after = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert count_after == count_before
    assert fts_after == fts_before
