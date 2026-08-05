from __future__ import annotations

from transcript_search_v2 import db
from transcript_search_v2.tools import health

from ..conftest import copy_fixture


def test_doctor_reports_environment_checks(tmp_env):
    result = health.doctor(fix=False)
    env = result["environment"]
    assert env["sqlite_fts5_ok"] is True
    assert env["sqlite_load_extension_ok"] is True
    assert env["missing_dependencies"] == []
    assert "mcp" in env["dependencies"]


def test_doctor_without_fix_reports_missing_index(tmp_env):
    result = health.doctor(fix=False)
    assert result["index"] is None
    assert "not built yet" in result["note"]
    assert result["fixes_applied"] == []


def test_doctor_with_fix_builds_missing_index(tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    result = health.doctor(fix=True)
    assert any(f["action"] == "backfill" for f in result["fixes_applied"])
    assert result["index"]["files_tracked"] == 1
    assert result["index"]["chunks_total"] > 0


def test_doctor_reports_pending_embeddings_without_fix(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    from transcript_search_v2 import ingest

    ingest.backfill(conn)
    conn.close()

    result = health.doctor(fix=False)
    assert result["index"]["chunks_pending_embedding"] == result["index"]["chunks_total"]
    assert not any(f["action"] == "embed_backlog" for f in result["fixes_applied"])


def test_doctor_cost_alerts_present_and_empty_without_llm_calls(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    from transcript_search_v2 import ingest

    ingest.backfill(conn)
    conn.close()

    result = health.doctor(fix=False)
    alerts = result["cost_alerts"]
    assert alerts["today_usd"] == 0.0
    assert alerts["expensive_sessions"] == []


def test_doctor_with_fix_drains_embedding_backlog(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    from transcript_search_v2 import ingest

    ingest.backfill(conn)
    pending_before = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()[0]
    assert pending_before > 0
    conn.close()

    result = health.doctor(fix=True)
    assert any(f["action"] == "embed_backlog" for f in result["fixes_applied"])
    assert result["index"]["chunks_pending_embedding"] == 0

    conn2 = db.connect(read_only=True)
    remaining = conn2.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()[0]
    conn2.close()
    assert remaining == 0


def test_doctor_db_integrity_ok_on_healthy_db(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    from transcript_search_v2 import ingest

    ingest.backfill(conn)
    conn.close()

    result = health.doctor(fix=False)
    assert result["db_integrity"]["ok"] is True


def test_doctor_with_fix_quarantines_and_rebuilds_corrupt_db_with_audit_row(tmp_env):
    from transcript_search_v2 import config

    config.ensure_data_dir()
    config.DB_PATH.write_bytes(b"not a real sqlite file")

    result = health.doctor(fix=True)

    assert any(f["action"] == "quarantine_corrupt_db" for f in result["fixes_applied"])
    assert any(f["action"] == "rebuild" for f in result["fixes_applied"])
    assert result["db_integrity"]["ok"] is True

    conn = db.connect(read_only=True)
    try:
        row = conn.execute(
            "SELECT trigger FROM run_history WHERE trigger = 'doctor_fix_rebuild'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()
