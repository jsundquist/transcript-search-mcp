"""doctor MCP tool: environment/index health checks with optional auto-fix.

Complements status()/coverage() (tools/admin.py), which report what's tracked
vs. discovered. This tool answers "why is nothing coming back from search" or
"is my install broken" -- checks that don't depend on the index already being
healthy, plus (with fix=True) the same auto-recovery actions a human would
reach for: backfill if the index is empty, drain the embedding backlog, quarantine
and rebuild a corrupt db.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
import sys
import time

from transcript_search_v2 import config, db, embed, ingest
from transcript_search_v2.timeutil import local_range_to_utc

_DEPENDENCIES = [
    ("mcp", "mcp"),
    ("pydantic", "pydantic"),
    ("sentence-transformers", "sentence_transformers"),
    ("sqlite-vec", "sqlite_vec"),
    ("watchfiles", "watchfiles"),
]

_MAX_FIX_EMBED_BATCHES = 20  # caps how much work an interactive tool call does


def _check_environment() -> dict:
    dependencies: dict[str, str | None] = {}
    missing: list[str] = []
    for label, import_name in _DEPENDENCIES:
        try:
            mod = __import__(import_name)
            dependencies[label] = getattr(mod, "__version__", "unknown")
        except ImportError:
            dependencies[label] = None
            missing.append(label)

    load_extension_ok = True
    fts5_ok = True
    sqlite_error: str | None = None
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.enable_load_extension(True)
        except AttributeError:
            load_extension_ok = False
        try:
            con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        except sqlite3.OperationalError:
            fts5_ok = False
        con.close()
    except Exception as e:  # noqa: BLE001 -- report, don't crash doctor()
        sqlite_error = str(e)

    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "dependencies": dependencies,
        "missing_dependencies": missing,
        "sqlite_load_extension_ok": load_extension_ok,
        "sqlite_fts5_ok": fts5_ok,
        "sqlite_error": sqlite_error,
        "timezone": str(config.LOCAL_TZ),
    }


def _check_data_paths() -> dict:
    return {
        "watch_root": {"path": str(config.WATCH_ROOT), "exists": config.WATCH_ROOT.exists()},
        "db_path": {"path": str(config.DB_PATH), "exists": config.DB_PATH.exists()},
    }


def _check_db_integrity(conn: sqlite3.Connection) -> dict:
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        ok = row is not None and row[0] == "ok"
        return {"ok": ok, "detail": None if ok else row[0]}
    except sqlite3.Error as e:
        return {"ok": False, "detail": str(e)}


def _cost_alerts(conn: sqlite3.Connection) -> dict:
    """Today's spend vs. 7-day average, and expensive sessions today --
    signals for context compaction / runaway agent loops, ported from the
    old project's server_status()."""
    has_pricing = conn.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0] > 0
    if not has_pricing:
        return {"today_usd": None, "seven_day_avg_usd": None, "expensive_sessions": []}

    today_utc, _ = local_range_to_utc(_dt.date.today().isoformat(), None)
    week_ago_utc, _ = local_range_to_utc((_dt.date.today() - _dt.timedelta(days=7)).isoformat(), None)

    def _cost_since(since: str, until: str | None = None) -> float:
        clauses = ["lc.timestamp >= ?"]
        params: list = [since]
        if until:
            clauses.append("lc.timestamp < ?")
            params.append(until)
        row = conn.execute(
            f"""
            SELECT sum(
                lc.input_tokens * mp.input_cost_per_token
                + lc.output_tokens * mp.output_cost_per_token
                + lc.cache_write_tokens * COALESCE(mp.cache_write_cost_per_token, 0)
                + lc.cache_read_tokens * COALESCE(mp.cache_read_cost_per_token, 0)
            )
            FROM llm_calls lc LEFT JOIN model_pricing mp ON lc.model = mp.model
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchone()
        return row[0] or 0.0

    today_cost = _cost_since(today_utc)
    week_cost_excl_today = _cost_since(week_ago_utc, today_utc)
    seven_day_avg = week_cost_excl_today / 7.0

    sessions_today = conn.execute(
        """
        SELECT lc.session_id,
               sum(
                   lc.input_tokens * mp.input_cost_per_token
                   + lc.output_tokens * mp.output_cost_per_token
                   + lc.cache_write_tokens * COALESCE(mp.cache_write_cost_per_token, 0)
                   + lc.cache_read_tokens * COALESCE(mp.cache_read_cost_per_token, 0)
               ) AS cost_usd,
               count(*) AS calls
        FROM llm_calls lc LEFT JOIN model_pricing mp ON lc.model = mp.model
        WHERE lc.timestamp >= ?
        GROUP BY lc.session_id
        ORDER BY cost_usd DESC
        LIMIT 10
        """,
        [today_utc],
    ).fetchall()

    pace_flag = None
    if seven_day_avg > 0 and today_cost > seven_day_avg * 2:
        pace_flag = f"{today_cost / seven_day_avg:.1f}x above 7-day avg"

    return {
        "today_usd": round(today_cost, 4),
        "seven_day_avg_usd": round(seven_day_avg, 4),
        "pace_flag": pace_flag,
        "expensive_sessions": [
            {"session_id": r["session_id"], "cost_usd": round(r["cost_usd"] or 0, 4), "calls": r["calls"]}
            for r in sessions_today
        ],
    }


def doctor(fix: bool = False) -> dict:
    """Environment + index health check. With fix=True, also attempts common
    auto-repairs: backfill if the index is empty, drain the embedding
    backlog, quarantine+rebuild a corrupt db."""
    result: dict = {
        "environment": _check_environment(),
        "data_paths": _check_data_paths(),
        "fixes_applied": [],
    }

    if not config.DB_PATH.exists():
        if fix:
            conn = db.connect()
            db.init_db(conn)
            backfill_result = ingest.backfill(conn)
            result["fixes_applied"].append({"action": "backfill", "result": backfill_result})
            conn.close()
        else:
            result["index"] = None
            result["note"] = "index not built yet -- call doctor(fix=True) or start the server"
            return result

    conn = db.connect()
    try:
        integrity = _check_db_integrity(conn)
        result["db_integrity"] = integrity

        if not integrity["ok"] and fix:
            conn.close()
            ts = time.strftime("%Y%m%d-%H%M%S")
            quarantine = config.DB_PATH.with_suffix(config.DB_PATH.suffix + f".corrupt-{ts}")
            config.DB_PATH.rename(quarantine)
            for sidecar_suffix in ("-wal", "-shm"):
                sidecar = config.DB_PATH.with_name(config.DB_PATH.name + sidecar_suffix)
                if sidecar.exists():
                    sidecar.rename(sidecar.with_name(sidecar.name + f".corrupt-{ts}"))
            result["fixes_applied"].append(
                {"action": "quarantine_corrupt_db", "quarantined_to": str(quarantine)}
            )
            conn = db.connect()
            db.init_db(conn)
            backfill_result = ingest.backfill(conn)
            result["fixes_applied"].append({"action": "rebuild", "result": backfill_result})
            result["db_integrity"] = _check_db_integrity(conn)

        files_tracked = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        chunks_embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
        sessions_total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        pending = chunks_total - chunks_embedded

        if fix and pending > 0:
            embedded_now = 0
            for _ in range(_MAX_FIX_EMBED_BATCHES):
                batch = db.pending_embeddings(conn, limit=config.EMBED_BATCH_SIZE)
                if not batch:
                    break
                rowids = [rowid for rowid, _ in batch]
                texts = [text for _, text in batch]
                vectors = embed.embed_texts(texts)
                for rowid, vector in zip(rowids, vectors):
                    db.upsert_embedding(conn, rowid, vector)
                conn.commit()
                embedded_now += len(batch)
            if embedded_now:
                result["fixes_applied"].append({"action": "embed_backlog", "chunks_embedded": embedded_now})
                chunks_embedded += embedded_now
                pending -= embedded_now

        result["index"] = {
            "files_tracked": files_tracked,
            "sessions_tracked": sessions_total,
            "chunks_total": chunks_total,
            "chunks_embedded": chunks_embedded,
            "chunks_pending_embedding": pending,
        }
        if files_tracked == 0:
            result["note"] = "no files tracked -- call doctor(fix=True) to run a backfill"

        result["cost_alerts"] = _cost_alerts(conn)
    finally:
        conn.close()

    return result
