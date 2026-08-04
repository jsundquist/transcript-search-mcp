"""status / coverage / reindex / prune MCP admin tools."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from transcript_search_v2 import config, db, ingest


def status(conn: sqlite3.Connection | None = None) -> dict:
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        files_tracked = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        chunks_embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
        sessions_total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        last_run = conn.execute(
            "SELECT started_at, finished_at, trigger, files_touched, chunks_added, error "
            "FROM run_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        db_size_bytes = config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0
        return {
            "files_tracked": files_tracked,
            "sessions_tracked": sessions_total,
            "chunks_total": chunks_total,
            "chunks_embedded": chunks_embedded,
            "embedding_coverage_pct": round(100 * chunks_embedded / chunks_total, 1)
            if chunks_total
            else 100.0,
            "db_size_bytes": db_size_bytes,
            "last_run": dict(last_run) if last_run else None,
        }
    finally:
        if own_conn:
            conn.close()


def coverage(project: str | None = None, conn: sqlite3.Connection | None = None) -> dict:
    """Compares files discovered on disk right now against what's tracked in
    the db, flagging anything whose on-disk size doesn't match what was last
    ingested -- i.e. pending re-ingest, independent of whether the watcher
    has caught up yet."""
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        discovered = ingest.discover_files()
        if project:
            discovered = [p for p in discovered if project in str(p.parent)]

        tracked_rows = conn.execute("SELECT path, last_seen_size, status FROM files").fetchall()
        tracked = {r["path"]: r for r in tracked_rows}

        untracked: list[str] = []
        pending_reindex: list[str] = []
        errored: list[str] = []
        for path in discovered:
            path_str = str(path)
            row = tracked.get(path_str)
            if row is None:
                untracked.append(path_str)
                continue
            if row["status"] != "ok":
                errored.append(path_str)
            elif row["last_seen_size"] != path.stat().st_size:
                pending_reindex.append(path_str)

        return {
            "files_discovered": len(discovered),
            "files_tracked": len(tracked_rows),
            "untracked": untracked,
            "pending_reindex": pending_reindex,
            "errored": errored,
        }
    finally:
        if own_conn:
            conn.close()


def prune(
    older_than_days: int,
    project: str | None = None,
    dry_run: bool = True,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Delete chunks (+ their FTS/vec rows) and llm_calls older than
    `older_than_days`, optionally scoped to a project (substring match).

    dry_run defaults to True -- this is destructive and irreversible via
    this tool, so the caller must explicitly pass dry_run=False to delete
    anything. `files`/`sessions` rows are left untouched, which means
    pruned content will NOT reappear on the next backfill even if its
    source .jsonl is unchanged -- the ingestion cursor (files.bytes_read)
    already advanced past those lines, so nothing re-triggers re-parsing
    them. Only re-run reindex() after resetting bytes_read if you actually
    want pruned lines re-indexed.
    """
    own_conn = conn is None
    conn = conn or db.connect(read_only=False)
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"

        clauses = ["timestamp < ?"]
        params: list = [cutoff]
        if project:
            clauses.append("project_cwd LIKE ?")
            params.append(f"%{project}%")
        where = " AND ".join(clauses)

        chunk_rowids = [r[0] for r in conn.execute(f"SELECT rowid FROM chunks WHERE {where}", params)]
        llm_call_count = conn.execute(f"SELECT COUNT(*) FROM llm_calls WHERE {where}", params).fetchone()[0]

        if dry_run:
            return {
                "dry_run": True,
                "cutoff": cutoff,
                "chunks_to_delete": len(chunk_rowids),
                "llm_calls_to_delete": llm_call_count,
            }

        if chunk_rowids:
            placeholders = ",".join("?" for _ in chunk_rowids)
            conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", chunk_rowids)
            conn.execute(f"DELETE FROM chunks_vec WHERE chunk_rowid IN ({placeholders})", chunk_rowids)
            conn.execute(f"DELETE FROM chunks WHERE rowid IN ({placeholders})", chunk_rowids)
        conn.execute(f"DELETE FROM llm_calls WHERE {where}", params)
        conn.commit()

        return {
            "dry_run": False,
            "cutoff": cutoff,
            "chunks_deleted": len(chunk_rowids),
            "llm_calls_deleted": llm_call_count,
        }
    finally:
        if own_conn:
            conn.close()
