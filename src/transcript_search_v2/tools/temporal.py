"""list_sessions / get_period / get_context / get_session MCP tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from transcript_search_v2 import db
from transcript_search_v2.extract import render_full
from transcript_search_v2.schema import OtherRecord, TranscriptRecord, parse_line
from transcript_search_v2.timeutil import local_range_to_utc


def list_sessions(
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        utc_from, utc_to = local_range_to_utc(date_from, date_to)
        clauses = []
        params: list = []
        if project:
            clauses.append("project_cwd LIKE ?")
            params.append(f"%{project}%")
        if utc_from:
            clauses.append("last_ts >= ?")
            params.append(utc_from)
        if utc_to:
            clauses.append("first_ts <= ?")
            params.append(utc_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT session_id, project_cwd, first_ts, last_ts, message_count, git_branch
            FROM sessions
            {where}
            ORDER BY last_ts DESC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def get_period(
    date_from: str,
    date_to: str,
    project: str | None = None,
    include_low_signal: bool = False,
    limit: int = 100,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        utc_from, utc_to = local_range_to_utc(date_from, date_to)
        clauses = ["timestamp >= ?", "timestamp <= ?"]
        params: list = [utc_from, utc_to]
        if not include_low_signal:
            clauses.append("signal != 'low'")
        if project:
            clauses.append("project_cwd LIKE ?")
            params.append(f"%{project}%")
        sql = f"""
            SELECT id, session_id, project_cwd, timestamp, role, block_type, signal, text
            FROM chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp ASC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own_conn:
            conn.close()


def _session_file_path(conn: sqlite3.Connection, session_id: str) -> Path | None:
    row = conn.execute(
        "SELECT path FROM files WHERE session_id = ? LIMIT 1", (session_id,)
    ).fetchone()
    return Path(row["path"]) if row else None


def _resolver(conn: sqlite3.Connection):
    def resolve(tool_use_id: str) -> str | None:
        return db.resolve_tool_name(conn, tool_use_id)

    return resolve


def _parse_all_lines(path: Path) -> list[tuple[int, TranscriptRecord]]:
    records: list[tuple[int, TranscriptRecord]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            try:
                raw = json.loads(raw_line)
                record = parse_line(raw)
            except Exception:
                continue
            if isinstance(record, OtherRecord):
                continue
            records.append((line_no, record))
    return records


def _render_message(conn: sqlite3.Connection, line_no: int, record: TranscriptRecord) -> dict:
    return {
        "source_line_no": line_no,
        "message_uuid": record.uuid,
        "parent_uuid": record.parentUuid,
        "role": record.type,
        "timestamp": record.timestamp,
        "is_sidechain": record.isSidechain,
        "blocks": render_full(record, resolve_tool_name=_resolver(conn)),
    }


def get_context(
    session_id: str,
    around_chunk_id: str | None = None,
    message_uuid: str | None = None,
    before: int = 5,
    after: int = 5,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Re-reads the original untruncated .jsonl lines around an anchor
    message, walking outward skipping sidechain/non-conversational lines,
    until `before`/`after` real messages are found on each side."""
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        if around_chunk_id:
            row = conn.execute(
                "SELECT message_uuid, source_line_no FROM chunks WHERE id = ?", (around_chunk_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown chunk_id: {around_chunk_id}")
            anchor_uuid, anchor_line = row["message_uuid"], row["source_line_no"]
        elif message_uuid:
            row = conn.execute(
                "SELECT message_uuid, source_line_no FROM chunks WHERE message_uuid = ? LIMIT 1",
                (message_uuid,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown message_uuid: {message_uuid}")
            anchor_uuid, anchor_line = row["message_uuid"], row["source_line_no"]
        else:
            raise ValueError("must provide around_chunk_id or message_uuid")

        path = _session_file_path(conn, session_id)
        if path is None or not path.exists():
            raise ValueError(f"no transcript file on record for session {session_id}")

        all_records = _parse_all_lines(path)
        # non-sidechain records only, for the walk
        conversational = [(ln, r) for ln, r in all_records if not r.isSidechain]

        anchor_pos = next(
            (i for i, (ln, r) in enumerate(conversational) if r.uuid == anchor_uuid), None
        )
        if anchor_pos is None:
            raise ValueError(f"anchor message {anchor_uuid} not found among conversational lines")

        before_slice = conversational[max(0, anchor_pos - before):anchor_pos]
        after_slice = conversational[anchor_pos + 1: anchor_pos + 1 + after]
        anchor_ln, anchor_record = conversational[anchor_pos]

        return {
            "session_id": session_id,
            "anchor_line": anchor_ln,
            "before": [_render_message(conn, ln, r) for ln, r in before_slice],
            "anchor": _render_message(conn, anchor_ln, anchor_record),
            "after": [_render_message(conn, ln, r) for ln, r in after_slice],
        }
    finally:
        if own_conn:
            conn.close()


def get_session(session_id: str, conn: sqlite3.Connection | None = None) -> dict:
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        path = _session_file_path(conn, session_id)
        if path is None or not path.exists():
            raise ValueError(f"no transcript file on record for session {session_id}")
        records = _parse_all_lines(path)
        return {
            "session_id": session_id,
            "path": str(path),
            "messages": [_render_message(conn, ln, r) for ln, r in records],
        }
    finally:
        if own_conn:
            conn.close()
