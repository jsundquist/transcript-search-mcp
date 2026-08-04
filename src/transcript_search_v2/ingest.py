"""Incremental, append-aware ingestion of Claude Code transcript files.

Backfill and ongoing re-indexing share the same code path: ingesting a file
that's never been seen (bytes_read=0) *is* the backfill case, not a special
one. Only complete lines are parsed -- a trailing line with no newline yet
(the file is still being appended to) is left for the next pass.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from transcript_search_v2 import config, db
from transcript_search_v2.extract import Chunk, extract_chunks, extract_llm_call
from transcript_search_v2.schema import OtherRecord, TranscriptRecord, parse_line

logger = logging.getLogger(__name__)


def discover_files(root: Path | None = None) -> list[Path]:
    root = root or config.WATCH_ROOT
    if not root.exists():
        return []
    return sorted(root.glob("*/*.jsonl"))


class _ToolNameCache:
    """In-memory cache over db.tool_uses, so tool_result truncation can look
    up its producing tool's name even across incremental/resumed ingestion
    runs where the originating tool_use line was processed in a prior run."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._cache: dict[str, str] = {}

    def resolve(self, tool_use_id: str) -> str | None:
        if tool_use_id in self._cache:
            return self._cache[tool_use_id]
        name = db.resolve_tool_name(self._conn, tool_use_id)
        if name:
            self._cache[tool_use_id] = name
        return name

    def record(self, tool_use_id: str, tool_name: str) -> None:
        self._cache[tool_use_id] = tool_name
        db.record_tool_use(self._conn, tool_use_id, tool_name)


@dataclass
class IngestResult:
    path: str
    lines_processed: int
    chunks_added: int
    skipped_malformed: int


def _read_new_complete_lines(path: Path, byte_offset: int) -> tuple[list[str], int]:
    """Read whole lines starting at byte_offset. Returns (lines, new_byte_offset).
    An unterminated trailing line (file still being appended to) is left unread."""
    with open(path, "rb") as f:
        f.seek(byte_offset)
        data = f.read()
    if not data:
        return [], byte_offset

    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        return [], byte_offset  # no complete line yet

    complete = data[: last_newline + 1]
    new_offset = byte_offset + len(complete)
    # complete always ends with exactly one trailing "\n", so split() always
    # yields one spurious empty trailing element -- drop only that one, not
    # genuinely blank physical lines in the middle, so line counting used for
    # source_line_no (physical line number) stays exact.
    lines = complete.decode("utf-8", errors="replace").split("\n")[:-1]
    return lines, new_offset


def ingest_file(conn: sqlite3.Connection, path: Path) -> IngestResult:
    path_str = str(path)
    session_id = path.stem
    project_dir = path.parent.name

    stat = path.stat()
    state = db.get_file_state(conn, path_str)

    if state and state.last_seen_mtime == stat.st_mtime and state.last_seen_size == stat.st_size:
        return IngestResult(path_str, 0, 0, 0)

    byte_offset = state.bytes_read if state else 0
    lines_read_so_far = state.lines_read if state else 0

    lines, new_offset = _read_new_complete_lines(path, byte_offset)
    if not lines:
        # Nothing new and complete to parse, but stat changed (e.g. truncated
        # write in progress) -- still refresh last_seen so we don't re-stat forever.
        db.upsert_file_state(
            conn, path_str, session_id, project_dir, None, byte_offset,
            lines_read_so_far, stat.st_mtime, stat.st_size,
        )
        return IngestResult(path_str, 0, 0, 0)

    tool_names = _ToolNameCache(conn)
    chunks_added = 0
    skipped = 0
    project_cwd: str | None = None
    session_timestamps: list[str] = []
    session_message_count = 0
    git_branch: str | None = None

    for raw_line in lines:
        lines_read_so_far += 1  # counts every physical line seen, malformed or not,
        # so source_line_no stays a true physical line number for get_context to re-read.
        current_line_no = lines_read_so_far

        if not raw_line.strip():
            continue

        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError:
            skipped += 1
            logger.warning("skipping malformed JSON line in %s", path_str)
            continue

        try:
            record = parse_line(raw)
        except Exception:
            skipped += 1
            logger.warning("skipping line failing schema validation in %s", path_str)
            continue

        if isinstance(record, OtherRecord):
            continue

        record: TranscriptRecord
        project_cwd = record.cwd or project_cwd
        session_timestamps.append(record.timestamp)
        session_message_count += 1
        git_branch = record.gitBranch or git_branch

        chunks = extract_chunks(
            record,
            source_line_no=current_line_no,
            project_cwd_fallback=project_dir,
            resolve_tool_name=tool_names.resolve,
        )
        for chunk in chunks:
            if chunk.block_type == "tool_use" and chunk.tool_use_id and chunk.tool_name:
                tool_names.record(chunk.tool_use_id, chunk.tool_name)
            db.upsert_chunk(conn, chunk)
            chunks_added += 1

        llm_call = extract_llm_call(record, project_cwd_fallback=project_dir)
        if llm_call is not None:
            db.upsert_llm_call(conn, llm_call)

    if session_timestamps:
        db.upsert_session(
            conn,
            session_id,
            project_cwd or project_dir,
            session_timestamps,
            session_message_count,
            git_branch,
        )

    db.upsert_file_state(
        conn, path_str, session_id, project_dir, project_cwd, new_offset,
        lines_read_so_far, stat.st_mtime, stat.st_size,
    )
    conn.commit()

    return IngestResult(path_str, len(lines), chunks_added, skipped)


def backfill(conn: sqlite3.Connection, root: Path | None = None, trigger: str = "backfill") -> dict:
    run_id = db.start_run(conn, trigger)
    files = discover_files(root)
    files_touched = 0
    chunks_added = 0
    error: str | None = None
    try:
        for path in files:
            try:
                result = ingest_file(conn, path)
            except Exception as exc:  # noqa: BLE001 -- never abort the whole backfill on one bad file
                logger.exception("failed to ingest %s", path)
                db.mark_file_error(conn, str(path), str(exc))
                conn.commit()
                continue
            if result.chunks_added or result.lines_processed:
                files_touched += 1
            chunks_added += result.chunks_added
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        raise
    finally:
        db.finish_run(conn, run_id, files_touched, chunks_added, error)
        conn.commit()

    return {"files_discovered": len(files), "files_touched": files_touched, "chunks_added": chunks_added}
