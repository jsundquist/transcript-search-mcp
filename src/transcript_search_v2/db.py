"""SQLite connection management, schema, and low-level read/write helpers.

One database file (WAL mode) holds everything: chunk rows, an FTS5 keyword
index, and a sqlite-vec vector index, all writable in the same transaction --
deliberately not a separate vector store, so the keyword/vector/source-of-
truth views can never drift out of sync with each other.

FTS5/vec0 rows are kept in sync explicitly in application code (upsert_chunk /
upsert_embedding) rather than via SQL triggers, so the sync logic is ordinary,
testable Python instead of implicit trigger behavior.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import sqlite_vec

from transcript_search_v2 import config
from transcript_search_v2.extract import Chunk

SCHEMA_VERSION = 1

_DDL = f"""
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_cwd TEXT,
    project_dir TEXT NOT NULL,
    bytes_read INTEGER NOT NULL DEFAULT 0,
    lines_read INTEGER NOT NULL DEFAULT 0,
    last_seen_mtime REAL,
    last_seen_size INTEGER,
    last_ingested_at TEXT,
    status TEXT NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_cwd);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project_cwd TEXT NOT NULL,
    first_ts TEXT,
    last_ts TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    git_branch TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_cwd);
CREATE INDEX IF NOT EXISTS idx_sessions_last_ts ON sessions(last_ts);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_cwd TEXT NOT NULL,
    message_uuid TEXT NOT NULL,
    parent_uuid TEXT,
    role TEXT NOT NULL,
    block_type TEXT NOT NULL,
    block_index INTEGER NOT NULL,
    signal TEXT NOT NULL,
    is_sidechain INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    text TEXT NOT NULL,
    full_length INTEGER NOT NULL,
    source_line_no INTEGER NOT NULL,
    tool_name TEXT,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project_ts ON chunks(project_cwd, timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_message ON chunks(message_uuid);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_rowid INTEGER PRIMARY KEY,
    embedding FLOAT[{config.EMBEDDING_DIM}]
);

-- tool_use_id -> tool_name, so tool_result chunks (which only carry a
-- back-reference to the tool_use that produced them) can be truncated with
-- the right type-aware limit even across incremental/resumed ingestion runs.
CREATE TABLE IF NOT EXISTS tool_uses (
    tool_use_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    trigger TEXT NOT NULL,
    files_touched INTEGER,
    chunks_added INTEGER,
    error TEXT
);
"""


def connect(read_only: bool = False) -> sqlite3.Connection:
    config.ensure_data_dir()
    if read_only:
        uri = f"file:{config.DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    if row[0] is None:
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
    conn.commit()


# --- files -----------------------------------------------------------------


@dataclass
class FileState:
    path: str
    session_id: str
    bytes_read: int
    lines_read: int
    last_seen_mtime: float | None
    last_seen_size: int | None


def get_file_state(conn: sqlite3.Connection, path: str) -> FileState | None:
    row = conn.execute(
        "SELECT path, session_id, bytes_read, lines_read, last_seen_mtime, last_seen_size "
        "FROM files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return None
    return FileState(*row)


def upsert_file_state(
    conn: sqlite3.Connection,
    path: str,
    session_id: str,
    project_dir: str,
    project_cwd: str | None,
    bytes_read: int,
    lines_read: int,
    mtime: float,
    size: int,
) -> None:
    conn.execute(
        """
        INSERT INTO files (path, session_id, project_cwd, project_dir, bytes_read, lines_read,
                            last_seen_mtime, last_seen_size, last_ingested_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'ok')
        ON CONFLICT(path) DO UPDATE SET
            project_cwd = excluded.project_cwd,
            bytes_read = excluded.bytes_read,
            lines_read = excluded.lines_read,
            last_seen_mtime = excluded.last_seen_mtime,
            last_seen_size = excluded.last_seen_size,
            last_ingested_at = excluded.last_ingested_at,
            status = 'ok'
        """,
        (path, session_id, project_cwd, project_dir, bytes_read, lines_read, mtime, size),
    )


def mark_file_error(conn: sqlite3.Connection, path: str, error: str) -> None:
    conn.execute("UPDATE files SET status = ? WHERE path = ?", (f"error: {error}"[:500], path))


# --- sessions ----------------------------------------------------------------


def upsert_session(
    conn: sqlite3.Connection,
    session_id: str,
    project_cwd: str,
    timestamps: list[str],
    message_count_delta: int,
    git_branch: str | None,
) -> None:
    if not timestamps:
        return
    min_ts, max_ts = min(timestamps), max(timestamps)
    conn.execute(
        """
        INSERT INTO sessions (session_id, project_cwd, first_ts, last_ts, message_count, git_branch)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            project_cwd = excluded.project_cwd,
            first_ts = MIN(sessions.first_ts, excluded.first_ts),
            last_ts = MAX(sessions.last_ts, excluded.last_ts),
            message_count = sessions.message_count + excluded.message_count,
            git_branch = COALESCE(excluded.git_branch, sessions.git_branch)
        """,
        (session_id, project_cwd, min_ts, max_ts, message_count_delta, git_branch),
    )


# --- tool_uses ---------------------------------------------------------------


def record_tool_use(conn: sqlite3.Connection, tool_use_id: str, tool_name: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO tool_uses (tool_use_id, tool_name) VALUES (?, ?)",
        (tool_use_id, tool_name),
    )


def resolve_tool_name(conn: sqlite3.Connection, tool_use_id: str) -> str | None:
    row = conn.execute(
        "SELECT tool_name FROM tool_uses WHERE tool_use_id = ?", (tool_use_id,)
    ).fetchone()
    return row[0] if row else None


# --- chunks --------------------------------------------------------------------


def upsert_chunk(conn: sqlite3.Connection, chunk: Chunk) -> int:
    """Insert or replace a chunk + its FTS row. Returns the chunk's rowid.
    Embeddings are written separately via upsert_embedding once computed."""
    cur = conn.cursor()
    existing = cur.execute("SELECT rowid FROM chunks WHERE id = ?", (chunk.id,)).fetchone()
    if existing is not None:
        old_rowid = existing[0]
        cur.execute("DELETE FROM chunks_fts WHERE rowid = ?", (old_rowid,))
        cur.execute("DELETE FROM chunks_vec WHERE chunk_rowid = ?", (old_rowid,))

    cur.execute(
        """
        INSERT OR REPLACE INTO chunks
            (id, session_id, project_cwd, message_uuid, parent_uuid, role, block_type,
             block_index, signal, is_sidechain, timestamp, text, full_length,
             source_line_no, tool_name, embedded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            chunk.id,
            chunk.session_id,
            chunk.project_cwd,
            chunk.message_uuid,
            chunk.parent_uuid,
            chunk.role,
            chunk.block_type,
            chunk.block_index,
            chunk.signal,
            int(chunk.is_sidechain),
            chunk.timestamp,
            chunk.text,
            chunk.full_length,
            chunk.source_line_no,
            chunk.tool_name,
        ),
    )
    new_rowid = cur.execute("SELECT rowid FROM chunks WHERE id = ?", (chunk.id,)).fetchone()[0]
    cur.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (new_rowid, chunk.text))
    return new_rowid


def upsert_embedding(conn: sqlite3.Connection, chunk_rowid: int, vector: list[float]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO chunks_vec (chunk_rowid, embedding) VALUES (?, ?)",
        (chunk_rowid, sqlite_vec.serialize_float32(vector)),
    )
    conn.execute("UPDATE chunks SET embedded = 1 WHERE rowid = ?", (chunk_rowid,))


def pending_embeddings(conn: sqlite3.Connection, limit: int = 1000) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT rowid, text FROM chunks WHERE embedded = 0 ORDER BY rowid LIMIT ?", (limit,)
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


# --- run_history -----------------------------------------------------------------


def start_run(conn: sqlite3.Connection, trigger: str) -> int:
    cur = conn.execute(
        "INSERT INTO run_history (started_at, trigger) VALUES (datetime('now'), ?)", (trigger,)
    )
    return cur.lastrowid


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    files_touched: int,
    chunks_added: int,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE run_history SET finished_at = datetime('now'), files_touched = ?, "
        "chunks_added = ?, error = ? WHERE id = ?",
        (files_touched, chunks_added, error, run_id),
    )
