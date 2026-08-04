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
from transcript_search_v2.extract import Chunk, LLMCall

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

-- Term/doc-frequency vocabulary over chunks_fts, used by fuzzy.py to find a
-- close-spelling replacement for a query token with no exact FTS5 match.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab('chunks_fts', 'row');

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

-- One row per real API response (deduped from the multiple JSONL lines that
-- share a single response's usage stats -- see extract.extract_llm_call).
-- cost_usd is intentionally NOT stored here: it's computed at query time via
-- a JOIN against model_pricing, so pricing updates apply retroactively to
-- historical calls with no backfill/re-pricing step needed.
CREATE TABLE IF NOT EXISTS llm_calls (
    message_uuid TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_cwd TEXT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_calls_project_ts ON llm_calls(project_cwd, timestamp);

CREATE TABLE IF NOT EXISTS model_pricing (
    model TEXT PRIMARY KEY,
    input_cost_per_token REAL NOT NULL,
    output_cost_per_token REAL NOT NULL,
    cache_write_cost_per_token REAL,
    cache_read_cost_per_token REAL,
    as_of TEXT
);
"""

# Seed pricing for models this index is likely to see. Unknown/future models
# just get cost_usd = NULL in analytics queries (LEFT JOIN) until a row is
# added here -- never blocks ingestion or search.
_PRICING_SEED = [
    ("claude-opus-4-6", 0.000015, 0.000075, 0.00001875, 0.0000015, "2025-07-01"),
    ("claude-sonnet-4-6", 0.000003, 0.000015, 0.00000375, 0.0000003, "2025-07-01"),
    ("claude-sonnet-4-6[1m]", 0.000003, 0.000015, 0.00000375, 0.0000003, "2025-07-01"),
    ("claude-sonnet-5", 0.000003, 0.000015, 0.00000375, 0.0000003, "2026-01-01"),
    ("claude-opus-4-8", 0.000015, 0.000075, 0.00001875, 0.0000015, "2026-01-01"),
    ("claude-haiku-4-5-20251001", 0.000001, 0.000005, 0.00000125, 0.0000001, "2026-04-08"),
    ("claude-haiku-4-5", 0.000001, 0.000005, 0.00000125, 0.0000001, "2026-04-08"),
]


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
    for row in _PRICING_SEED:
        conn.execute(
            "INSERT OR IGNORE INTO model_pricing "
            "(model, input_cost_per_token, output_cost_per_token, "
            "cache_write_cost_per_token, cache_read_cost_per_token, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            row,
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


# --- llm_calls -----------------------------------------------------------------


def upsert_llm_call(conn: sqlite3.Connection, call: LLMCall) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO llm_calls
            (message_uuid, session_id, project_cwd, timestamp, model,
             input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, stop_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call.message_uuid,
            call.session_id,
            call.project_cwd,
            call.timestamp,
            call.model,
            call.input_tokens,
            call.output_tokens,
            call.cache_write_tokens,
            call.cache_read_tokens,
            call.stop_reason,
        ),
    )


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
