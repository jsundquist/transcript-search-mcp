"""MCP entrypoint: registers tools, runs the initial backfill, then starts
the watcher + writer background tasks for the lifetime of the server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer

from transcript_search_v2 import config, db
from transcript_search_v2.tools import admin, analytics, health, search, temporal
from transcript_search_v2.watcher import watch_loop
from transcript_search_v2.writer import Writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    writer: Writer


# Set during lifespan startup; read by the reindex tool. A module-level
# singleton is simpler and more robust here than relying on SDK-internal
# request-context introspection, and this process only ever runs one server.
_app_ctx: AppContext | None = None


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    import asyncio

    global _app_ctx

    conn = db.connect()
    db.init_db(conn)
    writer = Writer(conn)

    logger.info("running startup backfill...")
    result = writer.run_startup_backfill()
    logger.info("startup backfill: %s", result)

    writer.start()
    watch_task = asyncio.create_task(watch_loop(writer), name="watch-loop")

    _app_ctx = AppContext(writer=writer)
    try:
        yield _app_ctx
    finally:
        _app_ctx = None
        watch_task.cancel()
        await writer.stop()
        conn.close()


mcp = MCPServer("transcript-search-v2", lifespan=lifespan)


# Bounds enforced at the tool boundary so a caller (human or LLM) passing a
# negative or absurdly large value can't request an unbounded result set --
# SQLite treats a negative LIMIT as "no limit", so this isn't just tidiness.
_MIN_LIMIT, _MAX_LIMIT = 1, 500
_MAX_OFFSET = 2000
_MAX_WINDOW = 100


def _clamp_limit(limit: int) -> int:
    return max(_MIN_LIMIT, min(_MAX_LIMIT, limit))


def _clamp_offset(offset: int) -> int:
    return max(0, min(_MAX_OFFSET, offset))


def _clamp_window(n: int) -> int:
    return max(0, min(_MAX_WINDOW, n))


@mcp.tool()
def keyword_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    roles: list[str] | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
    fuzzy_fallback: bool = True,
) -> list[dict]:
    """Full-text keyword/phrase search over your Claude Code transcripts.
    Quote the query (e.g. '"exact phrase"') for phrase search. If a strict
    search finds nothing, fuzzy_fallback retries once with typo'd terms
    corrected against the indexed vocabulary (results carry
    fuzzy_corrected_query when this kicks in). limit is capped at 500;
    offset pages past it."""
    return search.keyword_search(
        query,
        project,
        date_from,
        date_to,
        roles,
        include_low_signal,
        include_sidechains,
        _clamp_limit(limit),
        _clamp_offset(offset),
        fuzzy_fallback,
    )


@mcp.tool()
def semantic_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """Meaning-based search over your Claude Code transcripts -- finds
    relevant content even without literal keyword overlap. limit is capped
    at 500; offset pages past it."""
    return search.semantic_search(
        query,
        project,
        date_from,
        date_to,
        include_low_signal,
        include_sidechains,
        _clamp_limit(limit),
        _clamp_offset(offset),
    )


@mcp.tool()
def hybrid_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
    rerank: bool = True,
) -> list[dict]:
    """Combined keyword + semantic search (reciprocal-rank fusion), then
    reranked by signal/recency by default. Default choice when you're not
    sure whether a literal or a meaning-based match is more likely to find
    what you're after. limit is capped at 500; offset pages past it."""
    return search.hybrid_search(
        query,
        project,
        date_from,
        date_to,
        include_low_signal,
        include_sidechains,
        _clamp_limit(limit),
        _clamp_offset(offset),
        rerank,
    )


@mcp.tool()
def list_sessions(
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List Claude Code sessions, most recent first, optionally filtered by
    project (substring match) and/or date range. limit is capped at 500;
    offset pages past it."""
    return temporal.list_sessions(
        project, date_from, date_to, _clamp_limit(limit), _clamp_offset(offset)
    )


@mcp.tool()
def get_period(
    date_from: str,
    date_to: str,
    project: str | None = None,
    include_low_signal: bool = False,
    limit: int = 100,
) -> list[dict]:
    """All conversation content within a date range, without needing a
    keyword -- e.g. 'what did I work on last Tuesday'. limit is capped at
    500."""
    return temporal.get_period(date_from, date_to, project, include_low_signal, _clamp_limit(limit))


@mcp.tool()
def get_context(
    session_id: str,
    around_chunk_id: str | None = None,
    message_uuid: str | None = None,
    before: int = 5,
    after: int = 5,
) -> dict:
    """Full-fidelity (untruncated) conversation context surrounding a search
    result -- pass a chunk_id from a search result, or a message_uuid.
    before/after are each capped at 100 messages."""
    return temporal.get_context(
        session_id, around_chunk_id, message_uuid, _clamp_window(before), _clamp_window(after)
    )


@mcp.tool()
def get_session(session_id: str) -> dict:
    """Full untruncated dump of a single session's conversation."""
    return temporal.get_session(session_id)


@mcp.tool()
def status() -> dict:
    """Indexer status: files/sessions/chunks tracked, embedding coverage, last run."""
    return admin.status()


@mcp.tool()
def coverage(project: str | None = None) -> dict:
    """Compares transcript files discovered on disk against what's tracked in
    the index, flagging anything pending re-ingest."""
    return admin.coverage(project)


@mcp.tool()
def get_usage(
    group_by: str = "model",
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """LLM token usage and cost breakdown, grouped by "model" (default),
    "session", or "day". Answers "what did I spend today" / "which sessions
    were expensive". limit is capped at 500."""
    return analytics.get_usage(group_by, project, date_from, date_to, _clamp_limit(limit))


@mcp.tool()
def doctor(fix: bool = False) -> dict:
    """Environment and index health check beyond status()/coverage(): Python/
    dependency/sqlite-extension checks, db integrity, embedding backlog, and
    cost-anomaly alerts (today's spend vs 7-day average, expensive sessions).
    With fix=True, also attempts auto-repairs: backfill if the index is
    empty, drain the embedding backlog, quarantine+rebuild a corrupt db.
    Use this when the server is registered but searches return nothing."""
    return health.doctor(fix)


@mcp.tool()
def prune(
    older_than_days: int | None = None,
    project: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Delete chunks and LLM-call records older than older_than_days (falls
    back to TRANSCRIPT_SEARCH_RETENTION_DAYS if unset; raises if neither is
    given), optionally scoped to a project. dry_run=True (the default) only
    reports what would be deleted -- pass dry_run=False to actually delete.
    Deleted content will NOT reappear on a later reindex()/backfill even if
    its source file is unchanged, since the ingestion cursor already passed
    those lines; files/sessions bookkeeping is left untouched."""
    days = older_than_days if older_than_days is not None else config.DEFAULT_RETENTION_DAYS
    if days is None:
        raise ValueError(
            "older_than_days must be given (no TRANSCRIPT_SEARCH_RETENTION_DAYS default is set)"
        )
    return admin.prune(days, project, dry_run)


@mcp.tool()
async def reindex(project: str | None = None, session_id: str | None = None) -> str:
    """Manually trigger re-indexing of transcript files, optionally scoped to
    a project (substring match) or a specific session_id. Goes through the
    same write queue as the background watcher."""
    if _app_ctx is None:
        raise RuntimeError("server not fully started yet")
    count = await _app_ctx.writer.enqueue_reindex(project, session_id)
    return f"enqueued {count} file(s) for reindexing"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
