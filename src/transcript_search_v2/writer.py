"""The single writer: owns the one SQLite write connection.

Both the watcher and admin.reindex() funnel file-ingestion work through this
same queue/consumer, so there is exactly one code path that ever calls
ingest_file() for a given file after startup -- no two triggers can race on
the same file. The initial backfill scan runs synchronously before the
watcher/queue starts, so there's no concurrent writer at that point either.

Embedding is a second, decoupled loop: a chunk becomes keyword-searchable the
moment it's written, and semantically-searchable a little later once this
loop picks it up -- including chunks left un-embedded by a prior crash, since
"pending" is just "embedded = 0 in the db," not tracked separately in memory.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path

from transcript_search_v2 import config, db, embed, ingest

logger = logging.getLogger(__name__)

# Guards config.DB_PATH against doctor(fix=True)'s quarantine close/rename/
# reopen sequence running concurrently with a write here -- nothing else
# guarded DB_PATH access between the two before this existed. Writer holds
# it only around the actual write+commit, not the full ingest/embed work,
# so quarantine can't observe a half-written transaction; a threading.Lock
# (not asyncio.Lock) since doctor() may run on a different thread than the
# event loop (MCP tool dispatch).
DB_FILE_LOCK = threading.Lock()


def _apply_embeddings(conn: sqlite3.Connection, rowids: list[int], vectors: list) -> None:
    for rowid, vector in zip(rowids, vectors):
        db.upsert_embedding(conn, rowid, vector)
    conn.commit()


def drain_embeddings_batch(conn: sqlite3.Connection, batch_size: int) -> tuple[int, bool]:
    """Fetch, embed, and write one pending batch synchronously. Returns
    (chunks_embedded, had_batch) -- had_batch=False means the queue was
    empty; chunks_embedded=0 with had_batch=True means a batch existed but
    embedding it failed (logged here, not raised), so one bad chunk never
    aborts the caller's whole drain loop.

    Fully synchronous and does all its own DB access on the calling
    thread -- safe to call directly (as health.doctor(fix=True) does) but
    NOT via asyncio.to_thread against a connection created on the event
    loop thread, since sqlite3 connections are single-thread-bound by
    default. Writer._embed_loop instead offloads only the embed_texts call
    via to_thread and applies results with _apply_embeddings on the loop
    thread -- see that method."""
    batch = db.pending_embeddings(conn, limit=batch_size)
    if not batch:
        return 0, False

    rowids = [rowid for rowid, _ in batch]
    texts = [text for _, text in batch]
    try:
        vectors = embed.embed_texts(texts)
    except Exception:  # noqa: BLE001 -- one bad batch must not kill the caller's loop
        logger.exception("embedding batch failed")
        return 0, True

    _apply_embeddings(conn, rowids, vectors)
    return len(batch), True


class Writer:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ingest_queue: asyncio.Queue[Path] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    def run_startup_backfill(self) -> dict:
        """Synchronous initial catch-up scan, run before the queue-driven
        loops start (no concurrent writer yet, so this is safe without the
        queue)."""
        return ingest.backfill(self.conn, trigger="backfill")

    async def enqueue_file(self, path: Path) -> None:
        await self._ingest_queue.put(path)

    async def enqueue_reindex(
        self, project: str | None = None, session_id: str | None = None
    ) -> int:
        """Manual reindex trigger -- same queue as the watcher, so it can
        never race a concurrent watcher event for the same file."""
        files = ingest.discover_files()
        count = 0
        for path in files:
            if session_id and path.stem != session_id:
                continue
            if project and project not in str(path.parent):
                continue
            await self.enqueue_file(path)
            count += 1
        return count

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="writer-ingest-loop"),
            asyncio.create_task(self._embed_loop(), name="writer-embed-loop"),
        ]

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _ingest_loop(self) -> None:
        while True:
            path = await self._ingest_queue.get()
            try:
                with DB_FILE_LOCK:
                    ingest.ingest_file(self.conn, path)
            except Exception:  # noqa: BLE001 -- one bad file must not kill the loop
                logger.exception("failed to ingest %s", path)
                with DB_FILE_LOCK:
                    db.mark_file_error(self.conn, str(path), "ingest_loop error")
                    self.conn.commit()
            finally:
                self._ingest_queue.task_done()

    async def _embed_loop(self) -> None:
        while True:
            pending = db.pending_embeddings(self.conn, limit=config.EMBED_BATCH_SIZE)
            if not pending:
                await asyncio.sleep(1.0)
                continue

            rowids = [rowid for rowid, _ in pending]
            texts = [text for _, text in pending]
            try:
                vectors = await asyncio.to_thread(embed.embed_texts, texts)
            except Exception:  # noqa: BLE001
                logger.exception("embedding batch failed, will retry")
                await asyncio.sleep(2.0)
                continue

            with DB_FILE_LOCK:
                _apply_embeddings(self.conn, rowids, vectors)
