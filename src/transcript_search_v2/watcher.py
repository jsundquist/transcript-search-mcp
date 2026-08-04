"""Background filesystem watcher over ~/.claude/projects/.

watchfiles.awatch batches rapid successive writes (Claude Code can append
several lines per turn) into one debounced event via its own `debounce`
param, so no separate coalescing logic is needed here -- just filter to
.jsonl paths and enqueue each changed file exactly once per batch.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchfiles import awatch

from transcript_search_v2 import config
from transcript_search_v2.writer import Writer

logger = logging.getLogger(__name__)


async def watch_loop(writer: Writer, root: Path | None = None) -> None:
    root = root or config.WATCH_ROOT
    root.mkdir(parents=True, exist_ok=True)

    logger.info("watching %s for transcript changes", root)
    async for changes in awatch(root, debounce=int(config.WATCH_DEBOUNCE_SECONDS * 1000)):
        seen: set[Path] = set()
        for _change, path_str in changes:
            path = Path(path_str)
            if path.suffix != ".jsonl" or path in seen:
                continue
            seen.add(path)
            await writer.enqueue_file(path)
