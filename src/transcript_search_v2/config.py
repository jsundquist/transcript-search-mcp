"""Central configuration: paths, truncation limits, timezone."""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# --- Paths -------------------------------------------------------------

DATA_DIR = Path(os.environ.get("TRANSCRIPT_SEARCH_DATA_DIR", "~/.transcript-search-v2")).expanduser()
DB_PATH = DATA_DIR / "index.db"

WATCH_ROOT = Path(os.environ.get("TRANSCRIPT_SEARCH_WATCH_ROOT", "~/.claude/projects")).expanduser()

# --- Timezone ------------------------------------------------------------
# Transcript timestamps are stored/queried internally as UTC. date_from/date_to
# params on MCP tools are interpreted in this local timezone and converted to
# UTC at the query boundary, so "last Tuesday" means the day you experienced,
# not a UTC-shifted one.
LOCAL_TZ = ZoneInfo(os.environ.get("TRANSCRIPT_SEARCH_TZ", os.environ.get("TZ", "America/Chicago")))

# --- Truncation limits (chars) -------------------------------------------
# Deliberately leaner than the old project's (50K/5K/3K) for a lighter v1
# index; bump here if recall quality suffers on large tool outputs.
TRUNCATE_TEXT = 2000
TRUNCATE_THINKING = 1500
TRUNCATE_TOOL_USE = 1000

TRUNCATE_TOOL_RESULT_DEFAULT = 500
TRUNCATE_TOOL_RESULT_BASH = 1500
TRUNCATE_TOOL_RESULT_READLIKE = 3000  # Read, Grep, Glob, Bash-output-shaped results

READLIKE_TOOLS = {"Read", "Grep", "Glob"}
BASH_TOOLS = {"Bash", "BashOutput"}

# Tools whose tool_use block is promoted to signal=medium (an intentional,
# "interesting" action) rather than signal=low (routine/read-only).
MEDIUM_SIGNAL_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"}

# --- Ingestion -------------------------------------------------------------
WATCH_DEBOUNCE_SECONDS = 0.3
EMBED_BATCH_SIZE = 32

# --- Embedding -------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
