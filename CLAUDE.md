# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MCP server that indexes local Claude Code conversation transcripts (`~/.claude/projects/**/*.jsonl`)
into a SQLite index and exposes keyword/semantic/hybrid search, date-range browsing, full-fidelity
context recall, and usage/cost analytics as MCP tools. See `README.md` for the full tool list and
per-tool parameter semantics (those aren't repeated here).

## Commands

```bash
uv sync                 # install deps (dev group includes pytest)
uv run pytest           # run the full test suite
uv run pytest tests/unit/test_extract.py::test_name  # run a single test
uv run pytest tests/unit      # unit tests only (fast, no fixtures on disk)
uv run pytest tests/integration  # integration tests (full ingest -> search -> recall pipeline)
uv run python -m transcript_search_v2.server  # run the server directly (normally launched via MCP client, see mcp.json.example)
```

No lint/format/type-check tooling is configured in `pyproject.toml`.

Test config (`pyproject.toml`): `pythonpath = ["src"]`, `testpaths = ["tests"]` — tests import
`transcript_search_v2` directly from `src/` without an editable install.

## Architecture

### Data flow: one writer, two decoupled loops

`server.py`'s `lifespan()` is the only process entrypoint. On startup it opens the single SQLite
connection, runs a synchronous backfill (`ingest.backfill`), then starts a `Writer` (`writer.py`)
that owns that connection for the rest of the process lifetime, plus a `watchfiles`-based
`watch_loop` (`watcher.py`) on `~/.claude/projects/`.

- **Ingest loop**: consumes an `asyncio.Queue` of file paths fed by both the watcher and manual
  `reindex()` calls — this is deliberate: funneling both triggers through the same queue means
  exactly one code path ever calls `ingest_file()` for a given file, so they can't race on it.
- **Embed loop**: fully decoupled from ingest. It polls `chunks WHERE embedded = 0` rather than
  tracking pending work in memory, so a chunk is keyword-searchable the instant it's written and
  semantically-searchable once this loop catches up (including after a crash mid-batch — no
  separate "pending" bookkeeping to reconcile).
- Ingestion is incremental/append-aware: `files.bytes_read` tracks each file's cursor, so re-scans
  only parse new complete lines. Backfill and ongoing re-indexing share `ingest_file()`.

### Storage: one SQLite db, kept in sync in application code

`db.py` holds all DDL and low-level read/write helpers. `chunks` (source of truth), `chunks_fts`
(FTS5, porter-stemmed) and `chunks_vec` (sqlite-vec) are three views over the same conceptual data,
written together in `upsert_chunk`/`upsert_embedding` rather than via SQL triggers — sync logic is
ordinary Python, not implicit trigger behavior. When touching chunk writes, keep all three in step.

`chunks` is one row per *content block* (text/thinking/tool_use/tool_result), not per message —
see `extract.py` for the chunking/truncation/signal-classification logic. Each chunk gets a
`signal` of `high`/`medium`/`low` (config.py's `MEDIUM_SIGNAL_TOOLS` etc.) so routine tool noise
doesn't crowd out conversational content in default search results.

`tool_uses` is a side table mapping `tool_use_id -> tool_name` so that a later `tool_result` chunk
(which only carries a back-reference) can still be truncated with the type-aware limit
(`TRUNCATE_TOOL_RESULT_READLIKE` etc. in `config.py`) even across incremental/resumed ingestion.

`llm_calls` stores raw token counts only; `cost_usd` is computed at query time via a `model_pricing`
JOIN so pricing updates apply retroactively without a backfill step.

### Plugin boundary

`plugins/base.py` defines a `SourcePlugin` protocol for transcript sources; `plugins/claude_code.py`
is the only implementation. `ingest.py` goes through this abstraction, but `tools/temporal.py`'s
`get_context`/`get_session` currently bypass it and hardcode Claude-Code JSONL parsing directly —
know this if you're adding a second plugin or touching full-fidelity recall.

### Tool boundary (`server.py`)

All MCP tool functions live in `server.py` as thin wrappers that clamp `limit`/`offset`/window
params (`_clamp_limit`, `_clamp_offset`, `_clamp_window`) before delegating to `tools/*.py`. A
negative SQLite `LIMIT` means "no limit," so this clamping is a correctness requirement, not just
tidiness — always route new tools' numeric params through these (or equivalent) clamps.

`tools/` is organized by concern: `search.py` (keyword/semantic/hybrid), `temporal.py`
(sessions/period/context), `admin.py` (status/coverage/prune/reindex support), `health.py`
(doctor), `analytics.py` (usage/cost).

### Known gotchas

- FTS5 MATCH parses hyphenated terms like `sqlite-vec` as `NOT vec` unless quoted — handled by
  `_sanitize_fts_query` in `tools/search.py`. Malformed MATCH syntax is wrapped as `ValueError`.
- `chunks_fts` uses porter stemming, so raw query tokens often don't equal stemmed vocabulary
  terms — relevant when working on `fuzzy.py`'s correction logic.
- Embedding runs CPU-only (`embed.py`): MPS/GPU model init from a background thread hangs on Apple
  Silicon.
- `config.LOCAL_TZ` (`TRANSCRIPT_SEARCH_TZ` env var, falls back to `TZ`, then `America/Chicago`) is
  the boundary for all `date_from`/`date_to` interpretation — transcripts are stored/queried
  internally as UTC.
- `prune()` never runs automatically; `config.DEFAULT_RETENTION_DAYS` is only a suggested default
  for the tool's `older_than_days` param, sourced from `TRANSCRIPT_SEARCH_RETENTION_DAYS`.
- Env overrides: `TRANSCRIPT_SEARCH_DATA_DIR` (default `~/.transcript-search-v2/`),
  `TRANSCRIPT_SEARCH_WATCH_ROOT` (default `~/.claude/projects`).

## Tests

`tests/unit/` covers schema parsing and chunk extraction/truncation/signal classification in
isolation (no db). `tests/integration/` exercises the full ingest -> search -> context-recall
pipeline against synthetic fixtures under `tests/fixtures/sample_transcripts/`, using the `conn`/
`tmp_env` fixtures in `conftest.py` to redirect `config.DATA_DIR`/`DB_PATH`/`WATCH_ROOT` into a
temp dir per test. Fixture `.jsonl` files are copied into place by `copy_fixture()`, which renames
the destination to the `sessionId` found inside the file (not the fixture's own filename) to match
how real transcript files are named on disk.
