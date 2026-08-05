# transcript-search-v2

MCP server that indexes local Claude Code conversation transcripts
(`~/.claude/projects/**/*.jsonl`) and exposes keyword, phrase, semantic, and
date-range search over them, plus full-fidelity context/session recall.

## Architecture

- **Data dir**: `~/.transcript-search-v2/`.
- **One SQLite database** (`index.db`, WAL mode) holds everything: chunk
  rows, an FTS5 keyword index, and a [sqlite-vec](https://github.com/asg017/sqlite-vec)
  vector index, all writable in the same transaction -- deliberately not a
  separate vector store, so the keyword/vector/source-of-truth views can
  never drift out of sync with each other.
- **Chunking**: one chunk per content block (text/thinking/tool_use/
  tool_result), not per message, each independently truncated and classified
  by `signal` (`high`/`medium`/`low`) so routine tool noise doesn't crowd out
  conversational content in search results.
- **Embeddings**: local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim,
  CPU device -- MPS/GPU init from a background thread hangs on Apple
  Silicon, see `embed.py`), no API cost, works offline.
- **Ingestion**: incremental and append-aware -- each file's byte offset is
  tracked in the `files` table, so re-scans only parse new complete lines.
  Backfill (initial scan) and ongoing re-indexing share the same code path.
- **Watcher**: a `watchfiles` background task on `~/.claude/projects/` feeds
  a single-writer queue (`writer.py`), so the watcher, manual `reindex()`
  calls, and startup backfill can never race on the same file. A separate,
  decoupled embedding loop means a chunk is keyword-searchable immediately
  on write and semantically-searchable a little later.

## Setup

```bash
uv sync
```

The embedding model downloads once on first use (~80MB, cached under
`~/.cache/huggingface`).

## Register with Claude Code

Copy the relevant block from `mcp.json.example` into your `~/.claude.json`
`mcpServers` section (or wherever your MCP client reads server configs from).

## Tools

- `keyword_search` / `semantic_search` / `hybrid_search` -- full-text,
  meaning-based, and combined (reciprocal-rank fusion, with signal/recency
  reranking by default) search. Quote the query (e.g. `'"exact phrase"'`) for
  phrase search. All support `project` (substring match against the working
  directory a message was sent from), `date_from`/`date_to` (interpreted in
  local time, see `config.LOCAL_TZ`), `include_low_signal`,
  `include_sidechains`, `limit` (capped at 500), and `offset` (for paging).
  `keyword_search` also retries once with typo-corrected terms
  (`fuzzy_fallback`, see `fuzzy.py`) if a strict search finds nothing.
- `list_sessions`, `get_period` -- browse by recency or date range without a
  keyword.
- `get_context`, `get_session` -- full-fidelity (untruncated) recall,
  re-reading the original `.jsonl` lines rather than the truncated index.
- `status`, `coverage`, `reindex` -- indexer health and manual re-index
  trigger.
- `get_usage` -- LLM token/cost breakdown by model, session, or day.
- `doctor` -- environment/index health check, with `fix=True` auto-repair
  (backfill, drain embedding backlog, quarantine+rebuild a corrupt db).
- `prune` -- delete chunks/LLM-call records older than N days, optionally
  scoped to a project; `dry_run=True` by default.

## Tests

```bash
uv run pytest
```

Unit tests cover schema parsing and chunk extraction/truncation/signal
classification in isolation. Integration tests exercise the full
ingest -> search -> context-recall pipeline against synthetic fixtures under
`tests/fixtures/sample_transcripts/`, including malformed lines, sidechain
filtering, idempotent re-ingestion, and the FTS5 hyphenated-term gotcha
(`sqlite-vec` parses as `NOT vec` unless quoted -- see `_sanitize_fts_query`
in `tools/search.py`).
