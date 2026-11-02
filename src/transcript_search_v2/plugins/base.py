"""SourcePlugin protocol -- the vendor abstraction for session source ingestion.

Any agent session source (Claude Code, Cursor, Copilot, etc.) implements this
protocol to plug into ingest.py's generic incremental file-walking loop.
ingest.py owns byte-offset tracking, session/tool_use bookkeeping, and DB
writes; it has no knowledge of any source's line format. Adding a second
source is one new file here plus registering it -- ingest.py needs no changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from transcript_search_v2.extract import Chunk, LLMCall
from transcript_search_v2.schema import TranscriptRecord

ToolNameResolver = Callable[[str], "str | None"]


@dataclass
class LineResult:
    """Everything ingest.py needs from parsing+extracting one physical line.

    kind:
      "content"   -- a real conversational line; chunks/llm_call/session
                      fields are populated.
      "skip"      -- well-formed but not indexable content (e.g. a metadata
                      line type), not counted as an error.
      "malformed" -- couldn't be parsed at all; ingest.py logs and skips just
                      this line, never aborting the file.
    """

    kind: Literal["content", "skip", "malformed"]
    chunks: list[Chunk] = field(default_factory=list)
    llm_call: LLMCall | None = None
    project_cwd: str | None = None
    timestamp: str | None = None
    git_branch: str | None = None


class SourcePlugin(Protocol):
    """Protocol for session source plugins.

    Example skeleton::

        class MyPlugin:
            name = "my-tool"

            def watch_dirs(self) -> list[Path]:
                return [Path("~/.my-tool/sessions").expanduser()]

            def is_mine(self, path: Path) -> bool:
                return path.suffix == ".jsonl"

            def parse_and_extract(self, raw_line, *, source_line_no,
                                   project_cwd_fallback, resolve_tool_name) -> LineResult:
                ...
    """

    name: str

    def watch_dirs(self) -> list[Path]:
        """Directories to monitor for new/changed session files."""
        ...

    def is_mine(self, path: Path) -> bool:
        """Return True if this plugin owns and should parse this file."""
        ...

    def parse_and_extract(
        self,
        raw_line: str,
        *,
        source_line_no: int,
        project_cwd_fallback: str,
        resolve_tool_name: ToolNameResolver,
    ) -> LineResult:
        """Parse one physical line and extract its chunks/llm_call. Never
        raises on malformed input -- returns LineResult(kind="malformed")
        instead, so one bad line never aborts the whole file."""
        ...

    def parse_raw(self, raw_line: str) -> TranscriptRecord | None:
        """Parse one physical line to its raw record (uuid/parentUuid/
        isSidechain/etc.), or None if blank/malformed/not a conversational
        record. Used by full-fidelity context/session recall (tools/temporal.py),
        which needs the original record shape rather than parse_and_extract's
        ingestion-shaped LineResult."""
        ...
