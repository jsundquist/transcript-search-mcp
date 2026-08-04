"""Claude Code source plugin -- parses ~/.claude/projects/*/*.jsonl transcripts.

All Claude-Code-specific parsing (JSON decode, schema validation, chunk/llm_call
extraction) lives here, behind the SourcePlugin protocol. ingest.py's
byte-offset-tracked file loop is entirely source-format agnostic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transcript_search_v2 import config
from transcript_search_v2.extract import extract_chunks, extract_llm_call
from transcript_search_v2.plugins.base import LineResult, ToolNameResolver
from transcript_search_v2.schema import OtherRecord, TranscriptRecord, parse_line

logger = logging.getLogger(__name__)


class ClaudeCodePlugin:
    name = "claude-code"

    def watch_dirs(self) -> list[Path]:
        return [config.WATCH_ROOT]

    def is_mine(self, path: Path) -> bool:
        return path.suffix == ".jsonl"

    def parse_and_extract(
        self,
        raw_line: str,
        *,
        source_line_no: int,
        project_cwd_fallback: str,
        resolve_tool_name: ToolNameResolver,
    ) -> LineResult:
        if not raw_line.strip():
            return LineResult(kind="skip")

        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed JSON line (line %d)", source_line_no)
            return LineResult(kind="malformed")

        try:
            record = parse_line(raw)
        except Exception:
            logger.warning("skipping line failing schema validation (line %d)", source_line_no)
            return LineResult(kind="malformed")

        if isinstance(record, OtherRecord):
            return LineResult(kind="skip")

        record: TranscriptRecord
        chunks = extract_chunks(
            record,
            source_line_no=source_line_no,
            project_cwd_fallback=project_cwd_fallback,
            resolve_tool_name=resolve_tool_name,
        )
        llm_call = extract_llm_call(record, project_cwd_fallback=project_cwd_fallback)

        return LineResult(
            kind="content",
            chunks=chunks,
            llm_call=llm_call,
            project_cwd=record.cwd,
            timestamp=record.timestamp,
            git_branch=record.gitBranch,
        )
