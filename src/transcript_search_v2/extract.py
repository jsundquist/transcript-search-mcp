"""Turn a parsed TranscriptRecord into a list of indexable Chunks.

Chunking is per content block, not per message: a single assistant turn with
a short text reply plus a large tool_use becomes multiple independently
truncated/classified chunks, so one giant tool call doesn't crowd out the
conversational text sharing its message.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from transcript_search_v2 import config
from transcript_search_v2.schema import ContentBlock, TranscriptRecord

Signal = str  # "high" | "medium" | "low"

# Resolves a tool_use_id (from a tool_result block) to the tool name that
# produced it, so tool_result truncation can be type-aware. tool_result
# blocks don't carry the tool name themselves -- only a back-reference to the
# tool_use.id from an earlier message. ingest.py supplies the real resolver
# (in-memory cache backed by a persistent table, since incremental re-ingest
# can resume mid-file after the originating tool_use was already processed).
ToolNameResolver = Callable[[str], str | None]


def _no_resolver(_tool_use_id: str) -> str | None:
    return None


@dataclass
class Chunk:
    id: str
    session_id: str
    project_cwd: str
    message_uuid: str
    parent_uuid: str | None
    role: str
    block_type: str
    block_index: int
    signal: Signal
    is_sidechain: bool
    timestamp: str
    text: str
    full_length: int
    source_line_no: int
    tool_name: str | None
    tool_use_id: str | None = None  # raw "toolu_xxx" id, set only for tool_use blocks


@dataclass
class LLMCall:
    message_uuid: str
    session_id: str
    project_cwd: str
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    stop_reason: str | None


def extract_llm_call(record: TranscriptRecord, project_cwd_fallback: str) -> LLMCall | None:
    """Extract token/cost metadata for one real API response, or None.

    Claude Code splits a single API response into multiple JSONL lines --
    one per content block (thinking, text, tool_use) -- all sharing an
    identical `usage` object. Only the line carrying the final response has
    `usage.service_tier` set; the others are re-emissions of the same usage
    data and must NOT be counted again, or token/cost totals inflate 2-3x.
    """
    if record.type != "assistant":
        return None
    usage = record.message.usage
    if usage is None or usage.service_tier is None:
        return None
    return LLMCall(
        message_uuid=record.uuid,
        session_id=record.sessionId,
        project_cwd=record.cwd or project_cwd_fallback,
        timestamp=record.timestamp,
        model=record.message.model or "",
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        stop_reason=record.message.stop_reason,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


def _tool_result_limit(tool_name: str | None) -> int:
    if tool_name in config.READLIKE_TOOLS:
        return config.TRUNCATE_TOOL_RESULT_READLIKE
    if tool_name in config.BASH_TOOLS:
        return config.TRUNCATE_TOOL_RESULT_BASH
    return config.TRUNCATE_TOOL_RESULT_DEFAULT


def _tool_use_signal(tool_name: str | None) -> Signal:
    if tool_name in config.MEDIUM_SIGNAL_TOOLS:
        return "medium"
    return "low"


def _tool_result_text(content: object) -> tuple[str, bool]:
    """Extract text from a tool_result block's `content` (str or block list).
    Returns (text, has_non_text_content)."""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts: list[str] = []
        has_other = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                has_other = True
        return "\n".join(parts), has_other
    return str(content), False


_BlockResult = tuple[str, str, Signal, "str | None", int, "str | None"]
# (block_type, raw_text, signal, tool_name, truncate_limit, tool_use_id)


def _block_from_user_block(
    block: ContentBlock,
    block_index: int,
    resolve_tool_name: ToolNameResolver,
) -> _BlockResult | None:
    if block.type == "text":
        return "text", block.text or "", "high", None, config.TRUNCATE_TEXT, None

    if block.type == "tool_result":
        tool_name = resolve_tool_name(block.tool_use_id) if block.tool_use_id else None
        raw_text, has_other = _tool_result_text(block.content)
        if has_other and not raw_text:
            raw_text = "[non-text tool_result content]"
        return "tool_result", raw_text, "low", tool_name, _tool_result_limit(tool_name), None

    if block.type in ("image", "document"):
        media_type = None
        if isinstance(block.content, dict):
            media_type = block.content.get("media_type")
        placeholder = f"[attachment: {block.type}{f' ({media_type})' if media_type else ''}]"
        return (
            "attachment_note",
            placeholder,
            "low",
            None,
            config.TRUNCATE_TOOL_RESULT_DEFAULT,
            None,
        )

    # Unknown block type -- keep it, don't drop silently.
    text = json.dumps(block.model_dump(exclude_none=True), default=str)
    return block.type, text, "low", None, config.TRUNCATE_TOOL_RESULT_DEFAULT, None


def _block_from_assistant_block(
    block: ContentBlock,
    block_index: int,
) -> _BlockResult | None:
    if block.type == "text":
        return "text", block.text or "", "high", None, config.TRUNCATE_TEXT, None

    if block.type == "thinking":
        return "thinking", block.thinking or "", "medium", None, config.TRUNCATE_THINKING, None

    if block.type == "tool_use":
        tool_name = block.name
        input_json = json.dumps(block.input or {}, default=str)
        text = f"[{tool_name}] {input_json}"
        return (
            "tool_use",
            text,
            _tool_use_signal(tool_name),
            tool_name,
            config.TRUNCATE_TOOL_USE,
            block.id,
        )

    # Unknown block type -- keep it, don't drop silently.
    text = json.dumps(block.model_dump(exclude_none=True), default=str)
    return block.type, text, "low", None, config.TRUNCATE_TOOL_RESULT_DEFAULT, None


def _raw_blocks(
    record: TranscriptRecord,
    resolve_tool_name: ToolNameResolver,
) -> list[_BlockResult]:
    content = record.message.content
    if isinstance(content, str):
        return [("text", content, "high", None, config.TRUNCATE_TEXT, None)] if content else []
    blocks: list[_BlockResult] = []
    for i, block in enumerate(content):
        if record.type == "user":
            result = _block_from_user_block(block, i, resolve_tool_name)
        else:
            result = _block_from_assistant_block(block, i)
        if result is not None:
            blocks.append(result)
    return blocks


def render_full(
    record: TranscriptRecord,
    resolve_tool_name: ToolNameResolver = _no_resolver,
) -> list[dict]:
    """Untruncated block rendering for full-fidelity context/session recall --
    same block extraction as extract_chunks, minus the truncation caps."""
    return [
        {"block_type": bt, "text": text, "tool_name": tn}
        for bt, text, _signal, tn, _limit, _tool_use_id in _raw_blocks(record, resolve_tool_name)
        if text
    ]


def extract_chunks(
    record: TranscriptRecord,
    source_line_no: int,
    project_cwd_fallback: str,
    resolve_tool_name: ToolNameResolver = _no_resolver,
) -> list[Chunk]:
    project_cwd = record.cwd or project_cwd_fallback
    blocks = _raw_blocks(record, resolve_tool_name)

    chunks: list[Chunk] = []
    for block_index, (block_type, raw_text, signal, tool_name, limit, tool_use_id) in enumerate(
        blocks
    ):
        if not raw_text:
            continue
        chunks.append(
            Chunk(
                id=f"{record.uuid}:{block_index}",
                session_id=record.sessionId,
                project_cwd=project_cwd,
                message_uuid=record.uuid,
                parent_uuid=record.parentUuid,
                role=record.type,
                block_type=block_type,
                block_index=block_index,
                signal=signal,
                is_sidechain=record.isSidechain,
                timestamp=record.timestamp,
                text=_truncate(raw_text, limit),
                full_length=len(raw_text),
                source_line_no=source_line_no,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
            )
        )
    return chunks
