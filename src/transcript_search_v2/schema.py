"""Pydantic models for Claude Code's JSONL transcript envelope.

Only `type: "user"` / `type: "assistant"` lines carry conversational content;
other line types (file-history-snapshot, system, mode, etc.) are represented
by OtherRecord and skipped by the ingester. Content blocks use extra="allow"
so unrecognized block types are never silently dropped -- they just fall
through extract.py's classification as "unknown" signal instead of raising.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContentBlock(BaseModel):
    """A single block within message.content. Covers text, thinking, tool_use,
    tool_result, image, document, and any future/unknown block type."""

    model_config = ConfigDict(extra="allow")

    type: str

    # text
    text: str | None = None

    # thinking
    thinking: str | None = None
    signature: str | None = None

    # tool_use
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    # tool_result
    tool_use_id: str | None = None
    content: Any | None = None
    is_error: bool | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    service_tier: str | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    # User messages: plain string or content-block array.
    # Assistant messages: always a content-block array.
    content: str | list[ContentBlock]
    model: str | None = None
    usage: Usage | None = None
    stop_reason: str | None = None


class TranscriptRecord(BaseModel):
    """A `user` or `assistant` line -- the only line types with real content."""

    model_config = ConfigDict(extra="allow")

    type: Literal["user", "assistant"]
    uuid: str
    parentUuid: str | None = None
    timestamp: str
    sessionId: str
    cwd: str | None = None
    gitBranch: str | None = None
    isSidechain: bool = False
    message: Message


class OtherRecord(BaseModel):
    """Any line type other than user/assistant -- metadata/UI noise, skipped."""

    model_config = ConfigDict(extra="allow")

    type: str


def parse_line(raw: dict[str, Any]) -> TranscriptRecord | OtherRecord:
    """Parse a single decoded JSONL line into the appropriate record model."""
    line_type = raw.get("type")
    if line_type in ("user", "assistant"):
        return TranscriptRecord.model_validate(raw)
    return OtherRecord.model_validate(raw)
