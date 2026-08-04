from __future__ import annotations

from pathlib import Path

from transcript_search_v2 import ingest
from transcript_search_v2.extract import Chunk
from transcript_search_v2.plugins.base import LineResult
from transcript_search_v2.plugins.claude_code import ClaudeCodePlugin

from ..conftest import copy_fixture


class FakePlugin:
    """Minimal second SourcePlugin, proving ingest.py needs no changes to
    support a new source -- exactly the claim in plugins/base.py's docstring."""

    name = "fake-source"

    def watch_dirs(self) -> list[Path]:
        return []

    def is_mine(self, path: Path) -> bool:
        return path.suffix == ".fake"

    def parse_and_extract(self, raw_line, *, source_line_no, project_cwd_fallback, resolve_tool_name):
        if not raw_line.strip():
            return LineResult(kind="skip")
        chunk = Chunk(
            id=f"fake:{source_line_no}",
            session_id="fake-session",
            project_cwd=project_cwd_fallback,
            message_uuid=f"fake-msg-{source_line_no}",
            parent_uuid=None,
            role="user",
            block_type="text",
            block_index=0,
            signal="high",
            is_sidechain=False,
            timestamp="2026-07-20T10:00:00.000Z",
            text=raw_line.strip(),
            full_length=len(raw_line.strip()),
            source_line_no=source_line_no,
            tool_name=None,
        )
        return LineResult(kind="content", chunks=[chunk], timestamp=chunk.timestamp)


def test_backfill_routes_files_to_the_owning_plugin(conn, tmp_env):
    claude_code_path = copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")

    fake_dir = tmp_env["watch_root"] / "-fake-proj"
    fake_dir.mkdir()
    fake_path = fake_dir / "fake-session.fake"
    fake_path.write_text("hello from a different source\nsecond line\n")

    plugins = [ClaudeCodePlugin(), FakePlugin()]
    result = ingest.backfill(conn, root=tmp_env["watch_root"], plugins=plugins)
    assert result["files_discovered"] == 2
    assert result["files_touched"] == 2

    claude_rows = conn.execute(
        "SELECT text FROM chunks WHERE session_id = 'ne-session'"
    ).fetchall()
    assert len(claude_rows) > 0

    fake_rows = conn.execute(
        "SELECT text FROM chunks WHERE session_id = 'fake-session' ORDER BY source_line_no"
    ).fetchall()
    assert [r["text"] for r in fake_rows] == ["hello from a different source", "second line"]

    assert claude_code_path.exists()
    assert fake_path.exists()
