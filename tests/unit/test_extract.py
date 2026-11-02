from transcript_search_v2 import config
from transcript_search_v2.extract import extract_chunks
from transcript_search_v2.schema import parse_line


def _rec(**overrides):
    base = {
        "type": "assistant",
        "uuid": "a1",
        "parentUuid": None,
        "timestamp": "2026-07-20T10:00:00.000Z",
        "sessionId": "s1",
        "cwd": "/tmp/proj",
        "isSidechain": False,
        "message": {"role": "assistant", "content": []},
    }
    base.update(overrides)
    return parse_line(base)


def test_text_thinking_tool_use_produce_separate_chunks_with_expected_signal():
    rec = _rec(
        message={
            "role": "assistant",
            "content": [
                {"type": "text", "text": "here is my answer"},
                {"type": "thinking", "thinking": "pondering"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "x"}},
            ],
        }
    )
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert len(chunks) == 3
    by_type = {c.block_type: c for c in chunks}
    assert by_type["text"].signal == "high"
    assert by_type["thinking"].signal == "medium"
    # Read is routine/read-only -> low signal, unlike Edit/Write/Bash
    assert by_type["tool_use"].signal == "low"
    assert by_type["tool_use"].tool_name == "Read"


def test_edit_tool_use_is_medium_signal():
    rec = _rec(
        message={
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Edit", "input": {}}],
        }
    )
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert chunks[0].signal == "medium"


def test_tool_result_resolves_tool_name_and_type_aware_truncation():
    rec = _rec(
        type="user",
        uuid="u1",
        message={
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x" * 10_000}],
        },
    )
    chunks = extract_chunks(
        rec,
        source_line_no=1,
        project_cwd_fallback="/tmp/proj",
        resolve_tool_name=lambda _id: "Read",
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.tool_name == "Read"
    assert c.full_length == 10_000
    assert len(c.text) < 10_000
    assert c.text.startswith("x" * 10)


def test_tool_result_default_truncation_when_tool_unresolved():
    rec = _rec(
        type="user",
        uuid="u1",
        message={
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_unknown", "content": "y" * 5000}
            ],
        },
    )
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert chunks[0].tool_name is None
    assert len(chunks[0].text) <= config.TRUNCATE_TOOL_RESULT_DEFAULT + 60  # + truncation marker


def test_image_block_becomes_attachment_placeholder_not_base64():
    rec = _rec(
        type="user",
        uuid="u1",
        message={
            "role": "user",
            "content": [
                {"type": "image", "content": {"media_type": "image/png", "data": "base64stuff"}}
            ],
        },
    )
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert len(chunks) == 1
    assert chunks[0].block_type == "attachment_note"
    assert chunks[0].signal == "low"
    assert "base64stuff" not in chunks[0].text
    assert "image/png" in chunks[0].text


def test_plain_string_user_content_becomes_single_text_chunk():
    rec = _rec(type="user", uuid="u1", message={"role": "user", "content": "plain question"})
    chunks = extract_chunks(rec, source_line_no=7, project_cwd_fallback="/tmp/proj")
    assert len(chunks) == 1
    assert chunks[0].block_type == "text"
    assert chunks[0].text == "plain question"
    assert chunks[0].source_line_no == 7
    assert chunks[0].id == "u1:0"


def test_empty_text_block_produces_no_chunk():
    rec = _rec(message={"role": "assistant", "content": [{"type": "text", "text": ""}]})
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert chunks == []


def test_is_sidechain_propagates_to_chunk():
    rec = _rec(
        isSidechain=True,
        message={"role": "assistant", "content": [{"type": "text", "text": "subagent text"}]},
    )
    chunks = extract_chunks(rec, source_line_no=1, project_cwd_fallback="/tmp/proj")
    assert chunks[0].is_sidechain is True
