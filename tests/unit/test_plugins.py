from transcript_search_v2.plugins.claude_code import ClaudeCodePlugin

plugin = ClaudeCodePlugin()


def _extract(raw_line, line_no=1):
    return plugin.parse_and_extract(
        raw_line,
        source_line_no=line_no,
        project_cwd_fallback="/tmp/proj",
        resolve_tool_name=lambda _tool_use_id: None,
    )


def test_blank_line_is_skip():
    assert _extract("   \n").kind == "skip"


def test_malformed_json_is_malformed():
    assert _extract("{not json").kind == "malformed"


def test_line_failing_schema_validation_is_malformed():
    # missing required fields (uuid, timestamp, sessionId, message)
    assert _extract('{"type":"assistant"}').kind == "malformed"


def test_non_content_type_is_skip():
    result = _extract('{"type":"file-history-snapshot"}')
    assert result.kind == "skip"


def test_user_text_line_is_content_with_one_chunk():
    line = (
        '{"type":"user","uuid":"u1","parentUuid":null,'
        '"timestamp":"2026-07-20T10:00:00.000Z","sessionId":"s1",'
        '"cwd":"/tmp/proj","gitBranch":"main","isSidechain":false,'
        '"message":{"role":"user","content":"hello"}}'
    )
    result = _extract(line)
    assert result.kind == "content"
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "hello"
    assert result.project_cwd == "/tmp/proj"
    assert result.git_branch == "main"
    assert result.llm_call is None


def test_assistant_line_with_service_tier_yields_llm_call():
    line = (
        '{"type":"assistant","uuid":"a1","parentUuid":"u1",'
        '"timestamp":"2026-07-20T10:00:05.000Z","sessionId":"s1",'
        '"cwd":"/tmp/proj","gitBranch":"main","isSidechain":false,'
        '"message":{"role":"assistant","model":"claude-sonnet-5",'
        '"content":[{"type":"text","text":"hi"}],'
        '"usage":{"input_tokens":10,"output_tokens":5,"service_tier":"standard"}}}'
    )
    result = _extract(line)
    assert result.kind == "content"
    assert result.llm_call is not None
    assert result.llm_call.model == "claude-sonnet-5"
