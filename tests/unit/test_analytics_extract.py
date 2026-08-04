from transcript_search_v2.extract import extract_llm_call
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


def test_no_llm_call_for_user_messages():
    rec = _rec(type="user", uuid="u1", message={"role": "user", "content": "hi"})
    assert extract_llm_call(rec, project_cwd_fallback="/tmp/proj") is None


def test_no_llm_call_without_usage():
    rec = _rec(message={"role": "assistant", "content": [], "model": "claude-sonnet-5"})
    assert extract_llm_call(rec, project_cwd_fallback="/tmp/proj") is None


def test_no_llm_call_when_usage_missing_service_tier():
    """A split-response entry (thinking/tool_use block) shares the parent
    response's usage stats but lacks service_tier -- must not be counted,
    or token/cost totals inflate 2-3x across the split lines."""
    rec = _rec(
        message={
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "..."}],
            "model": "claude-sonnet-5",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
    )
    assert extract_llm_call(rec, project_cwd_fallback="/tmp/proj") is None


def test_llm_call_extracted_when_service_tier_present():
    rec = _rec(
        message={
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "model": "claude-sonnet-5",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 100,
                "service_tier": "standard",
            },
            "stop_reason": "end_turn",
        }
    )
    call = extract_llm_call(rec, project_cwd_fallback="/tmp/proj")
    assert call is not None
    assert call.message_uuid == "a1"
    assert call.model == "claude-sonnet-5"
    assert call.input_tokens == 20
    assert call.output_tokens == 15
    assert call.cache_write_tokens == 5
    assert call.cache_read_tokens == 100
    assert call.stop_reason == "end_turn"
