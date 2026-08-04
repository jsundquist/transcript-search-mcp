from transcript_search_v2.schema import OtherRecord, TranscriptRecord, parse_line


def test_parse_line_user_string_content():
    rec = parse_line(
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-07-20T10:00:00.000Z",
            "sessionId": "s1",
            "cwd": "/tmp/proj",
            "isSidechain": False,
            "message": {"role": "user", "content": "hello"},
        }
    )
    assert isinstance(rec, TranscriptRecord)
    assert rec.message.content == "hello"


def test_parse_line_skips_unknown_types():
    rec = parse_line({"type": "file-history-snapshot", "uuid": "fh1"})
    assert isinstance(rec, OtherRecord)
    assert rec.type == "file-history-snapshot"


def test_parse_line_unknown_content_block_type_does_not_raise():
    rec = parse_line(
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": None,
            "timestamp": "2026-07-20T10:00:00.000Z",
            "sessionId": "s1",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "content": [{"type": "some_future_block_type", "weird_field": 123}],
            },
        }
    )
    assert isinstance(rec, TranscriptRecord)
    assert rec.message.content[0].type == "some_future_block_type"
