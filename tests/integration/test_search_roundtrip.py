from __future__ import annotations

from transcript_search_v2 import db, embed, ingest
from transcript_search_v2.tools import search, temporal

from ..conftest import copy_fixture


def _embed_all_pending(conn):
    pending = db.pending_embeddings(conn, limit=1000)
    if not pending:
        return
    texts = [t for _, t in pending]
    vectors = embed.embed_texts(texts)
    for (rowid, _), vector in zip(pending, vectors):
        db.upsert_embedding(conn, rowid, vector)
    conn.commit()


def test_keyword_search_finds_known_text(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    results = search.keyword_search("zorbnax_widget", conn=conn)
    assert any("zorbnax_widget" in r["text"] for r in results)


def test_keyword_phrase_search_is_exact_not_scrambled(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl")
    ingest.backfill(conn)
    results = search.keyword_search('"file_a.txt and file_b.txt"', conn=conn)
    assert len(results) == 1
    # word-order-scrambled query should not match the phrase-quoted search
    scrambled = search.keyword_search('"file_b.txt and file_a.txt"', conn=conn)
    assert scrambled == []


def test_keyword_search_excludes_low_signal_by_default(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl")
    ingest.backfill(conn)
    default_results = search.keyword_search("file_a", conn=conn)
    assert all(r["signal"] != "low" for r in default_results)
    all_results = search.keyword_search("file_a", include_low_signal=True, conn=conn)
    assert any(r["signal"] == "low" for r in all_results)


def test_keyword_search_excludes_sidechains_by_default(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "sidechain.jsonl")
    ingest.backfill(conn)
    default_results = search.keyword_search("unique_sidechain_marker", conn=conn)
    assert default_results == []
    included = search.keyword_search(
        "unique_sidechain_marker", include_sidechains=True, conn=conn
    )
    assert len(included) == 1


def test_keyword_search_handles_hyphenated_terms(conn, tmp_env):
    # FTS5 MATCH treats a bare '-' as NOT/column-filter syntax, so an
    # unquoted "sqlite-vec"-style query must not be parsed as "sqlite NOT vec".
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    conn.execute(
        "UPDATE chunks SET text = 'we should use sqlite-vec for this' WHERE message_uuid = 'ne-a1'"
    )
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        "INSERT INTO chunks_fts(rowid, text) SELECT rowid, text FROM chunks"
    )
    conn.commit()
    results = search.keyword_search("sqlite-vec", conn=conn)
    assert any("sqlite-vec" in r["text"] for r in results)


def test_project_filter(conn, tmp_env):
    # project_cwd is taken from the record's own "cwd" field (here
    # "/tmp/fixture-proj", set in the fixture), not from the directory the
    # file happens to live under -- cwd is the more reliable signal since
    # directory-name decoding is lossy for paths containing literal dashes.
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    assert search.keyword_search("zorbnax_widget", project="fixture-proj", conn=conn)
    assert search.keyword_search("zorbnax_widget", project="nonexistent-project", conn=conn) == []


def test_date_range_filter(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    assert search.keyword_search(
        "zorbnax_widget", date_from="2026-07-20", date_to="2026-07-20", conn=conn
    )
    assert (
        search.keyword_search(
            "zorbnax_widget", date_from="2026-07-21", date_to="2026-07-22", conn=conn
        )
        == []
    )


def test_keyword_search_fuzzy_fallback_corrects_typo(conn, tmp_env):
    # underscore is a token separator for FTS5's unicode61 tokenizer, so
    # "zorbnax_widget" indexes as two tokens ("zorbnax", "widget") -- the
    # typo test uses a single-word token so the correction lookup lines up
    # with what's actually in chunks_vocab.
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)

    strict = search.keyword_search("dashbord", fuzzy_fallback=False, conn=conn)
    assert strict == []

    fuzzy_results = search.keyword_search("dashbord", conn=conn)
    assert any("dashboard" in r["text"] for r in fuzzy_results)
    assert all("fuzzy_corrected_query" in r for r in fuzzy_results)


def test_keyword_search_fuzzy_fallback_does_not_trigger_for_phrase_queries(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    # a quoted phrase with a typo should not get "corrected" into a
    # different phrase -- phrase search stays strict.
    results = search.keyword_search('"dashbord header"', conn=conn)
    assert results == []


def test_keyword_search_fuzzy_fallback_leaves_uncorrectable_query_empty(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    results = search.keyword_search("completely unrelated gibberish query", conn=conn)
    assert results == []


def test_semantic_search_finds_paraphrase_without_literal_overlap(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _embed_all_pending(conn)

    paraphrase = "what is the purpose of that dashboard component"
    results = search.semantic_search(paraphrase, conn=conn)
    assert any("zorbnax_widget" in r["text"] for r in results)
    # keyword search on the same paraphrase should find nothing (no literal overlap)
    assert search.keyword_search(paraphrase, conn=conn) == []


def test_hybrid_search_reranked_results_carry_rerank_score(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _embed_all_pending(conn)

    results = search.hybrid_search("zorbnax_widget", conn=conn)
    assert results
    assert all("rerank_score" in r for r in results)
    assert all("score" in r for r in results)  # original RRF score preserved


def test_hybrid_search_rerank_false_skips_rerank_score(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    _embed_all_pending(conn)

    results = search.hybrid_search("zorbnax_widget", rerank=False, conn=conn)
    assert results
    assert all("rerank_score" not in r for r in results)


def test_get_context_reads_full_untruncated_surrounding_messages(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "tool_use_result.jsonl")
    ingest.backfill(conn)
    row = conn.execute(
        "SELECT id, session_id FROM chunks WHERE message_uuid = 'tr-a2'"
    ).fetchone()
    ctx = temporal.get_context(
        row["session_id"], around_chunk_id=row["id"], before=5, after=5, conn=conn
    )
    assert ctx["anchor"]["message_uuid"] == "tr-a2"
    before_uuids = [m["message_uuid"] for m in ctx["before"]]
    assert before_uuids == ["tr-u1", "tr-a1", "tr-u2"]


def test_get_context_excludes_sidechain_messages(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "sidechain.jsonl")
    ingest.backfill(conn)
    row = conn.execute("SELECT id, session_id FROM chunks WHERE message_uuid = 'sc-a2'").fetchone()
    ctx = temporal.get_context(row["session_id"], around_chunk_id=row["id"], before=5, conn=conn)
    before_uuids = [m["message_uuid"] for m in ctx["before"]]
    assert "sc-a1" not in before_uuids  # sidechain message must not be spliced in
    assert before_uuids == ["sc-u1"]


def test_get_session_full_dump(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    row = conn.execute("SELECT session_id FROM sessions WHERE session_id = 'ne-session'").fetchone()
    sess = temporal.get_session(row["session_id"], conn=conn)
    assert len(sess["messages"]) == 2
    assert sess["messages"][0]["blocks"][0]["text"] == "what does the zorbnax_widget do"


def test_list_sessions_and_get_period(conn, tmp_env):
    copy_fixture(tmp_env["watch_root"], "normal_exchange.jsonl")
    ingest.backfill(conn)
    sessions = temporal.list_sessions(conn=conn)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "ne-session"

    period = temporal.get_period("2026-07-20", "2026-07-20", conn=conn)
    assert len(period) == 2
    out_of_range = temporal.get_period("2026-07-21", "2026-07-22", conn=conn)
    assert out_of_range == []
