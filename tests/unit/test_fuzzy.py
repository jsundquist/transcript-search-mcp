import sqlite3

import pytest

from transcript_search_v2 import db, fuzzy


@pytest.fixture
def vocab_conn(tmp_env):
    con = db.connect(read_only=False)
    db.init_db(con)
    yield con
    con.close()


def _seed(con: sqlite3.Connection, texts: list[str]):
    for i, text in enumerate(texts):
        con.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (i + 1, text))
    con.commit()


def test_edit_distance_basics():
    assert fuzzy._edit_distance("kitten", "kitten") == 0
    assert fuzzy._edit_distance("kitten", "sitten") == 1
    assert fuzzy._edit_distance("", "abc") == 3


def test_closest_vocab_term_exact_hit_needs_no_correction(tmp_env, vocab_conn):
    _seed(vocab_conn, ["the zorbnax widget renders fine", "another unrelated line"])
    match = fuzzy._closest_vocab_term(vocab_conn, "zorbnax", max_distance=2)
    assert match is None


def test_closest_vocab_term_corrects_a_typo(tmp_env, vocab_conn):
    _seed(vocab_conn, ["the zorbnax widget renders fine"])
    match = fuzzy._closest_vocab_term(vocab_conn, "zorbnaxx", max_distance=2)
    assert match == "zorbnax"


def test_closest_vocab_term_returns_none_when_nothing_close(tmp_env, vocab_conn):
    _seed(vocab_conn, ["completely different vocabulary here"])
    match = fuzzy._closest_vocab_term(vocab_conn, "zzzzzzzzzz", max_distance=2)
    assert match is None


def test_correct_query_only_touches_uncorrectable_free(tmp_env, vocab_conn):
    _seed(vocab_conn, ["the zorbnax widget renders fine"])
    corrected, changed = fuzzy.correct_query(vocab_conn, "zorbnaxx widget")
    assert changed is True
    assert corrected == "zorbnax widget"


def test_correct_query_no_change_when_already_exact(tmp_env, vocab_conn):
    _seed(vocab_conn, ["the zorbnax widget renders fine"])
    corrected, changed = fuzzy.correct_query(vocab_conn, "zorbnax widget")
    assert changed is False
