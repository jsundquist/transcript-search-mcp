"""Typo-tolerant fallback for keyword_search.

FTS5's tokenizer alone has no notion of "close enough" -- a misspelled query
term simply never matches. Rather than replacing FTS5 with a fuzzy engine,
this is a fallback invoked only when a strict FTS5 MATCH returns zero
results: each query token that has no exact match in the indexed vocabulary
(chunks_vocab, an fts5vocab view over chunks_fts) is replaced with its
closest vocabulary term by edit distance, if one exists within a small
threshold. Tokens with no close candidate are left as-is -- if nothing was
correctable, the caller's original empty result stands.

Correction operates on whitespace-split query tokens, but FTS5's
unicode61 tokenizer also splits on underscores/punctuation -- a typo'd
compound like "zorbnaxx_widget" won't line up with the vocabulary's split
"zorbnax"/"widget" entries and won't be corrected. Plain single-word typos
are the target case.
"""

from __future__ import annotations

import sqlite3

MAX_EDIT_DISTANCE = 2
# Bounds the worst-case per-token vocab-scan + edit-distance cost a
# many-token or pathologically long garbage query can trigger -- this runs
# by default on every zero-result strict search, so it needs a cap even
# though it's "only" a fallback path.
MAX_TOKENS = 20
MAX_TOKEN_LEN = 30


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def _closest_vocab_term(
    conn: sqlite3.Connection, token: str, max_distance: int
) -> str | None:
    """Best vocabulary replacement for `token`, or None if nothing is close
    enough. Ties broken by document frequency (more common term wins) --
    prefer the correction a real typo is more likely to have meant."""
    lower = token.lower()
    rows = conn.execute(
        "SELECT term, doc FROM chunks_vocab "
        "WHERE length(term) BETWEEN ? AND ?",
        (max(1, len(lower) - max_distance), len(lower) + max_distance),
    ).fetchall()

    best_term: str | None = None
    best_distance = max_distance + 1
    best_doc = -1
    for term, doc in rows:
        if term == lower:
            return None  # exact match already exists -- no correction needed
        distance = _edit_distance(lower, term)
        if distance > max_distance:
            continue
        if distance < best_distance or (distance == best_distance and doc > best_doc):
            best_term, best_distance, best_doc = term, distance, doc
    return best_term


def correct_query(
    conn: sqlite3.Connection, query: str, max_distance: int = MAX_EDIT_DISTANCE
) -> tuple[str, bool]:
    """Best-effort typo correction of an unquoted (non-phrase) query.

    Returns (corrected_query, changed). Only called when the original query
    already returned zero results, so a no-op correction is harmless.
    """
    tokens = query.split()
    corrected: list[str] = []
    changed = False
    for i, token in enumerate(tokens):
        if i >= MAX_TOKENS or len(token) > MAX_TOKEN_LEN:
            corrected.append(token)
            continue
        replacement = _closest_vocab_term(conn, token, max_distance)
        if replacement is not None:
            corrected.append(replacement)
            changed = True
        else:
            corrected.append(token)
    return " ".join(corrected), changed
