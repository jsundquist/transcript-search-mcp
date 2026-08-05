"""keyword_search / semantic_search / hybrid_search MCP tools."""

from __future__ import annotations

import sqlite3

import sqlite_vec

from transcript_search_v2 import db, embed, fuzzy, rerank as rerank_module
from transcript_search_v2.sqlutil import project_date_clause
from transcript_search_v2.timeutil import local_range_to_utc


def _sanitize_fts_query(query: str) -> str:
    """FTS5's MATCH syntax treats bare '-' as a NOT/column-filter operator and
    other punctuation specially, so an unquoted term like "sqlite-vec" parses
    as "sqlite NOT vec" instead of a literal token, silently returning wrong
    (usually empty) results. If the caller already used a `"` (explicit
    phrase search, per the documented '"exact phrase"' convention), pass
    through unchanged; otherwise quote each token individually so special
    characters are treated literally while preserving default AND-of-terms
    keyword search semantics (quoting a single token doesn't force phrase
    adjacency -- only a multi-word quoted string does)."""
    if '"' in query:
        return query
    tokens = query.split()
    return " ".join(f'"{t}"' for t in tokens)


def _signal_clause(include_low_signal: bool) -> str:
    return "" if include_low_signal else "AND c.signal != 'low'"


def _sidechain_clause(include_sidechains: bool) -> str:
    return "" if include_sidechains else "AND c.is_sidechain = 0"


def _filter_clauses(
    project: str | None, utc_from: str | None, utc_to: str | None, params: list
) -> str:
    clauses = project_date_clause(
        project, utc_from, utc_to, params, project_col="c.project_cwd", ts_col="c.timestamp"
    )
    return " ".join(f"AND {c}" for c in clauses)


def _row_to_result(row: sqlite3.Row, score: float | None, snippet: str | None = None) -> dict:
    return {
        "chunk_id": row["id"],
        "session_id": row["session_id"],
        "project": row["project_cwd"],
        "timestamp": row["timestamp"],
        "role": row["role"],
        "block_type": row["block_type"],
        "signal": row["signal"],
        "text": row["text"],
        "score": score,
        "snippet": snippet,
    }


def _run_keyword_query(
    conn: sqlite3.Connection,
    sanitized_query: str,
    project: str | None,
    utc_from: str | None,
    utc_to: str | None,
    roles: list[str] | None,
    include_low_signal: bool,
    include_sidechains: bool,
    limit: int,
    offset: int = 0,
) -> list[dict]:
    params: list = [sanitized_query]
    role_clause = ""
    if roles:
        placeholders = ",".join("?" for _ in roles)
        role_clause = f"AND c.role IN ({placeholders})"
    filters = _filter_clauses(project, utc_from, utc_to, params)
    if roles:
        params.extend(roles)
    sql = f"""
        SELECT c.id, c.session_id, c.project_cwd, c.timestamp, c.role, c.block_type,
               c.signal, c.text, bm25(chunks_fts) AS bm25_score,
               snippet(chunks_fts, 0, '<mark>', '</mark>', '…', 16) AS snippet
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        {_signal_clause(include_low_signal)}
        {_sidechain_clause(include_sidechains)}
        {role_clause}
        {filters}
        ORDER BY bm25_score
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        # a query containing a `"` bypasses _sanitize_fts_query's per-token
        # quoting (to leave native FTS5 phrase/boolean syntax reachable), so
        # a malformed expression here is a bad user query, not a server bug.
        raise ValueError(f"invalid search query syntax: {sanitized_query!r} ({e})") from e
    return [_row_to_result(r, r["bm25_score"], r["snippet"]) for r in rows]


def keyword_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    roles: list[str] | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
    fuzzy_fallback: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """FTS5 keyword/phrase search. Quote `query` (e.g. '"exact phrase"') for
    phrase search -- no separate phrase tool needed.

    fuzzy_fallback: if the strict search returns nothing and the query isn't
    a quoted phrase, retry once with typo'd tokens replaced by their closest
    indexed-vocabulary match (see fuzzy.py). Results from the fallback carry
    a `fuzzy_corrected_query` key; a strict-search hit never does.
    """
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        utc_from, utc_to = local_range_to_utc(date_from, date_to)
        results = _run_keyword_query(
            conn, _sanitize_fts_query(query), project, utc_from, utc_to,
            roles, include_low_signal, include_sidechains, limit, offset,
        )
        if results or not fuzzy_fallback or '"' in query:
            return results

        corrected, changed = fuzzy.correct_query(conn, query)
        if not changed:
            return results

        fallback_results = _run_keyword_query(
            conn, _sanitize_fts_query(corrected), project, utc_from, utc_to,
            roles, include_low_signal, include_sidechains, limit, offset,
        )
        for r in fallback_results:
            r["fuzzy_corrected_query"] = corrected
        return fallback_results
    finally:
        if own_conn:
            conn.close()


def semantic_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Embedding nearest-neighbor search. Over-fetches candidates from the
    vec0 KNN query, then applies project/date/signal/sidechain filters and
    truncates to `limit` -- vec0's KNN clause itself only understands the
    embedding+k constraint, so post-join filtering is how the rest applies."""
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        vector = embed.embed_query(query)
        vec_blob = sqlite_vec.serialize_float32(vector)
        # Cap sized for the server's _MAX_LIMIT (500) + _MAX_OFFSET (2000) --
        # a flat 500 cap meant fetch_k stopped growing with `offset` past
        # ~50, so a large-but-valid offset silently returned empty instead
        # of the results that were actually there further down.
        fetch_k = min(max((limit + offset) * 5, 50), 15000)

        utc_from, utc_to = local_range_to_utc(date_from, date_to)
        filter_params: list = []
        filters = _filter_clauses(project, utc_from, utc_to, filter_params)

        sql = f"""
            SELECT c.id, c.session_id, c.project_cwd, c.timestamp, c.role, c.block_type,
                   c.signal, c.text, v.distance AS vec_distance
            FROM chunks_vec v
            JOIN chunks c ON c.rowid = v.chunk_rowid
            WHERE v.embedding MATCH ? AND k = ?
            {_signal_clause(include_low_signal)}
            {_sidechain_clause(include_sidechains)}
            {filters}
            ORDER BY vec_distance
            LIMIT ? OFFSET ?
        """
        params = [vec_blob, fetch_k, *filter_params, limit, offset]
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_result(r, r["vec_distance"]) for r in rows]
    finally:
        if own_conn:
            conn.close()


def hybrid_search(
    query: str,
    project: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_low_signal: bool = False,
    include_sidechains: bool = False,
    limit: int = 20,
    offset: int = 0,
    rerank: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Reciprocal-rank fusion of keyword_search and semantic_search. Default
    tool to reach for when it's unclear whether a literal or a semantic match
    is more likely to find what you're after.

    rerank: after RRF fusion, re-score the fused candidates by RRF-score *
    signal-weight * recency-decay before truncating to `limit` -- RRF alone
    has no notion that a routine tool_result from months ago usually matters
    less than a high-signal chunk from yesterday at the same fused rank.
    Reranked results carry a "rerank_score" key alongside the RRF "score".
    """
    own_conn = conn is None
    conn = conn or db.connect(read_only=True)
    try:
        fetch_limit = max((limit + offset) * 3, 30)
        kw_results = keyword_search(
            query, project, date_from, date_to,
            include_low_signal=include_low_signal, include_sidechains=include_sidechains,
            limit=fetch_limit, conn=conn,
        )
        sem_results = semantic_search(
            query, project, date_from, date_to,
            include_low_signal=include_low_signal, include_sidechains=include_sidechains,
            limit=fetch_limit, conn=conn,
        )

        rrf_k = 60
        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for results in (kw_results, sem_results):
            for rnk, r in enumerate(results):
                cid = r["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rnk + 1)
                # semantic_search results never carry a snippet (no FTS match
                # to excerpt around) -- don't let a later semantic hit
                # clobber a snippet already found for the same chunk via
                # keyword_search.
                if cid in by_id and not r.get("snippet") and by_id[cid].get("snippet"):
                    r = {**r, "snippet": by_id[cid]["snippet"]}
                by_id[cid] = r

        # keep a wider pool than `limit`+`offset` pre-rerank, so signal/recency
        # can pull a lower-RRF-rank result above a higher one within the pool.
        page = limit + offset
        pool_size = max(page * 3, page) if rerank else page
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:pool_size]
        out = []
        for chunk_id, score in ranked:
            result = dict(by_id[chunk_id])
            result["score"] = score
            out.append(result)

        if rerank:
            out = rerank_module.rerank(out, limit=limit, offset=offset)
        else:
            out = out[offset:offset + limit]
        return out
    finally:
        if own_conn:
            conn.close()
