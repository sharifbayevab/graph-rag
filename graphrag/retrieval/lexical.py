"""Lexical retrieval: BM25 over the Postgres inverted index (spec §4.1).

score(c, q) = Σ_{t ∈ q} idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|c|/avgdl)),
idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5)); N, avgdl over all chunks, |c| = chunks.n_terms.
"""
from __future__ import annotations

from collections import Counter

from graphrag import config
from graphrag.data.lemma import tokens
from graphrag.data.normalize import query_internal
from graphrag.types import Chunk

_SQL = """
WITH q AS (SELECT DISTINCT unnest(%(terms)s::text[]) AS term),
st AS (SELECT count(*)::float8 AS n, greatest(avg(n_terms), 1)::float8 AS avgdl FROM chunks),
df AS (SELECT p.term, count(*)::float8 AS df FROM postings p JOIN q ON q.term = p.term GROUP BY p.term)
SELECT p.chunk_id,
       round(sum(ln(1 + (st.n - df.df + 0.5) / (df.df + 0.5))
           * p.tf * (%(k1)s + 1)
           / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * c.n_terms / st.avgdl)))::numeric, 9)::float8 AS score
FROM postings p
JOIN df ON df.term = p.term
JOIN chunks c ON c.chunk_id = p.chunk_id
CROSS JOIN st
GROUP BY p.chunk_id
ORDER BY score DESC, p.chunk_id COLLATE "C"
LIMIT %(k)s
"""


def postings_for(chunk: Chunk) -> Counter:
    """Term frequencies of what is indexed: index_text (breadcrumb + text_norm)."""
    return Counter(tokens(chunk.index_text or chunk.text_norm))


def query_terms(text: str, lang: str | None = None) -> list[str]:
    return sorted(set(tokens(query_internal(text, lang))))


def search_terms(conn, terms: list[str], k: int = 10) -> list[tuple[str, float]]:
    terms = sorted(set(terms))
    if not terms:
        return []
    bm = config.load("data")["bm25"]
    with conn.cursor() as cur:
        cur.execute(_SQL, {"terms": terms, "k": k, "k1": float(bm["k1"]), "b": float(bm["b"])})
        out = [(cid, float(s)) for cid, s in cur.fetchall()]
    conn.rollback()
    return out


def search(conn, text: str, k: int = 10, lang: str | None = None) -> list[tuple[str, float]]:
    return search_terms(conn, query_terms(text, lang), k)
