"""Postgres: document metadata, structure, μ (chunks.parent) and the inverted index."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from graphrag import config
from graphrag.types import DocRecord

_SCHEMA = Path(__file__).with_name("pg_schema.sql")


def connect(dsn: str | None = None):
    return psycopg2.connect(dsn or config.PG_DSN)


def init_schema(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(_SCHEMA.read_text(encoding="utf-8"))


def analyze(conn) -> None:
    """VACUUM ANALYZE the index tables (call after a bulk load): fresh planner statistics and
    a set visibility map, so BM25 reads the covering postings index without heap fetches.
    VACUUM cannot run in a transaction, so this runs in autocommit and restores the mode."""
    prev = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE postings")
            cur.execute("VACUUM ANALYZE chunks")
    finally:
        conn.autocommit = prev


def delete_document(conn, doc_id: str) -> None:
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))   # cascades


def upsert_document(conn, rec: DocRecord, postings: dict[str, Counter]) -> None:
    """Replace the document in one transaction (delete + insert): idempotent."""
    m = rec.meta
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (rec.doc_id,))
        cur.execute(
            "INSERT INTO documents (doc_id, section, path, type, date, organ, lang, status, title, text)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (rec.doc_id, m.section, rec.source.file, m.type, m.date, m.organ, m.lang, m.status,
             m.title, rec.text))
        execute_values(cur, "INSERT INTO nodes (node_id, doc_id, type, number, title, parent_id,"
                            " span_start, span_end) VALUES %s",
                       [(n.node_id, rec.doc_id, n.type, n.number, n.title, n.parent_id, n.start, n.end)
                        for n in rec.structure])
        execute_values(cur, "INSERT INTO chunks (chunk_id, doc_id, parent, par, tokens, src, text,"
                            " text_norm, index_text, span_start, span_end, n_terms) VALUES %s",
                       [(c.chunk_id, rec.doc_id, c.parent, c.par, c.tokens, c.src, c.text, c.text_norm,
                         c.index_text or c.text_norm, c.start, c.end, sum(postings[c.chunk_id].values()))
                        for c in rec.chunks])
        rows = [(t, cid, tf) for cid, cnt in postings.items() for t, tf in sorted(cnt.items())]
        if rows:
            execute_values(cur, "INSERT INTO postings (term, chunk_id, tf) VALUES %s", rows,
                           page_size=5000)


def chunk_rows(conn, ids: list[str]) -> dict[str, dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, doc_id, parent, text, text_norm, index_text FROM chunks"
                    " WHERE chunk_id = ANY(%s)", (list(ids),))
        rows = {r[0]: {"doc_id": r[1], "parent": r[2], "text": r[3], "text_norm": r[4],
                       "index_text": r[5]} for r in cur.fetchall()}
    conn.rollback()                      # end the read transaction (no idle-in-transaction)
    return rows


def counts(conn) -> dict[str, int]:
    with conn.cursor() as cur:
        out = {}
        for t in ("documents", "nodes", "chunks", "postings"):
            cur.execute(f"SELECT count(*) FROM {t}")
            out[t] = cur.fetchone()[0]
    conn.rollback()
    return out
