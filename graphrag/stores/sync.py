"""The ONLY write path for a document across all stores (spec §10.2).

upsert_document replaces the document in Postgres and Qdrant (delete + insert) and reconciles its
structure in Neo4j (GraphStore.reconcile_structure: MERGE with SET n += r, only vanished vertices and
stale CONTAINS removed), so incoming REFERS_TO / AMENDS / RELATED edges from other documents and the
enrichment properties written by the graph load survive a re-index (M1 requirement 2, Ruling R27). It
is idempotent, so a failed run is fixed by re-running it. Everything that can fail on the
input (tokenizing, strict encoding) runs before any store is touched. M7 calls these
functions from the Redis Streams consumer; nothing else may write to a store directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphrag import config
from graphrag.data.mapping import mu_inverse
from graphrag.kg.store import GraphStore
from graphrag.retrieval import lexical, vector
from graphrag.stores import pg
from graphrag.types import DocRecord


@dataclass
class Stores:
    pg: Any
    qdrant: Any
    graph: GraphStore
    encoder: Any
    collection: str

    @classmethod
    def open(cls, encoder, collection: str | None = None) -> "Stores":
        conn = pg.connect()
        try:
            graph = GraphStore()
        except Exception:
            conn.close()
            raise
        st = cls(conn, vector.client(), graph, encoder,
                 collection or vector.collection_name(encoder.slug))
        try:
            pg.init_schema(st.pg)
            vector.ensure_collection(st.qdrant, st.collection, encoder.dim)
            st.graph.init_schema()
        except Exception:
            st.close()
            raise
        return st

    def close(self) -> None:
        """Close every store even if one close fails; re-raise the first error."""
        errors = []
        for store in (self.pg, self.graph, self.qdrant):
            try:
                store.close()
            except Exception as e:                   # noqa: BLE001 — collected, re-raised below
                errors.append(e)
        if errors:
            raise errors[0]


def delete_document(st: Stores, doc_id: str) -> dict:
    """Remove the document everywhere; returns the graph report (enrichment edges removed with it)."""
    pg.delete_document(st.pg, doc_id)
    vector.delete_doc(st.qdrant, st.collection, doc_id)
    return st.graph.delete_document(doc_id)


def upsert_document(st: Stores, rec: DocRecord) -> dict:
    """Returns the graph reconcile report (vertices removed, dangling enrichment edges removed)."""
    field = config.load("data")["embed_field"]
    postings = {c.chunk_id: lexical.postings_for(c) for c in rec.chunks}
    # strict: an index build must never silently truncate a chunk
    vecs = st.encoder.encode_passages([getattr(c, field) or c.text_norm for c in rec.chunks], strict=True)
    inv = mu_inverse(rec.structure, rec.chunks)
    pg.delete_document(st.pg, rec.doc_id)
    vector.delete_doc(st.qdrant, st.collection, rec.doc_id)
    pg.upsert_document(st.pg, rec, postings)
    vector.upsert_chunks(st.qdrant, st.collection, rec, vecs)
    return st.graph.reconcile_structure(rec, {k: len(v) for k, v in inv.items()})
