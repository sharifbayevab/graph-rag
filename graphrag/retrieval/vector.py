"""Vector retrieval: ANN over chunk embeddings in Qdrant, cosine (spec §4.2)."""
from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from graphrag import config
from graphrag.data.normalize import query_internal
from graphrag.types import DocRecord


_PAYLOAD_INDEXES = ("doc_id", "section")


def collection_name(slug: str) -> str:
    return f"grl_chunks__{slug}"


def client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def ensure_collection(cl: QdrantClient, name: str, dim: int) -> None:
    """Create the collection, or verify an existing one (single unnamed vector, dim, cosine);
    payload indexes are (re)created idempotently either way."""
    if cl.collection_exists(name):
        info = cl.get_collection(name)
        vp = info.config.params.vectors
        if not isinstance(vp, models.VectorParams):
            raise ValueError(f"collection {name} has named vectors; expected a single vector")
        if vp.size != dim:
            raise ValueError(f"collection {name} has dim {vp.size}, encoder has {dim}")
        if vp.distance != models.Distance.COSINE:
            raise ValueError(f"collection {name} uses {vp.distance} distance, expected Cosine")
        have = set(info.payload_schema or {})
    else:
        cl.create_collection(name, vectors_config=models.VectorParams(size=dim,
                                                                      distance=models.Distance.COSINE))
        have = set()
    for field in _PAYLOAD_INDEXES:
        if field not in have:
            cl.create_payload_index(name, field, field_schema=models.PayloadSchemaType.KEYWORD, wait=True)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"grl:{chunk_id}"))


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def upsert_chunks(cl: QdrantClient, name: str, rec: DocRecord, vectors: np.ndarray) -> None:
    if len(vectors) != len(rec.chunks):
        raise ValueError(f"{rec.doc_id}: {len(vectors)} vectors for {len(rec.chunks)} chunks")
    pts = [models.PointStruct(id=point_id(c.chunk_id), vector=v.tolist(),
                              payload={"chunk_id": c.chunk_id, "doc_id": rec.doc_id,
                                       "parent": c.parent, "section": rec.meta.section})
           for c, v in zip(rec.chunks, vectors)]
    for i in range(0, len(pts), 256):
        cl.upsert(name, points=pts[i:i + 256], wait=True)


def delete_doc(cl: QdrantClient, name: str, doc_id: str) -> None:
    cl.delete(name, points_selector=models.FilterSelector(filter=_doc_filter(doc_id)), wait=True)


def search(cl: QdrantClient, name: str, vec: np.ndarray, k: int = 10,
           exclude_doc: str | None = None) -> list[tuple[str, float]]:
    flt = None
    if exclude_doc:
        flt = models.Filter(must_not=[models.FieldCondition(key="doc_id",
                                                            match=models.MatchValue(value=exclude_doc))])
    res = cl.query_points(name, query=np.asarray(vec).tolist(), limit=k, query_filter=flt,
                          with_payload=["chunk_id"])
    return [(p.payload["chunk_id"], float(p.score)) for p in res.points]


def search_text(cl: QdrantClient, name: str, encoder, text: str, k: int = 10,
                exclude_doc: str | None = None, lang: str | None = None) -> list[tuple[str, float]]:
    """Text query → ANN search: the query is normalized like indexed text (`query_internal`)
    and embedded with the encoder's query prefix."""
    vec = encoder.encode_queries([query_internal(text, lang)])[0]
    return search(cl, name, vec, k=k, exclude_doc=exclude_doc)
