"""Stores and the single write path (spec §4.1, §4.2, §10.2).

Unit tests run offline; `integration` tests need `docker compose up -d` and clean up
everything they write (pg rows / temp schema, neo4j nodes, the qdrant test collection)."""
import math
import uuid
from collections import Counter

import pytest

from graphrag import config
from graphrag.data.ingest import build_corpus
from graphrag.retrieval import lexical, vector
from graphrag.stores import pg
from graphrag.types import Chunk, DocRecord, Meta, Node, Source
from tests.helpers import FakeEncoder, words
from tests.test_ingest import _corpus

integration = pytest.mark.integration
_COLL = "grl_test__fake"


@pytest.fixture()
def rec(tmp_path):
    return build_corpus(_corpus(tmp_path), words)[0][0]


# ── unit ─────────────────────────────────────────────────────────────────────
def test_embed_field_is_index_text():
    assert config.load("data")["embed_field"] == "index_text"


def test_postings_tokenize_index_text_with_text_norm_fallback(rec):
    c = rec.chunks[0]
    assert c.index_text and c.index_text != c.text_norm
    assert lexical.postings_for(c) == Counter(lexical.tokens(c.index_text))
    bare = Chunk(**{**c.__dict__, "index_text": ""})
    assert lexical.postings_for(bare) == Counter(lexical.tokens(c.text_norm))


def test_query_terms_normalize_like_the_index():
    assert lexical.query_terms("Қонун 12-модда", "uz-Cyrl") == lexical.query_terms("qonun 12-modda", "uz-Latn")
    assert lexical.query_terms("") == []


def test_point_id_is_deterministic_uuid():
    assert vector.point_id("D#1") == vector.point_id("D#1") != vector.point_id("D#2")
    uuid.UUID(vector.point_id("D#1"))


def test_sync_upsert_encodes_index_text_strictly(rec, monkeypatch):
    from graphrag.stores import sync

    seen = {}

    class Enc(FakeEncoder):
        def encode_passages(self, texts, **kw):
            seen["texts"], seen["kw"] = list(texts), kw
            return super().encode_passages(texts)

    calls = []
    monkeypatch.setattr(sync.pg, "delete_document", lambda conn, d: calls.append(("pg-", d)))
    monkeypatch.setattr(sync.pg, "upsert_document", lambda conn, r, p: calls.append(("pg+", sorted(p))))
    monkeypatch.setattr(sync.vector, "delete_doc", lambda cl, n, d: calls.append(("qd-", d)))
    monkeypatch.setattr(sync.vector, "upsert_chunks", lambda cl, n, r, v: calls.append(("qd+", v.shape)))

    class G:                                        # Task 10 (R27): the graph is reconciled, never deleted
        def delete_document(self, d): raise AssertionError("graph delete on upsert")
        def reconcile_structure(self, r, n): calls.append(("g~", n))

    st = sync.Stores(pg=None, qdrant=None, graph=G(), encoder=Enc(), collection="x")
    sync.upsert_document(st, rec)
    assert seen["texts"] == [c.index_text for c in rec.chunks]
    assert seen["kw"].get("strict") is True
    assert [k for k, _ in calls] == ["pg-", "qd-", "pg+", "qd+", "g~"]
    assert calls[2][1] == sorted(c.chunk_id for c in rec.chunks)
    assert calls[3][1] == (len(rec.chunks), FakeEncoder.dim)
    n_chunks = calls[4][1]
    assert n_chunks[rec.doc_id] == len(rec.chunks)                  # root contains every chunk


def test_sync_encoding_failure_leaves_stores_untouched(rec):
    from graphrag.stores import sync

    class Boom(FakeEncoder):
        def encode_passages(self, texts, **kw):
            raise ValueError("would truncate")

    class G:
        def __getattr__(self, name):
            raise AssertionError("store touched")

    st = sync.Stores(pg=None, qdrant=None, graph=G(), encoder=Boom(), collection="x")
    with pytest.raises(ValueError):
        sync.upsert_document(st, rec)


# ── integration: the three stores via the single write path ──────────────────
@pytest.fixture()
def stores(rec):
    from graphrag.stores.sync import Stores, delete_document
    try:
        st = Stores.open(FakeEncoder(), collection=_COLL)
    except Exception as e:                           # services not running
        pytest.skip(f"stores unavailable: {e}")
    # a private pg schema: rankings and counts must not depend on the loaded corpus (the
    # shared public schema holds the real index after build_indexes.py), and the test never
    # writes to it; neo4j/qdrant are isolated by the test doc_id / test collection
    schema = f"grl_test_{uuid.uuid4().hex[:8]}"
    with st.pg, st.pg.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
    pg.init_schema(st.pg)
    yield st
    delete_document(st, rec.doc_id)                  # never leave test rows in neo4j
    st.qdrant.delete_collection(_COLL)
    st.pg.rollback()
    with st.pg, st.pg.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    st.close()


def _graph_edges(st, doc_id):
    with st.graph.driver.session() as s:
        return sorted((r["a"], r["b"]) for r in s.run(
            "MATCH (a:Node {doc_id: $d})-[:CONTAINS]->(b:Node) RETURN a.node_id AS a, b.node_id AS b",
            d=doc_id))


def _doc_counts(st, rec):
    with st.pg.cursor() as cur:
        out = {}
        for t in ("documents", "nodes", "chunks"):
            cur.execute(f"SELECT count(*) FROM {t} WHERE doc_id = %s", (rec.doc_id,))
            out[t] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM postings p JOIN chunks c USING (chunk_id) WHERE c.doc_id = %s",
                    (rec.doc_id,))
        out["postings"] = cur.fetchone()[0]
    out["qdrant"] = st.qdrant.count(_COLL, count_filter=vector._doc_filter(rec.doc_id), exact=True).count
    with st.graph.driver.session() as s:
        out["neo4j"] = s.run("MATCH (n:Node {doc_id: $d}) RETURN count(n) AS c", d=rec.doc_id).single()["c"]
    out["edges"] = len(_graph_edges(st, rec.doc_id))
    return out


@integration
def test_upsert_is_idempotent_and_found_three_ways(stores, rec):
    from graphrag.stores.sync import upsert_document
    upsert_document(stores, rec)
    first = _doc_counts(stores, rec)
    total = pg.counts(stores.pg)
    assert total["documents"] == 1                  # private schema: isolated from the loaded corpus
    upsert_document(stores, rec)                    # replaces, does not duplicate
    assert _doc_counts(stores, rec) == first
    assert pg.counts(stores.pg) == total
    postings = {c.chunk_id: lexical.postings_for(c) for c in rec.chunks}
    assert first == {"documents": 1, "nodes": len(rec.structure), "chunks": len(rec.chunks),
                     "postings": sum(len(p) for p in postings.values()),
                     "qdrant": len(rec.chunks), "neo4j": len(rec.structure),
                     "edges": sum(1 for n in rec.structure if n.parent_id)}
    c = rec.chunks[0]
    rows = pg.chunk_rows(stores.pg, [c.chunk_id])
    assert rows[c.chunk_id]["parent"] == c.parent
    assert rows[c.chunk_id]["index_text"] == c.index_text
    # lexical
    hits = lexical.search_terms(stores.pg, lexical.query_terms(c.text_norm, "uz-Latn"), k=5)
    assert c.chunk_id in [h[0] for h in hits]
    # vector (what is indexed is index_text)
    q = stores.encoder.encode_passages([c.index_text])[0]
    assert vector.search(stores.qdrant, stores.collection, q, k=1)[0][0] == c.chunk_id
    # graph
    assert stores.graph.node(c.parent)["doc_id"] == rec.doc_id
    assert stores.graph.path_to_document(c.parent)[0] == rec.doc_id
    assert stores.graph.node(rec.doc_id)["n_chunks"] == len(rec.chunks)


@integration
def test_contains_edges_follow_parent_id_exactly(stores, rec):
    from graphrag.stores.sync import upsert_document
    upsert_document(stores, rec)
    assert _graph_edges(stores, rec.doc_id) == sorted(
        (n.parent_id, n.node_id) for n in rec.structure if n.parent_id)
    by_id = {n.node_id: n for n in rec.structure}
    leaf = max(rec.structure, key=lambda n: len(stores.graph.path_to_document(n.node_id)))
    chain, x = [], leaf
    while x:
        chain.append(x.node_id)
        x = by_id.get(x.parent_id)
    assert stores.graph.path_to_document(leaf.node_id) == chain[::-1]


@integration
def test_delete_removes_everywhere(stores, rec):
    from graphrag.stores.sync import delete_document, upsert_document
    upsert_document(stores, rec)
    delete_document(stores, rec.doc_id)
    assert _doc_counts(stores, rec) == dict.fromkeys(
        ("documents", "nodes", "chunks", "postings", "qdrant", "neo4j", "edges"), 0)
    assert pg.chunk_rows(stores.pg, [rec.chunks[0].chunk_id]) == {}
    assert stores.graph.node(rec.doc_id) is None
    q = stores.encoder.encode_passages([rec.chunks[0].index_text])[0]
    assert all(h[0] != rec.chunks[0].chunk_id for h in vector.search(stores.qdrant, stores.collection, q, k=5))


@integration
def test_vector_search_can_exclude_own_document(stores, rec):
    from graphrag.stores.sync import upsert_document
    upsert_document(stores, rec)
    q = stores.encoder.encode_passages([rec.chunks[0].index_text])[0]
    assert vector.search(stores.qdrant, stores.collection, q, k=5)
    hits = vector.search(stores.qdrant, stores.collection, q, k=5, exclude_doc=rec.doc_id)
    assert all(not h[0].startswith(rec.doc_id) for h in hits)


# ── integration: BM25 in SQL == textbook BM25 on a hand-built example ────────
@pytest.fixture()
def pg_scratch():
    """A private Postgres schema: exact corpus statistics, dropped afterwards."""
    try:
        conn = pg.connect()
    except Exception as e:
        pytest.skip(f"postgres unavailable: {e}")
    schema = f"grl_test_{uuid.uuid4().hex[:8]}"
    with conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
    pg.init_schema(conn)
    yield conn
    conn.rollback()
    with conn, conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.close()


def _tiny_record(postings: dict[str, Counter]) -> DocRecord:
    root = Node("T", "документ", "t", None, None, 0, 30)
    chunks = [Chunk(cid, "T", i + 1, 1, "", "x", "x", i, i + 1, "x") for i, cid in enumerate(postings)]
    meta = Meta("qonun", "2020-01-02", "o", ["uz-Latn"], "active", "QONUNLAR", "t")
    return DocRecord("T", meta, Source("docs", "docs/t.md", None), "x" * 30, [root], chunks)


def _bm25(postings, terms, k1=1.2, b=0.75):
    n = len(postings)
    dl = {c: sum(p.values()) for c, p in postings.items()}
    avgdl = sum(dl.values()) / n
    out = {}
    for c, p in postings.items():
        s, hit = 0.0, False
        for t in terms:
            df = sum(1 for q in postings.values() if t in q)
            if t in p:
                hit = True
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * p[t] * (k1 + 1) / (p[t] + k1 * (1 - b + b * dl[c] / avgdl))
        if hit:
            out[c] = s
    return out


@integration
def test_sql_bm25_equals_textbook_formula(pg_scratch):
    # lengths 5, 3, 8 (avgdl 16/3); df(apple)=2, df(pear)=1, df(fig)=3, df(kiwi)=0
    postings = {"T#a": Counter(apple=2, fig=1, x=2), "T#b": Counter(pear=1, fig=2),
                "T#c": Counter(apple=1, fig=3, y=4)}
    pg.upsert_document(pg_scratch, _tiny_record(postings), postings)
    assert pg.counts(pg_scratch) == {"documents": 1, "nodes": 1, "chunks": 3, "postings": 8}
    for terms in (["apple"], ["pear"], ["fig"], ["apple", "pear"], ["apple", "fig", "kiwi"],
                  ["apple", "apple"], ["kiwi"]):
        got = dict(lexical.search_terms(pg_scratch, terms, k=10))
        want = _bm25(postings, set(terms))
        assert got.keys() == want.keys(), terms
        for c in want:
            assert abs(got[c] - want[c]) < 1e-9, (terms, c, got[c], want[c])
    # the hand numbers for one case: apple in T#a (tf 2, dl 5), N 3, df 2
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    tf_part = 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 5 / (16 / 3)))
    assert abs(dict(lexical.search_terms(pg_scratch, ["apple"]))["T#a"] - idf * tf_part) < 1e-9
    ranked = lexical.search_terms(pg_scratch, ["apple", "fig"], k=2)
    assert len(ranked) == 2 and ranked[0][1] >= ranked[1][1]


# ── fix round 1 ──────────────────────────────────────────────────────────────
import io  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def test_stores_close_closes_all_and_reraises_first_error():
    from graphrag.stores import sync

    closed = []

    class C:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(self.name)

    st = sync.Stores(pg=C("pg", True), qdrant=C("qd", True), graph=C("g"), encoder=None, collection="x")
    with pytest.raises(RuntimeError, match="^pg$"):
        st.close()
    assert sorted(closed) == ["g", "pg", "qd"]


@integration
def test_pg_stores_effective_index_text(pg_scratch):
    postings = {"T#a": Counter(apple=1)}
    rec = _tiny_record(postings)
    rec.chunks[0].index_text, rec.chunks[0].text_norm = "", "apple norm"
    pg.upsert_document(pg_scratch, rec, postings)
    assert pg.chunk_rows(pg_scratch, ["T#a"])["T#a"]["index_text"] == "apple norm"


@pytest.fixture()
def qdrant_scratch():
    try:
        cl = vector.client()
        cl.get_collections()
    except Exception as e:
        pytest.skip(f"qdrant unavailable: {e}")
    name = f"grl_test__{uuid.uuid4().hex[:8]}"
    yield cl, name
    if cl.collection_exists(name):
        cl.delete_collection(name)
    cl.close()


@integration
def test_ensure_collection_verifies_distance_and_payload_indexes(qdrant_scratch):
    from qdrant_client import models
    cl, name = qdrant_scratch
    cl.create_collection(name, vectors_config=models.VectorParams(size=8, distance=models.Distance.DOT))
    with pytest.raises(ValueError, match="distance"):
        vector.ensure_collection(cl, name, 8)
    cl.delete_collection(name)
    cl.create_collection(name, vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE))
    vector.ensure_collection(cl, name, 8)                  # existing, no payload indexes yet → added
    vector.ensure_collection(cl, name, 8)                  # idempotent
    assert set(cl.get_collection(name).payload_schema) == {"doc_id", "section"}
    with pytest.raises(ValueError, match="dim"):
        vector.ensure_collection(cl, name, 16)


_PERF_Q = {
    "rare": [f"t{x}" for x in range(20000, 20010)],
    "mixed": ["t1", "t3", "t10", "t40", "t150", "t600", "t2000", "t8000", "t20000", "t50000"],
    "common": [f"t{x}" for x in range(1, 11)],
    "20-term": [f"t{x}" for x in (1, 2, 5, 9, 17, 30, 55, 90, 150, 260, 400, 700, 1000, 1800, 3000,
                                  5000, 9000, 15000, 30000, 55000)],
}


def _load_synthetic(conn, n_docs=120, per_doc=140, vocab=60000, seed=0) -> int:
    """16.8k chunks, ~1.4M postings; Zipf(1.15) term draws, 60-300 tokens per chunk."""
    rng = np.random.default_rng(seed)
    bufs = {t: io.StringIO() for t in ("documents", "nodes", "chunks", "postings")}
    n_post = 0
    for d in range(n_docs):
        did = f"P{d}"
        bufs["documents"].write(f"{did}\tS\tp\t\\N\t\\N\t\\N\t\\N\t\\N\t\\N\tx\n")
        bufs["nodes"].write(f"{did}\t{did}\tдокумент\t\\N\t\\N\t\\N\t0\t1\n")
        for c in range(per_doc):
            cid = f"{did}#{c}"
            u, tf = np.unique(np.minimum(rng.zipf(1.15, int(rng.integers(60, 300))), vocab),
                              return_counts=True)
            bufs["chunks"].write(f"{cid}\t{did}\t{did}\t{c}\t1\t\\N\tx\tx\tx\t0\t1\t{int(tf.sum())}\n")
            bufs["postings"].write("".join(f"t{t}\t{cid}\t{f}\n" for t, f in zip(u, tf)))
            n_post += len(u)
    with conn, conn.cursor() as cur:
        for t, b in bufs.items():
            b.seek(0)
            cur.copy_from(b, t)
    return n_post


def _median_ms(conn, terms, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lexical.search_terms(conn, terms, k=10)
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


@integration
def test_bm25_latency_at_scale_with_covering_index(pg_scratch):
    conn = pg_scratch
    with conn, conn.cursor() as cur:                        # the pre-fix layout: non-covering PK
        cur.execute("ALTER TABLE postings DROP CONSTRAINT postings_pkey,"
                    " ADD CONSTRAINT postings_pkey PRIMARY KEY (term, chunk_id)")
    n_post = _load_synthetic(conn)
    assert pg.counts(conn)["chunks"] >= 10_000 and n_post >= 1_000_000
    pg.analyze(conn)
    before = {k: lexical.search_terms(conn, q, k=10) for k, q in _PERF_Q.items()}
    before_ms = {k: _median_ms(conn, q, reps=3) for k, q in _PERF_Q.items()}

    pg.init_schema(conn)                                    # migrates the PK to the covering layout
    pg.analyze(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT indnatts - indnkeyatts FROM pg_index WHERE indexrelid = to_regclass('postings_pkey')")
        assert cur.fetchone()[0] == 1                       # INCLUDE (tf)
        cur.execute("EXPLAIN " + lexical._SQL, {"terms": _PERF_Q["mixed"], "k": 10, "k1": 1.2, "b": 0.75})
        plan = "\n".join(r[0] for r in cur.fetchall())
    conn.rollback()
    assert "Index Only Scan using postings_pkey" in plan

    for k, q in _PERF_Q.items():                            # identical ranking and scores
        after = lexical.search_terms(conn, q, k=10)
        assert [c for c, _ in after] == [c for c, _ in before[k]], k
        assert all(abs(a - b) < 1e-9 for (_, a), (_, b) in zip(after, before[k])), k
    after_ms = {k: _median_ms(conn, q) for k, q in _PERF_Q.items()}
    print(f"\nBM25 {pg.counts(conn)['chunks']} chunks / {n_post} postings, median ms "
          + ", ".join(f"{k} {before_ms[k]:.0f}→{after_ms[k]:.0f}" for k in _PERF_Q))
    assert after_ms["mixed"] < 150, after_ms


# ── Task 10 (Ruling R27): non-destructive reconcile + versioned graph load ──────────────────────
import json  # noqa: E402

from graphrag.kg.schema import Evidence, GraphEdge, GraphNode  # noqa: E402


def test_dev_neo4j_guard_has_no_override(monkeypatch):
    from graphrag.stores.guard import dev_neo4j_problem
    monkeypatch.setenv("GRL_ALLOW_RESET", "1")
    assert dev_neo4j_problem("bolt://localhost:7688") is None
    assert dev_neo4j_problem("neo4j://127.0.0.1:7688") is None
    for bad in ("bolt://localhost:7687", "bolt://203.0.113.10:7688", "http://localhost:7688", "bolt://localhost"):
        assert dev_neo4j_problem(bad), bad


def test_edge_props_round_trip_and_identity():
    from graphrag.kg import store
    ev = Evidence("D", 1, 5, "q")
    e = GraphEdge("D/СТ.1", "X/СТ.2", "AMENDS",
                  {"evidence": ev, "evidence_all": [ev, ev], "op": "add", "ops": ["add", "edit"], "date": "2024-01-02"})
    p = store.edge_props(e)
    assert p["op"] == "add" and p["date"] == "2024-01-02" and isinstance(p["evidence"], str)
    assert p["json_attrs"] == ["evidence", "evidence_all", "ops"]
    assert store.edge_attrs_from_props({**p, "edge_key": "k", "graph": "g", "graph_version": "v"}) == \
        {"evidence": ev.to_json(), "evidence_all": [ev.to_json()] * 2, "op": "add", "ops": ["add", "edit"],
         "date": "2024-01-02"}
    r1 = GraphEdge("a", "b", "REFERS_TO", {"relation": "amended"})
    r2 = GraphEdge("a", "b", "REFERS_TO", {"relation": "cited_in_amendment"})
    assert store.edge_key_json(r1) != store.edge_key_json(r2)
    assert store.edge_key_json(GraphEdge("a", "b", "RELATED", {"weight": 0.7})) == '["a", "RELATED", "b"]'


def test_node_props_keep_structure_owned_fields_on_corpus_nodes():
    from graphrag.kg import store
    corpus = GraphNode("D/СТ.1", "element", "статья", "T", "D", False, {"number": "1"})
    p = store.node_props(corpus)
    assert "title" not in p and "type" not in p and "number" not in p and p["kg_title"] == "T"
    stub = GraphNode("EXT/x/СТ.2", "element", "статья", "2-модда", None, True, {"act": "EXT/x", "number": "2"})
    q = store.node_props(stub)
    assert q["number"] == "2" and q["act"] == "EXT/x" and "doc_id" not in q
    assert store.node_from_props(q, ["Node", "Element"]) == stub.to_json()


def _law(doc_id: str, articles: list[int]) -> DocRecord:
    """A law with one chunk per article (СТ.k); spans laid out back to back."""
    root_t = f"{doc_id} qonuni"
    text, structure, chunks = root_t + "\n", [], []
    for k in articles:
        body = f"{k}-modda. {doc_id} test matni number {k} " * 3
        s, e = len(text), len(text) + len(body)
        nid = f"{doc_id}/СТ.{k}"
        structure.append(Node(nid, "статья", f"{k}-modda", str(k), doc_id, s, e))
        chunks.append(Chunk(f"{nid}#1", nid, 1, len(body.split()), "абз. 1", body, body.lower(), s, e,
                            f"{root_t}\n{body.lower()}"))
        text += body
    structure.insert(0, Node(doc_id, "документ", root_t, None, None, 0, len(text)))
    meta = Meta("qonun", "2024-01-02", "o", ["uz-Latn"], "active", "QONUNLAR", root_t)
    return DocRecord(doc_id, meta, Source("docs", f"docs/{doc_id}.md", None), text, structure, chunks)


def _artifact(recs, extra_nodes=(), extra_edges=()):
    from graphrag.kg.extract_ref import structure_nodes
    nodes = [n for r in recs for n in structure_nodes(r)] + list(extra_nodes)
    edges = [GraphEdge(n.parent_id, n.node_id, "CONTAINS", {}) for r in recs for n in r.structure if n.parent_id]
    return nodes, edges + list(extra_edges)


def _ev(doc_id):
    return Evidence(doc_id, 0, 4, "test")


@pytest.fixture()
def kg():
    """Stores on a private pg schema + test qdrant collection; neo4j isolated by GRLTEST doc_ids and a
    test graph name; everything the test writes is removed afterwards."""
    from graphrag.stores.guard import dev_neo4j_problem
    from graphrag.stores.sync import Stores, delete_document
    if dev_neo4j_problem(config.NEO4J_URI):
        pytest.skip("neo4j is not the dev service")
    coll = f"grl_test__{uuid.uuid4().hex[:8]}"
    try:
        st = Stores.open(FakeEncoder(), collection=coll)
    except Exception as e:
        pytest.skip(f"stores unavailable: {e}")
    schema = f"grl_test_{uuid.uuid4().hex[:8]}"
    with st.pg, st.pg.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
    pg.init_schema(st.pg)
    tag = uuid.uuid4().hex[:8].upper()
    ids = {"A": f"GRLTEST{tag}A", "B": f"GRLTEST{tag}B", "graph": f"test_{tag.lower()}", "tag": tag}
    yield st, ids
    st.graph.drop_versions(ids["graph"], "all")
    with st.graph.driver.session() as s:
        s.run("MATCH (m:GraphMeta {graph: $g}) DELETE m", g=ids["graph"]).consume()
        s.run("MATCH (e:GraphEvent) WHERE e.doc_id IN $d DELETE e", d=[ids["A"], ids["B"]]).consume()
    for d in (ids["A"], ids["B"]):
        delete_document(st, d)
    with st.graph.driver.session() as s:
        s.run("MATCH (e:GraphEvent) WHERE e.doc_id IN $d DELETE e", d=[ids["A"], ids["B"]]).consume()
        left = s.run("MATCH (n:Node) WHERE n.node_id STARTS WITH $p OR n.node_id CONTAINS $t RETURN count(n) AS c",
                     p="GRLTEST" + ids["tag"], t=ids["tag"]).single()["c"]
    st.qdrant.delete_collection(coll)
    st.pg.rollback()
    with st.pg, st.pg.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    st.close()
    assert left == 0, "test vertices left in neo4j"


def _incoming(st, node_id, graph):
    with st.graph.driver.session() as s:
        return {r["t"]: r["c"] for r in s.run(
            "MATCH ()-[r]-(n:Node {node_id: $id}) WHERE r.graph = $g RETURN type(r) AS t, count(r) AS c",
            id=node_id, g=graph)}


def _graph_edges_of(st, graph):
    with st.graph.driver.session() as s:
        return sorted((r["s"], r["t"], r["d"], r["v"]) for r in s.run(
            "MATCH (a)-[r]->(b) WHERE r.graph = $g RETURN a.node_id AS s, type(r) AS t, b.node_id AS d, "
            "r.graph_version AS v", g=graph))


def _enrichment(ids):
    """B/СТ.1 -> A/СТ.1 REFERS_TO + AMENDS, RELATED A/СТ.2 ~ B/СТ.1, an entity and an EXT stub act."""
    a, b, t = ids["A"], ids["B"], ids["tag"]
    ent = GraphNode(f"ENT/organ/grltest {t}", "entity", "organ", "test organ", None, False,
                    {"aliases": ["test organ"], "docs": 1, "mentions": 2, "norm_key": f"grltest {t}", "sources": ["rule"]})
    ext = GraphNode(f"EXT/grltest {t}", "document", "документ", "Test act", None, True,
                    {"act_key": f"grltest {t}", "amending": False, "mentions": 1})
    ext_el = GraphNode(f"EXT/grltest {t}/СТ.5", "element", "статья", "5-modda", None, True,
                       {"act": ext.node_id, "number": "5"})
    edges = [GraphEdge(f"{b}/СТ.1", f"{a}/СТ.1", "REFERS_TO", {"evidence": _ev(b), "relation": "element"}),
             GraphEdge(f"{b}/СТ.1", f"{a}/СТ.1", "AMENDS", {"evidence": _ev(b), "op": "edit", "date": "2024-02-03"}),
             GraphEdge(f"{a}/СТ.2", f"{b}/СТ.1", "RELATED", {"weight": 0.71}),
             GraphEdge(f"{a}/СТ.1", ent.node_id, "REGULATES", {"evidence": _ev(a)}),
             GraphEdge(f"{a}/СТ.2", ext_el.node_id, "REFERS_TO", {"evidence": _ev(a), "relation": "act"}),
             GraphEdge(ext.node_id, ext_el.node_id, "CONTAINS", {})]
    return [ent, ext, ext_el], edges


@integration
def test_reupsert_keeps_incoming_edges_and_enrichment_props(kg):
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    nodes, edges = _artifact([A, B], xn, xe)
    rep = st.graph.load_graph(nodes, edges, "v1", graph=ids["graph"], strict_corpus=False, sample=100)
    assert rep["ok"] and rep["parity"]["ok"] and rep["active_version"] == "v1"
    a1 = f"{ids['A']}/СТ.1"
    before = _incoming(st, a1, ids["graph"])
    assert before == {"REFERS_TO": 1, "AMENDS": 1, "REGULATES": 1}
    assert _incoming(st, f"{ids['A']}/СТ.2", ids["graph"]) == {"RELATED": 1, "REFERS_TO": 1}
    props = st.graph.node(a1)
    assert props["kind"] == "element" and json.loads(props["attrs_json"]) == {"number": "1"}
    out = upsert_document(st, A)                              # M1 re-index of A: nothing of B's is lost
    assert out["nodes_removed"] == 0 and out["dangling_removed"] == {} and out["contains_removed"] == 0
    assert _incoming(st, a1, ids["graph"]) == before
    assert _incoming(st, f"{ids['A']}/СТ.2", ids["graph"]) == {"RELATED": 1, "REFERS_TO": 1}
    after = st.graph.node(a1)                                 # SET n += r keeps the enrichment props
    assert after["attrs_json"] == props["attrs_json"] and after["kind"] == "element"
    assert after["n_chunks"] == 1 and after["kg_title"] == "1-modda"
    assert st.graph.parity(nodes, edges, "v1", graph=ids["graph"], sample=100)["ok"]
    assert _graph_edges(st, ids["A"]) == sorted((n.parent_id, n.node_id) for n in A.structure if n.parent_id)


@integration
def test_removed_article_drops_its_edges_and_is_reported(kg):
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    st.graph.load_graph(*_artifact([A, B], xn, xe), "v1", graph=ids["graph"], strict_corpus=False)
    a1 = f"{ids['A']}/СТ.1"
    A2 = _law(ids["A"], [2])                                  # article 1 removed from A
    out = upsert_document(st, A2)
    assert out["nodes_removed"] == 1
    assert out["dangling_removed"] == {ids["graph"]: {"AMENDS": 1, "REFERS_TO": 1, "REGULATES": 1}}
    assert st.graph.node(a1) is None
    with st.graph.driver.session() as s:                      # nothing hangs: no edge names the removed node
        assert s.run("MATCH (n:Node {doc_id: $d}) RETURN count(n) AS c", d=ids["A"]).single()["c"] == 2
    assert _incoming(st, f"{ids['A']}/СТ.2", ids["graph"]) == {"RELATED": 1, "REFERS_TO": 1}
    assert _graph_edges(st, ids["A"]) == [(ids["A"], f"{ids['A']}/СТ.2")]
    # the next load (artifact rebuilt without A/СТ.1) reports the removal
    xe2 = [e for e in xe if a1 not in (e.src, e.dst)]
    rep = st.graph.load_graph(*_artifact([A2, B], xn, xe2), "v2", graph=ids["graph"], strict_corpus=False)
    assert rep["ok"] and rep["parity"]["ok"]
    assert [(d["doc_id"], d["node_ids"], d["removed"]) for d in rep["dangling_edge_removals"]] == \
        [(ids["A"], [a1], {"AMENDS": 1, "REFERS_TO": 1, "REGULATES": 1})]
    # the swap drops the 3 v1 edges that survived (the other 3 went with A/СТ.1); every v1 vertex is in v2
    assert rep["swap"] == {"edges": {"CONTAINS": 1, "REFERS_TO": 1, "RELATED": 1}, "edges_total": 3, "nodes": 0}
    rep3 = st.graph.load_graph(*_artifact([A2, B], xn, xe2), "v3", graph=ids["graph"], strict_corpus=False)
    assert rep3["dangling_edge_removals"] == []               # consumed by the v2 report


@integration
def test_new_version_replaces_old_edges_and_versioned_nodes(kg):
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    st.graph.load_graph(*_artifact([A, B], xn, xe), "v1", graph=ids["graph"], strict_corpus=False)
    v1 = _graph_edges_of(st, ids["graph"])
    assert {v for *_, v in v1} == {"v1"} and len(v1) == len(xe)
    ent2 = GraphNode(f"ENT/organ/grltest2 {ids['tag']}", "entity", "organ", "other organ", None, False, {})
    xn2 = [xn[0].__class__(**{**xn[0].__dict__, "attrs": {**xn[0].attrs, "mentions": 9}}), ent2]  # EXT act dropped
    xe2 = [xe[0], GraphEdge(f"{ids['A']}/СТ.1", ent2.node_id, "REGULATES", {"evidence": _ev(ids["A"])})]
    rep = st.graph.load_graph(*_artifact([A, B], xn2, xe2), "v2", graph=ids["graph"], strict_corpus=False)
    assert rep["ok"] and rep["previous_version"] == "v1" and st.graph.active_version(ids["graph"]) == "v2"
    v2 = _graph_edges_of(st, ids["graph"])
    assert {v for *_, v in v2} == {"v2"} and len(v2) == 2      # old version's edges removed after the swap
    assert rep["swap"]["edges_total"] == len(xe) and rep["swap"]["nodes"] == 2   # EXT act + its stub article
    assert st.graph.node(xn[1].node_id) is None and st.graph.node(xn[2].node_id) is None
    assert st.graph.node(ent2.node_id)["versions"] == [f"{ids['graph']}@v2"]
    assert json.loads(st.graph.node(xn[0].node_id)["attrs_json"])["mentions"] == 9
    assert st.graph.node(f"{ids['A']}/СТ.1")["doc_id"] == ids["A"]   # corpus structure never versioned
    assert len(_graph_edges(st, ids["A"])) == 2


@integration
def test_failed_load_leaves_old_version_intact(kg, monkeypatch):
    from graphrag.kg.store import GraphLoadError
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    art1 = _artifact([A, B], xn, xe)
    st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)
    v1 = _graph_edges_of(st, ids["graph"])
    ent2 = GraphNode(f"ENT/organ/grltest2 {ids['tag']}", "entity", "organ", "other organ", None, False, {})
    art2 = _artifact([A, B], xn + [ent2],
                     xe + [GraphEdge(f"{ids['B']}/СТ.1", ent2.node_id, "REGULATES", {"evidence": _ev(ids["B"])})])
    real, calls = st.graph._write_edge_batch, []

    def boom(etype, rows, graph, version):                    # dies after the first edge batch was written
        if calls:
            raise RuntimeError("simulated crash mid-load")
        calls.append(etype)
        real(etype, rows, graph, version)

    monkeypatch.setattr(st.graph, "_write_edge_batch", boom)
    with pytest.raises(GraphLoadError) as ei:
        st.graph.load_graph(*art2, "v2", graph=ids["graph"], strict_corpus=False, batch=2)
    rep = ei.value.report
    assert not rep["ok"] and "simulated crash" in rep["error"] and rep["written"]
    assert rep["rollback"]["edges_total"] >= 1 and rep["active_version"] == "v1"
    assert _graph_edges_of(st, ids["graph"]) == v1             # the old version is complete and live
    assert st.graph.node(ent2.node_id) is None                 # the new-only vertex went with the rollback
    assert st.graph.active_version(ids["graph"]) == "v1"
    assert st.graph.parity(*art1, "v1", graph=ids["graph"])["ok"]
    monkeypatch.setattr(st.graph, "_write_edge_batch", real)
    # a parity failure is a failed load as well: the new version is rolled back, v1 stays
    monkeypatch.setattr(st.graph, "parity", lambda *a, **k: {"ok": False})
    with pytest.raises(GraphLoadError, match="parity"):
        st.graph.load_graph(*art2, "v2", graph=ids["graph"], strict_corpus=False)
    assert _graph_edges_of(st, ids["graph"]) == v1 and st.graph.active_version(ids["graph"]) == "v1"


@integration
def test_load_refuses_when_corpus_differs_from_artifact(kg):
    from graphrag.kg.store import GraphLoadError
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A = _law(ids["A"], [1, 2])
    upsert_document(st, A)
    nodes, edges = _artifact([_law(ids["A"], [1, 2, 3])])     # the artifact knows an article Neo4j lacks
    with pytest.raises(GraphLoadError, match="precheck") as ei:
        st.graph.load_graph(nodes, edges, "v1", graph=ids["graph"], strict_corpus=False)
    assert ei.value.report["precheck"]["missing_corpus_nodes"] == 1
    assert _graph_edges_of(st, ids["graph"]) == [] and st.graph.active_version(ids["graph"]) is None


# ── Task 10 fix round 1 (Ruling R28) ─────────────────────────────────────────────────────────────
import dataclasses  # noqa: E402


def test_graph_write_guard_is_inside_the_store(monkeypatch):
    """R28 (4): load_graph / drop_versions refuse endpoints outside store.write_endpoints before any I/O."""
    from graphrag.kg.store import GraphStore
    from graphrag.stores.guard import graph_write_problem
    allowed = config.load("graph")["store"]["write_endpoints"]
    monkeypatch.setenv("GRL_ALLOW_RESET", "1")                # no env override
    assert graph_write_problem("bolt://localhost:7688", allowed) is None
    assert graph_write_problem("neo4j://127.0.0.1:7688", allowed) is None
    for bad in ("bolt://localhost:7687", "bolt://203.0.113.10:7688", "http://localhost:7688", "bolt://localhost",
                "bolt+s://localhost:7688"):
        assert graph_write_problem(bad, allowed), bad
    assert graph_write_problem("bolt://10.0.0.5:7688", ["10.0.0.5:7688"]) is None   # allowlist comes from config
    g = object.__new__(GraphStore)                           # no driver: a refused call must not touch it
    g.uri, g.driver = "bolt://203.0.113.10:7687", None
    with pytest.raises(PermissionError, match="allowlisted"):
        g.load_graph([], [], "v1")
    with pytest.raises(PermissionError, match="allowlisted"):
        g.drop_versions("main", "all")


def test_reset_all_clears_graph_meta_and_events():
    """R28 (5): build_indexes.py --reset goes through GraphStore.reset_all, which also drops GraphMeta/GraphEvent."""
    from graphrag.kg.store import GraphStore
    ran = []

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, q, **kw):
            ran.append(q)
            return type("R", (), {"consume": lambda self: None})()

    g = object.__new__(GraphStore)
    g.driver = type("D", (), {"session": lambda self: _S()})()
    g.reset_all()
    assert ran == ["MATCH (n:Node) DETACH DELETE n", "MATCH (m:GraphMeta) DELETE m", "MATCH (e:GraphEvent) DELETE e"]
    src = (config.ROOT / "scripts" / "build_indexes.py").read_text(encoding="utf-8")
    assert "st.graph.reset_all()" in src and "DETACH DELETE" not in src


def _snap(st, node_ids):
    with st.graph.driver.session() as s:
        return {r["id"]: (r["p"], sorted(r["l"])) for r in s.run(
            "UNWIND $ids AS id MATCH (n:Node {node_id: id}) RETURN id, properties(n) AS p, labels(n) AS l",
            ids=list(node_ids))}


def _probe_artifacts(ids):
    """v1 / v2 on the same corpus: v2 changes a shared entity's title and attrs and drops its `status`, adds a
    key and drops one on a corpus vertex, turns a concept into an entity (kind label change)."""
    from graphrag.stores.sync import upsert_document  # noqa: F401  (callers upsert A and B)
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    xn, xe = _enrichment(ids)
    ent = xn[0]
    ent1 = dataclasses.replace(ent, attrs={**ent.attrs, "status": "old"})
    con1 = GraphNode(f"CON/grltest {ids['tag']}", "concept", "concept", "test concept", None, False, {"mentions": 1})
    a1 = f"{ids['A']}/СТ.1"
    n1, e1 = _artifact([A, B], [ent1, con1] + xn[1:], xe)
    n1 = [dataclasses.replace(n, attrs={**n.attrs, "status": "old"}) if n.node_id == a1 else n for n in n1]
    ent2 = dataclasses.replace(ent, title="NEW", attrs={**ent.attrs, "mentions": 99, "fresh": "x"})
    con2 = dataclasses.replace(con1, kind="entity", type="organ", attrs={"mentions": 2})
    n2, e2 = _artifact([A, B], [ent2, con2] + xn[1:],
                       xe + [GraphEdge(f"{ids['B']}/СТ.1", con2.node_id, "REGULATES", {"evidence": _ev(ids["B"])})])
    n2 = [dataclasses.replace(n, attrs={**n.attrs, "fresh": "y"}) if n.node_id == a1 else n for n in n2]
    return A, B, (n1, e1), (n2, e2), [ent.node_id, con1.node_id, a1]


@integration
def test_failed_load_restores_the_old_version_exactly(kg, monkeypatch):
    """The reviewer's probe (R28 (2)): v2 changes + drops attrs on shared vertices and dies mid-load (and,
    separately, fails parity) -> Neo4j equals v1 exactly: full parity, every prop, label and edge."""
    from graphrag.kg.store import GraphLoadError
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B, art1, art2, watch = _probe_artifacts(ids)
    upsert_document(st, A)
    upsert_document(st, B)
    rep1 = st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)
    assert rep1["ok"] and rep1["parity"]["full"] and rep1["parity"]["ok"]
    all_ids = [n.node_id for n in art1[0]]
    before, v1_edges = _snap(st, all_ids), _graph_edges_of(st, ids["graph"])
    assert before[watch[0]][0]["status"] == "old" and before[watch[1]][1] == ["Concept", "Node"]
    real, calls = st.graph._write_edge_batch, []

    def boom(etype, rows, graph, version):
        if calls:
            raise RuntimeError("simulated crash mid-load")
        calls.append(etype)
        real(etype, rows, graph, version)

    monkeypatch.setattr(st.graph, "_write_edge_batch", boom)
    with pytest.raises(GraphLoadError) as ei:
        st.graph.load_graph(*art2, "v2", graph=ids["graph"], strict_corpus=False, batch=2)
    rep = ei.value.report
    assert rep["staged_vertices"] >= len(all_ids) and rep["rollback_restored_vertices"] == rep["staged_vertices"]
    par = st.graph.parity(*art1, "v1", graph=ids["graph"])
    assert par["full"] and par["ok"], par["mismatch_examples"]
    assert _snap(st, all_ids) == before                        # props (no staging slot left), labels
    assert _graph_edges_of(st, ids["graph"]) == v1_edges and st.graph.active_version(ids["graph"]) == "v1"
    monkeypatch.setattr(st.graph, "_write_edge_batch", real)
    real_parity = st.graph.parity
    monkeypatch.setattr(st.graph, "parity", lambda *a, **k: {"ok": False})
    with pytest.raises(GraphLoadError, match="parity"):
        st.graph.load_graph(*art2, "v2", graph=ids["graph"], strict_corpus=False)
    monkeypatch.setattr(st.graph, "parity", real_parity)
    assert _snap(st, all_ids) == before and st.graph.parity(*art1, "v1", graph=ids["graph"])["ok"]


@integration
def test_successful_swap_leaves_no_stale_keys(kg):
    """R28 (1)+(2): a successful v2 equals v2 exactly -- dropped attrs gone from the flat props, the
    staging slot cleared, one kind label; structure-owned props of corpus vertices untouched."""
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B, art1, art2, (ent, con, a1) = _probe_artifacts(ids)
    upsert_document(st, A)
    upsert_document(st, B)
    st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)
    struct_before = {k: v for k, v in st.graph.node(a1).items() if k in ("title", "type", "number", "n_chunks")}
    rep = st.graph.load_graph(*art2, "v2", graph=ids["graph"], strict_corpus=False)
    assert rep["ok"] and rep["parity"]["full"] and rep["parity"]["ok"] and rep["staging_cleared"] >= 3
    assert st.graph.parity(*art2, "v2", graph=ids["graph"])["ok"]
    snap = _snap(st, [ent, con, a1])
    e, c, a = (snap[i][0] for i in (ent, con, a1))
    assert "status" not in e and e["title"] == "NEW" and e["mentions"] == 99 and e["fresh"] == "x"
    assert "status" not in a and a["fresh"] == "y" and {k: a[k] for k in struct_before} == struct_before
    assert snap[con][1] == ["Entity", "Node"] and c["type"] == "organ"
    for p, _ in snap.values():
        assert "prev_json" not in p and "prev_for" not in p
    assert sorted(e["enr_keys"]) == sorted(k for k in store_node_props(art2[0], ent) if k != "node_id")


def store_node_props(nodes, node_id):
    from graphrag.kg.store import node_props
    return node_props(next(n for n in nodes if n.node_id == node_id))


@integration
def test_staging_left_by_a_dead_load_is_restored_by_the_next_load(kg):
    """A load that died without its rollback (process killed) leaves staged vertices; the next load restores them."""
    from graphrag.kg.store import node_props
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B, art1, art2, (ent, con, a1) = _probe_artifacts(ids)
    upsert_document(st, A)
    upsert_document(st, B)
    st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)
    all_ids = [n.node_id for n in art1[0]]
    before = _snap(st, all_ids)
    v2 = {n.node_id: n for n in art2[0]}
    st.graph._write_node_batch("Entity", [{"id": ent, "p": node_props(v2[ent])}], f"{ids['graph']}@v2")
    st.graph._write_corpus_batch([{"id": a1, "p": node_props(v2[a1])}], f"{ids['graph']}@v2")
    assert st.graph.node(ent)["title"] == "NEW" and st.graph.node(ent)["prev_for"] == f"{ids['graph']}@v2"
    assert not st.graph.parity(*art1, "v1", graph=ids["graph"])["ok"]
    rep = st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)   # idempotent reload of v1
    assert rep["ok"] and rep["purged_staging"] == {"restored": 2, "cleared": 0}
    assert _snap(st, all_ids) == before


@integration
def test_full_parity_sees_stale_props_labels_membership_and_extra_edges(kg):
    """R28 (3): what the old sampled parity could not see."""
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B, art1, _, (ent, con, a1) = _probe_artifacts(ids)
    upsert_document(st, A)
    upsert_document(st, B)
    st.graph.load_graph(*art1, "v1", graph=ids["graph"], strict_corpus=False)
    g = ids["graph"]
    breakages = [
        ("MATCH (n:Node {node_id: $e}) SET n.stale_key = 'x'", "MATCH (n:Node {node_id: $e}) REMOVE n.stale_key",
         "stale:stale_key"),
        ("MATCH (n:Node {node_id: $a}) SET n.status = 'changed'", "MATCH (n:Node {node_id: $a}) SET n.status = 'old'",
         "flat:status"),
        ("MATCH (n:Node {node_id: $e}) SET n:Concept", "MATCH (n:Node {node_id: $e}) REMOVE n:Concept", "labels:"),
        ("MATCH (n:Node {node_id: $e}) SET n.versions = n.versions + 'x@y', n.versions = ['x@y']",
         f"MATCH (n:Node {{node_id: $e}}) SET n.versions = ['{g}@v1']", "versions:tag_missing"),
        (f"MATCH (a:Node {{node_id: $a}}), (b:Node {{node_id: $c}}) CREATE (a)-[:REGULATES "
         f"{{edge_key: 'k', graph: '{g}', graph_version: 'v1', json_attrs: []}}]->(b)",
         "MATCH ()-[r:REGULATES {edge_key: 'k'}]->() DELETE r", "extra"),
    ]
    for brk, fix, what in breakages:
        with st.graph.driver.session() as s:
            s.run(brk, e=ent, a=a1, c=con).consume()
        par = st.graph.parity(*art1, "v1", graph=g)
        assert not par["ok"] and what in json.dumps(par["mismatch_examples"], ensure_ascii=False), what
        with st.graph.driver.session() as s:
            s.run(fix, e=ent, a=a1, c=con).consume()
        assert st.graph.parity(*art1, "v1", graph=g)["ok"], what


@integration
def test_load_lock_refuses_a_concurrent_load_and_breaks_only_stale_locks(kg):
    """R28 (7)."""
    from graphrag.kg.store import GraphLoadError
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A = _law(ids["A"], [1, 2])
    upsert_document(st, A)
    art = _artifact([A])
    g = ids["graph"]
    with st.graph.driver.session() as s:
        s.run("MERGE (m:GraphMeta {graph: $g}) SET m.lock_owner = 'other:1:x', m.lock_at = datetime()", g=g).consume()
    for brk in (False, True):                                # a fresh lock is never broken
        with pytest.raises(GraphLoadError, match="locked by other:1:x"):
            st.graph.load_graph(*art, "v1", graph=g, strict_corpus=False, break_stale_lock=brk)
    with st.graph.driver.session() as s:
        s.run("MATCH (m:GraphMeta {graph: $g}) SET m.lock_at = datetime() - duration('PT2H')", g=g).consume()
    with pytest.raises(GraphLoadError, match="locked"):
        st.graph.load_graph(*art, "v1", graph=g, strict_corpus=False)
    rep = st.graph.load_graph(*art, "v1", graph=g, strict_corpus=False, break_stale_lock=True)
    assert rep["ok"] and rep["lock"]["broke_stale_lock"]["owner"] == "other:1:x"
    with pytest.raises(GraphLoadError, match="precheck"):     # a failing load releases the lock as well
        st.graph.load_graph(*_artifact([_law(ids["A"], [1, 2, 3])]), "v2", graph=g, strict_corpus=False)
    with st.graph.driver.session() as s:
        m = s.run("MATCH (m:GraphMeta {graph: $g}) RETURN m.lock_owner AS o, m.lock_at AS at", g=g).single()
    assert m["o"] is None and m["at"] is None


@integration
def test_stale_active_version_is_not_protected(kg, monkeypatch):
    """R28 (5) / review M2: GraphMeta naming a version with no data (a reset before R28) does not stop the
    rollback of a failed load of that same version."""
    from graphrag.kg.store import GraphLoadError
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    with st.graph.driver.session() as s:
        s.run("MERGE (m:GraphMeta {graph: $g}) SET m.active_version = 'v1'", g=ids["graph"]).consume()
    monkeypatch.setattr(st.graph, "parity", lambda *a, **k: {"ok": False})
    with pytest.raises(GraphLoadError) as ei:
        st.graph.load_graph(*_artifact([A, B], xn, xe), "v1", graph=ids["graph"], strict_corpus=False)
    assert ei.value.report["stale_active_version"] == "v1" and ei.value.report["rollback"]["edges_total"] == len(xe)
    assert _graph_edges_of(st, ids["graph"]) == [] and st.graph.node(xn[0].node_id) is None


@integration
def test_version_indexes_exist_and_are_used(kg):
    """R28 (6): (graph, graph_version) index per edge type, used by the drop and parity queries."""
    from graphrag.kg.store import _VER_SEL
    st, _ = kg
    with st.graph.driver.session() as s:
        names = {r["name"] for r in s.run("SHOW INDEXES YIELD name RETURN name")}
        assert {f"rel_{t.lower()}_version" for t in config.load("graph")["edge_types"]} | {"node_prev_for"} <= names

        def ops(p):
            return [p["operatorType"]] + [o for c in p.get("children", []) for o in ops(c)]
        for q in (f"EXPLAIN MATCH ()-[r:AMENDS]->() WHERE r.graph = $g AND r.graph_version IS NOT NULL AND ({_VER_SEL}) "
                  "WITH r LIMIT 20000 DELETE r RETURN count(*) AS c",
                  "EXPLAIN MATCH (a:Node)-[r:REFERS_TO]->(b:Node) WHERE r.graph = $g AND r.graph_version = $v RETURN r"):
            plan = s.run(q, g="main", v="x", mode="keep").consume().plan
            assert any("RelationshipIndexSeek" in o for o in ops(plan)), ops(plan)


@integration
def test_orphans_after_corpus_removal_are_reported_and_cleaned(kg):
    """R28 (8): removing A/СТ.1 leaves the entity only it regulated with no edge; the next load reports it and,
    when the rebuilt artifact no longer has it, the swap removes it."""
    from graphrag.stores.sync import upsert_document
    st, ids = kg
    A, B = _law(ids["A"], [1, 2]), _law(ids["B"], [1])
    upsert_document(st, A)
    upsert_document(st, B)
    xn, xe = _enrichment(ids)
    st.graph.load_graph(*_artifact([A, B], xn, xe), "v1", graph=ids["graph"], strict_corpus=False)
    a1 = f"{ids['A']}/СТ.1"
    A2 = _law(ids["A"], [2])
    upsert_document(st, A2)
    assert _incoming(st, xn[0].node_id, ids["graph"]) == {}             # the entity is an orphan now
    rep = st.graph.load_graph(*_artifact([A2, B], xn[1:], [e for e in xe if a1 not in (e.src, e.dst)]), "v2",
                              graph=ids["graph"], strict_corpus=False)
    assert rep["ok"] and rep["orphans"] == {"previous_version": 1, "examples": [xn[0].node_id],
                                            "removed_by_swap": 1, "kept_in_new_version": 0}
    assert st.graph.node(xn[0].node_id) is None
