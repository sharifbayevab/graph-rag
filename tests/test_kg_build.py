# tests/test_kg_build.py -- M2 Task 9: graph build, node features, §3.4 validation, PyG export (Rulings R1, R11,
# R24, R25)
import copy
import json
from collections import Counter

import numpy as np
import pytest

from graphrag import config
from graphrag.data.mapping import mu_inverse
from graphrag.kg import build as B
from graphrag.kg import semantic as S
from graphrag.kg.acts import ActRegistry
from graphrag.kg.artifact import SchemaMismatch, read_graph, write_graph
from graphrag.kg.extract_ref import extract_refs
from graphrag.kg.features import VecCache, WordCutter, node_text, text_sha
from graphrag.kg.pyg import load_pyg, to_pyg
from graphrag.kg.schema import GraphEdge, GraphNode, Schema
from tests.helpers import FakeEncoder
from tests.test_kg_entities import DEFS
from tests.test_kg_refs import _rec

LAW_A = ["1-модда. Жиноят кодексига ўзгартишлар",
         "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",
         "1) 27-модданинг иккинчи қисми қуйидаги таҳрирда баён этилсин:",
         "«Жарима ушбу Кодекснинг 27-моддасининг иккинчи қисмида назарда тутилган тартибда тайинланади»;",
         "2-модда. Айрим қонун ҳужжатларини ўз кучини йўқотган деб топиш",
         "Қуйидагилар ўз кучини йўқотган деб топилсин:",
         "1) Ўзбекистон Республикасининг 1997 йил 24 апрелда қабул қилинган «Табиий монополиялар тўғрисида»ги "
         "398–I-сонли Қонуни;",
         "3-модда. Ижро", "Вазирлар Маҳкамаси ушбу Қонуннинг ижросини таъминласин.",
         "4-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]
A, B2 = "QONUNLAR-bbbb0001", "QONUNLAR-bbbb0002"
REL = GraphEdge(f"{A}/СТ.3", f"{B2}/СТ.3", "RELATED", {"weight": 0.71})


def _recs():
    return [_rec(LAW_A, A, "Жиноят кодексига ўзгартишлар киритиш тўғрисида", "2024-02-01"),
            _rec(DEFS, B2, "Давлат ижтимоий суғуртаси тўғрисида")]


def _build(recs=None, related=(REL,), cache=None):
    recs = recs or _recs()
    ex = B.extract_all(recs, llm_factory=lambda: None)
    return B.build_graph(recs, None, encoder=FakeEncoder(), related=list(related), extracted=ex, cache=cache)


@pytest.fixture(scope="module")
def graph():
    return _build()


# --- build ------------------------------------------------------------------------------------------

def test_two_records_give_all_seven_edge_types_and_validate(graph):
    types = Counter(e.type for e in graph.edges)
    assert set(types) == set(Schema.load().edge_types)
    assert B.validate(graph) == [] and graph.stats["validation"]["violations"] == 0
    assert {n.kind for n in graph.nodes} == {"document", "element", "entity", "concept"}
    assert [n.node_id for n in graph.nodes] == sorted(n.node_id for n in graph.nodes)
    rp = next(e for e in graph.edges if e.type == "REPEALS")
    assert rp.attrs["date"] == "2024-02-01"
    assert next(n for n in graph.nodes if n.node_id == rp.dst).attrs["repealed_on"] == "2024-02-01"


def test_r11_drops_amended_but_keeps_the_in_quote_citation(graph):
    """R11 + R24 (2): the amendment head REFERS_TO{amended} twin of AMENDS is dropped; the citation inside the
    quoted new text of the same (src, dst) is its own REFERS_TO{cited_in_amendment} edge and survives."""
    am = next(e for e in graph.edges if e.type == "AMENDS")
    refs = [e for e in graph.edges if e.type == "REFERS_TO" and (e.src, e.dst) == (am.src, am.dst)]
    assert [e.attrs["relation"] for e in refs] == ["cited_in_amendment"]
    assert graph.stats["r11_dropped_refers_amended"] == 1


def test_rel_priority_keeps_one_edge_per_relation_class():
    rec = _recs()[0]
    es = extract_refs(rec, ActRegistry.from_records([rec]))
    pair = [e for e in es if e.dst == "EXT/jinoyat kodeks/СТ.27/Ч.2"]
    assert sorted(e.attrs["relation"] for e in pair) == ["amended", "cited_in_amendment"]
    assert len({e.src for e in pair}) == 1


def test_edge_dedup_merges_evidence_and_keys_refers_to_by_relation():
    ev = lambda s: {"doc_id": "D", "start": s, "end": s + 2, "quote": "xx"}
    es = [GraphEdge("a", "b", "REFERS_TO", {"evidence": ev(5), "relation": "act"}),
          GraphEdge("a", "b", "REFERS_TO", {"evidence": ev(1), "relation": "act"}),
          GraphEdge("a", "b", "REFERS_TO", {"evidence": ev(9), "relation": "cited_in_amendment"}),
          GraphEdge("a", "b", "RELATED", {"weight": 0.4}), GraphEdge("a", "b", "RELATED", {"weight": 0.6})]
    out, merged = B.dedup_edges(es)
    assert merged == {"REFERS_TO": 1, "RELATED": 1} and len(out) == 3
    act = next(e for e in out if e.attrs.get("relation") == "act")
    assert act.attrs["evidence"]["start"] == 1 and [x["start"] for x in act.attrs["evidence_all"]] == [1, 5]
    assert next(e for e in out if e.type == "RELATED").attrs == {"weight": 0.6}


def test_entity_dedup_moves_edges_to_the_minimal_id():
    n = lambda i: GraphNode(i, "entity", "organ", i, None, False, {"norm_key": "k"})
    nodes = {"ENT/organ/a": n("ENT/organ/a"), "ENT/organ/b": n("ENT/organ/b")}
    ev = {"doc_id": "D", "start": 0, "end": 1, "quote": "x"}
    out_n, out_e, remap = B.dedup_entities(nodes, [GraphEdge("D/СТ.1", "ENT/organ/b", "REGULATES", {"evidence": ev})])
    assert remap == {"ENT/organ/b": "ENT/organ/a"} and set(out_n) == {"ENT/organ/a"}
    assert out_e[0].dst == "ENT/organ/a"


# --- isolated vertices (R1 / R24 (1)) ----------------------------------------------------------------

def _mini(docs=("D1", "D2"), linked=()):
    nodes = []
    for d in docs:
        nodes += [GraphNode(d, "document", "документ", d, d, False, {}),
                  GraphNode(f"{d}/СТ.1", "element", "статья", "1", d, False, {}),
                  GraphNode(f"{d}/СТ.2", "element", "статья", "2", d, False, {})]
    nodes.sort(key=lambda n: n.node_id)
    edges = [GraphEdge(d, f"{d}/СТ.1", "CONTAINS", {}) for d in docs]
    ev = {"doc_id": "D1", "start": 0, "end": 1, "quote": "x"}
    edges += [GraphEdge(a, b, "REFERS_TO", {"evidence": ev, "relation": "version_of"}) for a, b in linked]
    return B.Graph(nodes, edges, schema=Schema.load(), cfg=config.load("graph"))


def test_isolated_vertex_attaches_to_nearest_of_its_document_by_x():
    g = _mini()
    ids = [n.node_id for n in g.nodes]          # D1, D1/СТ.1, D1/СТ.2, D2, D2/СТ.1, D2/СТ.2
    X = np.eye(6, dtype=np.float32)
    X[ids.index("D1/СТ.2")] = [0, 0.8, 0, 0, 0.6, 0]            # closest to D2/СТ.1, but same doc wins
    X[ids.index("D2/СТ.2")] = [0, 0, 0, 1, 0, 0]
    info = B.attach_isolated(g, X)
    att = {(e.src, e.dst): e.attrs for e in g.edges if e.type == "RELATED"}
    assert att[("D1/СТ.1", "D1/СТ.2")] == {"weight": 0.8, "attached": True}
    assert att[("D2", "D2/СТ.2")] == {"weight": 1.0, "attached": True}
    assert info["attached"] == 2 and info["same_document"] == 2 and B.isolated(g) == []
    ls = B.latent_share(g)
    assert ls["related_latent"] == 0 and ls["related_attached"] == 2 and ls["share_without_attached"] == 0.0


def test_attach_falls_back_to_corpus_but_never_to_a_linked_version_document():
    doc = lambda d: GraphNode(d, "document", "документ", d, d, False, {})
    el = lambda d, k: GraphNode(f"{d}/СТ.{k}", "element", "статья", str(k), d, False, {})
    ev = {"doc_id": "D2", "start": 0, "end": 1, "quote": "x"}
    nodes = sorted([doc("D1"), doc("D2"), el("D2", 1), doc("D3"), el("D3", 1),
                    GraphNode("EXT/x/СТ.1", "element", "статья", "1-модда", None, True, {"act": "EXT/x"})],
                   key=lambda n: n.node_id)
    edges = [GraphEdge("D2", "D2/СТ.1", "CONTAINS", {}), GraphEdge("D3", "D3/СТ.1", "CONTAINS", {}),
             GraphEdge("D2", "D1", "REFERS_TO", {"evidence": ev, "relation": "version_of"})]
    g = B.Graph(nodes, edges, schema=Schema.load(), cfg=config.load("graph"))
    ids = [n.node_id for n in nodes]
    X = np.zeros((len(ids), 3), dtype=np.float32) + [1, 0, 0]
    X[ids.index("EXT/x/СТ.1")] = [0, 0, 1]
    X[ids.index("D3/СТ.1")] = [0, 0.6, 0.8]            # nearest to the stub (corpus: it has no document)
    B.attach_isolated(g, X)                              # D1 is linked (version_of) -> not isolated
    e = next(e for e in g.edges if e.attrs.get("attached"))
    assert (e.src, e.dst) == ("D3/СТ.1", "EXT/x/СТ.1") and e.attrs["weight"] == 0.8
    # D1 alone in its document: the corpus, but not D2 (its version document) although D2 is nearest
    g = B.Graph(nodes, edges[:2] + [GraphEdge("D2", "D1", "REFERS_TO", {"evidence": ev, "relation": "version_of"})],
                schema=Schema.load(), cfg=config.load("graph"))
    g.edges = [e for e in g.edges if e.type != "REFERS_TO"]
    g.edges.append(GraphEdge("D2/СТ.1", "D3/СТ.1", "RELATED", {"weight": 0.5}))
    groups_edges = [GraphEdge("D2", "D1", "REFERS_TO", {"evidence": ev, "relation": "version_of"})]
    X = np.zeros((len(ids), 3), dtype=np.float32) + [1, 0, 0]
    X[ids.index("D1")] = [0, 1, 0]
    X[ids.index("D2")] = [0, 1, 0]
    X[ids.index("D3")] = [0, 0.6, 0.8]
    orig = B.doc_groups_of
    B.doc_groups_of = lambda graph: S.doc_groups(["D1", "D2", "D3"], [("D2", "D1")])
    try:
        B.attach_isolated(g, X)
    finally:
        B.doc_groups_of = orig
    got = {(e.src, e.dst) for e in g.edges if e.attrs.get("attached")}
    assert ("D1", "D3") in got and not any("D2" in p and "D1" in p for p in got)
    assert groups_edges


# --- validation --------------------------------------------------------------------------------------

def _broken(graph, fn):
    g = copy.copy(graph)
    g.nodes, g.edges = list(graph.nodes), [GraphEdge(e.src, e.dst, e.type, copy.deepcopy(e.attrs)) for e in graph.edges]
    fn(g)
    return B.validate(g)


def test_validation_catches_repeal_without_date(graph):
    def f(g):
        e = next(e for e in g.edges if e.type == "REPEALS")
        del e.attrs["date"]
    assert any("no date" in m for m in _broken(graph, f))


def test_validation_catches_isolated_vertex(graph):
    def f(g):
        g.nodes = sorted(g.nodes + [GraphNode(f"{A}/СТ.99", "element", "статья", "99", A, False, {})],
                         key=lambda n: n.node_id)
    assert f"isolated vertex {A}/СТ.99" in _broken(graph, f)


def test_validation_catches_structural_vertex_without_chunks(graph):
    def f(g):
        g.rindex = copy.copy(graph.rindex)
        g.rindex.mu_inv = {**graph.rindex.mu_inv, f"{A}/СТ.3/Ч.1": []}
    assert _broken(graph, f) == [f"{A}/СТ.3/Ч.1: empty μ⁻¹ (no chunks)"]
    assert B.validate(graph) == []


def test_validation_catches_contains_not_to_a_child(graph):
    def f(g):
        g.edges.append(GraphEdge(A, f"{A}/СТ.1/Ч.1", "CONTAINS", {}))
        g.edges.append(GraphEdge("EXT/jinoyat kodeks", "EXT/jinoyat kodeks/СТ.27/Ч.2", "CONTAINS", {}))
    errs = _broken(graph, f)
    assert any("not the immediate parent" in m for m in errs)
    assert any("stub is not an immediate path child" in m for m in errs)


def test_validation_catches_evidence_not_a_substring(graph):
    def f(g):
        e = next(e for e in g.edges if e.type == "DEFINES")
        e.attrs["evidence"]["quote"] += " (tampered)"
    assert any("is not its quote" in m for m in _broken(graph, f))


def test_validation_catches_latent_share_above_the_cap(graph):
    def f(g):
        g.cfg = {**g.cfg, "related": {**g.cfg["related"], "max_latent_share": 0.01}}
    assert any("max_latent_share" in m for m in _broken(graph, f))
    assert not any("max_latent_share" in m for m in B.validate(graph))


def test_validation_catches_untyped_node_and_unknown_edge_type(graph):
    def f(g):
        g.nodes[0] = GraphNode(g.nodes[0].node_id, g.nodes[0].kind, "", "", None, False, {})
        g.edges.append(GraphEdge(A, B2, "SEES", {}))
    errs = _broken(graph, f)
    assert any("without type" in m for m in errs) and any("unknown edge type" in m for m in errs)


# --- features ----------------------------------------------------------------------------------------

def test_node_texts(graph):
    by = dict(zip([n.node_id for n in graph.nodes], graph.texts))
    assert by[A].startswith("jinoyat kodeksiga oʼzgartishlar kiritish toʼgʼrisida\nqonunlar")
    art = by[f"{A}/СТ.3"]
    assert art.startswith("jinoyat kodeksiga oʼzgartishlar kiritish toʼgʼrisida › 3-modda. ijro\n")
    assert "vazirlar mahkamasi" in art and art.count("3-modda. ijro") == 1
    assert by["EXT/jinoyat kodeks/СТ.27/Ч.2"] == "jinoyat kodeksi › 27-modda › 2-qism"
    assert by["ENT/organ/vazir mahkama"] == "vazirlar mahkamasi · davlat organi"
    con = next(t for i, t in by.items() if i.startswith("CON/sugʼurtalangan"))
    assert con.startswith("sugʼurtalangan shaxs – ")


def test_body_is_cut_to_body_tokens():
    recs = _recs()
    long = _rec(["1-модда. Узун", " ".join(["сўз"] * 900)], "QONUNLAR-cccc0001", "Узун")
    n = GraphNode("QONUNLAR-cccc0001/СТ.1", "element", "статья", "Узун", "QONUNLAR-cccc0001", False, {})
    t = node_text(n, recs + [long], WordCutter(), {"body_tokens": 50})
    assert len(t.split()) <= 50 + 12


def test_x_cache_by_text_sha(tmp_path, graph):
    enc = FakeEncoder()
    cache = VecCache(tmp_path / "node_vecs.npy", enc.model_name)
    g1 = _build(cache=cache)
    assert g1.feature_info["encoded"] == len(set(g1.texts)) and g1.x.shape == (len(g1.nodes), 8)
    c2 = VecCache(tmp_path / "node_vecs.npy", enc.model_name)
    assert c2.loaded == len(set(g1.texts))
    g2 = _build(cache=c2)
    assert g2.feature_info["encoded"] == 0 and np.array_equal(g1.x, g2.x)
    k = text_sha(g1.texts[0])
    assert np.array_equal(c2.vecs[k], FakeEncoder().encode_passages([g1.texts[0]])[0])


# --- PyG ---------------------------------------------------------------------------------------------

def test_to_pyg_shapes_types_and_mu_inverse(graph):
    d = to_pyg(graph, graph.x)
    N = len(graph.nodes)
    n_rel = sum(e.type == "RELATED" for e in graph.edges)
    assert d.edge_index.shape == (2, len(graph.edges) + n_rel)          # RELATED in both directions
    assert int(d.edge_type.min()) >= 0 and int(d.edge_type.max()) < 7
    assert tuple(d.x.shape) == (N, 8) and d.num_nodes == N
    rel = int(Schema.load().edge_type_id["RELATED"])
    m = d.edge_type == rel
    assert set(d.edge_weight[~m].tolist()) == {1.0} and d.edge_weight[m].tolist() == pytest.approx([0.71, 0.71])
    pairs = {tuple(x) for x in d.edge_index[:, m].t().tolist()}
    assert all((b, a) in pairs for a, b in pairs)
    # μ⁻¹ CSR == graphrag.data.mapping.mu_inverse of the records
    want = {}
    for r in _recs():
        want.update(mu_inverse(r.structure, r.chunks))
    for i, nid in enumerate(d.node_ids):
        got = [d.chunk_ids[j] for j in d.mu_inv_idx[d.mu_inv_ptr[i]:d.mu_inv_ptr[i + 1]].tolist()]
        assert sorted(got) == sorted(want.get(nid, [])), nid
    assert d.mu_inv_ptr[-1] == d.mu_inv_idx.numel()
    for c, k in zip(d.chunk_ids, d.chunk_node.tolist()):
        assert c.split("#")[0] == d.node_ids[k]
    assert d.node_kinds == ["document", "element", "entity", "concept"]
    assert [d.node_kinds[k] for k in d.node_kind.tolist()] == [n.kind for n in graph.nodes]


def test_two_builds_write_byte_identical_artifacts(tmp_path):
    shas = []
    for k in ("a", "b"):
        g = _build(list(reversed(_recs())) if k == "b" else None)
        stats = {"version": "test", **g.stats}
        shas.append(B.write_artifact(g, tmp_path / k, stats))
    assert shas[0] == shas[1]
    for f in ("nodes.jsonl", "edges.jsonl", "x.npy", "pyg.pt", "stats.json", "schema.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f
    d = load_pyg(tmp_path / "a" / "pyg.pt")
    assert d.edge_index.shape[0] == 2 and len(d.node_ids) == d.x.shape[0]
    nodes, edges = read_graph(tmp_path / "a")
    assert len(nodes) == d.x.shape[0]


# --- Ruling R25: one approved RELATED source ----------------------------------------------------------

def test_latent_edges_refuses_topic_mode_without_topic_vectors():
    ra = S_rec("QONUNLAR-x1")
    rb = S_rec("QONUNLAR-x2")
    vecs = {c.chunk_id: np.ones(4, dtype=np.float32) / 2 for r in (ra, rb) for c in r.chunks}
    with pytest.raises(ValueError, match="topic"):
        S.latent_edges([ra, rb], vecs, 0.5, 5, 5, cfg={"score_vectors": "topic"})
    assert S.latent_edges([ra, rb], vecs, 0.5, 5, 5, cfg={"score_vectors": "topic", "dedup": None, "template": None},
                          score_vectors=vecs)
    rel = config.load("graph")["related"]
    assert rel["score_vectors"] == "topic" and rel["theta_sim"] == 0.65 and rel["adopted_edges"] == 1938


def S_rec(doc_id):
    from tests.test_kg_semantic import _rec as srec
    return srec(doc_id, {1: 1})


def _write_cache(d, name, ids, digest, X):
    np.save(d / f"{name}.npy", X)
    import hashlib
    (d / f"{name}_ids.json").write_text(json.dumps({"ids": ids, "text_sha256": digest,
                                                     "npy_sha256": hashlib.sha256((d / f"{name}.npy").read_bytes()).hexdigest()}))


def test_related_edges_recomputes_and_checks_the_adopted_set(tmp_path):
    recs = [S_rec("QONUNLAR-x1"), S_rec("QONUNLAR-x2")]
    ids = sorted(c.chunk_id for r in recs for c in r.chunks)
    base = config.load("graph")
    cfg = {**base, "related": {**base["related"], "dedup": None, "template": None, "anisotropy_pcs": None,
                               "adopted_edges": None, "adopted_edges_sha256": None}}
    rc = {**S.DEFAULTS, **cfg["related"]}
    X = np.array([[1, 0, 0], [0.9, 0.436, 0]], dtype=np.float32)
    _write_cache(tmp_path, "chunk_vecs", ids, B.index_text_digest(recs), X)
    _write_cache(tmp_path, "chunk_vecs_topic", ids, B.topic_text_digest(recs, rc), X)
    es, info = B.related_edges(recs, [], cfg, tmp_path)
    assert len(es) == 1 and info["edge_set_sha256"] == B.edge_set_sha(es)
    ok = {**cfg, "related": {**cfg["related"], "adopted_edges": 1, "adopted_edges_sha256": info["edge_set_sha256"]}}
    assert B.related_edges(recs, [], ok, tmp_path)[1]["matches_adopted"]
    bad = {**cfg, "related": {**cfg["related"], "adopted_edges": 1, "adopted_edges_sha256": "0" * 64}}
    with pytest.raises(B.RelatedMismatch, match="sha256"):
        B.related_edges(recs, [], bad, tmp_path)
    with pytest.raises(B.RelatedMismatch, match="theta_sim"):
        B.related_edges(recs, [], {**cfg, "related": {**cfg["related"], "theta_sim": None}}, tmp_path)
    # a topic cache whose texts no longer match is refused (never re-encoded, never silently replaced)
    _write_cache(tmp_path, "chunk_vecs_topic", ids, "f" * 64, X)
    with pytest.raises(B.RelatedMismatch, match="topic vectors"):
        B.related_edges(recs, [], cfg, tmp_path)
    # no fallback to index_text vectors when the topic cache is missing
    (tmp_path / "chunk_vecs_topic.npy").unlink()
    with pytest.raises(B.RelatedMismatch, match="missing"):
        B.related_edges(recs, [], cfg, tmp_path)


# --- Ruling R24 (5): schema hardening ------------------------------------------------------------------

def _gcfg():
    return copy.deepcopy(config.load("graph"))


def test_schema_rejects_unknown_spec_keys_and_bad_vertex_types():
    cfg = _gcfg()
    cfg["edge_types"]["CONTAINS"]["attrs"] = []
    with pytest.raises(ValueError, match="unknown spec keys"):
        Schema(cfg)
    cfg = _gcfg()
    cfg["vertex_types"]["entity"] = "organ"
    with pytest.raises(ValueError, match="vertex_types.entity"):
        Schema(cfg)
    cfg = _gcfg()
    cfg["edge_types"]["CONTAINS"]["src"] = "document"
    with pytest.raises(ValueError, match="list of strings"):
        Schema(cfg)


def test_read_graph_checks_schema_meta_and_write_checks_before_mkdir(tmp_path):
    nodes = [GraphNode("D1", "document", "документ", "Қ", "D1", False, {}),
             GraphNode("D1/СТ.1", "element", "статья", "1", "D1", False, {})]
    d = write_graph(tmp_path / "g", nodes, [GraphEdge("D1", "D1/СТ.1", "CONTAINS", {})])
    cfg = _gcfg()
    cfg["vertex_kinds"] = cfg["vertex_kinds"] + ["extra"]
    with pytest.raises(SchemaMismatch, match="vertex_kinds"):
        read_graph(d, schema=Schema(cfg))
    cfg = _gcfg()
    cfg["version"] = 2
    with pytest.raises(SchemaMismatch, match="version"):
        read_graph(d, schema=Schema(cfg))
    (d / "schema.json").unlink()
    with pytest.raises(SchemaMismatch, match="no schema.json"):
        read_graph(d)
    assert len(read_graph(d, check_schema=False)[0]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        write_graph(tmp_path / "never", nodes + nodes[:1], [])
    assert not (tmp_path / "never").exists()
    # two REFERS_TO of one pair with different relations are distinct edges (R24 (2)); same relation: duplicate
    ev = {"doc_id": "D1", "start": 0, "end": 1, "quote": "Қ"}
    two = [GraphEdge("D1/СТ.1", "D1", "REFERS_TO", {"evidence": ev, "relation": r}) for r in ("amended", "cited_in_amendment")]
    write_graph(tmp_path / "ok", nodes, two)
    with pytest.raises(ValueError, match="duplicate edge"):
        write_graph(tmp_path / "dup", nodes, [two[0], GraphEdge(two[0].src, two[0].dst, "REFERS_TO",
                                                                  {"evidence": {**ev, "end": 0, "quote": ""},
                                                                   "relation": "amended"})])


# --- Ruling R26 (Task 9 fix round 1) -----------------------------------------------------------------------

PHANTOM = "EXT/phantom act toʼgʼrisida"


def test_orphan_ext_act_and_its_stubs_are_pruned_not_attached():
    recs = _recs()
    ex = B.extract_all(recs, llm_factory=lambda: None)
    kept_ext = sorted(n for n, v in ex.nodes.items() if v.kind == "document" and v.external)
    ex.nodes[PHANTOM] = GraphNode(PHANTOM, "document", "документ", "Phantom act", None, True,
                                  {"act_key": "phantom act", "amending": False, "mentions": 2})
    ex.nodes[f"{PHANTOM}/СТ.1"] = GraphNode(f"{PHANTOM}/СТ.1", "element", "статья", "1-модда", None, True,
                                            {"act": PHANTOM, "number": "1"})
    ex.stub_contains = list(ex.stub_contains) + [GraphEdge(PHANTOM, f"{PHANTOM}/СТ.1", "CONTAINS", {})]
    g = B.build_graph(recs, None, encoder=FakeEncoder(), related=[REL], extracted=ex)
    ids = {n.node_id for n in g.nodes}
    assert PHANTOM not in ids and f"{PHANTOM}/СТ.1" not in ids
    assert g.stats["pruned_orphan_ext_acts"] == 1 and g.stats["pruned_orphan_ext_stubs"] == 1
    assert g.stats["pruned_orphan_ext_ids"] == [PHANTOM]
    assert g.stats["isolated_attached"]["attached"] == 0 and g.stats["isolated_attached"]["attached_vertices"] == []
    assert not any(e.attrs.get("attached") for e in g.edges) and B.validate(g) == []
    assert kept_ext and set(kept_ext) <= ids                   # EXT acts with edges stay


def test_prune_keeps_an_ext_act_whose_stub_is_referenced():
    act, stub = "EXT/x", "EXT/x/СТ.1"
    nodes = {act: GraphNode(act, "document", "документ", "x", None, True, {}),
             stub: GraphNode(stub, "element", "статья", "1", None, True, {"act": act}),
             "D1/СТ.1": GraphNode("D1/СТ.1", "element", "статья", "1", "D1", False, {})}
    ev = {"doc_id": "D1", "start": 0, "end": 1, "quote": "x"}
    edges = [GraphEdge(act, stub, "CONTAINS", {}),
             GraphEdge("D1/СТ.1", stub, "REFERS_TO", {"evidence": ev, "relation": "element"})]
    n2, e2, st = B.prune_orphan_ext(dict(nodes), list(edges))
    assert set(n2) == set(nodes) and len(e2) == 2 and st["pruned_orphan_ext_acts"] == 0
    n3, e3, st = B.prune_orphan_ext(dict(nodes), edges[:1])
    assert set(n3) == {"D1/СТ.1"} and e3 == [] and (st["pruned_orphan_ext_acts"], st["pruned_orphan_ext_stubs"]) == (1, 1)


def test_version_digest_covers_all_graphrag_code_config_and_libraries(tmp_path):
    for f, body in {"graphrag/types.py": "a", "graphrag/data/lemma.py": "b", "graphrag/data/legacy/normalize.py": "c",
                    "graphrag/kg/build.py": "d", "scripts/build_graph.py": "e", "scripts/other.py": "f",
                    "config/graph.yaml": "g: 1", "config/data.yaml": "h: 1"}.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(body)
    (tmp_path / "graphrag/__pycache__").mkdir()
    (tmp_path / "graphrag/__pycache__/x.py").write_text("junk")
    code, conf = B.code_digest(tmp_path), B.config_digest(tmp_path)
    assert sorted(code) == ["graphrag/data/legacy/normalize.py", "graphrag/data/lemma.py", "graphrag/kg/build.py",
                            "graphrag/types.py", "scripts/build_graph.py"]
    assert sorted(conf) == ["config/data.yaml", "config/graph.yaml"]
    (tmp_path / "graphrag/data/lemma.py").write_text("b2")
    assert B.code_digest(tmp_path) != code
    (tmp_path / "config/data.yaml").write_text("h: 2")
    assert B.config_digest(tmp_path) != conf
    libs = B.library_versions(("numpy", "no-such-distribution-xyz"))
    assert libs["numpy"] == np.__version__ and libs["no-such-distribution-xyz"] is None
    real = B.code_digest(config.ROOT)
    assert {"graphrag/data/lemma.py", "graphrag/data/structure.py", "graphrag/data/ingest.py",
            "graphrag/data/legacy/normalize.py", "graphrag/types.py", "scripts/build_graph.py"} <= set(real)
    assert {"torch", "torch_geometric", "transformers", "sentence-transformers", "FlagEmbedding"} <= set(B.VERSION_LIBRARIES)


def test_same_version_dir_with_different_content_is_an_error(tmp_path):
    from graphrag.kg.artifact import ArtifactConflict
    g = _build()
    d = tmp_path / "v"
    shas = B.write_artifact(g, d, {"version": "v", **g.stats})
    assert B.write_artifact(g, d, {"version": "v", **g.stats}) == shas       # identical content: fine
    before = {f.name: f.read_bytes() for f in d.iterdir()}
    with pytest.raises(ArtifactConflict, match="stats.json"):
        B.write_artifact(g, d, {"version": "v", **g.stats, "extra": 1})
    g2 = copy.copy(g)
    g2.edges = [e for e in g.edges if e.type != "RELATED"]
    with pytest.raises(ArtifactConflict, match="edges.jsonl"):
        B.write_artifact(g2, d, {"version": "v", **g.stats})
    assert {f.name: f.read_bytes() for f in d.iterdir()} == before           # nothing was written
    B.write_artifact(g2, d, {"version": "v", **g.stats}, overwrite=True)      # explicit overwrite still possible


def test_validation_requires_evidence_in_the_sources_own_document(graph):
    def f(g):                                     # a valid span, but of the other document
        e = next(e for e in g.edges if e.type == "DEFINES")
        other = A if e.src.split("/", 1)[0] == B2 else B2
        t = graph.rindex.texts[other]
        e.attrs["evidence"] = {"doc_id": other, "start": 0, "end": 5, "quote": t[:5]}
    errs = _broken(graph, f)
    assert any("is not the source's document" in m for m in errs)
    assert not any("is not its quote" in m for m in errs)


def test_undeclared_edge_attrs_are_rejected(graph):
    s, nodes = Schema.load(), {n.node_id: n for n in graph.nodes}
    ev = next(e for e in graph.edges if e.type == "REFERS_TO")
    bad = GraphEdge(ev.src, ev.dst, "REFERS_TO", {**ev.attrs, "weight": 0.5})
    assert any("undeclared attr 'weight'" in m for m in s.validate_edge(bad, nodes))
    assert s.validate_edge(ev, nodes) == []
    rel = next(e for e in graph.edges if e.type == "RELATED")
    assert any("undeclared attr 'relation'" in m
               for m in s.validate_edge(GraphEdge(rel.src, rel.dst, "RELATED", {**rel.attrs, "relation": "act"}), nodes))

    def f(g):
        e = next(e for e in g.edges if e.type == "CONTAINS")
        e.attrs["attached"] = True
    assert any("undeclared attr 'attached' for CONTAINS" in m for m in _broken(graph, f))


def test_repeals_evidence_is_the_first_by_position(graph):
    ev = lambda s_, q: {"doc_id": "D1", "start": s_, "end": s_ + len(q), "quote": q}
    item, head = ev(50, "item"), ev(10, "head")
    e = GraphEdge("D1/СТ.2", "D2", "REPEALS", {"evidence": item, "evidence_all": [head, item], "date": "2024-01-01"})
    out, n = B.repeals_first_evidence([e, GraphEdge("D1", "D1/СТ.2", "CONTAINS", {})])
    assert n == 1 and out[0].attrs["evidence"] == head and out[0].attrs["evidence_all"] == [head, item]
    assert out[0].attrs["date"] == "2024-01-01" and out[1].type == "CONTAINS"
    assert B.repeals_first_evidence(out)[1] == 0
    for r in (e for e in graph.edges if e.type == "REPEALS"):
        assert r.attrs["evidence"] == B._evidences(r)[0]
