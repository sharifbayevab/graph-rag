# tests/test_kg_semantic.py -- M2 Task 8: RELATED latent edges (graphrag.kg.semantic, Ruling R21)
import json
import types

import numpy as np
import pytest

import scripts.research_related as R
from graphrag.kg import semantic as S
from graphrag.kg.llm import KgCacheMiss, KgLLM
from graphrag.kg.schema import GraphEdge, Schema
from graphrag.types import Chunk, DocRecord, Meta, Node, Source

TEXT = ("Давлат ижтимоий суғуртаси бўйича нафақа тайинлаш тартиби ва уни тўлаш муддатлари ҳамда "
        "суғурта бадалларини ҳисоблаш қоидалари ваколатли орган томонидан белгиланади")   # > 12 content tokens


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _rec(doc_id, arts, texts=None, types_=None):
    """A record with one article per entry of `arts` (article number -> number of chunks)."""
    nodes, chunks, pos = [Node(doc_id, "документ", "T", None, None, 0, 10 ** 6)], [], 0
    for a, n in arts.items():
        nid = f"{doc_id}/СТ.{a}"
        nodes.append(Node(nid, (types_ or {}).get(a, "статья"), f"Модда {a}", str(a), doc_id, pos, pos + 1000))
        for p in range(1, n + 1):
            t = (texts or {}).get((a, p), TEXT)
            chunks.append(Chunk(f"{nid}#{p}", nid, p, 40, "", t, t, pos + p, pos + p + 1, t))
        pos += 1000
    return DocRecord(doc_id, Meta("law_text", None, "x", ["uz-Cyrl"], "", "QONUNLAR", "T"),
                     Source("s", "f", None), "", nodes, chunks)


def _vec(*pairs, dim=16):
    """A unit vector with the given (axis, weight) components."""
    v = np.zeros(dim, dtype=np.float32)
    for ax, w in pairs:
        v[ax] = w
    return _unit(v)


# --- boilerplate ------------------------------------------------------------------------------------

def test_boilerplate_reasons():
    assert S.boilerplate_reason(TEXT, "статья") is None
    assert S.boilerplate_reason("Ўзбекистон Республикаси Президенти Ш. Мирзиёев", "реквизиты") == "signature"
    assert S.boilerplate_reason(TEXT, "заголовок") == "header"
    assert S.boilerplate_reason("Қисқа матн фақат", "статья") == "short"
    gaz = "2024 йил, № 1, 5, 7, 8, 9-моддалар, № 2, 102, 111-моддалар, № 3, 243-модда; 2025 йил, № 1, 12-модда"
    assert S.boilerplate_reason(gaz, "часть") == "gazette"
    adopt = ("Лойиҳа Ўзбекистон Республикасининг ҚОНУНИ Масъулияти чекланган жамиятлар тўғрисида "
             "Қонунчилик палатаси томонидан 2025 йил 10 июнда қабул қилинган")
    assert S.boilerplate_reason(adopt, "преамбула") == "adoption"
    chair = ("Raislik qiluvchi: Rahmat. Qoʼshimcha fikrlar boʼlmasa, qaror loyihasini ovozga qoʼyaman. "
             "(ovoz beriladi: qarshilar, betaraflar yoʼq)")
    assert S.boilerplate_reason(chair, "выступление") == "formula"
    clause = ("4. Ушбу қарорнинг ижросини назорат қилиш Ўзбекистон Республикаси Олий Мажлиси Сенати Раисининг "
              "биринчи ўринбосари зиммасига юклансин.")
    assert S.boilerplate_reason(clause, "пункт") == "clause"


# --- same document / groups / explicit ------------------------------------------------------------------

def _two_docs():
    """doc A: articles 1, 2; doc B: article 1. A/1 ~ A/2 (same doc, cos 1), A/1 ~ B/1 (cos ~0.9)."""
    ra, rb = _rec("QONUNLAR-a", {1: 1, 2: 1}), _rec("QONUNLAR-b", {1: 1})
    vecs = {"QONUNLAR-a/СТ.1#1": _vec((0, 1.0)), "QONUNLAR-a/СТ.2#1": _vec((0, 1.0)),
            "QONUNLAR-b/СТ.1#1": _vec((0, 0.9), (1, 0.436))}
    return [ra, rb], vecs


def test_no_same_document_edges_and_undirected():
    recs, vecs = _two_docs()
    es = S.latent_edges(recs, vecs, 0.5, k_ann=5, k_rel=5)
    assert es, "cross-document pairs must give edges"
    for e in es:
        assert e.src < e.dst and e.type == "RELATED"
        assert e.src.split("/")[0] != e.dst.split("/")[0]
    assert {(e.src, e.dst) for e in es} == {("QONUNLAR-a/СТ.1", "QONUNLAR-b/СТ.1"),
                                            ("QONUNLAR-a/СТ.2", "QONUNLAR-b/СТ.1")}
    sch = Schema.load()
    from graphrag.kg.extract_ref import structure_nodes
    nodes = {n.node_id: n for r in recs for n in structure_nodes(r)}
    assert all(not sch.validate_edge(e, nodes) for e in es)


def test_translation_version_pairs_excluded():
    recs, vecs = _two_docs()
    es = S.latent_edges(recs, vecs, 0.5, k_ann=5, k_rel=5, linked_docs=[("QONUNLAR-b", "QONUNLAR-a")])
    assert es == []


def test_translation_group_is_masked_before_ranking():
    """k_ann = 1: the linked (translation) document holds the nearest chunk; masking it BEFORE ranking lets
    the third document's chunk take the slot (a post-hoc filter would leave no edge)."""
    ra, rb, rc = _rec("QONUNLAR-a", {1: 1}), _rec("QONUNLAR-b", {1: 1}), _rec("QONUNLAR-c", {1: 1})
    vecs = {"QONUNLAR-a/СТ.1#1": _vec((0, 1.0)), "QONUNLAR-b/СТ.1#1": _vec((0, 1.0), (1, 0.05)),
            "QONUNLAR-c/СТ.1#1": _vec((0, 1.0), (2, 0.4))}
    es = S.latent_edges([ra, rb, rc], vecs, 0.5, k_ann=1, k_rel=5, linked_docs=[("QONUNLAR-a", "QONUNLAR-b")])
    assert {(e.src, e.dst) for e in es} == {("QONUNLAR-a/СТ.1", "QONUNLAR-c/СТ.1"),
                                            ("QONUNLAR-b/СТ.1", "QONUNLAR-c/СТ.1")}


def test_explicit_edge_pairs_excluded():
    recs, vecs = _two_docs()
    ex = [GraphEdge("QONUNLAR-b/СТ.1", "QONUNLAR-a/СТ.1", "REFERS_TO", {})]      # reversed direction
    es = S.latent_edges(recs, vecs, 0.5, k_ann=5, k_rel=5, explicit_edges=ex)
    assert {(e.src, e.dst) for e in es} == {("QONUNLAR-a/СТ.2", "QONUNLAR-b/СТ.1")}
    # REGULATES / DEFINES are not "joined" pairs of document elements; only REFERS_TO / AMENDS / REPEALS
    assert S.explicit_pairs([GraphEdge("x", "y", "REGULATES", {})]) == set()
    assert S.explicit_pairs([GraphEdge("y", "x", "AMENDS", {}), GraphEdge("p", "q", "REPEALS", {})]) == {("x", "y"), ("p", "q")}


def test_contains_ancestry():
    par = {"d": None, "d/ГЛ.1": "d", "d/ГЛ.1/СТ.2": "d/ГЛ.1", "e/СТ.1": "e"}
    assert S.is_ancestor_pair("d", "d/ГЛ.1/СТ.2", par) and S.is_ancestor_pair("d/ГЛ.1/СТ.2", "d/ГЛ.1", par)
    assert not S.is_ancestor_pair("d/ГЛ.1", "e/СТ.1", par)


def test_boilerplate_chunks_take_no_part():
    ra = _rec("QONUNLAR-a", {1: 1}, texts={(1, 1): "Қисқа матн"})
    rb = _rec("QONUNLAR-b", {1: 1})
    vecs = {"QONUNLAR-a/СТ.1#1": _vec((0, 1.0)), "QONUNLAR-b/СТ.1#1": _vec((0, 1.0))}
    assert S.latent_edges([ra, rb], vecs, 0.5, k_ann=5, k_rel=5) == []


def test_weight_is_max_over_chunk_pairs():
    ra, rb = _rec("QONUNLAR-a", {1: 2}), _rec("QONUNLAR-b", {1: 1})
    vecs = {"QONUNLAR-a/СТ.1#1": _vec((0, 0.8), (1, 0.6)), "QONUNLAR-a/СТ.1#2": _vec((0, 0.95), (2, 0.312)),
            "QONUNLAR-b/СТ.1#1": _vec((0, 1.0))}
    es = S.latent_edges([ra, rb], vecs, 0.5, k_ann=5, k_rel=5)
    assert len(es) == 1 and es[0].attrs["weight"] == pytest.approx(0.95, abs=1e-3)


# --- k_rel cap / tie-break ----------------------------------------------------------------------------

def _star(n_leaves=8, weights=None):
    """Hub doc h (one article) and n_leaves single-article docs; leaf i has cos w_i with the hub."""
    weights = weights or [0.9 - 0.02 * i for i in range(n_leaves)]
    recs = [_rec("QONUNLAR-h", {1: 1})] + [_rec(f"QONUNLAR-l{i:02d}", {1: 1}) for i in range(n_leaves)]
    vecs = {"QONUNLAR-h/СТ.1#1": _vec((0, 1.0))}
    for i, w in enumerate(weights):
        vecs[f"QONUNLAR-l{i:02d}/СТ.1#1"] = _vec((0, w), (1 + i, float(np.sqrt(1 - w * w))))
    return recs, vecs


def test_k_rel_cap_keeps_top_of_either_end():
    recs, vecs = _star(8)
    best = {("A", f"B{i}"): (0.9 - 0.01 * i, 0, 0) for i in range(8)}
    kept = S.cap_degree(best, 3)
    # every leaf has only one candidate -> it is in the leaf's own top-3 -> all survive (hub degree 8 > k_rel)
    assert len(kept) == 8
    es = S.latent_edges(recs, vecs, 0.5, k_ann=10, k_rel=3)
    deg = S.degrees((e.src, e.dst) for e in es)
    assert deg["QONUNLAR-h/СТ.1"] == 8                      # hubness allowed by "top-k of one end"
    # when both ends rank an edge outside their top-k it is dropped
    best2 = {("A", "B"): (0.5, 0, 0), ("A", "C"): (0.9, 0, 0), ("A", "D"): (0.8, 0, 0),
             ("B", "C"): (0.95, 0, 0), ("B", "D"): (0.94, 0, 0)}
    kept2 = S.cap_degree(best2, 2)
    assert ("A", "B") not in kept2 and len(kept2) == 4
    for x, d in S.degrees(kept2).items():
        assert d <= 3


def test_k_rel_cap_every_vertex_top_k():
    rng = np.random.default_rng(0)
    best = {}
    vs = [f"v{i:02d}" for i in range(30)]
    for i in range(30):
        for j in range(i + 1, 30):
            if rng.random() < 0.5:
                best[(vs[i], vs[j])] = (round(float(rng.random()), 3), 0, 0)
    k = 4
    kept = S.cap_degree(best, k)
    inc = {}
    for (u, v), t in best.items():
        inc.setdefault(u, []).append((-t[0], v, (u, v)))
        inc.setdefault(v, []).append((-t[0], u, (u, v)))
    expect = {p for x in inc for _, _, p in sorted(inc[x])[:k]}
    assert set(kept) == expect


def test_deterministic_tie_break_by_id():
    # hub with 4 leaves at IDENTICAL cosine, k_rel = 2 and the hub the only candidate of nobody else:
    # each leaf keeps its only edge, so use a second hub to force the tie on the hub's side
    best = {("H", "L3"): (0.8, 0, 0), ("H", "L1"): (0.8, 0, 0), ("H", "L2"): (0.8, 0, 0), ("H", "L0"): (0.8, 0, 0),
            ("L0", "M"): (0.99, 0, 0), ("L1", "M"): (0.99, 0, 0), ("L2", "M"): (0.99, 0, 0), ("L3", "M"): (0.99, 0, 0)}
    kept = S.cap_degree(best, 1)
    # H keeps its tied top-1 by smallest other id (L0); each L keeps M; M's top-1 is L0 (id)
    assert ("H", "L0") in kept and ("H", "L1") not in kept and ("H", "L3") not in kept
    # ANN ranking ties broken by chunk id (row order), independent of input order
    recs, vecs = _star(6, weights=[0.8] * 6)
    e1 = S.latent_edges(recs, vecs, 0.5, k_ann=2, k_rel=5)
    e2 = S.latent_edges(list(reversed(recs)), dict(reversed(list(vecs.items()))), 0.5, k_ann=2, k_rel=5)
    assert [x.to_json() for x in e1] == [x.to_json() for x in e2]
    # the hub's two ANN slots go to the smallest chunk ids among the tied leaves
    ids = sorted(vecs)
    idx = S.build_index(ids, np.stack([vecs[i] for i in ids]), [i.split("#")[0] for i in ids],
                        [i.split("/")[0] for i in ids], k_ann=2)
    h = idx.chunk_ids.index("QONUNLAR-h/СТ.1#1")
    assert [idx.chunk_ids[j] for j in idx.ann.nbr[h]] == ["QONUNLAR-l00/СТ.1#1", "QONUNLAR-l01/СТ.1#1"]
    assert idx.ann.score[h][0] == idx.ann.score[h][1]


def test_blockwise_equals_single_block_and_rounding():
    rng = np.random.default_rng(1)
    n, dim = 70, 8
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    doc = np.repeat(np.arange(14), 5)
    ids = [f"d{d:02d}/c#{i}" for i, d in enumerate(doc)]
    a = S.build_index(ids, X, [f"d{d:02d}/v{i % 2}" for i, d in enumerate(doc)], [f"d{d:02d}" for d in doc], k_ann=4,
                      block=7)
    b = S.build_index(ids, X, [f"d{d:02d}/v{i % 2}" for i, d in enumerate(doc)], [f"d{d:02d}" for d in doc], k_ann=4,
                      block=1024)
    assert np.array_equal(a.ann.nbr, b.ann.nbr) and np.array_equal(a.ann.score, b.ann.score)
    assert np.all(np.round(a.ann.score[np.isfinite(a.ann.score)], 6) == a.ann.score[np.isfinite(a.ann.score)])
    for i in range(n):                                    # never a neighbour from the row's own document
        assert all(doc[j] != doc[i] for j in a.ann.nbr[i] if j >= 0)


def test_near_duplicate_documents_grouped_and_templates():
    # docs a and b are copies (2 chunks each), doc c is different: a-b edges vanish, a-c / b-c stay
    ra, rb, rc = _rec("QONUNLAR-a", {1: 1, 2: 1}), _rec("QONUNLAR-b", {1: 1, 2: 1}), _rec("QONUNLAR-c", {1: 1})
    vecs = {"QONUNLAR-a/СТ.1#1": _vec((0, 1.0)), "QONUNLAR-a/СТ.2#1": _vec((1, 1.0)),
            "QONUNLAR-b/СТ.1#1": _vec((0, 1.0)), "QONUNLAR-b/СТ.2#1": _vec((1, 1.0)),
            "QONUNLAR-c/СТ.1#1": _vec((0, 0.9), (2, 0.436))}
    es = S.latent_edges([ra, rb, rc], vecs, 0.5, k_ann=5, k_rel=5)
    docs = {tuple(sorted((e.src.split("/")[0], e.dst.split("/")[0]))) for e in es}
    assert ("QONUNLAR-a", "QONUNLAR-b") not in docs and ("QONUNLAR-a", "QONUNLAR-c") in docs
    # without dedup the copies link
    es0 = S.latent_edges([ra, rb, rc], vecs, 0.5, k_ann=5, k_rel=5, cfg={"dedup": None, "template": None})
    assert ("QONUNLAR-a", "QONUNLAR-b") in {tuple(sorted((e.src.split("/")[0], e.dst.split("/")[0]))) for e in es0}
    # a chunk repeated near-verbatim in >= min_docs other documents is a template: no edges at all
    recs = [_rec(f"QONUNLAR-t{i}", {1: 1}) for i in range(6)]
    tv = {f"QONUNLAR-t{i}/СТ.1#1": _vec((0, 1.0), (1 + i, 0.05)) for i in range(6)}
    assert S.latent_edges(recs, tv, 0.5, k_ann=10, k_rel=5, cfg={"dedup": None}) == []
    assert S.latent_edges(recs, tv, 0.5, k_ann=10, k_rel=5, cfg={"dedup": None, "template": None})


# --- calibration --------------------------------------------------------------------------------------

def test_calibration_picks_minimal_theta_and_null_floor():
    grid = S.theta_grid({"start": 0.60, "stop": 0.70, "step": 0.01})
    assert grid[0] == 0.6 and grid[-1] == 0.7 and len(grid) == 11
    edges_at = {t: int(1000 - (t - 0.60) * 10000) for t in grid}          # 1000 at 0.60 ... 0 at 0.70

    def ev(t):
        return edges_at[t], 3.0
    # share L / (E + L) <= 0.30 with E = 1000 -> L <= 428.57 -> 1000 - 10000 (t - .6) <= 428 -> t >= 0.6572
    cal = S.calibrate_theta(ev, 1000, grid, 0.30, 5)
    assert cal["theta_grid"] == 0.66 and cal["theta"] == 0.66 and cal["binding"] == "grid"
    assert [r["ok"] for r in cal["table"]][:7] == [False] * 6 + [True]
    # null floor above the grid choice wins, snapped UP to the grid
    cal2 = S.calibrate_theta(ev, 1000, grid, 0.30, 5, null_floor=0.6712)
    assert cal2["theta"] == 0.68 and cal2["theta_null"] == 0.68 and cal2["binding"] == "null"
    # null floor below: grid wins, both recorded
    cal3 = S.calibrate_theta(ev, 1000, grid, 0.30, 5, null_floor=0.61)
    assert cal3["theta"] == 0.66 and cal3["theta_null"] == 0.61
    # the median-degree condition also binds
    cal4 = S.calibrate_theta(lambda t: (0, 6.0 if t < 0.64 else 5.0), 1000, grid, 0.30, 5)
    assert cal4["theta_grid"] == 0.64


def test_null_quantile_deterministic_cross_group():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    group = np.repeat(np.arange(4), 10)
    elig = np.ones(40, dtype=bool)
    a = S.null_quantile(X, group, elig, 500, 0.999, 7)
    b = S.null_quantile(X, group, elig, 500, 0.999, 7)
    assert a == b and a["pairs"] == 500 and -1 <= a["value"] <= 1


def test_histograms_count_above_theta():
    grid = S.theta_grid({"start": 0.60, "stop": 0.65, "step": 0.01})
    bins = S.grid_bins(grid)
    h = np.histogram(np.array([0.6, 0.600001, 0.63, 0.65, 0.99]), bins=bins)[0]
    assert S.count_above(h, grid, 0.60) == 4 and S.count_above(h, grid, 0.63) == 2 and S.count_above(h, grid, 0.65) == 1


# --- judge --------------------------------------------------------------------------------------------

def _edges(n, lo=0.78, hi=0.99):
    return [GraphEdge(f"a{i:03d}", f"b{i:03d}", "RELATED", {"weight": round(lo + (hi - lo) * i / (n - 1), 6)})
            for i in range(n)]


def test_sample_strata_deterministic():
    es = _edges(400)
    s1 = R.sample_strata(es, 0.78, {}, 1)
    s2 = R.sample_strata(list(reversed(es)), 0.78, {}, 1)
    assert [e.src for e in s1["near"]] == [e.src for e in s2["near"]]
    assert all(e.attrs["weight"] <= 0.80 + 1e-9 for e in s1["near"]) and len(s1["near"]) == 30
    assert len(s1["top"]) == 30 and all(e.attrs["weight"] >= s1["top_min_weight"] for e in s1["top"])
    assert not {e.src for e in s1["near"]} & {e.src for e in s1["top"]}


class _FakeClient:
    def __init__(self, answer_fn):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self._fn = answer_fn

    def _create(self, **kw):
        self.calls += 1
        inp = json.loads(kw["messages"][1]["content"])
        content = json.dumps(self._fn(inp))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


def _cfg():
    return {"llm": {"model": "gpt-4.1", "max_calls_total": 50,
                    "passes": {"audit_related": {"prompt_version": 1, "max_calls": 3}}}}


def test_judge_cache_budget_and_offline(tmp_path):
    items = [{"uid": f"u{i}", "view": {"a": {"path": "p", "text": f"t{i}"}, "b": {"path": "q", "text": "s"}}}
             for i in range(12)]
    batches = R.make_batches(items, 6, 1)
    assert [len(b) for b in batches] == [6, 6] and batches == R.make_batches(list(reversed(items)), 6, 1)
    fake = _FakeClient(lambda inp: {"verdicts": [{"id": x["id"], "reason": "r", "same_topic": x["a"]["text"] != "t0"}
                                                 for x in inp["items"]]})
    llm = KgLLM(_cfg(), cache_root=tmp_path, client=fake, allow_api=True)
    j = R.llm_judge(llm)
    v = {}
    for b in batches:
        v.update(j(b))
    assert fake.calls == 2 and len(v) == 12 and v["u0"]["same_topic"] is False and v["u1"]["same_topic"] is True
    assert R.precision(list(v.values()))["k"] == 11
    # offline re-run: cache hits, 0 calls; a new batch misses loudly
    off = KgLLM(_cfg(), cache_root=tmp_path, client=None, allow_api=False)
    j2 = R.llm_judge(off)
    v2 = {}
    for b in batches:
        v2.update(j2(b))
    assert v2 == v and off.stats["audit_related"]["api_calls"] == 0
    with pytest.raises(KgCacheMiss):
        j2(batches[0][:3])
    # a malformed answer (missing id) is rejected -> None verdicts
    bad = _FakeClient(lambda inp: {"verdicts": [{"id": "e1", "reason": "r", "same_topic": True}]})
    llm3 = KgLLM(_cfg(), cache_root=tmp_path / "b", client=bad, allow_api=True)
    assert all(x is None for x in R.llm_judge(llm3)(batches[0]).values())


# --- Ruling R22: sentence boilerplate classes, topic text, anisotropy, mutual kNN, dev calibration ------------

R22_RULES = {
    "implementation": {"patterns": ["ijrosini,? (?:ijrochilarga )?y?etkazilishini"]},
    "chair_procedure": {"sections": ["STENOGRAMMA"], "patterns": ["qaror qabul qilindi", "navbatdagi masala",
                                                                  "^masʼul ?[–-]", "soʼz [^.!?]{0,120}beriladi"]},
    "greeting": {"sections": ["STENOGRAMMA"], "patterns": ["assalomu? alaykum"]},
}
IMPL = ("2-modda. oʼzbekiston respublikasi adliya vazirligi va boshqa manfaatdor tashkilotlar ushbu qonunning ijrosini, "
        "ijrochilarga etkazilishini hamda mohiyati va ahamiyati aholi oʼrtasida tushuntirilishini taʼminlasin.")
AGENDA = ("qaror qabul qilindi. rahmat.\n\nnavbatdagi masala “oʼzbekiston respublikasining soliq kodeksiga oʼzgartirish "
          "kiritish hamda tadbirkorlik subyektlari faoliyatini qoʼllab-quvvatlash tizimini takomillashtirish, soliq "
          "maʼmuriyatchiligini soddalashtirish va raqamli nazorat mexanizmlarini joriy etish toʼgʼrisida”gi qonun haqida.\n\nmasʼul – budjet va iqtisodiy islohotlar masalalari qoʼmitasi.\n\n"
          "soʼz qoʼmita raisi erkin fayziyevich gadoyevga beriladi.")
SPEECH = ("assalomu alaykum hurmatli senatorlar!\n\nmahalliy budjet mablagʼlarini mustaqil boshqarish, budjet ijrosining "
          "markazga bogʼliqligini kamaytirish, hududlarni ijtimoiy-iqtisodiy rivojlantirish dasturlarini kafolatlangan "
          "moliyalashtirish manbalari bilan taʼminlash boʼyicha keng koʼlamli ishlar amalga oshirilmoqda va natijalar bor.")


def test_sentence_boilerplate_classes():
    rules = S.compile_sentence_rules(R22_RULES)
    assert [r.name for r in rules] == ["implementation", "chair_procedure", "greeting"]
    assert S.sentence_boilerplate_reason(IMPL, "QONUNLAR", rules, 20) == "implementation"
    # the chair agenda introduction is boilerplate in a stenogram ...
    assert S.sentence_boilerplate_reason(AGENDA, "STENOGRAMMA", rules, 20) == "chair_procedure"
    # ... its quoted act title is kept apart (residual + titles), not in the core residual
    with_titles, core, cls = S.strip_boilerplate(AGENDA, rules, "STENOGRAMMA")
    assert "soliq kodeksiga" in with_titles and "soliq" not in core and cls["chair_procedure"] == 4
    # section-restricted classes do not apply elsewhere
    assert S.sentence_boilerplate_reason(AGENDA, "QONUNLAR", rules, 20) is None
    # a greeting before a substantive speech: enough residual content -> not boilerplate
    assert S.sentence_boilerplate_reason(SPEECH, "STENOGRAMMA", rules, 20) is None
    assert S.sentence_boilerplate_reason(SPEECH, "STENOGRAMMA", rules, 200) == "greeting"
    # no class matched -> never boilerplate, however short; no rules -> None
    assert S.sentence_boilerplate_reason("qisqa matn", "STENOGRAMMA", rules, 20) is None
    assert S.sentence_boilerplate_reason(IMPL, "QONUNLAR", [], 20) is None
    # speaker-name words do not count as residual content
    assert S.residual_tokens("erkin fayziyevich gadoyev budjet", {"erkin", "fayziyevich", "gadoyev"}) == 1


def test_topic_text_drops_session_and_speaker_crumbs():
    sp = {"raislik qiluvchi", "matluba maxamajanovna gʼaniyeva"}
    norm = "matluba maxamajanovna gʼaniyeva: assalomu alaykum!\n\ntadbirkorlik subyektlari faoliyati"
    idx = ("senatning 35-yalpi majlisi (2022-yil 18 – 19-noyabr) › ayrim qonun hujjatlariga oʼzgartish kiritish toʼgʼrisida"
           " › matluba maxamajanovna gʼaniyeva\n" + norm)
    t = S.topic_text(idx, norm, sp, [r"^senatning \d+-yalpi majlisi"])
    assert t == "ayrim qonun hujjatlariga oʼzgartish kiritish toʼgʼrisida\nassalomu alaykum!\n\ntadbirkorlik subyektlari faoliyati"
    # a multi-speaker crumb is dropped only when every name is a speaker; laws keep their whole path
    assert S.topic_text("t › raislik qiluvchi, matluba maxamajanovna gʼaniyeva\nx", "x", sp) == "t\nx"
    assert S.topic_text("t › raislik qiluvchi, boshqa odam\nx", "x", sp) == "t › raislik qiluvchi, boshqa odam\nx"
    assert S.topic_text("qonun › 3-modda.\n3-modda. matn", "3-modda. matn") == "qonun › 3-modda.\n3-modda. matn"
    assert S.topic_text("matn", "matn") == "matn"
    # content_text removes only labels of known speakers
    assert S.content_text("raislik qiluvchi: rahmat.\nboshqa: gap", sp) == "rahmat.\nboshqa: gap"


def test_anisotropy_correct_all_but_the_top():
    rng = np.random.default_rng(1)
    common = np.zeros(32)
    common[0] = 3.0
    X = rng.normal(size=(200, 32)) * np.linspace(3, 0.5, 32) + common     # shared mean + one dominant direction
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    el = np.ones(200, dtype=bool)
    el[:10] = False
    Y, info = S.anisotropy_correct(X, el, 2)
    assert Y.dtype == np.float32 and np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-6)
    assert info["pcs"] == 2 and len(info["explained_variance_ratio"]) == 2
    # the eligible rows' top-2 principal directions are projected out (of every row)
    E = X[el].astype(np.float64)
    Ec = E - E.mean(0)
    w, V = np.linalg.eigh(Ec.T @ Ec)
    P = V[:, np.argsort(-w)[:2]]
    assert np.abs(Y.astype(np.float64) @ P).max() < 1e-5
    # centred: the mean pairwise cosine drops towards 0
    assert abs(float((Y[el] @ Y[el].T).mean())) < float((X[el] @ X[el].T).mean())
    Y2, _ = S.anisotropy_correct(X, el, 2)
    assert Y.tobytes() == Y2.tobytes()                                  # deterministic
    Y0, i0 = S.anisotropy_correct(X, el, 0)
    assert i0["explained_variance_ratio"] == [] and np.allclose(np.linalg.norm(Y0, axis=1), 1.0, atol=1e-6)


def test_cap_degree_mutual_knn():
    best = {("A", "B"): (0.5, 0, 0), ("A", "C"): (0.9, 0, 0), ("A", "D"): (0.8, 0, 0),
            ("B", "C"): (0.95, 0, 0), ("B", "D"): (0.94, 0, 0)}
    kept = S.cap_degree(best, 2, mutual=True)
    # A top2 {C, D}; B top2 {C, D}; C top2 {B, A}; D top2 {B, A}
    assert set(kept) == {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
    star = {("H", f"L{i}"): (0.9 - 0.01 * i, 0, 0) for i in range(8)}
    km = S.cap_degree(star, 3, mutual=True)
    assert set(km) == {("H", "L0"), ("H", "L1"), ("H", "L2")}               # no hub beyond k_rel
    assert len(S.cap_degree(star, 3)) == 8                                  # either-end keeps the hub
    rng = np.random.default_rng(0)
    vs = [f"v{i:02d}" for i in range(30)]
    rnd = {(vs[i], vs[j]): (round(float(rng.random()), 3), 0, 0) for i in range(30) for j in range(i + 1, 30)
           if rng.random() < 0.5}
    m = S.cap_degree(rnd, 4, mutual=True)
    assert max(S.degrees(m).values()) <= 4 and set(m) <= set(S.cap_degree(rnd, 4))
    recs, vecs = _star(8)
    es = S.latent_edges(recs, vecs, 0.5, k_ann=10, k_rel=3, cfg={"mutual_knn": True})
    assert S.degrees((e.src, e.dst) for e in es)["QONUNLAR-h/СТ.1"] == 3


def _steno(doc_id, texts):
    r = _rec(doc_id, {i + 1: 1 for i in range(len(texts))}, texts={(i + 1, 1): t for i, t in enumerate(texts)})
    r.meta.section = "STENOGRAMMA"
    return r


def test_latent_edges_r22_rules_and_score_vectors():
    ra, rb = _steno("STENOGRAMMA-a", [AGENDA, SPEECH]), _steno("STENOGRAMMA-b", [AGENDA, SPEECH])
    vecs = {"STENOGRAMMA-a/СТ.1#1": _vec((0, 1.0)), "STENOGRAMMA-b/СТ.1#1": _vec((0, 1.0)),
            "STENOGRAMMA-a/СТ.2#1": _vec((1, 1.0)), "STENOGRAMMA-b/СТ.2#1": _vec((1, 0.8), (2, 0.6))}
    cfg = {"dedup": None, "template": None}
    base = {(e.src, e.dst) for e in S.latent_edges([ra, rb], vecs, 0.5, 10, 5, cfg=cfg)}
    assert ("STENOGRAMMA-a/СТ.1", "STENOGRAMMA-b/СТ.1") in base                    # agenda formulas match (cos 1)
    r22 = {**cfg, "sentence_rules": R22_RULES, "residual_min_tokens": 20}
    es = S.latent_edges([ra, rb], vecs, 0.5, 10, 5, cfg=r22)
    assert {(e.src, e.dst) for e in es} == {("STENOGRAMMA-a/СТ.2", "STENOGRAMMA-b/СТ.2")}   # boilerplate left the ANN
    assert es[0].attrs["weight"] == pytest.approx(0.8, abs=1e-6)
    # score vectors replace the vectors for scoring
    sv = dict(vecs)
    sv["STENOGRAMMA-b/СТ.2#1"] = _vec((1, 0.6), (2, 0.8))
    es2 = S.latent_edges([ra, rb], vecs, 0.5, 10, 5, cfg=r22, score_vectors=sv)
    assert es2[0].attrs["weight"] == pytest.approx(0.6, abs=1e-6)
    assert S.latent_edges([ra, rb], vecs, 0.7, 10, 5, cfg=r22, score_vectors=sv) == []


def test_auc_logistic_and_dev_theta():
    assert R.auc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert R.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert R.auc([0.5, 0.5], [1, 0]) == 0.5 and R.auc([1.0], [1]) is None
    assert R.auc([0.9, -np.inf, 0.3], [1, 1, 0]) == 0.5                      # -inf (dropped) ranks last
    ci = R.bootstrap_auc([0.9, 0.8, 0.7, 0.3, 0.2, 0.6], [1, 1, 1, 0, 0, 0], 200, 1)
    assert ci == R.bootstrap_auc([0.9, 0.8, 0.7, 0.3, 0.2, 0.6], [1, 1, 1, 0, 0, 0], 200, 1) and ci[0] <= ci[1] <= 1
    rng = np.random.default_rng(0)
    s = rng.random(400)
    y = rng.random(400) < s                                                  # P(y | s) = s
    w = R.logistic_fit(s, y)
    assert w[1] > 0 and R.logistic_p(w, [0.2])[0] < 0.5 < R.logistic_p(w, [0.8])[0]
    assert R.logistic_fit(s, y) == w                                         # deterministic
    grid = [0.1 * i for i in range(10)]
    near = lambda t: np.array([t + 0.01, t + 0.02])
    d = R.dev_theta(near, w, grid, 0.8, lo=0.3)
    th = d["theta_dev"]
    assert th is not None and th >= 0.3
    est = {r["theta"]: r["est_near_precision"] for r in d["table"]}
    assert est[th] >= 0.8 and all(v < 0.8 for t, v in est.items() if t < th)
    assert R.dev_theta(near, w, grid, 0.9999)["theta_dev"] is None


def test_sample_strata_excludes_dev_pairs():
    es = _edges(400)
    dev = {(e.src, e.dst) for e in es[:5]} | {(e.src, e.dst) for e in es[-10:]}
    s = R.sample_strata(es, 0.78, {}, 1, tag="r22r0", exclude=dev)
    got = {(e.src, e.dst) for e in s["near"] + s["top"]}
    assert not got & dev and len(s["near"]) == 30 and len(s["top"]) == 20        # top pool: top 30 minus 10 dev
    assert s["excluded_dev_pairs"] == 15
    assert all(e.attrs["weight"] <= 0.80 + 1e-9 for e in s["near"])
    s2 = R.sample_strata(list(reversed(es)), 0.78, {}, 1, tag="r22r0", exclude=dev)
    assert [e.src for e in s["near"]] == [e.src for e in s2["near"]]
    # the top stratum is the top of ALL edges (dev pairs keep their rank), minus the dev pairs
    top_all = sorted(es, key=lambda e: -e.attrs["weight"])[:30]
    assert s["top_min_weight"] >= min(e.attrs["weight"] for e in top_all) - 1e-9


# --- Ruling R23: the five missed boilerplate classes (config data), class ablation, θ_a / θ_b adoption ---

def _r23_rules():
    from graphrag import config
    rel = config.load("graph")["related"]
    assert rel["r23_classes"] == ["chair_agenda", "vote_commission", "preamble_basis", "report_assignment",
                                  "dispute_resolution"]
    return S.compile_sentence_rules(rel["sentence_rules"])


R23_POS = {
    "chair_agenda": ("reglament: maʼruzalarga 10 daqiqagacha, muzokara soʼzlari va savol-javoblar uchun 5 daqiqadan vaqt "
                     "beriladi. ushbu reglamentga amal qilishingizni soʼrayman.\n\nkun tartibi loyihasi sizlarga tarqatilgan. "
                     "takliflar bormi?", "STENOGRAMMA"),
    "vote_commission": ("barchangizdan muhokamalarda faol boʼlishni soʼrab qolaman.\n\nyalpi majlis sanoq komissiyasi tarkibi "
                        "boʼyicha qaror loyihasi sizlarga tarqatilgan.\n\nanvar toʼychiyev – komissiya rahbari;", "STENOGRAMMA"),
    "preamble_basis": ("yuqoridagilardan kelib chiqib, “oʼzbekiston respublikasi oliy majlisining senati toʼgʼrisida”gi "
                       "oʼzbekiston respublikasining konstitutsiyaviy qonunining 9-moddasi va “parlament nazorati toʼgʼrisida”gi "
                       "oʼzbekiston respublikasi qonunining 14-moddasiga muvofiq oʼzbekiston respublikasi oliy majlisining senati"
                       "\n\nq a r o r q i l a d i :", "QARORLAR"),
    "report_assignment": ("3. ushbu masala boʼyicha oʼzbekiston respublikasi oliy majlisi senatining navbatdagi yalpi majlisida "
                          "maʼruza bilan soʼzga chiqish qoʼmita raisi oʼrinbosari zimmasiga yuklatilsin.", "XULOSALAR"),
    "dispute_resolution": ("20-modda. nizolarni hal etish\n\ndavlat ijtimoiy sugʼurtasi sohasidagi nizolar qonunchilikda "
                           "belgilangan tartibda hal etiladi.", "QONUNLAR"),
}


def test_r23_classes_match_their_formulas():
    rules = _r23_rules()
    for name, (text, sec) in R23_POS.items():
        _, _, cls = S.strip_boilerplate(text, rules, sec)
        assert name in cls, name
        assert S.sentence_boilerplate_reason(text, sec, rules, 20) is not None, name
    # the Russian variants
    _, _, cls = S.strip_boilerplate("статья 76. разрешение споров\n\nспоры, связанные с ликвидацией банков, разрешаются в "
                                    "порядке, установленном законодательством.", rules, "QONUNLAR")
    assert cls["dispute_resolution"] == 2
    # stenogram-only classes do not apply to other sections
    _, _, cls = S.strip_boilerplate(R23_POS["chair_agenda"][0], rules, "QONUNLAR")
    assert "chair_agenda" not in cls


def test_r23_classes_keep_topical_sentences():
    rules = _r23_rules()
    # a preamble naming the considered act is topical: the "… qonunini koʼrib chiqib, … muvofiq … senati" sentence
    # is not the competence-basis formula
    considered = ("oʼzbekiston respublikasi oliy majlisining qonunchilik palatasi tomonidan 2025-03-18 kuni qabul qilingan "
                  "“issiqxona gazlarining chiqarilishini cheklash toʼgʼrisida”gi oʼzbekiston respublikasi qonunini koʼrib "
                  "chiqib, oʼzbekiston respublikasi konstitutsiyasining 93-moddasiga muvofiq oʼzbekiston respublikasi oliy "
                  "majlisining senati")
    _, core, cls = S.strip_boilerplate(considered, rules, "QARORLAR")
    assert "preamble_basis" not in cls and "issiqxona" in core
    # "dispute resolution" inside a substantive sentence of a conclusion is not the generic final article
    concl = ("bitimda fuqaro muhofazasi mexanizmining maqsadi, vazifalari va qoʼllanilishi belgilangan hamda "
             "moliyalashtirish, nizolarni hal qilish masalalariga aniqlik kiritilgan.")
    assert "dispute_resolution" not in S.strip_boilerplate(concl, rules, "XULOSALAR")[2]
    # a closing speech thanking those active in the debate is not the vote-counting commission formula
    closing = "muhokamalarda faol boʼlgan barcha hamkasblarimizga, mehmonlarimizga rahmat."
    assert "vote_commission" not in S.strip_boilerplate(closing, rules, "STENOGRAMMA")[2]
    # a speech mentioning time ("… daqiqa …") with enough residual content stays eligible
    assert S.sentence_boilerplate_reason(R23_POS["chair_agenda"][0] + "\n\n" + SPEECH, "STENOGRAMMA", rules, 20) is None


def test_class_removals_ablation():
    spec = {"a": {"patterns": ["^alfa"]}, "new1": {"patterns": ["^beta"]}, "new2": {"patterns": ["^gamma"]}}
    texts = {"c1": "alfa", "c2": "beta", "c3": "beta", "c4": "gamma", "c5": "delta"}

    def reason_fn(sp):
        rs = S.compile_sentence_rules(sp)
        return {i: S.sentence_boilerplate_reason(t, None, rs, 1) for i, t in texts.items()}

    out = R.class_removals(reason_fn, spec, ["new1", "new2"])
    assert out == {"new1": {"removed": 2, "examples": ["c2", "c3"]}, "new2": {"removed": 1, "examples": ["c4"]}}


def test_adopt_theta_schedule():
    r = lambda t, p: {"theta": t, "near": {"precision": p}}
    assert R.adopt_theta([r(0.58, 0.81)], 0.80) == (0.58, "passed_theta_a")
    assert R.adopt_theta([r(0.58, 0.70), r(0.65, 0.833)], 0.80) == (0.65, "passed_theta_b")
    assert R.adopt_theta([r(0.58, 0.75), r(0.65, 0.70)], 0.80) == (0.58, "failed_best_judged")
    assert R.adopt_theta([r(0.58, 0.70), r(0.65, 0.70)], 0.80) == (0.65, "failed_best_judged")   # tie -> higher θ
