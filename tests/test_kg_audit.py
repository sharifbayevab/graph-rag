# tests/test_kg_audit.py -- M2 Task 7: edge-precision audit (sampling, planting, metrics, judge cache)
import json
import types

import pytest

from graphrag.kg import audit as A
from graphrag.kg.llm import KgLLM, KgCacheMiss
from graphrag.kg.schema import GraphEdge, GraphNode
from graphrag.types import Node

CFG = dict(A.DEFAULTS)


def _doc(doc_id: str, n_art: int = 12):
    """A synthetic law: n_art articles, each a long paragraph naming the next article and an organ."""
    paras, nodes, pos = [], [], 0
    for i in range(1, n_art + 1):
        p = (f"{i}-модда. Модда {i}\n\nВазирлик{i} ушбу Қонуннинг {i % n_art + 1}-моддасига мувофиқ иш юритади. "
             + "Матн сўзлари " * 25 + "\n\n")
        nodes.append(Node(f"{doc_id}/СТ.{i}", "статья", f"Модда {i}", str(i), doc_id, pos, pos + len(p) - 2))
        paras.append(p)
        pos += len(p)
    text = "".join(paras)
    nodes.insert(0, Node(doc_id, "документ", "Синов қонуни", None, None, 0, len(text)))
    return text, nodes


def _graph(n_docs: int = 3, n_art: int = 12):
    texts, titles, meta, struct, gnodes, edges = {}, {}, {}, {}, {}, []
    for d in range(n_docs):
        doc = f"QONUNLAR-{d:08x}"
        text, ns = _doc(doc, n_art)
        texts[doc], titles[doc], meta[doc] = text, f"Синов қонуни {d}", {"type": "law_text", "date": "2024-01-01"}
        for n in ns:
            struct[n.node_id] = n
            gnodes[n.node_id] = GraphNode(n.node_id, "document" if n.parent_id is None else "element", n.type,
                                          n.title, doc, False, {})
        for i in range(1, n_art + 1):
            n = struct[f"{doc}/СТ.{i}"]
            j = i % n_art + 1
            q = f"{j}-моддасига"
            a = text.index(q, n.start)
            edges.append(GraphEdge(n.node_id, f"{doc}/СТ.{j}", "REFERS_TO",
                                   {"evidence": {"doc_id": doc, "start": a, "end": a + len(q), "quote": q},
                                    "relation": "internal"}))
            q2 = f"Вазирлик{i}"
            b = text.index(q2, n.start)
            eid = f"ENT/organ/vazirlik{i}"
            gnodes[eid] = GraphNode(eid, "entity", "organ", q2, None, False, {"aliases": [q2]})
            edges.append(GraphEdge(n.node_id, eid, "REGULATES",
                                   {"evidence": {"doc_id": doc, "start": b, "end": b + len(q2), "quote": q2}}))
    g = A.AuditGraph(texts, titles, meta, struct, gnodes, edges)
    return g, edges


def test_strata_of_every_audited_class():
    ev = {"doc_id": "D", "start": 0, "end": 1, "quote": "x"}
    mk = lambda t, dst="D/СТ.1", **a: GraphEdge("D/СТ.2", dst, t, {"evidence": ev, **a})  # noqa: E731
    assert A.stratum_of(mk("REFERS_TO", relation="element")) == "REFERS_TO:element"
    assert A.stratum_of(mk("REFERS_TO", relation="translation_of")) == "REFERS_TO:version"
    assert A.stratum_of(mk("REFERS_TO", relation="version_of")) == "REFERS_TO:version"
    assert A.stratum_of(mk("AMENDS", op="renumber")) == "AMENDS:renumber"
    assert A.stratum_of(mk("REGULATES", "ENT/organ/x")) == "REGULATES:entity"
    assert A.stratum_of(mk("REGULATES", "CON/x")) == "REGULATES:concept"
    assert A.stratum_of(mk("REPEALS", date="2024-01-01")) == "REPEALS"
    assert A.stratum_of(mk("DEFINES", "CON/x")) == "DEFINES"
    assert A.stratum_of(GraphEdge("D", "D/СТ.1", "CONTAINS", {})) is None


def test_sample_sizes_major_full_and_small():
    sizes = A.sample_sizes({"REFERS_TO:act": 3000, "REPEALS": 83, "REFERS_TO:version": 2, "DEFINES": 100,
                            "AMENDS:add": 35}, major=40, full=("REPEALS",))
    assert sizes == {"AMENDS:add": 35, "DEFINES": 40, "REFERS_TO:act": 40, "REFERS_TO:version": 2, "REPEALS": 83}
    assert A.n_planted(40, 0.125) == 5 and A.n_planted(83, 0.125) == 11 and A.n_planted(2, 0.125) == 1
    assert A.n_planted(40, 0.125) / 40 >= 0.10


def test_sampling_is_deterministic_and_seed_dependent():
    g, edges = _graph()
    pops = A.populations(edges)
    sizes = {s: 10 for s in pops}
    s1 = A.draw_sample(pops, sizes, 7)
    s2 = A.draw_sample({s: list(reversed(v)) for s, v in pops.items()}, sizes, 7)   # input order irrelevant
    assert {s: [A.edge_sort_key(e) for e in v] for s, v in s1.items()} == {s: [A.edge_sort_key(e) for e in v] for s, v in s2.items()}
    s3 = A.draw_sample(pops, sizes, 8)
    assert any([A.edge_sort_key(e) for e in s1[s]] != [A.edge_sort_key(e) for e in s3[s]] for s in s1)
    assert all(len(v) == 10 for v in s1.values())


def test_planted_edges_are_wrong_new_and_unmarked():
    g, edges = _graph()
    pops = A.populations(edges)
    sample = A.draw_sample(pops, {s: 10 for s in pops}, 7)
    for s in pops:
        pl = A.plant_stratum(g, s, pops[s], sample[s], 4, 7, CFG)
        assert len(pl) == 4
        assert pl == A.plant_stratum(g, s, pops[s], sample[s], 4, 7, CFG)          # deterministic
        sample_keys = {A.edge_sort_key(e) for e in sample[s]}
        for p in pl:
            e, base = p["edge"], p["base"]
            assert A.edge_sort_key(base) not in sample_keys
            assert (e.src, e.type, e.dst) not in g.existing                          # never a real edge
            assert p["method"] in A.METHODS[e.type]
            if p["method"] == "shift_evidence":
                ev = e.attrs["evidence"]
                assert g.texts[ev["doc_id"]][ev["start"]:ev["end"]] == ev["quote"]
                assert abs(ev["start"] - base.attrs["evidence"]["start"]) >= 1500
            else:
                assert e.dst != base.dst
            view = A.render_item(g, e, CFG)
            assert "planted" not in json.dumps(view) and "method" not in view
        assert len({p["method"] for p in pl}) >= 2                                    # a mix of methods


def test_op_swap_changes_only_the_op():
    g, edges = _graph()
    e = edges[0]
    am = GraphEdge(e.src, e.dst, "AMENDS", {"evidence": e.attrs["evidence"], "op": "edit", "ops": ["edit", "add"]})
    pe = A._op_swap(g, am, None, None, None)
    assert pe.attrs["op"] == "delete" and "ops" not in pe.attrs and (pe.src, pe.dst) == (am.src, am.dst)


def test_render_item_marks_evidence_and_describes_target():
    g, edges = _graph()
    e = next(x for x in edges if x.type == "REGULATES")
    v = A.render_item(g, e, CFG)
    q = e.attrs["evidence"]["quote"]
    assert f"⟦{q}⟧" in v["context"] and v["evidence"] == q
    assert v["target"] == {"kind": "entity", "entity_type": "organ", "name": q, "aliases": [q]}
    assert v["source"]["node_path"] == ["статья 1: Модда 1"]
    r = next(x for x in edges if x.type == "REFERS_TO")
    t = A.render_item(g, r, CFG)["target"]
    assert t["kind"] == "corpus element" and t["path"] == ["статья 2: Модда 2"] and t["text"].startswith("2-модда")


def test_render_target_ext_act_and_stub():
    g, _ = _graph(1)
    g.nodes["EXT/jinoyat kodeks"] = GraphNode("EXT/jinoyat kodeks", "document", "документ", "Жиноят кодекси",
                                              None, True, {"numbers": ["2012-xii"]})
    g.nodes["EXT/jinoyat kodeks/СТ.27"] = GraphNode("EXT/jinoyat kodeks/СТ.27", "element", "статья", "27-модда",
                                                    None, True, {})
    g.nodes["EXT/jinoyat kodeks/СТ.27/Ч.2"] = GraphNode("EXT/jinoyat kodeks/СТ.27/Ч.2", "element", "часть",
                                                        "2-қисм", None, True, {})
    assert A.render_target(g, "EXT/jinoyat kodeks", CFG) == {
        "kind": "external act (not in the corpus)", "act_title": "Жиноят кодекси", "numbers": ["2012-xii"]}
    t = A.render_target(g, "EXT/jinoyat kodeks/СТ.27/Ч.2", CFG)
    assert t["element"] == "27-модда / 2-қисм" and t["act_title"] == "Жиноят кодекси"


def test_wilson_known_values():
    lo, hi = A.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = A.wilson(9, 10)
    assert (lo, hi) == (pytest.approx(0.5958, abs=1e-3), pytest.approx(0.9821, abs=1e-3))
    lo, hi = A.wilson(40, 40)
    assert lo == pytest.approx(0.9124, abs=1e-3) and hi == 1.0
    assert A.wilson(0, 0) == (0.0, 1.0)


def test_corrected_precision_rogan_gladen():
    assert A.corrected(0.8, 1.0, 1.0) == pytest.approx(0.8)
    assert A.corrected(0.9, 0.9, 0.95) == pytest.approx(0.8 / 0.85)
    assert A.corrected(0.05, 0.9, 0.95) == 0.0                     # clipped
    assert A.corrected(0.9, 0.5, 0.5) is None                      # an uninformative judge
    assert A.corrected(None, 0.9, 0.9) is None


def test_pooled_uses_population_weights():
    p = A.pooled({"a": {"N": 900, "n": 40, "k": 40}, "b": {"N": 100, "n": 40, "k": 20}})
    assert p["precision"] == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    assert p["N"] == 1000 and p["ci"][0] < p["precision"] < p["ci"][1]
    one = A.pooled({"a": {"N": 50, "n": 40, "k": 36}})
    assert one["precision"] == pytest.approx(0.9)


def test_evaluate_with_fake_judge_sensitivity_agreement_and_gate():
    items = [{"uid": f"g{i}", "stratum": "REPEALS", "planted": False} for i in range(20)] + \
            [{"uid": f"p{i}", "stratum": "REPEALS", "planted": True} for i in range(4)] + \
            [{"uid": f"h{i}", "stratum": "DEFINES", "planted": False} for i in range(10)]
    verdicts = {it["uid"]: {"correct": not it["planted"], "reason": "", "confidence": 1.0} for it in items}
    verdicts["g0"]["correct"] = False                              # one genuine edge rejected
    verdicts["p0"]["correct"] = True                               # one planted edge missed
    manual = {"g0": {"correct": True}, "g1": {"correct": True}, "h0": {"correct": True}, "h1": {"correct": False}}
    res = A.evaluate(items, verdicts, {"REPEALS": 83, "DEFINES": 100}, manual)
    j = res["judge"]
    assert j["sensitivity_planted"] == pytest.approx(0.75) and j["planted"] == 4
    assert j["manual_n"] == 4 and j["manual_agreement"] == pytest.approx(0.5)   # g0 and h1 disagree
    assert j["specificity_manual"] == pytest.approx(2 / 3)
    r = res["strata"]["REPEALS"]
    assert (r["n"], r["k"], r["planted"], r["planted_rejected"]) == (20, 19, 4, 3)
    assert r["precision"] == pytest.approx(0.95) and not r["below_gate"]
    assert res["types"]["DEFINES"]["precision"] == 1.0 and res["types"]["DEFINES"]["gate_raw"]


def test_make_batches_deterministic_per_stratum_ids_hide_planting():
    items = [{"uid": f"{s}{i}", "stratum": s, "planted": i % 5 == 0, "view": {"edge_type": "X", "n": i}}
             for s in ("A", "B") for i in range(13)]
    b1 = A.make_batches(items, 6, 1)
    b2 = A.make_batches(list(reversed(items)), 6, 1)
    assert [[x["uid"] for x in b] for b in b1] == [[x["uid"] for x in b] for b in b2]
    assert [len(b) for b in b1] == [6, 6, 1, 6, 6, 1]
    assert all(len({x["stratum"] for x in b}) == 1 for b in b1)
    inp = A.batch_input(b1[0])
    assert [x["id"] for x in inp["items"]] == [f"e{i}" for i in range(1, 7)]
    assert "planted" not in json.dumps(inp) and "uid" not in json.dumps(inp)


class _FakeClient:
    """OpenAI-shaped fake: rejects every item whose view carries bad=True."""

    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        items = json.loads(kw["messages"][1]["content"])["items"]
        ans = {"verdicts": [{"id": x["id"], "reason": "r", "correct": not x.get("bad"), "confidence": 1.7}
                            for x in items]}
        msg = types.SimpleNamespace(content=json.dumps(ans))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _llm_cfg(max_calls=5):
    return {"llm": {"model": "gpt-4.1", "max_calls_total": 10, "passes": {"audit": {"prompt_version": 1, "max_calls": max_calls}}}}


def test_llm_judge_caches_and_reruns_offline(tmp_path):
    items = [{"uid": f"u{i}", "stratum": "S", "planted": i == 2, "view": {"edge_type": "X", "k": i, "bad": i == 2}}
             for i in range(5)]
    batches = A.make_batches(items, 3, 1)
    fake = _FakeClient()
    live = A.llm_judge(KgLLM(_llm_cfg(), cache_root=tmp_path, client=fake, allow_api=True))
    v1 = {}
    for b in batches:
        v1.update(live(b))
    assert fake.calls == 2 and not v1["u2"]["correct"] and v1["u0"]["correct"]
    assert all(v["confidence"] == 1.0 for v in v1.values())       # clamped to [0, 1]
    off = KgLLM(_llm_cfg(), cache_root=tmp_path, allow_api=False)
    judge = A.llm_judge(off)
    v2 = {}
    for b in batches:
        v2.update(judge(b))
    assert v2 == v1 and off.stats["audit"]["api_calls"] == 0 and off.stats["audit"]["hits"] == 2
    with pytest.raises(KgCacheMiss):                               # a new batch offline fails loudly
        judge([{"uid": "z", "stratum": "S", "planted": False, "view": {"edge_type": "X", "k": 99}}])


def test_verify_verdicts_requires_every_id_once():
    chk = A.verify_verdicts(["e1", "e2"])
    ok = {"verdicts": [{"id": "e1", "correct": True, "confidence": 1, "reason": ""},
                       {"id": "e2", "correct": False, "confidence": 0.5, "reason": ""}]}
    assert chk(ok) == (True, None)
    assert chk({"verdicts": ok["verdicts"][:1]}) == (False, "ids")
    assert chk({"verdicts": [ok["verdicts"][0], ok["verdicts"][0]]}) == (False, "ids")


def test_contains_violations_counts_non_direct_children():
    text = "x" * 100
    rec = types.SimpleNamespace(doc_id="D", structure=[
        Node("D", "документ", "", None, None, 0, 100), Node("D/ГЛ.1", "глава", "", "1", "D", 0, 60),
        Node("D/ГЛ.1/СТ.1", "статья", "", "1", "D/ГЛ.1", 0, 30),
        Node("D/ГЛ.1/СТ.1/Ч.1", "часть", "", "1", "D", 0, 10),            # parent skips a level
        Node("D/СТ.9", "статья", "", "9", "D", 90, 120)])                 # span outside parent
    del text
    stubs = [GraphEdge("EXT/a", "EXT/a/СТ.1", "CONTAINS", {}), GraphEdge("EXT/a", "EXT/a/СТ.1/Ч.2", "CONTAINS", {})]
    nodes = {n: GraphNode(n, "element", "статья", "", None, True, {}) for n in ("EXT/a", "EXT/a/СТ.1", "EXT/a/СТ.1/Ч.2")}
    res = A.contains_violations([rec], stubs, nodes)
    assert res["total"] == 6
    assert res["by_check"] == {"not_direct_child_id": 2, "span_outside_parent": 1}


def test_manual_subset_quota_and_determinism():
    g, edges = _graph()
    pops = A.populations(edges)
    sample = A.draw_sample(pops, {s: 10 for s in pops}, 3)
    sub = A.manual_subset(sample, 3, {"REFERS_TO": 4, "REGULATES": 3})
    assert len(sub) == 7 and sub == A.manual_subset(sample, 3, {"REFERS_TO": 4, "REGULATES": 3})
    assert len({A.edge_sort_key(e) for _, e in sub}) == 7


def test_adjudication_overturns_judge_rejections_strict_and_lenient():
    items = [{"uid": f"g{i}", "stratum": "AMENDS:renumber", "planted": False} for i in range(10)] + \
            [{"uid": "p0", "stratum": "AMENDS:renumber", "planted": True}]
    verdicts = {it["uid"]: {"correct": i >= 4, "reason": "", "confidence": 1.0} for i, it in enumerate(items)}
    verdicts["p0"]["correct"] = False
    adj = {"g0": {"category": "judge_error"}, "g1": {"category": "true_error"},
           "g2": {"category": "relation_label"}}                   # g3 rejected, not adjudicated -> wrong
    res = A.evaluate(items, verdicts, {"AMENDS:renumber": 165}, {}, adj)
    r = res["strata"]["AMENDS:renumber"]
    assert (r["n"], r["k"]) == (10, 6)
    assert r["precision_adjudicated_strict"] == pytest.approx(0.7)       # 6 + judge_error
    assert r["precision_adjudicated_lenient"] == pytest.approx(0.8)      # + relation_label
    assert r["rejected_unadjudicated"] == 1 and r["adjudication"] == {"judge_error": 1, "relation_label": 1,
                                                                        "true_error": 1}
    t = res["types"]["AMENDS"]
    assert t["precision_adjudicated_strict"] == pytest.approx(0.7) and not t["gate_adjudicated_strict"]
    assert res["judge"]["genuine_rejected"] == 4 and res["judge"]["rejections_overturned_share"] == pytest.approx(0.25)


def test_corrected_ci_box_covers_point_estimate():
    ci = list(A.wilson(36, 40))
    box = A.corrected_ci(ci, [0.88, 0.98], [0.74, 0.97])
    point = A.corrected(0.9, 0.95, 0.9)
    assert box[0] <= point <= box[1]
    assert box[0] == pytest.approx(min(A.corrected(ci[0], a, b) for a in (0.88, 0.98) for b in (0.74, 0.97)))
    assert A.corrected_ci(None, [0.9, 1], [0.9, 1]) is None
