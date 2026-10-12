"""M3-prep round (Ruling R31): store reset guard, cumulative LLM budgets, the single extraction path, domain
regexes as config data, the additive PyG fields and the artifact pin / backup / restore."""
from __future__ import annotations

import json

import pytest

from graphrag import config


# --- (5) reset_all guarded inside the store; cumulative LLM budget ---------------------------------------

def _fake_store(uri):
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
    g.uri = uri
    g.driver = type("D", (), {"session": lambda self: _S()})()
    return g, ran


def test_reset_all_refuses_non_dev_endpoints(monkeypatch):
    monkeypatch.delenv("GRL_ALLOW_RESET", raising=False)
    for bad in ("bolt://localhost:7687", "bolt://203.0.113.10:7688", "bolt://localhost"):
        g, ran = _fake_store(bad)
        with pytest.raises(PermissionError, match="reset refused"):
            g.reset_all()
        assert ran == [], bad                                  # refused before any query
    g, ran = _fake_store("bolt://localhost:7688")
    g.reset_all()
    assert len(ran) == 3
    monkeypatch.setenv("GRL_ALLOW_RESET", "1")                # the M1 override of build_indexes.py --reset
    g, ran = _fake_store("bolt://10.1.2.3:7687")
    g.reset_all()
    assert len(ran) == 3


class _Fake:
    def __init__(self):
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        msg = type("M", (), {"content": json.dumps({"choice": 1})})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


SCH = {"type": "object", "additionalProperties": False, "required": ["choice"],
       "properties": {"choice": {"type": "integer"}}}


def _cfg(max_calls=3, total=100):
    return {"llm": {"model": "gpt-4.1", "max_calls_total": total,
                    "passes": {"ref_resolve": {"prompt_version": 1, "max_calls": max_calls}}}}


def test_per_pass_budget_is_cumulative_over_the_ledger(tmp_path):
    from graphrag.kg.llm import KgBudgetExceeded, KgLLM
    (tmp_path / "ledger.json").write_text(json.dumps({"ref_resolve": {"2026-09-27": 2}}), encoding="utf-8")
    fake = _Fake()
    a = KgLLM(_cfg(max_calls=3), tmp_path, client=fake, allow_api=True)
    assert a.cumulative_calls("ref_resolve") == 2
    a.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P")               # 3rd call ever: allowed
    assert a.cumulative_calls("ref_resolve") == 3 and fake.calls == 1
    with pytest.raises(KgBudgetExceeded, match="cumulative"):
        a.cached_json("ref_resolve", {"q": 2}, SCH, prompt="P")
    b = KgLLM(_cfg(max_calls=3), tmp_path, client=fake, allow_api=True)   # a new run sees the ledger
    with pytest.raises(KgBudgetExceeded, match="cumulative"):
        b.cached_json("ref_resolve", {"q": 3}, SCH, prompt="P")
    assert b.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P") == {"choice": 1}   # cache hits are free
    assert fake.calls == 1


def test_real_ledger_is_within_the_configured_budgets_or_exhausted():
    """The tracked ledger vs graph.yaml: a pass over its max_calls can make no further call."""
    from graphrag.kg.llm import KgLLM
    llm = KgLLM(allow_api=False)
    led = llm.ledger()
    assert led, "cache/kg/ledger.json is tracked"
    for p in led:
        assert p in llm.passes
        assert llm.cumulative_calls(p) == sum(led[p].values())


# --- (2) one edge-regeneration path; edge_key once; shared judge helpers ----------------------------------

def _e(src, dst, t, **attrs):
    from graphrag.kg.schema import GraphEdge
    return GraphEdge(src, dst, t, attrs)


def test_r11_filter_is_the_one_rule():
    from graphrag.kg import pipeline
    es = [_e("a", "b", "REFERS_TO", relation="amended"), _e("a", "b", "AMENDS", op="edit"),
          _e("a", "b", "REFERS_TO", relation="cited_in_amendment"), _e("a", "c", "REFERS_TO", relation="amended"),
          _e("c", "b", "REFERS_TO", relation="amended")]
    kept, dropped = pipeline.r11_filter(es)
    assert dropped == [es[0]] and kept == es[1:]


def test_build_audit_and_related_use_the_pipeline():
    """No hand-written copy of the extractor sequence or of R11 outside graphrag.kg.pipeline."""
    from graphrag.kg import build as B
    from graphrag.kg import pipeline
    assert B.extract_all is pipeline.extract_all and B.Extracted is pipeline.Extracted
    root = config.path(".")
    for f in ("graphrag/kg/build.py", "scripts/audit_graph_edges.py", "scripts/build_related.py",
              "scripts/build_graph.py"):
        src = (root / f).read_text(encoding="utf-8")
        assert "extract_ref.extract_all(" not in src and "extract_ent.build(" not in src, f
        assert '== "amended" and (e.src, e.dst) in amk' not in src, f
    assert "pipeline.extract_all(" in (root / "scripts/audit_graph_edges.py").read_text(encoding="utf-8")


def test_edge_key_is_defined_once():
    import re
    root = config.path(".")
    defs = [p.relative_to(root).as_posix() for p in (root / "graphrag").rglob("*.py")
            if re.search(r"(?m)^def edge_key\(", p.read_text(encoding="utf-8"))]
    assert defs == ["graphrag/kg/artifact.py"]
    from graphrag.kg.artifact import edge_key
    from graphrag.kg.store import edge_key_json
    e = _e("a", "b", "REFERS_TO", relation="act", evidence={"x": 1})
    assert edge_key(e) == ("a", "REFERS_TO", "b", "act")
    assert edge_key_json(e) == '["a", "REFERS_TO", "b", "act"]'
    assert edge_key(_e("a", "b", "RELATED", weight=0.7)) == ("a", "RELATED", "b")


def test_judge_prompt_digests_match_the_cached_answers():
    """The shared graphrag.kg.judge builds the same prompt + schema bytes: the cached answers stay hits."""
    import scripts.research_related as R
    from graphrag.kg import audit as A
    from graphrag.kg.llm import prompt_digest
    assert prompt_digest(A.JUDGE_PROMPT, A.judge_schema()) == \
        "616a3efc19b9a04dad555517715be2c2026762c0cbfa821415e597df850a70ab"
    assert prompt_digest(R.JUDGE_PROMPT, R.judge_schema()) == \
        "98f85f5e07212c1d3eb82a11bc40da35bab42abdf8a2cba24187d6e3fcdc767e"
    for p, dig in (("audit", "616a3efc"), ("audit_related", "98f85f5e")):
        f = sorted(config.path(f"cache/kg/{p}").glob("*.json"))[0]
        assert json.loads(f.read_text(encoding="utf-8"))["prompt_digest"].startswith(dig)


def test_research_helpers_left_the_library_module():
    from graphrag.kg import semantic as S
    for name in ("auc", "bootstrap_auc", "logistic_fit", "dev_theta", "adopt_theta", "class_removals",
                 "llm_judge", "sample_strata", "JUDGE_PROMPT"):
        assert not hasattr(S, name), name
    assert hasattr(S, "latent_edges") and hasattr(S, "calibrate_theta")


# --- (3) domain regexes and semantic.DEFAULTS are config data ------------------------------------------------

def test_act_title_classes_come_from_graph_yaml():
    from graphrag.kg import acts
    t = config.load("graph")["act_titles"]
    aw = t["amend_word"]
    assert acts._GENERIC.pattern == t["generic"].replace("{AMEND_WORD}", aw)
    assert acts._AMENDING.pattern == t["amending"] and acts._CODE_APPROVAL.pattern == t["code_approval"]
    assert acts._ADOPTED.pattern == t["adopted"] and acts.GENERIC_KEY == t["generic_key"]
    src = config.path("graphrag/kg/acts.py").read_text(encoding="utf-8")
    for lit in ("ayrim qonun hujjatlariga", "tasdiqlash {TG}", "Принят\\s+Законодательной", "kuchini yo{A}qotgan"):
        assert lit not in src, lit
    # behaviour: omnibus variants share the generic key; amending / code-approval / law-text header
    assert acts.act_key("Айрим қонун ҳужжатларига ўзгартиш ва қўшимчалар киритиш тўғрисида") == acts.GENERIC_KEY
    assert acts.is_generic(acts.GENERIC_KEY) and acts.is_amending(acts.GENERIC_KEY)
    assert acts._CODE_APPROVAL.match("soliq kodeks tasdiqlash toʼgʼrisida").group(1) == "soliq kodeks"
    assert acts._ADOPTED.search("ПРИНЯТ ЗАКОНОДАТЕЛЬНОЙ ПАЛАТОЙ 13 февраля 2025 года")


def test_semantic_defaults_and_boilerplate_classes_come_from_graph_yaml():
    from graphrag.kg import semantic as S
    g = config.load("graph")
    assert S.DEFAULTS == g["related_defaults"]
    bp = g["related_boilerplate"]
    for name, rx in (("formula", S._FORMULA), ("clause", S._CLAUSE), ("adoption", S._ADOPTION),
                     ("gazette_word", S._GAZ_WORD)):
        assert rx.pattern == bp[name].replace("{APO}", bp["apo"]), name
    src = config.path("graphrag/kg/semantic.py").read_text(encoding="utf-8")
    assert "raislik\\s+qiluvchi" not in src and '"min_content_tokens": 12' not in src
    assert S.boilerplate_reason("Раҳмат. Марҳамат.") == "formula"
    assert S.boilerplate_reason("Мазкур қарор маъқуллансин ва кучга киради.") == "clause"


# --- (4) additive PyG fields -------------------------------------------------------------------------------

def _mini_graph():
    from graphrag.kg import build as B
    from graphrag.kg.schema import GraphNode, Schema
    cfg = config.load("graph")
    n = [GraphNode("D1", "document", "документ", "d", "D1", False, {"doc_type": "law_text"}),
         GraphNode("D1/СТ.1", "element", "статья", "a", "D1", False, {"number": "1"}),
         GraphNode("ENT/organ/x", "entity", "organ", "x", None, False, {}),
         GraphNode("EXT/k", "document", "документ", "k", None, True, {}),
         GraphNode("EXT/k/СТ.2", "element", "статья", "k2", None, True, {"act": "EXT/k", "number": "2"})]
    ev = {"doc_id": "D1", "start": 0, "end": 1, "quote": "x"}
    e = [_e("D1", "D1/СТ.1", "CONTAINS"), _e("EXT/k", "EXT/k/СТ.2", "CONTAINS"),
         _e("D1/СТ.1", "EXT/k/СТ.2", "AMENDS", op="add", ops=["add", "renumber"], date="2025-01-02", evidence=ev),
         _e("D1/СТ.1", "EXT/k", "REFERS_TO", relation="amended", evidence=ev),
         _e("D1/СТ.1", "EXT/k", "REPEALS", date="1970-01-11", evidence=ev),
         _e("D1/СТ.1", "ENT/organ/x", "REGULATES", evidence=ev),
         _e("D1", "EXT/k", "RELATED", weight=0.7)]
    nodes = sorted(n, key=lambda x: x.node_id)
    return B.Graph(nodes=nodes, edges=e, schema=Schema(cfg), cfg=cfg)


def test_pyg_r31_fields_shape_dtype_and_values():
    import numpy as np
    import torch
    from graphrag.kg.pyg import pyg_payload
    g = _mini_graph()
    p = pyg_payload(g, np.eye(len(g.nodes), 4, dtype=np.float32))
    E, N = p["edge_index"].shape[1], len(p["node_ids"])
    assert E == 8                                                    # RELATED twice
    for f, dt, shape in (("edge_attr_relation", torch.long, (E,)), ("edge_attr_op", torch.long, (E,)),
                         ("edge_attr_ops", torch.bool, (E, len(p["amend_ops"]))), ("edge_date", torch.long, (E,)),
                         ("node_type", torch.long, (N,)), ("entity_type", torch.long, (N,)),
                         ("is_external", torch.bool, (N,)), ("node_doc", torch.long, (N,))):
        assert p[f].dtype == dt and tuple(p[f].shape) == shape, f
    types = [p["edge_types"][t] for t in p["edge_type"].tolist()]
    k = types.index("AMENDS")
    assert p["amend_ops"][p["edge_attr_op"][k]] == "add"
    assert {p["amend_ops"][j] for j in p["edge_attr_ops"][k].nonzero().flatten().tolist()} == {"add", "renumber"}
    assert p["edge_date"][k] == 20090 and p["edge_date"][types.index("REPEALS")] == 10
    assert p["edge_relations"][p["edge_attr_relation"][types.index("REFERS_TO")]] == "amended"
    other = [i for i, t in enumerate(types) if t not in ("REFERS_TO",)]
    assert all(p["edge_attr_relation"][i] == -1 for i in other)
    assert all(p["edge_attr_op"][i] == -1 and not p["edge_attr_ops"][i].any() for i, t in enumerate(types) if t != "AMENDS")
    row = {nid: i for i, nid in enumerate(p["node_ids"])}
    nt = lambda nid: p["node_types"][p["node_type"][row[nid]]] if p["node_type"][row[nid]] >= 0 else None  # noqa: E731
    assert nt("D1") == "document:law_text" and nt("EXT/k") == "document:external" and nt("D1/СТ.1") == "element:статья"
    assert nt("ENT/organ/x") is None
    assert p["entity_types"][p["entity_type"][row["ENT/organ/x"]]] == "organ" and p["entity_type"][row["D1"]] == -1
    assert p["is_external"].tolist() == [n.external for n in g.nodes]
    assert p["doc_ids"] == ["D1", "EXT/k"]
    assert p["doc_ids"][p["node_doc"][row["EXT/k/СТ.2"]]] == "EXT/k" and p["doc_ids"][p["node_doc"][row["D1/СТ.1"]]] == "D1"
    assert p["node_doc"][row["ENT/organ/x"]] == -1


@pytest.mark.corpus
def test_pyg_r31_fields_consistent_with_the_artifact():
    """pyg.pt of data/graph/latest.json vs its edges.jsonl / nodes.jsonl, and the vocabularies in stats.json."""
    import datetime as dt

    import torch
    from graphrag.kg.artifact import read_graph
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    d = config.path(latest["dir"])
    p = torch.load(d / "pyg.pt", weights_only=True)
    st = json.loads((d / "stats.json").read_text(encoding="utf-8"))["pyg"]
    for k in ("edge_relations", "amend_ops", "node_types", "entity_types"):
        assert st["vocabularies"][k] == p[k], k
    g = config.load("graph")
    assert p["edge_relations"] == g["ref_relations"] and p["amend_ops"] == g["amend_ops"]
    assert p["entity_types"] == g["vertex_types"]["entity"]
    nodes, edges = read_graph(d)
    row = {n.node_id: i for i, n in enumerate(nodes)}
    assert p["node_ids"] == [n.node_id for n in nodes]
    # every stored edge's attrs appear at its (type, src, dst) row(s)
    ei, et = p["edge_index"], p["edge_type"]
    idx = {(int(et[k]), int(ei[0, k]), int(ei[1, k])): k for k in range(ei.shape[1])}
    tid = {t: i for i, t in enumerate(p["edge_types"])}
    ep = dt.date(1970, 1, 1)
    n_rel = n_op = n_date = 0
    for e in edges:
        if e.type == "REFERS_TO":        # several relations may join one pair: checked as a multiset below
            n_rel += 1
            continue
        k = idx[(tid[e.type], row[e.src], row[e.dst])]
        if e.type == "AMENDS":
            assert p["amend_ops"][p["edge_attr_op"][k]] == e.attrs["op"]
            hot = {p["amend_ops"][j] for j in p["edge_attr_ops"][k].nonzero().flatten().tolist()}
            assert hot == set(e.attrs.get("ops") or [e.attrs["op"]])
            n_op += 1
        if e.type in ("AMENDS", "REPEALS"):
            want = (dt.date.fromisoformat(e.attrs["date"]) - ep).days if e.attrs.get("date") else -1
            assert int(p["edge_date"][k]) == want
            n_date += want >= 0
        else:
            assert int(p["edge_date"][k]) == -1 and int(p["edge_attr_op"][k]) == -1
    # REFERS_TO: the multiset of (src, dst, relation) equals the tensor rows'
    from collections import Counter
    want = Counter((row[e.src], row[e.dst], e.attrs["relation"]) for e in edges if e.type == "REFERS_TO")
    m = et == tid["REFERS_TO"]
    got = Counter((int(s), int(t), p["edge_relations"][int(r)])
                  for s, t, r in zip(ei[0, m], ei[1, m], p["edge_attr_relation"][m]))
    assert got == want and n_rel == sum(want.values())
    assert bool((p["edge_attr_relation"][~m] == -1).all())
    assert n_op == sum(1 for e in edges if e.type == "AMENDS") and st["counts_r31"]["edge_date_set"] == n_date
    # node arrays
    for n in nodes:
        i = row[n.node_id]
        assert bool(p["is_external"][i]) == n.external
        if n.kind == "entity":
            assert p["entity_types"][p["entity_type"][i]] == n.type and p["node_type"][i] == -1
        else:
            assert p["entity_type"][i] == -1
        if n.kind in ("entity", "concept"):
            assert p["node_doc"][i] == -1
        else:
            doc = n.node_id if n.kind == "document" else (n.attrs["act"] if n.external else n.doc_id)
            assert p["doc_ids"][p["node_doc"][i]] == doc
            name = f"element:{n.type}" if n.kind == "element" else \
                ("document:external" if n.external else f"document:{n.attrs.get('doc_type') or n.type}")
            assert p["node_types"][p["node_type"][i]] == name
    assert p["doc_ids"] == sorted(n.node_id for n in nodes if n.kind == "document")


# --- (1) pin / backup / restore -------------------------------------------------------------------------------

def test_restore_artifacts_round_trip(tmp_path):
    import scripts.restore_artifacts as RA
    root = tmp_path / "pkg"
    g = root / "data" / "graph"
    (g / "v1").mkdir(parents=True)
    (g / "v1" / "edges.jsonl").write_text("e\n")
    (g / "v1" / "x.npy").write_bytes(b"\x00\x01")
    (g / "node_vecs.npy").write_bytes(b"nv")
    (g / "audit.json").write_text("{}")
    (g / "latest.json").write_text('{"version": "v1"}')
    (g / "other.txt").write_text("not pinned")
    files = RA.pinned_files(["v1"], root)
    assert files == ["data/graph/audit.json", "data/graph/node_vecs.npy", "data/graph/v1/edges.jsonl",
                     "data/graph/v1/x.npy"]
    (root / "docs" / "m2").mkdir(parents=True)
    (root / RA.MANIFEST).write_text(RA.manifest_text(files, root))
    res = RA.backup("v1", tmp_path / "bak", root)
    arc = __import__("pathlib").Path(res["archive"])
    assert arc.exists() and (arc.parent / (arc.name + ".sha256")).exists() and res["files"] == 5
    out = RA.restore(arc, tmp_path / "scratch")
    assert out == {"extracted": 5, "verified": 4, "problems": []}
    assert (tmp_path / "scratch" / "data/graph/v1/x.npy").read_bytes() == b"\x00\x01"
    (tmp_path / "scratch" / "data/graph/v1/x.npy").write_bytes(b"changed")      # a clash is refused
    with pytest.raises(FileExistsError):
        RA.restore(arc, tmp_path / "scratch")
    (g / "v1" / "x.npy").write_bytes(b"drift")                                     # stale manifest: no backup
    with pytest.raises(ValueError, match="differ"):
        RA.backup("v1", tmp_path / "bak2", root)
    assert RA.verify(root, RA.read_manifest(root / RA.MANIFEST)) == ["sha256 differs: data/graph/v1/x.npy"]


def test_version_inputs_pin_the_node_vector_cache():
    import scripts.build_graph as BG
    b = config.load("graph")["build"]
    d = BG.node_vecs_digest(b)
    assert set(d) == {"data/graph/node_vecs.npy", "data/graph/node_vecs_keys.json"}
    src = config.path("scripts/build_graph.py").read_text(encoding="utf-8")
    assert "**node_vecs_digest(b)" in src
