"""M2 Task 7: edge-precision audit (gpt-4.1 judge, stratified, planted wrong edges, Wilson CI).

    python scripts/audit_graph_edges.py [--seed N] [--out data/graph] [--dry-run] [--manual-sheet FILE]
    GRL_KG_API=1 python scripts/audit_graph_edges.py        # the single live run (judge answers cached)

The audited edges are regenerated in-process from data/records by graphrag.kg.pipeline.extract_all
(the graph build's own extraction path, Ruling R31 (2): Task 4 REFERS_TO + Task 5 AMENDS / REPEALS on one
context, Task 6 DEFINES / REGULATES; LLM decisions from cache/kg, API calls off), then pipeline.r11_filter
(Ruling R11: a REFERS_TO{amended} edge with the (src, dst) of an AMENDS edge is dropped). Sampling / planting / metrics: graphrag.kg.audit.
Judge answers: KgLLM pass "audit" (cache/kg/audit, budget graph.yaml llm.passes.audit). Manual
labels of the fixed 30-edge subset: cache/kg/audit_manual.json ({uid: {correct, reason}}), made
before the live run; manual adjudication of every genuine edge the judge rejected:
cache/kg/audit_adjudication.json ({uid: {category, reason}}), made after it.

Fix round 1 (Ruling R20): the judged sample is FROZEN (data/graph/audit_sample.jsonl.gz: the 727 items
judged at commit 853565d -- uid, stratum, edge, judge view -- with the original population sizes; planting
marks stay in audit_key.json). By default the script replays it: the judge prompt and views are unchanged,
so every answer is a cache hit (no new LLM calls), and the judge-era metrics are reproduced. The frozen
genuine edges are then re-scored by rules against the CURRENT extractor output (`A.rescore`: target from
the judge + manual adjudication, relation label / AMENDS op / REGULATES adoption-line checks, current
population weights). `--fresh` draws a new sample from the current edges (new judge calls) and freezes it.
The programmatic AMENDS op check (`A.op_consistency`) runs over the whole current AMENDS population.

Outputs (--out): audit.json (provenance, per stratum / type metrics, judge sensitivity / agreement,
errors, CONTAINS check, rescore, op check), audit_key.json (private: planted edges and their methods),
audit_items.jsonl (judge input + verdict per item, without planting marks)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402
from graphrag.kg import audit as A  # noqa: E402

MANUAL_FILE = "cache/kg/audit_manual.json"
ADJ_FILE = "cache/kg/audit_adjudication.json"
FROZEN_FILE = "data/graph/audit_sample.jsonl.gz"


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001 -- provenance only
        return "?"


def regenerate():
    """(records, all nodes, audited edges, stub CONTAINS edges, stats) from graphrag.kg.pipeline (Ruling R31 (2):
    the build's extraction path) -- stats: extractor offline-LLM stats, the R11-dropped REFERS_TO{amended}
    count (pipeline.r11_filter) and the R20 relation / deputy counters."""
    from graphrag.data.ingest import read_records
    from graphrag.kg import extract_ref, pipeline
    recs = read_records(config.path("data/records"))
    ex = pipeline.extract_all(recs)
    nodes = {}
    for r in recs:
        nodes.update({n.node_id: n for n in extract_ref.structure_nodes(r)})
    for nid, n in ex.nodes.items():
        nodes.setdefault(nid, n)
    edges, dropped = pipeline.r11_filter(ex.edges)
    stats = {"ref_resolve": ex.llm["ref_resolve"], "entities": ex.llm["entities"],
             "r11_dropped_refers_amended": len(dropped),
             "r20": {"refers_to_relations": dict(sorted(Counter(e.attrs.get("relation") for e in edges
                                                               if e.type == "REFERS_TO").items())),
                     **ex.entity_stats.get("r20", {})}}
    return recs, nodes, edges, ex.stub_contains, stats


def build_items(g, pops, sample, seed, cfg):
    items = []
    for s in sorted(pops):
        for e in sample[s]:
            items.append({"uid": A.uid_of(s, e), "stratum": s, "planted": False, "edge": e})
        k = A.n_planted(len(sample[s]), cfg["plant_share"])
        for p in A.plant_stratum(g, s, pops[s], sample[s], k, seed, cfg):
            items.append({"uid": A.uid_of(s, p["edge"], True), "stratum": s, "planted": True, "edge": p["edge"],
                          "method": p["method"], "base": p["base"]})
    for it in items:
        it["view"] = A.render_item(g, it["edge"], cfg)
    return items


def _snip(g, e, w=150) -> str:
    ev = e.attrs["evidence"]
    t = g.texts[ev["doc_id"]]
    return (t[max(0, ev["start"] - w):ev["start"]] + "⟦" + A._cap(ev["quote"], 400) + "⟧" + t[ev["end"]:ev["end"] + w])


def in_open_quote(g, e) -> bool:
    """The evidence starts inside a quotation opened earlier in its paragraph («…» / “…”): a citation
    inside quoted new or replaced text of an amendment (not the element being amended)."""
    ev = e.attrs["evidence"]
    t = g.texts[ev["doc_id"]]
    pre = t[t.rfind("\n", 0, ev["start"]) + 1:ev["start"]]
    return pre.count("«") > pre.count("»") or pre.count("“") > pre.count("”")


def amended_label_check(g, pops, items, verdicts) -> dict:
    """The REFERS_TO{amended} mislabel of the judged sample (labels as judged, before R20): evidence inside
    a quotation of its SOURCE NODE (A.in_node_quote; the paragraph-scoped count of the first report is kept
    as `sample_in_quote_paragraph_scope`, it misses multi-paragraph quoted new text), how the judge treated
    them; and, for the CURRENT population, how the amended / cited_in_amendment labels agree with the
    node-scoped quote rule."""
    samp = [it for it in items if it["stratum"] == "REFERS_TO:amended" and not it["planted"]]
    q = [it for it in samp if A.in_node_quote(g, it["edge"])]
    cur = {}
    for rel in ("amended", "cited_in_amendment"):
        pop = pops.get(f"REFERS_TO:{rel}", [])
        cur[rel] = {"population": len(pop), "in_node_quote": sum(A.in_node_quote(g, e) for e in pop)}
    return {"sample": len(samp), "sample_in_quote": len(q),
            "sample_in_quote_paragraph_scope": sum(in_open_quote(g, it["edge"]) for it in samp),
            "sample_in_quote_judge_accepted": sum(1 for it in q if (verdicts.get(it["uid"]) or {}).get("correct")),
            "label_precision_strict_before_r20": (len(samp) - len(q)) / len(samp) if samp else None,
            "label_precision_strict_before_r20_ci": list(A.wilson(len(samp) - len(q), len(samp))) if samp else None,
            "current": cur, "sample_in_quote_uids": sorted(it["uid"] for it in q)}


def write_frozen(path: Path, meta: dict, items: list[dict]) -> None:
    """The judged sample (uid, stratum, edge, view; no planting marks), gzip with a fixed header (mtime 0)."""
    body = json.dumps({"_meta": meta}, ensure_ascii=False, sort_keys=True) + "\n" + "".join(
        json.dumps({"uid": it["uid"], "stratum": it["stratum"], "edge": it["edge"].to_json(), "view": it["view"]},
                   ensure_ascii=False, sort_keys=True) + "\n" for it in sorted(items, key=lambda x: x["uid"]))
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as f:
        f.write(body.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def read_frozen(path: Path, key: dict):
    """(meta, items, sha256 of the uncompressed file) with planting marks from the private key."""
    from graphrag.kg.schema import GraphEdge
    raw = gzip.decompress(path.read_bytes())
    lines = raw.decode("utf-8").splitlines()
    meta = json.loads(lines[0])["_meta"]
    planted = {p["uid"]: p for p in key["planted"]}
    items = []
    for ln in lines[1:]:
        d = json.loads(ln)
        it = {"uid": d["uid"], "stratum": d["stratum"], "planted": d["uid"] in planted,
              "edge": GraphEdge.from_json(d["edge"]), "view": d["view"]}
        if it["planted"]:
            it["method"] = planted[d["uid"]]["method"]
            it["base"] = GraphEdge.from_json(planted[d["uid"]]["base"])
        items.append(it)
    if len(planted) != sum(it["planted"] for it in items):
        raise ValueError("audit_key.json planted uids do not match the frozen sample")
    return meta, items, hashlib.sha256(raw).hexdigest()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    gcfg = config.load("graph")
    cfg = {**A.DEFAULTS, **(gcfg.get("audit") or {})}
    seed = int(opt("--seed", cfg["seed"]))
    out = config.path(opt("--out", "data/graph"))
    recs, nodes, edges, stub_contains, ext_stats = regenerate()
    g = A.AuditGraph.from_records(recs, nodes, [e for e in edges if e.type in A.AUDITED_TYPES])
    pops = A.populations(edges)
    cur_counts = {s: len(v) for s, v in pops.items()}
    frozen_path = config.path(opt("--frozen", FROZEN_FILE))
    fresh = "--fresh" in argv or not frozen_path.exists()
    if fresh:
        counts = cur_counts
        sizes = A.sample_sizes(counts, cfg["major"], tuple(cfg["full"]))
        sample = A.draw_sample(pops, sizes, seed)
        items = build_items(g, pops, sample, seed, cfg)
        frozen_meta, frozen_sha = None, None
    else:
        key0 = json.loads((out / "audit_key.json").read_text(encoding="utf-8"))
        frozen_meta, items, frozen_sha = read_frozen(frozen_path, key0)
        if frozen_meta["seed"] != seed:
            raise ValueError(f"frozen sample seed {frozen_meta['seed']} != {seed}; use --fresh for a new sample")
        counts, sizes = frozen_meta["populations"], frozen_meta["sample_sizes"]
        sample = {s: sorted((it["edge"] for it in items if it["stratum"] == s and not it["planted"]), key=A.edge_sort_key)
                  for s in sorted(counts)}
    man_sub = A.manual_subset(sample, seed)
    man_uids = [A.uid_of(s, e) for s, e in man_sub]
    by_uid = {it["uid"]: it for it in items}
    if opt("--manual-sheet"):
        Path(opt("--manual-sheet")).write_text(
            "".join(json.dumps({"uid": u, "stratum": by_uid[u]["stratum"], **by_uid[u]["view"]}, ensure_ascii=False,
                               sort_keys=True) + "\n" for u in man_uids), encoding="utf-8")
    mf = config.path(MANUAL_FILE)
    manual = json.loads(mf.read_text(encoding="utf-8"))["labels"] if mf.exists() else {}
    manual = {u: lab for u, lab in manual.items() if u in by_uid}
    af = config.path(ADJ_FILE)
    adjud = json.loads(af.read_text(encoding="utf-8"))["labels"] if af.exists() else {}
    adjud = {u: lab for u, lab in adjud.items() if u in by_uid}
    batches = A.make_batches(items, cfg["batch"], seed)
    print(json.dumps({"mode": "fresh" if fresh else "frozen_replay", "populations": counts,
                      "current_populations": cur_counts, "sizes": sizes, "items": len(items),
                      "planted": sum(it["planted"] for it in items), "batches": len(batches),
                      "manual_labels": len(manual)}, ensure_ascii=False, indent=1))
    if "--dry-run" in argv:
        return 0
    from graphrag.kg.llm import KgLLM
    llm = KgLLM()
    judge = A.llm_judge(llm)
    verdicts = {}
    for b in batches:
        verdicts.update(judge(b))
    res = A.evaluate(items, verdicts, counts, manual, adjud)
    # errors (genuine edges the judge rejects), judge misses on planted edges, manual disagreements
    errors = []
    for it in sorted(items, key=lambda x: (x["stratum"], x["uid"])):
        v = verdicts.get(it["uid"])
        if v is None or it["planted"] or v["correct"]:
            continue
        e = it["edge"]
        errors.append({"uid": it["uid"], "stratum": it["stratum"], "src": e.src, "dst": e.dst,
                       "attrs": {k: e.attrs[k] for k in ("relation", "op", "date") if k in e.attrs},
                       "target": it["view"]["target"], "context": _snip(g, e), "judge_reason": v["reason"],
                       "judge_confidence": v["confidence"], "manual": manual.get(it["uid"]),
                       "adjudication": adjud.get(it["uid"])})
    missed = [{"uid": it["uid"], "stratum": it["stratum"], "method": it["method"], "src": it["edge"].src,
               "dst": it["edge"].dst, "base_dst": it["base"].dst, "context": _snip(g, it["edge"]),
               "target": it["view"]["target"], "judge_reason": verdicts[it["uid"]]["reason"]}
              for it in items if it["planted"] and verdicts.get(it["uid"]) and verdicts[it["uid"]]["correct"]]
    disagree = [{"uid": u, "stratum": by_uid[u]["stratum"], "manual": manual[u], "judge": verdicts.get(u)}
                for u in man_uids if u in manual and verdicts.get(u) and verdicts[u]["correct"] != manual[u]["correct"]]
    from graphrag.kg import ent_rules
    audited = [e for e in edges if e.type in A.AUDITED_TYPES]
    resc = A.rescore(items, verdicts, adjud, g, audited, pops, ent_rules.adoption_line)
    opc = A.op_consistency(audited)
    relrule = A.relation_rule_population(g, audited)
    opc_sample = A.op_consistency([it["edge"] for it in items if not it["planted"]])
    op_plants = [it for it in items if it["planted"] and it.get("method") == "op_swap"]
    opc_planted = A.op_consistency([it["edge"] for it in op_plants])
    pop_digest = hashlib.sha256("".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                        for s in sorted(pops) for e in pops[s]).encode("utf-8")).hexdigest()
    dirty = _git("status", "--porcelain", "--", "graphrag", "config", "data/records")
    report = {
        "provenance": {"commit": _git("rev-parse", "HEAD"), "code_dirty": dirty.splitlines(),
                       "mode": "fresh" if fresh else "frozen_replay", "frozen_sample": FROZEN_FILE,
                       "frozen_sample_sha256": frozen_sha, "frozen_meta": frozen_meta,
                       "edges_sha256": pop_digest, "seed": seed, "config": cfg,
                       "extractors_offline_llm": ext_stats, "judge_model": llm.model,
                       "judge_prompt_version": llm.passes["audit"].get("prompt_version")},
        "populations": counts, "current_populations": cur_counts, "sample_sizes": sizes,
        "items": len(items), "planted": sum(it["planted"] for it in items),
        "planted_methods": dict(sorted(Counter(it["method"] for it in items if it["planted"]).items())),
        "planted_rejected_by_method": {m: [sum(1 for it in items if it["planted"] and it["method"] == m
                                               and verdicts.get(it["uid"]) and not verdicts[it["uid"]]["correct"]),
                                           sum(1 for it in items if it["planted"] and it["method"] == m)]
                                       for m in sorted({it["method"] for it in items if it["planted"]})},
        "batches": len(batches), "unjudged": sum(1 for it in items if verdicts.get(it["uid"]) is None),
        "llm": llm.stats_json(),
        **res,
        "manual_subset": man_uids, "manual_disagreements": disagree,
        "contains": A.contains_violations(recs, stub_contains, nodes),
        "amended_label_check": amended_label_check(g, pops, items, verdicts),
        "errors": errors, "planted_missed": missed,
        "rescore": resc, "relation_rule_population": relrule,
        "op_check": {"population": opc, "sample_genuine": {k: v for k, v in opc_sample.items() if k != "contradiction_list"},
                     "planted_op_swap": {"plants": len(op_plants), "flagged": opc_planted["contradicting_edges"]}},
    }
    out.mkdir(parents=True, exist_ok=True)
    if fresh:
        write_frozen(frozen_path, {"seed": seed, "populations": counts, "sample_sizes": sizes, "edges_sha256": pop_digest,
                                   "commit": _git("rev-parse", "HEAD")}, items)
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if fresh:                                             # replay: the key is an input, left as it is
        key = {"seed": seed, "planted": [{"uid": it["uid"], "stratum": it["stratum"], "method": it["method"],
                                          "edge": it["edge"].to_json(), "base": it["base"].to_json()}
                                         for it in items if it["planted"]]}
        (out / "audit_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                            encoding="utf-8")
    (out / "audit_items.jsonl").write_text("".join(
        json.dumps({"uid": it["uid"], "stratum": it["stratum"], "edge": it["edge"].to_json(), "view": it["view"],
                    "verdict": verdicts.get(it["uid"])}, ensure_ascii=False, sort_keys=True) + "\n"
        for it in sorted(items, key=lambda x: x["uid"])), encoding="utf-8")
    summary = {"judge": res["judge"], "llm": report["llm"], "unjudged": report["unjudged"],
               "planted_rejected_by_method": report["planted_rejected_by_method"],
               "precision_corrected_status": res["precision_corrected_status"],
               "rescore_types": resc["types"], "rescore_status": resc["status"],
               "relation_rule_population": {k: v for k, v in relrule.items() if k != "examples"},
               "op_check": {k: v for k, v in report["op_check"]["population"].items() if k != "contradiction_list"},
               "op_check_planted": report["op_check"]["planted_op_swap"],
               "types": {t: {k: r[k] for k in ("N", "n", "k", "precision", "ci", "precision_corrected",
                                                "ci_corrected", "precision_adjudicated_strict", "ci_adjudicated_strict",
                                                "precision_adjudicated_lenient", "sensitivity", "gate_raw",
                                                "gate_corrected", "gate_adjudicated_strict", "strata_below_0.90",
                                                "strata_below_0.90_adjudicated_strict")}
                         for t, r in res["types"].items()},
               "strata": {s: {k: r[k] for k in ("N", "n", "k", "precision", "ci", "precision_corrected",
                                                "precision_adjudicated_strict", "precision_adjudicated_lenient",
                                                "adjudication", "rejected_unadjudicated", "planted",
                                                "planted_rejected")} for s, r in res["strata"].items()},
               "contains": {k: report["contains"][k] for k in ("total", "violations", "by_check")},
               "amended_label_check": {k: v for k, v in report["amended_label_check"].items() if k != "sample_in_quote_uids"}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
