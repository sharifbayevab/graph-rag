"""M7 Task 6d-a (val + train only; the test splits stay sealed): the Q2 router branch «amendments of the named act»
(grl.hard.amend_lookup, a hand-written rule, no training) against the pilot-8 agent theta*, the fixed baselines and the
pilot-8 router system. The router predictions are the pilot-8 ones (runs/p8-local/router.json), not retrained.

    PYTHONPATH=. python scripts/m7_router_q2rule.py --out runs/p8-local/q2rule.json

Q2 table: all@5 / recall@5 per template for the rule, agent theta*, ce+fused@5 / @10 (val; the baselines exist for val
only). The rule alone on train too. System: val, router predicted `old` -> ranker, `count` -> agent, `version` ->
the rule (the agent's context where no root qualifies). all@10: the rule's fragments first, then the fused fill (p8 §7).
The EVIDENCE variant (candidates = quoted fragments of the edges) is a labelled DIAGNOSTIC only: R33 gold is built
from the same spans (R60), so it is circular and is not in the headline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()

from graphrag import config  # noqa: E402
from scripts.m7_pilot8 import GROUPS, _fill10, _group, _jsonl, _write, mean_se  # noqa: E402

TEMPLATES = ("what_law_changed", "which_deleted_year", "which_law_part", "which_laws_article")


def _m(ids, gold):
    S = set(ids)
    return len(S & gold) / len(gold), float(gold <= S)


def _evidence_rows(data, index, spans, edges_full):
    """DIAGNOSTIC (R60 circular): root -> fragment rows that overlap the evidence span of an incoming AMENDS edge."""
    ev = defaultdict(list)
    for e in edges_full:
        if e["type"] != "AMENDS":
            continue
        dv, sv = data.node_row.get(e["dst"]), data.node_row.get(e["src"])
        if dv is None or sv is None:
            continue
        r = int(index.root_row[dv])
        x = (e.get("attrs") or {}).get("evidence") or {}
        if r < 0 or not {"doc_id", "start", "end"} <= set(x):
            continue
        for c in index.mu_inv(sv):
            sp = spans.get(data.chunk_ids[int(c)])
            if sp and sp[0] == x["doc_id"] and sp[1] < x["end"] and x["start"] < sp[2]:
                ev[r].append(int(c))
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", default="data/m7/full/2ab8c949e85e/ship")
    ap.add_argument("--agent", default="runs/spark/p8-ev-g-k5/pq.jsonl")
    ap.add_argument("--ranker", default="runs/spark/p8-ranker-k5b/pq.jsonl")
    ap.add_argument("--baselines", default="runs/spark/p8-bl/baselines.json")
    ap.add_argument("--router", default="runs/p8-local/router.json")
    ap.add_argument("--out", default="runs/p8-local/q2rule.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--articles", action="store_true", help="the article-mode struct EnvData (struct.articles)")
    ap.add_argument("--amend-op", action="store_true", help="the struct env with struct.amend_op (Task 6d-b)")
    a = ap.parse_args(argv)
    from graphrag.data.ingest import read_records
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    from grl.hard.amend_lookup import amend_lookup, build_amend_index, load_amends_pairs, named_act_root
    from grl.hard.probe import STRUCT_PREFIX
    from grl.model.variants import StructEnv
    from grl.train.fbeta import grl_reward, grl_weights
    cfg = config.load("train")
    cfg["env"]["K_max"] = a.k
    ship = config.path(a.ship)
    rows = [r for r in _jsonl(ship / "pairs.jsonl") if r["split"] in ("train", "val")]
    if {r["split"] for r in rows} - {"train", "val"}:
        raise SystemExit("ship holds rows outside train / val")
    sdata = load_struct_env_data(articles=a.articles)           # fix 6d-b: the struct env flags from args
    ranks = load_rankings(ship / "rankings" / "latest.json", graph_version=sdata.version)
    senv = StructEnv(sdata, cfg["env"], None, articles=a.articles, amend_op=a.amend_op)
    L_max = int(cfg["env"]["L_max"])
    w = grl_weights(cfg)
    index = build_amend_index(sdata, senv._root_row, load_amends_pairs(sdata))
    # diagnostic inputs (evidence variant)
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    edges_full = _jsonl(config.path(latest["dir"]) / "edges.jsonl")
    recs = read_records(config.path(config.load("data")["records_dir"]))
    spans = {c.chunk_id: (r.doc_id, int(c.start), int(c.end)) for r in recs for c in r.chunks}
    ev_rows = _evidence_rows(sdata, index, spans, edges_full)

    val = {r["id"]: r for r in rows if r["split"] == "val"}
    agent = {r["qkey"][:-2]: r for r in _jsonl(a.agent)}
    ranker = {r["qkey"][:-2]: r for r in _jsonl(a.ranker)}
    bl = json.loads(Path(a.baselines).read_text(encoding="utf-8"))
    routes = json.loads(Path(a.router).read_text(encoding="utf-8"))["routes"]
    ids = sdata.chunk_ids

    def evaluate(r, ctx_rows, q, K10=False, fused=None):
        gold = {int(sdata.chunk_row[g]) for g in r["C_ref"]}
        ctx = [int(c) for c in ctx_rows]
        R = grl_reward(sdata, q.depth, ctx, frozenset(gold), w, senv.limits.T_max).total if ctx else 0.0
        r5, a5 = _m(ctx[:5], gold)
        out = {"R": R, "recall5": r5, "all5": a5, "k": len(ctx)}
        if fused is not None:
            fill = _fill10([ids[c] for c in ctx], fused, STRUCT_PREFIX)
            g_ids = set(r["C_ref"])
            out["recall10"], out["all10"] = _m(fill, g_ids)
        return out

    per = {}                                          # item id -> metrics
    for r in rows:
        k = r["id"]
        q = senv.query(ranks.get(f"{k}|q"))
        fb = agent[k]["context"] if k in agent else []
        rule = amend_lookup(index, q, sdata.chunk_tokens, k=a.k, l_max=L_max, fallback=fb)
        qual = named_act_root(index, q.v0) >= 0
        d = {"split": r["split"], "group": _group(r["pattern"]), "qualified": qual,
             "template": (r.get("provenance") or {}).get("m7", {}).get("template"),
             "rule": evaluate(r, rule, q, fused=bl["top"][k]["fused"] if k in bl["top"] else None)}
        # diagnostic: evidence variant (R60 circular)
        root = named_act_root(index, q.v0)
        cand = sorted(set(ev_rows.get(root, [])), key=lambda c: (-float(q.chunk_qsim[c]), c))[:a.k] if qual else fb
        d["diag_evidence"] = evaluate(r, cand, q)
        per[k] = d

    def agg(keys, f, metrics=("R", "recall5", "all5")):
        return {m: mean_se([f(k)[m] for k in keys]) for m in metrics}

    q2 = lambda split: [k for k, d in per.items() if d["group"] == "version" and d["split"] == split]
    out = {"k": a.k, "L_max": L_max, "q2": {}, "system": {}, "paired": {}}
    # ---- Q2 per template -------------------------------------------------------------------------------------
    for split in ("val", "train"):
        for t in (None,) + TEMPLATES:
            keys = [k for k in q2(split) if t is None or per[k]["template"] == t]
            if not keys:
                continue
            row = {"n": len(keys), "rule": agg(keys, lambda k: per[k]["rule"]),
                   "rule_qualified_share": sum(per[k]["qualified"] for k in keys) / len(keys),
                   "diag_evidence": agg(keys, lambda k: per[k]["diag_evidence"])}
            if split == "val":
                def blm(name, kk):
                    def f(k):
                        g = set(val[k]["C_ref"])
                        r_, a_ = _m(bl["top"][k][name][:kk], g)
                        return {"recall": r_, "all": a_}
                    return f
                row["agent"] = {"R": mean_se([agent[k]["reward"] for k in keys]),
                                "recall5": mean_se([agent[k]["recall"] for k in keys]),
                                "all5": mean_se([float(agent[k]["recall"] >= 1 - 1e-12) for k in keys])}
                for kk in (5, 10):
                    f = blm("ce+fused:0.6", kk)
                    row[f"cefused{kk}"] = {"recall": mean_se([f(k)["recall"] for k in keys]),
                                           "all": mean_se([f(k)["all"] for k in keys])}
                row["rule_at10"] = {"recall10": mean_se([per[k]["rule"]["recall10"] for k in keys]),
                                    "all10": mean_se([per[k]["rule"]["all10"] for k in keys])}
            out["q2"][f"{split}/{t or 'all'}"] = row
    # ---- router system (val) ---------------------------------------------------------------------------------
    def ag(k):
        r = agent[k]
        return {"R": r["reward"], "all5": float(r["recall"] >= 1 - 1e-12), "recall5": r["recall"]}

    def rk(k):
        r = ranker[k]
        return {"R": r["R"], "all5": r["all5"], "recall5": r["recall5"]}
    sysf = {"p8_router": lambda k: rk(k) if routes[k] == "old" else ag(k),
            "router_q2_rule": lambda k: rk(k) if routes[k] == "old" else (per[k]["rule"] if routes[k] == "version" else ag(k)),
            "p8_router_agent_only_q2": lambda k: ag(k)}
    vv = {k: {n: f(k) for n, f in sysf.items()} for k in val}
    for n in ("p8_router", "router_q2_rule"):
        out["system"][n] = {g: {m: mean_se([vv[k][n][m] for k in val if g == "all" or _group(val[k]["pattern"]) == g])
                                for m in ("R", "recall5", "all5")} for g in GROUPS + ("all",)}
    for g in GROUPS + ("all",):
        out["paired"][g] = {m: mean_se([vv[k]["router_q2_rule"][m] - vv[k]["p8_router"][m] for k in val
                                        if g == "all" or _group(val[k]["pattern"]) == g]) for m in ("R", "all5")}
    out["routed_to_q2"] = {"n": sum(routes[k] == "version" for k in val),
                           "of_which_old_misrouted": sum(routes[k] == "version" and _group(val[k]["pattern"]) == "old"
                                                         for k in val),
                           "no_qualifying_root_fallback_to_agent": sum(routes[k] == "version" and not per[k]["qualified"]
                                                                       for k in val)}
    out["per_item"] = per
    _write(a.out, out)
    for key, v in out["q2"].items():
        print(key, "n", v["n"], "rule all5 %.3f rec5 %.3f R %.3f" % tuple(v["rule"][m]["mean"] for m in ("all5", "recall5", "R")),
              "| agent all5 %.3f" % v["agent"]["all5"]["mean"] if "agent" in v else "",
              "| cef5 all %.3f cef10 all %.3f" % (v["cefused5"]["all"]["mean"], v["cefused10"]["all"]["mean"]) if "agent" in v else "",
              "| diag-evidence all5 %.3f" % v["diag_evidence"]["all5"]["mean"])
    for n, t in out["system"].items():
        print(n, {g: (t[g]["R"]["mean"], t[g]["all5"]["mean"]) for g in t})
    print(out["paired"]["all"], out["routed_to_q2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
