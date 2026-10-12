"""M7 pilot 9 (validation only; the test splits stay sealed): one combined relabel + SFT + GRPO on the joint ship with
article STRUCT rows (Task 6c, struct.articles) and the «amendments of act» operator (Task 6d-b, struct.amend_op, 34
actions). The Spark side reuses train.py (expert / sft / grpo / eval) and scripts/m7_pilot8.py (ranker --top5,
baselines, accum --set); the router is m7_pilot8.py router (refit on the new train rows), the Q2 rule is
scripts/m7_router_q2rule.py. This script only reads their outputs (Mac, offline).

    relabel --labels runs/spark/p9-expert/labels.jsonl --p8 runs/spark/p8-expert/labels.jsonl --out F
      expert reach (= all@5 at K_max 5), limit-final share, SFT-filtered rows, operator use, per group / template;
      against the pilot-8 labels on the shared ids (gains / losses).
    tables --ship S --baselines BL --ranker PQ --agent name=PQ ... --train-agent PQ --router R --q2rule-p8 J
           --q2rule J --out F
      per method and group (old / Q1 / Q1x per template / Q2 per template): R, recall@5, all@5, k, all@10 (+ fused
      fill), limit share; Q1x strict and any-of (R59) all@5, easy / non-easy; the router systems (a)-(d) with the
      old branch scored at the ranker's top-5 for @5 (top-1 column kept); the best single baseline per group with
      paired SE. System (d) picks agent vs rule for Q2 by TRAIN all@5 only."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402
from scripts.m7_pilot8 import _fill10, _group, _jsonl, _write, mean_se  # noqa: E402

Q1X = ("count_parts_article", "count_points_article", "count_codes_by_law", "count_laws_amended_code_year",
       "count_articles_deleted_by_law")
ARTICLE = ("count_parts_article", "count_points_article")
Q2T = ("what_law_changed", "which_deleted_year", "which_law_part", "which_laws_article")
BASELINES = ("dense", "bm25", "fused", "ce+fused:0.6")
AMEND_IDS = (31, 32, 33)
EPS = 1e-9


def _tpl(r) -> str | None:
    return ((r.get("provenance") or {}).get("m7") or {}).get("template")


def fine_group(r) -> str:
    """old / Q1 (the pilot-7 count templates) / Q1x (Task 6c templates) / Q2."""
    g = _group(r["pattern"])
    if g == "count":
        return "Q1x" if _tpl(r) in Q1X else "Q1"
    return "Q2" if g == "version" else "old"


def subgroups(r) -> list[str]:
    """Every reporting cell a row belongs to (all, its fine group, count = Q1 + Q1x, its template cell)."""
    fg, t = fine_group(r), _tpl(r)
    out = ["all", fg]
    if fg != "old":
        out.append("hard")                           # Task 9f: the hard set of the spec §1 criterion (Q1 + Q1x + Q2)
    if fg in ("Q1", "Q1x"):
        out.append("count")
    if fg in ("Q1x", "Q2"):
        out.append(f"{fg}/{t}")
    if fg == "Q1x" and t in ARTICLE:
        easy = bool(r["provenance"]["m7"].get("easy"))
        out += [f"Q1x/{t}/{'easy' if easy else 'non-easy'}", f"Q1x/article/{'easy' if easy else 'non-easy'}"]
    return out


CELL_ORDER = (["all", "old", "hard", "count", "Q1", "Q1x"] + [f"Q1x/{t}" for t in Q1X]
              + [f"Q1x/{t}/{e}" for t in ARTICLE for e in ("easy", "non-easy")]
              + ["Q1x/article/easy", "Q1x/article/non-easy", "Q2"] + [f"Q2/{t}" for t in Q2T])


def _cells(rows, keys, f, metrics):
    by = defaultdict(list)
    for k in keys:
        for c in subgroups(rows[k]):
            by[c].append(k)
    return {c: {"n": len(by[c]), **{m: mean_se([f(k)[m] for k in by[c]]) for m in metrics}}
            for c in CELL_ORDER if by.get(c)}


# --- relabel -------------------------------------------------------------------------------------------------------

def _exp(r) -> dict:
    return r["provenance"]["meta"]["expert"]


def _filtered(r) -> bool:
    """grl.train.data.sft_train_filter: score <= eps, an immediate STOP, or C* not reached."""
    e = _exp(r)
    stop1 = len(r["actions"]) == 1 and r["actions"][0] == 0
    return float(e["score"]) <= EPS or stop1 or not e["reached"]


def cmd_relabel(a) -> int:
    new = {r["id"]: r for r in _jsonl(a.labels)}
    old = {r["id"]: r for r in _jsonl(a.p8)}
    meta = json.loads(Path(str(a.labels).replace(".jsonl", ".stats.json")).read_text(encoding="utf-8"))

    def rec(r):
        e = _exp(r)
        return {"reach": float(bool(e["reached"])), "limit": float(e.get("final_by") == "limit"),
                "score": float(e["score"]), "filtered": float(_filtered(r)), "len": float(e["len"]),
                "amend": float(any(x in AMEND_IDS for x in r["actions"])), "capped": float(bool(e.get("capped"))),
                "n_states": float(e.get("n_states") or 0)}
    out = {"env_kw": meta.get("env_kw"), "K_max": meta.get("K_max"), "depth": meta.get("depth"),
           "seconds": meta.get("seconds"), "splits": {}, "vs_p8": {}}
    ms = ("reach", "limit", "score", "filtered", "len", "amend", "capped", "n_states")
    for split in ("train", "val"):
        keys = [k for k, r in new.items() if r["split"] == split]
        out["splits"][split] = _cells(new, keys, lambda k: rec(new[k]), ms)
    shared = [k for k in new if k in old]
    for split in ("train", "val"):
        keys = [k for k in shared if new[k]["split"] == split]
        f = lambda k: {"reach_p8": float(bool(_exp(old[k])["reached"])), "reach_p9": float(bool(_exp(new[k])["reached"])),
                       "gain": float(bool(_exp(new[k])["reached"]) and not _exp(old[k])["reached"]),
                       "loss": float(bool(_exp(old[k])["reached"]) and not _exp(new[k])["reached"]),
                       "dscore": float(_exp(new[k])["score"]) - float(_exp(old[k])["score"]),
                       "limit_p8": float(_exp(old[k]).get("final_by") == "limit"),
                       "limit_p9": float(_exp(new[k]).get("final_by") == "limit"),
                       "same_traj": float(new[k]["actions"] == old[k]["actions"])}
        out["vs_p8"][split] = _cells(new, keys, f, ("reach_p8", "reach_p9", "gain", "loss", "dscore", "limit_p8",
                                                     "limit_p9", "same_traj"))
    out["new_only"] = sorted(Counter(fine_group(new[k]) for k in new if k not in old).items())
    out["sft_train_kept"] = sum(1 for r in new.values() if r["split"] == "train" and not _filtered(r))
    _write(a.out, out)
    for split in ("train", "val"):
        for c, v in out["splits"][split].items():
            print(split, c, v["n"], "reach %.3f limit %.3f filt %.3f amend %.3f" % tuple(
                v[m]["mean"] for m in ("reach", "limit", "filtered", "amend")))
    return 0


# --- tables --------------------------------------------------------------------------------------------------------

def _metrics(ctx_ids, r, fused, R, reason=None) -> dict:
    """@5 / @10 (+ fused fill) / dynamic metrics of one context (chunk ids in selection order)."""
    from grl.hard.metrics import anyof_metrics
    from grl.hard.probe import STRUCT_PREFIX
    gold = set(r["C_ref"])
    top5 = list(dict.fromkeys(ctx_ids))[:5]
    S5 = set(top5)
    fill = _fill10(ctx_ids, fused, STRUCT_PREFIX)
    alt = ((r.get("provenance") or {}).get("m7") or {}).get("alt_gold")
    am = anyof_metrics(top5, gold, alt)
    return {"R": R, "recall5": len(S5 & gold) / len(gold), "all5": float(gold <= S5),
            "recall10": len(set(fill) & gold) / len(gold), "all10": float(gold <= set(fill)),
            "anyof5": float(am["anyof_all"]), "k": float(len(ctx_ids)),
            "limit": float(reason in ("k_max", "l_max")) if reason is not None else 0.0}


METRICS = ("R", "recall5", "all5", "recall10", "all10", "anyof5", "k", "limit")


def _paired(rows, keys, fa, fb, metrics=("R", "all5", "all10", "anyof5")):
    return _cells(rows, keys, lambda k: {m: fa(k)[m] - fb(k)[m] for m in metrics}, metrics)


def cmd_tables(a) -> int:
    from grl.env.struct_data import load_struct_env_data
    ship = config.path(a.ship)
    rows = {r["id"]: r for r in _jsonl(ship / "pairs.jsonl")}
    if {r["split"] for r in rows.values()} - {"train", "val"}:
        raise SystemExit("ship holds rows outside train / val")
    val = [k for k, r in rows.items() if r["split"] == "val"]
    train = [k for k, r in rows.items() if r["split"] == "train"]
    ids = load_struct_env_data(articles=True).chunk_ids
    bl = json.loads(Path(a.baselines).read_text(encoding="utf-8"))
    if sorted(bl["top"]) != sorted(val):
        raise SystemExit("baselines do not cover exactly the val rows")
    fused = {k: bl["top"][k]["fused"] for k in val}
    M = {}                                                           # method -> id -> metrics

    def load_agent(path, keys):
        per = {}
        for x in _jsonl(path):
            per[x["qkey"][:-2]] = x
        if not set(keys) <= set(per):
            raise SystemExit(f"{path}: does not cover the rows")
        return per
    for spec in a.agent:
        name, path = spec.split("=", 1)
        per = load_agent(path, val)
        M[name] = {k: _metrics([ids[int(c)] for c in per[k]["context"]], rows[k], fused[k], per[k]["reward"],
                               per[k]["reason"]) for k in val}
    rk = load_agent(a.ranker, val)
    if any("context_top5" not in rk[k] for k in val):
        raise SystemExit(f"{a.ranker}: no context_top5 (run m7_pilot8.py ranker --top5)")
    M["ranker_top1"] = {k: _metrics([ids[int(c)] for c in rk[k]["context"]], rows[k], fused[k], rk[k]["R"],
                                    rk[k]["reason"]) for k in val}
    M["ranker_top5"] = {k: {**_metrics([ids[int(c)] for c in rk[k]["context_top5"]], rows[k], fused[k], rk[k]["R"]),
                            "R": rk[k]["R"], "k": 1.0, "limit": 0.0} for k in val}   # R / k: the top-1 run's
    for b in BASELINES:
        for kk in (1, 2, 5):                         # @kk set metrics in the *5 fields; @10 (top-10) in the *10 fields
            M[f"bl:{b}@{kk}"] = {}
            for k in val:
                top = bl["top"][k][b]
                m, m10 = _metrics(top[:kk], rows[k], [], None), _metrics(top[:10], rows[k], [], None)
                M[f"bl:{b}@{kk}"][k] = {**m, "R": 0.0, "recall10": m10["recall10"], "all10": m10["all10"]}
    rules = {}
    for tag, path in (("rule_p8", a.q2rule_p8), ("rule", a.q2rule)):
        q = json.loads(Path(path).read_text(encoding="utf-8"))["per_item"]
        rules[tag] = q
        M[tag] = {k: {"R": q[k]["rule"]["R"], "recall5": q[k]["rule"]["recall5"], "all5": q[k]["rule"]["all5"],
                      "recall10": q[k]["rule"]["recall10"], "all10": q[k]["rule"]["all10"],
                      "anyof5": q[k]["rule"]["all5"], "k": float(q[k]["rule"]["k"]), "limit": 0.0} for k in val}
    # ---- router + systems ----------------------------------------------------------------------------------------
    router = json.loads(Path(a.router).read_text(encoding="utf-8"))
    routes = router["routes"]
    if set(routes) != set(val):
        raise SystemExit("router routes do not cover exactly the val rows")
    tr_agent = load_agent(a.train_agent, train)
    q2_train = [k for k in train if fine_group(rows[k]) == "Q2"]
    choice = {"agent_train_all5": mean_se([float(tr_agent[k]["recall"] >= 1 - 1e-12) for k in q2_train]),
              "rule_train_all5": mean_se([rules["rule"][k]["rule"]["all5"] for k in q2_train]),
              "n_train_q2": len(q2_train)}
    choice["pick"] = "rule" if choice["rule_train_all5"]["mean"] > choice["agent_train_all5"]["mean"] else "agent"
    for t in Q2T:
        kt = [k for k in q2_train if _tpl(rows[k]) == t]
        choice[t] = {"n": len(kt), "agent": mean_se([float(tr_agent[k]["recall"] >= 1 - 1e-12) for k in kt]),
                     "rule": mean_se([rules["rule"][k]["rule"]["all5"] for k in kt])}
    th, p8 = a.theta, a.p8_theta
    q2pick = "rule" if choice["pick"] == "rule" else th
    spec = {"a_p8": (p8, p8), "b_p8_rule": (p8, "rule_p8"), "c_agent": (th, th), "d_agent_q2best": (th, q2pick)}
    systems = {}
    for name, (cnt, ver) in spec.items():
        for old_src, sfx in (("ranker_top5", ""), ("ranker_top1", "_top1")):
            src = {"old": old_src, "count": cnt, "version": ver}
            systems[name + sfx] = {k: M[src[routes[k]]][k] for k in val}
    for sname, sv in systems.items():
        M["sys:" + sname] = sv
    # oracle router (true type) for (d)
    M["sys:d_oracle_route"] = {k: M[{"old": "ranker_top5", "count": th, "version": q2pick}[_group(rows[k]["pattern"])]][k]
                               for k in val}
    # ---- aggregate -------------------------------------------------------------------------------------------------
    vrows = {k: rows[k] for k in val}
    table = {name: _cells(vrows, val, lambda k, m=m: m[k], METRICS) for name, m in M.items()}
    paired = {
        "theta_minus_sft": _paired(vrows, val, lambda k: M[th][k], lambda k: M[a.sft][k]),
        "theta_minus_p8theta": _paired(vrows, val, lambda k: M[th][k], lambda k: M[p8][k]),
        "theta_minus_rule": _paired(vrows, val, lambda k: M[th][k], lambda k: M["rule"][k]),
        "b_minus_a": _paired(vrows, val, lambda k: M["sys:b_p8_rule"][k], lambda k: M["sys:a_p8"][k]),
        "c_minus_a": _paired(vrows, val, lambda k: M["sys:c_agent"][k], lambda k: M["sys:a_p8"][k]),
        "d_minus_a": _paired(vrows, val, lambda k: M["sys:d_agent_q2best"][k], lambda k: M["sys:a_p8"][k]),
        "d_minus_b": _paired(vrows, val, lambda k: M["sys:d_agent_q2best"][k], lambda k: M["sys:b_p8_rule"][k]),
        "d_minus_c": _paired(vrows, val, lambda k: M["sys:d_agent_q2best"][k], lambda k: M["sys:c_agent"][k]),
        "d_minus_ranker_top5": _paired(vrows, val, lambda k: M["sys:d_agent_q2best"][k], lambda k: M["ranker_top5"][k]),
        "d_minus_theta": _paired(vrows, val, lambda k: M["sys:d_agent_q2best"][k], lambda k: M[th][k]),
    }
    # best single baseline per cell (by val mean all@5 at k = 5 / all10 at k = 10): a reporting comparator only
    best = {}
    for cell in CELL_ORDER:
        if cell not in table["sys:d_agent_q2best"]:
            continue
        keys = [k for k in val if cell in subgroups(rows[k])]
        best[cell] = {}
        for at, metric in (("@5", "all5"), ("@10", "all10")):
            cands = {b: table[f"bl:{b}@5"][cell][metric]["mean"] for b in BASELINES}
            bb = max(cands, key=lambda b: (cands[b], b == "ce+fused:0.6"))
            best[cell][at] = {"baseline": bb, "baseline_mean": cands[bb],
                              "system": table["sys:d_agent_q2best"][cell][metric],
                              "diff": mean_se([M["sys:d_agent_q2best"][k][metric] - M[f"bl:{bb}@5"][k][metric]
                                               for k in keys])}
            if cell.startswith("Q1x"):
                best[cell][at]["anyof_system"] = table["sys:d_agent_q2best"][cell]["anyof5"]
                best[cell][at]["anyof_baselines"] = {b: table[f"bl:{b}@5"][cell]["anyof5"]["mean"] for b in BASELINES}
    misrouted = {"to_version_old": sum(routes[k] == "version" and _group(rows[k]["pattern"]) == "old" for k in val),
                 "to_count_old": sum(routes[k] == "count" and _group(rows[k]["pattern"]) == "old" for k in val),
                 "q1x_to_old": sum(routes[k] == "old" and fine_group(rows[k]) == "Q1x" for k in val)}
    out = {"n": {c: v["n"] for c, v in table["sys:d_agent_q2best"].items()}, "table": table, "paired": paired,
           "best_baseline": best, "q2_choice_train": choice, "systems": {k: list(v) for k, v in spec.items()},
           "router": {**router["classifier"], "misrouted": misrouted},
           "inputs": {k: v for k, v in vars(a).items() if k not in ("func",)},
           "note": "old branch @5 = the ranker's top-5 (no-STOP greedy list); *_top1 = its 1-fragment pick; R and k "
                   "of the ranker always the top-1 run; baselines carry no R"}
    _write(a.out, out)
    for name in ("sft", th, p8, "rule", "ranker_top1", "ranker_top5", "bl:ce+fused:0.6@5", "sys:a_p8", "sys:b_p8_rule",
                 "sys:c_agent", "sys:d_agent_q2best", "sys:d_agent_q2best_top1"):
        if name in table:
            t = table[name]
            print(f"{name:26s}", " ".join(f"{c}:{t[c]['all5']['mean']:.3f}/{t[c]['R']['mean']:.3f}"
                                         for c in ("all", "old", "Q1", "Q1x", "Q2") if c in t))
    print("q2 choice (train):", json.dumps({k: choice[k] for k in ("pick", "agent_train_all5", "rule_train_all5")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relabel")
    r.add_argument("--labels", required=True)
    r.add_argument("--p8", default="runs/spark/p8-expert/labels.jsonl")
    r.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--ship", default="data/m7/full/54aa9a000b4b/ship")
    t.add_argument("--baselines", required=True)
    t.add_argument("--ranker", required=True, help="m7_pilot8.py ranker --top5 per-query file")
    t.add_argument("--agent", action="append", required=True, help="name=train.py eval --per-query file (val)")
    t.add_argument("--sft", default="sft")
    t.add_argument("--theta", default="theta")
    t.add_argument("--p8-theta", default="p8theta")
    t.add_argument("--train-agent", required=True, help="theta's per-query file on TRAIN (the (d) Q2 choice)")
    t.add_argument("--router", required=True)
    t.add_argument("--q2rule-p8", required=True, help="m7_router_q2rule.py output, fallback = p8 theta")
    t.add_argument("--q2rule", required=True, help="m7_router_q2rule.py output, fallback = the new theta")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"relabel": cmd_relabel, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
