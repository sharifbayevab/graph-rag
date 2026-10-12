"""M7 Task 8: probe metrics at k = 5 (docs/m7 design §4, §5 step 2) and the oracle diagnostics (§8 risks).
Ruling R2: here, not in scripts/ (not a package); scripts/hard_probe_eval.py and the tests import it.

- set_metrics(context, gold): recall@5 = |ctx ∩ gold| / |gold|, all@5 = gold ⊆ ctx, on fragment ids (a context is a
  method's first 5 ids, or the oracle's final context at K_max = 5).
- mean ± SE = sd(ddof = 1) / sqrt(n) (scripts/rank_report.py), per question type and pooled Q1–Q3.
- go_numbers: the spec rule's numbers -- GO iff the best baseline has all@5 < 0.5 on at least two types (per type the
  best of the baselines, the conservative reading), plus the oracle all@5 per type (< 0.5: the graph does not reach
  the gold on that type). The numbers only; the user decides.
- oracle_record / oracle_diagnostics: per episode the path (action names + added ids), the number of retrieval steps
  (STOP excluded), whether `structure` / an AMENDS hop was used, and whether the needed EXT act was reached:
  ext_covered = a vertex of the act (root = the EXT vertex) is covered by the final state, ext_touched = covered or
  on the final frontier (its STRUCT fragment is then a `structure` candidate)."""
from __future__ import annotations

import math

TYPES = ("count", "version", "compare")
GO_THRESHOLD = 0.5
GO_MIN_TYPES = 2


def set_metrics(context, gold) -> dict:
    g = set(gold)
    if not g:
        raise ValueError("set_metrics needs a non-empty gold")
    hit = len(set(context) & g)
    return {"recall": hit / len(g), "all": hit == len(g)}


def anyof_metrics(context, gold, alt_gold=None) -> dict:
    """R59 (Task 6c fix round 1): strict = set_metrics on the STRUCT gold; any-of all@5 = the gold OR the item's
    alternative evidence set (alt_gold: every corpus chunk of the article, which together hold every part / point) is
    wholly in the context. Items without alt_gold: any-of = strict."""
    strict = set_metrics(context, gold)
    alt = set(alt_gold or ())
    return {"strict_all": strict["all"], "strict_recall": strict["recall"],
            "anyof_all": strict["all"] or (bool(alt) and alt <= set(context))}


def top_k_ids(ids, k: int = 5) -> list:
    """The first k distinct ids of an order."""
    out, seen = [], set()
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
        if len(out) == k:
            break
    return out


def mean_se(xs) -> tuple:
    xs = [float(x) for x in xs]
    n = len(xs)
    if not n:
        return None, None
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def _cell(rows) -> dict:
    r, r_se = mean_se([x["recall"] for x in rows])
    a, a_se = mean_se([float(x["all"]) for x in rows])
    return {"n": len(rows), "recall": r, "recall_se": r_se, "all": a, "all_se": a_se}


def summarize_methods(rows) -> dict:
    """{method: {type | "pooled": {n, recall, recall_se, all, all_se}}} from per-query rows {method, qtype, qid,
    recall, all}; a method may score a qid once only."""
    by: dict = {}
    for x in rows:
        by.setdefault(x["method"], []).append(x)
    out = {}
    for m, rs in by.items():
        q = [x["qid"] for x in rs]
        if len(q) != len(set(q)):
            raise ValueError(f"{m}: a qid is scored more than once")
        out[m] = {t: _cell([x for x in rs if x["qtype"] == t]) for t in TYPES}
        out[m]["pooled"] = _cell([x for x in rs if x["qtype"] in TYPES])
    return out


def go_numbers(table: dict, baselines, oracle: str) -> dict:
    best, name = {}, {}
    for t in TYPES:
        cand = [(table[b][t]["all"], b) for b in baselines if b in table and table[b][t]["all"] is not None]
        if not cand:
            raise ValueError(f"go_numbers: no baseline of {list(baselines)} has an all@5 for type {t!r}")
        v = max(x[0] for x in cand)
        tied = [b for x, b in cand if x == v]
        best[t] = v
        name[t] = tied[0] if len(tied) == 1 else ("tie(all 0)" if v == 0 else tied)
    below = [t for t in TYPES if best[t] < GO_THRESHOLD]
    oa = {t: table[oracle][t]["all"] for t in TYPES}
    return {"threshold": GO_THRESHOLD, "min_types": GO_MIN_TYPES, "best_baseline_all": best,
            "best_baseline_name": name, "types_below": below, "rule_go": len(below) >= GO_MIN_TYPES,
            "oracle_all": oa, "oracle_below": [t for t in TYPES if oa[t] < GO_THRESHOLD]}


def _action_name(a) -> str:
    if a.op in ("structure", "amendments"):          # + Task 6d-b amend_op
        return f"{a.op}:{a.m}"
    if a.op == "graph":
        return f"graph:{a.r}:{a.m}"
    return f"{a.op}:{a.k}" if a.k is not None else a.op


def oracle_record(env, res: dict, ext_targets) -> dict:
    """Diagnostics of one expert_search result in a StructEnv (module docstring). ext_targets: node ids of the EXT
    acts the question needs ([] -> the EXT fields are None)."""
    d, s = env.data, res["final"]
    path = [{"action": _action_name(env.actions[h.action]), "added": [d.chunk_ids[int(c)] for c in h.added]}
            for h in s.history if env.actions[h.action].op != "stop"]
    out = {"context": [d.chunk_ids[int(c)] for c in s.context], "path": path, "n_steps": len(path),
           "used_structure": any(p["action"].startswith("structure:") for p in path),
           "used_amends": any(p["action"].startswith("graph:AMENDS:") for p in path),
           "score": float(res["score"]), "exact": bool(res["exact"]), "capped": bool(res.get("capped")),
           "n_states": int(res["n_states"]), "ext_covered": None, "ext_touched": None}
    tg = {d.node_row[x] for x in ext_targets if x in d.node_row}
    if ext_targets:
        root = env._root_row
        cov = {int(root[v]) for v in s.covered}
        near = cov | {int(root[v]) for v in s.frontier}
        out["ext_covered"], out["ext_touched"] = bool(tg & cov), bool(tg & near)
    return out


def _share(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _diag(rs) -> dict:
    return {"n": len(rs), "all": _share([r["all"] for r in rs]),
            "structure_share": _share([r["used_structure"] for r in rs]),
            "amends_share": _share([r["used_amends"] for r in rs]),
            "ext_covered_share": _share([r["ext_covered"] for r in rs]),
            "ext_touched_share": _share([r["ext_touched"] for r in rs]),
            "steps_mean": _share([r["n_steps"] for r in rs]),
            "exact_share": _share([r["exact"] for r in rs]), "capped_share": _share([r["capped"] for r in rs]),
            "seconds_mean": _share([r["seconds"] for r in rs]),
            "seconds_max": max((float(r["seconds"]) for r in rs), default=None)}


def oracle_diagnostics(recs) -> dict:
    out = {t: _diag([r for r in recs if r["qtype"] == t]) for t in TYPES}
    out["pooled"] = _diag([r for r in recs if r["qtype"] in TYPES])
    return out
