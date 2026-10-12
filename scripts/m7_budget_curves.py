"""M7 budget-matched comparison of the frozen system (R62) with the baselines: recall@k / all@k curves for k = 1..20 and
a per-question matched budget (each baseline gets exactly as many fragments as the system put into the context).
Descriptive only: the frozen system and the baselines are fixed, nothing is selected here (no k, no weights).

    PYTHONPATH=. python scripts/m7_budget_curves.py --split val    # f9-dry-val2 run (val rows)
    PYTHONPATH=. python scripts/m7_budget_curves.py --split test   # f9-test run (the opened test; no selection)
        -> data/m7/final/budget_<split>.json, docs/m7/figures/budget_curves_<split>.png

System context per question (as in m7_final_eval.build_methods): router old -> RL ranker top-5; count (Q1/Q1x) -> struct
agent s01 (K_max 5); version (Q2) -> the 6d-a rule (fallback the agent). For k beyond its own context the system curve is
filled with the fused ranking (the @10 convention of m7_pilot9._fill10); for k below it, its selection order is cut."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from graphrag import config  # noqa: E402
from scripts.m7_final_eval import FROZEN_SEED, SHIP, _jsonl, _rule_contexts, load_router, route  # noqa: E402
from scripts.m7_pilot9 import BASELINES, fine_group  # noqa: E402

RUNS = {"val": "runs/spark/f9-dry-val2", "test": "runs/spark/f9-test"}
TEST_EVAL = "data/m7/full/f9test-d43805b09c2e/ship/test_eval"
KS = list(range(1, 21))
GROUPS = ("all", "old", "hard", "Q1", "Q1x", "Q2")
LABEL = {"dense": "dense", "bm25": "BM25", "fused": "fused (BM25+dense)", "ce+fused:0.6": "ce+fused (reranker)",
         "system": "наша система"}


def rows_of(split):
    if split == "val":
        allr = {r["id"]: r for r in _jsonl(config.path(SHIP) / "pairs.jsonl")}
        keys = sorted(k for k, r in allr.items() if r["split"] == "val")
        return {k: allr[k] for k in keys}, keys, config.path(SHIP) / "rankings" / "latest.json"
    te = config.path(TEST_EVAL)
    allr = {r["id"]: r for r in _jsonl(te / "test_eval.jsonl")}
    keys = sorted(allr)
    return allr, keys, te / "rankings" / "latest.json"


def in_group(r, g) -> bool:
    fg = fine_group(r)
    return g == "all" or fg == g or (g == "hard" and fg != "old")


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--kmax", type=int, default=5, help="system budget K_max (5 = the frozen system; >5: the agent's "
                    "K_max-24 run cut to K_max in selection order, the rule with k = K_max, the ranker top-5 filled "
                    "with ce+fused up to K_max); baselines are compared at k = K_max")
    ap.add_argument("--agent", help="override the count-branch agent: a per-query eval file of the same split and K_max "
                    "(e.g. another weight vector / seed); the ranker and the rule stay as in the frozen system")
    ap.add_argument("--tag", default="", help="suffix of the output json (with --agent)")
    a = ap.parse_args(argv)
    K = a.kmax
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    run = config.path(RUNS[a.split])
    rows, keys, rankings = rows_of(a.split)
    sdata = load_struct_env_data(articles=True)
    ids, tok = sdata.chunk_ids, sdata.chunk_tokens
    row_of = {c: i for i, c in enumerate(ids)}
    ranks = load_rankings(rankings)
    routes = dict(zip(keys, route(load_router(), np.stack([np.asarray(ranks.get(f"{k}|q").qvec, np.float32)
                                                           for k in keys]))))
    bl = json.loads((run / "baselines.json").read_text(encoding="utf-8"))["top"]
    rk = {x["qkey"][:-2]: x for x in _jsonl(run / "ranker.jsonl")}
    exact = config.path(f"runs/spark/k{K}-{a.split}")          # exact K_max runs (ranker top-K, agent at K_max = K)
    exact_ok = K != 5 and (exact / "ranker.jsonl").is_file() and (exact / f"agent-{FROZEN_SEED}-k{K}.jsonl").is_file()
    if exact_ok:
        rk = {x["qkey"][:-2]: x for x in _jsonl(exact / "ranker.jsonl")}
        ag = {x["qkey"][:-2]: x for x in _jsonl(exact / f"agent-{FROZEN_SEED}-k{K}.jsonl")}
    else:
        ag = {x["qkey"][:-2]: x for x in _jsonl(run / f"agent-{FROZEN_SEED}-k{5 if K == 5 else 24}.jsonl")}
    if a.agent:
        ag = {x["qkey"][:-2]: x for x in _jsonl(config.path(a.agent))}
        if set(ag) != set(keys):
            raise SystemExit(f"{a.agent}: does not cover exactly the {a.split} rows")
    print(f"K_max={K}: {'EXACT runs ' + str(exact) if exact_ok else ('frozen K=5 runs' if K == 5 else 'APPROXIMATION (agent K24 cut, ranker top-5 + ce+fused fill)')}"
          + (f"; agent = {a.agent}" if a.agent else ""))
    agc = {k: list(dict.fromkeys(ag[k]["context"]))[:K] for k in keys}
    rule = _rule_contexts([rows[k] for k in keys], rankings, {FROZEN_SEED: agc}, k=K)[FROZEN_SEED]
    sys_ctx = {}
    for k in keys:
        if routes[k] == "old":
            top = [ids[int(c)] for c in rk[k]["context_top5"]]
            ctx = list(dict.fromkeys(top + (list(bl[k]["ce+fused:0.6"]) if K > 5 and not exact_ok else [])))[:K]
        else:
            ctx = list(dict.fromkeys(ids[int(c)] for c in (agc[k] if routes[k] == "count" else rule[k][0])))[:K]
        sys_ctx[k] = ctx

    def fill(ctx, k):                                   # own selection, then the fused ranking (no duplicates)
        return list(dict.fromkeys(list(ctx) + list(bl[k]["fused"])))

    lists = {"system": {k: fill(sys_ctx[k], k) for k in keys}}
    for b in BASELINES:
        lists[b] = {k: list(bl[k][b]) for k in keys}

    def rec(lst, gold, n):
        s = set(lst[:n])
        return len(s & gold) / len(gold), float(gold <= s), sum(int(tok[row_of[c]]) for c in lst[:n] if c in row_of)

    out = {"split": a.split, "n": len(keys), "curves": {}, "matched": {}, "system_k": {}}
    for g in GROUPS:
        ks_g = [k for k in keys if in_group(rows[k], g)]
        if not ks_g:
            continue
        out["system_k"][g] = mean_se([len(sys_ctx[k]) for k in ks_g])
        out["curves"][g] = {m: {met: [mean_se([rec(lists[m][k], set(rows[k]["C_ref"]), n)[i] for k in ks_g])[0]
                                      for n in KS] for i, met in ((0, "recall"), (1, "all"))} for m in lists}
        # matched budget: every method gets k_q = |system context| fragments of its own ranking
        M = {}
        for m in lists:
            r = [rec(lists[m][k], set(rows[k]["C_ref"]), len(sys_ctx[k])) for k in ks_g]
            M[m] = {"recall": mean_se([x[0] for x in r]), "all": mean_se([x[1] for x in r]),
                    "tokens": mean_se([x[2] for x in r])}
        for b in BASELINES:                               # paired system - baseline, same budget per question
            d = [rec(lists["system"][k], set(rows[k]["C_ref"]), len(sys_ctx[k]))[0]
                 - rec(lists[b][k], set(rows[k]["C_ref"]), len(sys_ctx[k]))[0] for k in ks_g]
            M[b]["d_recall_sys_minus"] = mean_se(d)
        out["matched"][g] = {"n": len(ks_g), **M}
        # K_max table: the system's own adaptive context (K_min..K_max) vs every baseline at the full K_max = 5
        kq = np.array([len(sys_ctx[k]) for k in ks_g])
        T = {"k_min": int(kq.min()), "k_max": int(kq.max()), "k_mean": float(kq.mean()), "k_median": float(np.median(kq)),
             "k_hist": {int(v): int((kq == v).sum()) for v in np.unique(kq)}}
        own = [rec(lists["system"][k], set(rows[k]["C_ref"]), len(sys_ctx[k])) for k in ks_g]
        T["system"] = {"recall": mean_se([x[0] for x in own]), "all": mean_se([x[1] for x in own]),
                       "tokens": mean_se([x[2] for x in own])}
        for b in BASELINES:
            r5 = [rec(lists[b][k], set(rows[k]["C_ref"]), K) for k in ks_g]
            T[b] = {"recall": mean_se([x[0] for x in r5]), "all": mean_se([x[1] for x in r5]),
                    "tokens": mean_se([x[2] for x in r5]),
                    "d_all_sys_minus": mean_se([o[1] - x[1] for o, x in zip(own, r5)]),
                    "d_recall_sys_minus": mean_se([o[0] - x[0] for o, x in zip(own, r5)])}
        out.setdefault("kmax", {})[g] = {"n": len(ks_g), **T}
    out["K_max"] = K
    out["exact_runs"] = bool(exact_ok)
    out["agent"] = a.agent
    dst = config.path(f"data/m7/final/budget_{a.split}{'' if K == 5 else f'_k{K}'}{'_' + a.tag if a.tag else ''}.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, M in out["matched"].items():
        kk = out["system_k"][g][0]
        print(f"[{a.split} {g} n={M['n']} system k={kk:.2f}] " + "  ".join(
            f"{m}: rec {M[m]['recall'][0]:.3f} all {M[m]['all'][0]:.3f} tok {M[m]['tokens'][0]:.0f}" for m in lists))
    for g in ("all", "old", "hard"):
        c = out["curves"][g]
        print(f"[{a.split} {g}] recall@k k=1,2,3,5,10,20: " + " | ".join(
            f"{m}: " + ",".join(f"{c[m]['recall'][n - 1]:.2f}" for n in (1, 2, 3, 5, 10, 20)) for m in lists))
    for g, T in out["kmax"].items():
        print(f"[{a.split} K_max={K} {g} n={T['n']}] k {T['k_min']}..{T['k_max']} mean {T['k_mean']:.2f} median "
              f"{T['k_median']:.0f} hist {T['k_hist']} | system rec {T['system']['recall'][0]:.3f} all "
              f"{T['system']['all'][0]:.3f} tok {T['system']['tokens'][0]:.0f} | " + "  ".join(
                  f"{b}@{K} rec {T[b]['recall'][0]:.3f} all {T[b]['all'][0]:.3f} tok {T[b]['tokens'][0]:.0f} "
                  f"Δall {T[b]['d_all_sys_minus'][0]:+.3f}±{T[b]['d_all_sys_minus'][1]:.3f}" for b in BASELINES))
    if K == 5 and not a.agent:
        plot(out, a.split)
    return 0


def plot(out, split):
    from scripts.plot_m7_calib_figures import BLUE, INK, INK2, MUTED, ORANGE, plt, save
    col = {"dense": "#8a8984", "bm25": "#b8b6ae", "fused": ORANGE, "ce+fused:0.6": "#1baf7a", "system": BLUE}
    groups = [g for g in ("all", "old", "hard") if g in out["curves"]]
    fig, axs = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), sharey=True)
    title = {"all": "все вопросы", "old": "обычные вопросы", "hard": "сложные (Q1 + Q1x + Q2)"}
    for ax, g in zip(np.atleast_1d(axs), groups):
        c = out["curves"][g]
        for m in ("bm25", "dense", "fused", "ce+fused:0.6", "system"):
            ax.plot(KS, c[m]["recall"], color=col[m], lw=3 if m == "system" else 1.8,
                    label=LABEL[m], zorder=5 if m == "system" else 2)
        kk = out["system_k"][g][0]
        y = out["matched"][g]["system"]["recall"][0]
        ax.scatter([kk], [y], s=160, marker="o", color=BLUE, edgecolors=INK, zorder=6)
        ax.annotate(f"собственный контекст\nсистемы: k̄ = {kk:.1f}".replace(".", ","), (kk, y), xytext=(14, -38),
                    textcoords="offset points", fontsize=10, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": MUTED})
        ax.axvline(5, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{title[g]} (n = {out['matched'][g]['n']})", fontsize=13)
        ax.set_xlabel("k — число фрагментов в контексте LLM", fontsize=11)
        ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axs)[0].set_ylabel("recall@k", fontsize=12)
    np.atleast_1d(axs)[0].legend(loc="lower right", fontsize=10)
    fig.suptitle(f"recall@k: наша система против бейзлайнов ({'val' if split == 'val' else 'тест'})",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, f"budget_curves_{split}.png")


if __name__ == "__main__":
    sys.exit(main())
