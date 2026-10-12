"""M7 Task 6c fix round 1: the R58 diagnostic (where the gold STRUCT row ranks among the `structure` candidates, and
whether the struct expert still reaches it) and the R59 baseline table (strict vs any-of all@5) on the TRAIN / VAL Q1
items of a struct ship (old Q1 + the Task 6c items). Never reads a test row.

    PYTHONPATH=. python scripts/m7_q1x_diag.py --ship data/m7/full/<sha>/ship --ce-cache runs/<ce>/ce_scores.jsonl \\
        --set env.K_max=5 --workers 6 --out runs/<id>/diag.json        (Spark, one job)

R58 modes (grl.model.variants.StructEnv): off = the pilot 7 / 8 env (no article rows; old Q1 only), root = Phase A
(every article of the reached roots), reached = R58 (articles in F ∪ S), ancestor = reached + the article above every
vertex of F ∪ S. Per item and mode: rank0 = 1-based rank of the gold STRUCT row in struct_scores at reset (None: not
a candidate), n_cands0, best_rank = the best rank of a gold row along the expert's final path (before each step),
reached = every gold row in the expert's final context (expert_search: depth --depth, prune_zero False (R35),
limit finals (R34), train.yaml expert caps). Aggregates per (set old / new, template, split): shares of rank0 / best
rank within 1 / 3 / 5, reach share.

R59: the baselines dense / bm25 / fused / ce+fused:0.6 (grl.train.rank.baseline_ranking over the pilot-6 RankEnv
pool, scripts/m7_pilot8.py RANK_KW) @5, strict all@5 (STRUCT gold) vs any-of all@5 (gold OR the item's alt_gold, the
article's chunks; grl.hard.metrics.anyof_metrics), per template x easy / non-easy x split. The agent's numbers come
in Phase B (no article-mode agent is trained yet)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()

from graphrag import config  # noqa: E402

MODES = ("off", "root", "reached", "ancestor")
SPLITS = ("train", "val")
K = 5


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def q1_rows(ship: Path) -> list[dict]:
    rows = [r for r in _jsonl(ship / "pairs.jsonl") if r["pattern"] == "count"]
    bad = [r["id"] for r in rows if r["split"] not in SPLITS]
    if bad:
        raise SystemExit(f"rows outside train / val: {bad[:3]}")
    return sorted(rows, key=lambda r: r["id"])


def is_new(r) -> bool:
    from grl.hard.q1x import Q1X_TEMPLATES
    return r["provenance"]["m7"]["template"] in Q1X_TEMPLATES


def diag_context(rankings: str, cfg: dict) -> dict:
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    from grl.model.variants import StructEnv, struct_kw
    from grl.train.fbeta import grl_weights
    old, art = load_struct_env_data(), load_struct_env_data(articles=True)
    am = bool(struct_kw(cfg).get("amend_op"))              # fix 6d-b: train.yaml struct.amend_op (default off)
    envs = {"off": StructEnv(old, cfg["env"], amend_op=am)}
    for m in MODES[1:]:
        envs[m] = StructEnv(art, cfg["env"], articles=True, article_scope=m, amend_op=am)
    return {"envs": envs, "ranks": load_rankings(rankings, graph_version=art.version), "w": grl_weights(cfg),
            "cfg": cfg}


def _rank(env, q, s, gold_rows) -> tuple[int | None, int]:
    rows, _ = env.struct_scores(q, s)
    rows = [int(c) for c in rows]
    pos = [rows.index(g) + 1 for g in gold_rows if g in rows]
    return (min(pos) if pos else None), len(rows)


def diag_one(ctx, job):
    """job = (row dict, mode): the R58 record of one item in one mode."""
    from grl.train.expert import expert_config, expert_search
    r, mode = job
    env = ctx["envs"][mode]
    d = env.data
    gold = [g for g in r["C_ref"]]
    if any(g not in d.chunk_row for g in gold):
        return {"id": r["id"], "mode": mode, "skipped": "gold not a row in this mode"}
    grows = frozenset(int(d.chunk_row[g]) for g in gold)
    q = env.query(ctx["ranks"].get(f"{r['id']}|q"))
    s0 = env.reset(q)
    rank0, n0 = _rank(env, q, s0, grows)
    ec = expert_config(ctx["cfg"], "struct")
    t0 = time.perf_counter()
    res = expert_search(env, q, grows, ctx["w"], int(ctx["depth"]), False, None, ec["max_states"],
                        ec["cap_beam"], limit_finals=True, max_rss_gb=ec.get("max_rss_gb"),
                        rss_ceiling_gb=ec.get("rss_ceiling_gb"))
    final = res["final"]
    best, s = rank0, env.reset(q)
    path = []
    for h in final.history:
        rk, _ = _rank(env, q, s, grows) if not s.done else (None, 0)
        if rk is not None:
            best = rk if best is None else min(best, rk)
        path.append(env.actions[h.action].op)
        s = env.step(q, s, h.action)
    return {"id": r["id"], "mode": mode, "rank0": rank0, "n_cands0": n0, "best_rank": best,
            "reached": grows <= set(int(c) for c in final.context), "score": round(float(res["score"]), 6),
            "path": path, "exact": bool(res.get("exact")), "seconds": round(time.perf_counter() - t0, 3)}


def summarize(recs, rows) -> dict:
    meta = {r["id"]: r for r in rows}
    groups = defaultdict(list)
    for x in recs:
        if x.get("skipped"):
            continue
        r = meta[x["id"]]
        t = r["provenance"]["m7"]["template"]
        st = "new" if is_new(r) else "old"
        for key in (f"{x['mode']}|{st}|ALL|{r['split']}", f"{x['mode']}|{st}|{t}|{r['split']}"):
            groups[key].append(x)
    out = {}
    for k, xs in sorted(groups.items()):
        n = len(xs)
        out[k] = {"n": n, **{f"rank0_top{j}": round(sum(1 for x in xs if x["rank0"] and x["rank0"] <= j) / n, 4)
                             for j in (1, 3, 5)},
                  **{f"best_top{j}": round(sum(1 for x in xs if x["best_rank"] and x["best_rank"] <= j) / n, 4)
                     for j in (1, 5)},
                  "reach": round(sum(x["reached"] for x in xs) / n, 4),
                  "n_cands0_mean": round(sum(x["n_cands0"] for x in xs) / n, 2)}
    return out


def baselines(rows, ship: Path, ce_cache: str) -> tuple[dict, dict]:
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.metrics import anyof_metrics
    from grl.model.rank import RankEnv
    from grl.train.rank import baseline_ranking
    sys.path.insert(0, str(ROOT / "scripts"))
    from m7_pilot8 import BASELINES, RANK_KW
    cfg = config.load("train")
    data = load_env_data()
    ranks = load_rankings(ship / "rankings" / "latest.json", graph_version=data.version)
    env = RankEnv(data, cfg["env"], None, **RANK_KW, ce_cache=ce_cache)
    per, groups = {}, defaultdict(list)
    for r in rows:
        m7 = r["provenance"]["m7"]
        q = env.query(ranks.get(f"{r['id']}|q"))
        rec = {}
        for b in BASELINES:
            top = [data.chunk_ids[int(c)] for c in baseline_ranking(env, q, b)][:K]
            rec[b] = anyof_metrics(top, r["C_ref"], m7.get("alt_gold"))
        per[r["id"]] = rec
        ez = "easy" if m7.get("easy") else ("non_easy" if m7.get("alt_gold") else "-")
        for key in (f"{m7['template']}|{ez}|{r['split']}", f"{m7['template']}|ALL|{r['split']}",
                    f"{'new' if is_new(r) else 'old'}|ALL|{r['split']}"):
            groups[key].append(rec)
    out = {}
    for k, recs in sorted(groups.items()):
        n = len(recs)
        out[k] = {"n": n, **{f"{b}|{m}": round(sum(x[b][m] for x in recs) / n, 4)
                             for b in BASELINES for m in ("strict_all", "anyof_all")}}
    return out, per


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", required=True)
    ap.add_argument("--ce-cache", required=True)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import _set
    from graphrag.env.parallel import run as prun
    cfg = config.load("train")
    for kv in a.set:
        if not kv.startswith("env.K_max="):
            raise SystemExit("--set: only env.K_max")
        _set(cfg, kv)
    t0 = time.time()
    ship = config.path(a.ship)
    rows = q1_rows(ship)
    jobs = [(r, m) for r in rows for m in MODES if not (m == "off" and is_new(r))]
    init = partial(diag_context, str(ship / "rankings" / "latest.json"), cfg)
    recs = prun(diag_one, jobs, a.workers, (cfg.get("parallel") or {}).get("start_method"),
                **{"init": partial(_ctx_with_depth, init, a.depth)})
    bl, per_bl = baselines(rows, ship, a.ce_cache)
    out = {"ship": a.ship, "n_q1": len(rows), "K_max": int(cfg["env"]["K_max"]), "depth": a.depth,
           "ce_cache": a.ce_cache, "r58": summarize(recs, rows), "r59_baselines": bl, "records": recs,
           "baseline_per_item": per_bl, "seconds": round(time.time() - t0, 1)}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_q1", "K_max", "seconds")}))
    return 0


def _ctx_with_depth(init, depth):
    c = init()
    c["depth"] = depth
    return c


if __name__ == "__main__":
    sys.exit(main())
