"""M3 Task 13: time the oracle on real Q-real queries (stratified by |C*| and section), serially and with `--workers`
processes (graphrag.env.parallel, train.yaml parallel.start_method); check that the parallel results equal the serial
ones; project Tasks 20 and 21; exit 1 if a projection exceeds --budget-hours.
    python scripts/profile_oracle.py [--workers 8] [--n 20] [--budget-hours 2] [--start-method spawn|fork]

Per query: one BeamSearch (train.yaml oracle.*: both acceptance criteria, d5 and unconstrained, from one search,
ruling M3-R6), the greedy oracle, 4 random rollouts. Worker start-up (spawn: import + load_env_data + rankings in
each worker, once) is measured separately and included in the projections."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from graphrag import config
from graphrag.env.parallel import default_start_method, run
from graphrag.env.policies import BeamSearch, d5_accept, greedy_oracle, random_policy, rollout
from graphrag.eval.gold import read_gold

OUT = "data/m3/oracle_profile.json"


def _init() -> dict:
    """Heavy read-only context, built once per process (spawn workers and the serial path alike)."""
    from graphrag.env.data import load_env_data
    from graphrag.env.mdp import Env
    from graphrag.env.reward import weights_from_config
    from graphrag.retrieval.rankings import QREAL_LATEST, load_rankings
    data = load_env_data()
    return {"env": Env(data), "w": weights_from_config(), "ranks": load_rankings(QREAL_LATEST, graph_version=data.version)}


_HOLD = 0.5


def _ready(ctx, i):
    """Start-up probe: each worker holds one item for _HOLD s, so every worker takes one; returns (pid, wall clock)."""
    time.sleep(_HOLD)
    return os.getpid(), time.time()


def gold_rows(item, data, use_support: bool) -> frozenset:
    """C* as fragment rows: core ∪ support (train.yaml gold.use_support) or core only."""
    ids = item.gold if use_support else sorted(item.core)
    return frozenset(int(data.chunk_row[c]) for c in ids)


def _one(ctx, item):
    qk, gold = item
    env, w = ctx["env"], ctx["w"]
    q = env.query(ctx["ranks"].get(qk))
    t = {}
    t0 = time.perf_counter()
    bs = BeamSearch(env, q, gold, w)
    b5, ba = bs.best(d5_accept(gold)), bs.best()
    t["beam"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = greedy_oracle(env, q, gold, w)
    t["greedy"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rnd = [rollout(env, q, random_policy(sd, allow_stop=sd % 2 == 0)) for sd in range(4)]
    t["random4"] = time.perf_counter() - t0
    acts = lambda s: None if s is None else [h.action for h in s.history]
    R = lambda s: None if s is None else bs.R(s)
    res = {"qkey": qk, "d5": acts(b5), "R_d5": R(b5), "any": acts(ba), "R_any": R(ba), "greedy": acts(g),
           "random": [acts(s) for s in rnd], "n_finals": len(bs.finals)}
    return t, res


def _digest(results) -> str:
    return hashlib.sha256(json.dumps([r for _, r in results], sort_keys=True).encode()).hexdigest()


def pick(items, n: int) -> list:
    items = sorted(items, key=lambda i: (len(i.gold), i.section, hashlib.sha256(i.qid.encode()).hexdigest()))
    if n >= len(items):
        return items
    return [items[round(k * (len(items) - 1) / (n - 1))] for k in range(n)] if n > 1 else items[:1]


def main(argv=None) -> int:
    tcfg = config.load("train")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int((tcfg.get("parallel") or {}).get("workers", 8)))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget-hours", type=float, default=2.0)
    ap.add_argument("--start-method", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    method = args.start_method or default_start_method()
    t0 = time.perf_counter()
    ctx = _init()
    t_load = time.perf_counter() - t0
    data = ctx["env"].data
    items = read_gold(config.path(tcfg["gold"]["file"]))
    work = [(f"{i.qid}|cyr", gold_rows(i, data, bool(tcfg["gold"]["use_support"]))) for i in pick(items, args.n)]

    t0 = time.perf_counter()
    serial = run(_one, work, 1, env=ctx["env"], w=ctx["w"], ranks=ctx["ranks"])
    t_serial = time.perf_counter() - t0

    t0 = time.time()                                            # worker start-up alone (spawn: load once per worker)
    ready = run(_ready, list(range(args.workers)), args.workers, start_method=method, init=_init)
    t_startup = max(t for _, t in ready) - t0 - _HOLD           # the slowest worker ready (they start concurrently)
    t0 = time.perf_counter()
    par = run(_one, work, args.workers, start_method=method, init=_init)
    t_par = time.perf_counter() - t0                            # includes a fresh pool's start-up
    identical = _digest(par) == _digest(serial)

    beam = np.array([t["beam"] for t, _ in serial])
    per = {k: float(np.mean([t[k] for t, _ in serial])) for k in serial[0][0]}
    speedup = t_serial / max(t_par - t_startup, 1e-9)           # steady-state; start-up is added once below
    n20 = round(tcfg["dataset"]["size"] * 1.25)
    n21 = tcfg["dataset"]["size"] + 2 * len(items)
    su_h = t_startup / 3600
    proj20 = 2 * n20 * per["beam"] / speedup / 3600 + 2 * su_h                     # + determinism re-run
    proj21 = n21 * (per["beam"] + per["greedy"] + 2 * per["random4"]) / speedup / 3600 + su_h
    proj21c = n21 * (2 * per["beam"] + per["greedy"] + 2 * per["random4"]) / speedup / 3600 + su_h  # brief: 2 beams
    rep = {"n_queries": len(work), "graph_version": data.version, "gold_file": tcfg["gold"]["file"],
           "oracle": tcfg.get("oracle"), "start_method": method, "workers": args.workers, "cpu_count": os.cpu_count(),
           "distinct_worker_pids": len({p for p, _ in ready}),
           "load_env_s": round(t_load, 2), "worker_startup_s": round(t_startup, 2),
           "per_query_s": {k: round(v, 3) for k, v in per.items()},
           "beam_s": {"median": round(float(np.median(beam)), 3), "p95": round(float(np.percentile(beam, 95)), 3),
                      "max": round(float(beam.max()), 3)},
           "serial_s": round(t_serial, 2), "parallel_s": round(t_par, 2), "speedup": round(speedup, 2),
           "parallel_identical": identical, "results_sha256": _digest(serial),
           "projected_hours": {"task20": round(proj20, 3), "task21": round(proj21, 3),
                               "task21_two_beam_searches": round(proj21c, 3)},
           "projection_counts": {"task20_candidates": n20, "task20_runs": 2, "task21_queries": n21},
           "budget_hours": args.budget_hours}
    out = config.path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    if not identical:
        print("parallel results differ from the serial ones", file=sys.stderr)
        return 1
    return 0 if max(proj20, proj21, proj21c) <= args.budget_hours else 1


if __name__ == "__main__":
    sys.exit(main())
