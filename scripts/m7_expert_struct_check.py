"""M7 part 2, Task 2 (R34 / R35) -- one Spark job, two parts:

1. Probe cross-check (data/m7/probe/<sha>/ship, StructEnv at env.K_max = 5, depth 5): per probe query the new struct
   expert (grl.train.expert.expert_search, prune_zero=False, limit_finals=True, exact: no state cap) against
   - oracle_noprune of the probe run's out.json (grl.hard.oracle, ORDERED key; its R recomputed from the recorded
     final context with grl_reward and the run's weights) -- the final R must match wherever both complete;
   - diag_search on the SET key (modes noprune_set / all5_set added here): reachability by set key vs the ordered key
     (out.json oracle_all5 all / recall) must agree.
   Also the expert at the train.yaml cap (expert.max_states / cap_beam): capped or not, same R or not.
2. Step 5: train.py expert --variant struct --set env.K_max=5 --depth 5 --probe-n 40 on the pilot 5 ship
   (run_expert(probe_n=40), both prune modes, limit finals on), for expert.max_states in --ladder until the
   unpruned (struct default) mode has capped = 0 on all 40 queries; never above 2,000,000.

    PYTHONPATH=. python scripts/m7_expert_struct_check.py --probe-ship data/m7/probe/<sha>/ship \\
        --probe-out runs/m7-probe-diag2/out.json --pilot-ship data/m5/pilot/083e116775b3/ship --out runs/<name>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.hard import oracle as _oracle  # noqa: E402

_oracle.MODES.setdefault("noprune_set", {"prune": False, "ordered": False, "target": "R"})   # spawn workers too
_oracle.MODES.setdefault("all5_set", {"prune": False, "ordered": False, "target": "all"})
LADDER_MAX = 2_000_000
EPS = 1e-9


def _ctx(rankings: str, pairs: str, env_cfg: dict, cfg: dict) -> dict:
    return _oracle.diag_context(rankings, pairs, env_cfg, cfg)


def _final(env, gold, res) -> dict:
    f = res["final"]
    h = len(gold & f.ctx_set)
    return {"score": float(res["score"]), "reason": f.reason, "n_states": int(res["n_states"]),
            "all": h == len(gold), "recall": h / len(gold), "n_steps": len(f.history),
            "context": [env.data.chunk_ids[int(c)] for c in f.context],
            "actions": [int(x.action) for x in f.history]}


def one(ctx, job):
    from grl.train.expert import expert_search
    from grl.train.fbeta import grl_reward
    qid, ref_ctx, max_states, cap_beam = job
    env, it, w = ctx["env"], ctx["items"][qid], ctx["w"]
    q = env.query(ctx["ranks"].get(it.qkey))
    out = {"qid": qid}
    t = time.perf_counter()
    e = expert_search(env, q, it.gold, w, 5, prune_zero=False, beam=None, max_states=None, limit_finals=True)
    out["expert"] = {**_final(env, it.gold, e), "final_by": e["final_by"], "exact": e["exact"],
                     "seconds": round(time.perf_counter() - t, 3)}
    t = time.perf_counter()
    c = expert_search(env, q, it.gold, w, 5, prune_zero=False, beam=None, max_states=max_states, cap_beam=cap_beam,
                      limit_finals=True)
    out["expert_cap"] = {"score": float(c["score"]), "capped": bool(c["capped"]), "n_states": int(c["n_states"]),
                         "seconds": round(time.perf_counter() - t, 3)}
    rows = [int(env.data.chunk_row[x]) for x in ref_ctx]
    out["ref_noprune_R"] = float(grl_reward(env.data, q.depth, tuple(rows), it.gold, w, env.limits.T_max).total)
    for mode in ("noprune_set", "all5_set"):
        t = time.perf_counter()
        r = _oracle.diag_search(env, q, it.gold, w, 5, mode, max_states=LADDER_MAX)
        out[mode] = {**_final(env, it.gold, r), "seconds": round(time.perf_counter() - t, 3)}
    return out


def probe_check(a, cfg) -> dict:
    from functools import partial

    from graphrag.env.parallel import run
    from grl.train.expert import expert_config
    ship = Path(a.probe_ship)
    ref = json.loads(Path(a.probe_out).read_text(encoding="utf-8"))
    rows = {r["qid"]: r for r in ref["per_query"]}
    ec = expert_config(cfg)
    jobs = [(qid, r["methods"]["oracle_noprune"]["top5"], ec["max_states"], ec["cap_beam"])
            for qid, r in sorted(rows.items())][:a.n_queries]
    par = cfg.get("parallel") or {}
    t0 = time.time()
    res = run(one, jobs, int(a.workers), par.get("start_method"),
              init=partial(_ctx, str(ship / "rankings" / "latest.json"), str(ship / "pairs.jsonl"), cfg["env"], cfg))
    per = []
    for x in res:
        r = rows[x["qid"]]
        on, oa = r["methods"]["oracle_noprune"], r["methods"]["oracle_all5"]
        per.append({"qid": x["qid"], "qtype": r["qtype"], **x,
                    "ref_noprune": {k: on.get(k) for k in ("all", "recall", "reason", "n_states")},
                    "ref_all5": {k: oa.get(k) for k in ("all", "recall", "reason", "n_states")},
                    "R_match": abs(x["expert"]["score"] - x["ref_noprune_R"]) <= EPS,
                    "R_set_vs_ordered": abs(x["noprune_set"]["score"] - x["ref_noprune_R"]) <= EPS,
                    "reach_set_vs_ordered": (x["all5_set"]["all"], round(x["all5_set"]["recall"], 9))
                    == (bool(oa["all"]), round(float(oa["recall"]), 9))})
    return {"seconds": round(time.time() - t0, 1), "n": len(per), "per_query": per,
            "max_states_cap": ec["max_states"], "cap_beam": ec["cap_beam"]}


def rung_cfg(cfg: dict, ms: int) -> dict:
    """The cfg of one ladder rung: variant struct reads expert.struct_max_states (R38), so the rung sets that key."""
    from copy import deepcopy
    c = deepcopy(cfg)
    c.setdefault("expert", {})["struct_max_states"] = int(ms)
    return c


def ladder(a, cfg) -> list[dict]:
    from grl.train.expert import run_expert
    ship = Path(a.pilot_ship)
    out, steps = Path(a.out), []
    par = cfg.get("parallel") or {}
    for ms in a.ladder:
        if ms > LADDER_MAX:
            raise SystemExit(f"ladder step {ms} > {LADDER_MAX}")
        c = rung_cfg(cfg, ms)
        t0 = time.time()
        rep = run_expert(ship / "trajectories.jsonl", str(ship / "rankings" / "latest.json"), "struct",
                         out / f"probe40-ms{ms}.json", c, int(a.workers), par.get("start_method"), 5, None, None,
                         int(a.probe_n))
        kz = rep["modes"]["keep_zero"]
        steps.append({"max_states": int(ms), "seconds": round(time.time() - t0, 1),
                      "keep_zero_capped_share": kz["capped_share"], "keep_zero": kz,
                      "prune_zero": rep["modes"]["prune_zero"], "n_score_differs": rep["n_score_differs"],
                      "keep_zero_better": rep["keep_zero_better"]})
        print(f"ladder max_states={ms}: keep_zero capped_share {kz['capped_share']} n_states max "
              f"{kz['n_states_max']} median {kz.get('n_states_median')}", file=sys.stderr, flush=True)
        if kz["capped_share"] == 0:
            break
    return steps


def main(argv=None) -> int:
    from graphrag import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-ship", required=True)
    ap.add_argument("--probe-out", required=True)
    ap.add_argument("--pilot-ship", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--probe-n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=None, help="default: expert.struct_workers (R40)")
    ap.add_argument("--ladder", type=int, nargs="+", default=[20000, 200000, 1000000, 2000000])
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--n-queries", type=int, default=None, help="smoke: only the first N probe queries")
    a = ap.parse_args(argv)
    cfg = config.load("train")
    cfg["env"]["K_max"] = int(a.k_max)
    if a.workers is None:
        from grl.train.expert import expert_config
        a.workers = expert_config(cfg, "struct")["workers"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = {"K_max": int(a.k_max), "depth": 5}
    if not a.skip_probe:
        rep["probe"] = probe_check(a, cfg)
        (Path(a.out) / "check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        p = rep["probe"]["per_query"]
        print(f"probe: n {len(p)} R_match {sum(x['R_match'] for x in p)} set_vs_ordered R "
              f"{sum(x['R_set_vs_ordered'] for x in p)} reach {sum(x['reach_set_vs_ordered'] for x in p)} "
              f"({rep['probe']['seconds']}s)", file=sys.stderr, flush=True)
    rep["ladder"] = ladder(a, cfg)
    (Path(a.out) / "check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("DONE", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
