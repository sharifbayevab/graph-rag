"""M7 pilot 7 (interim, validation only): baselines on the M7 Q1 / Q2 val rows of the joint struct ship, and the
offline report tables (agent vs baselines on M7 val; old-val R at K_max 5 / 24; action shares).

Spark (one job; CE cache = scripts/ce_scores.py --set rerank.n=50 --set rerank.ce_n=50 over the ship pairs):
    PYTHONPATH=. python scripts/m7_pilot7.py baselines --ship data/m7/full/<sha>/ship \\
        --ce-cache runs/p7-ce/ce_scores.jsonl --out runs/p7-bl/baselines.json
  dense@5 / bm25@5 / fused@5 (w 0.7) / ce+fused:0.6@5 exactly as scripts/hard_probe_eval.py (grl.train.rank
  .baseline_ranking over the pilot-6 RankEnv pool: n 50, fuse_w 0.7, norm_top_n 50, k_max 20, static, ce feature),
  first 5 ids of each ranking, on the base EnvData (STRUCT rows are in no index: Q1 gold is missed by construction).
  Reads only val rows of origin m7_hard (grl.hard.ship.HARD_ORIGIN); the ship holds no test rows.
Mac (offline, from pulled files):
    PYTHONPATH=. python scripts/m7_pilot7.py report --ship ... --baselines F --agent-k5 PQ5.jsonl --agent-k24 PQ24.jsonl
        [--old-traj data/m5/pilot/083e116775b3/ship/trajectories.jsonl] --out F.json
  agent = greedy struct policy (train.py eval --per-query, val); its final context (<= K_max = 5 fragments) is its
  top-5. recall@5 / all@5 by type (grl.hard.metrics), old-val R (182 rows; and the 126 rows of the pilot-5 val
  population with an oracle trajectory, where the pilot-5 RL-ranker val R 0.7034 was measured), action shares."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()

from graphrag import config  # noqa: E402

K = 5
BASELINES = ("dense", "bm25", "fused", "ce+fused:0.6")
RANK_KW = {"n": 50, "fuse_w": 0.7, "norm_top_n": 50, "k_max": 20, "static": True, "use_ce": True}   # pilot-6 pool


def _write_json(path, obj) -> None:
    path = Path(path) if Path(path).is_absolute() else config.path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def hard_val_rows(pairs) -> list:
    """The M7 hard val Pair rows of a ship pairs file (train / val loader guards: test-eval / probe / Q-real refused)."""
    from grl.hard.ship import HARD_ORIGIN
    from grl.train.expert import expert_inputs
    return [p for p in expert_inputs(pairs) if p.split == "val" and p.origin == HARD_ORIGIN]


def cmd_baselines(a) -> int:
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.metrics import top_k_ids
    from grl.hard.probe import check_rank_rows
    from grl.model.rank import RankEnv
    from grl.model.rerank import rerank_settings
    from grl.train.rank import baseline_ranking
    t0 = time.time()
    cfg = config.load("train")
    ship = config.path(a.ship)
    rows = hard_val_rows(ship / "pairs.jsonl")
    data = load_env_data()
    ranks = load_rankings(ship / "rankings" / "latest.json", graph_version=data.version)
    keys = [f"{p.id}|q" for p in rows]
    rr = check_rank_rows(ranks, keys, len(data.chunk_ids))     # base EnvData: no STRUCT rows at all
    env = RankEnv(data, cfg["env"], None, **RANK_KW, ce_cache=a.ce_cache)
    out = {}
    for p in rows:
        q = env.query(ranks.get(f"{p.id}|q"))
        out[p.id] = {n: top_k_ids([data.chunk_ids[int(c)] for c in baseline_ranking(env, q, n)], K)
                     for n in BASELINES}
        out[p.id]["_pool_size"] = int(env.cand_rows(q).size)
    _write_json(a.out, {"k": K, "n": len(rows), "baselines": list(BASELINES), "top5": out,
                        "rank_env": {**RANK_KW, "ce_cache": a.ce_cache}, "settings": rerank_settings(env),
                        "rank_rows": rr, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({"n": len(rows), "seconds": round(time.time() - t0, 1)}))
    return 0


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _mean_se(xs):
    from grl.hard.metrics import mean_se
    m, se = mean_se(xs)
    return {"n": len(xs), "mean": None if m is None else round(m, 4), "se": None if se is None else round(se, 4)}


def _family(op: str) -> str:
    head = op.split(":", 1)[0]
    return head if head in ("structure", "graph") else op


def action_shares(pq) -> dict:
    c = Counter(_family(x) for r in pq for x in r["actions"])
    tot = sum(c.values())
    first = Counter(_family(r["actions"][0]) for r in pq if r["actions"])
    return {"n_queries": len(pq), "n_actions": tot, "share": {k: round(v / tot, 4) for k, v in sorted(c.items())},
            "first_action": dict(sorted(first.items())), "mean_len": round(tot / max(len(pq), 1), 3),
            "uses_structure_share": round(sum(any(_family(x) == "structure" for x in r["actions"]) for r in pq)
                                          / max(len(pq), 1), 4)}


def cmd_report(a) -> int:
    from grl.env.struct_data import load_struct_env_data
    from grl.hard.metrics import set_metrics, summarize_methods
    ship = config.path(a.ship)
    pairs = {r["id"]: r for r in _jsonl(ship / "pairs.jsonl")}
    if any(r["split"] not in ("train", "val") for r in pairs.values()):
        raise SystemExit("ship holds non train / val rows")
    data = load_struct_env_data()
    bl = json.loads(Path(a.baselines).read_text(encoding="utf-8"))
    pq5 = {r["qkey"][:-2]: r for r in _jsonl(a.agent_k5)}
    pq24 = {r["qkey"][:-2]: r for r in _jsonl(a.agent_k24)} if a.agent_k24 else {}
    for pq in (pq5, pq24):
        bad = [k for k, r in pq.items() if r["split"] != "val" or k not in pairs]
        if bad:
            raise SystemExit(f"per-query rows outside the ship val: {bad[:3]}")
    hard = sorted(k for k, r in pairs.items() if r.get("origin") == "m7_hard" and r["split"] == "val")
    old = sorted(k for k, r in pairs.items() if r.get("origin") != "m7_hard" and r["split"] == "val")
    if sorted(bl["top5"]) != hard or not set(hard) <= set(pq5):
        raise SystemExit("baselines / agent do not cover exactly the hard val rows")
    mrows, per = [], {}
    for k in hard:
        gold, qt = pairs[k]["C_ref"], pairs[k]["pattern"]
        m = {n: bl["top5"][k][n] for n in BASELINES}
        m["agent"] = [data.chunk_ids[int(c)] for c in pq5[k]["context"]][:K]
        if len(pq5[k]["context"]) > K:
            raise SystemExit(f"{k}: agent context longer than K_max 5")
        per[k] = {"qtype": qt, "gold": gold, **{n: {"top5": v, **set_metrics(v, gold)} for n, v in m.items()}}
        for n, v in m.items():
            mrows.append({"method": n, "qtype": qt, "qid": k, **set_metrics(v, gold)})
    table = summarize_methods(mrows)
    old_traj_val = sorted({r["id"] for r in _jsonl(a.old_traj) if r["split"] == "val"}) if a.old_traj else []
    oldr = {}
    for name, pq in (("K5", pq5), ("K24", pq24)):
        if not pq:
            continue
        oldr[name] = {"all182": _mean_se([pq[k]["reward"] for k in old]),
                      "traj126": _mean_se([pq[k]["reward"] for k in old_traj_val]) if old_traj_val else None,
                      "hard_val": _mean_se([pq[k]["reward"] for k in hard if k in pq]),
                      "hard_val_by_type": {t: _mean_se([pq[k]["reward"] for k in hard
                                                        if pairs[k]["pattern"] == t and k in pq])
                                           for t in ("count", "version")},
                      "val_all": _mean_se([r["reward"] for r in pq.values()])}
    out = {"k": K, "n_hard_val": len(hard), "n_old_val": len(old), "n_old_traj_val": len(old_traj_val),
           "table": table, "old_val_R": oldr,
           "actions": {"hard_val_K5": action_shares([pq5[k] for k in hard]),
                       "old_val_K5": action_shares([pq5[k] for k in old]),
                       **({"old_val_K24": action_shares([pq24[k] for k in old])} if pq24 else {})},
           "per_query": per,
           "inputs": {"ship": a.ship, "baselines": a.baselines, "agent_k5": a.agent_k5, "agent_k24": a.agent_k24}}
    _write_json(a.out, out)
    for n in BASELINES + ("agent",):
        print(n, {t: (table[n][t]["n"], table[n][t]["recall"] and round(table[n][t]["recall"], 3),
                      table[n][t]["all"] and round(table[n][t]["all"], 3)) for t in ("count", "version", "pooled")})
    print(json.dumps({"old_val_R": oldr, "actions": out["actions"]}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--ship", required=True)
    b.add_argument("--ce-cache", required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--ship", required=True)
    r.add_argument("--baselines", required=True)
    r.add_argument("--agent-k5", required=True)
    r.add_argument("--agent-k24")
    r.add_argument("--old-traj", default="data/m5/pilot/083e116775b3/ship/trajectories.jsonl")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_baselines(a) if a.cmd == "baselines" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
