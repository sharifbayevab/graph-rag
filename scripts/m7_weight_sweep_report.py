"""M7 reward-weight sweep: Mac-side summary of the pulled runs (scripts/m7_weight_sweep.py) -> data/m7/calib/sweep.json
and docs/m7/figures/ws_*.png. Validation only.

    PYTHONPATH=. python scripts/m7_weight_sweep_report.py [--pull] [--points eq,f1,...] [--seeds s01,s02]

Per (point, seed) the agent candidate follows the task-9 acceptance rule (scripts/m7_runs.acceptance): GRPO theta* if
it beats its SFT on val R (the run's OWN reward) by more than one paired SE and no alarm fired in the last 20 % of
iterations, else the SFT. Points are compared on reward-independent metrics of the candidate: all@5 (the whole gold
set inside the <= 5 fragments), recall@5, per group (old / count = Q1+Q1x / version = Q2). R_ref = the current-weight
reward of the candidate's final context (a common yardstick; biased towards the current weights by construction).
The current point reuses the task-9 candidates (runs/spark/f9-dry-val2/agent-sNN-k5.jsonl)."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.m7_weight_sweep import CURRENT, POINTS, STAGE1_SEEDS  # noqa: E402

LOCAL = ROOT / "runs" / "spark"
CUR_FILES = {s: LOCAL / f"f9-dry-val2/agent-{s}-k5.jsonl" for s in ("s01", "s02", "s03")}
GROUPS = ("old", "count", "version")
REF_W = dict(zip(("r_chunks", "r_coverage", "r_diversity", "r_coherence"), CURRENT))


def jl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def group(r) -> str:
    return r["pattern"] if r["pattern"] in ("count", "version") else "old"


def per_query(rows) -> dict:
    return {r["qkey"]: {"all5": float(r["recall"] >= 1 - 1e-9 and r["n_ctx"] <= 5), "rec": float(r["recall"]),
                        "rref": sum(w * float(r[k]) for k, w in REF_W.items()), "g": group(r)} for r in rows}


def accept(point: str, seed: str) -> dict:
    sft = {r["qkey"]: r["reward"] for r in jl(LOCAL / f"ws-{point}-evsft-{seed}/pq.jsonl")}
    grp = {r["qkey"]: r["reward"] for r in jl(LOCAL / f"ws-{point}-ev-{seed}/pq.jsonl")}
    d = np.array([grp[k] - sft[k] for k in sorted(sft)])
    se = d.std(ddof=1) / math.sqrt(len(d))
    met = jl(LOCAL / f"ws-{point}-grpo-{seed}/metrics.jsonl")
    gt = [x for x in met if x.get("phase") == "grpo"]
    n_it = max(x["step"] for x in gt)
    al = [(x["step"], a) for x in met if x.get("phase") in ("grpo", "grpo_val") for a in (x.get("alarms") or [])
          if x["step"] > 0.8 * n_it]
    ok = d.mean() > se and not al
    return {"diff": float(d.mean()), "se": float(se), "alarms_last20": len(al), "candidate": "grpo" if ok else "sft"}


def summarise(pq: dict) -> dict:
    out = {"n": len(pq)}
    for m in ("all5", "rec", "rref"):
        out[m] = float(np.mean([v[m] for v in pq.values()]))
        for g in GROUPS:
            vs = [v[m] for v in pq.values() if v["g"] == g]
            out[f"{m}_{g}"] = float(np.mean(vs))
    return out


def pull(points, seeds) -> None:
    """Only the small files the summary reads (no checkpoints / labels: the Mac disk is tight)."""
    from graphrag import config
    from scripts.spark import Remote
    files = [f"ws-{p}-{k}-{s}/{f}" for p in points for s in seeds
             for k, f in (("ev", "pq.jsonl"), ("evsft", "pq.jsonl"), ("grpo", "metrics.jsonl"), ("sft", "metrics.jsonl"))]
    files += [f"ws-{p}-expert/labels.stats.json" for p in points]
    r = Remote(config.load("train")["spark"])
    LOCAL.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(r.argv + [f"cd {r.home('runs')} && tar -c -f - --ignore-failed-read "
                                   + " ".join(files) + " 2>/dev/null"], env=r.env, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-f", "-", "-C", str(LOCAL)], stdin=p.stdout, check=False)
    p.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--points", default=",".join(POINTS))
    ap.add_argument("--seeds", default=",".join(STAGE1_SEEDS))
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args(argv)
    pts, seeds = [p for p in a.points.split(",") if p], [s for s in a.seeds.split(",") if s]
    if a.pull:
        pull(pts, seeds)

    runs = {"cur": {s: {"pq": per_query(jl(f)), "candidate": "task9"} for s, f in CUR_FILES.items()}}
    for p in pts:
        runs[p] = {}
        for s in seeds:
            try:
                acc = accept(p, s)
            except FileNotFoundError:
                continue
            f = LOCAL / (f"ws-{p}-ev-{s}/pq.jsonl" if acc["candidate"] == "grpo" else f"ws-{p}-evsft-{s}/pq.jsonl")
            runs[p][s] = {"pq": per_query(jl(f)), **acc}
    keys = sorted(runs["cur"]["s01"]["pq"])
    rng = np.random.default_rng(20261011)
    res = {}
    cur_q = {m: np.mean([[runs["cur"][s]["pq"][k][m] for k in keys] for s in runs["cur"]], 0) for m in ("all5", "rec")}
    for p, rs in runs.items():
        if not rs:
            continue
        per_seed = {s: summarise(r["pq"]) for s, r in rs.items()}
        row = {"w": list(CURRENT if p == "cur" else POINTS[p]), "seeds": per_seed,
               "candidates": {s: r.get("candidate") for s, r in rs.items()}}
        for m in ("all5", "rec", "rref", "all5_old", "all5_count", "all5_version", "rec_version"):
            v = np.array([x[m] for x in per_seed.values()])
            row[m] = float(v.mean())
            row[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        for m in ("all5", "rec"):                              # paired vs current, per question (seed-averaged)
            q = np.mean([[r["pq"][k][m] for k in keys] for r in rs.values()], 0)
            d = q - cur_q[m]
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
            row[f"d_{m}"] = float(d.mean())
            row[f"d_{m}_ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        res[p] = row
    out = ROOT / "data/m7/calib/sweep.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for p, r in sorted(res.items(), key=lambda kv: -kv[1]["all5"]):
        print(f"{p:5s} w={r['w']} n_seeds={len(r['seeds'])} all@5 {r['all5']:.3f}±{r['all5_sd']:.3f} "
              f"rec {r['rec']:.3f} old {r['all5_old']:.3f} count {r['all5_count']:.3f} ver {r['all5_version']:.3f} "
              f"| Δall@5 vs cur {r['d_all5']:+.3f} [{r['d_all5_ci'][0]:+.3f},{r['d_all5_ci'][1]:+.3f}] cand {r['candidates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
