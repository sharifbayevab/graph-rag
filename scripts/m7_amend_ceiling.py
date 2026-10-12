"""M7 Task 6d-b (train + val only; the test splits stay sealed): the expert ceiling of the struct env with and without
the `amendments` operator (struct.amend_op). Rows: every Q2 (pattern version) train / val row of the ship plus a seeded
sample of `--sample` old + Q1 (single / multi / count) train / val rows. Both modes run the struct expert exactly as
`train.py expert --variant struct` does (grl.train.expert.run_expert: depth 5, unpruned, limit finals, the run's
K_max), on the same subset file.

    PYTHONPATH=. python scripts/m7_amend_ceiling.py --out runs/m7-6db-ceiling [--sample 100] [--set env.K_max=5]

Writes <out>/subset.jsonl, <out>/{off,on}.jsonl (+ .stats.json) and <out>/ceiling.json: per mode and group (Q2
template / old / count) n, all@5 (= reached: C* within the final context at K_max 5), recall, limit-final share,
score, n_states, the share of finals that use `amendments`, and the per-row on-vs-off flips. With --ref (the pilot-8
labels, runs/p8-expert/labels.jsonl) the off mode is also compared row by row to it (it must be identical: the flag
off is the pilot-8 env)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()

SEED = 20261004
OLD_PATTERNS = ("single", "multi", "count")
AMEND_IDS = (31, 32, 33)                 # `amendments` m = 1 / 3 / 5 in the 34-action layout (StructEnv amend_op)


def _template(p) -> str:
    m7 = (p.provenance or {}).get("m7") or {}
    if p.pattern == "version":
        return "q2:" + str(m7.get("template"))
    return "count" if p.pattern == "count" else "old"


def subset(pairs, n_sample: int, seed: int = SEED, q2_max: int | None = None):
    q2 = sorted((p for p in pairs if p.pattern == "version"), key=lambda p: p.id)[:q2_max]
    pool = sorted((p for p in pairs if p.pattern in OLD_PATTERNS), key=lambda p: p.id)
    rng = random.Random(seed)
    pick = rng.sample(pool, min(n_sample, len(pool)))
    return sorted(q2 + pick, key=lambda p: p.id)


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def summarise(rows, groups) -> dict:
    """rows: id -> pair with the expert trajectory (provenance.meta.expert)."""
    by = defaultdict(list)
    for i, p in rows.items():
        by[groups[i]].append(p)
        by["q2:ALL" if groups[i].startswith("q2:") else "old+count"].append(p)
    out = {}
    for g in sorted(by):
        ps = by[g]
        ex = [p.provenance["meta"]["expert"] for p in ps]
        out[g] = {"n": len(ps), "all5": _mean(float(e["reached"]) for e in ex),
                  "score": _mean(e["score"] for e in ex),
                  "limit_share": _mean(float(e.get("final_by") == "limit") for e in ex),
                  "capped_share": _mean(float(e["capped"]) for e in ex),
                  "n_states_mean": _mean(e["n_states"] for e in ex),
                  "uses_amendments": _mean(float(any(int(a) in AMEND_IDS for a in p.actions)) for p in ps),
                  "len_mean": _mean(e["len"] for e in ex)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", default="data/m7/full/2ab8c949e85e/ship")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--depth", type=int, default=5, help="expert depth (pilot 8: 5)")
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--q2-max", type=int, help="smoke test only: the first N Q2 rows by id")
    ap.add_argument("--ref", help="pilot-8 struct expert labels: the off mode must equal them row by row")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    import yaml

    from graphrag import config
    from graphrag.dataset.finalize import read_pairs, write_pairs
    from grl.model.variants import struct_kw
    from grl.train.expert import expert_config, expert_inputs, run_expert
    cfg = config.load("train")
    for kv in a.set:
        k, v = kv.split("=", 1)
        node, *path = cfg, *k.split(".")
        for x in path[:-1]:
            node = node[x]
        node[path[-1]] = yaml.safe_load(v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ship = config.path(a.ship)
    pairs = expert_inputs(ship / "pairs.jsonl")                       # train + val only (test rows dropped)
    sub = subset(pairs, a.sample, q2_max=a.q2_max)
    sub_f = out / "subset.jsonl"
    write_pairs(sub_f, sub)
    groups = {p.id: _template(p) for p in sub}
    rankings = str(ship / "rankings" / "latest.json")
    workers = a.workers or expert_config(cfg, "struct")["workers"]
    rep = {"ship": str(a.ship), "seed": SEED, "sample": a.sample, "q2_max": a.q2_max, "K_max": int(cfg["env"]["K_max"]), "depth": a.depth,
           "n": len(sub), "n_q2": sum(g.startswith("q2:") for g in groups.values()), "set": a.set, "modes": {}}
    labels = {}
    for mode in a.modes.split(","):
        c = json.loads(json.dumps(cfg))
        c.setdefault("struct", {})["amend_op"] = mode == "on"
        stats = run_expert(sub_f, rankings, "struct", out / f"{mode}.jsonl", c, workers, depth=a.depth)
        rows = {p.id: p for p in read_pairs(out / f"{mode}.jsonl")}
        labels[mode] = rows
        rep["modes"][mode] = {"env_kw": struct_kw(c), "seconds": stats["seconds"], "groups": summarise(rows, groups)}
        print(json.dumps({mode: rep["modes"][mode]["groups"]}, ensure_ascii=False), file=sys.stderr, flush=True)
    if {"on", "off"} <= set(labels):
        flips = defaultdict(lambda: {"gain": 0, "loss": 0, "same": 0})
        for i, g in groups.items():
            x, y = (labels[m][i].provenance["meta"]["expert"]["reached"] for m in ("off", "on"))
            flips[g]["gain" if y and not x else "loss" if x and not y else "same"] += 1
        rep["flips"] = dict(flips)
        rep["score_delta"] = {g: _mean(labels["on"][i].provenance["meta"]["expert"]["score"]
                                       - labels["off"][i].provenance["meta"]["expert"]["score"]
                                       for i in groups if groups[i] == g) for g in sorted(set(groups.values()))}
    if a.ref and "off" in labels:
        ref = {p.id: p for p in read_pairs(a.ref) if p.id in groups}
        diff = [i for i in groups if i in ref and (ref[i].actions != labels["off"][i].actions
                                                   or ref[i].provenance["meta"] != labels["off"][i].provenance["meta"])]
        rep["off_vs_ref"] = {"n_compared": len(ref), "n_differ": len(diff), "differ": diff[:20]}
    if "on" in labels:
        first = defaultdict(lambda: defaultdict(int))
        for i, p in labels["on"].items():
            if p.actions:
                first[groups[i]][str(p.actions[0])] += 1
        rep["on_first_action_ids"] = {g: dict(v) for g, v in first.items()}
    (out / "ceiling.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "on_first_action_ids"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
