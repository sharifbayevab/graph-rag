"""M7 Task 9 (final, adapted after pilots 7-9; validation only, the test splits stay sealed): the 3-seed queue of the
pilot-9 accepted recipe -- SFT `struct` from scratch (articles + amend_op, 34 actions, wd 1e-3, dropout 0.2, early stop
on val R, patience 5) -> GRPO (beta .3, lr 5e-5, G_s 8, 300 it, val / 20, Q2 upweight .5) -> greedy eval of theta*
(= GRPO's best val point) on val (per-query). Seed s01 = pilot 9 itself (seed 20261001, the same code path; checked by
the `s01chk` GRPO rerun, which must reproduce p9-grpo's theta* exactly), s02 / s03 run fresh.

    PYTHONPATH=. python scripts/m7_runs.py jobs                 # the queue (deterministic names, never --split test)
    PYTHONPATH=. python scripts/m7_runs.py chain --seed s02     # on Spark: sft -> grpo -> eval, one step at a time
    PYTHONPATH=. python scripts/m7_runs.py chain --seed s01chk  # on Spark: GRPO seed 20261001 from p9-sft-scratch

Every step's argv is printed before it runs and the chain stops at the first non-zero exit code (rc.py prints it)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHIP = "data/m7/full/54aa9a000b4b/ship"
LABELS = "runs/p9-expert/labels.jsonl"
RANKINGS = f"{SHIP}/rankings/latest.json"
FLAGS = ["--set", "env.K_max=5", "--set", "struct.articles=true", "--set", "struct.amend_op=true"]
SFT_SET = ["sft.early_stop=reward", "sft.patience=5", "sft.eval_epoch0=true", "sft.train_eval_loss=true",
           "sft.weight_decay=1e-3", "model.dropout=0.2"]
GRPO_SET = ["grpo.beta_kl=0.3", "grpo.lr=5e-5", "grpo.G_s=8", "grpo.iterations=300", "grpo.val_every=20",
            "grpo.patience=100", "grpo.upweight_pattern=version", "grpo.upweight_share=0.5"]
SEEDS = {"s01": 20261001, "s02": 20261002, "s03": 20261003}
P9 = {"sft": "runs/p9-sft-scratch", "grpo": "runs/p9-grpo", "theta": "runs/p9-grpo/checkpoints/grpo-000220.pt"}


def _sets(keys) -> list[str]:
    return [x for k in keys for x in ("--set", k)]


def sft_run(seed: str) -> str:
    return P9["sft"] if seed == "s01" else f"runs/f9-sft-{seed}"


def grpo_run(seed: str) -> str:
    return P9["grpo"] if seed == "s01" else f"runs/f9-grpo-{seed}"


def steps(seed: str) -> list[tuple[str, list[str]]]:
    """(step name, train.py argv) of one seed; s01chk = the GRPO reproduction check of pilot 9 (seed 20261001)."""
    base = ["--variant", "struct", "--pairs", LABELS, "--rankings", RANKINGS, *FLAGS]
    if seed == "s01chk":
        return [("grpo", ["grpo", *base, "--run", "runs/f9-grpo-s01chk", "--sft", P9["sft"],
                          *_sets(GRPO_SET + [f"grpo.seed={SEEDS['s01']}"])])]
    if seed not in SEEDS or seed == "s01":
        raise SystemExit(f"--seed {seed}: one of s02 / s03 / s01chk (s01 = pilot 9)")
    s = SEEDS[seed]
    return [("sft", ["sft", *base, "--run", sft_run(seed), *_sets(SFT_SET + [f"sft.seed={s}"])]),
            ("grpo", ["grpo", *base, "--run", grpo_run(seed), "--sft", sft_run(seed),
                      *_sets(GRPO_SET + [f"grpo.seed={s}"])]),
            ("eval", ["eval", *base, "--ckpt", "{best}", "--split", "val",
                      "--per-query", f"runs/f9-ev-{seed}/pq.jsonl"])]


def jobs() -> list[dict]:
    out = [{"name": "f9-s01chk", "seed": "s01chk", "steps": steps("s01chk")}]
    out += [{"name": f"f9-{s}", "seed": s, "steps": steps(s)} for s in ("s02", "s03")]
    for j in out:
        for _, argv in j["steps"]:
            if any(x == "--split" and y != "val" for x, y in zip(argv, argv[1:])):
                raise SystemExit(f"{j['name']}: a step reads a split other than val")
    return out


def _last_json(lines: list[str]) -> dict:
    for ln in reversed(lines):
        ln = ln.strip()
        if ln.startswith("{") and '"best"' in ln:
            return json.loads(ln)
    raise SystemExit("no {\"best\": ...} line in the step's output")


def chain(seed: str) -> int:
    best = None
    for name, argv in steps(seed):
        argv = [best if x == "{best}" else x for x in argv]
        cmd = [sys.executable, "scripts/train.py", *argv]
        print("STEP", name, json.dumps(argv), flush=True)
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)
        lines = []
        for ln in p.stdout:
            sys.stdout.write(ln)
            sys.stdout.flush()
            lines.append(ln)
        rc = p.wait()
        print(f"STEP {name} RC {rc}", flush=True)
        if rc:
            return rc
        if name == "grpo":
            best = _last_json(lines)["best"]
    return 0


def _jl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


FAMILY = (("stop", ("stop",)), ("structure", ("structure",)), ("amendments", ("amendments",)),
          ("graph", ("graph",)), ("lexical", ("lexical", "bm25", "lex")), ("vector", ("vector", "dense", "vec")))


def action_family(name: str) -> str:
    n = str(name).lower()
    for fam, keys in FAMILY:
        if any(n.startswith(k) for k in keys):
            return fam
    return "other"


def seed_summary(seed: str, local: str = "runs/spark", eval_pq: str | None = None) -> dict:
    """Mac: one seed's SFT (epochs, best epoch by val R, val R) and GRPO (theta* = best val point, its val R, the SFT
    point, alarms anywhere / in the last 20 % of iterations, KL max) from the pulled metrics; the theta* action-family
    shares per group from a val per-query file."""
    sft = _jl(Path(local) / sft_run(seed)[len("runs/"):] / "metrics.jsonl")
    sv = [d for d in sft if d.get("phase") == "sft_val"]
    best = max(sv, key=lambda d: (d["val_reward_mean"], -d["step"]))
    g = _jl(Path(local) / grpo_run(seed)[len("runs/"):] / "metrics.jsonl")
    gt = [d for d in g if d.get("phase") == "grpo"]
    gv = [d for d in g if d.get("phase") == "grpo_val"]
    n_it = max(d["step"] for d in gt)
    th = max((d for d in gv if d["step"] > 0), key=lambda d: (d["val_reward_mean"], -d["step"]))
    tail = 0.8 * n_it
    al = [(d["step"], a) for d in gt + gv for a in (d.get("alarms") or [])]
    out = {"seed": seed, "seed_value": SEEDS[seed],
           "sft": {"run": sft_run(seed), "val_points": len(sv), "best_step": best["step"],
                   "best_val_R": best["val_reward_mean"], "val_R_curve": [d["val_reward_mean"] for d in sv]},
           "grpo": {"run": grpo_run(seed), "iterations": n_it, "theta_step": th["step"],
                    "theta_val_R": th["val_reward_mean"], "sft_val_R": next(d["val_reward_mean"] for d in gv
                                                                            if d["step"] == 0),
                    "val_R_curve": [(d["step"], d["val_reward_mean"]) for d in gv],
                    "alarms": al, "alarms_last20": [x for x in al if x[0] > tail],
                    "kl_max": max(d["kl_ref"] for d in gt),
                    "kl_mean_last20": sum(d["kl_ref"] for d in gt if d["step"] > tail) / max(1, sum(
                        1 for d in gt if d["step"] > tail))}}
    out["grpo"]["accepted"] = not out["grpo"]["alarms_last20"]
    if eval_pq:
        from collections import Counter, defaultdict
        fam = defaultdict(Counter)
        for r in _jl(eval_pq):
            grp = r["pattern"] if r["pattern"] in ("count", "version") else "old"
            for x in r["actions"]:
                fam[grp][action_family(x)] += 1
                fam["all"][action_family(x)] += 1
        out["action_share"] = {g: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items())}
                               for g, c in fam.items()}
    return out


def acceptance(seed: str, sft_pq: str, grpo_pq: str, local: str = "runs/spark") -> dict:
    """Plan Task 9 step 2 (fix round 1, M2): GRPO is accepted iff its theta* beats the SFT on val R by more than one
    paired SE (per question, the same val rows) AND no alarm fired in the last 20 % of its iterations; otherwise the
    seed's candidate is its SFT."""
    import math
    a = {r["qkey"]: r["reward"] for r in _jl(sft_pq)}
    b = {r["qkey"]: r["reward"] for r in _jl(grpo_pq)}
    if set(a) != set(b):
        raise SystemExit(f"{seed}: the SFT and GRPO per-query files cover different rows")
    d = [b[k] - a[k] for k in sorted(a)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    al = seed_summary(seed, local)["grpo"]["alarms_last20"]
    ok = m > se and not al
    return {"seed": seed, "n": n, "sft_R": sum(a.values()) / n, "grpo_R": sum(b.values()) / n, "diff": m, "se": se,
            "gt_1se": m > se, "alarms_last20": al, "accepted": ok, "candidate": "grpo" if ok else "sft",
            "sft_pq": sft_pq, "grpo_pq": grpo_pq}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    c = sub.add_parser("chain")
    c.add_argument("--seed", required=True)
    m = sub.add_parser("summary", help="Mac: per-seed SFT / GRPO summary of the pulled runs (JSON)")
    m.add_argument("--pq", action="append", default=[], help="seed=val per-query file of theta* (action shares)")
    m.add_argument("--out", required=True)
    ac = sub.add_parser("accept", help="Mac: the plan's GRPO acceptance rule per seed (JSON)")
    ac.add_argument("--pair", action="append", required=True, help="seed=SFT_val_pq,GRPO_theta_val_pq")
    ac.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "accept":
        res = {}
        for x in a.pair:
            sd, files = x.split("=", 1)
            res[sd] = acceptance(sd, *files.split(","))
            v = res[sd]
            print(sd, f"SFT {v['sft_R']:.4f} GRPO {v['grpo_R']:.4f} diff {v['diff']:+.4f} +- {v['se']:.4f}",
                  f"alarms_last20 {len(v['alarms_last20'])} -> {v['candidate'].upper()}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if a.cmd == "summary":
        pq = dict(x.split("=", 1) for x in a.pq)
        res = {s: seed_summary(s, eval_pq=pq.get(s)) for s in SEEDS}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        for s, v in res.items():
            print(s, json.dumps({"sft_best": (v["sft"]["best_step"], v["sft"]["best_val_R"]),
                                 "theta": (v["grpo"]["theta_step"], v["grpo"]["theta_val_R"]),
                                 "alarms_last20": len(v["grpo"]["alarms_last20"]), "kl_max": round(v["grpo"]["kl_max"], 3)}))
        return 0
    if a.cmd == "jobs":
        for j in jobs():
            print(json.dumps({"name": j["name"], "steps": [[n, " ".join(x)] for n, x in j["steps"]]}))
        return 0
    return chain(a.seed)


if __name__ == "__main__":
    sys.exit(main())
