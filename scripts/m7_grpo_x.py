"""M7 Task GRPO-X (user decision 2026-10-05; validation only, the test splits stay sealed): GRPO improvement
experiments against pilot 9 theta* (runs/p9-grpo it 220), each a Spark job run one at a time (`chain`), then the
Mac-side report (`report`: paired val R vs p9 theta*, per group, all@5, curves figure).

    PYTHONPATH=. python scripts/m7_grpo_x.py jobs                  # every job's steps (never --split test)
    PYTHONPATH=. python scripts/m7_grpo_x.py chain --job gx-noop   # on Spark: the p9 GRPO replay (20 it, new code)
    PYTHONPATH=. python scripts/m7_grpo_x.py check-noop            # Mac: gx-noop rows == p9-grpo rows 1..20
    PYTHONPATH=. python scripts/m7_grpo_x.py report --out F.json --fig F.png

Jobs (all from the pilot-9 recipe of scripts/m7_runs.py; seed 20261001 = pilot 9):
- gx-noop: p9 GRPO, 20 iterations, with the GRPO-X code (options off) -- must log exactly p9-grpo's first 20 rows;
- gx-e0: SFT from scratch stopped at epoch 16 (sft.max_epochs 16; the p9 SFT's val R there .4904) -> p9 GRPO -> eval;
- gx-e2: p9 GRPO from p9 SFT with beta .1, lr 1e-4, B 24, 1000 it, val / 50, grpo.adv_std false -> eval;
- gx-e3: p9 GRPO from p9 SFT + expert replay (lambda .2, version + count, 4 per iteration) + batch mix
  {version .5, count .3, rest} (instead of the single version upweight) -> eval of theta* (all val R) and of the
  count + Q2 pick (run.keep_top 20 keeps every validation checkpoint);
- gx-e1-rft: 32 samples per train query from p9 theta* (train.py rft; keep R_best > R_greedy + .05);
- gx-e1-ft: SFT fine-tune of p9 theta* (lr 1e-4, <= 10 epochs, early stop on val R, patience 3) on the kept rows +
  the Q2 / count expert labels -> eval;
- gx-e1-grpo (optional): short p9 GRPO (100 it, val / 10) from gx-e1-ft -> eval.
Every step's argv is printed before it runs; the chain stops at the first non-zero exit code."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.m7_runs import FLAGS, GRPO_SET, LABELS, P9, RANKINGS, SFT_SET, _last_json, _sets  # noqa: E402

SEED = 20261001
BASE = ["--variant", "struct", "--pairs", LABELS, "--rankings", RANKINGS, *FLAGS]
OVERRIDE = lambda base, extra: [k for k in base if k.split("=")[0] not in {e.split("=")[0] for e in extra}] + extra  # noqa: E731
GRPO_P9 = GRPO_SET + [f"grpo.seed={SEED}"]
E2_SET = ["grpo.beta_kl=0.1", "grpo.lr=1e-4", "grpo.batch_queries=24", "grpo.iterations=1000", "grpo.val_every=50",
          "grpo.adv_std=false"]
E3_DROP = ("grpo.upweight_pattern", "grpo.upweight_share")     # replaced by the three-way batch mix
E3_SET = ["grpo.upweight_mix=version=0.5,count=0.3",
          "grpo.replay_lambda=0.2", "grpo.replay_patterns=version,count", "grpo.replay_per_iter=4", "run.keep_top=20"]
E1_FT_SET = ["sft.lr=1e-4", "sft.max_epochs=10", "sft.patience=3"]
E1_GRPO_SET = ["grpo.iterations=100", "grpo.val_every=10"]
THETA = P9["theta"]


def _eval(ckpt: str, out: str) -> tuple[str, list[str]]:
    return ("eval", ["eval", *BASE, "--ckpt", ckpt, "--split", "val", "--per-query", out])


def steps(job: str) -> list[tuple[str, list[str]]]:
    r = f"runs/{job}"
    if job == "gx-noop":
        return [("grpo", ["grpo", *BASE, "--run", f"{r}/grpo", "--sft", P9["sft"],
                          *_sets(OVERRIDE(GRPO_P9, ["grpo.iterations=20"]))])]
    if job == "gx-e0":
        return [("sft", ["sft", *BASE, "--run", f"{r}/sft", *_sets(SFT_SET + [f"sft.seed={SEED}", "sft.max_epochs=16"])]),
                ("grpo", ["grpo", *BASE, "--run", f"{r}/grpo", "--sft", f"{r}/sft", *_sets(GRPO_P9)]),
                _eval("{best}", f"{r}/pq.jsonl")]
    if job == "gx-e2":
        return [("grpo", ["grpo", *BASE, "--run", f"{r}/grpo", "--sft", P9["sft"], *_sets(OVERRIDE(GRPO_P9, E2_SET))]),
                _eval("{best}", f"{r}/pq.jsonl")]
    if job == "gx-e3":
        return [("grpo", ["grpo", *BASE, "--run", f"{r}/grpo", "--sft", P9["sft"], *_sets(OVERRIDE([k for k in GRPO_P9 if k.split("=")[0] not in E3_DROP], E3_SET))]),
                _eval("{best}", f"{r}/pq.jsonl"),
                ("pick", [f"{r}/grpo"]),
                _eval("{pick}", f"{r}/pq_hard.jsonl")]
    if job == "gx-e1-rft":
        return [("rft", ["rft", *BASE, "--ckpt", THETA, "--split", "train", "--out", f"{r}/rft.jsonl",
                         "--rft-g", "32", "--rft-margin", "0.05", "--set", f"grpo.seed={SEED}"])]
    if job == "gx-e1-ft":
        return [("sft", ["sft", *BASE, "--run", f"{r}/sft", "--init-ckpt", THETA, "--rft", "runs/gx-e1-rft/rft.jsonl",
                         "--expert-patterns", "version,count",
                         *_sets(OVERRIDE(SFT_SET + [f"sft.seed={SEED}"], E1_FT_SET))]),
                _eval("{best}", f"{r}/pq.jsonl")]
    if job == "gx-e1-grpo":
        return [("grpo", ["grpo", *BASE, "--run", f"{r}/grpo", "--sft", "runs/gx-e1-ft/sft",
                          *_sets(OVERRIDE(GRPO_P9, E1_GRPO_SET))]),
                _eval("{best}", f"{r}/pq.jsonl")]
    raise SystemExit(f"--job {job}: one of {JOBS}")


JOBS = ("gx-noop", "gx-e0", "gx-e2", "gx-e3", "gx-e1-rft", "gx-e1-ft", "gx-e1-grpo")


def jobs() -> list[dict]:
    out = [{"name": j, "steps": steps(j)} for j in JOBS]
    for j in out:
        for _, argv in j["steps"]:
            if any(x == "--split" and y not in ("val", "train") for x, y in zip(argv, argv[1:])):
                raise SystemExit(f"{j['name']}: a step reads a split other than val / train")
    return out


def _jl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


HARD = ("count", "version")


def hard_r(v: dict) -> float:
    """Val R over count + Q2 (version) rows of a grpo_val / sft_val row (n-weighted over val_by_group)."""
    bg = v["val_by_group"]
    n = sum(bg[g]["n"] for g in HARD)
    return sum(bg[g]["reward_mean"] * bg[g]["n"] for g in HARD) / n


def pick_hard(run: str) -> str:
    """The checkpoint of the validation point with the best count + Q2 val R (ties: earlier step); it must still be on
    disk (run.keep_top large enough)."""
    rows = [d for d in _jl(Path(run) / "metrics.jsonl") if d.get("phase") == "grpo_val" and d["step"] > 0]
    best = max(rows, key=lambda d: (round(hard_r(d), 9), -d["step"]))
    f = Path(run) / "checkpoints" / f"grpo-{best['step']:06d}.pt"
    if not f.is_file():
        raise SystemExit(f"{f}: the count + Q2 pick is not on disk")
    print(json.dumps({"pick_hard": str(f), "step": best["step"], "hard_val_R": round(hard_r(best), 6),
                      "val_R": best["val_reward_mean"]}), flush=True)
    return str(f)


def chain(job: str) -> int:
    best = pick = None
    for name, argv in steps(job):
        if name == "pick":
            pick = pick_hard(argv[0])
            continue
        argv = [best if x == "{best}" else pick if x == "{pick}" else x for x in argv]
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
        if name in ("grpo", "sft"):
            out = _last_json(lines)
            best = out["best"] if name == "grpo" else str(Path(argv[argv.index("--run") + 1]) / "pi_ref.pt")
    return 0


# ---- Mac side -------------------------------------------------------------------------------------------------------

VOLATILE = ("time",)


def check_noop(local: str = "runs/spark") -> dict:
    """gx-noop (GRPO-X code, options off) vs p9-grpo: grpo rows 1..20 and grpo_val rows 0 / 20, all fields but time."""
    a = [d for d in _jl(Path(local) / "p9-grpo/metrics.jsonl") if d["step"] <= 20]
    b = _jl(Path(local) / "gx-noop/grpo/metrics.jsonl")
    strip = lambda rows: sorted((json.dumps({k: v for k, v in d.items() if k not in VOLATILE}, sort_keys=True)  # noqa: E731
                                 for d in rows))
    sa, sb = strip(a), strip(b)
    out = {"p9_rows": len(sa), "noop_rows": len(sb), "identical": sa == sb,
           "differing": len(set(sa) ^ set(sb))}
    print(json.dumps(out))
    return out


def _labels(path: str = "runs/spark/p9-expert/labels.jsonl") -> dict:
    return {r["id"]: r for r in _jl(path)}


def _pq(path) -> dict:
    return {r["qkey"].split("|")[0]: r for r in _jl(path)}


def mean_se_d(xs) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def compare(base: dict, other: dict, labels: dict) -> dict:
    """Per cell (all / old / Q1 / Q1x / Q2 / count+Q2): R and all@5 of both, paired ΔR ± SE and Δall@5 ± SE."""
    from scripts.m7_pilot9 import fine_group
    if set(base) != set(other):
        raise SystemExit(f"per-query files differ in keys ({len(set(base) ^ set(other))})")
    cells = defaultdict(list)
    for k in sorted(base):
        fg = fine_group(labels[k])
        for c in ("all", fg) + (("hard",) if fg != "old" else ()):
            cells[c].append(k)
    out = {}
    for c in ("all", "old", "Q1", "Q1x", "Q2", "hard"):
        ks = cells.get(c, [])
        if not ks:
            continue
        rb = [base[k]["reward"] for k in ks]
        ro = [other[k]["reward"] for k in ks]
        ab = [float(base[k]["recall"] >= 1 - 1e-9) for k in ks]
        ao = [float(other[k]["recall"] >= 1 - 1e-9) for k in ks]
        dr = mean_se_d([y - x for x, y in zip(rb, ro)])
        da = mean_se_d([y - x for x, y in zip(ab, ao)])
        out[c] = {"n": len(ks), "R_base": round(sum(rb) / len(ks), 6), "R": round(sum(ro) / len(ks), 6),
                  "dR": round(dr[0], 6), "dR_se": round(dr[1], 6), "all5_base": round(sum(ab) / len(ks), 6),
                  "all5": round(sum(ao) / len(ks), 6), "dall5": round(da[0], 6), "dall5_se": round(da[1], 6)}
    return out


def curve(run: str, phase: str = "grpo_val") -> list[tuple[int, float, float]]:
    rows = [d for d in _jl(Path(run) / "metrics.jsonl") if d.get("phase") == phase]
    return [(d["step"], d["val_reward_mean"], hard_r(d)) for d in rows]


def grpo_diag(run: str) -> dict:
    """theta* point, KL max / last-20 % mean, alarms (anywhere / last 20 %) of a GRPO run."""
    rows = _jl(Path(run) / "metrics.jsonl")
    gt = [d for d in rows if d.get("phase") == "grpo"]
    gv = [d for d in rows if d.get("phase") == "grpo_val" and d["step"] > 0]
    n = max(d["step"] for d in gt)
    th = max(gv, key=lambda d: (d["val_reward_mean"], -d["step"]))
    al = [(d["step"], a) for d in gt + gv for a in (d.get("alarms") or [])]
    tail = [d for d in gt if d["step"] > 0.8 * n]
    return {"iterations": n, "theta_step": th["step"], "theta_val_R": th["val_reward_mean"],
            "sft_val_R": next(d["val_reward_mean"] for d in rows if d.get("phase") == "grpo_val" and d["step"] == 0),
            "kl_max": round(max(d["kl_ref"] for d in gt), 4),
            "kl_last20": round(sum(d["kl_ref"] for d in tail) / len(tail), 4),
            "alarms": len(al), "alarms_last20": len([x for x in al if x[0] > 0.8 * n]),
            "alarm_kinds_last20": sorted({a.split(":")[0] for s, a in al if s > 0.8 * n})}


RUNS = {  # name -> (per-query file, run dir with the selection curve, phase)
    "E0": ("gx-e0/pq.jsonl", "gx-e0/grpo", "grpo_val"),
    "E1": ("gx-e1-ft/pq.jsonl", "gx-e1-ft/sft", "sft_val"),
    "E1+GRPO": ("gx-e1-grpo/pq.jsonl", "gx-e1-grpo/grpo", "grpo_val"),
    "E2": ("gx-e2/pq.jsonl", "gx-e2/grpo", "grpo_val"),
    "E3": ("gx-e3/pq.jsonl", "gx-e3/grpo", "grpo_val"),
    "E3 (count+Q2 pick)": ("gx-e3/pq_hard.jsonl", "gx-e3/grpo", "grpo_val"),
}


def report(out: str, fig: str, local: str = "runs/spark") -> dict:
    L = Path(local)
    labels = _labels(str(L / "p9-expert/labels.jsonl"))
    base = _pq(L / "p9-ev-g-k5/pq.jsonl")
    res = {"base": "p9 theta* (runs/spark/p9-grpo/checkpoints/grpo-000220.pt, val per-query p9-ev-g-k5)", "runs": {}}
    curves = {"p9": curve(str(L / "p9-grpo"))}
    for name, (pq, run, phase) in RUNS.items():
        if not (L / pq).is_file():
            continue
        res["runs"][name] = {"cmp": compare(base, _pq(L / pq), labels)}
        if phase == "grpo_val":
            res["runs"][name]["grpo"] = grpo_diag(str(L / run))
        if name != "E3 (count+Q2 pick)":
            curves[name] = curve(str(L / run), phase)
    if (L / "gx-e0/sft/metrics.jsonl").is_file():
        sv = {d["step"]: d["val_reward_mean"] for d in _jl(L / "gx-e0/sft/metrics.jsonl") if d.get("phase") == "sft_val"}
        p9 = {d["step"]: d["val_reward_mean"] for d in _jl(L / "p9-sft-scratch/metrics.jsonl")
              if d.get("phase") == "sft_val"}
        res["e0_sft_check"] = {"epoch16_val_R": sv.get(16), "p9_epoch16_val_R": p9.get(16),
                               "abs_diff": None if sv.get(16) is None else round(abs(sv[16] - p9[16]), 6),
                               "max_abs_diff_epochs_1_16": round(max(abs(sv[e] - p9[e]) for e in sv if e in p9), 6)}
    if (L / "gx-e1-rft/rft.jsonl").is_file():
        rows = _jl(L / "gx-e1-rft/rft.jsonl")
        by = defaultdict(list)
        for r in rows:
            by["all"].append(r)
            by[r["pattern"] if r["pattern"] in HARD else "old"].append(r)
        res["e1_rft"] = {g: {"n": len(v), "kept": sum(r["keep"] for r in v),
                             **{k: round(sum(r[k] for r in v) / len(v), 4) for k in ("r_greedy", "r_mean", "r_best")}}
                         for g, v in sorted(by.items())}
    res["curves"] = curves
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    plot(curves, res, fig)
    return res


def plot(curves: dict, res: dict, fig: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = max(r for s, r, _ in curves["p9"] if s > 0)
    f, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    colors = {"p9": "#555555", "E0": "#1f77b4", "E1": "#d62728", "E1+GRPO": "#ff7f0e", "E2": "#2ca02c", "E3": "#9467bd"}
    for name, pts in curves.items():
        if name == "E1":                                       # SFT epochs: x = epoch * 10 to share the axis? no --
            continue                                           # E1 is a fine-tune (epochs), drawn on the inset axis
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=colors.get(name, None),
                label="p9 GRPO" if name == "p9" else name)
    ax.axhline(th, color="#555555", ls="--", lw=1, label=f"p9 θ* ({th:.3f})")
    ax.set_xlabel("итерация GRPO")
    ax.set_ylabel("val R (greedy)")
    if "E1" in curves:
        ax2 = ax.inset_axes([0.62, 0.08, 0.35, 0.33])
        pts = curves["E1"]
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, color=colors["E1"])
        ax2.axhline(th, color="#555555", ls="--", lw=1)
        ax2.set_title("E1: эпохи RFT (0 = p9 θ*)", fontsize=7)
        ax2.tick_params(labelsize=6)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Val R по ходу обучения: E0–E3 против p9 θ*", fontsize=10)
    ax.grid(alpha=0.3)
    f.tight_layout()
    Path(fig).parent.mkdir(parents=True, exist_ok=True)
    f.savefig(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    c = sub.add_parser("chain")
    c.add_argument("--job", required=True, choices=JOBS)
    sub.add_parser("check-noop")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--fig", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        for j in jobs():
            print(json.dumps({"name": j["name"], "steps": [[n, " ".join(x)] for n, x in j["steps"]]}))
        return 0
    if a.cmd == "check-noop":
        return 0 if check_noop()["identical"] else 1
    if a.cmd == "report":
        res = report(a.out, a.fig)
        for k, v in res["runs"].items():
            c = v["cmp"]
            print(k, json.dumps({g: (c[g]["R"], c[g]["dR"], c[g]["dR_se"], c[g]["all5"]) for g in c}))
        return 0
    return chain(a.job)


if __name__ == "__main__":
    sys.exit(main())
