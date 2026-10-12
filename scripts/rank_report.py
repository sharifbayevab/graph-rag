"""Pilot 6 tables and figure from pulled Spark runs (runs/spark/*; light: JSONL / JSON only, no torch).

    PYTHONPATH=. python scripts/rank_report.py SPEC.json   -> markdown tables on stdout, docs/m5/figures/07_hit_at_k.png

SPEC: {"metrics": [...], "ref": "<label of the paired reference>", "pool_n": 50,
       "splits": {"val": {"ceiling": "<baselines run>", "methods": [[label, run, baseline name | null], ...]},
                  "test": {...}}}
A method row is the per-query file runs/spark/<run>/pq.jsonl (eval: one row per query; baselines: rows with "name").
Mean ± SE = sd / sqrt(n); paired difference to `ref` over the common queries, ± SE of the per-query differences."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "spark"


def load(run: str, name: str | None) -> dict:
    out = {}
    for line in (RUNS / run / "pq.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if name is None or r.get("name") == name:
                out[r["qkey"]] = r
    if not out:
        raise SystemExit(f"{run}: no per-query rows for {name!r}")
    return out


def hit_curve(rows: dict, kmax: int = 20) -> np.ndarray:
    fr = [r["first_rank"] for r in rows.values()]
    return np.array([np.mean([f is not None and f <= k for f in fr]) for k in range(1, kmax + 1)])


def stat(rows: dict, m: str) -> tuple[float, float]:
    x = np.array([r[m] for r in rows.values()], float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def paired(a: dict, b: dict, m: str) -> tuple[float, float, int]:
    ks = sorted(set(a) & set(b))
    d = np.array([a[k][m] - b[k][m] for k in ks], float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(ks)


def tables(spec: dict) -> dict:
    ms, ref = spec["metrics"], spec.get("ref")
    data = {}
    for split, sp in spec["splits"].items():
        rows = {lab: load(run, name) for lab, run, name in sp["methods"]}
        if sp.get("subset"):                              # e.g. the pilot-5 val-126 queries (trajectories.jsonl)
            sub = sp["subset"]
            keep = {json.loads(l)["id"] + "|q" for l in (ROOT / sub["file"]).read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l)["split"] == sub["split"]}
            rows = {lab: {k: v for k, v in r.items() if k in keep} for lab, r in rows.items()}
        data[split] = rows
        n = {len(r) for r in rows.values()}
        print(f"\n### {split} (n = {sorted(n)})\n")
        print("| method | " + " | ".join(ms) + " |")
        print("|---|" + "---|" * len(ms))
        for lab, r in rows.items():
            print(f"| {lab} | " + " | ".join("{:.3f} ± {:.3f}".format(*stat(r, m)) for m in ms) + " |")
        if ref and ref in rows:
            print(f"\nPaired difference to **{ref}** (± SE):\n")
            print("| method | " + " | ".join(ms) + " |")
            print("|---|" + "---|" * len(ms))
            for lab, r in rows.items():
                if lab != ref:
                    print(f"| {lab} | " + " | ".join("{:+.3f} ± {:.3f}".format(*paired(r, rows[ref], m)[:2])
                                                       for m in ms) + " |")
    return data


def figure(spec: dict, data: dict, out: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import plot_m5_figures as pf                       # palette / rcParams of the M5 figures
    plt = pf.plt
    colors = [pf.BLUE, pf.ORANGE, pf.AQUA, pf.YELLOW, pf.INK2, pf.MUTED, "#7d4fc4", "#c43d6b"]
    splits = list(spec["splits"])
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    k = np.arange(1, 21)
    for ax, split in zip(axes, splits):
        sp = spec["splits"][split]
        plot = spec.get("plot") or list(data[split])           # labels drawn (all by default)
        for i, lab in enumerate(plot):
            r = data[split][lab]
            st = (spec.get("styles") or {}).get(lab, {})
            ax.plot(k, hit_curve(r), color=st.get("color", colors[i % len(colors)]), lw=st.get("lw", 1.8),
                    ls=st.get("ls", "-"), marker=st.get("marker"), ms=4, label=lab)
        if spec.get("ylim"):
            ax.set_ylim(*spec["ylim"])
        if "ceiling_value" in sp:
            c = float(sp["ceiling_value"])
        else:
            ceil = json.loads((RUNS / sp["ceiling"] / "out.json").read_text(encoding="utf-8"))
            c = ceil["pool_ceiling"][str(spec["pool_n"])]["hit"]
        ax.axhline(c, color=pf.MUTED, ls="--", lw=1)
        ax.text(20, c - .006, f"потолок пула N={spec['pool_n']}: {c:.3f}", ha="right", va="top", color=pf.INK2,
                fontsize=9)
        n = len(next(iter(data[split].values())))
        ax.set_title(f"{sp.get('title', split)} (n = {n})")
        ax.set_xlabel("k (длина списка)")
        ax.set_xticks([1, 3, 5, 10, 15, 20])
    axes[0].set_ylabel("hit@k: доля запросов с эталоном в top-k")
    axes[-1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Пилот 6: RL-ранкер и базовые методы, hit@k", fontweight="bold", y=1.03)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", out)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    d = tables(spec)
    if spec.get("figure", True):
        figure(spec, d, ROOT / "docs" / "m5" / "figures" / "07_hit_at_k.png")
