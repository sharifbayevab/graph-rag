"""M7 reward-weight sweep by the final result: figures from data/m7/calib/sweep.json (scripts/m7_weight_sweep_report.py)
and the tau grid (data/m7/calib/simplex_grid.json). Validation only.

    PYTHONPATH=. python scripts/plot_m7_ws_final.py
        -> docs/m7/figures/ws_points.png   every weight vector: per-seed all@5, mean, current-weights band
           docs/m7/figures/ws_edge.png     the diversity = coherence = 0.1 edge: all@5 vs the F1 share (+ current)
           docs/m7/figures/ws_tau.png      calibration tau (in-sample) vs the final all@5: does tau predict training?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.plot_m7_calib_figures import BLUE, INK, INK2, MUTED, plt, save  # noqa: E402

RED, GREEN = "#d62728", "#1baf7a"
NAME = {"cur": "текущие", "eq": "равные", "f1": "F1-угол", "cov": "cov-угол", "div": "div-угол", "coh": "coh-угол"}


def lab(p, w):
    return f"{NAME.get(p, p)}  " + " / ".join(f"{x:g}".replace(".", ",") for x in w)


def main() -> int:
    S = json.loads((ROOT / "data/m7/calib/sweep.json").read_text(encoding="utf-8"))
    cur = S["cur"]
    seeds = lambda r: [v["all5"] for v in r["seeds"].values()]          # noqa: E731

    # ---- 1. every point
    order = sorted(S, key=lambda p: S[p]["all5"])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(order) + 1.8))
    ax.axvspan(cur["all5"] - cur["all5_sd"], cur["all5"] + cur["all5_sd"], color=RED, alpha=.08, lw=0)
    ax.axvline(cur["all5"], color=RED, lw=1.2, ls="--")
    for i, p in enumerate(order):
        r = S[p]
        v = seeds(r)
        c = RED if p == "cur" else (GREEN if r["d_all5_ci"][0] > 0 else (MUTED if r["d_all5_ci"][1] >= 0 else BLUE))
        ax.scatter(v, [i] * len(v), s=36, color=c, alpha=.55, edgecolors="none", zorder=3)
        ax.scatter([r["all5"]], [i], s=220 if p == "cur" else 110, marker="*" if p == "cur" else "D", color=c,
                   edgecolors=INK, zorder=4)
        ax.text(1.005, i, f"{r['all5']:.3f}".replace(".", ","), va="center", fontsize=10, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab(p, S[p]["w"]) for p in order], fontsize=10)
    ax.set_xlabel("val all@5 агента (кружки — сиды, ромб — среднее);  веса F1 / coverage / diversity / coherence",
                  fontsize=10)
    ax.set_title("Подбор весов по итоговому результату: каждый набор обучен заново (эксперт → SFT → GRPO)",
                 fontsize=13)
    ax.grid(axis="y", visible=False)
    save(fig, "ws_points.png")

    # ---- 2. the edge diversity = coherence = 0.1: all@5 vs the F1 share
    edge = sorted((p for p in S if p != "cur" and abs(S[p]["w"][2] - .1) < 1e-9 and abs(S[p]["w"][3] - .1) < 1e-9),
                  key=lambda p: S[p]["w"][0])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array([S[p]["w"][0] for p in edge])
    m = np.array([S[p]["all5"] for p in edge])
    sd = np.array([S[p]["all5_sd"] if np.isfinite(S[p]["all5_sd"]) else 0 for p in edge])
    ax.fill_between(x, m - sd, m + sd, color=BLUE, alpha=.12, lw=0)
    ax.plot(x, m, color=BLUE, lw=2.5, marker="o", ms=8, label="diversity = coherence = 0,1 (среднее ± sd по сидам)")
    for p in edge:
        v = seeds(S[p])
        ax.scatter([S[p]["w"][0]] * len(v), v, s=22, color=BLUE, alpha=.45, edgecolors="none")
    ax.errorbar([cur["w"][0]], [cur["all5"]], yerr=[cur["all5_sd"]], fmt="*", ms=18, color=RED, mec=INK,
                capsize=5, label="текущие 0,5 / 0,2 / 0,2 / 0,1 (± sd; вне ребра: diversity 0,2, показаны при F1 = 0,5)", zorder=5)
    ax.set_xlabel("вес F1 / вес coverage", fontsize=11)
    ax.set_ylabel("val all@5 агента", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g} / {.8 - a:.1f}".replace(".", ",") for a in x], fontsize=9)
    ax.set_title("Ребро «diversity = coherence = 0,1»: качество от раздела веса между F1 и coverage", fontsize=12)
    ax.legend(loc="lower center", fontsize=10)
    save(fig, "ws_edge.png")

    # ---- 3. calibration tau (in-sample grid) vs the final all@5
    G = json.loads((ROOT / "data/m7/calib/simplex_grid.json").read_text(encoding="utf-8"))
    pts, tau = np.array(G["grid"]), np.array(G["tau"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    X, Y = [], []
    for p, r in S.items():
        i = int(np.argmin(np.abs(pts - np.array(r["w"])).sum(1)))
        X.append(tau[i])
        Y.append(r["all5"])
        ax.scatter([tau[i]], [r["all5"]], s=200 if p == "cur" else 70, marker="*" if p == "cur" else "o",
                   color=RED if p == "cur" else BLUE, edgecolors=INK, zorder=3)
        ax.annotate(NAME.get(p, p), (tau[i], r["all5"]), xytext=(6, 4), textcoords="offset points", fontsize=9,
                    color=INK2)
    rx, ry = np.argsort(np.argsort(X)), np.argsort(np.argsort(Y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    ax.set_xlabel("Kendall τ калибровки (R против U, без обучения)", fontsize=11)
    ax.set_ylabel("val all@5 после полного обучения", fontsize=11)
    ax.set_title(f"Калибровка τ против итога обучения: Spearman ρ = {rho:.2f}".replace(".", ","), fontsize=12)
    save(fig, "ws_tau.png")
    print(f"spearman(tau, final all@5) = {rho:.3f} over {len(X)} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
