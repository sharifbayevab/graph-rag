"""M7 reward calibration under the design constraint "all four components take part": w_i >= EPS (0.1).

    PYTHONPATH=. python scripts/plot_m7_weight_constrained.py
        -> docs/m7/figures/rc_constrained_sweeps.png, rc_constrained_opt.png + data/m7/calib/constrained.json
In-sample tau on the informative questions of the 80-question subsample (same data as rc_simplex); bootstrap over
questions; nested 5-fold CV over the same document folds as `fit`."""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from scripts.plot_m7_calib_figures import (BLUE, GRID, INK, INK2, MUTED, ORANGE, CALIB, cr, load, plt, save)  # noqa: E402
from scripts.m7_reward_calib import _groups  # noqa: E402

N, EPS, B, SEED = 20, 0.1, 2000, 20261011
KEYS = ("f1", "coverage", "diversity", "coherence")
LAB = ("F1", "coverage", "diversity", "coherence")
CUR = np.array([cr.CURRENT[k] for k in KEYS])
RED = "#d62728"


def taus(G, w):
    return np.array([cr.kendall_tau(F @ w, U) for F, U in G])


def main() -> int:
    sub, names, util, folds, _ = load()
    groups = _groups(sub, names, util, 1.0)
    idx = [cr.feature_names(1.0).index(k) for k in KEYS]
    G = [(F[:, idx], U) for _, F, U, _ in groups]
    qid = [q for q, _, _, _ in groups]
    pts = np.array([c for c in product(range(N + 1), repeat=4) if sum(c) == N], float) / N
    T = np.array([taus(G, w) for w in pts])
    ok = ~np.isnan(T).any(0)
    T, G = T[:, ok], [g for g, o in zip(G, ok) if o]
    fo = np.array([folds[q] for q, o in zip(qid, ok) if o])
    nq = T.shape[1]
    tau = T.mean(1)
    ci = int(np.argmin(np.abs(pts - CUR).sum(1)))
    M = (pts >= EPS - 1e-9).all(1)

    # ---- in-sample constrained optimum
    best = float(tau[M].max())
    top = M & (tau >= best - 1e-12)
    order = np.flatnonzero(M)[np.argsort(-tau[M], kind="stable")]
    rank_strict = int((tau[M] > tau[ci] + 1e-12).sum()) + 1          # 1 = nobody strictly better

    # ---- bootstrap over questions: how often is each point the constrained argmax
    # tau(w) is piecewise constant: grid points with identical per-question tau form one ordering cell
    cell_of = {}
    for i in np.flatnonzero(M):
        cell_of.setdefault(T[i].round(9).tobytes(), []).append(i)
    cells = list(cell_of.values())
    Tc = T[[c[0] for c in cells]]
    c_cur = next(j for j, c in enumerate(cells) if ci in c)
    rng = np.random.default_rng(SEED)
    pcell, reg = np.zeros(len(cells)), []
    for _ in range(B):
        m = Tc[:, rng.integers(0, nq, nq)].mean(1)
        pcell += m >= m.max() - 1e-12
        reg.append(m.max() - m[c_cur])
    pcell /= B
    reg = np.array(reg)

    # ---- nested CV: grid search on 4 folds, evaluate the chosen (tie-averaged) weights on the 5th
    sel, chosen = np.zeros(nq), []
    for f in range(5):
        tr, te = fo != f, fo == f
        m = T[:, tr].mean(1)
        m[~M] = -np.inf
        tie = np.flatnonzero(m >= m.max() - 1e-12)
        sel[te] = T[tie][:, te].mean(0)
        chosen.append(pts[tie].mean(0).round(3).tolist())
    d = sel - T[ci]

    # ---- eps curve: best tau when every weight >= eps
    eps_grid = np.round(np.arange(0, 0.2001, 0.025), 3)
    eps_best = [float(tau[(pts >= e - 1e-9).all(1)].max()) for e in eps_grid]

    # ---- pairwise transfers through the current point: w_i += δ, w_j -= δ (other two fixed)
    PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    ds = np.round(np.arange(-0.5, 0.5001, 0.0125), 4)
    sweeps = {}
    for i, j in PAIRS:
        W = np.array([CUR + d * (np.eye(4)[i] - np.eye(4)[j]) for d in ds])
        valid = (W >= -1e-9).all(1)
        sweeps[f"{KEYS[i]}<-{KEYS[j]}"] = {"delta": ds[valid].tolist(), "w_i": W[valid, i].tolist(),
                                            "tau": [float(taus(G, w).mean()) for w in W[valid]],
                                            "allowed": (W[valid] >= EPS - 1e-9).all(1).tolist()}

    print(f"nq={nq} grid={M.sum()} best={best:.4f} cur={tau[ci]:.4f} strict_rank={rank_strict} ties_at_max={top.sum()} "
          f"cells={len(cells)} P_opt cur-cell={pcell[c_cur]:.3f} max={pcell.max():.3f} regret med={np.median(reg):.4f} q95={np.quantile(reg, .95):.4f} "
          f"nestedCV delta={d.mean():.4f}±{d.std(ddof=1) / np.sqrt(nq):.4f}")

    # ================= figure 1: pairwise transfers
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.6), sharey=True)
    for ax, (i, j) in zip(axs.flat, PAIRS):
        sw = sweeps[f"{KEYS[i]}<-{KEYS[j]}"]
        x, y, al = np.array(sw["w_i"]), np.array(sw["tau"]), np.array(sw["allowed"])
        ax.axvspan(x.min(), x[al].min(), color=MUTED, alpha=.13, lw=0)
        ax.axvspan(x[al].max(), x.max(), color=MUTED, alpha=.13, lw=0)
        ax.plot(x, y, color=MUTED, lw=1.3)
        ax.plot(x[al], y[al], color=BLUE, lw=2.8)
        ax.scatter([CUR[i]], [tau[ci]], marker="*", s=380, color=RED, edgecolors=INK, linewidths=1, zorder=5)
        ax.axhline(best, color=INK2, lw=1, ls=":")
        ax.set_title(f"{LAB[i]} ← {LAB[j]}", fontsize=14)
        ax.set_xlabel(f"w({LAB[i]});  w({LAB[j]}) = {CUR[i] + CUR[j]:.1f} − w({LAB[i]})".replace(".", ","), fontsize=10)
        ax.set_xlim(x.min(), x.max())
    for ax in axs[:, 0]:
        ax.set_ylabel("Kendall τ (R против U)", fontsize=12)
    axs[0, 0].plot([], [], color=BLUE, lw=2.8, label="все wᵢ ≥ 0,1")
    axs[0, 0].plot([], [], color=MUTED, lw=1.3, label="какой-то wᵢ < 0,1 (серая зона)")
    axs[0, 0].plot([], [], color=INK2, lw=1, ls=":", label="максимум τ при wᵢ ≥ 0,1")
    axs[0, 0].scatter([], [], marker="*", s=180, color=RED, edgecolors=INK, label="текущие 0,5 / 0,2 / 0,2 / 0,1")
    axs[0, 0].legend(loc="lower right", fontsize=9.5)
    fig.suptitle("Перенос веса между двумя компонентами (две другие неизменны): "
                 "на допустимом участке τ нигде не выше текущей точки",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "rc_constrained_sweeps.png")

    # ================= figure 2: eps curve | slice | bootstrap
    fig = plt.figure(figsize=(20, 6.4))
    ax1 = fig.add_axes([.04, .14, .25, .70])
    ax1.plot(eps_grid, eps_best, color=BLUE, lw=2.6, marker="o", ms=6, label="лучшая τ при всех wᵢ ≥ ε")
    ax1.axhline(tau[ci], color=RED, lw=2, ls="--", label="текущие веса, τ = " + f"{tau[ci]:.3f}".replace(".", ","))
    ax1.axvline(EPS, color=INK2, lw=1, ls=":")
    ax1.text(EPS + .004, max(eps_best) - .002, "ε = 0,1:\nвсе 4 компоненты\nне меньше 10 %", fontsize=10, color=INK2, va="top")
    ax1.set_xlabel("ε — минимальная доля каждой компоненты", fontsize=11)
    ax1.set_ylabel("Kendall τ", fontsize=11)
    ax1.set_title("(а) Цена требования «участвуют все 4»", fontsize=13)
    ax1.legend(loc="lower left", fontsize=10)

    ax2 = fig.add_axes([.33, .10, .28, .76])
    s = 0.1
    m = np.abs(pts[:, 3] - s) < 1e-9
    P, sl = pts[m], 1 - s
    px = P[:, 1] / sl + 0.5 * P[:, 2] / sl
    py = P[:, 2] / sl * np.sqrt(3) / 2
    tri = mtri.Triangulation(px, py)
    vmin, vmax = float(tau.min()), float(tau.max())
    cf = ax2.tricontourf(tri, tau[m], levels=14, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.tricontour(tri, (pts[m] >= EPS - 1e-9).all(1).astype(float), levels=[.5], colors=["white"], linewidths=2.6)
    tm = top[m]
    ax2.scatter(px[tm], py[tm], s=60, facecolors="none", edgecolors="white", linewidths=1.6, zorder=6)
    cx = (CUR[1] / sl + 0.5 * CUR[2] / sl, CUR[2] / sl * np.sqrt(3) / 2)
    ax2.scatter(*cx, marker="*", s=650, color=RED, edgecolors=INK, linewidths=1.2, zorder=10)
    ax2.annotate("текущие", cx, xytext=(28, 34), textcoords="offset points", color=RED, fontsize=15,
                 fontweight="bold", arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.4})
    ax2.plot([0, 1, .5, 0], [0, 0, np.sqrt(3) / 2, 0], color=INK2, lw=1.2)
    ax2.text(-.03, -.04, "F1", fontsize=15, fontweight="bold", ha="right", va="top")
    ax2.text(1.03, -.04, "coverage", fontsize=15, fontweight="bold", ha="center", va="top")
    ax2.text(.5, np.sqrt(3) / 2 + .03, "diversity", fontsize=15, fontweight="bold", ha="center", va="bottom")
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    ax2.set_xlim(-.1, 1.1)
    ax2.set_ylim(-.12, .96)
    ax2.set_title("(б) Срез coherence = 0,1: белая рамка — все wᵢ ≥ 0,1,\nкружки — точки максимума τ", fontsize=13)
    cb = fig.colorbar(cf, cax=fig.add_axes([.615, .2, .008, .55]))
    cb.set_label("Kendall τ", fontsize=11)

    ax3 = fig.add_axes([.765, .14, .22, .70])
    k6 = np.argsort(-pcell, kind="stable")[:6]
    lbl = []
    for j in k6:
        c = pts[cells[j]].mean(0)
        lbl.append(" / ".join(f"{v:.2f}" for v in c).replace(".", ",") + f"\n{len(cells[j])} т. сетки, τ = " + f"{Tc[j].mean():.3f}".replace(".", ","))
    ax3.barh(range(len(k6))[::-1], pcell[k6], color=[RED if j == c_cur else BLUE for j in k6])
    for r, j in zip(range(len(k6))[::-1], k6):
        ax3.text(pcell[j] + .003, r, f"{pcell[j]:.0%}", va="center", fontsize=11)
    ax3.set_yticks(range(len(k6))[::-1])
    ax3.set_yticklabels(lbl, fontsize=9.5)
    ax3.set_xlim(0, pcell.max() * 1.22)
    ax3.set_xlabel(f"доля bootstrap-выборок, где ячейка — максимум (B = {B})", fontsize=10)
    ax3.set_title(f"(в) Какая ячейка весов — максимум\n(центр ячейки F1 / cov / div / coh; всего {len(cells)} ячеек)", fontsize=13)
    ax3.grid(axis="y", visible=False)
    fig.suptitle("При условии «участвуют все 4 компоненты» (wᵢ ≥ 0,1) текущие веса — точка максимума",
                 fontsize=16, fontweight="bold", y=1.0)
    save(fig, "rc_constrained_opt.png")

    res = {"eps": EPS, "n_questions": nq, "n_points_constrained": int(M.sum()),
           "best_tau": best, "current_tau": float(tau[ci]), "current_strict_rank": rank_strict,
           "points_at_max": pts[top].tolist(),
           "top10": [{"w": pts[i].tolist(), "tau": float(tau[i])} for i in order[:10]],
           "bootstrap": {"B": B, "seed": SEED, "n_cells": len(cells), "p_opt_current_cell": float(pcell[c_cur]),
                         "top_cells": [{"center": pts[cells[j]].mean(0).tolist(), "n_points": len(cells[j]),
                                        "tau": float(Tc[j].mean()), "p_opt": float(pcell[j])} for j in k6],
                         "regret_median": float(np.median(reg)), "regret_q95": float(np.quantile(reg, .95))},
           "nested_cv": {"chosen": chosen, "heldout_tau": float(sel.mean()), "current_tau": float(T[ci].mean()),
                         "delta": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(nq))},
           "eps_curve": {"eps": eps_grid.tolist(), "best_tau": eps_best},
           "sweeps": sweeps,
           "note": "in-sample tau on informative questions of the 80-question subsample, F1 (beta=1); EPS = design constraint"}
    (ROOT / CALIB / "constrained.json").write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
