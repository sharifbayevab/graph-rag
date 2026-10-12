"""M7 reward calibration: tau over the weight simplex (F1, coverage, diversity, coherence), step 0.05.

    PYTHONPATH=. python scripts/plot_m7_weight_simplex.py -> docs/m7/figures/rc_simplex.png + data/m7/calib/simplex_grid.json
In-sample on all informative questions (U not constant) of the 80-question subsample; zone = tau* - tau <= 2*SE(paired diff)."""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.plot_m7_calib_figures import (BLUE, GRID, INK, INK2, ORANGE, SURFACE, CALIB, cr, load, plt, save)  # noqa: E402
from scripts.m7_reward_calib import _groups  # noqa: E402

N = 20
KEYS = ("f1", "coverage", "diversity", "coherence")
CUR = np.array([cr.CURRENT[k] for k in KEYS])
RED = "#d62728"


def bary(P):  # simplex points -> 3D tetrahedron coordinates
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return P @ V


def main() -> int:
    sub, names, util, folds, fit = load()
    G = [(F, U) for _, F, U, _ in _groups(sub, names, util, 1.0)]
    idx = [cr.feature_names(1.0).index(k) for k in KEYS]
    G = [(F[:, idx], U) for F, U in G]
    pts = np.array([c for c in product(range(N + 1), repeat=4) if sum(c) == N], float) / N
    assert len(pts) == 1771
    T = np.array([[cr.kendall_tau(F @ w, U) for F, U in G] for w in pts])   # points x questions
    ok = ~np.isnan(T).any(0)
    T = T[:, ok]
    nq = T.shape[1]
    tau = T.mean(1)
    se = T.std(1, ddof=1) / np.sqrt(nq)
    b = int(tau.argmax())
    d = T[b] - T                                   # paired per-question difference
    se_d = d.std(1, ddof=1) / np.sqrt(nq)
    gap = tau[b] - tau
    zone = gap <= 2 * se_d + 1e-12
    ci = int(np.argmin(np.abs(pts - CUR).sum(1)))
    tau_cur = float(tau[ci])
    ia = 0.5 * 0 + 0   # placeholder to keep names explicit
    print(f"nq={nq} best={pts[b]} tau*={tau[b]:.4f} se={se[b]:.4f} cur={tau_cur:.4f} se={se[ci]:.4f} "
          f"zone={zone.sum()}/{len(pts)} cur_in_zone={bool(zone[ci])} gap_cur={gap[ci]:.4f} se_d_cur={se_d[ci]:.4f}")
    dist = float(np.abs(pts[b] - CUR).sum() / 2)

    # ---- figure
    fig = plt.figure(figsize=(17, 8.2))
    ax = fig.add_axes([.10, .12, .40, .72], projection="3d")
    ax.computed_zorder = False
    ax.set_facecolor(SURFACE)
    X = bary(pts)
    V = bary(np.eye(4))
    for i, j in product(range(4), repeat=2):
        if i < j:
            ax.plot(*zip(V[i], V[j]), color=INK2, lw=0.8, zorder=1)
    out = ~zone
    vmin, vmax = tau.min(), tau.max()
    ax.scatter(*X[out].T, c=tau[out], cmap="viridis", vmin=vmin, vmax=vmax, s=7, alpha=.15, lw=0, depthshade=False, zorder=2)
    sc = ax.scatter(*X[zone].T, c=tau[zone], cmap="viridis", vmin=vmin, vmax=vmax, s=30, alpha=.7,
                    edgecolors="none", depthshade=False, zorder=3)
    xc, xb = bary(CUR[None])[0], X[b]
    ax.scatter(*xc, marker="*", s=700, color=RED, edgecolors=INK, linewidths=1.5, zorder=20, depthshade=False)
    ax.scatter(*xb, marker="X", s=240, color="black", edgecolors="white", linewidths=1.0, zorder=20, depthshade=False)
    ax.text(*(xc + np.array([.3, .0, .45])), "текущие", color=RED, fontsize=17, fontweight="bold")
    ax.text(*(xb + np.array([.0, .15, .02])), "← лучшая", color="black", fontsize=13, fontweight="bold", ha="left")
    for v, lab in zip(V, ("F1", "coverage", "diversity", "coherence")):
        off = v * (.12 if lab == "coverage" else .2)
        ax.text(*(v + off), lab, fontsize=18, fontweight="bold", ha="center", va="center")
    ax.set_axis_off()
    ax.view_init(elev=18, azim=62)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    fig.text(.27, .88, "(а) Симплекс весов: тетраэдр", fontsize=20, fontweight="bold", ha="center")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="viridis"), cax=fig.add_axes([.04, .2, .015, .55]))
    cb.set_label("Kendall τ (согласие с полезностью)", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    ax2 = fig.add_axes([.54, .12, .38, .72])
    s = 0.1
    m = np.abs(pts[:, 3] - s) < 1e-9
    P = pts[m]
    sl = 1 - s
    a, c = P[:, 0] / sl, P[:, 1] / sl                                  # F1 share, coverage share in the slice
    x2 = c + (1 - a - c) * 0.5 * 0 + 0                                  # placeholder, overwritten below
    div = P[:, 2] / sl
    xs = c + 0.5 * div                                                  # corners: F1 (0,0), coverage (1,0), diversity (.5,.87)
    ys = div * np.sqrt(3) / 2
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(xs, ys)
    cf = ax2.tricontourf(tri, tau[m], levels=14, cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.tricontour(tri, (gap - 2 * se_d)[m], levels=[0], colors=["white"], linewidths=3.2)
    ax2.tricontour(tri, (gap - 2 * se_d)[m], levels=[0], colors=[INK], linewidths=1.6, linestyles="--")
    ax2.scatter(xs, ys, s=5, color=INK, alpha=.25, lw=0)
    for k in (0, 1, 2):
        pass
    cx = lambda w: (w[1] / (1 - s) + 0.5 * w[2] / (1 - s), w[2] / (1 - s) * np.sqrt(3) / 2)   # noqa: E731
    ax2.scatter(*cx(CUR), marker="*", s=700, color=RED, edgecolors=INK, linewidths=1.2, zorder=10)
    ax2.annotate("текущие", cx(CUR), xytext=(30, 40), textcoords="offset points", color=RED, fontsize=17,
                 fontweight="bold", arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.5})
    bs = abs(pts[b][3] - s) < 1e-9
    if bs:
        ax2.scatter(*cx(pts[b]), marker="X", s=240, color="black", edgecolors="white", linewidths=1, zorder=10)
    ax2.plot([0, 1, .5, 0], [0, 0, np.sqrt(3) / 2, 0], color=INK2, lw=1.2)
    ax2.text(-.03, -.04, "F1", fontsize=18, fontweight="bold", ha="right", va="top")
    ax2.text(1.03, -.04, "coverage", fontsize=18, fontweight="bold", ha="center", va="top")
    ax2.text(.5, np.sqrt(3) / 2 + .03, "diversity", fontsize=18, fontweight="bold", ha="center", va="bottom")
    ax2.set_aspect("equal"); ax2.set_axis_off()
    ax2.set_xlim(-.1, 1.1); ax2.set_ylim(-.12, .96)
    fig.text(.73, .88, "(б) Срез coherence = 0,1 (через текущую точку)", fontsize=20, fontweight="bold", ha="center")
    cb2 = fig.colorbar(cf, cax=fig.add_axes([.94, .2, .015, .55]))
    cb2.set_label("Kendall τ", fontsize=16); cb2.ax.tick_params(labelsize=14)
    ax2.plot([], [], color=INK, ls="--", lw=1.6, label="граница зоны неотличимости")
    ax2.legend(loc="lower center", bbox_to_anchor=(.5, -.02), fontsize=15)

    share = float(zone.mean())
    if zone[ci]:
        take = (f"Текущие веса: τ = {tau_cur:.3f}, лучшая точка сетки τ* = {tau[b]:.3f}; разница в пределах 2 SE, "
                f"текущая точка в неотличимой зоне ({share:.0%} сетки) → менять веса незачем.")
    else:
        take = (f"Текущие веса: τ = {tau_cur:.3f}, лучшая точка сетки τ* = {tau[b]:.3f}; текущая точка вне зоны "
                f"неотличимости ({share:.0%} сетки): на этих 68 вопросах (in-sample) есть точка лучше.")
    fig.suptitle("Пространство весов награды R = w₁F1 + w₂coverage + w₃diversity + w₄coherence: τ по сетке с шагом 0,05",
                 fontsize=22, fontweight="bold", y=.985)
    fig.text(.5, .015, take, ha="center", fontsize=17, color=INK)
    
    save(fig, "rc_simplex.png")

    res = {"step": 1 / N, "features": list(KEYS), "n_points": len(pts), "n_questions": nq,
           "grid": pts.tolist(), "tau": tau.tolist(), "se": se.tolist(), "se_diff_vs_best": se_d.tolist(),
           "in_zone": zone.tolist(),
           "best": {"w": pts[b].tolist(), "tau": float(tau[b]), "se": float(se[b])},
           "current": {"w": CUR.tolist(), "tau": tau_cur, "se": float(se[ci]), "gap_to_best": float(gap[ci]),
                       "se_diff": float(se_d[ci]), "in_zone": bool(zone[ci])},
           "zone_share": share, "zone_size": int(zone.sum()), "l1_half_dist_best_to_current": dist,
           "note": "in-sample, all informative questions of the 80-question subsample, F1 (beta=1); zone: tau*-tau <= 2*SE(paired diff)"}
    (ROOT / CALIB / "simplex_grid.json").write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
