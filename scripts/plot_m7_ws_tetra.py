"""M7 reward-weight sweep: the Scheffé mixture model in the full weight tetrahedron (F1, coverage, diversity,
coherence), with the trained points, the current weights, the model optimum and its bootstrap cloud. Validation only.

    PYTHONPATH=. python scripts/plot_m7_ws_tetra.py   (after plot_m7_ws_surface.py)
        -> docs/m7/figures/ws_tetra.png
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402

from scripts.plot_m7_calib_figures import INK, INK2, SURFACE, plt, save  # noqa: E402
from scripts.plot_m7_ws_surface import design  # noqa: E402

RED, GOLD = "#d62728", "#ffb000"
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)        # F1, coverage, diversity, coherence
LAB = ("F1", "coverage", "diversity", "coherence")
NAME = {"cur": "текущие", "eq": "равные", "f1": "F1", "cov": "cov", "div": "div", "coh": "coh", "opt": "оптимум"}
HALO = [pe.withStroke(linewidth=3, foreground="white")]


FULL = "--full" in sys.argv          # full simplex 0..1 (the region w_i >= 0.1 drawn inside it)


def xyz(W):
    """Zoomed: the region w_i >= 0.1 mapped onto the whole tetrahedron (u = (w - 0.1) / 0.6); --full: w as is."""
    W = np.atleast_2d(W)
    return (W if FULL else (W - 0.1) / 0.6) @ V


KEY = {"cur", "f4c4", "opt", "eq"}
CORNER = ("f1", "cov", "div", "coh")


def panel(ax, S, grid, mu, opt, elev, azim, title, lo, hi):
    ax.set_facecolor(SURFACE)
    ax.computed_zorder = False
    for i, j in itertools.combinations(range(4), 2):
        ax.plot(*zip(V[i], V[j]), color=INK2, lw=1.0, zorder=1)
    if FULL:
        inner = xyz(np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]]))
        for i, j in itertools.combinations(range(4), 2):
            ax.plot(*zip(inner[i], inner[j]), color=RED, lw=1.0, ls="--", alpha=.6, zorder=1)
    P = xyz(grid)
    t1, t2 = np.quantile(mu, .80), np.quantile(mu, .93)
    mid, top = (mu >= t1) & (mu < t2), mu >= t2
    ax.scatter(*P[mid].T, c=mu[mid], cmap="viridis", vmin=lo, vmax=hi, s=6, alpha=.18, lw=0, depthshade=False,
               zorder=2)
    sc = ax.scatter(*P[top].T, c=mu[top], cmap="viridis", vmin=lo, vmax=hi, s=16, alpha=.75, lw=0,
                    depthshade=False, zorder=3)
    for p, r in S.items():
        q = xyz(np.array(r["w"]))[0]
        if p == "cur":
            ax.scatter(*q, marker="*", s=560, color=RED, edgecolors=INK, linewidths=1.2, depthshade=False, zorder=9)
        else:
            ax.scatter(*q, s=80, c=[r["all5"]], cmap="viridis", vmin=lo, vmax=hi, edgecolors=INK, linewidths=1,
                       depthshade=False, zorder=8)
        if p in KEY or (FULL and p in CORNER):
            ax.text(*(q + np.array([.07, .07, .07])), f"{NAME.get(p, p)} {r['all5']:.3f}".replace(".", ","),
                    fontsize=10, color=RED if p == "cur" else INK, fontweight="bold" if p in ("cur", "opt") else None,
                    path_effects=HALO, zorder=10)
    o = xyz(opt)[0]
    ax.scatter(*o, marker="X", s=300, color=GOLD, edgecolors=INK, linewidths=1.3, depthshade=False, zorder=9)
    if "opt" not in S:
        ax.text(*(o + np.array([.07, -.12, -.12])), "оптимум модели (прогноз)", fontsize=10, fontweight="bold",
                color=INK, path_effects=HALO, zorder=10)
    for v, lab, c in zip(V, LAB, CORNER):
        val = f"\n(обучено: {S[c]['all5']:.3f})".replace(".", ",") if c in S and not FULL else ""
        ax.text(*(v * 1.16), f"{lab} = 1" if FULL else f"{lab} 0,7{val}", fontsize=12, fontweight="bold", ha="center", va="center", zorder=10,
                path_effects=HALO)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        f(-1.1, 1.1)
    ax.set_title(title, fontsize=12, pad=0)
    return sc


def main() -> int:
    S = json.loads((ROOT / "data/m7/calib/sweep.json").read_text(encoding="utf-8"))
    R = json.loads((ROOT / "data/m7/calib/ws_surface.json").read_text(encoding="utf-8"))
    beta = np.array(R["beta"])
    grid = np.array([c for c in itertools.product(range(41), repeat=4) if sum(c) == 40], float) / 40
    grid = grid[(grid >= 0.1 - 1e-9).all(1)]
    mu = design(grid) @ beta
    opt = np.array(R["optimum"]["w"])
    lo, hi = 0.60, float(mu.max())
    fig = plt.figure(figsize=(18, 8.8))
    a1 = fig.add_axes([0.0, 0.05, 0.47, 0.83], projection="3d")
    a2 = fig.add_axes([0.45, 0.05, 0.47, 0.83], projection="3d")
    sc = panel(a1, S, grid, mu, opt, 16, 60, "вид 1", lo, hi)
    panel(a2, S, grid, mu, opt, 24, 150, "вид 2 (повёрнут)", lo, hi)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.93, 0.2, 0.012, 0.55]))
    cb.set_label("val all@5 (облако — модель Шеффе; кружки — измерения)", fontsize=11)
    fig.suptitle(("Полное пространство весов (0…1); красный пунктир — допустимая область wᵢ ≥ 0,1, облако — модель Шеффе в ней"
                  if FULL else "Пространство весов награды (область wᵢ ≥ 0,1): яркое облако — лучшие 7 % весов по модели Шеффе"),
                 fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .015, ("вершины — вес 1 у одной компоненты (остальные 0); вне пунктира модель не обучалась;  ★ текущие;  "
                        if FULL else "вершины — веса 0,7 у одной компоненты и 0,1 у остальных;  ★ текущие 0,5/0,2/0,2/0,1;  ")
             + 
             f"✕ оптимум модели {'/'.join(f'{x:g}' for x in opt)};  кружки — 12 обученных наборов (27 агентов)".replace(".", ","),
             ha="center", fontsize=11, color=INK2)
    save(fig, "ws_tetra_full.png" if FULL else "ws_tetra.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
