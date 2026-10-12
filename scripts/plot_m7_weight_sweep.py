"""M7 reward calibration: tau when ONE weight is swept 0..1 (others keep current proportions). In-sample, 68 questions.

    PYTHONPATH=. python scripts/plot_m7_weight_sweep.py -> docs/m7/figures/rc_weights_sweep.png
Needs data/m7/calib/simplex_grid.json (scripts/plot_m7_weight_simplex.py) for tau* and the zone threshold."""
from __future__ import annotations

import json
import sys

import numpy as np

from scripts.plot_m7_weight_simplex import CALIB, CUR, INK, INK2, GRID, KEYS, ROOT, _groups, cr, load, plt, save  # noqa: E402

COL = {"f1": "#2a78d6", "coverage": "#eb6834", "diversity": "#1baf7a", "coherence": "#8a4fd0"}
LAB = {"f1": "F1", "coverage": "coverage", "diversity": "diversity", "coherence": "coherence"}


def main() -> int:
    sub, names, util, _, _ = load()
    idx = [cr.feature_names(1.0).index(k) for k in KEYS]
    G = [(F[:, idx], U) for _, F, U, _ in _groups(sub, names, util, 1.0)]
    sg = json.loads((ROOT / CALIB / "simplex_grid.json").read_text(encoding="utf-8"))
    tau_star = sg["best"]["tau"]
    zt = min(t for t, z in zip(sg["tau"], sg["in_zone"]) if z)       # lowest tau in the paired-SE zone = tau* - 2*SE(diff)
    xs = np.round(np.arange(0, 1.0001, .02), 2)
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.axhspan(zt, tau_star, color="#9a9a96", alpha=.22, lw=0, zorder=0)
    ax.axhline(tau_star, color=INK2, ls="--", lw=1.4, zorder=1)
    ax.text(.5, tau_star + .004, f"лучшая точка сетки: τ* = {tau_star:.3f}", ha="center", va="bottom", fontsize=15, color=INK2)
    ax.annotate("серая зона: статистически не хуже\nлучшего (τ* − 2·SE)", (.33, zt), xytext=(.33, .17), ha="center", fontsize=15, color=INK2,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 1.2})
    cur_tau = None
    ends = []
    for j, k in enumerate(KEYS):
        rest = CUR.copy(); rest[j] = 0; rest /= rest.sum()
        m, se = [], []
        for x in xs:
            w = rest * (1 - x); w[j] = x
            t, s, _ = cr.mean_tau(G, w)
            m.append(t); se.append(s)
        m, se = np.array(m), np.array(se)
        ax.plot(xs, m, color=COL[k], lw=3, label=LAB[k], zorder=3)
        if k == "f1":
            ax.fill_between(xs, m - se, m + se, color=COL[k], alpha=.12, lw=0, zorder=2)
        c = int(np.argmin(abs(xs - CUR[j])))
        cur_tau = m[c]
        ax.plot([CUR[j]], [m[c]], "o", ms=13, color=COL[k], mec=INK, mew=1.5, zorder=5)
        ends.append((m[-1], k))
        print(k, "peak x=%.2f tau=%.3f; tau(0)=%.3f tau(1)=%.3f; at current=%.3f" % (xs[m.argmax()], m.max(), m[0], m[-1], m[c]))
    # direct labels at the right end, de-overlapped
    ends.sort()
    ys = []
    for y, k in ends:
        y = max(y, ys[-1] + .022) if ys else y
        ys.append(y)
    for (y0, k), y in zip(ends, ys):
        ax.text(1.015, y, LAB[k], color=COL[k], fontsize=17, fontweight="bold", va="center")
    ax.plot([], [], "o", ms=13, color="white", mec=INK, mew=1.5, label="текущие веса (τ = %.3f, у всех линий)" % cur_tau)
    ax.set_xlim(0, 1.0); ax.set_xlim(-.01, 1.14)
    ax.set_xticks(np.arange(0, 1.01, .1))
    ax.tick_params(labelsize=14)
    ax.set_xlabel("вес компоненты (остальные — в прежней пропорции)", fontsize=17)
    ax.set_ylabel("Kendall τ: согласие награды с реальной полезностью", fontsize=17)
    ax.set_title("Как меняется качество награды, если сдвигать один вес", fontsize=22, fontweight="bold")
    ax.legend(loc="lower left", fontsize=15, title="сдвигаемый вес", title_fontsize=15, framealpha=.9, frameon=True, facecolor="#fcfcfb", edgecolor=GRID)
    fig.text(.5, .01, "Текущие веса в серой зоне; около них кривые плоские → выбор устойчив. τ растёт лишь с весом F1 (до 0,455 при w=1); большой вес diversity вредит.", ha="center", fontsize=14, color=INK)
    fig.subplots_adjust(bottom=.14)
    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
    save(fig, "rc_weights_sweep.png")
