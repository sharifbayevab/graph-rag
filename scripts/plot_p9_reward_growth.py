"""Pilot 9: val reward growth over the whole training (SFT epochs 0..best, then GRPO iterations). Val only.

    python scripts/plot_p9_reward_growth.py   -> docs/m7/figures/p9_reward_growth.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p8_figures import AQUA, BLUE, INK, INK2, MUTED, ORANGE, plt, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "runs" / "spark"
BEST_EP, THETA_IT, GRPO_END, GRPO_W = 26, 220, 300, 26.0


def jl(p):
    return [json.loads(x) for x in (RS / p).read_text(encoding="utf-8").splitlines() if x.strip()]


def fmt(v, d=3, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def main() -> int:
    sft = {r["step"]: r["val_reward_mean"] for r in jl("p9-sft-scratch/metrics.jsonl") if r["phase"] == "sft_val"}
    gval = {r["step"]: r["val_reward_mean"] for r in jl("p9-grpo/metrics.jsonl") if r["phase"] == "grpo_val"}
    ex = float(np.mean([r["provenance"]["meta"]["expert"]["score"] for r in jl("p9-expert/labels.jsonl")
                        if r["split"] == "val"]))
    # paired SE of GRPO theta* vs SFT (same 267 val questions)
    q = lambda run: {r["qkey"]: r["reward"] for r in jl(f"{run}/pq.jsonl") if r["split"] == "val"}  # noqa: E731
    s, g = q("p9-ev-sft"), q("p9-ev-g-k5")
    d = np.array([g[k] - s[k] for k in sorted(s)])
    se = d.std(ddof=1) / np.sqrt(len(d))

    start, mid, end = sft[0], sft[BEST_EP], gval[THETA_IT]
    last = gval[GRPO_END]
    gain_s, gain_g = mid - start, end - mid
    sh_s, sh_g = gain_s / (ex - start), gain_g / (ex - mid)
    gx = lambda it: BEST_EP + it * GRPO_W / GRPO_END  # noqa: E731
    xmax = BEST_EP + GRPO_W

    fig = plt.figure(figsize=(18, 8.4))
    ax = fig.add_axes([0.065, 0.2, 0.60, 0.68])
    ax.axvspan(0, BEST_EP, color=BLUE, alpha=0.05, lw=0)
    ax.axvspan(BEST_EP, xmax, color=ORANGE, alpha=0.06, lw=0)
    ax.axvline(BEST_EP, color=MUTED, lw=2)
    ax.axhline(ex, color=AQUA, ls=":", lw=2.8)
    ax.text(0.4, ex + 0.012, f"эксперт (потолок): {fmt(ex)}", color=AQUA, fontsize=19, fontweight="bold", va="bottom")

    xs_s = list(range(BEST_EP + 1))
    its = sorted(gval)
    ax.plot(xs_s, [sft[e] for e in xs_s], color=BLUE, lw=4.8, marker="o", ms=6, zorder=3, solid_capstyle="round")
    ax.plot([BEST_EP] + [gx(i) for i in its[1:]], [mid] + [gval[i] for i in its[1:]], color=ORANGE, lw=4.8,
            marker="o", ms=6, zorder=3)
    ax.plot([0], [start], "o", color=BLUE, ms=15, zorder=4)
    ax.plot([gx(THETA_IT)], [end], "o", color=ORANGE, ms=17, mec="white", mew=2, zorder=5)

    ax.annotate(f"{fmt(start)}\nдо обучения", (0, start), xytext=(1.5, start - 0.13), fontsize=21, fontweight="bold",
                color=BLUE, ha="left", arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.5))
    ax.annotate(f"{fmt(end)}\nпосле", (gx(THETA_IT), end), xytext=(gx(THETA_IT) - 1, end + 0.065), fontsize=21,
                fontweight="bold", color=ORANGE, ha="center", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
    ymax = 0.78
    ax.text(BEST_EP / 2, ymax - 0.012, "SFT", fontsize=26, fontweight="bold", color=BLUE, ha="center", va="top")
    ax.text(BEST_EP + GRPO_W / 2, ymax - 0.012, "GRPO", fontsize=26, fontweight="bold", color=ORANGE, ha="center",
            va="top")
    ax.set_ylim(0, ymax); ax.set_xlim(0, xmax)
    ticks_s = [0, 5, 10, 15, 20, BEST_EP]
    ax.set_xticks(ticks_s + [gx(i) for i in (100, 200, 300)])
    ax.set_xticklabels([str(t) for t in ticks_s] + ["100", "200", "300"], fontsize=17)
    ax.set_xlabel("эпохи SFT (слева)  |  итерации GRPO (справа)", fontsize=19, labelpad=8)
    ax.set_ylabel("средняя награда R (val)", fontsize=21, labelpad=8)
    ax.tick_params(axis="y", labelsize=17); ax.grid(axis="x", visible=False)

    # braces at the right edge (outside the axes)
    xb = xmax + 0.9
    for lo, hi, col, lw, txt, y in (
        (start, mid, BLUE, 5, f"SFT: {fmt(gain_s, 3, True)}\n({sh_s:.0%} разрыва\nдо эксперта)", (start + mid) / 2),
        (mid, end, ORANGE, 12, f"GRPO: {fmt(gain_g, 3, True)} ± {fmt(se)}\n(≈{gain_g / se:.1f} SE;\n{sh_g:.0%} остатка)", (mid + end) / 2),
    ):
        ax.plot([xb - 0.35, xb, xb, xb - 0.35], [lo, lo, hi, hi], color=col, lw=lw, clip_on=False,
                solid_joinstyle="miter", solid_capstyle="butt")
        ax.text(xb + 0.9, y, txt.replace("3.5", "3,5"), color=col, fontsize=19, fontweight="bold",
                va="center", ha="left", clip_on=False)

    # zoom inset: GRPO only
    ins = ax.inset_axes([0.56, 0.13, 0.40, 0.27])
    base = mid
    xs = np.array(its); ys = np.array([gval[i] for i in its])
    ins.axhline(base, color=BLUE, ls="--", lw=2.5)
    ins.fill_between(xs, base, ys, where=ys >= base, color=ORANGE, alpha=0.3, interpolate=True)
    ins.plot(xs, ys, color=ORANGE, lw=3.5, marker="o", ms=5)
    ins.plot([THETA_IT], [end], "o", color=ORANGE, ms=13, mec="white", mew=2, zorder=5)
    ins.set_ylim(0.485, 0.535); ins.set_xlim(0, 305)
    ins.set_yticks([0.49, 0.50, 0.51, 0.52, 0.53])
    ins.set_yticklabels(["0,49", "0,50", "0,51", "0,52", "0,53"])
    ins.tick_params(labelsize=14); ins.grid(axis="x", visible=False)
    ins.set_xlabel("итерации GRPO", fontsize=14, labelpad=1)
    ins.text(60, base - 0.0025, f"уровень после SFT {fmt(base)}", color=BLUE, fontsize=14, fontweight="bold", va="top")
    ins.text(THETA_IT, end + 0.0035, f"θ* {fmt(end)}", color=ORANGE, fontsize=14, fontweight="bold", ha="center",
             va="bottom")
    ins.set_title("GRPO крупным планом (масштаб увеличен)", fontsize=14, fontweight="bold", color="#b03a00", pad=6)
    ins.set_facecolor("white")
    ax.add_patch(Rectangle((BEST_EP, 0.49), GRPO_W, 0.045, fc="none", ec=INK2, lw=1.6, ls="--", zorder=6))
    for xa, xb_ in ((BEST_EP, 0.0), (xmax, 1.0)):
        fig.add_artist(ConnectionPatch(xyA=(xa, 0.49), coordsA=ax.transData, xyB=(xb_, 1.0), coordsB=ins.transAxes,
                                       color=INK2, lw=1.3, ls="--"))

    fig.suptitle("Награда агента растёт на обоих этапах: SFT и GRPO", fontsize=28, fontweight="bold", y=0.975)
    fig.text(0.5, 0.075,
             f"SFT учит основам ({fmt(start)} → {fmt(mid)}), GRPO добавляет сверх уровня SFT "
             f"(→ {fmt(end)}, {fmt(gain_g, 3, True)} ≈ {str(round(gain_g / se, 1)).replace('.', ',')} SE); сильнее всего на вопросах о версиях (+0,082)",
             ha="center", va="top", fontsize=17, color=INK)
    fig.text(0.5, 0.025, f"θ* выбран по val как лучшая из 16 точек (оценка слегка завышена); на последней итерации R = {fmt(last)}.",
             ha="center", va="top", fontsize=14, color=INK2)
    save(fig, "p9_reward_growth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
