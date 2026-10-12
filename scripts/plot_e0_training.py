"""E0 (gx-e0): SFT 0..16 loss and SFT -> GRPO val R growth. Val only.   python scripts/plot_e0_training.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p8_figures import AQUA, BLUE, INK, INK2, MUTED, ORANGE, YELLOW, plt, save  # noqa: E402

RS = Path(__file__).resolve().parents[1] / "runs" / "spark"
SFT_END, GRPO_END, GRPO_W = 16, 300, 16.0


def jl(p):
    return [json.loads(x) for x in (RS / p).read_text(encoding="utf-8").splitlines() if x.strip()]


def f(v, d=3, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def main() -> int:
    sv = [r for r in jl("gx-e0/sft/metrics.jsonl") if r["phase"] == "sft_val"]
    ep = [r["step"] for r in sv]
    assert ep == list(range(SFT_END + 1))
    tl, vl, R = [r["train_loss_eval"] for r in sv], [r["val_loss"] for r in sv], [r["val_reward_mean"] for r in sv]
    g = jl("gx-e0/grpo/metrics.jsonl")
    gval = {r["step"]: r["val_reward_mean"] for r in g if r["phase"] == "grpo_val"}
    gtr = {r["step"]: r["reward_mean"] for r in g if r["phase"] == "grpo"}
    best_it = max(gval, key=gval.get)
    ex = float(np.mean([r["provenance"]["meta"]["expert"]["score"] for r in jl("p9-expert/labels.jsonl")
                        if r["split"] == "val"]))
    rnd = json.loads((RS / "p9-accum/accum.json").read_text())["policies"]["random"]["accum"]["all"]["R"][-1]["mean"]
    start, mid, best, last = R[0], R[-1], gval[best_it], gval[GRPO_END]
    imin = min(range(len(vl)), key=vl.__getitem__)

    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 16), gridspec_kw=dict(height_ratios=[1, 1.1], hspace=0.32))
    # ---- left: loss
    a.plot(ep, tl, color=BLUE, lw=3.5, marker="o", ms=6, label="train loss")
    a.plot(ep, vl, color=ORANGE, lw=3.5, marker="o", ms=6, label="val loss")
    a.scatter([imin], [vl[imin]], s=260, color=ORANGE, ec="white", lw=2, zorder=6)
    a.annotate(f"минимум val loss\n{f(vl[imin])} (эпоха {imin})", (imin, vl[imin]), xytext=(imin - 3.2, 2.15), ha="center",
               fontsize=17, fontweight="bold", color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.4))
    a.axvline(SFT_END, color=INK2, ls="--", lw=2.2)
    a.text(SFT_END - 0.3, 0.95, "старт GRPO\n(эпоха 16)", ha="right", va="bottom", fontsize=17, fontweight="bold", color=INK2)
    a.set_xlim(-0.4, SFT_END + 0.6); a.set_ylim(0.8, 3.3)
    a.set_xlabel("эпоха SFT", fontsize=20); a.set_ylabel("loss SFT (marginal)", fontsize=20)
    a.set_title("Loss SFT", fontsize=25, fontweight="bold", pad=12)
    a.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), fontsize=17)
    a.tick_params(labelsize=17)

    # ---- right: R growth
    gx = lambda it: SFT_END + it * GRPO_W / GRPO_END  # noqa: E731
    xmax = SFT_END + GRPO_W
    ymax = 0.78
    b.axvspan(0, SFT_END, color=BLUE, alpha=0.05, lw=0); b.axvspan(SFT_END, xmax, color=ORANGE, alpha=0.06, lw=0)
    b.axvline(SFT_END, color=INK2, ls="--", lw=2.2)
    b.axhline(ex, color=AQUA, ls=":", lw=2.8); b.axhline(rnd, color=YELLOW, ls=":", lw=2.8)
    b.text(0.3, ex + 0.012, f"эксперт (потолок): {fmt0 if False else f(ex)}", color=AQUA, fontsize=19, fontweight="bold", va="bottom")
    b.text(xmax - 0.2, rnd - 0.012, f"случайная политика: {f(rnd)}", color="#a87500", fontsize=19, fontweight="bold",
           va="top", ha="right")
    it_t = np.array(sorted(i for i in gtr if i <= GRPO_END)); y_t = np.array([gtr[i] for i in it_t])
    k = 15
    sm = np.convolve(np.pad(y_t, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    b.plot([gx(i) for i in it_t], sm, color=MUTED, lw=2, label="награда в обучении (train)")
    its = sorted(gval)
    b.plot(ep, R, color=BLUE, lw=4.8, marker="o", ms=6, zorder=3, label="R на val")
    b.plot([SFT_END] + [gx(i) for i in its[1:]], [mid] + [gval[i] for i in its[1:]], color=ORANGE, lw=4.8, marker="o", ms=6, zorder=3)
    b.plot([0], [start], "o", color=BLUE, ms=15, zorder=4)
    b.plot([gx(best_it)], [best], "o", color=ORANGE, ms=17, mec="white", mew=2, zorder=5)
    b.annotate(f"{f(start)}\nдо обучения", (0, start), xytext=(1.2, start - 0.14), fontsize=21, fontweight="bold",
               color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.5))
    b.annotate(f"{f(mid)}\nконец SFT", (SFT_END, mid), xytext=(SFT_END - 5.5, mid - 0.17), ha="center", fontsize=19, fontweight="bold",
               color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.4))
    b.annotate(f"{f(best)}\n(итерация {best_it})", (gx(best_it), best), xytext=(gx(best_it), best + 0.045), ha="center",
               fontsize=20, fontweight="bold", color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
    b.text(SFT_END / 2, ymax - 0.012, "SFT", fontsize=26, fontweight="bold", color=BLUE, ha="center", va="top")
    b.text(SFT_END + GRPO_W / 2, ymax - 0.012, "GRPO", fontsize=26, fontweight="bold", color=ORANGE, ha="center", va="top")
    b.set_ylim(0, ymax); b.set_xlim(0, xmax)
    ts = [0, 4, 8, 12, 16]
    b.set_xticks(ts + [gx(i) for i in (100, 200, 300)])
    b.set_xticklabels([str(t) for t in ts] + ["100", "200", "300"], fontsize=17)
    b.set_xlabel("эпохи SFT (слева)  |  итерации GRPO (справа)", fontsize=19)
    b.set_ylabel("средняя награда R (val)", fontsize=21)
    b.tick_params(axis="y", labelsize=17); b.grid(axis="x", visible=False)
    b.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=16)
    b.set_title("Рост награды: SFT → GRPO", fontsize=25, fontweight="bold", pad=12)

    fig.suptitle("Обучение агента: SFT → GRPO", fontsize=30, fontweight="bold", y=0.95)
    save(fig, "e0_training.png")
    print(start, mid, best, best_it, last, imin, vl[imin])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
