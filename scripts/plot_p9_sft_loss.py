"""Pilot 9: SFT (scratch, kept) loss vs val R per epoch. Val only.   python scripts/plot_p9_sft_loss.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p8_figures import BLUE, INK, INK2, MUTED, ORANGE, plt, save  # noqa: E402

RS = Path(__file__).resolve().parents[1] / "runs" / "spark"


def val(run):
    return [r for r in (json.loads(x) for x in (RS / run / "metrics.jsonl").read_text().splitlines() if x.strip())
            if r["phase"] == "sft_val"]


def f(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def main() -> int:
    v, c = val("p9-sft-scratch"), val("p9-sft-init")
    ep = [r["step"] for r in v]
    best = max(v, key=lambda r: r["val_reward_mean"])["step"]
    stop = ep[-1]
    tl, vl = [r["train_loss_eval"] for r in v], [r["val_loss"] for r in v]
    R = [r["val_reward_mean"] for r in v]
    imin = min(range(len(vl)), key=vl.__getitem__)
    gap = tl[best] - vl[best]

    fig, (a, b) = plt.subplots(1, 2, figsize=(21, 8), gridspec_kw=dict(wspace=0.22))
    a.plot(ep, tl, color=BLUE, lw=3.5, marker="o", ms=5, label="train loss (режим eval)")
    a.plot(ep, vl, color=ORANGE, lw=3.5, marker="o", ms=5, label="val loss")
    a.axvline(best, color=INK2, ls="--", lw=2)
    a.text(best + 0.3, 2.95, f"выбор:\nэпоха {best}\n(по R)", ha="left", va="top", fontsize=17, fontweight="bold", color=INK2)
    a.scatter([stop], [vl[-1]], marker="x", s=220, color=INK, lw=4, zorder=6)
    a.annotate(f"ранняя остановка\n(эпоха {stop})", (stop, vl[-1]), xytext=(stop - 2, 2.2), ha="center", fontsize=16,
               color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=1.3))
    a.annotate(f"минимум val loss\n{f(vl[imin])} (эпоха {imin})", (imin, vl[imin]), xytext=(imin - 3, 1.95), ha="center",
               fontsize=16, color=ORANGE, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.3))
    a.set_xlim(-0.5, stop + 0.8); a.set_ylim(0.8, 3.3)
    a.set_xlabel("эпоха SFT", fontsize=20); a.set_ylabel("loss SFT (marginal)", fontsize=20)
    a.set_title("Loss: val выходит на плато", fontsize=24, fontweight="bold", pad=12)
    a.legend(loc="upper left", bbox_to_anchor=(0.1, 1.0), fontsize=14)
    a.tick_params(labelsize=17)

    b.plot(ep, R, color=BLUE, lw=4, marker="o", ms=6)
    b.axvline(best, color=INK2, ls="--", lw=2)
    b.scatter([0], [R[0]], s=200, color=BLUE, zorder=5); b.scatter([best], [R[best]], s=260, color=ORANGE, zorder=6, ec="white", lw=2)
    b.annotate(f"{f(R[0])}\nэпоха 0", (0, R[0]), xytext=(3, 0.12), fontsize=20, fontweight="bold", color=BLUE,
               arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.3))
    b.annotate(f"{f(R[best])}\nэпоха {best}", (best, R[best]), xytext=(best - 5, 0.3), ha="center", fontsize=20,
               fontweight="bold", color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.3))
    b.set_xlim(-0.5, stop + 0.8); b.set_ylim(0, 0.6)
    b.set_xlabel("эпоха SFT", fontsize=20); b.set_ylabel("средняя награда R (val)", fontsize=20)
    b.set_title("Качество на val (награда R)", fontsize=24, fontweight="bold", pad=12)
    b.tick_params(labelsize=17)

    fig.suptitle("SFT пилота 9 (с нуля): loss и качество агента", fontsize=28, fontweight="bold", y=0.99)
    fig.text(0.5, 0.0, f"train loss падает, val loss выходит на плато около {f(vl[imin], 2)} (минимум на эпохе {imin}), "
             f"а R на val растёт до эпохи {best}: поэтому эпоху выбираем по R, а не по loss.",
             ha="center", va="top", fontsize=19, color=INK)
    fig.text(0.5, -0.05, f"На выбранной эпохе val loss − train loss = {f(-gap, 2)}. R между соседними эпохами шумит (±0,01–0,02), "
             "выбор эпохи 26 по максимуму R — слегка оптимистичен.", ha="center", va="top", fontsize=14, color=INK2)
    save(fig, "p9_sft_loss.png")
    print(vl[imin], imin, tl[best], vl[best], gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
