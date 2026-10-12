"""Pilot 9: contributions of SFT and GRPO to val R (3 panels). Val only.

    python scripts/plot_p9_contributions.py   -> docs/m7/figures/p9_contributions.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p8_figures import AQUA, BLUE, INK, INK2, MUTED, ORANGE, plt, save  # noqa: E402
from plot_p9_figures import _fine  # noqa: E402

RS = Path(__file__).resolve().parents[1] / "runs" / "spark"
RED = "#d03b3b"
THETA_IT = 220


def jl(p):
    return [json.loads(x) for x in (RS / p).read_text(encoding="utf-8").splitlines() if x.strip()]


def fmt(v, d=3, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def pq(run):
    return {r["qkey"]: r["reward"] for r in jl(f"{run}/pq.jsonl") if r["split"] == "val"}


def main() -> int:
    lab = {r["id"] + "|q": r for r in jl("p9-expert/labels.jsonl") if r["split"] == "val"}
    ex = float(np.mean([r["provenance"]["meta"]["expert"]["score"] for r in lab.values()]))
    s, g = pq("p9-ev-sft"), pq("p9-ev-g-k5")
    keys = sorted(lab)
    assert set(keys) == set(s) == set(g)
    sa = np.array([s[k] for k in keys]); ga = np.array([g[k] for k in keys])
    grp = np.array([_fine(lab[k]) for k in keys])
    init = next(r["val_reward_mean"] for r in jl("p9-sft-scratch/metrics.jsonl") if r["phase"] == "sft_val" and r["step"] == 0)
    d = ga - sa
    sft_gain, grpo_gain, grpo_se = sa.mean() - init, d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    fin = ga.mean()
    sh_sft = sft_gain / (ex - init)
    sh_grpo = grpo_gain / (ex - sa.mean())

    gval = {r["step"]: r["val_reward_mean"] for r in jl("p9-grpo/metrics.jsonl") if r["phase"] == "grpo_val"}
    its = sorted(gval)
    last = gval[its[-1]]

    fig, axs = plt.subplots(1, 3, figsize=(25, 8.6), gridspec_kw=dict(width_ratios=[1.45, 1.15, 1.05], wspace=0.32))
    # ---- (1) waterfall (full axis from 0)
    ax = axs[0]
    ax.bar(0, init, color=MUTED, width=0.86)
    ax.bar(1, sft_gain, bottom=init, color=BLUE, width=0.86)
    ax.bar(2, grpo_gain, bottom=sa.mean(), color=ORANGE, width=0.86)
    ax.bar(3, fin, color=INK2, width=0.86)
    ax.axhline(ex, color=AQUA, ls=":", lw=3)
    ax.text(-0.45, ex + 0.012, f"эксперт (потолок): {fmt(ex)}", color=AQUA, fontsize=17, fontweight="bold", ha="left", va="bottom")
    ax.text(0, init + 0.01, fmt(init), ha="center", va="bottom", fontsize=20, fontweight="bold")
    ax.text(1, init + sft_gain / 2, f"{fmt(sft_gain, 3, True)}\nзакрывает\n{sh_sft:.0%}\nразрыва", ha="center",
            va="center", fontsize=16, fontweight="bold", color="white")
    ax.annotate(f"{fmt(grpo_gain, 3, True)} ± {fmt(grpo_se)}\n({grpo_gain / grpo_se:.1f} SE)\nзакрывает {sh_grpo:.0%}\nостатка разрыва".replace(".", ","),
                (2, sa.mean() + grpo_gain), xytext=(2.55, 0.73), ha="center", va="bottom", fontsize=17, fontweight="bold",
                color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
    ax.text(3, fin + 0.01, fmt(fin), ha="center", va="bottom", fontsize=20, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(["до\nобучения", "+ SFT", "+ GRPO", "итог"], fontsize=19)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("средняя награда R (val)", fontsize=20)
    ax.tick_params(axis="y", labelsize=17); ax.grid(axis="x", visible=False)
    ax.set_title("Вклад этапов", fontsize=24, fontweight="bold", pad=14)

    # ---- (2) GRPO zoom
    ax = axs[1]
    base = sa.mean()
    xs = np.array(its); ys = np.array([gval[i] for i in its])
    ax.axhline(base, color=BLUE, ls="--", lw=3)
    ax.fill_between(xs, base, ys, where=ys >= base, color=ORANGE, alpha=0.3, interpolate=True)
    ax.plot(xs, ys, color=ORANGE, lw=4.5, marker="o", ms=7)
    ax.plot([THETA_IT], [gval[THETA_IT]], "o", color=ORANGE, ms=20, mec="white", mew=2.5, zorder=5)
    ax.plot([its[-1]], [last], "s", color=INK2, ms=14, mec="white", mew=2, zorder=5)
    ax.text(8, base - 0.0015, f"уровень после SFT: {fmt(base)}", color=BLUE, fontsize=18, fontweight="bold", va="top")
    ax.annotate(f"θ*, итерация {THETA_IT}: {fmt(gval[THETA_IT])}", (THETA_IT, gval[THETA_IT]), xytext=(135, 0.5385),
                fontsize=18, fontweight="bold", color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5))
    ax.annotate(f"конец, итерация {its[-1]}:\n{fmt(last)}", (its[-1], last), xytext=(262, 0.531), fontsize=17,
                fontweight="bold", color=INK2, ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.5))
    ax.set_ylim(0.485, 0.545); ax.set_xlim(0, 305)
    ax.set_xlabel("итерации GRPO", fontsize=20)
    ax.set_ylabel("R (val)", fontsize=20)
    ax.tick_params(labelsize=17); ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.98, "масштаб увеличен", transform=ax.transAxes, ha="left", va="top", fontsize=17,
            fontweight="bold", color=RED, bbox=dict(fc="white", ec=RED, boxstyle="round,pad=0.3"))
    ax.set_title("GRPO крупным планом", fontsize=24, fontweight="bold", pad=14)

    # ---- (3) per group paired deltas
    ax = axs[2]
    names = {"old": "старые", "Q1": "Q1", "Q1x": "Q1x", "Q2": "Q2", "all": "весь val"}
    rows = []
    for k in ("old", "Q1", "Q1x", "Q2", "all"):
        m = np.ones(len(d), bool) if k == "all" else grp == k
        dd = d[m]
        rows.append((k, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else 0.0, int(m.sum())))
    rows.sort(key=lambda r: r[1])
    for i, (k, mu, se, n) in enumerate(rows):
        col = RED if mu < 0 else (INK2 if k == "all" else ORANGE)
        ax.barh(i, mu, xerr=se, color=col, height=0.62, error_kw=dict(lw=2.5, capsize=7, ecolor=INK))
        ax.text(mu + se + 0.004 if mu >= 0 else mu - se - 0.004, i, fmt(mu, 3, True), va="center",
                ha="left" if mu >= 0 else "right", fontsize=18, fontweight="bold", color=col)
    ax.axvline(0, color=INK, lw=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{names[k]}\n(n={n})" for k, _, _, n in rows], fontsize=18)
    lo = min(0, min(r[1] - r[2] for r in rows)); hi = max(r[1] + r[2] for r in rows)
    ax.set_xlim(lo - 0.04, hi + 0.06)
    ax.set_xlabel("изменение R: SFT → GRPO (± парная SE)", fontsize=19)
    ax.tick_params(axis="x", labelsize=17); ax.grid(axis="y", visible=False)
    ax.set_title("Где помогает GRPO", fontsize=24, fontweight="bold", pad=14)

    fig.suptitle("Вклад SFT и GRPO в награду агента (пилот 9, val)", fontsize=30, fontweight="bold", y=1.0)
    fig.text(0.5, 0.0,
             f"SFT даёт основной рост (закрывает ~{sh_sft:.0%} разрыва до эксперта), GRPO — значимую добавку сверх SFT "
             f"({fmt(grpo_gain, 3, True)} ≈ {grpo_gain / grpo_se:.1f} SE; ещё ~{sh_grpo:.0%} оставшегося разрыва).",
             ha="center", va="top", fontsize=20, color=INK)
    fig.text(0.5, -0.045,
             f"θ* выбран по val как лучшая из 16 точек (оценка слегка завышена); на последней итерации R = {fmt(last)}. "
             "Оси панелей 1 и 3 от нуля; панель 2 увеличена намеренно.",
             ha="center", va="top", fontsize=14, color=INK2)
    save(fig, "p9_contributions.png")
    print({k: (round(m, 4), round(s_, 4), n) for k, m, s_, n in rows}, sh_sft, sh_grpo, grpo_gain, grpo_se)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
