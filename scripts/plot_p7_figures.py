"""M7 pilot 7 figures (PNG) from pulled Spark runs (runs/spark/<run>/metrics.jsonl) -- our struct agent only (no
method comparison), validation only. Style / palette of scripts/plot_m5_figures.py.

    PYTHONPATH=. python scripts/plot_p7_figures.py --sft p7-sft-a --grpo p7-grpo [--best-epoch N]
        [--report runs/spark/p7-report.json] [--report-grpo runs/spark/p7-report-grpo.json]
-> docs/m7/figures/p7_sft_loss.png, p7_sft_val.png, p7_grpo_reward.png, p7_reward_terms.png, p7_action_mix.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "spark"
OUT = ROOT / "docs" / "m7" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
})
LW = 1.4
GROUPS = (("count", "Q1 (подсчёт)", BLUE), ("version", "Q2 (версия)", ORANGE), ("old", "старый val", AQUA))
FAMILIES = (("structure", "structure", BLUE), ("graph", "граф", ORANGE), ("lexical", "лексический", AQUA),
            ("vector", "векторный", YELLOW))


def rows(run: str) -> list[dict]:
    p = RUNS / run / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def sft_series(run: str):
    rs = rows(run)
    tr = defaultdict(list)
    for r in rs:
        if r["phase"] == "sft":
            tr[int(r["epoch"])].append(float(r["loss"]))
    val = sorted((r for r in rs if r["phase"] == "sft_val"), key=lambda r: r["step"])
    fin = [r for r in rs if r["phase"] == "sft_final"]
    best = int(fin[-1]["best"].split("-")[1].split(".")[0]) if fin else None
    return tr, val, best


def fig_sft_loss(run: str) -> None:
    tr, val, best = sft_series(run)
    ep = sorted(tr)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ep, [np.mean(tr[e]) for e in ep], color=BLUE, lw=LW, label="train loss (среднее за эпоху)")
    ax.plot([v["step"] for v in val], [v["val_loss"] for v in val], color=ORANGE, lw=LW, label="val loss")
    if best:
        ax.axvline(best, color=MUTED, ls="--", lw=1, label=f"выбранная эпоха ({best}, по val R)")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss SFT (marginal)")
    ax.set_title("SFT struct-агента: функция потерь")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    save(fig, "p7_sft_loss.png")


def fig_sft_val(run: str) -> None:
    _, val, best = sft_series(run)
    ep = [v["step"] for v in val]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for g, lab, col in GROUPS[:2]:
        a1.plot(ep, [v["val_by_group"][g]["all_share"] for v in val], color=col, lw=LW, label=lab)
    a1.set_title("val all@5, вопросы M7")
    a1.set_ylabel("all@5")
    for g, lab, col in GROUPS:
        a2.plot(ep, [v["val_by_group"][g]["reward_mean"] for v in val], color=col, lw=LW, label=lab)
    a2.plot(ep, [v["val_reward_mean"] for v in val], color=INK2, lw=LW, ls=":", label="весь val (критерий выбора)")
    a2.set_title("val R по группам")
    a2.set_ylabel("награда R")
    for ax in (a1, a2):
        ax.set_xlabel("эпоха")
        if best:
            ax.axvline(best, color=MUTED, ls="--", lw=1, label=f"выбранная эпоха {best}")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    fig.suptitle("SFT struct-агента: метрики на val по эпохам", fontweight="bold")
    fig.tight_layout()
    save(fig, "p7_sft_val.png")


def _smooth(y, k: int = 10):
    y = np.asarray(y, float)
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - k + 1):i + 1].mean()
    return out


def grpo_series(run: str):
    rs = rows(run)
    it = sorted((r for r in rs if r["phase"] == "grpo"), key=lambda r: r["step"])
    val = sorted((r for r in rs if r["phase"] == "grpo_val"), key=lambda r: r["step"])
    return it, val


def fig_grpo_reward(run: str) -> None:
    it, val = grpo_series(run)
    x = [r["step"] for r in it]
    y = [r["reward_mean"] for r in it]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    a1.plot(x, y, color=BLUE, lw=0.7, alpha=0.35, label="награда батча (train)")
    a1.plot(x, _smooth(y), color=BLUE, lw=LW, label="скользящее среднее, 10 итераций")
    vx = [v["step"] for v in val]
    a1.plot(vx, [v["val_reward_mean"] for v in val], "o-", color=ORANGE, lw=1, ms=4, label="val R (весь val)")
    for g, lab, col, mk in (("count", "val all@5 Q1", AQUA, "s"), ("version", "val all@5 Q2", YELLOW, "^")):
        a1.plot(vx, [v["val_by_group"][g]["all_share"] for v in val], mk + "--", color=col, lw=0.9, ms=4, label=lab)
    a1.set_ylabel("R / all@5 (доля)")
    a1.set_ylim(0, 1)
    a1.set_title("GRPO struct-агента: награда по итерациям")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8.5)
    a2.plot(x, [r.get("kl_ref", 0.0) for r in it], color=MUTED, lw=LW, label="KL(π‖π_ref)")
    a2.set_ylabel("KL")
    a2.set_xlabel("итерация GRPO")
    a2.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "p7_grpo_reward.png")


def fig_reward_terms(run: str) -> None:
    it, _ = grpo_series(run)
    x = [r["step"] for r in it]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, lab, col in (("r_chunks", "R_chunks (F1 фрагментов, вес 0,5)", BLUE),
                        ("r_coverage", "coverage (вес 0,2)", ORANGE), ("r_diversity", "diversity (вес 0,2)", AQUA),
                        ("r_coherence", "coherence (вес 0,1)", YELLOW)):
        ax.plot(x, _smooth([r[k] for r in it]), color=col, lw=LW, label=lab)
    ax.set_xlabel("итерация GRPO")
    ax.set_ylabel("значение компоненты (скользящее среднее, 10)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Компоненты награды в батчах GRPO")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    save(fig, "p7_reward_terms.png")


def fig_action_mix(sft_val: dict, grpo_val: dict) -> None:
    """sft_val / grpo_val: {group: {family: share}} on val (the selected SFT checkpoint vs the GRPO checkpoint)."""
    cols = [("SFT", "count", "Q1"), ("GRPO", "count", "Q1"), ("SFT", "version", "Q2"), ("GRPO", "version", "Q2"),
            ("SFT", "old", "стар."), ("GRPO", "old", "стар.")]
    src = {"SFT": sft_val, "GRPO": grpo_val}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = [0, 1, 2.6, 3.6, 5.2, 6.2]
    for x, (who, g, lab) in zip(xs, cols):
        sh = src[who].get(g, {})
        tot = sum(sh.get(f, 0.0) for f, _, _ in FAMILIES) or 1.0
        bottom = 0.0
        for f, fl, col in FAMILIES:
            v = sh.get(f, 0.0) / tot
            ax.bar(x, v, 0.8, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=1,
                   label=fl if x == 0 else None)
            if v >= 0.07:
                ax.text(x, bottom + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color=SURFACE)
            bottom += v
    ax.set_xticks(xs, [f"{who}\n{lab}" for who, _, lab in cols])
    ax.set_ylabel("доля шагов поиска (без STOP)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Какие действия выбирает агент на val: SFT и после GRPO")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "p7_action_mix.png")


def last_val_groups(run: str, phase: str, step: int | None = None) -> dict:
    vs = [r for r in rows(run) if r["phase"] == phase and (step is None or r["step"] == step)]
    v = vs[-1]
    return {g: d["action_share"] for g, d in v["val_by_group"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--grpo")
    ap.add_argument("--grpo-step", type=int, help="the GRPO validation whose action mix is shown (default: best)")
    a = ap.parse_args(argv)
    fig_sft_loss(a.sft)
    fig_sft_val(a.sft)
    if a.grpo:
        fig_grpo_reward(a.grpo)
        fig_reward_terms(a.grpo)
        _, val, best = sft_series(a.sft)
        sft_mix = {g: d["action_share"] for g, d in next(v for v in val if v["step"] == best)["val_by_group"].items()}
        fig_action_mix(sft_mix, last_val_groups(a.grpo, "grpo_val", a.grpo_step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
