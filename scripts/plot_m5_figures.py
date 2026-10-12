"""M5 summary figures (PNG) from pulled Spark runs (runs/spark/*). Light: reads JSONL/JSON only.

    PYTHONPATH=. python scripts/plot_m5_figures.py   ->  docs/m5/figures/*.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "spark"
OUT = ROOT / "docs" / "m5" / "figures"

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


def rows(run: str) -> list[dict]:
    p = RUNS / run / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_methods() -> None:
    """Pilot 5: val vs strict consensus test, mean ± SE (report numbers)."""
    m = [  # name, val, val_se, test, test_se, trained
        ("Правило BM25+dense\n(порог 0,9)", .6731, .019, .6355, .021, False),
        ("Правило кросс-энкодер\n+ поиск", .6831, .020, .6606, .020, False),
        ("Адаптивный агент\nGRPO (без тревог)", .618, .020, .589, .021, True),
        ("RL-реранкер SFT+CE\n(3 сида)", .7005, .003, .6655, .004, True),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = range(len(m))
    h = .36
    ax.barh([i + h / 2 for i in y], [r[1] for r in m], h, xerr=[r[2] for r in m], color=BLUE,
            error_kw={"ecolor": INK2, "lw": 1}, label="val (182)", edgecolor=SURFACE, linewidth=2)
    ax.barh([i - h / 2 for i in y], [r[3] for r in m], h, xerr=[r[4] for r in m], color=ORANGE,
            error_kw={"ecolor": INK2, "lw": 1}, label="тест, строгий (136)", edgecolor=SURFACE, linewidth=2)
    for i, r in enumerate(m):
        ax.text(r[1] + r[2] + .004, i + h / 2, f"{r[1]:.3f}", va="center", color=INK2, fontsize=9)
        ax.text(r[3] + r[4] + .004, i - h / 2, f"{r[3]:.3f}", va="center", color=INK2, fontsize=9)
    ax.axvline(.773, color=MUTED, ls="--", lw=1)
    ax.text(.773, len(m) - .45, " потолок top-20 (тест) 0,773", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(list(y), [r[0] for r in m])
    ax.set_xlim(.5, .8)
    ax.set_xlabel("награда R (выше — лучше), ± стандартная ошибка")
    ax.set_title("Пилот 5: методы на val и строгом тесте")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "01_methods_val_test.png")


def fig_progress() -> None:
    """Best val reward per pilot (different val sets: P1 n=41, P2–P4 n=63, P5 n=126 trajectories / 182)."""
    pts = [("П1\nSFT фикс. k", .485, False), ("П2\nSFT +рег.", .520, False), ("П2b\nадаптив. RL\n(стабильный)", .629, False),
           ("П3\nSFT нов. эксперт", .671, False), ("П4\nRL-реранкер\nSFT+CE", .6865, False), ("П5\nRL-реранкер\n(все пары)", .7005, False)]
    alarmed = [(2, .663, "адаптив. RL\nс тревогами KL")]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = list(range(len(pts)))
    ax.plot(x, [p[1] for p in pts], color=BLUE, lw=2, marker="o", ms=8, label="лучшая обученная модель (val)")
    for i, p in enumerate(pts):
        ax.annotate(f"{p[1]:.3f}", (i, p[1]), textcoords="offset points", xytext=(0, 9), ha="center", color=INK2, fontsize=9)
    ax.scatter([a[0] for a in alarmed], [a[1] for a in alarmed], s=64, facecolors=SURFACE, edgecolors=ORANGE, linewidths=2,
               zorder=3, label="не принят: сигналы тревоги")
    ax.annotate(alarmed[0][2], (alarmed[0][0], alarmed[0][1]), textcoords="offset points", xytext=(34, -10), color=INK2, fontsize=8)
    ax.axhline(.479, color=MUTED, ls="--", lw=1)
    ax.text(len(pts) - .5, .479, "dense@3 (П2–П4) 0,479", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.axhline(.671, color=AQUA, ls=":", lw=1.5)
    ax.text(-.3, .671, "правило BM25+dense 0,671", color=INK2, fontsize=9, va="bottom")
    ax.set_xticks(x, [p[0] for p in pts], fontsize=8.5)
    ax.set_ylim(.44, .74)
    ax.set_ylabel("награда R на val")
    ax.set_title("Прогресс по пилотам (val; наборы val различаются по пилотам)")
    ax.legend(loc="center right", bbox_to_anchor=(1, .33))
    save(fig, "02_progress_by_pilot.png")


def fig_sft(run: str = "p5-rr-sft") -> None:
    r = rows(run)
    if not r:
        return
    tr = defaultdict(list)
    for x in r:
        if x.get("phase") == "sft":
            tr[x["epoch"]].append(x["loss"])
    val = [x for x in r if x.get("phase") == "sft_val"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    ep = sorted(tr)
    a1.plot(ep, [sum(tr[e]) / len(tr[e]) for e in ep], color=BLUE, lw=2, label="train loss")
    a1.plot([v["step"] for v in val], [v["val_loss"] for v in val], color=ORANGE, lw=2, label="val loss")
    a1.set_ylabel("loss (кросс-энтропия)")
    a1.set_title("SFT RL-реранкера (пилот 5, с кросс-энкодером): обучение")
    a1.legend()
    a2.plot([v["step"] for v in val], [v["val_reward_mean"] for v in val], color=BLUE, lw=2, marker="o", ms=4, label="награда val")
    a2.plot([v["step"] for v in val], [v["val_recall_mean"] for v in val], color=AQUA, lw=2, label="полнота val")
    best = max(val, key=lambda v: v["val_reward_mean"])
    a2.scatter([best["step"]], [best["val_reward_mean"]], s=80, color=BLUE, zorder=3)
    a2.annotate(f"лучшая эпоха {best['step']}: {best['val_reward_mean']:.3f}", (best["step"], best["val_reward_mean"]),
                textcoords="offset points", xytext=(8, -14), color=INK2, fontsize=9)
    a2.set_xlabel("эпоха")
    a2.set_ylabel("значение на val")
    a2.legend(loc="lower right")
    save(fig, "03_sft_reranker_curves.png")


def fig_grpo(run: str = "p5-rr-grpo", title: str = "GRPO поверх SFT-реранкера (пилот 5)") -> None:
    r = rows(run)
    g = [x for x in r if x.get("phase") == "grpo"]
    v = [x for x in r if x.get("phase") == "grpo_val"]
    if not g:
        return
    it = [x["step"] for x in g]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    panels = [
        (axs[0][0], "reward_mean", "награда (train, батч)", BLUE),
        (axs[0][1], "kl_ref", "KL к π_ref", ORANGE),
        (axs[1][0], "entropy", "энтропия политики", AQUA),
        (axs[1][1], "slot_top_share", "доля «первый кандидат»", YELLOW),
    ]
    for ax, key, lab, col in panels:
        ys = [x.get(key) for x in g]
        if all(y is None for y in ys):
            ax.set_visible(False)
            continue
        ax.plot(it, ys, color=col, lw=1.5)
        ax.set_title(lab, fontsize=10)
    if v:
        axs[0][0].plot([x["step"] for x in v], [x["val_reward_mean"] for x in v], color=INK, lw=2, marker="o", ms=4, label="val")
        axs[0][0].legend(loc="lower right")
    if any(x.get("collapse_threshold") for x in g):
        axs[1][1].plot(it, [x.get("collapse_threshold") for x in g], color=MUTED, ls="--", lw=1, label="порог коллапса")
        axs[1][1].legend(loc="lower right")
    axs[0][1].axhline(.5, color=MUTED, ls="--", lw=1)
    for ax in axs[1]:
        ax.set_xlabel("итерация")
    fig.suptitle(title, fontweight="bold")
    save(fig, "04_grpo_reranker_dynamics.png")


def fig_recall_k() -> None:
    p = RUNS / "p2-curves-val" / "out.json"
    if not p.exists():
        return
    c = json.loads(p.read_text())["curves"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for name, col, lab in (("dense", BLUE, "dense"), ("rrf", ORANGE, "RRF (BM25+dense)"), ("bm25", AQUA, "BM25")):
        k = [x["k"] for x in c[name]]
        a1.plot(k, [x["recall_mean"] for x in c[name]], color=col, lw=2, label=lab)
        a2.plot(k, [x["reward_mean"] for x in c[name]], color=col, lw=2, label=lab)
    a1.set_title("полнота @k: сколько правильных фрагментов найдено", fontsize=10)
    a2.set_title("награда @k: лишние фрагменты штрафуются", fontsize=10)
    a1.set_ylabel("полнота")
    a2.set_ylabel("награда R")
    for a in (a1, a2):
        a.set_xlabel("k (фрагментов в контексте)")
    a1.legend(loc="lower right")
    fig.suptitle("Фиксированный top-k на val (пилот 2, 63 вопроса, |C*| = 1)", fontweight="bold")
    save(fig, "05_recall_reward_vs_k.png")


def fig_reward_terms(run: str = "p5-rr-grpo") -> None:
    g = [x for x in rows(run) if x.get("phase") == "grpo"]
    if not g:
        return
    it = [x["step"] for x in g]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for key, col, lab in (("r_chunks", BLUE, "R_chunks (F1 по фрагментам), вес 0,5"), ("r_coverage", ORANGE, "R_coverage, вес 0,2"),
                          ("r_diversity", AQUA, "R_diversity, вес 0,2"), ("r_coherence", YELLOW, "R_coherence, вес 0,1")):
        ax.plot(it, [x.get(key) for x in g], color=col, lw=1.6, label=lab)
    ax.set_xlabel("итерация GRPO")
    ax.set_ylabel("значение слагаемого (до весов)")
    ax.set_title("Слагаемые награды по ходу GRPO (пилот 5, реранкер)")
    ax.legend(loc="center right", fontsize=8.5)
    save(fig, "06_reward_terms.png")


if __name__ == "__main__":
    fig_methods()
    fig_progress()
    fig_sft()
    fig_grpo()
    fig_recall_k()
    fig_reward_terms()
