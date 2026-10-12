"""M7 pilot 8 figures (PNG), validation only, our own policies only. Style / palette of scripts/plot_p7_figures.py
(m5 palette, Russian labels, legends, no dual axes).

    PYTHONPATH=. python scripts/plot_p8_figures.py --sft p8-sft-XX --grpo p8-grpo --accum runs/spark/p8-accum/accum.json
        --expert runs/spark/p8-expert/labels.jsonl --router runs/p8-local/router.json [--p7-sft p7-sft-b]
-> docs/m7/figures/p8_learning_curve.png, p8_loss.png, p8_episode_accumulation.png, p8_grpo_q2.png, p8_system.png,
   m7_showcase.png"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
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
LW = 1.6
GROUPS = (("count", "Q1 (подсчёт)", BLUE), ("version", "Q2 (версии)", ORANGE), ("old", "старый (одношаговый)", AQUA))
GNAME = {g: n for g, n, _ in GROUPS}
GCOL = {g: c for g, _, c in GROUPS}


def rows(run: str) -> list[dict]:
    p = RUNS / run / "metrics.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def _se(xs):
    xs = np.asarray(xs, float)
    return float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def sft_val(run: str):
    rs = rows(run)
    val = sorted((r for r in rs if r["phase"] == "sft_val"), key=lambda r: r["step"])
    fin = [r for r in rs if r["phase"] == "sft_final"]
    best = int(fin[-1]["best"].split("-")[1].split(".")[0]) if fin else None
    return val, best


def expert_ceiling(path) -> dict:
    out = defaultdict(lambda: {"R": [], "all": []})
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] != "val":
            continue
        e = r["provenance"]["meta"]["expert"]
        g = r["pattern"] if r["pattern"] in ("count", "version") else "old"
        for k in (g, "all"):
            out[k]["R"].append(float(e["score"]))
            out[k]["all"].append(float(bool(e["reached"])))
    return {g: {"R": float(np.mean(v["R"])), "all": float(np.mean(v["all"])), "R_se": _se(v["R"]),
                "all_se": _se(v["all"]), "n": len(v["R"])} for g, v in out.items()}


def random_ref(accum) -> dict:
    fin = accum["finals"]["random"]
    out = {}
    for g in ("count", "version", "old", "all"):
        v = [x for x in fin.values() if g == "all" or x["group"] == g]
        out[g] = {"R": float(np.mean([x["R"] for x in v])), "all": float(np.mean([x["all"] for x in v]))}
    return out


# --- 1. learning curve ---------------------------------------------------------------------------------------------

def fig_learning(run, exp, rnd) -> None:
    val, best = sft_val(run)
    ep = [v["step"] for v in val]
    stop = ep[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, ek, title in ((axes[0], "all_share", "all", "val all@5 по типам вопросов"),
                               (axes[1], "reward_mean", "R", "val награда R по типам вопросов")):
        for g, name, col in GROUPS:
            ax.plot(ep, [v["val_by_group"][g][key] for v in val], color=col, lw=LW, marker="o", ms=3,
                    label=f"агент: {name}")
            ax.axhline(exp[g][ek], color=col, ls="--", lw=1.0, alpha=0.9)
            ax.axhline(rnd[g][ek], color=col, ls=":", lw=1.0, alpha=0.9)
        if key == "reward_mean":
            ax.plot(ep, [v["val_reward_mean"] for v in val], color=INK, lw=LW, label="агент: весь val")
        ax.axvline(best, color=MUTED, ls="--", lw=1)
        ax.axvspan(best, stop, color=GRID, alpha=0.5, lw=0)
        ax.annotate(f"выбрана эпоха {best}", xy=(best, 0.02), xytext=(-4, 0), textcoords="offset points",
                    ha="right", color=INK2, fontsize=9)
        ax.annotate(f"ранняя остановка\n(эпоха {stop})", xy=(stop, 0.02), xytext=(-3, 0), textcoords="offset points",
                    ha="right", color=INK2, fontsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("эпоха SFT (0 = до обучения, инициализация)")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
    axes[0].set_ylabel("доля вопросов, где найден весь C*")
    axes[1].set_ylabel("R")
    h, l = axes[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=MUTED, ls="--"), Line2D([], [], color=MUTED, ls=":")]
    l += ["потолок эксперта (цвет = тип)", "случайная политика (цвет = тип)"]
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle("Обучение struct-агента (SFT, val): эпоха 0 = инициализация от SFT full пилота 5 до любого шага на M7",
                 fontweight="bold", y=1.02)
    save(fig, "p8_learning_curve.png")


# --- 2. loss -------------------------------------------------------------------------------------------------------

def fig_loss(run, p7) -> None:
    val, best = sft_val(run)
    ep = [v["step"] for v in val]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(ep, [v["train_loss_eval"] for v in val], color=BLUE, lw=LW, marker="o", ms=3,
            label="пилот 8: train loss (режим eval)")
    ax.plot(ep, [v["val_loss"] for v in val], color=ORANGE, lw=LW, marker="o", ms=3, label="пилот 8: val loss")
    if p7:
        v7, b7 = sft_val(p7)
        ax.plot([v["step"] for v in v7], [v["val_loss"] for v in v7], color=MUTED, lw=1.2, ls="--",
                label="пилот 7: val loss (без регуляризации)")
    ax.axvline(best, color=INK2, ls="--", lw=1, label=f"выбранная эпоха {best} (по val R)")
    ax.scatter([ep[-1]], [val[-1]["val_loss"]], marker="x", color=INK, zorder=5, s=50,
               label=f"ранняя остановка (эпоха {ep[-1]}, терпение 5 по val R)")
    ax.set_xlabel("эпоха SFT")
    ax.set_ylabel("loss SFT (marginal)")
    ax.set_title("SFT: train и val loss")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    save(fig, "p8_loss.png")


# --- 3. episode accumulation ---------------------------------------------------------------------------------------

POL = (("grpo", "обученный агент (SFT + GRPO θ*)", BLUE), ("init", "до обучения (эпоха 0, init-from)", MUTED), ("random", "случайная политика",
                                                                                       YELLOW))


def fig_accum(accum) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    for j, (g, name, _) in enumerate(GROUPS):
        for i, (key, lab) in enumerate((("recall", "накопленный recall C*"), ("R", "частичная награда R"))):
            ax = axes[i, j]
            for p, pname, col in POL:
                s = accum["policies"][p]["accum"][g][key]
                t = np.arange(len(s))
                m = np.array([x["mean"] for x in s])
                e = np.array([x["se"] for x in s])
                ax.plot(t, m, color=col, lw=LW, marker="o", ms=3, label=pname)
                ax.fill_between(t, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(f"{name} (n {accum['groups_n'][g]})")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 1:
                ax.set_xlabel("шаг эпизода t")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle("Накопление внутри эпизода (val, K_max 5): контекст после шага t; закончившийся эпизод держит "
                 "свой финал; полоса = ± SE", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "p8_episode_accumulation.png")


# --- 4. GRPO -------------------------------------------------------------------------------------------------------

def fig_grpo(run, theta) -> None:
    rs = rows(run)
    it = sorted((r for r in rs if r["phase"] == "grpo"), key=lambda r: r["step"])
    va = sorted((r for r in rs if r["phase"] == "grpo_val"), key=lambda r: r["step"])
    x = np.array([r["step"] for r in it])
    rw = np.array([r["reward_mean"] for r in it])
    k = 20
    sm = np.convolve(rw, np.ones(k) / k, mode="valid")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(x, rw, color=BLUE, lw=0.6, alpha=0.25, label="награда батча (Q2 ≈ 50 % батча)")
    a1.plot(x[k - 1:], sm, color=BLUE, lw=LW, label=f"скользящее среднее ({k} итераций)")
    vx = [r["step"] for r in va]
    a1.plot(vx, [r["val_by_group"]["version"]["all_share"] for r in va], color=ORANGE, marker="s", lw=1.2,
            label="val Q2 all@5")
    a1.plot(vx, [r["val_reward_mean"] for r in va], color=INK, marker="o", lw=1.2, label="val R (весь val)")
    if theta is not None:
        a1.axvline(theta, color=MUTED, ls="--", lw=1, label=f"лучшая val-точка (итерация {theta})")
    a1.set_ylabel("награда / доля")
    a1.set_ylim(0, 1)
    a1.set_title("GRPO с упором на Q2 (β 0.3, lr 5e-5, G_s 8): обучение и валидация")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)
    a2.plot(x, [r["kl_ref"] for r in it], color=AQUA, lw=1.0, label="KL(π‖π_ref)")
    a2.axhline(0.5, color=MUTED, ls=":", lw=1, label="порог тревоги §8.7 (0.5)")
    a2.set_xlabel("итерация GRPO")
    a2.set_ylabel("KL")
    a2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "p8_grpo_q2.png")


# --- 5. system -----------------------------------------------------------------------------------------------------

SYS = (("p8g", "struct-агент (θ*)", BLUE), ("ranker_k5", "RL-ранкер П5", AQUA), ("router", "роутер → по типу", ORANGE))


def fig_system(tables, exp) -> None:
    tab = tables["table"]
    panels = (("all5", "all", "all@5"), ("all10", None, "all@10 (агент + fused-дозаполнение)"),
              ("all_dyn", "all", "all на своём наборе (динамический k)"), ("R", "R", "награда R"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    gs = [g for g, _, _ in GROUPS] + ["all"]
    names = [GNAME.get(g, "весь val") for g in gs]
    xs = np.arange(len(gs))
    wdt = 0.2
    for ax, (m, em, title) in zip(axes.flat, panels):
        for i, (sname, lab, col) in enumerate(SYS):
            mv = [tab[sname][g][m]["mean"] for g in gs]
            se = [tab[sname][g][m]["se"] for g in gs]
            ax.bar(xs + (i - 1.5) * wdt, mv, wdt, yerr=se, color=col, label=lab, capsize=2,
                   error_kw={"lw": 0.8, "ecolor": INK2})
        if em:
            ax.bar(xs + 1.5 * wdt, [exp[g][em] for g in gs], wdt, fill=False, edgecolor=INK, lw=1.0, ls="--",
                   label="потолок эксперта")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=10)
    fig.suptitle("Системы на val (K_max 5, ± SE)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "p8_system.png")


# --- 6. showcase ---------------------------------------------------------------------------------------------------

TITLE3 = "3. Итог роутер-системы на val\nподсчёт решён; версии — слабое место"


def fig_showcase(run, accum, tables, exp) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
    val, best = sft_val(run)
    val = [v for v in val if v["step"] <= best]
    ep = [v["step"] for v in val]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    ax = axes[0]
    for g in ("count", "version"):
        ax.plot(ep, [v["val_by_group"][g]["all_share"] for v in val], color=GCOL[g], lw=2.6, marker="o", ms=5,
                label=GNAME[g])
        ax.axhline(exp[g]["all"], color=GCOL[g], ls="--", lw=1.4)
        ax.text(ep[-1], exp[g]["all"] + 0.015, f"эксперт {exp[g]['all']:.2f}", color=GCOL[g], ha="right",
                fontsize=11)
    ax.set_xlabel("эпоха обучения (0 = до обучения)")
    ax.set_ylabel("val all@5")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64))
    a0 = val[0]["val_by_group"]
    a1 = val[-1]["val_by_group"]
    ax.set_title("1. Агент учится находить весь C*\n" + f"Q1: {a0['count']['all_share']:.2f} → {a1['count']['all_share']:.2f}, "
                 f"Q2: {a0['version']['all_share']:.2f} → {a1['version']['all_share']:.2f}", loc="left")
    ax = axes[1]
    for p, lab, col in (("grpo", "обученный агент", BLUE), ("random", "случайная политика", YELLOW)):
        s = accum["policies"][p]["accum"]["all"]["R"]
        t = np.arange(len(s))
        m = np.array([x["mean"] for x in s])
        e = np.array([x["se"] for x in s])
        ax.plot(t, m, color=col, lw=2.6, marker="o", ms=5, label=lab)
        ax.fill_between(t, m - e, m + e, color=col, alpha=0.18, lw=0)
    ax.set_xlabel("шаг эпизода t")
    ax.set_ylabel("частичная награда R (весь val)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    tr = accum["policies"]["grpo"]["accum"]["all"]["R"][-1]["mean"]
    rd = accum["policies"]["random"]["accum"]["all"]["R"][-1]["mean"]
    ax.set_title(f"2. Награда копится за 2–3 шага эпизода\nагент {tr:.2f} против случайной {rd:.2f}", loc="left")
    ax = axes[2]
    tab = tables["table"]["router"]
    gs = ["count", "version", "old"]
    xs = np.arange(3)
    ax.bar(xs - 0.2, [exp[g]["all"] for g in gs], 0.42, fill=False, edgecolor=INK, lw=1.4, ls="--",
           label="потолок эксперта (@5)")
    for off, m, alpha, lab in ((-0.2, "all5", 1.0, "all@5"), (0.2, "all10", 0.45, "all@10 (+ fused-дозаполнение)")):
        mv = [tab[g][m]["mean"] for g in gs]
        se = [tab[g][m]["se"] for g in gs]
        ax.bar(xs + off, mv, 0.34, yerr=se, color=[GCOL[g] for g in gs], alpha=alpha, capsize=4, label=lab)
        for x, v, e in zip(xs + off, mv, se):
            ax.text(x, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([GNAME[g].replace(" (", "\n(") for g in gs])
    ax.set_ylabel("val, доля вопросов с полным C*")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=10)
    ax.set_title(TITLE3, loc="left")
    for a in axes:
        a.grid(alpha=0.5)
    fig.tight_layout()
    save(fig, "m7_showcase.png")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--p7-sft", default="p7-sft-b")
    ap.add_argument("--grpo")
    ap.add_argument("--grpo-theta", type=int)
    ap.add_argument("--accum", required=True)
    ap.add_argument("--expert", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--showcase-title2", default=None)
    a = ap.parse_args()
    exp = expert_ceiling(a.expert)
    accum = json.loads(Path(a.accum).read_text(encoding="utf-8"))
    tables = json.loads(Path(a.tables).read_text(encoding="utf-8"))
    rnd = random_ref(accum)
    fig_learning(a.sft, exp, rnd)
    fig_loss(a.sft, a.p7_sft)
    fig_accum(accum)
    if a.grpo:
        fig_grpo(a.grpo, a.grpo_theta)
    fig_system(tables, exp)
    fig_showcase(a.sft, accum, tables, exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
