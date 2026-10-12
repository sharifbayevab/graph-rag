"""M7 pilot 9 figures (PNG), validation only. Style / palette / helpers of scripts/plot_p8_figures.py.

    python scripts/plot_p9_figures.py --sft p9-sft-XX --accum runs/spark/p9-accum/accum.json
        --expert runs/spark/p9-expert/labels.jsonl --tables runs/p9-local/tables.json
-> docs/m7/figures/m7_showcase.png (refreshed), p9_system.png, p9_q2_templates.png"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_p8_figures as p8  # noqa: E402  (rcParams, palette, save, sft_val, _se)
from plot_p8_figures import AQUA, BLUE, GRID, INK, INK2, LW, MUTED, ORANGE, YELLOW, plt, save, sft_val  # noqa: E402

Q2T = (("what_law_changed", "что изменило\nакт"), ("which_deleted_year", "что исключено\nв году"),
       ("which_law_part", "какой закон\nизменил часть"), ("which_laws_article", "какие законы\nизменили статью"))
CELLS = (("old", "старые"), ("Q1", "Q1 (подсчёт,\nпилот 7)"), ("Q1x", "Q1x (статьи,\nкодексы)"), ("Q2", "Q2 (версии)"),
         ("all", "весь val"))
PURPLE = "#7d5ba6"


def _fine(r) -> str:
    t = ((r.get("provenance") or {}).get("m7") or {}).get("template")
    if r["pattern"] == "count":
        return "Q1x" if t in ("count_parts_article", "count_points_article", "count_codes_by_law",
                              "count_laws_amended_code_year", "count_articles_deleted_by_law") else "Q1"
    return "Q2" if r["pattern"] == "version" else "old"


def expert_cells(path) -> dict:
    """Val expert reach (= all@5) per fine group and Q2 template."""
    out = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] != "val":
            continue
        v = float(bool(r["provenance"]["meta"]["expert"]["reached"]))
        g = _fine(r)
        out[g].append(v)
        out["all"].append(v)
        if g == "Q2":
            out["Q2/" + r["provenance"]["m7"]["template"]].append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _bars(ax, groups, series, metric, exp=None, width=0.15, labels=True):
    xs = np.arange(len(groups))
    n = len(series) + (1 if exp else 0)
    for i, (src, lab, col) in enumerate(series):
        mv = [src[g][metric]["mean"] for g in groups]
        se = [src[g][metric]["se"] for g in groups]
        x = xs + (i - (n - 1) / 2) * width
        ax.bar(x, mv, width, yerr=se, color=col, label=lab, capsize=2, error_kw={"lw": 0.8, "ecolor": INK2})
        if labels:
            for xx, v, e in zip(x, mv, se):
                ax.text(xx, v + e + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK2, rotation=90)
    if exp:
        ax.bar(xs + (n - 1 - (n - 1) / 2) * width, [exp.get(g, 0) for g in groups], width, fill=False,
               edgecolor=INK, lw=1.0, ls="--", label="потолок эксперта (@5)")
    ax.set_xticks(xs)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.18)


def fig_system(t, exp) -> None:
    tab = t["table"]
    gs = [g for g, _ in CELLS]
    series = [(tab["sys:a_p8"], "(a) система пилота 8", MUTED),
              (tab["sys:b_p8_rule"], "(b) пилот 8 + правило Q2", YELLOW),
              (tab["sys:c_agent"], "(c) новый агент для Q1/Q2", BLUE),
              (tab["sys:d_agent_q2best"], "(d) новый агент; Q2 — лучшее по train", ORANGE),
              (tab["bl:ce+fused:0.6@5"], "ce+fused@5 (база)", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, (m, title) in zip(axes, (("all5", "all@5 (старые: top-5 ранкера)"), ("R", "награда R (старые: top-1 ранкера)"))):
        ser = series if m == "all5" else series[:-1]
        _bars(ax, gs, ser, m, exp if m == "all5" else None, width=0.14)
        ax.set_xticklabels([n for _, n in CELLS], fontsize=9)
        ax.set_title(title)
    axes[0].set_ylabel("val, доля вопросов с полным C*")
    axes[1].set_ylabel("R")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle("Роутер-системы на val (K_max 5, ± SE)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "p9_system.png")


def fig_q2(t, exp) -> None:
    tab = t["table"]
    cells = [f"Q2/{k}" for k, _ in Q2T] + ["Q2"]
    names = [n for _, n in Q2T] + ["все Q2"]
    series = [(tab[t["inputs"]["p8_theta"]], "агент пилота 8 (θ*)", MUTED),
              (tab["rule"], "правило «поправки акта» (6d-a)", YELLOW),
              (tab[t["inputs"]["theta"]], "новый агент (θ*, оператор amendments)", BLUE),
              (tab["bl:ce+fused:0.6@5"], "ce+fused@5", AQUA)]
    fig, ax = plt.subplots(figsize=(12, 5))
    _bars(ax, cells, series, "all5", {c: exp.get(c, 0) for c in cells}, width=0.16)
    ax.set_xticklabels([f"{n}\n(n {t['n'][c]})" for n, c in zip(names, cells)], fontsize=9)
    ax.set_ylabel("val all@5")
    ax.set_title("Q2 по шаблонам (val, ± SE)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    save(fig, "p9_q2_templates.png")


def fig_showcase(run, accum, t, exp) -> None:
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
    val, best = sft_val(run)
    val = [v for v in val if v["step"] <= best]
    ep = [v["step"] for v in val]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    ax = axes[0]
    for g, name, col in (("count", "Q1 + Q1x (подсчёт)", BLUE), ("version", "Q2 (версии)", ORANGE)):
        ax.plot(ep, [v["val_by_group"][g]["all_share"] for v in val], color=col, lw=2.6, marker="o", ms=5, label=name)
    ax.set_xlabel("эпоха SFT (0 = до обучения; SFT с нуля)")
    ax.set_ylabel("val all@5")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    a0, a1 = val[0]["val_by_group"], val[-1]["val_by_group"]
    ax.set_title("1. SFT на новой разметке\n" + f"подсчёт: {a0['count']['all_share']:.2f} → {a1['count']['all_share']:.2f}, "
                 f"Q2: {a0['version']['all_share']:.2f} → {a1['version']['all_share']:.2f}", loc="left")
    ax = axes[1]
    for p, lab, col in (("grpo", "обученный агент", BLUE), ("random", "случайная политика", YELLOW)):
        s = accum["policies"][p]["accum"]["all"]["R"]
        x = np.arange(len(s))
        m = np.array([q["mean"] for q in s])
        e = np.array([q["se"] for q in s])
        ax.plot(x, m, color=col, lw=2.6, marker="o", ms=5, label=lab)
        ax.fill_between(x, m - e, m + e, color=col, alpha=0.18, lw=0)
    ax.set_xlabel("шаг эпизода t")
    ax.set_ylabel("частичная награда R (весь val)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    tr = accum["policies"]["grpo"]["accum"]["all"]["R"][-1]["mean"]
    rd = accum["policies"]["random"]["accum"]["all"]["R"][-1]["mean"]
    ax.set_title(f"2. Награда копится за 2–3 шага\nагент {tr:.2f} против случайной {rd:.2f}", loc="left")
    ax = axes[2]
    gs = ["old", "Q1", "Q1x", "Q2"]
    sysd = t["table"]["sys:d_agent_q2best"]
    bb = t["best_baseline"]
    xs = np.arange(len(gs))
    sv = [sysd[g]["all5"]["mean"] for g in gs]
    se = [sysd[g]["all5"]["se"] for g in gs]
    bv = [bb[g]["@5"]["baseline_mean"] for g in gs]
    ax.bar(xs - 0.19, sv, 0.36, yerr=se, color=ORANGE, capsize=4, label="система (d)")
    ax.bar(xs + 0.19, bv, 0.36, color=AQUA, label="лучшая отдельная база @5")
    for x, v, e in zip(xs - 0.19, sv, se):
        ax.text(x, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    for x, v in zip(xs + 0.19, bv):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([dict(CELLS)[g] for g in gs], fontsize=11)
    ax.set_ylabel("val all@5")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11)
    ax.set_title("3. Итог системы на val (@5)\nстарые — top-5 ранкера", loc="left")
    for a in axes:
        a.grid(alpha=0.5)
    fig.tight_layout()
    save(fig, "m7_showcase.png")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--accum", required=True)
    ap.add_argument("--expert", required=True)
    ap.add_argument("--tables", required=True)
    a = ap.parse_args()
    exp = expert_cells(a.expert)
    t = json.loads(Path(a.tables).read_text(encoding="utf-8"))
    accum = json.loads(Path(a.accum).read_text(encoding="utf-8"))
    fig_system(t, exp)
    fig_q2(t, exp)
    fig_showcase(a.sft, accum, t, exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
