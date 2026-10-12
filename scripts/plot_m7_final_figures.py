"""M7 Task 10: the strict-test figures (style / palette / bar helper of scripts/plot_p9_figures.py).

    python scripts/plot_m7_final_figures.py --report runs/f9-local/test_report.json
-> docs/m7/figures/08_hard_all_at5.png, 09_hard_recall_at5.png (test, ± SE)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p8_figures import AQUA, BLUE, MUTED, ORANGE, YELLOW, plt, save  # noqa: E402
from plot_p9_figures import PURPLE, _bars  # noqa: E402

CELLS = (("old", "старые\n(136)"), ("Q1", "Q1\n(29)"), ("Q1x", "Q1x\n(32)"), ("Q2", "Q2\n(15)"),
         ("hard", "hard =\nQ1+Q1x+Q2 (76)"), ("all", "весь тест\n(212)"))
SERIES = (("sys:s01", "(B) система (роутер: ранкер / агент s01 / правило Q2)", BLUE),
          ("agent:s01", "(A) struct-агент s01", ORANGE),
          ("rule:s01", "правило 6d-a", PURPLE),
          ("ranker_top5", "RL-ранкер, top-5", YELLOW),
          ("bl:bm25@5", "BM25@5", MUTED),
          ("bl:ce+fused:0.6@5", "ce+fused@5", AQUA))


def fig(t: dict, metric: str, title: str, name: str) -> None:
    tab = t["table"]
    gs = [g for g, _ in CELLS]
    fig_, ax = plt.subplots(figsize=(14, 5.4))
    _bars(ax, gs, [(tab[k], lab, col) for k, lab, col in SERIES], metric, width=0.13)
    ax.set_xticklabels([n for _, n in CELLS], fontsize=9)
    ax.set_ylabel(f"строгий тест, {title}")
    ax.set_ylim(0, 1.25)
    h, l = ax.get_legend_handles_labels()
    fig_.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig_.suptitle(f"Строгий тест M7: {title} по типам (K_max 5, strict-эталон, ± SE)", fontweight="bold")
    save(fig_, name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/f9-local/test_report.json")
    a = ap.parse_args(argv)
    t = json.loads(Path(a.report).read_text(encoding="utf-8"))
    fig(t, "all5", "all@5", "08_hard_all_at5")
    fig(t, "recall5", "recall@5", "09_hard_recall_at5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
