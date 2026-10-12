"""M7 reward-calibration figures (PNG) from data/m7/calib (scripts/m7_reward_calib.py outputs). Light: JSON + numpy.

    PYTHONPATH=. python scripts/plot_m7_calib_figures.py   ->  docs/m7/figures/rc_*.png
Full U = 0.5·U_ret + 0.5·U_ans on the answered subsample; the same document folds as `fit`."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas(2, force=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grl.train import calibrate_reward as cr  # noqa: E402
from scripts.m7_reward_calib import OUT as CALIB, SEED, _groups, _jsonl, _w_vec  # noqa: E402

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
LABEL = {"f0.5": "F₀.₅", "f1": "F₁", "f2": "F₂", "f3": "F₃", "recall": "recall", "precision": "precision",
         "coverage": "coverage", "diversity": "diversity", "coherence": "coherence", "n": "n/8"}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def load():
    d = ROOT / CALIB
    cands = _jsonl(d / "candidates.jsonl")
    ans = {(r["id"], r["cand"]): r for r in _jsonl(d / "answers.jsonl")}
    names = lambda x: [n for n in cr.ANSWER_CANDIDATES if n in x["cands"] and (x["id"], n) in ans]   # noqa: E731
    util = lambda x, n: 0.5 * x["cands"][n]["u_ret"] + 0.5 * ans[(x["id"], n)]["u_ans_f1"]          # noqa: E731
    sub = [x for x in cands if len(names(x)) >= 2]
    folds = cr.doc_folds([{"id": x["id"], "docs": x["docs"]} for x in cands], 5, SEED)
    fit = json.loads((d / "fit.json").read_text(encoding="utf-8"))
    return sub, names, util, folds, fit


def fold_taus(G, folds, w_of):
    """Per-fold held-out mean τ; w_of(train_groups) -> weight vector."""
    out = []
    for f in range(5):
        tr = [(F, U) for q, F, U, _ in G if folds[q] != f]
        te = [(F, U) for q, F, U, _ in G if folds[q] == f]
        w = w_of(tr)
        t = [cr.kendall_tau(F @ w, U) for F, U in te]
        t = [x for x in t if not math.isnan(x)]
        out.append(float(np.mean(t)) if t else float("nan"))
    return np.array(out)


def ms(v):
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def fig_beta(sub, names, util, folds, fit):
    G1 = _groups(sub, names, util, 1.0)
    cur = ms(fold_taus(G1, folds, lambda tr: _w_vec(cr.CURRENT, 1.0)))
    rows = []
    for b in cr.BETAS:
        G = _groups(sub, names, util, b)
        rows.append((b, *ms(fold_taus(G, folds, cr.fit_simplex))))
    chosen = fit["u_full"]["final"]["beta"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(rows))
    ax.errorbar(xs, [r[1] for r in rows], yerr=[r[2] for r in rows], color=BLUE, lw=1.4, marker="o", ms=5,
                capsize=3, label="подобранные веса при фиксированном β")
    ax.axhline(cur[0], color=ORANGE, ls="--", lw=1.2)
    ax.axhspan(cur[0] - cur[1], cur[0] + cur[1], color=ORANGE, alpha=.08, lw=0)
    ax.text(-.3, cur[0] + cur[1] + .005, f"текущие веса: τ = {cur[0]:.3f} ± {cur[1]:.3f}", color=ORANGE, va="bottom",
            fontsize=9)
    i = [r[0] for r in rows].index(chosen)
    ax.plot([i], [rows[i][1]], marker="o", ms=11, mfc="none", mec=INK, mew=1.2)
    ax.annotate(f"выбрано β = {chosen:g}\n(на train-τ)", (i, rows[i][1]), xytext=(-60, 30),
                textcoords="offset points", fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED})
    for x, r in zip(xs, rows):
        ax.text(x + .08, r[1] - .012, f"{r[1]:.3f}", fontsize=8.5, color=INK2, va="top")
    ax.set_xticks(xs, [f"β = {r[0]:g}" for r in rows])
    ax.set_xlim(-.4, len(rows) - .2)
    ax.set_ylabel("held-out τ Кендалла (среднее по 5 фолдам ± SE)")
    ax.set_title("τ на отложенных фолдах в зависимости от β в F_β")
    ax.set_ylim(min(r[1] - r[2] for r in rows) - .03, cur[0] + cur[1] + .04)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=1)
    ax.grid(axis="x", visible=False)
    save(fig, "rc_beta_curve.png")


def fig_weights(fit):
    fw = fit["u_full"]["final"]["weights"]
    b = fit["u_full"]["final"]["beta"]
    fkey = cr.BETA_KEYS[b]
    comps = [("F-мера", "f1", fkey), ("recall", "recall", "recall"), ("precision", "precision", "precision"),
             ("coverage", "coverage", "coverage"), ("diversity", "diversity", "diversity"),
             ("coherence", "coherence", "coherence"), ("n/8", "n", "n")]
    cur = [cr.CURRENT.get(c, 0.0) for _, c, _ in comps]
    new = [fw.get(f, 0.0) for _, _, f in comps]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(comps))
    h = .38
    ax.bar(x - h / 2, cur, h, color=ORANGE, label="текущие (F₁)", edgecolor=SURFACE, lw=1.5)
    ax.bar(x + h / 2, new, h, color=BLUE, label=f"подобранные, не приняты ({LABEL[fkey]})", edgecolor=SURFACE, lw=1.5)
    for i in range(len(comps)):
        ax.text(x[i] - h / 2, cur[i] + .01, f"{cur[i]:.2f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(x[i] + h / 2, new[i] + .01, f"{new[i]:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [c[0] for c in comps])
    ax.set_ylim(0, max(cur + new) + .08)
    ax.set_ylabel("вес в R (Σ = 1)")
    ax.set_title("Веса награды: текущие и подобранные")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "rc_weights.png")


def fig_search(sub, names, util, fit):
    fin = fit["u_full"]["final"]
    b = fin["beta"]
    keys = cr.feature_names(b)
    w0 = _w_vec(fin["weights"], b)
    G = [(F, U) for _, F, U, _ in _groups(sub, names, util, b)]
    top = list(np.argsort(-w0)[:2])
    i, j = top
    rest = np.array(w0)
    rest[[i, j]] = 0
    rest = rest / rest.sum() if rest.sum() > 0 else rest
    grid = np.round(np.arange(0, 1.0001, .05), 2)
    T = np.full((len(grid), len(grid)), np.nan)
    for a_, a in enumerate(grid):
        for c_, c in enumerate(grid):
            if a + c > 1 + 1e-9:
                continue
            w = rest * (1 - a - c)
            w[i], w[j] = a, c
            if w.sum() < 1e-9:
                continue
            T[c_, a_] = cr.mean_tau(G, w)[0]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    m = ax.imshow(T, origin="lower", cmap="Blues", extent=[-.025, 1.025, -.025, 1.025], aspect="auto")
    cb = fig.colorbar(m, ax=ax, shrink=.85)
    cb.set_label("τ Кендалла (все 80 вопросов, in-sample)", color=INK2)
    ax.plot([w0[i]], [w0[j]], marker="o", ms=9, mfc=ORANGE, mec=INK, mew=1)
    ax.annotate(f"подобрано: {LABEL[keys[i]]} {w0[i]:.2f}, {LABEL[keys[j]]} {w0[j]:.2f}\nτ = {fin['tau_train']:.3f}",
                (w0[i], w0[j]), xytext=((-190, 40) if w0[i] > 0.5 else (20, 40)), textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK2}, bbox={"fc": SURFACE, "ec": GRID, "lw": .6})
    ax.set_xlabel(f"вес {LABEL[keys[i]]}")
    ax.set_ylabel(f"вес {LABEL[keys[j]]}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_title(f"τ по двум главным весам (остальные — в пропорции оптимума, β = {b:g})", fontsize=10.5)
    save(fig, "rc_weight_search.png")


def heldout_pairs(sub, names, util, folds, fit):
    """R (current, fold-fitted with the fold's β chosen on train τ) and U for every held-out context."""
    cur_R, new_R, U_all = [], [], []
    Gs = {b: _groups(sub, names, util, b) for b in cr.BETAS}
    for f in range(5):
        best = None
        for b, G in Gs.items():
            tr = [(F, U) for q, F, U, _ in G if folds[q] != f]
            w = cr.fit_simplex(tr)
            t = cr.mean_tau(tr, w)[0]
            if best is None or t > best[0] + 1e-12:
                best = (t, b, w)
        _, b, w = best
        wc = _w_vec(cr.CURRENT, 1.0)
        for (q, F, U, _), (_, F1, _, _) in zip(Gs[b], Gs[1.0]):
            if folds[q] == f:
                new_R += list(F @ w)
                cur_R += list(F1 @ wc)
                U_all += list(U)
    return np.array(cur_R), np.array(new_R), np.array(U_all)


def fig_scatter(sub, names, util, folds, fit):
    cur_R, new_R, U = heldout_pairs(sub, names, util, folds, fit)
    tau_c = fit["u_full"]["heldout"]["current"]["tau"]
    tau_n = fit["u_full"]["heldout"]["fitted"]["tau"]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-.008, .008, len(U))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, R, col, title, t in ((axs[0], cur_R, ORANGE, "текущие веса", tau_c),
                                 (axs[1], new_R, BLUE, "подобранные веса (не приняты)", tau_n)):
        ax.scatter(R + jit, U, s=14, color=col, alpha=.55, lw=0)
        rho = cr.spearman(R, U)
        ax.set_title(title)
        ax.text(.03, .97, f"Spearman по всем контекстам = {rho:.2f}\nсредний τ внутри вопроса = {t:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK2,
                bbox={"fc": SURFACE, "ec": GRID, "lw": .6})
        ax.set_xlabel("R контекста (held-out фолд)")
    axs[0].set_ylabel("полезность U = 0,5·U_ret + 0,5·U_ans")
    fig.suptitle("R против U на отложенных фолдах (80 вопросов, 300 контекстов)", fontweight="bold", y=1.02)
    save(fig, "rc_r_vs_u.png")


def fig_p5(fit):
    """Old-set contexts of the subsample (M3 train-like), R of the fitted (not adopted) and of the current weights."""
    sub, names, util, _, _ = load()
    fin = fit["u_full"]["final"]
    keys = cr.feature_names(fin["beta"])
    wv = _w_vec(fin["weights"], fin["beta"])
    Rn, Rc, U = [], [], []
    for x in sub:
        if x["set"] != "old":
            continue
        for n in names(x):
            f = x["cands"][n]["feats"]
            Rn.append(float(np.array([f[k] for k in keys]) @ wv))
            Rc.append(cr.score(f, cr.CURRENT))
            U.append(util(x, n))
    Rn, Rc, U = map(np.array, (Rn, Rc, U))
    g = U >= .5
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, R, title, thr, old in ((axs[0], Rc, "текущая R", fit["p5"]["current_q"]["0.1"]["threshold"], .55),
                                   (axs[1], Rn, "подобранная R (не принята)", fit["p5"]["new"]["0.1"]["threshold"],
                                    None)):
        ax.hist(R[g], bins, color=BLUE, alpha=.75, label=f"U ≥ 0,5 ({g.sum()})", edgecolor=SURFACE, lw=.6)
        ax.hist(R[~g], bins, color=ORANGE, alpha=.75, label=f"U < 0,5 ({(~g).sum()})", edgecolor=SURFACE, lw=.6)
        ax.axvline(thr, color=INK, lw=1.2)
        ax.text(thr - .01, .6, f"П5 = q10\n= {thr:.2f}", ha="right", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
        if old is not None:
            ax.axvline(old, color=MUTED, ls="--", lw=1.1)
            ax.text(old + .01, .45, f"старый П5 = {old:.2f}\nпропускает {np.mean(R[g] >= old):.0%}\nхороших",
                    fontsize=8.5, color=INK2, va="top", transform=ax.get_xaxis_transform())
        ax.set_title(title)
        ax.set_xlabel("R контекста")
        ax.legend(loc="upper right" if old is not None else "upper left")
    axs[0].set_ylabel("число контекстов (старые вопросы подвыборки)")
    fig.suptitle("Распределение R у хороших и плохих контекстов и порог П5", fontweight="bold", y=1.02)
    save(fig, "rc_p5_threshold.png")


def main() -> int:
    sub, names, util, folds, fit = load()
    fig_beta(sub, names, util, folds, fit)
    fig_weights(fit)
    fig_search(sub, names, util, fit)
    fig_scatter(sub, names, util, folds, fit)
    fig_p5(fit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
