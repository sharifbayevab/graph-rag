"""M7 reward-weight sweep: a Scheffé quadratic mixture model (Scheffé 1958) of the final val all@5 over the weight
simplex, from the per-seed results of data/m7/calib/sweep.json. Validation only.

    y(w) = Σ_i β_i w_i + Σ_{i<j} β_ij w_i w_j      (w on the simplex, no intercept; ridge 1e-3)

Leave-one-point-out CV, the predicted optimum on the constrained simplex (w_i >= 0.1, step 0.025) and its bootstrap
spread (seeds resampled within each point).

    PYTHONPATH=. python scripts/plot_m7_ws_surface.py
        -> docs/m7/figures/ws_surface.png (slice coherence = 0.1 + LOO), data/m7/calib/ws_surface.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from scripts.plot_m7_calib_figures import BLUE, INK, INK2, MUTED, plt, save  # noqa: E402

RED, GREEN = "#d62728", "#1baf7a"
LAM, B, SEED = 1e-3, 2000, 20261012
CUR = np.array([0.5, 0.2, 0.2, 0.1])
NAME = {"cur": "текущие", "eq": "равные", "f1": "F1-угол", "cov": "cov-угол", "div": "div-угол", "coh": "coh-угол"}


def design(W):
    W = np.atleast_2d(W)
    return np.stack([W[:, i] for i in range(4)] + [W[:, i] * W[:, j] for i, j in itertools.combinations(range(4), 2)], 1)


def fit(X, Y):
    A = design(X)
    return np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ Y)


def main() -> int:
    S = json.loads((ROOT / "data/m7/calib/sweep.json").read_text(encoding="utf-8"))
    obs = [(p, np.array(r["w"]), v["all5"]) for p, r in S.items() for v in r["seeds"].values()]
    P = np.array([o[0] for o in obs])
    X = np.stack([o[1] for o in obs])
    Y = np.array([o[2] for o in obs])
    pts = sorted(S)

    # leave-one-point-out
    loo = {}
    for p in pts:
        tr = P != p
        loo[p] = {"pred": float((design(S[p]["w"]) @ fit(X[tr], Y[tr]))[0]), "obs": S[p]["all5"]}
    inner = [p for p in pts if p not in ("div", "coh")]
    mae_in = float(np.mean([abs(loo[p]["pred"] - loo[p]["obs"]) for p in inner]))
    mae_all = float(np.mean([abs(loo[p]["pred"] - loo[p]["obs"]) for p in pts]))

    # constrained grid, fit, optimum
    grid = np.array([c for c in itertools.product(range(41), repeat=4) if sum(c) == 40], float) / 40
    grid = grid[(grid >= 0.1 - 1e-9).all(1)]
    beta = fit(X, Y)
    mu = design(grid) @ beta
    b = int(mu.argmax())
    ci = int(np.argmin(np.abs(grid - CUR).sum(1)))

    # bootstrap: resample seeds within every point
    rng = np.random.default_rng(SEED)
    idx = {p: np.flatnonzero(P == p) for p in pts}
    arg, best_val, cur_val = [], [], []
    for _ in range(B):
        sel = np.concatenate([rng.choice(idx[p], len(idx[p])) for p in pts])
        m = design(grid) @ fit(X[sel], Y[sel])
        arg.append(grid[int(m.argmax())])
        best_val.append(m.max())
        cur_val.append(m[ci])
    arg = np.array(arg)
    gain = np.array(best_val) - np.array(cur_val)
    res = {"model": "Scheffé quadratic mixture, ridge 1e-3", "n_obs": len(Y), "n_points": len(pts),
           "beta": beta.tolist(), "loo": loo, "loo_mae_inner": mae_in, "loo_mae_all": mae_all,
           "optimum": {"w": grid[b].tolist(), "pred": float(mu[b])},
           "current_pred": float(mu[ci]),
           "bootstrap": {"B": B, "argmax_mean": arg.mean(0).tolist(),
                         "argmax_q05": np.quantile(arg, .05, axis=0).tolist(),
                         "argmax_q95": np.quantile(arg, .95, axis=0).tolist(),
                         "gain_over_current_mean": float(gain.mean()),
                         "gain_q05": float(np.quantile(gain, .05)), "gain_q95": float(np.quantile(gain, .95))}}
    (ROOT / "data/m7/calib/ws_surface.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("loo_mae_inner", "loo_mae_all", "optimum", "current_pred")}, ensure_ascii=False))
    print("bootstrap", {k: np.round(v, 3).tolist() if isinstance(v, list) else round(v, 4)
                        for k, v in res["bootstrap"].items()})

    # ---- figure: (a) slice coherence = 0.1, (b) LOO predicted vs observed
    fig = plt.figure(figsize=(17, 7))
    ax = fig.add_axes([.03, .08, .5, .78])
    s = 0.1
    fine = np.array([c for c in itertools.product(range(81), repeat=3) if sum(c) == 72], float) / 80   # w1+w2+w3 = .9
    W = np.column_stack([fine, np.full(len(fine), s)])
    W = W[(W >= 0.1 - 1e-9).all(1)]
    z = design(W) @ beta
    sl = 1 - s
    tx = lambda w: (w[..., 1] / sl + 0.5 * w[..., 2] / sl, w[..., 2] / sl * np.sqrt(3) / 2)   # noqa: E731
    xs, ys = tx(W)
    tri = mtri.Triangulation(xs, ys)
    cf = ax.tricontourf(tri, z, levels=np.linspace(0.45, 0.71, 27), cmap="viridis", extend="both")
    ax.tricontour(tri, z, levels=[0.68, 0.69, 0.70], colors="white", linewidths=[0.8, 1.2, 1.8])
    ex, ey = tx(np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.7, .1, .1, .1]]))
    ax.plot(ex, ey, color=INK2, lw=1.2)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    k = 0
    for p, r in sorted(S.items(), key=lambda kv: kv[1]["w"][1]):
        w = np.array(r["w"])
        if abs(w[3] - s) > 1e-9:
            continue
        x, y = tx(w)
        ax.scatter([x], [y], s=300 if p == "cur" else 90, marker="*" if p == "cur" else "o",
                   color=RED if p == "cur" else "white", edgecolors=INK, zorder=5)
        on_edge = abs(w[2] - .1) < 1e-9
        dy = (-26 if k % 2 == 0 else -44) if on_edge else 0
        dx = 0 if on_edge else 16
        k += on_edge
        ax.annotate(f"{r['all5']:.3f}".replace(".", ","), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if on_edge else "left", va="center" if not on_edge else "baseline", fontsize=11, fontweight="bold", color=RED if p == "cur" else INK, path_effects=halo)
    a_on = arg[np.abs(arg[:, 3] - s) < 1e-9]
    ax.scatter(*tx(a_on), s=6, color="#ffb000", alpha=.25, lw=0, zorder=4)
    ox, oy = tx(grid[b])
    ax.scatter([ox], [oy], s=320, marker="X", color="#ffb000", edgecolors=INK, zorder=6)
    ax.annotate(f"оптимум модели {grid[b][0]:g}/{grid[b][1]:g}/{grid[b][2]:g}/{grid[b][3]:g} → {mu[b]:.3f}".replace(".", ","),
                (ox, oy), xytext=(18, 26), textcoords="offset points", fontsize=11, color=INK, fontweight="bold",
                path_effects=halo, arrowprops={"arrowstyle": "-", "color": INK})
    cx, cy = tx(np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1]]))
    ax.text(cx[0] - .03, cy[0] + .03, "F1 0,7", fontsize=13, fontweight="bold", ha="right", va="top")
    ax.text(cx[1] + .03, cy[1] + .03, "coverage 0,7", fontsize=13, fontweight="bold", ha="left", va="top")
    ax.text(cx[2], cy[2] + .045, "diversity 0,7", fontsize=13, fontweight="bold", ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(cx[0] - .12, cx[1] + .16)
    ax.set_ylim(cy[0] - .12, cy[2] + .09)
    ax.set_title("(а) Модель Шеффе: all@5 на срезе coherence = 0,1\n(жёлтые точки — оптимум в bootstrap по сидам)",
                 fontsize=13)
    cb = fig.colorbar(cf, cax=fig.add_axes([.53, .2, .01, .55]))
    cb.set_label("предсказанный val all@5", fontsize=11)

    ax2 = fig.add_axes([.64, .12, .33, .7])
    for p in [q for q in pts if q not in ("div", "coh")]:
        c = RED if p == "cur" else BLUE
        ax2.scatter([loo[p]["obs"]], [loo[p]["pred"]], s=90, color=c, edgecolors=INK, zorder=3)
        ax2.annotate(NAME.get(p, p), (loo[p]["obs"], loo[p]["pred"]), xytext=(5, 4), textcoords="offset points",
                     fontsize=9, color=INK2)
    lim = [0.58, 0.73]
    ax2.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax2.fill_between(lim, [x - .02 for x in lim], [x + .02 for x in lim], color=MUTED, alpha=.12, lw=0)
    far = "; ".join(f"{NAME[p]}: факт {loo[p]['obs']:.3f}, прогноз {loo[p]['pred']:.3f}" for p in ("div", "coh"))
    ax2.text(.02, .97, "вне графика (срыв обучения):\n" + far.replace(".", ","), transform=ax2.transAxes,
             fontsize=9, color=INK2, va="top")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("наблюдаемый val all@5", fontsize=11)
    ax2.set_ylabel("предсказание без этой точки (LOO)", fontsize=11)
    ax2.set_title(f"(б) Проверка: ошибка LOO {mae_in:.3f} внутри области\n(углы div / coh — срыв обучения, "
                  f"модель их не ловит)".replace(".", ","), fontsize=12)
    fig.suptitle("Аппроксимация результата по весам: квадратичная модель смеси Шеффе (12 наборов, 27 обученных агентов)",
                 fontsize=15, fontweight="bold", y=.99)
    save(fig, "ws_surface.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
