"""Reward-weight calibration (IMPLEMENTATION.md §7.3): λ* = argmax Corr(R_λ, M_eval) over the simplex on the VAL
split only (§9.1, §12 п. 7). M_eval comes from M6 (generator + ACC_L judge, plan decision D5); this module only
searches. Ties: closest (L1) to the spec weights, then lexicographically smallest."""
from __future__ import annotations

import numpy as np

from grl.train.data import SplitGuardError

SPEC_WEIGHTS = (0.5, 0.2, 0.2, 0.1)
PARTS = ("chunks", "coverage", "diversity", "coherence")
NAMES = ("alpha", "beta", "gamma", "delta")


def simplex_grid(step: float = 0.1) -> list[tuple]:
    n = int(round(1 / step))
    return [(a / n, b / n, c / n, (n - a - b - c) / n)
            for a in range(n + 1) for b in range(n + 1 - a) for c in range(n + 1 - a - b)]


def rankdata(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def correlation(x, y, method: str = "spearman") -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    elif method != "pearson":
        raise ValueError(f"method {method!r} is not spearman | pearson")
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calibrate(rows, step: float = 0.1, method: str = "spearman") -> dict:
    bad = sorted({r["split"] for r in rows} - {"val"})
    if bad:
        raise SplitGuardError(f"reward weights are calibrated on val only (§7.3, §9.1); got splits {bad}")
    P = np.array([[r["parts"][k] for k in PARTS] for r in rows], np.float64)
    y = np.array([r["m_eval"] for r in rows], np.float64)
    if y.std() == 0:
        raise ValueError("M_eval has no variance: nothing to correlate with")
    spec = np.array(SPEC_WEIGHTS)
    scored = []
    for lam in simplex_grid(step):
        c = correlation(P @ np.array(lam), y, method)
        if not np.isnan(c):
            scored.append((round(c, 9), -round(float(np.abs(np.array(lam) - spec).sum()), 9), tuple(-v for v in lam), lam))
    if not scored:
        raise ValueError("every R_λ is constant on these rows")
    c, _, _, lam = max(scored)
    return {"best": {"weights": dict(zip(NAMES, lam)), "corr": c},
            "spec": {"weights": dict(zip(NAMES, SPEC_WEIGHTS)), "corr": round(correlation(P @ spec, y, method), 9)},
            "n": len(rows), "method": method, "step": step}
