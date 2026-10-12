"""Reward-weight calibration on val (IMPLEMENTATION.md §7.3, §9.1)."""
import numpy as np
import pytest

from grl.train.calibrate import SPEC_WEIGHTS, calibrate, correlation, rankdata, simplex_grid
from grl.train.data import SplitGuardError


def _rows(m_eval, split="val", seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, m in enumerate(m_eval):
        out.append({"pair_id": f"p{i}", "split": split, "policy": "sft",
                    "parts": {"chunks": float(m), "coverage": float(rng.random()), "diversity": float(rng.random()),
                              "coherence": float(rng.random())}, "m_eval": float(m)})
    return out


def test_grid_and_ranks():
    g = simplex_grid(0.1)
    assert len(g) == 286 and all(abs(sum(p) - 1) < 1e-9 for p in g) and SPEC_WEIGHTS in g
    assert rankdata([3.0, 1.0, 3.0]).tolist() == [2.5, 1.0, 2.5]
    assert correlation([1, 2, 3], [2, 4, 6], "pearson") == pytest.approx(1.0)


def test_calibration_recovers_the_informative_component():
    res = calibrate(_rows(np.linspace(0, 1, 40)), step=0.1, method="spearman")
    assert res["best"]["weights"] == {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
    assert res["best"]["corr"] == pytest.approx(1.0) and res["spec"]["corr"] < 1.0 and res["n"] == 40


def test_calibration_refuses_non_val_rows_and_constant_targets():
    with pytest.raises(SplitGuardError):
        calibrate(_rows(np.linspace(0, 1, 5), split="train"))
    with pytest.raises(ValueError):
        calibrate(_rows(np.zeros(5)))
