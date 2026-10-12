import numpy as np
import pytest

from graphrag.env.start import calibrate_start, make_query, select_v0, v0_hit
from tests.m3_toy import TOY_START, toy_data, toy_rank


def test_toy_v0_scores_and_order():
    d = toy_data()
    q = make_query(d, toy_rank(), TOY_START, 10)
    assert q.node_lex.tolist() == [1.0, 0.8, 1.0, 0.2, 0.2, 1.0, 0.0, 0.0]
    assert q.node_sem[:5].tolist() == pytest.approx([0.8, 0.8, 0.0, 0.9, 0.9]) and np.isinf(q.node_sem[5:]).all()
    assert q.node_rel.tolist() == pytest.approx([0.8, 0.8, 0.0, 0.9, 0.9, 0.5, 0.1, 0.7])
    assert q.v0.tolist() == [2, 0, 5, 4, 3, 1]
    assert q.depth.tolist() == [0, 0, 0, 0, 0, 0, 2, 1]


def test_cap_and_empty_fallback():
    """Review Focus (V₀ never empty)."""
    lex = np.array([0.9, 0.9, 0.1])
    sem = np.array([0.2, -np.inf, 0.95])
    depth = np.array([0, 1, 2], np.int16)
    assert select_v0(lex, sem, depth, 0.5, 0.9, 2).tolist() == [2, 1]
    assert select_v0(lex, sem, depth, 0.99, 0.99, 64).tolist() == [2]


def test_hit_counts_the_gold_node_or_its_parent():
    d = toy_data()
    assert v0_hit(d, np.array([4]), frozenset({3})) and v0_hit(d, np.array([1]), frozenset({0}))
    assert v0_hit(d, np.array([0]), frozenset({0}))
    assert not v0_hit(d, np.array([5]), frozenset({4}))


def test_calibration_reads_only_the_train_split():
    d = toy_data()
    with pytest.raises(ValueError, match="train"):
        calibrate_start(d, [toy_rank()], [frozenset({4})], ["test"], [0.5], [0.85], 64)
    res = calibrate_start(d, [toy_rank()], [frozenset({4})], ["train"], [0.5, 0.9], [0.85, 0.95], 64)
    assert res["best"]["hit_rate"] == 1.0 and len(res["grid"]) == 4
    # tie on hit rate: fewer passing vertices wins (5 at τ_lex=.9 vs 6 at .5)
    assert (res["best"]["tau_lex"], res["best"]["tau_sem"], res["best"]["mean_passing"]) == (0.9, 0.85, 5.0)
    # cap=2 truncates V₀ to [2, 0] (score 1.0, deeper first) for every τ in the grid -> no hit
    capped = calibrate_start(d, [toy_rank()], [frozenset({4})], ["train"], [0.5, 0.9], [0.85, 0.95], 2)
    assert capped["best"]["hit_rate"] == 0.0


@pytest.mark.parametrize("ranks, golds, splits, cap", [
    ([], [], [], 64),                                              # no queries
    (["R", "R"], [frozenset({4})] * 2, ["train"], 64),             # fewer split labels than queries
    (["R"], [], ["train"], 64),                                    # no gold set
    (["R"], [frozenset({4})], ["train"], 0),                       # empty V₀ cap
])
def test_calibration_rejects_misaligned_or_empty_inputs(ranks, golds, splits, cap):
    ranks = [toy_rank() for _ in ranks]
    with pytest.raises(ValueError):
        calibrate_start(toy_data(), ranks, golds, splits, [0.5], [0.85], cap)
