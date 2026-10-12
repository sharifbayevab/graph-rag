import itertools

import numpy as np
import pytest

from graphrag.env.reward import Weights, r_chunks, r_coherence, r_coverage, r_diversity, reward
from tests.m3_toy import toy_data

DEPTH = np.array([0, 0, 0, 0, 2, 0, 2, 1], np.int16)
W = Weights()


def test_components_by_hand():
    d = toy_data()
    assert r_chunks({0, 4}, {4}) == pytest.approx(2 / 3)
    assert r_coverage(d, {0, 4}, {4}) == 1.0
    assert r_diversity([0, 2], 10) == pytest.approx(np.log(2) / np.log(11))
    assert r_coherence(d, {0, 4}) == 1.0 and r_coherence(d, {0, 2}) == 0.5
    assert reward(d, DEPTH, (0, 4), frozenset({4}), W, 10).total == pytest.approx(0.691146, abs=1e-6)
    assert reward(d, DEPTH, (0, 2), frozenset({4}), W, 10).total == pytest.approx(0.05, abs=1e-9)
    assert reward(d, DEPTH, (0, 2, 4), frozenset({4}), W, 10).total == pytest.approx(0.603089, abs=1e-6)
    assert reward(d, DEPTH, (4, 3), frozenset({4}), W, 10).total == pytest.approx(0.633333, abs=1e-6)


def test_empty_context_is_zero_and_empty_gold_is_an_error():
    d = toy_data()
    assert reward(d, DEPTH, (), frozenset({4}), W, 10).total == 0.0
    with pytest.raises(ValueError):
        reward(d, DEPTH, (0,), frozenset(), W, 10)


@pytest.mark.parametrize("context, gold", [
    ((-1,), frozenset({4})),          # executor padding must not alias the last chunk
    ((0, 1.0), frozenset({4})),       # float id
    ((0,), frozenset({99})),          # gold out of range
    ((True,), frozenset({4})),        # bool is not a chunk id
])
def test_invalid_chunk_ids_raise(context, gold):
    with pytest.raises(ValueError):
        reward(toy_data(), DEPTH, context, gold, W, 10)


def test_numpy_chunk_ids_are_accepted():
    d = toy_data()
    a = reward(d, DEPTH, (np.int64(0), np.int64(4)), frozenset({np.int64(4)}), W, 10).total
    assert a == reward(d, DEPTH, (0, 4), frozenset({4}), W, 10).total


def test_unreachable_depth_counts_as_t_max():
    assert reward(toy_data(), np.full(8, -1, np.int16), (0, 4), frozenset({4}), W, 10).diversity == 0.0


def test_adding_a_gold_fragment_always_raises_R_on_the_toy():
    """M3 acceptance row, exhaustively on the toy: 1280 (C, g, C*) triples."""
    d = toy_data()
    bad = []
    for gs in range(1, 6):
        for G in itertools.combinations(range(5), gs):
            G = frozenset(G)
            for cs in range(0, 6):
                for C in itertools.combinations(range(5), cs):
                    base = reward(d, DEPTH, C, G, W, 10).total
                    bad += [(C, g) for g in sorted(G - set(C)) if not reward(d, DEPTH, C + (g,), G, W, 10).total > base]
    assert bad == []


def test_weights_come_from_train_yaml():
    from graphrag.env.reward import weights_from_config
    assert weights_from_config() == Weights(0.5, 0.2, 0.2, 0.1)
