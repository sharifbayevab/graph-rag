import numpy as np
import pytest

from graphrag.env.mdp import Env, InvalidAction, action_space, check_invariants
from tests.m3_toy import EDGE_TYPES, TOY_CFG, TOY_START, toy_data, toy_rank


@pytest.fixture
def env():
    return Env(toy_data(), dict(TOY_CFG), TOY_START)


def test_action_space_is_the_28_of_the_roadmap():
    acts = action_space(EDGE_TYPES)
    assert len(acts) == 28 and acts[0].op == "stop"
    assert [a.k for a in acts[1:4]] == [3, 5, 10] and acts[4].op == "vector"
    assert (acts[7].r, acts[7].m) == ("CONTAINS", 1) and (acts[27].r, acts[27].m) == ("RELATED", 5)
    assert acts[10].arg() == {"r": "REFERS_TO", "m": 1} and acts[1].arg() == 3 and acts[0].arg() is None


def test_reset_puts_v0_in_the_frontier_not_in_the_context(env):
    q = env.query(toy_rank())
    s = env.reset(q)
    assert s.context == () and s.covered == frozenset() and s.frontier == frozenset({0, 1, 2, 3, 4, 5})


def test_mask_by_hand(env):
    q = env.query(toy_rank())
    assert np.flatnonzero(~env.mask(q, env.reset(q))).tolist() == [16, 17, 18, 22, 23, 24]   # REPEALS, DEFINES


def test_step_graph_updates_context_cover_and_frontier(env):
    q = env.query(toy_rank())
    s = env.step(q, env.reset(q), 10)                                      # graph REFERS_TO m=1
    assert s.context == (4, 3) and s.covered == frozenset({4}) and s.length == 140
    assert s.frontier == frozenset({0, 1, 2, 3, 5, 7}) and s.t == 1 and not s.done


def test_masked_action_and_finished_episode_raise(env):
    q = env.query(toy_rank())
    s = env.reset(q)
    with pytest.raises(InvalidAction):
        env.step(q, s, 16)
    s = env.step(q, s, 0)
    assert s.done and s.reason == "stop"
    with pytest.raises(InvalidAction):
        env.step(q, s, 1)


@pytest.mark.parametrize("bad", [-1, 28, 1.0, True, "3"])
def test_out_of_range_or_non_integer_action_id_raises(env, bad):
    q = env.query(toy_rank())
    with pytest.raises(InvalidAction):
        env.step(q, env.reset(q), bad)


def test_numpy_action_id_is_stored_as_int(env):
    q = env.query(toy_rank())
    s = env.step(q, env.reset(q), np.int64(10))
    assert type(s.history[-1].action) is int


def test_limits_end_the_episode(env):
    q = env.query(toy_rank())
    s = env.step(q, env.reset(q), 21)                                      # REGULATES m=3 -> |C| = K_max = 4
    assert s.done and s.reason == "k_max" and len(s.context) == 4
    short = Env(toy_data(), {**TOY_CFG, "T_max": 2}, TOY_START)
    q2 = short.query(toy_rank())
    s1 = short.step(q2, short.reset(q2), 13)                               # AMENDS m=1: stub, no fragments
    assert s1.context == () and s1.covered == frozenset({7}) and not s1.done
    s2 = short.step(q2, s1, 1)                                             # lexical k=3 at t = T_max
    assert s2.done and s2.reason == "t_max" and s2.t == 2


def test_random_episodes_respect_limits_and_invariants(env):
    from graphrag.env.policies import random_policy, rollout
    q = env.query(toy_rank())
    for seed in range(200):
        s = rollout(env, q, random_policy(seed, allow_stop=seed % 2 == 0))
        assert check_invariants(env, q, s) == [], seed


def test_env_refuses_another_graph_version():
    with pytest.raises(ValueError, match="graph"):
        Env(toy_data(), {**TOY_CFG, "graph_version": "other"}, TOY_START)
