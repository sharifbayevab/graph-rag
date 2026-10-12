import itertools

import numpy as np
import pytest

from graphrag.env.data import EnvData
from graphrag.env.mdp import Env, InvalidAction, check_invariants
from graphrag.env.policies import (BeamSearch, action_classes, beam_oracle, d5_accept, greedy_oracle, oracle_config,
                                   random_policy, rollout, successor_additions, trajectory_json, trajectory_labels, trajectory_steps)
from graphrag.env.reward import Weights, reward
from graphrag.retrieval.rankings import QueryRanking
from tests import m3_toy as T
from tests.m3_toy import QVEC, TOY_CFG, TOY_START, toy_data, toy_rank

W = Weights()


def _env():
    return Env(toy_data(), dict(TOY_CFG), TOY_START)


def test_greedy_oracle_by_hand():
    """CONTAINS m=1 from V₀ picks D2 (rel .9, row 3 < 4) -> fragments (#2 .9, #1 .6); R = 0.6333; nothing better."""
    env = _env()
    q = env.query(toy_rank())
    s = greedy_oracle(env, q, frozenset({4}), W)
    assert [h.action for h in s.history] == [7, 0] and s.context == (4, 3) and s.reason == "stop"
    assert reward(env.data, q.depth, s.context, frozenset({4}), W, 10).total == 0.633333333


def test_beam_is_never_worse_than_greedy_and_prefers_short():
    env = _env()
    q = env.query(toy_rank())
    for gold in (frozenset({4}), frozenset({0, 3}), frozenset({2}), frozenset({1, 2, 4})):
        g, b = greedy_oracle(env, q, gold, W), beam_oracle(env, q, gold, W, 8)
        assert reward(env.data, q.depth, b.context, gold, W, 10).total >= \
            reward(env.data, q.depth, g.context, gold, W, 10).total and b.done
    assert [h.action for h in beam_oracle(env, q, frozenset({4}), W, 8).history] == [7, 0]


def test_accept_filters_before_the_argmax():
    """Review I2: with gold {0, 3} the unconstrained best may miss a gold fragment; the accepted best reaches both
    and ends with STOP -- and is at least as good as any other accepted final."""
    env = _env()
    q = env.query(toy_rank())
    gold = frozenset({0, 3})
    b = beam_oracle(env, q, gold, W, 8, accept=d5_accept(gold))
    assert b is not None and b.reason == "stop" and gold <= b.ctx_set
    none = beam_oracle(env, q, frozenset({2}), W, 8, accept=lambda s: False)
    assert none is None


def test_beam_is_deterministic_across_runs():
    """Review C1: no id()-keyed cache -- repeated runs give the same trajectory."""
    env = _env()
    q = env.query(toy_rank())
    runs = {tuple(h.action for h in beam_oracle(env, q, frozenset({1, 2, 4}), W, 8).history) for _ in range(5)}
    assert len(runs) == 1


def test_equivalent_action_classes_by_hand():
    """At reset: lexical k=3/5/10 all add (2, 0, 3); AMENDS m=1/3/5 all select the one stub; REFERS_TO m=1,
    REGULATES m=1 and RELATED m=1 all select D2/СТ.5 and add (4, 3) with covered {4}."""
    env = _env()
    q = env.query(toy_rank())
    classes = action_classes(env, q, env.reset(q))
    assert [1, 2, 3] in classes and [13, 14, 15] in classes and [10, 19, 25] in classes and [0] in classes
    assert sorted(a for c in classes for a in c) == np.flatnonzero(env.mask(q, env.reset(q))).tolist()


def test_random_policy_is_seeded():
    env = _env()
    q = env.query(toy_rank())
    a = rollout(env, q, random_policy(3, allow_stop=False))
    assert a == rollout(env, q, random_policy(3, allow_stop=False))
    assert a.reason in ("k_max", "l_max", "t_max", "exhausted")


def test_oracle_on_gold_outside_the_rankings_does_not_break():
    """Review Focus 5 (toy form): rankings hold only fragment 0, V₀ = {D1/СТ.1}."""
    env = _env()
    rank = QueryRanking("toy|q", "q", QVEC, np.array([0]), np.array([1.0]), np.array([0]), np.array([0.8]),
                        np.zeros(0, np.int64), np.zeros(0))
    q = env.query(rank)
    q.v0 = np.array([1])
    q.depth = env.data.graph.depth_from(q.v0, 10)
    s = beam_oracle(env, q, frozenset({2}), W, 8)
    assert s.done and check_invariants(env, q, s) == []


def test_trajectory_is_exactly_the_8_1_shape_with_labels():
    env = _env()
    q = env.query(toy_rank())
    s = greedy_oracle(env, q, frozenset({4}), W)
    assert trajectory_steps(env, s) == [{"t": 1, "op": "graph", "arg": {"r": "CONTAINS", "m": 1}},
                                        {"t": 2, "op": "stop"}]
    equiv, masks = trajectory_labels(env, q, s)
    assert 7 in equiv[0] and equiv[1] == [0] and all(a in m for a, m in zip([7, 0], masks))
    j = trajectory_json(env, q, s, ["D2/СТ.5#2"], reward(env.data, q.depth, s.context, frozenset({4}), W, 10))
    assert j["C_ref"] == ["D2/СТ.5#2"] and j["q"] == "toy question" and j["actions"] == [7, 0]
    assert j["action_equiv"] == equiv and j["action_masks"] == masks
    assert j["C"] == ["D2/СТ.5#2", "D2/СТ.5#1"] and j["reason"] == "stop"


def test_accept_filter_changes_the_winner_when_the_best_final_auto_terminates():
    """Review I2, discriminating form. L_max = 500, gold {0}: lexical k=3 adds D1/СТ.2#1 (400) + D1/СТ.1#1 (100)
    = 500 -> auto-ends on l_max (no STOP) with the highest R; the accepted best is vector k=3 (4, 0, 1; 210 tokens)
    then STOP. Filtering after the argmax would return nothing."""
    env = Env(toy_data(), {**TOY_CFG, "L_max": 500}, TOY_START)
    q = env.query(toy_rank())
    gold = frozenset({0})
    free = beam_oracle(env, q, gold, W, 8)
    assert [h.action for h in free.history] == [1] and free.reason == "l_max" and not d5_accept(gold)(free)
    b = beam_oracle(env, q, gold, W, 8, accept=d5_accept(gold))
    assert [h.action for h in b.history] == [4, 0] and b.context == (4, 0, 1) and b.reason == "stop"
    assert reward(env.data, q.depth, b.context, gold, W, 10).total < reward(env.data, q.depth, free.context, gold,
                                                                            W, 10).total


def test_episode_ends_exhausted_when_no_retrieval_action_is_left():
    """K_max = 24 (never binds): AMENDS m=1 selects EXT/k/СТ.9 (chunkless); CONTAINS m=5 selects D2, D2/СТ.5, D1/СТ.1,
    D1, EXT/k and adds all five fragments; REGULATES m=1 selects ENT/organ/senat. Now every vertex is covered, both
    rankings are used up and no graph candidate is left -> "exhausted" (not STOP, not a limit)."""
    env = Env(toy_data(), {**TOY_CFG, "K_max": 24}, TOY_START)
    q = env.query(toy_rank())
    s = env.reset(q)
    for a in (13, 9, 19):
        assert not s.done
        s = env.step(q, s, a)
    assert s.done and s.reason == "exhausted" and s.context == (4, 3, 0, 1, 2) and s.length == 690
    assert s.covered == frozenset(range(8)) and s.frontier == frozenset()
    assert not env.mask(q, s).any() and check_invariants(env, q, s) == []


# ---------------------------------------------------------------- fix round M3-R6 (review I1, m2-m4)

def _stub_env(**over):
    """The toy plus X3/СТ.7 (node row 8; one fragment X3/СТ.7#1, row 5, 70 tokens, outside every ranking), reachable
    only by D2/СТ.5 ∈ V₀ -AMENDS-> EXT/k/СТ.9 (chunkless stub) -AMENDS-> X3/СТ.7 (the val-0212 shape). The names
    sort last, so the rows of the old nodes/fragments are unchanged."""
    d = EnvData.build(T.NODES + ["X3/СТ.7"], T.CHUNKS + ["X3/СТ.7#1"], T.EDGES + [("EXT/k/СТ.9", "AMENDS", "X3/СТ.7")],
                      {**T.CHUNK_PARENT, "X3/СТ.7#1": "X3/СТ.7"}, {**T.MU_INV, "X3/СТ.7": ["X3/СТ.7#1"]},
                      {**T.TOKENS, "X3/СТ.7#1": 70}, np.vstack([T.CHUNK_VECS, [[0.5, 0, 0, np.sqrt(0.75)]]]).astype(
                          np.float32), np.vstack([T.NODE_X, [[0, 0, 0, 1]]]).astype(np.float32), T.EDGE_TYPES)
    return Env(d, {**TOY_CFG, **over}, TOY_START)


def _all_finals(env, q):
    """Every reachable final, deduplicated on the state without history (the future depends only on it)."""
    finals, seen = {}, {}

    def walk(s):
        k = (s.context, s.covered)
        if seen.get(k, 99) <= s.t:
            return
        seen[k] = s.t
        for a in np.flatnonzero(env.mask(q, s)).tolist():
            c = env.step(q, s, a)
            if c.done:
                finals.setdefault((c.context, c.reason), c)
            else:
                walk(c)
    walk(env.reset(q))
    return list(finals.values())


def _exhaustive_best(env, q, gold, accept, finals=None):
    """Max true R over every accepted final (None if there is none)."""
    rs = [reward(env.data, q.depth, f.context, gold, W, 10).total for f in (finals or _all_finals(env, q)) if accept(f)]
    return max(rs) if rs else None


def test_two_amends_hops_through_a_chunkless_stub_reach_the_better_final():
    """Review I1 / M3-R6 (5). Gold {D3/СТ.7#1}: the only accepted finals go AMENDS m=1 (-> stub, adds nothing, R 0),
    then AMENDS (-> X3/СТ.7, adds the gold fragment; the best is m=3, which also takes D2/СТ.5 ∈ V₀), then STOP. Without lookahead the R-0 child is pruned at width 8 (every
    fragment-adding child has R > 0) and nothing is accepted; with the lookahead it survives and wins."""
    env = _stub_env()
    q = env.query(toy_rank())
    gold, acc = frozenset({5}), d5_accept(frozenset({5}))
    assert beam_oracle(env, q, gold, W, 8, accept=acc, lookahead=False, graph_share=0) is None
    b = beam_oracle(env, q, gold, W, 8, accept=acc)
    assert b is not None and [h.action for h in b.history] == [13, 14, 0] and b.context == (4, 3, 5)
    assert b.history[0].added == () and b.history[0].selected == (7,) and b.history[1].selected == (4, 8)
    assert reward(env.data, q.depth, b.context, gold, W, 10).total == _exhaustive_best(env, q, gold, acc)
    assert check_invariants(env, q, b) == []


@pytest.mark.parametrize("over", [{}, {"L_max": 500}, {"K_max": 24}, {"stub": True}])
def test_beam_equals_exhaustive_search_on_the_toy(over):
    """Every gold of size 1-3 (plus the stub graph): the default beam's accepted best R is exactly the exhaustive
    maximum over accepted finals, and it is None exactly when no accepted final exists."""
    over = dict(over)
    env = _stub_env(**over) if over.pop("stub", False) else Env(toy_data(), {**TOY_CFG, **over}, TOY_START)
    q = env.query(toy_rank())
    n, finals = len(env.data.chunk_ids), _all_finals(env, q)
    for g in (frozenset(c) for k in (1, 2, 3) for c in itertools.combinations(range(n), k)):
        acc = d5_accept(g)
        b, x = beam_oracle(env, q, g, W, accept=acc), _exhaustive_best(env, q, g, acc, finals)
        assert (b is None) == (x is None), sorted(g)
        assert b is None or reward(env.data, q.depth, b.context, g, W, 10).total == x, sorted(g)


def test_operator_diverse_prune_reserves_graph_slots():
    """⌈w/4⌉ kept children are graph children when that many exist; the rest go by score. Width 2 at reset: by score
    the two best are vector k=3 and k=5 (ids 4, 5); with one reserved graph slot the kept pair is vector k=3 plus the
    best-scored graph child."""
    env = _stub_env()
    q = env.query(toy_rank())
    s0 = env.reset(q)
    kids = [env.step(q, s0, a) for a in np.flatnonzero(env.mask(q, s0)).tolist() if a != 0]
    free = BeamSearch(env, q, frozenset({1}), W, 2, lookahead=False, graph_share=0)
    div = BeamSearch(env, q, frozenset({1}), W, 2, lookahead=False)
    assert free.graph_slots == 0 and div.graph_slots == 1 and BeamSearch(env, q, frozenset({1}), W, 8).graph_slots == 2
    assert [c.history[-1].action for c in free._prune(kids)] == [4, 5]
    kept = [c.history[-1].action for c in div._prune(kids)]
    best_graph = min((c for c in kids if env.actions[c.history[-1].action].op == "graph"), key=div._rank_key)
    assert kept == [4, best_graph.history[-1].action]


def test_width_comes_from_train_yaml_and_one_search_serves_every_criterion():
    assert oracle_config()["beam_width"] == 32 and oracle_config(beam_width=8)["beam_width"] == 8
    env = _env()
    q = env.query(toy_rank())
    gold = frozenset({1, 2, 4})
    bs = BeamSearch(env, q, gold, W)
    assert bs.width == 32 and bs.graph_slots == 8
    assert bs.best() == beam_oracle(env, q, gold, W) and bs.best(d5_accept(gold)) == beam_oracle(
        env, q, gold, W, accept=d5_accept(gold))
    runs = {tuple(h.action for h in beam_oracle(env, q, g, W).history) for g in [gold] * 3}
    assert len(runs) == 1


def test_trajectory_json_refuses_an_auto_terminated_final():
    """Review m2: {1, 2, 4} unconstrained ends on k_max (no STOP) -> not a §8.1 trajectory."""
    env = _env()
    q = env.query(toy_rank())
    s = beam_oracle(env, q, frozenset({1, 2, 4}), W, 8)
    assert s.reason == "k_max"
    with pytest.raises(ValueError, match="STOP"):
        trajectory_json(env, q, s, ["D1/СТ.1#2"], reward(env.data, q.depth, s.context, frozenset({1}), W, 10))


def test_masked_action_raises_even_without_the_mask_check():
    """Review m3: after lexical k=3 has taken the whole lexical ranking (2, 0, 3), every lexical action is masked;
    step(check=False) must refuse it (not return a silent no-op), and replaying a forged history raises too."""
    env = _env()
    q = env.query(toy_rank())
    s = env.step(q, env.reset(q), 1)
    assert s.context == (2, 0, 3) and not env.mask(q, s)[1:4].any()
    for a in (1, 2, 3):
        with pytest.raises(InvalidAction):
            env.step(q, s, a, check=False)
    step = type(s.history[0])
    forged = type(s)(**{**s.__dict__, "t": 2, "history": (s.history[0], step(2, 2, (), ()))})
    with pytest.raises(InvalidAction):
        trajectory_labels(env, q, forged)


def test_successor_additions_equal_env_effect_action_by_action():
    """The shared-Δ_V lookahead equals env.effect for every valid non-STOP action, on states along random rollouts
    of the plain toy, the stub toy and a tight-budget toy."""
    for env in (_env(), _stub_env(), Env(toy_data(), {**TOY_CFG, "K_max": 24, "L_max": 300}, TOY_START)):
        q = env.query(toy_rank())
        for seed in range(30):
            s, pol = env.reset(q), random_policy(seed, allow_stop=False)
            while not s.done:
                valid = [a for a in np.flatnonzero(env.mask(q, s)).tolist() if a]
                want = sorted(tuple(env.effect(q, s, env.actions[a])[0]) for a in valid)
                assert sorted(successor_additions(env, q, s)) == want
                s = env.step(q, s, pol(env, q, s))
