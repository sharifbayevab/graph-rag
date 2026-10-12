import numpy as np
import pytest

from graphrag.env.mdp import Env
from graphrag.env.parallel import run
from graphrag.env.policies import random_policy, rollout
from graphrag.retrieval.executor import delta_v
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank


def _delta_v_ref(data, q, cand, context, lambda_red, decimals):
    rel = np.asarray(q.node_rel, np.float64)[cand]
    red = np.zeros(len(cand))
    if len(context) and lambda_red:
        E = data.vecs64[list(context)]
        for i, v in enumerate(cand.tolist()):
            cs = data.mu_inv(v)
            if cs.size:
                red[i] = float((data.vecs64[cs] @ E.T).max())
    return np.round(rel - lambda_red * red, decimals)


def test_vectorised_delta_v_equals_the_reference():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    for cand, ctx in ((np.arange(8), (4,)), (np.array([0, 3, 5, 7]), (0, 2)), (np.array([1]), ()), (np.array([6, 7]), (1,))):
        assert delta_v(env.data, q, cand, ctx, 0.5, 6).tolist() == _delta_v_ref(env.data, q, cand, ctx, 0.5, 6).tolist()


def test_memoised_masks_equal_fresh_masks():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    for seed in range(50):
        s = rollout(env, q, random_policy(seed, allow_stop=False))
        fresh = Env(toy_data(), dict(TOY_CFG), TOY_START)
        q2 = fresh.query(toy_rank())
        t = fresh.reset(q2)
        for h in s.history:
            assert fresh.mask(q2, t).tolist() == env.mask(q, t).tolist()
            t = fresh.step(q2, t, h.action)


def _episode(ctx, seed):
    env, q = ctx["env"], ctx["q"]
    return tuple(h.action for h in rollout(env, q, random_policy(seed)).history)


def test_parallel_run_equals_serial_and_keeps_order():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    serial = run(_episode, list(range(30)), 1, env=env, q=q)
    assert run(_episode, list(range(30)), 2, env=env, q=q) == serial


# ---------------------------------------------------------------- Task 13 additions (memo transparency, spawn)

import itertools

from graphrag.env.policies import BeamSearch, d5_accept
from graphrag.env.reward import weights_from_config
from graphrag.env.start import query_from_vector
from graphrag.retrieval.executor import ContextSims, graph_candidates
from tests.test_m3_policies import _stub_env


def _envs():
    return [Env(toy_data(), dict(TOY_CFG), TOY_START), Env(toy_data(), {**TOY_CFG, "K_max": 24}, TOY_START),
            _stub_env()]


def test_memoised_candidates_scores_and_exhaustion_equal_the_plain_functions():
    for env in _envs():
        q = env.query(toy_rank())
        d = env.data
        for seed in range(40):
            s = env.reset(q)
            while not s.done:
                for r in range(len(d.graph.edge_types)):
                    want = graph_candidates(d, q, s.covered, r)
                    assert env.candidates(q, s.covered, r).tolist() == want.tolist()
                    cand, score = env.graph_scores(q, s, r)
                    assert cand.tolist() == want.tolist()
                    if want.size:
                        assert score.tolist() == delta_v(d, q, want, s.context, env.lambda_red, env.decimals).tolist()
                assert env._any_valid(q, s) == bool(env._mask(q, s)[1:].any())
                s = env.step(q, s, random_policy(seed)(env, q, s))


def test_context_sims_path_equals_the_plain_product_in_any_call_order():
    env = _stub_env()
    q = env.query(toy_rank())
    d, n = env.data, env.data.n_nodes
    calls = [(np.arange(n), ctx) for k in (1, 2, 3) for ctx in itertools.permutations(range(d.n_chunks), k)]
    for order in (calls, calls[::-1]):
        sims = ContextSims(d, cap=2)                       # also exercises the column growth
        for cand, ctx in order:
            assert delta_v(d, q, cand, ctx, 0.5, 6, sims=sims).tolist() == _delta_v_ref(d, q, cand, ctx, 0.5, 6).tolist()


def test_memo_belongs_to_one_query_context():
    """Two contexts with the same qkey (query_from_vector, other V₀) never read each other's memo entries."""
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    qa = query_from_vector(env.data, "same", np.array([1, 0, 0, 0], np.float32), [4])
    qb = query_from_vector(env.data, "same", np.array([1, 0, 0, 0], np.float32), [6])
    for _ in range(2):
        for q in (qa, qb):
            fresh = Env(toy_data(), dict(TOY_CFG), TOY_START)
            s = env.reset(q)
            assert env.mask(q, s).tolist() == fresh._mask(q, s).tolist()
            assert env.candidates(q, s.covered, 0).tolist() == graph_candidates(env.data, q, s.covered, 0).tolist()


def test_beam_is_the_same_with_a_one_entry_memo():
    W = weights_from_config()
    big, tiny = _stub_env(), _stub_env(memo_max=1)
    qb, qt = big.query(toy_rank()), tiny.query(toy_rank())
    for g in (frozenset({5}), frozenset({0, 3}), frozenset({1, 2, 4})):
        a, b = BeamSearch(big, qb, g, W), BeamSearch(tiny, qt, g, W)
        assert [(f.context, f.reason, [h.action for h in f.history]) for f in a.finals] == \
               [(f.context, f.reason, [h.action for h in f.history]) for f in b.finals]
        for acc in (None, d5_accept(g)):
            x, y = a.best(acc), b.best(acc)
            assert (x is None and y is None) or x.history == y.history
    assert len(tiny._memo_m) <= 1 and len(tiny._memo_c) <= 1 and len(tiny._memo_s) <= 1


def _toy_ctx():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    return {"env": env, "q": env.query(toy_rank())}


def test_parallel_run_with_a_per_worker_init_equals_serial():
    """spawn (train.yaml parallel.start_method): each worker builds its context once with `init`. (fork is not
    tested: CPython warns that forking this multi-threaded process may deadlock the child.)"""
    serial = run(_episode, list(range(24)), 1, init=_toy_ctx)
    assert run(_episode, list(range(24)), 3, start_method="spawn", init=_toy_ctx) == serial
    assert run(_episode, [], 3, start_method="spawn", init=_toy_ctx) == []


# ---------------------------------------------------------------- Task 13 fix (review I1, m1, m2): loud failure

import os
import signal
from contextlib import contextmanager

from graphrag.env.parallel import ParallelRunError


@contextmanager
def _deadline(sec: int):
    """Timeout guard: a hang fails the test instead of blocking the suite."""
    def boom(*_):
        raise TimeoutError(f"parallel.run did not return within {sec} s")
    old = signal.signal(signal.SIGALRM, boom)
    signal.alarm(sec)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _die_on_3(ctx, i):
    if i == 3:
        os._exit(7)                                    # what a segfault / OOM kill looks like to the pool
    return i


def _bad_init():
    raise RuntimeError("init failed in the worker")


def _raise_on_2(ctx, i):
    if i == 2:
        raise ValueError("bad item")
    return i


def test_worker_death_raises_instead_of_hanging():
    with _deadline(30), pytest.raises(ParallelRunError, match="pool broke"):
        run(_die_on_3, list(range(8)), 2, start_method="spawn")


def test_failing_init_raises_instead_of_respawning():
    with _deadline(30), pytest.raises(ParallelRunError, match="pool broke"):
        run(_episode, list(range(8)), 2, start_method="spawn", init=_bad_init)


def test_fn_exception_names_the_item():
    with _deadline(30), pytest.raises(ParallelRunError, match=r"item 2 \(2\).*ValueError: bad item"):
        run(_raise_on_2, list(range(6)), 2, start_method="spawn")


def test_memo_arrays_are_read_only_and_the_mask_key_has_length():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    s = env.reset(q)
    cand, score = env.graph_scores(q, s, 0)
    for a in (cand, score, env.candidates(q, s.covered, 1)):
        with pytest.raises(ValueError):
            a[...] = 0
    s1 = env.step(q, s, 1)                             # lexical k=3: length 550 of L_max 1000
    forged = type(s1)(**{**s1.__dict__, "length": 990})
    assert env.mask(q, s1).tolist() != env.mask(q, forged).tolist()
    assert env.mask(q, forged).tolist() == Env(toy_data(), dict(TOY_CFG), TOY_START)._mask(q, forged).tolist()
