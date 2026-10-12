"""Vectorised env over the M3 Env and policy rollouts (IMPLEMENTATION.md §6.5, §8.5, §9.6)."""
import numpy as np
import pytest
import torch

from graphrag.env.mdp import InvalidAction, check_invariants
from grl.model.policy import GraphPolicy
from grl.train.rollout import ProcVecEnv, SerialVecEnv, VecEnvError, rollout
from tests.m4_toy import MODEL_CFG, TOY_RANKS, toy_env, toy_graph, toy_init

ITEMS = [("toy|q", 0), ("toy|q", 1), ("toy|q2", 0), ("toy|q2", 1)]


def _scripted(vec, seed, items=ITEMS):
    rng, log = np.random.default_rng(seed), []
    descs = vec.reset(items)
    while any(d is not None for d in descs):
        acts = [None if d is None else int(rng.choice(np.flatnonzero(d.mask))) for d in descs]
        log.append(acts)
        descs = vec.step(acts)
    return log, vec.finals()


def test_serial_and_process_vec_env_agree():
    la, fa = _scripted(SerialVecEnv(**toy_init()), 0)
    pv = ProcVecEnv(toy_init, workers=2, start_method="spawn")
    try:
        lb, fb = _scripted(pv, 0)
    finally:
        pv.close()
    assert la == lb
    assert [(f.context, f.reason) for f in fa] == [(f.context, f.reason) for f in fb]


def test_a_masked_action_fails_loudly_in_both_vec_envs():
    sv = SerialVecEnv(**toy_init())
    d = sv.reset(ITEMS[:1])[0]
    bad = int(np.flatnonzero(~d.mask)[0])
    with pytest.raises(InvalidAction):
        sv.step([bad])
    pv = ProcVecEnv(toy_init, workers=1, start_method="spawn")
    try:
        pv.reset(ITEMS[:1])
        with pytest.raises(VecEnvError):
            pv.step([bad])
    finally:
        pv.close()


def test_rollout_encodes_once_per_query_and_respects_limits():
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, MODEL_CFG).eval()
    calls, orig = [], pol.encode
    pol.encode = lambda gg, qv: (calls.append(int(qv.shape[0])), orig(gg, qv))[1]
    eps, finals = rollout(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, ["toy|q", "toy|q2"], 4, greedy=False,
                          gen=torch.Generator().manual_seed(0), temperature=0.5)
    assert calls == [2] and len(eps) == 8 and [e.copy for e in eps[:4]] == [0, 1, 2, 3]
    env = toy_env()
    for e, f in zip(eps, finals):
        q = env.query(TOY_RANKS.get(e.qkey))
        assert f.done and check_invariants(env, q, f) == []
        assert [h.action for h in f.history] == e.actions
        assert all(bool(d.mask[a]) for d, a in zip(e.descs, e.actions))


def test_greedy_rollout_is_deterministic():
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, MODEL_CFG).eval()
    a = rollout(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, ["toy|q", "toy|q2"], 1, greedy=True)[0]
    b = rollout(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, ["toy|q", "toy|q2"], 1, greedy=True)[0]
    assert [e.actions for e in a] == [e.actions for e in b]


def test_process_vec_env_keeps_a_key_on_one_worker_across_resets():
    batches = [ITEMS, [("toy|q2", 0), ("toy|q2", 1)], [("p1|q", 0), ("toy|q2", 0)]]
    sv = SerialVecEnv(**toy_init())
    serial = [_scripted(sv, s, b) for s, b in enumerate(batches)]
    pv = ProcVecEnv(toy_init, workers=2, start_method="spawn")
    try:
        for s, b in enumerate(batches):
            lb, fb = _scripted(pv, s, b)
            la, fa = serial[s]
            assert la == lb
            assert [(f.context, f.reason) for f in fa] == [(f.context, f.reason) for f in fb]
            assert pv.owner["toy|q2"] == 1                    # i % W over this batch alone would move it to 0
        held = pv.env_keys()
    finally:
        pv.close()
    assert pv.owner == {"toy|q": 0, "toy|q2": 1, "p1|q": 0}
    assert held == [["p1|q", "toy|q"], ["toy|q2"]]          # every Env lives in exactly one worker


def test_a_worker_error_breaks_the_process_vec_env_for_good():
    pv = ProcVecEnv(toy_init, workers=2, start_method="spawn")
    try:
        descs = pv.reset(ITEMS)
        acts = [int(np.flatnonzero(d.mask)[0]) for d in descs]
        acts[0] = int(np.flatnonzero(~descs[0].mask)[0])     # worker 0 raises, worker 1 answers normally
        with pytest.raises(VecEnvError, match="InvalidAction"):
            pv.step(acts)
        for call in (pv.finals, lambda: pv.reset(ITEMS), lambda: pv.step([None] * len(ITEMS))):
            with pytest.raises(VecEnvError, match="broken"):
                call()
    finally:
        pv.close()


def test_sampling_rollout_needs_a_generator_and_a_positive_temperature():
    keys = ["toy|q", "toy|q2"]
    with pytest.raises(ValueError, match="Generator"):
        rollout(None, None, None, TOY_RANKS, keys, 2, greedy=False)
    for t in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            rollout(None, None, None, TOY_RANKS, keys, 2, greedy=False, gen=torch.Generator(), temperature=t)
