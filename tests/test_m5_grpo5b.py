"""Pilot 5b GRPO options (default-off): entropy bonus, PPO epochs with the clipped surrogate, larger groups."""
import copy
import math

import pytest
import torch

from graphrag.env.reward import Weights
from grl.train.data import load_items
from grl.train.evaluate import make_evaluator
from grl.train.grpo import (GRPO_OPTIONAL, _resume_cfg, grpo_loss, masked_entropy, ppo_surrogate, run_grpo,
                            update)
from grl.train.log import Checkpoints, RunLogger, comparable, read_rows
from grl.train.rollout import Episode, SerialVecEnv, rollout
from grl.train.sft import freeze_reference
from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_env, toy_init, toy_pairs
from tests.test_m5_grpo import ALARMS, GRPO_CFG, _pol, _state


def test_masked_entropy_is_finite_and_its_bonus_raises_entropy():
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[2.0, 0.0, 9.0]], requires_grad=True)
    lp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), -1)
    h = masked_entropy(lp, mask)
    p = torch.softmax(torch.tensor([2.0, 0.0]), -1)
    assert float(h) == pytest.approx(float(-(p * p.log()).sum()), abs=1e-6)
    h.sum().backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad[0, 0]) < 0 < float(logits.grad[0, 1])        # dH pushes towards uniform
    pol, g = _pol(0)
    ref = freeze_reference(pol)
    q, d = _state()
    qv = torch.from_numpy(q.qvec[None]).float()

    def ent():
        with torch.no_grad():
            z, qz = pol.encode(g, qv)
            return float(masked_entropy(pol.log_probs(z, qz, [0], [d]), torch.from_numpy(d.mask[None])))
    before = ent()
    m = update(pol, ref, g, qv, [Episode("toy|q", 0, 0, [d], [1])], torch.zeros(1),
               {"beta_kl": 0.0, "grad_clip": 10.0, "ent_coef": 1.0}, torch.optim.SGD(pol.parameters(), lr=1e-2), "cpu")
    assert math.isfinite(m["loss"]) and ent() > before


def test_clipped_surrogate_math():
    lp, old = torch.log(torch.tensor([1.5, 0.5, 1.5, 0.5])), torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    sur, ratio = ppo_surrogate(lp, old, adv, 0.2)
    assert torch.allclose(ratio, torch.tensor([1.5, 0.5, 1.5, 0.5]))
    # A>0: min(1.5, 1.2) = 1.2 ; min(0.5, 0.8) = 0.5 ; A<0: min(-1.5, -1.2) = -1.5 ; min(-0.5, -0.8) = -0.8
    assert torch.allclose(sur, -torch.tensor([1.2, 0.5, -1.5, -0.8]))
    s0, _ = ppo_surrogate(lp, old, adv, 0.0)
    assert torch.allclose(s0, -ratio * adv)
    x = torch.tensor([0.3, -0.2], requires_grad=True)          # ratio 1 at the first pass: same gradient as -A log π
    a = torch.tensor([1.0, -2.0])
    ppo_surrogate(x, x.detach(), a, 0.2)[0].sum().backward()
    assert torch.allclose(x.grad, -a)


def test_defaults_are_the_old_loss_and_update():
    args = (torch.tensor([-0.5, -1.0, -2.0]), torch.tensor([1.0, 1.0, -1.0]), torch.tensor([0.1, 0.3, 0.2]),
            torch.tensor([0, 0, 1]), 2, 0.5)
    assert torch.equal(grpo_loss(*args), grpo_loss(*args, None, 0.0, None))
    q, d = _state()
    qv = torch.from_numpy(q.qvec[None]).float()
    outs = []
    for cfg in ({"beta_kl": 0.1, "grad_clip": 1.0},
                {"beta_kl": 0.1, "grad_clip": 1.0, **GRPO_OPTIONAL}):
        pol, g = _pol(0)
        ref = freeze_reference(_pol(1)[0])
        eps = [Episode("toy|q", 0, 0, [d], [1]), Episode("toy|q", 0, 1, [d], [4])]
        m = update(pol, ref, g, qv, eps, torch.tensor([1.0, -1.0]), cfg, torch.optim.Adam(pol.parameters(), 1e-3), "cpu")
        outs.append((m, {k: v.clone() for k, v in pol.state_dict().items()}))
    assert outs[0][0] == outs[1][0] and set(outs[0][0]) == {"loss", "kl_ref", "entropy", "grad_norm", "n_steps"}
    assert all(torch.equal(outs[0][1][k], outs[1][1][k]) for k in outs[0][1])
    assert _resume_cfg({**GRPO_CFG, **GRPO_OPTIONAL}) == _resume_cfg(GRPO_CFG)      # old checkpoints still resume


def test_ppo_epochs_take_more_steps_and_check_on_policy():
    env = toy_env()
    pol, g = _pol(0)
    ref = freeze_reference(pol)
    vec = SerialVecEnv(**toy_init())
    eps, _ = rollout(pol, g, vec, TOY_RANKS, ["toy|q"], 4, greedy=False, gen=torch.Generator().manual_seed(0))
    assert all(len(e.logp_b) == len(e.actions) for e in eps)
    qv = torch.from_numpy(TOY_RANKS.get("toy|q").qvec[None]).float()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    cfg = {"beta_kl": 0.1, "grad_clip": 1.0, "ppo_epochs": 3, "clip_eps": 0.2}
    m = update(pol, ref, g, qv, eps, adv, cfg, torch.optim.Adam(pol.parameters(), 1e-2), "cpu")
    assert m["onpolicy_gap"] <= 1e-4 and m["ratio_max"] > 1.0 and 0.0 <= m["clip_frac"] <= 1.0
    bad = copy.deepcopy(eps)
    bad[0].logp_b[0] += 0.5                                   # a stale behaviour log-prob is refused
    with pytest.raises(RuntimeError, match="on-policy"):
        update(pol, ref, g, qv, bad, adv, cfg, torch.optim.Adam(pol.parameters(), 1e-2), "cpu")
    assert env is not None


def _run(tmp_path, name, iterations, resume, extra):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    cfg = {**SMALL_CFG, "dropout": 0.1}
    pol, g = _pol(0, cfg)
    ref = freeze_reference(_pol(0, cfg)[0])
    vec = SerialVecEnv(**toy_init())
    ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / name, {"t": 1}, resume=resume)
    res = run_grpo(pol, ref, g, vec, TOY_RANKS, items, {**GRPO_CFG, "iterations": iterations, **extra}, Weights(), 10,
                   lg, Checkpoints(tmp_path / name / "checkpoints", 3), "cpu", ev, env.actions, ALARMS, resume=resume)
    lg.close()
    return pol, res


def test_resume_is_exact_with_the_new_options(tmp_path):
    extra = {"G_s": 8, "ent_coef": 0.02, "ppo_epochs": 2, "clip_eps": 0.2}
    full, _ = _run(tmp_path, "a", 4, False, extra)
    rows = read_rows(tmp_path / "a")
    it = [r for r in rows if r["phase"] == "grpo"]
    assert all("clip_frac" in r and "onpolicy_gap" in r for r in it)
    _run(tmp_path, "b", 2, False, extra)
    part, _ = _run(tmp_path, "b", 4, True, extra)
    for (k, x), (_, y) in zip(full.state_dict().items(), part.state_dict().items()):
        assert torch.equal(x, y), k
    assert comparable(read_rows(tmp_path / "b")) == comparable(rows)
    with pytest.raises(ValueError, match="different grpo config"):     # the new options are part of the resume cfg
        _run(tmp_path, "b", 6, True, {**extra, "ent_coef": 0.01})


def test_cli_set_accepts_only_the_optional_grpo_keys():
    from tests.scripts_import import load_script
    tr = load_script("train")
    cfg = {"grpo": {"G_s": 4}}
    for kv in ("grpo.ent_coef=0.01", "grpo.ppo_epochs=2", "grpo.clip_eps=0.2", "grpo.G_s=16"):
        tr._set(cfg, kv)
    assert cfg["grpo"] == {"G_s": 16, "ent_coef": 0.01, "ppo_epochs": 2, "clip_eps": 0.2}
    with pytest.raises(KeyError):
        tr._set(cfg, "grpo.bogus=1")
    with pytest.raises(KeyError):
        tr._set({"sft": {}}, "sft.ent_coef=1")
