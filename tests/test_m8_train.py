# tests/test_m8_train.py -- M7 pilot 8 training options: SFT early stop on the val reward, the epoch-0 row and
# init_policy.pt, train loss in eval mode; GRPO pattern upweighting (batch share, resume exactness, refusals).
import dataclasses

import pytest
import torch

from graphrag.env.reward import Weights
from grl.model.policy import GraphPolicy
from grl.train.data import load_items, replay_examples
from grl.train.evaluate import make_evaluator
from grl.train.grpo import run_grpo
from grl.train.log import Checkpoints, RunLogger, comparable, read_rows
from grl.train.rollout import SerialVecEnv
from grl.train.sft import INIT_POLICY, freeze_reference, run_sft
from tests.m4_toy import DKW, SMALL_CFG, TOY_RANKS, toy_env, toy_graph, toy_init, toy_pairs
from tests.test_m5_grpo import ALARMS, GRPO_CFG, _pol
from tests.test_m5_sft import SFT_CFG


def _sft(tmp_path, name, cfg, resume=False):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    ex, expert = replay_examples(env, TOY_RANKS, items, DKW)
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, {**SMALL_CFG, "dropout": 0.1})
    ev = make_evaluator(g, SerialVecEnv(**toy_init()), TOY_RANKS, {"train": items, "val": items}, Weights(), 10, "cpu",
                        env.actions, {"train": expert, "val": expert})
    d = tmp_path / name
    lg = RunLogger(d, {"t": 1}, resume=resume)
    res = run_sft(pol, g, TOY_RANKS, ex, ex, {**SFT_CFG, **cfg}, lg, Checkpoints(d / "checkpoints", 3), "cpu", ev,
                  resume=resume)
    lg.close()
    return res, read_rows(d), d


def test_epoch0_row_and_init_policy(tmp_path):
    res, rows, d = _sft(tmp_path, "a", {"max_epochs": 3, "eval_epoch0": True, "train_eval_loss": True})
    val = [r for r in rows if r["phase"] == "sft_val"]
    assert [r["step"] for r in val] == [0, 1, 2, 3]
    assert all("train_loss_eval" in r and "val_loss" in r and "val_reward_mean" in r for r in val)
    assert (d / INIT_POLICY).is_file()
    torch.manual_seed(0)
    fresh = GraphPolicy(toy_graph(), {**SMALL_CFG, "dropout": 0.1}).state_dict()
    saved = torch.load(d / INIT_POLICY, map_location="cpu", weights_only=True)["model"]
    assert all(torch.equal(fresh[k], saved[k]) for k in fresh)              # the weights before any update


def test_defaults_log_no_epoch0(tmp_path):
    _, rows, d = _sft(tmp_path, "a", {"max_epochs": 2})
    assert [r["step"] for r in rows if r["phase"] == "sft_val"] == [1, 2]
    assert not (d / INIT_POLICY).exists() and "train_loss_eval" not in rows[-2]


def test_early_stop_on_reward(tmp_path):
    """min_delta 10 on the reward: no epoch after the first ever improves -> stop after 1 + patience epochs."""
    res, rows, _ = _sft(tmp_path, "a", {"max_epochs": 50, "early_stop": "reward", "patience": 2, "min_delta": 10.0})
    assert res.epochs == 3
    with pytest.raises(ValueError):
        _sft(tmp_path, "b", {"early_stop": "nope"})


def test_early_stop_reward_resume_is_exact(tmp_path):
    cfg = {"early_stop": "reward", "patience": 3, "eval_epoch0": True, "batch_queries": 1}
    straight = _sft(tmp_path, "a", {**cfg, "max_epochs": 4})
    _sft(tmp_path, "b", {**cfg, "max_epochs": 2})
    again = _sft(tmp_path, "b", {**cfg, "max_epochs": 4}, resume=True)
    assert sorted(map(str, comparable(straight[1]))) == sorted(map(str, comparable(again[1])))
    assert straight[0].epochs == again[0].epochs


def _grpo(tmp_path, name, cfg, resume=False):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    items = [dataclasses.replace(it, pattern="version") if it.pid == "p1" else it for it in items]
    pol, g = _pol(0)
    vec = SerialVecEnv(**toy_init())
    ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / name, {"t": 1}, resume=resume)
    try:
        run_grpo(pol, freeze_reference(pol), g, vec, TOY_RANKS, items, {**GRPO_CFG, **cfg}, Weights(), 10, lg,
                 Checkpoints(tmp_path / name / "checkpoints", 3), "cpu", ev, env.actions, ALARMS, resume=resume)
    finally:
        lg.close()
    return pol, read_rows(tmp_path / name)


UP = {"upweight_pattern": "version", "upweight_share": 0.5}


def test_upweight_share_in_every_batch(tmp_path):
    _, rows = _grpo(tmp_path, "a", {**UP, "iterations": 4})
    it = [r for r in rows if r["phase"] == "grpo"]
    assert len(it) == 4 and all(r["batch_up_share"] == 0.5 for r in it)


def test_upweight_resume_is_exact(tmp_path):
    full, rows = _grpo(tmp_path, "a", {**UP, "iterations": 4})
    _grpo(tmp_path, "b", {**UP, "iterations": 2})
    part, rows_b = _grpo(tmp_path, "b", {**UP, "iterations": 4}, resume=True)
    for (k, x), (_, y) in zip(full.state_dict().items(), part.state_dict().items()):
        assert torch.equal(x, y), k
    assert comparable(rows_b) == comparable(rows)


def test_upweight_refusals(tmp_path):
    with pytest.raises(ValueError, match="upweight"):
        _grpo(tmp_path, "a", {"upweight_pattern": "version", "upweight_share": 1.0})
    with pytest.raises(ValueError, match="upweight"):
        _grpo(tmp_path, "b", {"upweight_pattern": "nope", "upweight_share": 0.5})


def test_cli_accepts_optional_keys():
    from graphrag import config
    from tests.scripts_import import load_script
    cli = load_script("train")
    cfg = config.load("train")
    cli._set(cfg, "sft.early_stop=reward")
    cli._set(cfg, "sft.eval_epoch0=true")
    cli._set(cfg, "grpo.upweight_pattern=version")
    cli._set(cfg, "grpo.upweight_share=0.5")
    assert cfg["sft"]["early_stop"] == "reward" and cfg["grpo"]["upweight_share"] == 0.5
    with pytest.raises(KeyError):
        cli._set(cfg, "sft.nope=1")
