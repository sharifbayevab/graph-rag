"""GRPO (IMPLEMENTATION.md §8.3–8.6; U5 exact KL; plan Review Focus 3, 5)."""
import math

import numpy as np
import pytest
import torch

from graphrag.env.reward import Weights
from grl.model.observe import describe
from grl.model.policy import GraphPolicy
from grl.train.data import load_items
from grl.train.evaluate import make_evaluator
from grl.train.grpo import exact_kl, group_advantages, grpo_loss, run_grpo, update
from grl.train.log import Checkpoints, RunLogger, comparable, read_rows
from grl.train.rollout import Episode, SerialVecEnv
from grl.train.sft import freeze_reference
from tests.m3_toy import toy_rank
from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_env, toy_graph, toy_init, toy_pairs

ALARMS = {"tmax_share": 0.5, "op_collapse_share": 0.9, "kl_max": 0.5, "stop_share_min": 0.5}
GRPO_CFG = {"G_s": 4, "lr": 1.0e-3, "beta_kl": 0.01, "batch_queries": 2, "iterations": 4, "val_every": 2,
            "patience": 100, "temperature": 1.0, "adv_eps": 1.0e-6, "step_penalty": 0.0, "freeze_gnn": False,
            "grad_clip": 1.0, "seed": 0}


def _pol(seed=0, cfg=SMALL_CFG):
    torch.manual_seed(seed)
    g = toy_graph()
    return GraphPolicy(g, cfg).eval(), g


def test_group_advantages_and_degenerate_groups():
    A = group_advantages(torch.tensor([[1.0, 0, 0, 1], [0.3, 0.3, 0.3, 0.3]]), 1e-6)
    assert torch.allclose(A[0], torch.tensor([1.0, -1, -1, 1]), atol=1e-5)
    assert torch.equal(A[1], torch.zeros(4)) and not bool(torch.isnan(A).any())


def test_exact_kl_and_loss_have_finite_gradients_with_masked_actions():
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.0, 0.0, 5.0]], requires_grad=True)
    lp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), -1)
    lr = torch.log(torch.tensor([[0.25, 0.75, 0.0]]))
    kl = exact_kl(lp, lr, mask)
    assert float(kl) == pytest.approx(0.5 * math.log(2) + 0.5 * math.log(2 / 3), abs=1e-6)
    kl.sum().backward()
    assert bool(torch.isfinite(logits.grad).all())
    loss = grpo_loss(torch.tensor([-0.5, -1.0, -2.0]), torch.tensor([1.0, 1.0, -1.0]), torch.tensor([0.1, 0.3, 0.2]),
                     torch.tensor([0, 0, 1]), 2, 0.5)
    assert float(loss) == pytest.approx(((0.55 + 1.15) / 2 + (-2.0 + 0.1)) / 2)


def _state():
    env = toy_env()
    q = env.query(toy_rank())
    return q, describe(env, q, env.reset(q))


def test_update_moves_mass_towards_the_positive_advantage():
    pol, g = _pol()
    ref = freeze_reference(pol)
    q, d = _state()
    a1, a2 = 1, 4                                          # lexical k=3, vector k=3 -- both valid at reset
    assert d.mask[a1] and d.mask[a2]
    qv = torch.from_numpy(q.qvec[None]).float()

    def lp():
        with torch.no_grad():
            z, qz = pol.encode(g, qv)
            return pol.log_probs(z, qz, [0], [d])[0]
    before = lp()
    eps = [Episode("toy|q", 0, 0, [d], [a1]), Episode("toy|q", 0, 1, [d], [a2])]
    update(pol, ref, g, qv, eps, torch.tensor([1.0, -1.0]), {"beta_kl": 0.0, "grad_clip": 1.0},
           torch.optim.SGD(pol.parameters(), lr=1e-3), "cpu")
    after = lp()
    assert float(after[a1] - after[a2]) > float(before[a1] - before[a2])


def test_kl_term_pulls_towards_pi_ref():
    """Review Focus 3: a zero-advantage group (all-equal rewards) still yields a finite loss, the KL term and an
    optimiser step (the policy moves towards π_ref)."""
    pol, g = _pol(0)
    ref = freeze_reference(_pol(1)[0])
    q, d = _state()
    qv = torch.from_numpy(q.qvec[None]).float()

    def kl():
        with torch.no_grad():
            z, qz = pol.encode(g, qv)
            zr, qzr = ref.encode(g, qv)
            return float(exact_kl(pol.log_probs(z, qz, [0], [d]), ref.log_probs(zr, qzr, [0], [d]),
                                  torch.from_numpy(d.mask[None])))
    before = kl()
    m = update(pol, ref, g, qv, [Episode("toy|q", 0, 0, [d], [0])], torch.zeros(1), {"beta_kl": 1.0, "grad_clip": 10.0},
               torch.optim.SGD(pol.parameters(), lr=1e-3), "cpu")
    assert all(math.isfinite(m[k]) for k in ("loss", "kl_ref", "entropy", "grad_norm"))
    assert m["kl_ref"] > 0 and m["grad_norm"] > 0
    assert kl() < before


def _run(tmp_path, name, iterations, resume):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    cfg = {**SMALL_CFG, "dropout": 0.1}                    # Review Focus 5: dropout on (GRPO keeps eval mode)
    pol, g = _pol(0, cfg)
    ref = freeze_reference(_pol(0, cfg)[0])
    vec = SerialVecEnv(**toy_init())
    ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / name, {"t": 1}, resume=resume)
    res = run_grpo(pol, ref, g, vec, TOY_RANKS, items, {**GRPO_CFG, "iterations": iterations}, Weights(), 10, lg,
                   Checkpoints(tmp_path / name / "checkpoints", 3), "cpu", ev, env.actions, ALARMS, resume=resume)
    lg.close()
    return pol, res


def test_run_grpo_logs_8_6_and_resume_is_exact(tmp_path):
    full, res = _run(tmp_path, "a", 4, False)
    rows = read_rows(tmp_path / "a")
    it = [r for r in rows if r["phase"] == "grpo"]
    assert [r["step"] for r in it] == [1, 2, 3, 4] and res.iterations == 4
    assert {"reward_mean", "reward_group_std", "ep_len_mean", "len_hist", "op_hist", "stop_share", "tmax_share",
            "kl_ref", "entropy", "adv_zero_share", "loss", "grad_norm", "alarms"} <= set(it[0])
    assert all(0.0 <= r["adv_zero_share"] <= 1.0 and math.isfinite(r["loss"]) for r in it)
    assert [r["step"] for r in rows if r["phase"] == "grpo_val"] == [2, 4]
    _run(tmp_path, "b", 2, False)
    part, _ = _run(tmp_path, "b", 4, True)
    for (k, x), (_, y) in zip(full.state_dict().items(), part.state_dict().items()):
        assert torch.equal(x, y), k
    assert comparable(read_rows(tmp_path / "b")) == comparable(rows)


def test_run_grpo_refuses_non_unit_temperature(tmp_path):
    """Episodes keep no behaviour log-probs: the update is on-policy only at temperature 1 (§8.3)."""
    pol, g = _pol(0)
    with pytest.raises(ValueError, match="temperature"):
        run_grpo(pol, freeze_reference(pol), g, None, TOY_RANKS, [], {**GRPO_CFG, "temperature": 0.7}, Weights(), 10,
                 None, None, "cpu", None, None, ALARMS)


def test_run_grpo_resume_after_early_stop_runs_no_extra_iteration(tmp_path):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))

    def leg(resume):
        pol, g = _pol(0)
        vec = SerialVecEnv(**toy_init())
        ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
        lg = RunLogger(tmp_path / "r", {"t": 1}, resume=resume)
        res = run_grpo(pol, freeze_reference(pol), g, vec, TOY_RANKS, items,
                       {**GRPO_CFG, "iterations": 20, "val_every": 1, "patience": 1}, Weights(), 10, lg,
                       Checkpoints(tmp_path / "r" / "checkpoints", 3), "cpu", ev, env.actions, ALARMS, resume=resume)
        lg.close()
        return res, comparable(read_rows(tmp_path / "r"))
    first, rows = leg(False)
    assert first.stopped_early and first.iterations < 20
    again, rows2 = leg(True)
    assert again.stopped_early and again.iterations == first.iterations and rows2 == rows


def test_freeze_gnn_trains_only_aggregator_and_head():
    pol, g = _pol(0)
    ref = freeze_reference(_pol(1)[0])
    q, d = _state()
    qv = torch.from_numpy(q.qvec[None]).float()
    enc0 = {k: v.clone() for k, v in pol.encoder.state_dict().items()}
    head0 = {k: v.clone() for k, v in pol.head.state_dict().items()}
    update(pol, ref, g, qv, [Episode("toy|q", 0, 0, [d], [1])], torch.ones(1),
           {"beta_kl": 1.0, "grad_clip": 10.0, "freeze_gnn": True}, torch.optim.SGD(pol.parameters(), lr=1e-2), "cpu")
    assert all(torch.equal(enc0[k], v) for k, v in pol.encoder.state_dict().items())
    assert any(not torch.equal(head0[k], v) for k, v in pol.head.state_dict().items())


def test_train_cli_grpo_on_the_toy(tmp_path, capsys):
    """scripts/train.py grpo: θ₀ = π_ref from an SFT run dir (sha-checked), SFT baseline row grpo_val@0, run, CSV."""
    import importlib.util
    import json

    from graphrag import config
    from grl.train.sft import save_reference
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    pol, g = _pol(0)
    save_reference(pol, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"K_max": int(env.limits.K_max)}))   # SFT records K_max
    cfg = config.load("train")
    cfg["grpo"] = {**GRPO_CFG, "iterations": 2}
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": g, "vec": SerialVecEnv(**toy_init()), "ranks": TOY_RANKS,
         "train": items, "val": items, "w": Weights(), "env": env, "device": torch.device("cpu"), "det": {}}
    a = type("A", (), {"sft": str(tmp_path), "run": str(tmp_path / "run"), "resume": False, "pairs": str(f),
                       "rankings": str(f), "variant": "full"})()
    assert cli.COMMANDS["grpo"](a, S) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rows = read_rows(tmp_path / "run")
    assert out["iterations"] == 2 and out["best"].endswith(".pt")
    assert [(r["phase"], r["step"]) for r in rows if r["phase"] == "grpo_val"] == [("grpo_val", 0), ("grpo_val", 2)]
    assert (tmp_path / "run" / "grpo.csv").exists()


def _leg(tmp_path, name, cfg, resume=False, evaluate=None):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    pol, g = _pol(0)
    vec = SerialVecEnv(**toy_init())
    ev = evaluate or make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / name, {"t": 1}, resume=resume)
    try:
        return run_grpo(pol, freeze_reference(pol), g, vec, TOY_RANKS, items, {**GRPO_CFG, **cfg}, Weights(), 10, lg,
                        Checkpoints(tmp_path / name / "checkpoints", 3), "cpu", ev, env.actions, ALARMS,
                        resume=resume)
    finally:
        lg.close()


def test_hacking_alarm_fires_when_train_rises_and_val_falls(tmp_path, monkeypatch):
    """§8.7: train reward up, val reward down over the last 3 validations -> alarm_hacking in the grpo_val row; the
    history survives a resume (stored in last.pt)."""
    import grl.train.grpo as G
    real = G.summarize

    def rising(eps, *a, **k):
        out = real(eps, *a, **k)
        rising.n += 1
        return {**out, "reward_mean": float(rising.n)}
    rising.n = 0
    monkeypatch.setattr(G, "summarize", rising)
    vals = iter([0.9, 0.8, 0.7, 0.6])
    ev = lambda pol, split: {"reward_mean": next(vals)}            # noqa: E731
    _leg(tmp_path, "h", {"iterations": 2, "val_every": 1, "patience": 100}, evaluate=ev)
    _leg(tmp_path, "h", {"iterations": 4, "val_every": 1, "patience": 100}, resume=True, evaluate=ev)
    rows = [r for r in read_rows(tmp_path / "h") if r["phase"] == "grpo_val"]
    assert [r["alarm_hacking"] for r in rows] == [False, False, True, True]
    assert rows[2]["alarms"] and rows[2]["alarms"][0].startswith("hacking") and rows[0]["alarms"] == []


def test_resume_refuses_changed_cfg_and_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        _leg(tmp_path, "m", {"iterations": 2}, resume=True)
    _leg(tmp_path, "c", {"iterations": 1})
    with pytest.raises(ValueError, match="different grpo config"):
        _leg(tmp_path, "c", {"iterations": 2, "batch_queries": 1}, resume=True)


def test_patience_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="patience"):
        _leg(tmp_path, "p", {"patience": 0})
