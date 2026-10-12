"""SFT on expert trajectories with equivalent-action labels (IMPLEMENTATION.md §8.2; §11 step 8; plan D3)."""
import math

import numpy as np
import pytest
import torch

from graphrag.env.reward import Weights
from grl.model.policy import GraphPolicy
from grl.train.data import load_items, replay_examples
from grl.train.evaluate import make_evaluator
from grl.train.log import Checkpoints, RunLogger, read_rows
from grl.train.rollout import SerialVecEnv
from grl.train.sft import load_reference, run_sft, sft_loss, step_accuracy
from tests.m4_toy import DKW, SMALL_CFG, TOY_RANKS, toy_env, toy_graph, toy_init, toy_pairs

SFT_CFG = {"loss": "marginal", "lr": 3.0e-3, "weight_decay": 0.0, "batch_queries": 2, "max_epochs": 120,
           "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0}


def test_losses_by_hand():
    lp = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert float(sft_loss(lp, [[1, 2]], "marginal")) == pytest.approx(-math.log(0.5))
    assert float(sft_loss(lp, [[1, 2]], "uniform")) == pytest.approx(-(math.log(0.3) + math.log(0.2)) / 2)
    assert float(sft_loss(lp, [[2, 1]], "hard")) == pytest.approx(-math.log(0.3))
    assert step_accuracy(lp, [[0]]) == 1.0 and step_accuracy(lp, [[1, 2]]) == 0.0
    with pytest.raises(ValueError):
        sft_loss(lp, [[0]], "nope")


def test_sft_fits_toy_expert_trajectories_and_freezes_pi_ref(tmp_path):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    ex, expert = replay_examples(env, TOY_RANKS, items, DKW)
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, SMALL_CFG)
    ev = make_evaluator(g, SerialVecEnv(**toy_init()), TOY_RANKS, {"train": items, "val": items}, Weights(), 10, "cpu",
                        env.actions, {"train": expert, "val": expert})
    lg = RunLogger(tmp_path / "run", {"t": 1})
    res = run_sft(pol, g, TOY_RANKS, ex, ex, SFT_CFG, lg, Checkpoints(tmp_path / "run" / "checkpoints", 3), "cpu", ev)
    lg.close()
    final = [r for r in read_rows(tmp_path / "run") if r["phase"] == "sft_final"][0]
    assert final["train_step_acc"] == 1.0 and final["train_reproduction"] == 1.0
    ref = load_reference(g, SMALL_CFG, tmp_path / "run" / "pi_ref.json")
    assert not any(p.requires_grad for p in ref.parameters()) and not ref.training
    (tmp_path / "run" / "pi_ref.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        load_reference(g, SMALL_CFG, tmp_path / "run" / "pi_ref.json")


def test_train_cli_set_reaches_reward_weights():
    """--set parses YAML values into the loaded train config, and weights_from_config(cfg) sees the override
    (the R_chunks-only ablation of the controller's brief); unknown keys are refused."""
    import importlib.util

    from graphrag import config
    from graphrag.env.reward import weights_from_config
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = config.load("train")
    cli._set(cfg, "reward.weights={alpha: 1.0, beta: 0.0, gamma: 0.0, delta: 0.0}")
    cli._set(cfg, "sft.loss=uniform")
    assert weights_from_config(cfg) == Weights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    assert cfg["sft"]["loss"] == "uniform" and config.load("train")["sft"]["loss"] == "marginal"
    with pytest.raises(KeyError):
        cli._set(cfg, "sft.no_such_key=1")


def _resume_legs(tmp_path, cfg, model_cfg):
    """leg(d, epochs, resume) -> (comparable rows, SFTResult, final parameters); same torch seed / init every leg,
    like cmd_sft (which reseeds and rebuilds the model before run_sft restores the checkpoint)."""
    from grl.train.log import comparable
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    ex, expert = replay_examples(env, TOY_RANKS, items, DKW)
    g = toy_graph()

    def leg(d, epochs, resume=False):
        torch.manual_seed(0)
        pol = GraphPolicy(g, model_cfg)
        ev = make_evaluator(g, SerialVecEnv(**toy_init()), TOY_RANKS, {"train": items, "val": items}, Weights(), 10,
                            "cpu", env.actions, {"train": expert, "val": expert})
        lg = RunLogger(d, {"t": 1}, resume=resume)
        res = run_sft(pol, g, TOY_RANKS, ex, ex, {**cfg, "max_epochs": epochs}, lg, Checkpoints(d / "checkpoints", 3),
                      "cpu", ev, resume=resume)
        lg.close()
        return comparable(read_rows(d)), res, {k: v.clone() for k, v in pol.state_dict().items()}
    return leg


def _same(a, b):
    (ra, sa, pa), (rb, sb, pb) = a, b
    assert sum(r["phase"] == "sft_final" for r in rb) == 1
    assert sorted(map(str, ra)) == sorted(map(str, rb))
    assert sa.epochs == sb.epochs and sa.ref == sb.ref and sa.best_path.name == sb.best_path.name
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_sft_resume_reproduces_uninterrupted_run(tmp_path, dropout):
    """A run stopped after 2 epochs and resumed to 4 logs the same rows (wall-clock aside), ends with the same
    parameters and the same pi_ref.pt sha256 as a straight 4-epoch run -- with dropout on too (torch RNG state is
    checkpointed); exactly one sft_final row (the stale one of the first leg is dropped)."""
    leg = _resume_legs(tmp_path, {**SFT_CFG, "batch_queries": 1}, {**SMALL_CFG, "dropout": dropout})
    straight = leg(tmp_path / "a", 4)
    leg(tmp_path / "b", 2)
    _same(straight, leg(tmp_path / "b", 4, resume=True))


def test_sft_resume_after_early_stop_trains_no_extra_epoch(tmp_path):
    """Killed after the early-stop epoch's checkpoint (here: the run even finished), a resume trains nothing more:
    same rows, same epochs, same pi_ref; a no-op resume reports the epochs done, not one more."""
    cfg = {**SFT_CFG, "batch_queries": 1, "patience": 1, "min_delta": 10.0}     # epoch 2 never improves by 10 -> stop
    leg = _resume_legs(tmp_path, cfg, {**SMALL_CFG, "dropout": 0.1})
    straight = leg(tmp_path / "a", 6)
    assert straight[1].epochs == 2 and max(r["step"] for r in straight[0] if r["phase"] == "sft_val") == 2
    leg(tmp_path / "b", 6)
    _same(straight, leg(tmp_path / "b", 6, resume=True))


def test_sft_refuses_empty_val(tmp_path):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    ex, _ = replay_examples(env, TOY_RANKS, items, DKW)
    g = toy_graph()
    lg = RunLogger(tmp_path / "run", {"t": 1})
    with pytest.raises(ValueError, match="val"):
        run_sft(GraphPolicy(g, SMALL_CFG), g, TOY_RANKS, ex, [], SFT_CFG, lg, Checkpoints(tmp_path / "run" / "c", 3),
                "cpu", None)
    lg.close()


def test_every_torch_load_maps_to_cpu():
    """CUDA-saved checkpoints / π_ref / payloads must load on a CPU / MPS machine: every torch.load in grl/ and
    scripts/train.py passes map_location (and weights_only)."""
    import ast

    from graphrag import config
    files = sorted((config.ROOT / "grl").rglob("*.py")) + [config.ROOT / "scripts" / "train.py"]
    calls = []
    for f in files:
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load" \
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch":
                calls.append((f.name, n.lineno, {k.arg for k in n.keywords}))
    assert len(calls) >= 5
    assert [c for c in calls if not {"map_location", "weights_only"} <= c[2]] == []


def test_load_reference_moves_to_device(tmp_path):
    g = toy_graph()
    torch.manual_seed(0)
    pol = GraphPolicy(g, SMALL_CFG)
    meta = __import__("grl.train.sft", fromlist=["save_reference"]).save_reference(pol, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pi_ref.json", "pi_ref.pt"]      # no .tmp left behind
    ref = load_reference(g, SMALL_CFG, tmp_path / "pi_ref.json", device=torch.device("cpu"))
    assert meta["path"] == "pi_ref.pt" and all(p.device.type == "cpu" for p in ref.parameters())
    assert all(torch.equal(a, b) for a, b in zip(pol.state_dict().values(), ref.state_dict().values()))


def test_train_cli_grpo_refused_before_setup(monkeypatch):
    """Task 10: grpo runs now, but without --sft (π_ref) it is refused before the expensive setup()."""
    import importlib.util

    from graphrag import config
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "setup", lambda a: pytest.fail("setup() must not run for grpo without --sft"))
    with pytest.raises(SystemExit) as e:
        cli.main(["grpo", "--pairs", "p", "--rankings", "r", "--run", "d"])
    assert e.value.code == 2


def test_sft_patience_must_be_positive(tmp_path):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    ex, _ = replay_examples(env, TOY_RANKS, items, DKW)
    g = toy_graph()
    lg = RunLogger(tmp_path / "run", {"t": 1})
    with pytest.raises(ValueError, match="patience"):
        run_sft(GraphPolicy(g, SMALL_CFG), g, TOY_RANKS, ex, ex, {**SFT_CFG, "patience": 0}, lg,
                Checkpoints(tmp_path / "run" / "c", 3), "cpu", None)
    lg.close()
