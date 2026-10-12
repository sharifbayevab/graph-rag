"""Task GRPO-X (M7, before the test): default-off GRPO options (adv_std, upweight_mix, expert replay), the RFT
sampler / replay (grl.train.rft) and the train.py rft / sft --init-ckpt / --rft / --expert-patterns paths -- toy, CPU."""
import ast
import dataclasses
import inspect
import json

import pytest
import torch

from graphrag.env.reward import Weights
from grl.train import rft as R
from grl.train.data import SplitGuardError, load_items, replay_examples
from grl.train.evaluate import make_evaluator
from grl.train.grpo import (GRPO_OPTIONAL, _resume_cfg, collect, group_advantages, group_centered, parse_mix,
                            run_grpo)
from grl.train.log import Checkpoints, RunLogger, comparable, read_rows
from grl.train.rollout import SerialVecEnv
from grl.train.sft import freeze_reference
from tests.m4_toy import DKW, SMALL_CFG, TOY_RANKS, toy_env, toy_init, toy_pairs
from tests.test_m5_grpo import ALARMS, GRPO_CFG, _pol

NEW_KEYS = ("adv_std", "upweight_mix", "replay_lambda", "replay_patterns", "replay_per_iter")


def _items(tmp_path, patterns=None):
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    env = toy_env()
    items = load_items(f, env.data, ("train",))
    if patterns:
        items = [dataclasses.replace(it, pattern=patterns.get(it.pid, it.pattern)) for it in items]
    return env, items


def _run(tmp_path, name, cfg, iterations=4, resume=False, patterns=None, replay=False):
    env, items = _items(tmp_path, patterns)
    pol, g = _pol(0, {**SMALL_CFG, "dropout": 0.1})
    ref = freeze_reference(_pol(0, {**SMALL_CFG, "dropout": 0.1})[0])
    vec = SerialVecEnv(**toy_init())
    ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, Weights(), 10, "cpu", env.actions)
    rex = replay_examples(env, TOY_RANKS, items, DKW)[0] if replay else None
    lg = RunLogger(tmp_path / name, {"t": 1}, resume=resume)
    try:
        res = run_grpo(pol, ref, g, vec, TOY_RANKS, items, {**GRPO_CFG, "iterations": iterations, **cfg}, Weights(), 10,
                       lg, Checkpoints(tmp_path / name / "checkpoints", 3), "cpu", ev, env.actions, ALARMS,
                       resume=resume, replay_ex=rex)
    finally:
        lg.close()
    return pol, res, read_rows(tmp_path / name)


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# ---- defaults: byte-identical --------------------------------------------------------------------------------------

def test_new_keys_are_optional_and_default_off():
    assert set(NEW_KEYS) <= set(GRPO_OPTIONAL)
    assert GRPO_OPTIONAL["adv_std"] is True and GRPO_OPTIONAL["replay_lambda"] == 0.0
    assert GRPO_OPTIONAL["upweight_mix"] == "" and GRPO_OPTIONAL["replay_patterns"] == ""
    assert _resume_cfg({**GRPO_CFG, **GRPO_OPTIONAL}) == _resume_cfg(GRPO_CFG)          # old checkpoints resume
    assert _resume_cfg({**GRPO_CFG, "adv_std": False})["adv_std"] is False


def test_explicit_defaults_run_exactly_as_before(tmp_path):
    a, _, ra = _run(tmp_path, "a", {})
    b, _, rb = _run(tmp_path, "b", {k: GRPO_OPTIONAL[k] for k in NEW_KEYS})
    assert _same(a, b) and comparable(ra) == comparable(rb)
    assert not any(k in r for r in ra for k in ("replay_loss", "batch_mix_share"))


# ---- adv_std false (Dr. GRPO) ---------------------------------------------------------------------------------------

def test_group_centered_matches_r_minus_mean_and_zero_groups():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.1, 0.1, 0.1, 0.1]], dtype=torch.float64)
    c = group_centered(r)
    assert torch.allclose(c[0], torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=torch.float64))
    assert torch.equal(c[1], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(group_advantages(r, 1e-6) == 0, c == 0)                           # same zero groups


def test_collect_adv_std_false_is_centered_not_scaled(tmp_path):
    env, items = _items(tmp_path)
    pol, g = _pol(0)
    vec = SerialVecEnv(**toy_init())
    outs = []
    for flag in (True, False):
        _, _, _, r, adv, _ = collect(pol, g, vec, TOY_RANKS, items, {**GRPO_CFG, "adv_std": flag},
                                     torch.Generator().manual_seed(0), Weights(), 10, "cpu")
        outs.append((r, adv))
    (r1, a1), (r2, a2) = outs
    assert torch.equal(r1, r2)
    want = group_centered(r2.view(len(items), GRPO_CFG["G_s"])).view(-1).float()
    assert torch.equal(a2, want) and torch.equal(a1, group_advantages(r1.view(len(items), -1), 1e-6).view(-1).float())


def test_adv_std_false_run_and_resume(tmp_path):
    full, _, rows = _run(tmp_path, "a", {"adv_std": False})
    _run(tmp_path, "b", {"adv_std": False}, iterations=2)
    part, _, rows_b = _run(tmp_path, "b", {"adv_std": False}, resume=True)
    assert _same(full, part) and comparable(rows_b) == comparable(rows)


# ---- upweight_mix ---------------------------------------------------------------------------------------------------

def test_parse_mix():
    assert parse_mix("") == [] and parse_mix(None) == []
    assert parse_mix("version=0.5, count=0.3") == [("version", 0.5), ("count", 0.3)]
    for bad in ("version", "version=", "=0.5", "a=0.2,a=0.3"):
        with pytest.raises(ValueError):
            parse_mix(bad)


def test_upweight_mix_batches_and_resume(tmp_path):
    pats = {"p1": "version"}
    cfg = {"upweight_mix": "version=0.5"}
    full, _, rows = _run(tmp_path, "a", cfg, patterns=pats)
    it = [r for r in rows if r["phase"] == "grpo"]
    assert all(r["batch_mix_share"] == {"version": 0.5} for r in it)
    _run(tmp_path, "b", cfg, iterations=2, patterns=pats)
    part, _, rows_b = _run(tmp_path, "b", cfg, resume=True, patterns=pats)
    assert _same(full, part) and comparable(rows_b) == comparable(rows)


def test_upweight_mix_refusals(tmp_path):
    with pytest.raises(ValueError, match="exclude"):
        _run(tmp_path, "x", {"upweight_mix": "version=0.5", "upweight_pattern": "version", "upweight_share": 0.5},
             patterns={"p1": "version"})
    with pytest.raises(ValueError, match="upweight_mix"):          # no row of the pattern
        _run(tmp_path, "y", {"upweight_mix": "count=0.5"})
    with pytest.raises(ValueError, match="upweight_mix"):          # nothing left for the other rows
        _run(tmp_path, "z", {"upweight_mix": "version=0.9"}, patterns={"p1": "version"})


# ---- expert replay --------------------------------------------------------------------------------------------------

def test_replay_adds_the_sft_term_and_resumes_exactly(tmp_path):
    pats = {"p1": "version"}
    cfg = {"replay_lambda": 0.2, "replay_patterns": "version", "replay_per_iter": 1}
    base, _, _ = _run(tmp_path, "base", {}, patterns=pats)
    full, _, rows = _run(tmp_path, "a", cfg, patterns=pats, replay=True)
    it = [r for r in rows if r["phase"] == "grpo"]
    assert all(r["replay_loss"] > 0 and r["replay_steps"] > 0 and 0 <= r["replay_step_acc"] <= 1 for r in it)
    assert not _same(base, full)
    _run(tmp_path, "b", cfg, iterations=2, patterns=pats, replay=True)
    part, _, rows_b = _run(tmp_path, "b", cfg, resume=True, patterns=pats, replay=True)
    assert _same(full, part) and comparable(rows_b) == comparable(rows)


def test_replay_lambda_zero_ignores_the_examples(tmp_path):
    pats = {"p1": "version"}
    a, _, ra = _run(tmp_path, "a", {}, patterns=pats)
    b, _, rb = _run(tmp_path, "b", {"replay_patterns": "version"}, patterns=pats, replay=True)
    assert _same(a, b) and comparable(ra) == comparable(rb)


def test_replay_refusals(tmp_path):
    with pytest.raises(ValueError, match="replay examples"):
        _run(tmp_path, "a", {"replay_lambda": 0.2, "replay_patterns": "version"}, patterns={"p1": "version"})
    with pytest.raises(ValueError, match="replay queries"):
        _run(tmp_path, "b", {"replay_lambda": 0.2, "replay_patterns": "count", "replay_per_iter": 1},
             patterns={"p1": "version"}, replay=True)
    with pytest.raises(ValueError, match="replay_patterns"):
        _run(tmp_path, "c", {"replay_lambda": 0.2}, replay=True)


# ---- RFT sampler / replay -------------------------------------------------------------------------------------------

def test_sample_best_rows_keep_rule_and_determinism(tmp_path):
    env, items = _items(tmp_path)
    pol, g = _pol(0)
    rows = [R.sample_best(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, items, 8, torch.Generator().manual_seed(3),
                          Weights(), "cpu", margin=0.05) for _ in range(2)]
    assert rows[0] == rows[1] and [r["id"] for r in rows[0]] == [it.pid for it in items]
    for r in rows[0]:
        assert r["r_best"] >= r["r_mean"] - 1e-9 and r["n_best"] >= 1 and r["actions"]
        assert r["keep"] == (r["r_best"] > r["r_greedy"] + 0.05)
    all_keep = R.sample_best(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, items, 8, torch.Generator().manual_seed(3),
                             Weights(), "cpu", margin=-10.0)
    assert all(r["keep"] for r in all_keep)


def test_best_tie_break_prefers_fewer_steps_then_lower_copy():
    P = type("P", (), {})
    E = type("E", (), {})

    def part(t):
        p = P()
        p.total = t
        return p

    def ep(n, c):
        e = E()
        e.actions, e.copy = [1] * n, c
        return e
    assert R._best([part(0.5), part(0.7), part(0.7)], [ep(1, 0), ep(3, 1), ep(2, 2)]) == 2
    assert R._best([part(0.7), part(0.7)], [ep(2, 0), ep(2, 1)]) == 0


def test_rft_refuses_non_train_rows(tmp_path):
    env, items = _items(tmp_path)
    pol, g = _pol(0)
    val = [dataclasses.replace(items[0], split="val")]
    with pytest.raises(SplitGuardError):
        R.sample_best(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, val, 4, torch.Generator().manual_seed(0),
                      Weights(), "cpu")
    with pytest.raises(SplitGuardError):
        R.rft_examples(env, TOY_RANKS, [{"id": "p1", "qkey": "p1|q", "split": "val", "keep": True, "actions": [0]}],
                       DKW)


def test_no_gold_guard_sampling_and_replay(tmp_path, monkeypatch):
    """The policy side never sees C*: rollout() gets query keys only (gold enters through the env rewards after the
    episodes, as in GRPO collect), and the replay into SFT examples reads no gold / label name at all."""
    env, items = _items(tmp_path)
    pol, g = _pol(0)
    seen = []
    real = R.rollout

    def spy(*a, **k):
        seen.append((a, k))
        return real(*a, **k)
    monkeypatch.setattr(R, "rollout", spy)
    R.sample_best(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, items, 4, torch.Generator().manual_seed(0), Weights(),
                  "cpu")
    golds = [it.gold for it in items]
    for a, k in seen:
        assert a[4] == [it.qkey for it in items]
        assert not any(x is gd or x == gd for x in list(a) + list(k.values()) for gd in golds
                       if isinstance(x, frozenset))
    banned = ("gold", "c_ref", "y_ref", "answer", "evidence", "label", "target", "golds")
    for fn in (R.rft_examples,):
        tree = ast.parse(inspect.getsource(fn))
        names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not names & set(banned), (fn.__name__, names & set(banned))
    src = inspect.getsource(R.sample_best)                           # gold only in the reward call, after rollout
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "rollout"]
    assert calls and all("gold" not in ast.unparse(c) for c in calls)


def test_rft_examples_replay_the_sampled_episode(tmp_path):
    env, items = _items(tmp_path)
    pol, g = _pol(0)
    rows = R.sample_best(pol, g, SerialVecEnv(**toy_init()), TOY_RANKS, items, 8, torch.Generator().manual_seed(1),
                         Weights(), "cpu", margin=-10.0)
    ex = R.rft_examples(env, TOY_RANKS, rows, DKW)
    assert len(ex) == sum(len(r["actions"]) for r in rows)
    k = 0
    for r in rows:
        for t, a in enumerate(r["actions"]):
            assert ex[k].qkey == r["qkey"] and ex[k].t == t + 1 and a in ex[k].equiv and bool(ex[k].desc.mask[a])
            k += 1
    assert R.rft_examples(env, TOY_RANKS, [{**r, "keep": False} for r in rows], DKW) == []
    bad = dict(rows[0], context=[-1])
    with pytest.raises(ValueError, match="context"):
        R.rft_examples(env, TOY_RANKS, [bad], DKW)
    with pytest.raises(ValueError, match="did not end"):
        R.rft_examples(env, TOY_RANKS, [dict(rows[0], actions=rows[0]["actions"][:-1], context=None)
                                        if rows[0]["actions"][-1] == 0 else dict(rows[0], actions=[])], DKW)


# ---- CLI ------------------------------------------------------------------------------------------------------------

def _cli():
    from tests.scripts_import import load_script
    return load_script("train")


def test_cli_set_accepts_the_new_grpo_keys():
    tr = _cli()
    cfg = {"grpo": {}}
    for kv in ("grpo.adv_std=false", "grpo.upweight_mix=version=0.5,count=0.3", "grpo.replay_lambda=0.2",
               "grpo.replay_patterns=version,count", "grpo.replay_per_iter=4"):
        tr._set(cfg, kv)
    assert cfg["grpo"] == {"adv_std": False, "upweight_mix": "version=0.5,count=0.3", "replay_lambda": 0.2,
                           "replay_patterns": "version,count", "replay_per_iter": 4}


def test_cli_argument_guards():
    tr = _cli()
    base = ["--pairs", "x", "--rankings", "y"]
    for argv in (["rft", *base, "--out", "o.jsonl", "--split", "train"],          # no --ckpt
                 ["rft", *base, "--ckpt", "c", "--out", "o.jsonl"],              # split val
                 ["rft", *base, "--ckpt", "c", "--split", "train"],              # no --out
                 ["eval", *base, "--ckpt", "c", "--rft", "r.jsonl"],
                 ["eval", *base, "--ckpt", "c", "--init-ckpt", "c"]):
        with pytest.raises(SystemExit):
            tr.main(argv)


def test_cli_rft_then_sft_from_checkpoint(tmp_path, capsys, monkeypatch):
    """train.py rft writes the best-of-G rows; sft --init-ckpt --rft --expert-patterns fine-tunes on them (+ the
    expert rows of the pattern), records the inputs, and starts from the checkpoint's weights."""
    from graphrag import config
    from grl.hashing import sha256_file
    tr = _cli()
    env, items = _items(tmp_path, {"p1": "version"})
    pol, g = _pol(0)
    ck = tmp_path / "theta.pt"
    torch.save({"model": pol.state_dict()}, ck)
    cfg = config.load("train")
    cfg["grpo"] = {**GRPO_CFG}
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": g, "vec": SerialVecEnv(**toy_init()), "ranks": TOY_RANKS,
         "train": items, "val": items, "w": Weights(), "env": env, "device": torch.device("cpu"), "det": {},
         "dkw": DKW}
    out = tmp_path / "rft.jsonl"
    a = type("A", (), {"ckpt": str(ck), "out": str(out), "rft_g": 4, "rft_margin": -10.0, "variant": "full"})()
    assert tr.COMMANDS["rft"](a, S) == 0
    summ = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2 and summ["summary"]["all"]["kept"] == 2 and summ["G"] == 4
    seen = {}
    real = tr.__dict__.get("_evaluator")

    def ev(S_, expert=None):
        e = real(S_, expert)

        def wrapped(p, split):
            if "w0" not in seen:
                seen["w0"] = {k: v.clone() for k, v in p.state_dict().items()}
            return e(p, split)
        return wrapped
    monkeypatch.setattr(tr, "_evaluator", ev)
    monkeypatch.setattr(tr, "legacy_k_max", lambda: int(env.limits.K_max))     # toy M3 rows: the toy env's K_max
    cfg["sft"] = {"loss": "marginal", "lr": 1.0e-3, "weight_decay": 1e-3, "batch_queries": 2, "max_epochs": 1,
                  "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0, "eval_epoch0": True}
    a2 = type("A", (), {"run": str(tmp_path / "ft"), "resume": False, "pairs": "p", "rankings": "r",
                        "variant": "full", "init_from": None, "init_ckpt": str(ck), "rft": str(out),
                        "expert_patterns": "version", "keep_unreached": False})()
    assert tr.COMMANDS["sft"](a2, S) == 0
    err = capsys.readouterr().err
    info = json.loads([ln for ln in err.splitlines() if '"rft_rows"' in ln][-1])
    assert info["rft_rows"] == 2 and info["expert_train_queries"] == 1
    assert info["rft_steps"] == sum(len(r["actions"]) for r in rows)
    man = json.loads((tmp_path / "ft" / "manifest.json").read_text())
    assert man["init_ckpt"]["sha256"] == sha256_file(ck) and man["rft"]["sha256"] == sha256_file(out)
    assert man["expert_patterns"] == "version"
    assert all(torch.equal(seen["w0"][k], v) for k, v in pol.state_dict().items())   # epoch 0 = the checkpoint


def test_cli_grpo_builds_the_replay_examples(tmp_path, capsys):
    """train.py grpo with grpo.replay_lambda > 0: the expert rows of replay_patterns (SFT train filter) are replayed
    and the update logs the replay term; with the default the run is the plain GRPO run."""
    from grl.train.sft import save_reference
    tr = _cli()
    env, items = _items(tmp_path, {"p1": "version"})
    pol, g = _pol(0)
    save_reference(pol, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"K_max": int(env.limits.K_max)}))
    from graphrag import config
    cfg = config.load("train")
    cfg["grpo"] = {**GRPO_CFG, "iterations": 2, "replay_lambda": 0.2, "replay_patterns": "version",
                   "replay_per_iter": 1}
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": g, "vec": SerialVecEnv(**toy_init()), "ranks": TOY_RANKS,
         "train": items, "val": items, "w": Weights(), "env": env, "device": torch.device("cpu"), "det": {},
         "dkw": DKW}
    a = type("A", (), {"sft": str(tmp_path), "run": str(tmp_path / "run"), "resume": False, "pairs": "p",
                       "rankings": "r", "variant": "full"})()
    assert tr.COMMANDS["grpo"](a, S) == 0
    err = capsys.readouterr().err
    info = json.loads([ln for ln in err.splitlines() if '"replay_queries"' in ln][-1])
    assert info["replay_queries"] == 1 and info["replay_steps"] > 0
    it = [r for r in read_rows(tmp_path / "run") if r["phase"] == "grpo"]
    assert len(it) == 2 and all("replay_loss" in r for r in it)


# ---- scripts/m7_grpo_x.py --------------------------------------------------------------------------------------------

def _gx():
    from tests.scripts_import import load_script
    return load_script("m7_grpo_x")


def test_gx_jobs_use_the_brief_values_and_never_the_test_split():
    gx = _gx()
    js = {j["name"]: j["steps"] for j in gx.jobs()}
    flat = {n: " ".join(" ".join(a) for _, a in st) for n, st in js.items()}
    assert all("--split test" not in f for f in flat.values())
    for kv in ("grpo.beta_kl=0.1", "grpo.lr=1e-4", "grpo.batch_queries=24", "grpo.iterations=1000",
               "grpo.val_every=50", "grpo.adv_std=false"):
        assert f"--set {kv}" in flat["gx-e2"]
    assert flat["gx-e2"].count("grpo.beta_kl=") == 1 and flat["gx-e2"].count("grpo.lr=") == 1
    assert "grpo.replay_lambda=0.2" in flat["gx-e3"] and "upweight_pattern" not in flat["gx-e3"]
    assert "sft.max_epochs=16" in flat["gx-e0"] and "--sft runs/gx-e0/sft" in flat["gx-e0"]
    assert "--rft-g 32 --rft-margin 0.05" in flat["gx-e1-rft"] and "--split train" in flat["gx-e1-rft"]
    assert "--init-ckpt runs/p9-grpo/checkpoints/grpo-000220.pt" in flat["gx-e1-ft"]
    assert "grpo.iterations=20" in flat["gx-noop"] and flat["gx-noop"].count("grpo.iterations=") == 1


def test_gx_compare_paired_and_pick(tmp_path):
    gx = _gx()
    labels = {"a": {"pattern": "single"}, "b": {"pattern": "version", "provenance": {"m7": {"template": "x"}}},
              "c": {"pattern": "count", "provenance": {"m7": {"template": "count_parts_article"}}}}
    base = {"a": {"reward": 0.5, "recall": 1.0}, "b": {"reward": 0.2, "recall": 0.5}, "c": {"reward": 0.8, "recall": 1.0}}
    oth = {"a": {"reward": 0.6, "recall": 1.0}, "b": {"reward": 0.4, "recall": 1.0}, "c": {"reward": 0.8, "recall": 1.0}}
    c = gx.compare(base, oth, labels)
    assert c["all"]["dR"] == pytest.approx(0.1) and c["Q2"]["dall5"] == 1.0 and c["Q1x"]["dR"] == 0.0
    assert c["hard"]["n"] == 2 and c["old"]["R"] == 0.6
    run = tmp_path / "g"
    (run / "checkpoints").mkdir(parents=True)
    bg = lambda c_, v_: {"count": {"n": 3, "reward_mean": c_}, "version": {"n": 1, "reward_mean": v_},  # noqa: E731
                         "old": {"n": 6, "reward_mean": 0.1}}
    rows = [{"phase": "grpo_val", "step": 0, "val_reward_mean": 0.1, "val_by_group": bg(0, 0)},
            {"phase": "grpo_val", "step": 20, "val_reward_mean": 0.5, "val_by_group": bg(0.5, 0.1)},
            {"phase": "grpo_val", "step": 40, "val_reward_mean": 0.4, "val_by_group": bg(0.5, 0.5)}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        gx.pick_hard(str(run))
    (run / "checkpoints" / "grpo-000040.pt").write_bytes(b"x")
    assert gx.pick_hard(str(run)).endswith("grpo-000040.pt")
