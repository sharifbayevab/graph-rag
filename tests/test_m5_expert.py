"""M5 pre-RL: the variant-aware expert (grl.train.expert), the variant SFT gate (grl.train.data.check_variant) and
the gold-free search selector (grl.train.selector) -- toy, CPU."""
import importlib.util
import inspect
import json

import numpy as np
import pytest
import torch

from graphrag import config
from graphrag.dataset.finalize import Pair, read_pairs, write_pairs
from graphrag.env.mdp import Env
from graphrag.env.policies import action_classes
from graphrag.env.reward import Weights
from grl.model.variants import AdaptiveEnv, FusedEnv, adaptive_kw, env_spec
from grl.train import expert as ex
from grl.train import selector as sl
from grl.train.data import (DataMismatch, SplitGuardError, VariantMismatch, check_variant_file, check_variant_items,
                            load_items, replay_examples, replay_item)
from grl.train.fbeta import GRLWeights, grl_reward
from tests.m3_toy import TOY_START, toy_data, toy_rank
from tests.m4_toy import DKW, SMALL_CFG, TOY_RANKS, toy_graph, toy_init, toy_pairs, toy_rank2
from tests.test_m5_dynamic_k import CFG10, toy_rank3

W = GRLWeights()
SPEC_W = Weights()
FA_KW = adaptive_kw("fused_adaptive", None)                      # {"normalise", "norm_top_n", "fuse_w"} defaults
ENVS = {"fused_adaptive": FusedEnv, "adaptive": AdaptiveEnv, "full": Env}
GOLDS = ({1}, {2}, {4}, {3, 4})


def _env(variant):
    return ENVS[variant](toy_data(), CFG10, TOY_START)


def _cli():
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brute_force(env, q, gold, w, depth, prune_zero):
    """Every valid action sequence up to `depth` retrieval steps (no merging), STOP-completed; the same order."""
    best = None

    def R(s):
        return grl_reward(env.data, q.depth, s.context, gold, w, env.limits.T_max).total

    def visit(s, d):
        nonlocal best
        if s.done:
            return
        if env.mask(q, s)[0]:
            f = env.step(q, s, 0)
            k = (-R(f), len(f.history), tuple(h.action for h in f.history))
            if best is None or k < best:
                best = k
        if d == depth:
            return
        for a in np.flatnonzero(env.mask(q, s)).tolist():
            if a:
                c = env.step(q, s, a)
                if prune_zero and not c.history[-1].added:
                    continue
                visit(c, d + 1)
    visit(env.reset(q), 0)
    return best


# ---- 1. the search ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(ENVS))
@pytest.mark.parametrize("prune_zero", [True, False])
def test_expert_equals_brute_force_optimum(variant, prune_zero):
    env = _env(variant)
    for rank in (toy_rank(), toy_rank2(), toy_rank3()):
        q = env.query(rank)
        for gold in GOLDS:
            for depth in (1, 2, 3):
                res = ex.expert_search(env, q, frozenset(gold), W, depth, prune_zero)
                f = res["final"]
                want = brute_force(env, q, frozenset(gold), W, depth, prune_zero)
                assert (-res["score"], len(f.history), tuple(h.action for h in f.history)) == want
                assert f.done and f.reason == "stop" and res["exact"] and len(f.history) <= depth + 1


def test_expert_is_deterministic_beam_flagged_and_depth_capped():
    env = _env("fused_adaptive")
    q = env.query(toy_rank())
    a = ex.expert_search(env, q, frozenset({1}), W, 3)
    b = ex.expert_search(_env("fused_adaptive"), _env("fused_adaptive").query(toy_rank()), frozenset({1}), W, 3)
    assert a["final"].history == b["final"].history and a["score"] == b["score"]
    nb = ex.expert_search(env, q, frozenset({1}), W, 3, beam=1)
    assert not nb["exact"] and nb["score"] <= a["score"]
    assert ex.expert_search(env, q, frozenset({1}), W, 99)["depth"] == env.limits.T_max - 1
    with pytest.raises(ValueError):
        ex.expert_search(env, q, frozenset({1}), W, 3, beam=0)


def test_expert_uses_the_variant_search_not_fixed_k():
    """Some toy query / target: the fused expert's optimum is not reachable by the same ids in the fixed-k Env."""
    diff = 0
    for rank in (toy_rank(), toy_rank2(), toy_rank3()):
        for gold in GOLDS:
            fa, full = _env("fused_adaptive"), _env("full")
            a = ex.expert_search(fa, fa.query(rank), frozenset(gold), W, 3)
            b = ex.expert_search(full, full.query(rank), frozenset(gold), W, 3)
            diff += a["final"].context != b["final"].context
    assert diff > 0


# ---- 2. records, equivalence classes, replay ----------------------------------------------------------------------

def _expert_file(tmp_path, variant="fused_adaptive", env_kw=None, name="expert.jsonl", p2_split="val"):
    src = toy_pairs(tmp_path / f"m3-{name}", p2_split=p2_split)
    env = _env(variant)
    rows = []
    for p in read_pairs(src):
        if p.split == "test":
            rows.append(p)
            continue
        out, info = ex.expert_pair(env, TOY_RANKS, p, W, SPEC_W, variant, FA_KW if env_kw is None else env_kw)
        assert info["split"] == p.split and out.provenance["meta"]["variant"] == variant
        rows.append(out)
    f = tmp_path / name
    write_pairs(f, rows)
    return f, src


def _sig(s):
    return s.context, s.covered


def test_equiv_classes_are_computed_in_the_variant_env(tmp_path):
    f, _ = _expert_file(tmp_path)
    env = _env("fused_adaptive")
    for it in load_items(f, env.data, ("train", "val")):
        q = env.query(TOY_RANKS.get(it.qkey))
        s = env.reset(q)
        for t, a in enumerate(it.actions):
            m = np.flatnonzero(env.mask(q, s)).tolist()
            assert list(it.masks[t]) == m
            ref = env.step(q, s, a)
            same = [b for b in m if b == a or (b and a and _sig(env.step(q, s, b)) == _sig(ref))]
            assert list(it.equiv[t]) == same and a == min(same)        # the expert keeps the lowest id of a class
            s = ref
    fa, full = _env("fused_adaptive"), _env("full")
    q, q0 = fa.query(toy_rank()), full.query(toy_rank())
    lex = lambda cls: [c for c in cls if c[0] in (1, 2, 3)]                       # noqa: E731
    assert lex(action_classes(fa, q, fa.reset(q))) == [[1], [2, 3]]             # fused p .9 / .7 = .5 on the toy
    assert lex(action_classes(full, q0, full.reset(q0))) == [[1, 2, 3]]         # fixed k: 3 BM25 rows only


def test_expert_records_round_trip_through_replay(tmp_path):
    f, _ = _expert_file(tmp_path)
    env = _env("fused_adaptive")
    items = load_items(f, env.data, ("train", "val"))
    assert [i.pid for i in items] == ["p1", "p2"] and all(i.variant == "fused_adaptive" for i in items)
    for it in items:
        q, s = replay_item(env, TOY_RANKS, it)
        meta = [p for p in read_pairs(f) if p.id == it.pid][0].provenance["meta"]
        assert meta["expert"]["score"] == grl_reward(env.data, q.depth, s.context, it.gold, W, 10).total
        assert meta["env_kw"] == FA_KW and meta["expert"]["reached"] == it.reach
    ex_, expert = replay_examples(env, TOY_RANKS, items, DKW)
    assert len(ex_) == sum(len(i.actions) for i in items) and set(expert) == {"p1|q", "p2|q"}
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["split"] != "test":
            r["provenance"]["meta"]["expert"]["reached"] = not r["provenance"]["meta"]["expert"]["reached"]
    bad = tmp_path / "flipped.jsonl"
    bad.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(DataMismatch):
        replay_item(env, TOY_RANKS, load_items(bad, env.data, ("train",))[0])


def test_expert_replay_diverges_in_another_env(tmp_path):
    f, _ = _expert_file(tmp_path)
    data = toy_data()
    items = load_items(f, data, ("train", "val"))
    hit = 0
    for other in ("full", "adaptive"):
        for it in items:
            try:
                replay_item(_env(other), TOY_RANKS, it)
            except DataMismatch:
                hit += 1
    assert hit > 0                                                   # classes / masks differ in a fixed-k env


def test_expert_inputs_split_guard(tmp_path, monkeypatch):
    src = toy_pairs(tmp_path / "m3.jsonl")
    assert [p.id for p in ex.expert_inputs(src)] == ["p1", "p2"]           # the test row is dropped
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines()]
    rows[0]["origin"] = "qreal"
    q = tmp_path / "qreal.jsonl"
    q.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SplitGuardError):
        ex.expert_inputs(q)


def test_run_expert_writes_records_and_stats(tmp_path, monkeypatch):
    src = toy_pairs(tmp_path / "m3.jsonl")
    monkeypatch.setattr(ex, "expert_context", lambda rankings, variant, extra, env_cfg=None: {
        "env": ENVS[variant](toy_data(), CFG10, TOY_START, **extra), "ranks": TOY_RANKS})
    cfg = config.load("train")
    st = ex.run_expert(src, "unused", "fused_adaptive", tmp_path / "out" / "expert-fa.jsonl", cfg, workers=1)
    rows = read_pairs(tmp_path / "out" / "expert-fa.jsonl")
    assert [p.id for p in rows] == ["p1", "p2"] and all(p.provenance["meta"]["variant"] == "fused_adaptive"
                                                        for p in rows)
    saved = json.loads((tmp_path / "out" / "expert-fa.stats.json").read_text(encoding="utf-8"))
    assert saved["train"]["n"] == 1 and saved["val"]["n"] == 1 and saved["env_kw"] == FA_KW == st["env_kw"]
    with pytest.raises(ValueError):
        ex.run_expert(src, "unused", "hybrid_adaptive", tmp_path / "x.jsonl", cfg, workers=1)


# ---- 3. the SFT gate -----------------------------------------------------------------------------------------------

def test_sft_gate_by_meta_variant(tmp_path):
    f, m3 = _expert_file(tmp_path)
    assert check_variant_file(f, "fused_adaptive", FA_KW, False) == "expert"
    for variant, kw, ok in (("adaptive", adaptive_kw("adaptive", None), False),
                            ("fused_adaptive", {**FA_KW, "fuse_w": 0.5}, False),
                            ("full", {}, True)):
        with pytest.raises(VariantMismatch):
            check_variant_file(f, variant, kw, ok)
    assert check_variant_file(m3, "full", {}, True) == "m3"
    with pytest.raises(VariantMismatch):
        check_variant_file(m3, "fused_adaptive", FA_KW, False)          # fixed-k labels under fused: refused
    items = load_items(f, toy_data(), ("train", "val"))
    assert check_variant_items(items, "fused_adaptive", FA_KW, False) == "expert"
    mixed = load_items(m3, toy_data(), ("train",)) + load_items(f, toy_data(), ("val",))
    with pytest.raises(VariantMismatch):
        check_variant_items(mixed, "fused_adaptive", FA_KW, False)
    cli = _cli()
    for pairs, variant in ((m3, "fused_adaptive"), (m3, "adaptive"), (f, "adaptive"), (f, "full")):
        with pytest.raises(SystemExit):
            cli.main(["sft", "--pairs", str(pairs), "--rankings", "r", "--run", str(tmp_path / "r"),
                      "--variant", variant])
    with pytest.raises(SystemExit):                                      # fuse_w differs from the labels
        cli.main(["sft", "--pairs", str(f), "--rankings", "r", "--run", str(tmp_path / "r"),
                  "--variant", "fused_adaptive", "--set", "dynamic_k.fuse_w=0.5"])
    for argv in (["expert", "--pairs", "p", "--rankings", "r", "--variant", "fused_adaptive"],     # no --out
                 ["expert", "--pairs", "p", "--rankings", "r", "--variant", "hybrid", "--out", "o"],
                 ["selector", "--pairs", "p", "--rankings", "r", "--variant", "hybrid_adaptive"]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_sft_accepts_matching_expert_labels_under_fused(tmp_path, capsys):
    f, _ = _expert_file(tmp_path, p2_split="val")
    cli = _cli()
    cls, kw = env_spec("fused_adaptive")
    env = cls(toy_data(), CFG10, TOY_START, **kw)
    cfg = config.load("train")
    cfg["sft"] = {"loss": "hard", "lr": 3.0e-3, "weight_decay": 1e-3, "batch_queries": 2, "max_epochs": 3,
                  "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0}
    torch.manual_seed(0)
    vec = SerialVecEnvFused()
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": toy_graph(), "vec": vec, "ranks": TOY_RANKS, "dkw": DKW,
         "train": load_items(f, env.data, ("train",)), "val": load_items(f, env.data, ("val",)), "w": W, "env": env,
         "device": torch.device("cpu"), "det": {}}
    args = {"run": str(tmp_path / "run"), "resume": False, "pairs": str(f), "rankings": "r",
            "variant": "fused_adaptive", "set": []}
    a = type("A", (), args)()
    assert cli.COMMANDS["sft"](a, S) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["epochs"] == 3 and (tmp_path / "run" / "pi_ref.json").exists()
    S["train"] = load_items(toy_pairs(tmp_path / "m3x.jsonl", p2_split="train"), env.data, ("train",))
    with pytest.raises(VariantMismatch):                                 # defence in depth inside cmd_sft
        cli.COMMANDS["sft"](type("A", (), {**args, "run": str(tmp_path / "run2")})(), S)


def SerialVecEnvFused():
    from grl.train.rollout import SerialVecEnv
    cls, kw = env_spec("fused_adaptive")
    return SerialVecEnv(**{**toy_init(), "env_cfg": CFG10, "env_cls": cls, "env_kw": kw})


# ---- 4. the selector ----------------------------------------------------------------------------------------------

class Trap:
    """An item whose gold raises: the feature path must never touch it."""

    def __init__(self, qkey):
        self.qkey = qkey

    @property
    def gold(self):
        raise AssertionError("selector features read C*")


class Item:
    def __init__(self, qkey, gold, actions=(0,), variant=None):
        self.qkey, self.gold, self.actions, self.variant = qkey, frozenset(gold), tuple(actions), variant


KEYS = ["p1|q", "p2|q", "toy|q", "toy|q2"]


def test_selector_features_are_gold_free():
    env = _env("fused_adaptive")
    X, keys = sl.features(env, TOY_RANKS, [Trap(k) for k in KEYS])
    assert keys == KEYS and X.shape == (4, len(sl.FEATURE_NAMES)) and np.isfinite(X).all()
    X2, _ = sl.features(env, TOY_RANKS, [Item(k, {i % 5}) for i, k in enumerate(KEYS)])
    assert (X == X2).all()
    for fn in (sl.query_features, sl.features, sl._channel):
        assert not {"gold", "target", "C_ref"} & set(inspect.signature(fn).parameters)
    src = inspect.getsource(sl.query_features) + inspect.getsource(sl._channel)
    assert ".gold" not in src and "C_ref" not in src and "reward" not in src
    assert X[KEYS.index("p1|q")].tolist() == X[KEYS.index("toy|q")].tolist()     # same ranking -> same features


@pytest.mark.parametrize("label", ["switch", "best1"])
@pytest.mark.parametrize("model", ["logreg", "tree"])
def test_selector_runs_and_is_deterministic(label, model):
    env = _env("fused_adaptive")
    golds = [{1}, {2}, {4}, {3}]
    acts = [(4, 0), (1, 0), (4, 0), (1, 0)]
    train = [Item(k, g, a) for k, g, a in zip(KEYS, golds, acts)]
    val = [Item(k, g, a) for k, g, a in zip(KEYS[::-1], golds, acts)]
    a = sl.run_selector(env, TOY_RANKS, train, val, W, label, model)
    b = sl.run_selector(_env("fused_adaptive"), TOY_RANKS, train, val, W, label, model)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for split in ("train", "val"):
        r = a[split]
        assert r["n"] == 4 and 0 <= r["reward_mean"] <= 1 and 0 <= r["label_acc"] <= 1
        assert r["references"]["oracle:best1"]["reward_mean"] >= r["reward_mean"] - 1e-9
        assert r["references"]["oracle:best1"]["reward_mean"] >= r["references"]["oracle:switch"]["reward_mean"]
    with pytest.raises(ValueError):
        sl.run_selector(env, TOY_RANKS, train, val, W, "nope", model)


def test_selector_with_expert_labels(tmp_path):
    f, _ = _expert_file(tmp_path, p2_split="val")
    env = _env("fused_adaptive")
    tr, va = load_items(f, env.data, ("train",)), load_items(f, env.data, ("val",))
    out = sl.run_selector(env, TOY_RANKS, tr, va, W, "expert", "logreg")
    assert out["classes"] == [tr[0].actions[0]]
    exp = out["val"]["references"]["oracle:expert"]
    assert exp["oracle"] and exp["reward_mean"] >= out["val"]["reward_mean"] - 1e-9
    with pytest.raises(ValueError):                                       # M3 items carry no expert label
        sl.run_selector(env, TOY_RANKS, load_items(toy_pairs(tmp_path / "m3.jsonl", p2_split="train"), env.data,
                                                   ("train",)), va, W, "expert", "logreg")


# ---- fix round 1 ---------------------------------------------------------------------------------------------------

def test_sft_gate_compares_the_reward_the_expert_optimised(tmp_path):
    from grl.train.expert import reward_spec
    f1, _ = _expert_file(tmp_path, name="f1.jsonl")
    f2 = tmp_path / "f2.jsonl"
    src = toy_pairs(tmp_path / "m3-f2.jsonl")
    env, w2 = _env("fused_adaptive"), GRLWeights(chunks_beta=2.0)
    write_pairs(f2, [ex.expert_pair(env, TOY_RANKS, p, w2, SPEC_W, "fused_adaptive", FA_KW)[0]
                     for p in read_pairs(src) if p.split != "test"])
    assert check_variant_file(f1, "fused_adaptive", FA_KW, False, reward_spec(W)) == "expert"
    assert check_variant_file(f2, "fused_adaptive", FA_KW, False, reward_spec(w2)) == "expert"
    for f, run_w in ((f1, w2), (f2, W), (f1, GRLWeights(alpha=0.6))):
        with pytest.raises(VariantMismatch, match="different reward"):
            check_variant_file(f, "fused_adaptive", FA_KW, False, reward_spec(run_w))
    cli = _cli()
    with pytest.raises(SystemExit):                                        # F1 labels, F2 run
        cli.main(["sft", "--pairs", str(f1), "--rankings", "r", "--run", str(tmp_path / "r"),
                  "--variant", "fused_adaptive", "--set", "reward.chunks_beta=2"])
    with pytest.raises(SystemExit):                                        # F2 labels, F1 run
        cli.main(["sft", "--pairs", str(f2), "--rankings", "r", "--run", str(tmp_path / "r"),
                  "--variant", "fused_adaptive"])


def test_sft_train_filter_drops_score0_stop_only_and_unreached(tmp_path):
    from dataclasses import replace as dc_replace

    from grl.train.data import sft_train_filter
    f, m3 = _expert_file(tmp_path, p2_split="train")
    items = load_items(f, toy_data(), ("train",))
    good = items[0]
    crafted = [good, dc_replace(good, pid="z0", score=0.0), dc_replace(good, pid="z1", actions=(0,)),
               dc_replace(good, pid="z2", reach=False)]
    kept, cnt = sft_train_filter(crafted)
    assert [i.pid for i in kept] == [good.pid]
    assert cnt["dropped"] == 3 and cnt["score0"] == 1 and cnt["len1"] == 1 and cnt["unreached"] == 1
    assert cnt["reached_share"] == 0.75 and cnt["score0_share"] == 0.25 and cnt["len1_share"] == 0.25
    kept, cnt = sft_train_filter(crafted, keep=True)
    assert len(kept) == 4 and cnt["dropped"] == 0 and cnt["keep_unreached"]
    m3_items = load_items(m3, toy_data(), ("train",))                     # M3 rows are never dropped
    assert sft_train_filter(m3_items)[1]["dropped"] == 0
    assert ex.sft_drop(0.0, True, 3) and ex.sft_drop(0.5, True, 1) and ex.sft_drop(0.5, False, 2)
    assert not ex.sft_drop(0.5, True, 2)


def test_sft_manifest_records_the_filter(tmp_path, capsys):
    f, _ = _expert_file(tmp_path, p2_split="val")
    cli = _cli()
    cls, kw = env_spec("fused_adaptive")
    env = cls(toy_data(), CFG10, TOY_START, **kw)
    cfg = config.load("train")
    cfg["sft"] = {"loss": "hard", "lr": 3.0e-3, "weight_decay": 1e-3, "batch_queries": 2, "max_epochs": 1,
                  "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0}
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": toy_graph(), "vec": SerialVecEnvFused(), "ranks": TOY_RANKS,
         "dkw": DKW, "train": load_items(f, env.data, ("train",)), "val": load_items(f, env.data, ("val",)), "w": W,
         "env": env, "device": torch.device("cpu"), "det": {}}
    a = type("A", (), {"run": str(tmp_path / "run"), "resume": False, "pairs": str(f), "rankings": "r",
                       "variant": "fused_adaptive", "set": [], "keep_unreached": False})()
    assert cli.COMMANDS["sft"](a, S) == 0
    man = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert man["sft_labels"] == "expert" and man["sft_train_filter"]["n"] == 1
    assert {"reached_share", "score0_share", "len1_share", "dropped"} <= set(man["sft_train_filter"])
    assert man["sft_val_counts"]["dropped"] == 0


def test_state_cap_switches_the_query_to_a_flagged_beam(tmp_path, monkeypatch):
    env = _env("fused_adaptive")
    q = env.query(toy_rank())
    full = ex.expert_search(env, q, frozenset({1}), W, 3)
    capped = ex.expert_search(env, q, frozenset({1}), W, 3, max_states=5, cap_beam=1)
    assert not full["capped"] and full["exact"] and capped["capped"] and not capped["exact"]
    assert capped["beam"] == 1 and capped["score"] <= full["score"] and capped["n_states"] < full["n_states"]
    assert ex.expert_search(env, q, frozenset({1}), W, 3, max_states=10 ** 6)["exact"]
    p = [x for x in read_pairs(toy_pairs(tmp_path / "m3.jsonl")) if x.split == "train"][0]
    out, info = ex.expert_pair(env, TOY_RANKS, p, W, SPEC_W, "fused_adaptive", FA_KW, max_states=5, cap_beam=1)
    assert out.provenance["meta"]["expert"]["search"] == "beam1" and out.provenance["meta"]["expert"]["capped"]
    assert info["capped"]
    monkeypatch.setattr(ex, "expert_context", lambda rankings, variant, extra, env_cfg=None: {
        "env": ENVS[variant](toy_data(), CFG10, TOY_START, **extra), "ranks": TOY_RANKS})
    cfg = config.load("train")
    assert ex.expert_config(cfg) == {"depth": 3, "max_states": 20000, "cap_beam": 256}
    cfg["expert"] = {"depth": 3, "max_states": 5, "cap_beam": 1}
    st = ex.run_expert(tmp_path / "m3.jsonl", "unused", "fused_adaptive", tmp_path / "e.jsonl", cfg, workers=1)
    assert st["train"]["capped_share"] == 1.0 and st["max_states"] == 5
    assert {"score0_share", "len1_share", "reached_share", "drop_share", "seconds_max", "n_states_max"} <= set(
        st["train"])


def test_probe_compares_prune_and_keep_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "expert_context", lambda rankings, variant, extra, env_cfg=None: {
        "env": ENVS[variant](toy_data(), CFG10, TOY_START, **extra), "ranks": TOY_RANKS})
    src = toy_pairs(tmp_path / "m3.jsonl", p2_split="train")
    rep = ex.run_expert(src, "unused", "fused_adaptive", tmp_path / "probe.json", config.load("train"), workers=1,
                        probe_n=1)
    assert rep["probe_n"] == 1 and set(rep["modes"]) == {"prune_zero", "keep_zero"}
    for m in rep["modes"].values():
        assert {"n_states_mean", "n_states_max", "seconds_mean", "seconds_max", "score_mean", "capped_share"} <= set(m)
    assert rep["modes"]["keep_zero"]["score_mean"] >= rep["modes"]["prune_zero"]["score_mean"]   # superset search
    assert rep["modes"]["keep_zero"]["n_states_mean"] >= rep["modes"]["prune_zero"]["n_states_mean"]
    assert 0 <= rep["n_best_differs"] <= 1 and len(rep["per_query"]) == 1
    assert json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))["probe_n"] == 1
    assert not (tmp_path / "probe.stats.json").exists()                  # a probe writes no trajectories


def test_selector_slot_names_follow_the_variant():
    golds, acts = [{1}, {2}, {4}, {3}], [(4, 0), (1, 0), (4, 0), (1, 0)]
    items = [Item(k, g, a) for k, g, a in zip(KEYS, golds, acts)]
    want = {"fused_adaptive": ["cc0.7@0.9", "cc0.7@0.7", "cc0.7@0.5", "dense@0.9", "dense@0.7", "dense@0.5"],
            "adaptive": ["bm25@0.9", "bm25@0.7", "bm25@0.5", "dense@0.9", "dense@0.7", "dense@0.5"],
            "full": ["bm25@k3", "bm25@k5", "bm25@k10", "dense@k3", "dense@k5", "dense@k10"]}
    for variant, names in want.items():
        env = _env(variant)
        assert [sl.slot_name(env, env.actions[a]) for a in range(1, 7)] == names
        allnames = [sl.slot_name(env, a) for a in env.actions]
        assert len(set(allnames)) == len(allnames)                       # no collisions
        out = sl.run_selector(env, TOY_RANKS, items, items, W, "switch", "logreg")
        assert {k for k in out["val"]["references"] if k.startswith("always:")} == {f"always:{n}" for n in names}
