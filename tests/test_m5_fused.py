"""M5 fused (hybrid research R1): BM25 + dense convex fusion -- fused_scores, the fused_adaptive variant (FusedEnv),
the rel:cc<w>@p baselines, the oracle_switch ORACLE row and recall@<B>tok (toy, CPU)."""
import numpy as np
import pytest

from graphrag.env.policies import random_policy
from graphrag.env.policies import rollout as m3_rollout
from graphrag.env.reward import Weights
from grl.model.observe import describe
from grl.model.variants import (FUSE_W, AdaptiveEnv, FusedEnv, adaptive_kw, check_variant_invariants, env_spec,
                                fused_scores, is_adaptive, is_fused, model_config, threshold_take, with_scores)
from grl.train.baselines import (DYNAMIC_NAMES, FUSED_NAMES, ORACLE_SWITCH_NAMES, budget_recall, dynamic_context,
                                 reference_context, run_references)
from grl.train.fbeta import grl_reward
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import toy_rank2
from tests.test_m5_dynamic_k import CFG10, ITEMS, RANKS, _cli, _reachable, toy_rank3

W = Weights()


def _q(env, rank):
    return with_scores(env.query(rank), rank)


def test_fused_scores_hand_checked_and_limits():
    env = AdaptiveEnv(toy_data(), CFG10, TOY_START)
    q = _q(env, toy_rank())
    # cos: c0 .8 c1 .6 c2 0 c3 .6 c4 .9 (mm over the dense top: /.9); BM25 mm: c2 1, c0 .75, c3 0; c1, c4 absent -> 0
    rows, sc = fused_scores(q, 0.7)
    assert rows.tolist() == [0, 4, 1, 3, 2]                      # .847 .7 .467 .467 (row asc) .3
    assert sc[0] == pytest.approx(1.0) and sc[-1] == pytest.approx(0.0)
    assert sc[1] == pytest.approx((0.7 - 0.3) / (0.7 * 0.8 / 0.9 + 0.225 - 0.3))
    assert fused_scores(q, 1.0)[0].tolist() == [4, 0, 1, 3, 2]    # dense order
    assert fused_scores(q, 0.0)[0].tolist() == [2, 0, 1, 3, 4]    # BM25 first, absent rows tie at the floor
    assert set(rows.tolist()) >= set(np.asarray(q.lex).tolist()) | set(np.asarray(q.vec).tolist())
    fresh = _q(env, toy_rank())                                   # cache == recomputation; deterministic
    assert [x.tolist() for x in fused_scores(fresh, 0.7)] == [rows.tolist(), sc.tolist()]
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            fused_scores(q, bad)
    with pytest.raises(ValueError):
        fused_scores(q, 0.7, top_n=1)


def test_fused_variant_spec_effects_mask_and_preview():
    assert is_adaptive("fused_adaptive") and is_fused("fused_adaptive") and not is_fused("adaptive")
    cls, kw = env_spec("fused_adaptive")
    assert cls is FusedEnv and "require" not in kw                                    # no hybrid rule
    base = {"encoder": "gat", "graph": {"reverse_edges": True}}
    assert model_config(base, "fused_adaptive") == base                               # the full π_ref loads
    assert adaptive_kw("fused_adaptive", {"fuse_w": 0.85}) == {"normalise": True, "norm_top_n": 50, "fuse_w": 0.85}
    assert adaptive_kw("fused_adaptive", None)["fuse_w"] == FUSE_W == 0.7
    assert "fuse_w" not in adaptive_kw("adaptive", {"fuse_w": 0.5})
    with pytest.raises(ValueError):
        FusedEnv(toy_data(), CFG10, TOY_START, normalise=False)
    with pytest.raises(ValueError):
        FusedEnv(toy_data(), CFG10, TOY_START, fuse_w=2)
    env, ad = FusedEnv(toy_data(), CFG10, TOY_START), AdaptiveEnv(toy_data(), CFG10, TOY_START)
    q, aq = env.query(toy_rank()), ad.query(toy_rank())
    s, as_ = env.reset(q), ad.reset(aq)
    assert [env.effect(q, s, env.actions[a])[0] for a in (1, 2, 3)] == [[0], [0, 4], [0, 4]]   # p .9 .7 .5
    for a in range(4, 28):                                                     # vector + graph = AdaptiveEnv
        assert env.effect(q, s, env.actions[a]) == ad.effect(aq, as_, ad.actions[a])
    assert (env.mask(q, s) == ad.mask(aq, as_)).all()
    d = describe(env, q, s)
    assert d.preview[0].tolist() == sorted({int(env.data.chunk_node[c]) for c in [0]})   # fused ΔC at p = .9
    assert d.preview[1].tolist() == describe(ad, aq, as_).preview[1].tolist()
    s2 = env.step(q, s, 1)
    assert s2.context == (0,)
    rows, sc = fused_scores(q, 0.7)
    assert env.search_delta(q, s2, "lexical", 0.7) == threshold_take(rows, sc, {0}, 0.7, env._budget(s2),
                                                                     env.data.chunk_tokens)


@pytest.mark.parametrize("cfg", [TOY_CFG, CFG10, {**CFG10, "T_max": 3}, {**CFG10, "L_max": 200}])
@pytest.mark.parametrize("w", [0.0, 0.5, 0.7, 1.0])
def test_fused_valid_slots_add_and_invariants_hold(cfg, w):
    env = FusedEnv(toy_data(), cfg, TOY_START, fuse_w=w)
    for rank in (toy_rank(), toy_rank2(), toy_rank3()):
        q = env.query(rank)
        for s in _reachable(env, q, 4):
            m = env.mask(q, s)
            assert m.any() and m[0]                                   # no require: STOP always allowed
            for a in np.flatnonzero(m[1:4]) + 1:                       # a valid fused slot adds ≥ 1, in budget
                s2 = env.step(q, s, int(a))
                ctx = s2.context
                assert s2.history[-1].added and len(ctx) == len(set(ctx)) <= env.limits.K_max
                assert int(env.data.chunk_tokens[list(ctx)].sum()) <= env.limits.L_max
        for seed in range(4):
            fin = m3_rollout(env, q, random_policy(seed))
            assert fin.done and check_variant_invariants(env, q, fin) == []


def test_fused_env_never_reads_gold():
    """The fused search is C*-free: effects depend only on the query ranking (no gold argument anywhere)."""
    env = FusedEnv(toy_data(), CFG10, TOY_START)
    q = env.query(toy_rank())
    before = [env.effect(q, env.reset(q), env.actions[a])[0] for a in (1, 2, 3)]
    for g in ({0}, {4}, {1, 2}):
        run_references(["rel:cc0.7@0.9"], env, RANKS, [type(ITEMS[0])("toy|q", g)], W)
    assert [env.effect(q, env.reset(q), env.actions[a])[0] for a in (1, 2, 3)] == before


def test_fused_baselines_oracle_and_budget_recall():
    assert len(FUSED_NAMES) == 9 and len(ORACLE_SWITCH_NAMES) == 9
    assert set(FUSED_NAMES + ORACLE_SWITCH_NAMES) <= set(DYNAMIC_NAMES)
    env = AdaptiveEnv(toy_data(), CFG10, TOY_START)
    q = _q(env, toy_rank())
    assert dynamic_context(env, q, "rel:cc0.7@0.9") == (0,) and dynamic_context(env, q, "rel:cc0.7@0.7") == (0, 4)
    with pytest.raises(ValueError):
        dynamic_context(env, q, "rel_raw:cc0.7@0.9")                  # normalised only
    with pytest.raises(ValueError):
        dynamic_context(env, q, "oracle_switch:dense|cc0.7@0.9")      # needs C*
    with pytest.raises(ValueError):
        reference_context("oracle_switch:dense|cc0.7@0.9", env, q, frozenset(), W)
    for gold, want in (({4}, (4,)), ({0}, (0,))):                     # dense@.9 = (4,), cc0.7@.9 = (0,)
        assert reference_context("oracle_switch:dense|cc0.7@0.9", env, q, frozenset(gold), W) == want
    names = ["rel:dense@0.9", "rel:cc0.7@0.9", "oracle_switch:dense|cc0.7@0.9"]
    m = run_references(names, env, RANKS, ITEMS, W)
    assert m["oracle_switch:dense|cc0.7@0.9"]["oracle"] is True
    assert "oracle" not in m["rel:cc0.7@0.9"] and "oracle" not in m["rel:dense@0.9"]
    assert m[names[2]]["reward"] >= max(m[names[0]]["reward"], m[names[1]]["reward"]) - 1e-9
    for row in m.values():
        assert row["recall@300tok"] <= row["recall@600tok"] <= row["recall"] + 1e-9
    tok = env.data.chunk_tokens                                       # 100 50 400 80 60
    assert budget_recall(tok, (2, 0), {0}) == {"recall@300tok": 0.0, "recall@600tok": 1.0}   # 400 > 300
    assert budget_recall(tok, (0, 2), {0, 2}) == {"recall@300tok": 0.5, "recall@600tok": 1.0}
    assert budget_recall(tok, (), {0}) == {"recall@300tok": 0.0, "recall@600tok": 0.0}
    r = grl_reward(env.data, q.depth, (4,), {4}, W, env.limits.T_max).total
    assert r > grl_reward(env.data, q.depth, (0,), {4}, W, env.limits.T_max).total


@pytest.mark.parametrize("cmd,variant,key,refused", [
    ("grpo", "adaptive", "dynamic_k.fuse_w=0.5", True), ("baselines", "full", "dynamic_k.fuse_w=0.5", True),
    ("grpo", "fused_adaptive", "dynamic_k.fuse_w=0.85", False),
    ("grpo", "fused_adaptive", "dynamic_k.norm_top_n=20", False)])
def test_fuse_w_set_key_and_sft_refused(cmd, variant, key, refused):
    cli = _cli()
    a = type("A", (), {"set": [key], "variant": variant, "cmd": cmd})()
    if refused:
        with pytest.raises(SystemExit, match="--set"):
            cli.check_set_keys(a)
    else:
        cli.check_set_keys(a)
    with pytest.raises(SystemExit):                                   # oracle labels are fixed-k: SFT refused
        cli.main(["sft", "--pairs", "x", "--rankings", "y", "--run", "r", "--variant", "fused_adaptive"])
