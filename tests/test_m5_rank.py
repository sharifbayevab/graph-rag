"""Pilot 6: the RL reranker as a top-k ranker (grl.model.rank, grl.train.rank, variant "rank") -- metrics edge cases,
the ranking MDP (masked picks without replacement, no STOP), exact Plackett-Luce log-probabilities in static mode,
sampling frequencies, exact KL, listwise SFT examples / SFT / GRPO smoke, baselines, alarms, settings -- toy, CPU."""
import importlib.util
import math

import numpy as np
import pytest
import torch

from graphrag import config
from graphrag.env.mdp import InvalidAction
from grl.model.observe import describe
from grl.model.rank import RankEnv, check_rank_invariants, ranking
from grl.model.rerank import DYN_COLS, RerankPolicy, check_rerank_settings, rerank_settings
from grl.model.variants import VARIANTS, env_spec, is_rank, is_rerank, rank_kw, variant_env_kw
from grl.train.data import load_items
from grl.train.evaluate import evaluate_policy, make_evaluator
from grl.train.grpo import exact_kl, run_grpo
from grl.train.log import Checkpoints, RunLogger, alarms, read_rows
from grl.train.rank import (CE_RANK_NAMES, RANK_BASELINE_NAMES, RankReward, baseline_ranking, first_pick_stats,
                            pool_ceiling, rank_alarm_cfg, rank_metrics, rank_per_query, rank_sft_examples,
                            reward_value, run_rank_references, target_slot0_share)
from grl.train.rollout import SerialVecEnv, rollout
from grl.train.sft import freeze_reference, run_sft
from tests.m3_toy import TOY_CFG, TOY_START, toy_data
from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_graph, toy_pairs
from tests.test_m4_state import _violations

N, K = 5, 3
KW = {"n": N, "fuse_w": 0.7, "norm_top_n": 50, "k_max": K}
RR_CFG = {**SMALL_CFG, "rerank_head": {"cand_dim": 16, "hidden": [32]}}
RW = RankReward("ndcg@20", K)


def _env(**kw):
    return RankEnv(toy_data(), TOY_CFG, TOY_START, **{**KW, **kw})


def _init(**kw):
    return {"data": toy_data(), "ranks": TOY_RANKS, "env_cfg": TOY_CFG, "start_cfg": TOY_START,
            "env_cls": RankEnv, "env_kw": {**KW, **kw}}


def _items(tmp_path):
    f = toy_pairs(tmp_path / "m3.jsonl", p2_split="train")
    return load_items(f, toy_data(), ("train",))            # p1: C* rows {3, 4}; p2: C* row {2}


# ---- 1. metrics -------------------------------------------------------------------------------------------------

def test_rank_metrics_hand_values_and_edge_cases():
    m = rank_metrics([7, 3, 9, 1], {3})
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0 and m["recall@20"] == 1.0
    assert m["mrr"] == 0.5 and m["first_rank"] == 2 and m["ranks"] == [2]
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3)) and m["ndcg@20"] == m["ndcg@10"]
    two = rank_metrics([5, 1, 2, 6], {2, 6, 8})                    # one of three C* fragments outside the list
    assert two["recall@3"] == pytest.approx(1 / 3) and two["recall@5"] == pytest.approx(2 / 3)
    assert two["hit@1"] == 0.0 and two["mrr"] == pytest.approx(1 / 3)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert two["ndcg@10"] == pytest.approx((1 / math.log2(4) + 1 / math.log2(5)) / idcg)
    miss = rank_metrics([1, 2], {9})
    assert miss["hit@20"] == 0.0 and miss["mrr"] == 0.0 and miss["ndcg@20"] == 0.0 and miss["first_rank"] is None
    top = rank_metrics(list(range(25)), {0})
    assert top["hit@1"] == 1.0 and top["ndcg@10"] == 1.0 and top["mrr"] == 1.0
    late = rank_metrics(list(range(25)), {12})                     # beyond 10 but within 20
    assert late["hit@10"] == 0.0 and late["hit@15"] == 1.0 and late["ndcg@10"] == 0.0 and late["ndcg@20"] > 0
    many = rank_metrics(list(range(12)), set(range(12)))           # |C*| > 10: IDCG@10 over 10 positions
    assert many["ndcg@10"] == pytest.approx(1.0) and many["recall@10"] == pytest.approx(10 / 12)
    with pytest.raises(ValueError):
        rank_metrics([1, 2], set())
    with pytest.raises(ValueError):
        rank_metrics([1, 1], {1})
    assert reward_value(m, "mix") == pytest.approx((1 + 1 + 1 + 0.5) / 4)
    with pytest.raises(ValueError):
        RankReward("f1")


# ---- 2. the ranking MDP -----------------------------------------------------------------------------------------

def test_rank_mask_no_stop_picks_without_replacement_and_invariants():
    rng = np.random.default_rng(0)
    for k in (1, 3, 5):
        env = _env(k_max=k)
        for key in ("toy|q", "toy|q2"):
            q = env.query(TOY_RANKS.get(key))
            for _ in range(20):
                s = env.reset(q)
                while not s.done:
                    m = env.mask(q, s)
                    assert m.any() and not m[0]                     # STOP never valid while a candidate remains
                    picked = {env.cand_rows(q).tolist().index(c) + 1 for c in s.context}
                    assert not (set(np.flatnonzero(m).tolist()) & picked)   # placed candidates are masked
                    s = env.step(q, s, int(rng.choice(np.flatnonzero(m))))
                assert s.reason == "ranked" and len(s.context) == k == env.list_len(q)
                assert check_rank_invariants(env, q, s) == []
                assert ranking(s) == s.context
    env = _env()
    q = env.query(TOY_RANKS.get("toy|q"))
    s = env.step(q, env.reset(q), 1)
    for a in (0, 1):                                                # STOP / repeat refused even with check=False
        with pytest.raises(InvalidAction):
            env.step(q, s, a, check=False)
    with pytest.raises(ValueError):
        _env(k_max=N + 1)                                           # the pool must hold the list


def test_empty_pool_has_only_stop():
    env = RankEnv(toy_data(), {**TOY_CFG, "L_max": 40}, TOY_START, **{**KW, "k_max": 1})   # no fragment fits
    q = env.query(TOY_RANKS.get("toy|q"))
    assert env.cand_rows(q).size == 0
    s = env.reset(q)
    m = env.mask(q, s)
    assert m.tolist() == [True] + [False] * N
    s = env.step(q, s, 0)
    assert s.done and s.reason == "stop" and check_rank_invariants(env, q, s) == []


def _policy(seed=0):
    torch.manual_seed(seed)
    return RerankPolicy(toy_graph(), RR_CFG).eval()


def _step_logits(pol, env, q, states, qvec_key):
    from grl.train.rollout import qvecs
    g = toy_graph()
    z, qz = pol.encode(g, qvecs(TOY_RANKS, [qvec_key], "cpu"))
    descs = [describe(env, q, s) for s in states]
    lg, mask = pol.logits(z, qz, [0] * len(descs), descs)
    return lg, mask, pol.log_probs(z, qz, [0] * len(descs), descs)


def test_static_mode_is_an_exact_plackett_luce_and_dynamic_mode_is_not():
    pol = _policy()
    for static in (True, False):
        env = _env(k_max=N, static=static)
        q = env.query(TOY_RANKS.get("toy|q"))
        states, s = [], env.reset(q)
        for a in (2, 4, 1, 3):                                      # a fixed full ranking (slots 1, 3, 0, 2, then 4)
            states.append(s)
            s = env.step(q, s, a)
        states.append(s)
        with torch.no_grad():
            lg, mask, lp = _step_logits(pol, env, q, states, "toy|q")
        cand = lg[:, 1:]
        if static:
            assert torch.allclose(cand, cand[0].expand_as(cand))    # the same scores at every step
            assert all(d.covered.size == 0 and not d.feats[:, list(DYN_COLS)].any()
                       for d in (describe(env, q, x) for x in states))
            taken = [2, 4, 1, 3, 5]
            sc = cand[0]
            pl, left = 0.0, list(range(N))
            for a in taken:                                         # Plackett-Luce log-likelihood by hand
                j = a - 1
                pl += float(sc[j] - torch.logsumexp(sc[left], 0))
                left.remove(j)
            ours = sum(float(lp[t, a]) for t, a in enumerate(taken))
            assert ours == pytest.approx(pl, abs=1e-5)
            assert float(lp[-1, 5]) == pytest.approx(0.0, abs=1e-6)  # one candidate left: probability 1
        else:
            assert any(d.feats[:, list(DYN_COLS)].any() for d in (describe(env, q, x) for x in states[1:]))
        assert torch.isinf(lp[~mask]).all() and (lp[~mask] < 0).all()  # placed slots and STOP: probability 0


def test_sampling_is_masked_without_replacement_with_pl_first_pick_frequencies():
    pol = _policy(1)
    vec = SerialVecEnv(**_init(k_max=N))
    g = toy_graph()
    gen = torch.Generator().manual_seed(0)
    G = 400
    eps, finals = rollout(pol, g, vec, TOY_RANKS, ["toy|q"], G, greedy=False, gen=gen)
    for e, f in zip(eps, finals):
        assert 0 not in e.actions and len(set(e.actions)) == N == len(f.context) and f.reason == "ranked"
    env = _env(k_max=N)
    q = env.query(TOY_RANKS.get("toy|q"))
    with torch.no_grad():
        _, _, lp = _step_logits(pol, env, q, [env.reset(q)], "toy|q")
    p = lp[0, 1:].exp().numpy()
    freq = np.bincount([e.actions[0] - 1 for e in eps], minlength=N) / G
    assert np.abs(freq - p).max() < 0.08                            # 400 draws: ~3.5 SE of the largest p
    for e in eps:                                                   # behaviour log-probs = the PL log-likelihood
        assert sum(e.logp_b) <= 0 and all(np.isfinite(e.logp_b))
    greedy, gf = rollout(pol, g, vec, TOY_RANKS, ["toy|q"], 1, greedy=True)
    order = np.argsort(-lp[0, 1:].numpy(), kind="stable") + 1       # static: greedy = sort by score
    assert greedy[0].actions == order.tolist()


def test_exact_kl_over_masked_rank_steps():
    pol, ref = _policy(2), freeze_reference(_policy(3))
    env = _env(k_max=N)
    q = env.query(TOY_RANKS.get("toy|q"))
    s0 = env.reset(q)
    s1 = env.step(q, s0, 1)
    with torch.no_grad():
        _, mask, lp = _step_logits(pol, env, q, [s0, s1], "toy|q")
        _, _, lr = _step_logits(ref, env, q, [s0, s1], "toy|q")
        _, _, lp_same = _step_logits(ref, env, q, [s0, s1], "toy|q")
    kl = exact_kl(lp, lr, mask)
    hand = [sum(float(lp[t, a].exp() * (lp[t, a] - lr[t, a])) for a in np.flatnonzero(mask[t].numpy()))
            for t in range(2)]
    assert torch.isfinite(kl).all() and kl.tolist() == pytest.approx(hand, abs=1e-6) and (kl >= -1e-7).all()
    assert exact_kl(lp_same, lr, mask).abs().max() < 1e-7


# ---- 3. SFT / GRPO ----------------------------------------------------------------------------------------------

def test_listwise_sft_examples(tmp_path):
    items = _items(tmp_path)
    env = _env()
    exs, cnt = rank_sft_examples(env, TOY_RANKS, items, {})
    by = {}
    for e in exs:
        by.setdefault(e.qkey, []).append(e)
    pool1 = env.cand_rows(env.query(TOY_RANKS.get("p1|q"))).tolist()     # [0, 4, 1, 3, 2], C* {3, 4}
    assert [e.equiv for e in by["p1|q"]] == [tuple(sorted([pool1.index(4) + 1, pool1.index(3) + 1])),
                                             (pool1.index(3) + 1,)]         # then the C* fragment still unplaced
    assert [e.equiv for e in by["p2|q"]] == [(1,)]                          # C* row 2 = slot 0 of its pool
    assert cnt["no_target_in_pool"] == 0 and cnt["multi"] == 1 and cnt["examples"] == 3
    exs2, cnt2 = rank_sft_examples(env, TOY_RANKS, items, {}, neg_prefix=2)
    p2 = [e for e in exs2 if e.qkey == "p2|q"]
    assert len(p2) == 3 and all(e.equiv == (1,) for e in p2) and [e.desc.mask[1] for e in p2] == [True] * 3
    assert all(not e.desc.feats[:, list(DYN_COLS)].any() for e in exs2)     # static: no context features
    small = RankEnv(toy_data(), TOY_CFG, TOY_START, **{**KW, "n": 2, "k_max": 2})   # pool [0, 4] / [2, 3]
    _, c3 = rank_sft_examples(small, TOY_RANKS, items, {})
    assert c3["no_target_in_pool"] == 0 and c3["multi"] == 0
    tiny = RankEnv(toy_data(), TOY_CFG, TOY_START, **{**KW, "n": 1, "k_max": 1})    # p1 pool [0]: C* outside
    _, c4 = rank_sft_examples(tiny, TOY_RANKS, items, {})
    assert c4["no_target_in_pool"] == 1


SFT_CFG = {"loss": "marginal", "lr": 3.0e-3, "weight_decay": 0.0, "batch_queries": 2, "max_epochs": 60,
           "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0}


def test_sft_puts_the_target_first_and_eval_reports_ranking_metrics(tmp_path):
    items = _items(tmp_path)
    env = _env()
    exs, _ = rank_sft_examples(env, TOY_RANKS, items, {})
    pol = _policy()
    pol.train()
    g = toy_graph()
    vec = SerialVecEnv(**_init())
    ev = make_evaluator(g, vec, TOY_RANKS, {"train": items, "val": items}, RW, 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / "run", {"t": 1})
    run_sft(pol, g, TOY_RANKS, exs, exs, SFT_CFG, lg, Checkpoints(tmp_path / "run" / "checkpoints", 3), "cpu", ev)
    lg.close()
    final = [r for r in read_rows(tmp_path / "run") if r["phase"] == "sft_final"][0]
    assert final["train_step_acc"] == 1.0 and final["train_rank_hit@1"] == 1.0
    s, eps, finals = evaluate_policy(pol, g, vec, TOY_RANKS, items, RW, 10, "cpu", env.actions)
    assert s["rank_hit@1"] == 1.0 and s["reward_mean"] == pytest.approx(s["rank_ndcg@20"])
    assert s["first_slot0_share"] is not None and s["stop_share"] == 0.0
    rows = rank_per_query(items, eps, finals)
    assert [r["qkey"] for r in rows] == [i.qkey for i in items] and all(len(r["ranking"]) == K for r in rows)
    assert all(r["first_rank"] == 1 for r in rows)


GRPO_CFG = {"G_s": 4, "lr": 1.0e-3, "beta_kl": 0.1, "batch_queries": 2, "iterations": 4, "val_every": 2,
            "patience": 100, "temperature": 1.0, "adv_eps": 1.0e-6, "step_penalty": 0.0, "freeze_gnn": False,
            "grad_clip": 1.0, "seed": 0}
ALARMS = {"tmax_share": 0.5, "op_collapse_share": 0.9, "kl_max": 0.5, "stop_share_min": 0.5}


@pytest.mark.parametrize("static", [True, False])
def test_grpo_smoke_with_a_ranking_reward(tmp_path, static):
    items = _items(tmp_path)
    env = _env(static=static)
    pol = _policy()
    ref = freeze_reference(pol)
    g = toy_graph()
    vec = SerialVecEnv(**_init(static=static))
    ev = make_evaluator(g, vec, TOY_RANKS, {"train": items, "val": items}, RW, 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / "grpo", {"t": 1})
    acfg = rank_alarm_cfg(ALARMS, target_slot0_share(env, TOY_RANKS, items))
    res = run_grpo(pol, ref, g, vec, TOY_RANKS, items, GRPO_CFG, RW, 10, lg, Checkpoints(tmp_path / "grpo" / "ck", 3),
                   "cpu", ev, env.actions, acfg)
    lg.close()
    rows = [r for r in read_rows(tmp_path / "grpo") if r["phase"] == "grpo"]
    assert res.iterations == 4 and len(rows) == 4
    assert rows[0]["kl_ref"] == pytest.approx(0.0, abs=1e-6)
    for r in rows:
        assert np.isfinite(r["loss"]) and r["kl_ref"] >= -1e-9 and r["ep_len_mean"] == K
        assert 0 <= r["reward_mean"] <= 1 and r["reward_mean"] == pytest.approx(r["rank_ndcg@20"], abs=1e-6)
        assert "stop" not in r["op_hist"] and r["first_slot0_share"] is not None
        assert not any(a.startswith(("t_max", "stop")) for a in r["alarms"])      # switched off for ranking
    vals = [r for r in read_rows(tmp_path / "grpo") if r["phase"] == "grpo_val"]
    assert vals and all("val_rank_hit@10" in r for r in vals)


# ---- 4. baselines, alarms, settings, wiring ---------------------------------------------------------------------

class _It:
    def __init__(self, qkey, target):
        self.qkey, self.gold = qkey, frozenset(target)


def test_rank_baselines_orders_oracle_and_pool_ceiling():
    env = _env()
    q = env.query(TOY_RANKS.get("toy|q"))
    assert baseline_ranking(env, q, "dense") == [4, 0, 1]
    assert baseline_ranking(env, q, "bm25") == [2, 0, 3]
    assert baseline_ranking(env, q, "fused") == env.cand_rows(q).tolist()[:K]
    assert baseline_ranking(env, q, "pool_oracle", frozenset({3})) == [3, 0, 4]
    assert len(set(baseline_ranking(env, q, "rrf"))) == K
    items = [_It("toy|q", {3}), _It("toy|q2", {1}), _It("p1|q", {3, 4})]
    out = run_rank_references(RANK_BASELINE_NAMES, env, TOY_RANKS, items, rows_out=(rows := {}))
    assert out["skipped"] == CE_RANK_NAMES and out["pool_oracle"]["oracle"] is True
    for k in (1, 3):                                                # the ceiling of every reordering of the pool
        assert out["pool_oracle"][f"hit@{k}"] >= out["fused"][f"hit@{k}"]
        assert out["pool_oracle"][f"recall@{k}"] >= out["fused"][f"recall@{k}"]
    assert out["pool_oracle"]["hit@1"] == 1.0 and len(rows["fused"]) == 3
    pc = pool_ceiling(env, TOY_RANKS, items, ns=(1, 2, 5))
    assert pc["1"]["hit"] <= pc["2"]["hit"] <= pc["5"]["hit"] == 1.0 and pc["5"]["recall"] == 1.0
    assert pc["1"]["recall"] == pytest.approx(np.mean([0.0, 0.0, 0.0]))         # slot 0 never holds C* here


def test_rank_alarm_config_and_first_pick_collapse():
    acfg = rank_alarm_cfg(ALARMS, 0.77)
    assert acfg["first_pick_collapse"] == pytest.approx(0.9) and rank_alarm_cfg(ALARMS, 0.9)["first_pick_collapse"] \
        == pytest.approx(0.95)
    m = {"tmax_share": 1.0, "stop_share": 0.0, "op_hist": {"select:0": 99, "select:1": 1}, "kl_ref": 0.1,
         "first_slot0_share": 0.85}
    assert alarms(m, acfg) == []                                    # t_max / STOP / op-collapse do not apply
    assert any(a.startswith("collapse") for a in alarms({**m, "first_slot0_share": 0.95}, acfg))
    assert any(a.startswith("kl") for a in alarms({**m, "kl_ref": 0.6}, acfg))
    assert alarms({"stop_share": 1.0, "kl_ref": 0.0, "first_slot0_share": 0.99}, ALARMS) == []   # default path

    class E:
        def __init__(self, a):
            self.actions = a
    assert first_pick_stats([E([1, 2]), E([1, 3]), E([2, 1]), E([1])]) == {"first_slot0_share": 0.75,
                                                                           "first_top_share": 0.75}


def test_rank_settings_guard_variant_wiring_and_gold_free_model():
    env = _env()
    rs = rerank_settings(env)
    assert rs["rank_k_max"] == K and rs["rank_static"] is True
    check_rerank_settings(rs, rs, "x")
    with pytest.raises(ValueError, match="rank_k_max"):
        check_rerank_settings(rs, rerank_settings(_env(k_max=2)), "x")
    with pytest.raises(ValueError, match="rank_static"):
        check_rerank_settings(rs, rerank_settings(_env(static=False)), "x")
    assert is_rank("rank") and is_rerank("rank") and not is_rank("rerank") and VARIANTS["rank"]["rank"]
    assert env_spec("rank")[0] is RankEnv
    cfg = {"rerank": {"n": 50}, "dynamic_k": {}, "rank": {"k_max": 20, "static": False}}
    assert rank_kw(cfg) == {"k_max": 20, "static": False}
    assert variant_env_kw("rank", cfg)["k_max"] == 20 and "k_max" not in variant_env_kw("rerank", cfg)
    src = (config.ROOT / "grl" / "model" / "rank.py").read_text(encoding="utf-8")
    assert not _violations(src)


def _cli():
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_refuses_rank_keys_outside_rank_and_the_expert():
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.main(["eval", "--variant", "rerank", "--pairs", "x", "--rankings", "y", "--ckpt", "z",
                  "--set", "rank.k_max=10"])
    with pytest.raises(SystemExit):
        cli.main(["expert", "--variant", "rank", "--pairs", "x", "--rankings", "y", "--out", "o.jsonl"])


def test_rank_loader_takes_trajectory_free_rows_and_keeps_the_guards(tmp_path):
    from graphrag.dataset.finalize import read_pairs, write_pairs
    from dataclasses import replace
    from grl.train.data import SplitGuardError, load_rank_items
    src = toy_pairs(tmp_path / "m3.jsonl", p2_split="val")
    rows = [replace(p, actions=[], action_equiv=[], action_masks=[]) for p in read_pairs(src)]
    f = tmp_path / "pairs.jsonl"
    write_pairs(f, rows)
    d = toy_data()
    with pytest.raises(ValueError):
        load_items(f, d, ("train",))                                # the trajectory loader still needs trajectories
    tr, va = load_rank_items(f, d, ("train",)), load_rank_items(f, d, ("val",))
    assert [i.pid for i in tr] == ["p1"] and [i.pid for i in va] == ["p2"]
    assert tr[0].gold == frozenset({3, 4}) and tr[0].actions == ()
    assert all(i.split in ("train", "val") for i in tr + va)       # the toy file's test row is never loaded
    with pytest.raises(SplitGuardError):
        load_rank_items(f, d, ("test",))
    te = tmp_path / "test_eval.jsonl"
    write_pairs(te, rows)
    with pytest.raises(SplitGuardError):
        load_rank_items(te, d, ("train",))                          # the consensus test-eval file, by name
    marked = [replace(p, provenance={**(p.provenance or {}), "test_eval": True}) for p in rows]
    write_pairs(tmp_path / "m.jsonl", marked)
    with pytest.raises(SplitGuardError):
        load_rank_items(tmp_path / "m.jsonl", d, ("train",))        # ... and by row mark
    q = [replace(rows[0], origin="qreal")]
    write_pairs(tmp_path / "q.jsonl", q)
    with pytest.raises(SplitGuardError):
        load_rank_items(tmp_path / "q.jsonl", d, ("train",))        # Q-real by origin
