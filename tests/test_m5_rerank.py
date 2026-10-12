"""M5 RL reranker (grl.model.rerank, variant "rerank"): candidate list, pointer mask / effect, cross-encoder feature
and cache (scripts/ce_scores.py with a fake scorer), variant expert + SFT + GRPO smoke, baselines -- toy, CPU."""
import importlib.util
import json

import numpy as np
import pytest
import torch

from graphrag import config
from graphrag.dataset.finalize import Pair, read_pairs, write_pairs
from graphrag.env.mdp import InvalidAction
from graphrag.env.reward import Weights
from grl.model.observe import describe
from grl.model.rerank import (CAND_FEATURES, CE_COLS, DYN_COLS, N_CAND_FEATURES, RerankEnv, RerankPolicy,
                              check_rerank_invariants, check_rerank_settings, load_ce_cache, rerank_candidates,
                              rerank_settings)
from grl.model.variants import (VARIANTS, env_spec, fused_scores, is_rerank, rerank_kw, variant_env_kw,
                                with_scores)
from grl.train import expert as ex
from grl.train.data import (VariantMismatch, check_variant_items, load_items, replay_examples, replay_item,
                            sft_train_filter)
from grl.train.evaluate import make_evaluator
from grl.train.fbeta import GRLWeights
from grl.train.grpo import run_grpo
from grl.train.log import (Checkpoints, RunLogger, alarms, expert_slot0_share, read_rows, rerank_alarm_cfg,
                           slot_top_share)
from grl.train.rerank_baselines import CE_NAMES, RERANK_BASELINE_NAMES, rerank_context, run_rerank_references
from grl.train.rollout import SerialVecEnv, rollout
from grl.train.sft import freeze_reference, load_reference, run_sft
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_graph, toy_pairs, toy_rank2
from tests.test_m4_state import _violations

N = 4
KW = {"n": N, "fuse_w": 0.7, "norm_top_n": 50}
W = GRLWeights()
SPEC_W = Weights()
RR_CFG = {**SMALL_CFG, "rerank_head": {"cand_dim": 16, "hidden": [32]}}


def _env(**kw):
    return RerankEnv(toy_data(), TOY_CFG, TOY_START, **{**KW, **kw})


def _script():
    spec = importlib.util.spec_from_file_location("ce_scores", config.ROOT / "scripts" / "ce_scores.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli():
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_scorer(pairs):
    """Deterministic, order-free toy cross-encoder: a function of the (query, doc) texts only."""
    return [float(sum(map(ord, d)) % 97) / 10.0 - 4.0 + 0.01 * len(qt) for qt, d in pairs]


def _texts():
    d = toy_data()
    return [f"fragment {c} text" + "x" * i for i, c in enumerate(d.chunk_ids)]


def _ce_cache(tmp_path, n=N, keys=("toy|q", "toy|q2", "p1|q", "p2|q")):
    m = _script()
    env = _env(n=n)
    texts = _texts()
    reqs = m.ce_requests(env, TOY_RANKS, keys, lambda c: texts[c])
    lines = m.score_requests(reqs, fake_scorer, lambda c: texts[c], env.data.chunk_ids, 3)
    out = tmp_path / "ce" / "ce_scores.jsonl"
    m.write_cache(out, lines, {"model": "fake", "graph_version": env.data.version, "candidates": {**KW, "n": n}})
    return out, lines


# ---- 1. candidates, mask, effect ----------------------------------------------------------------------------------

def test_candidate_set_is_the_fused_prefix_deterministic_and_budget_aware():
    env = _env()
    q = env.query(toy_rank())
    rows, f = fused_scores(q, 0.7, 50)
    assert env.cand_rows(q).tolist() == rows[:N].tolist() and env.cand_rows(q).tolist() == [0, 4, 1, 3]
    assert _env().cand_rows(_env().query(toy_rank())).tolist() == env.cand_rows(q).tolist()   # deterministic
    small = {**TOY_CFG, "L_max": 300}                                        # fragment 2 (400 tokens) never fits
    e2 = RerankEnv(toy_data(), small, TOY_START, n=5)
    q2 = e2.query(toy_rank())
    assert 2 not in e2.cand_rows(q2).tolist() and len(e2.cand_rows(q2)) == 4
    r, _ = rerank_candidates(with_scores(e2.query(toy_rank()), toy_rank()), 2, 0.7, 50, e2.data.chunk_tokens, 300)
    assert r.tolist() == e2.cand_rows(q2)[:2].tolist()
    assert len(env.actions) == N + 1 and env.actions[0].op == "stop" and env.actions[3].k == 2


def test_pointer_mask_never_all_false_and_random_episodes_hold_the_invariants():
    rng = np.random.default_rng(0)
    for cfg in (TOY_CFG, {**TOY_CFG, "K_max": 2}, {**TOY_CFG, "L_max": 160}, {**TOY_CFG, "T_max": 2}):
        env = RerankEnv(toy_data(), cfg, TOY_START, n=5)
        for rank in (toy_rank(), toy_rank2()):
            q = env.query(rank)
            for _ in range(20):
                s = env.reset(q)
                while not s.done:
                    m = env.mask(q, s)
                    assert m[0] and m.any()                                   # STOP always valid
                    sel = np.flatnonzero(m[1:]) + 1
                    a = int(rng.choice(sel)) if sel.size and rng.random() < 0.8 else 0
                    s = env.step(q, s, a)
                assert check_rerank_invariants(env, q, s) == []
                assert s.reason in ("stop", "k_max", "l_max", "t_max", "exhausted")


def test_selection_effect_budget_and_refusals():
    env = _env()
    q = env.query(toy_rank())
    s = env.reset(q)
    s1 = env.step(q, s, 2)                                                    # slot 1 = fragment 4
    assert s1.context == (4,) and env.data.chunk_node[4] in s1.covered and s1.length == 60
    assert not env.mask(q, s1)[2] and s1.history[-1].added == (4,)
    with pytest.raises(InvalidAction):
        env.step(q, s1, 2)                                                    # already in the context
    with pytest.raises(InvalidAction):
        env.step(q, s1, 2, check=False)                                       # refused even unchecked
    tight = RerankEnv(toy_data(), {**TOY_CFG, "L_max": 160}, TOY_START, n=4)
    qt = tight.query(toy_rank())
    st = tight.step(qt, tight.reset(qt), 1)                                   # fragment 0: 100 tokens
    m = tight.mask(qt, st)
    assert m.tolist() == [True, False, True, True, False]                     # 4 (60) and 1 (50) fit, 3 (80) not
    k2 = RerankEnv(toy_data(), {**TOY_CFG, "K_max": 2}, TOY_START, n=4)
    qk = k2.query(toy_rank())
    end = k2.step(qk, k2.step(qk, k2.reset(qk), 1), 2)
    assert end.done and end.reason == "k_max"
    with pytest.raises(ValueError):
        env.describe_state(q, env.step(q, s1, 0))


def test_describe_dispatch_features_and_policy_log_probs():
    env = _env()
    q = env.query(toy_rank())
    s = env.reset(q)
    d = describe(env, q, s, preview=3, cand_top=64)                           # observe.describe dispatches
    assert d.feats.shape == (N, N_CAND_FEATURES) and d.mask.shape == (N + 1,) and d.cand_node.shape == (N,)
    assert d.cand_node.tolist() == env.data.chunk_node[[0, 4, 1, 3]].tolist()
    assert np.all(d.feats[:, list(CE_COLS)] == 0) and np.all(d.feats[:, list(DYN_COLS)] == 0)
    s1 = env.step(q, s, 1)                                                    # fragment 0 (vertex D1/СТ.1)
    d1 = env.describe_state(q, s1)
    same_node = d1.feats[:, CAND_FEATURES.index("same_node_ctx")]
    assert same_node.tolist() == [1, 0, 1, 0]                                 # fragments 0 and 1 share D1/СТ.1
    assert d1.feats[:, CAND_FEATURES.index("adj_covered")].tolist() == [0, 1, 0, 1]   # RELATED to D2/СТ.5
    torch.manual_seed(0)
    g = toy_graph()
    pol = RerankPolicy(g, RR_CFG).eval()
    z, qz = pol.encode(g, torch.from_numpy(np.stack([q.qvec])))
    with torch.no_grad():
        lp = pol.log_probs(z, qz, [0, 0], [d, d1])
    assert lp.shape == (2, N + 1) and torch.allclose(lp.exp().sum(1), torch.ones(2))
    assert float(lp[1, 1].exp()) == 0.0 and float(lp[1, 0].exp()) > 0


def test_variant_registration_and_kwargs():
    assert is_rerank("rerank") and VARIANTS["rerank"] == {"rerank": True}
    assert env_spec("rerank") == (RerankEnv, {})
    cfg = config.load("train")
    assert cfg["rerank"]["n"] == 20 and cfg["rerank"]["use_ce"] is False
    assert rerank_kw(cfg) == {"n": 20, "fuse_w": cfg["dynamic_k"]["fuse_w"], "norm_top_n": 50}
    assert variant_env_kw("rerank", cfg) == rerank_kw(cfg)
    assert variant_env_kw("rerank", cfg, observe=True) == {**rerank_kw(cfg), "use_ce": False, "ce_cache": None}
    assert variant_env_kw("fused_adaptive", cfg) == {"normalise": True, "norm_top_n": 50, "fuse_w": 0.7}
    assert variant_env_kw("full", cfg) == {}
    with pytest.raises(ValueError):
        rerank_kw({**cfg, "dynamic_k": {**cfg["dynamic_k"], "normalise": False}})
    with pytest.raises(ValueError):
        _env(use_ce=True)                                                     # use_ce needs a cache
    cli = _cli()
    assert not cli._fixed_k_ok("rerank") and cli._fixed_k_ok("full")
    assert cli._policy_cls("rerank") is RerankPolicy and cli._policy_cls("full").__name__ == "GraphPolicy"
    assert cli._model_cfg(cfg, "rerank")["rerank_head"] == cfg["rerank"]["head"]
    assert "rerank_head" not in cli._model_cfg(cfg, "full")
    for argv, ok in ((["--variant", "full", "--set", "rerank.n=10"], False),
                     (["--variant", "rerank", "--set", "rerank.n=10"], True),
                     (["--variant", "rerank", "--set", "dynamic_k.fuse_w=0.5"], True),
                     (["--variant", "rerank", "--set", "dynamic_k.normalise=false"], False)):
        a = type("A", (), {"set": argv[3:4], "variant": argv[1], "cmd": "sft"})()
        if ok:
            cli.check_set_keys(a)
        else:
            with pytest.raises(SystemExit):
                cli.check_set_keys(a)


def test_rerank_code_is_gold_free():
    src = (config.ROOT / "grl" / "model" / "rerank.py").read_text(encoding="utf-8")
    assert not _violations(src)
    import inspect
    assert list(inspect.signature(RerankEnv.describe_state).parameters) == ["self", "q", "s"]
    assert list(inspect.signature(rerank_candidates).parameters) == ["q", "n", "fuse_w", "norm_top_n", "tokens",
                                                                     "L_max"]


# ---- 2. cross-encoder cache and feature switch --------------------------------------------------------------------

def test_ce_scores_script_with_a_fake_scorer(tmp_path):
    m = _script()
    out, lines = _ce_cache(tmp_path)
    env = _env()
    texts = _texts()
    reqs = m.ce_requests(env, TOY_RANKS, ["toy|q2", "toy|q"], lambda c: texts[c])
    assert [r["qkey"] for r in reqs] == ["toy|q", "toy|q2"]
    a = m.score_requests(reqs, fake_scorer, lambda c: texts[c], env.data.chunk_ids, 1)
    b = m.score_requests(reqs, fake_scorer, lambda c: texts[c], env.data.chunk_ids, 64)
    assert a == b                                                             # batch size does not matter
    for r in a:
        q = env.query(TOY_RANKS.get(r["qkey"]))
        assert r["rows"] == env.cand_rows(q).tolist()
        assert r["scores"] == [round(fake_scorer([(q.text, texts[c])])[0], 6) for c in r["rows"]]
        assert r["chunk_ids"] == [env.data.chunk_ids[c] for c in r["rows"]]
    meta = json.loads((out.parent / "ce_scores.meta.json").read_text(encoding="utf-8"))
    assert meta["n_queries"] == 4 and meta["model"] == "fake" and meta["out_sha256"]
    table = load_ce_cache(out, "toy")
    assert table["toy|q"] == dict(zip(lines[2]["rows"], lines[2]["scores"]))   # sorted by qkey: p1, p2, toy|q
    with pytest.raises(ValueError):
        load_ce_cache(out, "other-graph")
    out.write_text(out.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ce_cache(out, "toy")                                             # sha256 differs from meta
    with pytest.raises(ValueError):
        m.score_requests(reqs, lambda p: [0.0], lambda c: texts[c], env.data.chunk_ids, 4)   # wrong count


def test_ce_feature_switch_on_off_and_missing_scores(tmp_path):
    out, _ = _ce_cache(tmp_path)
    off = _env(ce_cache=str(out))
    on = _env(ce_cache=str(out), use_ce=True)
    q_off, q_on = off.query(toy_rank()), on.query(toy_rank())
    d_off, d_on = off.describe_state(q_off, off.reset(q_off)), on.describe_state(q_on, on.reset(q_on))
    assert np.all(d_off.feats[:, list(CE_COLS)] == 0) and d_off.scalars[-1] == 0
    assert d_on.scalars[-1] == 1 and np.any(d_on.feats[:, list(CE_COLS)] != 0)
    other = [i for i in range(N_CAND_FEATURES) if i not in CE_COLS]
    assert np.array_equal(d_off.feats[:, other], d_on.feats[:, other])        # only the ce columns change
    ce = on.ce_scores(q_on)
    best = int(np.argmax(ce))
    assert d_on.feats[best, CAND_FEATURES.index("inv_rank_ce")] == 1.0 and d_on.feats[best, CE_COLS[0]] == 1.0
    assert np.array_equal(off.mask(q_off, off.reset(q_off)), on.mask(q_on, on.reset(q_on)))   # same MDP
    torch.manual_seed(0)
    g = toy_graph()
    pol = RerankPolicy(g, RR_CFG).eval()
    z, qz = pol.encode(g, torch.from_numpy(np.stack([q_on.qvec])))
    assert not torch.allclose(pol.log_probs(z, qz, [0], [d_off]), pol.log_probs(z, qz, [0], [d_on]))
    big = _env(n=5, ce_cache=str(out), use_ce=True)                           # cache scored 4 slots only
    with pytest.raises(ValueError, match="no cross-encoder score"):
        big.describe_state(big.query(toy_rank()), big.reset(big.query(toy_rank())))


# ---- 3. expert, SFT, GRPO -----------------------------------------------------------------------------------------

def _expert_pairs(tmp_path, p2_split="val"):
    src = toy_pairs(tmp_path / "m3.jsonl", p2_split=p2_split)
    env = _env()
    rows = []
    for p in read_pairs(src):
        if p.split in ("train", "val"):
            out, info = ex.expert_pair(env, TOY_RANKS, p, W, SPEC_W, "rerank", KW)
            rows.append(out)
    f = tmp_path / "rerank.jsonl"
    write_pairs(f, rows)
    return f, env


def test_expert_selects_the_target_inside_the_candidates_then_stops(tmp_path):
    env = _env()
    for rank, target in ((toy_rank(), {3, 4}), (toy_rank(), {1}), (toy_rank2(), {2})):
        q = env.query(rank)
        res = ex.expert_search(env, q, frozenset(target), W, 3)
        cands = set(env.cand_rows(q).tolist())
        assert res["exact"] and res["final"].reason == "stop"
        if target <= cands:
            assert set(res["final"].context) == target                        # the target fragments, then STOP
    f, env = _expert_pairs(tmp_path)
    items = load_items(f, env.data, ("train", "val"))
    assert {i.variant for i in items} == {"rerank"} and all(i.reach for i in items)
    assert check_variant_items(items, "rerank", KW, False, ex.reward_spec(W)) == "expert"
    with pytest.raises(VariantMismatch):
        check_variant_items(items, "rerank", {**KW, "n": 5}, False)
    m3 = load_items(tmp_path / "m3.jsonl", env.data, ("train", "val"))
    with pytest.raises(VariantMismatch):
        check_variant_items(m3, "rerank", KW, False)                          # 28-action labels refused
    for it in items:
        _, s = replay_item(env, TOY_RANKS, it)
        assert s.reason == "stop" and it.gold <= s.ctx_set


SFT_CFG = {"loss": "hard", "lr": 3.0e-3, "weight_decay": 0.0, "batch_queries": 2, "max_epochs": 80,
           "patience": 1000, "min_delta": 0.0, "grad_clip": 1.0, "seed": 0}


def _rr_init(**kw):
    return {"data": toy_data(), "ranks": TOY_RANKS, "env_cfg": TOY_CFG, "start_cfg": TOY_START,
            "env_cls": RerankEnv, "env_kw": {**KW, **kw}}


def test_sft_on_toy_reaches_the_toy_expert(tmp_path):
    f, env = _expert_pairs(tmp_path, p2_split="train")
    items = load_items(f, env.data, ("train",))
    exs, expert = replay_examples(env, TOY_RANKS, items, {})
    assert {len(e.desc.mask) for e in exs} == {N + 1}
    torch.manual_seed(0)
    g = toy_graph()
    pol = RerankPolicy(g, RR_CFG)
    ev = make_evaluator(g, SerialVecEnv(**_rr_init()), TOY_RANKS, {"train": items, "val": items}, W, 10, "cpu",
                        env.actions, {"train": expert, "val": expert})
    lg = RunLogger(tmp_path / "run", {"t": 1})
    run_sft(pol, g, TOY_RANKS, exs, exs, SFT_CFG, lg, Checkpoints(tmp_path / "run" / "checkpoints", 3), "cpu", ev)
    lg.close()
    final = [r for r in read_rows(tmp_path / "run") if r["phase"] == "sft_final"][0]
    assert final["train_step_acc"] == 1.0 and final["train_reproduction"] == 1.0
    ref = load_reference(g, RR_CFG, tmp_path / "run" / "pi_ref.json", policy_cls=RerankPolicy)
    assert isinstance(ref, RerankPolicy) and not ref.training
    eps, finals = rollout(ref, g, SerialVecEnv(**_rr_init()), TOY_RANKS, [i.qkey for i in items], 1, greedy=True)
    assert [frozenset(fi.ctx_set) for fi in finals] == [expert[i.qkey] for i in items]


GRPO_CFG = {"G_s": 4, "lr": 1.0e-3, "beta_kl": 0.1, "batch_queries": 2, "iterations": 4, "val_every": 2,
            "patience": 100, "temperature": 1.0, "adv_eps": 1.0e-6, "step_penalty": 0.0, "freeze_gnn": False,
            "grad_clip": 1.0, "seed": 0}
ALARMS = {"tmax_share": 0.5, "op_collapse_share": 0.9, "kl_max": 0.5, "stop_share_min": 0.5}


@pytest.mark.parametrize("use_ce", [False, True])
def test_grpo_smoke_with_the_pointer_head(tmp_path, use_ce):
    f, env = _expert_pairs(tmp_path, p2_split="train")
    items = load_items(f, env.data, ("train",))
    kw = {"use_ce": True, "ce_cache": str(_ce_cache(tmp_path)[0])} if use_ce else {}
    torch.manual_seed(0)
    g = toy_graph()
    pol = RerankPolicy(g, RR_CFG).eval()
    ref = freeze_reference(pol)
    vec = SerialVecEnv(**_rr_init(**kw))
    ev = make_evaluator(g, vec, TOY_RANKS, {"train": items, "val": items}, W, 10, "cpu", env.actions)
    lg = RunLogger(tmp_path / "grpo", {"t": 1})
    acfg = rerank_alarm_cfg(ALARMS, expert_slot0_share(items))
    res = run_grpo(pol, ref, g, vec, TOY_RANKS, items, GRPO_CFG, W, 10, lg, Checkpoints(tmp_path / "grpo" / "ck", 3),
                   "cpu", ev, env.actions, acfg)
    lg.close()
    rows = [r for r in read_rows(tmp_path / "grpo") if r["phase"] == "grpo"]
    assert res.iterations == 4 and len(rows) == 4
    assert all(np.isfinite(r["loss"]) and r["kl_ref"] >= -1e-9 for r in rows)
    assert all(set(r["op_hist"]) <= {"stop"} | {f"select:{j}" for j in range(N)} for r in rows)
    assert rows[0]["kl_ref"] == pytest.approx(0.0, abs=1e-6)                 # θ₀ = π_ref
    assert all(r["expert_slot0_share"] == acfg["expert_slot0_share"] and "slot_top_share" in r
               and r["collapse_threshold"] == acfg["op_collapse_share"] for r in rows)


# ---- 4. baselines -------------------------------------------------------------------------------------------------

class _It:
    def __init__(self, qkey, target):
        self.qkey, self.gold = qkey, frozenset(target)


def test_rerank_baselines_fused_ce_and_oracle(tmp_path):
    from grl.model.variants import AdaptiveEnv
    from grl.train.baselines import dynamic_context
    env = _env(n=5)
    ad = AdaptiveEnv(toy_data(), TOY_CFG, TOY_START)
    for rank in (toy_rank(), toy_rank2()):
        q, qa = env.query(rank), with_scores(ad.query(rank), rank)
        for p in (0.9, 0.7, 0.5):                                             # = rel:cc0.7@p inside the top-N
            assert rerank_context(env, q, f"fused@{p}") == dynamic_context(ad, qa, f"rel:cc0.7@{p}")
        assert rerank_context(env, q, "fused@top1") == (int(env.cand_rows(q)[0]),)
    items = [_It("toy|q", {3}), _It("toy|q2", {2}), _It("p1|q", {3, 4})]
    plain = run_rerank_references(RERANK_BASELINE_NAMES, env, TOY_RANKS, items, W)
    assert plain["skipped"] == CE_NAMES and "ce@top1" not in plain and plain["oracle@N"]["oracle"] is True
    out, _ = _ce_cache(tmp_path, n=5)
    cenv = _env(n=5, ce_cache=str(out))
    res = run_rerank_references(RERANK_BASELINE_NAMES, cenv, TOY_RANKS, items, W)
    assert res["skipped"] == [] and res["n"] == 3
    rows = {k: v for k, v in res.items() if isinstance(v, dict) and "reward" in v}
    assert set(rows) == set(RERANK_BASELINE_NAMES)
    assert all(rows["oracle@N"]["reward"] >= v["reward"] - 1e-9 for v in rows.values())
    assert all(v["cand_recall"] == 1.0 for v in rows.values())                 # every target is a candidate
    q = cenv.query(toy_rank())
    ce = cenv.ce_scores(q)
    assert rerank_context(cenv, q, "ce@top1") == (int(cenv.cand_rows(q)[int(np.argmax(ce))]),)
    assert rerank_context(cenv, q, "ce+fused:0.0@top1") == rerank_context(cenv, q, "fused@top1")
    assert rerank_context(cenv, q, "ce+fused:1.0@0.5") == rerank_context(cenv, q, "ce@0.5")


# ---- 5. fix round 1: pinned revision, settings guard, collapse alarm, provenance ----------------------------------

def test_ce_revision_is_pinned_and_checked():
    import inspect
    m = _script()
    rev = config.load("train")["rerank"]["ce_revision"]
    assert rev == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert m.check_revision(rev, rev) == rev
    for bad in (None, "0" * 40):
        with pytest.raises(ValueError):
            m.check_revision(bad, rev)
    src = inspect.getsource(m.HFCrossEncoder.__init__)
    assert src.count("revision=revision") == 2 and "check_revision(" in src
    with pytest.raises(ValueError):
        m.HFCrossEncoder("BAAI/bge-reranker-v2-m3", "")                      # unpinned: refused before loading


def test_ce_text_fallbacks_and_atomic_cache(tmp_path):
    from types import SimpleNamespace as NS
    m = _script()
    recs = [NS(chunks=[NS(chunk_id="a", text="A", text_norm="a", index_text=""),
                       NS(chunk_id="b", text="", text_norm="b", index_text="B")])]
    assert m.chunk_texts(recs, "text") == ({"a": "A", "b": "b"}, 1)
    assert m.chunk_texts(recs, "index_text") == ({"a": "a", "b": "B"}, 1)
    out, _ = _ce_cache(tmp_path)
    assert not list(out.parent.glob("*.tmp")) and load_ce_cache(out, "toy")


def test_rerank_settings_guard_both_directions(tmp_path):
    out, _ = _ce_cache(tmp_path)
    on = _env(ce_cache=str(out), use_ce=True)
    st = rerank_settings(on)
    assert st == {"use_ce": True, "n": N, "fuse_w": 0.7, "norm_top_n": 50, "L_max": TOY_CFG["L_max"],
                  "ce_sha256": json.loads((out.parent / "ce_scores.meta.json").read_text())["out_sha256"]}
    assert rerank_settings(_env(ce_cache=str(out)))["ce_sha256"] is None      # feature off: the cache is irrelevant
    check_rerank_settings(st, dict(st), "x")                                  # same settings: accepted
    for k, v in (("use_ce", False), ("n", 5), ("ce_sha256", "f" * 64), ("L_max", 999), ("fuse_w", 0.5)):
        with pytest.raises(ValueError, match=k):
            check_rerank_settings(st, {**st, k: v}, "x")
    with pytest.raises(ValueError):
        check_rerank_settings(None, st, "x")
    cli = _cli()
    run = tmp_path / "sft"
    cli._write_json(run / cli.RERANK_SETTINGS, st)
    (run / "checkpoints").mkdir()
    assert cli._settings_near(run / "pi_ref.pt") == st and cli._settings_near(run / "checkpoints" / "x.pt") == st
    a = type("A", (), {"variant": "rerank", "ckpt": str(run / "pi_ref.pt"), "split": "val"})()
    with pytest.raises(SystemExit, match="use_ce"):
        cli.cmd_eval(a, {"env": _env(ce_cache=str(out))})                   # trained with ce, evaluated without
    with pytest.raises(KeyError):                                             # matching: passes the guard (then
        cli.cmd_eval(a, {"env": on})                                          # needs the rest of setup)
    cli._write_json(run / "pi_ref.json", {"path": "pi_ref.pt", "sha256": "0", "rerank": st, "K_max": TOY_CFG["K_max"]})
    g = type("A", (), {"variant": "rerank", "sft": str(run), "run": str(tmp_path / "g"), "resume": False})()
    S = {"cfg": config.load("train"), "env": _env(n=3, ce_cache=str(out), use_ce=True), "train": []}
    with pytest.raises(SystemExit, match="'n'"):
        cli.cmd_grpo(g, S)
    with pytest.raises(KeyError):                                             # matching settings: past the guard
        cli.cmd_grpo(g, {**S, "env": on})


def test_manifest_records_the_ce_cache_and_settings(tmp_path):
    out, _ = _ce_cache(tmp_path)
    cli = _cli()
    cfg = config.load("train")
    cfg["rerank"]["ce_cache"] = str(out)
    env = _env(ce_cache=str(out), use_ce=True)
    a = type("A", (), {"variant": "rerank", "pairs": str(out), "rankings": str(out)})()
    man = cli._manifest(a, {"cfg": cfg, "env": env, "det": {}, "device": "cpu", "train": [], "val": [], "w": W}, "sft")
    assert man["inputs"]["ce_cache"] == rerank_settings(env)["ce_sha256"]
    assert man["rerank_settings"] == rerank_settings(env)


def test_expert_relative_collapse_alarm():
    hist = {"stop": 100, "select:0": 93, "select:1": 7}
    assert slot_top_share(hist) == pytest.approx(0.93) and slot_top_share({"stop": 3}) is None
    hi = rerank_alarm_cfg(ALARMS, 0.92)                                       # expert itself takes slot 0 92 %
    assert hi["op_collapse_share"] == pytest.approx(0.97) and hi["expert_slot0_share"] == 0.92
    assert not [x for x in alarms({"op_hist": hist}, hi) if x.startswith("collapse")]
    assert [x for x in alarms({"op_hist": {"select:0": 98, "select:1": 2}}, hi) if x.startswith("collapse")]
    lo = rerank_alarm_cfg(ALARMS, 0.5)                                        # floor: op_collapse_share 0.9
    assert lo["op_collapse_share"] == 0.9 and [x for x in alarms({"op_hist": hist}, lo) if x.startswith("collapse")]
    items = [type("I", (), {"actions": (1, 0)})(), type("I", (), {"actions": (2, 1, 0)})()]
    assert expert_slot0_share(items) == pytest.approx(2 / 3)


def test_target_outside_the_candidates_is_unreached_and_dropped_from_sft(tmp_path):
    src = toy_pairs(tmp_path / "m3.jsonl")
    env = _env(n=2)                                                           # toy|q candidates: fragments 0, 4
    p1 = next(p for p in read_pairs(src) if p.id == "p1")                     # C* = fragments 3, 4
    assert not {3, 4} <= set(env.cand_rows(env.query(TOY_RANKS.get("p1|q"))).tolist())
    out, info = ex.expert_pair(env, TOY_RANKS, p1, W, SPEC_W, "rerank", {**KW, "n": 2})
    assert info["reached"] is False and info["recall"] < 1.0
    f = tmp_path / "rr.jsonl"
    write_pairs(f, [out])
    items = load_items(f, env.data, ("train",))
    assert items[0].reach is False
    kept, cnt = sft_train_filter(items)
    assert kept == [] and cnt["unreached"] == 1 and cnt["dropped"] == 1
