"""M5 hybrid variant (user request): STOP masked until a lexical, a vector and a graph step were taken (or that became
impossible), the hybrid heuristic baseline, recall / context-size metrics, GRPO under the hybrid mask (toy, CPU)."""
import numpy as np
import pytest
import torch

from graphrag.env.mdp import InvalidAction
from graphrag.env.policies import random_policy
from graphrag.env.policies import rollout as m3_rollout
from graphrag.env.reward import Weights
from grl.model.observe import describe
from grl.model.variants import FilteredEnv, check_variant_invariants, env_spec, model_config
from grl.train.baselines import best_graph_action, hybrid_heuristic_policy, reference_context
from grl.train.evaluate import context_metrics, summarize
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import toy_env, toy_rank2

REQ = ("lexical", "vector", "graph")
CFG10 = {**TOY_CFG, "K_max": 10}                  # 5 toy fragments fit: episodes are not cut by k_max at once


def hyb(cfg=CFG10, **kw):
    return FilteredEnv(toy_data(), cfg, TOY_START, require=REQ, **kw)


def walk(env, q, acts):
    s = env.reset(q)
    for a in acts:
        s = env.step(q, s, a)
    return s


def test_hybrid_spec_and_model_config():
    cls, kw = env_spec("hybrid")
    assert cls is FilteredEnv and kw == {"ops": None, "drop_types": (), "require": REQ}
    base = {"encoder": "gat", "graph": {"reverse_edges": True}}
    assert model_config(base, "hybrid") == base                         # π_ref of the full SFT loads unchanged
    env = cls(toy_data(), CFG10, TOY_START, **kw)
    assert env.required == REQ and env.allowed.all() and env.all_types
    with pytest.raises(ValueError):
        FilteredEnv(toy_data(), CFG10, TOY_START, require=("nope",))


def test_stop_masked_until_all_three_families_are_used():
    env = hyb()
    q = env.query(toy_rank())
    s = env.reset(q)
    assert not env.mask(q, s)[0] and env.missing_ops(s) == REQ
    with pytest.raises(InvalidAction):
        env.step(q, s, 0)
    with pytest.raises(InvalidAction):
        env.step(q, s, 0, check=False)                                   # no bypass for the oracles' fast path
    s = walk(env, q, [1])                                                # lexical 3
    assert not s.done and not env.mask(q, s)[0] and env.missing_ops(s) == ("vector", "graph")
    s = env.step(q, s, 4)                                                # vector 3
    assert not s.done and not env.mask(q, s)[0] and env.missing_ops(s) == ("graph",)
    g = int(np.flatnonzero(env.mask(q, s) & (env.op_of == "graph"))[0])
    s = env.step(q, s, g)
    assert not s.done and env.missing_ops(s) == () and env.mask(q, s)[0]
    s = env.step(q, s, 0)
    assert s.done and s.reason == "stop" and check_variant_invariants(env, q, s) == []
    full = FilteredEnv(toy_data(), CFG10, TOY_START)                     # without `require` STOP is always valid
    fq = full.query(toy_rank())
    assert full.mask(fq, full.reset(fq))[0]


def test_stop_allowed_when_a_family_is_impossible():
    # (a) no graph action can ever be valid (ops filter): STOP opens after lexical + vector
    env = hyb({**CFG10, "k_grid": [1, 2, 3]}, ops=["stop", "lexical", "vector"])
    q = env.query(toy_rank())
    s = walk(env, q, [1])                                                # lexical 1: vector still missing, valid
    assert env.missing_ops(s) == ("vector", "graph") and not env.mask(q, s)[0]
    s = env.step(q, s, 4)                                                # vector 1
    assert not s.done and env.mask(q, s)[1:7].any()
    assert env.missing_ops(s) == ("graph",) and env.mask(q, s)[0]
    # (b) vector k=10 takes every fragment: lexical can no longer add one -> not required any more
    env = hyb()
    q = env.query(toy_rank())
    s = walk(env, q, [6])
    assert len(s.context) == 5 and not env.mask(q, s)[1:4].any()
    assert env.missing_ops(s) == ("lexical", "graph")
    gm = env.mask(q, s) & (env.op_of == "graph")
    assert not s.done and gm.any() and not env.mask(q, s)[0]            # graph still possible -> STOP masked
    s = env.step(q, s, int(np.flatnonzero(gm)[0]))
    assert not s.done and env.missing_ops(s) == ("lexical",) and env.mask(q, s)[0]   # lexical impossible -> STOP
    # (c) no step-budget escape: STOP stays masked up to the last step; the episode ends by t_max
    env = hyb({**CFG10, "T_max": 2})
    q = env.query(toy_rank())
    s = env.reset(q)
    assert not env.mask(q, s)[0]                                         # 3 missing, 2 steps: still masked
    s = env.step(q, s, 7)                                                # graph(CONTAINS, m=1)
    assert not s.done and not env.mask(q, s)[0]
    s = env.step(q, s, 10)                                               # graph again
    assert s.done and s.reason == "t_max"
    # (d) K_max: the budget is spent by one family -> the episode ends by the M3 limit, not by the mask
    env = hyb(TOY_CFG)                                                   # K_max = 4
    q = env.query(toy_rank())
    s = walk(env, q, [5])                                                # vector 5 fills K_max
    assert s.done and s.reason == "k_max"


def _reachable(env, q, depth):
    seen, frontier, out = set(), [env.reset(q)], []
    for _ in range(depth + 1):
        nxt = []
        for s in frontier:
            key = (s.ctx_set, s.covered, frozenset(env.actions[h.action].op for h in s.history), s.t)
            if s.done or key in seen:
                continue
            seen.add(key)
            out.append(s)
            nxt += [env.step(q, s, int(a)) for a in np.flatnonzero(env.mask(q, s)) if a != 0]
        frontier = nxt
    return out


@pytest.mark.parametrize("cfg", [CFG10, TOY_CFG, {**CFG10, "T_max": 3}, {**CFG10, "L_max": 200}])
def test_mask_is_never_all_false_and_stop_rule_is_exact(cfg):
    seen_stop = set()
    for rank in (toy_rank(), toy_rank2()):
        env = hyb(cfg)
        q = env.query(rank)
        states = _reachable(env, q, 4)
        assert len(states) > 5
        for s in states:
            m = env.mask(q, s)
            assert m.any()
            base = FilteredEnv.__mro__[1].mask(env, q, s)                # the M3 mask (all ops allowed)
            assert np.array_equal(m[1:], base[1:])                       # only STOP is touched
            miss = env.missing_ops(s)
            possible = bool((base & np.isin(env.op_of, miss)).any()) if miss else False
            want_masked = bool(miss) and possible
            assert m[0] == (not want_masked)
            seen_stop.add((bool(m[0]), bool(miss)))
    assert (False, True) in seen_stop                                    # STOP masked somewhere in every cfg
    assert any(allowed for allowed, _ in seen_stop)                      # ... and allowed somewhere


def test_random_episodes_satisfy_the_variant_invariants():
    for cfg in (CFG10, TOY_CFG):
        env = hyb(cfg)
        for rank in (toy_rank(), toy_rank2()):
            q = env.query(rank)
            for seed in range(30):
                f = m3_rollout(env, q, random_policy(seed))
                assert f.done and check_variant_invariants(env, q, f) == []
                if f.reason == "stop":
                    pre = walk(env, q, [h.action for h in f.history[:-1]])
                    assert env.stop_allowed(q, pre)


def test_check_variant_invariants_flags_a_forbidden_stop():
    env = hyb()
    q = env.query(toy_rank())
    plain = FilteredEnv(toy_data(), CFG10, TOY_START)
    pq = plain.query(toy_rank())
    early = walk(plain, pq, [1, 0])                                      # lexical, STOP: fine without `require`
    probs = check_variant_invariants(env, q, early)                      # reported, not raised
    assert probs == ["t=1: STOP before the required families ('vector', 'graph')"]


def test_repeating_used_families_never_unlocks_stop():
    """The step budget is no escape: repeating lexical k=1 (toy lexical ranking 2, 0, 3; vector fragments 4, 1 stay
    available) keeps STOP masked down to the last step before t_max -- also at t=2 of T_max=3, where only one step is
    left for two missing families -- and the episode ends by the M3 t_max limit."""
    for T in (2, 3):
        env = hyb({**CFG10, "T_max": T, "k_grid": [1, 2, 3]})
        q = env.query(toy_rank())
        s = env.reset(q)
        while not s.done:
            m = env.mask(q, s)
            assert m[1] and m[4] and not m[0], (T, s.t)                  # lexical 1 / vector 1 valid: STOP masked
            s = env.step(q, s, 1)                                        # lexical k=1 again and again
        assert s.reason == "t_max" and s.t == T and env.missing_ops(s) == ("vector", "graph")


def test_describe_carries_the_hybrid_mask():
    env = hyb()
    q = env.query(toy_rank())
    s = env.reset(q)
    d = describe(env, q, s)
    assert not d.mask[0] and np.array_equal(d.mask, env.mask(q, s))


# ── hybrid heuristic ────────────────────────────────────────────────────────────────────────────────────────────

def test_hybrid_heuristic_plan_on_toy():
    for rank in (toy_rank(), toy_rank2()):
        env = hyb()
        q = env.query(rank)
        f = m3_rollout(env, q, hybrid_heuristic_policy)                  # Env.step(check=True) raises on a masked id
        acts = [env.actions[h.action] for h in f.history]
        assert [(a.op, a.k) for a in acts[:2]] == [("lexical", 5), ("vector", 5)]
        assert acts[2].op == "graph" and acts[3].op == "stop" and f.reason == "stop" and len(acts) == 4
        assert check_variant_invariants(env, q, f) == []


def test_best_graph_action_picks_the_type_with_the_best_delta_v():
    env = hyb()
    q = env.query(toy_rank())
    s = walk(env, q, [2, 5])
    m = env.mask(q, s)
    a = best_graph_action(env, q, s)
    assert a is not None and m[a] and env.actions[a].op == "graph"
    tops = {}
    for r, name in enumerate(env.data.graph.edge_types):
        if (m & (env.op_of == "graph") & np.array([x.r == name for x in env.actions])).any():
            sc = env.graph_scores(q, s, r)[1]
            tops[name] = float(sc.max()) if sc.size else -np.inf
    best = max(tops.values())
    assert tops[env.actions[a].r] == best
    assert env.actions[a].r == next(n for n in env.data.graph.edge_types if tops.get(n) == best)   # tie -> low id
    no_g = FilteredEnv(toy_data(), CFG10, TOY_START, ops=["stop", "lexical", "vector"])
    nq = no_g.query(toy_rank())
    assert best_graph_action(no_g, nq, no_g.reset(nq)) is None


@pytest.mark.parametrize("make", [
    lambda: toy_env(),                                                     # full M3 env, K_max 4
    lambda: hyb(),
    lambda: hyb(TOY_CFG),
    lambda: hyb({**CFG10, "T_max": 2}),
    lambda: hyb({**CFG10, "L_max": 120}),
    lambda: FilteredEnv(toy_data(), CFG10, TOY_START, ops=["stop", "lexical", "vector"], require=REQ),
    lambda: FilteredEnv(toy_data(), CFG10, TOY_START, ops=["stop", "vector", "graph"], require=REQ),
])
def test_hybrid_heuristic_is_masked_safe(make):
    env = make()
    for rank in (toy_rank(), toy_rank2()):
        q = env.query(rank)
        f = m3_rollout(env, q, hybrid_heuristic_policy)
        assert f.done
        assert check_variant_invariants(env, q, f) == []
        ctx = reference_context("hybrid_heuristic", env, q, frozenset({3}), Weights())
        assert ctx == f.context


def test_oracles_are_refused_under_require():
    env = hyb()
    q = env.query(toy_rank())
    for name in ("greedy_oracle", "beam_oracle"):
        with pytest.raises(ValueError, match="require"):
            reference_context(name, env, q, frozenset({3}), Weights())


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_recall_and_context_size_metrics():
    env = toy_env()
    q = env.query(toy_rank())
    m = context_metrics(env.data, q, (2, 0, 3), frozenset({3, 4}), Weights(), env.limits.T_max)
    assert m["recall"] == pytest.approx(0.5) and m["precision"] == pytest.approx(1 / 3)
    assert m["n_ctx"] == 3 and m["length"] == 400 + 100 + 80
    from grl.train.rollout import Episode
    he = hyb()
    hq = he.query(toy_rank())
    f1 = m3_rollout(he, hq, hybrid_heuristic_policy)
    f2 = walk(env, q, [1, 0])
    eps = [Episode("a", 0, 0, [], [h.action for h in f1.history]), Episode("b", 1, 0, [], [1, 0])]
    golds = {"a": frozenset({3, 4}), "b": frozenset({1, 3})}
    parts = [type("P", (), {"total": 0.5, "chunks": 0, "coverage": 0, "diversity": 0, "coherence": 0})()] * 2
    out = summarize(eps, [f1, f2], parts, golds, env.actions, env.limits.T_max)
    r1 = len(set(f1.ctx_set) & golds["a"]) / 2
    assert out["recall_mean"] == pytest.approx((r1 + 0.5) / 2)          # f2 = (2, 0, 3): recall 1/2
    assert out["ctx_size_mean"] == pytest.approx((len(f1.context) + 3) / 2)
    assert out["ctx_tokens_mean"] == pytest.approx((f1.length + 580) / 2)
    assert out["all_ops_share"] == pytest.approx(0.5)                   # f1 used lexical, vector, graph; f2 not


# ── GRPO under the hybrid mask (π_ref unconstrained) ────────────────────────────────────────────────────────────

def test_grpo_smoke_and_cli_under_the_hybrid_variant(tmp_path, capsys):
    import importlib.util
    import json

    from graphrag import config
    from grl.model.policy import GraphPolicy
    from grl.train.data import load_items
    from grl.train.log import read_rows
    from grl.train.rollout import SerialVecEnv, rollout
    from grl.train.sft import save_reference
    from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_graph, toy_init, toy_pairs
    from tests.test_m5_grpo import GRPO_CFG
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit):                                      # SFT under the hybrid mask is refused
        cli.main(["sft", "--pairs", "x", "--rankings", "y", "--run", str(tmp_path / "s"), "--variant", "hybrid"])
    with pytest.raises(SystemExit):
        cli.main(["eval", "--pairs", "x", "--rankings", "y", "--ckpt", "c", "--variant", "nope"])
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    cls, kw = env_spec("hybrid")
    env = cls(toy_data(), CFG10, TOY_START, **kw)
    items = load_items(f, env.data, ("train",))
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, SMALL_CFG).eval()
    save_reference(pol, tmp_path)                                        # π_ref = an unconstrained SFT policy
    (tmp_path / "manifest.json").write_text(json.dumps({"K_max": int(env.limits.K_max)}))   # SFT records K_max
    vec = SerialVecEnv(**{**toy_init(), "env_cfg": CFG10, "env_cls": cls, "env_kw": kw})
    eps, finals = rollout(pol, g, vec, TOY_RANKS, [it.qkey for it in items], 4, greedy=False,
                          gen=torch.Generator().manual_seed(0))
    for e, fin in zip(eps, finals):
        q = env.query(TOY_RANKS.get(e.qkey))
        assert check_variant_invariants(env, q, fin) == []
        assert all(d.mask.any() for d in e.descs)
        if fin.reason == "stop":
            assert env.stop_allowed(q, walk(env, q, e.actions[:-1]))
    cfg = config.load("train")
    cfg["grpo"] = {**GRPO_CFG, "iterations": 2}
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": g, "vec": vec, "ranks": TOY_RANKS, "train": items, "val": items,
         "w": Weights(), "env": env, "device": torch.device("cpu"), "det": {}}
    a = type("A", (), {"sft": str(tmp_path), "run": str(tmp_path / "run"), "resume": False, "pairs": str(f),
                       "rankings": str(f), "variant": "hybrid"})()
    assert cli.COMMANDS["grpo"](a, S) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rows = read_rows(tmp_path / "run")
    it = [r for r in rows if r["phase"] == "grpo"]
    assert out["iterations"] == 2 and len(it) == 2
    assert all(np.isfinite(r["kl_ref"]) and np.isfinite(r["loss"]) for r in it)
    assert all("all_ops_share" in r and "ctx_tokens_mean" in r and "recall_mean" in r for r in it)
    val = [r for r in rows if r["phase"] == "grpo_val"]
    assert "val_all_ops_share" in val[0] and "val_recall_mean" in val[0]
