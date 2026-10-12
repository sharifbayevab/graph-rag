"""M6 hooks: ablation variants, fixed baselines, evaluation (IMPLEMENTATION.md §9.3–9.5)."""
import numpy as np
import pytest

from graphrag import config
from graphrag.env.mdp import Env, InvalidAction
from graphrag.env.policies import random_policy
from graphrag.env.policies import rollout as m3_rollout
from graphrag.env.reward import Weights
from grl.model.observe import describe
from grl.model.variants import FilteredEnv, env_spec, model_config
from grl.train.baselines import graph_heuristic_policy, reference_context, topk_context
from grl.train.evaluate import context_metrics
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import toy_env


def test_filtered_env_without_graph_actions():
    env = FilteredEnv(toy_data(), TOY_CFG, TOY_START, ops=["stop", "lexical", "vector"])
    q = env.query(toy_rank())
    s = env.reset(q)
    assert not env.mask(q, s)[7:].any() and env.allowed_types == set()
    final = m3_rollout(env, q, random_policy(0))
    assert all(env.actions[h.action].op != "graph" for h in final.history)
    assert all(c.size == 0 for c in describe(env, q, s).cand)
    with pytest.raises(InvalidAction):
        env.step(q, s, 7)


def test_no_related_masks_exactly_the_three_related_actions():
    env = FilteredEnv(toy_data(), TOY_CFG, TOY_START, drop_types=("RELATED",))
    rel = [i for i, a in enumerate(env.actions) if a.op == "graph" and a.r == "RELATED"]
    assert len(rel) == 3 and not env.allowed[rel].any() and int(env.allowed.sum()) == 25
    assert env.data.graph.type_id("RELATED") not in env.allowed_types
    with pytest.raises(ValueError):
        FilteredEnv(toy_data(), TOY_CFG, TOY_START, drop_types=("NOPE",))


def test_dropped_relation_is_masked_off_and_absent_from_candidates():
    """Task 3 carry-over: the filter reaches BOTH Env.mask and the observed candidates (int relation ids)."""
    full, flt = toy_env(), FilteredEnv(toy_data(), TOY_CFG, TOY_START, drop_types=("RELATED",))
    qa, qb = full.query(toy_rank()), flt.query(toy_rank())
    sa, sb = full.reset(qa), flt.reset(qb)
    r = flt.data.graph.type_id("RELATED")
    rel = [i for i, a in enumerate(flt.actions) if a.op == "graph" and a.r == "RELATED"]
    da, db = describe(full, qa, sa), describe(flt, qb, sb)
    assert full.mask(qa, sa)[rel].all() and da.cand[r].size > 0 and da.cand_n[r] > 0     # live in the full env
    assert not flt.mask(qb, sb)[rel].any() and not db.mask[rel].any()
    assert db.cand[r].size == 0 and db.cand_n[r] == 0 and db.cand_best[r] == 0
    assert np.array_equal(db.mask, full.mask(qa, sa) & flt.allowed)
    assert all(isinstance(t, int) for t in flt.allowed_types)
    assert flt.allowed_types == set(range(len(flt.data.graph.edge_types))) - {r}
    for t in flt.allowed_types:                                                          # other types untouched
        assert np.array_equal(da.cand[t], db.cand[t])
    for a in rel:
        with pytest.raises(InvalidAction):
            flt.step(qb, sb, a)
        with pytest.raises(InvalidAction):
            flt.step(qb, sb, a, check=False)                                             # no bypass of the filter


def test_variant_configs():
    base = config.load("train")["model"]
    assert model_config(base, "full") == base and env_spec("full") == (Env, {})
    assert model_config(base, "no_reverse")["graph"]["reverse_edges"] is False
    assert model_config(base, "no_related")["graph"]["drop_types"] == ["RELATED"]
    assert env_spec("no_related") == (FilteredEnv, {"ops": None, "drop_types": ("RELATED",)})
    assert model_config(base, "no_gnn")["encoder"] == "mlp" and env_spec("no_gnn") == (Env, {})
    assert model_config(base, "no_graph")["encoder"] == "mlp"
    assert env_spec("no_graph")[1]["ops"] == ["stop", "lexical", "vector"]
    assert base["graph"]["reverse_edges"] is True                      # model_config does not mutate the base
    with pytest.raises(KeyError):
        model_config(base, "nope")


def test_fixed_topk_contexts_on_toy():
    env = toy_env()
    q = env.query(toy_rank())
    assert topk_context(env, q, "bm25", 3) == (2, 0, 3)
    assert topk_context(env, q, "dense", 5) == (4, 0, 1, 3)              # toy K_max = 4
    assert topk_context(env, q, "rrf", 3) == (0, 2, 3)


def test_reference_contexts_and_the_beam_upper_bound():
    env, w, gold = toy_env(), Weights(), frozenset({3, 4})
    q = env.query(toy_rank())
    m = {}
    for name in ("bm25@3", "dense@3", "rrf@3", "graph_heuristic", "greedy_oracle", "beam_oracle", "random@0"):
        ctx = reference_context(name, env, q, gold, w)
        m[name] = context_metrics(env.data, q, ctx, gold, w, env.limits.T_max)
        assert 0 <= m[name]["reward"] <= 1 and 0 <= m[name]["recall"] <= 1 and len(ctx) <= env.limits.K_max
    assert m["beam_oracle"]["reward"] >= m["bm25@3"]["reward"]          # lexical(3) + STOP is a beam path
    with pytest.raises(ValueError):
        reference_context("nope@1", env, q, gold, w)


def test_graph_heuristic_follows_its_plan_and_skips_masked_steps():
    env = toy_env()
    q = env.query(toy_rank())
    final = m3_rollout(env, q, graph_heuristic_policy)               # Env.step raises InvalidAction on a masked id
    acts = [env.actions[h.action] for h in final.history]
    assert final.done and (acts[0].op, acts[0].k) == ("lexical", 5) and len(acts) <= 4
    graph = [(a.r, a.m) for a in acts if a.op == "graph"]
    assert graph in ([("REFERS_TO", 3), ("CONTAINS", 3)], [("REFERS_TO", 3)], [("CONTAINS", 3)], [])
    assert not [a for a in acts[1:] if a.op in ("lexical", "vector")]
    flt = FilteredEnv(toy_data(), TOY_CFG, TOY_START, ops=["stop", "lexical", "vector"])
    fq = flt.query(toy_rank())
    ops = [flt.actions[h.action].op for h in m3_rollout(flt, fq, graph_heuristic_policy).history]
    assert ops[0] == "lexical" and "graph" not in ops                # every graph step masked -> skipped


def test_evaluate_policy_summary_and_references_on_toy():
    """Greedy evaluation of a variant policy in its FilteredEnv (no_graph: mlp encoder, no graph actions), the §8.6
    summary keys, reproduction, make_evaluator and run_references."""
    from collections import namedtuple

    import torch

    from grl.model.policy import GraphPolicy
    from grl.train.baselines import run_references
    from grl.train.evaluate import evaluate_policy, make_evaluator
    from grl.train.rollout import SerialVecEnv
    from tests.m4_toy import SMALL_CFG, TOY_RANKS, toy_graph, toy_init

    Item = namedtuple("Item", "qkey gold")
    items = [Item("toy|q", frozenset({3, 4})), Item("toy|q2", frozenset({2}))]
    cls, kw = env_spec("no_graph")
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, model_config(SMALL_CFG, "no_graph")).train()
    vec = SerialVecEnv(**{**toy_init(), "env_cls": cls, "env_kw": kw})
    env, w = cls(toy_data(), TOY_CFG, TOY_START, **kw), Weights()
    s, eps, finals = evaluate_policy(pol, g, vec, TOY_RANKS, items, w, TOY_CFG["T_max"], "cpu", env.actions)
    assert pol.training                                                  # mode restored
    assert s["n"] == 2 and sum(s["reasons"].values()) == 2 and sum(s["len_hist"].values()) == 2
    assert sum(s["op_hist"].values()) == sum(len(e.actions) for e in eps)
    assert not [k for k in s["op_hist"] if k.startswith("graph")]
    assert 0 <= s["recall_mean"] <= 1 and 0 <= s["stop_share"] + s["tmax_share"] <= 1
    for k in ("reward_mean", "reward_std", "r_chunks", "r_coverage", "r_diversity", "r_coherence",
              "precision_mean", "ctx_size_mean", "ep_len_mean"):
        assert k in s
    expert = {"val": {i.qkey: frozenset(f.ctx_set) for i, f in zip(items, finals)}}
    ev = make_evaluator(g, vec, TOY_RANKS, {"val": items}, w, TOY_CFG["T_max"], "cpu", env.actions, expert)
    r = ev(pol, "val")
    want = sum(f.reason == "stop" for f in finals) / 2                   # greedy is deterministic
    assert r["reproduction"] == want and r["reward_mean"] == s["reward_mean"]
    ref = run_references(["bm25@3", "graph_heuristic"], env, TOY_RANKS, items, w)
    assert set(ref) == {"bm25@3", "graph_heuristic"} and 0 <= ref["bm25@3"]["recall"] <= 1


# ── fix round 1: variant-aware frontier, split guards, beam lookahead on a filtered env ──────────────────────────

def _walk(env, q, acts):
    s = env.reset(q)
    for a in acts:
        s = env.step(q, s, a)
    return s


def _allowed_nb(env, rows):
    g, out = env.data.graph, set()
    for t in env.allowed_types:
        out |= set(g.neighbors(np.asarray(sorted(rows), np.int64), t).tolist())
    return out


def test_no_related_frontier_drops_related_only_neighbours():
    from grl.model.variants import check_variant_invariants
    full, flt = toy_env(), FilteredEnv(toy_data(), TOY_CFG, TOY_START, drop_types=("RELATED",))
    qa, qb = full.query(toy_rank()), flt.query(toy_rank())
    fa, fb = _walk(full, qa, [10]), _walk(flt, qb, [10])          # graph(REFERS_TO, m=1): covers D2/СТ.5 only (row 4)
    assert fa.covered == fb.covered == frozenset({4}) and fa.context == fb.context
    v0 = {int(v) for v in qb.v0}
    assert fb.frontier == frozenset((v0 | _allowed_nb(flt, fb.covered)) - fb.covered)
    s = _walk(flt, qb, [1])                                       # lexical 3: covers rows 1, 2, 4
    want = (v0 | _allowed_nb(flt, s.covered)) - s.covered
    assert s.frontier == frozenset(want)
    final = m3_rollout(flt, qb, random_policy(3))
    assert check_variant_invariants(flt, qb, final) == []


def test_related_only_neighbour_never_enters_the_filtered_frontier():
    """A synthetic graph where X is reachable from A only through RELATED and is not in V₀."""
    from tests import m3_toy as T
    from graphrag.env.data import EnvData
    edges = T.EDGES + [("D2/СТ.5", "RELATED", "EXT/k")]                    # EXT/k (row 6): RELATED-only neighbour of 4
    data = EnvData.build(T.NODES, T.CHUNKS, edges, T.CHUNK_PARENT, T.MU_INV, T.TOKENS, T.CHUNK_VECS, T.NODE_X,
                         T.EDGE_TYPES)
    full = Env(data, TOY_CFG, TOY_START)
    flt = FilteredEnv(data, TOY_CFG, TOY_START, drop_types=("RELATED",))
    qa, qb = full.query(toy_rank()), flt.query(toy_rank())
    assert 6 not in {int(v) for v in qb.v0}
    sa, sb = _walk(full, qa, [4]), _walk(flt, qb, [4])                    # vector 3: rows 4, 0, 1 -> covers 4 and 1
    assert 4 in sa.covered and 6 in sa.frontier and 6 not in sb.frontier
    assert sa.covered == sb.covered and sa.context == sb.context


def test_no_graph_frontier_has_no_graph_neighbours():
    from grl.model.variants import check_variant_invariants
    flt = FilteredEnv(toy_data(), TOY_CFG, TOY_START, ops=["stop", "lexical", "vector"])
    q = flt.query(toy_rank())
    v0 = frozenset(int(v) for v in q.v0)
    s = flt.reset(q)
    assert s.frontier == v0
    while not s.done:
        s = flt.step(q, s, int(np.flatnonzero(flt.mask(q, s))[-1]))       # the last valid: never STOP while others
        assert s.frontier == v0 - s.covered                               # no neighbours of S at all
    assert check_variant_invariants(flt, q, s) == []


def test_full_env_frontier_and_invariants_are_unchanged():
    from graphrag.env.mdp import check_invariants
    from grl.model.variants import check_variant_invariants
    full = toy_env()
    every = FilteredEnv(toy_data(), TOY_CFG, TOY_START)                   # nothing filtered: identical to Env
    qa, qb = full.query(toy_rank()), every.query(toy_rank())
    for seed in range(5):
        fa, fb = m3_rollout(full, qa, random_policy(seed)), m3_rollout(every, qb, random_policy(seed))
        assert fa == fb and check_invariants(full, qa, fa) == [] and check_invariants(every, qb, fb) == []
        assert check_variant_invariants(full, qa, fa) == []


def test_reference_and_evaluator_split_guards():
    from collections import namedtuple

    from grl.train.baselines import run_references
    from grl.train.data import SplitGuardError
    from grl.train.evaluate import make_evaluator
    from tests.m4_toy import TOY_RANKS
    Item = namedtuple("Item", "qkey gold")
    with pytest.raises(SplitGuardError):
        run_references(["bm25@3"], toy_env(), TOY_RANKS, [Item("x|cyr", frozenset({1}))], Weights())
    with pytest.raises(ValueError):
        make_evaluator(None, None, TOY_RANKS, {"test": []}, Weights(), 10, "cpu", ())
    ev = make_evaluator(None, None, TOY_RANKS, {"train": [], "val": []}, Weights(), 10, "cpu", ())
    with pytest.raises(ValueError):
        ev(None, "test")


def test_beam_oracle_on_a_filtered_env_stays_inside_the_variant():
    from graphrag.env.policies import BeamSearch
    env, w, gold = FilteredEnv(toy_data(), TOY_CFG, TOY_START, drop_types=("RELATED",)), Weights(), frozenset({3, 4})
    q = env.query(toy_rank())
    ref = reference_context("beam_oracle", env, q, gold, w)
    best = BeamSearch(env, q, gold, w, lookahead=False).best()
    assert ref == tuple(best.context) and all(env.allowed[h.action] for f in [best] for h in f.history)
    for f in BeamSearch(env, q, gold, w, lookahead=False).finals:
        assert all(env.allowed[h.action] for h in f.history)
    with pytest.raises(ValueError):
        reference_context("beam_oracle", env, q, gold, w, lookahead=True)
    full = toy_env()
    fq = full.query(toy_rank())
    assert reference_context("beam_oracle", full, fq, gold, w) == \
        tuple(BeamSearch(full, fq, gold, w).best().context)            # full env: config lookahead unchanged
