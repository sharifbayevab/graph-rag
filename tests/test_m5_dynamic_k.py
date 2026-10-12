"""M5 "dynamic k" (user request): recall curves, dynamic-k baselines, the adaptive-k agent variant (alone and with the
hybrid rule) and the F-beta R_chunks option (toy, CPU)."""
import numpy as np
import pytest

from graphrag.env.mdp import Env, InvalidAction
from graphrag.env.policies import random_policy
from graphrag.env.policies import rollout as m3_rollout
from graphrag.env.reward import Weights, reward
from graphrag.retrieval.executor import Budget, take_fitting
from graphrag.retrieval.rankings import QueryRanking, rrf_rows
from grl.model.observe import describe
from grl.model.variants import (ADAPTIVE_P, AdaptiveEnv, FilteredEnv, check_variant_invariants, env_spec,
                                is_adaptive, model_config, threshold_take, with_scores)
from grl.train.baselines import (DYNAMIC_NAMES, dynamic_context, elbow_cut, recall_curves, reference_context,
                                 rrf_scored, run_references, topk_context)
from grl.train.fbeta import GRLWeights, chunks_label, grl_reward, grl_weights, r_chunks_fbeta
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import toy_rank2

REQ = ("lexical", "vector", "graph")
CFG10 = {**TOY_CFG, "K_max": 10}
W = Weights()


class Item:
    def __init__(self, qkey, target):
        self.qkey, self.gold = qkey, frozenset(target)


class Ranks:
    def __init__(self, *rs):
        self.r = {x.qkey: x for x in rs}

    def get(self, k):
        return self.r[k]


def toy_rank3(qkey: str = "toy|q3") -> QueryRanking:
    """Flat BM25 scores and a long tail: thresholds / elbow differ from fixed k."""
    return QueryRanking(qkey, "toy question 3", np.array([0, 1, 0, 0], np.float32), np.array([1, 3, 4, 0, 2]),
                        np.array([3.0, 2.9, 2.8, 0.5, 0.4]), np.array([2, 3, 0, 1, 4]),
                        np.array([1.0, 0.8, 0.6, 0.0, 0.0]), np.zeros(0, np.int64), np.zeros(0, np.float64))


RANKS = Ranks(toy_rank(), toy_rank2(), toy_rank3())
ITEMS = [Item("toy|q", {2, 0}), Item("toy|q2", {3, 1}), Item("toy|q3", {1, 4, 2})]


def _budget_ok(env, ctx):
    tok = env.data.chunk_tokens
    return (len(ctx) == len(set(ctx)) and len(ctx) <= env.limits.K_max
            and (int(tok[list(ctx)].sum()) if ctx else 0) <= env.limits.L_max)


# ---- 1. recall curves ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [TOY_CFG, CFG10, {**CFG10, "L_max": 200}])
def test_recall_curves_are_monotone_and_prefixes_of_topk(cfg):
    env = Env(toy_data(), cfg, TOY_START)
    cur = recall_curves(env, RANKS, ITEMS, W)
    assert set(cur) == {"bm25", "dense", "rrf"}
    for kind, rows in cur.items():
        assert [r["k"] for r in rows] == list(range(1, env.limits.K_max + 1))
        rec = [r["recall_mean"] for r in rows]
        assert all(b >= a for a, b in zip(rec, rec[1:])), (kind, rec)                    # monotone in k
        assert all(r["n_ctx_mean"] <= r["k"] for r in rows)
    for it in ITEMS:                                    # the k-prefix of the top-K_max context IS the top-k context
        q = env.query(RANKS.get(it.qkey))
        for kind in ("bm25", "dense", "rrf"):
            full = topk_context(env, q, kind, env.limits.K_max)
            assert all(topk_context(env, q, kind, k) == full[:k] for k in range(1, env.limits.K_max + 1))
    one = run_references(["bm25@3"], env, RANKS, ITEMS, W)["bm25@3"]
    assert cur["bm25"][2]["recall_mean"] == one["recall"] and cur["bm25"][2]["reward_mean"] == one["reward"]


# ---- 2. dynamic-k baselines ----------------------------------------------------------------------------------------

def test_rrf_scored_matches_rrf_rows_and_elbow_rule():
    for r in (toy_rank(), toy_rank2(), toy_rank3()):
        rows, sc = rrf_scored(r.lex, r.vec, top=len(r.lex) + len(r.vec))
        assert rows.tolist() == rrf_rows(r.lex, r.vec, top=len(r.lex) + len(r.vec)).tolist()
        assert all(a >= b for a, b in zip(sc, sc[1:]))
    assert elbow_cut([3.0, 2.9, 2.8, 0.5, 0.4]) == 3 and elbow_cut([1.0]) == 1 and elbow_cut([]) == 0
    assert elbow_cut([1.0, 1.0, 1.0]) == 1                                                  # first maximum


def test_dynamic_baselines_values_deterministic_and_budget_safe():
    env = Env(toy_data(), CFG10, TOY_START)
    rank = toy_rank3()
    q = with_scores(env.query(rank), rank)
    assert dynamic_context(env, q, "rel:bm25@0.9") == (1, 3, 4)                            # 3.0, 2.9, 2.8 >= 2.7
    assert dynamic_context(env, q, "rel:bm25@0.5") == (1, 3, 4)
    assert dynamic_context(env, q, "rel:dense@0.7") == (2, 3)                              # cos 1.0, .8 >= .7
    assert dynamic_context(env, q, "elbow:bm25") == (1, 3, 4)                              # gaps .1 .1 2.3 .1
    assert dynamic_context(env, q, "elbow:dense") == (2, 3, 0)                             # gaps .2 .2 .6 0 -> 3
    assert set(dynamic_context(env, q, "agree@10")) == {1, 2, 3, 4, 0}
    for cfg in (TOY_CFG, CFG10, {**CFG10, "L_max": 200}, {**CFG10, "K_max": 1}):
        env = Env(toy_data(), cfg, TOY_START)
        for r in (toy_rank(), toy_rank2(), toy_rank3()):
            q = with_scores(env.query(r), r)
            for name in DYNAMIC_NAMES:
                a = reference_context(name, env, q, frozenset({0}), W)
                b = reference_context(name, env, with_scores(env.query(r), r), frozenset({0}), W)
                assert a == b and len(a) >= 1 and _budget_ok(env, a), (cfg, r.qkey, name, a)
    m = run_references(DYNAMIC_NAMES, Env(toy_data(), CFG10, TOY_START), RANKS, ITEMS, W)
    assert set(m) == set(DYNAMIC_NAMES) and all(0 <= v["recall"] <= 1 for v in m.values())


def test_agreement_without_overlap_falls_back_to_the_rrf_top():
    env = Env(toy_data(), CFG10, TOY_START)
    r = QueryRanking("toy|x", "x", np.array([1, 0, 0, 0], np.float32), np.array([0, 1]), np.array([2.0, 1.0]),
                     np.array([3, 4]), np.array([0.9, 0.8]), np.zeros(0, np.int64), np.zeros(0, np.float64))
    q = with_scores(env.query(r), r)
    assert dynamic_context(env, q, "agree@10") == (int(rrf_rows(q.lex, q.vec)[0]),)


# ---- 3. adaptive-k agent variant -----------------------------------------------------------------------------------

def test_threshold_take_rules():
    tok = np.array([100, 50, 400, 80, 60])
    b = Budget(0, 0, 10, 1000)
    rows, sc = [2, 0, 3], [5.0, 4.0, 1.0]
    assert threshold_take(rows, sc, set(), 0.9, b, tok) == [2]
    assert threshold_take(rows, sc, set(), 0.7, b, tok) == [2, 0]
    assert threshold_take(rows, sc, set(), 0.1, b, tok) == [2, 0, 3]
    assert threshold_take(rows, sc, {2}, 0.9, b, tok) == [0]                    # best REMAINING score sets the bar
    assert threshold_take(rows, sc, set(), 0.1, Budget(0, 0, 10, 300), tok) == [0, 3]   # 2 does not fit: skipped
    assert threshold_take(rows, sc, set(), 0.1, Budget(9, 0, 10, 1000), tok) == [2]     # K_max
    assert threshold_take(rows, sc, {2, 0, 3}, 0.5, b, tok) == []
    assert threshold_take([1, 4], [-1.0, -1.0], set(), 0.5, b, tok) == [1]      # non-positive best: one fragment
    for p in (0.9, 0.7, 0.5):                                                    # prefix of take_fitting, nested in p
        got = threshold_take(rows, sc, set(), p, b, tok)
        assert got == take_fitting(rows, set(), 10, b, tok)[:len(got)]
    with pytest.raises(ValueError):
        threshold_take(rows, sc, set(), 0.0, b, tok)


def test_adaptive_spec_effect_and_preview():
    assert is_adaptive("adaptive") and is_adaptive("hybrid_adaptive") and not is_adaptive("hybrid")
    cls, kw = env_spec("adaptive")
    assert cls is AdaptiveEnv and kw == {"ops": None, "drop_types": ()}
    cls2, kw2 = env_spec("hybrid_adaptive")
    assert cls2 is AdaptiveEnv and kw2["require"] == REQ
    base = {"encoder": "gat", "graph": {"reverse_edges": True}}
    assert model_config(base, "adaptive") == base == model_config(base, "hybrid_adaptive")   # full π_ref loads
    env, full = AdaptiveEnv(toy_data(), CFG10, TOY_START), Env(toy_data(), CFG10, TOY_START)
    assert len(env.actions) == len(full.actions) == 28 and env.p_of == {3: 0.9, 5: 0.7, 10: 0.5}
    q, fq = env.query(toy_rank()), full.query(toy_rank())
    s = env.reset(q)
    assert [env.effect(q, s, env.actions[a])[0] for a in (1, 2, 3)] == [[2], [2, 0], [2, 0]]   # BM25 5, 4, 1
    assert [env.effect(q, s, env.actions[a])[0] for a in (4, 5, 6)] == [[4], [4, 0], [4, 0, 1, 3]]   # cos .9 .8 .6
    assert (env.mask(q, s) == full.mask(fq, full.reset(fq))).all()               # validity is the M3 mask
    for a in range(7, 28):                                                         # graph actions unchanged
        assert env.effect(q, s, env.actions[a]) == full.effect(fq, full.reset(fq), full.actions[a])
    d, fd = describe(env, q, s), describe(full, fq, full.reset(fq))
    assert [p.tolist() for p in d.preview] == [sorted({int(env.data.chunk_node[c]) for c in x})
                                               for x in ([2], [4])]               # ΔC of p = 0.9
    assert d.preview_sim.tolist() == pytest.approx([0.0, 0.9])
    assert [p.tolist() for p in fd.preview] == [[1, 2, 4], [1, 4]]                # M3 preview unchanged
    s2 = env.step(q, s, 2)
    assert s2.context == (2, 0) and check_variant_invariants(env, q, env.step(q, s2, 0)) == []
    with pytest.raises(ValueError):
        AdaptiveEnv(toy_data(), CFG10, TOY_START, p_grid=(0.9, 0.5))
    with pytest.raises(ValueError):
        AdaptiveEnv(toy_data(), CFG10, TOY_START, p_grid=(1.2, 0.7, 0.5))
    assert ADAPTIVE_P == (0.9, 0.7, 0.5)


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


@pytest.mark.parametrize("require", [(), REQ])
@pytest.mark.parametrize("cfg", [TOY_CFG, CFG10, {**CFG10, "T_max": 3}, {**CFG10, "L_max": 200}])
def test_adaptive_takes_one_or_more_within_budget_and_mask_never_empty(cfg, require):
    env = AdaptiveEnv(toy_data(), cfg, TOY_START, require=require)
    ref = FilteredEnv(toy_data(), cfg, TOY_START, require=require)
    for rank in (toy_rank(), toy_rank2(), toy_rank3()):
        q, rq = env.query(rank), ref.query(rank)
        for s in _reachable(env, q, 4):
            m = env.mask(q, s)
            assert m.any(), "all-False mask in a live state"
            assert env.stop_allowed(q, s) == ref.stop_allowed(rq, s)            # same hybrid rule as FilteredEnv
            for a in np.flatnonzero(m[1:7]) + 1:
                s2 = env.step(q, s, int(a))
                added = s2.history[-1].added
                assert len(added) >= 1 and _budget_ok(env, s2.context)
            if require and not env.stop_allowed(q, s):
                assert not m[0]
                with pytest.raises(InvalidAction):
                    env.step(q, s, 0)
        for seed in range(4):
            fin = m3_rollout(env, q, random_policy(seed))
            assert fin.done and check_variant_invariants(env, q, fin) == []
            if require and fin.reason == "stop":
                assert env.missing_ops(fin) == () or env.stop_allowed(q, fin) is True


def test_adaptive_in_baselines_and_lexical_scores_required():
    env = AdaptiveEnv(toy_data(), CFG10, TOY_START)
    q = env.query(toy_rank())
    assert reference_context("graph_heuristic", env, q, frozenset({0}), W)            # lexical slot k=5 -> p=0.7
    reference_context("beam_oracle", env, q, frozenset({0, 2}), W)                     # lookahead forced off
    with pytest.raises(ValueError):
        reference_context("beam_oracle", env, q, frozenset({0, 2}), W, lookahead=True)
    bare = Env(toy_data(), CFG10, TOY_START).query(toy_rank())                         # no BM25 scores attached
    with pytest.raises(ValueError):
        env.effect(bare, env.reset(bare), env.actions[1])


# ---- 4. F-beta R_chunks --------------------------------------------------------------------------------------------

def test_fbeta_equals_f1_at_one_and_favours_recall_at_two():
    d = toy_data()
    depth = np.zeros(len(d.chunk_node), np.int16)
    G = {0, 1, 2}
    w1, w2 = GRLWeights(), GRLWeights(chunks_beta=2.0)
    for C in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3, 4), (3,), (0, 3)]:
        base = reward(d, depth, C, G, W, 10)
        assert grl_reward(d, depth, C, G, w1, 10) == base                          # beta = 1: the spec reward
        assert grl_reward(d, depth, C, G, W, 10) == base                           # plain Weights: unchanged
        tp = len(set(C) & G)
        if tp:
            p, r = tp / len(C), tp / len(G)
            assert r_chunks_fbeta(set(C), G, 1.0) == pytest.approx(2 * p * r / (p + r))
            assert base.chunks == pytest.approx(r_chunks_fbeta(set(C), G, 1.0), abs=1e-9)
    # high recall / low precision beats low recall / high precision under F2, not under F1
    hi_rec, hi_prec = {0, 1, 2, 3, 4}, {0}
    assert r_chunks_fbeta(hi_rec, G, 2.0) > r_chunks_fbeta(hi_prec, G, 2.0)
    assert r_chunks_fbeta({0, 1, 2, 3, 4}, G, 2.0) > r_chunks_fbeta({0, 1, 2, 3, 4}, G, 1.0)   # p < r: F2 > F1
    assert r_chunks_fbeta({0}, G, 2.0) < r_chunks_fbeta({0}, G, 1.0)                           # p > r: F2 < F1
    p2 = grl_reward(d, depth, (0, 1, 2, 3, 4), G, w2, 10)
    p1 = reward(d, depth, (0, 1, 2, 3, 4), G, W, 10)
    assert p2.chunks > p1.chunks and (p2.coverage, p2.diversity, p2.coherence) == (p1.coverage, p1.diversity,
                                                                                    p1.coherence)
    assert p2.total == pytest.approx(p1.total + W.alpha * (p2.chunks - p1.chunks), abs=2e-9)
    cfg = {"reward": {"weights": {"alpha": 0.5, "beta": 0.2, "gamma": 0.2, "delta": 0.1}}}
    assert grl_weights(cfg) == GRLWeights() and chunks_label(grl_weights(cfg)) == "F1"   # default: no key -> F1
    cfg["reward"]["chunks_beta"] = 2
    assert grl_weights(cfg).chunks_beta == 2.0 and chunks_label(grl_weights(cfg)) == "F2"
    cfg["reward"]["chunks_beta"] = 0
    with pytest.raises(ValueError):
        grl_weights(cfg)


def test_train_yaml_default_is_f1_and_settable():
    from graphrag import config
    t = config.load("train")
    assert t["reward"]["chunks_beta"] == 1 and grl_weights(t).chunks_beta == 1.0
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_cli", config.path("scripts/train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._set(t, "reward.chunks_beta=2")
    assert grl_weights(t).chunks_beta == 2.0
    for v in ("adaptive", "hybrid_adaptive"):
        with pytest.raises(SystemExit):
            mod.main(["sft", "--pairs", "x", "--rankings", "y", "--run", "z", "--variant", v])


@pytest.mark.parametrize("variant", ["adaptive", "hybrid_adaptive", "fused_adaptive"])
def test_grpo_smoke_under_adaptive_with_f2(tmp_path, capsys, variant):
    import importlib.util
    import json

    import torch

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
    f = toy_pairs(tmp_path / "pairs.jsonl", p2_split="train")
    cls, kw = env_spec(variant)
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
        assert check_variant_invariants(env, q, fin) == [] and all(d.mask.any() for d in e.descs)
    cfg = config.load("train")
    cfg["grpo"] = {**GRPO_CFG, "iterations": 2}
    cli._set(cfg, "reward.chunks_beta=2")
    S = {"cfg": cfg, "mcfg": SMALL_CFG, "graph": g, "vec": vec, "ranks": TOY_RANKS, "train": items, "val": items,
         "w": grl_weights(cfg), "env": env, "device": torch.device("cpu"), "det": {}}
    a = type("A", (), {"sft": str(tmp_path), "run": str(tmp_path / "run"), "resume": False, "pairs": str(f),
                       "rankings": str(f), "variant": variant})()
    assert cli.COMMANDS["grpo"](a, S) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["iterations"] == 2
    assert all(np.isfinite(r["loss"]) for r in read_rows(tmp_path / "run") if r["phase"] == "grpo")
    man = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert man["reward_chunks"] == "F2" and man["chunks_beta"] == 2.0 and man["variant"] == variant


def test_cli_baselines_curves_writes_json(tmp_path, capsys):
    import importlib.util
    import json

    from graphrag import config
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    S = {"env": Env(toy_data(), CFG10, TOY_START), "ranks": RANKS, "val": ITEMS, "w": GRLWeights(chunks_beta=2.0),
         "cfg": config.load("train")}
    a = type("A", (), {"curves": True, "split": "val", "out": str(tmp_path / "c.json"), "variant": "full"})()
    assert cli.COMMANDS["baselines"](a, S) == 0
    cap = capsys.readouterr()
    out = json.loads(cap.out.strip().splitlines()[-1])
    assert out == json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert out["reward_chunks"] == "F2" and set(out["dynamic"]) == set(DYNAMIC_NAMES)
    assert [r["k"] for r in out["curves"]["rrf"]] == list(range(1, 11)) and "bm25 recall" in cap.err
    with pytest.raises(SystemExit):
        cli.main(["grpo", "--pairs", "x", "--rankings", "y", "--run", "r", "--sft", "s", "--curves"])


# ---- controller ruling: per-query min-max normalisation before thresholds -----------------------------------------

def _narrow_band(n=60):
    """Cosine-like narrow band: 0.75 .. 0.55 over the top 50, then a tail; all fragments small (K_max binds)."""
    sc = np.concatenate([np.linspace(0.75, 0.55, 50), np.linspace(0.54, 0.50, n - 50)])
    return np.arange(n), sc, np.full(n, 10)


def test_normalised_thresholds_do_not_fill_k_max_on_a_narrow_band():
    from grl.model.variants import minmax_scores
    rows, sc, tok = _narrow_band()
    b = Budget(0, 0, 24, 8000)
    assert len(threshold_take(rows, sc, set(), 0.7, b, tok)) == 24                 # raw: fills K_max at once
    assert len(threshold_take(rows, sc, set(), 0.5, b, tok)) == 24
    norm = minmax_scores(sc, 50)
    assert norm[0] == 1.0 and norm[49] == 0.0 and norm[50] < 0
    got = {p: threshold_take(rows, norm, set(), p, b, tok) for p in (0.9, 0.7, 0.5)}
    assert [len(got[p]) for p in (0.9, 0.7, 0.5)] == [5, 15, 24]                   # 10 % / 30 % / 50 % of the band
    assert len(got[0.7]) < 24 and got[0.9] == got[0.7][:5] and got[0.7] == got[0.5][:15]
    # best REMAINING: after the top 10 are in the context, p = 0.9 keeps 0.9 x the normalised score of row 10
    rest = threshold_take(rows, norm, set(range(10)), 0.9, b, tok)
    assert rest[0] == 10 and all(norm[c] >= 0.9 * norm[10] - 1e-12 for c in rest)
    assert norm[rest[-1] + 1] < 0.9 * norm[10]


def test_degenerate_window_takes_one_fragment():
    from grl.model.variants import minmax_scores
    tok = np.full(60, 10)
    flat = np.full(60, 0.7)
    assert (minmax_scores(flat, 50) == 0).all()
    for p in (0.9, 0.7, 0.5):
        assert threshold_take(np.arange(60), minmax_scores(flat, 50), set(), p, Budget(0, 0, 24, 8000), tok) == [0]
    assert (minmax_scores(np.array([np.nan, np.nan]), 50) == 0).all()
    with pytest.raises(ValueError):
        minmax_scores(flat, 1)
    env = AdaptiveEnv(toy_data(), CFG10, TOY_START)                                # a flat BM25 ranking on the toy
    r = QueryRanking("toy|flat", "flat", np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 3]),
                     np.array([2.0, 2.0, 2.0]), np.array([4, 0, 1, 3, 2]), np.array([0.9, 0.8, 0.6, 0.6, 0.0]),
                     np.zeros(0, np.int64), np.zeros(0, np.float64))
    q = env.query(r)
    s = env.reset(q)
    assert [env.effect(q, s, env.actions[a])[0] for a in (1, 2, 3)] == [[0], [0], [0]]
    raw = AdaptiveEnv(toy_data(), CFG10, TOY_START, normalise=False)
    rq = raw.query(r)
    assert raw.effect(rq, raw.reset(rq), raw.actions[1])[0] == [0, 1, 3]            # raw: equal scores all pass
    w = env.query(r)
    assert check_variant_invariants(env, w, m3_rollout(env, w, random_policy(0))) == []


def test_normalised_adaptive_env_preview_and_config():
    from grl.model.variants import adaptive_kw, minmax_scores
    from graphrag import config
    t = config.load("train")
    assert t["dynamic_k"] == {"normalise": True, "norm_top_n": 50, "fuse_w": 0.7}
    assert adaptive_kw("adaptive", t["dynamic_k"]) == {"normalise": True, "norm_top_n": 50}
    assert adaptive_kw("hybrid_adaptive", {"normalise": False}) == {"normalise": False, "norm_top_n": 50}
    assert adaptive_kw("hybrid", t["dynamic_k"]) == {} and adaptive_kw("full", None) == {}
    env = AdaptiveEnv(toy_data(), CFG10, TOY_START)
    assert env.normalise and env.norm_top_n == 50
    rank = toy_rank3()
    q = env.query(rank)
    s = env.step(q, env.reset(q), 4)                                                  # vector p = 0.9 -> [2]
    assert s.context == (2,)
    sc = minmax_scores(q.lex_score, 50)
    assert (env.channel_scores(q, "lexical") == sc).all()
    for op in ("lexical", "vector"):
        want = env.search_delta(q, s, op, 0.9)[:3]
        d = describe(env, q, s)
        i = 0 if op == "lexical" else 1
        assert d.preview[i].tolist() == sorted({int(env.data.chunk_node[c]) for c in want})
    with pytest.raises(ValueError):
        AdaptiveEnv(toy_data(), CFG10, TOY_START, norm_top_n=1)


def test_raw_and_normalised_baselines_both_reported():
    env = Env(toy_data(), CFG10, TOY_START)
    names = [n for n in DYNAMIC_NAMES if n.startswith(("rel:", "rel_raw:")) and ":cc" not in n]   # cc: fused
    assert len(names) == 18
    m = run_references(names, env, RANKS, ITEMS, W, norm_top_n=50)
    assert set(m) == set(names)
    rank = toy_rank()
    q = with_scores(env.query(rank), rank)
    assert dynamic_context(env, q, "rel_raw:bm25@0.5") == (2, 0)                    # 5, 4 >= 2.5
    assert dynamic_context(env, q, "rel:bm25@0.5") == (2, 0)                        # 1, .75 >= .5; 1 -> 0
    assert dynamic_context(env, q, "rel_raw:dense@0.5", 50) == (4, 0, 1, 3)
    assert dynamic_context(env, q, "rel:dense@0.9", 50) == (4,)


def _cli():
    import importlib.util

    from graphrag import config
    spec = importlib.util.spec_from_file_location("train_cli", config.ROOT / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.mark.parametrize("cmd,variant,key,refused", [
    ("grpo", "full", "env.L_max=100", True), ("baselines", "adaptive", "env.T_max=3", True),
    ("grpo", "full", "env.K_max=10", False),                     # ruling R12: the one supported env override
    ("grpo", "adaptive", "start.cap=3", True),
    ("grpo", "full", "dynamic_k.normalise=false", True), ("grpo", "hybrid", "dynamic_k.norm_top_n=20", True),
    ("baselines", "full", "dynamic_k.normalise=false", True),
    ("grpo", "adaptive", "dynamic_k.normalise=false", False), ("grpo", "hybrid_adaptive", "dynamic_k.norm_top_n=20", False),
    ("baselines", "full", "dynamic_k.norm_top_n=20", False), ("grpo", "full", "reward.chunks_beta=2", False),
    ("grpo", "full", "grpo.lr=1e-4", False)])
def test_no_op_set_keys_are_refused(cmd, variant, key, refused):
    cli = _cli()
    a = type("A", (), {"set": [key], "variant": variant, "cmd": cmd})()
    if refused:
        with pytest.raises(SystemExit, match="--set"):
            cli.check_set_keys(a)
    else:
        cli.check_set_keys(a)
    if refused:                                          # main() refuses before touching any artifact
        argv = [cmd, "--pairs", "x", "--rankings", "y", "--variant", variant, "--set", key]
        argv += ["--run", "r", "--sft", "s"] if cmd == "grpo" else []
        with pytest.raises(SystemExit, match="--set"):
            cli.main(argv)


@pytest.mark.parametrize("beta,bound", [(1.0, True), (2.0, False)])
def test_oracle_rows_flagged_when_not_upper_bounds(capsys, beta, bound):
    import json

    from graphrag import config
    cli = _cli()
    cfg = config.load("train")
    cfg["env"]["k_grid"] = [3]
    S = {"env": Env(toy_data(), CFG10, TOY_START), "ranks": RANKS, "val": ITEMS, "w": GRLWeights(chunks_beta=beta),
         "cfg": cfg}
    a = type("A", (), {"curves": False, "split": "val", "out": None, "variant": "full"})()
    assert cli.COMMANDS["baselines"](a, S) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    for n in ("greedy_oracle", "beam_oracle"):
        assert out[n]["upper_bound"] is bound and (("note" in out[n]) is not bound)
    assert "upper_bound" not in out["bm25@3"]
