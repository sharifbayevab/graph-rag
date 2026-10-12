"""M7 Task 3: the `struct` variant -- StructEnv, the 28 M3 actions + `structure` m ∈ m_grid (31), docs/m7 design §3.1.
Toy: document D with one article (one chunk), external act EXT/k with one chunkless element, AMENDS D/СТ.1 ->
EXT/k/СТ.61; both roots have STRUCT rows (extend_with_struct). V₀ = [D/СТ.1]."""
import numpy as np
import pytest

from graphrag.env.data import EnvData
from graphrag.env.mdp import InvalidAction
from graphrag.env.start import query_from_vector
from grl.env.struct_data import extend_with_struct
from grl.model.variants import (VARIANTS, StructEnv, check_variant_invariants, env_spec, is_struct)
from tests.m3_toy import EDGE_TYPES, TOY_CFG, TOY_START

NODES = ["D", "D/СТ.1", "EXT/k", "EXT/k/СТ.61"]
CHUNKS = ["D/СТ.1#1"]
EDGES = [("D", "CONTAINS", "D/СТ.1"), ("EXT/k", "CONTAINS", "EXT/k/СТ.61"), ("D/СТ.1", "AMENDS", "EXT/k/СТ.61")]
QVEC = np.array([1, 0, 0, 0], np.float32)
S_D, S_EXT = "STRUCT/D", "STRUCT/EXT/k"
# cos with e_q: chunk .6, STRUCT/D .3, STRUCT/EXT/k .8 -> after AMENDS the EXT fragment ranks first (Δ_V, no context)
VECS = {S_D: np.array([0.3, 0, np.sqrt(0.91), 0]), S_EXT: np.array([0.8, 0.6, 0, 0])}


def toy_struct_data(tokens=(50, 50)):
    d = EnvData.build(NODES, CHUNKS, EDGES, {"D/СТ.1#1": "D/СТ.1"}, {"D": CHUNKS, "D/СТ.1": CHUNKS},
                      {"D/СТ.1#1": 100}, np.array([[0.6, 0, 0, 0.8]]), np.array([[0, 0, 0, 1]] * 4), EDGE_TYPES)
    return extend_with_struct(d, {S_D: "d", S_EXT: "e"}, VECS, {S_D: tokens[0], S_EXT: tokens[1]})


def make(cfg_over=None, tokens=(50, 50)):
    data = toy_struct_data(tokens)
    env = StructEnv(data, {**TOY_CFG, "K_max": 5, "T_max": 10, "m_grid": [1, 3, 5], **(cfg_over or {})}, TOY_START)
    q = query_from_vector(data, "toy|struct", QVEC, [data.node_row["D/СТ.1"]])
    return env, q


@pytest.fixture
def toy_struct():
    return make()


@pytest.fixture
def toy_struct_kmax1():
    return make({"K_max": 1})


def amends_m1(env):
    return next(i for i, a in enumerate(env.actions) if a.op == "graph" and a.r == "AMENDS" and a.m == 1)


def ids(env, rows):
    return [env.data.chunk_ids[int(c)] for c in rows]


def test_action_space_31(toy_struct):
    env, q = toy_struct
    assert len(env.actions) == 31 and [a.op for a in env.actions[28:]] == ["structure"] * 3
    assert [a.m for a in env.actions[28:]] == [1, 3, 5] and env.allowed.shape == (31,) and env.allowed.all()
    assert env.mask(q, env.reset(q)).shape == (31,)


def test_structure_adds_root_fragment(toy_struct):
    env, q = toy_struct
    s = env.reset(q)
    s = env.step(q, s, 28)                                   # structure m=1 from V0 = D/СТ.1 -> STRUCT/D
    assert env.data.chunk_ids[s.context[-1]] == "STRUCT/D"
    assert env.data.node_row["D"] in s.covered


def test_ext_reached_after_amends(toy_struct):
    # Ruling R1: the AMENDS step must be valid in the reset state, is taken, and reaches the EXT root
    env, q = toy_struct
    s = env.reset(q)
    a = amends_m1(env)
    assert env.mask(q, s)[a]
    s = env.step(q, s, a)
    ext = env.data.node_row["EXT/k/СТ.61"]
    assert s.history[-1].selected == (ext,) and s.history[-1].added == () and ext in s.covered   # chunkless: μ⁻¹ = ∅
    assert "STRUCT/EXT/k" in set(ids(env, env.struct_candidates(q, s)))


def test_reset_frontier_is_v0(toy_struct):
    # M3: F_0 = V₀ (Env.reset), so a graph neighbour of V₀ (EXT/k/СТ.61) is not a frontier vertex yet and R10
    # gives the same reset candidates as covered ∪ V₀
    env, q = toy_struct
    s = env.reset(q)
    assert s.frontier == frozenset({env.data.node_row["D/СТ.1"]})
    assert ids(env, env.struct_candidates(q, s)) == [S_D]


def test_frontier_root_is_a_candidate(toy_struct):
    # R10: candidates = roots of frontier ∪ covered ∪ V₀. CONTAINS m=1 selects D and adds its chunk (μ = D/СТ.1),
    # so EXT/k/СТ.61 enters F_t through the AMENDS edge while neither it nor EXT/k is covered
    env, q = toy_struct
    contains_m1 = next(i for i, a in enumerate(env.actions) if a.op == "graph" and a.r == "CONTAINS" and a.m == 1)
    s = env.reset(q)
    assert env.mask(q, s)[contains_m1]
    s = env.step(q, s, contains_m1)
    r = env.data.node_row
    assert r["EXT/k/СТ.61"] in s.frontier and not {r["EXT/k/СТ.61"], r["EXT/k"]} & s.covered
    assert S_EXT in set(ids(env, env.struct_candidates(q, s)))
    assert env.mask(q, s)[28]
    s = env.step(q, s, 30)                                     # structure m=5 adds both root fragments
    assert set(ids(env, s.context[1:])) == {S_D, S_EXT} and r["EXT/k"] in s.covered
    assert check_variant_invariants(env, q, env.step(q, s, 0)) == []


def test_candidates_ranked_by_fragment_delta_v(toy_struct):
    env, q = toy_struct
    s = env.step(q, env.reset(q), amends_m1(env))
    assert ids(env, env.struct_candidates(q, s)) == [S_EXT, S_D]          # rel .8 > .3, empty context
    s = env.step(q, s, 29)                                                  # structure m=3: both, in that order
    assert ids(env, s.context) == [S_EXT, S_D]
    assert {env.data.node_row["D"], env.data.node_row["EXT/k"]} <= s.covered
    assert env.struct_candidates(q, s).size == 0 and not env.mask(q, s)[28:].any()


def test_redundancy_term_uses_the_fragment_vector():
    # Δ_V of a STRUCT fragment f = cos(e_q, e_f) − λ·max_{c'∈C} cos(e_f, e_c'): the fragment's own vector, as a vertex
    # with μ⁻¹ = {f} would score in graphrag delta_v
    env, q = make()
    v = env.data.vecs64
    c = env.data.chunk_row["D/СТ.1#1"]
    rows = np.array([env.data.chunk_row[S_D], env.data.chunk_row[S_EXT]])
    lam = env.lambda_red
    want = np.round(q.chunk_qsim[rows] - lam * (v[rows] @ v[c]), env.decimals)
    from grl.model.variants import struct_delta_v
    got = struct_delta_v(env.data, q, rows, (c,), lam, env.decimals)
    assert np.allclose(got, want)


def test_structure_respects_l_max():
    env, q = make({"L_max": 120}, tokens=(30, 200))           # EXT fragment (200 tokens) never fits
    s = env.step(q, env.reset(q), amends_m1(env))
    s = env.step(q, s, 30)                                     # structure m=5
    assert ids(env, s.context) == [S_D] and s.length == 30


def test_structure_masked_without_candidates(toy_struct):
    env, q = toy_struct
    s = env.step(q, env.reset(q), 28)
    assert not env.mask(q, s)[28:].any()                      # STRUCT/D is in C, EXT/k not reached
    with pytest.raises(InvalidAction):
        env.step(q, s, 28)
    with pytest.raises(InvalidAction):
        env.step(q, s, 28, check=False)                       # empty effect is refused on the fast path too


def test_graph_ops_never_add_struct(toy_struct):
    env, q = toy_struct
    s = env.reset(q)
    for a in range(1, 28):
        if env.mask(q, s)[a]:
            s2 = env.step(q, s, a)
            assert all(c < env.data.n_base_chunks for c in s2.context)


def test_structure_masked_when_full(toy_struct_kmax1):     # Review Focus 5: K_max reached -> structure closed
    env, q = toy_struct_kmax1
    s = env.step(q, env.reset(q), 28)
    assert s.done and s.reason in ("k_max", "exhausted")


def test_invariants_hold(toy_struct):
    env, q = toy_struct
    s = env.reset(q)
    s = env.step(q, s, 28)
    s = env.step(q, s, 0)
    assert check_variant_invariants(env, q, s) == []
    s = env.step(q, env.reset(q), amends_m1(env))
    s = env.step(q, s, 30)
    s = env.step(q, s, 0)
    assert check_variant_invariants(env, q, s) == []


def test_invariants_catch_wrong_rows(toy_struct, monkeypatch):
    env, q = toy_struct
    base = env.data.chunk_row["D/СТ.1#1"]
    s = env.step(q, env.reset(q), 28)
    s = env.step(q, s, 0)
    orig = StructEnv.effect
    monkeypatch.setattr(StructEnv, "effect",
                        lambda self, q_, s_, a: ([base], []) if a.op == "structure" else orig(self, q_, s_, a))
    env._memo_m.clear()
    assert any("structure added a base row" in p for p in check_variant_invariants(env, q, s))
    monkeypatch.setattr(StructEnv, "effect", orig)
    s = env.step(q, env.reset(q), amends_m1(env))
    s = env.step(q, s, 0)
    struct_row = env.data.chunk_row[S_D]
    monkeypatch.setattr(StructEnv, "effect",
                        lambda self, q_, s_, a: ([struct_row], [int(q_.v0[0])]) if a.op == "graph"
                        else orig(self, q_, s_, a))
    assert any("graph added a STRUCT row" in p for p in check_variant_invariants(env, q, s))


def test_m3_part_of_mask_equals_filtered_env():
    # StructEnv._mask re-implements Env._mask for the 28 M3 slots; it must agree with FilteredEnv on every state
    from grl.model.variants import FilteredEnv
    env, _ = make({"K_max": 10})
    ref = FilteredEnv(env.data, env.cfg, TOY_START)
    for q in (query_from_vector(env.data, "toy|a", QVEC, [env.data.node_row["D/СТ.1"]]),
              query_from_vector(env.data, "toy|b", QVEC, [env.data.node_row["EXT/k"]])):
        frontier = [env.reset(q)]
        while frontier:
            s = frontier.pop()
            m, mr = env.mask(q, s), ref.mask(q, s)
            assert np.array_equal(m[:28], mr), (s.context, s.covered)
            for a in np.flatnonzero(m):
                if a:
                    s2 = env.step(q, s, int(a))
                    if not s2.done:
                        frontier.append(s2)


def test_needs_struct_rows():
    from tests.m3_toy import toy_data
    with pytest.raises(ValueError):
        StructEnv(toy_data(), TOY_CFG, TOY_START)


def test_variant_registry():
    assert VARIANTS["struct"] == {"struct": True} and is_struct("struct") and not is_struct("full")
    cls, kw = env_spec("struct")
    assert cls is StructEnv and kw == {}


# --- real graph (Spark verify; also run once on the Mac) ----------------------------------------------------------

@pytest.mark.slow
def test_real_graph_roots_match_structure_py():
    """Every vertex that has a root under grl.env.structure (CONTAINS walk + id-prefix fallback, on nodes.jsonl /
    edges.jsonl) gets the same root row in StructEnv; vertices without one get -1; every root has a STRUCT row."""
    import json

    from graphrag import config
    from grl.env.struct_data import load_struct_env_data
    from grl.env.structure import _Resolver
    data = load_struct_env_data()
    env = StructEnv(data)
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    d = config.path(latest["dir"])
    rd = lambda f: [json.loads(x) for x in (d / f).read_text(encoding="utf-8").splitlines() if x.strip()]
    nodes, edges = rd("nodes.jsonl"), rd("edges.jsonl")
    roots = {n["node_id"]: n for n in nodes if n["kind"] == "document"}
    assert {data.node_row[r] for r in roots} == set(data.struct_rows)
    res = _Resolver(roots, edges)
    want = np.array([data.node_row[r] if (r := res(nid)) is not None else -1 for nid in data.node_ids], np.int64)
    bad = np.flatnonzero(want != env._root_row)
    assert bad.size == 0, [(data.node_ids[i], want[i], env._root_row[i]) for i in bad[:5]]


# ---- M7 part 2, Task 1: cheap structure mask (struct_any_fits) + struct_scores memo ----

def _struct_states(env, q):
    """Reset, after AMENDS (two candidates), after structure m=1, after structure m=5 (no candidate left)."""
    s0 = env.reset(q)
    s1 = env.step(q, s0, amends_m1(env))
    out = [s0, s1]
    for a in (28, 30):                                  # structure m=1 / m=5 when valid (else the states above only)
        if env.mask(q, s1)[a]:
            out.append(env.step(q, s1, a))
    return out


@pytest.mark.parametrize("over, tokens", [(None, (50, 50)), ({"K_max": 1}, (50, 50)), ({"K_max": 2}, (50, 50)),
                                          ({"L_max": 120}, (30, 200)), ({"L_max": 100}, (50, 60)),
                                          ({"L_max": 40}, (50, 60))])
def test_struct_any_fits_matches_take(over, tokens):
    env, q = make(over, tokens)
    for s in _struct_states(env, q):
        assert env.struct_any_fits(q, s) == bool(env._struct_take(q, s, 1)), (over, tokens, s.context)


def test_struct_any_fits_on_the_fixture(toy_struct):
    env, q = toy_struct
    s = env.reset(q)
    assert env.struct_any_fits(q, s) == bool(env._struct_take(q, s, 1)) is True


def test_mask_does_not_score_candidates(toy_struct, monkeypatch):
    env, q = toy_struct
    import grl.model.variants as V

    def boom(*a, **k):
        raise AssertionError("mask scored STRUCT candidates")
    monkeypatch.setattr(V, "struct_delta_v", boom)
    for s in (env.reset(q), env.step(q, env.reset(q), amends_m1(env))):
        env._memo_m.clear()
        assert env.mask(q, s)[28:].all()


def test_struct_scores_memoised_per_query(toy_struct, monkeypatch):
    env, q = toy_struct
    import grl.model.variants as V
    calls, real = [], V.struct_delta_v
    monkeypatch.setattr(V, "struct_delta_v", lambda *a, **k: calls.append(1) or real(*a, **k))
    s = env.step(q, env.reset(q), amends_m1(env))
    a = env.struct_scores(q, s)
    b = env.struct_scores(q, s)
    assert len(calls) == 1 and a[0] is b[0] and not a[0].flags.writeable    # read-only shared memo, as _memo_s
    env.step(q, s, 29)                                                       # the step itself reads the memo
    assert len(calls) == 1
    q2 = query_from_vector(env.data, "toy|struct2", QVEC, [env.data.node_row["D/СТ.1"]])
    s2 = env.step(q2, env.reset(q2), amends_m1(env))                         # another query: memo reset (_own)
    assert not env._memo_st or all(k[0] == "toy|struct2" for k in env._memo_st)
    env.struct_scores(q2, s2)
    assert len(calls) == 2
