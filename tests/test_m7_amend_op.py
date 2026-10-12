"""M7 Task 6d-b: the StructEnv operator «amendments of act» (struct.amend_op, 31 -> 34 actions).
Toy: law L (points L/СТ.1, L/СТ.2), law M (point M/СТ.1), internal act A (article A/СТ.1) and the chunkless external
act EXT/y (stubs EXT/y/СТ.5, EXT/y/СТ.7). AMENDS L/СТ.1 -> EXT/y/СТ.5, L/СТ.2 -> EXT/y/СТ.7, M/СТ.1 -> EXT/y/СТ.5,
M/СТ.1 -> A/СТ.1. Every root has a STRUCT row. e_q = (1, 0, 0, 0)."""
import ast
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from graphrag.env.data import EnvData
from graphrag.env.mdp import InvalidAction
from graphrag.env.start import query_from_vector
from grl.env.struct_data import extend_with_struct
from grl.model import variants as V
from grl.model.observe import describe
from grl.model.policy import GraphPolicy, policy_dims
from grl.model.variants import AMEND_OP, StructEnv, check_variant_invariants, struct_delta_v, struct_kw
from tests.m3_toy import EDGE_TYPES, TOY_CFG, TOY_START

NODES = ["A", "A/СТ.1", "EXT/y", "EXT/y/СТ.5", "EXT/y/СТ.7", "L", "L/СТ.1", "L/СТ.2", "M", "M/СТ.1"]
CHUNKS = ["A/СТ.1#1", "L/СТ.1#1", "L/СТ.1#2", "L/СТ.2#1", "M/СТ.1#1"]
PARENT = {"A/СТ.1#1": "A/СТ.1", "L/СТ.1#1": "L/СТ.1", "L/СТ.1#2": "L/СТ.1", "L/СТ.2#1": "L/СТ.2",
          "M/СТ.1#1": "M/СТ.1"}
MU = {"A": ["A/СТ.1#1"], "A/СТ.1": ["A/СТ.1#1"], "L": ["L/СТ.1#1", "L/СТ.1#2", "L/СТ.2#1"],
      "L/СТ.1": ["L/СТ.1#1", "L/СТ.1#2"], "L/СТ.2": ["L/СТ.2#1"], "M": ["M/СТ.1#1"], "M/СТ.1": ["M/СТ.1#1"]}
AMENDS = [("L/СТ.1", "EXT/y/СТ.5"), ("L/СТ.2", "EXT/y/СТ.7"), ("M/СТ.1", "EXT/y/СТ.5"), ("M/СТ.1", "A/СТ.1")]
EDGES = [("A", "CONTAINS", "A/СТ.1"), ("EXT/y", "CONTAINS", "EXT/y/СТ.5"), ("EXT/y", "CONTAINS", "EXT/y/СТ.7"),
         ("L", "CONTAINS", "L/СТ.1"), ("L", "CONTAINS", "L/СТ.2"), ("M", "CONTAINS", "M/СТ.1")] + \
        [(s, "AMENDS", d) for s, d in AMENDS]
# cos with e_q: A .2, L/СТ.1#1 .5, L/СТ.1#2 .9, L/СТ.2#1 .7, M/СТ.1#1 .3; L/СТ.1#2 and L/СТ.2#1 are near-duplicates
CVECS = np.array([[0.2, np.sqrt(0.96), 0, 0], [0.5, 0, np.sqrt(0.75), 0], [0.9, 0, 0, np.sqrt(0.19)],
                  [0.7, 0, 0, np.sqrt(0.51)], [0.3, 0, np.sqrt(0.91), 0]], np.float32)
TOK = {"A/СТ.1#1": 100, "L/СТ.1#1": 100, "L/СТ.1#2": 100, "L/СТ.2#1": 100, "M/СТ.1#1": 100}
QVEC = np.array([1, 0, 0, 0], np.float32)
ROOTS = ("A", "EXT/y", "L", "M")


def toy_data(tokens=None):
    d = EnvData.build(NODES, CHUNKS, EDGES, PARENT, MU, {**TOK, **(tokens or {})}, CVECS,
                      np.array([[0, 0, 0, 1]] * len(NODES)), EDGE_TYPES)
    ids = {f"STRUCT/{r}": r for r in ROOTS}
    e = extend_with_struct(d, {i: i for i in ids}, {i: np.array([0.1, 0, 0, np.sqrt(0.99)], np.float32) for i in ids},
                           {i: 40 for i in ids})
    e.amend_pairs = list(AMENDS)                        # the directed (src, dst) ids (real data: edges.jsonl)
    return e


def make(amend_op=True, cfg_over=None, v0=("EXT/y/СТ.5",), tokens=None, data=None):
    data = data or toy_data(tokens)
    env = StructEnv(data, {**TOY_CFG, "K_max": 5, "T_max": 10, "m_grid": [1, 3, 5], **(cfg_over or {})}, TOY_START,
                    amend_op=amend_op)
    q = query_from_vector(data, "toy|amend|" + ",".join(v0), QVEC, [data.node_row[v] for v in v0])
    return env, q


def ids(env, rows):
    return [env.data.chunk_ids[int(c)] for c in rows]


AM1, AM3, AM5 = 31, 32, 33


# ---- action space / mask ------------------------------------------------------------------------------------------

def test_action_space_34_and_layout():
    env, q = make()
    off, _ = make(amend_op=False)
    assert len(env.actions) == 34 and len(off.actions) == 31
    assert env.actions[:31] == off.actions                                   # ids 0..30 unchanged
    assert [a.op for a in env.actions[31:]] == [AMEND_OP] * 3 and [a.m for a in env.actions[31:]] == [1, 3, 5]
    assert env.allowed.shape == (34,) and env.allowed.all() and env.op_of.shape == (34,)
    m = env.mask(q, env.reset(q))
    assert m.shape == (34,) and m[31:].all()


def test_mask_closed_without_amended_root():
    env, q = make(v0=("L/СТ.1",))                         # root L: no incoming AMENDS
    s = env.reset(q)
    assert env.amend_candidates(q, s).size == 0 and not env.mask(q, s)[31:].any()


def test_mask_closed_when_full():
    env, q = make(cfg_over={"K_max": 1})
    s = env.step(q, env.reset(q), AM1)
    assert s.done and s.reason == "k_max"
    env2, q2 = make(tokens={c: 600 for c in TOK}, cfg_over={"L_max": 500})      # nothing fits L_max
    assert not env2.mask(q2, env2.reset(q2))[31:].any()


@pytest.mark.parametrize("over, tokens", [(None, None), ({"K_max": 2}, None), ({"L_max": 150}, {"L/СТ.1#2": 200}),
                                          ({"L_max": 50}, None)])
def test_any_fits_matches_take(over, tokens):
    env, q = make(cfg_over=over, tokens=tokens)
    s = env.reset(q)
    states = [s]
    if env.mask(q, s)[AM1]:
        states.append(env.step(q, s, AM1))
    for x in states:
        if not x.done:
            assert env.amend_any_fits(q, x) == bool(env._amend_take(q, x, 1))


# ---- effect -------------------------------------------------------------------------------------------------------

def test_candidates_are_amending_source_fragments_ranked_by_delta_v():
    env, q = make()
    s = env.reset(q)
    rows = env.amend_candidates(q, s)
    # root EXT/y <- L/СТ.1, L/СТ.2, M/СТ.1 (not A/СТ.1#1: A is amended, not amending); empty context: Δ_V = cos
    assert ids(env, rows) == ["L/СТ.1#2", "L/СТ.2#1", "L/СТ.1#1", "M/СТ.1#1"]
    sc = struct_delta_v(env.data, q, rows, s.context, env.lambda_red, env.decimals)
    assert np.allclose(env.amend_scores(q, s)[1], sc) and all(np.diff(sc) <= 0)


def test_step_adds_top_m_and_covers_sources():
    env, q = make()
    s = env.step(q, env.reset(q), AM1)
    assert ids(env, s.context) == ["L/СТ.1#2"] and s.history[-1].added == tuple(s.context)
    assert env.data.node_row["L/СТ.1"] in s.covered
    # redundancy: L/СТ.2#1 is near L/СТ.1#2 (cos .94), so with #2 in C the next pick is L/СТ.1#1 (.275 > .23)
    nxt = ids(env, env.amend_candidates(q, s))
    assert nxt[0] == "L/СТ.1#1" and "L/СТ.1#2" not in nxt
    s3 = env.step(q, env.reset(q), AM3)
    assert len(s3.context) == 3 and set(ids(env, s3.context)) <= {c for c in CHUNKS if c != "A/СТ.1#1"}
    assert check_variant_invariants(env, q, env.step(q, s3, 0)) == []


def test_root_set_is_frontier_covered_and_v0():
    # V0 = M/СТ.1 (root M, not amended): no candidate at reset. After AMENDS m=1 (-> EXT/y/СТ.5 or A/СТ.1, chunkless /
    # A's article) the amended roots are reached and their AMENDS sources become candidates
    env, q = make(v0=("M/СТ.1",))
    s = env.reset(q)
    assert not env.mask(q, s)[31:].any()
    ame = next(i for i, a in enumerate(env.actions) if a.op == "graph" and a.r == "AMENDS" and a.m == 5)
    s = env.step(q, s, ame)
    assert env.mask(q, s)[31:].any()
    got = set(ids(env, env.amend_candidates(q, s)))
    assert {"L/СТ.1#1", "L/СТ.1#2", "L/СТ.2#1"} <= got
    s = env.step(q, s, AM5)
    assert check_variant_invariants(env, q, s if s.done else env.step(q, s, 0)) == []


def test_never_adds_struct_rows_and_invariant_catches_outsiders(monkeypatch):
    env, q = make()
    s = env.step(q, env.reset(q), AM5)
    n0 = env.data.n_base_chunks
    assert all(int(c) < n0 for c in s.context)
    final = s if s.done else env.step(q, s, 0)
    assert check_variant_invariants(env, q, final) == []
    outsider = env.data.chunk_row["A/СТ.1#1"]
    orig = StructEnv.effect
    monkeypatch.setattr(StructEnv, "effect",
                        lambda self, q_, s_, a: ([outsider], []) if a.op == AMEND_OP else orig(self, q_, s_, a))
    env._memo_m.clear()
    f2 = env.step(q, env.step(q, env.reset(q), AM1), 0)
    assert any("outside the AMENDS sources" in p for p in check_variant_invariants(env, q, f2))


def test_composes_with_articles():
    """articles=True and amend_op=True together: 34 actions, the article rows stay `structure`-only."""
    from grl.env.struct_data import extend_with_articles
    base = toy_data()
    art = extend_with_articles(base, {"STRUCT/A/СТ.1": "a"}, {"STRUCT/A/СТ.1": np.array([0.4, 0, 0, np.sqrt(.84)])},
                               {"STRUCT/A/СТ.1": 30})
    art.amend_pairs = list(AMENDS)
    env2 = StructEnv(art, {**TOY_CFG, "K_max": 5, "m_grid": [1, 3, 5]}, TOY_START, articles=True, amend_op=True)
    q = query_from_vector(art, "toy|amend|art", QVEC, [art.node_row["A/СТ.1"], art.node_row["EXT/y/СТ.5"]])
    assert len(env2.actions) == 34
    s = env2.reset(q)
    assert "STRUCT/A/СТ.1" in ids(env2, env2.struct_candidates(q, s))
    assert not set(ids(env2, env2.amend_candidates(q, s))) & {"STRUCT/A/СТ.1", "A/СТ.1#1"}
    s = env2.step(q, s, AM3)
    assert check_variant_invariants(env2, q, env2.step(q, s, 0)) == []


# ---- flag off: the 31-action env exactly ---------------------------------------------------------------------------

def _walk(env, q, depth=3):
    out, stack = [], [(env.reset(q), ())]
    while stack:
        s, path = stack.pop()
        out.append((path, s))
        if s.done or len(path) >= depth:
            continue
        for a in np.flatnonzero(env.mask(q, s)):
            stack.append((env.step(q, s, int(a)), path + (int(a),)))
    return out


def test_flag_off_is_the_31_action_env():
    on, q_on = make()
    off, q_off = make(amend_op=False)
    assert struct_kw({}) == {} == struct_kw({"struct": {"articles": False, "amend_op": False}})
    assert struct_kw({"struct": {"amend_op": True}}) == {"amend_op": True}
    with pytest.raises(ValueError):
        struct_kw({"struct": {"amend_op": "yes"}})
    assert off._amend_rows == {}
    # every state reachable with the 31 actions: identical masks (first 31) and identical successors
    for path, s in _walk(off, q_off):
        s_on = on.reset(q_on)
        for a in path:
            s_on = on.step(q_on, s_on, a)
        assert s_on == replace(s, qkey=s_on.qkey)
        if not s.done:
            assert np.array_equal(on.mask(q_on, s_on)[:31], off.mask(q_off, s))
            d_on, d_off = describe(on, q_on, s_on, 3, 64), describe(off, q_off, s)
            assert len(d_on.cand) == 9 and len(d_off.cand) == 8
            assert all(np.array_equal(a, b) for a, b in zip(d_on.cand[:8], d_off.cand))
            assert np.array_equal(d_on.cand_n[:8], d_off.cand_n) and np.array_equal(d_on.cand_best[:8], d_off.cand_best)


def test_flag_off_reads_no_amend_data(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("amend data read with amend_op off")
    monkeypatch.setattr(V, "amend_pairs", boom)
    data = toy_data()
    del data.amend_pairs
    env = StructEnv(data, {**TOY_CFG, "m_grid": [1, 3, 5]}, TOY_START)
    assert len(env.actions) == 31


# ---- determinism --------------------------------------------------------------------------------------------------

def test_deterministic_across_instances_and_order():
    a, qa = make()
    b, qb = make()
    for path, s in _walk(a, qa, depth=2):
        t = b.reset(qb)
        for x in path:
            t = b.step(qb, t, x)
        assert t.context == s.context and t.covered == s.covered
        if not s.done:
            assert np.array_equal(a.amend_candidates(qa, s), b.amend_candidates(qb, t))
    # ties broken by row asc
    data = toy_data()
    data.chunk_vecs[:] = np.array([1, 0, 0, 0], np.float32)
    data.__dict__.pop("vecs64", None)
    env, q = make(data=data)
    rows = env.amend_candidates(q, env.reset(q))
    assert rows.tolist() == sorted(rows.tolist())


# ---- observation / policy / transfer ------------------------------------------------------------------------------

def test_describe_amendments_is_type_8():
    env, q = make()
    s = env.reset(q)
    d = describe(env, q, s, 3, 64)
    rows, sc = env.amend_scores(q, s)
    want = list(dict.fromkeys(env.data.chunk_node[rows].tolist()))
    assert d.cand[8].tolist() == want and d.cand_n[8] == len(rows) and d.cand_best[8] == pytest.approx(sc[0])
    assert d.mask.shape == d.hist.shape == (34,)
    d2 = describe(env, q, env.step(q, s, AM1), 3, 64)
    assert d2.hist[AM1] == pytest.approx(1 / env.limits.T_max)
    assert policy_dims(env) == {"n_actions": 34, "n_types": 9}
    assert policy_dims(make(amend_op=False)[0]) == {"n_actions": 31, "n_types": 8}


def _small_policy(n_actions, n_types, seed):
    from tests.m4_toy import SMALL_CFG, toy_graph
    torch.manual_seed(seed)
    g = toy_graph()
    pol = GraphPolicy(g, SMALL_CFG, n_actions=n_actions, n_types=n_types).eval()
    for p in pol.head.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return pol, g


def _desc(n_types, n_actions, seed=0):
    """A synthetic StateDesc on the M4 toy graph's vertices; candidates of types >= 8 empty."""
    from grl.model.observe import StateDesc
    rng = np.random.default_rng(seed)
    cand = [np.array([1, 3], np.int64) if r < 8 else np.zeros(0, np.int64) for r in range(n_types)]
    mask = np.zeros(n_actions, bool)
    mask[:31] = True
    hist = np.zeros(n_actions, np.float32)
    hist[[2, 29]] = 0.1
    cn = np.array([2.0 if r < 8 else 0.0 for r in range(n_types)], np.float32)
    cb = np.array([0.3 if r < 8 else 0.0 for r in range(n_types)], np.float32)
    return StateDesc(np.array([0], np.int64), np.array([1, 2], np.int64), np.array([0, 1], np.int64), cand, cn, cb,
                     [np.array([2], np.int64), np.zeros(0, np.int64)], np.array([.5, 0], np.float32),
                     rng.random(6).astype(np.float32), mask, hist)


def _logits(pol, g, desc):
    with torch.no_grad():
        z, qz = pol.encode(g, torch.from_numpy(QVEC[None]).float())
        return pol.head(pol.state(z, qz, [0], [desc]))[0]


def test_transfer_31_to_34_keeps_old_logits():
    from grl.model.transfer import expand_policy_state, widen_policy_state
    old, g = _small_policy(31, 8, 0)
    new_sd = expand_policy_state(old.state_dict(), 31, 34, 8, 9)
    new, _ = _small_policy(34, 9, 1)
    new.load_state_dict(new_sd, strict=True)
    d31, d34 = _desc(8, 31), _desc(9, 34)
    torch.testing.assert_close(_logits(new, g, d34)[:31], _logits(old, g, d31), rtol=1e-5, atol=1e-6)
    last = max(int(k.split(".")[2]) for k in new_sd if k.startswith("head.net.") and k.endswith(".weight"))
    w_old, w_new = old.state_dict()[f"head.net.{last}.weight"], new_sd[f"head.net.{last}.weight"]
    for j in range(3):                                  # new row 31 + j = mean of the 8 types' rows with that m
        torch.testing.assert_close(w_new[31 + j], w_old[[7 + 3 * r + j for r in range(8)]].mean(0))
    via = widen_policy_state(old.state_dict(), (31, 8), (34, 9))
    assert all(torch.equal(via[k], new_sd[k]) for k in new_sd)


def test_widen_28_to_34_chains_and_refuses_backwards():
    from grl.model.transfer import widen_policy_state
    old, g = _small_policy(28, 7, 0)
    sd = widen_policy_state(old.state_dict(), (28, 7), (34, 9))
    new, _ = _small_policy(34, 9, 2)
    new.load_state_dict(sd, strict=True)
    from grl.model.observe import StateDesc
    d34 = _desc(9, 34)
    d28 = StateDesc(d34.covered, d34.frontier, d34.v0, d34.cand[:7], d34.cand_n[:7], d34.cand_best[:7], d34.preview,
                    d34.preview_sim, d34.scalars, d34.mask[:28], np.where(np.arange(28) == 2, 0.1, 0).astype(np.float32))
    d34.cand[7] = np.zeros(0, np.int64)                  # no structure candidate either: old features only
    d34.cand_n[7] = d34.cand_best[7] = 0
    d34.hist[29] = 0
    torch.testing.assert_close(_logits(new, g, d34)[:28], _logits(old, g, d28), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        widen_policy_state(sd, (34, 9), (31, 8))
    with pytest.raises(ValueError):
        widen_policy_state(sd, (30, 8), (34, 9))
    same = widen_policy_state(old.state_dict(), (28, 7), (28, 7))
    assert all(torch.equal(same[k], v) for k, v in old.state_dict().items())


def test_init_from_struct_source_widens_to_34(tmp_path):
    import json

    from grl.train.sft import save_reference
    from tests.m4_toy import SMALL_CFG, toy_graph
    from tests.test_m5_expert import _cli
    cli = _cli()
    torch.manual_seed(0)
    g = toy_graph()
    src = tmp_path / "struct-sft"
    src.mkdir()
    save_reference(GraphPolicy(g, SMALL_CFG, n_actions=31, n_types=8), src)
    (src / "manifest.json").write_text(json.dumps({"phase": "sft", "variant": "struct", "K_max": 5,
                                                   "model": {**SMALL_CFG, "n_actions": 31, "n_types": 8}}))
    mcfg = {**SMALL_CFG, "n_actions": 34, "n_types": 9}
    s = cli._init_from_source(src)
    assert s["variant"] == "struct"
    sd, info = cli._init_from_state(s, g, mcfg)
    GraphPolicy(g, mcfg).load_state_dict(sd, strict=True)
    assert info["variant"] == "struct" and info["model_check"] == "recorded"
    with pytest.raises(SystemExit, match="34 / 9|expected 31 / 8"):               # a 34-action struct source
        (src / "manifest.json").write_text(json.dumps({"phase": "sft", "variant": "struct",
                                                       "model": {**SMALL_CFG, "n_actions": 34, "n_types": 9}}))
        cli._init_from_state(cli._init_from_source(src), g, mcfg)
    with pytest.raises(SystemExit):                                              # a struct source cannot shrink
        (src / "manifest.json").write_text(json.dumps({"phase": "sft", "variant": "struct",
                                                       "model": {**SMALL_CFG, "n_actions": 31, "n_types": 8}}))
        cli._init_from_state(cli._init_from_source(src), g, {**SMALL_CFG, "n_actions": 28, "n_types": 7})


def test_action_names():
    from grl.hard.metrics import _action_name
    from grl.train.evaluate import op_name
    from grl.train.expert import action_name
    env, _ = make()
    assert [action_name(a) for a in env.actions[31:]] == ["amendments:1", "amendments:3", "amendments:5"]
    assert [_action_name(a) for a in env.actions[31:]] == ["amendments:1", "amendments:3", "amendments:5"]
    assert op_name(env.actions[31]) == "amendments"
    assert len({action_name(a) for a in env.actions}) == 34


# ---- no gold (R60) ------------------------------------------------------------------------------------------------

def test_no_gold_guard():
    """The operator never reads C* / labels / the AMENDS evidence: no gold-like name in its code, no edge attribute,
    and the env runs on a QueryCtx (which carries no gold)."""
    banned = ("gold", "c_ref", "y_ref", "answer", "evidence", "attrs", "label", "split", "target")
    fns = (StructEnv._amend_index, StructEnv._amend_cand_rows, StructEnv.amend_scores, StructEnv.amend_any_fits,
           StructEnv._amend_take, StructEnv._take_fitting, V.amend_pairs)
    from grl.hard import amend_lookup as al
    fns += (al.build_amend_index, al.load_amends_pairs)
    for fn in fns:
        text = inspect.getsource(fn)
        tree = ast.parse(text.strip() if not text.startswith(" ") else "class _X:\n" + text)
        names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                {n.arg.lower() for n in ast.walk(tree) if isinstance(n, ast.arg)} | \
                {n.value.lower() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and len(n.value) < 20}
        assert not any(b in x for x in names for b in banned), (fn.__name__, sorted(names))
    sig = set(inspect.signature(StructEnv.amend_scores).parameters)
    assert sig == {"self", "q", "s"}


def test_invalid_amend_step_refused():
    env, q = make(v0=("L/СТ.1",))
    with pytest.raises(InvalidAction):
        env.step(q, env.reset(q), AM1)
