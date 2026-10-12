"""Observable state (IMPLEMENTATION.md §6.3, §6.6; U3, U4): a function of (q, C_t, G) only -- never of C*."""
import ast
import dataclasses
import inspect

import numpy as np
import pytest
import torch

from graphrag import config
from graphrag.env.start import QueryCtx
from grl.model.observe import describe
from grl.model.state import StateEncoder, segment_mean
from tests.m3_toy import toy_rank
from tests.m4_toy import toy_env

BANNED = {"gold", "golds", "C_ref", "C_star", "c_ref"}


def test_describe_reset_state_on_toy():
    env = toy_env()
    q = env.query(toy_rank())
    s = env.reset(q)
    d = describe(env, q, s, preview=3, cand_top=64)
    assert d.covered.size == 0 and d.frontier.tolist() == sorted(int(v) for v in q.v0)
    assert d.v0.tolist() == q.v0.tolist()
    for r in range(7):
        c, _ = env.graph_scores(q, s, r)
        assert sorted(d.cand[r].tolist()) == sorted(c.tolist()) and d.cand_n[r] == c.size
    assert d.preview[0].tolist() == [1, 2, 4] and d.preview_sim[0] == pytest.approx(0.8)    # lex 2, 0, 3
    assert d.preview[1].tolist() == [1, 4] and d.preview_sim[1] == pytest.approx(0.9)       # vec 4, 0, 1
    assert d.scalars[:4].tolist() == [0, 0, 0, 0] and float(d.hist.sum()) == 0.0
    assert d.mask.tolist() == env.mask(q, s).tolist()


def test_describe_tracks_budget_and_history_and_refuses_finished_states():
    env = toy_env()
    q = env.query(toy_rank())
    s = env.step(q, env.reset(q), 1)                                  # lexical k=3
    assert s.context == (2, 0, 3)
    d = describe(env, q, s)
    assert d.scalars[0] == pytest.approx(0.1) and d.scalars[1] == pytest.approx(0.75)
    assert d.scalars[2] == pytest.approx(0.58) and d.hist[1] == pytest.approx(0.1)
    assert d.covered.tolist() == sorted(s.covered)
    with pytest.raises(ValueError):
        describe(env, q, env.step(q, s, 0))


FORBIDDEN_SUBSTR = ("gold", "c_ref", "c_star", "cstar")
FORBIDDEN_MODULES = ("reward", "policies", "grl.train")


def _violations(source: str) -> list:
    """Identifier-bearing names / strings containing a gold-ish token, and imports of reward / policies / grl.train."""
    tree = ast.parse(source)
    bad = []
    for n in ast.walk(tree):
        idents = []
        if isinstance(n, ast.Name):
            idents.append(n.id)
        elif isinstance(n, ast.arg):
            idents.append(n.arg)
        elif isinstance(n, ast.Attribute):
            idents.append(n.attr)
        elif isinstance(n, ast.alias):
            idents += [n.name, n.asname or ""]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idents.append(n.name)
        elif isinstance(n, ast.keyword):
            idents.append(n.arg or "")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            idents.append(n.value)
        bad += [i for i in idents if any(t in i.lower() for t in FORBIDDEN_SUBSTR)]
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [n.module or ""] + [f"{n.module or ''}.{a.name}" for a in n.names]
        bad += [m for m in mods if any(t in m.lower() for t in FORBIDDEN_MODULES)]
    return bad


def test_policy_side_code_never_names_the_gold():
    files = sorted(config.path("grl/model").rglob("*.py"))          # also policy.py / variants.py once they exist
    assert {"graph.py", "gnn.py", "observe.py", "state.py"} <= {f.name for f in files}
    for f in files:
        assert not _violations(f.read_text(encoding="utf-8")), f
    assert list(inspect.signature(describe).parameters) == ["env", "q", "s", "preview", "cand_top"]
    assert not ({f.name for f in dataclasses.fields(QueryCtx)} & BANNED)


def test_no_gold_guard_can_fail():
    assert _violations('x = getattr(env, "gold_ids")')
    assert _violations("from graphrag.env.reward import reward")
    assert _violations("import grl.train.loop")
    assert _violations("def f(a):\n    return g(Gold=a)")
    assert not _violations("def f(a):\n    return a + 1")


def test_segment_mean_matches_a_loop_and_empty_is_zero():
    z = torch.randn(2, 8, 16)
    rows = [np.array([1, 3]), np.zeros(0, np.int64), np.array([0, 0, 7])]
    m = segment_mean(z, [0, 1, 1], rows)
    assert torch.allclose(m[0], z[0, [1, 3]].mean(0)) and torch.equal(m[1], torch.zeros(16))
    assert torch.allclose(m[2], z[1, [0, 0, 7]].mean(0))


def test_state_vector_dimension():
    env = toy_env()
    q = env.query(toy_rank())
    d = describe(env, q, env.reset(q))
    st = StateEncoder(dim=256, pool_hidden=128, cand_dim=32, n_types=7, n_actions=28)
    x = st(torch.randn(1, 8, 256), torch.randn(1, 256), [0], [d])
    assert st.out_dim == 1198 and x.shape == (1, 1198) and bool(torch.isfinite(x).all())
