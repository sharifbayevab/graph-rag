"""Task 6d-a: the «amendments of the named act» rule (grl.hard.amend_lookup). Toy index, no data."""
import ast
import inspect
from types import SimpleNamespace

import numpy as np

from grl.hard import amend_lookup as al
from grl.hard.amend_lookup import AmendIndex, amend_lookup

# vertices 0..4; roots: 0,1 -> root 0 (act A), 2 -> root 2 (act B, amended), 3,4 -> root 3 (amending law)
ROOT = [0, 0, 2, 3, 3]
MU = {3: [10, 11, 12], 4: [13, 14], 0: [0], 1: [1], 2: [2]}
TOK = np.full(20, 100)


def index():
    return AmendIndex(ROOT, {2: [3, 4]}, lambda v: MU[int(v)])


def q(v0, qs=None):
    qs = np.zeros(20) if qs is None else qs
    return SimpleNamespace(v0=v0, chunk_qsim=qs)


def test_named_act_is_first_amended_root_and_ranked_by_qsim():
    qs = np.zeros(20)
    qs[[13, 11, 12, 10, 14]] = [.9, .8, .7, .6, .5]
    assert amend_lookup(index(), q([0, 1, 2], qs), TOK, k=3) == [13, 11, 12]       # root 0 has no AMENDS: skipped
    assert amend_lookup(index(), q([0, 1, 2], qs), TOK, k=5) == [13, 11, 12, 10, 14]


def test_fallback_when_no_root_qualifies():
    fb = [7, 8]
    assert amend_lookup(index(), q([0, 1, 3]), TOK, fallback=fb) == fb             # roots 0, 3: no incoming AMENDS
    assert amend_lookup(index(), q([]), TOK) == []


def test_budget_skips_oversized_and_caps_k():
    tok = TOK.copy()
    tok[13] = 5000
    qs = np.zeros(20)
    qs[[13, 11, 12, 10, 14]] = [.9, .8, .7, .6, .5]
    assert amend_lookup(index(), q([2], qs), tok, k=5, l_max=300) == [11, 12, 10]


def test_deterministic_ties_by_row():
    a = amend_lookup(index(), q([2]), TOK, k=4)
    assert a == amend_lookup(index(), q([2]), TOK, k=4) == [10, 11, 12, 13]


def test_no_gold_guard():
    """The rule never reads C*: no gold-like parameter, no gold / evidence attribute or name in the code, and it runs
    on a query object that carries only v0 and chunk_qsim."""
    banned = ("gold", "c_ref", "c*", "evidence", "y_ref", "answer", "split")
    params = set(inspect.signature(amend_lookup).parameters)
    assert params == {"index", "q", "chunk_tokens", "k", "l_max", "fallback"}
    for fn in (amend_lookup, al.named_act_root, al.build_amend_index, al.load_amends_pairs):
        text = inspect.getsource(fn)
        src = text.lower()
        tree = ast.parse(text)
        names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr.lower() for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                {n.arg.lower() for n in ast.walk(tree) if isinstance(n, ast.arg)}
        assert not any(b in x for x in names for b in banned), (fn.__name__, names)
        assert "attrs" not in src            # R60: the edge attributes (evidence) are never read
    only = SimpleNamespace(v0=[2], chunk_qsim=np.arange(20.0))
    assert amend_lookup(index(), only, TOK, k=2) == [14, 13]
