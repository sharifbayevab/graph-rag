"""M7 Task 6c (a): article-level STRUCT fragments -- grl.env.structure.article_structure / article_struct_text,
grl.env.struct_data.extend_with_articles, StructEnv(articles=True), the struct.articles config plumbing."""
import numpy as np
import pytest

from graphrag.env.data import EnvData
from graphrag.env.start import query_from_vector
from grl.env.struct_data import extend_with_articles, extend_with_struct
from grl.env.structure import (article_structure, article_struct_text, build_article_struct_texts, struct_doc,
                               struct_id)
from grl.model.variants import StructEnv, check_variant_invariants, data_kw, load_variant_data, variant_env_kw
from tests.m3_toy import EDGE_TYPES, TOY_CFG, TOY_START


def toks(s):
    return len(s.split())


def _n(i, typ, num="", ext=False, title="", kind="element"):
    doc = "/".join(i.split("/")[:2]) if i.startswith("EXT/") else i.split("/")[0]
    return {"node_id": i, "kind": kind, "type": typ, "external": ext, "title": title, "doc_id": None if ext else doc,
            "attrs": {"number": num} if num else {}}


NODES = [_n("Q-1", "документ", kind="document", title="Qonun A"), _n("Q-1/ГЛ.1", "глава", "1"),
         _n("Q-1/ГЛ.1/СТ.1", "статья", "1", title="Maqsad"), _n("Q-1/ГЛ.1/СТ.1/Ч.1", "часть", "1"),
         _n("Q-1/ГЛ.1/СТ.1/Ч.2", "часть", "2"), _n("Q-1/ГЛ.1/СТ.1/Ч.1/П.1", "пункт", "1"),
         _n("Q-1/ГЛ.1/СТ.1/Ч.1/П.2", "пункт", "2"), _n("Q-1/ГЛ.1/СТ.2", "статья", "2"),
         _n("Q-2", "документ", kind="document", title="Qonun B"), _n("Q-2/СТ.3", "статья", "3"),
         _n("Q-2/СТ.3/Ч.1", "часть", "1"), _n("Q-2/СТ.3/Ч.1/П.1", "пункт", "1"),
         _n("EXT/k", "документ", ext=True, kind="document", title="Kodeks"), _n("EXT/k/СТ.61", "статья", "61", ext=True),
         _n("EXT/k/СТ.61/Ч.1", "часть", "1", ext=True)]
C = "CONTAINS"
EDGES = [{"src": a, "dst": b, "type": C, "attrs": {}} for a, b in [
    ("Q-1", "Q-1/ГЛ.1"), ("Q-1/ГЛ.1", "Q-1/ГЛ.1/СТ.1"), ("Q-1/ГЛ.1/СТ.1", "Q-1/ГЛ.1/СТ.1/Ч.2"),
    ("Q-1/ГЛ.1/СТ.1", "Q-1/ГЛ.1/СТ.1/Ч.1"), ("Q-1/ГЛ.1/СТ.1/Ч.1", "Q-1/ГЛ.1/СТ.1/Ч.1/П.2"),
    ("Q-1/ГЛ.1/СТ.1/Ч.1", "Q-1/ГЛ.1/СТ.1/Ч.1/П.1"), ("Q-1/ГЛ.1", "Q-1/ГЛ.1/СТ.2"), ("Q-2", "Q-2/СТ.3"),
    ("Q-2/СТ.3", "Q-2/СТ.3/Ч.1"), ("Q-2/СТ.3/Ч.1", "Q-2/СТ.3/Ч.1/П.1"), ("EXT/k", "EXT/k/СТ.61"),
    ("EXT/k/СТ.61", "EXT/k/СТ.61/Ч.1")]] + [
    {"src": "Q-2/СТ.3/Ч.1/П.1", "dst": "Q-1/ГЛ.1/СТ.1/Ч.2", "type": "AMENDS",
     "attrs": {"date": "2025-01-10", "op": "edit"}},
    {"src": "Q-2/СТ.3/Ч.1/П.1", "dst": "EXT/k/СТ.61/Ч.1", "type": "AMENDS", "attrs": {"date": "2025-01-10", "op": "delete"}}]


def test_article_rows_only_internal_with_parts():
    s = article_structure(NODES, EDGES)
    assert sorted(s) == ["Q-1/ГЛ.1/СТ.1", "Q-2/СТ.3"]           # СТ.2: no part child; EXT/k/СТ.61: an EXT stub
    a = s["Q-1/ГЛ.1/СТ.1"]
    assert (a["doc"], a["number"], a["title"], a["parts"], a["n_parts"], a["n_points"]) == \
        ("Q-1", "1", "Maqsad", ["1", "2"], 2, 2)
    assert a["points"] == {"1": ["1", "2"]} and a["points_unique"]
    assert a["amended_by"] == [("Q-2", "2025-01-10", "edit", "Q-1/ГЛ.1/СТ.1/Ч.2")] and a["repealed_by"] == []


def test_article_text_counts_first_and_deterministic():
    t1 = build_article_struct_texts(NODES, EDGES, toks)
    t2 = build_article_struct_texts(list(reversed(NODES)), list(reversed(EDGES)), toks)
    assert t1 == t2
    t = t1[struct_id("Q-1/ГЛ.1/СТ.1")]
    lines = t.split("\n")
    assert lines[:3] == ["Qonun A", "1-модда. Maqsad", "қисмлар: 2; бандлар: 2"]
    assert "қисм рақамлари: 1, 2" in lines and "банд рақамлари: 1-қисм: 1, 2" in lines
    assert any(ln.startswith("Q-2 «Qonun B» 2025-01-10: edit 1") for ln in lines)


def test_article_text_cap_cuts_lists_not_counts():
    a = article_structure(NODES, EDGES)["Q-1/ГЛ.1/СТ.1"]
    a = {**a, "parts": [str(i) for i in range(1, 400)], "n_parts": 399}
    t = article_struct_text(a, toks, max_tokens=40)
    assert toks(t) <= 40 and "қисмлар: 399; бандлар: 2" in t and "қисм рақамлари" not in t
    with pytest.raises(ValueError):
        article_struct_text({**a, "doc_title": "w " * 50}, toks, max_tokens=40)


def test_struct_doc():
    assert struct_doc("STRUCT/Q-1") == "Q-1" and struct_doc("STRUCT/Q-1/ГЛ.1/СТ.1") == "Q-1"
    assert struct_doc("STRUCT/EXT/k") == "EXT/k" and struct_doc("STRUCT/EXT/k/СТ.61") == "EXT/k"
    with pytest.raises(ValueError):
        struct_doc("Q-1/СТ.1#1")


# --- data + env --------------------------------------------------------------------------------------------------

ENODES = ["D", "D/СТ.1", "D/СТ.1/Ч.1", "D/СТ.2", "D/СТ.2/Ч.1"]
ECHUNKS = ["D/СТ.1/Ч.1#1", "D/СТ.2/Ч.1#1"]
EEDGES = [("D", "CONTAINS", "D/СТ.1"), ("D/СТ.1", "CONTAINS", "D/СТ.1/Ч.1"), ("D", "CONTAINS", "D/СТ.2"),
          ("D/СТ.2", "CONTAINS", "D/СТ.2/Ч.1")]
QVEC = np.array([1, 0, 0, 0], np.float32)
A1, A2, SD = "STRUCT/D/СТ.1", "STRUCT/D/СТ.2", "STRUCT/D"


def base_data():
    d = EnvData.build(ENODES, ECHUNKS, EEDGES, {c: c.split("#")[0] for c in ECHUNKS},
                      {"D/СТ.1/Ч.1": ECHUNKS[:1], "D/СТ.2/Ч.1": ECHUNKS[1:]}, {c: 100 for c in ECHUNKS},
                      np.array([[0.6, 0, 0, 0.8], [0.5, 0, 0.866, 0]]), np.array([[0, 0, 0, 1]] * 5), EDGE_TYPES)
    return extend_with_struct(d, {SD: "d"}, {SD: np.array([0.3, 0, np.sqrt(0.91), 0])}, {SD: 40})


def art_data():
    v = {A1: np.array([0.9, np.sqrt(0.19), 0, 0]), A2: np.array([0.2, 0, 0, np.sqrt(0.96)])}
    return extend_with_articles(base_data(), {A1: "a1", A2: "a2"}, v, {A1: 30, A2: 30})


def make(data, articles, v0="D/СТ.1/Ч.1", **kw):
    env = StructEnv(data, {**TOY_CFG, "K_max": 5, "T_max": 10, "m_grid": [1, 3, 5]}, TOY_START, articles=articles,
                    **kw)
    return env, query_from_vector(data, "toy|art", QVEC, [data.node_row[v0]])


def test_article_rows_after_document_rows():
    b, e = base_data(), art_data()
    assert e.chunk_ids[:b.n_chunks] == b.chunk_ids and e.chunk_ids[b.n_chunks:] == [A1, A2]
    assert e.n_base_chunks == b.n_base_chunks == 2 and e.struct_rows == b.struct_rows == {e.node_row["D"]: 2}
    assert e.struct_art_rows == {e.node_row["D/СТ.1"]: 3, e.node_row["D/СТ.2"]: 4}
    assert np.array_equal(e.mu_idx, b.mu_idx) and np.allclose(e.chunk_vecs[:3], b.chunk_vecs)
    assert e.chunk_node[3] == e.node_row["D/СТ.1"] and list(e.chunk_tokens[3:]) == [30, 30]


def test_extend_with_articles_guards():
    with pytest.raises(ValueError):                                   # no document rows yet
        extend_with_articles(EnvData.build(ENODES, ECHUNKS, EEDGES, {c: c.split("#")[0] for c in ECHUNKS},
                                           {}, {c: 1 for c in ECHUNKS}, np.ones((2, 4)), np.ones((5, 4)), EDGE_TYPES),
                             {A1: "x"}, {A1: np.ones(4)}, {A1: 1})
    with pytest.raises(ValueError):                                   # twice
        extend_with_articles(art_data(), {A1: "x"}, {A1: np.ones(4)}, {A1: 1})
    with pytest.raises(ValueError):                                   # a root is not an article row
        extend_with_articles(base_data(), {"STRUCT/D": "x"}, {"STRUCT/D": np.ones(4)}, {"STRUCT/D": 1})
    with pytest.raises(ValueError):                                   # unknown node
        extend_with_articles(base_data(), {"STRUCT/D/СТ.9": "x"}, {"STRUCT/D/СТ.9": np.ones(4)}, {"STRUCT/D/СТ.9": 1})


def test_env_flag_must_match_data():
    with pytest.raises(ValueError):
        make(art_data(), False)
    with pytest.raises(ValueError):
        make(base_data(), True)


def test_old_mode_candidates_unchanged():
    env, q = make(base_data(), False)
    s = env.reset(q)
    assert [env.data.chunk_ids[c] for c in env.struct_candidates(q, s)] == [SD]
    assert len(env.actions) == 31


def test_article_scope_root_offers_all_articles_of_the_root():
    env, q = make(art_data(), True, article_scope="root")
    s = env.reset(q)
    cands = [env.data.chunk_ids[c] for c in env.struct_candidates(q, s)]
    assert set(cands) == {SD, A1, A2} and cands[0] == A1                 # Δ_V = cos: A1 .9 > SD .3 > A2 .2
    assert len(env.actions) == 31 and env.mask(q, s)[28]
    s = env.step(q, s, 28)                                               # structure m=1
    assert env.data.chunk_ids[s.context[-1]] == A1 and env.data.node_row["D/СТ.1"] in s.covered
    s = env.step(q, s, 29)                                               # structure m=3: the two left
    assert {env.data.chunk_ids[c] for c in s.context} >= {A1, A2, SD}
    assert not env.mask(q, s)[28]                                        # nothing left to add
    assert check_variant_invariants(env, q, env.step(q, s, 0)) == []


def test_article_mode_struct_take_respects_budget():
    env = StructEnv(art_data(), {**TOY_CFG, "K_max": 2, "T_max": 10, "m_grid": [1, 3, 5]}, TOY_START, articles=True,
                    article_scope="root")
    q = query_from_vector(env.data, "toy|art", QVEC, [env.data.node_row["D/СТ.1/Ч.1"]])
    s = env.step(q, env.reset(q), 30)                                    # structure m=5 under K_max 2
    assert len(s.context) == 2


def test_r58_reached_scope_only_articles_in_f_or_s():
    ids = lambda env, q, s: {env.data.chunk_ids[c] for c in env.struct_candidates(q, s)}
    env, q = make(art_data(), True)                                   # default scope "reached"; V0 = a part of СТ.1
    assert env.article_scope == "reached"
    s = env.reset(q)
    assert ids(env, q, s) == {SD}                                     # СТ.1 itself is not in F_0 ∪ S_0 yet
    env, q = make(art_data(), True, v0="D/СТ.1")                       # the article vertex reached
    assert ids(env, q, env.reset(q)) == {SD, A1}                      # never СТ.2: not reached
    env, q = make(art_data(), True, article_scope="ancestor")         # + the article above a reached vertex
    assert ids(env, q, env.reset(q)) == {SD, A1}
    with pytest.raises(ValueError):
        make(art_data(), True, article_scope="all")


def test_r58_reached_after_a_graph_step():
    env, q = make(art_data(), True)
    s = env.reset(q)
    for i, a in enumerate(env.actions):                               # a CONTAINS step that covers / fronts СТ.1
        if a.op == "graph" and a.r == "CONTAINS" and env.mask(q, s)[i]:
            s2 = env.step(q, s, i)
            if env.data.node_row["D/СТ.1"] in set(s2.frontier) | set(s2.covered):
                assert A1 in {env.data.chunk_ids[c] for c in env.struct_candidates(q, s2)}
                assert "STRUCT/D/СТ.2" not in {env.data.chunk_ids[c] for c in env.struct_candidates(q, s2)} or \
                    env.data.node_row["D/СТ.2"] in set(s2.frontier) | set(s2.covered)
                return
    pytest.fail("no CONTAINS step reaches D/СТ.1 in the toy")


def test_config_plumbing():
    assert variant_env_kw("struct", {"struct": {"articles": True}}) == {"articles": True}
    assert variant_env_kw("struct", {"struct": {"articles": False}}) == {} == variant_env_kw("struct", {})
    with pytest.raises(ValueError):
        variant_env_kw("struct", {"struct": {"articles": "yes"}})
    assert data_kw({"articles": True}) == {"articles": True} and data_kw({}) == {} == data_kw(None)
    with pytest.raises(ValueError):
        load_variant_data("full", articles=True)


def test_train_yaml_default_off():
    from graphrag import config
    assert config.load("train")["struct"] == {"articles": False, "amend_op": False}       # + Task 6d-b


def test_gold_docs_of_article_struct_ids():
    from grl.hard.checks import _doc_of
    from grl.hard.splits import gold_docs
    assert gold_docs(["STRUCT/Q-1/ГЛ.1/СТ.1", "STRUCT/EXT/k", "c#1"], {"c#1": "Q-9"}) == ["EXT/k", "Q-1", "Q-9"]
    assert _doc_of("STRUCT/Q-1/ГЛ.1/СТ.1") == "Q-1" and _doc_of("STRUCT/EXT/k") == "EXT/k"


# --- Spark verify only: real data, article rows after the document rows, in no ranking / μ⁻¹ ----------------------

@pytest.mark.slow
def test_real_article_rows():
    from grl.env.struct_data import load_struct_env_data
    a, b = load_struct_env_data(), load_struct_env_data(articles=True)
    nb = a.n_base_chunks
    assert b.chunk_ids[:a.n_chunks] == a.chunk_ids and b.struct_rows == a.struct_rows and b.n_base_chunks == nb
    assert len(b.struct_art_rows) == b.n_chunks - a.n_chunks == 903
    assert sorted(b.struct_art_rows.values()) == list(range(a.n_chunks, b.n_chunks))
    assert int(b.mu_idx.max()) < nb and np.array_equal(b.mu_idx, a.mu_idx)
    assert all(b.node_ids[v].split("/")[0] in {b.node_ids[r] for r in b.struct_rows} for v in b.struct_art_rows)


def test_struct_data_for_follows_the_flag(monkeypatch):
    from grl.env import struct_data
    from grl.model.variants import struct_data_for
    seen = []
    monkeypatch.setattr(struct_data, "load_struct_env_data", lambda **kw: seen.append(kw) or "d")
    assert struct_data_for({"struct": {"articles": True}}) == "d" and struct_data_for({}) == "d"
    assert seen == [{"articles": True}, {}]


def test_spark_source_head_flags_dirty(tmp_path):
    import subprocess
    from scripts.spark import source_head
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / "f").write_text("1")
    run("add", "f")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    h = source_head(tmp_path)
    assert len(h) == 40 and not h.endswith("dirty")
    (tmp_path / "f").write_text("2")
    assert source_head(tmp_path) == h + " dirty"
