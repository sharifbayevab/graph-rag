import numpy as np
import pytest

from graphrag.env.start import make_query, query_from_vector
from graphrag.retrieval.executor import Budget, delta_v, graph_candidates, graph_delta, ranked_delta
from tests.m3_toy import TOY_START, toy_data, toy_rank

B = Budget(0, 0, 4, 1000)


@pytest.fixture
def toy():
    d = toy_data()
    return d, make_query(d, toy_rank(), TOY_START, 10)


def test_lexical_and_vector_take_the_first_k_unseen(toy):
    d, q = toy
    assert ranked_delta(q.lex, frozenset(), 3, B, d.chunk_tokens) == [2, 0, 3]
    assert ranked_delta(q.vec, frozenset({4}), 3, B, d.chunk_tokens) == [0, 1, 3]
    assert ranked_delta(q.lex, frozenset({2, 0, 3}), 3, B, d.chunk_tokens) == []


def test_a_head_fragment_longer_than_the_rest_of_L_max_is_skipped(toy):
    """Review Focus 4."""
    d, q = toy
    assert ranked_delta(q.lex, frozenset(), 3, Budget(0, 0, 4, 200), d.chunk_tokens) == [0, 3]
    assert ranked_delta(q.lex, frozenset(), 10, Budget(3, 0, 4, 1000), d.chunk_tokens) == [2]


def test_graph_candidates_by_type_both_directions_minus_covered(toy):
    d, q = toy
    g = d.graph
    assert graph_candidates(d, q, frozenset(), g.type_id("REFERS_TO")).tolist() == [2, 4]
    assert graph_candidates(d, q, frozenset(), g.type_id("AMENDS")).tolist() == [7]
    assert graph_candidates(d, q, frozenset(), g.type_id("REPEALS")).tolist() == []
    assert graph_candidates(d, q, frozenset({4}), g.type_id("REGULATES")).tolist() == [1, 5]


def test_graph_traversal_ranks_by_delta_v_and_caps_fragments_per_vertex(toy):
    d, q = toy
    g = d.graph
    assert graph_delta(d, q, frozenset(), (), frozenset(), g.type_id("REFERS_TO"), 1, B, 3, 0.5, 6) == ([4], [4, 3])
    assert graph_delta(d, q, frozenset(), (), frozenset(), g.type_id("AMENDS"), 1, B, 3, 0.5, 6) == ([7], [])
    assert graph_delta(d, q, frozenset(), (), frozenset(), g.type_id("REGULATES"), 3, B, 3, 0.5, 6) == \
        ([4, 1, 5], [4, 3, 0, 1])
    assert graph_delta(d, q, frozenset(), (), frozenset(), g.type_id("REGULATES"), 3, B, 1, 0.5, 6) == \
        ([4, 1, 5], [4, 0])


def test_delta_v_redundancy_by_hand(toy):
    d, q = toy
    cand = graph_candidates(d, q, frozenset({4}), d.graph.type_id("RELATED"))
    assert cand.tolist() == [1]
    assert delta_v(d, q, cand, (4,), 0.5, 6).tolist() == pytest.approx([0.44])          # .8 − .5·max(.72, .54)
    assert delta_v(d, q, np.array([5]), (4,), 0.5, 6).tolist() == pytest.approx([0.5])


def test_hub_source_is_bounded_and_deterministic():
    """A hub as source (EXT/КоАП has degree 458): bounded by m · per_vertex_chunks, same result twice."""
    from graphrag.env.data import EnvData
    nodes = ["H"] + [f"H/СТ.{i:03d}" for i in range(300)]
    chunks = [f"H/СТ.{i:03d}#1" for i in range(300)]
    edges = [("H", "CONTAINS", f"H/СТ.{i:03d}") for i in range(300)]
    v = np.random.default_rng(0).normal(size=(300, 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    d = EnvData.build(nodes, chunks, edges, {c: c[:-2] for c in chunks}, {c[:-2]: [c] for c in chunks} | {"H": chunks},
                      {c: 10 for c in chunks}, v, np.tile(np.eye(4, dtype=np.float32)[:1], (301, 1)), ["CONTAINS"])
    q = query_from_vector(d, "hub", np.array([1, 0, 0, 0], np.float32), [0])
    run = lambda: graph_delta(d, q, frozenset(), (), frozenset(), 0, 5, Budget(0, 0, 24, 8000), 3, 0.5, 6)
    a = run()
    assert len(a[0]) == 5 and len(a[1]) == 5 and a == run()


@pytest.mark.corpus
def test_manual_corpus_examples():
    """docs/m2/report.md §11: AMENDS from the amending part reaches the amended Criminal Code article stub (no
    fragments: cover grows, context does not); CONTAINS from an article reaches its document and its part."""
    from graphrag.env.data import load_env_data
    d = load_env_data()
    src, dst = d.node_row["QONUNLAR-a325e7c6/СТ.1/Ч.1"], d.node_row["EXT/jinoyat kodeks/СТ.122.1"]
    q = query_from_vector(d, "manual-amends", d.node_x[dst], [src])
    r = d.graph.type_id("AMENDS")
    assert dst in graph_candidates(d, q, frozenset(), r).tolist()
    assert graph_delta(d, q, frozenset(), (), frozenset(), r, 1, Budget(0, 0, 24, 8000), 3, 0.5, 6) == ([dst], [])
    art = d.node_row["QONUNLAR-a325e7c6/СТ.1"]
    q2 = query_from_vector(d, "manual-contains", d.node_x[art], [art])
    assert {d.node_row["QONUNLAR-a325e7c6"], src} <= set(graph_candidates(d, q2, frozenset(), d.graph.type_id("CONTAINS")).tolist())
