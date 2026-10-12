import numpy as np
import pytest

from tests.m3_toy import toy_data


def test_typed_neighbours_are_undirected_unique_and_sorted():
    g = toy_data().graph
    assert g.neighbors([2], g.type_id("REFERS_TO")).tolist() == [4]
    assert g.neighbors([4], g.type_id("REFERS_TO")).tolist() == [2]
    assert g.neighbors([0, 1, 2, 3, 4, 5], g.type_id("REGULATES")).tolist() == [1, 4, 5]
    assert g.neighbors([0], g.type_id("REPEALS")).tolist() == []
    assert g.all_neighbors([4]).tolist() == [1, 2, 3, 5, 7]


def test_contains_parent_ancestors_and_tree_depth():
    g = toy_data().graph
    assert g.parent.tolist() == [-1, 0, 0, -1, 3, -1, -1, 6]
    assert g.ancestors(4) == [3] and g.ancestors(0) == []
    assert g.tree_depth.tolist() == [0, 1, 1, 0, 1, 0, 0, 1]


def test_bfs_depth_from_sources():
    g = toy_data().graph
    assert g.depth_from([2, 0, 5, 4, 3, 1], 10).tolist() == [0, 0, 0, 0, 0, 0, 2, 1]
    assert g.depth_from([6], 1).tolist() == [-1, -1, -1, -1, -1, -1, 0, 1]
    assert g.depth_from([], 10).tolist() == [-1] * 8


def test_two_contains_parents_are_rejected():
    from graphrag.retrieval.graph_walk import ArtifactGraph
    with pytest.raises(ValueError, match="forest"):
        ArtifactGraph(3, np.array([[0, 1], [2, 2]]), np.array([0, 0]), ["CONTAINS"])


def test_env_data_mu_inverse_and_tokens():
    d = toy_data()
    assert d.mu_inv(0).tolist() == [0, 1, 2] and d.mu_inv(5).tolist() == []
    assert d.chunk_node.tolist() == [1, 1, 2, 4, 4]
    assert d.chunk_tokens.tolist() == [100, 50, 400, 80, 60]


@pytest.mark.corpus
def test_corpus_env_data_matches_records_and_pyg():
    """m10: one definition of μ⁻¹ -- pyg.pt CSR == graphrag.data.mapping.mu_inverse per record; CONTAINS parents ==
    Node.parent_id for corpus vertices; unit-norm 1024-d fragment vectors."""
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.data.mapping import mu_inverse
    from graphrag.env.data import load_env_data
    d = load_env_data()
    for r in read_records(config.path(config.load("data")["records_dir"])):
        mi = mu_inverse(r.structure, r.chunks)
        for n in r.structure:
            v = d.node_row[n.node_id]
            assert sorted(d.chunk_ids[c] for c in d.mu_inv(v)) == sorted(mi[n.node_id]), n.node_id
            assert int(d.graph.parent[v]) == (d.node_row[n.parent_id] if n.parent_id else -1), n.node_id
    assert d.chunk_vecs.shape == (len(d.chunk_ids), 1024)
    assert np.allclose(np.linalg.norm(d.chunk_vecs, axis=1), 1.0, atol=1e-3)


def _parity_rows(d, seed: int = 20260928, per_side: int = 25) -> list[int]:
    """Stratified parity rows: 300 random vertices + the 5 largest hubs + every endpoint of the sparse types
    (REPEALS, DEFINES) + per other edge type `per_side` random sources and `per_side` random targets of the
    directed pyg.pt edges (both directions; >= 50 non-empty rows per type). Fixed seed."""
    import json
    import torch
    from graphrag import config
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    p = torch.load(config.path(latest["dir"]) / "pyg.pt", weights_only=True)
    src, dst = p["edge_index"][0].numpy(), p["edge_index"][1].numpy()
    et = p["edge_type"].numpy()
    rng = np.random.default_rng(seed)
    rows = set(rng.choice(d.n_nodes, 300, replace=False).tolist())
    rows |= set(np.argsort(-d.graph.degree(), kind="stable")[:5].tolist())
    for t, r in enumerate(d.graph.edge_types):
        for side in (src[et == t], dst[et == t]):
            ends = np.unique(side)
            if r in ("REPEALS", "DEFINES") or ends.size <= per_side:
                rows |= set(ends.tolist())
            else:
                rows |= set(rng.choice(ends, per_side, replace=False).tolist())
    return sorted(rows)


@pytest.mark.integration
@pytest.mark.corpus
def test_artifact_neighbours_equal_the_neo4j_active_version():
    """Roadmap §5b item 4 (D4 (a)): stratified rows (_parity_rows), all 7 edge types; every type is exercised by
    >= 50 rows with a non-empty neighbour list."""
    from graphrag.env.data import load_env_data
    from graphrag.kg.store import GraphStore
    from graphrag.retrieval.graph_walk import Neo4jGraph, neighbor_parity
    d = load_env_data()
    rows = _parity_rows(d)
    for t, r in enumerate(d.graph.edge_types):
        assert sum(d.graph.neighbors([v], t).size > 0 for v in rows) >= 50, r
    store = GraphStore()
    try:
        neo = Neo4jGraph(store, d.version)
        assert neighbor_parity(d.graph, d.node_ids, neo, rows) == []
    finally:
        store.close()


@pytest.mark.integration
def test_neo4j_graph_refuses_a_version_that_is_not_active():
    from graphrag.kg.store import GraphStore
    from graphrag.retrieval.graph_walk import Neo4jGraph
    store = GraphStore()
    try:
        with pytest.raises(ValueError, match="active version"):
            Neo4jGraph(store, "000000000000")
    finally:
        store.close()


def test_neo4j_graph_refuses_non_dev_endpoints():
    from types import SimpleNamespace
    from graphrag.retrieval.graph_walk import Neo4jGraph
    with pytest.raises(PermissionError):
        Neo4jGraph(SimpleNamespace(uri="bolt://localhost:7687"), "x")
