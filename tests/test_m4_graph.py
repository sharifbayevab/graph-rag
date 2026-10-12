"""Graph view for the GAT (IMPLEMENTATION.md §6.2; M2 report §14; R24 reverse edges; M2 §5b п. 2 hubs)."""
import pytest
import torch

from graphrag import config
from tests.m3_toy import EDGE_TYPES
from tests.m4_toy import toy_graph


def test_reverse_edges_for_directed_types_only():
    g = toy_graph()
    assert g.n_edges == 18 and g.rel_names[:7] == EDGE_TYPES
    assert g.rel_names[7:] == [f"rev_{t}" for t in EDGE_TYPES if t != "RELATED"]
    assert int((g.rel >= 7).sum()) == 8 and int((g.rel == 6).sum()) == 2
    pairs = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist(), g.rel.tolist()))
    assert {(4, 2, 8), (7, 4, 9), (1, 0, 7), (2, 4, 1), (4, 7, 2)} <= pairs


def test_no_reverse_and_drop_related():
    assert toy_graph(reverse_edges=False).n_edges == 10
    g = toy_graph(drop_types=("RELATED",))
    assert g.n_edges == 16 and not bool((g.rel == 6).any())
    with pytest.raises(ValueError):
        toy_graph(drop_types=("NOPE",))


def test_degree_features_and_edge_attributes():
    g = toy_graph()
    assert torch.allclose(g.logdeg, torch.log1p(torch.tensor([2.0, 3, 2, 1, 5, 2, 1, 2])))
    assert torch.equal(g.src_logdeg, g.logdeg[g.edge_index[0]])
    ref = (g.rel == 1) | (g.rel == 8)
    assert g.relation[ref].tolist() == [1, 1] and int(g.relation[~ref].sum()) == 0
    am = (g.rel == 2) | (g.rel == 9)
    assert int(g.ops[am].sum()) == 2 and int(g.ops[~am].sum()) == 0
    assert g.weight[g.rel == 6].tolist() == pytest.approx([0.8, 0.8])
    assert (g.n_relations, g.n_ops, g.n_kinds, g.n_node_types, g.n_entity_types) == (3, 5, 4, 4, 5)
    assert int(g.node_type.min()) >= 0 and int(g.entity_type.min()) >= 0


def test_edge_order_is_dst_rel_src():
    g = toy_graph()
    s, d, r = g.edge_index[0], g.edge_index[1], g.rel
    key = (d * len(g.rel_names) + r) * g.n_nodes + s
    assert torch.equal(key, torch.sort(key).values)


def test_hub_cap_keeps_the_smallest_sources():
    g = toy_graph(hub_cap=1)
    key = g.edge_index[1] * len(g.rel_names) + g.rel
    assert key.unique().numel() == key.numel() == 16
    assert g.edge_index[0][(g.edge_index[1] == 5) & (g.rel == 4)].tolist() == [1]


@pytest.mark.corpus
def test_real_graph_view():
    from grl.model.graph import build_graph
    g = build_graph(config.load("train")["model"]["graph"])
    assert g.n_nodes == 22047 and g.x.shape == (22047, 1024)
    assert g.n_edges == 72578 and int((g.rel >= 7).sum()) == 34351 and len(g.rel_names) == 13
    assert g.n_relations == 10 and g.n_ops == 5 and g.n_node_types == 27 and g.n_entity_types == 5
    assert float(torch.expm1(g.logdeg).max()) >= 300                      # the M2 hubs
    assert build_graph({"reverse_edges": False, "drop_types": [], "hub_cap": None}).n_edges == 38227
