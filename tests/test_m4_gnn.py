"""GAT with query injection (IMPLEMENTATION.md §6.2, §12 п. 5; U2)."""
from dataclasses import replace

import torch

from grl.model.gnn import GraphEncoder, encoder_params
from tests.m4_toy import MODEL_CFG, toy_graph


def _enc(**over):
    cfg = {**MODEL_CFG, **over}
    torch.manual_seed(0)
    g = toy_graph(**cfg["graph"])
    return GraphEncoder(g, cfg).eval(), g


def _changed(enc, g, node: int) -> list[bool]:
    q = torch.eye(4)[:1]
    with torch.no_grad():
        z0, _ = enc(g, q)
        g2 = replace(g, x=g.x.clone())
        g2.x[node] += 1.0
        z1, _ = enc(g2, q)
    return ((z1[0] - z0[0]).abs().sum(1) > 1e-6).tolist()


def test_shapes_and_query_dependence_of_every_vertex():
    enc, g = _enc()
    with torch.no_grad():
        z, qz = enc(g, torch.eye(4)[:2])
    assert z.shape == (2, 8, 256) and qz.shape == (2, 256)
    assert bool(((z[0] - z[1]).abs().sum(1) > 0).all())           # §12 п. 5


def test_batched_equals_single_queries():
    enc, g = _enc()
    q = torch.eye(4)[:2]
    with torch.no_grad():
        z, _ = enc(g, q)
        for b in range(2):
            zb, _ = enc(g, q[b:b + 1])
            assert torch.allclose(z[b], zb[0], atol=1e-5)


def test_receptive_field_follows_reverse_edges():
    enc2, g2 = _enc(layers=2)
    ch = _changed(enc2, g2, 7)            # 7 -> 4 (rev AMENDS) -> 2 (rev REFERS) -> 0 (rev CONTAINS): 3 hops to D1
    assert ch[7] and ch[4] and ch[2] and not ch[0]
    enc3, g3 = _enc()
    assert _changed(enc3, g3, 7)[0]
    encf, gf = _enc(graph={**MODEL_CFG["graph"], "reverse_edges": False})
    assert _changed(encf, gf, 7) == [i == 7 for i in range(8)]   # 7 has no stored out-edge


def test_attention_is_a_distribution_over_in_edges():
    enc, g = _enc()
    layer = enc.layers[0]
    with torch.no_grad():
        h, ea = enc.nodes(g), enc.edges(g)
        _, (ei, alpha) = layer.conv(h, g.edge_index, edge_attr=ea, return_attention_weights=True)
    s = torch.zeros(g.n_nodes, alpha.shape[1]).index_add_(0, ei[1], alpha)
    has = torch.bincount(g.edge_index[1], minlength=g.n_nodes) > 0
    assert torch.allclose(s[has], torch.ones_like(s[has]), atol=1e-5)


def test_mlp_encoder_ignores_neighbours():
    enc, g = _enc(encoder="mlp")
    assert enc.edges is None and _changed(enc, g, 4) == [i == 4 for i in range(8)]


def test_parameter_breakdown():
    enc, _ = _enc()
    p = encoder_params(enc)
    assert p["layers"] == 3 * (74496 + 65792 + 65536 + 512)
    assert p["q_proj"] == 4 * 256 + 256 and p["edges"] == 13 * 32 + 3 * 32 + 5 * 32 + 2 * 32 + 32
    assert p["total"] == p["nodes"] + p["edges"] + p["q_proj"] + p["layers"]
