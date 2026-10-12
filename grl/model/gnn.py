"""GAT with query injection (IMPLEMENTATION.md §6.2; roadmap U2):
    h_v^(l+1) = ELU(LayerNorm(W₀·h_v + Σ_u α_uv·W₁·h_u + W₂·e_q)),  3 layers, 8 heads × 32 = 256,
α conditioned on edge features (relation incl. reverse, REFERS_TO class, AMENDS ops, weight, source log-degree --
hub normalisation). The query enters every layer (§12 п. 5). `encoder: mlp` drops the neighbour sum (§9.5 ablation
"без GNN-кодировщика"). A batch of B queries runs as B copies of the graph (PyG Batch layout)."""
from __future__ import annotations

import torch
from torch import nn

from grl.model.graph import GraphTensors


class NodeEncoder(nn.Module):
    def __init__(self, g: GraphTensors, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.kind = nn.Embedding(g.n_kinds, dim)
        self.ntype = nn.Embedding(g.n_node_types, dim)
        self.etype = nn.Embedding(g.n_entity_types, dim)
        self.ext = nn.Embedding(2, dim)
        self.deg = nn.Linear(1, dim)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        return (self.proj(g.x) + self.kind(g.node_kind) + self.ntype(g.node_type) + self.etype(g.entity_type)
                + self.ext(g.is_external) + self.deg(g.logdeg.unsqueeze(1)))


class EdgeEncoder(nn.Module):
    def __init__(self, g: GraphTensors, dim: int):
        super().__init__()
        self.rel = nn.Embedding(len(g.rel_names), dim)
        self.relation = nn.Embedding(g.n_relations, dim)
        self.ops = nn.Linear(g.n_ops, dim, bias=False)
        self.num = nn.Linear(2, dim)

    def forward(self, g: GraphTensors) -> torch.Tensor:
        return (self.rel(g.rel) + self.relation(g.relation) + self.ops(g.ops)
                + self.num(torch.stack([g.weight, g.src_logdeg], 1)))


class QueryGATLayer(nn.Module):
    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float, message: bool = True):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv = (GATConv(dim, dim // heads, heads=heads, concat=True, edge_dim=edge_dim, add_self_loops=False,
                             dropout=dropout, bias=False) if message else None)
        self.root = nn.Linear(dim, dim)                  # W₀
        self.query = nn.Linear(dim, dim, bias=False)     # W₂
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ELU()

    def forward(self, h, edge_index, edge_attr, qn):
        out = self.root(h) + self.query(qn)
        if self.conv is not None:
            out = out + self.conv(h, edge_index, edge_attr=edge_attr)
        return self.act(self.norm(out))


class GraphEncoder(nn.Module):
    def __init__(self, g: GraphTensors, cfg: dict):
        super().__init__()
        dim, heads = int(cfg["hidden"]), int(cfg["heads"])
        if dim % heads:
            raise ValueError(f"hidden {dim} is not divisible by heads {heads}")
        kind = cfg.get("encoder", "gat")
        if kind not in ("gat", "mlp"):
            raise ValueError(f"encoder {kind!r} is not gat | mlp")
        self.dim = dim
        self.nodes = NodeEncoder(g, int(cfg["in_dim"]), dim)
        self.edges = EdgeEncoder(g, int(cfg["edge_dim"])) if kind == "gat" else None
        self.q_proj = nn.Linear(int(cfg["in_dim"]), dim)
        self.layers = nn.ModuleList([QueryGATLayer(dim, heads, int(cfg["edge_dim"]), float(cfg.get("dropout", 0.0)),
                                                   message=kind == "gat") for _ in range(int(cfg["layers"]))])
        self._ei: dict = {}

    def _batched(self, ei: torch.Tensor, B: int, N: int) -> torch.Tensor:
        k = (B, N, str(ei.device), ei.data_ptr())
        if k not in self._ei:
            self._ei = {k: torch.cat([ei + b * N for b in range(B)], 1)}
        return self._ei[k]

    def forward(self, g: GraphTensors, qvec: torch.Tensor):
        B, N = int(qvec.shape[0]), g.n_nodes
        qz = self.q_proj(qvec)
        h = self.nodes(g).repeat(B, 1)                    # copy b = rows b·N … b·N + N − 1
        qn = qz.repeat_interleave(N, 0)
        ei = self._batched(g.edge_index, B, N) if self.edges is not None else None
        ea = self.edges(g).repeat(B, 1) if self.edges is not None else None
        for layer in self.layers:
            h = layer(h, ei, ea, qn)
        return h.view(B, N, self.dim), qz


def _count(m) -> int:
    return sum(p.numel() for p in m.parameters()) if m is not None else 0


def encoder_params(enc: GraphEncoder) -> dict[str, int]:
    out = {"nodes": _count(enc.nodes), "edges": _count(enc.edges), "q_proj": _count(enc.q_proj),
           "layers": _count(enc.layers)}
    out["total"] = sum(out.values())
    return out
