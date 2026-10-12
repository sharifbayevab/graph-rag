"""Graph view of the pinned artifact for the M4 GAT (IMPLEMENTATION.md §6.2; M2 report §14; M2 ruling R24 left the
reverse relations to M4). Message direction is src -> dst (PyG). Relation ids: forward = graph.yaml edge type id
(0..6); reverse of each directed type = 7 + its rank among the directed types (RELATED is stored in both directions
already and gets no reverse). Edge order (dst, rel, src) -- deterministic. Hub normalisation (M2 §5b п. 2): log1p of
the in-degree as a vertex feature and log1p of the source in-degree inside edge_attr, i.e. in the attention logit.
edge_date is not used (an amendment date does not depend on the query)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace

import torch

from graphrag import config

UNDIRECTED = ("RELATED",)


@dataclass
class GraphTensors:
    x: torch.Tensor              # float32 [N, d]
    edge_index: torch.Tensor     # int64 [2, E]  src -> dst
    rel: torch.Tensor            # int64 [E] relation id (forward / reverse)
    relation: torch.Tensor       # int64 [E] REFERS_TO class + 1 (0 = none)
    ops: torch.Tensor            # float32 [E, n_ops] AMENDS multi-hot
    weight: torch.Tensor         # float32 [E] RELATED cosine, else 1
    src_logdeg: torch.Tensor     # float32 [E] log1p(in-degree of the source)
    node_kind: torch.Tensor      # int64 [N]
    node_type: torch.Tensor      # int64 [N] fine type + 1 (0 = none)
    entity_type: torch.Tensor    # int64 [N] entity type + 1 (0 = none)
    is_external: torch.Tensor    # int64 [N] 0 / 1
    logdeg: torch.Tensor         # float32 [N] log1p(in-degree)
    rel_names: list
    n_relations: int
    n_ops: int
    n_kinds: int
    n_node_types: int
    n_entity_types: int

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device) -> "GraphTensors":
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})


def graph_tensors(p: dict, reverse_edges: bool = True, drop_types=(), hub_cap: int | None = None) -> GraphTensors:
    types = list(p["edge_types"])
    unknown = set(drop_types) - set(types)
    if unknown:
        raise ValueError(f"unknown edge types {sorted(unknown)}")
    n = int(p["x"].shape[0])
    src, dst = p["edge_index"][0].long(), p["edge_index"][1].long()
    et = p["edge_type"].long()
    keep = torch.ones_like(et, dtype=torch.bool)
    for t in drop_types:
        keep &= et != types.index(t)
    src, dst, et = src[keep], dst[keep], et[keep]
    relation = p["edge_attr_relation"].long()[keep] + 1
    ops = p["edge_attr_ops"][keep].float()
    w = p["edge_weight"][keep].float()
    rel = et.clone()
    directed = [i for i, t in enumerate(types) if t not in UNDIRECTED]
    rel_names = types + [f"rev_{types[i]}" for i in directed]
    if reverse_edges:
        rev = torch.full((len(types),), -1, dtype=torch.long)
        for j, i in enumerate(directed):
            rev[i] = len(types) + j
        m = rev[et] >= 0
        src, dst = torch.cat([src, dst[m]]), torch.cat([dst, src[m]])
        rel = torch.cat([rel, rev[et[m]]])
        relation = torch.cat([relation, relation[m]])
        ops = torch.cat([ops, ops[m]])
        w = torch.cat([w, w[m]])
    R = len(rel_names)
    order = torch.argsort((dst * R + rel) * n + src, stable=True)
    src, dst, rel, relation, ops, w = src[order], dst[order], rel[order], relation[order], ops[order], w[order]
    if hub_cap is not None:
        key = dst * R + rel
        start = torch.ones_like(key, dtype=torch.bool)
        start[1:] = key[1:] != key[:-1]
        idx = torch.arange(key.numel())
        first = torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0).values
        k = (idx - first) < int(hub_cap)
        src, dst, rel, relation, ops, w = src[k], dst[k], rel[k], relation[k], ops[k], w[k]
    logdeg = torch.log1p(torch.bincount(dst, minlength=n).float())
    return GraphTensors(p["x"].float(), torch.stack([src, dst]), rel, relation, ops, w, logdeg[src],
                        p["node_kind"].long(), p["node_type"].long() + 1, p["entity_type"].long() + 1,
                        p["is_external"].long(), logdeg, rel_names, len(p["edge_relations"]) + 1,
                        len(p["amend_ops"]), len(p["node_kinds"]), len(p["node_types"]) + 1,
                        len(p["entity_types"]) + 1)


def load_payload(version: str | None = None, latest_path: str = "data/graph/latest.json") -> dict:
    latest = json.loads(config.path(latest_path).read_text(encoding="utf-8"))
    want = version or config.load("train")["env"]["graph_version"]
    if latest["version"] != want:
        raise ValueError(f"graph {latest['version']} in {latest_path} != pinned {want} (train.yaml env.graph_version)")
    f = config.path(latest["dir"]) / "pyg.pt"
    if hashlib.sha256(f.read_bytes()).hexdigest() != latest["sha256"]["pyg.pt"]:
        raise ValueError(f"{f}: sha256 differs from {latest_path}")
    return torch.load(f, map_location="cpu", weights_only=True)


def build_graph(graph_cfg: dict, payload: dict | None = None) -> GraphTensors:
    return graph_tensors(payload if payload is not None else load_payload(), bool(graph_cfg.get("reverse_edges", True)),
                         tuple(graph_cfg.get("drop_types") or ()), graph_cfg.get("hub_cap"))
