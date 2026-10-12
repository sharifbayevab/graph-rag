"""The environment's view of the pinned knowledge graph (IMPLEMENTATION.md §3.3, §5): rows of pyg.pt, μ (chunk_node)
and μ⁻¹ (CSR, span-overlap definition, roadmap §3 item 1), fragment lengths, the frozen bge-m3 fragment vectors
(= the Qdrant vectors, index_text) and vertex features X. Read-only, offline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from graphrag import config
from graphrag.retrieval.graph_walk import ArtifactGraph


@dataclass
class EnvData:
    version: str
    node_ids: list[str]
    chunk_ids: list[str]
    node_row: dict[str, int]
    chunk_row: dict[str, int]
    graph: ArtifactGraph
    chunk_node: np.ndarray        # int64 [C]  μ
    mu_ptr: np.ndarray            # int64 [N+1]
    mu_idx: np.ndarray            # int64 [nnz]
    chunk_tokens: np.ndarray      # int32 [C]
    chunk_vecs: np.ndarray        # float32 [C, d]
    node_x: np.ndarray            # float32 [N, d]

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @cached_property
    def vecs64(self) -> np.ndarray:
        return self.chunk_vecs.astype(np.float64)

    @cached_property
    def x64(self) -> np.ndarray:
        return self.node_x.astype(np.float64)

    def mu_inv(self, v: int) -> np.ndarray:
        return self.mu_idx[self.mu_ptr[v]:self.mu_ptr[v + 1]]

    @classmethod
    def build(cls, node_ids, chunk_ids, edges, chunk_parent, mu_inv, tokens, chunk_vecs, node_x, edge_types,
              version: str = "toy") -> "EnvData":
        if list(node_ids) != sorted(node_ids) or list(chunk_ids) != sorted(chunk_ids):
            raise ValueError("node_ids / chunk_ids must be sorted (pyg.pt row order)")
        node_row = {n: i for i, n in enumerate(node_ids)}
        chunk_row = {c: i for i, c in enumerate(chunk_ids)}
        tid = {t: i for i, t in enumerate(edge_types)}
        ei = np.array([[node_row[s] for s, _, _ in edges], [node_row[d] for _, _, d in edges]], np.int64).reshape(2, -1)
        et = np.array([tid[t] for _, t, _ in edges], np.int64)
        ptr, idx = [0], []
        for n in node_ids:
            idx += sorted(chunk_row[c] for c in mu_inv.get(n, []))
            ptr.append(len(idx))
        return cls(version, list(node_ids), list(chunk_ids), node_row, chunk_row,
                   ArtifactGraph(len(node_ids), ei, et, list(edge_types)),
                   np.array([node_row[chunk_parent[c]] for c in chunk_ids], np.int64),
                   np.array(ptr, np.int64), np.array(idx, np.int64),
                   np.array([tokens[c] for c in chunk_ids], np.int32),
                   np.asarray(chunk_vecs, np.float32), np.asarray(node_x, np.float32))


def load_env_data(version: str | None = None, latest_path: str = "data/graph/latest.json") -> EnvData:
    import torch
    from graphrag.data.ingest import read_records
    from graphrag.kg.build import _cached_vectors, index_text_digest
    latest = json.loads(config.path(latest_path).read_text(encoding="utf-8"))
    want = version or config.load("train")["env"]["graph_version"]
    if latest["version"] != want:
        raise ValueError(f"graph {latest['version']} in {latest_path} != pinned {want} (train.yaml env.graph_version)")
    d = config.path(latest["dir"])
    for f in ("pyg.pt", "x.npy"):
        if hashlib.sha256((d / f).read_bytes()).hexdigest() != latest["sha256"][f]:
            raise ValueError(f"{d / f}: sha256 differs from {latest_path}")
    p = torch.load(d / "pyg.pt", weights_only=True)
    records = read_records(config.path(config.load("data")["records_dir"]))
    tok = {c.chunk_id: c.tokens for r in records for c in r.chunks}
    chunk_ids = list(p["chunk_ids"])
    if sorted(tok) != chunk_ids:
        raise ValueError("record chunk ids differ from pyg.pt chunk_ids: rebuild the graph from these records")
    vec = _cached_vectors(config.path("data/graph/chunk_vecs.npy"), chunk_ids, index_text_digest(records),
                          "index_text vectors")
    node_ids = list(p["node_ids"])
    return EnvData(want, node_ids, chunk_ids, {n: i for i, n in enumerate(node_ids)},
                   {c: i for i, c in enumerate(chunk_ids)},
                   ArtifactGraph(len(node_ids), p["edge_index"].numpy(), p["edge_type"].numpy(), list(p["edge_types"])),
                   p["chunk_node"].numpy().astype(np.int64), p["mu_inv_ptr"].numpy().astype(np.int64),
                   p["mu_inv_idx"].numpy().astype(np.int64), np.array([tok[c] for c in chunk_ids], np.int32),
                   np.stack([vec[c] for c in chunk_ids]).astype(np.float32), p["x"].numpy().astype(np.float32))
