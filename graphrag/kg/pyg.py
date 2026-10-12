"""PyG export of the knowledge graph (M2 Task 9, Ruling R24 (3)/(4); consumed by the M4 GAT, IMPLEMENTATION.md §6).

    to_pyg(graph, X) -> torch_geometric.data.Data with
      x            float32 [N, d]      node features (graph.nodes order = sorted node_id)
      edge_index   int64   [2, E']     edges as stored (directed src -> dst); RELATED in BOTH directions
      edge_type    int64   [E']        graph.yaml pinned edge ids (CONTAINS 0 … RELATED 6)
      edge_weight  float32 [E']        RELATED: cosine (attrs.weight); every other type 1.0
      edge_attached bool   [E']        RELATED added by the isolated-vertex step (attrs.attached)
      node_ids     list[str] [N];  node_kind int64 [N] (index into node_kinds = graph.yaml vertex_kinds)
      chunk_ids    list[str] [C]  (sorted);  chunk_node int64 [C]  μ: chunk -> node row (Chunk.parent)
      mu_inv_ptr   int64 [N + 1], mu_inv_idx int64 [nnz]   μ⁻¹ CSR: chunks of node i are
                   chunk_ids[mu_inv_idx[mu_inv_ptr[i]:mu_inv_ptr[i + 1]]] (graphrag.data.mapping.mu_inverse,
                   span overlap; empty for stubs / external acts / entities / concepts)
      edge_types, node_kinds  list[str]   the id -> name tables
    Ruling R31 (4), additive fields for M3/M4 (vocabularies below, also listed in stats.json pyg.vocabularies):
      edge_attr_relation int64 [E']    REFERS_TO attrs.relation -> index into edge_relations (graph.yaml
                                        ref_relations order); -1 for every other type
      edge_attr_op       int64 [E']    AMENDS attrs.op (primary op) -> index into amend_ops; -1 otherwise
      edge_attr_ops      bool  [E', len(amend_ops)]  AMENDS multi-hot of attrs.ops (or [op]); all False otherwise
      edge_date          int64 [E']    attrs.date (AMENDS / REPEALS, ISO) as days since 1970-01-01; -1 when absent
      node_type          int64 [N]     fine type of document / element vertices -> index into node_types
                                        ("element:<type>", "document:<attrs.doc_type>", "document:external");
                                        -1 for entities / concepts
      entity_type        int64 [N]     entity type -> index into entity_types (graph.yaml vertex_types.entity);
                                        -1 for every non-entity
      is_external        bool  [N]     GraphNode.external (EXT acts and their element stubs)
      node_doc           int64 [N]     the vertex's document -> index into doc_ids (all document vertices, sorted):
                                        a document itself, a corpus element its doc_id, an EXT stub its attrs.act;
                                        -1 for entities / concepts
      edge_relations, amend_ops, node_types, entity_types, doc_ids   list[str]
Reverse relations of the directed types are NOT added here: the M4 GAT decides (e.g. ToUndirected /
per-type reverse edges as a transform). Edge order: by (edge_type, src row, dst row).

The file (pyg.pt) is a plain dict of tensors / lists (torch.load(weights_only=True) safe); load_pyg rebuilds
the Data object."""
from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path

import numpy as np
import torch

from graphrag.kg.schema import Schema

LIST_FIELDS = ("node_ids", "chunk_ids", "edge_types", "node_kinds", "edge_relations", "amend_ops", "node_types",
               "entity_types", "doc_ids")
EPOCH = _dt.date(1970, 1, 1)


def days_since_epoch(iso: str | None) -> int:
    if not iso:
        return -1
    return (_dt.date.fromisoformat(iso) - EPOCH).days


def node_type_name(n) -> str | None:
    if n.kind == "element":
        return f"element:{n.type}"
    if n.kind == "document":
        return "document:external" if n.external else f"document:{(n.attrs or {}).get('doc_type') or n.type}"
    return None


def node_document(n) -> str | None:
    if n.kind == "document":
        return n.node_id
    if n.kind == "element":
        return (n.attrs or {}).get("act") if n.external else n.doc_id
    return None


def pyg_payload(graph, X: np.ndarray, schema: Schema | None = None) -> dict:
    schema = schema or graph.schema or Schema.load()
    ids = [n.node_id for n in graph.nodes]
    if ids != sorted(ids):
        raise ValueError("graph.nodes must be sorted by node_id")
    if X.shape[0] != len(ids):
        raise ValueError(f"X has {X.shape[0]} rows for {len(ids)} nodes")
    pos = {nid: i for i, nid in enumerate(ids)}
    relations = list(schema.cfg.get("ref_relations") or [])
    ops = list(schema.cfg.get("amend_ops") or [])
    rel_id = {r: i for i, r in enumerate(relations)}
    op_id = {o: i for i, o in enumerate(ops)}
    rows = []
    for e in graph.edges:
        t = schema.edge_type_id[e.type]
        w = float(e.attrs["weight"]) if e.type == "RELATED" else 1.0
        att = bool(e.attrs.get("attached")) if e.type == "RELATED" else False
        rel = rel_id[e.attrs["relation"]] if e.type == "REFERS_TO" else -1
        op = op_id[e.attrs["op"]] if e.type == "AMENDS" else -1
        hot = tuple(sorted({op_id[o] for o in (e.attrs.get("ops") or [e.attrs["op"]])})) if e.type == "AMENDS" else ()
        date = days_since_epoch(e.attrs.get("date")) if e.type in ("AMENDS", "REPEALS") else -1
        s, d = pos[e.src], pos[e.dst]
        rows.append((t, s, d, w, att, rel, op, hot, date))
        if e.type == "RELATED":
            rows.append((t, d, s, w, att, rel, op, hot, date))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    kinds = list(schema.cfg["vertex_kinds"])
    kind_id = {k: i for i, k in enumerate(kinds)}
    ridx = graph.rindex
    chunk_ids = list(ridx.chunk_ids) if ridx is not None else []
    cpos = {c: i for i, c in enumerate(chunk_ids)}
    ptr, idx = [0], []
    for nid in ids:
        n = graph.nodes[pos[nid]]
        cs = ridx.mu_inv.get(nid, []) if (ridx is not None and not n.external) else []
        idx += sorted(cpos[c] for c in cs)
        ptr.append(len(idx))
    chunk_node = [pos[ridx.chunk_parent[c]] for c in chunk_ids] if ridx is not None else []
    E = len(rows)
    ops_hot = torch.zeros((E, len(ops)), dtype=torch.bool)
    for k, r in enumerate(rows):
        for j in r[7]:
            ops_hot[k, j] = True
    ntypes = sorted({x for x in (node_type_name(n) for n in graph.nodes) if x is not None})
    ntype_id = {x: i for i, x in enumerate(ntypes)}
    etypes = list((schema.cfg.get("vertex_types") or {}).get("entity") or [])     # graph.yaml order
    etype_id = {x: i for i, x in enumerate(etypes)}
    doc_ids = [n.node_id for n in graph.nodes if n.kind == "document"]          # sorted: graph.nodes is
    doc_id_ix = {x: i for i, x in enumerate(doc_ids)}
    return {
        "x": torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        "edge_index": torch.tensor([[r[1] for r in rows], [r[2] for r in rows]], dtype=torch.long).reshape(2, E),
        "edge_type": torch.tensor([r[0] for r in rows], dtype=torch.long),
        "edge_weight": torch.tensor([r[3] for r in rows], dtype=torch.float32),
        "edge_attached": torch.tensor([r[4] for r in rows], dtype=torch.bool),
        "node_kind": torch.tensor([kind_id[n.kind] for n in graph.nodes], dtype=torch.long),
        "mu_inv_ptr": torch.tensor(ptr, dtype=torch.long),
        "mu_inv_idx": torch.tensor(idx, dtype=torch.long),
        "chunk_node": torch.tensor(chunk_node, dtype=torch.long),
        "node_ids": ids, "chunk_ids": chunk_ids,
        "edge_types": list(schema.edge_types), "node_kinds": kinds,
        # Ruling R31 (4)
        "edge_attr_relation": torch.tensor([r[5] for r in rows], dtype=torch.long),
        "edge_attr_op": torch.tensor([r[6] for r in rows], dtype=torch.long),
        "edge_attr_ops": ops_hot,
        "edge_date": torch.tensor([r[8] for r in rows], dtype=torch.long),
        "node_type": torch.tensor([ntype_id.get(node_type_name(n), -1) if node_type_name(n) else -1
                                   for n in graph.nodes], dtype=torch.long),
        "entity_type": torch.tensor([etype_id[n.type] if n.kind == "entity" else -1 for n in graph.nodes],
                                    dtype=torch.long),
        "is_external": torch.tensor([bool(n.external) for n in graph.nodes], dtype=torch.bool),
        "node_doc": torch.tensor([doc_id_ix[node_document(n)] if node_document(n) else -1 for n in graph.nodes],
                                 dtype=torch.long),
        "edge_relations": relations, "amend_ops": ops, "node_types": ntypes, "entity_types": etypes,
        "doc_ids": doc_ids,
    }


def _data(payload: dict):
    from torch_geometric.data import Data
    d = Data(**payload)
    d.num_nodes = len(payload["node_ids"])
    return d


def to_pyg(graph, X: np.ndarray, schema: Schema | None = None):
    return _data(pyg_payload(graph, X, schema))


def pyg_bytes(payload: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def load_pyg(path: Path | str):
    payload = torch.load(Path(path), weights_only=True)
    return _data(payload)
