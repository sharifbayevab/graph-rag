"""Graph access of the M3 operators (IMPLEMENTATION.md §4.3; roadmap §3 item 8): typed neighbours in both
directions, CONTAINS parents / ancestors, BFS depth. ArtifactGraph reads the pinned PyG artifact (rows = sorted
node_id); Neo4jGraph (Task 4) answers the neighbour question from the dev Neo4j active version."""
from __future__ import annotations

import numpy as np

EMPTY = np.zeros(0, np.int64)


def _csr(n: int, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, de-duplicated adjacency a -> b without self loops as (ptr [n+1], idx)."""
    keep = a != b
    a, b = a[keep], b[keep]
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    if a.size:
        u = np.ones(a.size, bool)
        u[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
        a, b = a[u], b[u]
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, a + 1, 1)
    return np.cumsum(ptr), b.astype(np.int64)


class ArtifactGraph:
    def __init__(self, n_nodes: int, edge_index, edge_type, edge_types: list[str]):
        self.n = int(n_nodes)
        self.edge_types = list(edge_types)
        src = np.asarray(edge_index[0], np.int64)
        dst = np.asarray(edge_index[1], np.int64)
        et = np.asarray(edge_type, np.int64)
        self._typed = [_csr(self.n, np.concatenate([src[et == t], dst[et == t]]),
                            np.concatenate([dst[et == t], src[et == t]])) for t in range(len(self.edge_types))]
        self._all = _csr(self.n, np.concatenate([src, dst]), np.concatenate([dst, src]))
        parent = np.full(self.n, -1, np.int64)
        if "CONTAINS" in self.edge_types:
            m = et == self.edge_types.index("CONTAINS")
            if np.unique(dst[m]).size != int(m.sum()):
                raise ValueError("CONTAINS is not a forest: a vertex has two parents")
            parent[dst[m]] = src[m]
        self.parent = parent
        depth = np.zeros(self.n, np.int16)
        for v in range(self.n):
            u, h = v, 0
            while parent[u] >= 0:
                u, h = parent[u], h + 1
            depth[v] = h
        self.tree_depth = depth

    def type_id(self, r: str) -> int:
        return self.edge_types.index(r)

    @staticmethod
    def _gather(ptr: np.ndarray, idx: np.ndarray, rows) -> np.ndarray:
        parts = [idx[ptr[v]:ptr[v + 1]] for v in np.asarray(rows, np.int64).reshape(-1)]
        return np.unique(np.concatenate(parts)) if parts else EMPTY

    def neighbors(self, rows, r: int) -> np.ndarray:
        return self._gather(*self._typed[r], rows)

    def all_neighbors(self, rows) -> np.ndarray:
        return self._gather(*self._all, rows)

    def ancestors(self, v: int) -> list[int]:
        out, u = [], int(self.parent[v])
        while u >= 0:
            out.append(u)
            u = int(self.parent[u])
        return out

    def depth_from(self, sources, max_depth: int) -> np.ndarray:
        d = np.full(self.n, -1, np.int16)
        front = np.unique(np.asarray(sources, np.int64).reshape(-1))
        d[front] = 0
        for h in range(1, max_depth + 1):
            if front.size == 0:
                break
            nb = self.all_neighbors(front)
            nb = nb[d[nb] < 0]
            d[nb] = h
            front = nb
        return d

    def degree(self) -> np.ndarray:
        return np.diff(self._all[0])


class Neo4jGraph:
    """Neighbours from the dev Neo4j (R27 layout): corpus CONTAINS unversioned (no r.graph); every other edge
    carries r.graph / r.graph_version; RELATED stored once (src < dst) -- the undirected match covers both."""

    def __init__(self, store, version: str, graph: str = "main"):
        from graphrag.stores.guard import dev_neo4j_problem
        p = dev_neo4j_problem(store.uri)
        if p:
            raise PermissionError(f"graph_walk reads only the dev Neo4j: {p}")
        active = store.active_version(graph)
        if active != version:
            raise ValueError(f"Neo4j active version {active} != pinned graph version {version}")
        self.store, self.version, self.graph = store, version, graph

    def neighbors_ids(self, node_id: str, r: str) -> list[str]:
        if not (r.replace("_", "").isalpha() and r.isupper()):
            raise ValueError(f"bad edge type {r!r}")
        q = (f"MATCH (a:Node {{node_id: $id}})-[e:{r}]-(b:Node) "
             "WHERE b <> a AND (e.graph IS NULL OR (e.graph = $g AND e.graph_version = $v)) "
             "RETURN DISTINCT b.node_id AS id")
        with self.store.driver.session() as s:
            return sorted(x["id"] for x in s.run(q, id=node_id, g=self.graph, v=self.version))


def neighbor_parity(ag: ArtifactGraph, node_ids: list[str], neo: Neo4jGraph, rows) -> list[str]:
    problems = []
    for v in rows:
        for t, r in enumerate(ag.edge_types):
            a = sorted(node_ids[u] for u in ag.neighbors([v], t).tolist())
            b = neo.neighbors_ids(node_ids[v], r)
            if a != b:
                problems.append(f"{node_ids[v]} {r}: artifact {len(a)} vs neo4j {len(b)}; only artifact "
                                f"{sorted(set(a) - set(b))[:3]}, only neo4j {sorted(set(b) - set(a))[:3]}")
    return problems
