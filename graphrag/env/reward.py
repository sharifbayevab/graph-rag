"""Terminal reward R(q, C, C*) = α·R_chunks + β·R_coverage + γ·R_diversity + δ·R_coherence (IMPLEMENTATION.md §7.1;
roadmap §3 item 9). Fragments only -- the generator is never called (§7.2); deterministic (9 decimals).
U11 (step penalty) is M5, not part of this function."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from graphrag import config


@dataclass(frozen=True)
class Weights:
    alpha: float = 0.5
    beta: float = 0.2
    gamma: float = 0.2
    delta: float = 0.1


def weights_from_config(cfg: dict | None = None) -> Weights:
    w = (cfg if cfg is not None else config.load("train"))["reward"]["weights"]
    return Weights(**{k: float(v) for k, v in w.items()})


@dataclass(frozen=True)
class RewardParts:
    chunks: float
    coverage: float
    diversity: float
    coherence: float
    total: float


def covered_nodes(data, rows) -> set[int]:
    out: set[int] = set()
    for c in rows:
        v = int(data.chunk_node[c])
        out.add(v)
        out.update(data.graph.ancestors(v))
    return out


def r_chunks(C: set, G: set) -> float:
    tp = len(C & G)
    if not tp:
        return 0.0
    p, r = tp / len(C), tp / len(G)
    return 2 * p * r / (p + r)


def r_coverage(data, C: set, G: set) -> float:
    rel = covered_nodes(data, G)
    return len(covered_nodes(data, C) & rel) / len(rel) if rel else 0.0


def chunk_depths(data, depth, C, t_max: int) -> list[int]:
    out = []
    for c in C:
        h = int(depth[int(data.chunk_node[c])])
        out.append(t_max if h < 0 or h > t_max else h)
    return out


def r_diversity(depths: list[int], t_max: int) -> float:
    cnt = Counter(depths)
    if len(cnt) <= 1:
        return 0.0
    n = len(depths)
    return -sum((k / n) * math.log(k / n) for k in cnt.values()) / math.log(t_max + 1)


def r_coherence(data, C: set) -> float:
    nodes = sorted({int(data.chunk_node[c]) for c in C})
    if not nodes:
        return 0.0
    inside, parent = set(nodes), {v: v for v in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in nodes:
        for u in data.graph.all_neighbors([v]).tolist():
            if u in inside:
                a, b = find(u), find(v)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return max(Counter(find(v) for v in nodes).values()) / len(nodes)


def reward(data, depth, context, gold, w: Weights, t_max: int) -> RewardParts:
    n = len(data.chunk_node)
    for name, ids in (("context", context), ("gold", gold)):
        bad = [c for c in ids if isinstance(c, bool) or not isinstance(c, (int, np.integer)) or not 0 <= int(c) < n]
        if bad:                                      # -1 padding or floats would silently alias real chunks
            raise ValueError(f"{name} has chunk ids outside [0, {n}) or non-integer: {bad[:5]}")
    C = {int(c) for c in context}
    G = {int(g) for g in gold}
    if not G:
        raise ValueError("empty gold set C*")
    parts = (r_chunks(C, G), r_coverage(data, C, G),
             r_diversity(chunk_depths(data, depth, sorted(C), t_max), t_max), r_coherence(data, C))
    total = w.alpha * parts[0] + w.beta * parts[1] + w.gamma * parts[2] + w.delta * parts[3]
    return RewardParts(*(round(x, 9) for x in parts), round(total, 9))
