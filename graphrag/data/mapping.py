"""μ: chunk → node and μ⁻¹: node → chunks (spec §3.3), both by char-span overlap."""
from __future__ import annotations

import bisect

from graphrag.types import Chunk, Node


def mu(chunks: list[Chunk]) -> dict[str, str]:
    return {c.chunk_id: c.parent for c in chunks}


def mu_inverse(nodes: list[Node], chunks: list[Chunk]) -> dict[str, list[str]]:
    cs = sorted(chunks, key=lambda c: c.start)
    starts = [c.start for c in cs]
    out: dict[str, list[str]] = {}
    for n in nodes:
        i = max(bisect.bisect_right(starts, n.start) - 1, 0)
        ids = []
        while i < len(cs) and cs[i].start < n.end:
            if cs[i].end > n.start:
                ids.append(cs[i].chunk_id)
            i += 1
        out[n.node_id] = ids
    return out
