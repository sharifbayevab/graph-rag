"""M7 Task 6d-a: the Q2 router branch «amendments of the named act» (hand-written rule, no training).

Rule (no gold, no labels): the named act is the FIRST root in V0 (V0 is in score order) that has incoming AMENDS
edges. The candidates are the fragments (μ⁻¹) of the source vertices of those edges; they are ranked by the
query-fragment cosine the env already holds (q.chunk_qsim, the dense / fused vector score) and the first <= K that fit
L_max are returned. With no such root the caller's fallback (the struct agent's context) is returned unchanged.

R60: the AMENDS edges' evidence spans are NOT read here (the R33 Q2 gold is built from them: using them in the
ranking would be circular). build_amend_index reads only (src, dst) of AMENDS edges. The evidence variant is a labelled
diagnostic in scripts/m7_router_q2rule.py, never in the headline."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


class AmendIndex:
    """root node row -> source vertex rows of its incoming AMENDS edges (edge order, deduplicated), plus the env's
    vertex -> root map and μ⁻¹."""

    def __init__(self, root_row, sources: dict[int, list[int]], mu_inv):
        self.root_row = np.asarray(root_row, np.int64)
        self.sources = {int(r): [int(v) for v in vs] for r, vs in sources.items()}
        self.mu_inv = mu_inv


def load_amends_pairs(data) -> list[tuple[str, str]]:
    """(src node_id, dst node_id) of every AMENDS edge of the pinned graph. Only the two ids are kept."""
    from graphrag import config
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    if latest["version"] != data.version:
        raise ValueError(f"graph {latest['version']} != EnvData {data.version}")
    out = []
    with open(config.path(latest["dir"]) / "edges.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["type"] == "AMENDS":
                    out.append((e["src"], e["dst"]))
    return out


def build_amend_index(data, root_row, pairs) -> AmendIndex:
    src: dict[int, list[int]] = defaultdict(list)
    for s, d in pairs:
        sv, dv = data.node_row.get(s), data.node_row.get(d)
        if sv is None or dv is None:
            continue
        r = int(root_row[dv])
        if r >= 0 and int(sv) not in src[r]:
            src[r].append(int(sv))
    return AmendIndex(root_row, src, data.mu_inv)


def named_act_root(index: AmendIndex, v0) -> int:
    """First root of V0 (in V0 order) with incoming AMENDS, or -1."""
    for v in v0:
        r = int(index.root_row[int(v)])
        if r >= 0 and r in index.sources:
            return r
    return -1


def amend_lookup(index: AmendIndex, q, chunk_tokens, k: int = 5, l_max: int = 8000, fallback=()) -> list[int]:
    """<= k fragment rows ranked by q.chunk_qsim; reads only q.v0 and q.chunk_qsim. Deterministic (ties by row)."""
    r = named_act_root(index, q.v0)
    if r < 0:
        return [int(c) for c in fallback]
    cands = {int(c) for sv in index.sources[r] for c in index.mu_inv(sv)}
    qs = q.chunk_qsim
    out, ln = [], 0
    for c in sorted(cands, key=lambda c: (-float(qs[c]), c)):
        if len(out) >= k:
            break
        if ln + int(chunk_tokens[c]) > l_max:
            continue
        out.append(c)
        ln += int(chunk_tokens[c])
    return out
