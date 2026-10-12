"""Operator executor (IMPLEMENTATION.md §5.1, §5.3; roadmap §3 item 8): an operator -> the fragments it adds (ΔC)
and the vertices it selects. Deterministic and trained by nothing (§12 item 9): scores rounded, ties by row.
Never exceeds K_max fragments or L_max tokens."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Budget:
    n_ctx: int
    length: int
    K_max: int
    L_max: int


def take_fitting(rows, exclude, k: int, budget: Budget, tokens: np.ndarray) -> list[int]:
    out: list[int] = []
    length, n = budget.length, budget.n_ctx
    for c in rows:
        c = int(c)
        if len(out) >= k or n >= budget.K_max:
            break
        if c < 0 or c in exclude or c in out:
            continue
        t = int(tokens[c])
        if length + t > budget.L_max:
            continue
        out.append(c)
        length += t
        n += 1
    return out


def ranked_delta(ranking, ctx_set, k: int, budget: Budget, tokens) -> list[int]:
    return take_fitting(ranking, ctx_set, k, budget, tokens)


def _arr(s) -> np.ndarray:
    return np.fromiter(s, np.int64, len(s)) if s else np.zeros(0, np.int64)


def graph_candidates(data, q, covered, r: int) -> np.ndarray:
    nb = data.graph.neighbors(np.union1d(_arr(covered), np.asarray(q.v0, np.int64)), r)
    return nb[~np.isin(nb, _arr(covered))] if covered else nb


class ContextSims:
    """Per-query cache of cos(e_c, e_c') for every fragment c against each context fragment c' seen so far (Task 13).
    A column is `vecs64 @ vecs64[c']` -- one fixed-shape product per c', so its values do not depend on which other
    columns exist or on the order of the calls (the cache is transparent). A query touches only tens of distinct
    context fragments, so the whole beam search costs a few dozen columns instead of a matmul per Δ_V call."""

    def __init__(self, data, cap: int = 64):
        self.data = data
        self.slot: dict[int, int] = {}
        self.m = np.empty((data.n_chunks, cap), np.float64)

    def slots(self, context) -> np.ndarray:
        out = []
        for c in context:
            c = int(c)
            j = self.slot.get(c)
            if j is None:
                j = len(self.slot)
                if j == self.m.shape[1]:
                    self.m = np.concatenate([self.m, np.empty_like(self.m)], axis=1)
                self.m[:, j] = self.data.vecs64 @ self.data.vecs64[c]
                self.slot[c] = j
            out.append(j)
        return np.asarray(out, np.int64)


def delta_v(data, q, cand: np.ndarray, context, lambda_red: float, decimals: int,
            sims: ContextSims | None = None) -> np.ndarray:
    """Δ_V(v|C) = rel(v) − λ_red · max_{c∈μ⁻¹(v), c'∈C} cos(e_c, e_c'): one product over the concatenated μ⁻¹ of all
    candidates, then a segment max (np.maximum.reduceat); chunkless vertices have no redundancy term (Task 13: no
    Python loop over the vertices). `sims` (the Env's per-query ContextSims) replaces the product by a gather from
    cached columns. Both equal the per-vertex reference up to float64 summation order (tested on the corpus)."""
    cand = np.asarray(cand, np.int64)
    rel = np.asarray(q.node_rel, np.float64)[cand]
    red = np.zeros(len(cand))
    if len(context) and lambda_red and cand.size:
        lo = data.mu_ptr[cand]
        lens = data.mu_ptr[cand + 1] - lo
        nz = np.flatnonzero(lens > 0)
        if nz.size:
            ln = lens[nz]
            starts = np.concatenate(([0], np.cumsum(ln)[:-1]))
            idx = data.mu_idx[np.repeat(lo[nz] - starts, ln) + np.arange(int(ln.sum()))]
            if sims is None:
                m = (data.vecs64[idx] @ data.vecs64[list(context)].T).max(axis=1)
            else:
                j = sims.slots(context)                  # first: it may grow sims.m
                m = sims.m[np.ix_(idx, j)].max(axis=1)
            red[nz] = np.maximum.reduceat(m, starts)
    return np.round(rel - lambda_red * red, decimals)


def graph_delta(data, q, covered, context, ctx_set, r: int, m: int, budget: Budget, per_vertex: int,
                lambda_red: float, decimals: int, cand: np.ndarray | None = None,
                score: np.ndarray | None = None) -> tuple[list[int], list[int]]:
    """`cand` / `score` (Task 13): the precomputed graph_candidates(covered, r) and their delta_v(context) -- the
    Env's per-query memo passes them; None computes them here, as before."""
    if cand is None:
        cand = graph_candidates(data, q, covered, r)
    if cand.size == 0:
        return [], []
    if score is None:
        score = delta_v(data, q, cand, context, lambda_red, decimals)
    sel = [int(v) for v in cand[np.lexsort((cand, -score))][:m]]
    added: list[int] = []
    for v in sel:
        cs = sorted((int(c) for c in data.mu_inv(v)), key=lambda c: (-float(q.chunk_qsim[c]), c))
        used = int(data.chunk_tokens[added].sum()) if added else 0
        added += take_fitting(cs, set(ctx_set) | set(added), per_vertex,
                              Budget(budget.n_ctx + len(added), budget.length + used, budget.K_max, budget.L_max),
                              data.chunk_tokens)
    return sel, added
