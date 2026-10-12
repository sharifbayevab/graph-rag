"""Start point V₀ and the per-query context of an episode (IMPLEMENTATION.md §4.4; roadmap §3 item 7).
Computed once per query (§6.5): cosines, vertex scores, V₀, BFS depths from V₀ (R_diversity)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphrag.retrieval.rankings import QueryRanking


@dataclass
class QueryCtx:
    qkey: str
    text: str
    qvec: np.ndarray
    lex: np.ndarray            # chunk rows ranked by BM25 (U9)
    vec: np.ndarray            # chunk rows ranked by cosine (U9)
    chunk_qsim: np.ndarray     # float64 [C]
    node_sem: np.ndarray       # float64 [N], -inf when μ⁻¹(v) = ∅
    node_rel: np.ndarray       # float64 [N]
    node_lex: np.ndarray       # float64 [N]
    v0: np.ndarray             # int64 node rows, ranked
    depth: np.ndarray          # int16 [N]


def node_max(ptr: np.ndarray, idx: np.ndarray, vals, empty: float = -np.inf) -> np.ndarray:
    out = np.full(len(ptr) - 1, empty, np.float64)
    nz = np.flatnonzero(np.diff(ptr) > 0)
    if nz.size:
        out[nz] = np.maximum.reduceat(np.asarray(vals, np.float64)[idx], ptr[nz])
    return out


def select_v0(node_lex, node_sem, tree_depth, tau_lex: float, tau_sem: float, cap: int,
              decimals: int = 6) -> np.ndarray:
    sem = np.where(np.isfinite(node_sem), node_sem, -1.0)
    score = np.round(np.maximum(node_lex, sem), decimals)
    ok = np.flatnonzero((node_lex > tau_lex) | (sem > tau_sem))
    n = cap
    if ok.size == 0:
        ok, n = np.arange(score.size), 1
    order = np.lexsort((ok, -np.asarray(tree_depth, np.int64)[ok], -score[ok]))
    return ok[order][:n].astype(np.int64)


def make_query(data, rank: QueryRanking, start_cfg: dict | None, t_max: int, decimals: int = 6,
               v0: np.ndarray | None = None) -> QueryCtx:
    qv = np.asarray(rank.qvec, np.float64)
    qsim = np.round(data.vecs64 @ qv, decimals)
    sem = node_max(data.mu_ptr, data.mu_idx, qsim)
    rel = np.where(np.isfinite(sem), sem, np.round(data.x64 @ qv, decimals))
    chunk_lex = np.zeros(data.n_chunks)
    if rank.lex.size and np.nanmax(rank.lex_score) > 0:
        chunk_lex[rank.lex] = rank.lex_score / np.nanmax(rank.lex_score)
    lex = node_max(data.mu_ptr, data.mu_idx, chunk_lex, empty=0.0)
    if rank.title.size and np.nanmax(rank.title_score) > 0:
        t = np.zeros(data.n_nodes)
        t[rank.title] = rank.title_score / np.nanmax(rank.title_score)
        lex = np.maximum(lex, t)
    lex = np.round(lex, decimals)
    if v0 is None:
        v0 = select_v0(lex, sem, data.graph.tree_depth, float(start_cfg["tau_lex"]), float(start_cfg["tau_sem"]),
                       int(start_cfg["cap"]), decimals)
    v0 = np.asarray(v0, np.int64)
    return QueryCtx(rank.qkey, rank.text, rank.qvec, np.asarray(rank.lex, np.int64), np.asarray(rank.vec, np.int64),
                    qsim, sem, rel, lex, v0, data.graph.depth_from(v0, t_max))


def query_from_vector(data, qkey: str, qvec, v0, t_max: int = 10) -> QueryCtx:
    return make_query(data, QueryRanking.empty(qkey, qvec), None, t_max, v0=np.asarray(v0, np.int64))


def v0_hit(data, v0, gold) -> bool:
    s = {int(v) for v in v0}
    for c in gold:
        v = int(data.chunk_node[c])
        if v in s or int(data.graph.parent[v]) in s:
            return True
    return False


def calibrate_start(data, ranks, golds, splits, grid_lex, grid_sem, cap: int, t_max: int = 10) -> dict:
    """τ maximizing the TRAIN hit rate of V₀; ties: smaller mean |{v passing}|, then larger τ_lex, τ_sem (§9.1)."""
    splits = list(splits)
    if not (len(splits) == len(ranks) == len(golds) > 0):
        raise ValueError("τ calibration needs one split label and one gold set per query, and ≥ 1 query")
    if any(s != "train" for s in splits):
        raise ValueError("τ calibration reads the train split only (IMPLEMENTATION.md §9.1)")
    if cap < 1:
        raise ValueError("V₀ cap must be ≥ 1")
    pre = [make_query(data, r, None, t_max, v0=np.zeros(0, np.int64)) for r in ranks]
    rows = []
    for tl in grid_lex:
        for ts in grid_sem:
            hits, sizes = [], []
            for q, g in zip(pre, golds):
                sem = np.where(np.isfinite(q.node_sem), q.node_sem, -1.0)
                sizes.append(int(((q.node_lex > tl) | (sem > ts)).sum()))
                hits.append(v0_hit(data, select_v0(q.node_lex, q.node_sem, data.graph.tree_depth, tl, ts, cap), g))
            rows.append({"tau_lex": tl, "tau_sem": ts, "hit_rate": round(float(np.mean(hits)), 4),
                         "mean_passing": round(float(np.mean(sizes)), 2)})
    best = max(rows, key=lambda r: (r["hit_rate"], -r["mean_passing"], r["tau_lex"], r["tau_sem"]))
    return {"grid": rows, "best": best, "n_queries": len(pre), "cap": cap}
