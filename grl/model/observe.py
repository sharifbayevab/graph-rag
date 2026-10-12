"""Observable state of an episode for the policy (IMPLEMENTATION.md §6.3, §6.6; roadmap U3, U4; M3 ledger Task 7:
graph candidates include N(V₀), which the frontier does not). Everything here is a function of (q, C_t, G) -- the
target set never enters (plan Review Focus 2). Produced inside the env worker as numpy (picklable), pooled on the
device by grl.model.state.StateEncoder. An ablation env (grl.model.variants.FilteredEnv) exposes `allowed_types`;
candidates of a forbidden edge type are reported empty. The search preview follows the env's search effect
(grl.model.variants.search_preview: M3 next-k fragments, or the AdaptiveEnv threshold rule). A StructEnv (M7) adds the
`structure` operator as one more candidate type after the edge types (id len(edge_types) = 7): its candidates are the
STRUCT rows of StructEnv.struct_scores (Δ_V = struct_delta_v, desc, row asc), reported as their root vertices
(chunk_node of a STRUCT row = its root, μ of the fragment), so the pooled node embedding is the root's; n = number of
STRUCT candidates, best = their best struct_delta_v. With StructEnv.amend_op (Task 6d-b) the `amendments` operator is
one more type after it (id 8): its candidate fragments (StructEnv.amend_scores) reported as their vertices μ(f) (the
AMENDS sources; first occurrence, best first), n = number of candidate fragments, best = their best Δ_V."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grl.model.variants import search_preview

EMPTY = np.zeros(0, np.int64)
N_SCALARS = 6


@dataclass
class StateDesc:
    covered: np.ndarray          # int64 node rows S_t
    frontier: np.ndarray         # int64 node rows F_t
    v0: np.ndarray               # int64 node rows V₀
    cand: list                   # per edge type: int64 candidate rows, best Δ_V first (<= cand_top); StructEnv: + type
                                 # R = `structure` (the root vertices μ(f) of its STRUCT candidates f)
    cand_n: np.ndarray           # float32 [R(+1)] number of candidates per type
    cand_best: np.ndarray        # float32 [R(+1)] best Δ_V per type (0 when none)
    preview: list                # [lexical, vector]: int64 node rows μ(c) of the next fitting fragments
    preview_sim: np.ndarray      # float32 [2] max cos(e_q, e_c) of those fragments (0 when none)
    scalars: np.ndarray          # float32 [6] t/T, |C|/K, len/L, log1p|S|, log1p|F|, log1p|V₀|
    mask: np.ndarray             # bool [A] valid actions (U4)
    hist: np.ndarray             # float32 [A] actions taken so far / T_max


def _rows(s) -> np.ndarray:
    return np.array(sorted(s), np.int64) if s else EMPTY


def describe(env, q, s, preview: int = 3, cand_top: int = 64) -> StateDesc:
    """An env with its own observation (grl.model.rerank.RerankEnv.describe_state) describes itself."""
    own = getattr(env, "describe_state", None)
    if own is not None:
        return own(q, s)
    if s.done:
        raise ValueError("describe() is for live states: the episode is over")
    d, L = env.data, env.limits
    allowed = getattr(env, "allowed_types", None)
    cand, n, best = [], [], []
    for r in range(len(d.graph.edge_types)):
        c, sc = (EMPTY, EMPTY) if allowed is not None and r not in allowed else env.graph_scores(q, s, r)
        if c.size:
            order = np.lexsort((c, -sc))[:cand_top]
            cand.append(np.asarray(c[order], np.int64))
            n.append(c.size)
            best.append(float(sc[order[0]]))
        else:
            cand.append(EMPTY)
            n.append(0)
            best.append(0.0)
    own_struct = getattr(env, "struct_scores", None)          # grl.model.variants.StructEnv: type id 7
    if own_struct is not None:
        rows, sc = own_struct(q, s)                          # ranked: Δ_V desc, row asc
        cand.append(np.asarray(d.chunk_node[rows[:cand_top]], np.int64) if rows.size else EMPTY)
        n.append(rows.size)
        best.append(float(sc[0]) if rows.size else 0.0)
    if getattr(env, "amend_op", False):                       # Task 6d-b: `amendments`, type id 8 (after structure)
        rows, sc = env.amend_scores(q, s)                    # ranked: Δ_V desc, row asc
        top = d.chunk_node[rows[:cand_top]] if rows.size else EMPTY
        cand.append(np.array(list(dict.fromkeys(top.tolist())), np.int64) if rows.size else EMPTY)
        n.append(rows.size)
        best.append(float(sc[0]) if rows.size else 0.0)
    prev, psim = [], []
    for op in ("lexical", "vector"):
        rows = search_preview(env, q, s, op, preview)
        prev.append(np.unique(d.chunk_node[rows]).astype(np.int64) if rows else EMPTY)
        psim.append(float(np.max(q.chunk_qsim[rows])) if rows else 0.0)
    hist = np.zeros(len(env.actions), np.float32)
    for h in s.history:
        hist[h.action] += 1.0 / L.T_max
    scalars = np.array([s.t / L.T_max, len(s.context) / L.K_max, s.length / L.L_max, np.log1p(len(s.covered)),
                        np.log1p(len(s.frontier)), np.log1p(len(q.v0))], np.float32)
    return StateDesc(_rows(s.covered), _rows(s.frontier), np.asarray(q.v0, np.int64), cand,
                     np.array(n, np.float32), np.array(best, np.float32), prev, np.array(psim, np.float32),
                     scalars, env.mask(q, s), hist)
