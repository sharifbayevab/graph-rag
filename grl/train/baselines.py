"""Fixed and oracle reference contexts for M6 (IMPLEMENTATION.md §9.3–9.4): BM25 / dense / RRF top-k under the same
K_max / L_max budget (first k fitting fragments of the U9 ranking), the M3 greedy and beam oracles (know C*: upper
bounds), the uniform random policy, and the fixed graph heuristic of §9.3 ("графовое с эвристикой": lexical k=5 ->
graph(REFERS_TO, m=3) -> graph(CONTAINS, m=3) -> STOP; a masked step is skipped; no C*) and the M5 hybrid heuristic
(lexical k=5 -> vector k=5 -> the best-scoring valid graph action -> STOP; masked steps skipped; no C*).
Names: bm25@k, dense@k, rrf@k, graph_heuristic, hybrid_heuristic, greedy_oracle, beam_oracle, random@seed.
M5 "dynamic k" (no learning, no C*, deterministic, inside K_max / L_max): rel:<kind>@p (threshold_take on the
<kind> ranking, scores min-max normalised over its top norm_top_n: fragments while score ≥ p·top score),
rel_raw:<kind>@p (the same on raw scores), elbow:<kind> (min-max is affine: the cut is the same on raw scores) (cut at the largest score gap of the first
ELBOW_N fitting fragments), agree@N (fragments in both BM25 top-N and dense top-N, RRF order); <kind> = bm25 (U9 BM25
score) | dense (cosine) | rrf (RRF score, k = 60) | cc<w> (convex fusion w·mm(cos) + (1−w)·mm(BM25),
grl.model.variants.fused_scores; rel: only). oracle_switch:dense|cc<w>@p is an ORACLE (reads C*): per query the better
reward of rel:dense@p and rel:cc<w>@p -- the upper bound of a per-query dense / fused choice, not a deployable
baseline (its row carries oracle: true). Every reference row also reports recall@<B>tok: the recall of the longest
prefix of the context (in its order) within B tokens (TOKEN_BUDGETS). recall_curves: recall / reward of bm25 / dense / rrf @ k = 1..K_max."""
from __future__ import annotations

import numpy as np

from graphrag.env.policies import BeamSearch, greedy_oracle, random_policy
from graphrag.env.policies import rollout as m3_rollout
from graphrag.retrieval.executor import Budget, take_fitting
from collections import defaultdict

from graphrag.retrieval.rankings import rrf_rows
from grl.model.variants import (NORM_TOP_N, AdaptiveEnv, FilteredEnv, StructEnv, fused_scores, minmax_scores,
                                search_scores, threshold_take, with_scores)
from grl.train.data import guard_qkeys
from grl.train.evaluate import context_metrics
from grl.train.fbeta import grl_reward

HEURISTIC_PLAN = (("lexical", None, 5), ("graph", "REFERS_TO", 3), ("graph", "CONTAINS", 3))   # (op, r, k | m)


def _plan_ids(env) -> list[int]:
    ids = []
    for op, r, n in HEURISTIC_PLAN:
        ids += [i for i, a in enumerate(env.actions)
                if a.op == op and a.r == r and (a.m if op == "graph" else a.k) == n]
    return ids


def graph_heuristic_policy(env, q, s) -> int:
    """Next plan step after the last one taken that the mask allows; STOP once the plan is used up."""
    ids, m = _plan_ids(env), env.mask(q, s)
    start = max((ids.index(h.action) + 1 for h in s.history if h.action in ids), default=0)
    for a in ids[start:]:
        if m[a]:
            return a
    return 0 if m[0] else int(np.flatnonzero(m)[0])


HYBRID_M = 3                                                     # m of the hybrid heuristic's graph step


def best_graph_action(env, q, s, m=None) -> int | None:
    """The valid graph action of the edge type whose single best candidate Δ_V (env.graph_scores, no C*) is
    highest; ties -> lower type id. m = HYBRID_M when that action is valid, else the smallest valid m of the type.
    None when no graph action is valid."""
    m = env.mask(q, s) if m is None else m
    by_type: dict = {}                                   # edge type -> its graph action ids, one pass over the space
    for i, a in enumerate(env.actions):
        if a.op == "graph":
            by_type.setdefault(a.r, []).append(i)
    best = None
    for r, name in enumerate(env.data.graph.edge_types):
        ids = [i for i in by_type.get(name, ()) if m[i]]
        if not ids:
            continue
        _, sc = env.graph_scores(q, s, r)
        top = float(np.max(sc)) if sc.size else -np.inf
        if best is None or top > best[0]:
            pick = [i for i in ids if env.actions[i].m == HYBRID_M] or sorted(ids, key=lambda i: env.actions[i].m)
            best = (top, pick[0])
    return None if best is None else best[1]


def hybrid_heuristic_policy(env, q, s) -> int:
    """lexical k=5 -> vector k=5 -> best_graph_action -> STOP. The stage follows the last op taken (a masked stage
    is skipped: it never enters the history); masked-safe: a forbidden STOP falls back to the first valid action."""
    m = env.mask(q, s)
    last = env.actions[s.history[-1].action].op if s.history else None
    stage = {None: 0, "lexical": 1, "vector": 2}.get(last, 3)
    for st in range(stage, 3):
        if st < 2:
            op = ("lexical", "vector")[st]
            a = next((i for i, x in enumerate(env.actions) if x.op == op and x.k == 5), None)
            if a is not None and m[a]:
                return a
        else:
            a = best_graph_action(env, q, s, m)
            if a is not None:
                return a
    return 0 if m[0] else int(np.flatnonzero(m)[0])


def _rows(q, kind: str):
    if kind == "bm25":
        return q.lex
    if kind == "dense":
        return q.vec
    if kind == "rrf":
        return rrf_rows(q.lex, q.vec, top=len(q.lex) + len(q.vec))
    raise ValueError(f"unknown ranking {kind!r}")


def _budget0(env) -> Budget:
    return Budget(0, 0, env.limits.K_max, env.limits.L_max)


def topk_context(env, q, kind: str, k: int) -> tuple[int, ...]:
    return tuple(take_fitting(_rows(q, kind), set(), int(k), _budget0(env), env.data.chunk_tokens))


RRF_K = 60                                                      # graphrag.retrieval.rankings.rrf_rows default
ELBOW_N = 24                                                    # elbow: largest gap within the first 24 fragments
REL_P = (0.9, 0.7, 0.5)
AGREE_N = (10, 20)
DYNAMIC_NAMES = ([f"rel:{k}@{p}" for k in ("bm25", "dense", "rrf") for p in REL_P]
                 + [f"rel_raw:{k}@{p}" for k in ("bm25", "dense", "rrf") for p in REL_P]
                 + [f"elbow:{k}" for k in ("bm25", "dense", "rrf")] + [f"agree@{n}" for n in AGREE_N])
CC_W = (0.5, 0.7, 0.85)                                          # dense weights of the fused baselines
FUSED_NAMES = [f"rel:cc{w}@{p}" for w in CC_W for p in REL_P]
ORACLE_SWITCH_NAMES = [f"oracle_switch:dense|cc{w}@{p}" for w in CC_W for p in REL_P]
DYNAMIC_NAMES = DYNAMIC_NAMES + FUSED_NAMES + ORACLE_SWITCH_NAMES
TOKEN_BUDGETS = (300, 600)


def is_oracle(name: str) -> bool:
    return name.startswith("oracle_switch:") or name in ("greedy_oracle", "beam_oracle")


def rrf_scored(a, b, k: int = RRF_K, top: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """rrf_rows with its fused scores (same rows, same order: score desc, row asc)."""
    s: dict = defaultdict(float)
    for lst in (a, b):
        for r, c in enumerate(np.asarray(lst)[:top].tolist()):
            s[int(c)] += 1.0 / (k + r + 1)
    items = sorted(s.items(), key=lambda x: (-x[1], x[0]))[:top]
    return np.array([c for c, _ in items], np.int64), np.array([v for _, v in items], np.float64)


def ranking_scores(q, kind: str, norm_top_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(rows, scores) of a ranking: real U9 BM25 scores (q via with_scores), real cosines, RRF fused scores;
    norm_top_n: min-max normalised over the ranking's top norm_top_n (minmax_scores), None = raw."""
    if kind == "bm25":
        rows, sc = np.asarray(q.lex, np.int64), search_scores(q, "lexical")
    elif kind == "dense":
        rows, sc = np.asarray(q.vec, np.int64), search_scores(q, "vector")
    elif kind == "rrf":
        rows, sc = rrf_scored(q.lex, q.vec, top=len(q.lex) + len(q.vec))
    elif kind.startswith("cc"):
        if norm_top_n is None:
            raise ValueError("cc<w> is defined on min-max normalised scores only (no rel_raw:cc)")
        return fused_scores(q, float(kind[2:]), norm_top_n)               # already normalised
    else:
        raise ValueError(f"unknown ranking {kind!r}")
    return rows, (sc if norm_top_n is None else minmax_scores(sc, norm_top_n))


def elbow_cut(scores) -> int:
    """Fragments kept by the elbow rule: 1 + argmax of the gaps s_i − s_{i+1} (first maximum); len when < 2."""
    s = np.asarray(scores, np.float64)
    if s.size < 2:
        return int(s.size)
    return int(np.argmax(s[:-1] - s[1:])) + 1


def dynamic_context(env, q, name: str, norm_top_n: int = NORM_TOP_N) -> tuple[int, ...]:
    tok, b = env.data.chunk_tokens, _budget0(env)
    kind, _, arg = name.partition("@")
    if kind.startswith("oracle_switch:"):
        raise ValueError(f"{name} reads C*: use reference_context with the gold set")
    if kind.startswith(("rel:", "rel_raw:")):
        raw = kind.startswith("rel_raw:")
        rows, sc = ranking_scores(q, kind.split(":", 1)[1], None if raw else norm_top_n)
        return tuple(threshold_take(rows, sc, set(), float(arg), b, tok))
    if kind.startswith("elbow:"):
        rows, sc = ranking_scores(q, kind[6:])
        fit = take_fitting(rows, set(), ELBOW_N, b, tok)
        by_row = dict(zip(rows.tolist(), sc.tolist()))
        return tuple(fit[:elbow_cut([by_row[c] for c in fit])])
    if kind == "agree":
        n = int(arg)
        both = set(np.asarray(q.lex)[:n].tolist()) & set(np.asarray(q.vec)[:n].tolist())
        fused = _rows(q, "rrf")
        out = take_fitting([c for c in fused if int(c) in both], set(), env.limits.K_max, b, tok)
        return tuple(out or take_fitting(fused, set(), 1, b, tok))      # no overlap: the RRF top fragment
    raise ValueError(f"unknown dynamic-k baseline {name!r}")


def _filtered(env) -> bool:
    return isinstance(env, FilteredEnv) and not bool(env.allowed.all())


def _no_lookahead(env) -> bool:
    """successor_additions enumerates every edge type and the fixed-k search effect: not on a filtered or an
    adaptive env, nor on a StructEnv (it never enumerates `structure`; M7 part 2 Task 1)."""
    return _filtered(env) or isinstance(env, (AdaptiveEnv, StructEnv))


def reference_context(name: str, env, q, gold, w, lookahead: bool | None = None,
                      norm_top_n: int = NORM_TOP_N) -> tuple[int, ...]:
    """`lookahead` (beam_oracle): None = train.yaml oracle.lookahead on the full Env, False on a FilteredEnv
    (successor_additions enumerates every edge type, so its score would see forbidden graph steps), on an
    AdaptiveEnv (it computes the fixed-k search effect) and on a StructEnv (it ignores `structure`; train.py
    baselines skips both oracle rows on struct, grl.train.cli_checks.baseline_names)."""
    kind, _, arg = name.partition("@")
    if kind in ("bm25", "dense", "rrf"):
        return topk_context(env, q, kind, int(arg))
    if kind.startswith("oracle_switch:"):                    # ORACLE: the better of two contexts under C*
        a, _, b = kind.split(":", 1)[1].partition("|")
        if a != "dense" or not b.startswith("cc") or not gold:
            raise ValueError(f"bad oracle {name!r} (oracle_switch:dense|cc<w>@p, needs C*)")
        best = None
        for n in (f"rel:dense@{arg}", f"rel:{b}@{arg}"):           # tie -> dense (the simpler channel)
            ctx = dynamic_context(env, q, n, norm_top_n)
            r = grl_reward(env.data, q.depth, ctx, gold, w, env.limits.T_max).total
            if best is None or r > best[0]:
                best = (r, ctx)
        return best[1]
    if kind.startswith(("rel:", "rel_raw:", "elbow:")) or kind == "agree":
        return dynamic_context(env, q, name, norm_top_n)
    if kind == "graph_heuristic":
        return tuple(m3_rollout(env, q, graph_heuristic_policy).context)
    if kind == "hybrid_heuristic":
        return tuple(m3_rollout(env, q, hybrid_heuristic_policy).context)
    if kind in ("greedy_oracle", "beam_oracle") and getattr(env, "required", ()):
        raise ValueError(f"{kind} STOPs freely (step check=False): run it on an env without `require`")
    if kind == "greedy_oracle":
        return tuple(greedy_oracle(env, q, gold, w).context)
    if kind == "beam_oracle":
        if _no_lookahead(env):
            if lookahead:
                raise ValueError("beam lookahead enumerates every edge type and fixed-k search: not on a "
                                 "filtered / adaptive env")
            lookahead = False
        return tuple(BeamSearch(env, q, gold, w, lookahead=lookahead).best().context)
    if kind == "random":
        return tuple(m3_rollout(env, q, random_policy(int(arg or 0))).context)
    raise ValueError(f"unknown reference {name!r}")


def run_references(names, env, ranks, items, w, norm_top_n: int = NORM_TOP_N, rows_out: dict | None = None) -> dict:
    """Mean metrics per reference name; rows_out (pilot 5), when given, also gets {name: per-query rows}."""
    guard_qkeys([it.qkey for it in items])
    out = {}
    for name in names:
        rows = []
        for it in items:
            rank = ranks.get(it.qkey)
            q = with_scores(env.query(rank), rank)
            ctx = reference_context(name, env, q, it.gold, w, norm_top_n=norm_top_n)
            rows.append({**context_metrics(env.data, q, ctx, it.gold, w, env.limits.T_max),
                         **budget_recall(env.data.chunk_tokens, ctx, it.gold)})
            if rows_out is not None:
                rows_out.setdefault(name, []).append({"qkey": it.qkey, "context": [int(c) for c in ctx], **rows[-1]})
        out[name] = {k: round(float(np.mean([r[k] for r in rows])), 6) for k in rows[0]} if rows else {}
        if rows and name.startswith("oracle_switch:"):
            out[name]["oracle"] = True                       # reads C*: an upper bound, not a deployable baseline
    return out


def budget_recall(tokens, context, gold, budgets=TOKEN_BUDGETS) -> dict:
    """{recall@<B>tok}: recall of the longest prefix of `context` (in its order) whose token sum is ≤ B."""
    out, cum = {}, np.cumsum([int(tokens[int(c)]) for c in context]) if len(context) else np.zeros(0)
    for B in budgets:
        n = int(np.searchsorted(cum, B, side="right"))
        out[f"recall@{B}tok"] = len({int(c) for c in context[:n]} & set(gold)) / len(gold)
    return out


CURVE_KINDS = ("bm25", "dense", "rrf")
CURVE_KEYS = ("recall", "reward", "r_chunks", "precision", "n_ctx", "length")


def recall_curves(env, ranks, items, w, k_max: int | None = None) -> dict:
    """{kind: [{k, recall_mean, reward_mean, r_chunks_mean, precision_mean, n_ctx_mean, length_mean}, k = 1..k_max]}
    for the fixed bm25 / dense / rrf top-k (k_max = K_max by default). take_fitting is greedy in ranking order, so
    the top-k context is the k-prefix of the top-k_max one (tested): one ranking pass per query and kind."""
    guard_qkeys([it.qkey for it in items])
    k_max = int(k_max or env.limits.K_max)
    acc = {kind: [[] for _ in range(k_max)] for kind in CURVE_KINDS}
    for it in items:
        q = env.query(ranks.get(it.qkey))
        for kind in CURVE_KINDS:
            full = topk_context(env, q, kind, k_max)
            for k in range(1, k_max + 1):
                acc[kind][k - 1].append(context_metrics(env.data, q, full[:k], it.gold, w, env.limits.T_max))
    out = {}
    for kind, per_k in acc.items():
        out[kind] = [{"k": k, **{f"{m}_mean": round(float(np.mean([r[m] for r in rs])), 6) if rs else 0.0
                                 for m in CURVE_KEYS}} for k, rs in enumerate(per_k, start=1)]
    return out


def curves_table(curves: dict) -> str:
    """Plain-text table: one row per k, recall / reward per ranking."""
    kinds = list(curves)
    lines = ["  k " + "".join(f"{k + ' recall':>14}{k + ' reward':>14}" for k in kinds)]
    for i in range(len(curves[kinds[0]]) if kinds else 0):
        row = [curves[k][i] for k in kinds]
        lines.append(f"{i + 1:>3} " + "".join(f"{r['recall_mean']:>14.4f}{r['reward_mean']:>14.4f}" for r in row))
    return "\n".join(lines)
