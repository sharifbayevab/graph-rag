"""Rerank baselines (M5 RL reranker, no learning): one-shot selections from the rerank candidate list (the fused
top-N, grl.model.rerank.RerankEnv), all inside K_max / L_max (threshold_take: budget-fitting, ranking order).
- fused@top1 / fused@p: the candidates' own fused normalised scores (fused@0.9 = rel:cc<w>@0.9 whenever its take
  stays inside the top-N);
- ce@top1 / ce@p: the candidates re-ordered by the cross-encoder logit (ties -> fused position), p a relative
  threshold on the ce scores min-max normalised over the candidates;
- ce+fused:<a>@top1 / @p: the convex a·mm(ce) + (1−a)·fused (min-max normalised again over the candidates);
- oracle@N (ORACLE, reads C*, row flagged oracle: true): the exact variant expert (grl.train.expert.expert_search)
  inside the candidate list, capped at expert.depth = 3 selections + STOP (exact below the cap; a context of > 3
  fragments is outside it) -- the reward ceiling of rerank policies with <= 3 selections; every row also reports cand_recall, the
  share of C* inside the candidate list (recall@N, the recall ceiling).
ce rows need the env's ce cache (rerank.ce_cache); without one they are listed under "skipped"."""
from __future__ import annotations

import numpy as np

from graphrag.retrieval.executor import Budget
from grl.model.variants import minmax_scores, threshold_take
from grl.train.baselines import REL_P, budget_recall
from grl.train.data import guard_qkeys
from grl.train.evaluate import context_metrics

CE_FUSED_A = (0.3, 0.5, 0.7)
FUSED_NAMES = ["fused@top1"] + [f"fused@{p}" for p in REL_P]
CE_NAMES = (["ce@top1"] + [f"ce@{p}" for p in REL_P]
            + [f"ce+fused:{a}@{x}" for a in CE_FUSED_A for x in ("top1",) + REL_P])
ORACLE_NAMES = ["oracle@N"]
RERANK_BASELINE_NAMES = FUSED_NAMES + CE_NAMES + ORACLE_NAMES


def _norm(sc) -> np.ndarray:
    sc = np.asarray(sc, np.float64)
    if sc.size < 2:
        return np.ones_like(sc)
    return minmax_scores(sc, sc.size)


def _ordered(rows, score):
    """rows / scores sorted by score desc, ties -> candidate position (the fused order)."""
    order = np.lexsort((np.arange(len(rows)), -np.asarray(score, np.float64)))
    return np.asarray(rows, np.int64)[order], np.asarray(score, np.float64)[order]


def candidate_scores(env, q, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows, normalised scores) of the candidate list, ordered for threshold_take. kind: fused | ce | ce+fused:<a>."""
    c = env._cands(q)
    rows = c.rows
    fused = c.fused
    if kind == "fused":
        return rows, fused
    ce = env.ce_scores(q)
    if ce is None:
        raise ValueError("ce baselines need rerank.ce_cache")
    if kind == "ce":
        return _ordered(rows, _norm(ce))
    if kind.startswith("ce+fused:"):
        a = float(kind.split(":", 1)[1])
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"ce+fused weight must be in [0, 1], got {a}")
        return _ordered(rows, _norm(a * _norm(ce) + (1.0 - a) * fused))
    raise ValueError(f"unknown rerank score {kind!r}")


def rerank_context(env, q, name: str) -> tuple[int, ...]:
    kind, _, arg = name.partition("@")
    rows, sc = candidate_scores(env, q, kind)
    if not rows.size:
        return ()
    b = Budget(0, 0, env.limits.K_max, env.limits.L_max)
    p = 1.0 if arg == "top1" else float(arg)
    out = threshold_take(rows, sc, set(), p, b, env.data.chunk_tokens)
    return tuple(out[:1] if arg == "top1" else out)


def oracle_context(env, q, target, w, depth: int, max_states, cap_beam: int) -> tuple[int, ...]:
    from grl.train.expert import expert_search
    return tuple(expert_search(env, q, frozenset(target), w, depth, True, None, max_states, cap_beam)["final"].context)


def run_rerank_references(names, env, ranks, items, w, expert_cfg: dict | None = None,
                          rows_out: dict | None = None) -> dict:
    """Mean metrics per rerank reference; rows_out (pilot 5), when given, also gets {name: per-query rows} (incl. the
    chosen candidate slots)."""
    from grl.train.expert import expert_config
    guard_qkeys([it.qkey for it in items])
    ec = expert_config({"expert": expert_cfg or {}})
    has_ce = env.ce_cache is not None
    out, skipped = {}, []
    for name in names:
        if name in CE_NAMES and not has_ce:
            skipped.append(name)
            continue
        rows = []
        for it in items:
            q = env.query(ranks.get(it.qkey))
            if name == "oracle@N":
                ctx = oracle_context(env, q, it.gold, w, ec["depth"], ec["max_states"], ec["cap_beam"])
            else:
                ctx = rerank_context(env, q, name)
            cands = set(env.cand_rows(q).tolist())
            rows.append({**context_metrics(env.data, q, ctx, it.gold, w, env.limits.T_max),
                         **budget_recall(env.data.chunk_tokens, ctx, it.gold),
                         "cand_recall": len(cands & set(it.gold)) / len(it.gold)})
            if rows_out is not None:
                order = [int(c) for c in env.cand_rows(q).tolist()]
                rows_out.setdefault(name, []).append({"qkey": it.qkey, "context": [int(c) for c in ctx],
                                                      "slots": [order.index(int(c)) if int(c) in order else -1
                                                                for c in ctx], **rows[-1]})
        out[name] = {k: round(float(np.mean([r[k] for r in rows])), 6) for k in rows[0]} if rows else {}
        if rows and name in ORACLE_NAMES:
            out[name]["oracle"] = True
    out["skipped"] = skipped
    out["n"] = len(items)
    out["candidates"] = {"n": env.n, "fuse_w": env.fuse_w, "norm_top_n": env.norm_top_n,
                         "ce_cache": env.ce_cache}
    return out
