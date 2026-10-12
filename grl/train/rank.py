"""Pilot 6 -- the RL reranker as a top-k ranker (grl.model.rank.RankEnv, variant "rank"); the train / eval side, which
reads C*. User decision 2026-10-02: score hit@k / recall@k of the ranked list for k in {10, 15, 20} (plus @1/@3/@5,
MRR, nDCG@10) instead of the top-1 F1 reward.

Metrics (rank_metrics; binary relevance = membership in C*; C* non-empty, the ranking duplicate-free):
- hit@k = 1 iff some C* fragment is among the first k; recall@k = |C* ∩ top-k| / |C*| (C* outside the pool counts as
  missed, so the pool ceiling is part of the metric);
- MRR = 1 / rank of the first C* fragment within the list (0 if none) -- the list has rank.k_max = 20 entries for every
  method, so this is MRR@20;
- nDCG@k = DCG@k / IDCG@k with DCG = Σ_{i<=k} rel_i / log2(i + 1) and IDCG over min(|C*|, k) relevant positions.
Rewards (RankReward, train.yaml rank.reward): ndcg@20 (main), ndcg@10, or mix = mean(hit@10, hit@15, hit@20, MRR). The
reward of an episode is a function of its ranked list only; GRPO / evaluation reach it through RankReward.episode_parts
(grl.train.evaluate.episode_parts hook), so the rollout workers never see C*.
SFT (rank_sft_examples): listwise -- the state before any pick, labelled with the C* slots inside the pool; loss
sft.loss = marginal gives −log Σ_{g ∈ C*∩pool} π(g | s), i.e. the softmax cross-entropy over the pool with C* as target
(= −log of the Plackett-Luce probability that a C* fragment is ranked first). Multi-fragment C*: one more state after
each C* pick (fused order), labelled with the C* slots still unplaced. rank.sft_neg_prefix m > 0 adds the states with
the first 1..m non-C* candidates (fused order) already placed, labelled with C* (useful only when rank.static = false).
Queries without a C* fragment in the pool give no example (counted).
Baselines (run_rank_references, no learning): dense / bm25 / rrf (U9 rankings), fused (the pool order), ce (pool by
the cross-encoder logit), ce+fused:<a> (a·mm(ce) + (1−a)·fused over the pool, grl.train.rerank_baselines), and the
ORACLE pool_oracle (C* ∩ pool first) = the pool ceiling; pool_ceiling() reports hit / recall of the pool for several N."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from graphrag.retrieval.rankings import rrf_rows
from grl.model.observe import describe
from grl.model.rerank import rerank_candidates
from grl.train.data import SFTExample, guard_qkeys

RANK_KS = (1, 3, 5, 10, 15, 20)
NDCG_KS = (10, 20)
REWARD_KINDS = ("ndcg@20", "ndcg@10", "mix")
CE_FUSED_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RANK_BASELINE_NAMES = (["dense", "bm25", "rrf", "fused", "ce"] + [f"ce+fused:{a}" for a in CE_FUSED_GRID]
                       + ["pool_oracle"])
CE_RANK_NAMES = ["ce"] + [f"ce+fused:{a}" for a in CE_FUSED_GRID]
POOL_NS = (20, 50, 100)


def rank_metrics(ranking, target, ks=RANK_KS) -> dict:
    """hit@k / recall@k (k in ks), mrr, ndcg@10 / ndcg@20, first_rank (1-based, None if absent), ranks."""
    t = {int(g) for g in target}
    if not t:
        raise ValueError("rank_metrics needs a non-empty C*")
    r = [int(c) for c in ranking]
    if len(set(r)) != len(r):
        raise ValueError(f"ranking repeats a fragment: {r}")
    pos = [i for i, c in enumerate(r) if c in t]                      # 0-based positions of C* fragments
    out = {}
    for k in ks:
        h = sum(1 for p in pos if p < k)
        out[f"hit@{k}"] = float(h > 0)
        out[f"recall@{k}"] = h / len(t)
    out["mrr"] = 1.0 / (pos[0] + 1) if pos else 0.0
    for k in NDCG_KS:
        dcg = sum(1.0 / math.log2(p + 2) for p in pos if p < k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(t), k)))
        out[f"ndcg@{k}"] = dcg / idcg
    out["first_rank"] = pos[0] + 1 if pos else None
    out["ranks"] = [p + 1 for p in pos]
    return out


def reward_value(m: dict, kind: str) -> float:
    if kind == "ndcg@20":
        return m["ndcg@20"]
    if kind == "ndcg@10":
        return m["ndcg@10"]
    if kind == "mix":
        return (m["hit@10"] + m["hit@15"] + m["hit@20"] + m["mrr"]) / 4.0
    raise ValueError(f"rank.reward {kind!r} is not one of {REWARD_KINDS}")


@dataclass(frozen=True)
class RankParts:
    """RewardParts stand-in for summarize(): total = the ranking reward; chunks mirrors it, the M3 graph parts are 0."""
    total: float
    chunks: float
    coverage: float = 0.0
    diversity: float = 0.0
    coherence: float = 0.0


def _mean(xs) -> float:
    return round(float(np.mean(xs)), 6) if len(xs) else 0.0


@dataclass(frozen=True)
class RankReward:
    """The `w` of a rank run: reward kind + list length. episode_parts / summary are the hooks of
    grl.train.evaluate.episode_parts / rank_summary."""
    kind: str = "ndcg@20"
    k_max: int = 20
    is_rank: bool = True

    def __post_init__(self):
        if self.kind not in REWARD_KINDS:
            raise ValueError(f"rank.reward {self.kind!r} is not one of {REWARD_KINDS}")

    def value(self, ranking, target) -> float:
        return round(reward_value(rank_metrics(ranking, target), self.kind), 9)

    def episode_parts(self, eps, finals, golds) -> list[RankParts]:
        out = []
        for e, f in zip(eps, finals):
            v = self.value(f.context, golds[e.qkey])
            out.append(RankParts(v, v))
        return out

    def summary(self, eps, finals, golds) -> dict:
        """Mean ranking metrics over the episodes + first-pick diagnostics (§8.7 rank collapse) + the share of query
        groups (G > 1 copies) whose rankings are all identical."""
        ms = [rank_metrics(f.context, golds[e.qkey]) for e, f in zip(eps, finals)]
        keys = [f"{p}@{k}" for k in RANK_KS for p in ("hit", "recall")] + ["mrr"] + [f"ndcg@{k}" for k in NDCG_KS]
        out = {f"rank_{k}": _mean([m[k] for m in ms]) for k in keys}
        out.update(first_pick_stats(eps))
        groups: dict = {}
        for e, f in zip(eps, finals):
            groups.setdefault(e.qkey, []).append(tuple(f.context))
        multi = [g for g in groups.values() if len(g) > 1]
        out["rank_identical_share"] = _mean([float(len(set(g)) == 1) for g in multi]) if multi else None
        return out


def first_pick_stats(eps) -> dict:
    """First pick of every episode: share of the fused top candidate (action 1 = slot 0) and of the most used slot."""
    first = [int(e.actions[0]) for e in eps if e.actions]
    if not first:
        return {"first_slot0_share": None, "first_top_share": None}
    c = Counter(first)
    return {"first_slot0_share": round(c.get(1, 0) / len(first), 6),
            "first_top_share": round(max(c.values()) / len(first), 6)}


def rank_reward(cfg: dict) -> RankReward:
    rk = cfg.get("rank") or {}
    return RankReward(str(rk.get("reward", "ndcg@20")), int(rk.get("k_max", 20)))


def rank_alarm_cfg(cfg: dict, slot0_share: float, margin: float = 0.05) -> dict:
    """§8.7 adapted to ranking: t_max / STOP-share / per-action collapse do not apply (every episode places k_max
    picks, no STOP) and are switched off (thresholds outside [0, 1]); collapse = the first pick is the fused top
    candidate in more than max(op_collapse_share, share of train queries whose C* holds that candidate + margin) of the
    episodes (log.alarms first_pick_collapse); the KL ceiling is unchanged."""
    return {**cfg, "tmax_share": 1.01, "stop_share_min": -1.0, "op_collapse_share": 1.01,
            "first_pick_collapse": max(float(cfg["op_collapse_share"]), float(slot0_share) + margin),
            "target_slot0_share": round(float(slot0_share), 6)}


def target_slot0_share(env, ranks, items) -> float:
    """Share of the items whose C* contains the fused top candidate (pool slot 0)."""
    hits = []
    for it in items:
        rows = env.cand_rows(env.query(ranks.get(it.qkey)))
        hits.append(float(rows.size > 0 and int(rows[0]) in it.gold))
    return float(np.mean(hits)) if hits else 0.0


# ---- SFT ----------------------------------------------------------------------------------------------------------

def rank_sft_examples(env, ranks, items, describe_kw: dict, neg_prefix: int = 0) -> tuple[list[SFTExample], dict]:
    """Listwise SFT states (module docstring); returns (examples, counts)."""
    out, cnt = [], {"n": len(items), "no_target_in_pool": 0, "examples": 0, "multi": 0}
    for it in items:
        q = env.query(ranks.get(it.qkey))
        s0 = env.reset(q)
        rows = [int(c) for c in env.cand_rows(q).tolist()]
        slot = {c: j + 1 for j, c in enumerate(rows)}                  # action id of a pool member
        tg = [c for c in rows if c in it.gold]                         # C* ∩ pool, fused order
        if not tg:
            cnt["no_target_in_pool"] += 1
            continue
        cnt["multi"] += int(len(tg) > 1)

        def add(t, s, labels):
            out.append(SFTExample(it.qkey, t, describe(env, q, s, **describe_kw), tuple(sorted(slot[c] for c in labels))))
        add(1, s0, tg)
        s = s0
        for i, c in enumerate(tg[:-1]):                                # multi-fragment C*: after each C* pick
            s = env.step(q, s, slot[c])
            if s.done:
                break
            add(s.t + 1, s, tg[i + 1:])
        s = s0
        for c in [c for c in rows if c not in it.gold][:max(0, int(neg_prefix))]:
            s = env.step(q, s, slot[c])
            if s.done:
                break
            add(s.t + 1, s, tg)
    cnt["examples"] = len(out)
    return out, cnt


# ---- baselines ----------------------------------------------------------------------------------------------------

def baseline_ranking(env, q, name: str, target=None) -> list[int]:
    """The first k_max rows of a no-learning ranking (target only for the ORACLE pool_oracle)."""
    k = env.k_max
    if name in ("dense", "bm25", "rrf"):
        if name == "dense":
            rows = q.vec
        elif name == "bm25":
            rows = q.lex
        else:
            rows = rrf_rows(q.lex, q.vec, top=len(q.lex) + len(q.vec))
        out, seen = [], set()
        for c in rows:
            c = int(c)
            if c >= 0 and c not in seen:
                out.append(c)
                seen.add(c)
            if len(out) == k:
                break
        return out
    pool = [int(c) for c in env.cand_rows(q).tolist()]
    if name == "fused":
        return pool[:k]
    if name == "pool_oracle":
        if target is None:
            raise ValueError("pool_oracle reads C*")
        return ([c for c in pool if c in target] + [c for c in pool if c not in target])[:k]
    from grl.train.rerank_baselines import candidate_scores
    rows, _ = candidate_scores(env, q, name)                          # ce | ce+fused:<a>, ties -> fused position
    return [int(c) for c in rows][:k]


def pool_ceiling(env, ranks, items, ns=POOL_NS) -> dict:
    """{N: {hit, recall}}: share of items with a C* fragment in the fused top-N pool / mean share of C* inside it."""
    out = {}
    for n in ns:
        h, r = [], []
        for it in items:
            q = env.query(ranks.get(it.qkey))
            rows, _ = rerank_candidates(q, int(n), env.fuse_w, env.norm_top_n, env.data.chunk_tokens, env.limits.L_max)
            inn = len(set(int(c) for c in rows) & set(it.gold))
            h.append(float(inn > 0))
            r.append(inn / len(it.gold))
        out[str(n)] = {"hit": _mean(h), "recall": _mean(r), "n_items": len(items)}
    return out


METRIC_KEYS = tuple([f"{p}@{k}" for k in RANK_KS for p in ("hit", "recall")] + ["mrr"] + [f"ndcg@{k}" for k in NDCG_KS])


def run_rank_references(names, env, ranks, items, rows_out: dict | None = None) -> dict:
    """Mean ranking metrics per baseline (ce rows skipped without a ce cache) + pool_ceiling; rows_out gets per-query
    rows {name: [{qkey, ranking, metrics...}]}."""
    guard_qkeys([it.qkey for it in items])
    has_ce = env.ce_cache is not None
    out, skipped = {}, []
    for name in names:
        if name in CE_RANK_NAMES and not has_ce:
            skipped.append(name)
            continue
        rows = []
        for it in items:
            q = env.query(ranks.get(it.qkey))
            rk = baseline_ranking(env, q, name, it.gold)
            m = rank_metrics(rk, it.gold)
            rows.append(m)
            if rows_out is not None:
                rows_out.setdefault(name, []).append({"qkey": it.qkey, "ranking": rk, **m})
        out[name] = {k: _mean([r[k] for r in rows]) for k in METRIC_KEYS} if rows else {}
        if name == "pool_oracle" and rows:
            out[name]["oracle"] = True
    out["skipped"] = skipped
    out["n"] = len(items)
    out["pool_ceiling"] = pool_ceiling(env, ranks, items)
    out["pool"] = {"n": env.n, "k_max": env.k_max, "fuse_w": env.fuse_w, "norm_top_n": env.norm_top_n,
                   "ce_cache": env.ce_cache}
    return out


def rank_per_query(items, eps, finals) -> list[dict]:
    """One row per item of a greedy rank evaluation (order = items): ranking, picked slots, metrics."""
    by = {e.qkey: (e, f) for e, f in zip(eps, finals)}
    out = []
    for it in items:
        e, f = by[it.qkey]
        out.append({"qkey": it.qkey, "split": it.split, "pattern": it.pattern, "lang": it.lang,
                    "ranking": [int(c) for c in f.context], "slots": [int(a) - 1 for a in e.actions],
                    "reason": f.reason, **rank_metrics(f.context, it.gold)})
    return out
