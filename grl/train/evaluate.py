"""Evaluation shared by SFT / GRPO validation and the M6 hooks (IMPLEMENTATION.md §8.6, §9.4): episode summaries
(reward and its parts, recall / precision of C*, context size in fragments and tokens, share of episodes that took a
lexical, a vector and a graph step, episode length histogram, operator histogram, STOP vs limit share),
context metrics of fixed baselines, greedy policy evaluation, expert-trajectory reproduction (§11 step 8)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from grl.train.fbeta import grl_reward
from grl.train.rollout import rollout


def op_name(a) -> str:
    """graph:<r>, select:<slot> (rerank: the §8.7 collapse alarm then flags one candidate slot), else the op."""
    return f"graph:{a.r}" if a.op == "graph" else f"select:{a.k}" if a.op == "select" else a.op


def _mean(xs) -> float:
    return round(float(np.mean(xs)), 6) if len(xs) else 0.0


def summarize(eps, finals, parts, golds, actions, t_max: int) -> dict:
    n = len(eps)
    lens = [len(e.actions) for e in eps]
    reasons = Counter(f.reason for f in finals)
    ops = Counter(op_name(actions[a]) for e in eps for a in e.actions)
    rec = [len(set(f.ctx_set) & golds[e.qkey]) / len(golds[e.qkey]) for e, f in zip(eps, finals)]
    prec = [len(set(f.ctx_set) & golds[e.qkey]) / len(f.context) if f.context else 0.0 for e, f in zip(eps, finals)]
    tot = [p.total for p in parts]
    fams = [{actions[a].op for a in e.actions} for e in eps]
    hyb = [float({"lexical", "vector", "graph"} <= f) for f in fams]
    return {"n": n, "reward_mean": _mean(tot), "reward_std": round(float(np.std(tot)), 6) if n else 0.0,
            "r_chunks": _mean([p.chunks for p in parts]), "r_coverage": _mean([p.coverage for p in parts]),
            "r_diversity": _mean([p.diversity for p in parts]), "r_coherence": _mean([p.coherence for p in parts]),
            "recall_mean": _mean(rec), "precision_mean": _mean(prec),
            "ctx_size_mean": _mean([len(f.context) for f in finals]),
            "ctx_tokens_mean": _mean([f.length for f in finals]), "all_ops_share": _mean(hyb),
            "ep_len_mean": _mean(lens),
            "len_hist": {str(k): v for k, v in sorted(Counter(lens).items())},
            "stop_share": round(reasons.get("stop", 0) / n, 6) if n else 0.0,
            "tmax_share": round(reasons.get("t_max", 0) / n, 6) if n else 0.0,
            "reasons": dict(sorted(reasons.items())), "op_hist": dict(sorted(ops.items()))}


def context_metrics(data, q, context, gold, w, t_max: int) -> dict:
    p = grl_reward(data, q.depth, context, gold, w, t_max)          # F-beta R_chunks if w.chunks_beta != 1
    C = set(int(c) for c in context)
    return {"reward": p.total, "r_chunks": p.chunks, "r_coverage": p.coverage, "r_diversity": p.diversity,
            "r_coherence": p.coherence, "recall": len(C & gold) / len(gold),
            "precision": len(C & gold) / len(C) if C else 0.0, "n_ctx": len(C),
            "length": int(data.chunk_tokens[list(C)].sum()) if C else 0}


EVAL_CHUNK = 128      # queries encoded at once by a greedy evaluation (pilot 5: the GNN pass over 427 train queries
#                       at once OOMs the GB10); item lists <= EVAL_CHUNK run exactly as before (one rollout)


def episode_parts(vec, eps, finals, golds, w) -> list:
    """Reward parts of finished episodes: vec.rewards (the M3 / grl reward in the workers) unless `w` brings its own
    (pilot 6 grl.train.rank.RankReward: a function of the ranked list, computed here)."""
    own = getattr(w, "episode_parts", None)
    return list(own(eps, finals, golds)) if own is not None else list(vec.rewards(golds, w))


def extra_summary(w, eps, finals, golds) -> dict:
    """Variant metrics beyond summarize() (pilot 6: ranking metrics of RankReward.summary); {} otherwise."""
    own = getattr(w, "summary", None)
    return own(eps, finals, golds) if own is not None else {}


def evaluate_policy(policy, graph, vec, ranks, items, w, t_max, device, actions, chunk: int = EVAL_CHUNK):
    was = policy.training
    policy.eval()
    golds = {i.qkey: i.gold for i in items}
    eps, finals, parts = [], [], []
    for b in range(0, len(items), int(chunk)):            # rewards read the vec env's current episodes: per chunk
        e, f = rollout(policy, graph, vec, ranks, [i.qkey for i in items[b:b + int(chunk)]], 1, greedy=True,
                       device=device)
        eps, finals, parts = eps + list(e), finals + list(f), parts + episode_parts(vec, e, f, golds, w)
    out = summarize(eps, finals, parts, golds, actions, t_max)
    out.update(extra_summary(w, eps, finals, golds))
    groups = group_summary(items, eps, finals, parts, actions)
    if groups:
        out["by_group"] = groups
    policy.train(was)
    return out, eps, finals


HARD_PATTERNS = ("count", "version")      # M7 Q1 / Q2 rows (grl.hard.ship: pattern = qtype)


def _family(a) -> str:
    return a.op if a.op in ("structure", "graph", "lexical", "vector", "stop") else str(a.op)


def group_summary(items, eps, finals, parts, actions) -> dict:
    """M7 pilot 7: per group (old = non-hard rows, count = Q1, version = Q2) of a greedy evaluation -- n, reward_mean,
    all_share (C* within the final context: all@K_max), recall_mean, action family shares. {} when no item is a
    hard row (every earlier run logs exactly as before)."""
    if not any(getattr(i, "pattern", None) in HARD_PATTERNS for i in items):     # items without a pattern: none
        return {}
    by = {e.qkey: (e, f, p) for e, f, p in zip(eps, finals, parts)}
    out = {}
    for g in ("old",) + HARD_PATTERNS:
        its = [i for i in items if (i.pattern if i.pattern in HARD_PATTERNS else "old") == g]
        if not its:
            continue
        rows = [by[i.qkey] for i in its]
        fam = Counter(_family(actions[a]) for e, _, _ in rows for a in e.actions)
        tot = sum(fam.values())
        out[g] = {"n": len(its), "reward_mean": _mean([p.total for _, _, p in rows]),
                  "all_share": _mean([float(i.gold <= set(f.ctx_set)) for i, (_, f, _) in zip(its, rows)]),
                  "recall_mean": _mean([len(i.gold & set(f.ctx_set)) / len(i.gold) for i, (_, f, _) in zip(its, rows)]),
                  "action_share": {k: round(v / tot, 6) for k, v in sorted(fam.items())} if tot else {}}
    return out


def per_query(env, ranks, items, eps, finals, w, actions) -> list[dict]:
    """One row per item of a greedy evaluation: actions (op_name), final context rows, end reason and the
    context_metrics of that context (reward, recall, precision, ...) -- for SE / paired comparisons and choice
    diagnostics (pilot 5). Order = items."""
    by = {e.qkey: (e, f) for e, f in zip(eps, finals)}
    out = []
    for it in items:
        e, f = by[it.qkey]
        q = env.query(ranks.get(it.qkey))
        out.append({"qkey": it.qkey, "split": it.split, "pattern": it.pattern, "lang": it.lang,
                    "actions": [op_name(actions[a]) for a in e.actions], "action_ids": [int(a) for a in e.actions],
                    "context": [int(c) for c in f.context],
                    "reason": f.reason, **context_metrics(env.data, q, list(f.context), it.gold, w, t_max=env.limits.T_max)})
    return out


def reproduction(finals, items, expert: dict) -> float:
    """§11 step 8: the greedy episode ends with STOP on exactly the expert's final context set; for a row whose
    expert final is limit-ended (final_by "limit", R34 / R41) it ends with k_max / l_max on exactly that set."""
    from grl.train.data import LIMIT_FINAL_REASONS

    def ended(f, i) -> bool:
        if getattr(i, "final_by", None) == "limit":
            return f.reason in LIMIT_FINAL_REASONS
        return f.reason == "stop"
    hits = [float(ended(f, i) and frozenset(f.ctx_set) == expert[i.qkey]) for f, i in zip(finals, items)]
    return _mean(hits)


EVAL_SPLITS = ("train", "val")


def _check_split(split) -> None:
    if split not in EVAL_SPLITS:
        raise ValueError(f"make_evaluator serves only {EVAL_SPLITS} (got {split!r}); test is evaluated by M6 itself")


def make_evaluator(graph, vec, ranks, items_by_split: dict, w, t_max, device, actions, expert: dict | None = None):
    for split in items_by_split:
        _check_split(split)

    def ev(policy, split: str) -> dict:
        _check_split(split)
        items = items_by_split[split]
        s, _, finals = evaluate_policy(policy, graph, vec, ranks, items, w, t_max, device, actions)
        if expert and split in expert:
            s["reproduction"] = reproduction(finals, items, expert[split])
        return s
    return ev
