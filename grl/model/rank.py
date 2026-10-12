"""Pilot 6 (user decision 2026-10-02): the rerank pointer policy as a TOP-K RANKER (variant "rank"). The episode
orders the candidate pool -- the first rerank.n rows of the fused BM25 + dense ranking, exactly the RerankEnv list --
by picking one candidate per step WITHOUT replacement until rank.k_max are placed (no STOP while a candidate can still
be placed); the final context, in pick order, is the ranked list. Sampling the policy step by step is then a
Plackett-Luce process over the pool, and the log-probability of a ranking is the sum of the per-step masked
log-softmax values (exact, the existing rollout / GRPO machinery).

Mask: select j is valid iff slot j holds a candidate not yet placed and fewer than min(k_max, |pool|) are placed;
STOP is valid only when no select is (an empty pool), so a live mask is never all-False and STOP is never sampled
while a candidate remains. There is no token / K_max / T_max budget: a ranked list is not a context to be read.
The episode ends ("ranked") after min(k_max, |pool|) picks.

Observation: RerankDesc as in RerankEnv. With rank.static = true (default) the observation does not depend on what
was already placed -- no covered vertices, the three context feature columns zeroed -- so the candidate logits of
RerankPolicy are fixed per query and the process is an exact Plackett-Luce model with those scores (greedy decoding =
sort by logit). With static = false the context features and the covered-vertex pool are live (a state-dependent,
sequential pointer model). The STOP head only ever scores a masked action except on an empty pool. Nothing here sees
the target set (tests/test_m4_state guard)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from graphrag.env.mdp import InvalidAction, State, Step
from grl.model.rerank import DYN_COLS, EMPTY, RerankDesc, RerankEnv

RANK_K_MAX = 20                          # train.yaml rank.k_max


class RankEnv(RerankEnv):
    """The ranking MDP over the RerankEnv candidate pool (module docstring)."""

    def __init__(self, data, cfg=None, start_cfg=None, k_max: int = RANK_K_MAX, static: bool = True, **kw):
        super().__init__(data, cfg, start_cfg, **kw)
        if not 1 <= int(k_max) <= self.n:
            raise ValueError(f"rank.k_max must be in [1, rerank.n = {self.n}] (the pool must hold the list), "
                             f"got {k_max}")
        self.k_max, self.static = int(k_max), bool(static)

    def list_len(self, q) -> int:
        """Length of a complete ranked list of q: min(k_max, |pool|)."""
        return min(self.k_max, int(self.cand_rows(q).size))

    def mask(self, q, s) -> np.ndarray:
        m = np.zeros(self.n + 1, bool)
        if s.done:
            return m
        rows = self._cands(q).rows
        if len(s.context) < min(self.k_max, rows.size):
            for j, c in enumerate(rows.tolist()):
                m[j + 1] = c not in s.ctx_set
        if not m[1:].any():                        # nothing (left) to rank: STOP is the only valid action
            m[0] = True
        return m

    def _any_valid(self, q, s) -> bool:
        return bool(self.mask(q, s)[1:].any())

    def step(self, q, s, a, check: bool = True) -> State:
        """Masked actions are refused even with check=False (a pick always adds exactly one new candidate)."""
        if s.done:
            raise InvalidAction("the episode is over")
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= int(a) < len(self.actions):
            raise InvalidAction(f"action id {a!r} is not in [0, {len(self.actions)})")
        a = int(a)
        if not self.mask(q, s)[a]:
            raise InvalidAction(f"action {a} {self.actions[a]} is masked in this state")
        t = s.t + 1
        if a == 0:
            return replace(s, t=t, history=s.history + (Step(t, a, (), ()),), done=True, reason="stop")
        d = self.data
        c = int(self._cands(q).rows[a - 1])
        v = int(d.chunk_node[c])
        covered = s.covered | {v}
        frontier = s.frontier if v in s.covered else (s.frontier | set(d.graph.all_neighbors(
            np.array([v], np.int64)).tolist()))
        s2 = State(s.qkey, t, s.context + (c,), s.ctx_set | {c}, frozenset(covered), frozenset(frontier - covered),
                   s.length + int(d.chunk_tokens[c]), s.history + (Step(t, a, (c,), ()),))
        return replace(s2, done=True, reason="ranked") if len(s2.context) >= self.list_len(q) else s2

    def describe_state(self, q, s) -> RerankDesc:
        d = super().describe_state(q, s)
        if self.static:                            # placement-independent observation: exact Plackett-Luce
            d.covered = EMPTY
            d.feats[:, list(DYN_COLS)] = 0.0
        d.scalars[0] = s.t / self.k_max            # t relative to the list length (STOP head input only)
        return d


def ranking(final) -> tuple[int, ...]:
    """The ranked list of a finished rank episode (pick order)."""
    return tuple(int(c) for c in final.context)


def check_rank_invariants(env: RankEnv, q, final) -> list[str]:
    """Replay check of a rank episode: never an all-False live mask, no repeated pick, every pick a pool member, the
    list complete (min(k_max, |pool|)), STOP only on an empty pool, deterministic replay."""
    probs, s = [], env.reset(q)
    pool = set(env.cand_rows(q).tolist())
    for h in final.history:
        m = env.mask(q, s)
        if not m.any():
            probs.append(f"t={s.t}: all-False mask in a live state")
        if m[0] and m[1:].any():
            probs.append(f"t={s.t}: STOP valid while a candidate can be placed")
        s = env.step(q, s, h.action)
    if s != final:
        probs.append("replay differs from the recorded episode")
    if not final.done:
        probs.append("episode not terminated")
    if len(set(final.context)) != len(final.context):
        probs.append("a candidate placed twice")
    if not set(final.context) <= pool:
        probs.append("a pick outside the pool")
    if len(final.context) != env.list_len(q):
        probs.append(f"list length {len(final.context)} != {env.list_len(q)}")
    return probs
