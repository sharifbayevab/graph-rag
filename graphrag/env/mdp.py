"""The MDP of context selection (IMPLEMENTATION.md §5; roadmap §3 item 8, U4). s_t = (q, C_t, G);
C_{t+1} = C_t ∪ ΔC_t; reward only at the end (graphrag.env.reward); γ = 1. Constructive limits (§5.2, §12 item 8),
masked invalid actions (U4), V₀ in the frontier never in the context (§4.4, §12 item 2). Immutable states."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from graphrag import config
from graphrag.env.start import QueryCtx, make_query
from graphrag.retrieval.executor import Budget, ContextSims, delta_v, graph_candidates, graph_delta, ranked_delta


class InvalidAction(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    op: str
    k: int | None = None
    r: str | None = None
    m: int | None = None

    def arg(self):
        if self.op in ("lexical", "vector"):
            return self.k
        if self.op == "graph":
            return {"r": self.r, "m": self.m}
        return None


def action_space(edge_types, k_grid=(3, 5, 10), m_grid=(1, 3, 5)) -> tuple[Action, ...]:
    acts = [Action("stop")]
    acts += [Action("lexical", k=int(k)) for k in k_grid]
    acts += [Action("vector", k=int(k)) for k in k_grid]
    acts += [Action("graph", r=r, m=int(m)) for r in edge_types for m in m_grid]
    return tuple(acts)


@dataclass(frozen=True)
class Limits:
    T_max: int
    K_max: int
    L_max: int


@dataclass(frozen=True)
class Step:
    t: int
    action: int
    added: tuple[int, ...]
    selected: tuple[int, ...]


@dataclass(frozen=True)
class State:
    qkey: str
    t: int
    context: tuple[int, ...]
    ctx_set: frozenset
    covered: frozenset
    frontier: frozenset
    length: int
    history: tuple[Step, ...] = ()
    done: bool = False
    reason: str | None = None


class Env:
    def __init__(self, data, cfg: dict | None = None, start_cfg: dict | None = None):
        cfg = cfg if cfg is not None else config.load("train")["env"]
        if cfg["graph_version"] != data.version:
            raise ValueError(f"env is configured for graph {cfg['graph_version']}, the data is {data.version}")
        self.data, self.cfg = data, cfg
        self.start_cfg = start_cfg if start_cfg is not None else config.load("data")["start"]
        self.actions = action_space(data.graph.edge_types, cfg["k_grid"], cfg["m_grid"])
        self.limits = Limits(int(cfg["T_max"]), int(cfg["K_max"]), int(cfg["L_max"]))
        self.decimals = int(cfg["score_decimals"])
        self.per_vertex = int(cfg["per_vertex_chunks"])
        self.lambda_red = float(cfg["lambda_red"])
        self.memo_max = int(cfg.get("memo_max", 200000))
        self._memo_c: dict = {}              # (qkey, covered, r) -> graph candidates
        self._memo_m: dict = {}              # (qkey, ctx_set, length, covered) -> action mask
        self._memo_s: dict = {}              # (qkey, context, covered, r) -> (candidates, delta_v scores)
        self._sims = None                    # ContextSims of the memo's query (fragment x context-fragment cosines)
        self._base: dict = {}                # r -> set N_r(V₀) of the memo's query
        self._nb: dict = {}                  # (r, u) -> frozenset N_r(u); the graph is static, kept across queries
        self._memo_q = None                  # the QueryCtx the memo belongs to

    def query(self, rank) -> QueryCtx:
        q = make_query(self.data, rank, self.start_cfg, self.limits.T_max, self.decimals)
        self._clear_memo(q)
        return q

    def _clear_memo(self, q) -> None:
        self._memo_c.clear()
        self._memo_m.clear()
        self._memo_s.clear()
        self._base.clear()
        self._sims = ContextSims(self.data)
        self._memo_q = q

    def _own(self, q) -> None:
        """Task 13: the memo is per query. A QueryCtx made outside Env.query (query_from_vector, another Env) or an
        interleaved query resets it, so a qkey shared by two different contexts can never read the other's entries."""
        if q is not self._memo_q:
            self._clear_memo(q)

    def _put(self, memo: dict, k, v):
        for a in (v if isinstance(v, tuple) else (v,)):
            if isinstance(a, np.ndarray):
                a.setflags(write=False)      # review m1: a caller cannot poison the shared memo in place
        if len(memo) < self.memo_max:
            memo[k] = v
        return v

    def _neighbors(self, u: int, r: int) -> frozenset:
        k = (r, u)
        nb = self._nb.get(k)
        if nb is None:
            nb = self._put(self._nb, k, frozenset(self.data.graph.neighbors(np.array([u], np.int64), r).tolist()))
        return nb

    def candidates(self, q: QueryCtx, covered: frozenset, r: int) -> np.ndarray:
        """graph_candidates(covered, r) = sorted N_r(S ∪ V₀) \\ S, memoised per query (read-only result: do not
        modify). Computed as N_r(V₀) (once per query) ∪ N_r(u) for the covered u outside V₀ (per-vertex sets), minus
        S: the same sorted unique int64 rows as graph_candidates (tested)."""
        self._own(q)
        k = (q.qkey, covered, r)
        c = self._memo_c.get(k)
        if c is None:
            base = self._base.get(r)
            if base is None:
                base = self._base[r] = (frozenset(int(v) for v in q.v0),
                                        frozenset(self.data.graph.neighbors(np.asarray(q.v0, np.int64), r).tolist()))
            v0, acc = base[0], set(base[1])
            for u in covered:
                if u not in v0:
                    acc |= self._neighbors(u, r)
            acc -= covered
            c = self._put(self._memo_c, k, np.array(sorted(acc), np.int64))
        return c

    def graph_scores(self, q: QueryCtx, s: State, r: int) -> tuple[np.ndarray, np.ndarray]:
        """(candidates, Δ_V) of edge type r in `s`, memoised per query by (context in order, covered, r): the same
        arrays graph_delta computes itself, shared by the three m of a type and by the oracle's lookahead."""
        self._own(q)
        k = (q.qkey, s.context, s.covered, r)
        v = self._memo_s.get(k)
        if v is None:
            cand = self.candidates(q, s.covered, r)
            score = (delta_v(self.data, q, cand, s.context, self.lambda_red, self.decimals, sims=self._sims)
                     if cand.size else cand)
            v = self._put(self._memo_s, k, (cand, score))
        return v

    def reset(self, q: QueryCtx) -> State:
        return State(q.qkey, 0, (), frozenset(), frozenset(), frozenset(int(v) for v in q.v0), 0)

    def _budget(self, s: State) -> Budget:
        return Budget(len(s.context), s.length, self.limits.K_max, self.limits.L_max)

    def effect(self, q: QueryCtx, s: State, a: Action) -> tuple[list[int], list[int]]:
        d, b = self.data, self._budget(s)
        if a.op == "lexical":
            return ranked_delta(q.lex, s.ctx_set, a.k, b, d.chunk_tokens), []
        if a.op == "vector":
            return ranked_delta(q.vec, s.ctx_set, a.k, b, d.chunk_tokens), []
        if a.op == "graph":
            r = d.graph.type_id(a.r)
            cand, score = self.graph_scores(q, s, r)
            sel, added = graph_delta(d, q, s.covered, s.context, s.ctx_set, r, a.m, b,
                                     self.per_vertex, self.lambda_red, self.decimals, cand=cand, score=score)
            return added, sel
        raise ValueError(f"no effect for {a.op}")

    def mask(self, q: QueryCtx, s: State) -> np.ndarray:
        """Valid actions of `s` (U4). Memoised per query by (ctx_set, length, covered): the budget is the fragment
        count and length, the graph candidates depend on covered only (Task 13; length is part of the key so a state
        whose length is not the token sum of ctx_set cannot read another's mask, review m2). Returns a fresh copy."""
        if s.done:
            return np.zeros(len(self.actions), bool)
        self._own(q)
        k = (q.qkey, s.ctx_set, s.length, s.covered)
        m = self._memo_m.get(k)
        if m is None:
            m = self._put(self._memo_m, k, self._mask(q, s))
        return m.copy()

    def _mask(self, q: QueryCtx, s: State) -> np.ndarray:
        m = np.zeros(len(self.actions), bool)
        m[0] = True
        d, b = self.data, self._budget(s)
        ok = {"lexical": bool(ranked_delta(q.lex, s.ctx_set, 1, b, d.chunk_tokens)),
              "vector": bool(ranked_delta(q.vec, s.ctx_set, 1, b, d.chunk_tokens))}
        graph_ok = {r: self.candidates(q, s.covered, d.graph.type_id(r)).size > 0 for r in d.graph.edge_types}
        for i, a in enumerate(self.actions[1:], start=1):
            m[i] = graph_ok[a.r] if a.op == "graph" else ok[a.op]
        return m

    def _any_valid(self, q: QueryCtx, s: State) -> bool:
        """mask(q, s)[1:].any() with early exit (the end-of-episode "exhausted" check of every step)."""
        m = self._memo_m.get((q.qkey, s.ctx_set, s.length, s.covered)) if q is self._memo_q else None
        if m is not None:
            return bool(m[1:].any())
        d, b = self.data, self._budget(s)
        if ranked_delta(q.lex, s.ctx_set, 1, b, d.chunk_tokens) or ranked_delta(q.vec, s.ctx_set, 1, b, d.chunk_tokens):
            return True
        return any(self.candidates(q, s.covered, d.graph.type_id(r)).size for r in d.graph.edge_types)

    def _limit_reason(self, s: State) -> str | None:
        L = self.limits
        if len(s.context) >= L.K_max:
            return "k_max"
        if s.length >= L.L_max:
            return "l_max"
        if s.t >= L.T_max:
            return "t_max"
        return None

    def step(self, q: QueryCtx, s: State, a: int, check: bool = True) -> State:
        if s.done:
            raise InvalidAction("the episode is over")
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= int(a) < len(self.actions):
            raise InvalidAction(f"action id {a!r} is not in [0, {len(self.actions)})")
        a = int(a)                                   # history stays JSON-serialisable
        if check and not self.mask(q, s)[a]:
            raise InvalidAction(f"action {a} {self.actions[a]} is masked in this state")
        t = s.t + 1
        act = self.actions[a]
        if act.op == "stop":
            return replace(s, t=t, history=s.history + (Step(t, a, (), ()),), done=True, reason="stop")
        added, sel = self.effect(q, s, act)
        if not added and not sel:                    # cheap exact mask check (check=False): a valid retrieval
            raise InvalidAction(f"action {a} {act} is masked in this state")   # action adds or selects ≥ 1
        d = self.data
        new_cov = set(sel) | {int(d.chunk_node[c]) for c in added}
        covered = s.covered | new_cov
        fresh = np.array(sorted(new_cov - s.covered), np.int64)
        frontier = (s.frontier | set(d.graph.all_neighbors(fresh).tolist())) - covered
        length = s.length + (int(d.chunk_tokens[added].sum()) if added else 0)
        s2 = State(s.qkey, t, s.context + tuple(added), s.ctx_set | set(added), frozenset(covered),
                   frozenset(frontier), length, s.history + (Step(t, a, tuple(added), tuple(sel)),))
        reason = self._limit_reason(s2)
        if reason is None and not self._any_valid(q, s2):
            reason = "exhausted"
        return replace(s2, done=True, reason=reason) if reason else s2


def check_invariants(env: Env, q: QueryCtx, final: State) -> list[str]:
    """§9.6 on one episode by replaying it: limits, C_t prefix of C_t+1, no duplicates, length, μ (fragment vertex
    covered; graph fragments inside μ⁻¹ of the selected vertices), frontier = (V₀ ∪ N(S)) \\ S, determinism, end."""
    probs: list[str] = []
    d, L = env.data, env.limits
    s = env.reset(q)
    v0 = {int(v) for v in q.v0}
    for h in final.history:
        prev = s
        s = env.step(q, s, h.action)
        if s.context[:len(prev.context)] != prev.context:
            probs.append(f"t={s.t}: C_t not a prefix of C_t+1")
        if s.t > L.T_max or len(s.context) > L.K_max or s.length > L.L_max:
            probs.append(f"t={s.t}: limit exceeded (t {s.t}, |C| {len(s.context)}, len {s.length})")
        if len(set(s.context)) != len(s.context):
            probs.append(f"t={s.t}: duplicate fragment")
        if s.length != (int(d.chunk_tokens[list(s.context)].sum()) if s.context else 0):
            probs.append(f"t={s.t}: length mismatch")
        if any(int(d.chunk_node[c]) not in s.covered for c in s.context):
            probs.append(f"t={s.t}: a fragment's vertex is not covered (μ)")
        if env.actions[h.action].op == "graph":
            own = {int(c) for v in h.selected for c in d.mu_inv(v)}
            if not set(h.added) <= own:
                probs.append(f"t={s.t}: graph fragment outside μ⁻¹ of the selected vertices")
        want = (v0 | set(d.graph.all_neighbors(np.array(sorted(s.covered), np.int64)).tolist())) - set(s.covered)
        if set(s.frontier) != want:
            probs.append(f"t={s.t}: frontier differs from (V0 ∪ N(S)) \\ S")
    if s != final:
        probs.append("replay differs from the recorded episode (non-deterministic step)")
    if not final.done:
        probs.append("episode not terminated")
    return probs
