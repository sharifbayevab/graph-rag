"""Reference policies (roadmap §5 M3 row; §3 item 10; U6): random (seeded, uniform over valid actions), greedy
oracle, beam oracle (train.yaml oracle.*, depth ≤ T_max; max R among ACCEPTED finals, ties shortest, then
lexicographic; lookahead + operator-diverse pruning, ruling M3-R6) -- the expert trajectories of §8.1, labelled with the per-step class of equivalent actions and the mask for SFT (M4).
The oracle knows C*: an upper bound and a teacher, never a deployable policy."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from graphrag.env.mdp import InvalidAction
from graphrag.env.reward import reward
from graphrag.retrieval.executor import Budget, ranked_delta, take_fitting


def rollout(env, q, choose):
    s = env.reset(q)
    while not s.done:
        s = env.step(q, s, choose(env, q, s))
    return s


def random_policy(seed: int, allow_stop: bool = True):
    rng = np.random.default_rng(seed)

    def choose(env, q, s) -> int:
        m = env.mask(q, s)
        if not allow_stop:
            m = m.copy()
            m[0] = not m[1:].any()
        return int(rng.choice(np.flatnonzero(m)))
    return choose


def _R(env, q, s, gold, w) -> float:        # `s` needs only `.context`
    return reward(env.data, q.depth, s.context, gold, w, env.limits.T_max).total


def greedy_oracle(env, q, gold, w):
    s, cur = env.reset(q), 0.0
    while not s.done:
        best = None
        for a in np.flatnonzero(env.mask(q, s)).tolist():
            if a == 0:
                continue
            c = env.step(q, s, a, check=False)
            r = _R(env, q, c, gold, w)
            if best is None or r > best[0] + 1e-12:
                best = (r, c)
        if best is None or best[0] <= cur + 1e-12:
            return env.step(q, s, 0, check=False)
        cur, s = best
    return s


def d5_accept(gold: frozenset):
    """Decision П5 without the R threshold (applied by the caller): the episode ends with STOP and reaches all C*."""
    return lambda s: s.reason == "stop" and gold <= s.ctx_set


def oracle_config(**over) -> dict:
    """train.yaml `oracle:` (M3-R6): beam_width (32), graph_share (≥ ⌈share·w⌉ graph slots), lookahead (bool)."""
    from graphrag import config
    cfg = {"beam_width": 32, "graph_share": 0.25, "lookahead": True}
    cfg.update(config.load("train").get("oracle") or {})
    cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


class BeamSearch:
    """One beam search per (q, gold): `finals` are all candidate finals (the STOP-at-reset final, the greedy final and
    the STOP-completion of every kept child); `best(accept)` filters, then takes the argmax of true R -- so any number
    of acceptance criteria (d5, unconstrained) share one search (review m4).

    Ranking inside the beam (M3-R6) uses a score, not R: a child whose last step added no fragment (a graph hop
    onto chunkless vertices -- EXT stubs, ENT entities) is scored max(R, best R one fragment-adding step further),
    so two-hop paths through a stub survive the prune; ≥ ⌈graph_share·w⌉ slots go to graph-action children
    (operator-diverse beam). Finals are always judged by true R."""

    def __init__(self, env, q, gold, w, width: int | None = None, graph_share: float | None = None,
                 lookahead: bool | None = None):
        cfg = oracle_config(beam_width=width, graph_share=graph_share, lookahead=lookahead)
        self.env, self.q, self.gold, self.w = env, q, gold, w
        self.width, self.lookahead = int(cfg["beam_width"]), bool(cfg["lookahead"])
        if self.width < 1:
            raise ValueError(f"beam width {self.width} < 1")
        self.graph_slots = min(self.width, int(np.ceil(float(cfg["graph_share"]) * self.width)))
        self._r: dict[frozenset, float] = {}           # Review C1: R depends only on ctx_set (q, gold, depth fixed)
        self._look: dict[tuple, float] = {}           # lookahead: successors depend on (ctx_set, covered) only
        self.finals = self._search()

    def R(self, s) -> float:
        return self.R_set(s.ctx_set, s.context)

    def R_set(self, ctx_set: frozenset, context) -> float:
        r = self._r.get(ctx_set)
        if r is None:
            r = self._r[ctx_set] = _R(self.env, self.q, _Ctx(tuple(context)), self.gold, self.w)
        return r

    def score(self, s) -> float:
        r = self.R(s)
        if not self.lookahead or s.done or not s.history or s.history[-1].added:
            return r
        k = (s.ctx_set, s.covered)
        la = self._look.get(k)
        if la is None:
            la = r
            for added in successor_additions(self.env, self.q, s):
                if added:
                    la = max(la, self.R_set(s.ctx_set | set(added), s.context + tuple(added)))
            self._look[k] = la
        return la

    def _rank_key(self, s):
        return (-self.score(s), len(s.history), tuple(h.action for h in s.history))

    def _final_key(self, s):
        return (-self.R(s), len(s.history), tuple(h.action for h in s.history))

    def _finish(self, s):
        return s if s.done else self.env.step(self.q, s, 0, check=False)

    def _prune(self, children):
        seen, uniq = set(), []
        for c in sorted(children, key=self._rank_key):
            sig = (c.ctx_set, c.covered)
            if sig not in seen:
                seen.add(sig)
                uniq.append(c)
        is_graph = [self.env.actions[c.history[-1].action].op == "graph" for c in uniq]
        reserved = set([i for i, g in enumerate(is_graph) if g][:self.graph_slots])
        rest = [i for i in range(len(uniq)) if i not in reserved][:max(0, self.width - len(reserved))]
        return [uniq[i] for i in sorted(reserved | set(rest))]    # rank order among the kept

    def _search(self):
        env, q = self.env, self.q
        s0 = env.reset(q)
        finals = [self._finish(s0), greedy_oracle(env, q, self.gold, self.w)]
        beam = [s0]
        while beam:
            children = [env.step(q, s, a, check=False) for s in beam
                        for a in np.flatnonzero(env.mask(q, s)).tolist() if a != 0]
            kept = self._prune(children)
            finals += [self._finish(c) for c in kept]
            beam = [c for c in kept if not c.done]
        return finals

    def best(self, accept=None):
        finals = self.finals if accept is None else [f for f in self.finals if accept(f)]   # Review I2: filter first
        return min(finals, key=self._final_key) if finals else None


def successor_additions(env, q, s) -> list[tuple[int, ...]]:
    """ΔC of every valid non-STOP action of `s`, equal to `env.effect(q, s, a)[0]` action by action (tested), but
    one Δ_V and one selection loop per edge type serve all m of the grid: graph_delta's loop over the selected
    vertices is sequential, so ΔC for m is the state after the first m iterations of the loop for max(m)."""
    d, b, out = env.data, env._budget(s), []
    for a in env.actions[1:]:
        if a.op in ("lexical", "vector"):
            add = tuple(ranked_delta(q.lex if a.op == "lexical" else q.vec, s.ctx_set, a.k, b, d.chunk_tokens))
            if add:                                  # empty <=> the operator is masked
                out.append(add)
    ms = sorted({a.m for a in env.actions if a.op == "graph"})
    for r in d.graph.edge_types:
        cand, score = env.graph_scores(q, s, d.graph.type_id(r))       # Task 13: the Env's per-query memo
        if cand.size == 0:
            continue
        sel = [int(v) for v in cand[np.lexsort((cand, -score))][:ms[-1]]]
        added: list[int] = []
        after = []
        for v in sel:
            cs = sorted((int(c) for c in d.mu_inv(v)), key=lambda c: (-float(q.chunk_qsim[c]), c))
            used = int(d.chunk_tokens[added].sum()) if added else 0
            added += take_fitting(cs, set(s.ctx_set) | set(added), env.per_vertex,
                                  Budget(b.n_ctx + len(added), b.length + used, b.K_max, b.L_max), d.chunk_tokens)
            after.append(len(added))
        out += [tuple(added[:after[min(m, len(sel)) - 1]]) for m in ms]
    return out


class _Ctx:
    __slots__ = ("context",)

    def __init__(self, context):
        self.context = context


def beam_oracle(env, q, gold, w, width: int | None = None, accept=None, **kw):
    """Best accepted final of one beam search (None if no final passes `accept`); width defaults to
    train.yaml oracle.beam_width. For several criteria on one query use BeamSearch(...).best(accept) directly."""
    return BeamSearch(env, q, gold, w, width, **kw).best(accept)


def action_classes(env, q, s) -> list[list[int]]:
    """Valid actions of `s` grouped by their effect (ΔC in order, the new covered set); sorted by smallest id."""
    groups: dict[tuple, list[int]] = {}
    for a in np.flatnonzero(env.mask(q, s)).tolist():
        if a == 0:
            groups[("stop",)] = [0]
            continue
        added, sel = env.effect(q, s, env.actions[a])
        cov = frozenset(s.covered | set(sel) | {int(env.data.chunk_node[c]) for c in added})
        groups.setdefault((tuple(added), cov), []).append(a)
    return sorted(groups.values(), key=lambda g: g[0])


def trajectory_labels(env, q, final) -> tuple[list[list[int]], list[list[int]]]:
    equiv, masks = [], []
    s = env.reset(q)
    for h in final.history:
        masks.append(np.flatnonzero(env.mask(q, s)).tolist())
        cls = next((c for c in action_classes(env, q, s) if h.action in c), None)
        if cls is None:                                          # review m3: a foreign / masked history
            raise InvalidAction(f"t={h.t}: action {h.action} is not valid in the replayed state")
        equiv.append(cls)
        s = env.step(q, s, h.action, check=False)
    return equiv, masks


def trajectory_steps(env, s) -> list[dict]:
    """IMPLEMENTATION.md §8.1: [{t, op, arg}, …, {t, op: "stop"}]; arg = k (lexical / vector), {r, m} (graph)."""
    out = []
    for h in s.history:
        a = env.actions[h.action]
        out.append({"t": h.t, "op": a.op} if a.op == "stop" else {"t": h.t, "op": a.op, "arg": a.arg()})
    return out


def _require_stop(s):
    if not (s.done and s.reason == "stop" and s.history and s.history[-1].action == 0):
        raise ValueError(f"a §8.1 trajectory must end with STOP (episode reason {s.reason!r})")   # review m2


def trajectory_json(env, q, s, gold_ids: list[str], parts) -> dict:
    """The §8.1 record (+ service fields) of a STOP-ended final; an auto-terminated final is refused."""
    _require_stop(s)
    equiv, masks = trajectory_labels(env, q, s)
    return {"qkey": q.qkey, "q": q.text, "C_ref": sorted(gold_ids), "C": [env.data.chunk_ids[c] for c in s.context],
            "trajectory": trajectory_steps(env, s), "actions": [h.action for h in s.history],
            "action_equiv": equiv, "action_masks": masks, "reward": asdict(parts), "reason": s.reason}
