"""Ablation hooks for M6 (IMPLEMENTATION.md §9.3, §9.5; plan Task 6). A variant = a model-config overlay (GNN graph /
encoder kind) + an env spec (FilteredEnv: actions outside `allowed` are masked; STOP always allowed unless the variant
`require`s operator families, see FilteredEnv.stop_allowed). "full" is the unchanged M3 Env. "hybrid" (M5, user
request): STOP is masked while a lexical, vector or graph family not yet taken has a valid action. "adaptive" (M5, user
request "dynamic k"): AdaptiveEnv reinterprets the k of the lexical / vector slots as a relative score threshold p
(threshold_take); "hybrid_adaptive" = adaptive + the hybrid `require` rule. "fused_adaptive" (M5, hybrid research R1):
FusedEnv = adaptive with the lexical slots searching the convex BM25 + dense fusion (fused_scores) instead of BM25
alone; no `require` -- the lexical signal enters the ranking, not extra fragments. "struct" (M7, docs/m7 design
§3.1): StructEnv = the 28 M3 actions + the `structure` operator (31), which adds the virtual STRUCT/<root> fragment of
the document / external-act roots of frontier ∪ covered ∪ V₀ (grl.env.struct_data); struct.amend_op (Task 6d-b) adds
the `amendments` operator (34): the AMENDS-source fragments of those roots."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from graphrag.env.mdp import Action, Env, InvalidAction, check_invariants
from graphrag.retrieval.executor import Budget, delta_v, ranked_delta

ALL_OPS = ("stop", "lexical", "vector", "graph")


class FilteredEnv(Env):
    """`allowed` is ANDed into Env.mask (so every mask reader -- rollouts, oracles, describe -- sees it) and
    `allowed_types` (int relation ids) empties the observed candidates of a forbidden type (grl.model.observe).
    The frontier follows the variant too: F_t = (V₀ ∪ N_allowed(S_t)) \\ S_t, N_allowed = typed neighbours (both
    directions) over `allowed_types` only -- no_related never sees a RELATED-only neighbour, no_graph sees no graph
    neighbourhood (F_t = V₀ \\ S_t; V₀ is selected from the query node rankings, not by an edge walk). With every
    type allowed this is exactly the M3 frontier. Nothing else in the M3 step reads the frontier, so the episode is
    otherwise unchanged; check_variant_invariants is the M3 §9.6 check with this frontier rule."""

    def __init__(self, data, cfg=None, start_cfg=None, ops=None, drop_types=(), require=()):
        super().__init__(data, cfg, start_cfg)
        bad_req = set(require) - {"lexical", "vector", "graph"}
        if bad_req:
            raise ValueError(f"unknown required operator families {sorted(bad_req)}")
        bad = set(drop_types) - set(data.graph.edge_types)
        if bad:
            raise ValueError(f"unknown edge types {sorted(bad)}")
        ops = set(ops or ALL_OPS) | {"stop"}
        self.allowed = np.array([a.op in ops and not (a.op == "graph" and a.r in drop_types) for a in self.actions])
        self.allowed_types = {int(data.graph.type_id(a.r)) for a, ok in zip(self.actions, self.allowed)
                              if ok and a.op == "graph"}
        self.all_types = self.allowed_types == set(range(len(data.graph.edge_types)))
        self.required = tuple(op for op in ("lexical", "vector", "graph") if op in set(require))
        self.op_of = np.array([a.op for a in self.actions])

    def allowed_neighbors(self, rows) -> set[int]:
        rows = np.array(sorted(int(v) for v in rows), np.int64)
        if not rows.size:
            return set()
        if self.all_types:
            return set(self.data.graph.all_neighbors(rows).tolist())
        out: set[int] = set()
        for t in sorted(self.allowed_types):
            out.update(self.data.graph.neighbors(rows, t).tolist())
        return out

    def missing_ops(self, s) -> tuple[str, ...]:
        """Required operator families not yet taken in s.history (every history step is a valid, applied action:
        Env.step refuses masked or empty retrieval steps)."""
        used = {self.actions[h.action].op for h in s.history}
        return tuple(op for op in self.required if op not in used)

    def stop_allowed(self, q, s, base=None) -> bool:
        """The `require` rule (hybrid). For a live state with missing families M = required \\ ops(history), STOP is
        masked iff BOTH
          (1) M is non-empty,
          (2) some action of a family in M is valid now (in `base` = the M3 mask & `allowed`).
        Otherwise STOP stays allowed, so the requirement relaxes exactly when no missing family can be taken from
        here (budget K_max / L_max exhausted for it, no graph candidate, filtered out). There is no step-budget
        escape: repeating already-used families never unlocks STOP; the episode can still end by the M3 limits
        (t_max / k_max / l_max / exhausted, unchanged). When STOP is masked, (2) leaves a valid action, so a live
        state's mask is never all-False. A family that is invalid now but may become valid later (e.g. graph after a
        retrieval step reaches new vertices) does not keep STOP masked by itself; the rule is re-evaluated in every
        state."""
        if s.done or not self.required:
            return True
        miss = self.missing_ops(s)
        if not miss:
            return True
        if base is None:
            base = super().mask(q, s) & self.allowed
        return not bool((base & np.isin(self.op_of, miss)).any())

    def mask(self, q, s):
        m = super().mask(q, s) & self.allowed
        if self.required and not s.done and not self.stop_allowed(q, s, m):
            m[0] = False
        return m

    def _any_valid(self, q, s) -> bool:
        return bool((super().mask(q, s) & self.allowed)[1:].any())

    def step(self, q, s, a, check: bool = True):
        """A forbidden action is refused even with check=False (the M3 oracles' fast path skips the mask)."""
        if (not s.done and not isinstance(a, bool) and isinstance(a, (int, np.integer))
                and 0 <= int(a) < len(self.actions) and not self.allowed[int(a)]):
            raise InvalidAction(f"action {int(a)} {self.actions[int(a)]} is outside this variant")
        if (self.required and not s.done and not isinstance(a, bool) and isinstance(a, (int, np.integer))
                and int(a) == 0 and not self.stop_allowed(q, s)):
            raise InvalidAction(f"STOP is masked: required operator families {self.missing_ops(s)} not taken yet")
        s2 = super().step(q, s, a, check)
        if self.all_types:
            return s2
        frontier = (set(s.frontier) | self.allowed_neighbors(s2.covered - s.covered)) - s2.covered
        return replace(s2, frontier=frozenset(frontier))


def check_variant_invariants(env, q, final) -> list[str]:
    """M3 check_invariants (§9.6) with the variant frontier rule of FilteredEnv; identical to it for Env. With
    `require` (hybrid) also: every live state's mask has a valid action, and a STOP is taken only where
    stop_allowed holds (all required families taken, or no longer possible). On EnvData with STRUCT rows (M7) also:
    `structure` adds only rows >= n_base_chunks and every other operator only rows < n_base_chunks (the μ⁻¹ check of
    check_invariants applies to `graph` only; a STRUCT row is outside μ⁻¹ by construction); `amendments` (Task 6d-b)
    adds only candidates of the state it was taken in (AMENDS-source fragments of the reached roots)."""
    if isinstance(env, FilteredEnv) and env.required:
        s, probs = env.reset(q), []                      # the require rule first: a forbidden STOP is reported, not
        for h in final.history:                          # raised by the replays below
            if not env.mask(q, s).any():
                probs.append(f"t={s.t}: all-False mask in a live state")
            if h.action == 0 and not env.stop_allowed(q, s):
                probs.append(f"t={s.t}: STOP before the required families {env.missing_ops(s)}")
                return probs
            try:
                s = env.step(q, s, h.action)
            except InvalidAction as e:
                probs.append(f"t={s.t}: action {h.action} refused on replay ({e})")
                return probs
        if probs:
            return probs
    probs = check_invariants(env, q, final)
    probs += _struct_row_probs(env, q, final)
    if not isinstance(env, FilteredEnv) or env.all_types:
        return probs
    probs = [p for p in probs if "frontier differs" not in p]
    v0, s = {int(v) for v in q.v0}, env.reset(q)
    for h in final.history:
        s = env.step(q, s, h.action)
        want = (v0 | env.allowed_neighbors(s.covered)) - set(s.covered)
        if set(s.frontier) != want:
            probs.append(f"t={s.t}: frontier differs from (V0 ∪ N_allowed(S)) \\ S")
    return probs

def _struct_row_probs(env, q, final) -> list[str]:
    """The base / STRUCT row split per operator, on a replay of the episode (EnvData without STRUCT rows: none)."""
    n0 = getattr(env.data, "n_base_chunks", None)
    if n0 is None:
        return []
    probs, s = [], env.reset(q)
    for h0 in final.history:
        prev = s
        s = env.step(q, s, h0.action)
        h = s.history[-1]
        op = env.actions[h.action].op
        if op == "structure" and any(int(c) < n0 for c in h.added):
            probs.append(f"t={h.t}: structure added a base row")
        if op != "structure" and any(int(c) >= n0 for c in h.added):
            probs.append(f"t={h.t}: {op} added a STRUCT row")
        if op == AMEND_OP and not set(h.added) <= {int(c) for c in env._amend_cand_rows(q, prev).tolist()}:
            probs.append(f"t={h.t}: {AMEND_OP} added a fragment outside the AMENDS sources of the reached roots")
    return probs


def struct_delta_v(data, q, rows, context, lambda_red: float, decimals: int, sims=None) -> np.ndarray:
    """Δ_V of STRUCT fragments (docs/m7 §3.1 "order as the graph operators"): graphrag delta_v applied to each
    fragment f as a vertex with μ⁻¹ = {f}, i.e. rel = cos(e_q, e_f) (q.chunk_qsim, the node_rel of such a vertex) and
    the redundancy max_{c'∈C} cos(e_f, e_c') -- the fragment that is added, not the root's corpus chunks."""
    rows = np.asarray(rows, np.int64)
    view = SimpleNamespace(mu_ptr=np.arange(len(rows) + 1, dtype=np.int64), mu_idx=rows, vecs64=data.vecs64)
    qv = SimpleNamespace(node_rel=np.asarray(q.chunk_qsim, np.float64)[rows])
    return delta_v(view, qv, np.arange(len(rows), dtype=np.int64), context, lambda_red, decimals, sims=sims)


AMEND_OP = "amendments"                  # Task 6d-b: the «amendments of act» operator (struct.amend_op)


def amend_pairs(data) -> list[tuple[str, str]]:
    """(src node id, dst node id) of every AMENDS edge, directed (ArtifactGraph keeps typed neighbours undirected). Read
    once from the pinned graph's edges.jsonl (grl.hard.amend_lookup.load_amends_pairs: the two ids only, no attributes)
    and cached on the EnvData as `amend_pairs`; a toy EnvData sets the attribute itself."""
    got = getattr(data, "amend_pairs", None)
    if got is None:
        from grl.hard.amend_lookup import load_amends_pairs
        got = data.amend_pairs = [(str(s), str(d)) for s, d in load_amends_pairs(data)]
    return list(got)


class StructEnv(FilteredEnv):
    """M7 (docs/m7 design §3.1): the 28 M3 actions + `structure` m ∈ m_grid (31, ids 28..30). `structure` m adds the
    STRUCT rows (grl.env.struct_data) of the roots of F_t ∪ S_t ∪ V₀ (ruling R10, spec §3.1: the document / EXT act
    of the frontier vertices and of their CONTAINS ancestors; covered ∪ V₀ keeps the earlier candidates as a subset;
    at reset F_0 = V₀) whose row is not in C yet, ranked by
    struct_delta_v desc, row asc; the first m that fit K_max / L_max (a fragment over L_max is skipped, as in
    take_fitting). The root of a vertex is grl.env.structure's: walk CONTAINS up to a `document` vertex (= a vertex
    with a STRUCT row), else the id prefix. Each added row's μ (the root) becomes covered and the frontier grows from
    it, by Env.step. Masked iff no candidate fits. Graph operators never reach a STRUCT row (μ⁻¹ unchanged).
    No _any_valid override: FilteredEnv._any_valid reads Env.mask -> self._mask, which includes `structure`.
    articles=True (Task 6c; EnvData of load_struct_env_data(articles=True)): the candidates also hold article
    STRUCT/<article> rows, ranked together with the root rows; the action space stays 31. Which articles
    (`article_scope`): "reached" (ruling R58, the default) -- the article vertices in F_t ∪ S_t, i.e. an article the
    agent has already reached (V₀ ⊆ F_0); "ancestor" -- also the article ancestor of every vertex of F_t ∪ S_t;
    "root" -- every article of the roots of F_t ∪ S_t ∪ V₀ (the Phase A rule; crowded, kept only for the R58
    diagnostic, scripts/m7_q1x_diag.py). The flag must match the data (an article row in the data with
    articles=False, or none with articles=True, is refused), so articles=False is the pilot 7 / 8 env exactly.
    amend_op=True (Task 6d-b, train.yaml struct.amend_op): + `amendments` m ∈ m_grid (34 actions, ids 31..33).
    `amendments` m adds the corpus fragments μ⁻¹(u) of the SOURCE vertices u of the AMENDS edges u -> w whose
    destination w lies under a root of F_t ∪ S_t ∪ V₀ (the `structure` root set), not in C yet, ranked by the same
    fragment Δ_V (struct_delta_v) desc, row asc; the first m that fit K_max / L_max. Only (src, dst) of the edges is
    read (R60: never the evidence spans the R33 Q2 reference set is built from). Composes with `articles`;
    amend_op=False is the 31-action env exactly (no slot, no index, no data read)."""

    ARTICLE_SCOPES = ("reached", "ancestor", "root")

    def __init__(self, data, cfg=None, start_cfg=None, articles: bool = False, article_scope: str = "reached",
                 amend_op: bool = False):
        if not hasattr(data, "struct_rows") or not hasattr(data, "n_base_chunks"):
            raise ValueError("StructEnv needs EnvData with structure rows (grl.env.struct_data.load_struct_env_data)")
        art = dict(getattr(data, "struct_art_rows", None) or {})
        if bool(articles) != bool(art):
            raise ValueError(f"StructEnv(articles={bool(articles)}) on EnvData with {len(art)} article STRUCT rows "
                             "(load_struct_env_data(articles=...) must match the env's `articles`)")
        if article_scope not in self.ARTICLE_SCOPES:
            raise ValueError(f"article_scope {article_scope!r} not in {self.ARTICLE_SCOPES}")
        self.articles, self.article_scope = bool(articles), article_scope
        self.amend_op = bool(amend_op)
        super().__init__(data, cfg, start_cfg)
        new = [Action("structure", m=int(m)) for m in self.cfg["m_grid"]]
        if self.amend_op:                                     # Task 6d-b: ids 31..33 after the structure slots
            new += [Action(AMEND_OP, m=int(m)) for m in self.cfg["m_grid"]]
        self.actions = tuple(self.actions) + tuple(new)
        self.allowed = np.concatenate([self.allowed, np.ones(len(new), bool)])
        self.op_of = np.array([a.op for a in self.actions])
        self._root_row = self._roots()
        self._art = {int(v): int(r) for v, r in art.items()}  # article node row -> its STRUCT row
        self._art_rows = self._articles_by_root(art)          # root node row -> its article STRUCT rows (sorted)
        self._art_anc = self._article_ancestors() if art else None
        self._memo_st: dict = {}             # (qkey, context, covered) -> (STRUCT rows, struct_delta_v), per query
        self._memo_am: dict = {}             # (qkey, context, covered) -> (amendment rows, struct_delta_v), per query
        self._amend_rows = self._amend_index() if self.amend_op else {}   # root node row -> sorted fragment rows

    def _clear_memo(self, q) -> None:
        super()._clear_memo(q)
        self._memo_st.clear()
        if hasattr(self, "_memo_am"):
            self._memo_am.clear()

    # ---- Task 6d-b: «amendments of act» ----------------------------------------------------------------------------

    def _amend_index(self) -> dict[int, np.ndarray]:
        """root node row -> the sorted corpus fragment rows (μ⁻¹, rows < n_base_chunks) of the SOURCE vertices of the
        AMENDS edges whose destination lies under that root (grl.hard.amend_lookup.build_amend_index on this env's
        root map). Only (src, dst) of the edges is read -- never their attributes / evidence spans (R60: the R33 Q2
        reference set is built from them)."""
        from grl.hard.amend_lookup import build_amend_index
        d, n0 = self.data, int(self.data.n_base_chunks)
        idx = build_amend_index(d, self._root_row, amend_pairs(d))
        out = {}
        for r, srcs in idx.sources.items():
            rows = sorted({int(c) for v in srcs for c in d.mu_inv(int(v)) if int(c) < n0})
            if rows:
                out[int(r)] = np.array(rows, np.int64)
        return out

    def _amend_cand_rows(self, q, s) -> np.ndarray:
        """Unranked candidates: the AMENDS-source fragments of the roots of F_t ∪ S_t ∪ V₀ (the `structure` root
        set) that have incoming AMENDS, not in C yet; sorted rows."""
        reached = set(s.frontier) | set(s.covered) | {int(v) for v in q.v0}
        roots = sorted({int(self._root_row[u]) for u in reached} - {-1})
        parts = [self._amend_rows[r] for r in roots if r in self._amend_rows]
        if not parts:
            return np.zeros(0, np.int64)
        rows = np.unique(np.concatenate(parts))
        if s.ctx_set:
            rows = rows[~np.isin(rows, np.fromiter(s.ctx_set, np.int64, len(s.ctx_set)))]
        return rows

    def amend_scores(self, q, s) -> tuple[np.ndarray, np.ndarray]:
        """(fragment rows, Δ_V) of the `amendments` candidates ranked by Δ_V desc, row asc -- the fragment-level Δ_V
        of the structure operator (struct_delta_v: cos(e_q, e_f) − λ_red · max cos to C). Memoised per query by
        (context in order, covered), as struct_scores. Empty without amend_op."""
        self._own(q)
        k = (q.qkey, s.context, s.covered)
        v = self._memo_am.get(k)
        if v is not None:
            return v
        rows = self._amend_cand_rows(q, s) if self.amend_op else np.zeros(0, np.int64)
        if not rows.size:
            return self._put(self._memo_am, k, (rows, np.zeros(0, np.float64)))
        sc = np.asarray(struct_delta_v(self.data, q, rows, s.context, self.lambda_red, self.decimals,
                                       sims=self._sims))
        order = np.lexsort((rows, -sc))
        return self._put(self._memo_am, k, (rows[order], sc[order]))

    def amend_candidates(self, q, s) -> np.ndarray:
        return self.amend_scores(q, s)[0]

    def amend_any_fits(self, q, s) -> bool:
        """bool(_amend_take(q, s, 1)) without scoring (the mask's existence test)."""
        b = self._budget(s)
        if b.n_ctx >= b.K_max:
            return False
        rows = self._amend_cand_rows(q, s)
        return bool(rows.size) and bool((self.data.chunk_tokens[rows] <= b.L_max - b.length).any())

    def _amend_take(self, q, s, m: int) -> list[int]:
        return self._take_fitting(s, self.amend_candidates(q, s), m)

    def _take_fitting(self, s, ranked, m: int) -> list[int]:
        b, tok = self._budget(s), self.data.chunk_tokens
        out, length = [], b.length
        for c in ranked.tolist():
            if b.n_ctx + len(out) >= b.K_max or len(out) >= m:
                break
            if length + int(tok[c]) > b.L_max:
                continue
            out.append(int(c))
            length += int(tok[c])
        return out

    def _roots(self) -> np.ndarray:
        from grl.env.structure import _prefix_root
        d = self.data
        roots, parent = set(d.struct_rows), d.graph.parent
        out = np.full(d.n_nodes, -1, np.int64)
        for v, nid in enumerate(d.node_ids):
            x = v
            while x not in roots and parent[x] >= 0:          # CONTAINS is a forest (ArtifactGraph): no cycle
                x = int(parent[x])
            if x not in roots:
                x = d.node_row.get(_prefix_root(nid), -1)
            out[v] = x if x in roots else -1
        return out

    def _articles_by_root(self, art: dict) -> dict[int, tuple[int, ...]]:
        """Task 6c: root node row -> the STRUCT rows of its articles (article node row -> chunk row in `art`), by the
        same root walk as _roots; {} without article rows."""
        out: dict[int, list[int]] = {}
        for v, row in art.items():
            r = int(self._root_row[int(v)])
            if r < 0:
                raise ValueError(f"article STRUCT row of {self.data.node_ids[int(v)]} has no root")
            out.setdefault(r, []).append(int(row))
        return {r: tuple(sorted(rows)) for r, rows in out.items()}

    def _article_ancestors(self) -> np.ndarray:
        """node row -> the nearest article (a vertex with an article STRUCT row) on its CONTAINS path up, itself
        included; -1 if none."""
        d, parent = self.data, self.data.graph.parent
        out = np.full(d.n_nodes, -1, np.int64)
        for v in range(d.n_nodes):
            x = v
            while x not in self._art and parent[x] >= 0:
                x = int(parent[x])
            out[v] = x if x in self._art else -1
        return out

    def _struct_rows(self, q, s) -> list[int]:
        """Unranked candidate STRUCT rows: the rows of the roots of F_t ∪ S_t ∪ V₀ (+ with articles: the article rows
        of `article_scope`) not in C yet."""
        d = self.data
        reached = set(s.frontier) | set(s.covered)
        roots = {int(self._root_row[u]) for u in reached | {int(v) for v in q.v0}} - {-1}
        rows = [d.struct_rows[r] for r in roots]
        if self._art:
            if self.article_scope == "root":
                rows += [c for r in roots for c in self._art_rows.get(r, ())]
            else:
                arts = {int(u) for u in reached if int(u) in self._art}
                if self.article_scope == "ancestor":
                    arts |= {int(self._art_anc[u]) for u in reached} - {-1}
                rows += [self._art[v] for v in sorted(arts)]
        return [c for c in rows if c not in s.ctx_set]

    def struct_scores(self, q, s) -> tuple[np.ndarray, np.ndarray]:
        """(STRUCT rows, struct_delta_v) of the candidates, ranked by Δ_V desc, row asc (grl.model.observe reads it as
        the 8th candidate type). Memoised per query by (context in order, covered), as Env.graph_scores: the
        frontier is a function of covered ((V₀ ∪ N_allowed(S)) \\ S), ctx_set and Δ_V of the context (read-only
        arrays)."""
        self._own(q)
        k = (q.qkey, s.context, s.covered)
        v = self._memo_st.get(k)
        if v is not None:
            return v
        d = self.data
        rows = np.array(sorted(self._struct_rows(q, s)), np.int64)
        if not rows.size:
            return self._put(self._memo_st, k, (rows, np.zeros(0, np.float64)))
        sc = np.asarray(struct_delta_v(d, q, rows, s.context, self.lambda_red, self.decimals, sims=self._sims))
        order = np.lexsort((rows, -sc))
        return self._put(self._memo_st, k, (rows[order], sc[order]))

    def struct_any_fits(self, q, s) -> bool:
        """bool(_struct_take(q, s, 1)) without scoring: some candidate STRUCT row fits L_max − length while
        |C| < K_max (the mask's existence test; no struct_delta_v, no ranking)."""
        b = self._budget(s)
        if b.n_ctx >= b.K_max:
            return False
        room, tok = b.L_max - b.length, self.data.chunk_tokens
        return any(int(tok[c]) <= room for c in self._struct_rows(q, s))

    def struct_candidates(self, q, s) -> np.ndarray:
        return self.struct_scores(q, s)[0]

    def _struct_take(self, q, s, m: int) -> list[int]:
        return self._take_fitting(s, self.struct_candidates(q, s), m)

    def effect(self, q, s, a):
        if a.op == "structure":
            return self._struct_take(q, s, a.m), []
        if a.op == AMEND_OP:
            return self._amend_take(q, s, a.m), []
        return super().effect(q, s, a)

    def _mask(self, q, s):
        """Env._mask (which looks every non-graph op up in a lexical / vector dict) on the 28 M3 actions, then the
        structure slots (struct_any_fits: an existence test, no candidate scoring); tests/test_m7_struct_env.py checks
        the M3 part equals FilteredEnv's mask."""
        m = np.zeros(len(self.actions), bool)
        m[0] = True
        d, b = self.data, self._budget(s)
        ok = {"lexical": bool(ranked_delta(q.lex, s.ctx_set, 1, b, d.chunk_tokens)),
              "vector": bool(ranked_delta(q.vec, s.ctx_set, 1, b, d.chunk_tokens)),
              "structure": self.struct_any_fits(q, s)}
        if self.amend_op:
            ok[AMEND_OP] = self.amend_any_fits(q, s)
        graph_ok = {r: self.candidates(q, s.covered, d.graph.type_id(r)).size > 0 for r in d.graph.edge_types}
        for i, a in enumerate(self.actions[1:], start=1):
            m[i] = graph_ok[a.r] if a.op == "graph" else ok[a.op]
        return m


ADAPTIVE_P = (0.9, 0.7, 0.5)            # p of the lexical / vector slots k_grid[0], k_grid[1], k_grid[2] (3, 5, 10)
SCORE_EPS = 1e-12                        # float tolerance of the threshold comparison (ties at p = 1 stay in)
NORM_TOP_N = 50                          # train.yaml dynamic_k.norm_top_n: min-max window of a channel's ranking


def minmax_scores(scores, top_n: int = NORM_TOP_N) -> np.ndarray:
    """Per-query, per-channel min-max normalisation (controller ruling, dynamic k): lo / hi = min / max of the first
    top_n scores of the ranking, every score -> (s − lo) / (hi − lo) (rows past top_n may go below 0). So p means the
    same share of the score range for BM25 and for cosine. Degenerate window (hi = lo, or no finite score): all 0, so
    threshold_take takes exactly 1 fragment."""
    if int(top_n) < 2:
        raise ValueError(f"dynamic_k.norm_top_n must be >= 2, got {top_n}")
    s = np.asarray(scores, np.float64)
    head = s[:int(top_n)]
    head = head[np.isfinite(head)]
    if not head.size or not float(head.max()) - float(head.min()) > SCORE_EPS:
        return np.zeros_like(s)
    lo, hi = float(head.min()), float(head.max())
    return (s - lo) / (hi - lo)


def threshold_take(rows, scores, exclude, p: float, budget: Budget, tokens) -> list[int]:
    """Dynamic-k ΔC over a ranking with scores (descending, as U9 stores them): the first unseen fragment that fits
    the budget is always taken; its score s₁ (the best remaining score) sets the threshold p·s₁, and the following
    unseen fragments are taken in ranking order while their score ≥ p·s₁ (a fragment that does not fit L_max is
    skipped, as in take_fitting), up to K_max fragments / L_max tokens. Empty iff take_fitting(rows, exclude, 1, ...)
    is empty, so the M3 mask of a search family is unchanged. Deterministic; a prefix of take_fitting(rows, exclude,
    K_max, ...) (nested in p: a smaller p takes a longer prefix). s₁ ≤ 0 or non-finite: only that one fragment."""
    if not 0.0 < float(p) <= 1.0:
        raise ValueError(f"threshold p must be in (0, 1], got {p}")
    out: list[int] = []
    seen: set[int] = set()
    length, n, thr = budget.length, budget.n_ctx, None
    for c, sc in zip(rows, scores):
        c = int(c)
        if n >= budget.K_max:
            break
        if c < 0 or c in exclude or c in seen:
            continue
        sc = float(sc)
        if thr is not None and not sc >= thr:
            break
        t = int(tokens[c])
        if length + t > budget.L_max:
            continue
        out.append(c)
        seen.add(c)
        length += t
        n += 1
        if thr is None:
            thr = float(p) * sc - SCORE_EPS if np.isfinite(sc) and sc > 0 else np.inf
    return out


def with_scores(q, rank):
    """Attach the U9 BM25 scores (aligned with q.lex) to a QueryCtx; idempotent. The dense scores need no copy:
    q.chunk_qsim is the cosine the vector ranking was sorted by."""
    sc = np.asarray(rank.lex_score, np.float64)
    if len(sc) != len(q.lex):
        raise ValueError(f"{q.qkey}: {len(sc)} lexical scores for {len(q.lex)} ranked fragments")
    q.lex_score = sc
    return q


def search_scores(q, op: str) -> np.ndarray:
    """Scores aligned with the ranking of a search family: BM25 (lexical, via with_scores) or cosine (vector)."""
    if op == "vector":
        return np.asarray(q.chunk_qsim, np.float64)[np.asarray(q.vec, np.int64)]
    if op == "lexical":
        sc = getattr(q, "lex_score", None)
        if sc is None:
            raise ValueError("lexical scores missing: build the query with env.query(rank) of an AdaptiveEnv or "
                             "grl.model.variants.with_scores")
        return sc
    raise ValueError(f"no ranking scores for {op!r}")


class AdaptiveEnv(FilteredEnv):
    """Adaptive-k search (M5 "dynamic k"): the same 28 action ids; lexical slots 1–3 and vector slots 4–6 take
    threshold_take with p = ADAPTIVE_P[i] for the slot of k_grid[i] (k is only the slot label), on the channel's
    scores min-max normalised over its top `norm_top_n` (minmax_scores; `normalise=False` = raw BM25 / cosine). The
    threshold is p × the (normalised) score of the best REMAINING fragment. Graph actions, the
    mask (a search family is valid iff it can add ≥ 1 fragment, as in M3), the limits and the `require` rule are
    unchanged, so every live mask has a valid action exactly as in FilteredEnv. The state preview of a family is the
    ΔC of its most conservative action (largest p), cut to `preview` fragments (search_preview) -- the fragments any
    adaptive search of that family adds first; for the fixed-k Env the same rule (smallest k) is the M3 preview."""

    def __init__(self, data, cfg=None, start_cfg=None, ops=None, drop_types=(), require=(), p_grid=ADAPTIVE_P,
                 normalise: bool = True, norm_top_n: int = NORM_TOP_N):
        super().__init__(data, cfg, start_cfg, ops=ops, drop_types=drop_types, require=require)
        if int(norm_top_n) < 2:
            raise ValueError(f"dynamic_k.norm_top_n must be >= 2, got {norm_top_n}")
        self.normalise, self.norm_top_n = bool(normalise), int(norm_top_n)
        ks = [a.k for a in self.actions if a.op == "lexical"]
        p_grid = tuple(float(p) for p in p_grid)
        if len(ks) != len(p_grid) or len(set(ks)) != len(ks):
            raise ValueError(f"adaptive needs one p per k slot: k_grid {ks}, p {list(p_grid)}")
        if not all(0.0 < p <= 1.0 for p in p_grid):
            raise ValueError(f"adaptive p must be in (0, 1], got {list(p_grid)}")
        self.p_of = dict(zip(ks, p_grid))
        self.p_preview = max(p_grid)

    def query(self, rank):
        return with_scores(super().query(rank), rank)

    def channel_scores(self, q, op: str) -> np.ndarray:
        sc = search_scores(q, op)
        return minmax_scores(sc, self.norm_top_n) if self.normalise else sc

    def search_delta(self, q, s, op: str, p: float) -> list[int]:
        rows = q.lex if op == "lexical" else q.vec
        return threshold_take(rows, self.channel_scores(q, op), s.ctx_set, p, self._budget(s), self.data.chunk_tokens)

    def effect(self, q, s, a):
        if a.op in ("lexical", "vector"):
            return self.search_delta(q, s, a.op, self.p_of[a.k]), []
        return super().effect(q, s, a)

    def search_preview(self, q, s, op: str, n: int) -> list[int]:
        return self.search_delta(q, s, op, self.p_preview)[:int(n)]


FUSE_W = 0.7                             # train.yaml dynamic_k.fuse_w: dense weight w of the fused lexical slots


def check_fuse_w(w) -> float:
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"dynamic_k.fuse_w must be in [0, 1], got {w}")
    return w


def _affine_minmax(head) -> tuple[float, float] | None:
    head = np.asarray(head, np.float64)
    head = head[np.isfinite(head)]
    if not head.size or not float(head.max()) - float(head.min()) > SCORE_EPS:
        return None
    return float(head.min()), float(head.max())


def fused_scores(q, w: float, top_n: int = NORM_TOP_N) -> tuple[np.ndarray, np.ndarray]:
    """Convex combination of the two channels (Bruch et al. 2023, TOIS): rows = q.lex ∪ q.vec (a superset of the BM25
    ranking, so a lexical slot is never emptier than in M3), fused = w·mm(cos) + (1−w)·mm(BM25), then min-max
    normalised over its own top `top_n` (minmax_scores) so p means the same share of the range as for one channel.
    mm = the per-query, per-channel min-max of minmax_scores (lo / hi of the channel's top top_n), applied to every
    row: the cosine is known for every fragment (q.chunk_qsim); a row outside the BM25 list gets the lowest BM25 value
    min(0, min mm over the list). A degenerate channel window contributes 0. Sorted by fused score desc, row asc.
    Deterministic; no C*. Cached on q per (w, top_n). w = 1: the dense order (ties aside); w = 0: BM25 first."""
    w, top_n = check_fuse_w(w), int(top_n)
    if top_n < 2:
        raise ValueError(f"dynamic_k.norm_top_n must be >= 2, got {top_n}")
    key = (w, top_n)
    cache = getattr(q, "_fused", None)
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    lex = np.asarray(q.lex, np.int64)
    vec = np.asarray(q.vec, np.int64)
    rows = np.array(sorted(set(lex.tolist()) | set(vec.tolist())), np.int64)
    cos = np.asarray(q.chunk_qsim, np.float64)
    vs = cos[vec]
    lim = _affine_minmax(vs[:top_n])
    d = (cos[rows] - lim[0]) / (lim[1] - lim[0]) if lim else np.zeros(len(rows))
    bm = minmax_scores(search_scores(q, "lexical"), top_n)
    floor = min(0.0, float(bm.min())) if bm.size else 0.0
    by_row = dict(zip(lex.tolist(), bm.tolist()))
    b = np.array([by_row.get(int(c), floor) for c in rows], np.float64)
    f = w * d + (1.0 - w) * b
    order = np.lexsort((rows, -f))
    rows, f = rows[order], f[order]
    f = minmax_scores(f, top_n)
    try:
        q._fused = (key, rows, f)
    except AttributeError:                       # a frozen QueryCtx: no cache, same result
        pass
    return rows, f


class FusedEnv(AdaptiveEnv):
    """fused_adaptive: AdaptiveEnv whose lexical slots 1–3 take threshold_take (p = ADAPTIVE_P) on fused_scores(q, w)
    -- BM25 + dense, convex weight w (dense) -- instead of BM25; the vector slots stay dense-adaptive, graph, the
    limits and the mask are unchanged. Mask: lexical valid iff the M3 BM25 ranking can add ≥ 1 fragment; the fused
    rows ⊇ the BM25 rows, so a valid lexical slot always adds ≥ 1 (conservative: a fused fragment outside BM25 never
    unmasks it). The state preview of the lexical family is the fused ΔC at the largest p (AdaptiveEnv rule)."""

    def __init__(self, data, cfg=None, start_cfg=None, fuse_w: float = FUSE_W, **kw):
        super().__init__(data, cfg, start_cfg, **kw)
        if not self.normalise:
            raise ValueError("fused_adaptive needs dynamic_k.normalise: a convex mix of raw BM25 and cosine is "
                             "scale-dependent")
        self.fuse_w = check_fuse_w(fuse_w)

    def search_delta(self, q, s, op: str, p: float) -> list[int]:
        if op != "lexical":
            return super().search_delta(q, s, op, p)
        rows, sc = fused_scores(q, self.fuse_w, self.norm_top_n)
        return threshold_take(rows, sc, s.ctx_set, p, self._budget(s), self.data.chunk_tokens)


def search_preview(env, q, s, op: str, n: int) -> list[int]:
    """Fragments a search of family `op` would add first in `s` (grl.model.observe preview): the env's own rule if it
    has one (AdaptiveEnv), else the M3 one (next n fitting fragments = ΔC of the smallest k, cut to n)."""
    own = getattr(env, "search_preview", None)
    if own is not None:
        return own(q, s, op, n)
    return ranked_delta(q.lex if op == "lexical" else q.vec, s.ctx_set, int(n), env._budget(s), env.data.chunk_tokens)


VARIANTS = {
    "full": {},
    "no_graph": {"ops": ["stop", "lexical", "vector"], "model": {"encoder": "mlp"}},
    "no_lexical": {"ops": ["stop", "vector", "graph"]},
    "no_related": {"drop_types": ["RELATED"], "model": {"graph": {"drop_types": ["RELATED"]}}},
    "no_reverse": {"model": {"graph": {"reverse_edges": False}}},
    "no_gnn": {"model": {"encoder": "mlp"}},
    "hybrid": {"require": ["lexical", "vector", "graph"]},      # M5: lexical + vector + graph before STOP
    "adaptive": {"adaptive": True},                              # M5: search slots = relative thresholds p
    "hybrid_adaptive": {"require": ["lexical", "vector", "graph"], "adaptive": True},
    "fused_adaptive": {"adaptive": True, "fused": True},       # M5: lexical slots = BM25 + dense convex fusion
    "rerank": {"rerank": True},                                  # M5: pointer policy over the fused top-N (rerank.py)
    "rank": {"rerank": True, "rank": True},                      # M5 pilot 6: the same policy as a top-k ranker (rank.py)
    "struct": {"struct": True},                                  # M7: + the `structure` operator (31 actions)
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def model_config(base: dict, name: str) -> dict:
    return _merge(base, VARIANTS[name].get("model", {}))


def env_spec(name: str):
    v = VARIANTS[name]
    if v.get("struct"):
        return StructEnv, {}
    if v.get("rank"):
        from grl.model.rank import RankEnv
        return RankEnv, {}
    if v.get("rerank"):
        from grl.model.rerank import RerankEnv
        return RerankEnv, {}
    if not v.get("ops") and not v.get("drop_types") and not v.get("require") and not v.get("adaptive"):
        return Env, {}
    kw = {"ops": v.get("ops"), "drop_types": tuple(v.get("drop_types", ()))}
    if v.get("require"):
        kw["require"] = tuple(v["require"])
    if v.get("fused"):
        return FusedEnv, kw
    return (AdaptiveEnv if v.get("adaptive") else FilteredEnv), kw


def is_adaptive(name: str) -> bool:
    return bool(VARIANTS[name].get("adaptive"))


def is_fused(name: str) -> bool:
    return bool(VARIANTS[name].get("fused"))


def is_rerank(name: str) -> bool:
    return bool(VARIANTS[name].get("rerank"))


def is_rank(name: str) -> bool:
    return bool(VARIANTS[name].get("rank"))


def is_struct(name: str) -> bool:
    return bool(VARIANTS[name].get("struct"))


def load_variant_data(name: str, version: str | None = None, articles: bool = False):
    """EnvData of a variant: + the STRUCT rows for `struct` (grl.env.struct_data.load_struct_env_data; articles=True,
    Task 6c: + the STRUCT/<article> rows), the pinned load_env_data otherwise."""
    if is_struct(name):
        from grl.env import struct_data
        if articles:
            return struct_data.load_struct_env_data(version=version, articles=True)
        return struct_data.load_struct_env_data(version=version)
    if articles:
        raise ValueError(f"articles=True is a struct-variant option (variant {name!r})")
    from graphrag.env import data
    return data.load_env_data(version)


def rank_kw(cfg: dict) -> dict:
    """The rank env's own kwargs (train.yaml rank.k_max / rank.static), pilot 6."""
    rk = cfg.get("rank") or {}
    return {"k_max": int(rk.get("k_max", 20)), "static": bool(rk.get("static", True))}


def rerank_kw(cfg: dict) -> dict:
    """The rerank env's EFFECT kwargs (candidate list: rerank.n, dynamic_k.fuse_w / norm_top_n) -- what the expert
    labels depend on and the SFT gate compares (grl.train.data.check_variant)."""
    rr, dk = cfg.get("rerank") or {}, cfg.get("dynamic_k") or {}
    if not bool(dk.get("normalise", True)):
        raise ValueError("rerank fuses min-max normalised scores: dynamic_k.normalise must stay true")
    return {"n": int(rr.get("n", 20)), "fuse_w": check_fuse_w(dk.get("fuse_w", FUSE_W)),
            "norm_top_n": int(dk.get("norm_top_n", NORM_TOP_N))}


def rerank_obs_kw(cfg: dict) -> dict:
    """The rerank env's observation-only kwargs (rerank.use_ce, rerank.ce_cache): not part of the label key."""
    rr = cfg.get("rerank") or {}
    return {"use_ce": bool(rr.get("use_ce", False)), "ce_cache": rr.get("ce_cache") or None}


def variant_env_kw(name: str, cfg: dict, observe: bool = False) -> dict:
    """Extra env kwargs of a variant from train.yaml: adaptive_kw (dynamic_k) for the adaptive ones, rerank_kw
    (+ rerank_obs_kw when observe) for rerank, {} otherwise."""
    if is_rank(name):
        return {**rerank_kw(cfg), **rank_kw(cfg), **(rerank_obs_kw(cfg) if observe else {})}
    if is_rerank(name):
        return {**rerank_kw(cfg), **(rerank_obs_kw(cfg) if observe else {})}
    if is_struct(name):
        return struct_kw(cfg)
    return adaptive_kw(name, cfg.get("dynamic_k"))


def struct_kw(cfg: dict) -> dict:
    """Task 6c / 6d-b: the struct env's kwargs from train.yaml `struct` -- "articles": True when struct.articles is on,
    "amend_op": True when struct.amend_op is on (the label key then records them, so the SFT gate refuses labels of
    the other mode); a flag that is off adds no key, so both off is {} (the pilot 7 / 8 key, unchanged)."""
    st = cfg.get("struct") or {}
    out = {}
    for k in ("articles", "amend_op"):
        v = st.get(k, False)
        if not isinstance(v, bool):
            raise ValueError(f"struct.{k} must be true / false, got {v!r}")
        if v:
            out[k] = True
    return out


def struct_data_for(cfg: dict):
    """The struct EnvData a train.yaml config asks for (struct.articles; fix round 1: every M7 loader goes through
    this or data_kw, so an article STRUCT C* id is a row whenever the run is in article mode)."""
    from grl.env.struct_data import load_struct_env_data
    return load_struct_env_data(**data_kw(struct_kw(cfg)))


def data_kw(env_kw: dict | None) -> dict:
    """load_variant_data kwargs implied by a variant's env kwargs (Task 6c: articles)."""
    return {"articles": True} if (env_kw or {}).get("articles") else {}


def adaptive_kw(name: str, dk: dict | None) -> dict:
    """Extra AdaptiveEnv kwargs from train.yaml `dynamic_k` (empty for a non-adaptive variant; + fuse_w for a fused
    one)."""
    if not is_adaptive(name):
        return {}
    dk = dk or {}
    kw = {"normalise": bool(dk.get("normalise", True)), "norm_top_n": int(dk.get("norm_top_n", NORM_TOP_N))}
    if is_fused(name):
        kw["fuse_w"] = check_fuse_w(dk.get("fuse_w", FUSE_W))
    return kw
