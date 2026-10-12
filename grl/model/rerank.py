"""RL reranker (M5 variant "rerank", user-approved spec deviation: its own action space; the 28-action variants stay
intact). The policy SELECTS fragments from a fixed candidate list -- the top-N of the fused BM25 + dense ranking
(grl.model.variants.fused_scores, dynamic_k.fuse_w / norm_top_n) -- instead of choosing search operators.

Candidates (rerank_candidates): the first N rows of the fused ranking (fused score desc, row asc) whose own length
fits L_max; deterministic, a function of the query ranking only. N = train.yaml rerank.n.
Actions: 0 = STOP (always valid in a live state, so a mask is never all-False), j + 1 = select candidate j (valid iff
j < |candidates|, not yet in the context, and it fits the remaining budget: |C| < K_max, length + tokens ≤ L_max).
Effect of selecting j: the fragment is appended to C, its vertex μ(c) becomes covered (frontier as in M3). Limits
K_max / L_max / T_max end the episode as in M3; "exhausted" when no candidate can be added any more.
Observation (RerankDesc, describe_state): per candidate the vertex row (z_node from the query-injected GAT on the
policy side) and the CAND_FEATURES -- dense / BM25 / fused scores and ranks, the cross-encoder score (only when
use_ce; zeros otherwise), document-structure features (same-document count among the candidates, CONTAINS depth,
distance to V₀, AMENDS / REFERS_TO degree -- undirected: EnvData keeps the symmetric typed CSR only; the edge
directions reach the policy through z_node, whose GAT has reverse relations), position, tokens, and three
context features (same vertex / same document as a selected fragment, adjacent to a covered vertex).
Cross-encoder scores come from the cache written by scripts/ce_scores.py (load_ce_cache): a candidate without a
cached score is a loud error when the score is needed. Settings that change what a trained policy sees (rerank_settings:
use_ce, n, fuse_w, norm_top_n, L_max, the ce cache sha256) are recorded with every SFT / GRPO run and checked by grpo /
eval (check_rerank_settings). The variant expert (train.py expert) searches exactly up to expert.depth = 3 selections
then STOP: contexts of > 3 fragments are outside its labels and outside the oracle@N ceiling of the baselines. Nothing here sees the target set (tests/test_m4_state guard)."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from graphrag.env.mdp import Action, Env, InvalidAction, State, Step
from grl.model.gnn import GraphEncoder
from grl.model.graph import GraphTensors
from grl.model.policy import masked_log_softmax
from grl.model.state import segment_mean
from grl.model.variants import FUSE_W, NORM_TOP_N, check_fuse_w, fused_scores, minmax_scores, search_scores, \
    with_scores

RERANK_N = 20                            # train.yaml rerank.n
CAND_FEATURES = ("dense_cos", "dense_norm", "bm25_norm", "bm25_in", "fused_norm", "inv_rank_dense", "inv_rank_bm25",
                 "inv_rank_fused", "pos", "ce_norm", "ce_sigmoid", "inv_rank_ce", "same_doc_cands", "tree_depth",
                 "dist_v0", "deg_amends", "deg_refers_to", "tokens", "same_node_ctx", "same_doc_ctx", "adj_covered")
N_CAND_FEATURES = len(CAND_FEATURES)
CE_COLS = (CAND_FEATURES.index("ce_norm"), CAND_FEATURES.index("ce_sigmoid"), CAND_FEATURES.index("inv_rank_ce"))
DYN_COLS = (CAND_FEATURES.index("same_node_ctx"), CAND_FEATURES.index("same_doc_ctx"),
            CAND_FEATURES.index("adj_covered"))
RERANK_SCALARS = ("t", "n_ctx", "length", "n_valid", "n_cands", "use_ce")
N_RERANK_SCALARS = len(RERANK_SCALARS)
EMPTY = np.zeros(0, np.int64)


def rerank_candidates(q, n: int, fuse_w: float, norm_top_n: int, tokens, L_max: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows, fused normalised scores) of the first n fragments of fused_scores(q) whose own length fits L_max."""
    if int(n) < 1:
        raise ValueError(f"rerank.n must be >= 1, got {n}")
    rows, f = fused_scores(q, check_fuse_w(fuse_w), int(norm_top_n))
    keep = [i for i, c in enumerate(rows.tolist()) if int(tokens[c]) <= int(L_max)][:int(n)]
    return rows[keep].astype(np.int64), np.asarray(f, np.float64)[keep]


# ---- cross-encoder cache ------------------------------------------------------------------------------------------

def ce_meta_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ".meta.json")


_CE_TABLES: dict = {}                    # (path, size, mtime_ns, graph version) -> table: one load per process


def load_ce_cache(path, graph_version: str | None = None) -> dict:
    """{qkey: {chunk row: raw cross-encoder logit}} from a scripts/ce_scores.py .jsonl; the .meta.json beside it must
    name the same graph version (rows are graph-version rows) and the .jsonl sha256. Read-only result, shared by
    every env of the process (one Env per query in the vec env)."""
    import hashlib
    p = Path(path)
    if not p.is_absolute():
        from graphrag import config
        p = config.path(str(p))
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns, graph_version)
    if key in _CE_TABLES:
        return _CE_TABLES[key]
    meta = json.loads(ce_meta_path(p).read_text(encoding="utf-8"))
    if graph_version is not None and meta.get("graph_version") != graph_version:
        raise ValueError(f"{p}: ce scores for graph {meta.get('graph_version')}, the env uses {graph_version}")
    if hashlib.sha256(p.read_bytes()).hexdigest() != meta.get("out_sha256"):
        raise ValueError(f"{p}: sha256 differs from its meta.json")
    out: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["qkey"])] = {int(c): float(s) for c, s in zip(r["rows"], r["scores"])}
    _CE_TABLES[key] = out
    return out


@dataclass
class RerankDesc:
    covered: np.ndarray          # int64 node rows S_t
    v0: np.ndarray               # int64 node rows V₀
    cand_node: np.ndarray        # int64 [N] vertex μ(c) of every candidate slot (0 for an empty slot, masked)
    feats: np.ndarray            # float32 [N, N_CAND_FEATURES]
    scalars: np.ndarray          # float32 [N_RERANK_SCALARS]
    mask: np.ndarray             # bool [N + 1] valid actions (0 = STOP)


@dataclass
class _QCands:
    key: tuple
    rows: np.ndarray
    fused: np.ndarray            # float64 fused normalised scores of the rows (the ranking's own values)
    nodes: np.ndarray
    docs: np.ndarray
    static: np.ndarray           # float32 [N, F] (dynamic columns 0)
    nbrs: list                   # frozenset of all neighbours of each candidate vertex
    ce: np.ndarray | None        # raw ce logits aligned with rows, None without a cache


class RerankEnv(Env):
    """The rerank MDP over the M3 Env's data, limits and State (module docstring)."""

    def __init__(self, data, cfg=None, start_cfg=None, n: int = RERANK_N, fuse_w: float = FUSE_W,
                 norm_top_n: int = NORM_TOP_N, use_ce: bool = False, ce_cache: str | None = None):
        super().__init__(data, cfg, start_cfg)
        if int(n) < 1:
            raise ValueError(f"rerank.n must be >= 1, got {n}")
        if int(norm_top_n) < 2:
            raise ValueError(f"dynamic_k.norm_top_n must be >= 2, got {norm_top_n}")
        if use_ce and not ce_cache:
            raise ValueError("rerank.use_ce needs rerank.ce_cache (scripts/ce_scores.py output)")
        self.n, self.fuse_w, self.norm_top_n = int(n), check_fuse_w(fuse_w), int(norm_top_n)
        self.use_ce, self.ce_cache = bool(use_ce), ce_cache
        self.actions = (Action("stop"),) + tuple(Action("select", k=j) for j in range(self.n))
        self._ce = None
        self._doc = None
        tid = data.graph.edge_types
        self._deg_types = tuple(tid.index(t) for t in ("AMENDS", "REFERS_TO") if t in tid)

    # -- query / candidates --
    def query(self, rank):
        return with_scores(super().query(rank), rank)

    def ce_table(self) -> dict | None:
        if self.ce_cache is None:
            return None
        if self._ce is None:
            self._ce = load_ce_cache(self.ce_cache, self.data.version)
        return self._ce

    def doc_of(self) -> np.ndarray:
        """Root of every vertex in the CONTAINS forest (the document; a vertex outside it is its own root)."""
        if self._doc is None:
            cached = getattr(self.data.graph, "_rerank_doc_root", None)   # static graph: shared by every env
            if cached is not None:
                self._doc = cached
                return cached
            parent = np.asarray(self.data.graph.parent, np.int64)
            root = np.arange(parent.size, dtype=np.int64)
            for _ in range(parent.size + 1):                  # a forest has depth < |V|; more = a cycle
                up = parent[root]
                m = up >= 0
                if not m.any():
                    break
                root = np.where(m, up, root)
            else:
                raise ValueError("CONTAINS parent pointers contain a cycle")
            root.setflags(write=False)
            self._doc = root
            try:
                self.data.graph._rerank_doc_root = root
            except AttributeError:
                pass
        return self._doc

    def ce_scores(self, q) -> np.ndarray | None:
        """Raw ce logits of the candidates (None without a cache); a candidate missing from the cache raises."""
        return self._cands(q).ce

    def cand_rows(self, q) -> np.ndarray:
        """The candidate fragment rows of q (slot j = action j + 1)."""
        return self._cands(q).rows

    def _cands(self, q) -> _QCands:
        key = (self.n, self.fuse_w, self.norm_top_n, self.ce_cache, self.use_ce, self.limits.L_max)
        c = getattr(q, "_rerank", None)
        if c is not None and c.key == key:
            return c
        d, L = self.data, self.limits
        rows, fsc = rerank_candidates(q, self.n, self.fuse_w, self.norm_top_n, d.chunk_tokens, L.L_max)
        n = rows.size
        nodes = d.chunk_node[rows].astype(np.int64) if n else EMPTY
        docs = self.doc_of()[nodes] if n else EMPTY
        F = np.zeros((self.n, N_CAND_FEATURES), np.float32)
        col = {k: i for i, k in enumerate(CAND_FEATURES)}
        ce = None
        table = self.ce_table()
        if table is not None and n:
            got = table.get(q.qkey)
            miss = [int(c) for c in rows if got is None or int(c) not in got]
            if miss:
                raise ValueError(f"{q.qkey}: no cross-encoder score for candidate rows {miss[:5]} in "
                                 f"{self.ce_cache} (rescore with ce_n >= rerank.n under the same fusion)")
            ce = np.array([got[int(c)] for c in rows], np.float64)
        if n:
            cos = np.asarray(q.chunk_qsim, np.float64)
            vec = np.asarray(q.vec, np.int64)
            lo_hi = (float(np.min(cos[vec][:self.norm_top_n])), float(np.max(cos[vec][:self.norm_top_n]))) \
                if vec.size else (0.0, 0.0)
            span = lo_hi[1] - lo_hi[0]
            lex = np.asarray(q.lex, np.int64)
            bm = minmax_scores(search_scores(q, "lexical"), self.norm_top_n) if lex.size else np.zeros(0)
            bm_of = dict(zip(lex.tolist(), bm.tolist()))
            floor = min(0.0, float(bm.min())) if bm.size else 0.0
            rk_d = {int(c): i for i, c in enumerate(vec.tolist())}
            rk_b = {int(c): i for i, c in enumerate(lex.tolist())}
            for j, c in enumerate(rows.tolist()):
                F[j, col["dense_cos"]] = cos[c]
                F[j, col["dense_norm"]] = (cos[c] - lo_hi[0]) / span if span > 1e-12 else 0.0
                F[j, col["bm25_norm"]] = bm_of.get(c, floor)
                F[j, col["bm25_in"]] = float(c in bm_of)
                F[j, col["fused_norm"]] = fsc[j]
                F[j, col["inv_rank_dense"]] = 1.0 / (1 + rk_d[c]) if c in rk_d else 0.0
                F[j, col["inv_rank_bm25"]] = 1.0 / (1 + rk_b[c]) if c in rk_b else 0.0
                F[j, col["inv_rank_fused"]] = 1.0 / (1 + j)
                F[j, col["pos"]] = j / self.n
                F[j, col["tokens"]] = np.log1p(float(d.chunk_tokens[c])) / 10.0
            same = np.array([int((docs == x).sum()) - 1 for x in docs], np.float32)
            F[:n, col["same_doc_cands"]] = np.log1p(same)
            F[:n, col["tree_depth"]] = np.asarray(d.graph.tree_depth, np.float32)[nodes] / 10.0
            dist = np.asarray(q.depth, np.int64)[nodes]
            T = L.T_max + 1
            F[:n, col["dist_v0"]] = np.where(dist >= 0, np.minimum(dist, T), T) / T
            for name, t in zip(("deg_amends", "deg_refers_to"), self._deg_types):
                F[:n, col[name]] = [np.log1p(d.graph.neighbors(np.array([v], np.int64), t).size) for v in nodes]
            if ce is not None and self.use_ce:
                cn = minmax_scores(ce, max(2, n)) if n > 1 else np.ones(1)
                F[:n, col["ce_norm"]] = cn
                F[:n, col["ce_sigmoid"]] = 1.0 / (1.0 + np.exp(-ce))
                order = np.lexsort((np.arange(n), -ce))
                rk = np.empty(n, np.int64)
                rk[order] = np.arange(n)
                F[:n, col["inv_rank_ce"]] = 1.0 / (1 + rk)
        nbrs = [frozenset(d.graph.all_neighbors(np.array([v], np.int64)).tolist()) for v in nodes.tolist()]
        c = _QCands(key, rows, fsc, nodes, docs, F, nbrs, ce)
        try:
            q._rerank = c
        except AttributeError:
            pass
        return c

    # -- MDP --
    def reset(self, q) -> State:
        self._own(q)
        return super().reset(q)

    def _fits(self, s, c: int) -> bool:
        L = self.limits
        return len(s.context) < L.K_max and s.length + int(self.data.chunk_tokens[c]) <= L.L_max

    def mask(self, q, s) -> np.ndarray:
        m = np.zeros(self.n + 1, bool)
        if s.done:
            return m
        m[0] = True
        for j, c in enumerate(self._cands(q).rows.tolist()):
            m[j + 1] = c not in s.ctx_set and self._fits(s, c)
        return m

    def _any_valid(self, q, s) -> bool:
        return bool(self.mask(q, s)[1:].any())

    def effect(self, q, s, a: Action):
        if a.op != "select":
            raise ValueError(f"no effect for {a.op}")
        rows = self._cands(q).rows
        j = int(a.k)
        if j >= rows.size:
            return [], []
        c = int(rows[j])
        return ([c] if c not in s.ctx_set and self._fits(s, c) else []), []

    def step(self, q, s, a, check: bool = True) -> State:
        if s.done:
            raise InvalidAction("the episode is over")
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= int(a) < len(self.actions):
            raise InvalidAction(f"action id {a!r} is not in [0, {len(self.actions)})")
        a = int(a)
        if check and not self.mask(q, s)[a]:
            raise InvalidAction(f"action {a} {self.actions[a]} is masked in this state")
        t = s.t + 1
        if a == 0:
            return replace(s, t=t, history=s.history + (Step(t, a, (), ()),), done=True, reason="stop")
        added, _ = self.effect(q, s, self.actions[a])
        if not added:                                # refused even with check=False: a select always adds one
            raise InvalidAction(f"action {a} {self.actions[a]} is masked in this state")
        d = self.data
        c = added[0]
        v = int(d.chunk_node[c])
        covered = s.covered | {v}
        frontier = s.frontier if v in s.covered else (s.frontier | set(d.graph.all_neighbors(
            np.array([v], np.int64)).tolist()))
        s2 = State(s.qkey, t, s.context + (c,), s.ctx_set | {c}, frozenset(covered), frozenset(frontier - covered),
                   s.length + int(d.chunk_tokens[c]), s.history + (Step(t, a, (c,), ()),))
        reason = self._limit_reason(s2)
        if reason is None and not self._any_valid(q, s2):
            reason = "exhausted"
        return replace(s2, done=True, reason=reason) if reason else s2

    # -- observation --
    def describe_state(self, q, s) -> RerankDesc:
        if s.done:
            raise ValueError("describe_state() is for live states: the episode is over")
        c = self._cands(q)
        n, L = c.rows.size, self.limits
        F = c.static.copy()
        if n:
            ctx_nodes = {int(self.data.chunk_node[x]) for x in s.context}
            ctx_docs = {int(self.doc_of()[v]) for v in ctx_nodes}
            F[:n, DYN_COLS[0]] = [float(int(v) in ctx_nodes) for v in c.nodes]
            F[:n, DYN_COLS[1]] = [float(int(x) in ctx_docs) for x in c.docs]
            F[:n, DYN_COLS[2]] = [float(bool(nb & s.covered)) for nb in c.nbrs]
        nodes = np.zeros(self.n, np.int64)
        nodes[:n] = c.nodes
        m = self.mask(q, s)
        sc = np.array([s.t / L.T_max, len(s.context) / L.K_max, s.length / L.L_max, float(m[1:].sum()) / self.n,
                       n / self.n, float(self.use_ce)], np.float32)
        return RerankDesc(np.array(sorted(s.covered), np.int64) if s.covered else EMPTY,
                          np.asarray(q.v0, np.int64), nodes, F, sc, m)


SETTINGS_KEYS = ("use_ce", "n", "fuse_w", "norm_top_n", "L_max", "ce_sha256")


def ce_cache_sha(path) -> str | None:
    """out_sha256 of a ce cache (from its .meta.json, verified against the file), None for no cache."""
    if not path:
        return None
    import hashlib
    p = Path(path)
    if not p.is_absolute():
        from graphrag import config
        p = config.path(str(p))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    meta = json.loads(ce_meta_path(p).read_text(encoding="utf-8"))
    if meta.get("out_sha256") != sha:
        raise ValueError(f"{p}: sha256 differs from its meta.json")
    return sha


def rerank_settings(env: "RerankEnv") -> dict:
    """What a trained rerank policy depends on beyond its weights: the candidate list (n, fuse_w, norm_top_n, L_max)
    and the ce feature (use_ce + the sha256 of the cache it read; None when the feature is off)."""
    out = {"use_ce": bool(env.use_ce), "n": int(env.n), "fuse_w": float(env.fuse_w), "norm_top_n": int(env.norm_top_n),
           "L_max": int(env.limits.L_max), "ce_sha256": ce_cache_sha(env.ce_cache) if env.use_ce else None}
    if hasattr(env, "k_max") and hasattr(env, "static"):          # pilot 6 rank variant (grl.model.rank.RankEnv)
        out.update({"rank_k_max": int(env.k_max), "rank_static": bool(env.static)})
    return out


def check_rerank_settings(recorded: dict | None, current: dict, where: str) -> None:
    """Refuse a policy trained under other rerank settings (ValueError naming every differing key)."""
    if recorded is None:
        raise ValueError(f"{where}: no recorded rerank settings (rerank_settings.json) -- not a rerank run?")
    keys = list(SETTINGS_KEYS) + sorted((set(recorded) | set(current)) - set(SETTINGS_KEYS))   # + rank keys (pilot 6)
    diff = {k: (recorded.get(k), current.get(k)) for k in keys if recorded.get(k) != current.get(k)}
    if diff:
        raise ValueError(f"{where}: rerank settings differ from the run that trained the policy "
                         f"(key: (trained, now)) {diff} -- pass the same --set rerank.use_ce / rerank.ce_cache / "
                         "rerank.n / dynamic_k.* as that run")


def check_rerank_invariants(env: RerankEnv, q, final) -> list[str]:
    """Replay check of a rerank episode: limits, prefix, no duplicates, length, every fragment a candidate, the
    mask never all-False in a live state, determinism, termination."""
    probs, s = [], env.reset(q)
    cands = set(env.cand_rows(q).tolist())
    L = env.limits
    for h in final.history:
        if not env.mask(q, s).any():
            probs.append(f"t={s.t}: all-False mask in a live state")
        prev = s
        s = env.step(q, s, h.action)
        if s.context[:len(prev.context)] != prev.context:
            probs.append(f"t={s.t}: C_t not a prefix of C_t+1")
        if s.t > L.T_max or len(s.context) > L.K_max or s.length > L.L_max:
            probs.append(f"t={s.t}: limit exceeded")
        if len(set(s.context)) != len(s.context):
            probs.append(f"t={s.t}: duplicate fragment")
        if s.length != (int(env.data.chunk_tokens[list(s.context)].sum()) if s.context else 0):
            probs.append(f"t={s.t}: length mismatch")
        if not set(s.context) <= cands:
            probs.append(f"t={s.t}: a fragment outside the candidate list")
    if s != final:
        probs.append("replay differs from the recorded episode")
    if not final.done:
        probs.append("episode not terminated")
    return probs


# ---- policy -------------------------------------------------------------------------------------------------------

def _mlp(d_in: int, hidden, d_out: int, dropout: float) -> nn.Sequential:
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
        d = int(h)
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


HEAD_DEFAULTS = {"cand_dim": 64, "hidden": [256, 128]}


class RerankPolicy(nn.Module):
    """Pointer policy: logit_j = MLP([W_n z_node(j), W_q z_q, ctx, feats_j]) for every candidate slot, STOP logit =
    MLP([W_q z_q, ctx, v0, mean of the valid candidates' W_n z_node, scalars]); masked log-softmax over STOP + slots
    (masked_log_softmax). The GAT encoder (query-injected) is the full variant's one. model cfg key rerank_head:
    {cand_dim, hidden}. Same interface as GraphPolicy (encode, log_probs), so SFT / GRPO / rollouts are reused."""

    def __init__(self, g: GraphTensors, cfg: dict):
        super().__init__()
        h = {**HEAD_DEFAULTS, **(cfg.get("rerank_head") or {})}
        self.encoder = GraphEncoder(g, cfg)
        D, cd, drop = self.encoder.dim, int(h["cand_dim"]), float(cfg.get("dropout", 0.0))
        self.q_proj = nn.Linear(D, cd)
        self.node_proj = nn.Linear(D, cd)
        self.ctx_proj = nn.Sequential(nn.Linear(D, cd), nn.ReLU())
        self.v0_proj = nn.Sequential(nn.Linear(D, cd), nn.ReLU())
        self.cand_head = _mlp(3 * cd + N_CAND_FEATURES, h["hidden"], 1, drop)
        self.stop_head = _mlp(4 * cd + N_RERANK_SCALARS, h["hidden"], 1, drop)

    def encode(self, g: GraphTensors, qvec: torch.Tensor):
        return self.encoder(g, qvec)

    def logits(self, z, qz, qidx, descs) -> tuple[torch.Tensor, torch.Tensor]:
        dev, dt = z.device, z.dtype
        qi = torch.as_tensor(np.asarray(qidx, np.int64), device=dev)
        q = self.q_proj(qz[qi])                                                    # [E, cd]
        ctx = self.ctx_proj(segment_mean(z, qidx, [d.covered for d in descs]))
        v0 = self.v0_proj(segment_mean(z, qidx, [d.v0 for d in descs]))
        nodes = torch.from_numpy(np.stack([d.cand_node for d in descs])).to(dev)   # [E, N]
        zn = self.node_proj(z[qi[:, None], nodes])                                 # [E, N, cd]
        feats = torch.from_numpy(np.stack([d.feats for d in descs])).to(dev, dt)   # [E, N, F]
        mask = torch.from_numpy(np.stack([d.mask for d in descs])).to(dev)         # [E, N + 1]
        N = nodes.shape[1]
        x = torch.cat([zn, q[:, None].expand(-1, N, -1), ctx[:, None].expand(-1, N, -1), feats], -1)
        cand = self.cand_head(x).squeeze(-1)                                       # [E, N]
        valid = mask[:, 1:].to(dt)
        pooled = (zn * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        sc = torch.from_numpy(np.stack([d.scalars for d in descs])).to(dev, dt)
        stop = self.stop_head(torch.cat([q, ctx, v0, pooled, sc], -1))             # [E, 1]
        return torch.cat([stop, cand], 1), mask

    def log_probs(self, z, qz, qidx, descs) -> torch.Tensor:
        lg, mask = self.logits(z, qz, qidx, descs)
        return masked_log_softmax(lg, mask)


def rerank_params(p: RerankPolicy) -> dict[str, int]:
    out = {"encoder": sum(t.numel() for t in p.encoder.parameters())}
    out["head"] = sum(t.numel() for t in p.parameters()) - out["encoder"]
    out["total"] = out["encoder"] + out["head"]
    return out
