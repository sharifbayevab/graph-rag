"""RELATED «тематически связан» latent edges (IMPLEMENTATION.md §3.2 cycle 2, M2 Task 8, Ruling R21).

Chunk vectors (bge-m3, unit length) -> for every eligible chunk its top-`k_ann` cosine neighbours among
eligible chunks of OTHER documents (blockwise dense cosine, 1024 rows per block; the document mask is
applied before ranking) -> a chunk pair with cos > θ is a candidate edge between the vertices μ(c_i),
μ(c_j) (μ = Chunk.parent, graphrag.data.mapping), weight = max cos over the pairs of those vertices ->
each vertex keeps its top-`k_rel` candidates by weight; an edge survives when it is in the top-k of at
least one end. Edges are undirected, stored with src < dst.

Accuracy guards (Ruling R21):
  * documents linked as translation / version of one act (acts.ActRegistry: editions, ru/uz) form one
    document GROUP; chunk pairs inside a group are masked exactly like same-document pairs; documents that
    are near-copies of each other (unlinked editions, script variants, duplicate files, fill-in template
    resolutions: near_duplicate_docs on a first ANN pass) join the group as well;
  * vertex pairs already joined by an explicit edge (REFERS_TO / AMENDS / REPEALS in either direction,
    CONTAINS ancestry) are not latent: dropped and counted;
  * boilerplate chunks (signature / requisites, document headers, adoption headers, gazette citation
    lists, procedural formulas, standard clauses, < `min_content_tokens` content tokens; and template
    chunks repeated near-verbatim in many documents, found on a first ANN pass) take no part in the ANN;
  * θ = max(θ_grid, θ_null): θ_grid = the smallest grid value with latent share <= max_latent_share and
    median RELATED degree <= k_rel; θ_null = the q-quantile (q99.9) of cosine over random cross-group
    pairs of eligible chunks, snapped up to the grid;
  * hubness: max / p99 degree are reported (top-k of one end lets hubs exceed k_rel).

Determinism: vectors are float32; cosine is accumulated in float64 on the CPU (float32 BLAS results depend
on the block shape at ~1e-7, enough to flip a 1e-6 rounding) and rounded to 1e-6 before any ranking or
threshold; ties are broken by chunk id (the row order is the sorted chunk-id order) and by vertex id.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from graphrag import config
from graphrag.data.lemma import STOP
from graphrag.kg.schema import GraphEdge

EXPLICIT_TYPES = ("REFERS_TO", "AMENDS", "REPEALS")
DECIMALS = 6

# config/graph.yaml related_defaults (Ruling R31 (3)): the defaults for keys a caller's cfg does not set
_GRAPH_CFG = config.load("graph")
DEFAULTS: dict[str, Any] = _GRAPH_CFG["related_defaults"]

# --- boilerplate ---------------------------------------------------------------------------------------

_WORD = re.compile(r"[^\W\d_]{2,}")
# chunk-level boilerplate classes (config/graph.yaml related_boilerplate; {APO} = any Uzbek apostrophe)
_BP = _GRAPH_CFG["related_boilerplate"]
_APO = _BP["apo"]


def _bp_rx(key: str) -> re.Pattern:
    return re.compile(_BP[key].replace("{APO}", _APO))


_FORMULA = _bp_rx("formula")     # procedural formulas of stenograms / resolutions (only for short chunks)
_CLAUSE = _bp_rx("clause")       # standard clauses: approval, "for information", execution control, entry into force
_ADOPTION = _bp_rx("adoption")
_GAZ_WORD = _bp_rx("gazette_word")
_NUMBER = re.compile(r"\d+")


_LABEL = re.compile(r"(?m)^[ \t]*([^\W\d_][^\n:.!?;,\d]{1,80}?)[ \t]*:[ \t]*")


def content_text(text_norm: str, speakers: set[str] | frozenset = frozenset()) -> str:
    """The chunk text for the RELATED score (Ruling R22 v2): text_norm WITHOUT the breadcrumb that
    index_text prepends (document / agenda item / speaker path) and without stenogram speaker labels
    ("<speaker>: " at a turn start, speaker in `speakers`: lowercased speech-node titles + chair)."""
    if not speakers:
        return text_norm.strip()
    return _LABEL.sub(lambda m: "" if m.group(1).strip().lower() in speakers else m.group(0), text_norm).strip()


def content_tokens(text: str) -> list[str]:
    """Alphabetic words of >= 2 letters, lowercased, without stop words (numbers never count)."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def boilerplate_reason(text: str, parent_type: str | None = None, cfg: dict | None = None) -> str | None:
    """Why a chunk is excluded from the ANN (Ruling R21 (2)), or None: "signature" / "header" by its parent
    node type (реквизиты / заголовок), "gazette" (a list of gazette issue / article numbers), "adoption"
    (an act title + "… томонидан … қабул қилинган" header), "formula" (a short procedural formula of a
    speech: chair turns, thanks, voting), "clause" (a short standard clause: approval "… маъқуллансин",
    execution control "… зиммасига юклансин", entry into force "… кучга киради"), "short" (< min_content_tokens content tokens)."""
    c = {**DEFAULTS, **(cfg or {})}
    by_type = c["boilerplate_parent_types"]
    if parent_type in by_type:
        return by_type[parent_type]
    toks = content_tokens(text)
    n = len(toks)
    if text.count("№") >= c["gazette_min_numbers"] and len(_NUMBER.findall(text)) >= 4 \
            and sum(1 for w in toks if not _GAZ_WORD.match(w)) < c["min_content_tokens"]:
        return "gazette"
    if n < c["adoption_max_tokens"] and _ADOPTION.search(text) and parent_type == "преамбула":
        return "adoption"
    if n < c["formula_max_tokens"] and _FORMULA.search(text):
        return "formula"
    if n < c["clause_max_tokens"] and _CLAUSE.search(text):
        return "clause"
    if n < c["min_content_tokens"]:
        return "short"
    return None


# --- R22: sentence-level boilerplate classes, topic text, anisotropy correction -----------------------

_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
_QUOTED = re.compile(r"“[^”]{3,400}”|«[^»]{3,400}»|\"[^\"]{3,400}\"")


@dataclass(frozen=True)
class SentenceRule:
    name: str
    rx: re.Pattern
    sections: frozenset


def compile_sentence_rules(spec: dict | None) -> list[SentenceRule]:
    """graph.yaml `related.sentence_rules` {class: {patterns: [regex], sections: [section] (optional)}} ->
    rules in config order (the first matching class names a sentence). Patterns are matched against a
    sentence of text_norm (lowercase; Uzbek in Latin script, Russian Cyrillic)."""
    out = []
    for name, v in (spec or {}).items():
        pats = v["patterns"] if isinstance(v, dict) else v
        secs = frozenset((v.get("sections") or []) if isinstance(v, dict) else [])
        out.append(SentenceRule(name, re.compile("|".join(f"(?:{p})" for p in pats)), secs))
    return out


def strip_boilerplate(text_norm: str, rules: list[SentenceRule], section: str | None = None
                      ) -> tuple[str, str, Counter]:
    """Split into sentences (after . ! ? ; and at line breaks) and drop those matching a boilerplate class
    (a class restricted to `sections` applies only there). Returns (residual + the quoted titles of the
    dropped sentences, residual alone, Counter of matched classes): a quoted act title inside a procedural
    sentence ("навбатдаги масала «…» тўғрисида") is topical, the formula around it is not."""
    keep, titles, cls = [], [], Counter()
    for s in _SENT.split(text_norm):
        s = s.strip()
        if not s:
            continue
        hit = next((r.name for r in rules if (not r.sections or section in r.sections) and r.rx.search(s)), None)
        if hit is None:
            keep.append(s)
        else:
            cls[hit] += 1
            titles += [m.group(0) for m in _QUOTED.finditer(s)]
    return " ".join(keep + titles), " ".join(keep), cls


def residual_tokens(text: str, names: set[str] | frozenset = frozenset()) -> int:
    """Content tokens (content_tokens) that are neither gazette words (modda / yil / son / axborotnoma …)
    nor person-name words (`names`: words of speaker names)."""
    return sum(1 for w in content_tokens(text) if not _GAZ_WORD.match(w) and w not in names)


def sentence_boilerplate_reason(text_norm: str, section: str | None, rules: list[SentenceRule],
                                min_tokens: int, names: set[str] | frozenset = frozenset()) -> str | None:
    """Ruling R22 (v4): the chunk is boilerplate when at least one sentence matches a boilerplate class and
    fewer than `min_tokens` residual content tokens remain without the matched sentences (quoted titles of
    matched sentences do not count). The reason is the most frequent matched class (ties: config order)."""
    if not rules:
        return None
    _, core, cls = strip_boilerplate(text_norm, rules, section)
    if not cls or residual_tokens(core, names) >= min_tokens:
        return None
    order = {r.name: k for k, r in enumerate(rules)}
    return min(cls, key=lambda c: (-cls[c], order[c]))


def topic_text(index_text: str, text_norm: str, speakers: set[str] | frozenset = frozenset(),
               drop_crumb: Iterable[str] = ()) -> str:
    """Ruling R22 (v2): the text embedded for RELATED. index_text = breadcrumb + "\\n" + text_norm; the
    breadcrumb parts (" › ") that only identify the session (`drop_crumb` regexes, e.g. "senatning N-yalpi
    majlisi (date)") or the speaker(s) (every comma-separated name in `speakers`) are dropped -- they made
    speeches of one senator / one session similar whatever the subject -- and the topical parts (act /
    agenda-item title, article path) are kept; the body is content_text (speaker labels removed)."""
    body = content_text(text_norm, speakers)
    crumb = ""
    if index_text and index_text != text_norm and index_text.endswith(text_norm):
        crumb = index_text[: len(index_text) - len(text_norm)].rstrip("\n")
    rxs = [re.compile(p) for p in drop_crumb]
    parts = [p.strip() for p in crumb.split(" › ")] if crumb else []
    keep = [p for p in parts if p and not any(r.search(p) for r in rxs)
            and not all(x.strip() in speakers for x in p.split(","))]
    head = " › ".join(keep)
    return f"{head}\n{body}" if head else body


def anisotropy_correct(X: np.ndarray, eligible: np.ndarray, pcs: int) -> tuple[np.ndarray, dict]:
    """Ruling R22 (v1), "all-but-the-top" (Mu & Viswanath 2018): subtract the mean of the eligible rows,
    remove the projection on their top-`pcs` principal directions, renormalize. Float64 throughout
    (eigh of the d x d scatter matrix; eigenvector signs do not matter for a projection), float32 out."""
    Y = np.asarray(X, dtype=np.float64)
    E = Y[np.asarray(eligible, dtype=bool)]
    mu = E.mean(axis=0)
    Y = Y - mu
    ev = []
    if pcs > 0:
        Ec = E - mu
        w, V = np.linalg.eigh(Ec.T @ Ec)
        P = V[:, np.argsort(-w, kind="stable")[:pcs]]
        Y = Y - (Y @ P) @ P.T
        tot = float(w.sum())
        ev = [round(float(x) / tot, 6) for x in sorted(w, reverse=True)[:pcs]]
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return (Y / nrm).astype(np.float32), {"pcs": pcs, "explained_variance_ratio": ev,
                                          "mean_norm": round(float(np.linalg.norm(mu)), 6)}


# --- groups / explicit pairs ---------------------------------------------------------------------------

def doc_groups(doc_ids: Iterable[str], linked: Iterable[tuple[str, str]]) -> dict[str, str]:
    """doc_id -> group representative (the smallest doc_id of its connected component) over `linked`
    document pairs (translation / version of one act). Unlinked documents are their own group."""
    parent: dict[str, str] = {d: d for d in doc_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in linked:
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo
    return {d: find(d) for d in sorted(parent)}


def pair(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u < v else (v, u)


def explicit_pairs(edges: Iterable[GraphEdge], types: tuple = EXPLICIT_TYPES) -> set[tuple[str, str]]:
    """Undirected vertex pairs joined by an explicit edge of `types`."""
    return {pair(e.src, e.dst) for e in edges if e.type in types and e.src != e.dst}


def is_ancestor_pair(u: str, v: str, parent_of: dict[str, str | None]) -> bool:
    """CONTAINS ancestry: one vertex is an ancestor of the other in the structure tree."""
    for a, b in ((u, v), (v, u)):
        x = parent_of.get(b)
        seen = 0
        while x is not None and seen < 64:
            if x == a:
                return True
            x = parent_of.get(x)
            seen += 1
    return False


# --- ANN ----------------------------------------------------------------------------------------------

def _round(a: np.ndarray) -> np.ndarray:
    return np.round(a.astype(np.float64), DECIMALS)


@dataclass
class Ann:
    """Top-k cross-group neighbours of every chunk row (-1 / -inf where fewer exist)."""
    nbr: np.ndarray                    # int64 [n, k]
    score: np.ndarray                  # float64 [n, k], rounded to 1e-6, descending, ties by column index
    group_hist: np.ndarray             # chunk pairs (i < j, both eligible) of linked documents (translation / version)
    dup_hist: np.ndarray               # ... of documents grouped only as near-duplicates
    same_doc_hist: np.ndarray          # chunk pairs (i < j, both eligible) of one document, by bin
    bins: np.ndarray                   # histogram bin edges (grid values and 1.0 + eps)


def ann_topk(X: np.ndarray, doc: np.ndarray, group: np.ndarray, eligible: np.ndarray, k: int,
             block: int = 1024, bins: np.ndarray | None = None, device: str = "cpu",
             link_group: np.ndarray | None = None) -> Ann:
    """Blockwise dense cosine (X rows unit length, float32). Columns of the row's own group (same document
    or a translation / version of it) and ineligible columns are masked BEFORE ranking. Scores are rounded
    to 1e-6; order = (-score, column index), i.e. chunk-id order when rows are sorted by chunk id."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    X64 = X.astype(np.float64)                             # float64 accumulation: block-shape independent at 1e-6
    n = len(X)
    bins = np.asarray(bins if bins is not None else [0.0, 1.0 + 1e-9], dtype=np.float64)
    nbr = np.full((n, k), -1, dtype=np.int64)
    score = np.full((n, k), -np.inf, dtype=np.float64)
    gh = np.zeros(len(bins) - 1, dtype=np.int64)
    dh = np.zeros(len(bins) - 1, dtype=np.int64)
    link_group = group if link_group is None else link_group
    sh = np.zeros(len(bins) - 1, dtype=np.int64)
    kk = min(k + 16, n)                                   # margin so rounding ties at the cut are ranked exactly
    col = np.arange(n)
    Xt = None
    if device != "cpu":
        import torch
        Xt = torch.from_numpy(X).to(device)
    for a in range(0, n, block):
        b = min(a + block, n)
        if Xt is not None:                                # float32 on the GPU: fast, not bit-stable across blocks
            S = (Xt[a:b] @ Xt.T).cpu().numpy()
        else:
            S = X64[a:b] @ X64.T
        S = _round(S)
        rows = np.arange(a, b)
        same_group = group[rows, None] == group[None, :]
        both = eligible[rows, None] & eligible[None, :]
        upper = col[None, :] > rows[:, None]
        same_doc = doc[rows, None] == doc[None, :]
        linked = link_group[rows, None] == link_group[None, :]
        gh += np.histogram(S[linked & ~same_doc & both & upper], bins=bins)[0]
        dh += np.histogram(S[same_group & ~linked & both & upper], bins=bins)[0]
        sh += np.histogram(S[same_doc & both & upper], bins=bins)[0]
        S[same_group | ~both] = -np.inf
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk] if kk < n else np.tile(col, (b - a, 1))
        for r in range(b - a):
            i = a + r
            if not eligible[i]:
                continue
            cand = part[r]
            sc = S[r, cand]
            order = np.lexsort((cand, -sc))
            cand, sc = cand[order], sc[order]
            ok = np.isfinite(sc)
            cand, sc = cand[ok][:k], sc[ok][:k]
            nbr[i, :len(cand)] = cand
            score[i, :len(sc)] = sc
    return Ann(nbr, score, gh, dh, sh, bins)


def null_quantile(X: np.ndarray, group: np.ndarray, eligible: np.ndarray, n_pairs: int, q: float,
                  seed: int) -> dict:
    """Cosine of `n_pairs` random pairs of eligible chunks from different groups (the null distribution
    of unrelated cross-document pairs); its q-quantile is the θ floor."""
    idx = np.flatnonzero(eligible)
    rng = np.random.default_rng(seed)
    got_i, got_j = [], []
    need = n_pairs
    while need > 0:
        i = rng.choice(idx, size=need * 2)
        j = rng.choice(idx, size=need * 2)
        ok = group[i] != group[j]
        i, j = i[ok][:need], j[ok][:need]
        got_i.append(i)
        got_j.append(j)
        need -= len(i)
    i, j = np.concatenate(got_i), np.concatenate(got_j)
    s = _round(np.einsum("ij,ij->i", X[i].astype(np.float64), X[j].astype(np.float64)))
    return {"pairs": int(len(s)), "quantile": q, "value": float(np.quantile(s, q, method="higher")),
            "mean": round(float(s.mean()), DECIMALS), "p50": float(np.quantile(s, 0.5, method="higher")),
            "p99": float(np.quantile(s, 0.99, method="higher")), "max": float(s.max()), "seed": seed}


# --- edges --------------------------------------------------------------------------------------------

@dataclass
class Candidates:
    best: dict[tuple[str, str], tuple[float, int, int]]      # vertex pair -> (weight, chunk i, chunk j)
    excluded: Counter = field(default_factory=Counter)       # reason -> vertex pairs
    excluded_chunk_pairs: Counter = field(default_factory=Counter)


def candidates(ann: Ann, vertex: list[str], theta: float, explicit: set[tuple[str, str]],
               parent_of: dict[str, str | None] | None = None) -> Candidates:
    """Chunk pairs of the ANN lists with cos > θ -> vertex pairs, weight = max cos (ties: smallest chunk
    pair). Pairs joined by an explicit edge / CONTAINS ancestry are dropped and counted."""
    rows, cols = np.nonzero(ann.score > theta)
    chunk_w: dict[tuple[int, int], float] = {}           # a pair may be in both rows' lists: max score
    for i, c in zip(rows.tolist(), cols.tolist()):
        j = int(ann.nbr[i, c])
        cp = (i, j) if i < j else (j, i)
        w = float(ann.score[i, c])
        if w > chunk_w.get(cp, -np.inf):
            chunk_w[cp] = w
    best: dict[tuple[str, str], tuple[float, int, int]] = {}
    drop: dict[str, set] = defaultdict(set)
    drop_c: Counter = Counter()
    for (ci, cj), w in sorted(chunk_w.items()):
        u, v = vertex[ci], vertex[cj]
        if u == v:
            drop_c["same_vertex"] += 1
            drop["same_vertex"].add((u, v))
            continue
        p = pair(u, v)
        if p in explicit:
            drop_c["explicit_edge"] += 1
            drop["explicit_edge"].add(p)
            continue
        if parent_of is not None and is_ancestor_pair(u, v, parent_of):
            drop_c["contains_ancestry"] += 1
            drop["contains_ancestry"].add(p)
            continue
        cur = best.get(p)
        if cur is None or w > cur[0]:                     # pairs visited in (ci, cj) order: ties keep the smallest
            best[p] = (w, ci, cj)
    return Candidates(best, Counter({k: len(v) for k, v in drop.items()}), drop_c)


def cap_degree(best: dict[tuple[str, str], tuple], k_rel: int, mutual: bool = False) -> dict[tuple[str, str], tuple]:
    """Each vertex keeps its top-k_rel incident candidates by (weight desc, other vertex id asc); an
    edge survives when it is in the top-k of at least one end (mutual=False) or of both ends (mutual=True,
    Ruling R22 v5: mutual kNN, degree <= k_rel, no hubs)."""
    inc: dict[str, list] = defaultdict(list)
    for (u, v), t in best.items():
        inc[u].append((-t[0], v, (u, v)))
        inc[v].append((-t[0], u, (u, v)))
    top = {x: {p for _, _, p in sorted(inc[x])[:k_rel]} for x in inc}
    if mutual:
        keep = {p for p in best if p in top[p[0]] and p in top[p[1]]}
    else:
        keep = set().union(*top.values()) if top else set()
    return {p: best[p] for p in sorted(keep)}


def to_edges(kept: dict[tuple[str, str], tuple]) -> list[GraphEdge]:
    return [GraphEdge(u, v, "RELATED", {"weight": min(1.0, float(t[0]))}) for (u, v), t in sorted(kept.items())]


def degrees(kept: Iterable[tuple[str, str]]) -> Counter:
    d: Counter = Counter()
    for u, v in kept:
        d[u] += 1
        d[v] += 1
    return d


def degree_stats(deg: Counter, n_vertices: int | None = None) -> dict:
    vals = np.array(sorted(deg.values()), dtype=np.int64)
    if not len(vals):
        return {"vertices": 0, "median": 0.0, "mean": 0.0, "p90": 0, "p99": 0, "max": 0, "hist": {}}
    out = {"vertices": int(len(vals)), "median": float(np.median(vals)), "mean": round(float(vals.mean()), 4),
           "p90": int(np.quantile(vals, 0.90, method="higher")), "p99": int(np.quantile(vals, 0.99, method="higher")),
           "max": int(vals.max()), "hist": {str(k): int(v) for k, v in sorted(Counter(vals.tolist()).items())}}
    if n_vertices is not None:
        out["median_over_all_chunk_vertices"] = float(np.median(np.concatenate(
            [vals, np.zeros(max(0, n_vertices - len(vals)), dtype=np.int64)])))
    return out


@dataclass
class RelatedIndex:
    """ANN computed once; `edges(θ)` for any θ (calibration scans the grid on the same ANN lists)."""
    chunk_ids: list[str]
    vertex: list[str]                 # μ(c) per row
    doc: list[str]
    ann: Ann
    explicit: set
    parent_of: dict | None
    k_rel: int
    eligible: np.ndarray
    group: np.ndarray
    near_duplicates: list = field(default_factory=list)
    templates: list = field(default_factory=list)       # chunk ids made ineligible by template_rows
    mutual: bool = False                                # cap_degree(mutual=...)

    def result(self, theta: float) -> tuple[dict, Candidates]:
        c = candidates(self.ann, self.vertex, theta, self.explicit, self.parent_of)
        return cap_degree(c.best, self.k_rel, self.mutual), c

    def edges(self, theta: float) -> list[GraphEdge]:
        return to_edges(self.result(theta)[0])


def near_duplicate_docs(ann: Ann, doc: np.ndarray, docs: list[str], eligible: np.ndarray, sim: float = 0.97,
                        max_ratio: float = 0.6, min_ratio: float = 0.45) -> list[dict]:
    """Document pairs that are (near-)copies or fill-in templates of each other: in one document at least
    `max_ratio`, in the other at least `min_ratio` of the eligible chunks have an ANN neighbour with
    cos >= sim in the other document (unlinked editions / script variants of one act, duplicate files,
    template resolutions approving different generic-titled laws). Their latent edges carry no topic."""
    n_el = Counter(doc[eligible].tolist())
    hit: dict[tuple[int, int], set] = defaultdict(set)
    rows, cols = np.nonzero(ann.score >= sim - 1e-12)
    for i, c in zip(rows.tolist(), cols.tolist()):
        hit[(int(doc[i]), int(doc[ann.nbr[i, c]]))].add(i)
    out = []
    for (a, b) in sorted({tuple(sorted(k)) for k in hit}):
        ra = len(hit.get((a, b), ())) / n_el[a] if n_el[a] else 0.0
        rb = len(hit.get((b, a), ())) / n_el[b] if n_el[b] else 0.0
        if max(ra, rb) >= max_ratio and min(ra, rb) >= min_ratio:
            out.append({"docs": [docs[a], docs[b]], "ratios": [round(ra, 4), round(rb, 4)],
                        "eligible_chunks": [n_el[a], n_el[b]]})
    return out


def template_rows(ann: Ann, doc: np.ndarray, sim: float = 0.90, min_docs: int = 5) -> np.ndarray:
    """Rows whose ANN list holds neighbours with cos >= sim from >= min_docs distinct other documents: text
    repeated near-verbatim across many documents (session opening / closing speeches, "Вазирлар Маҳкамаси
    … ушбу Қонунга мувофиқлаштирсин" final articles, amending-law heads) -- boilerplate for the ANN."""
    out = []
    for i in range(len(ann.nbr)):
        ok = ann.score[i] >= sim - 1e-12
        if ok.sum() >= min_docs and len(set(doc[ann.nbr[i][ok]].tolist())) >= min_docs:
            out.append(i)
    return np.array(out, dtype=np.int64)


def build_index(chunk_ids: list[str], X: np.ndarray, vertex: list[str], doc: list[str],
                group_of: dict[str, str] | None = None, eligible: np.ndarray | None = None,
                explicit: set | None = None, parent_of: dict | None = None, k_ann: int = 10, k_rel: int = 5,
                block: int = 1024, bins=None, device: str = "cpu", dedup: dict | None = None,
                template: dict | None = None, score_X: np.ndarray | None = None,
                anisotropy_pcs: int | None = None, mutual: bool = False) -> RelatedIndex:
    """Rows are sorted by chunk id first (deterministic order and id tie-break). With `dedup`
    ({sim, max_ratio, min_ratio}) and / or `template` ({sim, min_docs}) a first ANN pass on X finds
    near-duplicate documents (near_duplicate_docs: they join one group) and template chunks (template_rows:
    they become ineligible). The final ANN scores `score_X` (R22: topic-text vectors; default X), after
    anisotropy_correct over the final eligible rows when `anisotropy_pcs` is not None."""
    order = sorted(range(len(chunk_ids)), key=lambda i: chunk_ids[i])
    ids = [chunk_ids[i] for i in order]
    X = np.asarray(X, dtype=np.float32)[order]
    vertex = [vertex[i] for i in order]
    doc = [doc[i] for i in order]
    group_of = dict(group_of or {})
    elig = np.ones(len(ids), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)[order]
    docs = sorted(set(doc))
    dcode = {d: i for i, d in enumerate(docs)}
    doc_arr = np.array([dcode[d] for d in doc], dtype=np.int64)

    def groups_arr(gof):
        gcode = {g: i for i, g in enumerate(sorted({gof.get(d, d) for d in docs}))}
        return np.array([gcode[gof.get(d, d)] for d in doc], dtype=np.int64)

    grp_arr = groups_arr(group_of)
    ann = ann_topk(X, doc_arr, grp_arr, elig, k_ann, block, bins, device)
    dups: list[dict] = []
    tmpl = np.zeros(0, dtype=np.int64)
    link_arr = grp_arr
    if dedup:
        dups = near_duplicate_docs(ann, doc_arr, docs, elig, **dedup)
        if dups:
            group_of = doc_groups(docs, [(d, group_of.get(d, d)) for d in docs] + [tuple(x["docs"]) for x in dups])
            grp_arr = groups_arr(group_of)
    if template:
        tmpl = template_rows(ann, doc_arr, **template)
        elig = elig.copy()
        elig[tmpl] = False
    Z = X if score_X is None else np.asarray(score_X, dtype=np.float32)[order]
    aniso = None
    if anisotropy_pcs is not None:
        Z, aniso = anisotropy_correct(Z, elig, int(anisotropy_pcs))
    if dups or len(tmpl) or Z is not X:
        ann = ann_topk(Z, doc_arr, grp_arr, elig, k_ann, block, bins, device, link_group=link_arr)
    idx = RelatedIndex(ids, vertex, doc, ann, explicit or set(), parent_of, k_rel, elig, grp_arr, mutual=mutual)
    idx.near_duplicates = dups
    idx.templates = [ids[i] for i in tmpl.tolist()]
    idx.score_X = Z
    idx.anisotropy = aniso
    return idx


def speaker_names(records, cfg: dict | None = None) -> set[str]:
    """Lowercased speaker names (titles of speech nodes, as written and in internal Latin script) plus the
    chair label(s): the stenogram labels content_text removes and the name words residual_tokens ignores."""
    from graphrag.data.normalize import to_internal
    c = {**DEFAULTS, **(cfg or {})}
    out = {x.lower() for x in c["chair_labels"]}
    for r in records:
        for n in r.structure:
            if n.type == c["speech_node_type"] and n.title:
                out.add(n.title.lower().strip())
                out.add(to_internal(n.title, "uz").lower().strip())
    return out


def chunk_reason(chunk, parent_type: str | None, section: str | None, cfg: dict,
                 rules: list[SentenceRule], speakers: set[str], names: set[str]) -> str | None:
    """boilerplate_reason (R21 rules on the chunk text), then sentence_boilerplate_reason (R22 classes on
    content_text of text_norm; speaker labels removed in speech sections)."""
    why = boilerplate_reason(chunk.text, parent_type, cfg)
    if why is None and rules:
        sp = speakers if section in cfg["speech_sections"] else frozenset()
        why = sentence_boilerplate_reason(content_text(chunk.text_norm, sp), section, rules,
                                          int(cfg["residual_min_tokens"]), names)
    return why


def latent_edges(records, vectors: dict[str, np.ndarray], theta: float, k_ann: int, k_rel: int, *,
                 explicit_edges: Iterable[GraphEdge] = (), linked_docs: Iterable[tuple[str, str]] = (),
                 cfg: dict | None = None, block: int = 1024,
                 score_vectors: dict[str, np.ndarray] | None = None) -> list[GraphEdge]:
    """RELATED edges of `records` (graphrag.types.DocRecord) from chunk vectors {chunk_id: unit vector}:
    cross-group ANN (k_ann), score > θ, vertex = μ(c) = chunk.parent, weight = max score, explicit pairs and
    boilerplate chunks excluded, degree cap k_rel (top-k of either end; of both with cfg mutual_knn),
    src < dst. R22: cfg sentence_rules (boilerplate classes), anisotropy_pcs (all-but-the-top on the score
    vectors); `score_vectors` (topic-text vectors) are scored instead of `vectors` when given (the
    near-duplicate / template pass always uses `vectors`). Ruling R25: cfg score_vectors "topic" without
    `score_vectors` raises (no silent fallback to the index_text vectors)."""
    c = {**DEFAULTS, **(cfg or {})}
    if c.get("score_vectors") == "topic" and score_vectors is None:
        raise ValueError("related.score_vectors is 'topic' but no topic vectors were passed (score_vectors=None)")
    rules = compile_sentence_rules(c.get("sentence_rules"))
    speakers = speaker_names(records, c) if rules else set()
    names = {w for s in speakers for w in s.split()}
    ids, X, Z, vertex, doc, elig, parent_of = [], [], [], [], [], [], {}
    for r in records:
        ptype = {n.node_id: n.type for n in r.structure}
        for n in r.structure:
            parent_of[n.node_id] = n.parent_id
        for ch in r.chunks:
            ids.append(ch.chunk_id)
            X.append(vectors[ch.chunk_id])
            if score_vectors is not None:
                Z.append(score_vectors[ch.chunk_id])
            vertex.append(ch.parent)
            doc.append(r.doc_id)
            elig.append(chunk_reason(ch, ptype.get(ch.parent), r.meta.section, c, rules, speakers, names) is None)
    if not ids:
        return []
    groups = doc_groups(sorted({r.doc_id for r in records}), linked_docs)
    idx = build_index(ids, np.stack(X), vertex, doc, groups, np.array(elig), explicit_pairs(explicit_edges),
                      parent_of, k_ann, k_rel, block, dedup=c.get("dedup"), template=c.get("template"),
                      score_X=np.stack(Z) if Z else None, anisotropy_pcs=c.get("anisotropy_pcs"),
                      mutual=bool(c.get("mutual_knn")))
    return idx.edges(theta)


# --- calibration --------------------------------------------------------------------------------------

def theta_grid(spec: dict) -> list[float]:
    n = int(round((spec["stop"] - spec["start"]) / spec["step"]))
    return [round(spec["start"] + i * spec["step"], 6) for i in range(n + 1)]


def grid_bins(grid: list[float]) -> np.ndarray:
    """Histogram edges such that bins[k:] counts scores > grid[k] (scores are rounded to 1e-6)."""
    return np.array([g + 5e-7 for g in grid] + [1.0 + 1e-5], dtype=np.float64)


def count_above(hist: np.ndarray, grid: list[float], theta: float) -> int:
    """Pairs with score > θ from a grid_bins histogram (θ on the grid)."""
    k = min(range(len(grid)), key=lambda i: abs(grid[i] - theta))
    return int(hist[k:].sum())


def snap_up(x: float, grid: list[float]) -> float:
    """The smallest grid value >= x (x itself beyond the grid)."""
    for g in grid:
        if g >= x - 1e-12:
            return g
    return x


def calibrate_theta(evaluate: Callable[[float], tuple[int, float]], n_explicit: int, grid: list[float],
                    max_share: float, k_rel: int, null_floor: float | None = None) -> dict:
    """θ_grid = the smallest grid θ with latent share L / (n_explicit + L) <= max_share and median RELATED
    degree <= k_rel, where evaluate(θ) -> (L, median degree); θ = max(θ_grid, null floor snapped up to the
    grid). Returns {theta, theta_grid, theta_null, table}."""
    table, th_grid = [], None
    for t in grid:
        L, med = evaluate(t)
        share = L / (n_explicit + L) if (n_explicit + L) else 0.0
        ok = share <= max_share and med <= k_rel
        table.append({"theta": t, "edges": L, "share": round(share, 6), "median_degree": med, "ok": ok})
        if ok and th_grid is None:
            th_grid = t
    th_null = snap_up(null_floor, grid) if null_floor is not None else None
    cands = [x for x in (th_grid, th_null) if x is not None]
    theta = max(cands) if cands else None
    return {"theta": theta, "theta_grid": th_grid, "theta_null": th_null, "null_floor_raw": null_floor,
            "binding": None if theta is None else ("grid" if theta == th_grid and (th_null is None or th_grid >= th_null)
                                                   else "null"),
            "table": table}
