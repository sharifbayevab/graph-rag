"""Graph walks of the synthetic dataset (decision M3-R10): 1-3 hops over verified edge types only (M2 audit:
REGULATES / RELATED excluded). A walk starts at the fragment holding the evidence of an anchor edge (REFERS_TO, AMENDS or
REPEALS), follows up to max_hops edges in either direction and ends at a corpus vertex with an eligible fragment; the
fact-bearing fragments are the start, the end, and every intermediate corpus vertex that is an endpoint of a
REFERS_TO / AMENDS / REPEALS hop (a pass through CONTAINS adds no fact). Chunkless vertices (EXT stubs, concepts) are
intermediate only.

Enumeration is deterministic: neighbours in sha256(seed:node) order, capped by max_branch. Ruling M3-R11: for every
anchor edge ALL valid walks (1..max_hops) are enumerated and one is picked seeded-uniformly instead of the first valid
one in (hops, order), so the neighbour order biases nothing; M3-R12 (3) stratifies the pick by hop count (a length
uniformly among the anchor's valid lengths, then a walk of that length) so 3-hop CONTAINS padding does not dominate.
A walk is valid when its end is a corpus vertex with an eligible fragment, its pattern is defined (classify), it bears
2-4 fragments and (with split_of) all its corpus documents lie in one split (a walk never crosses splits, П3)."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

WALK_EDGES = ("CONTAINS", "REFERS_TO", "AMENDS", "REPEALS", "DEFINES")
ANCHOR_EDGES = ("REFERS_TO", "AMENDS", "REPEALS")
FACT_EDGES = ANCHOR_EDGES                                   # hops whose corpus endpoints bear a fact
SKIP_RELATIONS = ("version_of", "translation_of")          # same content in another edition / language
MAX_CHUNKS = 4


@dataclass(frozen=True)
class Walk:
    pattern: str
    nodes: tuple[str, ...]
    hops: tuple[tuple[str, str, str], ...]                 # (src, type, dst) per hop, in walk order
    chunks: tuple[str, ...]                                # fact-bearing fragments, in walk order
    docs: tuple[str, ...]


STUB_PREFIXES = ("EXT", "CON", "ENT")                     # chunkless vertices: external acts, concepts, entities


def classify(hops, doc_of) -> str | None:
    """amends if any AMENDS / REPEALS hop; else, with a REFERS_TO hop, refers_internal when every corpus vertex of the
    walk lies in ONE document and there are at least two of them (M3-R12 (2): two parts of one act linked directly or
    through an external act), refers_cross otherwise; None without REFERS_TO / AMENDS / REPEALS."""
    types = [t for _, t, _ in hops]
    if "AMENDS" in types or "REPEALS" in types:
        return "amends"
    if "REFERS_TO" not in types:
        return None
    corpus = {v for s, _, d in hops for v in (s, d) if doc_of(v) not in STUB_PREFIXES}
    return "refers_internal" if len(corpus) >= 2 and len({doc_of(v) for v in corpus}) == 1 else "refers_cross"


def _h(seed, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _chunk_at(rec, start: int, end: int) -> str | None:
    hold = [c for c in rec.chunks if c.start <= start and end <= c.end]
    if hold:
        return hold[0].chunk_id
    ov = sorted((c for c in rec.chunks if c.start < end and c.end > start), key=lambda c: c.start)
    return ov[0].chunk_id if ov else None


def _skip(e) -> bool:
    return e.type == "REFERS_TO" and (e.attrs or {}).get("relation") in SKIP_RELATIONS


def is_act_level(v: str, corpus_docs) -> bool:
    """A document root of the corpus or an EXT act root ("EXT/<key>"), not an element stub ("EXT/<key>/СТ.9")."""
    if v.startswith("EXT/"):
        return "/" not in v[4:]
    return "/" not in v and v in corpus_docs


def hub_vertices(edges, corpus_docs, max_referrer_docs: int = 6, top_degree_share: float = 0.01) -> tuple[set, dict]:
    """M3-R14 (1), Task 10's hub rule: an act-level vertex is a hub when more than `max_referrer_docs` other documents
    refer to it (REFERS_TO / AMENDS / REPEALS to the act or any of its elements) or its degree over the walk edges is
    in the top `top_degree_share` of act-level vertices (rank by (-degree, id)). Returns (hubs, audit)."""
    import math
    act_of = lambda v: v if v.startswith("EXT/") and "/" not in v[4:] else (
        "EXT/" + v[4:].split("/", 1)[0] if v.startswith("EXT/") else v.split("/", 1)[0])
    refs: dict[str, set] = defaultdict(set)
    deg: Counter = Counter()
    for e in edges:
        if e.type not in WALK_EDGES or _skip(e):
            continue
        deg[e.src] += 1
        deg[e.dst] += 1
        if e.type in ANCHOR_EDGES:
            a, d = act_of(e.dst), e.src.split("/", 1)[0]
            if a != d:
                refs[a].add(d)
    acts = sorted({v for v in deg if is_act_level(v, corpus_docs)} | {a for a in refs if is_act_level(a, corpus_docs)})
    by_ref = {a for a in acts if len(refs.get(a, ())) > max_referrer_docs}
    k = math.ceil(top_degree_share * len(acts))
    by_deg = set(sorted(acts, key=lambda a: (-deg[a], a))[:k])
    hubs = by_ref | by_deg
    audit = {"act_level_vertices": len(acts), "hubs": len(hubs), "by_referrers": len(by_ref), "by_degree": len(by_deg),
             "degree_cut": min((deg[a] for a in by_deg), default=None),
             "top": [[a, len(refs.get(a, ())), deg[a]] for a in sorted(hubs, key=lambda a: (-len(refs.get(a, ())), a))[:15]]}
    return hubs, audit


def enumerate_walks(data, records, edges, ok: set[str], max_hops: int = 3, max_branch: int = 8,
                    seed=0, split_of: dict[str, str] | None = None, stats: dict | None = None,
                    hubs: set[str] | frozenset = frozenset()) -> dict[str, list[Walk]]:
    """pattern -> walks (one per anchor edge at most; identical fragment tuples kept once). `stats`, when given, is
    filled with anchor / candidate counts for the report. M3-R14 (1): a walk with an intermediate vertex in `hubs`
    (hub act-level vertices, hub_vertices) is invalid."""
    recs = {r.doc_id: r for r in records}
    doc_of = lambda n: n.split("/", 1)[0]
    adj: dict[str, list[tuple[str, tuple[str, str, str]]]] = defaultdict(list)
    for e in edges:
        if e.type in WALK_EDGES and not _skip(e):
            hop = (e.src, e.type, e.dst)
            adj[e.src].append((e.dst, hop))
            adj[e.dst].append((e.src, hop))
    for v in adj:
        adj[v] = sorted(set(adj[v]), key=lambda x: (_h(seed, x[0] + "|" + x[1][1]), x))[:max_branch]

    memo: dict[tuple[str, str], str | None] = {}

    def best_chunk(v: str, ref: str) -> str | None:
        key = (v, ref)
        if key not in memo:
            row = data.node_row.get(v)
            out = None
            if row is not None and doc_of(v) in recs:
                cand = sorted(data.chunk_ids[c] for c in data.mu_inv(row)
                              if data.chunk_ids[c] in ok and data.chunk_ids[c] != ref)
                if cand:
                    va = data.vecs64[data.chunk_row[ref]]
                    out = min(cand, key=lambda x: (-round(float(data.vecs64[data.chunk_row[x]] @ va), 6), x))
            memo[key] = out
        return memo[key]

    def one_split(nodes) -> bool:
        if split_of is None:
            return True
        return len({split_of[doc_of(v)] for v in nodes if doc_of(v) in recs}) <= 1

    def realise(nodes, hops, a) -> Walk | None:
        pat = classify(hops, doc_of)
        if pat is None or not one_split(nodes) or any(v in hubs for v in nodes[1:-1]):
            return None
        b = best_chunk(nodes[-1], a)
        if b is None:
            return None
        mids = [best_chunk(v, a) for i, v in enumerate(nodes[1:-1], start=1)
                if hops[i - 1][1] in FACT_EDGES or hops[i][1] in FACT_EDGES]
        ch = tuple(dict.fromkeys([a, *[m for m in mids if m], b]))
        if not 2 <= len(ch) <= MAX_CHUNKS:
            return None
        return Walk(pat, tuple(nodes), tuple(hops), ch, tuple(dict.fromkeys(doc_of(c) for c in ch)))

    out: dict[str, list[Walk]] = defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    st: Counter = Counter()
    for e in sorted(edges, key=lambda e: (e.src, e.type, e.dst)):
        if e.type not in ANCHOR_EDGES or _skip(e):
            continue
        st["anchor_edges"] += 1
        ev = (e.attrs or {}).get("evidence") or {}
        if ev.get("doc_id") not in recs:
            st["anchor_no_evidence_doc"] += 1
            continue
        a = _chunk_at(recs[ev["doc_id"]], ev["start"], ev["end"])
        if a is None or a not in ok:
            st["anchor_chunk_not_eligible"] += 1
            continue
        valid: list[Walk] = []
        frontier = [((e.src, e.dst), ((e.src, e.type, e.dst),))]
        while frontier:
            nxt = []
            for nodes, hops in frontier:
                w = realise(nodes, hops, a)
                if w is not None:
                    valid.append(w)
                if len(hops) < max_hops:
                    for v, hop in adj.get(nodes[-1], ()):
                        if v not in nodes:
                            nxt.append((nodes + (v,), hops + (hop,)))
            frontier = nxt
        if not valid:
            st["anchor_no_valid_walk"] += 1
            continue
        st["anchor_with_valid_walk"] += 1
        st["valid_walks_total"] += len(valid)
        anchor = f"{e.src}|{e.type}|{e.dst}"
        # M3-R12 (3): hop count first, uniformly among the lengths this anchor has valid walks for; then the M3-R11
        # seeded-uniform walk of that length (the next length / walk in the same order only if all are duplicates)
        lens = sorted({len(w.hops) for w in valid}, key=lambda n: (_h(seed, f"{anchor}#len{n}"), n))
        order = sorted(valid, key=lambda w: (lens.index(len(w.hops)),
                                             _h(seed, anchor + "#" + "|".join("~".join(h) for h in w.hops)), w.hops))
        pick = next((w for w in order if w.chunks not in seen), None)
        if pick is None:
            st["anchor_all_duplicate"] += 1
            continue
        seen.add(pick.chunks)
        out[pick.pattern].append(pick)
    if stats is not None:
        stats.update(sorted(st.items()))
    return {p: v for p, v in sorted(out.items())}



# ---- M3-R20 (research M3-R19 R4): walk classes and the quality filter ---------------------------------------------------
import re as _re

_ELEM = _re.compile(r"/(?:СТ|ГЛ|Ч|П|ПП|§)\.")
_ART = _re.compile(r"/СТ\.(\d+)")
WALK_CLASSES = ("bridge", "aggregation")


def _corpus(v: str) -> bool:
    return v.split("/", 1)[0] not in STUB_PREFIXES


def walk_class(hops) -> str:
    """bridge: every REFERS_TO / AMENDS / REPEALS hop links two corpus legal elements directly (article -> article:
    the source fragment cites the provision the target fragment states -- a genuine chain); aggregation: every such
    hop targets a legal element (a corpus or an external article / part / item): two documents about the SAME
    provision (what each act changed, before vs after, which acts ...); act: a hop targets a whole act / document
    (a topic-only link -- research M3-R19: 1 of 20 such walks chainable; dropped)."""
    fh = [h for h in hops if h[1] in FACT_EDGES]
    if not fh:
        return "act"
    if all(_corpus(a) and _corpus(b) and _ELEM.search(a) and _ELEM.search(b) for a, _, b in fh):
        return "bridge"
    if all(_ELEM.search(b) for _, _, b in fh):
        return "aggregation"
    return "act"


def anchored(chunks, hops, text_of) -> bool:
    """Every fact-bearing fragment is about a targeted provision: it lies at the target article (or inside it) or
    mentions the article's number as a whole number. Removes CONTAINS-sibling padding (a neighbouring item of the
    amending act picked by similarity) and fragments of a commentary that do not touch the provision."""
    arts = {(b, m.group(1)) for a, t, b in hops if t in FACT_EDGES for m in [_ART.search(b)] if m}
    if not arts:
        return False
    for c in chunks:
        loc = c.split("#", 1)[0]
        at = any(loc == b or loc.startswith(b + "/") or b.startswith(loc + "/") for b, _ in arts)
        if not (at or any(_re.search(rf"(?<![\d.,]){n}(?!\d)", text_of(c) or "") for _, n in arts)):
            return False
    return True


def element_referrers(edges) -> dict[str, int]:
    """Legal element -> number of distinct documents whose REFERS_TO / AMENDS / REPEALS edges target it."""
    refs: dict[str, set] = defaultdict(set)
    for e in edges:
        if e.type in FACT_EDGES:
            refs[e.dst].add(e.src.split("/", 1)[0])
    return {k: len(v) for k, v in refs.items()}


def quality_walks(walks: dict, text_of, referrers: dict | None = None, max_referrer_docs: int = 6) -> tuple[list, dict]:
    """M3-R20: the one `multi` pattern = bridge + aggregation walks of every legacy pattern (refers_internal /
    refers_cross / amends) that are anchored and whose targeted provisions are not procedural hubs (an element cited
    by > max_referrer_docs documents, e.g. the Senate-law article every committee conclusion cites as its basis).
    Returns ([(chunks, docs, hops)], report)."""
    from collections import Counter as _C
    out, rep = [], _C()
    for p, ws in sorted(walks.items()):
        for w in ws:
            ch, docs, hops = (w.chunks, w.docs, w.hops) if hasattr(w, "chunks") else w
            k = walk_class(hops)
            hub = referrers is not None and any(referrers.get(b, 0) > int(max_referrer_docs)
                                                 for _, t, b in hops if t in FACT_EDGES)
            if hub and k in WALK_CLASSES:
                k += "-hub"
            ok = k in WALK_CLASSES and anchored(ch, hops, text_of)
            rep[f"{p}|{k}|{'kept' if ok else 'dropped'}"] += 1
            if ok:
                out.append((ch, docs, hops))
    return out, dict(sorted(rep.items()))
