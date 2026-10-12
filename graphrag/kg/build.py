"""Knowledge-graph build (IMPLEMENTATION.md §3.2 "завершение", §3.4; M2 Task 9, Rulings R1 / R11 / R24).

    pipeline.extract_all(records, cfg)  -- Tasks 3-6 extractors, LLM answers from cache/kg only (a miss raises
                                           graphrag.kg.llm.KgCacheMiss); the one path (Ruling R31 (2))
    assemble(records, cfg, related, extracted) -> Graph   -- vertices + edges, dedup, R11, entity dedup
    embed_nodes(graph, encoder)          -- X (graphrag.kg.features), BEFORE the isolated-vertex step (R1)
    attach_isolated(graph, X)            -- RELATED {weight: cos, attached: true} to the nearest vertex by X
    validate(graph) -> list[str]         -- §3.4 checks; empty = passed
    build_graph(records, cfg, ...)       -- all of the above

Vertices: the structure of every record (документ -> kind document, the rest -> element), the act registry's
corpus documents and external acts (EXT/<key>), EXT element stubs (Tasks 4 / 5), entities (ENT/) and concepts
(CON/). Edges: CONTAINS from parent_id + stub chains, REFERS_TO (incl. version_of / translation_of),
AMENDS / REPEALS, REGULATES / DEFINES, RELATED (Task 8, data/graph/related_edges.jsonl at the adopted θ).

Dedup: one edge per artifact.edge_key -- (src, type, dst), and the relation for REFERS_TO (R24 (2)); the
evidence of merged edges is united in attrs.evidence_all (by position) and attrs.evidence is the first.
R11: a REFERS_TO {relation: amended} with the (src, dst) of an AMENDS edge is dropped (AMENDS supersedes);
cited_in_amendment citations are kept (R24 (2)).
Ruling R26: external acts whose subtree has no edge but its own stub CONTAINS are pruned before X (registry
leftovers, prune_orphan_ext); REPEALS attrs.evidence = the first by position; §3.4 evidence must lie in the
source vertex's own document; edge attrs must be declared for their type in graph.yaml."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graphrag import config
from graphrag.kg.artifact import edge_key
from graphrag.kg.features import RecordIndex
from graphrag.kg.pipeline import Extracted, extract_all, r11_filter  # noqa: F401 -- the one extraction path (R31 (2))
from graphrag.kg.schema import GraphEdge, GraphNode, Schema

EVIDENCE_TYPES = ("REFERS_TO", "AMENDS", "REPEALS", "REGULATES", "DEFINES")
ATTACHABLE = ("document", "element")         # RELATED endpoints (graph.yaml)
DECIMALS = 6


@dataclass
class Graph:
    nodes: list[GraphNode]                   # sorted by node_id (= PyG row order)
    edges: list[GraphEdge]                   # canonical artifact order
    stats: dict = field(default_factory=dict)
    rindex: RecordIndex | None = None
    schema: Schema | None = None
    cfg: dict = field(default_factory=dict)
    x: np.ndarray | None = None
    texts: list[str] | None = None
    feature_info: dict | None = None


def _cfg(cfg: dict | None) -> dict:
    return cfg if cfg is not None else config.load("graph")


def _canon(edges: list[GraphEdge]) -> list[GraphEdge]:
    from graphrag.kg.artifact import canonical
    return canonical([], edges)[1]


# --- RELATED (Ruling R25: one approved source) -------------------------------------------------------

class RelatedMismatch(ValueError):
    """The recomputed RELATED set is not the adopted one (graph.yaml related.adopted_edges[_sha256])."""


def edge_set_sha(edges: list[GraphEdge]) -> str:
    """sha256 of the canonical RELATED jsonl: edges sorted by (src, dst), json sort_keys, one per line."""
    import hashlib
    body = "".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                   for e in sorted(edges, key=lambda e: (e.src, e.dst, e.type)))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _cached_vectors(npy: Path, ids: list[str], digest: str, what: str) -> dict[str, np.ndarray]:
    """Rows of a Task 8 vector cache (<name>.npy + <name>_ids.json) keyed by chunk id, after checking the
    row order (= sorted chunk ids), the sha256 of the embedded texts and the npy sha256 recorded with it;
    never re-encodes (a mismatch raises RelatedMismatch)."""
    import hashlib
    meta_f = npy.with_name(npy.stem + "_ids.json")
    if not npy.exists() or not meta_f.exists():
        raise RelatedMismatch(f"{what}: {npy} / {meta_f.name} missing (run scripts/build_related.py)")
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    if meta.get("ids") != ids:
        raise RelatedMismatch(f"{what}: {meta_f.name} row ids differ from the sorted chunk ids of data/records")
    if meta.get("text_sha256") != digest:
        raise RelatedMismatch(f"{what}: text sha256 {meta.get('text_sha256')} != recomputed {digest}")
    raw = npy.read_bytes()
    if meta.get("npy_sha256") and hashlib.sha256(raw).hexdigest() != meta["npy_sha256"]:
        raise RelatedMismatch(f"{what}: {npy.name} sha256 differs from {meta_f.name}")
    X = np.load(npy)
    if X.shape[0] != len(ids):
        raise RelatedMismatch(f"{what}: {X.shape[0]} rows for {len(ids)} chunks")
    return {i: X[k] for k, i in enumerate(ids)}


def index_text_digest(records) -> str:
    """sha256 of the embedded chunk texts (data.yaml embed_field) in chunk-id order (scripts/build_related.text_digest)."""
    import hashlib
    field_ = config.load("data")["embed_field"]
    h = hashlib.sha256()
    for c in sorted((c for r in records for c in r.chunks), key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8") + b"\0" + (getattr(c, field_) or c.text_norm).encode("utf-8") + b"\n")
    return h.hexdigest()


def topic_texts(records, rc: dict) -> dict[str, str]:
    """chunk id -> semantic.topic_text, exactly as scripts/build_related.load_topic_vectors (speaker labels
    removed in STENOGRAMMA only, related.topic_drop_crumb)."""
    from graphrag.kg import semantic as S
    speakers = S.speaker_names(records, rc)
    drop = rc.get("topic_drop_crumb") or []
    return {c.chunk_id: S.topic_text(c.index_text, c.text_norm,
                                     speakers if r.meta.section == "STENOGRAMMA" else frozenset(), drop)
            for r in records for c in r.chunks}


def topic_text_digest(records, rc: dict) -> str:
    import hashlib
    tt = topic_texts(records, rc)
    h = hashlib.sha256()
    for i in sorted(tt):
        h.update(i.encode("utf-8") + b"\0" + tt[i].encode("utf-8") + b"\n")
    return h.hexdigest()


def related_edges(records, explicit_edges: list[GraphEdge], cfg: dict | None = None,
                  vec_dir: Path | str | None = None) -> tuple[list[GraphEdge], dict]:
    """THE source of RELATED edges for the graph (Ruling R25): semantic.latent_edges at graph.yaml
    related.theta_sim with the Task 8 inputs -- index_text vectors (data/graph/chunk_vecs.npy, for the
    near-duplicate / template pass) and topic-text vectors (chunk_vecs_topic.npy, the score; texts recomputed
    with semantic.topic_text exactly as scripts/build_related.py and checked by sha256), explicit pairs of the
    extracted REFERS_TO / AMENDS / REPEALS edges, translation / version document links. The result must be
    the adopted set: related.adopted_edges edges with sha256 related.adopted_edges_sha256 (RelatedMismatch)."""
    from graphrag.kg import semantic as S
    cfg = _cfg(cfg)
    rc = {**S.DEFAULTS, **(cfg.get("related") or {})}
    theta = rc.get("theta_sim")
    if theta is None:
        raise RelatedMismatch("graph.yaml related.theta_sim is not set (the adopted θ)")
    recs = sorted(records, key=lambda r: r.doc_id)
    vdir = config.path(vec_dir or "data/graph")
    ids = sorted(c.chunk_id for r in recs for c in r.chunks)
    vecs = _cached_vectors(vdir / "chunk_vecs.npy", ids, index_text_digest(recs), "index_text vectors")
    topic = None
    if rc.get("score_vectors") == "topic":
        topic = _cached_vectors(vdir / "chunk_vecs_topic.npy", ids, topic_text_digest(recs, rc), "topic vectors")
    linked = [(e.src, e.dst) for e in explicit_edges if e.type == "REFERS_TO"
              and e.attrs.get("relation") in ("version_of", "translation_of")]
    edges = S.latent_edges(recs, vecs, float(theta), int(rc["k_ann"]), int(rc["k_rel"]),
                           explicit_edges=explicit_edges, linked_docs=linked, cfg=rc, block=int(rc["block"]),
                           score_vectors=topic)
    sha = edge_set_sha(edges)
    info = {"theta": float(theta), "score_vectors": rc.get("score_vectors") or "index_text", "edges": len(edges),
            "edge_set_sha256": sha, "adopted_edges": rc.get("adopted_edges"),
            "adopted_edges_sha256": rc.get("adopted_edges_sha256")}
    if rc.get("adopted_edges") is not None and len(edges) != int(rc["adopted_edges"]):
        raise RelatedMismatch(f"recomputed {len(edges)} RELATED edges != adopted {rc['adopted_edges']}")
    if rc.get("adopted_edges_sha256") and sha != rc["adopted_edges_sha256"]:
        raise RelatedMismatch(f"recomputed RELATED edge-set sha256 {sha} != adopted {rc['adopted_edges_sha256']}")
    info["matches_adopted"] = bool(rc.get("adopted_edges_sha256"))
    return edges, info


def explicit_for_related(ex: "Extracted") -> list[GraphEdge]:
    """The explicit edges Task 8 excluded RELATED pairs with (the extracted edges after pipeline.r11_filter);
    only their REFERS_TO / AMENDS / REPEALS pairs and version / translation links are used."""
    return r11_filter(ex.edges)[0]


def check_related_stats(edges: list[GraphEdge], stats_path: Path | str) -> dict:
    """Cross-check of the RELATED set with Task 8 related_stats.json (final: θ, count, weight min / p50 / max,
    degree vertices / median / p99 / max)."""
    from graphrag.kg import semantic as S
    st = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    fin = st["final"]
    ws = np.array([e.attrs["weight"] for e in edges]) if edges else np.zeros(0)
    problems = []
    if len(edges) != fin["edges"]:
        problems.append(f"{len(edges)} edges, related_stats final {fin['edges']}")
    if edges and {"min": float(ws.min()), "p50": float(np.median(ws)), "max": float(ws.max())} != fin["weight"]:
        problems.append("weights differ from related_stats final.weight")
    deg = S.degree_stats(S.degrees((e.src, e.dst) for e in edges))
    if any(deg[k] != fin["degree"][k] for k in ("vertices", "median", "p99", "max")):
        problems.append("degree stats differ from related_stats final.degree")
    return {"theta": st["theta"], "gate_status": st.get("gate_status"), "edges": fin["edges"], "problems": problems}


# --- assembly -----------------------------------------------------------------------------------------

def _evidences(e: GraphEdge) -> list[dict]:
    evs = list(e.attrs.get("evidence_all") or [])
    if e.attrs.get("evidence") is not None:
        evs.append(e.attrs["evidence"])
    uniq = {(x["doc_id"], x["start"], x["end"], x["quote"]): x for x in evs}
    return [uniq[k] for k in sorted(uniq)]


def merge_group(group: list[GraphEdge]) -> GraphEdge:
    """One edge from edges with the same artifact.edge_key: evidence united (by position), evidence = first;
    RELATED: the max weight (attached only when every copy is)."""
    if len(group) == 1:
        return group[0]
    g0 = group[0]
    if g0.type == "RELATED":
        w = max(e.attrs["weight"] for e in group)
        attrs = {"weight": w}
        if all(e.attrs.get("attached") for e in group):
            attrs["attached"] = True
        return GraphEdge(g0.src, g0.dst, g0.type, attrs)
    if g0.type == "AMENDS":                  # one extractor, keyed by (src, AMENDS, dst): never duplicated
        raise ValueError(f"duplicate AMENDS {g0.src} -> {g0.dst}: ops cannot be merged safely")
    if g0.type not in EVIDENCE_TYPES:
        return g0
    evs = sorted({(x["doc_id"], x["start"], x["end"], x["quote"]): x for e in group for x in _evidences(e)}.items())
    evs = [x for _, x in evs]
    attrs = {}
    for e in group:                           # other attrs: the first edge's (dates: the latest)
        for k, v in e.attrs.items():
            if k in ("evidence", "evidence_all"):
                continue
            if k == "date" and v and attrs.get("date"):
                attrs["date"] = max(attrs["date"], v)
            else:
                attrs.setdefault(k, v)
    attrs["evidence"] = evs[0]
    if len(evs) > 1:
        attrs["evidence_all"] = evs
    return GraphEdge(g0.src, g0.dst, g0.type, attrs)


def dedup_edges(edges: list[GraphEdge]) -> tuple[list[GraphEdge], Counter]:
    groups: dict[tuple, list[GraphEdge]] = defaultdict(list)
    for e in _canon(edges):
        groups[edge_key(e)].append(e)
    merged = Counter({k[1]: 0 for k in groups})
    out = []
    for k in sorted(groups):
        g = groups[k]
        if len(g) > 1:
            merged[k[1]] += len(g) - 1
        out.append(merge_group(g))
    return out, Counter({t: c for t, c in merged.items() if c})


def dedup_entities(nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> tuple[dict, list, dict]:
    """§3.2: entity / concept vertices with the same (kind, type, norm_key) are one vertex -- the minimal id
    survives, its edges are moved onto it (the extractors already key ids by (type, norm_key), so normally
    nothing merges). Returns (nodes, edges, {dropped id: survivor})."""
    by: dict[tuple, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        if n.kind in ("entity", "concept") and n.attrs.get("norm_key"):
            by[(n.kind, n.type, n.attrs["norm_key"])].append(nid)
    remap = {}
    for k, ids in by.items():
        if len(ids) > 1:
            keep = min(ids)
            for x in ids:
                if x != keep:
                    remap[x] = keep
    if not remap:
        return nodes, edges, {}
    out_nodes = {nid: n for nid, n in nodes.items() if nid not in remap}
    for x, keep in sorted(remap.items()):
        k = out_nodes[keep]
        al = sorted(set(k.attrs.get("aliases") or []) | set(nodes[x].attrs.get("aliases") or []) | {nodes[x].title})
        out_nodes[keep] = GraphNode(k.node_id, k.kind, k.type, k.title, k.doc_id, k.external, {**k.attrs, "aliases": al})
    moved = [GraphEdge(remap.get(e.src, e.src), remap.get(e.dst, e.dst), e.type, e.attrs) for e in edges]
    return out_nodes, [e for e in moved if e.src != e.dst], dict(sorted(remap.items()))


def structural(records) -> tuple[list[GraphNode], list[GraphEdge]]:
    from graphrag.kg.extract_ref import structure_nodes
    nodes, edges = [], []
    for r in sorted(records, key=lambda r: r.doc_id):
        nodes += structure_nodes(r)
        edges += [GraphEdge(n.parent_id, n.node_id, "CONTAINS", {}) for n in r.structure if n.parent_id]
    return nodes, edges


def assemble(records, cfg: dict | None = None, related: list[GraphEdge] = (), extracted: Extracted | None = None,
             schema: Schema | None = None) -> Graph:
    """Vertices + edges before features / the isolated-vertex step (§3.2 steps 1-3)."""
    from graphrag.kg.extract_amend import apply_repealed_on
    cfg = _cfg(cfg)
    schema = schema or Schema(cfg)
    ridx = records if isinstance(records, RecordIndex) else RecordIndex(records)
    ex = extracted if extracted is not None else extract_all(ridx.records, cfg)
    s_nodes, s_edges = structural(ridx.records)
    nodes: dict[str, GraphNode] = {n.node_id: n for n in s_nodes}
    for nid in sorted(ex.nodes):
        nodes.setdefault(nid, ex.nodes[nid])
    nodes = {n.node_id: n for n in apply_repealed_on([nodes[k] for k in sorted(nodes)], ex.repealed_on)}
    kept, r11 = r11_filter(ex.edges)
    edges = s_edges + list(ex.stub_contains) + kept + list(related)
    nodes, edges, remap = dedup_entities(nodes, edges)
    n_in = Counter(e.type for e in edges)
    edges, merged = dedup_edges(edges)
    edges, n_rp = repeals_first_evidence(edges)
    nodes, edges, pruned = prune_orphan_ext(nodes, edges)
    g = Graph([nodes[k] for k in sorted(nodes)], edges, rindex=ridx, schema=schema, cfg=cfg)
    g.stats = {"input_edges_by_type": dict(sorted(n_in.items())), "merged_duplicate_edges": dict(sorted(merged.items())),
               "r11_dropped_refers_amended": len(r11), "entity_dedup_merged": len(remap),
               "entity_dedup_remap": remap, "extract_llm": ex.llm, "extract": ex.stats,
               "repeals_evidence_first_by_position": n_rp, **pruned}
    return g


def repeals_first_evidence(edges: list[GraphEdge]) -> tuple[list[GraphEdge], int]:
    """Ruling R26 (5): REPEALS attrs.evidence = the first of its evidence by position (doc_id, start, end,
    quote), as for every other merged type (Task 5 had put the repeal item first); evidence_all keeps all of
    them in position order. Returns (edges, how many REPEALS changed their primary evidence)."""
    out, changed = [], 0
    for e in edges:
        if e.type == "REPEALS" and e.attrs.get("evidence") is not None:
            evs = _evidences(e)
            if evs[0] != e.attrs["evidence"]:
                changed += 1
            attrs = {k: v for k, v in e.attrs.items() if k not in ("evidence", "evidence_all")}
            attrs["evidence"] = evs[0]
            if len(evs) > 1:
                attrs["evidence_all"] = evs
            e = GraphEdge(e.src, e.dst, e.type, attrs)
        out.append(e)
    return out, changed


def prune_orphan_ext(nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> tuple[dict, list, dict]:
    """Ruling R26 (1): an external act (EXT document vertex) whose subtree -- the act and its EXT element
    stubs -- touches no edge but its own stub CONTAINS carries no evidence-backed provenance (a registry
    leftover: every mention re-resolved to a corpus act); the act, its stubs and those CONTAINS are dropped
    before X and the isolated-vertex step. Returns (nodes, edges, stats)."""
    sub: dict[str, set[str]] = defaultdict(set)
    acts = {nid for nid, n in nodes.items() if n.kind == "document" and n.external}
    for nid, n in nodes.items():
        if n.kind == "element" and n.external:
            act = n.attrs.get("act") or next((a for a in sorted(acts) if nid.startswith(a + "/")), None)
            if act in acts:
                sub[act].add(nid)
    touched: dict[str, bool] = {a: False for a in acts}
    owner = {x: a for a, xs in sub.items() for x in xs} | {a: a for a in acts}
    for e in edges:
        a, b = owner.get(e.src), owner.get(e.dst)
        if a is None and b is None:
            continue
        if e.type == "CONTAINS" and a is not None and a == b:
            continue
        for o in (a, b):
            if o is not None:
                touched[o] = True
    orphan = sorted(a for a, t in touched.items() if not t)
    drop = set(orphan) | {x for a in orphan for x in sub[a]}
    out_nodes = {nid: n for nid, n in nodes.items() if nid not in drop}
    out_edges = [e for e in edges if e.src not in drop and e.dst not in drop]
    return out_nodes, out_edges, {"pruned_orphan_ext_acts": len(orphan),
                                  "pruned_orphan_ext_stubs": len(drop) - len(orphan),
                                  "pruned_orphan_ext_ids": orphan}


# --- isolated vertices (§3.2 "присоединить изолированные вершины", R1 / R24 (1)) ------------------------------

def doc_groups_of(graph: Graph) -> dict[str, str]:
    from graphrag.kg import semantic as S
    docs = sorted({n.doc_id for n in graph.nodes if n.doc_id})
    linked = [(e.src, e.dst) for e in graph.edges if e.type == "REFERS_TO"
              and e.attrs.get("relation") in ("version_of", "translation_of")]
    return S.doc_groups(docs, [(a.split("/", 1)[0], b.split("/", 1)[0]) for a, b in linked])


def isolated(graph: Graph) -> list[str]:
    seen = {x for e in graph.edges for x in (e.src, e.dst)}
    return [n.node_id for n in graph.nodes if n.node_id not in seen]


def attach_isolated(graph: Graph, X: np.ndarray) -> dict:
    """Every vertex without edges gets RELATED {weight: cos, attached: true} (src < dst) to its nearest vertex
    by X (float64 cosine rounded to 1e-6; ties: the smaller node id): of its own document when it has other
    vertices, else of the corpus -- never a vertex of another document of its translation / version group.
    Only document / element vertices can carry RELATED (graph.yaml); others are left for validate()."""
    ids = [n.node_id for n in graph.nodes]
    pos = {nid: i for i, nid in enumerate(ids)}
    Z = X.astype(np.float64)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    ok = np.array([n.kind in ATTACHABLE for n in graph.nodes])
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(graph.nodes):
        if n.doc_id and ok[i]:
            by_doc[n.doc_id].append(i)
    groups = doc_groups_of(graph)
    group_of = np.array([groups.get(n.doc_id, "") if n.doc_id else "" for n in graph.nodes], dtype=object)
    doc_of = np.array([n.doc_id or "" for n in graph.nodes], dtype=object)
    iso = isolated(graph)
    added, info = [], {"isolated_before": len(iso), "attached": 0, "same_document": 0, "corpus": 0,
                       "by_kind": Counter(), "unattachable": [],
                       "attached_vertices": []}
    for nid in iso:
        i = pos[nid]
        n = graph.nodes[i]
        if not ok[i]:
            info["unattachable"].append(nid)
            continue
        cand = [j for j in by_doc.get(n.doc_id, []) if j != i] if n.doc_id else []
        scope = "same_document"
        if not cand:
            scope = "corpus"
            m = ok.copy()
            m[i] = False
            if n.doc_id and group_of[i]:          # other documents of its translation / version group
                m &= ~((group_of == group_of[i]) & (doc_of != n.doc_id))
            cand = np.flatnonzero(m).tolist()
        if not cand:
            info["unattachable"].append(nid)
            continue
        c = np.array(cand)
        s = np.round(Z[c] @ Z[i], DECIMALS)
        best = s.max()
        j = min((ids[k] for k in c[s == best]))  # ties: the smallest node id
        w = float(min(1.0, max(0.0, best)))
        a, b = sorted((nid, j))
        added.append(GraphEdge(a, b, "RELATED", {"weight": w, "attached": True}))
        info["attached_vertices"].append(nid)
        info[scope] += 1
        info["by_kind"][n.kind] += 1
    edges, merged = dedup_edges(graph.edges + added)
    graph.edges = edges
    info["attached"] = len(added)
    info["merged_attach_pairs"] = sum(merged.values())
    info["by_kind"] = dict(sorted(info["by_kind"].items()))
    return info


# --- validation (§3.4) --------------------------------------------------------------------------------

def latent_share(graph: Graph) -> dict:
    rel = [e for e in graph.edges if e.type == "RELATED"]
    att = sum(1 for e in rel if e.attrs.get("attached"))
    E = len(graph.edges)
    return {"related_latent": len(rel) - att, "related_attached": att, "edges": E,
            "share_without_attached": round((len(rel) - att) / E, 6) if E else 0.0,
            "share_with_attached": round(len(rel) / E, 6) if E else 0.0}


def validate(graph: Graph, max_examples: int | None = None) -> list[str]:
    """§3.4: every problem as a message; [] = the graph passed."""
    schema = graph.schema or Schema(graph.cfg or config.load("graph"))
    ridx = graph.rindex
    errs: list[str] = []
    nodes = {}
    for n in graph.nodes:
        if n.node_id in nodes:
            errs.append(f"duplicate node {n.node_id}")
        nodes[n.node_id] = n
        errs += schema.validate_node(n)
    ids = Counter(edge_key(e) for e in graph.edges)
    errs += [f"duplicate edge {k}" for k, c in sorted(ids.items()) if c > 1]
    for e in graph.edges:
        errs += schema.validate_edge(e, nodes)
    # no isolated vertices
    errs += [f"isolated vertex {nid}" for nid in isolated(graph)]
    # CONTAINS -> the immediate child: parent_id for corpus vertices, the id path for EXT stubs
    struct = ridx.struct if ridx else {}
    for e in graph.edges:
        if e.type != "CONTAINS":
            continue
        d = nodes.get(e.dst)
        if d is None:
            continue
        if e.dst in struct and not d.external:
            if struct[e.dst][1].parent_id != e.src:
                errs.append(f"CONTAINS {e.src} -> {e.dst}: not the immediate parent "
                            f"(parent_id {struct[e.dst][1].parent_id!r})")
        else:
            rest = e.dst[len(e.src) + 1:] if e.dst.startswith(e.src + "/") else None
            if not d.external or not rest or "/" in rest:
                errs.append(f"CONTAINS {e.src} -> {e.dst}: stub is not an immediate path child")
    has_parent = {e.dst for e in graph.edges if e.type == "CONTAINS"}
    for nid, (r, n) in sorted(struct.items()):
        if n.parent_id is not None and nid in nodes and nid not in has_parent:
            errs.append(f"CONTAINS missing for {nid} (parent {n.parent_id})")
    # REPEALS dated
    errs += [f"REPEALS {e.src} -> {e.dst}: no date" for e in graph.edges if e.type == "REPEALS" and not e.attrs.get("date")]
    # μ⁻¹ non-empty for every corpus structural vertex (stubs / external acts have no chunks by construction)
    if ridx is not None:
        errs += [f"{n.node_id}: empty μ⁻¹ (no chunks)" for n in graph.nodes
                 if n.kind in ("document", "element") and not n.external and n.node_id in struct
                 and not ridx.mu_inv.get(n.node_id)]
    # latent share
    cap = float((graph.cfg.get("related") or {}).get("max_latent_share", 0.30))
    ls = latent_share(graph)
    if ls["share_without_attached"] > cap:
        errs.append(f"RELATED share {ls['share_without_attached']} > max_latent_share {cap}")
    # evidence is the exact text span of the source vertex's own document
    texts = ridx.texts if ridx else {}
    for e in graph.edges:
        src_doc = nodes[e.src].doc_id if e.src in nodes else None
        for ev in _evidences(e) if e.type in EVIDENCE_TYPES else []:
            t = texts.get(ev["doc_id"])
            if ev["doc_id"] != src_doc:           # R26 (3): "текст своего документа" -- the source's own text
                errs.append(f"{e.type} {e.src} -> {e.dst}: evidence doc {ev['doc_id']!r} is not the source's "
                            f"document {src_doc!r}")
            if t is None:
                errs.append(f"{e.type} {e.src} -> {e.dst}: evidence doc {ev['doc_id']!r} not in the corpus")
            elif t[ev["start"]:ev["end"]] != ev["quote"]:
                errs.append(f"{e.type} {e.src} -> {e.dst}: evidence [{ev['start']}:{ev['end']}] is not its quote")
    return errs if max_examples is None else errs[:max_examples]


# --- statistics ---------------------------------------------------------------------------------------

def degree_summary(graph: Graph) -> dict:
    deg = Counter()
    for e in graph.edges:
        deg[e.src] += 1
        deg[e.dst] += 1
    vals = np.array([deg.get(n.node_id, 0) for n in graph.nodes], dtype=np.int64)
    if not len(vals):
        return {}
    top = sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    def q(p):
        return int(np.quantile(vals, p, method="higher"))
    return {"vertices": int(len(vals)), "min": int(vals.min()), "median": float(np.median(vals)),
            "mean": round(float(vals.mean()), 4), "p90": q(0.90), "p99": q(0.99), "max": int(vals.max()),
            "degree_1": int((vals == 1).sum()), "top": [[k, v] for k, v in top],
            "by_kind": {k: {"median": float(np.median(v)), "max": int(np.max(v))}
                        for k, v in sorted(_by_kind(graph, deg).items())}}


def _by_kind(graph, deg):
    out = defaultdict(list)
    for n in graph.nodes:
        out[n.kind].append(deg.get(n.node_id, 0))
    return out


def graph_stats(graph: Graph) -> dict:
    kinds = Counter(n.kind for n in graph.nodes)
    ktype = Counter((n.kind, n.type) for n in graph.nodes)
    ext = Counter(n.kind for n in graph.nodes if n.external)
    et = Counter(e.type for e in graph.edges)
    rel = Counter(e.attrs.get("relation") for e in graph.edges if e.type == "REFERS_TO")
    ops = Counter(e.attrs.get("op") for e in graph.edges if e.type == "AMENDS")
    ops_all = Counter(o for e in graph.edges if e.type == "AMENDS" for o in (e.attrs.get("ops") or [e.attrs.get("op")]))
    rp = [e for e in graph.edges if e.type == "REPEALS"]
    schema = graph.schema
    return {
        "nodes": len(graph.nodes), "edges": len(graph.edges),
        "nodes_by_kind": dict(sorted(kinds.items())),
        "nodes_by_kind_type": {k: dict(sorted((t, c) for (kk, t), c in ktype.items() if kk == k)) for k in sorted(kinds)},
        "external_nodes_by_kind": dict(sorted(ext.items())),
        "stub_elements": sum(1 for n in graph.nodes if n.kind == "element" and n.external),
        "edges_by_type": {t: et.get(t, 0) for t in (schema.edge_types if schema else sorted(et))},
        "refers_to_by_relation": dict(sorted((str(k), v) for k, v in rel.items())),
        "amends_by_op": dict(sorted((str(k), v) for k, v in ops.items())),
        "amends_ops_all_evidence": dict(sorted((str(k), v) for k, v in ops_all.items())),
        "amends_with_date": sum(1 for e in graph.edges if e.type == "AMENDS" and e.attrs.get("date")),
        "repeals": len(rp), "repeals_with_date": sum(1 for e in rp if e.attrs.get("date")),
        "related": latent_share(graph),
        "degree": degree_summary(graph),
    }


# --- version inputs (Ruling R26 (2)) --------------------------------------------------------------------

VERSION_CODE_GLOBS = ("graphrag/**/*.py", "scripts/build_graph.py")
VERSION_CONFIG_GLOB = "config/*.yaml"
# M3 Task 1: packages the graph build never imports (tests/test_m3_version_isolation.py proves it) and configs that
# never shape the artifact are outside the version digest, so M3+ work does not move the graph version
VERSION_CODE_EXCLUDE = ("graphrag/retrieval/", "graphrag/env/", "graphrag/eval/", "graphrag/dataset/")
VERSION_CONFIG_SKIP = ("config/train.yaml",)
VERSION_LIBRARIES = ("numpy", "torch", "torch_geometric", "transformers", "tokenizers", "sentence-transformers",
                     "FlagEmbedding", "pymorphy3", "pymorphy3-dicts-ru", "PyYAML")


def _sha_bytes(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files_digest(root: Path | str, globs) -> dict[str, str]:
    """relative posix path -> sha256 of every file matching `globs` under `root` (no __pycache__), sorted."""
    root = Path(root)
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    return {p.relative_to(root).as_posix(): _sha_bytes(p) for p in files}


def code_digest(root: Path | str) -> dict[str, str]:
    """Every graph-affecting source file (graphrag/ minus VERSION_CODE_EXCLUDE, plus the build script)."""
    return {p: d for p, d in files_digest(root, VERSION_CODE_GLOBS).items() if not p.startswith(VERSION_CODE_EXCLUDE)}


# Ruling R29 E: top-level config sections that do not shape the artifact (graph.yaml store: the Neo4j write
# allowlist / load lock of Task 10) are left out of the version digest
VERSION_CONFIG_EXCLUDE = {"graph.yaml": ("store",), "data.yaml": ("start",)}   # R29 E + M3 Task 1


def _without_sections(text: str, names) -> str:
    """`text` (YAML) without the top-level blocks `names`: from "<name>:" at column 0 up to the next column-0 line."""
    for n in names:
        text = re.sub(rf"(?ms)^{re.escape(n)}:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)(?:\n|\Z))*", "", text)
    return text


def config_digest(root: Path | str) -> dict[str, str]:
    import hashlib
    out = {p: d for p, d in files_digest(root, (VERSION_CONFIG_GLOB,)).items() if p not in VERSION_CONFIG_SKIP}
    for name, sections in VERSION_CONFIG_EXCLUDE.items():
        rel = f"config/{name}"
        if rel in out:
            body = _without_sections((Path(root) / rel).read_text(encoding="utf-8"), sections)
            out[rel] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return out


def library_versions(names=VERSION_LIBRARIES) -> dict[str, str | None]:
    """Installed distribution versions (None = not installed): tokenizer / encoder / lemmatizer / PyG code that
    shapes vertex texts, X and the export."""
    import importlib.metadata as md
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


# --- the whole build ----------------------------------------------------------------------------------

def build_graph(records, cfg: dict | None = None, *, encoder, related: list[GraphEdge] = (),
                extracted: Extracted | None = None, cache=None, schema: Schema | None = None,
                round_decimals: int | None = None) -> Graph:
    """§3.2 completion: assemble -> X (R1: before attachment) -> attach isolated -> stats. graph.x = X in
    graph.nodes order; graph.stats["validation"] = validate(graph)."""
    from graphrag.kg.features import embed_nodes
    cfg = _cfg(cfg)
    g = assemble(records, cfg, related, extracted, schema)
    fcfg = (cfg.get("build") or {}).get("features") or {}
    X = embed_nodes(g, encoder, cache, fcfg, round_decimals)
    att = attach_isolated(g, X)
    g.x = X
    viol = validate(g)
    g.stats = {**g.stats, **graph_stats(g), "isolated_attached": att,
               "validation": {"violations": len(viol), "examples": viol[:50]}}
    return g


def write_artifact(graph: Graph, d: Path | str, stats: dict, schema: Schema | None = None,
                   overwrite: bool = False) -> dict:
    """Write the artifact directory (graphrag.kg.artifact.write_graph + x.npy, pyg.pt, stats.json; stats gets
    the PyG summary); returns the sha256 of every file. A version directory is immutable (Ruling R26 (2)):
    existing files with different bytes raise artifact.ArtifactConflict unless overwrite=True."""
    import hashlib
    import io
    from graphrag.kg.artifact import write_graph
    from graphrag.kg.pyg import pyg_bytes, pyg_payload
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(graph.x, dtype=np.float32))
    payload = pyg_payload(graph, graph.x, schema)
    extra = {"x.npy": buf.getvalue(), "pyg.pt": pyg_bytes(payload)}
    stats["pyg"] = {"num_nodes": len(payload["node_ids"]), "num_edges": int(payload["edge_index"].shape[1]),
                    "x_shape": list(payload["x"].shape), "chunks": len(payload["chunk_ids"]),
                    "mu_inv_nnz": int(payload["mu_inv_idx"].numel()),
                    "edge_types": payload["edge_types"], "node_kinds": payload["node_kinds"],
                    "edges_by_type": {t: int((payload["edge_type"] == i).sum()) for i, t in enumerate(payload["edge_types"])},
                    "direction": "edges exported as stored (directed src -> dst); RELATED in both directions; reverse "
                                 "relations of the other types are NOT added (the M4 GAT adds them as a transform)",
                    "edge_weight": "RELATED = cosine (attrs.weight), every other type 1.0",
                    "edge_order": "(edge_type, src row, dst row)",
                    # Ruling R31 (4): additive fields and their vocabularies (index -> name; -1 = not applicable)
                    "fields_r31": {"edge_attr_relation": "REFERS_TO relation idx (edge_relations), -1 otherwise",
                                   "edge_attr_op": "AMENDS primary op idx (amend_ops), -1 otherwise",
                                   "edge_attr_ops": "AMENDS multi-hot [E', len(amend_ops)] of attrs.ops (or [op])",
                                   "edge_date": "AMENDS / REPEALS attrs.date as days since 1970-01-01, -1 otherwise",
                                   "node_type": "document / element fine type idx (node_types), -1 entity / concept",
                                   "entity_type": "entity type idx (entity_types), -1 otherwise",
                                   "is_external": "bool: EXT act or EXT element stub",
                                   "node_doc": "document row idx (doc_ids) of the vertex, -1 entity / concept"},
                    "vocabularies": {k: payload[k] for k in ("edge_relations", "amend_ops", "node_types",
                                                              "entity_types")},
                    "doc_ids": len(payload["doc_ids"]),
                    "counts_r31": {
                        "edge_attr_relation": {payload["edge_relations"][i]: int((payload["edge_attr_relation"] == i).sum())
                                               for i in range(len(payload["edge_relations"]))},
                        "edge_attr_op": {payload["amend_ops"][i]: int((payload["edge_attr_op"] == i).sum())
                                         for i in range(len(payload["amend_ops"]))},
                        "edge_date_set": int((payload["edge_date"] >= 0).sum()),
                        "node_type": {payload["node_types"][i]: int((payload["node_type"] == i).sum())
                                      for i in range(len(payload["node_types"]))},
                        "entity_type": {payload["entity_types"][i]: int((payload["entity_type"] == i).sum())
                                        for i in range(len(payload["entity_types"]))},
                        "is_external": int(payload["is_external"].sum()),
                        "node_doc_unset": int((payload["node_doc"] < 0).sum())}}
    extra["stats.json"] = (json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    d = Path(d)
    schema = schema or graph.schema
    write_graph(d, graph.nodes, graph.edges, extra_files=extra, schema=schema, overwrite=overwrite)
    return {name: hashlib.sha256((d / name).read_bytes()).hexdigest()
            for name in ("nodes.jsonl", "edges.jsonl", "schema.json", "x.npy", "pyg.pt", "stats.json")}
