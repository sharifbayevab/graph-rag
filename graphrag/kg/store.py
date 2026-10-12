"""Neo4j store for the knowledge graph (Task 10, Rulings R27, R28).

Two owners write to Neo4j, and neither destroys what the other wrote:

* reconcile_structure (M1, called from graphrag.stores.sync.upsert_document -- the single write path for
  a document) owns the corpus structure: Document/Element vertices of the records and their CONTAINS
  edges (unversioned: no `versions` on the node, no `graph` on the edge). It MERGEs with SET n += r, so
  enrichment properties written by load_graph survive; it deletes only the document's vertices that
  disappeared from the record (their incoming/outgoing enrichment edges go with them and are counted as
  dangling removals, logged as a :GraphEvent for the next load report) and only the stale CONTAINS.
* load_graph (M2) owns everything else in the graph artifact: EXT document/element stubs, entities,
  concepts, stub CONTAINS and every non-CONTAINS edge. These are versioned per named graph (default
  "main"): an edge carries r.graph / r.graph_version / r.edge_key, a vertex carries n.versions (a list
  of "<graph>@<version>" tags). The new version is written next to the old one, checked for parity
  with the artifact, and only then are the other versions' edges and vertices removed (a swap). A load
  that fails (an exception mid-load, or a parity failure) removes its own partial writes and leaves the
  old version intact. The active version of a graph is recorded on (:GraphMeta {graph}).

Property layout (R27 (4)): a vertex keeps the artifact attrs verbatim as n.attrs_json (plus flat scalar /
scalar-list props for filters); an edge stores nested attrs (evidence, evidence_all, ops) as JSON strings
(names listed in r.json_attrs) and scalar attrs (relation, op, date, weight, attached) flat. RELATED is
stored once, src < dst. Edge identity = the artifact dedup key (src, type, dst[, relation]) as r.edge_key.

Ruling R28 (Task 10 fix round 1):
* managed keys: n.enr_keys lists the props load_graph wrote on a vertex; a later write sends an explicit
  null for every managed key its row no longer has, so no stale flat prop survives a version that drops an
  attr. Structure-owned props of corpus vertices (title / type / number / parent_id / n_chunks) are never
  managed.
* staging: before a load first overwrites an existing vertex it copies the old managed props (and, for a
  versioned vertex, its kind label) to the single slot n.prev_json / n.prev_for = "<graph>@<version>". A
  failed load restores them (keys it added removed, keys it changed or dropped put back); a successful load
  clears the slot. A slot left by a load that died is resolved by the next load (restored, or only
  cleared when it belongs to the active version).
* parity is full by default: every vertex (attrs, flat props, enr_keys and no stale key, exactly one kind
  label, version membership) and every edge (endpoints, attrs; extra edges too).
* guard: load_graph / drop_versions refuse endpoints outside config/graph.yaml store.write_endpoints.
* lock: one load per graph at a time (GraphMeta lock_owner / lock_at); a lock older than
  store.lock_stale_s may be broken explicitly (break_stale_lock).
* orphans: removing a corpus article (reconcile_structure) can leave a versioned vertex of the active
  version with no edge (e.g. an entity only that article mentioned). Neo4j then no longer equals the active
  artifact (parity would fail) until build_graph + load_graph run again; the next load reports them
  (report["orphans"]) and its swap removes those the new artifact no longer has.
"""
from __future__ import annotations

import json
import os
import random
import socket
import time
import uuid
from collections import Counter, defaultdict

from neo4j import GraphDatabase

from graphrag import config
from graphrag.kg.artifact import edge_key
from graphrag.kg.schema import GraphEdge, GraphNode
from graphrag.stores.guard import graph_write_problem, neo4j_reset_problem
from graphrag.types import DocRecord

DEFAULT_GRAPH = "main"
BATCH = 2000
LABELS = {"document": "Document", "element": "Element", "entity": "Entity", "concept": "Concept"}
KIND_OF_LABEL = {v: k for k, v in LABELS.items()}
# vertex props owned by the structure (reconcile_structure) or the store itself: never set from artifact attrs
_NODE_CORE = frozenset({"node_id", "kind", "type", "title", "doc_id", "external", "parent_id", "n_chunks",
                        "number", "versions", "attrs_json", "kg_title"})
_EDGE_CORE = frozenset({"edge_key", "graph", "graph_version", "json_attrs"})
# vertex props of the store's bookkeeping (never managed, never compared as attrs)
_STORE_KEYS = frozenset({"versions", "enr_keys", "prev_json", "prev_for"})
# props reconcile_structure writes on a corpus vertex (never managed by load_graph)
_STRUCT_KEYS = frozenset({"node_id", "type", "title", "number", "parent_id", "doc_id", "n_chunks"})
_KIND_LABELS_REMOVE = "REMOVE n:Document, n:Element, n:Entity, n:Concept"
RESET_QUERIES = ("MATCH (n:Node) DETACH DELETE n", "MATCH (m:GraphMeta) DELETE m", "MATCH (e:GraphEvent) DELETE e")


class GraphLoadError(RuntimeError):
    """load_graph failed (precheck, write error or parity); .report holds the load report."""

    def __init__(self, msg: str, report: dict):
        super().__init__(msg)
        self.report = report


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      default=lambda o: o.to_json() if hasattr(o, "to_json") else str(o))


def _plain(obj):
    """attrs with Evidence instances turned into dicts (what the artifact files hold)."""
    return json.loads(_dumps(obj))


def is_corpus(n: GraphNode) -> bool:
    """A vertex of the corpus structure (owned by reconcile_structure, never versioned)."""
    return not n.external and n.kind in ("document", "element") and bool(n.doc_id)


def edge_key_json(e: GraphEdge) -> str:
    """r.edge_key in Neo4j: the JSON of artifact.edge_key (the artifact dedup key)."""
    return json.dumps(list(edge_key(e)), ensure_ascii=False)


def _scalar_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def node_props(n: GraphNode) -> dict:
    """Properties load_graph writes on a vertex (never None; a managed key the vertex had but this dict
    lacks is removed by the write, R28 (1))."""
    attrs = _plain(n.attrs)
    p = {"kind": n.kind, "external": n.external, "attrs_json": _dumps(attrs)}
    if is_corpus(n):
        p["kg_title"] = n.title                    # title / type / number stay the structure's
    else:
        p.update(node_id=n.node_id, type=n.type, title=n.title)
        if n.doc_id:
            p["doc_id"] = n.doc_id
        if isinstance(attrs.get("number"), str):
            p["number"] = attrs["number"]
    for k, v in attrs.items():
        if k in _NODE_CORE or v is None:
            continue
        if isinstance(v, (str, int, float, bool)) or _scalar_list(v):
            p[k] = v
    return p


def edge_props(e: GraphEdge) -> dict:
    p, js = {}, []
    for k, v in _plain(e.attrs).items():
        if v is None:
            continue
        if k in _EDGE_CORE:
            raise ValueError(f"{e.type} {e.src} -> {e.dst}: attr {k!r} clashes with a store property")
        if isinstance(v, (dict, list)):
            p[k] = _dumps(v)
            js.append(k)
        else:
            p[k] = v
    p["json_attrs"] = sorted(js)
    return p


def edge_attrs_from_props(p: dict) -> dict:
    js = set(p.get("json_attrs") or [])
    return {k: (json.loads(v) if k in js else v) for k, v in p.items() if k not in _EDGE_CORE}


def node_from_props(p: dict, labels) -> dict:
    kind = next((KIND_OF_LABEL[l] for l in labels if l in KIND_OF_LABEL), None)
    return {"node_id": p.get("node_id"), "kind": kind, "type": p.get("type"),
            "title": p.get("kg_title", p.get("title")), "doc_id": p.get("doc_id"),
            "external": p.get("external"),
            "attrs": json.loads(p["attrs_json"]) if "attrs_json" in p else None}


def _batches(rows: list, n: int):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def _etype(t: str) -> str:
    if not t.isidentifier() or not t.isupper():
        raise ValueError(f"bad edge type {t!r}")
    return t


def _managed(props: dict) -> set[str]:
    """Keys load_graph owns on a stored vertex: its n.enr_keys; for a vertex written before R28 (no
    enr_keys) every prop that is neither the store's nor, on a corpus vertex, the structure's."""
    if props.get("enr_keys") is not None:
        return set(props["enr_keys"])
    own = _STORE_KEYS | {"node_id"} | (_STRUCT_KEYS if "versions" not in props else frozenset())
    return set(props) - own


def _stage_rows(tx, rows: list[dict], tag: str) -> tuple[list[dict], int]:
    """Vertex rows {id, p} -> write rows {id, p (+ null for every managed key the row dropped), keys (the new
    n.enr_keys), prev (old managed props to stage, or None)}; reads the batch in the write transaction."""
    cur = {r["id"]: (r["p"], r["l"]) for r in tx.run(
        "UNWIND $ids AS id MATCH (n:Node {node_id: id}) RETURN id, properties(n) AS p, labels(n) AS l",
        ids=[r["id"] for r in rows])}
    out, staged = [], 0
    for r in rows:
        p = {k: v for k, v in r["p"].items() if k != "node_id"}
        keys, prev = sorted(p), None
        if r["id"] in cur:
            props, labels = cur[r["id"]]
            old = _managed(props)
            p.update({k: None for k in old - set(p)})
            if props.get("prev_for") != tag:       # first write of this load: stage the old state
                label = next((l for l in labels if l in KIND_OF_LABEL), None) if "versions" in props else None
                prev = _dumps({"props": {k: props[k] for k in sorted(old) if k in props},
                               "enr_keys": props.get("enr_keys"), "label": label})
                staged += 1
        out.append({"id": r["id"], "p": p, "keys": keys, "prev": prev})
    return out, staged


def _node_problems(n: GraphNode, props: dict | None, labels, tag: str) -> list[str]:
    """Why the stored vertex differs from artifact vertex `n` ([] when it is exact)."""
    if props is None:
        return ["missing"]
    probs = []
    have = node_from_props(props, labels)
    have["external"] = bool(have["external"])
    if have != _plain(n.to_json()):
        probs.append("attrs")
    want = {k: v for k, v in node_props(n).items() if k != "node_id"}
    probs += [f"flat:{k}" for k in sorted(want) if props.get(k) != want[k]]
    if sorted(props.get("enr_keys") or []) != sorted(want):
        probs.append("enr_keys")
    corpus = is_corpus(n)
    stale = set(props) - set(want) - _STORE_KEYS - ({"node_id"} | (_STRUCT_KEYS if corpus else frozenset()))
    if stale:
        probs.append(f"stale:{','.join(sorted(stale))}")
    if sorted(labels) != sorted(["Node", LABELS[n.kind]]):
        probs.append(f"labels:{','.join(sorted(labels))}")
    if corpus and "versions" in props:
        probs.append("versions:corpus_vertex_versioned")
    if not corpus and tag not in (props.get("versions") or []):
        probs.append("versions:tag_missing")
    return probs


# predicates shared by the version-drop queries: $mode is all | keep | drop
_VER_SEL = ("$mode = 'all' OR ($mode = 'keep' AND r.graph_version <> $v) "
            "OR ($mode = 'drop' AND r.graph_version = $v)")
_TAG_SEL = ("t STARTS WITH $prefix AND ($mode = 'all' OR ($mode = 'keep' AND t <> $tag) "
            "OR ($mode = 'drop' AND t = $tag))")


class GraphStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.uri = uri or config.NEO4J_URI
        self.driver = GraphDatabase.driver(self.uri,
                                           auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
                                           notifications_min_severity="OFF")
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE").consume()
            s.run("CREATE INDEX node_doc IF NOT EXISTS FOR (n:Node) ON (n.doc_id)").consume()
            s.run("CREATE CONSTRAINT graph_meta IF NOT EXISTS FOR (m:GraphMeta) REQUIRE m.graph IS UNIQUE").consume()
            # R28 (6): version drops, parity and staging look edges up by (graph, graph_version), vertices by prev_for
            s.run("CREATE INDEX node_prev_for IF NOT EXISTS FOR (n:Node) ON (n.prev_for)").consume()
            for et in sorted(config.load("graph")["edge_types"]):
                s.run(f"CREATE INDEX rel_{_etype(et).lower()}_version IF NOT EXISTS "
                      f"FOR ()-[r:{et}]-() ON (r.graph, r.graph_version)").consume()

    # ── corpus structure (M1 write path) ─────────────────────────────────────
    @staticmethod
    def _remove_nodes(tx, doc_id: str, ids: list[str]) -> dict:
        """DETACH DELETE corpus vertices `ids`; count the enrichment edges this removes (dangling
        otherwise) and log them as a :GraphEvent for the next load report."""
        if not ids:
            return {}
        rows = tx.run("MATCH (n:Node)-[r]-() WHERE n.node_id IN $ids AND r.graph IS NOT NULL "
                      "RETURN r.graph AS g, type(r) AS t, count(DISTINCT r) AS c", ids=ids).data()
        tx.run("MATCH (n:Node) WHERE n.node_id IN $ids AND n.versions IS NULL DETACH DELETE n", ids=ids).consume()
        dangling: dict[str, dict[str, int]] = defaultdict(dict)
        for r in rows:
            dangling[r["g"]][r["t"]] = r["c"]
        if dangling:
            tx.run("CREATE (:GraphEvent {kind: 'dangling_removed', doc_id: $d, graphs: $gs, counts_json: $cj, "
                   "node_ids: $ids, at: datetime()})", d=doc_id, gs=sorted(dangling),
                   cj=_dumps(dangling), ids=ids[:100]).consume()
        return {g: dict(sorted(v.items())) for g, v in sorted(dangling.items())}

    def reconcile_structure(self, rec: DocRecord, n_chunks: dict[str, int]) -> dict:
        """Make the document's structure in Neo4j equal `rec.structure` without touching anything else:
        MERGE vertices (SET n += r keeps enrichment props), delete only the document's vertices that are
        gone (and, with them, their now-dangling enrichment edges, counted), reconcile the CONTAINS edges."""
        rows = [{"node_id": n.node_id, "type": n.type, "title": n.title, "number": n.number,
                 "parent_id": n.parent_id, "doc_id": rec.doc_id,
                 "n_chunks": n_chunks.get(n.node_id, 0)} for n in rec.structure]
        docs = [r for r in rows if r["type"] == "документ"]
        elems = [r for r in rows if r["type"] != "документ"]
        ids = [r["node_id"] for r in rows]
        pairs = [f"{n.parent_id}\t{n.node_id}" for n in rec.structure if n.parent_id]
        edges = [{"src": n.parent_id, "dst": n.node_id} for n in rec.structure if n.parent_id]
        out = {"doc_id": rec.doc_id, "nodes": len(rows)}

        def write(tx):
            gone = [r["id"] for r in tx.run(
                "MATCH (n:Node {doc_id: $d}) WHERE n.versions IS NULL AND NOT n.node_id IN $ids "
                "RETURN n.node_id AS id", d=rec.doc_id, ids=ids)]
            out["nodes_removed"] = len(gone)
            out["dangling_removed"] = self._remove_nodes(tx, rec.doc_id, gone)
            tx.run("UNWIND $rows AS r MERGE (n:Node {node_id: r.node_id}) SET n += r, n:Document "
                   "REMOVE n:Element", rows=docs).consume()
            tx.run("UNWIND $rows AS r MERGE (n:Node {node_id: r.node_id}) SET n += r, n:Element "
                   "REMOVE n:Document", rows=elems).consume()
            out["contains_removed"] = tx.run(
                "MATCH (a:Node {doc_id: $d})-[r:CONTAINS]->(b:Node) WHERE r.graph IS NULL "
                "AND NOT a.node_id + '\t' + b.node_id IN $pairs DELETE r RETURN count(*) AS c",
                d=rec.doc_id, pairs=pairs).single()["c"]
            made = tx.run("UNWIND $edges AS e MATCH (a:Node {node_id: e.src}), (b:Node {node_id: e.dst}) "
                          "MERGE (a)-[:CONTAINS]->(b) RETURN count(*) AS c", edges=edges).single()["c"]
            if made != len(edges):                   # a parent_id that names no node: fail loudly
                raise ValueError(f"{rec.doc_id}: {len(edges) - made} CONTAINS edges have no endpoint")

        with self.driver.session() as s:
            s.execute_write(write)
        return out

    def delete_document(self, doc_id: str) -> dict:
        """Remove the document's structure vertices (and the enrichment edges on them, counted)."""
        out = {"doc_id": doc_id}

        def write(tx):
            ids = [r["id"] for r in tx.run("MATCH (n:Node {doc_id: $d}) WHERE n.versions IS NULL "
                                           "RETURN n.node_id AS id", d=doc_id)]
            out["nodes_removed"] = len(ids)
            out["dangling_removed"] = self._remove_nodes(tx, doc_id, ids)

        with self.driver.session() as s:
            s.execute_write(write)
        return out

    def reset_all(self) -> None:
        """Wipe every graph vertex and the load bookkeeping (build_indexes.py --reset, which has its own
        endpoint guard). R28 (5): GraphMeta / GraphEvent go as well, so no active_version outlives the data
        it names and no dangling-removal event is reported for a graph that no longer exists.
        R31 (5): guarded here as well -- only the local v2 Neo4j (localhost:7688) unless GRL_ALLOW_RESET=1
        (guard.neo4j_reset_problem); a store built without __init__ is checked against config.NEO4J_URI,
        the endpoint __init__ would have used."""
        p = neo4j_reset_problem(getattr(self, "uri", None) or config.NEO4J_URI)
        if p:
            raise PermissionError(f"graph reset refused: {p}")
        with self.driver.session() as s:
            for q in RESET_QUERIES:
                s.run(q).consume()

    # ── graph enrichment (M2 artifact load) ──────────────────────────────────
    def _check_write(self) -> None:
        """R28 (4): the artifact load and version drops write only to an allowlisted endpoint
        (config/graph.yaml store.write_endpoints). reconcile_structure -- the document write path that M7
        runs against prod -- is deliberately not guarded here; M7 adds its own explicit allow."""
        p = graph_write_problem(self.uri, config.load("graph")["store"]["write_endpoints"])
        if p:
            raise PermissionError(f"graph write refused: {p}")

    def active_version(self, graph: str = DEFAULT_GRAPH) -> str | None:
        with self.driver.session() as s:
            r = s.run("MATCH (m:GraphMeta {graph: $g}) RETURN m.active_version AS v", g=graph).single()
            return r["v"] if r else None

    @staticmethod
    def _edge_types(s) -> list[str]:
        """Every relationship type in the database (typed queries use the (graph, graph_version) indexes)."""
        return sorted(_etype(r["t"]) for r in s.run(
            "CALL db.relationshipTypes() YIELD relationshipType AS t RETURN t"))

    def drop_versions(self, graph: str, mode: str, version: str | None = None) -> dict:
        """mode all: every version of `graph`; keep: all but `version`; drop: only `version`. Deletes
        the selected versions' edges, strips their tags from vertices and deletes vertices left
        without a tag. Corpus structure (unversioned) is never touched."""
        self._check_write()
        assert mode in ("all", "keep", "drop") and (mode == "all" or version is not None)
        prm = {"g": graph, "mode": mode, "v": version or "", "prefix": f"{graph}@", "tag": f"{graph}@{version}"}
        edges: Counter = Counter()
        with self.driver.session() as s:
            for et in self._edge_types(s):
                q = (f"MATCH ()-[r:{et}]->() WHERE r.graph = $g AND r.graph_version IS NOT NULL AND ({_VER_SEL}) "
                     "WITH r LIMIT 20000 DELETE r RETURN count(*) AS c")
                while True:
                    c = s.execute_write(lambda tx: tx.run(q, **prm).single()["c"])
                    if not c:
                        break
                    edges[et] += c
            s.execute_write(lambda tx: tx.run(
                f"MATCH (n:Node) WHERE any(t IN n.versions WHERE {_TAG_SEL}) "
                f"SET n.versions = [t IN n.versions WHERE NOT ({_TAG_SEL})]", **prm).consume())
            nodes = 0
            while True:
                c = s.execute_write(lambda tx: tx.run(
                    "MATCH (n:Node) WHERE n.versions = [] WITH n LIMIT 10000 DETACH DELETE n "
                    "RETURN count(*) AS c").single()["c"])
                if not c:
                    break
                nodes += c
        return {"edges": dict(sorted(edges.items())), "edges_total": sum(edges.values()), "nodes": nodes}

    def _precheck(self, corpus: list[GraphNode], versioned: list[GraphNode], strict_corpus: bool) -> dict:
        docs = sorted({n.doc_id for n in corpus})
        want = {n.node_id for n in corpus}
        with self.driver.session() as s:
            have = {r["id"] for r in s.run("MATCH (n:Node) WHERE n.doc_id IN $docs AND n.versions IS NULL "
                                           "RETURN n.node_id AS id", docs=docs)}
            clash = [r["id"] for r in s.run("MATCH (n:Node) WHERE n.node_id IN $ids AND n.versions IS NULL "
                                            "RETURN n.node_id AS id", ids=[n.node_id for n in versioned])]
            extra_docs = [r["d"] for r in s.run(
                "MATCH (n:Document) WHERE n.versions IS NULL AND n.doc_id IS NOT NULL AND NOT n.doc_id IN $docs "
                "RETURN n.doc_id AS d ORDER BY d", docs=docs)] if strict_corpus else []
        missing, extra = sorted(want - have), sorted(have - want)
        return {"corpus_docs": len(docs), "corpus_nodes": len(want),
                "missing_corpus_nodes": len(missing), "missing_examples": missing[:20],
                "extra_corpus_nodes": len(extra), "extra_examples": extra[:20],
                "versioned_id_clashes": clash[:20], "corpus_docs_not_in_artifact": extra_docs[:20],
                "ok": not (missing or extra or clash or extra_docs)}

    # R28 (1)+(2): every vertex write reads the batch's current props in the same transaction, sends an
    # explicit null for each managed key the new row no longer has, records the new managed keys as
    # n.enr_keys and, the first time this load touches an existing vertex, stages its old managed props
    # (and kind label) in the single slot n.prev_json / n.prev_for = tag.
    _NODE_SET = ("SET n += r.p, n.enr_keys = r.keys, n.prev_json = coalesce(r.prev, n.prev_json), "
                 "n.prev_for = CASE WHEN r.prev IS NULL THEN n.prev_for ELSE $tag END")

    def _write_node_batch(self, label: str, rows: list[dict], tag: str) -> int:
        assert label in KIND_OF_LABEL

        def write(tx):
            wr, staged = _stage_rows(tx, rows, tag)
            tx.run(f"UNWIND $rows AS r MERGE (n:Node {{node_id: r.id}}) {self._NODE_SET}, "
                   "n.versions = CASE WHEN $tag IN coalesce(n.versions, []) THEN n.versions "
                   f"ELSE coalesce(n.versions, []) + $tag END {_KIND_LABELS_REMOVE} SET n:{label}",
                   rows=wr, tag=tag).consume()
            return staged
        with self.driver.session() as s:
            return s.execute_write(write)

    def _write_corpus_batch(self, rows: list[dict], tag: str) -> int:
        def write(tx):
            wr, staged = _stage_rows(tx, rows, tag)
            c = tx.run(f"UNWIND $rows AS r MATCH (n:Node {{node_id: r.id}}) WHERE n.versions IS NULL "
                       f"{self._NODE_SET} RETURN count(n) AS c", rows=wr, tag=tag).single()["c"]
            if c != len(rows):
                raise ValueError(f"{len(rows) - c} corpus vertices missing in Neo4j")
            return staged
        with self.driver.session() as s:
            return s.execute_write(write)

    def _restore_staged(self, tag: str) -> int:
        """Put back the staged props (and kind label) of every vertex staged for `tag`: keys the load added
        are removed, keys it changed or removed get their old values; the slot is cleared."""
        def work(tx):
            rows = tx.run("MATCH (n:Node) WHERE n.prev_for = $tag WITH n LIMIT $lim "
                          "RETURN n.node_id AS id, properties(n) AS p", tag=tag, lim=BATCH).data()
            by_label = defaultdict(list)
            for r in rows:
                prev = json.loads(r["p"]["prev_json"])
                p = {k: None for k in _managed(r["p"]) | set(prev["props"])}
                p.update(prev["props"])
                by_label[prev["label"]].append({"id": r["id"], "p": p, "keys": prev["enr_keys"]})
            for label, rs in sorted(by_label.items(), key=lambda kv: kv[0] or ""):
                assert label is None or label in KIND_OF_LABEL
                lab = f" {_KIND_LABELS_REMOVE} SET n:{label}" if label else ""
                tx.run("UNWIND $rows AS r MATCH (n:Node {node_id: r.id}) SET n += r.p, n.enr_keys = r.keys "
                       "REMOVE n.prev_json, n.prev_for" + lab, rows=rs).consume()
            return len(rows)
        n = 0
        with self.driver.session() as s:
            while c := s.execute_write(work):
                n += c
        return n

    def _clear_staged(self, tag: str) -> int:
        n = 0
        with self.driver.session() as s:
            while c := s.execute_write(lambda tx: tx.run(
                    "MATCH (n:Node) WHERE n.prev_for = $tag WITH n LIMIT 10000 REMOVE n.prev_json, n.prev_for "
                    "RETURN count(*) AS c", tag=tag).single()["c"]):
                n += c
        return n

    def _resolve_dead_staging(self, graph: str, active: str | None) -> dict:
        """Staging slots left by a load of `graph` that died: a slot for the active version is only cleared
        (the vertex already holds the active version's props); any other slot is restored."""
        with self.driver.session() as s:
            tags = sorted(r["t"] for r in s.run("MATCH (n:Node) WHERE n.prev_for STARTS WITH $p "
                                                "RETURN DISTINCT n.prev_for AS t", p=f"{graph}@"))
        out = {"restored": 0, "cleared": 0}
        for t in tags:
            if active is not None and t == f"{graph}@{active}":
                out["cleared"] += self._clear_staged(t)
            else:
                out["restored"] += self._restore_staged(t)
        return out

    def _version_present(self, graph: str, version: str) -> bool:
        with self.driver.session() as s:
            if s.run("MATCH (n:Node) WHERE $t IN n.versions RETURN 1 LIMIT 1", t=f"{graph}@{version}").single():
                return True
            return any(s.run(f"MATCH ()-[r:{et}]->() WHERE r.graph = $g AND r.graph_version = $v RETURN 1 LIMIT 1",
                             g=graph, v=version).single() for et in self._edge_types(s))

    def _orphans(self, graph: str, version: str) -> list[str]:
        """Vertices of `version` with no edge of `graph` (R28 (8): e.g. an entity whose only mentions were
        in a corpus article that reconcile_structure removed)."""
        with self.driver.session() as s:
            return [r["id"] for r in s.run(
                "MATCH (n:Node) WHERE $t IN n.versions AND NOT EXISTS { MATCH (n)-[r]-() WHERE r.graph = $g } "
                "RETURN n.node_id AS id ORDER BY id", t=f"{graph}@{version}", g=graph)]

    def _acquire_lock(self, graph: str, break_stale_lock: bool) -> dict:
        """R28 (7): one load per graph at a time. The lock lives on (:GraphMeta {graph}) as lock_owner /
        lock_at; a held lock refuses the load unless break_stale_lock is set AND the lock is older than
        config store.lock_stale_s (a load that died without releasing it)."""
        owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        stale_s = int(config.load("graph")["store"]["lock_stale_s"])

        def work(tx):
            # the write takes the node's exclusive lock first, so the read below sees any concurrent taker
            tx.run("MERGE (m:GraphMeta {graph: $g}) SET m.lock_seq = coalesce(m.lock_seq, 0) + 1", g=graph).consume()
            r = tx.run("MATCH (m:GraphMeta {graph: $g}) RETURN m.lock_owner AS owner, toString(m.lock_at) AS at, "
                       "CASE WHEN m.lock_at IS NULL THEN NULL "
                       "ELSE duration.inSeconds(m.lock_at, datetime()).seconds END AS age_s", g=graph).single().data()
            if r["owner"] is not None and not (break_stale_lock and r["age_s"] is not None and r["age_s"] >= stale_s):
                return {"taken": False, "held": r, "stale_after_s": stale_s}
            tx.run("MATCH (m:GraphMeta {graph: $g}) SET m.lock_owner = $o, m.lock_at = datetime()",
                   g=graph, o=owner).consume()
            return {"taken": True, "owner": owner, "broke_stale_lock": r if r["owner"] is not None else None}
        with self.driver.session() as s:
            return s.execute_write(work)

    def _release_lock(self, graph: str, owner: str) -> None:
        with self.driver.session() as s:
            s.run("MATCH (m:GraphMeta {graph: $g}) WHERE m.lock_owner = $o REMOVE m.lock_owner, m.lock_at",
                  g=graph, o=owner).consume()

    def _consume_events(self, graph: str, delete: bool) -> list[dict]:
        with self.driver.session() as s:
            rows = s.run("MATCH (e:GraphEvent) WHERE $g IN e.graphs RETURN e.doc_id AS doc_id, "
                         "e.counts_json AS c, e.node_ids AS node_ids, toString(e.at) AS at ORDER BY at",
                         g=graph).data()
            if delete:
                s.run("MATCH (e:GraphEvent) WHERE $g IN e.graphs DELETE e", g=graph).consume()
        return [{"doc_id": r["doc_id"], "at": r["at"], "node_ids": r["node_ids"],
                 "removed": json.loads(r["c"]).get(graph, {})} for r in rows]

    def load_graph(self, nodes: list[GraphNode], edges: list[GraphEdge], version: str,
                   graph: str = DEFAULT_GRAPH, batch: int = BATCH, strict_corpus: bool = True,
                   sample: int | None = None, seed: int = 0, break_stale_lock: bool = False) -> dict:
        """Load the artifact as `version` of `graph`, verify parity (full unless `sample`), then activate it
        and drop the other versions. Returns the load report; raises GraphLoadError (with .report) when
        the lock, the precheck, a write or the parity check fails -- the partial new version is removed,
        staged vertex props are restored and the previous version stays active, exactly as it was."""
        self._check_write()
        for e in edges:
            if e.type == "RELATED" and not e.src < e.dst:
                raise ValueError(f"RELATED {e.src} -> {e.dst}: stored once with src < dst")
        lock = self._acquire_lock(graph, break_stale_lock)
        if not lock["taken"]:
            h = lock["held"]
            msg = (f"graph {graph!r} is locked by {h['owner']} since {h['at']} ({h['age_s']} s; a lock older than "
                   f"{lock['stale_after_s']} s may be broken with break_stale_lock)")
            raise GraphLoadError(msg, {"graph": graph, "version": version, "ok": False, "error": msg, "lock": lock})
        try:
            return self._load(nodes, edges, version, graph, batch, strict_corpus, sample, seed, lock)
        finally:
            self._release_lock(graph, lock["owner"])

    def _load(self, nodes, edges, version, graph, batch, strict_corpus, sample, seed, lock) -> dict:
        t0 = time.time()
        tag = f"{graph}@{version}"
        active = self.active_version(graph)
        report: dict = {"graph": graph, "version": version, "previous_version": active, "ok": False,
                        "lock": {"owner": lock["owner"], "broke_stale_lock": lock["broke_stale_lock"]}}
        if active is not None and not self._version_present(graph, active):
            report["stale_active_version"] = active     # GraphMeta outlived its data: nothing to protect
            active = None
        corpus = [n for n in nodes if is_corpus(n)]
        versioned = [n for n in nodes if not is_corpus(n)]
        corpus_ids = {n.node_id for n in corpus}
        # leftovers of a load that died without cleanup (only versions other than the active one)
        report["purged_staging"] = self._resolve_dead_staging(graph, active)
        report["purged_leftovers"] = self.drop_versions(graph, "keep", active) if active else \
            self.drop_versions(graph, "all")
        orphans = self._orphans(graph, active) if active else []
        report["precheck"] = pre = self._precheck(corpus, versioned, strict_corpus)
        if not pre["ok"]:
            report["error"] = "precheck failed (corpus in Neo4j differs from the artifact)"
            report["dangling_edge_removals"] = self._consume_events(graph, delete=False)
            raise GraphLoadError(report["error"], report)
        written: Counter = Counter()
        staged = 0
        try:
            t = time.time()
            by_label = defaultdict(list)
            for n in versioned:
                by_label[LABELS[n.kind]].append({"id": n.node_id, "p": node_props(n)})
            for label, rows in sorted(by_label.items()):
                for b in _batches(rows, batch):
                    staged += self._write_node_batch(label, b, tag)
                    written[f"nodes:{label}"] += len(b)
            for b in _batches([{"id": n.node_id, "p": node_props(n)} for n in corpus], batch):
                staged += self._write_corpus_batch(b, tag)
                written["nodes:corpus_enriched"] += len(b)
            report["staged_vertices"] = staged
            report["t_nodes_s"] = round(time.time() - t, 2)
            t = time.time()
            by_type = defaultdict(list)
            for e in edges:
                if e.type == "CONTAINS" and e.src in corpus_ids and e.dst in corpus_ids:
                    continue                       # corpus CONTAINS: owned by reconcile_structure
                by_type[e.type].append({"src": e.src, "dst": e.dst, "key": edge_key_json(e), "p": edge_props(e)})
            for etype, rows in sorted(by_type.items()):
                for b in _batches(rows, batch):
                    self._write_edge_batch(etype, b, graph, version)
                    written[f"edges:{etype}"] += len(b)
            report["t_edges_s"] = round(time.time() - t, 2)
            report["written"] = dict(sorted(written.items()))
            t = time.time()
            report["parity"] = par = self.parity(nodes, edges, version, graph, sample=sample, seed=seed)
            report["t_parity_s"] = round(time.time() - t, 2)
            if not par["ok"]:
                raise GraphLoadError("parity check failed", report)
        except Exception as e:
            report["written"] = dict(sorted(written.items()))
            report["staged_vertices"] = staged
            report["error"] = f"{type(e).__name__}: {e}"
            if version != active:                  # never drop the version that is live
                report["rollback"] = self.drop_versions(graph, "drop", version)
            report["rollback_restored_vertices"] = self._restore_staged(tag)
            report["active_version"] = self.active_version(graph)
            report["dangling_edge_removals"] = self._consume_events(graph, delete=False)
            report["t_total_s"] = round(time.time() - t0, 2)
            if isinstance(e, GraphLoadError):
                raise
            raise GraphLoadError(report["error"], report) from e
        # activate first, then clear the staging and drop the other versions: a crash from here on
        # leaves the new version active and complete; the next load purges what is left of the old one
        t = time.time()
        with self.driver.session() as s:
            s.run("MERGE (m:GraphMeta {graph: $g}) SET m.active_version = $v, m.previous_version = $p, "
                  "m.loaded_at = datetime()", g=graph, v=version, p=active).consume()
        report["staging_cleared"] = self._clear_staged(tag)
        report["swap"] = self.drop_versions(graph, "keep", version)
        report["t_swap_s"] = round(time.time() - t, 2)
        with self.driver.session() as s:
            still = {r["id"] for r in s.run("MATCH (n:Node) WHERE n.node_id IN $ids RETURN n.node_id AS id",
                                            ids=orphans)}
        report["orphans"] = {"previous_version": len(orphans), "examples": orphans[:20],
                             "removed_by_swap": len(orphans) - len(still), "kept_in_new_version": len(still)}
        report["dangling_edge_removals"] = self._consume_events(graph, delete=True)
        report["active_version"] = version
        report["ok"] = True
        report["t_total_s"] = round(time.time() - t0, 2)
        return report

    def _write_edge_batch(self, etype: str, rows: list[dict], graph: str, version: str) -> None:
        _etype(etype)

        def write(tx):
            c = tx.run(f"UNWIND $rows AS e MATCH (a:Node {{node_id: e.src}}) MATCH (b:Node {{node_id: e.dst}}) "
                       f"MERGE (a)-[r:{etype} {{edge_key: e.key, graph: $g, graph_version: $v}}]->(b) "
                       "SET r += e.p RETURN count(r) AS c", rows=rows, g=graph, v=version).single()["c"]
            if c != len(rows):
                raise ValueError(f"{etype}: {len(rows) - c} edges have no endpoint in Neo4j")
        with self.driver.session() as s:
            s.execute_write(write)

    def parity(self, nodes: list[GraphNode], edges: list[GraphEdge], version: str,
               graph: str = DEFAULT_GRAPH, sample: int | None = None, seed: int = 0) -> dict:
        """Neo4j (corpus structure of the artifact's documents + `version` of `graph`) vs the artifact:
        counts by kind/type, edge type, REFERS_TO relation, AMENDS op; then every vertex and edge (or
        `sample` random ones of each) compared: attrs, flat props and n.enr_keys (no stale key), exactly
        one kind label, version membership (R27 (7), R28 (3)); a full check also finds extra edges."""
        tag = f"{graph}@{version}"
        corpus_ids = {n.node_id for n in nodes if is_corpus(n)}
        docs = sorted({n.doc_id for n in nodes if is_corpus(n)})

        def is_struct(e):
            return e.type == "CONTAINS" and e.src in corpus_ids and e.dst in corpus_ids

        want_nodes = Counter(f"{n.kind}/{n.type}" for n in nodes)
        want_edges = Counter(e.type for e in edges)
        want_rel = Counter(str(e.attrs.get("relation")) for e in edges if e.type == "REFERS_TO")
        want_op = Counter(str(e.attrs.get("op")) for e in edges if e.type == "AMENDS")
        have_nodes, have_edges, have_rel, have_op = Counter(), Counter(), Counter(), Counter()
        rng = random.Random(seed)
        full = sample is None
        chk_nodes = nodes if full or sample >= len(nodes) else rng.sample(nodes, sample)
        chk_edges = edges if full or sample >= len(edges) else rng.sample(edges, sample)
        lab = "head([l IN labels(n) WHERE l <> 'Node'])"
        got_e: dict[tuple, list] = defaultdict(list)
        with self.driver.session() as s:
            for q, kw in ((f"MATCH (n:Node) WHERE $tag IN n.versions RETURN {lab} AS l, n.type AS t, count(*) AS c",
                           {"tag": tag}),
                          (f"MATCH (n:Node) WHERE n.versions IS NULL AND n.doc_id IN $docs "
                           f"RETURN {lab} AS l, n.type AS t, count(*) AS c", {"docs": docs})):
                for r in s.run(q, **kw):
                    have_nodes[f"{KIND_OF_LABEL.get(r['l'], r['l'])}/{r['t']}"] += r["c"]
            for et in self._edge_types(s):
                for r in s.run(f"MATCH (a:Node)-[r:{et}]->(b:Node) WHERE r.graph = $g AND r.graph_version = $v "
                               "RETURN a.node_id AS s, b.node_id AS d, r.edge_key AS k, properties(r) AS p",
                               g=graph, v=version):
                    got_e[(et, r["k"])].append((r["s"], r["d"], edge_attrs_from_props(r["p"])))
                    have_edges[et] += 1
                    if et == "REFERS_TO":
                        have_rel[str(r["p"].get("relation"))] += 1
                    if et == "AMENDS":
                        have_op[str(r["p"].get("op"))] += 1
            struct_have = [(r["s"], r["d"]) for r in s.run(
                "MATCH (a:Node)-[r:CONTAINS]->(b:Node) WHERE r.graph IS NULL AND a.versions IS NULL "
                "AND a.doc_id IN $docs RETURN a.node_id AS s, b.node_id AS d", docs=docs)]
            have_edges["CONTAINS"] += len(struct_have)
            got_n = {}
            for b in _batches([n.node_id for n in chk_nodes], 5000):
                got_n.update({r["id"]: (r["p"], r["l"]) for r in s.run(
                    "UNWIND $ids AS id MATCH (n:Node {node_id: id}) RETURN id, properties(n) AS p, labels(n) AS l",
                    ids=b)})
        node_bad = []
        for n in chk_nodes:
            probs = _node_problems(n, *got_n.get(n.node_id, (None, None)), tag)
            if probs:
                node_bad.append({"node_id": n.node_id, "problems": probs})
        edge_bad = []
        struct_set = set(struct_have)
        for e in chk_edges:
            if is_struct(e):
                if (e.src, e.dst) not in struct_set:
                    edge_bad.append({"edge": [e.src, e.type, e.dst], "missing": True})
                continue
            have, want = got_e.get((e.type, edge_key_json(e))), [(e.src, e.dst, _plain(e.attrs))]
            if have != want:
                edge_bad.append({"edge": [e.src, e.type, e.dst], "artifact": want, "neo4j": have})
        if full:                                    # anything in Neo4j the artifact does not have
            keys = {(e.type, edge_key_json(e)) for e in edges if not is_struct(e)}
            edge_bad += [{"extra": list(k)} for k in sorted(set(got_e) - keys)]
            want_struct = {(e.src, e.dst) for e in edges if is_struct(e)}
            edge_bad += [{"extra": [a, "CONTAINS", b]} for a, b in sorted(struct_set - want_struct)]

        def diff(a: Counter, b: Counter) -> dict:
            return {k: {"artifact": a.get(k, 0), "neo4j": b.get(k, 0)} for k in sorted(set(a) | set(b))
                    if a.get(k, 0) != b.get(k, 0)}

        d = {"nodes": diff(want_nodes, have_nodes), "edges": diff(want_edges, have_edges),
             "refers_to_relation": diff(want_rel, have_rel), "amends_op": diff(want_op, have_op)}
        return {"ok": not any(d.values()) and not node_bad and not edge_bad, "full": full,
                "nodes_artifact": sum(want_nodes.values()), "nodes_neo4j": sum(have_nodes.values()),
                "edges_artifact": sum(want_edges.values()), "edges_neo4j": sum(have_edges.values()),
                "edges_by_type": {k: {"artifact": want_edges[k], "neo4j": have_edges.get(k, 0)}
                                  for k in sorted(want_edges)},
                "count_diffs": d, "checked_nodes": len(chk_nodes), "checked_edges": len(chk_edges),
                "node_mismatches": len(node_bad), "edge_mismatches": len(edge_bad),
                "mismatch_examples": (node_bad + edge_bad)[:10]}

    # ── reads ────────────────────────────────────────────────────────────────
    def node(self, node_id: str) -> dict | None:
        with self.driver.session() as s:
            r = s.run("MATCH (n:Node {node_id: $id}) RETURN n", id=node_id).single()
            return dict(r["n"]) if r else None

    def path_to_document(self, node_id: str) -> list[str]:
        """Node ids from the Document root down to node_id ([] if unknown)."""
        with self.driver.session() as s:
            r = s.run("MATCH p = (d:Document)-[:CONTAINS*0..32]->(n:Node {node_id: $id}) "
                      "RETURN [x IN nodes(p) | x.node_id] AS ids ORDER BY length(p) DESC LIMIT 1",
                      id=node_id).single()
            return r["ids"] if r else []

    def count_nodes(self) -> int:
        """Corpus structure vertices (what the records hold); versioned enrichment vertices excluded."""
        with self.driver.session() as s:
            return s.run("MATCH (n:Node) WHERE n.versions IS NULL RETURN count(n) AS c").single()["c"]
