"""Graph schema (IMPLEMENTATION.md §3.1): the vertex kinds, edge types, their allowed endpoints and
required attributes live in config/graph.yaml; this module only holds the record types and
validates records against that data."""
import datetime as _dt
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from graphrag import config

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EVIDENCE_FIELDS = ("doc_id", "start", "end", "quote")


@dataclass(frozen=True)
class Evidence:
    """A text span backing an extracted edge: [start, end) character offsets in doc_id's text."""
    doc_id: str
    start: int
    end: int
    quote: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Evidence":
        return cls(d["doc_id"], d["start"], d["end"], d["quote"])


@dataclass
class GraphNode:
    node_id: str
    kind: str                  # graph.yaml vertex_kinds
    type: str                  # fine type: документ / статья / organ / понятие ...
    title: str
    doc_id: str | None
    external: bool             # referenced but not in the corpus
    attrs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GraphNode":
        return cls(d["node_id"], d["kind"], d["type"], d["title"], d.get("doc_id"),
                   d.get("external", False), dict(d.get("attrs") or {}))


@dataclass
class GraphEdge:
    src: str
    dst: str
    type: str                  # graph.yaml edge_types key
    attrs: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GraphEdge":
        return cls(d["src"], d["dst"], d["type"], dict(d.get("attrs") or {}))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _as_evidence_dict(ev: Any) -> Any:
    return ev.to_json() if isinstance(ev, Evidence) else ev


def _check_evidence(ev: Any, schema: "Schema | None" = None) -> list[str]:
    ev = _as_evidence_dict(ev)
    if not isinstance(ev, dict):
        return [f"evidence must be an Evidence or a dict with {list(_EVIDENCE_FIELDS)}, "
                f"got {type(ev).__name__}"]
    errs = [f"evidence missing field {k!r}" for k in _EVIDENCE_FIELDS if k not in ev]
    if errs:
        return errs
    if not isinstance(ev["doc_id"], str) or not ev["doc_id"]:
        errs.append("evidence.doc_id must be a non-empty string")
    if not (_is_int(ev["start"]) and _is_int(ev["end"]) and 0 <= ev["start"] <= ev["end"]):
        errs.append(f"evidence span must be ints 0 <= start <= end, got {ev['start']!r}..{ev['end']!r}")
    if not isinstance(ev["quote"], str):
        errs.append("evidence.quote must be a string")
    return errs


def _check_evidence_list(v: Any, schema: "Schema | None" = None) -> list[str]:
    if not isinstance(v, list):
        return [f"evidence_all must be a list of evidence, got {type(v).__name__}"]
    return [f"evidence_all[{i}]: {m}" for i, ev in enumerate(v) for m in _check_evidence(ev)]


def _check_date(v: Any, schema: "Schema | None" = None) -> list[str]:
    if not isinstance(v, str) or not _DATE_RE.match(v):
        return [f"date must be an ISO YYYY-MM-DD string, got {v!r}"]
    try:
        _dt.date.fromisoformat(v)
    except ValueError:
        return [f"date is not a valid calendar date: {v!r}"]
    return []


def _check_op(v: Any, schema: "Schema | None" = None) -> list[str]:
    ops = schema.amend_ops if schema is not None else frozenset()
    return [] if isinstance(v, str) and v in ops else [f"op {v!r} not in amend_ops {sorted(ops)}"]


def _check_ops(v: Any, schema: "Schema | None" = None) -> list[str]:
    """Merged AMENDS operations: a list of amend_ops, index-aligned with evidence_all (Task 5, R12)."""
    if not isinstance(v, list) or not v:
        return [f"ops must be a non-empty list of amend_ops, got {v!r}"]
    return [m for x in v for m in _check_op(x, schema)]


def _check_relation(v: Any, schema: "Schema | None" = None) -> list[str]:
    """REFERS_TO relation class: a string; one of graph.yaml ref_relations when the schema lists them."""
    if not isinstance(v, str):
        return [f"relation must be str, got {type(v).__name__}"]
    rels = schema.ref_relations if schema is not None else frozenset()
    return [] if not rels or v in rels else [f"relation {v!r} not in ref_relations {sorted(rels)}"]


def _check_weight(v: Any, schema: "Schema | None" = None) -> list[str]:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
        return []
    return [f"weight must be a float in [0, 1], got {v!r}"]


def _check_type(name: str, t: type):
    def check(v: Any, schema: "Schema | None" = None) -> list[str]:
        return [] if isinstance(v, t) else [f"{name} must be {t.__name__}, got {type(v).__name__}"]
    return check


# Every edge attr name graph.yaml may use; an attr present on an edge must be declared (required /
# optional) for its edge type in graph.yaml (Ruling R26 (4)) and is type-checked whenever its name is
# known here. Names without a checker are rejected, both in graph.yaml (Schema.__init__) and on edges.
_ATTR_CHECKERS = {
    "evidence": _check_evidence,
    "evidence_all": _check_evidence_list,
    "date": _check_date,
    "op": _check_op,
    "ops": _check_ops,
    "weight": _check_weight,
    "relation": _check_relation,
    "attached": _check_type("attached", bool),
}


_SPEC_KEYS = frozenset({"id", "label", "src", "dst", "required", "optional"})


@dataclass(frozen=True)
class EdgeSpec:
    id: int
    label: str
    src: frozenset
    dst: frozenset
    required: tuple
    optional: tuple


class Schema:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.version = cfg.get("version")
        kinds = cfg["vertex_kinds"]
        if not isinstance(kinds, list) or not all(isinstance(k, str) and k for k in kinds):
            raise ValueError(f"graph.yaml: vertex_kinds must be a list of non-empty strings, got {kinds!r}")
        self.vertex_kinds: frozenset[str] = frozenset(kinds)
        vt = cfg.get("vertex_types") or {}
        if not isinstance(vt, dict):
            raise ValueError(f"graph.yaml: vertex_types must be a mapping kind -> [types], got {vt!r}")
        for k, v in vt.items():          # a bare string would silently become a set of characters
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                raise ValueError(f"graph.yaml: vertex_types.{k} must be a list of non-empty strings, got {v!r}")
        self.vertex_types: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in vt.items()}
        bad = set(self.vertex_types) - self.vertex_kinds
        if bad:
            raise ValueError(f"graph.yaml: vertex_types for unknown kinds {sorted(bad)}")
        self.amend_ops: frozenset[str] = frozenset(cfg.get("amend_ops") or [])
        self.ref_relations: frozenset[str] = frozenset(cfg.get("ref_relations") or [])
        self.specs: dict[str, EdgeSpec] = {}
        for t, spec in cfg["edge_types"].items():
            if not isinstance(spec, dict):
                raise ValueError(f"graph.yaml: edge {t} spec must be a mapping, got {spec!r}")
            unknown_keys = sorted(set(spec) - _SPEC_KEYS)
            if unknown_keys:             # e.g. the pre-Task-2 `attrs:` key: never silently ignored
                raise ValueError(f"graph.yaml: edge {t} has unknown spec keys {unknown_keys} "
                                 f"(allowed {sorted(_SPEC_KEYS)})")
            for key in ("src", "dst", "required", "optional"):
                v = spec.get(key)
                if v is not None and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                    raise ValueError(f"graph.yaml: edge {t} {key} must be a list of strings, got {v!r}")
            src, dst = frozenset(spec["src"]), frozenset(spec["dst"])
            unknown = (src | dst) - self.vertex_kinds
            if unknown:
                raise ValueError(f"graph.yaml: edge {t} uses unknown vertex kinds {sorted(unknown)}")
            req, opt = tuple(spec.get("required") or ()), tuple(spec.get("optional") or ())
            nochk = [a for a in req + opt if a not in _ATTR_CHECKERS]
            if nochk:
                raise ValueError(f"graph.yaml: edge {t} attrs without a checker: {nochk}")
            if set(req) & set(opt):
                raise ValueError(f"graph.yaml: edge {t} attrs both required and optional")
            if not _is_int(spec.get("id")):
                raise ValueError(f"graph.yaml: edge {t} needs an integer id")
            self.specs[t] = EdgeSpec(spec["id"], spec.get("label", t), src, dst, req, opt)
        ids = sorted(s.id for s in self.specs.values())
        if ids != list(range(len(ids))):
            raise ValueError(f"graph.yaml: edge ids must be unique and exactly 0..{len(ids) - 1}, got {ids}")
        # canonical on-disk order and edge_type id (PyG, checkpoints): sorted by the pinned yaml id
        self.edge_types: list[str] = sorted(self.specs, key=lambda t: self.specs[t].id)
        self.edge_type_id: dict[str, int] = {t: self.specs[t].id for t in self.edge_types}
        self.labels: dict[str, str] = {t: self.specs[t].label for t in self.edge_types}

    @property
    def entity_types(self) -> frozenset[str]:
        return self.vertex_types.get("entity", frozenset())

    @classmethod
    def load(cls, cfg: dict | None = None) -> "Schema":
        """config/graph.yaml (or an explicit / `config.overridden` dict)."""
        return cls(cfg if cfg is not None else config.load("graph"))

    def meta(self) -> dict:
        """What an artifact / checkpoint records to detect a schema mismatch."""
        return {"version": self.version, "edge_types": list(self.edge_types),
                "vertex_kinds": sorted(self.vertex_kinds)}

    def validate_node(self, n: GraphNode) -> list[str]:
        errs = []
        if not isinstance(n.node_id, str) or not n.node_id:
            errs.append("node_id must be a non-empty string")
        if n.kind not in self.vertex_kinds:
            errs.append(f"{n.node_id}: unknown vertex kind {n.kind!r}")
        if not isinstance(n.type, str) or not n.type:
            errs.append(f"{n.node_id}: node without type")
        elif n.kind in self.vertex_types and n.type not in self.vertex_types[n.kind]:
            errs.append(f"{n.node_id}: type {n.type!r} not allowed for kind {n.kind!r} "
                        f"(expected {sorted(self.vertex_types[n.kind])})")
        if n.doc_id is not None and not isinstance(n.doc_id, str):
            errs.append(f"{n.node_id}: doc_id must be a string or None")
        if not isinstance(n.external, bool):
            errs.append(f"{n.node_id}: external must be a bool, got {type(n.external).__name__}")
        if not isinstance(n.attrs, dict):
            errs.append(f"{n.node_id}: attrs must be a dict")
        return errs

    def validate_edge(self, e: GraphEdge, nodes: dict[str, GraphNode]) -> list[str]:
        """Human-readable problems with `e` (empty = valid); O(1) given the node_id -> node dict.
        Evidence may be given as an Evidence instance or its to_json() dict."""
        tag = f"{e.type} {e.src} -> {e.dst}"
        spec = self.specs.get(e.type)
        if spec is None:
            return [f"{tag}: unknown edge type {e.type!r}"]
        errs = []
        if e.src == e.dst:
            errs.append(f"{tag}: self-loop")
        for end, nid, allowed in (("src", e.src, spec.src), ("dst", e.dst, spec.dst)):
            n = nodes.get(nid)
            if n is None:
                errs.append(f"{tag}: {end} node {nid!r} not in graph")
            elif n.kind not in allowed:
                errs.append(f"{tag}: {end} kind {n.kind!r} not allowed (expected {sorted(allowed)})")
        if not isinstance(e.attrs, dict):
            return errs + [f"{tag}: attrs must be a dict"]
        for a in spec.required:
            if e.attrs.get(a) is None:
                errs.append(f"{tag}: missing required attr {a!r}")
        declared = set(spec.required) | set(spec.optional)
        for a, v in e.attrs.items():
            check = _ATTR_CHECKERS.get(a)
            if check is None:
                errs.append(f"{tag}: unknown attr {a!r}")
                continue
            if a not in declared:        # Ruling R26 (4): only the attrs graph.yaml declares for this edge type
                errs.append(f"{tag}: undeclared attr {a!r} for {e.type} (declared {sorted(declared)})")
            if v is not None:
                errs += [f"{tag}: {m}" for m in check(v, self)]
        return errs


EDGE_TYPES: list[str] = Schema.load().edge_types
"""Canonical on-disk order of edge types (index = edge_type id, pinned by `id` in graph.yaml).
Import-time snapshot of config/graph.yaml; code that may run under `config.overridden` should use
`schema.edge_types` / `schema.edge_type_id` of its own Schema instance."""
