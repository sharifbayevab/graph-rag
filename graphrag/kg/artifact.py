"""Deterministic on-disk graph artifact: nodes.jsonl + edges.jsonl + schema.json (+ extra files).

Same graph in, same bytes out: nodes sorted by node_id, edges by (src, type, dst, attrs json);
every line is json.dumps(ensure_ascii=False, sort_keys=True). Each file is written to a .tmp
sibling and moved into place with os.replace, so a reader never sees a half-written file.
schema.json records the schema meta (edge_types in id order) so a reader / checkpoint can detect
a mismatch with the current graph.yaml."""
import json
import os
from pathlib import Path

from graphrag.kg.schema import GraphEdge, GraphNode, Schema

NODES_FILE = "nodes.jsonl"
EDGES_FILE = "edges.jsonl"
SCHEMA_FILE = "schema.json"
_RESERVED = (NODES_FILE, EDGES_FILE, SCHEMA_FILE)


class SchemaMismatch(ValueError):
    pass


class ArtifactConflict(ValueError):
    """write_graph(overwrite=False) into a directory that already holds different bytes for one of the files
    (Ruling R26 (2): a version directory is immutable -- same version, same content)."""


def _default(o):
    if hasattr(o, "to_json"):          # Evidence instances inside attrs
        return o.to_json()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_default)


def node_key(n: GraphNode) -> str:
    return n.node_id


def edge_sort_key(e: GraphEdge) -> tuple[str, str, str, str]:
    """The canonical on-disk ORDER of edges: (src, type, dst, attrs json). Not an identity (see edge_key)."""
    return (e.src, e.type, e.dst, _dumps(e.attrs))


def edge_key(e: GraphEdge) -> tuple:
    """THE edge key (Ruling R31 (2): defined once) = the artifact dedup key, what makes an edge unique in a
    graph: (src, type, dst), plus attrs.relation for REFERS_TO (Ruling R24 (2): one REFERS_TO edge per
    relation class, e.g. an amendment head `amended` and a `cited_in_amendment` citation inside its quoted
    new text may join the same pair). The Neo4j r.edge_key is its JSON (store.edge_key_json)."""
    if e.type == "REFERS_TO":
        return (e.src, e.type, e.dst, str(e.attrs.get("relation") or ""))
    return (e.src, e.type, e.dst)


def canonical(nodes, edges) -> tuple[list[GraphNode], list[GraphEdge]]:
    return sorted(nodes, key=node_key), sorted(edges, key=edge_sort_key)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _jsonl(records) -> bytes:
    return "".join(_dumps(r.to_json()) + "\n" for r in records).encode("utf-8")


def write_graph(dir, nodes, edges, extra_files: dict[str, bytes] | None = None,
                schema: Schema | None = None, overwrite: bool = True) -> Path:
    """Write the graph to `dir` (created if needed); returns the directory. `extra_files` maps a
    plain file name (no directories) to its bytes, e.g. stats.json. Duplicate node ids and
    duplicate (src, type, dst) edges are errors: dedup / evidence merging is the builder's job.
    overwrite=False: a file already in `dir` whose bytes differ from the new ones raises ArtifactConflict
    (nothing is written); identical files are left as they are."""
    schema = schema or Schema.load()
    d = Path(dir)
    nodes, edges = canonical(nodes, edges)
    ids = [n.node_id for n in nodes]
    dup = next((a for a, b in zip(ids, ids[1:]) if a == b), None)
    if dup is not None:
        raise ValueError(f"duplicate node_id {dup!r}")
    keys = sorted(edge_key(e) for e in edges)
    dup_e = next((a for a, b in zip(keys, keys[1:]) if a == b), None)   # sorted: dups adjacent
    if dup_e is not None:
        raise ValueError(f"duplicate edge (src, type, dst[, relation]) {dup_e!r}")
    for name in extra_files or {}:
        if Path(name).name != name or name in _RESERVED or name.endswith(".tmp"):
            raise ValueError(f"bad extra file name {name!r}")
    files = {NODES_FILE: _jsonl(nodes), EDGES_FILE: _jsonl(edges),
             SCHEMA_FILE: (json.dumps(schema.meta(), ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8"),
             **dict(extra_files or {})}
    if not overwrite:
        differ = sorted(n for n, data in files.items() if (d / n).exists() and (d / n).read_bytes() != data)
        if differ:
            raise ArtifactConflict(f"{d} already holds different {differ}: same version, different content "
                                   f"(a version input is missing from the digest, or the build is not deterministic)")
    d.mkdir(parents=True, exist_ok=True)          # only after every check: a rejected write leaves nothing
    for name in (NODES_FILE, EDGES_FILE, SCHEMA_FILE, *sorted(extra_files or {})):
        _atomic_write(d / name, files[name])
    return d


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_schema_meta(dir) -> dict | None:
    p = Path(dir) / SCHEMA_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_graph(dir, schema: Schema | None = None,
               check_schema: bool = True) -> tuple[list[GraphNode], list[GraphEdge]]:
    """(nodes, edges) from `dir`, in canonical order. With check_schema, a missing schema.json or one
    whose version / edge_types / vertex_kinds differ from `schema` (default: current graph.yaml) raises
    SchemaMismatch."""
    d = Path(dir)
    if check_schema:
        meta = read_schema_meta(d)
        if meta is None:
            raise SchemaMismatch(f"{d}: no {SCHEMA_FILE} (pass check_schema=False to read it anyway)")
        want = (schema or Schema.load()).meta()
        diff = sorted(k for k in want if meta.get(k) != want[k])
        if diff:
            raise SchemaMismatch(f"{d}: artifact schema differs in {diff}: "
                                 + "; ".join(f"{k} {meta.get(k)!r} != {want[k]!r}" for k in diff))
    nodes = [GraphNode.from_json(r) for r in _read_jsonl(d / NODES_FILE)]
    edges = [GraphEdge.from_json(r) for r in _read_jsonl(d / EDGES_FILE)]
    return canonical(nodes, edges)
