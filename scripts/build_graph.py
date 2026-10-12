"""M2 Task 9: build the knowledge graph artifact (IMPLEMENTATION.md §3.2 completion, §3.4 validation; Ruling R24).

    python scripts/build_graph.py [--out data/graph] [--device mps|cpu] [--round N]

Offline only: the extractors read their LLM answers from cache/kg (a miss -> exit 1), RELATED are the Task 8
edges (graph.yaml build.related_edges, checked against related_stats.json), node vectors come from
data/graph/node_vecs.npy (bge-m3 is loaded only for texts missing from it, encode_passages(strict=True)).
Writes <out>/<version>/{nodes.jsonl, edges.jsonl, schema.json, x.npy, pyg.pt, stats.json} and <out>/latest.json;
version = sha256(records + manifest + config/*.yaml + cache digests + graphrag/**/*.py + this script + library
versions)[:12] (Ruling R26 (2)); an existing <out>/<version>/ with different bytes is an error (exit 1).
Exit 1 when §3.4 validation reports any violation or an LLM answer is missing from the cache."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the encoder / tokenizer come from the local HF cache
os.environ.pop("GRL_KG_API", None)                    # never an LLM call from a graph build

import numpy as np  # noqa: E402

from graphrag import config  # noqa: E402


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dir_digest(d: Path, pattern: str = "*") -> dict:
    files = sorted(p for p in d.rglob(pattern) if p.is_file()) if d.exists() else []
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(d).as_posix().encode("utf-8") + b"\t" + _sha_file(p).encode() + b"\n")
    return {"files": len(files), "sha256": h.hexdigest()}


def version_inputs(cfg: dict, root: Path = ROOT) -> dict:
    """Everything the artifact depends on (Ruling R26 (2)): the records, the extractor LLM caches, the Task 8
    vector-cache metadata and related_stats, the node-vector cache (R31 (1)), every graphrag/**/*.py + this script, config/*.yaml, and the key
    library versions."""
    from graphrag.kg import build as B
    b = cfg.get("build") or {}
    return {
        "records": dir_digest(config.path("data/records"), "*.json"),
        "records_manifest_sha256": _sha_file(config.path("data/records/manifest.jsonl")),
        "config": B.config_digest(root),
        "caches": {**{f"cache/kg/{p}": dir_digest(config.path(f"cache/kg/{p}")) for p in b.get("cache_passes") or []},
                   **{f"{b['vec_dir']}/{f}": _sha_file(config.path(f"{b['vec_dir']}/{f}"))
                      for f in ("chunk_vecs_ids.json", "chunk_vecs_topic_ids.json")},   # row ids + text / npy sha256
                   **node_vecs_digest(b),                                               # R31 (1): X's cache
                   b["related_stats"]: _sha_file(config.path(b["related_stats"]))},
        "code": B.code_digest(root),
        "libraries": B.library_versions(),
    }


def node_vecs_digest(b: dict) -> dict:
    """Ruling R31 (1): the node-vector cache X is read from (build.features.cache, e.g. data/graph/node_vecs.npy)
    and its key file (sha256 of each embedded text, row order) -- both by sha256, so X cannot change under
    one version name."""
    npy = config.path((b.get("features") or {}).get("cache", "data/graph/node_vecs.npy"))
    keys = npy.with_name(npy.stem + "_keys.json")
    rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)  # noqa: E731
    return {rel(p): (_sha_file(p) if p.exists() else None) for p in (npy, keys)}


def version_of(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


class LazyEncoder:
    """bge-m3 (graphrag.data.embed.Encoder) constructed on the first encode: a fully cached build never
    loads the model. The tokenizer (for the vertex-text token budget) is the model's own, at the pinned encoder
    revision (config/model.yaml encoder.revision; M3-R2)."""

    def __init__(self, device: str | None):
        from transformers import AutoTokenizer
        self.cfg = config.load("model")["encoder"]
        self.model_name = self.cfg["name"]
        self.dim = self.cfg["dim"]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.cfg.get("revision"))
        self._enc = None
        self.used_device = None

    def encode_passages(self, texts, strict=False, progress=None):
        if self._enc is None:
            from graphrag.data.embed import Encoder
            self._enc = Encoder(device=self.device)
            self.used_device = str(self._enc._m.device)
        return self._enc.encode_passages(texts, strict=strict, progress=True if progress is None else progress)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    from graphrag.data.ingest import read_records
    from graphrag.kg import build as B
    from graphrag.kg.features import VecCache
    from graphrag.kg.llm import KgCacheMiss
    from graphrag.kg.schema import Schema
    t0 = time.time()
    cfg = config.load("graph")
    b = cfg["build"]
    schema = Schema(cfg)
    out = config.path(opt("--out", b["out"]))
    inputs = version_inputs(cfg)
    version = version_of(inputs)
    recs = read_records(config.path("data/records"))
    try:
        ex = B.extract_all(recs, cfg)
    except KgCacheMiss as e:
        print(f"LLM cache miss (offline build): {e}", file=sys.stderr)
        return 1
    try:                                                  # Ruling R25: the one approved RELATED source
        related, rinfo = B.related_edges(recs, B.explicit_for_related(ex), cfg, config.path(b["vec_dir"]))
    except B.RelatedMismatch as e:
        print(f"RELATED mismatch: {e}", file=sys.stderr)
        return 1
    rinfo["related_stats"] = B.check_related_stats(related, config.path(b["related_stats"]))
    if rinfo["related_stats"]["problems"]:
        print(f"RELATED differs from related_stats.json: {rinfo['related_stats']['problems']}", file=sys.stderr)
        return 1
    misses = sum(v.get("offline_miss", 0) for s in ex.llm.values() if s for v in s.values())
    api = sum(v.get("api_calls", 0) for s in ex.llm.values() if s for v in s.values())
    t_ex = time.time() - t0
    enc = LazyEncoder(opt("--device"))
    fcfg = b.get("features") or {}
    cache = VecCache(config.path(fcfg.get("cache", "data/graph/node_vecs.npy")), enc.model_name)
    rnd = int(opt("--round")) if opt("--round") else None
    g = B.build_graph(recs, cfg, encoder=enc, related=related, extracted=ex, cache=cache, schema=schema,
                      round_decimals=rnd)
    t_build = time.time() - t0
    viol = g.stats["validation"]
    fi = g.feature_info or {}
    stats = {
        "version": version, "version_inputs": inputs,
        **{k: v for k, v in g.stats.items() if k not in ("extract_llm",)},
        "extract_llm_offline": ex.llm, "related_source": rinfo,
        "features": {"model": enc.model_name, "dim": fi.get("dim"), "texts": fi.get("texts"),
                     "unique_texts": fi.get("unique_texts"), "cutter": fi.get("cutter"),
                     "cache": fcfg.get("cache", "data/graph/node_vecs.npy"), "round_decimals": rnd,
                     "text_config": {k: v for k, v in fcfg.items() if k != "cache"},
                     "encode": "encode_passages(strict=True), unit norm, float32; cached by sha256(text), X read "
                               "from the cache in node order"},
    }
    d = out / version
    from graphrag.kg.artifact import ArtifactConflict
    try:
        shas = B.write_artifact(g, d, stats, schema)
    except ArtifactConflict as e:
        print(f"version directory conflict: {e}", file=sys.stderr)
        return 1
    (out / "latest.json").write_text(json.dumps({"version": version, "dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d),
                                                 "sha256": shas}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"version": version, "dir": str(d), "nodes": stats["nodes"], "edges": stats["edges"],
               "nodes_by_kind": stats["nodes_by_kind"], "edges_by_type": stats["edges_by_type"],
               "refers_to_by_relation": stats["refers_to_by_relation"], "amends_by_op": stats["amends_by_op"],
               "related": stats["related"], "isolated_attached": stats["isolated_attached"],
               "degree": {k: stats["degree"][k] for k in ("median", "mean", "p90", "p99", "max")},
               "x_shape": stats["pyg"]["x_shape"], "pyg_edges": stats["pyg"]["num_edges"],
               "validation": viol, "llm_offline": {"misses": misses, "api_calls": api},
               "features_run": {"encoded": fi.get("encoded"), "cache_rows_loaded": fi.get("cache_rows_loaded"),
                                "device": enc.used_device},
               "seconds": {"extract": round(t_ex, 1), "build": round(t_build, 1), "total": round(time.time() - t0, 1)},
               "sha256": shas}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if viol["violations"] or misses or api:
        print(f"FAILED: {viol['violations']} validation violations, {misses} LLM cache misses, {api} API calls",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
