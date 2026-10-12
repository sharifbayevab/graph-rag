"""Corpus → document records (spec §2.1), deterministic.

Walks the configured sections in NFC-path order, extracts and normalizes paragraphs,
parses structure, chunks, and fills metadata via the vendored legacy parser. Files with
no text are skipped ("empty"); byte-identical texts keep the first file and list the rest
as aliases. `validate_record` implements the stage checks of spec §2.6.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Callable

from graphrag import config
from graphrag.data.chunker import chunk_document, is_unit
from graphrag.data import extract
from graphrag.data.extract import read_paragraphs
from graphrag.data.legacy.parse import parse_document
from graphrag.data.lemma import build_vocab
from graphrag.data.mapping import mu_inverse
from graphrag.data.normalize import normalize_paragraphs, to_internal
from graphrag.data.structure import parse_structure
from graphrag.data.titles import fix_title
from graphrag.data.tokens import TokenCounter
from graphrag.types import DocRecord, Meta, Source


def doc_id_for(section: str, relpath: str) -> str:
    return f"{section}-{hashlib.sha1(relpath.encode('utf-8')).hexdigest()[:8]}"


def iter_sources(cfg: dict) -> list[tuple[str, Path, str]]:
    root = config.path(cfg["corpus_root"])
    out = []
    for sec in cfg["sections"]:
        base = root / sec
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in cfg["extensions"]:
                continue
            if any(part in cfg["exclude_dirs"] for part in p.relative_to(root).parts):
                continue
            rel = unicodedata.normalize("NFC", p.relative_to(root.parent).as_posix())
            out.append((sec, p, rel))
    return sorted(out, key=lambda x: x[2])


_AGENDA: dict[str, object] = {}


def agenda_segmenter(cfg: dict):
    """One AgendaSegmenter per (model, cache dir) and process, so its API budget covers the run;
    None when `agenda` is absent or disabled. API calls only with `agenda.api: true` or
    GRL_AGENDA_API=1 (otherwise cached answers only, regex fallback on a miss)."""
    a = cfg.get("agenda") or {}
    if not a.get("enabled"):
        return None
    key = f"{a['model']}|{a['cache_dir']}"
    if key not in _AGENDA:
        from graphrag.data.agenda import AgendaSegmenter
        allow = bool(a.get("api")) or os.environ.get("GRL_AGENDA_API") == "1"
        _AGENDA[key] = AgendaSegmenter(a["model"], config.path(a["cache_dir"]),
                                       max_calls=int(a.get("max_calls_per_run", 60)), allow_api=allow)
    return _AGENDA[key]


BUILD_INFO: dict[str, dict] = {}    # doc_id → {"engine": ..., "agenda": "llm"|"regex"} of the last build


def build_record(section: str, path: Path, relpath: str, cfg: dict,
                 count: Callable[[str], int]) -> DocRecord | None:
    paras = normalize_paragraphs(read_paragraphs(path))
    if not paras:
        return None
    doc_id = doc_id_for(section, relpath)
    info = {}
    if str(path) in extract.PDF_ENGINE:
        info["engine"] = extract.PDF_ENGINE[str(path)]
    seg = agenda_segmenter(cfg)
    seen = (seg.stats["docs"], seg.stats["used"]) if seg is not None else None
    pd = parse_document(relpath, section, "\n\n".join(p.text for p in paras),
                        unicodedata.normalize("NFC", path.name))   # NFD on macOS otherwise
    lang = pd.language
    title = fix_title(pd.title_raw, [p.text for p in paras], section, relpath)
    dt, nodes = parse_structure(doc_id, section, pd.doc_type, paras, title or "",
                                cfg["max_tokens"], count, agenda_segmenter(cfg))
    chunks = chunk_document(dt, nodes, cfg["max_tokens"], count, lambda s: to_internal(s, lang))
    if not chunks:
        return None
    if seg is not None and seg.stats["docs"] > seen[0]:
        info["agenda"] = "llm" if seg.stats["used"] > seen[1] else "regex"
    BUILD_INFO[doc_id] = info
    meta = Meta(type=pd.doc_type, date=pd.doc_date.isoformat() if pd.doc_date else None,
                organ=pd.issuing_body, lang=[lang], status="проект" if pd.is_draft else "принят",
                section=section, title=title)
    pages = [q for p in paras for q in (p.page, p.page_end) if q is not None]
    span = f"{min(pages)}-{max(pages)}" if pages else None
    return DocRecord(doc_id, meta, Source("docs", relpath, span, []), dt.text, nodes, chunks)


_DL_SUFFIX = re.compile(r"\s*\(\d{1,2}\)$")                 # "(1)".."(99)", never a year "(2020)"


def _has_download_suffix(relpath: str) -> bool:
    """'3. Қонун матни (1).docx' / 'x(2).pdf' — a browser's duplicate-download copy."""
    return bool(_DL_SUFFIX.search(PurePosixPath(relpath).stem))


def build_corpus(cfg: dict | None = None, count: Callable[[str], int] | None = None):
    """Records in NFC-relpath order. Among byte-identical texts the survivor is the file without
    a trailing "(N)" download suffix, then the first by relpath; the rest become its aliases."""
    cfg = cfg or config.load("data")
    count = count or TokenCounter(cfg["tokenizer"], cfg.get("tokenizer_revision"))
    BUILD_INFO.clear()
    extract.PDF_ENGINE.clear()
    built, skipped, groups = [], [], {}
    for sec, p, rel in iter_sources(cfg):
        try:
            rec = build_record(sec, p, rel, cfg, count)
        except Exception as e:                      # keep building; reported by build_corpus.py
            skipped.append({"file": rel, "reason": f"error: {e!r}"})
            continue
        if rec is None:
            skipped.append({"file": rel, "reason": "empty"})
            continue
        key = hashlib.sha256(rec.text.encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(rec)
        built.append(rec)
    survivors = set()
    for recs in groups.values():
        best = min(recs, key=lambda r: (_has_download_suffix(r.source.file), r.source.file))
        survivors.add(id(best))
        for r in recs:                              # recs are in relpath order
            if r is not best:
                best.source.aliases.append(r.source.file)
                skipped.append({"file": r.source.file, "reason": f"duplicate of {best.source.file}"})
    records = [r for r in built if id(r) in survivors]
    return records, skipped


def _unit_coverage(rec: DocRecord, n, chunks) -> str:
    """Text of unit `n` covered by chunks that lie inside it or contain it whole (a merged
    stenogram chunk spans several выступление units), restricted to the unit span."""
    parts = []
    for c in chunks:
        if (n.start <= c.start and c.end <= n.end) or (c.start <= n.start and n.end <= c.end):
            a, b = max(c.start, n.start), min(c.end, n.end)
            parts.append(c.text[a - c.start:b - c.start])
    return "".join(parts)


def validate_record(rec: DocRecord, max_tokens: int) -> list[str]:
    errs = []
    by_id = {n.node_id: n for n in rec.structure}
    if len(by_id) != len(rec.structure):
        errs.append("duplicate node ids")
    if len({c.chunk_id for c in rec.chunks}) != len(rec.chunks):
        errs.append("duplicate chunk ids")
    kids: dict[str, list] = {}
    for n in rec.structure:
        if n.parent_id:
            kids.setdefault(n.parent_id, []).append(n)
    chunks = sorted(rec.chunks, key=lambda c: c.start)
    prev = -1
    for c in chunks:
        if c.parent not in by_id:
            errs.append(f"{c.chunk_id}: parent missing")
            continue
        if rec.text[c.start:c.end] != c.text:
            errs.append(f"{c.chunk_id}: not an exact substring")
        if c.start < prev:
            errs.append(f"{c.chunk_id}: overlaps previous chunk")
        prev = c.end
        p = by_id[c.parent]
        if not (p.start <= c.start and c.end <= p.end):
            errs.append(f"{c.chunk_id}: outside parent span")
        elif any(k.start <= c.start and c.end <= k.end for k in kids.get(c.parent, [])):
            errs.append(f"{c.chunk_id}: parent is not the deepest node containing the chunk")
        if c.tokens > max_tokens:
            errs.append(f"{c.chunk_id}: {c.tokens} tokens > {max_tokens}")
        if not c.index_text or not c.index_text.endswith(c.text_norm):
            errs.append(f"{c.chunk_id}: index_text empty or does not end with text_norm")
    inv = mu_inverse(rec.structure, rec.chunks)
    for n in rec.structure:
        if not inv[n.node_id]:
            errs.append(f"{n.node_id}: no chunks in μ⁻¹")
        if is_unit(n, by_id):
            inside = _unit_coverage(rec, n, chunks)
            if "".join(inside.split()) != "".join(rec.text[n.start:n.end].split()):
                errs.append(f"{n.node_id}: unit text not fully covered by its chunks")
    return errs


def write_records(records: list[DocRecord], out_dir: Path, info: dict[str, dict] | None = None) -> None:
    """Record JSONs + manifest.jsonl; `info` (doc_id → e.g. {"engine", "agenda"}) is added to the
    manifest lines only, never to the records."""
    info = info or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {f"{r.doc_id}.json" for r in records}
    for f in out_dir.glob("*.json"):
        if f.name not in keep:
            f.unlink()
    for r in records:
        (out_dir / f"{r.doc_id}.json").write_text(r.to_json(), encoding="utf-8")
    manifest = [{"doc_id": r.doc_id, "file": r.source.file, "n_nodes": len(r.structure),
                 "n_chunks": len(r.chunks), **info.get(r.doc_id, {})} for r in sorted(records, key=lambda r: r.doc_id)]
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False, sort_keys=True) + "\n" for m in manifest),
        encoding="utf-8")


def read_records(out_dir: Path) -> list[DocRecord]:
    recs = [DocRecord.from_json(f.read_text(encoding="utf-8")) for f in sorted(out_dir.glob("*.json"))]
    return sorted(recs, key=lambda r: r.source.file)


def write_lexicon(records: list[DocRecord], path: Path) -> None:
    vocab = build_vocab(c.text_norm for r in records for c in r.chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True, indent=0) + "\n",
                    encoding="utf-8")
