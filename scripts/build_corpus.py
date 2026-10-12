"""Build document records + Uzbek lexicon from ../docs; print stats. Records and lexicon are
written only when every record validates and no file failed; otherwise exit 1, outputs untouched.

The build must be reproducible offline: it also exits 1 (outputs untouched) when it was degraded,
i.e. an agenda answer was not cached (regex fallback), a vision answer was not cached, or a PDF
fell back to pypdf, unless --allow-degraded is given. With API calls off (the default:
`agenda.api` / `pdf.vision.api` false, no GRL_AGENDA_API / GRL_VISION_API) a vision miss fails
its file; with --allow-degraded it is dropped instead. The extraction engine per PDF and the
agenda mode per stenogram (llm | regex) are recorded in manifest.jsonl.

--set KEY=VALUE overrides config/data.yaml for this run (dotted key, YAML value), e.g.
  --set records_dir=/tmp/rec --set lexicon.path=/tmp/lex.json --set pdf.vision.cache_dir=/tmp/empty"""
import argparse
import sys
from collections import Counter

import yaml

from graphrag import config
from graphrag.data import extract
from graphrag.data.ingest import (BUILD_INFO, agenda_segmenter, build_corpus, validate_record,
                                  write_lexicon, write_records)


def apply_overrides(cfg: dict, pairs: list[str]) -> None:
    """In place on `cfg` (a copy: config.load returns a fresh dict); main() makes it the process-wide
    "data" config for the run only (config.overridden), so the cached config is never mutated."""
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        *parents, leaf = key.split(".")
        d = cfg
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)


def _misses(cfg: dict) -> tuple[int, int]:
    """(agenda, vision) offline-miss counters of the process-wide segmenter / transcriber."""
    seg, vision = agenda_segmenter(cfg), extract._VISION
    return (seg.stats["offline_miss"] if seg is not None else 0,
            vision.stats["offline_miss"] if vision is not None else 0)


def degradations(agenda_miss: int, vision_miss: int, engines: dict[str, str]) -> list[str]:
    out = []
    if agenda_miss:
        out.append(f"agenda: {agenda_miss} uncached answers (regex fallback used)")
    if vision_miss:
        out.append(f"vision: {vision_miss} uncached answers (dropped)")
    fb = sorted(p for p, e in engines.items() if e == "pypdf_fallback")
    if fb:
        out.append(f"pdf: {len(fb)} files fell back to pypdf (e.g. {fb[0]})")
    return out


def main(argv: list[str] = ()) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-degraded", action="store_true",
                    help="write even if cache misses or the pypdf fallback changed the records")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(list(argv))
    cfg = config.load("data")                        # a copy: overrides never leak into the cache
    apply_overrides(cfg, args.set)
    if args.allow_degraded:
        cfg["pdf"]["vision"]["on_miss"] = "skip"
    with config.overridden("data", cfg):             # deep readers (extract, pdf, chunker) see them
        return _run(cfg, args)


def _run(cfg: dict, args) -> int:
    base = _misses(cfg)                              # counters are per process: count this run only
    records, skipped = build_corpus(cfg)
    misses = [a - b for a, b in zip(_misses(cfg), base)]
    types = Counter(n.type for r in records for n in r.structure)
    per_sec = Counter(r.meta.section for r in records)
    n_chunks = sum(len(r.chunks) for r in records)
    print(f"records={len(records)} chunks={n_chunks} per_section={dict(per_sec)}")
    print(f"node types={dict(types)}")
    seg = agenda_segmenter(cfg)
    if seg is not None:
        print(f"agenda: {dict(sorted(seg.stats.items()))}")
    vision = extract._VISION
    if vision is not None:
        print(f"vision: {dict(sorted(vision.stats.items()))}")
    print(f"pdf engines: {dict(Counter(extract.PDF_ENGINE.values()))}")
    for s in skipped:
        print(f"skipped: {s['file']} — {s['reason']}")
    errors = {r.doc_id: validate_record(r, cfg["max_tokens"]) for r in records}
    bad = {k: v for k, v in errors.items() if v}
    for k, v in bad.items():
        print(f"INVALID {k}: {v[:5]}")
    hard = [s for s in skipped if s["reason"].startswith("error")]
    if bad or hard:
        print(f"NOT WRITTEN: {len(bad)} invalid records, {len(hard)} failed files")
        return 1
    degraded = degradations(*misses, extract.PDF_ENGINE)
    for d in degraded:
        print(f"DEGRADED {d}")
    if degraded and not args.allow_degraded:
        print("NOT WRITTEN: degraded, non-reproducible build (pass --allow-degraded to write anyway)")
        return 1
    write_records(records, config.path(cfg["records_dir"]), BUILD_INFO)
    write_lexicon(records, config.path(cfg["lexicon"]["path"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
