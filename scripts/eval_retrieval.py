"""Retrieval regression on Q-real (docs/m1/research-encoder-report.md §1): BM25, dense, RRF.

Q-real = the `type == "content"` items of ../eval/validation_set.jsonl (340), each asked twice
(question_cyr, question_lat). Gold = chunk(s) of the item's source document (source_files[0],
NFC; matched on record source.file / aliases, basename fallback) whose whitespace/case/quote-
normalized `text` contains supporting_quote; fallbacks, in order: a span over two adjacent chunks,
the same two tests on the internal (to_internal) form against text_norm, and a fuzzy match
(longest common block >= 60 % of the quote). Items without gold are reported and skipped.

Systems (live stores, read-only): BM25 = lexical.search_terms(lexical.query_terms(q)); dense =
vector.search_text (query_internal + encoder query prefix, Qdrant cosine); RRF(k=60) over the
top-100 of each (ties by corpus order). Metrics R@1/5/10/50 (first gold hit within rank k) and
MRR@10. Deterministic: no sampling; same stores + encoder give the same numbers.

Writes data/eval/retrieval_<collection>.json (or --out FILE) and prints a table.
    .venv/bin/python scripts/eval_retrieval.py [--collection NAME] [--limit N] [--out FILE]
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from graphrag import config
from graphrag.data.embed import Encoder
from graphrag.data.ingest import read_records
from graphrag.data.normalize import detect_language, to_internal
from graphrag.retrieval import lexical, vector
from graphrag.stores import pg

KS = (1, 5, 10, 50)
TOP, RRF_K = 100, 60
VALIDATION = config.ROOT.parent / "eval" / "validation_set.jsonl"
_APOS = re.compile(r"[‘’ʼʻ`'´]")
_QUOT = re.compile(r"[«»“”„\"]")


def wsn(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return " ".join(_APOS.sub("'", _QUOT.sub('"', s)).lower().split())


def build_qreal(records) -> tuple[list[str], list[dict], dict]:
    """(chunk ids in corpus order, Q-real items with gold chunk ids, gold-tier counts)."""
    chunks = [(r, c) for r in records for c in r.chunks]
    ids = [c.chunk_id for _, c in chunks]
    wtext = [wsn(c.text) for _, c in chunks]
    by_file: dict[str, list[int]] = {}
    for i, (r, c) in enumerate(chunks):
        for f in {unicodedata.normalize("NFC", x) for x in [r.source.file, *r.source.aliases]}:
            by_file.setdefault(f, []).append(i)
    for f in by_file:
        by_file[f].sort(key=lambda i: chunks[i][1].start)
    items = [json.loads(line) for line in open(VALIDATION, encoding="utf-8")]
    items = [x for x in items if x["type"] == "content"]
    out, tiers = [], {}
    for x in items:
        f = unicodedata.normalize("NFC", x["source_files"][0])
        idx = by_file.get(f)
        if idx is None:
            b = f.rsplit("/", 1)[-1]
            cand = [k for k in by_file if k.rsplit("/", 1)[-1] == b]
            idx = by_file[cand[0]] if len(cand) == 1 else None
        gold, tier = [], "no-file"
        if idx:
            q = wsn(x["supporting_quote"])
            gold, tier = [i for i in idx if q in wtext[i]], "exact"
            if not gold:
                gold, tier = next(([a, b] for a, b in zip(idx, idx[1:])
                                   if q in wtext[a] + " " + wtext[b]), []), "span2"
            if not gold:
                qi = wsn(to_internal(x["supporting_quote"], detect_language(x["supporting_quote"])))
                nt = [wsn(chunks[i][1].text_norm) for i in idx]
                gold, tier = [i for i, t in zip(idx, nt) if qi in t], "internal"
                if not gold:
                    gold, tier = next(([idx[k], idx[k + 1]] for k in range(len(idx) - 1)
                                       if qi in nt[k] + " " + nt[k + 1]), []), "internal-span2"
                if not gold:
                    best = (0, None)
                    for k, i in enumerate(idx):
                        m = SequenceMatcher(None, qi, nt[k], autojunk=False).find_longest_match(
                            0, len(qi), 0, len(nt[k]))
                        if m.size > best[0]:
                            best = (m.size, i)
                    if best[1] is not None and best[0] >= 0.6 * len(qi):
                        gold, tier = [best[1]], "fuzzy60"
            if not gold:
                tier = "unmapped"
        tiers[tier] = tiers.get(tier, 0) + 1
        if gold:
            out.append({"id": x["id"], "section": x["section"], "cyr": x["question_cyr"],
                        "lat": x.get("question_lat"), "gold": sorted(ids[i] for i in gold)})
    return ids, out, tiers


def rrf(a: list[str], b: list[str], order: dict[str, int]) -> list[str]:
    s: dict[str, float] = {}
    for lst in (a, b):
        for r, c in enumerate(lst[:TOP]):
            s[c] = s.get(c, 0.0) + 1.0 / (RRF_K + r + 1)
    return [c for c, _ in sorted(s.items(), key=lambda x: (-x[1], order[x[0]]))][:TOP]


def metrics(rankings: list[list[str]], golds: list[set[str]]) -> dict:
    rows = []
    for r, g in zip(rankings, golds):
        rank = next((i + 1 for i, c in enumerate(r[:TOP]) if c in g), None)
        rows.append([float(rank is not None and rank <= k) for k in KS]
                    + [1.0 / rank if rank and rank <= 10 else 0.0])
    m = np.mean(rows, 0) if rows else np.zeros(len(KS) + 1)
    return {**{f"R@{k}": round(float(v), 4) for k, v in zip(KS, m)},
            "MRR@10": round(float(m[-1]), 4), "n": len(rows)}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", help="Qdrant collection (default: the configured encoder's)")
    ap.add_argument("--limit", type=int, help="first N items only (smoke runs)")
    ap.add_argument("--out", help="result JSON (default data/eval/retrieval_<collection>.json); M3 comparisons "
                                  "write their own file instead of overwriting the baseline")
    return ap.parse_args(argv)


def out_path(args: argparse.Namespace, coll: str):
    from pathlib import Path
    return Path(args.out) if args.out else config.path(f"data/eval/retrieval_{coll}.json")


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    records = read_records(config.path(config.load("data")["records_dir"]))
    ids, qreal, tiers = build_qreal(records)
    order = {c: i for i, c in enumerate(ids)}
    if args.limit:
        qreal = qreal[:args.limit]
    print(f"Q-real: {len(qreal)} items with gold; tiers {tiers}", flush=True)
    enc = Encoder()
    coll = args.collection or vector.collection_name(enc.slug)
    cl, conn = vector.client(), pg.connect()
    n_pts = cl.count(coll, exact=True).count
    if n_pts != len(ids):
        print(f"WARNING: collection {coll} has {n_pts} points, records have {len(ids)} chunks")
    lat = {"bm25": [], "dense": []}
    results: dict = {}
    per_query: dict = {}
    for script in ("cyr", "lat"):
        items = [x for x in qreal if x[script]]
        golds = [set(x["gold"]) for x in items]
        runs = {"bm25": [], "dense": [], "rrf": []}
        for x in items:
            q = x[script]
            t = time.perf_counter()
            b = [c for c, _ in lexical.search_terms(conn, lexical.query_terms(q), TOP)]
            lat["bm25"].append(time.perf_counter() - t)
            t = time.perf_counter()
            d = [c for c, _ in vector.search_text(cl, coll, enc, q, TOP)]
            lat["dense"].append(time.perf_counter() - t)
            runs["bm25"].append(b)
            runs["dense"].append(d)
            runs["rrf"].append(rrf(b, d, order))
        for sysname, rk in runs.items():
            results[f"{sysname}|{script}"] = metrics(rk, golds)
        per_query[script] = {x["id"]: {s: next((i + 1 for i, c in enumerate(runs[s][j])
                                                if c in golds[j]), None) for s in runs}
                             for j, x in enumerate(items)}
        print(f"{script}: {len(items)} queries done ({time.time() - t0:.0f}s)", flush=True)
    conn.close()
    pct = {k: {"p50_ms": round(float(np.percentile(v, 50) * 1000), 1),
               "p95_ms": round(float(np.percentile(v, 95) * 1000), 1)} for k, v in lat.items()}
    out = {"collection": coll, "encoder": enc.model_name, "dim": enc.dim, "points": n_pts,
           "chunks": len(ids), "n_items": len(qreal), "gold_tiers": tiers, "top": TOP,
           "rrf_k": RRF_K, "metrics": results, "latency": pct,
           "gold_rank": per_query, "seconds": round(time.time() - t0, 1)}
    dst = out_path(args, coll)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\ncollection {coll} ({n_pts} points), encoder {enc.model_name}")
    print(f"| system | script | {' | '.join(f'R@{k}' for k in KS)} | MRR@10 | n |")
    print("|---|---|" + "---|" * (len(KS) + 2))
    for key, m in results.items():
        s, sc = key.split("|")
        print(f"| {s} | {sc} | " + " | ".join(f"{m[f'R@{k}']:.3f}" for k in KS)
              + f" | {m['MRR@10']:.3f} | {m['n']} |")
    print(f"latency {pct}; total {out['seconds']}s; wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
