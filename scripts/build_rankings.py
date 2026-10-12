"""U9 rankings. Task 11: --family qreal (every Q-real query, both scripts) -> data/rankings/qreal/;
Task 19: --family dataset (every dataset candidate, --candidates FILE) -> data/rankings/dataset/ (the pilot of Task 17
uses --candidates data/dataset/pilot/candidates.jsonl --out data/rankings/pilot); M7 Task 7: --family probe (the probe
HardItems, split "probe", --candidates <snap>/pairs.jsonl; qkey "<qid>|q") -- the same path and settings otherwise;
M7 pilot 7: --family hard (the M7 Q1 / Q2 train / val rows as §8.1 Pair rows, grl.hard.ship.read_hard_pairs;
qkey "<id>|q") -- the same path and settings. M7 Task 9f: --family hard_test (the hard TEST rows, read only by
scripts/m7_final_eval.py open behind the once-token; grl.hard.ship.read_hard_test_pairs; report beside --out).
    python scripts/build_rankings.py --family qreal|dataset|probe|hard|hard_test [--candidates F] [--out DIR] [--qdrant] [--eval]"""
import argparse
import hashlib
import json
import sys

import numpy as np

from graphrag import config
from graphrag.data.embed import Encoder
from graphrag.env.data import load_env_data
from graphrag.eval.gold import read_gold
from graphrag.kg.artifact import read_graph
from graphrag.kg.build import files_digest
from graphrag.retrieval.rankings import (TOP, TitleIndex, build_rankings, load_rankings, qdrant_agreement,
                                         queries_from_gold, queries_from_pairs, rrf_rows, write_rankings)
from graphrag.stores import pg


def _sha(p) -> str:
    return hashlib.sha256(config.path(p).read_bytes()).hexdigest()


def evaluate(ranks, data, gold: dict[str, set]) -> dict:
    """R@k / MRR@10 by system and query family (cyr, lat: Q-real; q: dataset)."""
    out = {}
    for fam in ("cyr", "lat", "q"):
        keys = [k for k in ranks.keys() if k.endswith(f"|{fam}")]
        for name in ("bm25", "dense", "rrf"):
            hits = {1: [], 5: [], 10: [], 50: []}
            mrr = []
            for k in keys:
                r = ranks.get(k)
                rows = r.lex if name == "bm25" else r.vec if name == "dense" else rrf_rows(r.lex, r.vec)
                g = gold[k.split("|")[0]]
                rank = next((i + 1 for i, c in enumerate(rows[:100].tolist()) if data.chunk_ids[c] in g), None)
                for kk in hits:
                    hits[kk].append(float(rank is not None and rank <= kk))
                mrr.append(1.0 / rank if rank and rank <= 10 else 0.0)
            if keys:
                out[f"{name}|{fam}"] = {**{f"R@{kk}": round(float(np.mean(v)), 4) for kk, v in hits.items()},
                                        "MRR@10": round(float(np.mean(mrr)), 4), "n": len(keys)}
    return out


def family_queries(family: str, candidates) -> list[tuple[str, str]]:
    """M7 probe queries ("<qid>|q", question) of a probe pairs file (grl.hard.probe.read_probe_rows guards it);
    family hard: ("<id>|q", q) of a hard Pair file (grl.hard.ship.read_hard_pairs guards it)."""
    if family == "hard":
        from grl.hard.ship import read_hard_pairs
        return sorted(queries_from_pairs(read_hard_pairs(config.path(str(candidates)))))
    if family == "hard_test":                          # Task 9f / 10: the one final test opening only
        from grl.hard.ship import read_hard_test_pairs
        return sorted(queries_from_pairs(read_hard_test_pairs(config.path(str(candidates)))))
    if family != "probe":
        raise ValueError(f"family_queries: {family!r} is built in main()")
    from grl.hard.probe import probe_queries, read_probe_rows
    return probe_queries(read_probe_rows(config.path(str(candidates))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qreal", "dataset", "probe", "hard", "hard_test"], required=True)
    ap.add_argument("--candidates", default="data/dataset/candidates.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--qdrant", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    tcfg = config.load("train")
    data = load_env_data()
    real = read_gold(config.path(tcfg["gold"]["file"]))
    pairs = []
    if args.family == "dataset":                       # graphrag.dataset exists from Task 14 on
        from graphrag.dataset.finalize import read_pairs
        pairs = read_pairs(config.path(args.candidates))
    if args.family in ("probe", "hard", "hard_test"):
        if not args.out:
            ap.error(f"--family {args.family} needs --out (the snapshot dir; never data/rankings/)")
        if args.eval:
            ap.error("--eval has no M7 metrics (STRUCT gold is in no index); see scripts/hard_probe.py")
        queries = family_queries(args.family, args.candidates)
    else:
        queries = sorted(queries_from_gold(real)) if args.family == "qreal" else sorted(queries_from_pairs(pairs))
    root = config.path(args.out or f"data/rankings/{args.family}")
    nodes, _ = read_graph(config.path(f"data/graph/{data.version}"))
    assert [n.node_id for n in nodes] == data.node_ids
    bm = config.load("data")["bm25"]
    titles = TitleIndex([n.title for n in nodes], float(bm["k1"]), float(bm["b"]))
    enc = Encoder(device="cpu")
    conn = pg.connect()
    try:
        arrays = build_rankings(queries, data, conn, enc, titles, TOP)
    finally:
        conn.close()
    d = config.load("data")
    inputs = {"graph_version": data.version, "top": TOP, "bm25": bm,
              "records_manifest": _sha(f"{d['records_dir']}/manifest.jsonl"), "lexicon": _sha(d["lexicon"]["path"]),
              "encoder": {"name": enc.model_name, "revision": config.load("model")["encoder"].get("revision"), "device": "cpu"},
              "family": args.family, "gold": _sha(tcfg["gold"]["file"]),
              "candidates": _sha(args.candidates) if args.family in ("dataset", "probe", "hard", "hard_test") else None,
              "code": files_digest(config.ROOT, ("graphrag/retrieval/*.py", "graphrag/data/normalize.py",
                                                 "graphrag/data/lemma.py", "graphrag/data/embed.py"))}
    ver = write_rankings(root, queries, arrays, inputs)
    rep = {"version": ver, "n_queries": len(queries)}
    if args.qdrant:
        from graphrag.retrieval import vector
        rep["qdrant"] = qdrant_agreement(vector.client(), vector.collection_name(enc.slug), arrays, data.chunk_ids)
    if args.eval:
        gold = {i.qid: set(i.core) for i in real} | {p.id: set(p.C_ref) for p in pairs}
        rep["metrics"] = evaluate(load_rankings(root / "latest.json", graph_version=data.version), data, gold)
    out = (root.parent / f"rankings_report_{args.family}.json" if args.family == "hard_test"   # 9f: beside it
           else config.path(f"data/m3/rankings_report_{args.family}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
