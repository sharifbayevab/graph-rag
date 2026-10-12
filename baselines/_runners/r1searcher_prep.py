"""R1-Searcher baseline, step 1 (Mac, light): export the inputs the Spark runner needs.

  PYTHONPATH=. python scripts/ext_baselines/r1searcher_prep.py --out /path/to/bundle_dir

Writes
  corpus.jsonl  one row per BASE chunk (load_struct_env_data(articles=True).chunk_ids[:n_base_chunks], STRUCT rows
                excluded), in env row order: {id, title, text, tokens}. title = the chunk's breadcrumb (index_text
                minus text_norm), text = the original chunk text -- the KILT-style "title text" passage R1-Searcher reads.
  vecs.npy      the env chunk_vecs[:n_base_chunks] (bge-m3, 1024-d, L2-normalised, = data/graph/chunk_vecs.npy rows).
  val.jsonl     val rows of the M7 ship pairs: {id, q, lang, C_ref}.
  test.jsonl    the sealed test rows (test_eval.jsonl, read only), same fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = "data/m7/full/54aa9a000b4b/ship/pairs.jsonl"
TEST = "data/m7/full/f9test-d43805b09c2e/ship/test_eval/test_eval.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from graphrag import config
    from graphrag.data.ingest import read_records
    from grl.env.struct_data import load_struct_env_data

    data = load_struct_env_data(articles=True)
    nb = data.n_base_chunks
    ids = list(data.chunk_ids[:nb])
    assert nb == 16643 and not any(c.startswith("STRUCT/") for c in ids), nb
    chunks = {c.chunk_id: c for r in read_records(config.path(config.load("data")["records_dir"])) for c in r.chunks}
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for k, cid in enumerate(ids):
            c = chunks[cid]
            it, tn = c.index_text, c.text_norm
            title = it[: len(it) - len(tn)].strip() if it.endswith(tn) else ""
            assert int(c.tokens) == int(data.chunk_tokens[k]), cid
            f.write(json.dumps({"id": cid, "title": title, "text": c.text, "tokens": int(data.chunk_tokens[k])},
                               ensure_ascii=False) + "\n")
    np.save(out / "vecs.npy", np.ascontiguousarray(data.chunk_vecs[:nb], dtype=np.float32))

    n = {}
    for name, src, split in (("val", PAIRS, "val"), ("test", TEST, "test")):
        n[name] = 0
        with open(src, encoding="utf-8") as fi, open(out / f"{name}.jsonl", "w", encoding="utf-8") as fo:
            for line in fi:
                r = json.loads(line)
                if r["split"] != split:
                    continue
                fo.write(json.dumps({"id": r["id"], "q": r["q"], "lang": r["lang"], "C_ref": r["C_ref"]},
                                    ensure_ascii=False) + "\n")
                n[name] += 1
    print(f"{nb} base chunks, vecs {data.chunk_vecs[:nb].shape}, {n} questions -> {out}")


if __name__ == "__main__":
    main()
