"""Export the base corpus (16,643 chunks, no STRUCT rows) and the val questions for the external baselines.

    PYTHONPATH=. python scripts/ext_baselines/export_corpus.py           -> runs/ext_baselines/data/{corpus,val,test}.jsonl
    PYTHONPATH=. python scripts/ext_baselines/export_corpus.py --struct  -> runs/ext_baselines/data_struct/ (+ the 1,892
        STRUCT rows of load_struct_env_data(articles=True): document + article fragments, text = their index_text from
        grl.env.struct_data.build_struct_inputs, tokens = the env's chunk_tokens)

Passage text = Chunk.index_text (breadcrumb + text_norm: what our dense/BM25 baselines index). Query text =
query_internal(q) (the same mapping our baselines apply to queries, graphrag/retrieval/rankings.py). Both methods key
passages by md5(content): the 61 chunks whose index_text duplicates another chunk's (46 groups, no val gold among them)
become one passage; scripts/ext_baselines/common.py maps a retrieved passage back to all its ids (sorted)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from graphrag import config
from graphrag.data.normalize import query_internal
from grl.env.struct_data import load_struct_env_data

PAIRS = "data/m7/full/54aa9a000b4b/ship/pairs.jsonl"                     # val rows (split == "val")
TEST = "data/m7/full/f9test-d43805b09c2e/ship/test_eval/test_eval.jsonl"   # test rows (read only)
STRUCT = "--struct" in sys.argv
OUT = Path("runs/ext_baselines/data_struct" if STRUCT else "runs/ext_baselines/data")


def main() -> None:
    from graphrag.data.ingest import read_records
    data = load_struct_env_data(articles=True)
    ids = list(data.chunk_ids[:data.n_base_chunks])
    assert len(ids) == 16643 and not any(i.startswith("STRUCT/") for i in ids)
    chunks = {c.chunk_id: c for r in read_records(config.path(config.load("data")["records_dir"])) for c in r.chunks}
    texts = [chunks[i].index_text for i in ids]
    print(f"{len(texts) - len(set(texts))} chunks duplicate another chunk's passage text")
    if STRUCT:
        from grl.env.struct_data import build_struct_inputs
        sids = list(data.chunk_ids[data.n_base_chunks:])
        assert len(sids) == 1892 and all(i.startswith("STRUCT/") for i in sids)
        index = {**build_struct_inputs(data.version)[1], **build_struct_inputs(data.version, articles=True)[1]}
        ids, texts = ids + sids, texts + [index[i] for i in sids]
    tok = {c: int(data.chunk_tokens[data.chunk_row[c]]) for c in ids}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i, t in zip(ids, texts):
            f.write(json.dumps({"id": i, "text": t, "tokens": tok[i]}, ensure_ascii=False) + "\n")
    print(f"corpus {len(ids)} passages, {sum(tok.values())} tokens")
    for split, src in (("val", PAIRS), ("test", TEST)):
        rows = [json.loads(l) for l in config.path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = sorted((r for r in rows if split == "test" or r["split"] == "val"), key=lambda r: r["id"])
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "q": r["q"], "query": query_internal(r["q"]), "C_ref": r["C_ref"]},
                                   ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} questions")


if __name__ == "__main__":
    main()
