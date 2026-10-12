"""Re-score the per-question files fetched from Spark and write the meta.json files.

    python scripts/ext_baselines/finalize.py      (after copying Spark's out/ to runs/ext_baselines/out/)
    -> runs/ext_baselines/<method>-{val,test}-k12.jsonl + .meta.json"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import K, LLM, PRICE, ROOT, cache_usage, corpus, jsonl, questions  # noqa: E402

VERSION = {
    "hipporag": {"package": "hipporag 2.0.0a4 (PyPI; = OSU-NLP-Group/HippoRAG git 242abfb, 2025-06-24)",
                 "repo": "https://github.com/OSU-NLP-Group/HippoRAG", "commit": "242abfb",
                 "config": "HippoRAG main.py BaseConfig: retrieval_top_k=200, linking_top_k=5, max_qa_steps=3, qa_top_k=5, "
                           "graph_type=facts_and_sim_passage_node_unidirectional, embedding_batch_size=8, "
                           "max_new_tokens=None, rerank_dspy_file_path=filter_llama3.3-70B-Instruct.json, "
                           "openie_mode=online; BaseConfig defaults otherwise (damping 0.5, passage_node_weight 0.05, "
                           "synonymy_edge_topk 2047, synonymy_edge_sim_threshold 0.8, temperature 0)"},
    "proprag": {"package": "PropRAG (no release)", "repo": "https://github.com/ReLink-Inc/PropRAG",
                "commit": "3ec103488abd5589e569ee0fdd6e0c7067e5b783",
                "config": "PropRAG main.py (no flags): retrieval_top_k=200, linking_top_k=5, max_qa_steps=3, qa_top_k=5, "
                          "graph_type=facts_and_sim_passage_node_unidirectional, embedding_batch_size=4, beam_width=4, "
                          "max_path_length=3, second_stage_filter_k=40, use_propositions=False (EnhancedOpenIE with "
                          "propositions is always used); retrieve() defaults (two-stage beam search, focus_top_k=50); "
                          "BaseConfig defaults otherwise (damping 0.5, passage_node_weight 0.05, "
                          "synonymy_edge_sim_threshold 0.8, max_new_tokens 8192, temperature 0)"},
}
COMMON_DEV = [
    "LLM: gpt-4.1-mini, temperature 0 (user decision; papers: Llama-3.3-70B-Instruct / gpt-4o-mini)",
    "Embedder: BAAI/bge-m3 via a minimal sentence-transformers adapter (scripts/ext_baselines/bge_m3.py; no query "
    "instruction, L2-normalised, max_seq_length = embedding_max_seq_len 2048, encode batch 64) instead of "
    "nvidia/NV-Embed-v2 (user decision: same embedder as all our methods)",
    "Corpus: our 16,643 base chunks (no STRUCT rows); passage text = Chunk.index_text (breadcrumb + text_norm, Latin "
    "lower case, what our dense/BM25 baselines index); 61 chunks share their text with another chunk (46 groups, no "
    "val/test gold among them) -> 16,582 unique passages; a retrieved passage maps to all its chunk ids (sorted)",
    "Queries: query_internal(q) (Cyrillic -> Latin, lower case, canonical numbers), the same mapping our baselines apply",
    f"Top-{K}: the first {K} chunk ids of the method's ranking (retrieve() returns 200 passages)",
    "Prompts / few-shot demos: official, verbatim (English); no tuning on our data; one frozen config for val and test",
    "Extraction LLM calls were made on the Mac with the method's own extraction code and sqlite LLM cache (concurrency "
    "capped at 16 threads; PropRAG restarted once at 32 threads, resuming from its cache), then replayed from that cache during indexing on Spark (0 new calls)",
    "Spark env: torch 2.14.0+cu130 (GB10 / aarch64; the pinned torch 2.5.1 has no build for it), transformers 4.45.2 "
    "(pinned), openai 1.91.0; vllm replaced by an import shim (offline vLLM mode unused)",
]
METHOD_DEV = {
    "hipporag": [
        "Version: the latest PyPI release 2.0.0a4 (`pip install hipporag`), not git HEAD (2.0.0a5-dev): HEAD aborts the "
        "whole indexing when any NER response fails to parse (13 of a 100-passage pilot with gpt-4.1-mini), while the "
        "release keeps such passages with no entities",
        "Extraction ran with BaseConfig's max_new_tokens=2048 (main.py sets None); the cache key ignores it and it only "
        "matters for truncated outputs (see finish_reason_length)",
        "Observation, not a change: with the release's NER parser 3,333 of 16,582 passages (20.1%) got no entities "
        "(gpt-4.1-mini often answers with a bare JSON list or a list of objects, which the 2.0.0a4 regex does not "
        "accept; 17 responses raised a parse error) and 2,517 (15.2%) no triples (20 parse errors); kept as-is, "
        "official prompts unchanged",
    ],
    "proprag": [
        "CacheOpenAI.infer always sends extra_body={provider: {order: [Nebius]}} (OpenRouter routing); dropped for the "
        "OpenAI endpoint (the OpenAI API rejects the unknown field)",
        "API key passed via config.api_key from OPENAI_API_KEY instead of openrouter_api_key.txt",
        "Hard-coded ThreadPoolExecutor(max_workers=300) capped to 16/32 during extraction (no effect on outputs)",
        "Observation: NER and proposition extraction returned non-empty results for all 16,582 passages",
    ],
}


def length_truncations(sqlite_path) -> int:
    import sqlite3
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    n = sum(1 for (m,) in con.execute("SELECT metadata FROM cache") if json.loads(m).get("finish_reason") == "length")
    con.close()
    return n


def main() -> None:
    _, t2i, tokens = corpus()
    for method in ("hipporag", "proprag"):
        ret = json.loads((ROOT / "out" / f"{method}-retrieve.json").read_text())
        idx = json.loads((ROOT / "out" / f"{method}-index.json").read_text())
        mac_cache = ROOT / method / "llm_cache" / f"{LLM}_cache.sqlite"
        extraction = cache_usage(mac_cache)
        query_time = {k: ret["llm_cache_after_retrieve"][k] - ret["llm_cache_after"][k]
                      for k in ("calls", "prompt_tokens", "completion_tokens")}
        query_time["usd_upper_bound"] = round(query_time["prompt_tokens"] / 1e6 * PRICE["input"]
                                              + query_time["completion_tokens"] / 1e6 * PRICE["output"], 2)
        for split in ("val", "test"):
            qs = {q["id"]: q for q in questions(split)}
            src = ROOT / "out" / f"{method}-{split}-k12.jsonl"
            rows = jsonl(src)
            assert sorted(r["id"] for r in rows) == sorted(qs), f"{src}: ids differ from the {split} questions"
            for r in rows:                                     # independent re-score
                gold, ids = set(qs[r["id"]]["C_ref"]), r["retrieved"]
                assert len(ids) <= K and all(i in tokens for i in ids)
                assert abs(r["recall12"] - len(gold & set(ids)) / len(gold)) < 1e-12
                assert r["tokens12"] == sum(tokens[i] for i in ids)
            dst = ROOT / f"{method}-{split}-k12.jsonl"
            shutil.copyfile(src, dst)
            s = ret[split]
            meta = {
                "method": method, "split": split, "n_questions": len(rows), "k": K,
                "recall12_mean": sum(r["recall12"] for r in rows) / len(rows),
                "tokens12_mean": sum(r["tokens12"] for r in rows) / len(rows),
                "unmapped_passages": s["unmapped_passages"],
                "recall12_ceiling_base_corpus": sum(len(set(q["C_ref"]) & tokens.keys()) / len(q["C_ref"])
                                                    for q in qs.values()) / len(qs),
                "questions_with_STRUCT_gold": sum(any(c not in tokens for c in q["C_ref"]) for q in qs.values()),
                "recall12_mean_on_questions_without_STRUCT_gold": (
                    lambda v: sum(v) / len(v))([r["recall12"] for r in rows
                                                if all(c in tokens for c in qs[r["id"]]["C_ref"])]),
                "gold_ids_in_merged_duplicate_passages": sum(len(t2i[t]) > 1 for q in qs.values() for t in
                                                             {t for t, v in t2i.items() for c in q["C_ref"]
                                                              if c in v}),
                "version": VERSION[method],
                "llm": f"{LLM} (OpenAI API), temperature 0",
                "embedding_model": "BAAI/bge-m3 (sentence-transformers adapter, 1024-d, normalised)",
                "indexing": {"spark_index_seconds_embed_and_graph": idx["index_seconds"],
                             "mac_extraction_log": f"runs/ext_baselines/openie_{method}.log",
                             "graph": idx["graph"]},
                "api_cost": {"price_usd_per_1M": PRICE, "extraction": extraction,
                             "extraction_finish_reason_length": length_truncations(mac_cache),
                             "query_time_val_plus_test": query_time,
                             "note": "usd_upper_bound bills every prompt token at the uncached input price"},
                "retrieve_seconds": s["retrieve_seconds"],
                "deviations_from_defaults": COMMON_DEV + METHOD_DEV[method],
            }
            (ROOT / f"{method}-{split}-k12.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            print(f"{method} {split}: n={len(rows)} recall@12 {meta['recall12_mean']:.4f} tokens12 "
                  f"{meta['tokens12_mean']:.0f} unmapped {s['unmapped_passages']} | ceiling "
                  f"{meta['recall12_ceiling_base_corpus']:.4f}, STRUCT-gold q {meta['questions_with_STRUCT_gold']}, "
                  f"recall@12 w/o them {meta['recall12_mean_on_questions_without_STRUCT_gold']:.4f}")


if __name__ == "__main__":
    main()
