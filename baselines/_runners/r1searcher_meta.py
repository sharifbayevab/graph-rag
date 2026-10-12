"""R1-Searcher baseline, step 3 (Mac, light): summarise a run file and write its meta.json.

  python scripts/ext_baselines/r1searcher_meta.py --run runs/ext_baselines/r1searcher-val-k12.jsonl \\
      --traces <...>.traces.jsonl --runtime-s 1234 --split val
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

CYR = re.compile(r"[Ѐ-ӿ]")
LAT = re.compile(r"[A-Za-zʻʼ]")


def script(q: str) -> str:
    c, l = len(CYR.findall(q)), len(LAT.findall(q))
    if c and l:
        return "mixed"
    return "cyrillic" if c else ("latin" if l else "other")


def summary(rows: list[dict], traces: list[dict]) -> dict:
    n = len(rows)
    qs = [q for r in rows for q in r["queries"]]
    base_only = []
    for r in rows:      # recall over the gold ids reachable by a base-chunk retriever (STRUCT/... gold excluded)
        g = [c for c in r["gold"] if not c.startswith("STRUCT/")] if "gold" in r else None
        if g:
            base_only.append(len(set(g) & set(r["retrieved"])) / len(g))
    return {
        "n": n,
        "mean_recall12": sum(r["recall12"] for r in rows) / n,
        "mean_tokens12": sum(r["tokens12"] for r in rows) / n,
        "mean_n_search": sum(r["n_search"] for r in rows) / n,
        "share_answered": sum(r["answered"] for r in rows) / n,
        "share_zero_search": sum(r["n_search"] == 0 for r in rows) / n,
        "mean_n_retrieved": sum(len(r["retrieved"]) for r in rows) / n,
        "stop_reason_final": dict(collections.Counter(t["stop_reason_final"] for t in traces)),
        "n_queries": len(qs),
        "query_script": dict(collections.Counter(script(q) for q in qs)),
        **({"mean_recall12_base_gold_only": sum(base_only) / len(base_only), "n_base_gold_questions": len(base_only)}
           if base_only else {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--questions", required=True, help="bundle val.jsonl / test.jsonl (for gold ids)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--runtime-s", type=float, required=True)
    ap.add_argument("--config-sha", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8")]
    gold = {q["id"]: q["C_ref"] for q in map(json.loads, open(a.questions, encoding="utf-8"))}
    traces = [json.loads(l) for l in open(a.traces, encoding="utf-8")]
    s = summary([{**r, "gold": gold[r["id"]]} for r in rows], traces)
    meta = {
        "baseline": "R1-Searcher (RUCAIBox, arXiv 2503.05592), released checkpoint, no fine-tuning, no tuning on our data",
        "split": a.split,
        "checkpoint": {"hf_repo": "XXsongLALA/Qwen-2.5-7B-base-RAG-RL",
                       "revision": "bc1b424ee27133445a6430df9c93d46a9453a6eb", "dtype": "bfloat16"},
        "official_code": "github.com/RUCAIBox/R1-Searcher evaluation/eval_search_loacl.py (main), README command "
                         "--prompt_type v0 --temp 0.0",
        "prompt": "base_prompt_v0 verbatim (English), raw text completion, no chat template; question inserted as is "
                  "(Uzbek Cyrillic / Uzbek Latin / Russian), not translated",
        "generation": {"temperature": 0.0, "top_p": 0.95, "decoding": "greedy", "max_new_tokens_per_round": 512,
                       "stop": ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"],
                       "max_rounds": "official loop: round k==9 ends as many_retrieve -> at most 9 generations / 9 searches",
                       "query_cleanup": "remove \" ' ... ; tab -> space (official)",
                       "doc_insertion": "<|end_of_query|>\\n\\n<|begin_of_documents|>\\n(1)doc\\n...(5)doc\\n"
                                        "<|end_of_documents|>\\n\\n appended to prompt + generated_text.strip() (official)"},
        "retriever": {
            "ours": "naive dense: BAAI/bge-m3 (rev 5617a9f61b028005a4858fdac845db406aefb181), CLS dense, L2-normalised, "
                    "exact inner product over the 16,643 base chunks (load_struct_env_data(articles=True)."
                    "chunk_ids[:n_base_chunks], STRUCT rows excluded); passage vectors = data/graph/chunk_vecs.npy "
                    "(index_text, sha256 46505680...bb903), query = bge-m3 encode of graphrag query_internal(query), "
                    "no prefix",
            "top_n_per_search": 5,
            "doc_text_shown": "'<breadcrumb> <original chunk text>' whitespace-collapsed to one line (KILT-style "
                              "'title text' passage); breadcrumb = index_text minus text_norm (internal normalised form)",
            "paper": "BAAI/bge-large-en-v1.5 (FlagModel, query instruction 'Represent this sentence for searching "
                     "relevant passages:', max_length 300) + FAISS Flat IP over KILT 2019 English Wikipedia split into "
                     "100-word passages, k=5"},
        "metric": "retrieved = docs of all search calls in order received, deduplicated, first 12; recall12 = "
                  "|first12 ∩ C_ref| / |C_ref| (C_ref includes STRUCT/... ids, which no base-chunk retriever can "
                  "return); tokens12 = sum chunk_tokens of the first 12",
        "engine": "vLLM 0.31.0 (official engine), torch 2.13.0+cu130, LLM(model, tensor_parallel_size=1, "
                  "gpu_memory_utilization=0.30, kv_cache_memory_bytes=18 GiB, trust_remote_code=True); all active prompts of a round in one "
                  "llm.generate call, as in the official loop",
        "deviations_from_official": [
            "vLLM gpu_memory_utilization 0.30 + a fixed kv_cache_memory_bytes (18 GiB = 337k tokens) instead of "
            "0.95: the DGX Spark's 121 GB unified memory is shared with other jobs (enough for every round's batch; this "
            "only affects scheduling, not greedy outputs beyond batch-level float noise). vLLM 0.31.0 is newer "
            "than the version the authors used.",
            "Round 9 (k==9) is not generated: the official code discards its output and marks many_retrieve, so the "
            "result is identical and compute is saved.",
            "Search backend = our corpus + bge-m3 dense retriever (the environment), not bge-large-en-v1.5 over KILT "
            "Wikipedia; top-5 per search kept as official.",
            "Document text: our chunks (up to ~512 bge-m3 tokens, Uzbek/Russian legal text) instead of 100-word "
            "English Wikipedia passages; formatted 'title text' on one line like the official corpus; the "
            "official re.sub(r'^\\d+\\s+', '', doc) is still applied.",
            "Questions are Uzbek/Russian (the model was RL-trained on English HotpotQA/2Wiki); used as is per the "
            "user's choice, no translation, no prompt adaptation.",
            "The official script reads the 'question' field; ours is the M7 'q' field (same role).",
        ],
        "frozen_config_sha256": a.config_sha,
        "frozen_config_note": "sha256 of scripts/ext_baselines/r1searcher_run.py; val and test were run back to back "
                              "by one script with this identical config (runs/ext_baselines/r1searcher.frozen_config"
                              ".sha256 lists the input bundle hashes). An earlier HF-transformers attempt was aborted "
                              "mid-val (too slow, ~35 min/round) before any output; no test question was run with it.",
        "runtime_s": a.runtime_s,
        "hardware": "DGX Spark (NVIDIA GB10, aarch64, 121 GB unified memory)",
        "summary": s,
    }
    out = Path(a.run).with_suffix(".meta.json")
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
