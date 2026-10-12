"""Index + retrieve with HippoRAG 2 (PyPI 2.0.0a4) or PropRAG (git HEAD), configured as each repo's main.py does,
except the user's fixed choices: LLM gpt-4.1-mini (temperature 0), embedder BAAI/bge-m3 (bge_m3.py adapter), corpus =
our 16,643 base chunks, top-12. Runs on Spark (GPU for bge-m3).

    python run_method.py hipporag index       # OPENAI_API_KEY=dummy: every extraction call must hit the Mac-built cache
    python run_method.py hipporag retrieve    # real key (HippoRAG's recognition-memory filter calls the LLM per query)
    python run_method.py proprag  index|retrieve

retrieve: frozen config, val then test, -> <ROOT>/out/<method>-{val,test}-k12.jsonl + <method>-retrieve.json (stats)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bge_m3 import MODEL, make_adapter  # noqa: E402
from common import LLM, ROOT, SRC, cache_usage, corpus, map_and_score, questions  # noqa: E402
from openie_mac import guard, save_dir  # noqa: E402

RETRIEVE_BUDGET_USD = 5.0          # query-time LLM spend cap (HippoRAG filter: 479 short calls)


def build_hipporag(n_docs, variant):
    root = SRC / "HippoRAG-2.0.0a4"
    os.chdir(root)                                   # main.py's rerank_dspy_file_path is relative to the repo root
    sys.path.insert(0, str(root / "src"))
    import hipporag  # noqa: F401  (package __init__ rebinds the name hipporag.HippoRAG to the class)
    H = sys.modules["hipporag.HippoRAG"]
    from hipporag.embedding_model import BaseEmbeddingModel
    from hipporag.utils.config_utils import BaseConfig
    Adapter = make_adapter(BaseEmbeddingModel)
    H._get_embedding_model_class = lambda embedding_model_name=None: Adapter
    if variant == "v2":                              # uzlegal prompts / parsers + Unicode text_processing
        import hipporag.information_extraction.openie_openai as oo
        import uzlegal
        uzlegal.install_hipporag(oo, H)
    elif variant == "kg":                            # names from our KG keep Cyrillic / ʼ (original drops non-ASCII)
        import uzlegal
        H.text_processing = uzlegal.text_processing_unicode
    cfg = BaseConfig(                                # = HippoRAG main.py, except llm_name / embedding / save_dir
        save_dir=str(save_dir("hipporag", variant)),
        llm_base_url="https://api.openai.com/v1",
        llm_name=LLM,
        dataset=None,
        embedding_model_name=MODEL,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=n_docs,
        openie_mode="online",
    )
    rag = H.HippoRAG(global_config=cfg)
    return rag, cfg, rag.llm_model


def build_proprag(n_docs, variant):
    root = SRC / "PropRAG"
    os.chdir(root)                                   # BaseConfig reads ./openrouter_api_key.txt (empty)
    sys.path.insert(0, str(root))
    import src.proprag  # noqa: F401
    P = sys.modules["src.proprag.PropRAG"]
    from src.proprag.embedding_model import BaseEmbeddingModel
    from src.proprag.utils.config_utils import BaseConfig
    Adapter = make_adapter(BaseEmbeddingModel)
    P._get_embedding_model_class = lambda config, use_cache=True: Adapter(global_config=config,
                                                                          embedding_model_name=MODEL)
    if variant == "v2":
        import src.proprag.information_extraction.enhanced_openie as eo
        import src.proprag.prompts.prompt_template_manager as ptm
        import uzlegal
        uzlegal.install_proprag(eo, ptm)
    cfg = BaseConfig(                                # = PropRAG main.py defaults (no flags), except llm / embedding
        llm_base_url=None,
        llm_name=LLM,
        dataset=None,
        save_dir=str(save_dir("proprag", variant)),
        embedding_model_name=MODEL,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=4,
        beam_width=4,
        max_path_length=3,
        second_stage_filter_k=40,
        corpus_len=n_docs,
        openie_mode="online",
        use_propositions=False,
    )
    cfg.api_key = os.environ["OPENAI_API_KEY"]       # instead of openrouter_api_key.txt
    rag = P.PropRAG(global_config=cfg)
    return rag, cfg, rag.llm_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("hipporag", "proprag"))
    ap.add_argument("phase", choices=("index", "retrieve"))
    ap.add_argument("--variant", choices=("base", "struct", "v2", "kg"), default="base",
                    help="base = faithful (data/); struct, v2, kg use data_struct/ (base + STRUCT rows)")
    a = ap.parse_args()
    assert (os.environ.get("EXT_DATA", "data") == "data_struct") == (a.variant != "base"), "EXT_DATA vs --variant"
    assert a.variant != "kg" or a.method == "hipporag", "kg variant is HippoRAG only"
    tag = a.method if a.variant == "base" else f"{a.method}-{a.variant}"
    logging.basicConfig(level=logging.INFO)
    docs, t2i, tokens = corpus()
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    rag, cfg, llm = (build_hipporag if a.method == "hipporag" else build_proprag)(len(docs), a.variant)
    budget = 0.01 if a.phase == "index" else RETRIEVE_BUDGET_USD
    guard(llm.openai_client, budget, 3, a.method == "proprag")
    t0 = time.time()
    rag.index(docs)
    t_index = time.time() - t0
    stats = {"method": a.method, "variant": a.variant, "phase": a.phase, "index_seconds": round(t_index, 1),
             "llm_cache_after": cache_usage(llm.cache_file_name)}
    if a.phase == "retrieve":
        for split in ("val", "test"):                # one frozen config for both splits
            qs = questions(split)
            t0 = time.time()
            res = rag.retrieve([q["query"] for q in qs])
            rows, unmapped = map_and_score(qs, [r.docs for r in res], t2i, tokens)
            with open(out / f"{tag}-{split}-k12.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats[split] = {"n": len(rows), "unmapped_passages": unmapped,
                            "recall12": sum(r["recall12"] for r in rows) / len(rows),
                            "tokens12": sum(r["tokens12"] for r in rows) / len(rows),
                            "retrieve_seconds": round(time.time() - t0, 1)}
            print(split, stats[split], flush=True)
        stats["llm_cache_after_retrieve"] = cache_usage(llm.cache_file_name)
    stats["graph"] = {"nodes": rag.graph.vcount(), "edges": rag.graph.ecount()}
    (out / f"{tag}-{a.phase}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
