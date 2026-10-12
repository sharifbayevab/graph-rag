"""Offline extraction (the only paid LLM step of indexing) on the Mac, with each method's OWN extraction code, prompts and
sqlite LLM cache. The Spark index run (run_method.py) later replays every call from this cache.

    venv_mac_hippo/bin/python scripts/ext_baselines/openie_mac.py hipporag [--limit N] [--workers 8] [--budget 60]
    venv_mac_prop/bin/python  scripts/ext_baselines/openie_mac.py proprag  [--limit N] [--workers 8] [--budget 90]

OPENAI_API_KEY comes from the environment. Only the method packages' __init__ files are bypassed (they import
torch/igraph/the embedder, none of which extraction uses). Guards: a USD budget on billed tokens and a cap on how often
one identical prompt is re-sent (PropRAG re-asks with use_cache=False in unbounded loops); either one exits the process,
nothing paid is lost (the cache is written per call)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LLM, PRICE, ROOT, SRC, cache_usage, corpus, stub_packages  # noqa: E402

_lock = threading.Lock()
_spent = {"prompt": 0, "cached_prompt": 0, "completion": 0, "calls": 0}


def spent_usd() -> float:
    """Billed USD of this process, prompt-cache discount applied (OpenAI usage.prompt_tokens_details.cached_tokens)."""
    return ((_spent["prompt"] - _spent["cached_prompt"]) / 1e6 * PRICE["input"]
            + _spent["cached_prompt"] / 1e6 * PRICE["cached_input"] + _spent["completion"] / 1e6 * PRICE["output"])


def save_dir(method: str, variant: str) -> Path:
    """base -> <ROOT>/<method> (the faithful runs); struct / v2 / v2pilot -> <ROOT>/<method>-<variant>."""
    return ROOT / (method if variant == "base" else f"{method}-{variant}")
_repeats: dict[str, int] = {}


def guard(client, budget: float, max_repeat: int, drop_extra_body: bool):
    """Wrap client.chat.completions.create: budget / repeat guards (and PropRAG's OpenRouter-only extra_body)."""
    create = client.chat.completions.create

    def wrapped(**params):
        if drop_extra_body:
            params.pop("extra_body", None)           # {"provider": {"order": ["Nebius"]}}: OpenRouter routing only
        h = hashlib.sha256(json.dumps(params["messages"], sort_keys=True, default=str).encode()).hexdigest()
        with _lock:
            _repeats[h] = _repeats.get(h, 0) + 1
            if _repeats[h] > max_repeat:
                print(f"\nFATAL: identical prompt sent {_repeats[h]} times (unbounded retry loop); stopping",
                      file=sys.stderr, flush=True)
                os._exit(3)
        try:
            r = create(**params)
        except Exception as e:                        # a 400/401 would make PropRAG's infer() loop forever
            name = type(e).__name__
            if name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                print(f"\nFATAL: {name}: {e}", file=sys.stderr, flush=True)
                os._exit(4)
            raise
        with _lock:
            _spent["calls"] += 1
            _spent["prompt"] += r.usage.prompt_tokens
            _spent["completion"] += r.usage.completion_tokens
            det = getattr(r.usage, "prompt_tokens_details", None)
            _spent["cached_prompt"] += int(getattr(det, "cached_tokens", 0) or 0)
            usd = spent_usd()
            if usd > budget:
                print(f"\nFATAL: budget ${budget} exceeded (${usd:.2f}, {_spent}); stopping", file=sys.stderr,
                      flush=True)
                os._exit(5)
        return r

    client.chat.completions.create = wrapped


class _Pool(ThreadPoolExecutor):
    """PropRAG hard-codes ThreadPoolExecutor(max_workers=300), HippoRAG uses the default pool size; cap the concurrency (no effect on results)."""
    workers = 8

    def __init__(self, max_workers=None, *a, **kw):
        super().__init__(self.workers, *a, **kw)


def run_hipporag(docs, a):
    src = SRC / "HippoRAG-2.0.0a4/src/hipporag"        # = PyPI hipporag 2.0.0a4 (git 242abfb)
    stub_packages({"hipporag": src, "hipporag.llm": src / "llm"})
    import hipporag.information_extraction.openie_openai as oo
    from hipporag.llm.openai_gpt import CacheOpenAI
    from hipporag.utils.config_utils import BaseConfig
    from hipporag.utils.misc_utils import compute_mdhash_id
    _Pool.workers = a.workers
    oo.ThreadPoolExecutor = _Pool
    if a.variant.startswith("v2"):
        import uzlegal
        uzlegal.install_hipporag(oo)
    cfg = BaseConfig(llm_name=LLM, save_dir=str(save_dir("hipporag", a.variant)))
    llm = CacheOpenAI.from_experiment_config(cfg)
    guard(llm.openai_client, a.budget, a.max_repeat, False)
    ie = oo.OpenIE(llm_model=llm)
    chunks = {compute_mdhash_id(t, prefix="chunk-"): {"content": t} for t in docs}
    ner, tri = ie.batch_openie(chunks)
    print(f"NER errors {sum(1 for r in ner.values() if r.metadata.get('error'))}, empty NER "
          f"{sum(1 for r in ner.values() if not r.unique_entities)}, triple errors "
          f"{sum(1 for r in tri.values() if r.metadata.get('error'))}, empty triples "
          f"{sum(1 for r in tri.values() if not r.triples)}")
    return llm.cache_file_name


def run_proprag(docs, a):
    root = SRC / "PropRAG"
    stub_packages({"src": root / "src", "src.proprag": root / "src/proprag"})
    sys.path.insert(0, str(root))
    import src.proprag.information_extraction.enhanced_openie as eo
    import src.proprag.information_extraction.proposition_extraction as pe
    import src.proprag.utils.misc_utils as mu
    from src.proprag.llm.openai_gpt import CacheOpenAI
    from src.proprag.utils.config_utils import BaseConfig
    _Pool.workers = a.workers
    eo.ThreadPoolExecutor = pe.ThreadPoolExecutor = _Pool
    if a.variant.startswith("v2"):
        import src.proprag.prompts.prompt_template_manager as ptm
        import uzlegal
        uzlegal.install_proprag(eo, ptm)
    os.chdir(root)            # PropRAG runs from its repo root (BaseConfig reads ./openrouter_api_key.txt, empty there)
    cfg = BaseConfig(llm_name=LLM, llm_base_url=None, save_dir=str(save_dir("proprag", a.variant)))
    cfg.api_key = os.environ["OPENAI_API_KEY"]
    llm = CacheOpenAI.from_experiment_config(cfg)
    guard(llm.openai_client, a.budget, a.max_repeat, True)
    ie = eo.EnhancedOpenIE(llm_model=llm)
    chunks = {mu.compute_mdhash_id(t, prefix="chunk-"): {"content": t} for t in docs}
    ner, _, props = ie.batch_openie(chunks)
    print(f"NER errors {sum(1 for r in ner.values() if r.metadata.get('error'))}, empty NER "
          f"{sum(1 for r in ner.values() if not r.unique_entities)}, proposition errors "
          f"{sum(1 for r in props.values() if r.metadata.get('error'))}, empty propositions "
          f"{sum(1 for r in props.values() if not r.propositions)}")
    return llm.cache_file_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=("hipporag", "proprag"))
    ap.add_argument("--limit", type=int, default=0, help="pilot: N evenly spaced passages")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, required=True, help="USD cap on tokens billed in THIS process")
    ap.add_argument("--max-repeat", type=int, default=6)
    ap.add_argument("--variant", choices=("base", "struct", "v2", "v2pilot"), default="base",
                    help="struct / v2 / v2pilot use the base+STRUCT corpus (EXT_DATA=data_struct); v2* install uzlegal")
    ap.add_argument("--ids", help="v2pilot: file with one chunk id per line (only these passages)")
    a = ap.parse_args()
    docs, t2i, _ = corpus()
    assert (os.environ.get("EXT_DATA", "data") == "data_struct") == (a.variant != "base"), "EXT_DATA vs --variant"
    if a.ids:
        want = set(Path(a.ids).read_text().split())
        docs = [t for t in docs if want & set(t2i[t])]
    if a.limit:
        docs = docs[::len(docs) // a.limit][:a.limit]
    path = (run_hipporag if a.method == "hipporag" else run_proprag)(docs, a)
    summary = {"method": a.method, "variant": a.variant, "passages": len(docs), "billed_this_run": dict(_spent),
               "billed_usd_this_run": round(spent_usd(), 4), "cache_totals": cache_usage(path)}
    with open(save_dir(a.method, a.variant) / "billing.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
