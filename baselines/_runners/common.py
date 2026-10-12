"""Shared helpers for the external baselines (HippoRAG 2, PropRAG): corpus / questions, passage -> chunk-id mapping,
scoring, import stubs and the gpt-4.1-mini price. No dependency on the project venv: runs in the methods' own venvs."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("EXT_ROOT", HERE.parents[2] / "runs" / "ext_baselines"))   # advanced-rag-v2/runs/ext_baselines
DATA = ROOT / os.environ.get("EXT_DATA", "data")          # data (base chunks) | data_struct (base + STRUCT rows)
SRC = ROOT / "src"
K = 12
LLM = "gpt-4.1-mini"
PRICE = {"input": 0.40, "cached_input": 0.10, "output": 1.60}     # USD per 1M tokens, gpt-4.1-mini (OpenAI list price)


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def corpus():
    """(docs, text_to_ids, tokens): docs = unique passage texts in corpus order; text_to_ids = text -> sorted chunk ids
    sharing that text (61 chunks duplicate another's text; both methods key passages by md5(content))."""
    rows = jsonl(DATA / "corpus.jsonl")
    t2i: dict[str, list[str]] = {}
    for r in rows:
        t2i.setdefault(r["text"], []).append(r["id"])
    return list(t2i), {t: sorted(v) for t, v in t2i.items()}, {r["id"]: r["tokens"] for r in rows}


def questions(split: str):
    assert split in ("val", "test")
    return jsonl(DATA / f"{split}.jsonl")


def map_and_score(qs, retrieved_texts, t2i, tokens, k: int = K):
    """Rows {id, retrieved, recall12, tokens12} + number of passages that failed to map to a chunk id."""
    out, unmapped = [], 0
    for q, texts in zip(qs, retrieved_texts):
        ids = []
        for t in texts:
            if t not in t2i:
                unmapped += 1
                continue
            ids.extend(i for i in t2i[t] if i not in ids)
        ids = ids[:k]
        gold = set(q["C_ref"])
        out.append({"id": q["id"], "retrieved": ids, "recall12": len(gold & set(ids)) / len(gold),
                    "tokens12": int(sum(tokens[i] for i in ids))})
    return out, unmapped


def stub_packages(names_paths: dict[str, Path]) -> None:
    """Register packages without running their __init__ (the OpenIE-only Mac run must not import torch/igraph)."""
    for name, path in names_paths.items():
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m


def cache_usage(sqlite_path) -> dict:
    """Token totals over every response in an llm_cache sqlite (each row was paid exactly once)."""
    n = p = c = 0
    if not Path(sqlite_path).is_file():
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd_upper_bound": 0.0}
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    for (meta,) in con.execute("SELECT metadata FROM cache"):
        m = json.loads(meta)
        n += 1
        p += int(m.get("prompt_tokens", 0))
        c += int(m.get("completion_tokens", 0))
    con.close()
    return {"calls": n, "prompt_tokens": p, "completion_tokens": c,
            "usd_upper_bound": round(p / 1e6 * PRICE["input"] + c / 1e6 * PRICE["output"], 2)}
