"""Cross-encoder scores for the M5 RL reranker (grl.model.rerank): every (query, candidate) of the snapshot's train +
val queries, candidates = the rerank candidate list with rerank.ce_n slots (>= rerank.n, same fusion), scored by
rerank.ce_model (default BAAI/bge-reranker-v2-m3) through transformers AutoModelForSequenceClassification (no
FlagEmbedding): eval mode, no grad, fp16 on CUDA, fixed batch order (sorted by fragment length, then position), raw
logit per pair, rounded to 6 decimals.
    PYTHONPATH=. python scripts/ce_scores.py --pairs P --rankings R --out runs/<id>/ce_scores.jsonl [--limit N]
        [--device auto|cuda|cpu|mps] [--set rerank.ce_n=30 ...]
Output: <out>.jsonl -- one line per query {"qkey", "rows", "chunk_ids", "scores"} sorted by qkey, rows in candidate
order -- and <out stem>.meta.json (model id + revision commit sha, transformers / torch versions, dtype, device,
text field, max_length, candidate config, graph version, input sha256s, out_sha256), read by
grl.model.rerank.load_ce_cache (rerank.ce_cache). Test / Q-real rows of --pairs are never read
(grl.train.expert.expert_inputs). Pilot 5: --test-eval <ship>/test_eval/test_eval.jsonl --test-rankings <its
latest.json> also scores the consensus test queries into the SAME cache (query + candidate texts only, no C*), so a
trained policy's recorded ce_sha256 matches at the final test evaluation; training only ever looks up its own keys.
M7 Task 7: --probe reads --pairs as a probe ship pairs.jsonl (split "probe", grl.hard.probe) -- a separate cache.
On Spark the model is cached under HF_HOME = ~/abdurakhmon/.cache/huggingface (scripts/spark.py REMOTE_ENV). Heavy:
run it on Spark only; Mac tests use a fake scorer."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

TEXT_FIELDS = ("text", "text_norm", "index_text")


def ce_requests(env, ranks, qkeys, chunk_text) -> list[dict]:
    """[{qkey, text, rows}] in qkey order; rows = env.cand_rows(q) (the rerank candidate list, no C*)."""
    out = []
    for k in sorted(qkeys):
        q = env.query(ranks.get(k))
        rows = [int(c) for c in env.cand_rows(q)]
        missing = [c for c in rows if chunk_text(c) is None]
        if missing:
            raise ValueError(f"{k}: no text for chunk rows {missing[:5]}")
        out.append({"qkey": k, "text": str(q.text or ""), "rows": rows})
    return out


def score_requests(reqs, scorer, chunk_text, chunk_ids, batch: int) -> list[dict]:
    """Every (query, fragment) pair scored by scorer(list[(query, doc)]) -> list[float], in batches of `batch`
    ordered by (fragment text length, flat position) -- deterministic; results mapped back to candidate order."""
    if int(batch) < 1:
        raise ValueError(f"rerank.ce_batch must be >= 1, got {batch}")
    flat = [(i, j, r["text"], chunk_text(c)) for i, r in enumerate(reqs) for j, c in enumerate(r["rows"])]
    order = sorted(range(len(flat)), key=lambda x: (len(flat[x][3]), x))
    scores = [None] * len(flat)
    for b in range(0, len(order), int(batch)):
        idx = order[b:b + int(batch)]
        got = list(scorer([(flat[x][2], flat[x][3]) for x in idx]))
        if len(got) != len(idx):
            raise ValueError(f"scorer returned {len(got)} scores for {len(idx)} pairs")
        for x, v in zip(idx, got):
            scores[x] = round(float(v), 6)
    out = [{"qkey": r["qkey"], "rows": list(r["rows"]), "chunk_ids": [chunk_ids[c] for c in r["rows"]],
            "scores": [None] * len(r["rows"])} for r in reqs]
    for (i, j, _, _), v in zip(flat, scores):
        out[i]["scores"][j] = v
    return out


def query_keys(pairs_path, probe: bool = False, limit: int | None = None) -> list[str]:
    """The query keys to score, sorted: the train + val pairs of a snapshot (grl.train.expert.expert_inputs: test /
    Q-real never read, a probe file refused) or, with probe=True (M7 Task 7), the probe rows of a probe ship
    pairs.jsonl (grl.hard.probe.read_probe_rows: split "probe" only, no Q-real)."""
    if probe:
        from grl.hard.probe import read_probe_rows
        keys = sorted(f"{r['qid']}|q" for r in read_probe_rows(pairs_path))
    else:
        from grl.train.expert import expert_inputs
        keys = sorted(f"{p.id}|q" for p in expert_inputs(pairs_path))
    return keys[:limit] if limit else keys


def write_cache(out_path, lines, meta: dict) -> dict:
    """The .jsonl, then its .meta.json, each written to a temp file and renamed (atomic): a kill leaves either the old
    pair or a new .jsonl whose sha256 the old meta refuses (load_ce_cache), never a torn file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in sorted(lines, key=lambda r: r["qkey"]))
    data = text.encode("utf-8")
    from grl.model.rerank import ce_meta_path
    meta = {**meta, "out_sha256": hashlib.sha256(data).hexdigest(), "n_queries": len(lines),
            "n_pairs": sum(len(r["rows"]) for r in lines)}
    _atomic(out_path, data)
    _atomic(ce_meta_path(out_path),
            (json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return meta


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def chunk_texts(records, field: str) -> tuple[dict, int]:
    """chunk id -> text of `field`; an empty field falls back to text_norm -- (texts, number of fallbacks)."""
    out, fb = {}, 0
    for r in records:
        for c in r.chunks:
            v = getattr(c, field)
            if not v:
                v, fb = c.text_norm, fb + 1
            out[c.chunk_id] = v
    return out, fb


def check_revision(got, want: str) -> str:
    """The loaded model's commit must be exactly the pinned rerank.ce_revision (None = unknown: refused)."""
    if got is None or str(got) != str(want):
        raise ValueError(f"cross-encoder revision {got!r} != pinned rerank.ce_revision {want!r}")
    return str(got)


class HFCrossEncoder:
    """transformers cross-encoder: logits[:, 0] of AutoModelForSequenceClassification over (query, doc) pairs."""

    def __init__(self, model_id: str, revision: str, device: str = "auto", max_length: int = 512,
                 fp16: bool | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from grl.train.log import pick_device
        self.torch = torch
        self.device = pick_device(device)
        self.fp16 = (self.device.type == "cuda") if fp16 is None else bool(fp16)
        if not revision:
            raise ValueError("rerank.ce_revision must pin the model commit")
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        check_revision(getattr(self.model.config, "_commit_hash", None), revision)
        if self.fp16:
            self.model = self.model.half()
        self.model.to(self.device).eval()
        self.max_length = int(max_length)
        self.info = {"model": model_id, "revision": getattr(self.model.config, "_commit_hash", None),
                     "dtype": "float16" if self.fp16 else "float32", "device": str(self.device),
                     "max_length": self.max_length, "tokenizer_fast": bool(getattr(self.tok, "is_fast", False))}

    def __call__(self, pairs):
        torch = self.torch
        with torch.no_grad():
            enc = self.tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            return self.model(**enc).logits.view(-1).float().cpu().tolist()


def _set(cfg: dict, kv: str) -> None:
    """--set a.b=v (YAML value) on an existing key, as scripts/train.py."""
    import yaml
    k, v = kv.split("=", 1)
    node, *path = cfg, *k.split(".")
    for p in path[:-1]:
        node = node[p]
    if path[-1] not in node:
        raise KeyError(f"--set {k}: no such config key")
    node[path[-1]] = yaml.safe_load(v)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, help="only the first N queries (smoke test)")
    ap.add_argument("--test-eval", help="pilot 5: also score the queries of this test_eval.jsonl")
    ap.add_argument("--test-rankings", help="pilot 5: the rankings latest.json of --test-eval")
    ap.add_argument("--probe", action="store_true", help="M7: --pairs is a probe ship pairs.jsonl (split probe)")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    if bool(a.test_eval) != bool(a.test_rankings):
        ap.error("--test-eval and --test-rankings go together")
    if a.probe and a.test_eval:
        ap.error("--probe scores only the probe queries (no --test-eval)")
    import torch
    import transformers
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.rerank import RerankEnv
    from grl.model.variants import rerank_kw
    from grl.train.log import set_determinism
    cfg = config.load("train")
    for kv in a.set:
        _set(cfg, kv)
    rr = cfg["rerank"]
    if rr["ce_text"] not in TEXT_FIELDS:
        raise SystemExit(f"rerank.ce_text must be one of {TEXT_FIELDS}")
    kw = rerank_kw(cfg)
    n_score = int(rr["ce_n"])
    if n_score < kw["n"]:
        raise SystemExit(f"rerank.ce_n {n_score} < rerank.n {kw['n']}: the env would miss scores")
    set_determinism(0, "warn")
    t0 = time.time()
    data = load_env_data()
    ranks = load_rankings(a.rankings, graph_version=data.version)
    qkeys = query_keys(a.pairs, a.probe, a.limit)
    recs = read_records(config.path(config.load("data")["records_dir"]))
    by_id, fallbacks = chunk_texts(recs, rr["ce_text"])
    texts = [by_id.get(c) for c in data.chunk_ids]
    env = RerankEnv(data, None, None, **{**kw, "n": n_score})
    reqs = ce_requests(env, ranks, qkeys, lambda c: texts[c])
    if a.test_eval:
        from grl.train.data import load_test_items
        from grl.train.data import TEST_EVAL_NAME
        if Path(a.test_eval).name != TEST_EVAL_NAME:   # the name guard first; no other file is parsed (9f M6)
            raise SystemExit(f"--test-eval reads only a {TEST_EVAL_NAME} file")
        gdata = data                                   # only the row guard reads C*; M7 Task 9f: STRUCT gold rows
        if any(g.startswith("STRUCT/") for r in (json.loads(x) for x in Path(a.test_eval).read_text(
                encoding="utf-8").splitlines() if x.strip()) for g in r.get("C_ref") or ()):
            from grl.env.struct_data import load_struct_env_data
            gdata = load_struct_env_data(articles=True)
        tkeys = [i.qkey for i in load_test_items(a.test_eval, gdata)]
        if set(tkeys) & set(qkeys):
            raise SystemExit(f"test-eval keys overlap the train / val keys: {sorted(set(tkeys) & set(qkeys))[:3]}")
        tranks = load_rankings(a.test_rankings, graph_version=data.version)
        reqs += ce_requests(env, tranks, tkeys, lambda c: texts[c])
    scorer = HFCrossEncoder(rr["ce_model"], rr["ce_revision"], a.device, int(rr["ce_max_length"]))
    lines = score_requests(reqs, scorer, lambda c: texts[c], data.chunk_ids, int(rr["ce_batch"]))
    lt = Path(a.rankings) if Path(a.rankings).is_absolute() else config.path(a.rankings)
    test_in = {}
    if a.test_eval:
        tl = Path(a.test_rankings) if Path(a.test_rankings).is_absolute() else config.path(a.test_rankings)
        test_in = {"test_eval": str(a.test_eval), "test_eval_sha256": _sha(a.test_eval),
                   "test_rankings": str(a.test_rankings), "test_rankings_sha256": _sha(tl), "n_test": len(tkeys)}
    meta = write_cache(a.out, lines, {
        **scorer.info, "transformers": transformers.__version__, "torch": torch.__version__,
        "text_field": rr["ce_text"], "text_fallbacks": fallbacks, "pinned_revision": rr["ce_revision"],
        "graph_version": data.version,
        "candidates": {**kw, "n": n_score}, "limit": a.limit, "seconds": round(time.time() - t0, 1),
        "probe": bool(a.probe),
        "inputs": {"pairs": str(a.pairs), "pairs_sha256": _sha(a.pairs), "rankings": str(a.rankings),
                   "rankings_sha256": _sha(lt), **test_in}})
    print(json.dumps({k: meta[k] for k in ("model", "revision", "dtype", "device", "n_queries", "n_pairs", "seconds",
                                           "out_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
