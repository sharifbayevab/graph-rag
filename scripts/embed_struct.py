"""M7 Task 2: bge-m3 vectors of the structure fragments (docs/m7 design §3.2), on Spark.

    PYTHONPATH=. python scripts/spark.py run --name m7-struct-vecs -- scripts/embed_struct.py \
        --out runs/m7-struct-vecs/struct_vecs.npz
    (pull; copy to data/hard/struct_vecs.npz; grl.env.struct_data.load_struct_env_data reads it)
Task 6c: --articles embeds the STRUCT/<article> fragments instead (copy to data/hard/struct_art_vecs.npz; read by
load_struct_env_data(articles=True)); same encoder, same checks, same npz layout.

Encoding = the index build's: graphrag.data.embed.Encoder (config/model.yaml name / revision / prefix /
max_seq_length, normalize_embeddings=True) .encode_passages(strict=True) over the corpus-format index strings of
grl.env.struct_data.build_struct_inputs. Sanity: --check-corpus N corpus chunks (evenly spaced) are re-encoded the
same way and compared with data/graph/chunk_vecs.npy (= the Qdrant index vectors).
npz: ids, vecs [S, 1024] float32, tokens, texts_sha256, index_sha256, graph_version, model, revision; plus
<out>.meta.json (counts, norms, corpus cosine)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402
from grl.env.struct_data import build_struct_inputs, texts_digest  # noqa: E402


def corpus_check(enc, n: int) -> dict:
    from graphrag.data.ingest import read_records
    from graphrag.kg.build import _cached_vectors, index_text_digest
    records = read_records(config.path(config.load("data")["records_dir"]))
    field = config.load("data")["embed_field"]
    chunks = sorted((c for r in records for c in r.chunks), key=lambda c: c.chunk_id)
    vec = _cached_vectors(config.path("data/graph/chunk_vecs.npy"), [c.chunk_id for c in chunks],
                          index_text_digest(records), "index_text vectors")
    pick = [chunks[i] for i in np.linspace(0, len(chunks) - 1, n).round().astype(int)]
    got = enc.encode_passages([getattr(c, field) or c.text_norm for c in pick], strict=True)
    cos = np.einsum("ij,ij->i", got.astype(np.float64), np.stack([vec[c.chunk_id] for c in pick]).astype(np.float64))
    return {"n": n, "cos_min": round(float(cos.min()), 6), "cos_mean": round(float(cos.mean()), 6)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help="default: cuda if available, else the Encoder default")
    ap.add_argument("--check-corpus", type=int, default=64)
    ap.add_argument("--articles", action="store_true", help="Task 6c: the STRUCT/<article> fragments")
    a = ap.parse_args(argv)
    import torch
    from graphrag.data.embed import Encoder
    t0 = time.time()
    texts, index, tokens, ver = build_struct_inputs(articles=a.articles)
    ids = sorted(texts)
    cfg = config.load("model")["encoder"]
    dev = a.device or ("cuda" if torch.cuda.is_available() else None)
    enc = Encoder(device=dev)
    vecs = enc.encode_passages([index[i] for i in ids], strict=True, progress=False)
    norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
    if vecs.shape != (len(ids), int(cfg["dim"])) or not np.isfinite(vecs).all():
        raise SystemExit(f"bad vectors {vecs.shape}")
    if np.abs(norms - 1).max() > 1e-3:
        raise SystemExit(f"vectors not unit norm: {norms.min()}..{norms.max()}")
    meta = {"graph_version": ver, "n": len(ids), "dim": int(vecs.shape[1]), "model": enc.model_name,
            "revision": cfg.get("revision"), "device": str(enc._m.device), "max_seq_length": enc.max_seq_length,
            "n_ext": sum(i.startswith("STRUCT/EXT/") for i in ids), "articles": bool(a.articles),
            "norm_min": round(float(norms.min()), 6), "norm_max": round(float(norms.max()), 6),
            "tokens_max": max(tokens.values()), "texts_sha256": texts_digest(texts),
            "index_sha256": texts_digest(index),
            "corpus_check": corpus_check(enc, a.check_corpus) if a.check_corpus else None,
            "torch": torch.__version__}
    out = config.path(a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), vecs=vecs.astype(np.float32), tokens=np.array([tokens[i] for i in ids], np.int32),
             texts_sha256=np.array(meta["texts_sha256"]), index_sha256=np.array(meta["index_sha256"]),
             graph_version=np.array(ver), model=np.array(enc.model_name), revision=np.array(str(cfg.get("revision"))))
    from grl.hashing import sha256_file
    meta["npz_sha256"] = sha256_file(out)
    meta["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
