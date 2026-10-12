"""EnvData + the M7 structure fragments as extra chunk rows at the end (docs/m7 design §3.2). μ⁻¹ is unchanged, so no
M3 graph operator and no ranking ever yields a STRUCT row; only the `structure` operator adds them.

Rows [0, n_base_chunks) are the corpus chunks in pyg.pt order (sorted ids); rows [n_base_chunks, n_chunks) are the
STRUCT/<root> fragments in sorted id order. chunk_ids is therefore NOT globally sorted -- nothing downstream of
EnvData.build relies on that (only build() checks it, and extend_with_struct does not go through build()).

Vectors (scripts/embed_struct.py, Spark): bge-m3 via graphrag.data.embed.Encoder.encode_passages(strict=True), the
call the index build makes (config/model.yaml prefix / max_seq_length, L2-normalised). The embedded string has the
corpus index_text format (graphrag.data.chunker): to_internal(breadcrumb) + "\\n" + to_internal(text), where a
document-level breadcrumb is the root title, capped like chunker._breadcrumb (BREADCRUMB_MAX_TOKENS, leading words
dropped, prefix + text + 2 special tokens <= ENCODER_MAX_TOKENS). `lang` of a document root = its record's
meta.lang[0]; an EXT root has no record and uses "uz"."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np

from grl.env.structure import PREFIX, build_article_struct_texts, build_struct_texts, struct_doc

DEFAULT_CACHE = "data/hard/struct_vecs.npz"
DEFAULT_ART_CACHE = "data/hard/struct_art_vecs.npz"      # Task 6c: STRUCT/<article> rows (embed_struct.py --articles)
EXT_LANG = "uz"


def extend_with_struct(data, texts: dict, vecs: dict, tokens: dict):
    ids = sorted(texts)
    roots = [i[len(PREFIX):] for i in ids]
    bad = [r for r in roots if r not in data.node_row]
    if bad:
        raise ValueError(f"structure roots not in the graph: {bad[:5]}")
    clash = [c for c in ids if c in data.chunk_row]
    if clash:
        raise ValueError(f"structure ids already chunk rows (extended twice?): {clash[:5]}")
    n0 = data.n_chunks
    e = replace(data,
                chunk_ids=list(data.chunk_ids) + ids,
                chunk_row={**data.chunk_row, **{c: n0 + i for i, c in enumerate(ids)}},
                chunk_node=np.concatenate([data.chunk_node, np.array([data.node_row[r] for r in roots], np.int64)]),
                chunk_tokens=np.concatenate([data.chunk_tokens, np.array([tokens[c] for c in ids], np.int32)]),
                chunk_vecs=np.concatenate([data.chunk_vecs,
                                           np.stack([vecs[c] for c in ids]).reshape(len(ids), -1).astype(np.float32)]))
    e.n_base_chunks = n0
    e.struct_rows = {data.node_row[r]: n0 + i for i, r in enumerate(roots)}
    return e


def extend_with_articles(data, texts: dict, vecs: dict, tokens: dict):
    """Task 6c: the STRUCT/<article> rows appended AFTER the document STRUCT rows of extend_with_struct (whose ids,
    rows and order stay as they are); n_base_chunks and struct_rows are kept, struct_art_rows = article node row ->
    its chunk row. μ⁻¹ unchanged: only the `structure` operator reaches these rows."""
    if not hasattr(data, "struct_rows") or not hasattr(data, "n_base_chunks"):
        raise ValueError("extend_with_articles needs the document STRUCT rows first (extend_with_struct)")
    if getattr(data, "struct_art_rows", None):
        raise ValueError("article STRUCT rows already present (extended twice?)")
    ids = sorted(texts)
    arts = [i[len(PREFIX):] for i in ids]
    bad = [a for a in arts if a not in data.node_row]
    if bad:
        raise ValueError(f"article ids not in the graph: {bad[:5]}")
    roots = {data.node_ids[r] for r in data.struct_rows}
    bad = [a for a in arts if a in roots or struct_doc(PREFIX + a) not in roots]
    if bad:
        raise ValueError(f"article rows must lie under a document root and not be one: {bad[:5]}")
    clash = [c for c in ids if c in data.chunk_row]
    if clash:
        raise ValueError(f"article structure ids already chunk rows: {clash[:5]}")
    n0, nb, srows = data.n_chunks, data.n_base_chunks, dict(data.struct_rows)
    e = replace(data,
                chunk_ids=list(data.chunk_ids) + ids,
                chunk_row={**data.chunk_row, **{c: n0 + i for i, c in enumerate(ids)}},
                chunk_node=np.concatenate([data.chunk_node, np.array([data.node_row[a] for a in arts], np.int64)]),
                chunk_tokens=np.concatenate([data.chunk_tokens, np.array([tokens[c] for c in ids], np.int32)]),
                chunk_vecs=np.concatenate([data.chunk_vecs,
                                           np.stack([vecs[c] for c in ids]).reshape(len(ids), -1).astype(np.float32)]))
    e.n_base_chunks, e.struct_rows = nb, srows
    e.struct_art_rows = {data.node_row[a]: n0 + i for i, a in enumerate(arts)}
    return e


def texts_digest(texts: dict) -> str:
    """sha256 of id \\0 text \\n in id order (the graphrag.kg.build.index_text_digest layout)."""
    h = hashlib.sha256()
    for i in sorted(texts):
        h.update(i.encode("utf-8") + b"\0" + texts[i].encode("utf-8") + b"\n")
    return h.hexdigest()


def _drop_leading(words: list[str], cap: int, count) -> str:
    lo, hi = 0, len(words)                     # smallest number of leading words to drop (chunker._breadcrumb)
    while lo < hi:
        mid = (lo + hi) // 2
        if count(" ".join(words[mid:])) <= cap:
            hi = mid
        else:
            lo = mid + 1
    return " ".join(words[lo:])


def struct_index_text(title: str, text: str, lang: str, count) -> str:
    from graphrag.data.chunker import BREADCRUMB_MAX_TOKENS, ENCODER_MAX_TOKENS, PASSAGE_PREFIX
    from graphrag.data.normalize import to_internal
    norm = to_internal(text, lang)
    cap = min(BREADCRUMB_MAX_TOKENS, ENCODER_MAX_TOKENS - 2 - count(PASSAGE_PREFIX) - 1 - count(norm))
    crumb = to_internal(" ".join(title.split()), lang) if cap > 0 else ""
    if crumb and count(crumb) > cap:
        crumb = _drop_leading(crumb.split(), cap, count)
    while crumb and count(f"{PASSAGE_PREFIX}{crumb}\n{norm}") + 2 > ENCODER_MAX_TOKENS:
        crumb = crumb.split(None, 1)[1] if " " in crumb.strip() else ""   # tokenization is not additive
    return f"{crumb}\n{norm}" if crumb else norm


def _jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_struct_inputs(version: str | None = None, latest_path: str = "data/graph/latest.json", count=None,
                        articles: bool = False):
    """(texts, index_texts, tokens, version) of every root of the pinned graph, deterministic. texts = what the agent
    reads (Task 1, <= data.yaml max_tokens by the corpus tokenizer); index_texts = what bge-m3 encodes.
    articles=True (Task 6c): the STRUCT/<article> fragments instead (grl.env.structure.build_article_struct_texts;
    breadcrumb = the article's document title, lang = the document's)."""
    from graphrag import config
    from graphrag.data.ingest import read_records
    latest = json.loads(config.path(latest_path).read_text(encoding="utf-8"))
    want = version or config.load("train")["env"]["graph_version"]
    if latest["version"] != want:
        raise ValueError(f"graph {latest['version']} in {latest_path} != pinned {want} (train.yaml env.graph_version)")
    d = config.path(latest["dir"])
    for f in ("nodes.jsonl", "edges.jsonl"):
        if hashlib.sha256((d / f).read_bytes()).hexdigest() != latest["sha256"][f]:
            raise ValueError(f"{d / f}: sha256 differs from {latest_path}")
    dc = config.load("data")
    if count is None:
        from graphrag.data.tokens import TokenCounter
        count = TokenCounter(dc["tokenizer"], dc.get("tokenizer_revision"))
    nodes, edges = _jsonl(d / "nodes.jsonl"), _jsonl(d / "edges.jsonl")
    build = build_article_struct_texts if articles else build_struct_texts
    texts = build(nodes, edges, count, int(dc["max_tokens"]))
    langs = {r.doc_id: (r.meta.lang[0] if r.meta.lang else EXT_LANG)
             for r in read_records(config.path(dc["records_dir"]))}
    title = {n["node_id"]: n.get("title") or "" for n in nodes if n["kind"] == "document"}
    index, tokens = {}, {}
    for sid, t in texts.items():
        root = struct_doc(sid)
        index[sid] = struct_index_text(title[root], t, langs.get(root, EXT_LANG), count)
        tokens[sid] = int(count(t))
    return texts, index, tokens, want


def _checked_cache(cache: str, data, articles: bool):
    """(texts, vecs by id, tokens) of a STRUCT vector cache, refused when its ids / texts / embedded strings / graph /
    token counts / shape differ from the ones rebuilt now."""
    from graphrag import config
    texts, index, tokens, ver = build_struct_inputs(data.version, articles=articles)
    z = np.load(config.path(cache), allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    if ids != sorted(texts):
        raise ValueError(f"{cache}: {len(ids)} ids differ from the {len(texts)} rebuilt structure ids")
    if str(z["graph_version"]) != ver:
        raise ValueError(f"{cache}: built on graph {z['graph_version']}, the environment uses {ver}")
    if str(z["texts_sha256"]) != texts_digest(texts):
        raise ValueError(f"{cache}: texts_sha256 {z['texts_sha256']} != rebuilt {texts_digest(texts)} -- rerun "
                         "scripts/embed_struct.py")
    if str(z["index_sha256"]) != texts_digest(index):
        raise ValueError(f"{cache}: index_sha256 differs from the rebuilt embedded strings -- rerun embed_struct.py")
    if [int(x) for x in z["tokens"]] != [tokens[i] for i in ids]:
        raise ValueError(f"{cache}: token counts differ from the corpus tokenizer's")
    vecs = z["vecs"]
    if vecs.shape != (len(ids), data.chunk_vecs.shape[1]):
        raise ValueError(f"{cache}: vecs {vecs.shape} for {len(ids)} ids, dim {data.chunk_vecs.shape[1]}")
    return texts, {i: vecs[k] for k, i in enumerate(ids)}, tokens


def load_struct_env_data(cache: str = DEFAULT_CACHE, version: str | None = None, articles: bool = False,
                         art_cache: str = DEFAULT_ART_CACHE):
    """load_env_data() + the document STRUCT rows; articles=True (Task 6c) + the STRUCT/<article> rows after them
    (art_cache). Each vector cache is refused when it differs from the inputs rebuilt now. articles=False is the
    pilot 7 / 8 data exactly (same rows, same order)."""
    from graphrag.env.data import load_env_data
    data = load_env_data(version)
    e = extend_with_struct(data, *_checked_cache(cache, data, False))
    if articles:
        e = extend_with_articles(e, *_checked_cache(art_cache, data, True))
    return e
