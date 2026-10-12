"""Node features X (IMPLEMENTATION.md §3.2 "сформировать признаки X", §6.1 h⁰): one bge-m3 passage vector
per graph vertex, from a short vertex text (M2 Task 9, Ruling R24).

Vertex text (internal representation of graphrag.data.normalize.to_internal: Latin Uzbek, ISO dates,
lower case -- the same space as the chunk index_text and the queries):
  document  -- title + section · document type;
  element   -- breadcrumb (document title › … › the element's label, graphrag.data.chunker._label) + title +
               the first `body_tokens` (~300) tokens of its own text span;
  external act (document, external) -- its display name;
  stub (element, external) -- "<act> › N-модда › …" (the stub chain under its act);
  entity    -- name + type word; concept -- term – definition.
Every text is cut to fit the encoder (`encode_passages(strict=True)` never truncates silently).

Vectors are cached in data/graph/node_vecs.npy (+ node_vecs_keys.json: the sha256 of each row's text, the
model name), keyed by the sha256 of the text: a rebuild re-encodes only new / changed texts, and X of a
graph is read back from the cache in node order (the same bytes on every build)."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from graphrag.data.normalize import to_internal
from graphrag.kg.schema import GraphNode

log = logging.getLogger(__name__)

DEFAULTS = {"body_tokens": 300, "title_tokens": 200, "crumb_tokens": 96, "max_tokens": 510,
            "entity_type_words": {"organ": "davlat organi", "position": "lavozim", "procedure": "tartib-taomil",
                                  "legal_category": "huquqiy kategoriya"}}


# --- token budget ---------------------------------------------------------------------------------

class WordCutter:
    """Whitespace "tokens" (tests / no tokenizer): head(s, n) keeps the first n words."""
    name = "words"

    def count(self, s: str) -> int:
        return len(s.split())

    def head(self, s: str, n: int) -> str:
        ws = s.split()
        return s if len(ws) <= n else " ".join(ws[:n])


class TokenCutter:
    """Encoder tokenizer (bge-m3): head(s, n) = the prefix of s covered by its first n tokens (offsets).
    Whole element bodies are tokenized to be cut (body_tokens / max_tokens), so a long body exceeds the model's
    8192 positions by design: the tokenizer's "sequence length is longer than the maximum" warning is silenced
    (verbose=False) -- nothing longer than max_tokens ever reaches the encoder (Ruling R26 (7))."""

    def __init__(self, tokenizer, name: str = "tokenizer"):
        self.tok = tokenizer
        self.name = name

    def count(self, s: str) -> int:
        return len(self.tok(s, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])

    def head(self, s: str, n: int) -> str:
        enc = self.tok(s, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
        offs = enc["offset_mapping"]
        if len(offs) <= n:
            return s
        return s[:offs[n - 1][1]].rstrip() if n > 0 else ""


def cutter_for(encoder) -> WordCutter | TokenCutter:
    """The encoder's own tokenizer when it has one (graphrag.data.embed.Encoder), else words."""
    m = getattr(encoder, "_m", None)
    tok = getattr(m, "tokenizer", None) if m is not None else getattr(encoder, "tokenizer", None)
    return TokenCutter(tok, getattr(encoder, "model_name", "tokenizer")) if tok is not None else WordCutter()


# --- record index ---------------------------------------------------------------------------------

class RecordIndex:
    """Lookups over the corpus records for vertex texts, validation and μ⁻¹."""

    def __init__(self, records):
        from graphrag.data.mapping import mu_inverse
        self.records = sorted(records, key=lambda r: r.doc_id)
        self.by_doc = {r.doc_id: r for r in self.records}
        self.texts = {r.doc_id: r.text for r in self.records}
        self.struct: dict[str, tuple] = {}
        self.mu_inv: dict[str, list[str]] = {}
        self.chunk_parent: dict[str, str] = {}
        for r in self.records:
            for n in r.structure:
                self.struct[n.node_id] = (r, n)
            inv = mu_inverse(r.structure, r.chunks)
            for nid in sorted(inv):
                self.mu_inv[nid] = sorted(inv[nid])
            for c in r.chunks:
                self.chunk_parent[c.chunk_id] = c.parent
        self.chunk_ids = sorted(self.chunk_parent)


def _index(records) -> RecordIndex:
    return records if isinstance(records, RecordIndex) else RecordIndex(records)


def _lang(rec) -> str:
    return "ru" if (rec.meta.lang or [""])[0] == "ru" else "uz"


def _ws(s: str) -> str:
    return " ".join((s or "").split())


def _crumb(idx: RecordIndex, nid: str) -> str:
    from graphrag.data.chunker import _label
    rec, n = idx.struct[nid]
    dt = SimpleNamespace(text=rec.text)
    chain = []
    while n is not None:
        chain.append(n)
        n = idx.struct[n.parent_id][1] if n.parent_id else None
    labels = []
    for k in reversed(chain):
        lb = _ws(rec.meta.title or k.title) if k.parent_id is None else _ws(_label(dt, k))
        if lb:
            labels.append(lb)
    return " › ".join(labels)


def node_text(node: GraphNode, records, cutter=None, cfg: dict | None = None,
              nodes: dict[str, GraphNode] | None = None) -> str:
    """The text encoded for `node` (see the module docstring). `records`: DocRecords or a RecordIndex;
    `nodes`: node_id -> GraphNode (for a stub's act title; without it the act id is used)."""
    c = {**DEFAULTS, **(cfg or {})}
    cut = cutter or WordCutter()
    idx = _index(records)
    if node.node_id in idx.struct and not node.external:
        rec, n = idx.struct[node.node_id]
        lang = _lang(rec)
        if n.parent_id is None:
            title = cut.head(to_internal(_ws(rec.meta.title or n.title), lang), c["title_tokens"])
            text = f"{title}\n{rec.meta.section.lower()} · {rec.meta.type}"
        else:
            crumb = to_internal(_crumb(idx, n.node_id), lang)
            if cut.count(crumb) > c["crumb_tokens"]:          # keep the element end of the path
                ws = crumb.split()
                while len(ws) > 1 and cut.count(" ".join(ws)) > c["crumb_tokens"]:
                    ws = ws[1:]
                crumb = " ".join(ws)
            title = to_internal(_ws(n.title), lang)
            head = crumb if not title or title in crumb else f"{crumb}\n{title}"
            body = to_internal(_ws(rec.text[n.start:n.end]), lang)
            last = crumb.rsplit(" › ", 1)[-1]
            if last and body.startswith(last) and len(body) > len(last):   # the header line is already in the crumb
                body = body[len(last):].lstrip(" .:")
            body = cut.head(body, c["body_tokens"])
            text = f"{head}\n{body}" if body else head
    elif node.kind == "document":                             # external act
        text = to_internal(_ws(node.title or node.node_id), "uz")
    elif node.kind == "element":                              # EXT stub chain: <act> › 27-modda › 2-qism
        nodes = nodes or {}
        parts, x, pid = [], node, node.node_id
        while x is not None and x.kind == "element":
            parts.append(_ws(x.title) or x.node_id.rsplit("/", 1)[-1])
            pid = x.node_id.rsplit("/", 1)[0]
            x = nodes.get(pid)
        act = nodes.get(node.attrs.get("act") or pid)
        parts.append(_ws(act.title) if act is not None else (node.attrs.get("act") or pid))
        text = to_internal(" › ".join(reversed(parts)), "uz")
    elif node.kind == "entity":
        word = c["entity_type_words"].get(node.type, node.type)
        text = f"{to_internal(_ws(node.title), 'uz')} · {word}"
    else:                                                     # concept
        lang = node.attrs.get("lang") or "uz"
        d = _ws(node.attrs.get("definition") or "")
        text = to_internal(f"{_ws(node.title)} – {d}" if d else _ws(node.title), "ru" if lang == "ru" else "uz")
    text = text.strip() or node.node_id
    return cut.head(text, c["max_tokens"]) if cut.count(text) > c["max_tokens"] else text


def text_sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# --- cache ------------------------------------------------------------------------------------------

class VecCache:
    """sha256(text) -> float32 vector, stored as <path>.npy + <path stem>_keys.json (rows sorted by sha)."""

    def __init__(self, path: Path | str | None, model: str):
        self.path = Path(path) if path else None
        self.model = model
        self.vecs: dict[str, np.ndarray] = {}
        self.loaded = 0
        if self.path is not None and self.path.exists() and self.keys_path.exists():
            meta = json.loads(self.keys_path.read_text(encoding="utf-8"))
            if meta.get("model") == model:
                X = np.load(self.path)
                if X.shape[0] == len(meta["keys"]):
                    self.vecs = {k: X[i] for i, k in enumerate(meta["keys"])}
                    self.loaded = len(self.vecs)
            else:
                log.warning("node vector cache %s is for model %r, not %r: ignored", self.path, meta.get("model"), model)

    @property
    def keys_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_keys.json")

    def save(self) -> None:
        if self.path is None:
            return
        keys = sorted(self.vecs)
        dim = next(iter(self.vecs.values())).shape[0] if keys else 0
        X = np.stack([self.vecs[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, dim), np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npy")
        np.save(tmp, X)
        tmp.replace(self.path)
        self.keys_path.write_text(json.dumps({"model": self.model, "dim": int(dim), "rows": len(keys), "keys": keys},
                                             ensure_ascii=False, indent=0) + "\n", encoding="utf-8")


def embed_texts(texts: list[str], encoder, cache: VecCache | None = None, round_decimals: int | None = None,
                batch: int = 4096) -> tuple[np.ndarray, dict]:
    """[len(texts), dim] float32 rows (unit norm), encoding only the texts missing from `cache` (in sha
    order, so the batches do not depend on the node order). Returns (X, info)."""
    cache = cache if cache is not None else VecCache(None, getattr(encoder, "model_name", "?"))
    shas = [text_sha(t) for t in texts]
    todo = sorted({s: t for s, t in zip(shas, texts) if s not in cache.vecs}.items())
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        V = encoder.encode_passages([t for _, t in part], strict=True)
        V = np.asarray(V, dtype=np.float32)
        if round_decimals is not None:
            V = np.round(V, round_decimals).astype(np.float32)
        for (s, _), v in zip(part, V):
            cache.vecs[s] = v
        if cache.path is not None:
            cache.save()                                      # progress survives an interruption
    dim = getattr(encoder, "dim", None) or (next(iter(cache.vecs.values())).shape[0] if cache.vecs else 0)
    X = np.stack([cache.vecs[s] for s in shas]).astype(np.float32) if shas else np.zeros((0, dim), np.float32)
    return X, {"texts": len(texts), "unique_texts": len(set(shas)), "encoded": len(todo),
               "cache_rows_loaded": cache.loaded, "cache_rows": len(cache.vecs)}


def embed_nodes(graph, encoder, cache: VecCache | None = None, cfg: dict | None = None,
                round_decimals: int | None = None) -> np.ndarray:
    """X [N, dim] in graph.nodes order (graph.rindex: the RecordIndex of its records); graph.feature_info
    gets the text / cache counts."""
    cut = cutter_for(encoder)
    by_id = {n.node_id: n for n in graph.nodes}
    texts = [node_text(n, graph.rindex, cut, cfg, by_id) for n in graph.nodes]
    X, info = embed_texts(texts, encoder, cache, round_decimals)
    graph.texts = texts
    graph.feature_info = {**info, "cutter": cut.name, "dim": int(X.shape[1]) if X.ndim == 2 else 0}
    return X
