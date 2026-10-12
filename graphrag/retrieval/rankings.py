"""U9 per-query rankings (roadmap §4 U9). Task 3: the container; Task 11: build / write / load."""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graphrag.data.lemma import tokens
from graphrag.data.normalize import query_internal

TOP = 200


@dataclass
class QueryRanking:
    qkey: str                 # "<qid>|cyr" / "<qid>|lat" (Q-real), "<pair id>|q" (dataset)
    text: str
    qvec: np.ndarray          # float32 [d]
    lex: np.ndarray           # chunk rows, ranked
    lex_score: np.ndarray
    vec: np.ndarray           # chunk rows, ranked
    vec_score: np.ndarray
    title: np.ndarray         # node rows, ranked
    title_score: np.ndarray

    @classmethod
    def empty(cls, qkey: str, qvec, text: str = "") -> "QueryRanking":
        z, f = np.zeros(0, np.int64), np.zeros(0, np.float64)
        return cls(qkey, text, np.asarray(qvec, np.float32), z, f, z, f, z, f)


ARRAYS = ("qvec", "lex_idx", "lex_score", "vec_idx", "vec_score", "title_idx", "title_score")


class TitleIndex:
    """BM25 (data.yaml k1, b) over vertex titles in the internal representation (M2 requirement 3: headings are
    vertex text); in memory, summed in sorted term order, rounded to 9 decimals, ties by row."""

    def __init__(self, titles: list[str], k1: float, b: float):
        self.k1, self.b = k1, b
        docs = [Counter(tokens(query_internal(t))) if t else Counter() for t in titles]
        self.length = np.array([sum(d.values()) for d in docs], np.float64)
        nz = self.length > 0
        n = int(nz.sum())
        self.avgdl = max(float(self.length[nz].mean()) if n else 1.0, 1.0)
        df = Counter(t for d in docs for t in d)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, tf in sorted(d.items()):
                self.post[t].append((i, tf))

    def search(self, text: str, k: int) -> list[tuple[int, float]]:
        s: dict[int, float] = defaultdict(float)
        for t in sorted(set(tokens(query_internal(text)))):
            for i, tf in self.post.get(t, ()):
                s[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.length[i] / self.avgdl))
        return sorted(((i, round(v, 9)) for i, v in s.items() if v > 0), key=lambda x: (-x[1], x[0]))[:k]


def dense_topk(vecs64: np.ndarray, qvec, k: int, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    s = np.round(vecs64 @ np.asarray(qvec, np.float64), decimals)
    order = np.lexsort((np.arange(s.size), -s))[:k]
    return order.astype(np.int64), s[order]


def rrf_rows(a, b, k: int = 60, top: int = 100) -> np.ndarray:
    s: dict[int, float] = defaultdict(float)
    for lst in (a, b):
        for r, c in enumerate(np.asarray(lst)[:top].tolist()):
            s[int(c)] += 1.0 / (k + r + 1)
    return np.array([c for c, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))][:top], np.int64)


def queries_from_gold(items) -> list[tuple[str, str]]:
    out = []
    for it in items:
        out.append((f"{it.qid}|cyr", it.question_cyr))
        if it.question_lat:
            out.append((f"{it.qid}|lat", it.question_lat))
    return sorted(out)


def queries_from_pairs(pairs) -> list[tuple[str, str]]:
    return sorted((f"{p.id}|q", p.q) for p in pairs)


def _pad(rows, scores, top: int):
    i = np.full(top, -1, np.int32)
    s = np.full(top, np.nan, np.float32)
    i[:len(rows)] = np.asarray(rows, np.int32)[:top]
    s[:len(scores)] = np.asarray(scores, np.float32)[:top]
    return i, s


def build_rankings(queries, data, conn, encoder, titles: TitleIndex, top: int = TOP) -> dict[str, np.ndarray]:
    from graphrag.retrieval import lexical
    Q = len(queries)
    qvec = encoder.encode_queries([query_internal(t) for _, t in queries]).astype(np.float32)
    out = {"qvec": qvec, **{n: np.zeros((Q, top), np.int32 if n.endswith("idx") else np.float32) for n in ARRAYS[1:]}}
    for i, (_, text) in enumerate(queries):
        lx = lexical.search(conn, text, k=top)
        out["lex_idx"][i], out["lex_score"][i] = _pad([data.chunk_row[c] for c, _ in lx], [s for _, s in lx], top)
        rows, s = dense_topk(data.vecs64, qvec[i], top)
        out["vec_idx"][i], out["vec_score"][i] = _pad(rows, s, top)
        tl = titles.search(text, top)
        out["title_idx"][i], out["title_score"][i] = _pad([r for r, _ in tl], [s for _, s in tl], top)
    return out


def _npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a))
    return buf.getvalue()


def write_rankings(root, queries, arrays: dict, inputs: dict) -> str:
    from graphrag import config
    root = Path(root)
    files = {"queries.jsonl": "".join(json.dumps({"qkey": k, "text": t}, ensure_ascii=False) + "\n"
                                      for k, t in queries).encode("utf-8"),
             **{f"{n}.npy": _npy(arrays[n]) for n in ARRAYS}}
    h = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for name in sorted(files):
        h.update(name.encode() + hashlib.sha256(files[name]).digest())
    ver = h.hexdigest()[:12]
    files["meta.json"] = (json.dumps({"version": ver, "inputs": inputs, "n_queries": len(queries)},
                                     ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    d = root / ver
    d.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        (d / name).write_bytes(b)
    rel = str(d.relative_to(config.ROOT)) if d.is_relative_to(config.ROOT) else str(d)
    (root / "latest.json").write_text(json.dumps({"version": ver, "dir": rel, "sha256": {
        n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}}, ensure_ascii=False, indent=1,
        sort_keys=True) + "\n", encoding="utf-8")
    return ver


class Rankings:
    def __init__(self, d: Path, meta: dict):
        self.dir, self.meta = d, meta
        qs = [json.loads(l) for l in (d / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self._keys = [q["qkey"] for q in qs]
        self._text = {q["qkey"]: q["text"] for q in qs}
        self._pos = {k: i for i, k in enumerate(self._keys)}
        self._a = {n: np.load(d / f"{n}.npy") for n in ARRAYS}

    def keys(self) -> list[str]:
        return list(self._keys)

    def get(self, qkey: str) -> QueryRanking:
        i = self._pos[qkey]

        def cut(name):
            idx, sc = self._a[f"{name}_idx"][i], self._a[f"{name}_score"][i]
            m = idx >= 0
            return idx[m].astype(np.int64), sc[m].astype(np.float64)

        (lx, ls), (vx, vs), (tx, ts) = cut("lex"), cut("vec"), cut("title")
        return QueryRanking(qkey, self._text[qkey], self._a["qvec"][i], lx, ls, vx, vs, tx, ts)


QREAL_LATEST = "data/rankings/qreal/latest.json"
DATASET_LATEST = "data/rankings/dataset/latest.json"


class RankingsUnion:
    """Several ranking artifacts behind one keys() / get() (key sets are disjoint by construction)."""

    def __init__(self, parts: list):
        self.parts = parts
        self._owner = {}
        for r in parts:
            for k in r.keys():
                if k in self._owner:
                    raise ValueError(f"query key {k} in two ranking artifacts")
                self._owner[k] = r
        self.meta = {"version": "+".join(r.meta["version"] for r in parts)}

    def keys(self) -> list[str]:
        return sorted(self._owner)

    def get(self, qkey: str):
        return self._owner[qkey].get(qkey)


def load_union(graph_version: str | None = None) -> RankingsUnion:
    return RankingsUnion([load_rankings(QREAL_LATEST, graph_version), load_rankings(DATASET_LATEST, graph_version)])


def load_rankings(latest=QREAL_LATEST, graph_version: str | None = None) -> Rankings:
    from graphrag import config
    lf = Path(latest) if Path(str(latest)).is_absolute() else config.path(str(latest))
    lt = json.loads(lf.read_text(encoding="utf-8"))
    d = Path(lt["dir"]) if Path(lt["dir"]).is_absolute() else config.path(lt["dir"])
    for name, sha in lt["sha256"].items():
        if hashlib.sha256((d / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"{d / name}: sha256 differs from {lf}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    gv = meta["inputs"].get("graph_version")
    if graph_version is not None and gv != graph_version:
        raise ValueError(f"rankings were built on graph {gv}, the environment uses {graph_version}")
    return Rankings(d, meta)


def qdrant_agreement(cl, coll: str, arrays: dict, chunk_ids: list[str], ks=(10, 200)) -> dict:
    from graphrag.retrieval import vector
    ov = {k: [] for k in ks}
    for i in range(arrays["qvec"].shape[0]):
        ann = [c for c, _ in vector.search(cl, coll, arrays["qvec"][i], k=max(ks))]
        exact = [chunk_ids[r] for r in arrays["vec_idx"][i] if r >= 0]
        for k in ks:
            ov[k].append(len(set(ann[:k]) & set(exact[:k])) / max(len(exact[:k]), 1))
    return {f"overlap@{k}": round(float(np.mean(v)), 4) for k, v in ov.items()}
