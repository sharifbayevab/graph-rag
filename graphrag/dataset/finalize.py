"""Finalization before trajectories (M3-R3 пп. 7, 9, 11, 13): exact / cosine / 4-gram near-duplicate removal, leakage
guard against Q-real (qreal_exclusion; question cos >= qreal_question_cos_max) and across splits (every document of a
pair is assigned to the pair's split -- the document-group split of Task 10), a manual sample stratified over every
split × pattern × language cell for the INDEPENDENT reviewer, and the §8.1 records with complete provenance (incl. the
ds_facts cache keys the pair's facts came from)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from graphrag.data.normalize import query_internal
from graphrag.dataset.seeds import HOP
from graphrag.eval.gold import Anchor, text_sha, wsn


def _h(seed, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def question_vectors(encoder, texts: list[str]) -> np.ndarray:
    return encoder.encode_queries([query_internal(t) for t in texts]).astype(np.float64)


TEMPLATE_TOKENS = 3


def template(q: str) -> str:
    """The question opening (M3-R3 п. 7; ruling M3-R18 (b)): the first 3 tokens of the question in the internal form
    (query_internal: Uzbek Cyrillic -> Latin, lower case, canonical numbers), punctuation stripped, stop-words kept --
    deterministic and language-agnostic (Cyrillic and Latin spellings of one Uzbek opening are the same template)."""
    return " ".join(re.findall(r"\w+", query_internal(q))[:TEMPLATE_TOKENS])


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    def grams(s):
        w = re.findall(r"\w+", s.lower())
        return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(len(ga | gb), 1)


def _doc(cid: str) -> str:
    return cid.split("/", 1)[0].split("#", 1)[0]


def dedup_and_leak(cands, qv, real_qv, excluded: set[str], cfg: dict, seed,
                   split_of: dict[str, str] | None = None) -> tuple[list, dict[str, str]]:
    """Leakage first (split_leak, qreal_fragment, qreal_question), then duplicates (exact wsn, cos >= dedup_cos_max,
    word 4-gram Jaccard >= ngram_jaccard_max) against the pairs kept so far, pairs of scarcer split × pattern × language
    cells first, then sha256(seed:seed_id) order -- dedup is global, across splits. Returns (kept sorted by seed_id, {seed_id: reason})."""
    from collections import Counter
    size = Counter((c.seed.split, c.seed.pattern, c.seed.lang) for c in cands)
    cell = lambda c: (c.seed.split, c.seed.pattern, c.seed.lang)
    # scarcer cells first (review m5): of two near-duplicates the one from the thinner cell survives; then sha256
    order = sorted(range(len(cands)), key=lambda i: (size[cell(cands[i])], _h(seed, cands[i].seed.seed_id)))
    real_qv = np.asarray(real_qv, dtype=np.float64)
    kept, seen, dropped = [], set(), {}
    for i in order:
        c, sid = cands[i], cands[i].seed.seed_id
        docs = set(c.seed.docs) | {_doc(x) for x in c.seed.chunks}
        if split_of is not None and any(split_of.get(d) != c.seed.split for d in docs):
            dropped[sid] = "split_leak"
        elif set(c.seed.chunks) & excluded:
            dropped[sid] = "qreal_fragment"
        elif real_qv.size and float((real_qv @ qv[i]).max()) >= float(cfg["qreal_question_cos_max"]):
            dropped[sid] = "qreal_question"
        elif wsn(c.question) in seen or (kept and float((qv[kept] @ qv[i]).max()) >= float(cfg["dedup_cos_max"])) \
                or any(ngram_jaccard(c.question, cands[k].question) >= float(cfg["ngram_jaccard_max"]) for k in kept):
            dropped[sid] = "duplicate"
        else:
            seen.add(wsn(c.question))
            kept.append(i)
    return [cands[i] for i in sorted(kept, key=lambda i: cands[i].seed.seed_id)], dropped


def manual_sample(cands, n: int, seed) -> list:
    cells: dict[tuple, list] = {}
    for c in sorted(cands, key=lambda c: _h(seed, "manual|" + c.seed.seed_id)):
        cells.setdefault((c.seed.split, c.seed.pattern, c.seed.lang), []).append(c)
    out, keys = [], sorted(cells)
    while len(out) < n and any(cells.values()):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop(0))
    return sorted(out, key=lambda c: c.seed.seed_id)


def facts_cache_keys(cand, block_of: dict[str, str], block_chunks: dict[str, list[str]], chunks_by_id, index,
                     llm) -> dict[str, str]:
    """{block_id: ds_facts cache key} of the blocks holding the pair's C_ref: the input extract_block sends (same
    fragments, labels, act / where), keyed like KgLLM.cached_json (model, prompt_version, prompt digest, input)."""
    from graphrag.dataset.facts import PASS, PROMPT_FACTS, SCHEMA_FACTS
    from graphrag.kg.llm import _canon_input, prompt_digest
    digest = prompt_digest(PROMPT_FACTS, SCHEMA_FACTS)
    out = {}
    for b in sorted({block_of[c] for c in cand.seed.chunks}):
        frags = [{"id": f"C{i + 1}", "act": index.act[chunks_by_id[c].parent.split("/", 1)[0]],
                  "where": index.format(chunks_by_id[c].parent), "text": chunks_by_id[c].text}
                 for i, c in enumerate(block_chunks[b])]
        out[b] = llm.cache_key(PASS, _canon_input({"block": b, "fragments": frags}), digest)
    return out


REQUIRED_PROVENANCE = ("seed_id", "chunks", "walk", "fact_ids", "facts.prompt_digest", "facts.cache_keys",
                       "generate.prompt_digest", "generate.cache_key", "generate.prompt_version",
                       "verify.prompt_digest", "verify.cache_key", "verify.prompt_version", "verify.verdicts",
                       "verify.report", "verify.report.facts_recovered", "verify.report.ablation",
                       "copy.max_run", "copy.overlap", "seed")


def provenance_missing(pair) -> list[str]:
    """M3-R3 п. 11: the provenance paths a pair lacks (empty list = complete); an empty facts.cache_keys is missing."""
    out = []
    for path in REQUIRED_PROVENANCE:
        x = pair.provenance
        for k in path.split("."):
            x = x.get(k, None) if isinstance(x, dict) else None
        if x is None or (path == "facts.cache_keys" and not x):
            out.append(path)
    return out


@dataclass
class Pair:
    id: str
    q: str
    lang: str
    pattern: str
    hop: str
    y_ref: dict                                   # {"text", "sources"}
    C_ref: list[str]
    C_ref_anchors: list[Anchor]
    facts: list[dict]                             # [{fact_id, chunk_id, statement, value, quote}]
    docs: list[str]
    split: str
    section: str
    seed_id: str
    origin: str = "synthetic"
    trajectory: list[dict] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    action_equiv: list[list[int]] = field(default_factory=list)
    action_masks: list[list[int]] = field(default_factory=list)
    reward: dict | None = None
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Pair":
        return cls(**{**d, "C_ref_anchors": [Anchor(**a) for a in d["C_ref_anchors"]]})


def to_pair(cand, index, data, chunks_by_id: dict) -> Pair:
    s = cand.seed
    ref = sorted(s.chunks)
    anchors = [Anchor(_doc(c), chunks_by_id[c].start, chunks_by_id[c].end, text_sha(chunks_by_id[c].text)) for c in ref]
    prov = json.loads(json.dumps({**cand.provenance, "seed": s.to_json()}, ensure_ascii=False))   # JSON-stable
    from graphrag.dataset.opus import pair_id
    return Pair(pair_id(s.seed_id), cand.question, s.lang, s.pattern, HOP[s.pattern],
                {"text": cand.answer, "sources": index.sources_for(data, ref)}, ref, anchors,
                [asdict(f) for f in cand.facts], list(s.docs), s.split, s.section, s.seed_id, provenance=prov)


def write_pairs(path, pairs) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                         for x in sorted(pairs, key=lambda x: x.id)), encoding="utf-8")


def read_pairs(path) -> list[Pair]:
    return [Pair.from_json(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
