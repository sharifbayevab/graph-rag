"""gold_complete (M3 Task 12, D6): for the Q-real TEST set, gpt-4.1 names the fragments of the SAME document without
which the reference answer is not fully supported; each addition quotes a verbatim span (>= 20 normalized characters)
of that candidate, else the whole answer is rejected. Cached in cache/gold/, API only with GRL_GOLD_API=1."""
from __future__ import annotations

from dataclasses import replace

from graphrag.eval.gold import Anchor, GoldItem, norm_map, text_sha
from graphrag.retrieval.rankings import rrf_rows

PASS = "gold_complete"
PROMPT = """You check the gold evidence of a question-answer pair over Uzbek / Russian legal documents.
You get a QUESTION, its reference ANSWER, the CORE fragment(s) that contain the answer's supporting quote, and up to
six CANDIDATE fragments of the same document.
1. core_supports: true if the CORE fragments state what the ANSWER asserts (possibly incompletely); false if they do
   not support the ANSWER at all.
2. needed: the CANDIDATE fragments WITHOUT which the ANSWER cannot be fully supported -- e.g. the article number, the
   act title or the organ the ANSWER cites appears only in that candidate. For each, copy a verbatim evidence span of
   at least 20 characters, exactly as written in that candidate. Do not list candidates that only repeat, paraphrase
   or add background. If nothing is needed, return an empty list.
Answer only with the JSON object."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["core_supports", "needed"],
          "properties": {"core_supports": {"type": "boolean"},
                         "needed": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["id", "evidence"],
                             "properties": {"id": {"type": "string"}, "evidence": {"type": "string"}}}}}}
MIN_EVIDENCE = 20


def verify(ans: dict, cand_text: dict[str, str]) -> tuple[bool, str | None]:
    for n in ans.get("needed", []):
        t = cand_text.get(n["id"])
        if t is None:
            return False, "unknown_id"
        ev = norm_map(n["evidence"])[0]
        if len(ev) < MIN_EVIDENCE:
            return False, "short_evidence"
        if ev not in norm_map(t)[0]:
            return False, "evidence_not_in_candidate"
    return True, None


def candidates(item: GoldItem, rec, rank, data, max_n: int = 6) -> list[str]:
    """Same-document candidates: the neighbours of each core chunk (document order), then the item's RRF(lex, vec)
    ranking restricted to the document; core chunks excluded; at most max_n."""
    cs = sorted(rec.chunks, key=lambda c: c.start)
    pos = {c.chunk_id: i for i, c in enumerate(cs)}
    core, out = set(item.core), []
    for cid in item.core:
        for j in (pos[cid] - 1, pos[cid] + 1):
            if 0 <= j < len(cs) and cs[j].chunk_id not in core and cs[j].chunk_id not in out:
                out.append(cs[j].chunk_id)
    if rank is not None:
        for r in rrf_rows(rank.lex, rank.vec, top=200).tolist():
            cid = data.chunk_ids[r]
            if cid in pos and cid not in core and cid not in out:
                out.append(cid)
    return out[:max_n]


def complete_one(llm, question, answer, core: dict, cands: dict, ids: dict):
    why = {}

    def vf(a):
        ok, r = verify(a, cands)
        why["r"] = r
        return ok, r

    inp = {"question": question, "answer": answer, "core": [{"id": k, "text": v} for k, v in core.items()],
           "candidates": [{"id": k, "text": v} for k, v in cands.items()]}
    ans = llm.cached_json(PASS, inp, SCHEMA, vf, prompt=PROMPT)
    if ans is None:
        return [], None, why.get("r") or "bad_json"
    return sorted({ids[n["id"]] for n in ans["needed"]}), bool(ans["core_supports"]), None


def complete(items: list[GoldItem], records, ranks, data, llm) -> tuple[list[GoldItem], dict]:
    recs = {r.doc_id: r for r in records}
    keys = set(ranks.keys())
    st = {"items": 0, "with_candidates": 0, "added": 0, "items_with_support": 0, "rejected": 0, "core_not_supporting": []}
    out = []
    for it in sorted(items, key=lambda i: i.qid):
        st["items"] += 1
        rec = recs[it.doc_id]
        by_id = {c.chunk_id: c for c in rec.chunks}
        rank = ranks.get(f"{it.qid}|cyr") if f"{it.qid}|cyr" in keys else None
        cand = candidates(it, rec, rank, data)
        if not cand:
            out.append(it)
            continue
        st["with_candidates"] += 1
        ids = {f"K{i + 1}": c for i, c in enumerate(cand)}
        added, core_ok, reason = complete_one(llm, it.question_cyr, it.gold_answer,
                                              {f"G{i + 1}": by_id[c].text for i, c in enumerate(it.core)},
                                              {k: by_id[c].text for k, c in ids.items()}, ids)
        st["rejected"] += bool(reason)
        if core_ok is False:
            st["core_not_supporting"].append(it.qid)
        sup = [Anchor(rec.doc_id, by_id[c].start, by_id[c].end, text_sha(by_id[c].text)) for c in added]
        st["added"] += len(added)
        st["items_with_support"] += bool(added)
        out.append(replace(it, support_anchors=sup, support=added))
    return out, st
