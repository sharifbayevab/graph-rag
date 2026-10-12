"""ds_verify (decisions M3-R10, M3-R3 пп. 4, 6, 8; ruling M3-R17): an independent gpt-4.1 judge that does NOT see
the generator's answer or facts. Per view (full C_ref; for multi-hop also every C_ref minus one fragment) it says
whether the question is answerable and quotes the spans it needs. Code accepts a pair iff (a) on the full view every
used fact's value lies, as whole words, in a verbatim judge quote (length-capped) from that fact's fragment and (b)
every ablated view is unanswerable.

Batching (M3-R17 / review I1): a call never holds two views of the same question, nor a plant together with its base
pair; fragments are renumbered F1..Fk per view in a seeded shuffle (no id gaps, no walk order) and view keys are
opaque (V1..Vn), so a reduced view cannot be told apart by structure. Calls are packed pilot-first (review I4): the
pilot's candidates are planned and packed on their own, so the full run replays the pilot's calls from the cache.

Plants (review I2): sensitivity is scored on the JUDGE's own verdict for JUDGE_KINDS --
- question_swap: another pair's question (same document, disjoint C_ref) over this pair's fragments -> must be
  unanswerable;
- padded_single: a single-hop pair plus one unrelated fragment, shown as a 2-hop pair (full / drop:A / drop:B) -> the
  view with only the real fragment must be answerable AND the view with only the pad unanswerable (tests п. 6: a judge
  that calls every reduced view unanswerable, or every view answerable, fails it);
- neighbor_swap (third judge-level kind, proposed here): one fragment replaced by an ADJACENT fragment of the same
  document that does not contain that fragment's fact values -- same act, same topic, the needed fact is absent from
  every fragment -> must be unanswerable (the hard, near-miss version of chunk_swap).
CODE_KINDS (chunk_swap, value_swap) are kept as checks of the code rule on judge quotes; they are reported, not gated.
Gate: every judge kind has >= min_kind_n plants and point sensitivity >= min_sensitivity_kind."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict

from graphrag.eval.gold import norm_map, wsn
from graphrag.kg.audit import wilson

PASS = "ds_verify"
JUDGE_KINDS = ("question_swap", "padded_single", "neighbor_swap")
CODE_KINDS = ("chunk_swap", "value_swap")
PLANT_KINDS = ("chunk_swap", "value_swap", "question_swap", "padded_single", "neighbor_swap")
KIND_WEIGHT = {"chunk_swap": 1, "value_swap": 1, "question_swap": 2, "padded_single": 2, "neighbor_swap": 2}
PROMPT_VER = """You independently check questions for a legal retrieval dataset (Uzbek / Russian).
Each VIEW gives a QUESTION and some FRAGMENTS (id, act, where, text). Using ONLY these fragments:
- answerable: true only if the fragments contain everything needed to answer the question FULLY;
- evidence: ONE quote per needed value, with its fragment id: the shortest span of that fragment that contains the
  value (one clause), at most 300 characters -- longer quotes are ignored; copied character by character in the
  fragment's own script and spelling (never transliterated or translated), never shortened with "…" or "...";
  never a whole fragment;
- answer: your answer from these fragments only, in the question's language, stating every value the question asks
  for (quote key values exactly as written in the fragments); if the fragments support only part of it, give that part;
- complete: true only if your answer fully answers EVERY part of the question; false if any part is missing, guessed or
  only partly supported by these fragments.
Judge every view on its own fragments only. Return {"views": [...]} with one entry per view key."""
SCHEMA_VER = {"type": "object", "additionalProperties": False, "required": ["views"], "properties": {"views": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["key", "answerable", "evidence", "answer", "complete"],
                               "properties": {"key": {"type": "string"}, "answerable": {"type": "boolean"},
                                              "answer": {"type": "string"}, "complete": {"type": "boolean"},
                                              "evidence": {"type": "array", "items": {
                                                  "type": "object", "additionalProperties": False,
                                                  "required": ["id", "quote"],
                                                  "properties": {"id": {"type": "string"},
                                                                 "quote": {"type": "string"}}}}}}}}}
_NO = {"answerable": False, "evidence": []}
# M3-R20 (M3-R3 п. 4 revised): the agreement pass -- the same independent judge model, a separate prompt, shown the
# candidate answer ONLY here (after the blind views): does the answer state the source values correctly?
PROMPT_AGREE = """You check answers of a legal question-answer dataset over Uzbek / Russian parliamentary documents.
Each ITEM gives a QUESTION, a candidate ANSWER and SOURCE spans copied verbatim from the documents (the values the
answer must convey and an independent judge's evidence). The answer may use another language or script than the
source: translation and transliteration are allowed.
- agrees: true only if the ANSWER conveys every SOURCE value correctly (same meaning, same numbers, same bodies and
  acts, nothing reversed, narrowed or widened), adds no claim the source spans do not support, and answers the QUESTION;
- reason: one short sentence.
Return {"items": [{"key", "agrees", "reason"}]} with exactly one entry per item key."""
SCHEMA_AGREE = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["key", "agrees", "reason"],
                               "properties": {"key": {"type": "string"}, "agrees": {"type": "boolean"},
                                              "reason": {"type": "string"}}}}}}


def needs_agreement(c) -> bool:
    """A pair whose values are not all code-checked (clause, translated name / term, number with reworded words)."""
    return any(v != "code" for v in (c.provenance.get("value_checks") or {}).values())


def _parts(cands, pilot_ids=(), parts=None) -> list[tuple[str, list]]:
    """[(part name, candidates in seed_id order)]: the pilot, then (M3-R23) every named part (e.g. one Opus generator
    packet) in name order, then the rest -- each part planned / packed on its own, so adding a part never changes
    the calls of another (their cache is reused). Without `parts` this is the M3-R17 pilot / rest split."""
    pilot, parts = set(pilot_ids), parts or {}
    of = lambda c: "pilot" if c.seed.seed_id in pilot else parts.get(c.seed.seed_id, "rest")
    names = ["pilot"] + sorted({of(c) for c in cands} - {"pilot", "rest"}) + ["rest"]
    return [(n, sorted((c for c in cands if of(c) == n), key=lambda c: c.seed.seed_id)) for n in names]


def agree_candidates(cands, llm, per_call: int, pilot_ids=(), parts=None) -> tuple[list, dict]:
    """The agreement pass over `cands` (blind-judge accepted): pairs that need it (needs_agreement), pilot part first,
    each part in seed_id order, `per_call` items per call (deterministic, so the full run replays the pilot's calls).
    A pair is kept iff it needs no agreement or the judge agrees. guard_total before every uncached call."""
    from graphrag.dataset.facts import guard_total
    from graphrag.kg.llm import _canon_input, prompt_digest
    digest = prompt_digest(PROMPT_AGREE, SCHEMA_AGREE)
    need = [c for c in cands if needs_agreement(c)]
    verdict, n_calls, bad = {}, 0, 0
    for _, cs in _parts(need, pilot_ids, parts):
        for i in range(0, len(cs), int(per_call)):
            chunk = cs[i:i + int(per_call)]
            items, back = [], {}
            for n, c in enumerate(chunk):
                vc = c.provenance.get("value_checks") or {}
                vals = [f.value for f in c.facts if vc.get(f.fact_id, "code") != "code"]
                ev = [e["quote"] for e in ((c.provenance.get("verify") or {}).get("verdicts", {}).get("full") or {})
                      .get("evidence", [])]
                key = f"A{n + 1}"
                back[key] = c
                items.append({"key": key, "question": c.question, "answer": c.answer,
                              "source": list(dict.fromkeys(vals + ev))})
            inp = {"items": items}
            guard_total(llm, PASS, inp, PROMPT_AGREE, SCHEMA_AGREE)
            ans = llm.cached_json(PASS, inp, SCHEMA_AGREE,
                                  lambda a: (sorted(x["key"] for x in a["items"]) == sorted(back), "keys"),
                                  prompt=PROMPT_AGREE)
            n_calls += 1
            if ans is None:
                bad += 1
                continue
            ck = llm.cache_key(PASS, _canon_input(inp), digest)
            for x in ans["items"]:
                verdict[back[x["key"]].seed.seed_id] = (bool(x["agrees"]), x["reason"], ck)
    kept, why = [], Counter()
    for c in cands:
        if not needs_agreement(c):
            kept.append(c)
            continue
        v = verdict.get(c.seed.seed_id)
        if v is None:
            why[f"{c.seed.pattern}|agree_bad_answer"] += 1
            continue
        c.provenance["agree"] = {"pass": PASS, "prompt_digest": digest, "cache_key": v[2], "agrees": v[0],
                                 "reason": v[1]}
        if v[0]:
            kept.append(c)
        else:
            why[f"{c.seed.pattern}|disagree"] += 1
    return kept, {"checked": len(need), "kept": len(kept), "rejected": dict(sorted(why.items())), "calls": n_calls,
                  "bad_calls": bad}


def _u(seed, key: str) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _cell(c) -> str:
    return f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}"


def _has(quote_n: str, value: str, lemma: bool = False) -> bool:
    """`value` occurs in the quote as whole canonical tokens (graphrag.dataset.facts.value_in): not "5" inside "15",
    "0,5" or "1 500" (Task 15 concern, pre-pilot re-review I1). Exact surface form by default -- a judge quote and a
    fact value are both verbatim from the fragment."""
    from graphrag.dataset.facts import value_in
    return bool(value) and value_in(value, quote_n, lemma=lemma)


def views(cand) -> list[tuple[str, list[str]]]:
    ids = [f"F{i + 1}" for i in range(len(cand.seed.chunks))]
    out = [("full", ids)]
    if len(ids) > 1:
        out += [(f"drop:{f}", [x for x in ids if x != f]) for f in ids]
    return out


def quote_ok(quote_n: str, text_n: str, cap=None) -> bool:
    """A judge quote counts only if verbatim in its fragment and, with cap = (max_chars, max_share, floor_chars), not
    longer than max_chars nor than max_share of the fragment (quotes up to floor_chars always pass): quoting whole
    fragments does not recover facts (review m1)."""
    if not quote_n:
        return False
    if ("…" in quote_n or "..." in quote_n) and quote_n not in text_n:   # M3-R30: an elided join is no quote
        return False
    if quote_n not in text_n:          # M3-R25 A: the judge re-scripted / re-typed a verbatim span (Cyrl <-> Latn)
        from graphrag.dataset.facts import ctoks, find_seq
        q = ctoks(quote_n)
        if not q or not find_seq(ctoks(text_n), q, lemma=False):
            return False
    if cap is None:
        return True
    mx, share, floor = cap
    return len(quote_n) <= int(mx) and (len(quote_n) <= int(floor) or len(quote_n) <= float(share) * len(text_n))


def judge_ok(cand, verdicts: dict[str, dict], frag_texts: dict[str, str], values=None,
             full_only: bool = False, quote_cap=None) -> tuple[bool, dict]:
    """verdicts: view key ("full", "drop:F2", …) -> judge entry; frag_texts: fragment label -> text.
    values (plants): the expected values instead of the candidate's facts.
    (a) every fact: answerable on `full` and its value (wsn) inside a judge quote that is verbatim in THAT fact's
    fragment; (b) every expected ablation view (views(cand)) answered unanswerable -- a missing verdict fails.
    full_only (chunk_swap / question_swap plants, which have one view): (a) only -- a plant must not be "rejected"
    merely because it has no ablation views."""
    label = {c: f"F{i + 1}" for i, c in enumerate(cand.seed.chunks)}
    full = verdicts.get("full") or _NO
    good, any_len = defaultdict(list), defaultdict(list)
    for e in full.get("evidence", []):
        t = frag_texts.get(e["id"])
        q = norm_map(e["quote"])[0]
        if t is not None and quote_ok(q, norm_map(t)[0], None):
            any_len[e["id"]].append(wsn(e["quote"]))
            if quote_ok(q, norm_map(t)[0], quote_cap):
                good[e["id"]].append(wsn(e["quote"]))
    vals = values or [f.value for f in cand.facts]
    # M3-R25 B: a typed value (number / name / term, <= 8 tokens) counts inside any verbatim quote -- the length cap
    # only stops whole-fragment quotes from "recovering" clause values
    # a clause value may itself be close to the cap (ingest allows <= 300 chars): a verbatim quote at most `floor`
    # characters longer than the value localises it as well as a capped quote does (M3-R29 diagnosis; a whole-fragment
    # quote still does not count)
    slack = int(quote_cap[2]) if quote_cap else None

    def pool(f):
        if typed_value(f) or slack is None:
            return any_len if typed_value(f) else good
        lim = len(wsn(f.value)) + slack
        out = defaultdict(list)
        for k in set(good) | set(any_len):
            out[k] = list(dict.fromkeys(good[k] + [q for q in any_len[k] if len(q) <= lim]))
        return out
    # M3-R25 D: a PATH fact (a multi-hop link fact the question itself states) is not an answer value -- the judge
    # need not quote it; the ablation still has to show that its fragment is needed
    path = set((getattr(cand, "provenance", None) or {}).get("path_fact_ids") or ())
    pairs = [(f, v) for f, v in zip(cand.facts, vals)
             if not (len(cand.seed.chunks) > 1 and (f.fact_id in path or _has(wsn(cand.question), f.value, lemma=True)))]
    rec = [bool(full.get("answerable")) and any(_has(q, v) for q in pool(f)[label[f.chunk_id]]) for f, v in pairs]
    # M3-R28: the drop views are read by their evidence only when the judge's own full answer demonstrably carries the
    # typed answer values (a lazy / blind judge writing no values keeps the raw verdict: answerable = answerable)
    fa = wsn(full.get("answer") or "")
    typed_ans = [f for f, _ in pairs if typed_value(f)]
    # cross-language (M3-R29 diagnosis): a Russian answer over an Uzbek fragment translates its values -- numbers are
    # still code-checked, a translated name / term is not held against the judge
    from graphrag.dataset.generate import cross_language, text_lang
    cross = {c for c in cand.seed.chunks if cross_language(cand.seed.lang, text_lang(frag_texts.get(label[c], "")))}
    trust = bool(fa.strip()) and all(value_evidence(f, fa, f.chunk_id in cross) is not False for f in typed_ans) and \
        (full.get("complete") is not False)
    raw = lambda v: (v or {"answerable": True})["answerable"]
    abl = {} if full_only else {k: not (drop_answerable(cand, k, verdicts.get(k), path, cross) if trust
                                        else raw(verdicts.get(k)))
                                for k, _ in views(cand)[1:]}
    return bool(rec) and all(rec) and all(abl.values()), {"facts_recovered": rec, "ablation": abl}


def value_evidence(f, answer: str, cross: bool) -> bool | None:
    """Does the judge's answer state fact f? True / False by code; None = cannot be code-checked (a name / term
    translated into another language -- the `complete` flag decides). A value with numbers counts when all its
    numbers are in the answer (M3-R20 num+judge: "до 1 января 2040 года" states "2040-yil 1-yanvarga qadar")."""
    from graphrag.dataset.facts import numbers
    if _stated(f.value, answer):
        return True
    nv = numbers(f.value)
    if nv:
        return nv <= numbers(answer)
    return None if cross else False


def _stated(value: str, answer: str) -> bool:
    """The judge's answer states the value: whole canonical tokens (lemmas), list joiners ignored ("A ва B" = "A и B")
    -- the ingest value check (generate.value_status)."""
    from graphrag.dataset.generate import _in_wo_conj
    return _has(wsn(answer), value, lemma=True) or _in_wo_conj(value, answer)


def drop_answerable(cand, key: str, v: dict | None, path=(), cross=frozenset()) -> bool:
    """M3-R28 (evidence-based ablation, strengthens п. 6): a drop view "drop:Fi" counts as ANSWERABLE only if the
    judge says answerable AND its answer carries the removed fragment's answer facts -- a typed value code-checked
    (whole canonical tokens in the judge's answer), a clause value by its numbers (code) and the judge's `complete`
    flag (its answer fully answers the question, gold withheld). A removed fragment holding only question-stated (path) facts, or an empty
    answer text, falls back to answerable AND complete. A missing verdict is answerable (fails the pair)."""
    if v is None:
        return True
    if not v.get("answerable"):
        return False
    complete = v.get("complete")
    complete = True if complete is None else bool(complete)          # a pre-M3-R28 verdict has no flag
    lab = key.split(":", 1)[1]
    gone = [f for i, c in enumerate(cand.seed.chunks) if f"F{i + 1}" == lab for f in cand.facts if f.chunk_id == c]
    gone = [f for f in gone if f.fact_id not in set(path) and not _has(wsn(cand.question), f.value, lemma=True)]
    ans = v.get("answer") or ""
    if not gone or not ans.strip():
        return complete
    from graphrag.dataset.facts import numbers
    typed = [f for f in gone if typed_value(f)]
    clause = [f for f in gone if not typed_value(f)]
    ev = [value_evidence(f, ans, f.chunk_id in cross) for f in typed]
    if any(e is False for e in ev):
        return False
    clause += [f for f, e in zip(typed, ev) if e is None]           # translated name / term: by `complete`
    if any(not numbers(f.value) <= numbers(ans) for f in clause):
        return False                  # a clause's own numbers are code-checked as well (its words by `complete`)
    return complete if clause else True


def typed_value(f) -> bool:
    from graphrag.dataset.facts import MAX_TYPED_TOKENS, ctoks, value_type
    return (getattr(f, "vtype", "") or value_type(f.value)) != "clause" and len(ctoks(f.value)) <= MAX_TYPED_TOKENS


def _perturb(v: str, other: str | None) -> str:
    m = re.search(r"\d+", v)
    if m:
        return v[:m.start()] + str(int(m.group()) + 1) + v[m.end():]
    return other if other and wsn(other) != wsn(v) else v + " эмас"




def _single(c) -> bool:
    return len(c.seed.chunks) == 1


def plan_batches(cands, batch: int, plant_share: float, seed, unrelated, pilot_ids=(), neighbor=None,
                 pilot_target: int | None = None, parts=None, verdicts=None) -> list[dict]:
    """[{cands: [...], plants: [...], part}] -- plant groups of `batch` candidates, pilot candidates first and planned
    on their own (partners come from the same part, so the pilot's plan is unchanged by the rest; review I4).
    Rest (and the pilot when pilot_target is None): one plant per group chosen with `plant_share`, kind by the deficit
    rule. Pilot with pilot_target (ruling M3-R18 (a)): every judge kind gets `pilot_target` plants and every code kind
    ceil(pilot_target / 2), bases in sha256(seed:pk|kind|seed_id) order -- a base may carry several plants of DISTINCT
    kinds, so a small pilot still measures >= min_kind_n per kind; the resulting share is reported by the caller.
    Plant fields: kind, cand (base), and chunks (chunk_swap / padded_single / neighbor_swap), question (+ other,
    partner) for question_swap, values for value_swap.
    M3-R25 C: `verdicts` (who -> view -> judge entry, the candidates' own calls) restrict a multi base's neighbor_swap
    to a slot whose drop view the judge called unanswerable (a fragment the pair needs); when the pilot's first pass
    leaves a kind short, a second neighbour plant per base (another slot / the other adjacent fragment) is tried."""
    out = []
    for part, cs in _parts(cands, pilot_ids, parts):          # M3-R23: + named parts between pilot and rest
        by_doc = defaultdict(list)
        for c in cs:
            by_doc[c.seed.docs[0]].append(c)
        groups = [{"cands": cs[i:i + batch], "plants": [], "part": part} for i in range(0, len(cs), batch)]
        if part == "pilot" and pilot_target is not None:
            where = {c.seed.seed_id: g for g in groups for c in g["cands"]}
            for kind in PLANT_KINDS:
                want = int(pilot_target) if kind in JUDGE_KINDS else math.ceil(int(pilot_target) / 2)
                n = 0
                for variant in ((0, 1) if kind == "neighbor_swap" else (0,)):
                    for c in sorted(cs, key=lambda c: _u(seed, f"pk|{kind}|{c.seed.seed_id}")):
                        if n >= want:
                            break
                        pl = _plant(kind, c, [c], by_doc, seed, unrelated, neighbor, cs, variant, verdicts)
                        if pl is not None and pl["cand"] is c:
                            where[c.seed.seed_id]["plants"].append(pl)
                            n += 1
        else:
            n_kind = Counter()
            for g in groups:
                c = g["cands"][0]
                if _u(seed, "plant|" + c.seed.seed_id) < plant_share:
                    # deficit rule: the applicable kind furthest below its weighted share goes first (judge kinds
                    # weigh 2, code kinds 1), so a kind that is rarely applicable (padded_single) catches up
                    for kind in sorted(PLANT_KINDS, key=lambda k: (n_kind[k] / KIND_WEIGHT[k], PLANT_KINDS.index(k))):
                        pl = _plant(kind, c, g["cands"], by_doc, seed, unrelated, neighbor, cs, 0, verdicts)
                        if pl is not None:
                            g["plants"].append(pl)
                            n_kind[kind] += 1
                            break
        out += groups
    return out


def _nb(neighbor, c, i, k):
    if k == 0:
        return neighbor(c, i)
    try:
        return neighbor(c, i, k)
    except TypeError:                      # a one-option picker
        return None


def plant_who(p) -> str:
    return f"p:{p['kind']}:{p['cand'].seed.seed_id}" + (f"#{p['variant']}" if p.get("variant") else "")


def _plant(kind, c, group, by_doc, seed, unrelated, neighbor, part_cs=(), variant: int = 0, verdicts=None):
    sid = c.seed.seed_id
    if kind == "chunk_swap":
        ch = list(c.seed.chunks)
        ch[int(_u(seed, "swap|" + sid) * len(ch)) % len(ch)] = unrelated(c)       # not always the last (review m3)
        return {"kind": kind, "cand": c, "chunks": ch}
    if kind == "question_swap":
        ok = lambda o: o is not c and o.question != c.question and not set(o.seed.chunks) & set(c.seed.chunks)
        other, partner = next((o for o in by_doc[c.seed.docs[0]] if ok(o)), None), "same_doc"
        if other is None:                     # fallback: same split and section, another document (an easier plant)
            other, partner = next((o for o in part_cs if ok(o) and o.seed.docs[0] != c.seed.docs[0]
                                   and o.seed.split == c.seed.split and o.seed.section == c.seed.section), None), \
                "same_section"
        return None if other is None else {"kind": kind, "cand": c, "question": other.question,
                                           "other": other.seed.seed_id, "partner": partner}
    if kind == "padded_single":
        b = next((x for x in group if _single(x)), None)
        if b is None:
            return None
        pad = unrelated(b)
        ch = [b.seed.chunks[0], pad] if _u(seed, "pad|" + b.seed.seed_id) < 0.5 else [pad, b.seed.chunks[0]]
        return {"kind": kind, "cand": b, "chunks": ch, "real": b.seed.chunks[0], "pad": pad}
    if kind == "neighbor_swap":
        if neighbor is None:
            return None
        idx = sorted(range(len(c.seed.chunks)), key=lambda i: _u(seed, f"nb|{sid}|{i}"))
        vv = (verdicts or {}).get("c:" + sid)
        if vv is not None and len(idx) > 1:          # only a fragment the pair needs (its drop view unanswerable)
            idx = [i for i in idx if (vv.get(f"drop:F{i + 1}") or {}).get("answerable") is False]
        opts = []
        for k in (0, 1):
            for i in idx:
                x = _nb(neighbor, c, i, k)
                if x is not None and (i, x) not in opts:
                    opts.append((i, x))
        if variant >= len(opts):
            return None
        i, x = opts[variant]
        ch = list(c.seed.chunks)
        ch[i] = x
        return {"kind": kind, "cand": c, "chunks": ch, "slot": i, "variant": variant}
    own = {wsn(f.value) for f in c.facts}
    other = next((f.value for o in by_doc[c.seed.docs[0]] if o is not c for f in o.facts if wsn(f.value) not in own), None)
    return {"kind": "value_swap", "cand": c, "values": [_perturb(f.value, other) for f in c.facts]}


def unrelated_picker(chunks_by_id: dict, split_of: dict[str, str], seed, min_tokens: int, section_of=None,
                     allowed=None):
    """chunk_swap / padded_single source (brief: same split, other document): per candidate, the first fragment in
    sha256(seed:unrelated|seed_id|chunk) order of another document of the same split, a fragment of >= min_tokens
    tokens (in `allowed` when given), from a document of the same section when one exists (a harder plant)."""
    doc = lambda cid: cid.split("/", 1)[0].split("#", 1)[0]
    pool = [x for x in sorted(chunks_by_id) if chunks_by_id[x].tokens >= int(min_tokens)
            and (allowed is None or x in allowed)]

    def pick(c) -> str:
        sec = (section_of or {}).get(c.seed.docs[0])
        cand = [x for x in pool if split_of.get(doc(x)) == c.seed.split and doc(x) not in c.seed.docs]
        same = [x for x in cand if section_of is not None and section_of.get(doc(x)) == sec]
        src = same or cand
        if not src:
            raise ValueError(f"no unrelated fragment for {c.seed.seed_id} (split {c.seed.split})")
        return min(src, key=lambda x: hashlib.sha256(f"{seed}:unrelated|{c.seed.seed_id}|{x}".encode()).hexdigest())
    return pick


NB_QUOTE_SHARE_MAX = 0.30          # M3-R25 C (ii): a replacement sharing more of the removed facts' quote content is no plant


def quote_share(quotes, text: str) -> float:
    """Share of the content tokens of `quotes` (canonical, >= 4 letters, no numbers / generic act words) that occur in
    `text` (canonical surface or M1 lemma; script-insensitive)."""
    from graphrag.dataset.facts import _generic, ctoks
    gen = _generic()
    q = {t.s: t.l for x in quotes for t in ctoks(x) if not t.num and len(t.s) >= 4 and t.s not in gen}
    if not q:
        return 0.0
    tt = ctoks(text)
    have = {t.s for t in tt} | {x for t in tt for x in t.l}
    return sum(1 for s, l in q.items() if s in have or l & have) / len(q)


def neighbor_picker(doc_order: dict[str, list[str]], chunks_by_id: dict, seed, min_tokens: int, allowed=None):
    """neighbor_swap source: for fragment i of a candidate, the previous or next fragment of the same document (seeded
    order) that is not in the pair, has >= min_tokens tokens (in `allowed` when given) and contains none of the values
    of the facts on fragment i (whole-word match) -- the needed fact is then absent from every shown fragment.
    M3-R25 C (ii): the match is on canonical tokens (script- and lemma-insensitive), and a fragment sharing more than
    NB_QUOTE_SHARE_MAX of those facts' quote content tokens is no replacement; k = 0 / 1 picks the first / second
    valid adjacent fragment (a second plant per base)."""
    pos = {c: (d, k) for d, cs in doc_order.items() for k, c in enumerate(cs)}

    def pick(c, i, k=0):
        x = c.seed.chunks[i]
        if x not in pos:
            return None
        d, at = pos[x]
        fs = [f for f in c.facts if f.chunk_id == x]
        near = [doc_order[d][j] for j in (at - 1, at + 1) if 0 <= j < len(doc_order[d])]
        ok = []
        for y in sorted(near, key=lambda y: _u(seed, f"neighbor|{c.seed.seed_id}|{y}")):
            if y in c.seed.chunks or chunks_by_id[y].tokens < int(min_tokens) or (allowed is not None and y not in allowed):
                continue
            t = wsn(chunks_by_id[y].text)
            if any(_has(t, f.value, lemma=True) for f in fs):
                continue
            if quote_share([f.quote for f in fs], chunks_by_id[y].text) > NB_QUOTE_SHARE_MAX:
                continue
            ok.append(y)
        return ok[k] if k < len(ok) else None
    return pick


def sensitivity_gate(st: dict, min_sensitivity_kind: float) -> list[str]:
    """M3-R3 п. 8 stop reasons: a judge-level plant kind below the sensitivity floor, or with too few plants."""
    out = [f"{k}: sensitivity {v} < {min_sensitivity_kind}" for k, v in sorted(st["sensitivity_by_kind"].items())
           if v < float(min_sensitivity_kind)]
    if st.get("kinds_below_min_n"):
        out.append(f"too few plants of kinds {st['kinds_below_min_n']}")
    return out


def _plant_views(p) -> list[tuple[str, str, list[str]]]:
    """(view key, question, chunk ids) of a plant."""
    c = p["cand"]
    if p["kind"] in ("chunk_swap", "neighbor_swap"):
        return [("full", c.question, p["chunks"])]
    if p["kind"] == "question_swap":
        return [("full", p["question"], list(c.seed.chunks))]
    if p["kind"] == "padded_single":
        return [("full", c.question, p["chunks"]), ("drop:real", c.question, [p["pad"]]),
                ("drop:pad", c.question, [p["real"]])]
    return []                                                                  # value_swap: no view of its own


def pack_calls(groups, per_call: int, seed, cands: bool = True, plants: bool = True) -> list[list[dict]]:
    """View items -> calls of <= per_call views; within a part (pilot / rest), items in sha256(seed:view|who|view)
    order go first-fit to a call holding no view of the same question family (a pair, its plants, a question_swap's
    donor) -- never two views of one question, never a plant beside its base pair. Items: {who, view, question,
    chunks, fam}."""
    calls: list[list[dict]] = []
    for part in dict.fromkeys(g["part"] for g in groups):     # plan order: pilot, named parts, rest
        items = []
        for gi, g in enumerate(x for x in groups if x["part"] == part):
            for c in (g["cands"] if cands else ()):
                for key, ids in views(c):
                    items.append({"part": part, "who": "c:" + c.seed.seed_id, "view": key, "question": c.question,
                                  "chunks": [c.seed.chunks[int(x[1:]) - 1] for x in ids], "orig": list(ids),
                                  "fam": {c.seed.seed_id}})
            for j, p in enumerate(g["plants"] if plants else ()):
                fam = {p["cand"].seed.seed_id} | ({p["other"]} if p.get("other") else set())
                for key, q, ch in _plant_views(p):
                    items.append({"part": part, "who": plant_who(p), "view": key,
                                  "question": q,
                                  "chunks": list(ch), "orig": [f"F{i + 1}" for i in range(len(ch))], "fam": fam})
        items.sort(key=lambda it: (_u(seed, f"view|{it['who']}|{it['view']}"), it["who"], it["view"]))
        mine: list[tuple[list[dict], set]] = []
        for it in items:
            for call, fams in mine:
                if len(call) < per_call and not (fams & it["fam"]):
                    call.append(it)
                    fams |= it["fam"]
                    break
            else:
                mine.append(([it], set(it["fam"])))
        calls += [call for call, _ in mine]
    return calls


def plan_digest(calls, part: str = "pilot") -> str | None:
    """sha256 of the `part`'s call plan (views per call: who, view, chunks, question) -- the frozen pilot plan
    (ruling M3-R18): the full run must reproduce it exactly, else the pilot's cached calls would be re-paid."""
    mine = lambda x: x == part or x.startswith(part + "|") or x == "plants:" + part
    sig = [[[it["who"], it["view"], list(it["chunks"]), it["question"]] for it in call]
           for call in calls if call and mine(call[0]["part"])]
    if not sig:
        return None
    return hashlib.sha256(json.dumps(sig, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


TOPUP_MARGIN = 2           # top-up aims at min_kind_n + 2 valid plants (contested misses are excluded afterwards)


def plan_topup(acc, short: dict, groups, seed, unrelated, neighbor, verdicts) -> list[dict]:
    """Top-up plants for judge kinds short of their minimum: kind -> deficit; bases = accepted candidates in
    sha256(seed:topup|kind|seed_id) order, a base / variant never planted twice for one kind; 1.5 x the deficit (some
    plants turn out contested or invalid). Returns [{cands: [], plants, part: "plants:topup"}]."""
    used = {(p["kind"], p["cand"].seed.seed_id, p.get("variant", 0)) for g in groups for p in g["plants"]}
    by_doc = defaultdict(list)
    for c in acc:
        by_doc[c.seed.docs[0]].append(c)
    plants = []
    for kind, deficit in sorted(short.items()):
        want, n = math.ceil(1.5 * deficit), 0
        for variant in ((0, 1) if kind == "neighbor_swap" else (0,)):
            for c in sorted(acc, key=lambda c: _u(seed, f"topup|{kind}|{c.seed.seed_id}")):
                if n >= want:
                    break
                if (kind, c.seed.seed_id, variant) in used:
                    continue
                pl = _plant(kind, c, [c], by_doc, seed, unrelated, neighbor, acc, variant, verdicts)
                if pl is None or pl["cand"] is not c:
                    continue
                used.add((kind, c.seed.seed_id, variant))
                plants.append(pl)
                n += 1
    return [{"cands": [], "plants": plants, "part": "plants:topup"}] if plants else []


def _cand_part(c, pilot: set, parts: dict) -> str:
    sid = c.seed.seed_id
    if sid in pilot:
        return "pilot" + (f"|{parts[sid]}" if sid in parts else "")
    return parts.get(sid, "rest")


def verify_candidates(cands, frag_of, llm, batch: int, plant_share: float, seed, unrelated,
                      min_kind_n: int = 10, views_per_call: int = 5, neighbor=None, pilot_ids=(),
                      quote_cap=None, pilot_target: int | None = None, frozen_plan: str | None = None,
                      agree_per_call: int | None = None, parts=None, topup: bool = False) -> tuple[list, dict]:
    """frag_of(chunk_id, label) -> {id, act, where, text}. frozen_plan: the pilot's plan_digest; a full run whose
    pilot part plans differently raises before any non-pilot call (the pilot's cache would not be reused).

    M3-R25: two phases. (1) the candidates' own views, packed per part (pilot [per named part], named parts, rest) --
    no plant beside them, so a plant change never re-pays a candidate call; the blind verdicts decide which bases are
    accepted. (2) plants only on ACCEPTED bases (every kind; multi neighbor_swap only on a needed slot), planned and
    packed in their own calls ("plants:<part>"). A neighbor_swap miss whose judge evidence lies in the replacement
    fragment is `contested` (manual adjudication), not a miss."""
    from graphrag.dataset.facts import guard_total
    from graphrag.kg.llm import _canon_input, prompt_digest
    digest = prompt_digest(PROMPT_VER, SCHEMA_VER)
    pilot, parts = set(pilot_ids), dict(parts or {})
    by_part: dict[str, list] = defaultdict(list)
    for c in sorted(cands, key=lambda c: c.seed.seed_id):
        by_part[_cand_part(c, pilot, parts)].append(c)
    order = sorted(by_part, key=lambda n: (0 if n.startswith("pilot") else 2 if n == "rest" else 1, n))
    groups1 = [{"cands": by_part[n], "plants": [], "part": n} for n in order]
    verdict: dict[str, dict] = defaultdict(dict)          # who -> view -> judge entry (evidence ids in pair labels)
    call_key: dict[tuple, str] = {}
    labels: dict[tuple, dict] = {}
    cnt = Counter()

    def run(calls):
        for call in calls:
            vs, back = [], {}
            for n, it in enumerate(call):
                k = len(it["chunks"])
                orig = it["orig"]
                perm = sorted(range(k), key=lambda i: _u(seed, f"frag|{it['who']}|{it['view']}|{i}"))
                frs, lab = [], {}
                for pos, i in enumerate(perm):
                    f = frag_of(it["chunks"][i], f"F{pos + 1}")
                    frs.append(f)
                    lab[f"F{pos + 1}"] = orig[i]
                name = f"V{n + 1}"
                back[name] = (it, lab)
                vs.append({"key": name, "question": it["question"], "fragments": frs})
            inp = {"views": vs}
            keys = set(back)
            guard_total(llm, PASS, inp, PROMPT_VER, SCHEMA_VER)          # M3-R3 п. 14: 2500 over the whole ledger
            ans = llm.cached_json(PASS, inp, SCHEMA_VER,
                                  lambda a: (sorted(x["key"] for x in a["views"]) == sorted(keys), "keys"),
                                  prompt=PROMPT_VER)
            cnt["calls"] += 1
            if ans is None:
                cnt["bad"] += 1
                continue
            key = llm.cache_key(PASS, _canon_input(inp), digest)
            for x in ans["views"]:
                it, lab = back[x["key"]]
                ev = [{"id": lab[e["id"]], "quote": e["quote"]} for e in x["evidence"] if e["id"] in lab]
                verdict[it["who"]][it["view"]] = {"answerable": bool(x["answerable"]), "answer": x["answer"],
                                                  "complete": x.get("complete"),
                                                  "evidence": ev}
                call_key[(it["who"], it["view"])] = key
                labels[(it["who"], it["view"])] = lab
    texts = {}
    for c in cands:
        texts[c.seed.seed_id] = {f"F{i + 1}": frag_of(ch, f"F{i + 1}")["text"] for i, ch in enumerate(c.seed.chunks)}
    status: dict[str, tuple] = {}

    def judge(cs):
        for c in cs:
            who = "c:" + c.seed.seed_id
            if any(k not in verdict[who] for k, _ in views(c)):
                status[c.seed.seed_id] = ("bad_answer", None)
                continue
            ok, rep = judge_ok(c, verdict[who], texts[c.seed.seed_id], quote_cap=quote_cap)
            status[c.seed.seed_id] = (None if ok else ("facts" if not all(rep["facts_recovered"]) else "ablation"), rep)
    all_calls, groups2 = [], []
    for phase_pilot in (True, False):
        g1 = [g for g in groups1 if g["part"].startswith("pilot") == phase_pilot]
        calls1 = pack_calls(g1, views_per_call, seed)
        cs = [c for g in g1 for c in g["cands"]]
        if phase_pilot:
            # the frozen-plan check needs the pilot's plant plan, which needs the pilot's (cached) verdicts
            run(calls1)
            judge(cs)
        else:
            run(calls1)
            judge(cs)
        acc = [c for c in cs if status[c.seed.seed_id][0] is None]
        gp = plan_batches(acc, batch, plant_share, seed, unrelated, pilot_ids=pilot if phase_pilot else (),
                          neighbor=neighbor, pilot_target=pilot_target if phase_pilot else None,
                          parts=None if phase_pilot else parts, verdicts=verdict)
        for g in gp:
            g["part"] = "plants:" + ("pilot" if phase_pilot else g["part"])
        calls2 = pack_calls(gp, views_per_call, seed, cands=False)
        if phase_pilot:
            pdig = plan_digest(calls1 + calls2)
            if frozen_plan is not None and pdig != frozen_plan:
                raise ValueError(f"ds_verify: the pilot part plans to {pdig} but the frozen pilot plan is "
                                 f"{frozen_plan} -- same pilot candidates, seed, verify_* config and pilot_target are "
                                 f"required (M3-R18)")
        run(calls2)
        all_calls += calls1 + calls2
        groups2 += gp
    kept, why = [], Counter()
    by_cell: dict[str, Counter] = defaultdict(Counter)
    accepted = set()

    def reject(c, reason):
        why[f"{c.seed.pattern}|{reason}"] += 1
        by_cell[_cell(c)][reason] += 1

    for c in sorted(cands, key=lambda c: c.seed.seed_id):
        who = "c:" + c.seed.seed_id
        r, rep = status[c.seed.seed_id]
        if r:
            reject(c, r)
            continue
        accepted.add(c.seed.seed_id)
        vk = {k: call_key[(who, k)] for k, _ in views(c)}
        c.provenance["verify"] = {"pass": PASS, "model": llm.model,
                                  "prompt_version": llm.passes[PASS].get("prompt_version"), "prompt_digest": digest,
                                  "cache_key": vk["full"], "cache_keys": vk,
                                  "labels": {k: labels[(who, k)] for k, _ in views(c)},
                                  "report": rep, "verdicts": verdict[who]}
        kept.append(c)
    def score():
        n_j, caught_j, n_c, caught_c, n_all, invalid = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
        pj, pc, contested, pcont = Counter(), Counter(), Counter(), Counter()
        contested_items = []
        for g in groups2:
            is_pilot = g["part"] == "plants:pilot"
            for p in g["plants"]:
                kind, c = p["kind"], p["cand"]
                who = plant_who(p)
                n_all[kind] += 1
                if kind == "value_swap":
                    ts = [wsn(t) for t in texts[c.seed.seed_id].values()]
                    if any(_has(t, v) for t in ts for v in p["values"]):
                        invalid[kind] += 1                    # the "wrong" value is in the fragments: not a wrong pair
                        continue
                    ok, _ = judge_ok(c, verdict["c:" + c.seed.seed_id], texts[c.seed.seed_id], p["values"],
                                     quote_cap=quote_cap)
                    n_c[kind] += 1
                    caught_c[kind] += not ok
                    continue
                vv = verdict.get(who, {})
                if any(k not in vv for k, _, _ in _plant_views(p)):
                    invalid[kind] += 1                        # a bad call: no verdict for the plant
                    continue
                if kind == "padded_single":
                    caught = vv["drop:pad"]["answerable"] and not vv["drop:real"]["answerable"]
                else:
                    caught = not vv["full"]["answerable"]
                if kind == "neighbor_swap" and not caught and \
                        any(e["id"] == f"F{p['slot'] + 1}" for e in vv["full"].get("evidence", [])):
                    contested[kind] += 1                      # the judge answered from the replacement fragment itself
                    pcont[kind] += is_pilot
                    contested_items.append({"who": who, "replacement": p["chunks"][p["slot"]],
                                            "evidence": vv["full"]["evidence"], "answer": vv["full"].get("answer")})
                    continue
                if kind in JUDGE_KINDS:
                    n_j[kind] += 1
                    caught_j[kind] += bool(caught)
                    if is_pilot:
                        pj[kind] += 1
                        pc[kind] += bool(caught)
                else:                                         # chunk_swap: code rule on the judge's quotes + judge verdict
                    ptexts = {f"F{i + 1}": frag_of(ch, f"F{i + 1}")["text"] for i, ch in enumerate(p["chunks"])}
                    ok, _ = judge_ok(c, {"full": vv["full"]}, ptexts, full_only=True, quote_cap=quote_cap)
                    n_c[kind] += 1
                    caught_c[kind] += not ok
                    n_j[kind + "@judge"] += 1
                    caught_j[kind + "@judge"] += bool(caught)
        return n_j, caught_j, n_c, caught_c, n_all, invalid, pj, pc, contested, pcont, contested_items

    n_j, caught_j, n_c, caught_c, n_all, invalid, pj, pc, contested, pcont, contested_items = score()
    # cumulative sizing (M3-R3 п. 8): a judge kind short of min_kind_n VALID plants (invalid / contested excluded) gets
    # one top-up round of its own plants over all accepted bases, in their own calls ("plants:topup")
    short = {k: int(min_kind_n) + TOPUP_MARGIN - n_j[k] for k in JUDGE_KINDS if n_j[k] < int(min_kind_n)}
    topup_n = 0
    if topup and short:
        acc = sorted((c for c in cands if status[c.seed.seed_id][0] is None), key=lambda c: c.seed.seed_id)
        gt = plan_topup(acc, short, groups2, seed, unrelated, neighbor, verdict)
        calls3 = pack_calls(gt, views_per_call, seed, cands=False)
        run(calls3)
        all_calls += calls3
        groups2 += gt
        topup_n = sum(len(g["plants"]) for g in gt)
        n_j, caught_j, n_c, caught_c, n_all, invalid, pj, pc, contested, pcont, contested_items = score()
    agree = None
    if agree_per_call:                                   # M3-R20: value correctness of judge-checked values
        before = kept
        kept, agree = agree_candidates(kept, llm, int(agree_per_call), pilot_ids, parts)
        left = {c.seed.seed_id for c in kept}
        for c in before:
            if c.seed.seed_id not in left:
                reject(c, "disagree" if "agree" in c.provenance else "agree_bad_answer")
    sens = {k: round(caught_j[k] / n_j[k], 4) for k in JUDGE_KINDS if n_j[k] >= min_kind_n}
    wil = {k: [round(x, 4) for x in wilson(caught_j[k], n_j[k])] for k in n_j if n_j[k]}
    pilot_groups = [g for g in groups2 if g["part"] == "plants:pilot"]
    n_pilot = sum(len(g["cands"]) for g in groups1 if g["part"].startswith("pilot"))
    n_pp = sum(len(g["plants"]) for g in pilot_groups)
    return kept, {"checked": len(cands), "kept": len(kept), "rejected": dict(sorted(why.items())),
                  "rejected_by_cell": {c: dict(sorted(v.items())) for c, v in sorted(by_cell.items())},
                  "calls": cnt["calls"], "bad_calls": cnt["bad"], "views": sum(len(c) for c in all_calls),
                  "views_plants": sum(1 for c in all_calls for it in c if it["who"].startswith("p:")),
                  "calls_plants": sum(1 for c in all_calls if c and c[0]["part"].startswith("plants:")),
                  "plants": sum(len(g["plants"]) for g in groups2),
                  "planted": {k: n_j[k] for k in sorted(n_j)}, "planted_total": {k: n_all[k] for k in PLANT_KINDS if n_all[k]},
                  "plants_invalid": dict(sorted(invalid.items())),
                  "contested_by_kind": dict(contested), "contested": contested_items,
                  "sensitivity_by_kind": sens, "sensitivity_wilson_by_kind": wil,
                  "judge_rate_by_kind": {k: round(caught_j[k] / n_j[k], 4) for k in sorted(n_j) if n_j[k]},
                  "code_sensitivity_by_kind": {k: round(caught_c[k] / n_c[k], 4) for k in CODE_KINDS if n_c[k]},
                  "min_kind_n": min_kind_n,
                  "kinds_below_min_n": sorted(k for k in JUDGE_KINDS if n_j[k] < min_kind_n),
                  "pilot": {"candidates": n_pilot, "plants": n_pp,
                            "plant_share": round(n_pp / max(1, math.ceil(n_pilot / max(1, batch))), 4),
                            "planted": {k: pj[k] for k in JUDGE_KINDS if pj[k]},
                            "contested_by_kind": dict(pcont),
                            "sensitivity_by_kind": {k: round(pc[k] / pj[k], 4) for k in JUDGE_KINDS
                                                    if pj[k] >= min_kind_n},
                            "kinds_below_min_n": sorted(k for k in JUDGE_KINDS if pj[k] < min_kind_n)},
                  "plan_digest_pilot": pdig, "agree": agree, "topup": {"short_before": short, "plants": topup_n}}
