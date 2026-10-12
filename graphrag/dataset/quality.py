"""ds_quality (ruling M3-R22 (3)): a rubric quality judge, gpt-4.1, blind to the generator (it sees only the question,
the answer, the target language and the C_ref fragments -- no facts, no generator output other than the pair, no
verify verdicts). It applies the strict reader's 7-criterion rubric (repilot-reading.md) and returns GOOD / BORDERLINE
/ BAD with a reason; only GOOD passes. Batched `per_call` pairs per call, pilot part first, each part in seed_id order
(deterministic: the full run replays the pilot's calls); versioned, cached, guard_total before every uncached call."""
from __future__ import annotations

from collections import Counter

from graphrag.kg.llm import _canon_input, prompt_digest

PASS = "ds_quality"
VERDICTS = ("GOOD", "BORDERLINE", "BAD")
# prompt_version 1 (M3-R22): the strict reader's rubric, one verdict -- calibration 3/17 exact, 4 BAD accepted (too
# lenient: 16/17 GOOD). prompt_version 2: adversarial per-criterion checks; GOOD is derived by code only when all seven
# checks pass; each fragment carries its document kind. No example pair from the calibration set in either version.
PROMPT_QUALITY = """You are a strict, adversarial reviewer of a legal question-answer dataset over Uzbek parliamentary documents.
Generated pairs usually have at least one defect: your job is to find it. Each ITEM gives the target LANG, a QUESTION,
an ANSWER and the FRAGMENTS the pair was built from (kind: law / Senate decision / committee conclusion / plenary
transcript / other document; act = the document's own title; where; text). Check each criterion separately and set
ok=false whenever there is a real doubt:
c1 supported: every name, place, number, body and claim of the question and answer is in the fragment text (not
   only in your knowledge); nothing reversed, narrowed or widened.
c2 complete and specific: the answer fully answers the question; if the fragment lists several items / priorities /
   conditions, the question asks for all of them or names the one it wants; the answer is not a single vague word
   ("комитетом", "обязательны") where the question expects content.
c3 self-contained: a reader who has NEVER seen the fragment knows exactly which draft, law, article, village, body or
   event is meant; "which article was excluded" without naming the amended law, "who reviewed the draft" without
   naming the draft, "residents" without naming the place -> fail. A question built on a throwaway transcript line fails.
c4 act identity: the act the question names is the fragment's own act (its `act` title, faithfully rendered) -- for
   an amending law the amended act and its article. Fail if it names an act, decree or programme the text only
   mentions, the year's slogan instead of the act, a garbled or wrong title, the amending law's own item number
   instead of the amended provision, or misattributes the kind (a conclusion or a speech called a law / decision /
   "draft"; a transcript called a "session" document where it is a plenary sitting).
c5 paraphrased: the question is not the fragment's sentence (nor its translation / transliteration) with one element
   blanked out; it does not restate almost the whole fragment so that the answer only repeats one phrase.
c6 fit: the interrogative matches the answer type and the rule's logic (threshold -> "up to what amount", fee /
   manner -> "on what terms / is it paid", term -> "within what period"); the question's condition is not inverted.
c7 language: fluent, grammatical LANG with the correct legal terms (тафтиш инстанцияси = ревизионная инстанция, not
   надзорная; Солиқ қўмитаси = Налоговый комитет, not Государственный налоговый комитет; баённома = протокол;
   ялпи мажлис = пленарное заседание); pure script (uz-Cyrl: no Latin letters, no ʻ / ʼ; uz-Latn: no Cyrillic); in
   ru, act titles rendered in Russian (an Uzbek or Latin-script title alone inside a Russian question fails).
verdict: GOOD only if all seven are ok; BAD if c1, c2, c3 or c4 fails in a way that makes the answer wrong, ambiguous
or unanswerable without the fragment; otherwise BORDERLINE.
Return {"items": [{"key", "checks": {"c1".."c7": {"ok", "note"}}, "verdict", "reason"}]}, one per key; note = a few words."""
_CHECK = {"type": "object", "additionalProperties": False, "required": ["ok", "note"],
          "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}}}
CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
SCHEMA_QUALITY = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["key", "checks", "verdict", "reason"],
                               "properties": {"key": {"type": "string"},
                                              "checks": {"type": "object", "additionalProperties": False,
                                                         "required": list(CRITERIA),
                                                         "properties": {c: _CHECK for c in CRITERIA}},
                                              "verdict": {"type": "string", "enum": list(VERDICTS)},
                                              "reason": {"type": "string"}}}}}}
KIND = {"QONUNLAR": "law", "QARORLAR": "Senate decision", "XULOSALAR": "committee conclusion",
        "STENOGRAMMA": "plenary transcript", "BOSHQA": "other document"}


def final_verdict(x: dict) -> tuple[str, list[int]]:
    """GOOD only when every criterion check is ok (code, not the model's own verdict); a model GOOD with a failed
    check becomes BORDERLINE."""
    failed = [i + 1 for i, c in enumerate(CRITERIA) if not x["checks"][c]["ok"]]
    if not failed:
        return ("GOOD" if x["verdict"] == "GOOD" else x["verdict"]), failed
    return ("BORDERLINE" if x["verdict"] == "GOOD" else x["verdict"]), failed


def quality_item(key: str, lang: str, question: str, answer: str, frags: list[dict], kinds=None) -> dict:
    return {"key": key, "lang": lang, "question": question, "answer": answer,
            "fragments": [{"kind": (kinds or [None] * len(frags))[i] or "document", "act": f["act"], "where": f["where"],
                           "text": f["text"]} for i, f in enumerate(frags)]}


def judge_quality(cands, frag_of, llm, per_call: int, pilot_ids=(), prompt: str = PROMPT_QUALITY) -> tuple[list, dict]:
    """cands: generate.Candidate (after ds_verify); frag_of(chunk_id, label) -> {act, where, text}. Returns (GOOD
    candidates with provenance["quality"], stats {checked, kept, verdicts, rejected by pattern, calls, bad_calls})."""
    from graphrag.dataset.facts import guard_total
    digest = prompt_digest(prompt, SCHEMA_QUALITY)
    pilot = set(pilot_ids)
    parts = [sorted((c for c in cands if c.seed.seed_id in pilot), key=lambda c: c.seed.seed_id),
             sorted((c for c in cands if c.seed.seed_id not in pilot), key=lambda c: c.seed.seed_id)]
    verdict, n_calls, bad = {}, 0, 0
    for cs in parts:
        for i in range(0, len(cs), int(per_call)):
            chunk = cs[i:i + int(per_call)]
            back = {f"Q{n + 1}": c for n, c in enumerate(chunk)}
            items = [quality_item(k, c.seed.lang, c.question, c.answer,
                                  [frag_of(ch, f"F{j + 1}") for j, ch in enumerate(c.seed.chunks)],
                                  [KIND.get(ch.split("-", 1)[0]) for ch in c.seed.chunks])
                     for k, c in back.items()]
            inp = {"items": items}
            guard_total(llm, PASS, inp, prompt, SCHEMA_QUALITY)
            ans = llm.cached_json(PASS, inp, SCHEMA_QUALITY,
                                  lambda a: (sorted(x["key"] for x in a["items"]) == sorted(back), "keys"),
                                  prompt=prompt)
            n_calls += 1
            if ans is None:
                bad += 1
                continue
            ck = llm.cache_key(PASS, _canon_input(inp), digest)
            for x in ans["items"]:
                v, failed = final_verdict(x)
                verdict[back[x["key"]].seed.seed_id] = (v, failed, x["reason"], ck,
                                                       {c: x["checks"][c] for c in CRITERIA})
    kept, why, dist = [], Counter(), Counter()
    for c in cands:
        v = verdict.get(c.seed.seed_id)
        if v is None:
            why[f"{c.seed.pattern}|quality_bad_answer"] += 1
            continue
        dist[f"{c.seed.pattern}|{v[0]}"] += 1
        c.provenance["quality"] = {"pass": PASS, "model": llm.model,
                                   "prompt_version": llm.passes[PASS].get("prompt_version"), "prompt_digest": digest,
                                   "cache_key": v[3], "verdict": v[0], "failed": v[1], "reason": v[2],
                                   "checks": v[4]}
        if v[0] == "GOOD":
            kept.append(c)
        else:
            why[f"{c.seed.pattern}|quality_{v[0].lower()}"] += 1
    return kept, {"checked": len(cands), "kept": len(kept), "verdicts": dict(sorted(dist.items())),
                  "rejected": dict(sorted(why.items())), "calls": n_calls, "bad_calls": bad}


def calibration(labels: dict[str, str], got: dict[str, str]) -> dict:
    """Agreement of the judge's verdicts `got` with the strict reader's `labels` (seed_id -> GOOD / BORDERLINE / BAD):
    exact 3-class agreement, pass/fail agreement (GOOD vs not) and the reader's BAD the judge accepted (GOOD)."""
    ids = sorted(labels)
    exact = sum(got.get(i) == labels[i] for i in ids)
    binary = sum((got.get(i) == "GOOD") == (labels[i] == "GOOD") for i in ids)
    bad_acc = [i for i in ids if labels[i] == "BAD" and got.get(i) == "GOOD"]
    nongood_acc = [i for i in ids if labels[i] != "GOOD" and got.get(i) == "GOOD"]
    return {"n": len(ids), "exact": exact, "binary": binary, "bad_accepted": bad_acc,
            "non_good_accepted": nongood_acc,
            "confusion": dict(sorted(Counter(f"{labels[i]}->{got.get(i)}" for i in ids).items()))}
