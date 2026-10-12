"""M7 part 2, Task 5: the ds_hard gpt-4.1 judge of Q1 / Q2 paraphrases (docs/m7/2026-10-02-hard-questions-design.md
§2, §6) -- «is the paraphrase the same question with the same answer as the template».

* One call judges up to 10 pairs «template <-> paraphrase» (keys P1..Pn); schema {verdicts: [{key, same, why}]};
  an answer whose keys are not exactly the call's keys is refused (cached, reproducible) and the call is asked once
  more with a retry note; a pair still without a verdict counts as not judged (same = None).
* Plants (the M3 verify plants mechanism, graphrag.dataset.verify, re-implemented small): a base pair's paraphrase
  corrupted by code, paired with the same template --
    number_swap  a year (preferred), an article / part number or a day replaced by another number;
    act_swap     the «title» replaced by another graph title of the same ctoks length and script;
    op_swap      the kind of change replaced by another one (deletion <-> addition, the rest -> deletion).
  Per kind max(min_n, ceil(plant_share * pairs)) plants, bases in sha256(seed:pk|kind|qid) order (a base may carry
  plants of several kinds). `pack_calls`: items in sha256(seed:view|who) order go first-fit to a call holding no item
  of the same qid -- a plant never shares a call with its base or another plant of that base.
* Gate (M3-R25 style): per kind >= min_n judged plants and sensitivity (share judged «not same») >= 0.90; a failure
  is a loud STOP, the threshold is never lowered.
* Budget: graphrag.dataset.facts.guard_total (the whole cache/hard ledger vs hard_llm.max_calls_total 900) before
  every cached_json; KgLLM keeps the per-pass cumulative cap."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

from graphrag.dataset.facts import ctoks, guard_total
from grl.hard.paraphrase import _QUOTE_RE, _u
from grl.hard.templates import title_key as _title_key


@lru_cache(maxsize=None)
def title_key(title: str) -> tuple[str, ...]:
    return _title_key(title)

PASS = "ds_hard"
PER_CALL = 10
MIN_N = 30
MIN_SENSITIVITY = 0.90
PLANT_KINDS = ("number_swap", "act_swap", "op_swap")
Q1X_PLANT_KINDS = PLANT_KINDS + ("object_swap",)   # Task 6c: + what is counted (the new templates' main risk)
AMENDING = {"kiritish", "vnesenii", "vnesenie", "kiritilishi"}   # folded words of amending-law titles

PROMPT = """You check paraphrases of questions for a legal retrieval benchmark over Uzbek legislation. Questions are \
in Russian, Uzbek (Cyrillic) or Uzbek (Latin); act titles in «» may stay in Uzbek inside a Russian question.

Each pair has an ORIGINAL question (written from a template) and a PARAPHRASE. For every pair decide: is the \
paraphrase the same question, so that its correct answer is exactly the same as the original's?

same = true only if ALL of these hold:
- the same act: the same title (a different script or an inflected suffix outside the quotes is fine), and the same \
act number if the original gives one;
- the same years, dates, article and part numbers -- no number changed, dropped or added;
- the same kind of change if the original names one (addition, deletion / exclusion, new wording, word replacement, \
renumbering), and the same scope (e.g. articles vs. all elements; which law vs. what was changed);
- the same kind of answer (a count vs. a list), nothing added, removed or narrowed that would change the answer;
- the paraphrase does not state or hint at the answer.
Different wording, word order or grammar is fine. If any detail differs, same = false.

Return exactly one verdict per pair with its key, same (true / false), and why: a short reason in English \
(at most 25 words)."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
          "properties": {"verdicts": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["key", "same", "why"],
              "properties": {"key": {"type": "string"}, "same": {"type": "boolean"}, "why": {"type": "string"}}}}}}

RETRY_NOTE = "The previous answer did not return exactly one verdict per pair key; return exactly the keys given."


def make_pair(item: dict, paraphrase: str, anchors: dict) -> dict:
    return {"qid": item["qid"], "who": "c:" + item["qid"], "original": item["question"], "paraphrase": paraphrase,
            "lang": item["lang"], "anchors": anchors, "template": item.get("template")}


# --- plants --------------------------------------------------------------------------------------------------------

def _number_swap(pair: dict, seed) -> str | None:
    a, text = pair["anchors"], pair["paraphrase"]
    keep = set(a["numbers"]) | {x.lstrip("0") or "0" for d in a["dates"] for x in d.split("-")}
    core = set(a.get("number_core") or ())
    years = sorted({d[:4] for d in a["dates"]} | {n for n in a["numbers"] if len(n) == 4})
    groups = [years, sorted(n for n in a["numbers"] if len(n) < 4), sorted({d[8:].lstrip("0") for d in a["dates"]})]
    r = int(_u(seed, f"numrot|{pair['qid']}") * 3)              # rotate: year / article-part / day first (by seed)
    order = [n for g in groups[r:] + groups[:r] for n in g]
    for n in dict.fromkeys(order):
        if n in core or not n.isdigit() or not re.search(rf"(?<!\d){n}(?!\d)", text):
            continue
        k = int(n)
        options = [k - 1, k + 1] if len(n) == 4 else ([k + 1, k - 1] if k > 1 else [k + 1])
        if len(n) < 4 and n in {d[8:].lstrip("0") for d in a["dates"]}:
            options = [x for x in options if 1 <= x <= 28] or [k + 1]
        options = [x for x in options if str(x) not in keep]
        if not options:
            continue
        new = options[0] if _u(seed, f"num|{pair['qid']}|{n}") < 0.5 else options[-1]
        out = re.sub(rf"(?<!\d){n}(?!\d)", str(new), text)
        if out != text:
            return out
    return None


def _cyr(s: str) -> bool:
    return any("Ѐ" <= c <= "ӿ" for c in s)


def _act_like(title: str) -> bool:
    """A swap title reads like an act title (not an amending law): one line, no digits or quotes, «... to'g'risida» / «... kodeksi»."""
    if any(c.isdigit() or c in "«»\n\"“”„" for c in title):
        return False
    key = title_key(title)
    if any(t in AMENDING for t in key):                 # «... внесении изменений / киритиш ...»: an amending law
        return False
    return any(t in ("togrisida", "haqida") or t.startswith("kodeks") for t in key)


def _act_swap(pair: dict, seed, titles) -> str | None:
    a, text = pair["anchors"], pair["paraphrase"]
    key = tuple(a["title_key"])
    span = next((m for m in _QUOTE_RE.finditer(text) if title_key(m.group(0)[1:-1]) == key), None)
    if span is None:
        return None
    old = span.group(0)[1:-1]
    cands = sorted({t for t in titles if len(title_key(t)) == len(key) and title_key(t) != key
                    and _cyr(t) == _cyr(old) and _act_like(t)})
    if not cands:
        return None
    new = min(cands, key=lambda t: (_u(seed, f"act|{pair['qid']}|{t}"), t))
    return text[:span.start()] + f"«{new}»" + text[span.end():]


_AP = "[ʻʼ'‘`’]?"
# (pattern, replacement) per language and op; a replacement is a fixed word / phrase of the target op, or a stem swap
# (ru) keeping the suffix
_RU_STEM = {"delete": (r"(исключ|удал)(\w*)", "дополн"), "add": (r"(дополн|добавл)(\w*)", "исключ")}
_RU_WORD = {"replace": r"замен\w*(\s+слов)?", "edit": r"(нов\w+\s+)?редакц\w*", "renumber": r"перенумер\w*"}
_UZ = {"uz-Latn": {"delete": (r"chiqarib\s+tashla\w*", ("qoʻshilgan", "qoʻshish", "qoʻshimcha")),
                   "add": (rf"qo{_AP}sh(imcha|il|ish)\w*", ("chiqarib tashlangan", "chiqarib tashlash",
                                                            "chiqarib tashlash")),
                   "replace": (rf"(so{_AP}zlarni\s+)?almashtir\w*", None), "edit": (r"(yangi\s+)?tahrir\w*", None),
                   "renumber": (rf"raqam\w*(\s+o{_AP}zgartir\w*)?", None)},
       "uz-Cyrl": {"delete": (r"чиқариб\s+ташла\w*", ("қўшилган", "қўшиш", "қўшимча")),
                   "add": (r"қўш(имча|ил|иш)\w*", ("чиқариб ташланган", "чиқариб ташлаш", "чиқариб ташлаш")),
                   "replace": (r"(сўзларни\s+)?алмаштир\w*", None), "edit": (r"(янги\s+)?таҳрир\w*", None),
                   "renumber": (r"рақам\w*(\s+ўзгартир\w*)?", None)}}
_UZ_DEL = {"uz-Latn": "chiqarib tashlash", "uz-Cyrl": "чиқариб ташлаш"}


def _uz_form(word: str, forms: tuple) -> str:
    w = word.lower()
    if w.endswith(("gan", "ган")) or "il" in w[-6:] or "ил" in w[-6:]:
        return forms[0]
    if w.endswith(("sh", "ш")):
        return forms[1]
    return forms[2]


def _op_swap(pair: dict, seed) -> str | None:
    op, lang, text = pair["anchors"].get("op"), pair["lang"], pair["paraphrase"]
    if not op:
        return None
    if lang == "ru":
        if op in _RU_STEM:
            rx, to = _RU_STEM[op]
            out = re.sub(rx, lambda m: (to + m.group(2)) if m.group(1)[0].islower() else to.capitalize() + m.group(2),
                         text, flags=re.IGNORECASE)
        else:
            out = re.sub(_RU_WORD[op], "исключение", text, flags=re.IGNORECASE)
    else:
        rx, forms = _UZ[lang][op]
        out = re.sub(rx, lambda m: _uz_form(m.group(0), forms) if forms else _UZ_DEL[lang], text,
                     flags=re.IGNORECASE)
    return out if out != text else None


# Task 6c object_swap: the counted object outside «» replaced by another one, per template and language
_W = r"(?<![\wЀ-ӿ])"
OBJECT_SWAP = {
    "count_parts_article": {"ru": (_W + r"част(ь|и|ей|ям|ями|ях)(?![\wЀ-ӿ])", "пунктов"),
                            "uz-Cyrl": (_W + r"қисм", "банд"), "uz-Latn": (r"\bqism", "band")},
    "count_points_article": {"ru": (_W + r"пункт(ов|а|ы|ам|ами|ах)?(?![\wЀ-ӿ])", "частей"),
                             "uz-Cyrl": (_W + r"банд(?!лик)", "қисм"), "uz-Latn": (r"\bband(?!lik)", "qism")},
    "count_codes_by_law": {"ru": (_W + r"кодекс(ов|а|ы|ам|ах)?(?![\wЀ-ӿ])", "законов"),
                           "uz-Cyrl": (_W + r"кодекс", "қонун"), "uz-Latn": (r"\bkodeks", "qonun")},
    "count_laws_amended_code_year": {"ru": (_W + r"закон(ов|а|ы|ам|ами|ах)?(?![\wЀ-ӿ])", "кодексов"),
                                     "uz-Cyrl": (_W + r"қонун", "кодекс"), "uz-Latn": (r"\bqonun", "kodeks")},
    "count_articles_deleted_by_law": {"ru": (_W + r"стат(ей|ьи|ья|ьям|ьях)(?![\wЀ-ӿ])", "пунктов"),
                                      "uz-Cyrl": (_W + r"модда", "банд"), "uz-Latn": (r"\bmodda", "band")}}


def _object_swap(pair: dict, seed) -> str | None:
    rule = OBJECT_SWAP.get(pair.get("template") or "", {}).get(pair["lang"])
    if not rule:
        return None
    rx, to = rule
    text, out, last = pair["paraphrase"], [], 0
    for m in list(_QUOTE_RE.finditer(text)) + [None]:          # outside the «title» only
        end = m.start() if m else len(text)
        out.append(re.sub(rx, to, text[last:end], flags=re.IGNORECASE))
        if m:
            out.append(m.group(0))
            last = m.end()
    new = "".join(out)
    return new if new != text else None


def make_plant(kind: str, pair: dict, seed, titles=()) -> dict | None:
    text = {"number_swap": lambda: _number_swap(pair, seed), "act_swap": lambda: _act_swap(pair, seed, titles),
            "op_swap": lambda: _op_swap(pair, seed), "object_swap": lambda: _object_swap(pair, seed)}[kind]()
    if text is None or [t.s for t in ctoks(text)] == [t.s for t in ctoks(pair["paraphrase"])]:
        return None
    return {"who": f"p:{kind}:{pair['qid']}", "kind": kind, "qid": pair["qid"], "original": pair["original"],
            "paraphrase": text}


def plan_plants(pairs, seed, plant_share: float = 0.15, min_n: int = MIN_N, titles=(), kinds=PLANT_KINDS,
                topup=()) -> list[dict]:
    """Per kind `want` plants from `pairs` (sha256 order); Task 6c: a kind that the pairs cannot fill is topped up
    from `topup` bases (pairs of an earlier, already judged pass -- e.g. Task 5's accepted Q1 / Q2 paraphrases for
    op_swap, which only 9 Task 6c items can carry). Top-up plants are marked topup = True; they are judged in the
    same calls by the same prompt, so the gate (min_n, sensitivity) measures this pass's judge, thresholds unchanged."""
    want = max(int(min_n), math.ceil(plant_share * len(pairs)))
    out = []
    for kind in kinds:
        n = 0
        for src, flag in ((pairs, False), (topup, True)):
            for p in sorted(src, key=lambda p: (_u(seed, f"pk|{kind}|{p['qid']}"), p["qid"])):
                if n >= want:
                    break
                pl = make_plant(kind, p, seed, titles)
                if pl is not None:
                    out.append({**pl, "topup": True} if flag else pl)
                    n += 1
    return out


def call_items(pairs, plants) -> list[dict]:
    return [{"who": p["who"], "qid": p["qid"], "original": p["original"], "paraphrase": p["paraphrase"]}
            for p in list(pairs) + list(plants)]


def pack_calls(items, per_call: int = PER_CALL, seed=0) -> list[list[dict]]:
    """Items in sha256(seed:view|who) order, first-fit to a call of < per_call items holding no item of the same qid."""
    calls: list[tuple[list[dict], set]] = []
    for it in sorted(items, key=lambda it: (_u(seed, "view|" + it["who"]), it["who"])):
        for call, fams in calls:
            if len(call) < per_call and it["qid"] not in fams:
                call.append(it)
                fams.add(it["qid"])
                break
        else:
            calls.append(([it], {it["qid"]}))
    return [c for c, _ in calls]


# --- calls ---------------------------------------------------------------------------------------------------------

def _keys_ok(keys):
    want = sorted(keys)

    def verify(ans):
        got = sorted(v["key"] for v in ans.get("verdicts", []))
        return (got == want, None if got == want else "keys")
    return verify


def judge_calls(calls, llm) -> dict[str, dict]:
    """who -> {same: bool | None, why, call}; guard_total before every cached_json (cache/hard ledger, cap 900)."""
    out = {}
    for ci, call in enumerate(calls):
        keys = {f"P{i + 1}": it for i, it in enumerate(call)}
        inp = {"pairs": [{"key": k, "original": it["original"], "paraphrase": it["paraphrase"]}
                         for k, it in keys.items()]}
        ans = None
        for attempt in (inp, {**inp, "retry": RETRY_NOTE}):
            guard_total(llm, PASS, attempt, PROMPT, SCHEMA)
            ans = llm.cached_json(PASS, attempt, SCHEMA, _keys_ok(keys), prompt=PROMPT)
            if ans is not None:
                break
        got = {v["key"]: v for v in (ans or {}).get("verdicts", [])}
        for k, it in keys.items():
            v = got.get(k)
            out[it["who"]] = {"same": bool(v["same"]) if v else None, "why": v["why"] if v else "no verdict",
                              "call": ci}
    return out


def judge_paraphrase(pairs, llm, plant_share: float = 0.15, seed=20261003, titles=(), per_call: int = PER_CALL,
                     min_n: int = MIN_N, plants: bool = True, roots: dict | None = None, kinds=PLANT_KINDS,
                     topup=()) -> dict:
    """pairs (make_pair) -> {verdicts: qid -> {same, why}, plants: [{who, kind, qid, paraphrase, same, why}],
    calls: n}. kinds / topup: plan_plants (Task 6c); the defaults reproduce Task 5's call plan exactly."""
    pl = plan_plants(pairs, seed, plant_share, min_n, titles, kinds, topup) if plants else []
    calls = pack_calls(call_items(pairs, pl), per_call, seed)
    res = judge_calls(calls, llm)
    verdicts = {p["qid"]: {k: res[p["who"]][k] for k in ("same", "why")} for p in pairs}
    prow = [{**{k: p[k] for k in ("who", "kind", "qid", "paraphrase")}, **({"topup": True} if p.get("topup") else {}),
             **{k: res[p["who"]][k] for k in ("same", "why")}} for p in pl]
    return {"verdicts": verdicts, "plants": prow, "calls": len(calls)}


def gate(plant_rows, min_n: int = MIN_N, min_sensitivity: float = MIN_SENSITIVITY, kinds=PLANT_KINDS) -> dict:
    """Per kind: n judged plants (same is not None), caught (same is False), sensitivity; ok iff every kind has
    >= min_n judged plants and sensitivity >= min_sensitivity (never lowered: a failure is a STOP). `kinds`: the
    plant kinds of the pass (Q1 / Q2 paraphrase: PLANT_KINDS; Q3 compare: CMP_PLANT_KINDS, `same` = accepted)."""
    by_kind, reasons = {}, []
    for k in kinds:
        rows = [r for r in plant_rows if r["kind"] == k]
        judged = [r for r in rows if r["same"] is not None]
        caught = sum(r["same"] is False for r in judged)
        sens = round(caught / len(judged), 4) if judged else 0.0
        by_kind[k] = {"planted": len(rows), "judged": len(judged), "caught": caught, "sensitivity": sens,
                      "missed": [r["who"] for r in judged if r["same"]]}
        if len(judged) < min_n:
            reasons.append(f"{k}: {len(judged)} judged plants < {min_n}")
        if judged and sens < min_sensitivity:
            reasons.append(f"{k}: sensitivity {sens} < {min_sensitivity}")
    return {"ok": not reasons, "reasons": reasons, "min_n": min_n, "min_sensitivity": min_sensitivity,
            "by_kind": by_kind}


# --- Q3 compare judge (plan M7 part 2, Task 6) -----------------------------------------------------------------------
# One call judges up to PER_CALL_CMP items (keys Q1..Qn): the question, the answer and the GOLD fragments labelled by
# document (A / B, with title / type / date). Schema {verdicts: [{key, needs_both, supported, why}]}; an item is
# accepted iff needs_both and supported. Plants (caught = not accepted):
#   doc_drop     the gold of one document only (the other document's fragments removed), the same answer;
#   answer_swap  the answer of another item (same language, another packet; sha256 order), the item's own gold.
# Per kind max(min_n, ceil(plant_share * items)) plants; bases in sha256(seed:cpk|kind|qid) order. Gate: gate(...,
# kinds=CMP_PLANT_KINDS) -- the Task 5 gate, unchanged thresholds. Budget: guard_total before every cached_json.

PER_CALL_CMP = 5                     # gold texts are long (up to 5 chunks per item): 5 items per call
CMP_PLANT_KINDS = ("doc_drop", "answer_swap")

CMP_PROMPT_VERSION = 2               # R53: answer summary first, one-label rule, root-disjoint swap donors

CMP_PROMPT = """[compare judge, prompt version 2] You check comparative questions for a legal retrieval benchmark over Uzbek parliamentary documents \
(laws, Senate committee conclusions, Senate resolutions, plenary transcripts). Questions are in Russian, Uzbek \
(Cyrillic) or Uzbek (Latin).

Each item names two documents, A and B (title, type, date), and gives a QUESTION, its ANSWER and the GOLD \
FRAGMENTS: the only evidence, each labelled with its document. Judge each item on its own: never use the question, \
answer or fragments of another item. For every item, in this order:

1. answer_about: summarise in ONE sentence what THIS item's ANSWER is about (its subject, act and point), reading \
the answer itself, not what the question leads you to expect. Then check: is that the subject the QUESTION asks \
about? If not, supported = false.

2. Count the document labels the shown fragments carry. If every fragment carries the same label (only A, or only \
B), needs_both MUST be false.

needs_both = true only if answering the question needs facts from BOTH documents: a fragment of A and a fragment \
of B each contribute something the answer states, and neither document's fragments alone could answer it. If the \
fragments of only one document are shown, or the question is answerable from one document alone, or the comparison \
is only that both documents share a title or topic, needs_both = false.

supported = true only if every name, number, date, body and claim in the ANSWER is stated in the gold fragments \
(or in the documents' title / type / date), nothing is reversed, widened or added from outside knowledge, and the \
answer actually answers this question. An answer about a different subject, or relying on facts that no shown \
fragment states, is supported = false.

Judge strictly; if in doubt, false. Return exactly one verdict per item with its key, answer_about, needs_both, \
supported and why: a short reason in English (at most 25 words)."""

CMP_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdicts"],
              "properties": {"verdicts": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["key", "answer_about", "needs_both", "supported", "why"],
                  "properties": {"key": {"type": "string"}, "answer_about": {"type": "string"},
                                 "needs_both": {"type": "boolean"},
                                 "supported": {"type": "boolean"}, "why": {"type": "string"}}}}}}

CMP_RETRY_NOTE = "The previous answer did not return exactly one verdict per item key; return exactly the keys given."


def _cmp_view(item: dict, packet: dict, keep_docs=None) -> tuple[list[dict], list[dict]]:
    """(documents [{key, title, type, date}], fragments [{doc, text}]) of an item's gold, gold order; keep_docs: the
    document keys whose fragments are shown (doc_drop)."""
    keys = {d["doc_id"]: k for k, d in zip("AB", packet["docs"])}
    text = {c["chunk_id"]: (d["doc_id"], c["text"]) for d in packet["docs"] for c in d["chunks"]}
    docs = [{"key": keys[d["doc_id"]], "title": d.get("full_title") or d["title"], "type": d["doc_type"],
             "date": d.get("date")} for d in packet["docs"]]
    frags = [{"doc": keys[text[g][0]], "text": text[g][1]} for g in item["gold"]
             if keep_docs is None or keys[text[g][0]] in keep_docs]
    return docs, frags


def cmp_entry(item: dict, packet: dict, who: str | None = None, answer: str | None = None, keep_docs=None) -> dict:
    docs, frags = _cmp_view(item, packet, keep_docs)
    ans = item["answer"] if answer is None else answer
    return {"who": who or "c:" + item["qid"], "qid": item["qid"], "question": item["question"],
            "answer": ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False),
            "documents": docs, "fragments": frags}


def _pid(item: dict) -> str:
    return (item.get("meta") or {}).get("packet_id") or item["packet_id"]


def item_roots(item: dict, roots: dict | None) -> set:
    """R53: an item's documents plus their roots (roots: doc -> set of root ids, e.g. the amended acts and the folded
    title; None -> the documents only)."""
    out = set(item["docs"])
    for d in item["docs"]:
        out |= set((roots or {}).get(d, ()))
    return out


def plan_cmp_plants(items, packets: dict, seed, plant_share: float = 0.15, min_n: int = MIN_N,
                    roots: dict | None = None) -> list[dict]:
    want = max(int(min_n), math.ceil(plant_share * len(items)))
    out = []
    for kind in CMP_PLANT_KINDS:
        n = 0
        for it in sorted(items, key=lambda it: (_u(seed, f"cpk|{kind}|{it['qid']}"), it["qid"])):
            if n >= want:
                break
            p = packets[_pid(it)]
            who = f"p:{kind}:{it['qid']}"
            if kind == "doc_drop":
                keep = "A" if _u(seed, f"drop|{it['qid']}") < 0.5 else "B"
                e = cmp_entry(it, p, who, keep_docs={keep})
                if not e["fragments"]:
                    continue
            else:
                mine = item_roots(it, roots)
                donors = [d for d in items if d["lang"] == it["lang"] and _pid(d) != _pid(it)
                          and d["answer"] != it["answer"] and not item_roots(d, roots) & mine]   # R53 (c)
                if not donors:
                    continue
                donor = min(donors, key=lambda d: (_u(seed, f"swap|{it['qid']}|{d['qid']}"), d["qid"]))
                e = cmp_entry(it, p, who, answer=donor["answer"])
                e["donor"] = donor["qid"]
            e["kind"] = kind
            out.append(e)
            n += 1
    return out


def cmp_call_items(items, packets: dict, plants=()) -> list[dict]:
    return [cmp_entry(it, packets[_pid(it)]) for it in items] + [dict(p) for p in plants]


def judge_cmp_calls(calls, llm) -> dict[str, dict]:
    """who -> {needs_both, supported, ok, why, call}; None fields when an item got no verdict."""
    out = {}
    for ci, call in enumerate(calls):
        keys = {f"Q{i + 1}": it for i, it in enumerate(call)}
        inp = {"items": [{"key": k, "documents": it["documents"], "question": it["question"], "answer": it["answer"],
                          "fragments": it["fragments"]} for k, it in keys.items()]}
        ans = None
        for attempt in (inp, {**inp, "retry": CMP_RETRY_NOTE}):
            guard_total(llm, PASS, attempt, CMP_PROMPT, CMP_SCHEMA)
            ans = llm.cached_json(PASS, attempt, CMP_SCHEMA, _keys_ok(keys), prompt=CMP_PROMPT)
            if ans is not None:
                break
        got = {v["key"]: v for v in (ans or {}).get("verdicts", [])}
        for k, it in keys.items():
            v = got.get(k)
            nb = bool(v["needs_both"]) if v else None
            sp = bool(v["supported"]) if v else None
            out[it["who"]] = {"needs_both": nb, "supported": sp, "ok": (nb and sp) if v else None,
                              "answer_about": v.get("answer_about") if v else None,
                              "why": v["why"] if v else "no verdict", "call": ci}
    return out


def judge_compare(items, packets: dict, llm, plant_share: float = 0.15, seed=20261003, per_call: int = PER_CALL_CMP,
                  min_n: int = MIN_N, plants: bool = True, roots: dict | None = None) -> dict:
    """items (HardItem dicts with meta.packet_id), packets: packet_id -> packet -> {verdicts: qid -> {needs_both,
    supported, ok, why}, plants: [{who, kind, qid, donor?, same (= accepted), needs_both, supported, why}], calls}."""
    pl = plan_cmp_plants(items, packets, seed, plant_share, min_n, roots) if plants else []
    calls = pack_calls(cmp_call_items(items, packets, pl), per_call, seed)
    res = judge_cmp_calls(calls, llm)
    verdicts = {it["qid"]: {k: res["c:" + it["qid"]][k] for k in ("answer_about", "needs_both", "supported", "ok",
                                                                    "why")}
                for it in items}
    prow = [{"who": p["who"], "kind": p["kind"], "qid": p["qid"], **({"donor": p["donor"]} if "donor" in p else {}),
             "same": res[p["who"]]["ok"],
             **{k: res[p["who"]][k] for k in ("answer_about", "needs_both", "supported", "why")}}
            for p in pl]
    return {"verdicts": verdicts, "plants": prow, "calls": len(calls)}
