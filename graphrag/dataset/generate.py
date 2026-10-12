"""ds_generate (decision M3-R10): a single-turn question composed from the fact chain of a walk (single: facts of one
fragment). The generator sees facts + act / where metadata, not the raw fragment text. Every item is checked by code
(check_item); a bad item is rejected with its reason, the others in the batch are kept. Batches keep the pilot's
composition, so the full run replays the pilot from the cache.

M3-R16: the question and the answer are wholly in the target script (uz-Latn: values and titles transliterated
Cyrl -> Latn; ru: Cyrillic, Uzbek titles / original wordings inside «»); every value check compares canonical tokens
(graphrag.dataset.facts.ctoks: internal normalized form, whole tokens, M1 lemmas, number words = digits)."""
from __future__ import annotations

import re
from functools import lru_cache
from collections import Counter
from dataclasses import dataclass, field

from graphrag.data.lemma import tokens
from graphrag.data.normalize import query_internal, segment_is_russian
from graphrag.dataset.facts import (DEFAULT_MAX_FACTS, SCHEMA_FACTS, Tok, ctoks, facts_prompt, find_seq, guard_total,
                                    numbers, value_in, value_type)
from graphrag.dataset.seeds import HOP, Seed
from graphrag.dataset.walks import walk_class
from graphrag.kg.llm import _canon_input, prompt_digest

PASS = "ds_generate"
LANG_NAME = {"uz-Cyrl": "Uzbek in Cyrillic script", "uz-Latn": "Uzbek in Latin script (oʻ gʻ with ʻ)", "ru": "Russian"}
# prompt_version 4 (M3-R20, research M3-R19 R5): fragment text shown, typed facts, walk-class templates (bridge /
# aggregation), path facts, act naming, attribution; one information need
# prompt_version 5 (M3-R22, strict re-pilot reading): act identity, self-contained, names from the fragment only,
# near-copy, interrogative vs answer type, script / fluency + GLOSSARY (config dataset.legal_glossary)
PROMPT_GEN = """You write single-turn question-answer pairs for a legal retrieval dataset over Uzbek parliamentary documents
(laws, Senate decisions, committee conclusions, plenary transcripts).
Each ITEM gives a target LANG, a PATTERN (single / multi), for multi a WALK_CLASS (bridge / aggregation) and the walk,
and for each fragment its act, where, TEXT and FACTS (label, type, statement, value; values are verbatim from the text).
GLOSSARY (when given) maps Uzbek legal terms to their standard Russian renderings.
Write ONE question and its answer per item:
- ONE information need. Never join two questions ("X ким ва Y қанча?", "кто ... и сколько ...?", "... and what ...?").
- single: ask about one fact of the fragment.
- multi / bridge: fragment A cites a provision that fragment B states. Ask for B's fact and identify B only through
  A's reference (e.g. "the order that Art. 24 of Law X refers to -- in which place are the Agency's claims met?").
  A's fact that identifies the provision is a PATH fact: it may appear in the question; it is never a made-up
  condition ("if ...") that the text does not state.
- multi / aggregation: two documents deal with the SAME provision. Ask one question whose answer needs one value from
  EACH fragment: what each act changed in it, how it read before vs after, which acts / bodies deal with it. The
  answer states one value from every fragment.
- If the fragments do not support a genuine question of the item's kind, still write your best item: code rejects it.
- the question must not reveal any value it asks for, in any form (verbatim, inflected, translated, transliterated,
  digits or words, synonym). No pointers ("ушбу / мазкур / bu hujjat / данный / настоящий / этот документ").
- self-contained: a reader who has NEVER seen the fragment must understand exactly what is asked. Name the concrete
  subject (which draft, which villages, which law, which body). If the fragment lists several items, ask for all of
  them or name the specific one. Do not build a pair on a transcript filler line.
- act identity: name the act by the fragment's own `act` title (verbatim, transliterated, or -- in Russian -- its
  standard Russian rendering); for an amending law also name the amended act and its article (e.g. "ст. 347
  Таможенного кодекса"). Never name an act, decree or programme that the text only mentions, nor the year's slogan.
  attribute correctly -- a speech is what a senator / official said at a Senate plenary session (ru: "на пленарном
  заседании Сената"), a conclusion is a committee's, a decision is the Senate's; do not call a speech a law or a
  decision, nor a conclusion on an adopted law a "draft".
- every proper name (person, place, organisation, act) in the question and the answer must occur in the fragment text
  or its act title; never import names from elsewhere.
- paraphrase: ask about the rule's effect in your own words; do not reproduce the sentence with one word blanked out
  (at most ~40 % of the question's content words may come from the fragment); do not copy more than 5 consecutive
  words of the text.
- the interrogative must fit the answer type and keep the rule's logic: threshold -> "up to what amount / at what
  size" (not "from what amount ... not"), fee or manner -> "is it paid / on what terms", term -> "within what period",
  body -> "which body"; never invert a condition.
- ask as a citizen or official would; vary the form (ким / нима / қачон / қанча муддатда / қандай шартда / қайси
  ҳолда; kim / nima / qachon; кто / что / когда / в какой срок / при каком условии); end with "?"; no number that
  is not in the fragments.
- the answer (1-3 sentences) states every value you use, nothing unsupported, no number that is not in the fragments.
- script and language: write the question and the answer ENTIRELY in LANG, fluently and grammatically --
  uz-Latn: Latin letters only (oʻ gʻ with ʻ); transliterate Cyrillic values and titles letter by letter;
  uz-Cyrl: Cyrillic only (ў ғ қ ҳ; never ʻ / ʼ or any Latin letter, also inside quotes); transliterate Latin values
  and titles letter by letter;
  ru: natural Russian in Cyrillic; translate values, names and act titles (keep numbers exact) with the standard legal
  terms of GLOSSARY; you may add the original wording of a name or term once in «» after it.
- used: the labels of the facts whose values the answer states, plus (bridge) the path fact.
Return {"items": [{"key", "question", "answer", "used": [fact labels]}]}."""
SCHEMA_GEN = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["key", "question", "answer", "used"],
                               "properties": {"key": {"type": "string"}, "question": {"type": "string"},
                                              "answer": {"type": "string"},
                                              "used": {"type": "array", "items": {"type": "string"}}}}}}}
TEXT_CAP = 2500                         # M3-R20: fragment text shown to the generator (characters)
_CYR = re.compile(r"[Ѐ-ӿ]")
_LAT = re.compile(r"[A-Za-z]")
_LAT_APOS = re.compile(r"[ʻʼ‘`]")
_DEICTIC = re.compile(r"(?i)\b(?:ушбу|мазкур|ushbu|mazkur|bu)\s+(?:ҳужжат|фрагмент|матн|hujjat|matn|fragment)|"
                      r"\bданн(?:ый|ом|ого)\s+(?:документ|фрагмент|текст)|\bэтом\s+(?:документе|фрагменте)|"
                      r"\bthis\s+(?:document|fragment)")
_QUOTED = re.compile(r"«[^«»]*»|“[^“”]*”|„[^„“]*“|\"[^\"]*\"")
RETRY_NOTE = "The previous answer did not return exactly one item per key; return exactly the keys given."


@dataclass
class Candidate:
    seed: Seed
    question: str
    answer: str
    facts: list                          # graphrag.dataset.facts.Fact, one or more per C_ref fragment
    provenance: dict = field(default_factory=dict)


def _unquoted(text: str) -> str:
    """`text` without its quoted spans (nested «…«…»…» included): cited titles / original wordings."""
    prev = None
    while prev != text:
        prev, text = text, _QUOTED.sub(" ", text)
    return text


_ROMAN = re.compile(r"(?<![A-Za-z])[IVXLC]{1,6}(?![A-Za-z])")


def lang_ok(text: str, lang: str) -> bool:
    """The whole text is in the target script (M3-R16): uz-Latn has no Cyrillic; uz-Cyrl and ru have no Latin letters
    outside «…» and route as Uzbek / Russian (segment_is_russian, which ignores quoted titles)."""
    if lang != "uz-Latn":            # Roman numerals ("II–V хавфлилик синфи", "XII-сонли") are not Latin script
        text = _ROMAN.sub(" ", text or "")
    if lang == "uz-Latn":
        return not _CYR.search(text) and bool(_LAT.search(text))
    if lang == "ru":
        return bool(_CYR.search(text)) and not _LAT.search(_unquoted(text)) and segment_is_russian(text, False)
    if lang == "uz-Cyrl":                          # M3-R22: pure Cyrillic -- no Latin letter nor ʻ / ʼ, also in quotes
        return bool(_CYR.search(text)) and not _LAT.search(text) and not _LAT_APOS.search(text) \
            and not segment_is_russian(text, True)
    raise ValueError(lang)


class ExemptWords(set):
    """The words of the fragments' act / where metadata (a set, for name_only) plus `phrases`: the canonical token
    runs of each act title, where string and its location part -- the only spans the copy rule exempts."""
    phrases: tuple = ()


_TITLE_HINT = re.compile(r"(?i)тўғрисида|ҳақида|toʻgʻrisida|toʼgʼrisida|to'g'risida|haqida|кодекс|kodeks|"
                         r"закон|кодекс|о\s|об\s")


def text_titles(text: str) -> list[str]:
    """Act titles the fragment TEXT cites: quoted spans («…», “…”, "…") that read as a title ("… тўғрисида",
    "… кодекси", «О …»), >= 2 words (the act-title exemption must cover the amended law the text names)."""
    out = []
    for m in _QUOTED.finditer(text or ""):
        span = m.group(0)[1:-1].strip()
        if len(span.split()) >= 2 and _TITLE_HINT.search(span):
            out.append(span)
    return out


def exempt_words(frags: list[dict]) -> ExemptWords:
    out = ExemptWords(w.lower() for f in frags for w in re.findall(r"\w+", f"{f.get('act', '')} {f.get('where', '')}"))
    ph = set()
    for f in frags:
        act, where = f.get("act", "") or "", f.get("where", "") or ""
        loc = where[len(act):].lstrip(" ,") if act and where.startswith(act) else ""
        for s in (act, where, loc, *text_titles(f.get("text", ""))):      # + titles the text cites (amended acts)
            t = ctoks(s)
            if t:
                ph.add(t)
    out.phrases = tuple(sorted(ph, key=lambda t: (-len(t), [x.s for x in t])))
    return out


def _exempt_mask(ta: tuple[Tok, ...], exempt) -> list[bool]:
    phrases = getattr(exempt, "phrases", None)
    if phrases is None:                                    # a plain word set: each word is a one-token span
        phrases = tuple(t for w in exempt for t in (ctoks(w),) if t)
    mask = [t.num for t in ta]                             # numbers never count
    for ph in phrases:
        for i in find_seq(ta, ph, lemma=True):             # an (inflected) whole title / place is exempt
            for k in range(i, i + len(ph)):
                mask[k] = True
    return mask


def max_copy_run(a: str, b: str, exempt=frozenset()) -> int:
    """Longest run of consecutive tokens of `a` copied verbatim from `b` (canonical tokens, so a transliterated copy
    counts). Every token keeps the run going; tokens inside an exempt span (a whole act title / where string) or
    numbers add 0 to its length, so a title word inside a copied span no longer resets it (review I1)."""
    ta, tb = ctoks(a), ctoks(b)
    mask = _exempt_mask(ta, exempt)
    best, prev = 0, [0] * (len(tb) + 1)
    for i, x in enumerate(ta):
        cur = [0] * (len(tb) + 1)
        for j, y in enumerate(tb, 1):
            if x.s == y.s:
                cur[j] = prev[j - 1] + (0 if mask[i] else 1)
                best = max(best, cur[j])
        prev = cur
    return best


def lexical_overlap(question: str, texts: list[str], exempt_text: str) -> float:
    ex = set(tokens(query_internal(exempt_text)))
    q = {t for t in tokens(query_internal(question)) if t not in ex and not t.isdigit() and "#" not in t}
    if not q:
        return 0.0
    f = set().union(*(set(tokens(query_internal(t))) for t in texts))
    return len(q & f) / len(q)


def lang_family(lang: str) -> str:
    return "ru" if lang == "ru" else "uz"


def copy_limits(cfg: dict, lang: str) -> tuple[int, float]:
    """M3-R21 (2): (max copy run, max lexical overlap) for the question language's family (uz / ru) from
    cfg["copy_calibration"][family] (P90 of the Q-real questions vs their gold fragments); a family without a
    calibration (no real questions) keeps the legacy max_copy_words / max_lexical_overlap."""
    cal = ((cfg.get("copy_calibration") or {}).get("families") or {}).get(lang_family(lang)) or {}
    return (int(cal.get("max_copy_words", cfg["max_copy_words"])),
            float(cal.get("max_lexical_overlap", cfg["max_lexical_overlap"])))


def calibrate_copy(questions, q: float = 0.90) -> dict:
    """M3-R21 (2): questions = [(family, question, frags)] (frags: {act, where, text} of the gold fragments). Per
    family: the copy-run and lexical-overlap distributions (titles and numbers exempt, as in check_item_ex) and their
    q-quantile (nearest-rank, so the threshold is an observed value) as the new thresholds."""
    import math
    by: dict = {}
    for fam, qu, frags in questions:
        ex = exempt_words(frags)
        meta = " ".join(f"{x['act']} {x['where']}" for x in frags)
        d = by.setdefault(fam, {"run": [], "overlap": []})
        d["run"].append(max(max_copy_run(qu, x["text"], ex) for x in frags))
        d["overlap"].append(round(lexical_overlap(qu, [x["text"] for x in frags], meta), 4))

    def rank(v, p):
        v = sorted(v)
        return v[max(0, math.ceil(p * len(v)) - 1)]
    out = {}
    for fam, d in sorted(by.items()):
        out[fam] = {"n": len(d["run"]), "max_copy_words": int(rank(d["run"], q)),
                    "max_lexical_overlap": float(rank(d["overlap"], q)),
                    "run_hist": {str(k): d["run"].count(k) for k in sorted(set(d["run"]))},
                    "overlap_quantiles": {f"p{int(p * 100)}": float(rank(d["overlap"], p))
                                          for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}}
    return out


# ---- M3-R20 checks (research M3-R19 R2) -------------------------------------------------------------------------------
_MULTIPLIERS = {"100", "1000", "1000000", "1000000000"}          # "104 тысячи" for "104 mingta": a scale word, not a number


def _months(text: str, allowed: set[str]) -> set[str]:
    """Month numbers of dates the text writes in words ("21 декабря 2024 года" -> 2024-12-21) whose year and day are
    supported: the month is then no invented number (the source wrote it as "21-dekabr")."""
    from graphrag.data.normalize import iso_dates
    return {str(int(m.group(2))) for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", iso_dates(text or ""))
            if str(int(m.group(1))) in allowed and str(int(m.group(3))) in allowed}


def text_lang(text: str) -> str:
    """Language of a fragment: uz-Latn (Latin majority), ru (Russian by segment routing) or uz-Cyrl."""
    c, l = len(_CYR.findall(text or "")), len(_LAT.findall(text or ""))
    if l > c:
        return "uz-Latn"
    return "ru" if segment_is_russian(text or "", False) else "uz-Cyrl"


def quoted(text: str) -> str:
    return " ".join(m.group(0)[1:-1] for m in _QUOTED.finditer(text or ""))


def cross_language(target: str, frag_lang: str) -> bool:
    """Russian vs Uzbek (either script): a value must be translated; Uzbek Latin vs Cyrillic is transliteration."""
    return (target == "ru") != (frag_lang == "ru")


_CONJ_TOK = {"va", "i", "hamda", "ham", "yoki", "ili"}           # folded: ва / va, и, ҳамда, ёки / или


def _in_wo_conj(value: str, text: str) -> bool:
    """A list value stated with other joiners ("A, B" as "A va B"): whole-token match with conjunctions dropped."""
    v = tuple(t for t in ctoks(value) if t.s not in _CONJ_TOK)
    return len(v) >= 2 and bool(find_seq(tuple(t for t in ctoks(text) if t.s not in _CONJ_TOK), v))


def _loose(x: str) -> str:
    """Orthographic skeleton of a folded token: Uzbek Latin vs Cyrillic spelling variants (ts/s, yu/u, dj/j, x/h, q/k,
    doubled letters, a/o/u, iy/i, ye/e) -- "Korrupsiyaga" = "Коррупцияга", "budjet" = "бюджет", "sunniy" = "сунъий"."""
    x = x.replace("dzh", "j").replace("dj", "j").replace("ts", "s").replace("yu", "u").replace("ya", "a")
    x = x.replace("iy", "i").replace("ye", "e").replace("x", "h").replace("q", "k").replace("gh", "g")
    x = re.sub(r"(.)\1+", r"\1", x)
    return re.sub(r"[aou]", "a", x)


def _loose_keys(t) -> set:
    return {_loose(t.s)} | {_loose(l) for l in t.l}


def loose_value_in(value: str, text: str) -> bool:
    """`value` occurs in `text` as a contiguous token run under the orthographic skeleton (surface or M1 lemma);
    numbers must be equal. A fallback after the exact canonical match (script / spelling variants of one word)."""
    v, t = ctoks(value), ctoks(text)
    if not v or len(v) > len(t):
        return False
    vk = [(x.num, x.s if x.num else _loose_keys(x)) for x in v]
    tk = [(x.num, x.s if x.num else _loose_keys(x)) for x in t]
    eq = lambda a, b: a[0] == b[0] and (a[1] == b[1] if a[0] else bool(a[1] & b[1]))
    return any(all(eq(vk[k], tk[i + k]) for k in range(len(vk))) for i in range(len(tk) - len(vk) + 1))


def value_status(f, answer: str, target: str, frag_lang: str) -> str | None:
    """How the answer states fact f (M3-R20, M3-R3 п. 4 revised):
    code -- whole canonical tokens of the value are in the answer (any script, inflection, number words);
    num+judge -- a number value whose numbers are all in the answer but whose words are not (translated / reworded):
      numbers code-checked, words by the judge's agreement pass;
    judge -- a clause value, or a name / term that must be translated (Russian vs Uzbek) and is not quoted in its
      original wording: checked by the independent judge's agreement pass;
    None -- a typed value (number / name / term) that the answer does not state."""
    from graphrag.dataset.facts import value_type
    if value_in(f.value, answer) or _in_wo_conj(f.value, answer) or loose_value_in(f.value, answer):
        return "code"
    vt = getattr(f, "vtype", "") or value_type(f.value)
    if vt == "number":
        nv = numbers(f.value)
        return "num+judge" if nv and nv <= numbers(answer) else None
    if vt == "clause":
        return "judge"
    if cross_language(target, frag_lang):
        return "code" if value_in(f.value, quoted(answer)) else "judge"
    return None


_WH_UZ = ("қайси", "қандай", "қанча", "қачон", "қаер", "ким", "нима", "неча", "нечт", "qaysi", "qanday", "qancha",
          "qachon", "qayer", "kim", "nima", "necha", "necht")
_WH_RU = {"какой", "какая", "какое", "какие", "каким", "какого", "какую", "каком", "каких", "какими", "сколько", "кто",
          "кого", "кому", "кем", "когда", "где", "куда", "откуда", "почему", "зачем", "чем", "чего", "чему"}
_WH_RU_HEAD = {"что", "как"}                     # interrogative only as the first word (also conjunctions)
_CONJ = {"ва", "va", "и", "ҳамда", "hamda", "а", "шунингдек", "shuningdek", "также"}


def is_compound(question: str) -> bool:
    """Two interrogatives joined by a conjunction: two information needs glued into one question (research M3-R19:
    about 20 of 73 items; the judge ablation passes them, so code rejects them)."""
    w = [x.lower() for x in re.findall(r"[^\W\d_]+(?:[ʻʼ'`’‘][^\W\d_]+)*", question or "")]
    wh = [i for i, x in enumerate(w) if x.startswith(_WH_UZ) or x in _WH_RU or (i == 0 and x in _WH_RU_HEAD)]
    wh = [i for i in wh if not (w[i].startswith(("ким", "kim")) and len(w[i]) > 6)]   # кимё / kimyoviy ...
    if len(wh) < 2:
        return False
    return any(x in _CONJ for x in w[wh[0] + 1:wh[-1]])


_ARTICLE = re.compile(r"(?i)\d+[¹²³⁴⁵⁶⁷⁸⁹⁰]*\s*-?\s*(?:модд|modd|бо[бв]|bob)|(?:стать|пункт|част|глав)\w*\s+\d+|"
                      r"\d+[¹²³⁴⁵⁶⁷⁸⁹⁰]*\s*-?\s*(?:банд|band|қисм|qism|хатбош|xatbosh)")
_CODE = re.compile(r"(?i)кодекс|kodeks|конституц|konstitutsiya")
_OMNIBUS = ("айрим", "қонун", "ҳужжат", "ўзгартириш", "ўзгартиш", "қўшимча", "киритиш", "тўғрисида", "ҳақида",
            "муносабати", "билан", "ayrim", "qonun", "hujjat", "oʻzgartirish", "oʼzgartirish", "oʻzgartish", "qoʻshimcha",
            "qoʼshimcha", "kiritish", "toʻgʻrisida", "toʼgʼrisida", "haqida", "munosabati", "bilan", "закон", "внесени",
            "изменени", "дополнени", "некотор", "отдельн", "законодательн", "акт", "республик", "узбекистан", "связи",
            "ўзбекистон", "республикаси", "республикасининг", "oʻzbekiston", "respublikasi", "respublikasining", "ва",
            "va", "янада", "yanada", "такомиллаштирилиши", "takomillashtirilishi", "совершенствован", "мер", "сфер")


def _content(text: str) -> set[str]:
    from graphrag.dataset.facts import _generic
    gen = _generic()
    om = frozenset(t.s for w in _OMNIBUS for t in ctoks(w))
    out = set()
    for t in ctoks(text):
        if t.num or len(t.s) < 4 or t.s in gen or t.s in om or any(t.s.startswith(x) for x in om if len(x) >= 5):
            continue
        out |= {t.s} | set(t.l)
    return out


def article_without_act(question: str, frags: list[dict], walk=()) -> bool:
    """The question cites an article / part / item number but names no act distinctively (research M3-R19: [60] "21-
    модданинг учинчи қисмида" under an omnibus title). Named = a code / constitution, or >= 2 distinctive content
    words shared with a fragment's act title or an external act of the walk (omnibus words such as "айрим қонун
    ҳужжатларига ўзгартириш" do not count)."""
    if not _ARTICLE.search(question or ""):
        return False
    if _CODE.search(question):
        return False
    q = _content(question)
    lang_q = "ru" if segment_is_russian(question or "", False) else "uz"
    if any((lang_q == "ru") != (text_lang(x.get("text", "")) == "ru") for x in frags) and \
            any(len(m.group(0)[1:-1].split()) >= 1 and (ru_span(m.group(0)[1:-1]) or _TITLE_HINT.search(m.group(0)))
                for m in _QUOTED.finditer(question or "")):
        return False                 # a translated title in «» (cannot be code-checked across languages; the readers do)
    names = [x.get("act", "") for x in frags] + [v.split("/")[1] for h in walk for v in (h[0], h[2])
                                                 if v.startswith("EXT/") and "/" in v]
    names += [t for x in frags for t in text_titles(x.get("text", ""))]               # acts the fragment text names
    names += [n for x in frags for n in (x.get("amended") or ()) if _named_in_text(n, x.get("text", ""))]
    names += [x[k] for x in frags for k in ("discussed_act", "amended_act") if x.get(k)]
    return not any(_content(n) and len(q & _content(n)) >= min(2, _n_content_words(n)) for n in names)


def _n_content_words(text: str) -> int:
    """Distinct content words (surfaces) of a name -- a one-word title ("Qurol toʻgʻrisida") needs one shared word."""
    from graphrag.dataset.facts import _generic
    gen = _generic()
    om = frozenset(t.s for w in _OMNIBUS for t in ctoks(w))
    return len({t.s for t in ctoks(text) if not t.num and len(t.s) >= 4 and t.s not in gen and t.s not in om
                and not any(t.s.startswith(x) for x in om if len(x) >= 5)})


# ---- M3-R22 (strict re-pilot reading): names only from the fragments, the act named is the fragment's own ------------
_NAME_STOP = ("qaysi", "qanday", "qancha", "qachon", "qayerda", "kim", "kimlar", "nima", "necha", "nechta", "ha", "yoʻq",
              "senat", "senati", "senatning", "senatining", "oliy", "majlis", "majlisi", "majlisining", "oʻzbekiston",
              "respublikasi", "respublikasining", "qonunchilik", "palatasi", "vazirlar", "mahkamasi", "prezident",
              "prezidenti", "qonun", "qonuni", "qonunga", "qonunda", "kodeks", "kodeksi", "qaror", "qarori", "ushbu",
              "mazkur", "bu", "unga", "ular", "javob", "maʼlumotga", "koʻra", "binoan", "muvofiq", "mazkur", "shu")


@lru_cache(maxsize=1)
def _name_stop() -> frozenset:
    from graphrag.dataset.facts import _generic
    return frozenset(t.s for w in _NAME_STOP for t in ctoks(w)) | _generic()


_CAPRUN = re.compile(r"(?:(?<=^)|(?<=[\s(«“\"„-]))((?:[A-ZА-ЯЁЎҚҒҲ][\w'ʻʼ’‘`-]*)(?:\s+[A-ZА-ЯЁЎҚҒҲ][\w'ʻʼ’‘`-]*)*)")
_RU_PROPER = ("Name", "Surn", "Patr", "Geox", "Orgn", "Trad")


_ABBR = {"бмт", "bmt", "оон", "мдҳ", "mdh", "снг", "ққс", "qqs", "ндс", "яим", "yaim", "ввп", "еи", "ес", "жст", "вто",
         "юнеско", "unesco", "юнисеф", "unicef", "хмт", "мот", "шҳт", "шос", "ақш", "aqsh", "сша", "ехҳт", "обсе", "мвф", "хвф",
         "жсст", "воз", "атм", "aj", "аж", "мчж", "mchj", "ооо", "ао"}


def _known_abbreviation(w: str, ref: str) -> bool:
    """An all-caps abbreviation that is standard (БМТ, ҚҚС, ООН ...) or whose letters are the initials of consecutive
    words of the reference text (ҚҚС <- "қўшилган қиймат солиғи")."""
    x = re.sub(r"[^\w]", "", w)
    if not (2 <= len(x) <= 6 and x.upper() == x):
        return False
    if x.lower() in _ABBR:
        return True
    if len(x) < 3:
        return False
    ini = "".join(t[0] for t in re.findall(r"[^\W\d_]+", (ref or "").lower()))
    return x.lower() in ini


def _ru_geo(w: str) -> bool:
    """A Russian dictionary place name (pymorphy Geox): a translated country / city."""
    from graphrag.data.lemma import _morph
    p = _morph().parse(w)[0]
    return "Geox" in p.tag


def _ru_common(w: str) -> bool:
    """A Russian word the dictionary knows as a common word (not a person / place / organisation name)."""
    from graphrag.data.lemma import _morph
    p = _morph().parse(w)[0]
    return p.is_known and not any(t in p.tag for t in _RU_PROPER)


def _ref_text(frags: list[dict]) -> str:
    """Reference text for names: act, where, text, amended acts and -- M3-R29 -- the transcript's speaker and
    session title (record meta)."""
    return " ".join(f"{x.get('act', '')} {x.get('where', '')} {x.get('text', '')} {' '.join(x.get('amended') or ())} "
                    f"{x.get('speaker') or ''} {(x.get('session') or {}).get('title') or ''} "
                    f"{x.get('discussed_act') or ''} {x.get('amended_act') or ''}" for x in frags)


_UZ_CYR = re.compile(r"[ўқғҳЎҚҒҲ]")


def ru_span(span: str) -> bool:
    """A quoted span that is Russian (a translated title): segment routing, or -- short titles such as «О банках»,
    «О СМИ» that routing cannot decide -- Cyrillic only, no Uzbek letter (ў қ ғ ҳ), and a Russian dictionary word or
    an abbreviation among its words."""
    if segment_is_russian(span, False):
        return True
    if _LAT.search(span) or _UZ_CYR.search(span) or not _CYR.search(span):
        return False
    if re.match(r"(?:О|Об|Обо)\s+\S", span):
        return True                  # a Russian law title «О пробации», «Об электронной коммерции»
    words = [w for w in re.findall(r"[А-Яа-яЁё]+", span) if len(w) > 2 or re.fullmatch(r"[А-ЯЁ]{2}", w)]
    ok = [bool(re.fullmatch(r"[А-ЯЁ]{2,6}", w)) or _ru_common(w.lower()) for w in words]
    return 0 < len(words) <= 6 and sum(ok) >= 0.6 * len(words)


def _stem_near(toks, ref_toks) -> bool:
    """Every token shares a >= 4-letter prefix (its length minus 3) with a reference token (folded surfaces)."""
    surf = {t.s for t in ref_toks if not t.num}
    for t in toks:
        k = max(4, len(t.s) - 3)
        if len(t.s) < 4 or not any(x[:k] == t.s[:k] for x in surf if len(x) >= k):
            return False
    return True


_RU_NAME_ENDINGS = ("ыми", "ому", "ого", "ему", "ыми", "ой", "ым", "ом", "ем", "ую", "ая", "ий", "а", "у", "е", "ы")


def name_key(word: str) -> str:
    """An orthography-tolerant key of a proper noun: Latin fold (Cyrillic transliterated, apostrophes dropped),
    ў/у (o/u), қ/к (q/k), ғ/г, ҳ/х (x/h), "iy"/"i" merged, a Russian case ending of a surname dropped
    (Туйчиеву, Тўйчиев, Tuychiyev, Toʻychiyev -> one key)."""
    w = re.sub(r"[^\w]", "", (word or "").lower())
    uz = re.match(r"(.{4,}?(?:ев|ов|ин|ий|ёв|а|я))(нинг|дан|га|да|ни)$", w) if _CYR.search(w) else None
    if uz:                                               # Uzbek case suffix on a Cyrillic name (Тўйчиевга)
        w = uz.group(1)
    elif _CYR.search(w):
        for e in _RU_NAME_ENDINGS:
            if len(w) > len(e) + 4 and w.endswith(e) and (w[:-len(e)].endswith(("ев", "ов", "ин", "ёв")) or e in ("а", "у", "е")):
                w = w[:-len(e)]
                break
    t = ctoks(w)
    f = t[0].s if t else w
    f = re.sub(r"(?<=[a-z]{4})(ning|nin|dan|ga|da|ni|lar)$", "", f)     # Uzbek case suffixes (Ҳамраевнинг)
    f = re.sub(r"(ev|ov)a$", r"\1", f)
    return _loose(f)                       # Бурханов = Burxonov, Темурджон = Temurjon (orthographic skeleton)


@lru_cache(maxsize=1)
def _role_stems() -> tuple:
    from graphrag import config as _cfg
    try:
        return tuple(_cfg.load("train")["dataset"].get("role_words") or ())
    except Exception:
        return ()


def unsupported_roles(text: str, frags: list[dict]) -> list[str]:
    """Instructions v7: role / title words of a speaker (config dataset.role_words, folded stems: senator, deputat,
    rais, vazir ...) in `text` that neither the fragment text nor its meta (act, where, speaker, session, discussed /
    amended act) contains -- "сенатор Мирзаев" when nothing says he is a senator."""
    stems = _role_stems()
    if not stems:
        return []
    ref = {t.s for t in ctoks(_ref_text(frags)) if not t.num}
    out = []
    for t in ctoks(text or ""):
        if t.num:
            continue
        st = next((x for x in stems if t.s.startswith(x)), None)
        if st and not any(r.startswith(st) for r in ref):
            out.append(t.s)
    return out


@lru_cache(maxsize=1)
def _gazetteer() -> dict:
    from graphrag import config as _cfg
    try:
        g = _cfg.load("train")["dataset"].get("name_gazetteer") or {}
    except Exception:
        g = {}
    return {k.lower(): [name_key(v) for v in vs] for k, vs in g.items()}


def name_variant_ok(word: str, ref_keys: set[str]) -> bool:
    """A proper noun is supported by an orthographic / transliteration / declension variant in the reference text, or
    -- a Russian region / city name -- by the gazetteer's Uzbek form ("Ташкентская" <- "Toshkent")."""
    k = name_key(word)
    if len(k) >= 4 and (k in ref_keys or any(len(r) >= 4 and (r.startswith(k) or k.startswith(r)) and
                                             abs(len(r) - len(k)) <= 2 for r in ref_keys)):
        return True
    low = (word or "").lower()
    return any(low.startswith(stem) and any(u in ref_keys for u in forms) for stem, forms in _gazetteer().items())


def foreign_names(text: str, frags: list[dict], lang: str) -> list[str]:
    """M3-R22 (2): proper nouns of `text` -- quoted spans and capitalised token runs -- that are not in the fragment
    texts / act titles / amended acts, compared as canonical tokens (to_internal fold, transliteration, M1 lemmas).
    Institutional words (Senat, Oliy Majlis, Qonun ...) and wh-words are exempt. A Russian text over Uzbek fragments
    (translation) checks only its genuine names (dictionary-unknown or tagged person / place / organisation) and
    its quoted Uzbek originals; a Russian quoted rendering is left to ds_quality."""
    ref = _ref_text(frags)
    rt = ctoks(ref)
    ref_keys = {name_key(w) for w in re.findall(r"[^\W\d_]+(?:[ʻʼ'`’‘][^\W\d_]+)*", ref)}
    stop = _name_stop()
    cross = any(cross_language(lang, text_lang(x.get("text", ""))) for x in frags)
    bad = []
    for m in _QUOTED.finditer(text or ""):
        span = m.group(0)[1:-1].strip()
        if not span or (cross and lang == "ru" and ru_span(span)) or (cross and lang != "ru" and not ru_span(span)):
            continue                 # a title rendered in the question's language over a fragment in the other one
        if not find_seq(rt, ctoks(span), lemma=True):
            words = [t for t in ctoks(span) if not t.num and t.s not in stop]
            if words and sum(1 for t in words if find_seq(rt, (t,), lemma=True)) < 0.8 * len(words):
                bad.append(span)
    rest = _unquoted(text or "")
    for m in _CAPRUN.finditer(rest):
        words = m.group(1).split()
        before = rest[:m.start(1)].rstrip(" «“\"„(-")
        if len(words) == 1 and (not before or before[-1] in ".?!:;"):
            continue                        # a lone sentence-initial capital is ordinary capitalisation
        if (not before or before[-1] in ".?!:;") and find_seq(rt, tuple(t for t in ctoks(words[0].lower()) if not t.num),
                                                              lemma=True):
            words = words[1:]               # a sentence-initial common word the text uses (Хулосада Россия, …)
        for w in words:
            toks = [t for t in ctoks(w) if not t.num]
            if not toks or all(t.s in stop or (set(t.l) & stop) for t in toks):
                continue
            if find_seq(rt, tuple(toks), lemma=True):
                continue
            if lang == "ru" and cross and _stem_near(toks, rt):
                continue                                   # a transliterated, Russian-inflected name (Тошовой / Toshova)
            if lang == "ru" and _CYR.search(w) and _ru_common(re.sub(r"[^\w-]", "", w)):
                continue                                   # a translated common word, not a name
            if lang == "ru" and cross and re.fullmatch(r"[А-ЯЁ]{2,6}", re.sub(r"[^\w]", "", w)):
                continue                                   # a Russian abbreviation (СНГ, ОБСЕ): a translation
            if name_variant_ok(w, ref_keys):
                continue                                   # a spelling / script / declension variant, or a gazetteer name
            if _known_abbreviation(w, ref):
                continue                                   # БМТ / ҚҚС / ООН, or initials of words the text spells out
            if lang == "ru" and cross and _CYR.search(w) and _ru_geo(re.sub(r"[^\w-]", "", w)):
                continue                                   # a translated country / place name (Китае, Южной Корее)
            bad.append(w)
    return bad


def _named_in_text(name: str, text: str) -> bool:
    """M3-R29: an act name occurs in the fragment text -- >= 60 % of its distinctive content words (lemma-folded)."""
    c = _content(name)
    return bool(c) and len(c & _content(text)) >= 0.6 * len(c)


_ACT_TAIL = re.compile(r"(?i)(?:тўғрисида|ҳақида|toʻgʻrisida|toʼgʼrisida|to'g'risida|haqida)(?:ги|gi)\s+"
                       r"(?:қонун|qonun|кодекс|kodeks)")


def wrong_act(question: str, frags: list[dict], lang: str) -> bool:
    """M3-R22 (2): the act the question names (a quoted title of >= 3 content words, "<X> кодекси / kodeksi", or -- M3-R29
    -- an unquoted "<X> тўғрисидаги Қонун / қарор") is the fragment's own act title or, for an amending law, an
    amended act (frag["amended"]) -- >= 60 % of its content words in one of them. M3-R29: an amended act counts only
    if its name occurs in the fragment TEXT (else the question must reach the provision through the amending act).
    A Russian rendering over Uzbek titles cannot be code-checked (left to the reader)."""
    names = [x.get("act", "") for x in frags] + [n for x in frags for n in (x.get("amended") or ())
                                                 if _named_in_text(n, x.get("text", ""))]
    names += [x[k] for x in frags for k in ("discussed_act", "amended_act") if x.get(k)]     # graph meta
    names += [t for x in frags for t in text_titles(x.get("text", ""))]                    # acts the text cites
    ok = lambda span: any(_content(span) and len(_content(span) & _content(n)) >= 0.6 * len(_content(span))
                          for n in names)
    cross = any(cross_language(lang, text_lang(x.get("text", ""))) for x in frags)
    for m in _QUOTED.finditer(question or ""):
        span = m.group(0)[1:-1]
        if not _TITLE_HINT.search(span) and not _TITLE_HINT.search((question or "")[m.end():m.end() + 4]):
            continue                 # a quoted fund / project / programme name («Ёшлар дафтари»), not an act title
        if len(_content(span)) >= 3 and not (cross and (ru_span(span) or lang != "ru")) and not ok(span):
            return True
    if not cross:
        stop = _name_stop()
        for m in re.finditer(r"(?i)([^\W\d_]+(?:\s+[^\W\d_]+)?)\s+(?:кодекс|kodeks)", _unquoted(question or "")):
            head = " ".join(w for w in m.group(1).split() if not ({t.s for t in ctoks(w)} & stop)
                            and not w.lower().startswith(_WH_UZ))
            if _content(head) and not any(_content(head) & _content(n) for n in names) and \
                    not any(loose_value_in(head, t) for t in names + [x.get("text", "") for x in frags]):
                return True
        uq = _unquoted(question or "")
        for m in _ACT_TAIL.finditer(uq):                                   # M3-R29: unquoted "<X> тўғрисидаги Қонун"
            head = _content(" ".join(re.findall(r"[^\W\d_]+(?:[ʻʼ'`’‘][^\W\d_]+)*", uq[:m.start()])[-2:]))
            if head and not any(head & _content(n) for n in names):
                return True
    return False


# ---- M3-R29: blanked-sentence near-copies; stenogram session dates ---------------------------------------------------
_SENT = re.compile(r"(?<=[.!?;:])\s+|\n+|\s+[–—-]\s+(?=\d+\))")


def sentences(text: str) -> list[str]:
    return [x for x in _SENT.split(text or "") if len([t for t in ctoks(x) if not t.num]) >= 3]


def _lemma_key(t) -> str:
    return min(t.l) if t.l else t.s


def near_copy_ratio(question: str, frags: list[dict]) -> float:
    """M3-R29: how much of the question is ONE fragment sentence in order -- the share of the question's tokens (act
    titles / where and numbers exempt, M1 lemmas, any script) covered by the longest common subsequence
    (difflib.SequenceMatcher matching blocks) with any single sentence of any fragment. A blanked-sentence near-copy
    ("… қачон келиб тушган?" over "… 18 февралда келиб тушган") scores near 1."""
    from difflib import SequenceMatcher
    qt = ctoks(question)
    mask = _exempt_mask(qt, exempt_words(frags))
    q = [_lemma_key(t) for t, m in zip(qt, mask) if not m]
    if len(q) < 3:
        return 0.0
    best = 0.0
    for x in frags:
        for sent in sentences(x.get("text", "")):
            st = [_lemma_key(t) for t in ctoks(sent) if not t.num]
            m = SequenceMatcher(None, q, st, autojunk=False)
            best = max(best, sum(b.size for b in m.get_matching_blocks()) / len(q))
    return round(best, 4)


def calibrate_near_copy(questions, q: float = 0.90) -> dict:
    """questions = [(family, question, frags)] -> per family {n, max_ratio (nearest-rank q-quantile), quantiles}."""
    import math
    by: dict = {}
    for fam, qu, frags in questions:
        by.setdefault(fam, []).append(near_copy_ratio(qu, frags))
    rank = lambda v, p: sorted(v)[max(0, math.ceil(p * len(v)) - 1)]
    return {fam: {"n": len(v), "max_ratio": float(rank(v, q)),
                  "quantiles": {f"p{int(p * 100)}": float(rank(v, p)) for p in (0.5, 0.75, 0.9, 0.95, 0.99)}}
            for fam, v in sorted(by.items())}


_TITLE_DATES = re.compile(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\.(\d{1,2})\.(\d{2,4})")


def _title_dates(text: str) -> set[str]:
    """Numbers of the dates written in a session title ("29-30.04.25", "28-29.05.2025"): days, month and the FULL
    year (a two-digit year yy is 20yy) -- so "2025" in a question is supported by a title that says "25"."""
    out = set()
    for m in _TITLE_DATES.finditer(text or ""):
        a, b, mo, y = int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3)), int(m.group(4))
        if not (1 <= mo <= 12 and 1 <= a <= 31 and a <= b <= 31):
            continue
        out |= {str(d) for d in range(a, b + 1)} | {str(mo), str(y if y >= 100 else 2000 + y)}
    return out


def session_numbers(frags: list[dict]) -> set[str]:
    """M3-R29 fix: numbers a question may use to date a plenary session -- the session's year, month and days (meta
    date and title range, two-digit years expanded) and the dates written in act / where titles."""
    out = set()
    for x in frags:
        sd = session_days(x.get("session"))
        if sd:
            out |= {sd[0], sd[1]} | sd[2]
        for t in (x.get("act"), x.get("where"), (x.get("session") or {}).get("title")):
            out |= _title_dates(t)
    return out


def session_iso(session: dict) -> list[str]:
    """Every day of the sitting as a full ISO date (2025-05-28, 2025-05-29) from meta date + title range."""
    sd = session_days(session)
    return sorted(f"{sd[0]}-{int(sd[1]):02d}-{int(d):02d}" for d in sd[2]) if sd else []


def session_days(session: dict) -> tuple[str, str, set[str]] | None:
    """(year, month, {days}) of a plenary session from its meta date and title range ("29-30.04.25")."""
    d = (session or {}).get("date") or ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", d)
    if not m:
        return None
    y, mo, days = str(int(m.group(1))), str(int(m.group(2))), {str(int(m.group(3)))}
    title = (session or {}).get("title") or ""
    t = _TITLE_DATES.search(title)
    if t and str(int(t.group(3))) == mo:
        a, b = int(t.group(1)), int(t.group(2) or t.group(1))
        days |= {str(k) for k in range(a, b + 1)}
    u = _UZ_TITLE.search(title)                  # "(2024-yil 18-19- va 21-dekabr)", "(2025-yil 23-, 24 – 25-yanvar)"
    if u and u.group(1) == y and _month_of(u.group(3)) == int(mo):
        days |= _day_set(u.group(2))
    return y, mo, days


_UZ_TITLE = re.compile(r"\((\d{4})-yil\s+([\d\s,\-–va]+?)\s*-?\s*([^\W\d_]+)\)")


def _month_of(word: str) -> int | None:
    return next((m for m, rx in _MONTH_RE.items() if rx.fullmatch(word or "")), None)


def _day_set(part: str) -> set[str]:
    """Days of "3 – 4", "17-, 19", "18-19- va 21", "23-, 24 – 25": ranges a–b and single days."""
    out = set()
    for m in re.finditer(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?", part or ""):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if 1 <= a <= b <= 31:
            out |= {str(k) for k in range(a, b + 1)}
    return out


def session_undated(question: str, frags: list[dict]) -> bool:
    """M3-R29: a question over a plenary transcript names the session date (year, month and a day of the session;
    any script / format that graphrag normalizes to a date)."""
    for x in frags:
        sd = session_days(x.get("session"))
        if sd is None:
            continue
        y, mo, days = sd
        nums = numbers(question) | _title_dates(question)        # "2025 йил 28–29 май" or the title form "28-29.05.25"
        month = mo in nums or bool(_MONTH_RE[int(mo)].search(question or ""))    # "29–30 апрелдаги / aprel / апреля"
        if not (y in nums and month and days & nums):
            return True
        if _days_near_month(question, int(mo)) - days:           # M3-R31: a day outside the sitting (meta)
            return True
    return False


def session_range_gap(question: str, frags: list[dict]) -> bool:
    """Instructions v8: a written day range "a–b" next to the session month spans a day that is not a sitting day
    ("18–21 декабря" for 18, 19, 21 December; "17–19 марта" for 17 and 19 March)."""
    for x in frags:
        sd = session_days(x.get("session"))
        if sd is None:
            continue
        _, mo, days = sd
        for m in _MONTH_RE[int(mo)].finditer(question or ""):
            win = re.sub(r"\d{4}", " ", (question or "")[max(0, m.start() - 16):m.start()])
            for a, b in re.findall(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})(?=\D*$)", win):
                a, b = int(a), int(b)
                if 1 <= a < b <= 31 and any(str(d) not in days for d in range(a, b + 1)):
                    return True
    return False


_RELATIVE_YEAR = re.compile(r"(?i)\b(?:жорий\s+йил|joriy\s+yil|бу\s+йил|bu\s+yil|ўтган\s+йил|oʻtgan\s+yil|oʼtgan\s+yil|"
                            r"o'tgan\s+yil|келгуси\s+йил|kelgusi\s+yil|текущ\w*\s+год|(?:в\s+)?этом\s+году|этого\s+года|"
                            r"прошл\w*\s+год\w*|следующ\w*\s+год\w*)\w*")


def relative_year(text: str) -> bool:
    """Instructions v8: a relative year («жорий йил», «текущий год», «в прошлом году» ...) without the actual year
    (a 4-digit year within ~30 characters after or before it)."""
    for m in _RELATIVE_YEAR.finditer(text or ""):
        near = (text or "")[max(0, m.start() - 30):m.end() + 30]
        if not re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", near):
            return True
    return False


def _days_near_month(question: str, mo: int) -> set[str]:
    """Day numbers written right before the session month's name ("20–21 сентября", "23–25-yanvar")."""
    out = set()
    for m in _MONTH_RE[mo].finditer(question or ""):
        win = re.sub(r"\d{4}", " ", (question or "")[max(0, m.start() - 16):m.start()])
        mm = re.search(r"((?:\d{1,2}\s*[-–,]?\s*(?:va|и|ва)?\s*)+)-?\s*$", win)
        if mm:                       # a written range a–b names its first and last day ("17–19 март" for 17, 19)
            out |= {str(int(d)) for d in re.findall(r"\d{1,2}", mm.group(1)) if 1 <= int(d) <= 31}
    return out


_MONTH_STEMS = {1: ("январ", "yanvar"), 2: ("феврал", "fevral"), 3: ("март", "mart"), 4: ("апрел", "aprel"),
                5: ("май", "мая", "мае", "may"), 6: ("июн", "iyun"), 7: ("июл", "iyul"), 8: ("август", "avgust"),
                9: ("сентябр", "sentabr", "sentyabr"), 10: ("октябр", "oktabr", "oktyabr"),
                11: ("ноябр", "noyabr"), 12: ("декабр", "dekabr")}
_MONTH_RE = {m: re.compile(r"(?i)\b(?:" + "|".join(v) + r")\w*") for m, v in _MONTH_STEMS.items()}


_UZ_LETTER = re.compile(r"[ўқғҳЎҚҒҲ]")
RU_GLOSS_MAX_WORDS = 6


def ru_impure(text: str, answer: bool) -> bool:
    """M3-R31: a Russian question / answer is entirely Russian -- no Uzbek-specific Cyrillic letter (ў қ ғ ҳ) outside
    «»; at most ONE quoted Uzbek original (Uzbek letters or Latin script), in an answer at most 6 words and no digits
    (a name or term after its translation, not a pasted clause or figure such as «23 та лойиҳа»)."""
    if _UZ_LETTER.search(_unquoted(text or "")):
        return True
    uz = [m.group(0)[1:-1] for m in _QUOTED.finditer(text or "")
          if _UZ_LETTER.search(m.group(0)) or _LAT.search(m.group(0))]
    if len(uz) > 1:
        return True
    return answer and any(len(x.split()) > RU_GLOSS_MAX_WORDS or re.search(r"\d", x) for x in uz)


_AMENDING = re.compile(r"(?i)(?:ўзгартириш|ўзгартиш|қўшимча|oʻzgartirish|oʼzgartirish|o'zgartirish|oʻzgartish|"
                       r"qoʻshimcha|qoʼshimcha|изменени|дополнени)")


def amending_locus(question: str, frags: list[dict]) -> bool:
    """M3-R31: over an AMENDING law, a question that cites an article / item must name the AMENDED act (a code, or an
    amended act the fragment text names) -- the amending law's own article is never the locus."""
    if not _ARTICLE.search(question or ""):
        return False
    ru_q = segment_is_russian(question or "", False)
    for x in frags:
        if not _AMENDING.search(x.get("act") or ""):
            continue
        if x.get("amended_act") and ru_q != (text_lang(x.get("text", "")) == "ru") and \
                any(ru_span(m.group(0)[1:-1]) or _TITLE_HINT.search(m.group(0)) for m in _QUOTED.finditer(question)):
            continue                 # the amended act (graph meta) named in translation -- the readers check it
        if _CODE.search(question):
            continue
        named = [n for n in (x.get("amended") or ()) if _named_in_text(n, x.get("text", ""))]
        named += [x["amended_act"]] if x.get("amended_act") else []           # the graph's single AMENDS target
        if not any(_content(n) and len(_content(n) & _content(question)) >= 0.6 * len(_content(n)) for n in named):
            return True
    return False


def check_item_ex(item: dict, seed: Seed, facts_by_label: dict, frags: list[dict], cfg: dict,
                  path_chunks: set | None = None) -> tuple[str | None, dict]:
    """M3-R20 check of one generated item. Returns (reject reason | None, info) with info = {answer: [facts the answer
    states], path: [multi-hop facts the question relies on], status: {fact_id: code | num+judge | judge}}.
    - used = the listed facts the answer STATES (value_status); a listed fact it does not state is dropped, not fatal;
    - bridge: a listed fact of the citing (anchor) fragment whose value is in the question is a PATH fact (it identifies
      the cited provision; any other fact in the question is a leak); multi: every C_ref
      fragment needs a stated or path fact whose value no other fragment's text contains (else `redundant`);
    - the leak rule is unchanged for every fact the answer states; numbers may come from any C_ref fragment."""
    q, a = item["question"].strip(), item["answer"].strip()
    info = {"answer": [], "path": [], "status": {}}
    if not q.endswith("?") or len(q) < 15 or not a:
        return "shape", info
    if any(u not in facts_by_label for u in item["used"]):
        return "unknown_fact", info
    multi = len(seed.chunks) > 1
    # a PATH fact comes only from the citing (anchor) fragment of a bridge walk; any other fact in the question leaks
    path_ok = {seed.chunks[0]} if multi and seed.walk and walk_class(seed.walk) == "bridge" else set()
    if path_chunks is not None:                  # M3-R21 ds_bridge: the anchor's fact identifies the neighbour
        path_ok = set(path_chunks)
    lang_of = {x["id"]: text_lang(x["text"]) for x in frags}
    label_of = {c: f"F{i + 1}" for i, c in enumerate(seed.chunks)}
    listed = [facts_by_label[u] for u in dict.fromkeys(item["used"])]
    ans, path, status = [], [], {}
    for f in listed:
        if f.chunk_id in path_ok and value_in(f.value, q):
            path.append(f)
            continue
        st = value_status(f, a, seed.lang, lang_of.get(label_of.get(f.chunk_id), "uz-Cyrl"))
        if st:
            ans.append(f)
            status[f.fact_id] = st
    info.update(answer=ans, path=path, status=status)
    if not ans:
        return "value_missing_in_answer", info
    if multi:
        text_of = {c: x["text"] for c, x in zip(seed.chunks, frags)}
        cover = {}
        for f in ans + path:
            others = [t for c, t in text_of.items() if c != f.chunk_id]
            if path_chunks is not None and f in path:     # ds_bridge: the anchor's link fact names the cited
                cover.setdefault(f.chunk_id, []).append(True)   # provision (the target states it); ablation tests it
                continue
            cover.setdefault(f.chunk_id, []).append(not any(value_in(f.value, t) for t in others))
        if set(cover) != set(seed.chunks):
            return "chain_incomplete", info
        if not all(any(v) for v in cover.values()):
            return "redundant", info
    elif {f.chunk_id for f in ans} != set(seed.chunks):
        return "chain_incomplete", info
    meta = " ".join(f"{x['act']} {x['where']}" for x in frags)
    # the act names the packet meta gives (discussed / amended act, amended unit) carry numbers a question may use
    meta_nums = numbers(meta) | numbers(" ".join(f"{x.get('discussed_act') or ''} {x.get('amended_act') or ''} "
                                                 f"{x.get('amended_unit') or ''}" for x in frags))
    q_nums = numbers(q)
    if any(value_in(f.value, q) for f in ans) or any((numbers(f.value) - meta_nums) & q_nums for f in ans):
        return "answer_in_question", info
    allowed = set().union(*(numbers(x["text"]) for x in frags)) | meta_nums | _MULTIPLIERS | session_numbers(frags)
    allowed |= _months(q, allowed) | _months(a, allowed)
    if not numbers(a) <= allowed or not q_nums <= allowed:
        return "unsupported_number", info
    if not lang_ok(q, seed.lang) or not lang_ok(a, seed.lang):
        return "language", info
    if _DEICTIC.search(q):
        return "deictic", info
    if is_compound(q):
        return "compound", info
    if article_without_act(q, frags, seed.walk):
        return "article_without_act", info
    if foreign_names(q, frags, seed.lang) or foreign_names(a, frags, seed.lang):     # M3-R22 (2)
        return "foreign_name", info
    if unsupported_roles(q, frags) or unsupported_roles(a, frags):                    # instructions v7
        return "unsupported_role", info
    if wrong_act(q, frags, seed.lang):
        return "wrong_act", info
    if session_undated(q, frags):                                                    # M3-R29
        return "session_undated", info
    if session_range_gap(q, frags) or session_range_gap(a, frags):                   # instructions v8
        return "session_range_gap", info
    if relative_year(q) or relative_year(a):                                         # instructions v8
        return "relative_year", info
    if seed.lang == "ru" and (ru_impure(q, False) or ru_impure(a, True)):             # M3-R31
        return "ru_impure", info
    if amending_locus(q, frags):                                                     # M3-R31
        return "amending_locus", info
    nc = (cfg.get("near_copy") or {}).get("max_ratio")
    if nc is not None and near_copy_ratio(q, frags) >= float(nc):                    # M3-R29 blanked-sentence copy
        return "near_copy", info
    ex = exempt_words(frags)
    max_copy, max_overlap = copy_limits(cfg, seed.lang)
    if max(max_copy_run(q, x["text"], ex) for x in frags) > max_copy:
        return "copied", info
    titles = " ".join(t for x in frags for t in text_titles(x.get("text", "")))       # cited act titles are exempt too
    if lexical_overlap(q, [x["text"] for x in frags], f"{meta} {titles}") > max_overlap:
        return "not_paraphrased", info
    return None, info


def check_item(item: dict, seed: Seed, facts_by_label: dict, frags: list[dict], cfg: dict) -> str | None:
    return check_item_ex(item, seed, facts_by_label, frags, cfg)[0]


def redundant_fragment(seed: Seed, facts: dict, text_of: dict) -> bool:
    """A multi-hop seed with a fragment every fact of which is stated by another C_ref fragment (a committee
    conclusion restating the law, two amendments of one wording): no genuine chain; skipped before any call."""
    for c in seed.chunks:
        others = [t for o, t in text_of.items() if o != c]
        if facts.get(c) and all(any(value_in(f.value, t) for t in others) for f in facts[c]):
            return True
    return False


def gen_batches(seeds: list[Seed], size: int, pilot_ids) -> list[list[Seed]]:
    pilot = sorted((s for s in seeds if s.seed_id in set(pilot_ids)), key=lambda s: s.seed_id)
    rest = sorted((s for s in seeds if s.seed_id not in set(pilot_ids)), key=lambda s: s.seed_id)
    return [pilot[i:i + size] for i in range(0, len(pilot), size)] + [rest[i:i + size] for i in range(0, len(rest), size)]


def _frags(seed, chunks_by_id, index) -> list[dict]:
    return [{"id": f"F{i + 1}", "act": index.act[chunks_by_id[c].parent.split("/", 1)[0]],
             "where": index.format(chunks_by_id[c].parent), "text": chunks_by_id[c].text, "cid": c}
            for i, c in enumerate(seed.chunks)]


def cell_of(seed: Seed) -> str:
    """The M3-R3 п. 1 cell of a seed: split × pattern × language (the rejection log is kept per cell, п. 12)."""
    return f"{seed.split}|{seed.pattern}|{seed.lang}"


def generate(seeds, facts: dict, chunks_by_id, index, llm, cfg: dict, pilot_ids=(), amended=None) -> tuple[list[Candidate], dict]:
    """amended (M3-R22): chunk_id -> names of the acts the fragment amends / cites (graph), for the wrong_act and
    foreign_names checks; the GLOSSARY (cfg legal_glossary) goes into every call's input."""
    out, why = [], Counter()
    by_cell: dict[str, Counter] = {}

    def reject(s: Seed, reason: str, n: int = 1, prefixed: bool = True) -> None:
        why[f"{s.pattern}|{reason}" if prefixed else reason] += n
        by_cell.setdefault(cell_of(s), Counter())[reason] += n

    digest = prompt_digest(PROMPT_GEN, SCHEMA_GEN)
    facts_digest = prompt_digest(facts_prompt(int(cfg.get("max_facts_per_chunk", DEFAULT_MAX_FACTS))), SCHEMA_FACTS)
    usable = []
    for s in seeds:
        if not all(facts.get(c) for c in s.chunks):  # a C_ref fragment without a fact cannot enter the chain
            reject(s, "no_facts")
        elif len(s.chunks) > 1 and redundant_fragment(s, facts, {c: chunks_by_id[c].text for c in s.chunks}):
            reject(s, "redundant")                   # M3-R20: skipped before any call
        else:
            usable.append(s)
    for batch in gen_batches(usable, int(cfg["gen_batch"]), pilot_ids):
        items, meta = [], {}
        for j, s in enumerate(batch):
            frags = _frags(s, chunks_by_id, index)
            labels = {f"F{i + 1}.{k}": f for i, c in enumerate(s.chunks) for k, f in enumerate(facts[c])}
            walk = [{"from": index.format(a) if not a.startswith(("EXT/", "ENT/", "CON/")) else a, "edge": t,
                     "to": index.format(b) if not b.startswith(("EXT/", "ENT/", "CON/")) else b} for a, t, b in s.walk]
            it = {"key": f"k{j + 1}", "lang": s.lang, "LANG": LANG_NAME[s.lang],
                  "pattern": "single" if len(s.chunks) == 1 else "multi", "walk": walk,
                  "fragments": [{"id": x["id"], "act": x["act"], "where": x["where"], "text": x["text"][:TEXT_CAP],
                                 "facts": [{"label": lab, "type": f.vtype or value_type(f.value),
                                            "statement": f.statement, "value": f.value}
                                           for lab, f in labels.items() if lab.startswith(x["id"] + ".")]}
                                for x in frags]}
            if s.walk:
                it["walk_class"] = walk_class(s.walk)
            items.append(it)
            meta[f"k{j + 1}"] = (s, labels, frags)
        ans = key = None
        for f_ in (m[2] for m in meta.values()):
            for x in f_:
                x.setdefault("amended", amended(x["cid"]) if amended else [])
        base = {"items": items, **({"GLOSSARY": cfg["legal_glossary"]} if cfg.get("legal_glossary") else {})}
        for inp in (base, {**base, "retry": RETRY_NOTE}):   # one retry with a distinct, cached input
            guard_total(llm, PASS, inp, PROMPT_GEN, SCHEMA_GEN)
            ans = llm.cached_json(PASS, inp, SCHEMA_GEN,
                                  lambda a: (sorted(x["key"] for x in a["items"]) == sorted(meta), "keys"),
                                  prompt=PROMPT_GEN)
            key = llm.cache_key(PASS, _canon_input(inp), digest)
            if ans is not None:
                break
        if ans is None:
            for s in batch:
                reject(s, "bad_answer", prefixed=False)
            continue
        for it in ans["items"]:
            s, labels, frags = meta[it["key"]]
            r, info = check_item_ex(it, s, labels, frags, cfg)
            if r:
                reject(s, r)
                continue
            q = it["question"].strip()
            ex = exempt_words(frags)
            used = list(dict.fromkeys(info["answer"] + info["path"]))       # M3-R20: the facts actually stated
            lab_of = {id(f): lab for lab, f in labels.items()}
            fp = llm.passes.get("ds_facts") or {}
            out.append(Candidate(s, q, it["answer"].strip(), used, {
                "seed_id": s.seed_id, "cell": cell_of(s), "chunks": list(s.chunks), "walk": [list(h) for h in s.walk],
                "fact_ids": [f.fact_id for f in used], "fact_labels": [lab_of[id(f)] for f in used],
                "listed_labels": list(dict.fromkeys(it["used"])),
                "answer_fact_ids": [f.fact_id for f in info["answer"]], "path_fact_ids": [f.fact_id for f in info["path"]],
                "value_checks": dict(info["status"]),
                "walk_class": walk_class(s.walk) if s.walk else None,
                "facts": {"pass": "ds_facts", "model": llm.model, "prompt_version": fp.get("prompt_version"),
                          "prompt_digest": facts_digest,
                          "cache_keys": {f.block: f.cache_key for f in sorted(used, key=lambda f: f.block)
                                         if f.block}},
                "generate": {"pass": PASS, "model": llm.model, "prompt_version": llm.passes[PASS].get("prompt_version"),
                             "prompt_digest": digest, "cache_key": key, "item": it["key"]},
                "copy": {"max_run": max(max_copy_run(q, x["text"], ex) for x in frags),
                         "overlap": round(lexical_overlap(q, [x["text"] for x in frags],
                                                          " ".join(f"{x['act']} {x['where']}" for x in frags)), 4)}}))
    by = Counter(s.pattern for s in seeds)
    ok = Counter(c.seed.pattern for c in out)
    return out, {"seeds": len(seeds), "accepted": len(out), "rejected": dict(sorted(why.items())),
                 "rejected_by_cell": {c: dict(sorted(v.items())) for c, v in sorted(by_cell.items())},
                 "accept_rate": {p: round(ok[p] / by[p], 4) for p in sorted(by)}}
