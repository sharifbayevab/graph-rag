"""ds_facts (decision M3-R10): gpt-4.1 extracts quote-grounded atomic facts from the fragments of one block (8
consecutive fragments of one document; the cache unit, shared by the pilot and the full run). Every fact is checked by
code; a bad fact is dropped with its reason, the rest of the answer is kept. Facts are stored per fragment and reused by
every walk.

Value matching (M3-R16): every value comparison runs on canonical tokens (`ctoks`): the steps of the internal
normalized form (`normalize.to_internal`: ISO dates, canonical numbering, the deterministic Cyrl -> Latn
transliteration `cyr_to_lat`, lower case) applied per token -- so a value folds the same alone and inside a sentence,
whatever script routing the sentence gets -- apostrophes dropped (oʻ / o' / oʼ agree), number words -> digits, and the
M1 lemmas of each token (Uzbek stemmer; for a Cyrillic word also the Russian lemma). Matching is by whole tokens: "5 кун" never matches inside "15 кун"; a word matches its
inflected form only when the M1 lemmas are equal."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from graphrag.data.lemma import get_stemmer, ru_lemma
from graphrag.data.normalize import canon_numbering, cyr_to_lat, iso_dates
from graphrag.eval.gold import wsn

PASS = "ds_facts"
MIN_QUOTE = 20
# prompt_version 2 (M3-R16, Task 15 review I4 / m5): the fact cap comes from config (max_facts_per_chunk)
PROMPT_FACTS_TEMPLATE = """You extract ATOMIC FACTS from fragments of Uzbek / Russian parliamentary and legal documents.
For each fragment (id, act, where, text) list at most <MAX_FACTS> facts that the TEXT states explicitly. A fact is one
self-contained statement with one VALUE: an organ, an amount, a term, a date, a condition, a duty or a right.
- statement: the fact in one short sentence, in the fragment's language; self-contained: replace "ушбу Қонун",
  "мазкур модда", "настоящий Закон" with the act or article named in `where`;
- value: the shortest span that answers the fact on its own, copied exactly from the text. A number always keeps its
  unit or noun ("30 кун", "5 фоиз", "2023 йил 1 январь", "10 процентов"), never a bare number; never a single generic
  word ("банк", "қарор", "орган"), a pronoun, or the act's / session's name;
- quote: a verbatim span of the text (at least 20 characters) that contains the value and states the fact.
Do not infer; skip procedural boilerplate (enactment formulas, signatures, gazette citations). In transcripts, facts
are decisions, proposals, votes and figures together with who stated them; skip greetings, chair formulas and agenda
boilerplate.
Return {"chunks": [{"id", "facts": [...]}]} with exactly one entry per fragment id (facts may be empty)."""
SCHEMA_FACTS = {"type": "object", "additionalProperties": False, "required": ["chunks"], "properties": {"chunks": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "facts"],
                               "properties": {"id": {"type": "string"}, "facts": {"type": "array", "items": {
                                   "type": "object", "additionalProperties": False,
                                   "required": ["statement", "value", "quote"],
                                   "properties": {"statement": {"type": "string"}, "value": {"type": "string"},
                                                  "quote": {"type": "string"}}}}}}}}}


def facts_prompt(max_facts: int) -> str:
    """The ds_facts system prompt for a fact cap (config max_facts_per_chunk); its digest enters the cache key."""
    return PROMPT_FACTS_TEMPLATE.replace("<MAX_FACTS>", str(int(max_facts)))


DEFAULT_MAX_FACTS = 6                     # config dataset.max_facts_per_chunk (a test keeps them equal)
PROMPT_FACTS = facts_prompt(DEFAULT_MAX_FACTS)


@dataclass(frozen=True)
class Fact:
    fact_id: str          # "<chunk_id>@<i>"
    chunk_id: str
    statement: str
    value: str
    quote: str
    block: str = ""       # ds_facts provenance (M3-R3 п. 11): the block and the cache key of the accepted answer
    cache_key: str = ""
    vtype: str = ""       # M3-R20: number (dates included) / name / term (<= 8 tokens) / clause -- value_type()


# ---- canonical tokens (M3-R16) -----------------------------------------------------------------------------------------
_APOS = re.compile(r"[ʻʼ'`’‘ʹ]")
_WORD = re.compile(r"[^\W_]+(?:[ʻʼ'`’‘ʹ][^\W_]+)*")
# a number literal is ONE token (pre-pilot re-review I1): thousands grouped by (thin / no-break) spaces with 3-digit
# groups ("1 500", "1 000 000") and decimals with "," or "." ("0,5", "1.5", "12,5%") -- so "5" never matches inside
# "0,5" and "500" never inside "1 500". Canonical form: digits, "." as the decimal mark, no leading zeros.
_NUMLIT = re.compile(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u2009\u202f]\d{3})+(?:[.,]\d+)?(?![\d])|(?<![\d.,])\d+[.,]\d+(?![\d])")
_TOKEN = re.compile(f"(?P<n>{_NUMLIT.pattern})|(?P<w>{_WORD.pattern})")


def _num_canon(x: str) -> str:
    x = re.sub(r"[ \u00a0\u2009\u202f]", "", x).replace(",", ".")
    whole, _, frac = x.partition(".")
    return str(int(whole)) + (f".{frac}" if frac else "")


def _lit(x: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?", x) is not None
_CYR = re.compile(r"[Ѐ-ӿ]")
_UZ_NUM = {"bir": 1, "ikki": 2, "uch": 3, "tort": 4, "besh": 5, "olti": 6, "yetti": 7, "etti": 7, "etmish": 70, "sakkiz": 8, "toqqiz": 9,
           "on": 10, "yigirma": 20, "ottiz": 30, "qirq": 40, "ellik": 50, "oltmish": 60, "yetmish": 70, "sakson": 80,
           "toqson": 90, "yuz": 100, "ming": 1000}
_RU_NUM = {"один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
           "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
           "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30,
           "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
           "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600,
           "семьсот": 700, "восемьсот": 800, "девятьсот": 900, "тысяча": 1000}
_RU_NOT_UZ = {"он"}             # Russian words that fold onto an Uzbek number word ("он" -> "on" = 10)
_WEAK = {1, 100}                # "bir" / "один" (an article) and "yuz" (also "face") are numbers only next to a number


@dataclass(frozen=True)
class Tok:
    s: str                      # folded surface (Latin, no apostrophes) or the digits of a number
    l: frozenset                # M1 lemma candidates, folded the same way ({s} for numbers)
    num: bool


def _fold(w: str) -> str:
    return _APOS.sub("", cyr_to_lat(w) if _CYR.search(w) else w)


def _compose(vals: list[int]) -> int:
    total = cur = 0
    for v in vals:
        if v in (100, 1000):
            cur = (cur or 1) * v
            if v == 1000:
                total, cur = total + cur, 0
        else:
            cur += v
    return total + cur


@lru_cache(maxsize=200_000)
def ctoks(text: str) -> tuple[Tok, ...]:
    """Canonical tokens of `text` (see the module docstring)."""
    st = get_stemmer()
    t = canon_numbering(iso_dates(unicodedata.normalize("NFC", text or ""))).lower()
    raw = []                                               # (surface, lemmas, number value | None)
    for m in _TOKEN.finditer(t):
        if m.group("n"):
            n = _num_canon(m.group("n"))
            raw.append((n, frozenset([n]), n))
            continue
        w = _APOS.sub("", m.group("w"))
        if not w:
            continue
        if w.isdigit():                                     # superscripts ("126²", a lone "¹") are digits too
            d = "".join(ch for ch in w if ch.isdecimal())
            if not d:
                continue
            raw.append((str(int(d)), frozenset([str(int(d))]), int(d)))
            continue
        surf = _fold(w)
        if _CYR.search(w):                                  # Uzbek or Russian: both lemmas are candidates
            lem = ru_lemma(w)
            n = _RU_NUM.get(lem)
            if n is None and w not in _RU_NOT_UZ:
                n = _UZ_NUM.get(surf)
            raw.append((surf, frozenset([st.stem(surf), _fold(lem)]), n))
        else:
            raw.append((surf, frozenset([st.stem(surf)]), _UZ_NUM.get(surf)))
    out: list[Tok] = []
    i = 0
    while i < len(raw):
        if raw[i][2] is not None and not _lit(raw[i][0]):
            j = i
            while j < len(raw) and raw[j][2] is not None and not _lit(raw[j][0]):
                j += 1
            vals = [x[2] for x in raw[i:j]]
            if len(vals) == 1 and vals[0] in _WEAK:          # a lone "bir" / "yuz" / "один" stays a word
                out.append(Tok(raw[i][0], raw[i][1], False))
            else:
                n = str(_compose(vals))
                out.append(Tok(n, frozenset([n]), True))
            i = j
            continue
        s, l, n = raw[i]
        out.append(Tok(s, l, n is not None))
        i += 1
    return tuple(out)


def _eq(a: Tok, b: Tok, lemma: bool) -> bool:
    if a.num or b.num:
        return a.num and b.num and a.s == b.s
    return a.s == b.s or (lemma and bool(a.l & b.l))


def find_seq(hay: tuple[Tok, ...], needle: tuple[Tok, ...], lemma: bool = True) -> list[int]:
    """Start positions of `needle` as a contiguous whole-token run of `hay`."""
    n = len(needle)
    if not n:
        return []
    return [i for i in range(len(hay) - n + 1) if all(_eq(hay[i + k], needle[k], lemma) for k in range(n))]


def value_in(value: str, text: str, lemma: bool = True) -> bool:
    """`value` occurs in `text` as whole canonical tokens (script-, apostrophe- and number-word-insensitive)."""
    return bool(find_seq(ctoks(text), ctoks(value), lemma))


def numbers(text: str) -> set[str]:
    return {t.s for t in ctoks(text) if t.num}


# generic words of act / session names: a value made only of these and of the act's own words names the act (M3-R9)
_GENERIC_NAME_WORDS = ("ги", "даги", "gi", "dagi", "Қонун", "Қонуни", "Қонунга", "кодекс", "кодекси", "қарор", "қарори",
                       "фармон", "фармони", "фармойиш", "низом", "низоми", "Закон", "Кодекс", "Постановление", "Указ",
                       "Положение", "Ўзбекистон", "Республикаси", "Республикасининг", "Республики", "Узбекистан",
                       "Олий", "Мажлис", "Мажлиси", "Сенат", "Сенати", "Қонунчилик", "палатаси", "сессия", "мажлис",
                       "йиғилиш", "заседание", "пленарное", "O'zbekiston", "Respublikasi", "Respublikasining", "qonun",
                       "qonuni", "kodeksi", "qarori", "farmoni")


@lru_cache(maxsize=1)
def _generic() -> frozenset:
    return frozenset(t.s for w in _GENERIC_NAME_WORDS for t in ctoks(w)) | \
        frozenset(x for w in _GENERIC_NAME_WORDS for t in ctoks(w) for x in t.l)


def _names(act_words) -> tuple[frozenset, frozenset]:
    toks = [t for w in act_words for t in ctoks(w)]
    return frozenset(t.s for t in toks) | _generic(), frozenset(x for t in toks for x in t.l) | _generic()


def check_fact(f: dict, text: str, act_words: set[str]) -> str | None:
    q = wsn(f["quote"])                                     # NFC + whitespace / case / quote normalization
    if len(q) < MIN_QUOTE:
        return "short_quote"
    if q not in wsn(text):
        return "quote_not_verbatim"
    vt = ctoks(f["value"])
    if not wsn(f["value"]) or not vt or not find_seq(ctoks(f["quote"]), vt, lemma=False):
        return "value_not_in_quote"                         # whole tokens: "5 кун" is not in "15 кун"
    words = [t for t in vt if not t.num]                    # a number is content, even if it equals the article no.
    ns, nl = _names(act_words)
    own = {t.s for w in act_words for t in ctoks(w)} - _generic()      # the act's own (non-generic) words
    if words and not any(t.num for t in vt) and all(t.s in ns or t.l & nl for t in words) and \
            any(t.s in own for t in words):
        return "name_only"                                  # the value IS the act's name (a body name is an answer)
    return None


# ---- M3-R20: typed values and snap-to-source --------------------------------------------------------------------------
MAX_TYPED_TOKENS = 8


def value_type(value: str) -> str:
    """number: the value carries a number or a date (code-checks its numbers); name / term: <= 8 canonical tokens
    (code-checked as whole tokens); clause: longer wording (checked by the independent judge, M3-R20)."""
    toks = ctoks(value)
    if any(t.num for t in toks) or re.search(r"\d", value or ""):
        return "number"
    if len(toks) <= MAX_TYPED_TOKENS:
        return "name" if (value or "").strip("«»“”\"' ")[:1].isupper() else "term"
    return "clause"


def _spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]


def snap(target: str, text: str, lemma: bool) -> str | None:
    """The exact substring of `text` (NFC) whose canonical tokens equal those of `target` (whole tokens; with `lemma`
    an inflected form of the same M1 lemma counts) -- so a model's re-inflected value or a quote retyped over the
    corpus's mixed-script words is replaced by the SOURCE span itself (still verbatim, M3-R20 / M3-R3 п. 4). None if
    no such span exists."""
    t = unicodedata.normalize("NFC", text or "")
    want = ctoks(target)
    if not want:
        return None
    sp = _spans(t)
    n = len(_spans(unicodedata.normalize("NFC", target)))
    for i in range(len(sp)):
        head = ctoks(t[sp[i][0]:sp[i][1]])
        if not head or not _eq(head[0], want[0], lemma):
            continue
        for j in range(i, min(len(sp), i + n + 3)):
            sub = t[sp[i][0]:sp[j][1]]
            got = ctoks(sub)
            if len(got) == len(want) and all(_eq(a, b, lemma) for a, b in zip(got, want)):
                return sub
            if len(got) > len(want):
                break
    return None


def snap_fact(f: dict, text: str, act_words: set[str]) -> tuple[dict | None, str | None]:
    """M3-R20 R1: check a raw fact, first snapping its quote (token-equal over the corpus's mixed-script / punctuation
    noise) and its value (same lemmas inside the quote) to the exact source spans. Returns (fact with verbatim quote
    and value, None) or (None, reason). A quote that joins spans ("…"), paraphrases or invents text still fails, as
    does a value that is not in its quote (absent, reworded, other number)."""
    g = dict(f)
    if len(wsn(g["quote"])) < MIN_QUOTE:
        return None, "short_quote"
    if wsn(g["quote"]) not in wsn(text):
        q2 = snap(g["quote"], text, lemma=False)
        if q2 is None or wsn(q2) not in wsn(text):
            return None, "quote_not_verbatim"
        g["quote"] = q2
    vt = ctoks(g["value"])
    if wsn(g["value"]) and vt and not find_seq(ctoks(g["quote"]), vt, lemma=False):
        v2 = snap(g["value"], g["quote"], lemma=True)
        if v2 is not None:
            g["value"] = v2
    r = check_fact(g, text, act_words)
    return (None, r) if r else (g, None)


def guard_total(llm, pass_name: str, inp, prompt: str, schema: dict) -> None:
    """M3-R3 п. 14 (Task 15 review m2): before a call that will hit the API (cache miss, API on), the dataset passes
    count `max_calls_total` over the whole ledger of their cache (cache/dataset/ledger.json: every run of every pass)
    plus this run's calls not yet recorded there. KgLLM itself caps the total per run only (graph passes, R31); the
    per-pass caps stay cumulative in KgLLM."""
    from graphrag.kg.llm import KgBudgetExceeded, _canon_input, prompt_digest
    if not llm.allow_api:
        return
    key = llm.cache_key(pass_name, _canon_input(inp), prompt_digest(prompt, schema))
    if llm.cache_file(pass_name, key).exists():
        return
    led = sum(int(v) for p in llm.ledger().values() for v in p.values())
    used = led + llm.total_api_calls() - sum(llm._ledgered.values())
    if used + 1 > llm.max_total:
        raise KgBudgetExceeded(f"{pass_name}: cumulative dataset budget of {llm.max_total} API calls would be exceeded "
                               f"({used} used, ledger {llm.cache_root / 'ledger.json'})")


def _ids_ok(ans: dict, labels: dict) -> bool:
    """Exactly one entry per fragment label: a foreign, missing or repeated id rejects the whole answer."""
    ids = [x["id"] for x in ans["chunks"]]
    return sorted(ids) == sorted(labels)


RETRY_NOTE = "The previous answer did not return exactly one entry per fragment id; return exactly the ids given."


def extract_block(llm, block_id: str, chunk_ids: list[str], chunks_by_id: dict, index, max_facts: int):
    """Facts of one block. An answer with foreign / missing / repeated ids is re-validated (and rejected) on every read,
    never used; the block is asked once more with a distinct input (`retry`), which is cached too, so an offline
    replay reproduces both the rejection and the retry (Task 15 review m7)."""
    from graphrag.dataset.generate import exempt_words
    from graphrag.kg.llm import _canon_input, prompt_digest
    labels = {f"C{i + 1}": cid for i, cid in enumerate(chunk_ids)}
    frags = [{"id": k, "act": index.act[chunks_by_id[c].parent.split("/", 1)[0]],
              "where": index.format(chunks_by_id[c].parent), "text": chunks_by_id[c].text} for k, c in labels.items()]
    prompt = facts_prompt(max_facts)
    digest = prompt_digest(prompt, SCHEMA_FACTS)
    why = Counter()
    ans = key = None
    for attempt, inp in enumerate(({"block": block_id, "fragments": frags},
                                   {"block": block_id, "fragments": frags, "retry": RETRY_NOTE})):
        guard_total(llm, PASS, inp, prompt, SCHEMA_FACTS)
        ans = llm.cached_json(PASS, inp, SCHEMA_FACTS, lambda a: (_ids_ok(a, labels), "ids"), prompt=prompt)
        key = llm.cache_key(PASS, _canon_input(inp), digest)
        if ans is not None:
            break
        if attempt == 0:
            why["retried_bad_answer"] += 1
    out: list[Fact] = []
    if ans is None:
        why["bad_answer"] += 1
        return out, why
    for x in sorted(ans["chunks"], key=lambda x: list(labels).index(x["id"])):   # C1..C8, not "C10" < "C2"
        cid = labels[x["id"]]
        frag = next(f for f in frags if f["id"] == x["id"])
        aw = exempt_words([frag])
        kept = 0
        for f in x["facts"]:
            g, r = snap_fact(f, chunks_by_id[cid].text, aw)
            r = r or ("too_many" if kept >= max_facts else None)
            if r:
                why[r] += 1
                continue
            if g["quote"] != f["quote"] or g["value"] != f["value"]:
                why["snapped"] += 1
            out.append(Fact(f"{cid}@{kept}", cid, g["statement"].strip(), g["value"].strip(), g["quote"],
                            block_id, key, value_type(g["value"])))
            kept += 1
    return out, why


def extract(blocks_needed, block_chunks: dict[str, list[str]], chunks_by_id, index, llm, max_facts: int):
    facts: dict[str, list[Fact]] = {}
    why = Counter()
    for b in sorted(blocks_needed):
        fs, w = extract_block(llm, b, block_chunks[b], chunks_by_id, index, max_facts)
        why.update(w)
        for cid in block_chunks[b]:
            facts[cid] = [f for f in fs if f.chunk_id == cid]
    n = sum(len(v) for v in facts.values())
    return facts, {"blocks": len(blocks_needed), "chunks": len(facts), "facts": n,
                   "chunks_without_facts": sum(not v for v in facts.values()), "rejected": dict(sorted(why.items()))}


def write_facts(path, facts: dict[str, list[Fact]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"chunk_id": c, "facts": [asdict(f) for f in facts[c]]}, ensure_ascii=False,
                                    sort_keys=True) + "\n" for c in sorted(facts)), encoding="utf-8")


def read_facts(path) -> dict[str, list[Fact]]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["chunk_id"]] = [Fact(**f) for f in r["facts"]]
    return out
