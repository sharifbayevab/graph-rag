"""M7 part 2, Task 5: paraphrases of the Q1 (count) / Q2 (version) template questions (docs/m7/
2026-10-02-hard-questions-design.md §2 «перефраз», §6).

The Opus paraphrasers (dispatched by the controller, no API) read packets of ~40 items of one language: the template
question, its answer, and its anchors -- what the paraphrase must keep, verbatim or in a normalised form:
  * the act: its title (compared by templates.title_key, a contiguous ctoks run) and, when the template names it, its
    graph number (the digits and suffix, e.g. 415-I; the ЗРУ / ЎРҚ / OʻRQ prefix is free);
  * the numbers (year, article, part ...) and the law date, compared in a normalised form: `scan` reads dates as
    ctoks canonical (Y, M, D) triples or «day + month word + year» runs (any of the three languages), every other
    number as an int;
  * the kind of change (op) when the question names one (meta.op, or «delete» for the *_deleted_year templates),
    by language-independent folded stems (OP_STEMS).

`para_ingest` refuses a paraphrase when an anchor is lost or changed, the answer number appears (a Q1 answer, or a
number of a Q2 answer's dates / elements not in the template), another new number appears, leak_ok (R14b, the STRUCT
title line exempt) fails, the language changed (`detect_lang`, act titles in «» ignored), it is empty or longer than
60 words, or it is the template unchanged. A rejected item gets one more paraphrase round (round 1)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from graphrag.dataset.facts import ctoks
from grl.hard.checks import leak_ok
from grl.hard.templates import _OPS, title_key

MAX_WORDS = 60
MONTHS = ("yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentyabr", "oktyabr", "noyabr",
          "dekabr")
# folded (ctoks: Cyrillic -> Latin, apostrophes dropped) stems naming a kind of change, ru + uz
OP_STEMS = {"delete": ("isklyuch", "udalen", "udali", "udalya", "udalyo", "chiqarib"),
            "add": ("dopoln", "dobavl", "qoshimcha", "qoshil", "qoshish"),
            "replace": ("zamen", "almashtir"),
            "edit": ("redakts", "redakci", "tahrir"),
            "renumber": ("perenumer",)}           # + RENUMBER_UZ: «raqam…» + its own change verb (has_op)
NOT_OP = ("dopolnitel", "isklyuchitel")      # «дополнительно», «исключительно»: not a kind of change
RENUMBER_UZ = "raqam"
# R51: a count question must not ask «how many times» (folded consecutive tokens; count_* templates, rounds >= 2)
# pattern -> the first paraphrase round it is checked in (earlier rounds keep their frozen judge plan; their rows are
# forced into the next retry round instead). «nechta / qancha ... bor» (= how many are there) is NOT a pattern.
TIMES = {("skolko", "raz"): 2, ("necha", "marta"): 2, ("nechta", "marta"): 2, ("nechchi", "marta"): 2,
         ("necha", "bor"): 3}                                          # «неча бор» = how many times (round 3)
DAY_SUFFIX = {"go", "e", "ye", "chi", "inchi", "nchi"}          # «17-го июня», «17-е», «17-chi»
DELETE_TEMPLATES = ("count_deleted_year", "which_deleted_year", "count_articles_deleted_by_law")
NO_ACT_TEMPLATES = ("count_codes_by_law",)     # Task 6c: the law is named by its date only; no act title in «»
NUM_PREFIX = {"zru", "orq"}
_NUMBER_RE = (re.compile(r"№\s*(\S+)\s+«"), re.compile(r"(\S+)-(?:сонли|sonli)\s+«"))
_QUOTE_RE = re.compile(r"«[^«»]*(?:«[^«»]*»[^«»]*)*»")
UZ_CHARS = set("ўқғҳЎҚҒҲ")
RU_CHARS = set("ыщЫЩ")
UZ_WORDS = {"нечта", "қайси", "нима", "нималар", "йил", "йилда", "ва", "билан", "бор", "қонун", "модда", "моддаси",
            "ҳужжат", "ўзгартиш", "қачон", "неча", "қандай", "киритилган", "киритган", "тўғрисида", "ҳақида",
            "бўйича", "мавжуд", "ичида", "давомида", "қанча", "қилинган"}
RU_WORDS = {"сколько", "какой", "какие", "какая", "каких", "что", "был", "была", "было", "были", "году", "года", "в",
            "из", "и", "акта", "акте", "акт", "статей", "статьи", "статью", "статья", "закон", "закона", "законом",
            "законе", "когда", "изменения", "изменений", "внесено", "внесены", "вносил", "изменил", "ли", "на", "по",
            "от", "для", "как", "какими", "какому", "раз", "года?", "есть", "всего", "содержит", "кодекс"}


def _u(seed, s: str) -> float:
    return int(hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()[:15], 16) / 16 ** 15


def _norm_num(s: str) -> str:
    return str(int(s)) if s.isdigit() else s


def _month(s: str) -> int | None:
    for k, stem in enumerate(MONTHS):
        if s.startswith(stem):
            return k + 1
    return None


def scan(text: str) -> tuple[set[str], Counter]:
    """(ISO dates, Counter of the other numbers) of `text` in canonical tokens: a date is a ctoks canonical (Y, M, D)
    triple or a month word with a day (1..31) next to it and a 4-digit year within 3 tokens; its tokens are consumed,
    the remaining numeric tokens count as numbers (int-normalised)."""
    toks = [(t.s, t.num) for t in ctoks(text)]
    used, dates = set(), set()
    isint = lambda i: 0 <= i < len(toks) and toks[i][1] and toks[i][0].isdigit() and i not in used
    for i in range(len(toks) - 2):
        if isint(i) and isint(i + 1) and isint(i + 2) and len(toks[i][0]) == 4 \
                and 1 <= int(toks[i + 1][0]) <= 12 and 1 <= int(toks[i + 2][0]) <= 31:
            dates.add(f"{toks[i][0]}-{int(toks[i + 1][0]):02d}-{int(toks[i + 2][0]):02d}")
            used |= {i, i + 1, i + 2}
    for i, (s, num) in enumerate(toks):
        m = None if num else _month(s)
        if not m:
            continue
        near = (i - 1, i + 1) + ((i - 2,) if i >= 2 and toks[i - 1][0] in DAY_SUFFIX else ())
        day = next((j for j in near if isint(j) and len(toks[j][0]) <= 2 and 1 <= int(toks[j][0]) <= 31),
                   None)
        if day is None:
            continue
        year = next((j for j in sorted(range(i - 5, i + 6), key=lambda j: abs(j - i))
                     if j != day and isint(j) and len(toks[j][0]) == 4), None)
        if year is None:
            continue
        dates.add(f"{toks[year][0]}-{m:02d}-{int(toks[day][0]):02d}")
        used |= {day, year}
    nums = Counter(_norm_num(s) for i, (s, num) in enumerate(toks) if num and i not in used)
    return dates, nums


def _strip_quotes(text: str) -> str:
    return _QUOTE_RE.sub(" ", text)


def detect_lang(text: str) -> str:
    """ru | uz-Cyrl | uz-Latn of a question, act titles in «» ignored (they stay untranslated)."""
    t = _strip_quotes(text)
    lat = sum(1 for c in t if "a" <= c.lower() <= "z")
    cyr = sum(1 for c in t if "Ѐ" <= c <= "ӿ")
    if lat > cyr:
        return "uz-Latn"
    words = re.findall(r"[\wЀ-ӿ]+", t.lower())
    uz = sum(c in UZ_CHARS for c in t) + sum(w in UZ_WORDS for w in words) + sum(
        bool(re.search(r"(нинг|ни|га|да|дан|лар|ган)$", w)) and len(w) > 4 for w in words) * 0.5
    ru = sum(c in RU_CHARS for c in t) + sum(w in RU_WORDS for w in words)
    return "uz-Cyrl" if uz > ru else "ru"


def _title(question: str) -> str:
    m = _QUOTE_RE.search(question)
    if not m:
        raise ValueError(f"no «title» in {question!r}")
    return m.group(0)[1:-1]


def _act_number(question: str) -> str | None:
    for rx in _NUMBER_RE:
        m = rx.search(question)
        if m:
            return m.group(1)
    return None


def _number_core(number: str) -> list[str]:
    return [t.s for t in ctoks(number) if t.s not in NUM_PREFIX]


def item_op(item: dict) -> str | None:
    return item["meta"].get("op") or ("delete" if item["template"] in DELETE_TEMPLATES else None)


def anchors(item: dict) -> dict:
    """What the paraphrase of `item` must keep (from its template question and meta)."""
    q = item["question"]
    title = "" if item["template"] in NO_ACT_TEMPLATES else _title(q)
    number = _act_number(q)
    dates, nums = scan(_strip_quotes(q))
    core = _number_core(number) if number else []
    return {"title": title, "title_key": list(title_key(title)), "number": number, "number_core": core,
            "dates": sorted(dates), "numbers": sorted(set(nums) - set(core) if core else set(nums)),
            "op": item_op(item), "lang": item["lang"]}


def _contains(seq: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n == 0 or any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def answer_numbers(item: dict) -> set[str]:
    """Numbers a paraphrase must not show: a Q1 answer; the dates' parts and element numbers of a Q2 answer."""
    a = item["answer"]
    if isinstance(a, str):
        return {_norm_num(a)}
    out = set()
    for _law, date, _op, el in a:
        out |= {_norm_num(x) for x in str(date).split("-") if x.isdigit()}
        parts = el.split("/")
        rest = "/".join(parts[2:] if parts[0] == "EXT" else parts[1:])      # the element path, not the doc id
        out |= {_norm_num(x) for x in re.findall(r"\d+", rest)}
    return out


def anchor_problems(q: str, item: dict, anch: dict) -> list[str]:
    """Anchor / number reasons of paraphrase `q` (empty: every anchor kept, no answer or new number)."""
    out = []
    toks = [t.s for t in ctoks(q)]
    bare = [t.s for t in ctoks(_strip_quotes(q))]                      # op words outside the act title
    t_bare = [t.s for t in ctoks(_strip_quotes(item["question"]))]
    if not _contains(toks, anch["title_key"]) or (anch["number_core"] and not _contains(toks, anch["number_core"])):
        out.append("anchor_act")
    dates, nums = scan(_strip_quotes(q))
    if set(anch["dates"]) - dates:
        out.append("anchor_date")
    if set(anch["numbers"]) - set(nums):
        out.append("anchor_number")
    t_dates, t_nums = scan(_strip_quotes(item["question"]))
    allowed = set(t_nums) | set(anch["number_core"])
    new = set(nums) - allowed
    if "anchor_date" not in out:            # a changed anchor date is reported as anchor_date only, not as new numbers
        new |= {p for d in dates - t_dates for p in (_norm_num(x) for x in d.split("-"))} - allowed
    if new & answer_numbers(item) - allowed:
        out.append("answer_in_question")
    elif new:
        out.append("new_number")
    has = has_op
    if anch["op"] and not has(bare, anch["op"]):
        out.append("anchor_op")
    if any(op != anch["op"] and has(bare, op) and not has(t_bare, op) for op in OP_STEMS):
        out.append("op_changed")          # another kind of change named (a broadened / different scope)
    return out


def has_op(toks: list[str], op: str) -> bool:
    """Folded tokens name the kind of change `op`: a stem at a word start (not «дополнительно» / «исключительно»);
    uz renumbering = a «raqam…» word (not «raqamli») whose own change verb follows it directly («рақами
    ўзгартирилган») or «qayta» right before it («qayta raqamlangan»); «ЗРУ рақами билан ўзгартирилган» is not one."""
    if any(t.startswith(st) and not t.startswith(NOT_OP) for t in toks for st in OP_STEMS[op]):
        return True
    if op == "renumber":
        for i, t in enumerate(toks):
            if t.startswith(RENUMBER_UZ) and not t.startswith("raqamli") and (
                    (i + 1 < len(toks) and toks[i + 1].startswith("ozgar"))
                    or (i > 0 and toks[i - 1].startswith("qayta"))):
                return True
    return False


def asks_times(q: str, rnd: int | None = None) -> bool:
    """R51: «сколько раз / necha marta / неча марта / неча бор» -- the number of amending acts, not of changed
    elements. `rnd`: only the patterns checked from that round (None: every pattern)."""
    toks = [t.s for t in ctoks(_strip_quotes(q))]
    pats = {p for p, k in TIMES.items() if rnd is None or k <= rnd}
    return any(tuple(toks[i:i + 2]) in pats for i in range(len(toks) - 1))


def check_paraphrase(q, item: dict, anch: dict, gold: dict | None, rnd: int = 0) -> list[str]:
    """Every reason `q` is refused as the paraphrase of `item` (empty: accepted). A count question asking «how many
    times» is refused by the TIMES patterns active in its round (R51; earlier rounds keep their frozen judge plan --
    their «times» rows are forced into a later retry round)."""
    if not isinstance(q, str) or not q.strip():
        return ["empty"]
    out = []
    if len(q.split()) > MAX_WORDS:
        out.append("length")
    if [t.s for t in ctoks(q)] == [t.s for t in ctoks(item["question"])]:
        out.append("unchanged")
    out += anchor_problems(q, item, anch)
    if detect_lang(q) != item["lang"]:
        out.append("lang")
    if item["qtype"] == "count" and asks_times(q, rnd):
        out.append("count_times")
    if gold is None:
        out.append("no_gold_texts")
    elif not leak_ok(q, dict(gold["texts"]), ids=item["gold"], exempt=gold.get("exempt") or {}):
        out.append("leak")
    return out


def para_ingest(raw_lines, items, gold: dict | None = None, retry_qids=None) -> tuple[list[dict], list[dict]]:
    """raw lines ({qid, question}, optionally `round` 0 / 1 set by the loader from the file name) -> (accepted,
    rejected). `gold`: qid -> {texts: {fragment id: text}, exempt: {fragment id: title prefix}} (R14b);
    `retry_qids`: round -> the qids of that round's retry packets (a set = round 1); a retry-round line of another
    qid is refused. The first line of a
    (qid, round) counts; later ones are duplicates."""
    by_qid = {it["qid"]: it for it in items}
    seen, acc, rej = set(), [], []
    for raw in raw_lines:
        if not isinstance(raw, dict) or not isinstance(raw.get("qid"), str):
            rej.append({"qid": None, "round": None, "reasons": ["malformed"], "raw": raw})
            continue
        qid, rnd = raw["qid"], int(raw.get("round", 0))
        it = by_qid.get(qid)
        if it is None:
            rej.append({"qid": qid, "round": rnd, "reasons": ["unknown_qid"], "raw": raw})
            continue
        if rnd >= 1 and retry_qids is not None and qid not in (
                retry_qids.get(rnd, set()) if isinstance(retry_qids, dict) else retry_qids):
            rej.append({"qid": qid, "round": rnd, "reasons": ["not_retry"], "raw": raw})
            continue
        if (qid, rnd) in seen:
            rej.append({"qid": qid, "round": rnd, "reasons": ["duplicate"], "raw": raw})
            continue
        seen.add((qid, rnd))
        q = raw.get("question")
        why = check_paraphrase(q, it, anchors(it), (gold or {}).get(qid) if gold is not None else None, rnd)
        if why:
            rej.append({"qid": qid, "round": rnd, "reasons": why, "raw": raw})
        else:
            acc.append({"qid": qid, "round": rnd, "question": " ".join(q.split()),
                        "template_question": it["question"], "lang": it["lang"], "qtype": it["qtype"],
                        "template": it["template"]})
    return acc, rej


# --- packets -------------------------------------------------------------------------------------------------------

def _display_anchors(item: dict, anch: dict) -> dict:
    m, lang = item["meta"], item["lang"]
    out = {"act": f"«{anch['title']}»" if anch["title"] else None, "act_number": anch["number"],
           "year": m.get("year") if m.get("year") and m["year"] in anch["numbers"] else None,
           "law_date": m.get("date"), "article": m.get("article"), "part": m.get("part"),
           "change_kind": _OPS[lang].get(anch["op"]) if anch["op"] else None,
           "numbers_to_keep": anch["numbers"]}
    return {k: v for k, v in out.items() if v not in (None, [])}


def packet_item(item: dict) -> dict:
    return {"qid": item["qid"], "qtype": item["qtype"], "template": item["template"], "question": item["question"],
            "answer": item["answer"], "anchors": _display_anchors(item, anchors(item))}


def para_packets(items, n_per_packet: int = 40, seed: int = 20261003, prefix: str = "para",
                 previous: dict | None = None) -> list[dict]:
    """Packets of one language each, ~n_per_packet items (round(n / n_per_packet) packets per language, sizes
    differ by <= 1), items in sha256(seed:qid) order so templates mix. `previous` (round 1): qid -> {question,
    problem} shown to the paraphraser."""
    by_lang = defaultdict(list)
    for it in items:
        by_lang[it["lang"]].append(it)
    out, k = [], 0
    for lang in sorted(by_lang):
        xs = sorted(by_lang[lang], key=lambda it: (_u(seed, "para|" + it["qid"]), it["qid"]))
        n = max(1, round(len(xs) / n_per_packet))
        for j in range(n):
            part = xs[j * len(xs) // n:(j + 1) * len(xs) // n]
            k += 1
            rows = []
            for it in part:
                r = packet_item(it)
                if previous and it["qid"] in previous:
                    r["previous"] = previous[it["qid"]]
                rows.append(r)
            out.append({"packet_id": f"{prefix}-{k:03d}", "lang": lang, "n": len(part), "items": rows})
    return out


INSTRUCTIONS = """# PARAPHRASER INSTRUCTIONS: M7 Q1 / Q2 question paraphrases (plan M7 part 2, Task 5)

You rewrite template questions of a legal retrieval benchmark (Uzbek legislation) into natural questions. A code
check and an independent judge verify every line you write.

## 0. Files

- Packets: `{packets}/{prefix}-NNN.json` (listed in `{packets}/{index}`). Do only the packets the controller
  assigns you.
- Output: the file the controller names, `{out}`. One JSON object per line, one line per item, in packet order:
  `{{"qid": "...", "question": "..."}}`. Write nothing else to that file. Append each line as soon as it is final.
- Use only the packet. Do not read other repository files, do not run code that calls an LLM API.

## 1. The packet

`lang` is the packet's language (`ru`, `uz-Cyrl` = Uzbek in Cyrillic, `uz-Latn` = Uzbek in Latin script). Each item
has `qid`, the template `question`, its `answer` (for your information only), and `anchors`:

- `act` -- the act title in «» (sometimes Uzbek inside a Russian question: keep it untranslated, as given);
- `act_number` -- the act's number when the template gives one (e.g. `415-I`, `ЗРУ-642`): keep it;
- `year`, `law_date` (ISO; write it as a normal date of the language, e.g. «15 апреля 2025 года» / «2025 йил 15
  апрелдаги» / «2025-yil 15-apreldagi»), `article`, `part`, and `numbers_to_keep` -- every number must stay;
- `change_kind` -- the kind of change (addition, deletion, new wording, word replacement, renumbering): keep it, in a
  normal word of the language (e.g. «дополнение» / «дополнены», «қўшимча», «chiqarib tashlangan»).
{retry}
## 2. Rules

1. Write ONE natural question in the packet's language, the way a lawyer or a citizen would ask it.
2. Keep every anchor exactly or in a normalised form: the act title verbatim inside «», the act number, the year, the
   law date, the article / part numbers, the kind of change. Do not add any other number, date, act or detail, and do
   not drop one. The question must ask exactly the same thing as the template (same scope, same kind of answer: a
   number for a count, a list for «what / which»). Write the act title character for character as given (no
   transliteration, no translation), and write numbers, years and dates with digits.
3. Never reveal the answer: never write the answer number or any number / date / law from `answer` that the template
   does not already contain.
4. Do not copy 8 or more consecutive words from legal text (the fragments behind the answer are checked by code; the
   act title is exempt). Do not quote from the law.
5. Vary the wording across items: do not start most questions the same way, and avoid template-like openings
   (e.g. not every question «Сколько ...» / «Nechta ...»). Change word order and phrasing, keep the meaning.
6. At most 60 words; one question per item; no answer, no explanation, no extra fields.
7. A count question asks for the number of changed elements (статьи / части / пункты, moddalar / qismlar / bandlar),
   never «how many times» («сколько раз», «necha marta», «неча марта»): that would count amending acts instead.
"""

RETRY_NOTE = """
Retry round (packets `{prefix}-NNN`): each item also has `previous` -- your earlier paraphrase and why it was refused
or must be redone (a code check, the judge, or a review rule). Write a NEW paraphrase that fixes that problem; this is
the item's last chance.
"""


Q1X_NOTES = """
## 3. Task 6c items (Q1 expansion)

These packets hold only count questions of five new kinds. Keep exactly WHAT is counted -- the answer is a number of
that object, so another object word changes the question:
- `count_parts_article`: parts (части / қисмлар / qismlar) of article `article` of the act;
- `count_points_article`: points (пункты / бандлар / bandlar) of article `article` of the act;
- `count_codes_by_law`: codes (кодексы / кодекслар / kodekslar) changed by the law of `law_date` -- there is no act
  title here: keep the law's date, do not name any code or law title;
- `count_laws_amended_code_year`: laws / documents (законы / қонунлар / qonunlar) that changed the code in `year`;
- `count_articles_deleted_by_law`: articles (статьи / моддалар / moddalar) of the act deleted (исключены / чиқариб
  ташланган / chiqarib tashlangan) by the law of `law_date`.
Rule 7 applies to all of them: never «сколько раз» / «necha marta» / «неча марта» / «неча бор».
"""


def instructions(packets_dir: str, out: str, prefix: str = "para", index: str = "index.json",
                 retry: bool = False, notes: bool = False) -> str:
    return INSTRUCTIONS.format(packets=packets_dir, prefix=prefix, index=index, out=out,
                               retry=RETRY_NOTE.format(prefix=prefix) if retry else "") + (Q1X_NOTES if notes else "")
