"""M7 Q1 (count) / Q2 (version) question templates with gold computed from the pinned graph (docs/m7/
2026-10-02-hard-questions-design.md §2; plan M7 part 1, Task 5; rulings R5a, R6, R16, R17).

Q1 (`count_items`): the answer is a number read from `doc_structure`; gold = [STRUCT/<root>] (|C*| = 1).
  count_articles / count_chapters   «сколько статей / глав в X» -- internal documents with >= 3 articles;
  count_articles_by_law             «сколько статей Y изменено законом Z» -- distinct articles of Y in Z's edges;
  count_amends_year                 «сколько изменений элементов Y внесено в году G» -- AMENDS edges (element-level
                                    edits) of Y dated in G (R17: not «раз»);
  count_deleted_year                «сколько исключений внесено в Y в году G» -- AMENDS edges of Y with op delete in G
                                    (R17: delete edits, word / digit level included -- not «нормы»).
  The «частей в статье N» template is dropped (R5a: structure fragments exist only for document / EXT roots).
  R6: an item is kept only if the fact its answer needs is printed in the STRUCT text (the header counts, or every
  per-law group line of Z with its article list -- the 384-token cap may cut those lines).

Q2 (`version_items`): the answer is the list [law, date, op, element] of the selected AMENDS edges; gold = the sorted
union of μ⁻¹ of their source elements (fragments of the amending laws; EXT elements own none).
  which_laws_article   «какие законы изменили статью N Y»
  what_law_changed     «что закон Z изменил в Y»
  which_deleted_year   «что было исключено из Y в G» (R17)
  which_law_part       «какой закон изменил часть M статьи N Y (и когда)»
  |gold| ∈ [2, 5] (which_law_part: [1, 5], |gold| = 1 on at most 20 % of the Q2 items). A set with |gold| > 5 is
  narrowed by year, then by op within the year; a subset that is still > 5 is refused, and so is an empty ("")
  year / op bucket. Every selected edge must have a
  source with >= 1 fragment; with `chunk_texts`, every edge's evidence quote must be in its source's fragments (the
  Q2 reading of R6: the fact is visible in the gold).
  Part 2, R33 (`chunk_spans` given): an edge's gold is only the μ⁻¹ fragments of its source overlapping its
  attrs.evidence span (evidence_gold); an edge with no such fragment drops the candidate (r33_no_quote); the quote must
  then be visible in those fragments. Part 2 also: per-type per-root caps (R43, `full_items`), the probe qids
  excluded in every language (`exclude`), a candidate filter before selection (`pre_drop`, R44), the |gold| = 1 share
  as a selection fixpoint (no post-trim) and one Q2 item per (root, answer list) (R48, `dup_answer`).

R16 (both types): a root is named only if its ctoks-folded title is unique among all document / EXT roots; a shared
title is disambiguated by the root's graph number (attrs.numbers, exactly one, unique within the title group) and the
root is dropped otherwise (internal documents carry no date / number in the graph, so they are dropped). Translated
twins -- internal roots with >= 3 articles, the same doc_type and the same (chapters, articles, parts, points) -- keep
only the lexicographically first root. The source of an edit is worded by its type: «закон / қонун / qonun» for
QONUNLAR, «документ / ҳужжат / hujjat» otherwise; which_laws_article / which_law_part use the singular for one source.

Both: roots with an empty title are skipped; the filters «документ >= 3 статей, акт >= 3 поправок»; at most
`max_per_root` items per root; a law Z is named by its date, which must be unique among Y's amending laws (graph
titles of amending laws are long and mostly identical boilerplate). Languages follow `lang_shares` (largest remainder
over n, shuffled by the seed). Titles are the graph's `title`, untranslated. Every rendering must pass `leak_ok`
against the gold texts and must not contain the answer number; a (candidate, language) that fails is not used.
Deterministic: random.Random(rng_seed), every walk over sorted keys."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from graphrag.dataset.facts import ctoks
from grl.env.structure import _article, _Resolver, build_struct_texts, doc_structure, struct_id
from grl.hard.checks import leak_ok

LANGS = ("ru", "uz-Cyrl", "uz-Latn")
# language counts of the old set: data/dataset/fable/stats.json ingest_gen.accepted_by_cell summed over split / kind
# (the brief's source); scripts/hard_probe.py recomputes them from the file.
OLD_SET_LANG_COUNTS = {"uz-Cyrl": 538, "ru": 363, "uz-Latn": 313}
MIN_ARTICLES, MIN_AMENDS, MAX_GOLD = 3, 3, 5
SINGLE_SHARE = 0.2

_MONTHS = {"ru": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
                  "ноября", "декабря"],
           "uz-Cyrl": ["январ", "феврал", "март", "апрел", "май", "июн", "июл", "август", "сентябр", "октябр", "ноябр",
                       "декабр"],
           "uz-Latn": ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentyabr", "oktyabr",
                       "noyabr", "dekabr"]}
_OPS = {"ru": {"replace": "замена слов", "edit": "новая редакция", "add": "дополнение", "delete": "исключение",
               "renumber": "перенумерация"},
        "uz-Cyrl": {"replace": "сўзларни алмаштириш", "edit": "янги таҳрир", "add": "қўшимча",
                    "delete": "чиқариб ташлаш", "renumber": "рақамини ўзгартириш"},
        "uz-Latn": {"replace": "soʻzlarni almashtirish", "edit": "yangi tahrir", "add": "qoʻshimcha",
                    "delete": "chiqarib tashlash", "renumber": "raqamini oʻzgartirish"}}
_SRC = {"ru": {"law": ("закон", "законом"), "doc": ("документ", "документом")},
        "uz-Cyrl": {"law": ("қонун", "қонун"), "doc": ("ҳужжат", "ҳужжат")},
        "uz-Latn": {"law": ("qonun", "qonun"), "doc": ("hujjat", "hujjat")}}
_WHO = {"ru": {("law", False): ("Какой закон", "вносил", "изменил"), ("law", True): ("Какие законы", "вносили", "изменили"),
               ("doc", False): ("Какой документ", "вносил", "изменил"),
               ("doc", True): ("Какие документы", "вносили", "изменили")},
        "uz-Cyrl": {("law", False): ("қонун",) * 3, ("law", True): ("қонунлар",) * 3,
                    ("doc", False): ("ҳужжат",) * 3, ("doc", True): ("ҳужжатлар",) * 3},
        "uz-Latn": {("law", False): ("qonun",) * 3, ("law", True): ("qonunlar",) * 3,
                    ("doc", False): ("hujjat",) * 3, ("doc", True): ("hujjatlar",) * 3}}
_DOC_LOC = {"ru": {"law": "законе", "doc": "документе"}, "uz-Cyrl": {"law": "қонуни", "doc": "ҳужжати"},
            "uz-Latn": {"law": "qonuni", "doc": "hujjati"}}
# R16b: a root whose title is shared, but unique among law_text roots with articles, is named only by templates that
# carry the type word («в законе «X»» / «X» қонунида / «X» qonunida)
TYPE_NAMED = "@law_text"
TYPE_TEMPLATES = ("count_articles", "count_chapters", "count_parts_article", "count_points_article")   # + Task 6c
_DOC_GEN = {"ru": {"law": "закона", "doc": "документа"}, "uz-Cyrl": _DOC_LOC["uz-Cyrl"], "uz-Latn": _DOC_LOC["uz-Latn"]}
_SRC_PL = {"ru": {"law": "законов", "doc": "документов"}, "uz-Cyrl": {"law": "қонун", "doc": "ҳужжат"},
           "uz-Latn": {"law": "qonun", "doc": "hujjat"}}
TEMPLATES = {
    "count_articles": {"ru": "Сколько статей в {doc_loc} {Yq}?",
                       "uz-Cyrl": "{Yq} {doc_loc}да нечта модда бор?",
                       "uz-Latn": "{Yq} {doc_loc}da nechta modda bor?"},
    "count_chapters": {"ru": "Сколько глав в {doc_loc} {Yq}?",
                       "uz-Cyrl": "{Yq} {doc_loc}да нечта боб бор?",
                       "uz-Latn": "{Yq} {doc_loc}da nechta bob bor?"},
    "count_articles_by_law": {"ru": "Сколько статей акта {Yq} было изменено {src_by} {date}?",
                              "uz-Cyrl": "{date} {src} билан {Yq}нинг нечта моддасига ўзгартиш киритилган?",
                              "uz-Latn": "{date} {src} bilan {Yq}ning nechta moddasiga oʻzgartish kiritilgan?"},
    "count_amends_year": {"ru": "Сколько изменений отдельных элементов (статей, частей, пунктов) акта {Yq} было внесено "
                                "в {G} году?",
                          "uz-Cyrl": "{G} йилда {Yq}нинг алоҳида элементларига (моддалар, қисмлар, бандлар) нечта "
                                     "ўзгартиш киритилган?",
                          "uz-Latn": "{G}-yilda {Yq}ning alohida elementlariga (moddalar, qismlar, bandlar) nechta "
                                     "oʻzgartish kiritilgan?"},
    "count_deleted_year": {"ru": "Сколько исключений было внесено в акт {Yq} в {G} году?",
                           "uz-Cyrl": "{G} йилда {Yq}га чиқариб ташлаш тарзида нечта ўзгартиш киритилган?",
                           "uz-Latn": "{G}-yilda {Yq}ga chiqarib tashlash tarzida nechta oʻzgartish kiritilgan?"},
    # Task 6c (Q1 expansion, grl.hard.q1x): article structure and amendment counts
    "count_parts_article": {"ru": "Сколько частей в статье {N} {doc_gen} {Yq}?",
                            "uz-Cyrl": "{Yq} {doc_loc}нинг {N}-моддасида нечта қисм бор?",
                            "uz-Latn": "{Yq} {doc_loc}ning {N}-moddasida nechta qism bor?"},
    "count_points_article": {"ru": "Сколько пунктов в статье {N} {doc_gen} {Yq}?",
                             "uz-Cyrl": "{Yq} {doc_loc}нинг {N}-моддасида нечта банд бор?",
                             "uz-Latn": "{Yq} {doc_loc}ning {N}-moddasida nechta band bor?"},
    "count_codes_by_law": {"ru": "Сколько кодексов изменил {src} {date}?",
                           "uz-Cyrl": "{date} {src} билан нечта кодексга ўзгартиш киритилган?",
                           "uz-Latn": "{date} {src} bilan nechta kodeksga oʻzgartish kiritilgan?"},
    "count_laws_amended_code_year": {"ru": "Сколько {src_pl} внесли изменения в {Yq} в {G} году?",
                                     "uz-Cyrl": "{G} йилда {Yq}га нечта {src_pl} билан ўзгартиш киритилган?",
                                     "uz-Latn": "{G}-yilda {Yq}ga nechta {src_pl} bilan oʻzgartish kiritilgan?"},
    "count_articles_deleted_by_law": {"ru": "Сколько статей акта {Yq} было исключено {src_by} {date}?",
                                      "uz-Cyrl": "{date} {src} билан {Yq}нинг нечта моддаси чиқариб ташланган?",
                                      "uz-Latn": "{date} {src} bilan {Yq}ning nechta moddasi chiqarib tashlangan?"},
    "which_laws_article": {"ru": "{who} {who_v} изменения в статью {N} акта {Yq}{when}{op}?",
                           "uz-Cyrl": "{when}{Yq}нинг {N}-моддасига қайси {who} ўзгартиш киритган{op}?",
                           "uz-Latn": "{when}{Yq}ning {N}-moddasiga qaysi {who} oʻzgartish kiritgan{op}?"},
    "what_law_changed": {"ru": "Что изменил в акте {Yq} {src} {date}{op}?",
                         "uz-Cyrl": "{date} {src} {Yq}га қандай ўзгартишлар киритган{op}?",
                         "uz-Latn": "{date} {src} {Yq}ga qanday oʻzgartishlar kiritgan{op}?"},
    "which_deleted_year": {"ru": "Что было исключено из акта {Yq} в {G} году?",
                           "uz-Cyrl": "{G} йилда {Yq}дан нималар чиқариб ташланган?",
                           "uz-Latn": "{G}-yilda {Yq}dan nimalar chiqarib tashlangan?"},
    "which_law_part": {"ru": "{who} и когда {who_p} часть {M} статьи {N} акта {Yq}{when}{op}?",
                       "uz-Cyrl": "{when}{Yq}нинг {N}-моддаси {M}-қисмига қайси {who} ва қачон ўзгартиш киритган{op}?",
                       "uz-Latn": "{when}{Yq}ning {N}-moddasi {M}-qismiga qaysi {who} va qachon oʻzgartish "
                                  "kiritgan{op}?"},
}
COUNT_TEMPLATES = ("count_articles", "count_chapters", "count_articles_by_law", "count_amends_year",
                   "count_deleted_year")
VERSION_TEMPLATES = ("which_laws_article", "what_law_changed", "which_deleted_year", "which_law_part")


@dataclass
class HardItem:
    qid: str
    qtype: str              # "count" | "version" | "compare"
    template: str
    lang: str               # "uz-Cyrl" | "uz-Latn" | "ru"
    question: str
    answer: Any             # Q1: the number as a string; Q2: [[law, date, op, element], ...]
    gold: list[str]         # fragment ids (STRUCT/<root> or corpus chunk ids), sorted
    docs: list[str]         # documents the question names: [root] (+ the law Z)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class _Cand:
    template: str
    root: str
    title: str
    number: str | None      # R16: the graph number naming a root whose title is shared, else None
    answer: Any
    gold: list[str]
    docs: list[str]
    meta: dict
    params: dict            # template slots other than Y (language-specific values are rendered later)
    gold_texts: list[str] = field(default_factory=list)
    exempt: dict = field(default_factory=dict)          # R14b: fragment id -> its exempt title prefix (STRUCT line 1)
    questions: dict = field(default_factory=dict)       # lang -> question that passed the checks

    @property
    def key(self):
        return (self.template, self.root, json.dumps(self.meta, sort_keys=True, ensure_ascii=False))

    @property
    def single(self) -> bool:
        return len(self.gold) == 1


# --- rendering -----------------------------------------------------------------------------------------------------

def _date(iso: str, lang: str) -> str:
    y, m, d = (int(x) for x in iso.split("-"))
    mon = _MONTHS[lang][m - 1]
    if lang == "ru":
        return f"от {d} {mon} {y} года"
    if lang == "uz-Cyrl":
        return f"{y} йил {d} {mon}даги"
    return f"{y}-yil {d}-{mon}dagi"


def _when(year: str | None, lang: str) -> str:
    if not year:
        return ""
    return {"ru": f" в {year} году", "uz-Cyrl": f"{year} йилда ", "uz-Latn": f"{year}-yilda "}[lang]


def _op(op: str | None, lang: str) -> str:
    if not op:
        return ""
    name = _OPS[lang].get(op, op)
    return {"ru": f" (вид изменения: {name})", "uz-Cyrl": f" (ўзгартиш тури: {name})",
            "uz-Latn": f" (oʻzgartish turi: {name})"}[lang]


def _num(number: str, lang: str) -> str:
    """A graph act number ('oʼrq-642', '560-ii') as written in the language: ЗРУ-642 / ЎРҚ-642 / OʻRQ-642, 560-II."""
    m = re.fullmatch(r"o[ʼ'ʻ]?rq-(.+)", number, re.IGNORECASE)
    if m:
        return {"ru": "ЗРУ-", "uz-Cyrl": "ЎРҚ-", "uz-Latn": "OʻRQ-"}[lang] + m.group(1).upper()
    return number.upper()


def _quoted(title: str, number: str | None, lang: str) -> str:
    if not number or number == TYPE_NAMED:
        return f"«{title}»"
    n = _num(number, lang)
    return {"ru": f"№ {n} «{title}»", "uz-Cyrl": f"{n}-сонли «{title}»", "uz-Latn": f"{n}-sonli «{title}»"}[lang]


def src_kind(doc_id: str) -> str:
    return "law" if doc_id.startswith("QONUNLAR") else "doc"


def render(template: str, lang: str, title: str, params: dict, number: str | None = None) -> str:
    src = _SRC[lang][params.get("src", "law")]
    who = _WHO[lang][(params.get("who", "law"), bool(params.get("plural", True)))]
    slots = {"Yq": _quoted(title, number, lang), "N": params.get("article", ""), "M": params.get("part", ""),
             "G": params.get("year", ""), "doc_loc": _DOC_LOC[lang][params.get("doctype", "doc")],
             "doc_gen": _DOC_GEN[lang][params.get("doctype", "doc")], "src_pl": _SRC_PL[lang][params.get("src", "law")],
             "src": src[0], "src_by": src[1], "who": who[0], "who_v": who[1],
             "who_p": who[2], "date": _date(params["date"], lang) if params.get("date") else "",
             "when": _when(params.get("when"), lang), "op": _op(params.get("op"), lang)}
    return " ".join(TEMPLATES[template][lang].format(**slots).split())


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text)) | {t.s for t in ctoks(text) if t.num}


def _qid(qtype: str, c: _Cand, lang: str) -> str:
    h = hashlib.sha1("|".join([c.template, c.root, json.dumps(c.meta, sort_keys=True, ensure_ascii=False),
                               lang]).encode("utf-8")).hexdigest()[:12]
    return f"m7-{qtype}-{h}"


def lang_quotas(n: int, shares: dict) -> dict[str, int]:
    """Largest-remainder split of n over the languages in `shares` (counts or fractions); ties by language name."""
    shares = {k: float(v) for k, v in shares.items() if float(v) > 0}
    tot = sum(shares.values())
    raw = {k: n * v / tot for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def _check_langs(c: _Cand, stats: Counter):
    """Render every language; keep those that pass leak_ok against the gold texts and do not state the answer."""
    nums = {str(c.answer)} if isinstance(c.answer, str) else set()
    for lang in LANGS:
        q = render(c.template, lang, c.title, c.params, c.number)
        stats["renderings"] += 1
        if nums & _numbers(q):
            stats["answer_in_question"] += 1
            continue
        stats["leak_checked"] += 1
        stats[f"leak_checked:{c.template}"] += 1
        if not leak_ok(q, dict(zip(c.gold, c.gold_texts)) if c.gold_texts else [], ids=c.gold, exempt=c.exempt):
            stats["leak_rejected"] += 1
            stats[f"leak_rejected:{c.template}"] += 1
            continue
        c.questions[lang] = q
    if not c.questions:
        stats["candidates_no_language"] += 1


def _select(cands: list[_Cand], n: int, rng: random.Random, shares: dict, max_per_root: int, unique,
            max_single: int | None, qtype: str, root_counts: Counter | None = None,
            stats: Counter | None = None, dedup=None) -> list[HardItem]:
    """Round-robin over templates (one item per slot), within a template the first eligible candidate of the
    shuffled queue with the fewest items on its root so far; each slot's language comes from the shuffled quotas.
    `root_counts` (part 2): a per-root counter the caller passes in, updated in place (full_items gives each type its
    own, R43). stats["cap_blocked"]: askable candidates left unselected whose root is at the cap at the end (a lower
    bound of what the cap cost). `dedup` (R48): candidate -> key; a candidate whose key an earlier selected item has
    is skipped (the first in selection order is kept, the slot is refilled); stats["dup_answer"] counts them."""
    quotas = lang_quotas(n, shares)
    langs = [lang for lang in sorted(quotas) for _ in range(quotas[lang])]
    rng.shuffle(langs)
    queues = defaultdict(list)
    for c in sorted(cands, key=lambda c: c.key):
        if c.questions:
            queues[c.template].append(c)
    templates = sorted(queues)
    for t in templates:
        rng.shuffle(queues[t])
    per_root = root_counts if root_counts is not None else Counter()
    used, used_dup, singles, out, ti = set(), set(), 0, [], 0
    for lang in langs:
        chosen = None
        for k in range(len(templates)):
            t = templates[(ti + k) % len(templates)]
            ok = [c for c in queues[t] if lang in c.questions and unique(c) not in used
                  and (dedup is None or dedup(c) not in used_dup) and per_root[c.root] < max_per_root
                  and not (c.single and max_single is not None and singles >= max_single)]
            if ok:
                chosen = min(ok, key=lambda c: per_root[c.root])      # stable: first of the least-used roots
                ti = (ti + k + 1) % len(templates)
                break
        if chosen is None:
            continue
        used.add(unique(chosen))
        if dedup is not None:
            used_dup.add(dedup(chosen))
        per_root[chosen.root] += 1
        singles += chosen.single
        out.append(HardItem(_qid(qtype, chosen, lang), qtype, chosen.template, lang, chosen.questions[lang],
                            chosen.answer, list(chosen.gold), list(chosen.docs), dict(chosen.meta)))
    if stats is not None:
        left = [c for t in templates for c in queues[t] if unique(c) not in used]
        dup = {unique(c) for c in left if dedup is not None and dedup(c) in used_dup}
        stats["dup_answer"] += len(dup)
        stats["cap_blocked"] += len({unique(c) for c in left
                                     if unique(c) not in dup and per_root[c.root] >= max_per_root})
    return out


def _pre_drop(cands: list[_Cand], pre_drop, stats: Counter) -> list[_Cand]:
    """R44: a caller's candidate filter run before selection -- pre_drop(cand) -> a drop reason (counted in stats,
    total and per template) or None to keep."""
    if pre_drop is None:
        return cands
    out = []
    for c in cands:
        why = pre_drop(c)
        if why:
            stats[why] += 1
            stats[f"{why}:{c.template}"] += 1
        else:
            out.append(c)
    return out


def _drop_excluded(cands: list[_Cand], qtype: str, exclude, stats: Counter) -> list[_Cand]:
    """Part 2: a candidate whose qid in any language is in `exclude` (the probe ids) is dropped in every language --
    a translation of a probe question is the probe question. qids hash template|root|meta|lang (no seed), so the
    match is exact."""
    if not exclude:
        return cands
    out = []
    for c in cands:
        if any(_qid(qtype, c, lang) in exclude for lang in LANGS):
            stats["probe_excluded"] += 1
            stats[f"probe_excluded:{c.template}"] += 1
        else:
            out.append(c)
    return out


def _ws_count(s: str) -> int:
    return len(s.split())


# --- R16: naming roots -------------------------------------------------------------------------------------------

def title_key(title: str) -> tuple[str, ...]:
    return tuple(t.s for t in ctoks(title))


def _law_with_articles(info: dict | None) -> bool:
    return bool(info) and info["kind"] == "document" and info["doc_type"] == "law_text" and info["n_articles"] >= 1


def root_names(nodes, structure: dict) -> tuple[dict[str, str | None], dict[str, str]]:
    """R16 / R16b. (named, dropped): named = root -> None (unique title), the graph number that disambiguates it, or
    TYPE_NAMED (R16b: the title is shared, but no other law_text root with articles has it -- a resolution / committee
    conclusion of the same title is another type, and the question names the type; an EXT act of the same title is a
    law too, so it keeps the title ambiguous -- the review's «Рақобат тўғрисида» case); dropped = root -> "r16_title"
    (still ambiguous) | "r16_twin" (translated twin of a smaller root id)."""
    roots = {n["node_id"]: n for n in nodes if n["kind"] == "document" and (n.get("title") or "").strip()}
    groups = defaultdict(list)
    for r in sorted(roots):
        groups[title_key(roots[r]["title"])].append(r)
    named, dropped = {}, {}
    for key, rs in sorted(groups.items()):
        if len(rs) == 1:
            named[rs[0]] = None
            continue
        nums = {r: ((roots[r].get("attrs") or {}).get("numbers") or []) for r in rs}
        own = Counter(x for r in rs for x in set(nums[r]))
        for r in rs:
            if len(nums[r]) == 1 and own[nums[r][0]] == 1:
                named[r] = nums[r][0]
            elif _law_with_articles(structure.get(r)) and not any(
                    o != r and (_law_with_articles(structure.get(o)) or o.startswith("EXT/")) for o in rs):
                named[r] = TYPE_NAMED
            else:
                dropped[r] = "r16_title"
    sig = defaultdict(list)
    for r in sorted(structure):
        i = structure[r]
        if i["kind"] == "document" and i["n_articles"] >= MIN_ARTICLES:
            sig[(i["doc_type"], i["n_chapters"], i["n_articles"], i["n_parts"], i["n_points"])].append(r)
    for rs in sig.values():
        for r in rs[1:]:
            dropped.setdefault(r, "r16_twin")
            named.pop(r, None)
    return named, dropped


def _r16_ok(r: str, named: dict, dropped: dict, template: str, stats: Counter) -> bool:
    if r in named and (named[r] != TYPE_NAMED or template in TYPE_TEMPLATES):
        return True
    reason = dropped.get(r, "r16_title")
    stats[f"{reason}_dropped"] += 1
    stats[f"{reason}_dropped:{template}"] += 1
    return False


# --- Q1 ------------------------------------------------------------------------------------------------------------

def _law_dates(amended_by) -> dict[str, set[str]]:
    d = defaultdict(set)
    for src, date, _, _ in amended_by:
        d[src].add(date)
    return d


def _unique_date_laws(amended_by) -> dict[str, str]:
    """Law -> its date, for laws with one date that no other law amending the same root shares."""
    dates = _law_dates(amended_by)
    per_date = Counter(next(iter(v)) for v in dates.values() if len(v) == 1)
    return {z: next(iter(v)) for z, v in sorted(dates.items())
            if len(v) == 1 and per_date[next(iter(v))] == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", next(iter(v)))}


def _law_line_articles(text: str, law: str, date: str) -> set[str] | None:
    """The articles printed on Z's group line of the STRUCT text (None if the line is not there)."""
    lines = [ln for ln in text.split("\n") if ln.startswith(law + " ") and f" {date}: " in ln]
    if len(lines) != 1:
        return None
    tail = lines[0].split("; ўзгартирилган моддалар ", 1)
    return set(tail[1].split(", ")) if len(tail) == 2 else set()


def _year_ops_visible(text: str, year: str, total: int, op: str, k: int) -> bool:
    m = re.search(rf"(?:^|; | ){re.escape(year)}: {total} \(([^)]*)\)", text)
    return bool(m) and f"{op} {k}" in [x.strip() for x in m.group(1).split(",")]


def count_candidates(nodes, edges, struct_texts: dict | None = None, stats: Counter | None = None) -> list[_Cand]:
    stats = stats if stats is not None else Counter()
    s = doc_structure(nodes, edges)
    texts = struct_texts if struct_texts is not None else build_struct_texts(nodes, edges, _ws_count)
    named, dropped = root_names(nodes, s)
    out = []

    def add(template, r, info, answer, visible, meta, params):
        stats["candidates"] += 1
        if not _r16_ok(r, named, dropped, template, stats):
            return
        if not visible:
            stats["r6_dropped"] += 1
            stats[f"r6_dropped:{template}"] += 1
            return
        docs = [r] + ([meta["law"]] if meta.get("law") else [])
        params = {**params, "doctype": "law" if info["doc_type"] == "law_text" else "doc"}
        text = texts.get(struct_id(r), "")
        out.append(_Cand(template, r, info["title"], named[r], str(answer), [struct_id(r)], docs, meta, params,
                         [text], {struct_id(r): text.split("\n", 1)[0]}))

    for r in sorted(s):
        info, text = s[r], texts.get(struct_id(r), "")
        if not info["title"].strip():
            continue
        if info["kind"] == "document" and info["n_articles"] >= MIN_ARTICLES:
            n = info["n_articles"]
            add("count_articles", r, info, n, f"моддалар: {n}" in text, {}, {})
            if info["n_chapters"] >= 2:
                n = info["n_chapters"]
                add("count_chapters", r, info, n, f"боблар: {n}" in text, {}, {})
        if len(info["amended_by"]) < MIN_AMENDS:
            continue
        for g, c in info["amend_counts"]["by_year"].items():
            if re.fullmatch(r"\d{4}", g):
                add("count_amends_year", r, info, c, bool(re.search(rf"(?:^|; | ){g}: {c}\b", text)),
                    {"year": g}, {"year": g})
        for g, ops in info["by_year_op"].items():
            k = ops.get("delete", 0)
            if k and re.fullmatch(r"\d{4}", g):
                add("count_deleted_year", r, info, k,
                    _year_ops_visible(text, g, info["amend_counts"]["by_year"][g], "delete", k),
                    {"year": g}, {"year": g})
        for law, date in _unique_date_laws(info["amended_by"]).items():
            arts = {a for src, _, _, dst in info["amended_by"] if src == law for a in [_article(dst)] if a}
            if not arts:
                continue
            seen = _law_line_articles(text, law, date)
            add("count_articles_by_law", r, info, len(arts), seen == arts, {"law": law, "date": date},
                {"date": date, "src": src_kind(law)})
    return out


def count_items(nodes, edges, rng_seed: int, n: int, *, struct_texts: dict | None = None,
                lang_shares: dict | None = None, max_per_root: int = 4, stats: dict | None = None,
                root_counts: Counter | None = None, exclude=None, pre_drop=None) -> list[HardItem]:
    """Q1 items. One item per (template, root) -- so the answer is a function of (template, root) in any language.
    `struct_texts`: the STRUCT texts (corpus tokenizer, data.yaml max_tokens); default: whitespace tokens, cap 384.
    `root_counts`: a per-root counter the caller may share; `exclude`: qids never emitted (the probe), dropped in
    every language; `pre_drop(cand) -> reason | None`: a candidate filter before selection (R44: R32 conflicts)."""
    st = Counter()
    cands = _pre_drop(_drop_excluded(count_candidates(nodes, edges, struct_texts, st), "count", exclude, st),
                      pre_drop, st)
    for c in cands:
        _check_langs(c, st)
    items = _select(cands, n, random.Random(rng_seed), lang_shares or OLD_SET_LANG_COUNTS, max_per_root,
                    lambda c: (c.template, c.root), None, "count", root_counts, st)
    st["selected"] = len(items)
    if stats is not None:
        stats.update(st)
    return items


# --- Q2 ------------------------------------------------------------------------------------------------------------

def _part(node_id: str) -> tuple[str, str] | None:
    """(article, part) when the element is a part (or lies under a part) directly under an article: …/СТ.N/Ч.M[/…]."""
    p = node_id.split("/")
    for i in range(len(p) - 1):
        if p[i].startswith("СТ.") and p[i + 1].startswith("Ч."):
            return p[i][3:], p[i + 1][2:]
    return None


def _amends_by_root(nodes, edges) -> dict[str, list[tuple]]:
    """root -> [(law, date, op, dst, src, evidence quote, evidence span (doc_id, start, end) | None)] of its incoming
    AMENDS edges, sorted."""
    roots = {n["node_id"]: n for n in nodes if n["kind"] == "document"}
    by_id = {n["node_id"]: n for n in nodes}
    root_of = _Resolver(roots, edges)
    out = defaultdict(list)
    for e in edges:
        if e["type"] != "AMENDS":
            continue
        r = root_of(e["dst"])
        if r is None:
            continue
        a = e.get("attrs") or {}
        law = by_id.get(e["src"], {}).get("doc_id") or root_of(e["src"]) or e["src"].split("/")[0]
        ev = a.get("evidence") or {}
        quote = ev.get("quote") or ""
        span = (ev["doc_id"], int(ev["start"]), int(ev["end"])) if {"doc_id", "start", "end"} <= set(ev) else None
        out[r].append((law, a.get("date") or "", a.get("op") or "", e["dst"], e["src"], quote, span))
    return {r: sorted(v, key=lambda x: (x[:6], x[6] or ("", -1, -1))) for r, v in sorted(out.items())}


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def evidence_gold(edge: dict, mu_inv_chunks, chunk_spans: dict) -> list[str]:
    """R33: the fragments of the edge's source element (`mu_inv_chunks`, μ⁻¹ of edge.src) whose (doc_id, [start, end))
    overlaps the edge's attrs.evidence span; [] when none does or the edge has no evidence span. Order of
    `mu_inv_chunks`."""
    ev = (edge.get("attrs") or {}).get("evidence") or {}
    if not {"doc_id", "start", "end"} <= set(ev):
        return []
    d, s, e = ev["doc_id"], int(ev["start"]), int(ev["end"])
    out = []
    for c in mu_inv_chunks:
        sp = chunk_spans.get(c)
        if sp and sp[0] == d and int(sp[1]) < e and s < int(sp[2]):
            out.append(c)
    return out


def version_candidates(nodes, edges, mu_inv: dict, chunk_texts: dict | None = None,
                       stats: Counter | None = None, *, chunk_spans: dict | None = None) -> list[_Cand]:
    """`chunk_spans` given (R33): an edge's gold is evidence_gold (its quoted fragments); a candidate with an edge whose
    span meets none of its source's fragments is dropped (stats r33_no_quote). Without it: part-1 gold (all μ⁻¹)."""
    stats = stats if stats is not None else Counter()
    titles = {n["node_id"]: n.get("title") or "" for n in nodes if n["kind"] == "document"}
    named, dropped = root_names(nodes, doc_structure(nodes, edges))
    out = []

    def rec_gold(rec):
        own = mu_inv.get(rec[4], [])
        if chunk_spans is None:
            return list(own)
        span = rec[6]
        return evidence_gold({"attrs": {"evidence": {"doc_id": span[0], "start": span[1], "end": span[2]}}}, own,
                             chunk_spans) if span else []

    def gold_of(recs):
        return sorted({c for rec in recs for c in rec_gold(rec)})

    def accept(template, r, recs, lo, meta, params):
        stats["candidates"] += 1
        if not _r16_ok(r, named, dropped, template, stats):
            return
        if any(not mu_inv.get(rec[4]) for rec in recs):
            stats["no_chunks"] += 1
            return
        if chunk_spans is not None and any(not rec_gold(rec) for rec in recs):
            stats["r33_no_quote"] += 1
            stats[f"r33_no_quote:{template}"] += 1
            return
        gold = gold_of(recs)
        if not lo <= len(gold) <= MAX_GOLD:
            stats["gold_out_of_range"] += 1
            return
        gt = [chunk_texts.get(c, "") for c in gold] if chunk_texts is not None else []
        if chunk_texts is not None:
            for rec in recs:
                own = _norm_ws(" ".join(chunk_texts.get(c, "") for c in rec_gold(rec)))
                if rec[5] and _norm_ws(rec[5]) not in own:
                    stats["evidence_invisible"] += 1
                    stats[f"evidence_invisible:{template}"] += 1
                    return
        answer = [[law, date, op, dst] for law, date, op, dst, *_ in sorted(recs, key=lambda x: (x[1], x[0], x[3],
                                                                                                    x[2]))]
        docs = [r] + ([meta["law"]] if meta.get("law") else [])
        laws = sorted({rec[0] for rec in recs})
        params = {**params, "who": "law" if all(src_kind(z) == "law" for z in laws) else "doc",
                  "plural": len(laws) > 1}
        out.append(_Cand(template, r, titles[r], named[r], answer, gold, docs, {**meta, "n_edges": len(recs)}, params,
                         gt))

    def narrowed(template, r, recs, lo, meta, params):
        """|gold| > 5 -> by year, then by op within the year; refused if still > 5."""
        if len(gold_of(recs)) <= MAX_GOLD:
            return accept(template, r, recs, lo, meta, params)
        stats["narrowed"] += 1
        by_year = defaultdict(list)
        for rec in recs:
            by_year[rec[1][:4]].append(rec)
        for y in sorted(by_year):
            yr = by_year[y]
            if not y:
                stats["empty_bucket_refused"] += 1
                continue
            m_y, p_y = ({**meta, "year": y}, {**params, "when": y}) if "year" not in meta else (meta, params)
            if len(gold_of(yr)) <= MAX_GOLD:
                accept(template, r, yr, lo, m_y, p_y)
                continue
            by_op = defaultdict(list)
            for rec in yr:
                by_op[rec[2]].append(rec)
            for op in sorted(by_op):
                if not op:
                    stats["empty_bucket_refused"] += 1
                elif len(gold_of(by_op[op])) <= MAX_GOLD:
                    accept(template, r, by_op[op], lo, {**m_y, "op": op}, {**p_y, "op": op})
                else:
                    stats["too_large_refused"] += 1

    for r, recs in _amends_by_root(nodes, edges).items():
        if len(recs) < MIN_AMENDS or not titles.get(r, "").strip():
            continue
        by_art, by_part, by_law, del_year = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for rec in recs:
            a = _article(rec[3])
            if a:
                by_art[a].append(rec)
            p = _part(rec[3])
            if p:
                by_part[p].append(rec)
            by_law[rec[0]].append(rec)
            if rec[2] == "delete" and re.fullmatch(r"\d{4}", rec[1][:4]):
                del_year[rec[1][:4]].append(rec)
        for a in sorted(by_art):
            narrowed("which_laws_article", r, by_art[a], 2, {"article": a}, {"article": a})
        for (a, p) in sorted(by_part):
            narrowed("which_law_part", r, by_part[(a, p)], 1, {"article": a, "part": p}, {"article": a, "part": p})
        for law, date in _unique_date_laws([x[:4] for x in recs]).items():
            narrowed("what_law_changed", r, by_law[law], 2, {"law": law, "date": date},
                     {"date": date, "src": src_kind(law)})
        for y in sorted(del_year):
            if len(gold_of(del_year[y])) > MAX_GOLD:
                stats["too_large_refused"] += 1
                stats["candidates"] += 1
                continue
            accept("which_deleted_year", r, del_year[y], 2, {"year": y}, {"year": y})
    return out


def _share_floor(k: int) -> int:
    """floor(SINGLE_SHARE * k), exact for the decimal share (no float drift at 0.2 * 175)."""
    return int(round(SINGLE_SHARE * k, 9))


def _answer_key(c: _Cand) -> tuple:
    """R48: Q2 items with the same root and answer list are one question, whatever the template."""
    return c.root, json.dumps(c.answer, ensure_ascii=False)


def version_items(nodes, edges, mu_inv: dict[str, list[str]], rng_seed: int, n: int, *,
                  chunk_texts: dict | None = None, lang_shares: dict | None = None, max_per_root: int = 4,
                  stats: dict | None = None, chunk_spans: dict | None = None, root_counts: Counter | None = None,
                  exclude=None, pre_drop=None, dedup_answers: bool = False) -> list[HardItem]:
    """Q2 items. `mu_inv`: element id -> its fragment ids (μ⁻¹); `chunk_texts`: fragment id -> text (leak check and
    evidence visibility; without it neither is checked against texts -- leak_ok still refuses fragment ids).
    `chunk_spans` (R33): fragment id -> (doc_id, start, end); gold = the quoted fragments only (evidence_gold).
    `root_counts` / `exclude` / `pre_drop`: as in count_items. `dedup_answers` (R48, part 2): one item per (root,
    answer list) across templates. The |gold| = 1 share is a fixpoint of the selection: max_single = floor(0.2 * n),
    then floor(0.2 * selected) until the selected set satisfies it -- nothing is removed after selection."""
    st = Counter()
    cands = _pre_drop(_drop_excluded(version_candidates(nodes, edges, mu_inv, chunk_texts, st,
                                                        chunk_spans=chunk_spans), "version", exclude, st),
                      pre_drop, st)
    for c in cands:
        _check_langs(c, st)
    rc0 = Counter(root_counts or {})
    max_single = _share_floor(n)
    for rounds in range(1, n + 2):                  # the |gold| = 1 share as a fixpoint over the selected set
        rc, sel_st = Counter(rc0), Counter()
        items = _select(cands, n, random.Random(rng_seed), lang_shares or OLD_SET_LANG_COUNTS, max_per_root,
                        lambda c: c.key, max_single, "version", rc, sel_st,
                        dedup=_answer_key if dedup_answers else None)
        if sum(len(it.gold) == 1 for it in items) <= _share_floor(len(items)):
            break
        max_single = _share_floor(len(items))       # strictly smaller: the loop ends
    st.update(sel_st)
    st["singles_rounds"] = rounds
    if root_counts is not None:
        root_counts.clear()
        root_counts.update(rc)
    st["selected"] = len(items)
    st["singles"] = sum(len(it.gold) == 1 for it in items)
    if stats is not None:
        stats.update(st)
    return items


# --- part 2: full generation (ruling R43) --------------------------------------------------------------------------

COUNT_CAP, VERSION_CAP = 4, 8      # R43: Q1 <= 4 per root within Q1 (spec §2); Q2 <= 8 per amended act within Q2


def full_items(nodes, edges, mu_inv, rng_seed: int, n_count: int, n_version: int, *, struct_texts=None,
               chunk_texts=None, chunk_spans=None, lang_shares=None, exclude=None, pre_drop=None,
               count_cap: int = COUNT_CAP, version_cap: int = VERSION_CAP):
    """R43: Q1 and Q2 with separate per-root caps (each counted within its own type, so Q1 filling a root never blocks
    Q2 on it); R33 gold when `chunk_spans` is given; probe qids excluded in every language.
    Returns (count, version, count_stats, version_stats, count_roots, version_roots)."""
    sc, sv, rc_c, rc_v = Counter(), Counter(), Counter(), Counter()
    count = count_items(nodes, edges, rng_seed, n_count, struct_texts=struct_texts, lang_shares=lang_shares,
                        max_per_root=count_cap, stats=sc, root_counts=rc_c, exclude=exclude, pre_drop=pre_drop)
    ver = version_items(nodes, edges, mu_inv, rng_seed, n_version, chunk_texts=chunk_texts, lang_shares=lang_shares,
                        max_per_root=version_cap, stats=sv, chunk_spans=chunk_spans, root_counts=rc_v,
                        exclude=exclude, pre_drop=pre_drop, dedup_answers=True)
    return count, ver, sc, sv, rc_c, rc_v



def item_key(it) -> tuple:
    """The candidate key (_Cand.key) of an emitted item -- template, root, meta (HardItem or its json dict)."""
    g = (lambda k: it[k]) if isinstance(it, dict) else (lambda k: getattr(it, k))
    return g("template"), g("docs")[0], json.dumps(g("meta"), sort_keys=True, ensure_ascii=False)
