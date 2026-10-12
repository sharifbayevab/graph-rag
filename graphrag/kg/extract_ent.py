"""Entities (organ / position / procedure / legal_category) and REGULATES ("регулирует") edges
(M2 Task 6).

* Dictionary + rules (deterministic, every record): graph.yaml `dictionaries` -- fixed names
  (Олий Мажлис, Сенат, Вазирлар Маҳкамаси, Президент, Бош прокурор, …; Cyrillic, Latin, Russian) and
  head words whose name is the capitalised run of words before them ("Иқтисодиёт ва молия вазирлиги",
  "Budjet va iqtisodiy islohotlar masalalari qoʼmitasi", "… қўмитаси раиси"). Stenograms and other
  non-norm records only contribute vertices (speaker roles etc.), never REGULATES.
* Verified gpt-4.1 pass (norms only): norm units (a top norm node, split at paragraphs into pieces of
  <= UNIT_CHARS) go in batches of `llm.passes.entities.batch_units`; the strict-JSON answer
  {units: [{unit_id, entities: [{type, name, evidence}]}]} is accepted per entity iff evidence is a
  substring of the unit text AND name is a substring of evidence (case / whitespace / apostrophe
  normalized) AND type is in graph.yaml vertex_types.entity. Cached in cache/kg/entities (KgLLM).
* Vertices: one per (type, norm_key), id ENT/<type>/<norm_key>; a leading "Ўзбекистон
  Республикаси" is dropped from names first.
* REGULATES: src = μ(evidence span) (a norm element), dst = the entity; dictionary evidence = the name
  span, LLM evidence = the evidence span. Concept mentions (graphrag.kg.concepts) add REGULATES ->
  concept. One edge per (src, dst), all evidence in evidence_all.

CLI: python -m graphrag.kg.extract_ent [--llm] [--out data/kg/ent_off]  (API only with GRL_KG_API=1)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from graphrag import config
from graphrag.kg.concepts import (_edge, concept_mentions, extract_concepts, lang_of, merge_concepts, norm_ids,
                                  norm_key, norm_tops, WORD_RX)
from graphrag.kg import ent_rules as R
from graphrag.kg.schema import Evidence, GraphEdge, GraphNode, Schema

__all__ = ["extract_concepts", "extract_entities", "norm_key", "build", "verify_entity"]

PASS = "entities"
ENT_PREFIX = "ENT/"
UNIT_CHARS = 3000
CTX_TITLE = 160
API_WORKERS = 8
_CYR_RU = re.compile(r"[ыщэё]", re.I)
_CYR = re.compile(r"[а-яёўқғҳ]", re.I)
_APOS = str.maketrans({c: "ʼ" for c in "ʻʼ'`‘’"})

# --- keys / ids -------------------------------------------------------------------------------------

_D: dict | None = None


def _dicts() -> dict:
    global _D
    if _D is None:
        cfg = config.load("graph")["dictionaries"]
        rules = cfg["name_rules"]
        fixed = []
        for typ, group in (("organ", "organs"), ("position", "positions")):
            for ent in cfg.get(group) or []:
                for p in ent["patterns"]:
                    fixed.append((typ, ent["title"], re.compile(r"(?i)(?<!\w)(?:" + p + r")\w*")))
        heads = {typ: re.compile(r"(?i)(?<!\w)(?:" + "|".join(rx) + r")") for typ, rx in cfg["heads"].items()}
        head_full = {typ: re.compile(r"(?i)^(?:" + "|".join(rx) + r")$") for typ, rx in cfg["heads"].items()}
        _D = {"fixed": fixed, "heads": heads, "head_full": head_full,
              "stop": {w.lower().translate(_APOS) for w in rules["stop_words"]},
              "inner": {w.lower() for w in rules["inner_words"]},
              "stop_suffix": re.compile("(?i)" + rules["stop_suffix"]),
              "prefix": re.compile("(?i)" + rules["prefix"]),
              "head_block": re.compile("(?i)" + rules["head_block"]),
              "rewrites": [(re.compile(a), b) for a, b in rules.get("key_rewrites") or []],
              "generic": {w.lower().translate(_APOS) for w in rules["generic_owners"]},
              "max_words": int(rules.get("max_words", 10))}
    return _D


def strip_prefix(name: str) -> tuple[int, str]:
    """(chars dropped, name) without a leading "Ўзбекистон Республикаси(нинг)"."""
    m = _dicts()["prefix"].match(name)
    if m and m.end() < len(name):
        return m.end(), name[m.end():]
    return 0, name


_ALIASES: dict | None = None


def canon_key(key: str) -> str:
    global _ALIASES
    for rx, rep in _dicts()["rewrites"]:
        key = rx.sub(rep, key)
    key = " ".join(key.split())
    if _ALIASES is None:
        from graphrag.kg import ent_rules
        _ALIASES = {}
        _ALIASES = {a: canon_key(b) for a, b in ent_rules.alias_map().items()}
    return _ALIASES.get(key, key)


def entity_key(name: str, lang: str | None = None) -> str:
    return canon_key(norm_key(strip_prefix(name)[1], lang))


def entity_id(typ: str, key: str) -> str:
    return f"{ENT_PREFIX}{typ}/{key}"


# --- dictionary + rules -----------------------------------------------------------------------------


def _is_cap(w: str) -> bool:
    return w[:1].isupper()


_INITIAL = set(".!?:;«“\"(–—-•")


def _sentence_initial(text: str, pos: int) -> bool:
    i = pos - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or text[i] == "\n" or text[i] in _INITIAL


class Casing:
    """Corpus casing of word forms outside sentence starts: a sentence-initial capitalised word may
    begin a name only when it is mostly written capitalised mid-sentence."""

    def __init__(self, texts=()):
        self.cap, self.low = Counter(), Counter()
        for t in texts:
            for m in WORD_RX.finditer(t):
                w = m.group(0)
                if _sentence_initial(t, m.start()):
                    continue
                lw = w.lower().translate(_APOS)
                if _is_cap(w):
                    self.cap[lw] += 1
                elif w[:1].islower():
                    self.low[lw] += 1

    def proper(self, w: str) -> bool:
        lw = w.lower().translate(_APOS)
        return self.cap[lw] > self.low[lw]


def _name_start(text: str, hs: int, allow_generic: bool, casing: Casing | None = None) -> int | None:
    """Start of the name that ends with the head at `hs`: the nearest capitalised word before the head
    (lower-case words, commas and "ва" may stand between), extended back through capitalised words
    (never across a comma); the walk stops at punctuation, a stop word, a genitive / case-marked word
    or another head. A sentence-initial capitalised word must be `casing.proper`."""
    D = _dicts()
    lo = max(0, hs - 400)
    words = list(WORD_RX.finditer(text, lo, hs))
    acc = []                                  # accepted words, nearest first: (start, word, comma after)
    nxt = hs
    for m in reversed(words):
        sep = text[m.end():nxt]
        if sep.strip(" \t,") or sep.count(",") > 1 or "\n" in sep:
            break
        w = m.group(0)
        lw = w.lower().translate(_APOS)
        if (lw in D["stop"] or w.isdigit() or D["stop_suffix"].search(lw)
                or any(r.match(w) for r in D["head_full"].values())):
            break
        acc.append((m.start(), w, "," in sep))
        nxt = m.start()
        if len(acc) >= D["max_words"]:
            break

    def ok_cap(i):
        s, w, _ = acc[i]
        if not _is_cap(w) or w.lower() in D["inner"]:
            return False
        return casing is None or not _sentence_initial(text, s) or casing.proper(w)

    cap = next((i for i in range(len(acc)) if ok_cap(i)), None)
    if cap is not None and any(_is_cap(acc[i][1]) and not ok_cap(i) for i in range(cap)):
        cap = None                            # a rejected capitalised word nearer the head
    if cap is None:
        if allow_generic and acc and acc[0][1].lower().translate(_APOS) in D["generic"]:
            return acc[0][0]
        return None
    j = cap
    while j + 1 < len(acc) and not acc[j + 1][2] and ok_cap(j + 1):
        j += 1
    return acc[j][0]


def dictionary_matches(text: str, lo: int = 0, hi: int | None = None, casing: Casing | None = None) -> list[dict]:
    """Entity mentions in text[lo:hi]: [{type, start, end, name, title (fixed only), rule}], sorted."""
    D = _dicts()
    hi = len(text) if hi is None else hi
    cands = []
    for typ, title, rx in D["fixed"]:
        for m in rx.finditer(text, lo, hi):
            cands.append({"type": typ, "start": m.start(), "end": m.end(), "title": title, "rule": "fixed"})
    organ_ends = {}
    for m in D["heads"]["organ"].finditer(text, lo, hi):
        e = m.end()
        while e < hi and (text[e].isalnum()):
            e += 1
        if D["head_block"].search(text[m.start():e]):
            continue
        s = _name_start(text, m.start(), False, casing)
        if s is None or s < lo:
            continue
        drop, _ = strip_prefix(text[s:e])
        s += drop
        if s >= m.start():
            continue
        cands.append({"type": "organ", "start": s, "end": e, "title": None, "rule": "head"})
    for c in cands:
        if c["type"] == "organ":
            organ_ends.setdefault(c["end"], []).append(c)
    for m in D["heads"]["position"].finditer(text, lo, hi):
        hs, e = m.start(), m.end()
        gap_end = hs
        while gap_end > lo and text[gap_end - 1] in " \t":
            gap_end -= 1
        owner = None
        if gap_end < hs and gap_end in organ_ends:
            owner = min(organ_ends[gap_end], key=lambda c: c["start"])
            s = owner["start"]
        else:
            s = _name_start(text, hs, True, casing)
            if s is None or s < lo:
                continue
            drop, _ = strip_prefix(text[s:e])
            s += drop
            if s >= hs:
                continue
        cands.append({"type": "position", "start": s, "end": e, "title": None, "rule": "owner" if owner else "head",
                      "owner": owner, "head": (hs, e)})
    # overlaps: a longer match wins; a match nested in a longer one of another type is kept
    # ("Budjet … qoʼmitasi" inside "Budjet … qoʼmitasi raisi"), partial overlaps are dropped
    cands.sort(key=lambda c: (-(c["end"] - c["start"]), c["start"], c["type"], c["rule"] != "fixed", c["title"] or ""))
    kept: list[dict] = []
    for c in cands:
        ok = True
        for k in kept:
            if c["start"] < k["end"] and k["start"] < c["end"]:
                nested = k["start"] <= c["start"] and c["end"] <= k["end"]
                if not (nested and k["type"] != c["type"]):
                    ok = False
                    break
        if ok:
            kept.append(c)
    for c in kept:
        c["name"] = text[c["start"]:c["end"]]
    return sorted(kept, key=lambda c: (c["start"], c["end"], c["type"]))


def match_key(text: str, m: dict, lang: str | None = None) -> str:
    """norm_key of a dictionary match: a fixed entry's canonical title; an owner + head position
    ("Senatining Raisi", "Oliy Majlisi Senatining Raisi") = the owner organ's key + the head lemma."""
    if m.get("title"):
        return canon_key(norm_key(m["title"]))
    if m.get("owner"):
        ok = match_key(text, m["owner"], lang)
        hk = norm_key(text[m["head"][0]:m["head"][1]], lang)
        return canon_key(f"{ok} {hk}") if ok and hk else ""
    return entity_key(m["name"], lang)


# --- LLM pass ---------------------------------------------------------------------------------------

ENT_PROMPT = """You extract entities from normative units of documents of the Oliy Majlis (parliament) of \
the Republic of Uzbekistan: articles, parts and points of laws, points of Senate resolutions, blocks of \
committee conclusions. Texts are in Uzbek (Cyrillic or Latin script) or Russian. Each unit has a unit_id, \
a context (document title | unit title) and its text.

For every unit return the entities that its norm concerns: bodies and officials it empowers, obliges or \
addresses, procedures it establishes or governs, legal categories of subjects, objects, documents or \
statuses it deals with. Entity types:
- organ: a state body, authority, organisation or institution with official competence, including \
collegial governing bodies and state funds / agencies, e.g. "Вазирлар Маҳкамаси", "Адлия вазирлиги", \
"Марказий банк", "Бош прокуратура", "судлар", "ҳокимликлар", "ваколатли давлат органи", "Олий Мажлис \
Сенати", "Божхона қўмитаси", "банк кузатув кенгаши", "Омонатларни кафолатлаш агентлиги", "Davlat soliq \
qoʼmitasi".
- position: an official post or role held by one person, e.g. "Президент", "вазир", "Сенат Раиси", "Бош \
прокурор", "ҳоким", "судья", "нотариус", "махсус бошқарувчи", "мансабдор шахс", "инспектор".
- procedure: a legal procedure or process the norm establishes or governs, e.g. "давлат рўйхатидан \
ўтказиш", "лицензиялаш", "санация қилиш", "аккредитация", "ратификация", "экспертиза", "давлат харидлари", \
"шикоятларни кўриб чиқиш". Not ordinary activities or events (a speech, a meeting, a visit).
- legal_category: a category of persons, objects, rights, documents or statuses the norm deals with, e.g. \
"юридик шахс", "жисмоний шахслар", "солиқ тўловчи", "акциядорлар", "кредиторлар", "давлат мулки", \
"лицензия", "суғурталанган шахс", "тадбиркорлик субъекти", "ер участкаси".

Rules:
1. name: a noun phrase naming the entity exactly as it is written in the unit text: a contiguous substring \
of the text, same script and spelling, the inflected form is fine ("Вазирлар Маҳкамасининг"), the fullest \
contiguous form of the name. Never translate, transliterate, normalise, reorder or paraphrase.
2. evidence: a contiguous substring of the same unit text, copied character for character (at most 250 \
characters: the clause or sentence), that contains the name.
3. Do not return: titles of laws, codes, resolutions or other acts; dates, numbers, amounts; generic words \
("ушбу Қонун", "модда", "тартиб", "ҳужжат", "масала"); personal names of people.
4. Each distinct entity once per unit (its first clear occurrence); at most 12 per unit; an empty list \
when the unit has none.
5. Return exactly one object per input unit, with the same unit_id, in the input order."""


def ent_schema(types) -> dict:
    ent = {"type": "object", "additionalProperties": False, "required": ["type", "name", "evidence"],
           "properties": {"type": {"type": "string", "enum": sorted(types)},
                          "name": {"type": "string"}, "evidence": {"type": "string"}}}
    unit = {"type": "object", "additionalProperties": False, "required": ["unit_id", "entities"],
            "properties": {"unit_id": {"type": "string"}, "entities": {"type": "array", "items": ent}}}
    return {"type": "object", "additionalProperties": False, "required": ["units"],
            "properties": {"units": {"type": "array", "items": unit}}}


def _split(text: str, a: int, b: int, limit: int) -> list[tuple[int, int]]:
    """Pieces of text[a:b] of <= limit chars, cut at paragraph ends (a longer paragraph at its last
    sentence end / whitespace before the limit)."""
    paras, pos = [], a
    for part in text[a:b].split("\n\n"):
        paras.append((pos, pos + len(part)))
        pos += len(part) + 2
    pieces, cur = [], None
    for s, e in paras:
        while e - s > limit:                                  # an overlong paragraph
            if cur:
                pieces.append(cur)
                cur = None
            chunk = text[s:s + limit]
            cut = max(chunk.rfind(". "), chunk.rfind("; "))
            cut = cut + 1 if cut > limit // 2 else (chunk.rfind(" ") if chunk.rfind(" ") > limit // 2 else limit)
            pieces.append((s, s + cut))
            s += cut
            while s < e and text[s].isspace():
                s += 1
        if s >= e:
            continue
        if cur and e - cur[0] > limit:
            pieces.append(cur)
            cur = None
        cur = (cur[0], e) if cur else (s, e)
    if cur:
        pieces.append(cur)
    out = []
    for s, e in pieces:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def llm_units(records) -> list[dict]:
    """Norm units of the LLM pass, in (doc_id, structure) order: {unit_id, doc_id, start, end, context}."""
    out = []
    for rec in sorted(records, key=lambda r: r.doc_id):
        title = " ".join((rec.meta.title or "").split())[:CTX_TITLE]
        for t in norm_tops(rec):
            pieces = _split(rec.text, t.start, t.end, UNIT_CHARS)
            for k, (s, e) in enumerate(pieces):
                uid = t.node_id if len(pieces) == 1 else f"{t.node_id}#{k + 1}"
                out.append({"unit_id": uid, "doc_id": rec.doc_id, "start": s, "end": e,
                            "context": f"{title} | {' '.join((t.title or '').split())[:CTX_TITLE]}"})
    return out


def batches(units: list[dict], size: int) -> list[list[dict]]:
    return [units[i:i + size] for i in range(0, len(units), size)]


def _norm_map(s: str) -> tuple[str, list[int]]:
    """Case / whitespace / apostrophe normalized s and, per normalized char, its index in s."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
            continue
        prev_space = False
        out.append(ch.lower().translate(_APOS))
        idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _norm(s: str) -> str:
    return _norm_map(s)[0]


def _locate(hay: str, needle: str) -> tuple[int, int] | None:
    """[start, end) in hay of the first normalized occurrence of needle, or None."""
    nh, idx = _norm_map(hay)
    nn = _norm(needle)
    if not nn:
        return None
    p = nh.find(nn)
    if p < 0:
        return None
    return idx[p], idx[p + len(nn) - 1] + 1


def verify_entity(unit_text: str, ent: dict, types) -> tuple[bool, str, tuple | None]:
    """(accepted, reason, (ev_start, ev_end, name_start, name_end) offsets in unit_text)."""
    typ, name, ev = ent.get("type"), (ent.get("name") or "").strip(), (ent.get("evidence") or "").strip()
    if typ not in types:
        return False, "bad_type", None
    if not name or not ev:
        return False, "empty", None
    loc = _locate(unit_text, ev)
    if loc is None:
        return False, "evidence_not_in_unit", None
    es, ee = loc
    nl = _locate(unit_text[es:ee], name)
    if nl is None:
        return False, "name_not_in_evidence", None
    return True, "ok", (es, ee, es + nl[0], es + nl[1])


def _shape_ok(ans) -> tuple[bool, str | None]:
    if not isinstance(ans, dict) or not isinstance(ans.get("units"), list):
        return False, "bad_shape"
    for u in ans["units"]:
        if not isinstance(u, dict) or not isinstance(u.get("unit_id"), str) or not isinstance(u.get("entities"), list):
            return False, "bad_shape"
    return True, None


def run_llm(records, llm, schema: Schema, cfg: dict | None = None, only: set | None = None):
    """Accepted LLM entity mentions [{doc_id, type, name, ev_start, ev_end, name_start, name_end, unit_id}]
    and the per-entity log (accepted and rejected, with reasons)."""
    g = cfg if cfg is not None else config.load("graph")
    size = int(g["llm"]["passes"][PASS].get("batch_units", 8))
    types = schema.entity_types
    js = ent_schema(types)
    recs = {r.doc_id: r for r in records}
    units = llm_units(records)
    bl = batches(units, size)
    if only is not None:                                  # a pilot on some batches (same cache keys)
        bl = [b for i, b in enumerate(bl) if i in only]

    def payload(b):
        return {"units": [{"unit_id": u["unit_id"], "context": u["context"],
                           "text": recs[u["doc_id"]].text[u["start"]:u["end"]]} for u in b]}

    def ask(b):
        return llm.cached_json(PASS, payload(b), js, _shape_ok, prompt=ENT_PROMPT)

    if getattr(llm, "allow_api", False):
        with ThreadPoolExecutor(API_WORKERS) as ex:
            answers = list(ex.map(ask, bl))
    else:
        answers = [ask(b) for b in bl]
    accepted, log = [], []
    for b, ans in zip(bl, answers):
        by = {u["unit_id"]: u for u in b}
        if ans is None:
            log.append({"batch": b[0]["unit_id"], "reason": "batch_rejected"})
            continue
        seen_units = set()
        for ua in ans["units"]:
            u = by.get(ua["unit_id"])
            if u is None or ua["unit_id"] in seen_units:
                for ent in ua["entities"]:
                    log.append({"unit_id": ua["unit_id"], "reason": "unknown_unit", **_ent_fields(ent)})
                continue
            seen_units.add(ua["unit_id"])
            rec = recs[u["doc_id"]]
            text = rec.text[u["start"]:u["end"]]
            for ent in ua["entities"]:
                ok, why, off = verify_entity(text, ent, types)
                row = {"unit_id": u["unit_id"], "doc_id": rec.doc_id, "reason": why, **_ent_fields(ent)}
                if ok:
                    es, ee, ns, ne = (u["start"] + x for x in off)
                    key = entity_key(rec.text[ns:ne], lang_of(rec))
                    if not key:
                        row["reason"] = "empty_key"
                        log.append(row)
                        continue
                    accepted.append({"doc_id": rec.doc_id, "unit_id": u["unit_id"], "type": ent["type"],
                                     "name": rec.text[ns:ne], "key": key, "ev_start": es, "ev_end": ee,
                                     "name_start": ns, "name_end": ne})
                log.append(row)
        missing = [uid for uid in by if uid not in seen_units]
        for uid in missing:
            log.append({"unit_id": uid, "reason": "unit_missing"})
    info = {"units": len(units), "batches": len(bl), "batch_units": size}
    return accepted, log, info


def _ent_fields(ent) -> dict:
    return {k: ent.get(k) for k in ("type", "name", "evidence")} if isinstance(ent, dict) else {"raw": ent}


# --- assembly ---------------------------------------------------------------------------------------


# --- fix round 1 (R15): context / name filters, retyping, verification ---------------------------

_F: dict | None = None


def _fix_cfg() -> dict:
    global _F
    if _F is None:
        d = config.load("graph")["dictionaries"]
        nf, rt = d["name_filter"], d["retype"]
        last = {}
        for typ in ("organ", "position"):
            for lem in rt[typ]:
                last[lem] = typ
        _F = {"boiler": re.compile("(?i)" + d["context"]["boilerplate"]),
              "max_words": int(nf["max_words"]), "patronymic": re.compile(nf["patronymic"]),
              "retype": last, "order": list(rt["order"]), "documents": set(rt.get("documents") or [])}
    return _F


class _Context:
    """Per record: gazette parentheses, act-title spans, author headings (spans where a mention is dropped)."""

    def __init__(self, rec):
        import bisect
        from graphrag.kg.acts import scan_act_mentions
        from graphrag.kg.extract_ref import _gazette_spans
        self._b = bisect
        self.gaz = _gazette_spans(rec.text)
        self.act = sorted((m.start, m.end) for m in scan_act_mentions(rec.text))
        self.boiler = []
        F = _fix_cfg()
        pos, prev = 0, 0
        for part in rec.text.split("\n\n"):
            m = F["boiler"].search(part)
            if m:                   # the heading's own text before it and the paragraph right before it
                self.boiler.append((prev, pos + m.start()))
            prev = pos
            pos += len(part) + 2

    @staticmethod
    def _inside(spans, a, b):
        return any(x <= a and b <= y for x, y in spans if x <= a)

    def reason(self, a: int, b: int) -> str | None:
        if self._inside(self.gaz, a, b):
            return "gazette"
        if self._inside(self.act, a, b):
            return "act_title"
        if self._inside(self.boiler, a, b):
            return "author_heading"
        return None


def _truncated(rec, ns: int, ne: int) -> bool:
    """The name starts inside a word, or ends inside one whose lemma differs from the name's last word
    (an inflected end such as "Маҳкамаси" in "Маҳкамасининг" is not a truncation)."""
    from graphrag.kg.concepts import word_lemmas
    t = rec.text
    if ns > 0 and t[ns - 1].isalnum():
        return True
    if ne < len(t) and t[ne].isalnum():
        a = ne
        while a > ns and (t[a - 1].isalnum() or t[a - 1] in "ʼʻ'’-"):
            a -= 1
        b = ne
        while b < len(t) and (t[b].isalnum() or t[b] in "ʼʻ'’"):
            b += 1
        lang = lang_of(rec)
        part, full = t[a:ne], t[a:b]
        return bool(part) and word_lemmas(part, lang) != word_lemmas(full, lang)
    return False


def name_reason(name: str, lang: str = "uz") -> str | None:
    """R15 rule 4 / R16 rule 5: a person (patronymic) or a clause (verb-final name)."""
    F = _fix_cfg()
    if F["patronymic"].search(name):
        return "patronymic"
    if R.is_clause(name, lang):
        return "clause"
    return None


def head_type(key: str) -> str | None:
    """organ / position by the key's last lemma (graph.yaml dictionaries.retype), else None."""
    return _fix_cfg()["retype"].get(key.split()[-1]) if key else None


VERIFY_PASS = "entity_verify"
VERIFY_PROMPT = """You check entities extracted from documents of the Oliy Majlis (parliament) of the \
Republic of Uzbekistan (Uzbek in Cyrillic or Latin script, some Russian). Each item has an id, the entity \
name, its assigned type, all surface forms seen in the corpus, and 1-2 text snippets where it occurs.

Types (an entity must be a genuine, reusable thing of exactly one type):
- organ: a state or public institution with official competence: a ministry, committee, agency, court, \
prosecutor's office, khokimiyat, chamber, council, commission, inspection, department, central or state \
bank, an institution or its collegial body. Correct: "Олий суд Маъмурий ишлар бўйича судлов ҳайъати", \
"жазони ижро этувчи муассаса", "ҳудудий ҳарбий судлар". Wrong: "манфаатдор ташкилотлар" (generic), \
"Оралиқ банк акциядорлари умумий йиғилиши" (a corporate meeting, not an institution), "Божхона \
қўмитасининг махсус жамғармаси" (a fund / account).
- position: an official public post held by one person. Correct: "Қонунчилик палатаси Спикери \
ўринбосарлари", "давлат ижрочиси", "Oliy sud raisi". Wrong: "ижтимоий ҳимоя органининг ходимлари" \
(employees), "ташкилотнинг муассиси ёки раҳбари" (founders / private roles), "бош бухгалтер" (a private \
role).
- procedure: a NAMED legal procedure (registration, licensing, accreditation, certification, expertise, \
inspection, appeal / cassation review, sanation, liquidation, bankruptcy, mediation, ratification, a \
customs regime, "… тартиби"). Correct: "денонсация", "давлат рўйхатига олиш", "Принудительная ликвидация". \
Wrong: "кўргазмали семинарлар ўтказиш" (an activity), "сунъий интеллект йўналишига инвестицияларни жалб \
қилиш" (a policy action), "гаровга қўйиш" (a transaction, not a procedure).
- legal_category: a category of persons, legal entities, objects, rights, documents or statuses. Correct: \
"юридик шахс", "масъулияти чекланган жамият", "доимий яшайдиган фуқаролиги бўлмаган шахслар". Wrong: \
"уй-жой фондида одамларнинг соғлиғига зарар етиши" (an event), "санитария қонунчиликни бузиш" (a \
violation), "ular oʼrnini bosuvchi shaxs" (depends on a pronoun).

For every item decide: keep=true with the correct type (it may differ from the assigned one) when the name \
is a genuine entity; keep=false when it is not an entity of any type (a person's personal name, the title \
of a law / code / resolution, a country or place, a date or amount, an activity, event, obligation, \
sanction or measure, a clause or text fragment, a list of several entities, a generic word). reason: one \
short sentence that quotes, verbatim, at least 4 consecutive words of a snippet (not only the name) as \
the justification. Return exactly one object per input item, with the same id, in the input order."""


def verify_schema(types) -> dict:
    it = {"type": "object", "additionalProperties": False, "required": ["id", "keep", "type", "reason"],
          "properties": {"id": {"type": "string"}, "keep": {"type": "boolean"},
                         "type": {"type": "string", "enum": sorted(types)}, "reason": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": it}}}


def _verify_shape(ans):
    if not isinstance(ans, dict) or not isinstance(ans.get("items"), list):
        return False, "bad_shape"
    return all(isinstance(x, dict) and isinstance(x.get("id"), str) for x in ans["items"]), "bad_shape"


def reason_supported(reason: str, name: str, snippets: list[str]) -> bool:
    """R16: the reason quotes >= 4 consecutive words of a snippet, at least one of them outside the name's
    occurrences in that snippet (the name alone never suffices). Normalized (case / space / apostrophes)."""
    def words(x):
        return [t for t in (w.strip("ʼ-") for w in re.findall(r"[^\W_]+(?:[ʼ-][^\W_]+)*", _norm(x))) if t]
    r = " " + " ".join(words(reason)) + " "
    # R17: stem-aware masking -- any snippet token sharing a stem with a name token (inflected occurrences
    # of the name) never counts toward the >= 4-word quote
    stems = {x[:max(3, len(x) - 4)] for x in words(name) if len(x) >= 3}
    for s in snippets:
        w = words(s)
        inside = [any(t.startswith(st) for st in stems) for t in w]
        for i in range(len(w) - 3):
            if all(inside[i:i + 4]):
                continue
            if " " + " ".join(w[i:i + 4]) + " " in r:
                return True
    return False


def _snippet(text: str, a: int, b: int, width: int = 100) -> str:
    lo = text.rfind("\n\n", 0, a)
    lo = 0 if lo < 0 else lo + 2
    hi = text.find("\n\n", b)
    hi = len(text) if hi < 0 else hi
    s, e = max(lo, a - width), min(hi, b + width)
    return " ".join(text[s:e].split())


def run_verify(items: list[dict], llm, schema: Schema, cfg: dict | None = None):
    """items: [{eid, name, type, snippets}] sorted by eid. Returns ({eid: (action, new_type)}, log)."""
    g = cfg if cfg is not None else config.load("graph")
    size = int(g["llm"]["passes"][VERIFY_PASS].get("batch_items", 40))
    types = schema.entity_types
    js = verify_schema(types)
    bl = [items[i:i + size] for i in range(0, len(items), size)]

    def payload(b):
        return {"items": [{"id": f"e{k + 1}", "name": it["name"], "type": it["type"],
                           "surface_forms": it.get("surfaces", []), "snippets": it["snippets"]}
                          for k, it in enumerate(b)]}

    def ask(b):
        return llm.cached_json(VERIFY_PASS, payload(b), js, _verify_shape, prompt=VERIFY_PROMPT)

    if getattr(llm, "allow_api", False):
        with ThreadPoolExecutor(API_WORKERS) as ex:
            answers = list(ex.map(ask, bl))
    else:
        answers = [ask(b) for b in bl]
    out, log = {}, []
    for b, ans in zip(bl, answers):
        by = {f"e{k + 1}": it for k, it in enumerate(b)}
        got = {}
        for x in (ans or {}).get("items", []):
            if x.get("id") in by and x["id"] not in got:
                got[x["id"]] = x
        for iid, it in by.items():
            x = got.get(iid)
            row = {"eid": it["eid"], "name": it["name"], "type": it["type"]}
            if x is None:
                out[it["eid"]] = ("keep", it["type"])
                log.append({**row, "verdict": "missing"})
                continue
            row.update({"keep": x.get("keep"), "new_type": x.get("type"), "reason": x.get("reason")})
            nt = x.get("type")
            if x.get("keep") and nt == it["type"]:
                out[it["eid"]] = ("keep", it["type"])
                row["verdict"] = "keep"
            elif not reason_supported(x.get("reason") or "", it["name"], it["snippets"]):
                out[it["eid"]] = ("keep", it["type"])
                row["verdict"] = "unverified"
            elif not x.get("keep") and nt == it["type"]:
                out[it["eid"]] = ("drop", None)
                row["verdict"] = "drop"
            elif nt in types:                          # a retype (keep=true, or keep=false with another type)
                if it.get("key") is None or R.allowed(it["key"], nt):
                    out[it["eid"]] = ("retype", nt)
                    row["verdict"] = "retype"
                else:                                  # R17: against the head-noun lists -> keep the type
                    out[it["eid"]] = ("keep", it["type"])
                    row["verdict"] = "retype_conflict_keep"
            else:
                out[it["eid"]] = ("keep", it["type"])
                row["verdict"] = "bad_type"
            log.append(row)
    return out, log, {"items": len(items), "batches": len(bl), "batch_items": size}


# --- assembly ---------------------------------------------------------------------------------------


def owner_rewrites() -> list[tuple]:
    """graph.yaml dictionaries.post_verify.fixed_owner_rewrites: [(title, type, key, before rx, after rx)]."""
    pv = config.load("graph")["dictionaries"]["post_verify"]
    return [(x["title"], x["type"], x["key"], re.compile(x["before"]) if x.get("before") else None,
             re.compile(x["after"].replace("^", "", 1)) if x.get("after") else None)   # used with .match(text, pos)
            for x in pv.get("fixed_owner_rewrites") or []]


def owner_post(name: str, typ: str, key: str, orw: list[tuple]) -> tuple[str, str] | None:
    """Ruling R29 B: (type, key) of the owner's post when a non-dictionary mention `name` (type / key as
    extracted) is the generic post of a fixed owner rewrite whose owner stands inside the name, before the post
    word ("Қонунчилик палатаси депутати", key "deputat" -> the position "qonunchilik palata deputat"); else None."""
    for title, t, k, bef, _ in orw:
        gen = canon_key(norm_key(title))
        if bef is None or typ != t or key != gen:
            continue
        w = list(re.finditer(r"(?i)(?<!\w)(?:" + re.escape(title[:5]) + "|" + re.escape(gen[:5]) + ")", name))
        if w and bef.search(name[:w[-1].start()]):
            return t, k
    return None


def _vertices_and_edges(mentions: list[dict]):
    """Entity vertices by (type, key) and REGULATES evidence by (src, dst) from the kept mentions."""
    ent: dict[str, dict] = {}
    evs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    kind: dict[tuple[str, str], set] = defaultdict(set)
    for m in mentions:
        eid = m["dst"]
        if eid.startswith(ENT_PREFIX):
            e = ent.setdefault(eid, {"type": m["type"], "key": m["key"], "surfaces": Counter(), "docs": set(),
                                     "sources": set(), "mentions": 0, "title": None, "fixed": False,
                                     "canon_surfaces": Counter()})
            _, s = strip_prefix(m["name"].strip())
            e["surfaces"][" ".join(s.split())] += 1
            if m.get("orig_key", m["key"]) == m["key"]:
                e["canon_surfaces"][" ".join(s.split())] += 1
            e["docs"].add(m["doc_id"])
            e["sources"].add(m["source"])
            e["mentions"] += 1
            if m.get("title"):
                e["fixed"] = True
                if e["title"] is None or m["title"] < e["title"]:
                    e["title"] = m["title"]
        if m.get("src"):
            evs[(m["src"], eid)].append(m["ev"])
            kind[(m["src"], eid)].add(m["source"])
    return ent, evs, kind


# a case suffix on the last word (genitive / accusative / dative / locative / ablative), stem >= 3 letters
_CASE_END = re.compile(r"(?<=\w{3})(?:нинг|ning|ни|ni|га|ga|да|da|дан|dan|даги|dagi)$")


def _case_cut(s: str) -> int | None:
    """Offset of a case suffix ending s, when stripping it keeps the last word's lemma (a real inflection:
    "омбудсманини" -> "омбудсмани"; not "нафақа"), else None."""
    from graphrag.kg.concepts import word_lemmas
    m = _CASE_END.search(s)
    if not m:
        return None
    words = WORD_RX.findall(s)
    if not words:
        return None
    last, cut = words[-1], words[-1][:len(words[-1]) - (m.end() - m.start())]
    lang = "ru" if _CYR_RU.search(last) else "uz"
    if cut and word_lemmas(last, lang) == word_lemmas(cut, lang):
        return m.start()
    return None


def shortest_surface(surfaces: Counter) -> str:
    """The shortest surface form (then the most frequent): the name the verification pass sees."""
    return sorted(surfaces.items(), key=lambda x: (len(x[0]), -x[1], x[0]))[0][0]


_OWNER_PREFIX = re.compile(r"(?i)^.*?(?:ҳузуридаги|huzuridagi|қошидаги|qoshidagi)\s+")
_POSS1 = re.compile(r"(?<=\w{3})(?:си|и|si|i)$")


def display_title(surfaces: Counter) -> str:
    """R15 fix 7 / R16 rule 8 / R17: a nominative surface (no case suffix) when one exists, the most frequent
    one (ties: the shortest); without an "X ҳузуридаги" owner prefix when another surface has none; a
    one-word possessive ("Ҳукумати") and a case suffix of an only-inflected name are stripped when the
    lemma is unchanged."""
    from graphrag.kg.concepts import word_lemmas
    nom = Counter({s: c for s, c in surfaces.items() if _case_cut(s) is None})
    pool = nom or surfaces
    plain = Counter({s: c for s, c in pool.items() if not _OWNER_PREFIX.match(s)})
    pool = plain or pool
    t = sorted(pool.items(), key=lambda x: (-x[1], len(x[0]), x[0]))[0][0]
    if not nom:
        cut = _case_cut(t)
        t = t[:cut] if cut else t
    if not plain:
        t = _OWNER_PREFIX.sub("", t) or t
    if len(t.split()) == 1 and (not _CYR.search(t) or re.search(r"[ўқғҳЎҚҒҲ]", t)):   # never Russian words
        m = _POSS1.search(t)
        if m:
            lang = "ru" if _CYR_RU.search(t) else "uz"
            if word_lemmas(t, lang) == word_lemmas(t[:m.start()], lang):
                t = t[:m.start()]
    return t


def _entity_nodes(ent: dict) -> list[GraphNode]:
    out = []
    for eid in sorted(ent):
        e = ent[eid]
        common = sorted(e["surfaces"].items(), key=lambda x: (-x[1], x[0]))
        # R18: a merged vertex is titled from the surfaces of its canonical key; Russian titles nominative
        title = e["title"] or display_title(e.get("canon_surfaces") or e["surfaces"])
        if not e["title"] and R.is_ru_key(e["key"]):
            title = R.ru_nominative(title)
        out.append(GraphNode(eid, "entity", e["type"], title, None, False,
                             {"norm_key": e["key"], "aliases": [s for s, _ in common[:20]],
                              "mentions": e["mentions"], "docs": len(e["docs"]),
                              "sources": sorted(e["sources"])}))
    return out


def _r18_stage(mentions: list[dict], recs: dict, r18) -> list[dict]:
    """Fix round 4 (R18), after verification (no LLM input changes): corrected head distribution, fused
    category lists split, "<ombudsman>нинг <органлари>" split, named institutions dropped, then the dedup
    merge of (type, dedup_key) groups onto one of their original keys."""
    out = []
    for m in mentions:
        ent = m["dst"].startswith(ENT_PREFIX) and m["rule"] != "fixed"
        lang = lang_of(recs[m["doc_id"]])
        if ent and m.get("composed"):
            # "kichik va oʼrta biznes subyektlari" -> "kichik biznes subyektlari" (not "kichik subyektlari")
            full = m["composed"]
            seps = R._top_level_seps(full)
            first = R._words_nopar(full[:seps[0][0]])[0]
            sw = R._words_nopar(full[seps[0][1]:])
            # the word parallel to the first conjunct: the first strict adjective of the second conjunct
            # ("жисмоний ҳамда нодавлат юридик шахслар" -> юридик), else its first word
            j = next((i for i, w in enumerate(sw[:-1]) if R.strict_adjective(w)), 0)
            nm = first + " " + " ".join(sw[j + 1:])
            if nm != m["name"]:
                k = entity_key(nm, lang)
                if k and R.head_type(k, "legal_category") == "legal_category":
                    r18["4_redistributed"][entity_id(m["type"], k)] += 1
                    m = dict(m, name=nm, key=k, dst=entity_id(m["type"], k))
        if ent and m["type"] == "legal_category" and not m.get("composed"):
            parts = R.split_category_list(m["name"])
            if parts:
                r18["4_category_list_split"][m["dst"]] += 1
                for a, b in parts:
                    nm = m["name"][a:b]
                    k = entity_key(nm, lang)
                    if not k or R.head_type(k, "legal_category") != "legal_category":
                        continue
                    x = dict(m, name=nm, key=k, ns=m["ns"] + a, ne=m["ns"] + b, dst=entity_id("legal_category", k))
                    if m["source"] == "dictionary":
                        x["es"], x["ee"] = x["ns"], x["ne"]
                    out.append(x)
                continue
        if ent and m["type"] == "organ":
            sp = R.split_owner_post(m["name"])
            if sp:
                own, rest, off = sp
                r18["4_owner_post_split"][m["dst"]] += 1
                pk = next((match_key(own, d, lang) for d in dictionary_matches(own)
                           if d["type"] == "position" and d["end"] == len(own)), None) or entity_key(own, lang)
                ok = entity_key(rest, lang)
                for typ, nm, k, a in (("position", own, pk, 0), ("organ", rest, ok, off)):
                    if not k:
                        continue
                    x = dict(m, type=typ, name=nm, key=k, ns=m["ns"] + a, ne=m["ns"] + a + len(nm), dst=entity_id(typ, k))
                    if m["source"] == "dictionary":
                        x["es"], x["ee"] = x["ns"], x["ne"]
                    out.append(x)
                continue
        if ent and m["type"] == "legal_category" and R.named_institution(m["key"], strip_prefix(m["name"].strip())[1]):
            r18["4_named_institution_dropped"][m["dst"]] += 1
            continue
        out.append(m)
    # dedup: (type, dedup_key) groups -> one original key
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for m in out:
        if m["dst"].startswith(ENT_PREFIX):
            groups[(m["type"], R.dedup_key(m["key"], m["type"]))][m["key"]] += 1
    canon = {}
    merged_log = {}
    for (typ, gk), keys in groups.items():
        def rank(kv):
            k, c = kv
            return (R.fold_key(k, typ) != gk, R.is_ru_key(k), -c, len(k), k)
        best = sorted(keys.items(), key=rank)[0][0]
        for k in keys:
            canon[(typ, k)] = best
        if len(keys) > 1:
            merged_log[f"{typ}/{best}"] = sorted(k for k in keys if k != best)
    for m in out:
        if m["dst"].startswith(ENT_PREFIX):
            ck = canon[(m["type"], m["key"])]
            if ck != m["key"]:
                r18["1_dedup_merged"][entity_id(m["type"], ck)] += 1
                m["orig_key"] = m["key"]
                m["key"] = ck
                m["dst"] = entity_id(m["type"], ck)
        if m["src"]:
            m["ev"] = Evidence(m["doc_id"], m["es"], m["ee"], recs[m["doc_id"]].text[m["es"]:m["ee"]]).to_json()
    r18["_groups"] = dict(sorted(merged_log.items()))
    return out


def llm_key(a: dict, lang: str | None = None) -> str:
    """The key of an accepted LLM name: the dictionary's key when a dictionary match of the same type ends
    the name ("Oliy Majlisining Senati" -> senat, "Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари
    бўйича вакили" -> inson huquq boʼyicha vakil), else entity_key."""
    name = a["name"].rstrip()
    for m in dictionary_matches(name):
        if m["type"] == a["type"] and m["end"] == len(name):
            k = match_key(name, m, lang)
            if k:
                return k
    return canon_key(a["key"])


def _citing_tops(records) -> dict[str, set[str]]:
    """corpus doc_id -> top-level norm ids whose text mentions that act (acts.resolve_record)."""
    from graphrag.kg.acts import ActRegistry, resolve_record
    reg = ActRegistry.from_records(records)
    out: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        tops = norm_tops(rec)
        if not tops:
            continue
        for m, ref in resolve_record(rec, reg):
            if ref is None or not ref.doc_id or ref.doc_id == rec.doc_id:
                continue
            for t in tops:
                if t.start <= m.start < t.end:
                    out[ref.doc_id].add(t.node_id)
    return out


def extract_entities(records, llm=None, concepts: list[GraphNode] | None = None,
                     defines: list[GraphEdge] | None = None, schema: Schema | None = None, stats: dict | None = None,
                     verify: bool | None = None):
    """(entity vertices, REGULATES edges to entities and concepts). llm=None: dictionary only.
    verify (default: llm given): the entity_verify LLM pass (fix 9)."""
    from graphrag.kg.extract_ref import DocIndex
    schema = schema or Schema.load()
    stats = stats if stats is not None else {}
    fx = stats.setdefault("fix_round_1", {})
    fx2 = stats.setdefault("fix_round_2", {})
    records = sorted(records, key=lambda r: r.doc_id)
    recs = {r.doc_id: r for r in records}
    docs = {r.doc_id: DocIndex(r) for r in records}
    norms_by = {r.doc_id: norm_ids(r) for r in records}
    ctxs = {r.doc_id: _Context(r) for r in records}
    casing = Casing(r.text for r in records)
    dict_counts = Counter()
    mentions: list[dict] = []
    for rec in records:
        lang = lang_of(rec)
        di = docs[rec.doc_id]
        for m in dictionary_matches(rec.text, casing=casing):
            key = match_key(rec.text, m, lang)
            if not key:
                continue
            dict_counts[f"{m['type']}:{m['rule']}"] += 1
            src = di.node_at(m["start"], m["end"])
            mentions.append({"doc_id": rec.doc_id, "type": m["type"], "key": key, "name": m["name"],
                             "source": "dictionary", "rule": m["rule"], "title": m["title"],
                             "ns": m["start"], "ne": m["end"], "es": m["start"], "ee": m["end"],
                             "src": src.node_id if src is not None and src.node_id in norms_by[rec.doc_id] else None})
    stats["dictionary"] = dict(sorted(dict_counts.items()))
    if llm is not None:
        accepted, log, info = run_llm(records, llm, schema)
        for a in accepted:
            rec = recs[a["doc_id"]]
            src = docs[rec.doc_id].node_at(a["ev_start"], a["ev_end"])
            mentions.append({"doc_id": rec.doc_id, "type": a["type"], "key": llm_key(a, lang_of(rec)),
                             "name": a["name"], "source": "llm", "rule": "llm", "title": None,
                             "ns": a["name_start"], "ne": a["name_end"], "es": a["ev_start"], "ee": a["ev_end"],
                             "src": src.node_id if src is not None and src.node_id in norms_by[rec.doc_id] else None})
        reasons = Counter(r["reason"] for r in log)
        stats["llm"] = {**info, "entities_returned": sum(v for k, v in reasons.items() if k not in ("unit_missing", "batch_rejected")),
                        "accepted": reasons.get("ok", 0),
                        "rejected": {k: v for k, v in sorted(reasons.items()) if k != "ok"},
                        "kgllm": llm.stats_json() if hasattr(llm, "stats_json") else None}
        stats["_llm_log"] = log
        stats["_llm_accepted"] = accepted

    def edges_of(ms):
        return len({(m["src"], (m["type"], m["key"])) for m in ms if m["src"]})

    fx["start"] = {"mentions": len(mentions), "regulates_entity": edges_of(mentions)}
    # fix 1: context filter; fix 4: name shape
    kept, drop1, drop4 = [], Counter(), Counter()
    for m in mentions:
        why = ctxs[m["doc_id"]].reason(m["ns"], m["ne"])
        if why:
            drop1[why + (":norm" if m["src"] else ":other")] += 1
            continue
        why = name_reason(m["name"], lang_of(recs[m["doc_id"]]))
        if not why and _truncated(recs[m["doc_id"]], m["ns"], m["ne"]):
            why = "truncated"                      # R17: the name starts / ends inside a word ("…предметом зало")
        if why:
            drop4[why + ":" + m["source"]] += 1
            continue
        kept.append(m)
    fx["1_context_filter"] = {"dropped_mentions": dict(sorted(drop1.items())), "regulates_entity_after": edges_of(kept)}
    fx["4_name_filter"] = {"dropped_mentions": dict(sorted(drop4.items()))}
    mentions = kept
    # R16 rule 4: REGULATES context (the mention stays for its vertex, the edge is dropped)
    ctx2 = {r.doc_id: R.Context2(r.text, r.meta.type) for r in records}
    c2 = Counter()
    for m in mentions:
        if m["src"]:
            why = ctx2[m["doc_id"]].reason(m["ns"], m["ne"])
            if why:
                c2[why] += 1
                m["src"] = None
    fx2["4_regulates_context"] = {"edges_dropped_mentions": dict(sorted(c2.items())), "regulates_entity_after": edges_of(mentions)}
    # R16 rule 2: coordinated lists -> separate entities (or dropped)
    new_m, sp = [], Counter()
    for m in mentions:
        parts = None if m["rule"] == "fixed" else R.split_coordinated(m["name"], m["type"])
        if parts is None:
            new_m.append(m)
            continue
        sp["fused_mentions"] += 1
        sp["conjuncts_kept"] += len(parts)
        if not parts:
            sp["dropped"] += 1
        lang = lang_of(recs[m["doc_id"]])
        for a, b in parts:
            nm = m["name"][a:b].strip()
            k = entity_key(nm, lang)
            if not k:
                continue
            x = dict(m)
            x.update({"name": nm, "key": k, "ns": m["ns"] + a, "ne": m["ns"] + a + len(nm)})
            if m["source"] == "dictionary":
                x["es"], x["ee"] = x["ns"], x["ne"]
            new_m.append(x)
    mentions = new_m
    fx2["2_coordinated_lists"] = dict(sorted(sp.items()))
    # fix 2: retype by head lemma, then a majority vote per key
    rt = Counter()
    for m in mentions:
        t = head_type(m["key"])
        if t and t != m["type"]:
            rt[f"{m['type']}->{t}"] += 1
            m["type"] = t
    order = _fix_cfg()["order"]
    votes: dict[str, Counter] = defaultdict(Counter)
    for m in mentions:
        votes[m["key"]][m["type"]] += 1
    win = {k: sorted(c.items(), key=lambda x: (-x[1], order.index(x[0]) if x[0] in order else 99, x[0]))[0][0]
           for k, c in votes.items()}
    split = sum(1 for c in votes.values() if len(c) > 1)
    mv = Counter()
    for m in mentions:
        if win[m["key"]] != m["type"]:
            mv[f"{m['type']}->{win[m['key']]}"] += 1
            m["type"] = win[m["key"]]
    fx["2_retype"] = {"head_lemma": dict(sorted(rt.items())), "split_keys": split, "majority_vote": dict(sorted(mv.items()))}
    # R16 rule 1: the head noun decides the type (named procedures, official posts, institutions only)
    hd, kept, stash, stash18 = Counter(), [], [], []
    for m in mentions:
        if m["rule"] == "fixed":
            kept.append(m)
            continue
        t = R.head_type(m["key"], m["type"])
        if t is None:
            hd[f"{m['type']}->drop"] += 1
            t18 = R.restore_head_drop(m["key"])
            if t18:                                # R18: restored after verification (never verified)
                m["type"] = t18
                m["r18"] = "head_drop_restore"
                stash18.append(m)
            continue
        if R.list_like(m["name"], t):
            parts = R.distribute_head(m["name"]) if t == "legal_category" else None
            if not parts:
                hd[f"{t}:list_or_fragment_drop"] += 1
                if R.oblique_short(m["name"]) and not R.head_level_fused(m["name"]):
                    m["type"] = t
                    stash.append(m)                # restored after verification if its entity survives
                elif t == "legal_category" and R.modifier_level_coordination(m["name"]):
                    m["type"] = t                  # R18: "товар ва молия бозорлари иштирокчилари"
                    m["r18"] = "modifier_coordination_restore"
                    stash18.append(m)
                continue
            hd["legal_category:head_distributed"] += 1
            lang = lang_of(recs[m["doc_id"]])
            for nm, off in parts:
                k2 = entity_key(nm, lang)
                if not k2 or R.head_type(k2, "legal_category") != "legal_category":
                    continue
                x = dict(m)
                x.update({"name": nm, "key": k2, "type": "legal_category"})
                if off is None:
                    x["composed"] = m["name"]      # R18: re-distributed after verification
                if off is not None:
                    x["ns"] = m["ns"] + off
                    x["ne"] = x["ns"] + len(nm)
                    if m["source"] == "dictionary":
                        x["es"], x["ee"] = x["ns"], x["ne"]
                kept.append(x)
            continue
        if t != m["type"]:
            hd[f"{m['type']}->{t}"] += 1
            m["type"] = t
        kept.append(m)
    mentions = kept
    fx2["1_head_typing"] = {"mentions": dict(sorted(hd.items())), "entities_after": len({(m["type"], m["key"]) for m in mentions})}
    # fix 6: entity keys that are concept keys -> the concept
    def _twin(k):
        return re.sub(r"^oʼzbekiston respublika ", "", k)
    ckeys = {_twin(c.attrs["norm_key"]): c.node_id for c in (concepts or [])}
    twin_keys = set()
    for m in mentions:
        cid = ckeys.get(_twin(m["key"]))
        m["dst"] = cid if cid else entity_id(m["type"], m["key"])
        if cid:
            twin_keys.add((m["type"], m["key"]))
    fx["6_concept_twins"] = {"entity_vertices_mapped": len(twin_keys)}
    for m in mentions:
        if m["src"]:
            m["ev"] = Evidence(m["doc_id"], m["es"], m["ee"], recs[m["doc_id"]].text[m["es"]:m["ee"]]).to_json()
    # fix 8: an entity without REGULATES is dropped unless a fixed dictionary name
    ent, evs, _ = _vertices_and_edges(mentions)
    connected = {dst for (_, dst) in evs}
    iso = {eid for eid, e in ent.items() if eid not in connected and not e["fixed"]}
    fx["8_isolated_dropped"] = {"vertices": len(iso),
                                "by_type": dict(sorted(Counter(ent[e]["type"] for e in iso).items()))}
    mentions = [m for m in mentions if m["dst"] not in iso]
    # fix 9: LLM verification of each remaining entity (keep / retype / drop, reason-verified)
    if verify is None:
        verify = llm is not None
    r18 = defaultdict(Counter)
    stats["fix_round_4"] = {}
    if verify and llm is not None:
        ent, _, _ = _vertices_and_edges(mentions)
        by_eid = defaultdict(list)
        for m in mentions:
            if m["dst"] in ent:
                by_eid[m["dst"]].append(m)
        items = []
        for eid in sorted(ent):
            ms = sorted(by_eid[eid], key=lambda m: (m["src"] is None, m["doc_id"], m["ns"], m["ne"]))
            snips = []
            for m in ms:
                s = _snippet(recs[m["doc_id"]].text, m["ns"], m["ne"])
                if s not in snips:
                    snips.append(s)
                if len(snips) == 2:
                    break
            title = ent[eid]["title"] or display_title(ent[eid]["surfaces"])
            surf = [x for x, _ in sorted(ent[eid]["surfaces"].items(), key=lambda x: (-x[1], x[0]))][:8]
            items.append({"eid": eid, "name": title, "type": ent[eid]["type"], "surfaces": surf,
                          "snippets": snips, "key": ent[eid]["key"]})
        verdicts, vlog, vinfo = run_verify(items, llm, schema)
        new = []
        for m in mentions:
            v = verdicts.get(m["dst"])
            if v is None or v[0] == "keep":
                new.append(m)
            elif v[0] == "drop":
                if R.restore_verify_drop(m["key"], m["type"]):
                    r18["3_verify_drop_restored"][m["dst"]] += 1      # R18 item 3
                    new.append(m)
                continue
            else:
                m["type"] = v[1]
                m["dst"] = entity_id(v[1], m["key"])
                new.append(m)
        mentions = new
        vc = Counter(r["verdict"] for r in vlog)
        fx["9_verify"] = {**vinfo, "verdicts": dict(sorted(vc.items())),
                          "retypes": dict(sorted(Counter(f"{r['type']}->{r['new_type']}" for r in vlog
                                                         if r["verdict"] == "retype").items())),
                          "kgllm": llm.stats_json().get(VERIFY_PASS) if hasattr(llm, "stats_json") else None}
        stats["_verify_log"] = vlog
    # R17: inflected short names set aside before verification come back when their entity survived it
    alive = {m["dst"]: m["type"] for m in mentions}
    back = 0
    for m in stash:
        eid = entity_id(m["type"], m["key"])
        cid = ckeys.get(_twin(m["key"]))
        dst = cid or eid
        if dst in alive:
            m["dst"] = dst
            if m["src"]:
                m["ev"] = Evidence(m["doc_id"], m["es"], m["ee"], recs[m["doc_id"]].text[m["es"]:m["ee"]]).to_json()
            mentions.append(m)
            back += 1
    fx2["oblique_restored"] = {"stashed": len(stash), "restored": back}
    for m in stash18:                               # R18 item 3: deterministic restores (not verified)
        cid = ckeys.get(_twin(m["key"]))
        m["dst"] = cid or entity_id(m["type"], m["key"])
        if m["src"]:
            m["ev"] = Evidence(m["doc_id"], m["es"], m["ee"], recs[m["doc_id"]].text[m["es"]:m["ee"]]).to_json()
        mentions.append(m)
        r18["3_" + m["r18"]][m["dst"]] += 1
    # fix 2 (after verification): a procedure named by a document noun -> legal_category
    docs_n = Counter()
    for m in mentions:
        if m["dst"].startswith(ENT_PREFIX) and m["type"] == "procedure" and m["key"].split()[-1] in _fix_cfg()["documents"]:
            m["type"] = "legal_category"
            m["dst"] = entity_id("legal_category", m["key"])
            docs_n[m["key"]] += 1
    fx["2_retype"]["document_procedures"] = {"vertices": len(docs_n), "mentions": sum(docs_n.values())}
    # R16 post-verification filters (deterministic, after the LLM pass: its inputs are unchanged)
    pv = config.load("graph")["dictionaries"]["post_verify"]
    ana = {w.lower() for w in pv["anaphora_first"]}
    cnt = re.compile(pv["count_words"])
    frag = re.compile(pv["fragment_words"]) if pv.get("fragment_words") else None
    qty = [q for q in R._cfg()["quantity"] if len(q) >= 5]
    gen = set(pv["generic_organ_keys"])
    genp = set(pv.get("generic_position_keys") or [])
    corp = set(pv.get("corporate_lemmas") or [])
    catorg = set(pv.get("category_organ_heads") or [])
    fixed_keys = {canon_key(norm_key(t)) for _, t, _ in _dicts()["fixed"]}
    plist = re.compile(pv["position_list"]) if pv.get("position_list") else None
    pvc, kept = Counter(), []
    # R20 (d): fixed-dictionary exclusions ("депутатлик") and owner rewrites ("Қонунчилик палатаси
    # депутатлари" -> the chamber's deputy post), after verification: the LLM inputs are unchanged
    fx_ex = {t: re.compile(rx) for t, rx in (pv.get("fixed_exclude") or {}).items()}
    orw = owner_rewrites()
    r20d = stats.setdefault("r20", {}).setdefault("deputat", {"excluded": [], "rewritten": []})
    kept_d = []
    for m in mentions:
        if m["rule"] == "fixed" and m.get("title") in fx_ex and fx_ex[m["title"]].search(m["name"]):
            r20d["excluded"].append({"doc_id": m["doc_id"], "src": m["src"], "name": m["name"]})
            continue
        if m["rule"] == "fixed":
            t_ = recs[m["doc_id"]].text
            for title, typ, key, bef, aft in orw:
                if m.get("title") != title:
                    continue
                mb = bef.search(t_[max(0, m["ns"] - 80):m["ns"]]) if bef is not None else None
                ma = aft.match(t_, m["ne"]) if aft is not None and mb is None else None
                if mb is None and ma is None:
                    continue
                old_dst = m["dst"]
                ns = m["ns"] - (len(mb.group(0)) if mb else 0)
                ne = ma.end() if ma else m["ne"]
                m.update({"ns": ns, "ne": ne, "es": ns, "ee": ne, "name": t_[ns:ne], "key": key, "type": typ,
                          "title": None, "rule": "owner_rewrite", "dst": entity_id(typ, key)})
                if m["src"]:
                    m["ev"] = Evidence(m["doc_id"], ns, ne, t_[ns:ne]).to_json()
                r20d["rewritten"].append({"doc_id": m["doc_id"], "src": m["src"], "name": m["name"], "from": old_dst,
                                          "to": m["dst"]})
                break
        elif m["dst"].startswith(ENT_PREFIX) and (got := owner_post(m["name"], m["type"], m["key"], orw)):
            # Ruling R29 B: a non-dictionary mention whose name holds the owner ("Қонунчилик палатаси депутати",
            # LLM, key stripped to the generic "deputat": QONUNLAR-3f9b4f02/СТ.11/Ч.1/П.24) is the owner's post too
            old_dst = m["dst"]
            m.update({"key": got[1], "type": got[0], "rule": "owner_rewrite", "dst": entity_id(*got)})
            r20d["rewritten"].append({"doc_id": m["doc_id"], "src": m["src"], "name": m["name"], "from": old_dst,
                                      "to": m["dst"]})
        kept_d.append(m)
    mentions = kept_d
    for m in mentions:
        if m["dst"].startswith(ENT_PREFIX) and m["rule"] != "fixed" and m["type"] == "organ":
            last_w = (WORD_RX.findall(m["name"]) or [""])[-1].lower()
            if re.search(r"(?:лари|lari)$", last_w) and re.search(r"(?i)[,]|\s(?:ва|va|ҳамда|hamda)\s", m["name"]) \
                    and R.split_coordinated(m["name"], "organ") is None and re.search(r"(?i)\s(?:ва|va)\s+[A-ZА-ЯЎҚҒҲ]", m["name"]):
                pvc["organ:fused_plural"] += 1               # "Қишлоқ ва Сув хўжалиги вазирликлари": two ministries
                continue
            if m["key"].split()[-1] in catorg and m["key"] not in fixed_keys:
                pvc["organ:bank->legal_category"] += 1
                m["type"] = "legal_category"
                m["dst"] = entity_id("legal_category", m["key"])
        if m["dst"].startswith(ENT_PREFIX) and m["rule"] != "fixed" and m["type"] in ("organ", "position") \
                and set(m["key"].split()) & corp:
            pvc[f"{m['type']}:corporate->legal_category"] += 1
            m["type"] = "legal_category"
            m["dst"] = entity_id("legal_category", m["key"])
        if m["dst"].startswith(ENT_PREFIX) and m["rule"] != "fixed" and m["type"] == "position" and \
                (m["key"] in genp or (plist is not None and plist.search(m["name"]))):
            pvc["position:" + ("generic" if m["key"] in genp else "list")] += 1
            continue
        if m["dst"].startswith(ENT_PREFIX) and m["rule"] != "fixed":
            first = (WORD_RX.findall(m["name"]) or [""])[0].lower().translate(_APOS)
            lastlem = (m["key"].split() or [""])[-1]
            why = ("anaphora" if first in ana else "count" if cnt.search(m["name"]) else
                   "quantity" if m["type"] == "legal_category" and any(lastlem.startswith(q) for q in qty) else
                   "fragment" if frag is not None and frag.search(m["name"]) else
                   "generic_organ" if m["type"] == "organ" and m["key"] in gen else
                   "alternatives" if m["type"] in ("organ", "position") and
                   re.search(r"(?i)(?<!\w)(?:ёки|yoki|ёхуд|yoxud)(?!\w)", m["name"]) else None)
            if why:
                pvc[f"{m['type']}:{why}"] += 1
                continue
        kept.append(m)
    mentions = kept
    # a signature line ("Ўзбекистон Республикаси / Олий Мажлиси Сенатининг Раиси / Т. Норбаева"): a short
    # paragraph that is (almost) only the position's name -> no REGULATES
    for m in mentions:
        if m["src"] and m["type"] in ("position", "organ"):
            t = recs[m["doc_id"]].text
            lo = t.rfind("\n\n", 0, m["ns"])
            hi = t.find("\n\n", m["ne"])
            para = t[(lo + 2 if lo >= 0 else 0):(hi if hi >= 0 else len(t))].strip()
            if len(para) - (m["ne"] - m["ns"]) <= 25 and para.endswith(":"):
                r18["5_list_intro_not_signature"][m["dst"]] += 1   # R18 item 5: "Вазирлар Маҳкамаси:" + duties
                continue
            if len(para) - (m["ne"] - m["ns"]) <= 25:
                m["src"] = None
                pvc[f"{m['type']}:signature_edge"] += 1
    # R20 (b): adoption / approval / signature lines ("… Қонунчилик палатаси томонидан қабул қилинди",
    # "Одобрен Сенатом") are legislative history -> no REGULATES (the mention keeps its vertex)
    r20 = stats.setdefault("r20", {}).setdefault("adoption_line_edges", [])
    for m in mentions:
        if m["src"]:
            why = R.adoption_line(recs[m["doc_id"]].text, m["ns"], m["ne"])
            if why:
                r20.append({"src": m["src"], "dst": m["dst"], "rule": why, "quote": recs[m["doc_id"]].text[m["ns"]:m["ne"]]})
                m["src"] = None
                pvc[f"{m['type']}:adoption_line_edge"] += 1
    # a position nested in an organ name ("халқ депутатлари маҳаллий Кенгашлари") is a modifier; a name
    # followed by "… тўғрисидаги кодекс" is inside an unquoted act title: no REGULATES
    ata = re.compile(pv["act_title_after"].replace("^", "", 1)) if pv.get("act_title_after") else None
    organ_spans = defaultdict(list)
    for m in mentions:
        if m["type"] == "organ":
            organ_spans[m["doc_id"]].append((m["ns"], m["ne"]))
    # R17: owner / modifier checks with the name extended to its word end ("Марказий банки|нинг қайта
    # молиялаш ставкаси"); an organ nested in a position name ("Олий Мажлис Сенати Раисининг") is its owner
    c2cfg = R._cfg()
    pos_spans = defaultdict(list)
    for m in mentions:
        if m["type"] == "position":
            pos_spans[m["doc_id"]].append((m["ns"], m["ne"]))
    for m in mentions:
        if not m["src"] or m["type"] != "organ":
            continue
        t_ = recs[m["doc_id"]].text
        e2 = m["ne"]
        while e2 < len(t_) and (t_[e2].isalnum() or t_[e2] in "ʼʻ'’"):
            e2 += 1
        if c2cfg["owner_after"].match(t_, e2) or c2cfg["modifier_after"].match(t_, e2):
            m["src"] = None
            pvc["organ:owner_or_modifier_edge"] += 1
        elif any(a <= m["ns"] and m["ne"] <= b and (a, b) != (m["ns"], m["ne"]) for a, b in pos_spans[m["doc_id"]]):
            m["src"] = None
            pvc["organ:owner_of_position_edge"] += 1
    # R17: a mention nested in another organ / position name of the same doc is that name's owner
    # ("Олий судининг Иқтисодий ишлар бўйича судлов ҳайъати", "Сенати Раисининг биринчи ўринбосари"); an
    # addressee of an Address ("Олий Мажлис ва Ўзбекистон халқига Мурожаатномаси") is no regulation
    all_spans = defaultdict(list)
    for m in mentions:
        if m["type"] in ("organ", "position"):
            all_spans[m["doc_id"]].append((m["ns"], m["ne"], m["dst"]))
    addressee = re.compile(r"(?i)(?:[^.;:\n]|(?<=\b[А-ЯA-ZЎҚҒҲ])\.){0,90}?(?:Мурожаатнома|Murojaatnoma|Послани)")
    deputy = re.compile(r"(?i)\s+(?:биринчи\s+|birinchi\s+)?(?:ўринбосар|oʼrinbosar|o'rinbosar|ёрдамчи|yordamchi)")
    for m in mentions:
        if not m["src"] or m["type"] not in ("organ", "position"):
            continue
        t_ = recs[m["doc_id"]].text
        nested = any(a <= m["ns"] and m["ne"] <= b and (a, b) != (m["ns"], m["ne"]) and d != m["dst"]
                     for a, b, d in all_spans[m["doc_id"]])
        if nested and re.search(r"(?i)(?:нинг|ning)$", m["name"]):
            m["src"] = None
            pvc[f"{m['type']}:nested_owner_edge"] += 1
        elif m["type"] == "position" and re.search(r"(?i)(?:нинг|ning)$", m["name"]) and deputy.match(t_, m["ne"]):
            m["src"] = None
            pvc["position:owner_of_deputy_edge"] += 1
        elif addressee.match(t_, m["ne"]):
            m["src"] = None
            pvc[f"{m['type']}:addressee_edge"] += 1
        elif m["type"] == "organ" and re.search(r"(?i)(?:нинг|ning)$", m["name"]) and \
                any(0 <= a - m["ne"] <= 3 and d.startswith(ENT_PREFIX + "position/") for a, b, d in all_spans[m["doc_id"]]):
            m["src"] = None                        # "Олий Мажлисининг Бола ҳуқуқлари бўйича вакили": the post's owner
            pvc["organ:owner_of_next_position_edge"] += 1
    ddc = config.load("graph")["dictionaries"]["dedup"]
    cm_before, cm_after = re.compile(ddc["council_modifier_before"]), re.compile(ddc["council_modifier_after"])
    organ_starts = defaultdict(list)
    for m in mentions:
        if m["type"] == "organ":
            organ_starts[m["doc_id"]].append(m["ns"])
    for m in mentions:
        if m["src"] and m["type"] == "organ" and re.search(r"(?i)(?:нинг|ning)$", m["name"]) and \
                any(0 < s0 - m["ne"] <= 60 and not re.search(r"[.;:,\n]", recs[m["doc_id"]].text[m["ne"]:s0])
                    for s0 in organ_starts[m["doc_id"]]):
            m["src"] = None                        # R17: a genitive owner of the next organ ("Олий судининг … ҳайъати")
            pvc["organ:owner_of_organ_edge"] += 1
    for m in mentions:
        if not m["src"]:
            continue
        if m["type"] == "position" and any(a <= m["ns"] and m["ne"] <= b and (a, b) != (m["ns"], m["ne"])
                                           for a, b in organ_spans[m["doc_id"]]):
            m["src"] = None
            pvc["position:nested_in_organ_edge"] += 1
        elif m["type"] == "position" and cm_before.search(recs[m["doc_id"]].text[max(0, m["ns"] - 12):m["ns"]]) and \
                cm_after.match(recs[m["doc_id"]].text[m["ne"]:m["ne"] + 120]):
            m["src"] = None                        # R18: "халқ депутатлари маҳаллий Кенгашлари" (a council name)
            r18["1b_council_name_modifier_edge"][m["dst"]] += 1
        elif ata is not None and ata.match(recs[m["doc_id"]].text, m["ne"]):
            m["src"] = None
            pvc[f"{m['type']}:act_title_edge"] += 1
    mentions = _r18_stage(mentions, recs, r18)
    # R18: an entity equal to a concept up to transliteration folds / a trailing light verb ("санация қилиш
    # тартиб-таомилини амалга ошириш" = the concept "санация қилиш тартиб-таомиллари") is that concept
    cfold = defaultdict(set)
    for c in concepts or []:
        cfold[R.fold_key(_twin(c.attrs["norm_key"]), "procedure")].add(c.node_id)
    for m in mentions:
        if not m["dst"].startswith(ENT_PREFIX):
            continue
        f = R.fold_key(_twin(m.get("orig_key", m["key"])), "procedure")
        cands = [f] + [f[:-len(t) - 1] for t in ("amal oshirish", "oʼtkazish") if f.endswith(" " + t)]
        hit = next((cfold[x] for x in cands if len(cfold.get(x, ())) == 1), None)
        if hit:
            r18["1c_concept_twin"][m["dst"]] += 1
            m["dst"] = next(iter(hit))
    ent_t, evs_t, _ = _vertices_and_edges(mentions)
    conn = {dst for (_, dst) in evs_t}
    iso2 = {eid for eid, e in ent_t.items() if eid not in conn and not e["fixed"]}
    mentions = [m for m in mentions if m["dst"] not in iso2]
    fx2["post_verify_filters"] = {"dropped_mentions": dict(sorted(pvc.items())), "isolated_after": len(iso2)}
    stats["fix_round_4"].update({k: {"mentions": sum(v.values()), "vertices": len(v),
                                     **({"ids": sorted(v)} if len(v) <= 40 else {})}
                                 for k, v in sorted(r18.items()) if not k.startswith("_")})
    stats["fix_round_4"]["dedup_groups"] = r18.get("_groups", {})
    # Ruling R29 B: REGULATES evidence = the mention's own name span (the shortest exact span naming the entity),
    # not the LLM's clause / sentence (up to 250 chars, often naming several entities); src = the deepest node
    # holding that span (the μ rule, unchanged when the clause lay in one node)
    shrunk = Counter()
    for m in mentions:
        if m["src"] and (m["es"], m["ee"]) != (m["ns"], m["ne"]):
            m["es"], m["ee"] = m["ns"], m["ne"]
            m["ev"] = Evidence(m["doc_id"], m["ns"], m["ne"], recs[m["doc_id"]].text[m["ns"]:m["ne"]]).to_json()
            n = docs[m["doc_id"]].node_at(m["ns"], m["ne"])
            src = n.node_id if n is not None and n.node_id in norms_by[m["doc_id"]] else m["src"]
            shrunk["src_moved" if src != m["src"] else "src_kept"] += 1
            m["src"] = src
    stats["r29_evidence_name_span"] = dict(sorted(shrunk.items()))
    ent, evs, kind = _vertices_and_edges(mentions)
    nodes = _entity_nodes(ent)
    ent_edges = {k: v for k, v in evs.items() if k[1].startswith(ENT_PREFIX)}
    twin_evs = {k: v for k, v in evs.items() if k[1].startswith("CON/")}
    edges = [_edge(src, dst, "REGULATES", ev) for (src, dst), ev in sorted(ent_edges.items())]
    stats["regulates_entity_by_source"] = dict(sorted(Counter("+".join(sorted(kind[(e.src, e.dst)])) for e in edges).items()))
    if concepts is not None:
        ospans = defaultdict(list)
        for m in mentions:
            if m["type"] == "organ":
                ospans[m["doc_id"]].append((m["ns"], m["ne"]))
        cedges, cstats = concept_mentions(records, concepts, defines or [], docs, _citing_tops(records), ospans)
        kept_c = []
        for e in cedges:                           # R20 (b) for concept mentions as well
            evs = list(e.attrs.get("evidence_all") or [e.attrs["evidence"]])
            ok = []
            for ev in evs:
                why = R.adoption_line(recs[ev["doc_id"]].text, ev["start"], ev["end"])
                if why:
                    r20.append({"src": e.src, "dst": e.dst, "rule": why, "quote": ev["quote"]})
                else:
                    ok.append(ev)
            if len(ok) == len(evs):
                kept_c.append(e)
            elif ok:
                kept_c.append(_edge(e.src, e.dst, "REGULATES", ok))
        cedges = kept_c
        stats["concept_mentions"] = cstats
        has_def = {(e.src, e.dst) for e in defines or []}
        merged = {(e.src, e.dst): list(e.attrs.get("evidence_all") or [e.attrs["evidence"]]) for e in cedges}
        tw = 0
        for k, v in twin_evs.items():
            if k in has_def:
                continue
            tw += k not in merged
            merged.setdefault(k, []).extend(v)
        fx["6_concept_twins"]["new_concept_edges"] = tw
        edges += [_edge(src, dst, "REGULATES", ev) for (src, dst), ev in sorted(merged.items())]
    fx["end"] = {"entities": len(nodes), "regulates_entity": len(ent_edges)}
    return nodes, edges


def build(records, llm=None, schema: Schema | None = None, verify: bool | None = None):
    """Concepts + DEFINES, entities + REGULATES for a corpus; every edge Schema-validated against the
    structure / concept / entity vertices. Returns (nodes, edges, stats)."""
    from graphrag.kg.extract_ref import structure_nodes
    schema = schema or Schema.load()
    records = sorted(records, key=lambda r: r.doc_id)
    cn, defines = [], []
    for r in records:
        n, e = extract_concepts(r)
        cn += n
        defines += e
    concepts = merge_concepts(cn)
    stats: dict = {}
    ents, regs = extract_entities(records, llm, concepts, defines, schema, stats, verify)
    graph_nodes = {n.node_id: n for r in records for n in structure_nodes(r)}
    new_nodes = concepts + ents
    for n in new_nodes:
        graph_nodes[n.node_id] = n
    texts = {r.doc_id: r.text for r in records}
    errs = [m for n in new_nodes for m in schema.validate_node(n)]
    edges = []
    for e in defines + regs:
        ee = schema.validate_edge(e, graph_nodes)
        for ev in [e.attrs["evidence"]] + list(e.attrs.get("evidence_all") or []):
            if texts[ev["doc_id"]][ev["start"]:ev["end"]] != ev["quote"]:
                ee.append(f"{e.type} {e.src} -> {e.dst}: evidence quote is not the text span")
        if ee:
            errs += ee
            continue
        edges.append(e)
    stats["schema_errors"] = len(errs)
    stats["schema_error_sample"] = errs[:10]
    stats["concepts"] = len(concepts)
    stats["entities_by_type"] = dict(sorted(Counter(n.type for n in ents).items()))
    stats["defines"] = len(defines)
    stats["regulates"] = sum(1 for e in edges if e.type == "REGULATES")
    return new_nodes, edges, stats


def _src_type(nid: str, nodes_by_id) -> str:
    n = nodes_by_id.get(nid)
    return n.type if n else "?"


def main(argv=None) -> int:
    from graphrag.data.ingest import read_records
    from graphrag.kg.extract_ref import structure_nodes
    from graphrag.kg.llm import KgLLM
    argv = list(argv or [])
    recs = read_records(config.path("data/records"))
    llm = KgLLM() if "--llm" in argv else None
    nodes, edges, stats = build(recs, llm, verify=False if "--no-verify" in argv else None)
    log = stats.pop("_llm_log", [])
    vlog = stats.pop("_verify_log", [])
    accepted = stats.pop("_llm_accepted", [])
    struct = {n.node_id: n for r in recs for n in structure_nodes(r)}
    allnodes = {**struct, **{n.node_id: n for n in nodes}}
    by = Counter()
    for e in edges:
        by[(e.type, _src_type(e.src, allnodes), allnodes[e.dst].kind)] += 1
    stats["edges_by_type_src_dst"] = {f"{t} {s}->{k}": v for (t, s, k), v in sorted(by.items())}
    deg = Counter(e.dst for e in edges if e.type == "REGULATES")
    stats["top30_entities_by_degree"] = [(nid, allnodes[nid].title, c) for nid, c in
                                         sorted(((n, c) for n, c in deg.items() if n.startswith(ENT_PREFIX)),
                                                key=lambda x: (-x[1], x[0]))[:30]]
    print(json.dumps({k: v for k, v in stats.items()}, ensure_ascii=False, indent=1))
    if "--out" in argv:
        d = config.path(argv[argv.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "ent_nodes.jsonl").write_text("".join(json.dumps(n.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                                   for n in nodes), encoding="utf-8")
        (d / "ent_edges.jsonl").write_text("".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                                   for e in edges), encoding="utf-8")
        (d / "ent_llm_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                     for r in log), encoding="utf-8")
        (d / "ent_llm_accepted.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                          for r in accepted), encoding="utf-8")
        (d / "ent_verify_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                        for r in vlog), encoding="utf-8")
        (d / "ent_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return 1 if stats["schema_errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
