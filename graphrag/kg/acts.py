"""Act registry (M2 Task 3): normalized act keys and resolution of act titles / code names /
the Constitution to corpus Document vertices or external act stubs.

act_key(title) is identical across Cyrillic/Latin script, quote styles, case endings, the
"Ўзбекистон Республикасининг" boilerplate and the "…ги Қонуни" wrapper, and keeps different acts
apart (Жиноят vs Жиноят-процессуал кодекси; "X тўғрисида" laws with different X).

ActRegistry.from_records(records) indexes the corpus laws (meta.type == "law_text"):
  * editions: several standing-law documents with one key are ONE act vertex (the latest dated
    one); the others get REFERS_TO {relation: "version_of", evidence: their title span} to it;
  * amending laws ("… ўзгартириш ва қўшимчалар киритиш тўғрисида") are never merged: same-key
    amending laws are distinct acts, told apart by resolve(date_hint) or listed in `ambiguous`
    for the LLM ref_resolve pass (Task 4);
  * anything else resolves to an external stub EXT/<act_key>.
No API calls; everything here is deterministic.
"""
from __future__ import annotations

import datetime as _dt
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from graphrag import config
from graphrag.data.legacy.normalize import APOS, skeleton_key
from graphrag.kg.schema import Evidence, GraphEdge, GraphNode, Schema

A = APOS                                   # canonical apostrophe of skeleton_key ("ʼ")
TG = f"to{A}g{A}risida"                    # canonical "тўғрисида"
DOC_TYPE = "документ"
EXT_PREFIX = "EXT/"

# --- act_key -------------------------------------------------------------------------------------

_ASCII_APOS = re.compile(r"(?<=[A-Za-z])['‘’ʻ`´](?=[A-Za-z])")      # o'zbek -> oʼzbek (skeleton strips ')
_PAREN = re.compile(r"\([^()]*\)")                                  # "(Тошкент, 2024 йил 18 июль)", "(омбудсман)"
_BOILER = re.compile(rf"\bo{A}?zbekiston respublikasi(?:ning)?\b")
_CITATION = [
    re.compile(r".*\btasdiqlangan\s+"),                               # "... билан тасдиқланган X" -> X
    re.compile(r"\b\d{4} yil\w* \d{1,2} [a-z]+\w*"),                  # "2023 йил 1 майдаги" (not "2024 йил учун")
    re.compile(rf"\b(?:o{A}?rq|orq|zru|pf|pq|xii|xi)(?: \d+)*(?: son\w*)?\b"),  # ЎРҚ-837-сон
    re.compile(r"\b\d+(?: [a-z]+)? son\w*\b"),                         # "837 xii sonli"
    re.compile(r"\byangi tahrir\w*\b"),
]
# тўғрисида + forms, and the source typos "toʼgʼrisadi" / "toʼgʼrisada" / "toʼgʼrsida" / "toʼgʼrsi"
_TG_FORMS = re.compile(rf"\bto{A}?g{A}?ri?s(?:ida|ada|adi|i)\w*|\bhaqida\w*")
_KODEKS = re.compile(r"\bkodeks\w*")
_KONST = re.compile(r"\bkonstitu(?:t)?siya(?:si|miz)?(?:ning|ga|ni|da|dan|dagi)?\b")
_QONUN = re.compile(r"\bqonun(?:i|ining|iga|ini|ida|idan|ga|ning|ni|da|lari|larining)?\b")
# Act-title classes as data (config/graph.yaml act_titles, Ruling R31 (3)): the amendment-word tail folded to
# amend_canonical, amending / repealing titles, subject-less omnibus titles (-> generic_key), code approval laws,
# the law-text adoption line (_ADOPTED, below)
_TITLES = config.load("graph")["act_titles"]


def _title_rx(key: str) -> re.Pattern:
    return re.compile(_TITLES[key].replace("{AMEND_WORD}", _TITLES["amend_word"]))


_AMEND_TAIL = _title_rx("amend_tail")
_HEADS = re.compile(r"^(reglament|nizom|bitim|shartnoma|konventsiya|konvensiya|protokol|"
                    r"dastur|farmon|qaror|kodeks|konstitutsiya)(?:s?i)?(?:ning|ni|ga|da|dagi|gi|dan)?$")
_CASE_END = re.compile(r"(?:ning|ga|ni|gi|da)$")
_AMEND_CANON = _TITLES["amend_canonical"]
_AMENDING = _title_rx("amending")
_GENERIC = _title_rx("generic")
_CODE_APPROVAL = _title_rx("code_approval")


# Orthographic variants that the Cyrillic->Latin mapping cannot settle, folded on the final key:
# word-initial е/э/e/ye (Ер -> "er" = Latin "Yer"/"Er"), ц = ts/s (акциядорлик = aksiyadorlik,
# конвенция = konvensiya), consonant + ю = yu/u (бюджет = budjet, валюта = valuta), ҳамда = ва,
# consonant + ъе = ye (субъект -> subʼekt = Latin subyekt).
_ORTHO = [(re.compile(rf"(?<=[bdfjklmnpqrstvxz]){A}e"), "ye"),     # субъект = subyekt, объект = obyekt
          (re.compile(r"\bye"), "e"), (re.compile(r"ts"), "s"),
          (re.compile(r"(?<=[bdfgjklmnpqrstvxz])yu"), "u"), (re.compile(r"\bhamda\b"), "va")]
# possessive -(s)i of the word right before "тўғрисида": "Давлат божи" = "давлат бож",
# "Миллий гвардияси" = "Миллий гвардия", "Марказий банки" = "Марказий банк"
_POSS = re.compile(rf"(?:(?<=[aeiou{A}])si|(?<=[bdfghjklmnpqrstvxz])i)(?= {TG}\b)")


def _prefold(title: str) -> str:
    s = unicodedata.normalize("NFKC", title)
    s = _ASCII_APOS.sub(A, s)
    return _PAREN.sub(" ", s)


def act_key(title: str) -> str:
    """Normalized, script- and case-insensitive identity key of an act name ("" for empty input)."""
    if not title or not title.strip():
        return ""
    k = skeleton_key(_prefold(title), drop_boilerplate=True)
    for rx in _CITATION:
        k = rx.sub(" ", k)
    k = _BOILER.sub(" ", k)
    k = _TG_FORMS.sub(TG, k)
    k = _KODEKS.sub("kodeks", k)
    k = k.replace(f"{TG} kodeks", "kodeks")             # "Маъмурий жавобгарлик тўғрисидаги кодекси"
    k = _KONST.sub("konstitutsiya", k)
    k = _QONUN.sub("qonun", k)
    k = re.sub(r"\bgi\b", " ", k)                       # «…»ги
    k = _AMEND_TAIL.sub(_AMEND_CANON, k)
    k = _POSS.sub("", k)
    words = k.split()
    # leading "Лойиҳа", "(Ўзбекистон Республикасининг) ҚОНУНИ", a stray "Respublikasining"
    while words and (words[0] in ("loyiha", "qonun") or words[0].startswith("respublikasi")):
        words.pop(0)
    while words and words[-1] == "qonun":              # "… тўғрисида»ги Қонуни"
        words.pop()
    if words:
        last = words[-1]
        m = _HEADS.match(last)
        if m:
            words[-1] = m.group(1)
        elif last != TG and len(last) >= 7:
            words[-1] = _CASE_END.sub("", last)
    k = " ".join(words)
    for rx, rep in _ORTHO:
        k = rx.sub(rep, k)
    if _GENERIC.match(k):                              # omnibus variants ("… oʼzgartirish oʼzgartirish kiritish")
        return GENERIC_KEY                             # share one canonical key (registry lookups, R7)
    return k


GENERIC_KEY = _TITLES["generic_key"]


def is_amending(key: str) -> bool:
    """An amending / repealing law: its title names what it changes, not a standing act."""
    return bool(_AMENDING.search(key))


def is_generic(key: str) -> bool:
    """A subject-less omnibus title ("айрим қонун ҳужжатларига … киритиш тўғрисида"): many
    different acts share it, so it never makes an external stub."""
    return bool(_GENERIC.match(key))


_BOILER_SURFACE = re.compile(r"^\s*(?:Ўзбекистон|O[ʻʼ'‘’`]?zbekiston)\s+(?:Республикаси|Respublikasi)"
                             r"(?:нинг|ning)?\s+", re.I)
_CODE_SURFACE = re.compile(r"(кодекс|kodeks)\w*$", re.I)
_KONST_SURFACE = re.compile(r"^(Конституция|Konstitutsiya)\w*$", re.I)


def display_name(surface: str) -> str:
    """Readable nominative name of a mention: boilerplate prefix off, codes as "X кодекси"."""
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", surface)).strip().strip("«»“”„\"")
    s = _BOILER_SURFACE.sub("", s)
    m = _CODE_SURFACE.search(s)
    if m:
        s = s[:m.start()] + ("кодекси" if m.group(1)[0] in "кК" else "kodeksi")
    m = _KONST_SURFACE.match(s)
    if m:
        s = "Конституция" if m.group(1)[0] == "К" else "Konstitutsiya"
    return s[:1].upper() + s[1:]


# --- mention scanning (config/graph.yaml patterns + code_names) ----------------------------------

@dataclass(frozen=True)
class Mention:
    start: int          # span of the mention in the scanned text
    end: int
    name: str           # what to resolve (quoted title / "X кодекси" / "Конституция…")
    kind: str           # act_title | code_name | code_list | constitution
    number: str | None = None   # "oʼrq-850" / "310-ii": the cited act's registration number (act_title)
    date: str | None = None     # ISO adoption date written right before the title ("2011 йил 21 декабрда қабул қилинган")
    date_kind: str | None = None  # "chamber" ("Қонунчилик палатаси томонидан … қабул қилинган") | "stated"


_SCAN: dict | None = None
_TOKEN = re.compile(r"[\w’ʼʻ'`‘–—-]+|[,;]")
# words act_key drops at the start of a name: never the first word of a code name
_NOT_FIRST = re.compile(rf"^(?:qonun\w*|loyiha|respublika\w*|o{A}?zbekiston\w*)$")


def _scan_cfg() -> dict:
    """Compiled patterns and the code-name lexicon {act_key("<name> кодекс"): canonical name}."""
    global _SCAN
    if _SCAN is None:
        cfg = config.load("graph")
        pats = {k: re.compile("(?i)" + v) for k, v in (cfg.get("patterns") or {}).items()
                if isinstance(v, str)}                  # patterns.refs is extract_ref's (Task 4)
        lex: dict[str, str] = {}
        max_words = 1
        for entry in cfg.get("code_names") or []:
            names = [n.strip() for n in entry.split("|") if n.strip()]
            for n in names:
                lex[act_key(f"{n} кодекс")] = names[0]
                max_words = max(max_words, len(n.split()))
        _SCAN = {"p": pats, "lex": lex, "max_words": max_words}
    return _SCAN


_QUOTE_CHARS = set("«»“”„\"")


def _quoted_mentions(text: str, p: dict) -> list[Mention]:
    """«X»ги / “X”gi titles followed by a law/code head; every member of a list
    "“X”gi, “Y”gi va “Z”gi qonunlarga" is a mention."""
    out = []
    qs = list(p["act_quoted"].finditer(text))
    i = 0
    while i < len(qs):
        j = i
        while j + 1 < len(qs) and p["act_list_sep"].match(text[qs[j].end():qs[j + 1].start()]):
            j += 1
        chain, i = qs[i:j + 1], j + 1
        h = p["act_head"].match(text, chain[-1].end())
        if h and any(p["act_head_block"].match(w) for w in h.group("gap").split()):
            h = None
        if not h:
            before = text[max(0, chain[0].start() - 60):chain[0].start()]
            if not (p["act_paren_head"].search(before) and text[chain[-1].end():].lstrip().startswith(")")):
                continue
        num = None
        if h:
            n = p["act_number"].search(h.group("gap"))
            num = _norm_number(n.group(0)) if n else None
        cd = p["act_context_date"].search(text[max(0, chain[0].start() - 120):chain[0].start()])
        cdate = _parse_date(cd.group("date")) if cd else None
        ckind = ("chamber" if cd.group("chamber") else "stated") if cdate else None
        for k, m in enumerate(chain):
            last = k == len(chain) - 1
            end = h.end() if h and last else m.end()
            out.append(Mention(m.start(), end, m.group("name"), "act_title", num if last else None,
                               cdate if k == 0 else None, ckind if k == 0 else None))
            if _QUOTE_CHARS & set(m.group("name")):      # a nested title: «X»ги inside “… киритиш”ги
                a = m.start("name")
                out += [replace(x, start=a + x.start, end=a + x.end) for x in _quoted_mentions(m.group("name"), p)]
    return out


_OUTER_CLOSE = re.compile(r"[»”\"]\s*[-‐]?\s*(?:ги|gi)\b")


def _outer_unquoted(text: str, p: dict, ms: list[Mention]) -> list[Mention]:
    """Ruling R29 A: an amending-law title whose opening quote is missing ("… «X»ги Қонунига ўзгартиш
    киритиш ҳақида»ги ЎРҚ–733-сонли Қонуни", repeal items of QONUNLAR-475d92b7 / d0d0b450 / eca1044a /
    f79c702c) is one mention of the outer law, [start of «X», end of its head noun], named by the text up to
    the unmatched closing quote, with the number and the adoption date written before «X»; «X» stays a
    nested mention (as inside a properly quoted outer title). Only with a registration number
    ("N-сонли") and a law / code head right after the closing quote."""
    out = list(ms)
    for m in ms:
        if m.kind != "act_title" or any(o.start < m.start and m.end <= o.end or o.start <= m.start and m.end < o.end
                                        for o in ms if o is not m):
            continue                                     # nested in (or containing) another quoted title
        eol = text.find("\n", m.end)
        oc = _OUTER_CLOSE.search(text, m.end, min(len(text) if eol < 0 else eol, m.end + 400))
        if oc is None or _QUOTE_CHARS & set(text[m.end:oc.start()]):
            continue
        h = p["act_head"].match(text, oc.end())
        if h is None or any(p["act_head_block"].match(w) for w in h.group("gap").split()):
            continue
        n = p["act_number"].search(h.group("gap"))
        if n is None:
            continue
        out.remove(m)
        out.append(replace(m, date=None, date_kind=None))
        out.append(Mention(m.start, h.end(), text[m.start:oc.start()], "act_title", _norm_number(n.group(0)),
                           m.date, m.date_kind))
    return out


def _norm_number(s: str) -> str:
    """"ЎРҚ–850" / "OʻRQ-850" -> "oʼrq-850", "310–II" -> "310-ii"."""
    return skeleton_key(s, False).replace(" ", "-")


def _code_mentions(text: str, s: dict) -> list[Mention]:
    """Match backwards from each кодекс/kodeks against the code_names lexicon (longest name
    first); a list "Жиноят ҳамда Жиноят-ижроия кодексларига" yields every member. Unknown names
    ("Biroq kodeksda", "ikkita kodeks") yield nothing."""
    p, lex, mw = s["p"], s["lex"], s["max_words"]
    out = []
    for h in p["code_head"].finditer(text):
        head = h.group(0)
        sing = "кодекси" if head[0] in "кК" else "kodeksi"
        toks = [(t.group(0), t.start(), t.end()) for t in _TOKEN.finditer(text, max(0, h.start() - 300), h.start())]
        found = []                                  # (start, end, canonical) right to left
        pos = len(toks)
        while pos > 0:
            hit = None
            for n in range(min(mw, pos), 0, -1):
                words = toks[pos - n:pos]
                if any(w[0] in ",;" for w in words) or _NOT_FIRST.match(skeleton_key(words[0][0], False)):
                    continue
                k = act_key(" ".join(w[0] for w in words) + " кодекс")
                name = lex.get(k)
                if name:
                    # tight span: drop leading words act_key ignores ("билан тасдиқланган Ўзбекистон
                    # Республикасининг Солиқ кодекси", "– Soliq", "Yangi tahrirdagi Soliq")
                    # (the shortest suffix of the n-gram with the same key)
                    words = next(words[j:] for j in range(len(words) - 1, -1, -1)
                                 if j == 0 or act_key(" ".join(w[0] for w in words[j:]) + " кодекс") == k)
                    hit = (words[0][1], words[-1][2], name, " ".join(w[0] for w in words), len(words))
                    break
            if not hit:
                break
            found.append(hit)
            pos -= hit[4]
            # another member only after a list separator (", " / "ва" / "ҳамда"), e.g. "Жиноят, "
            if pos > 0 and p["code_list_sep"].match(toks[pos - 1][0]):
                pos -= 1
                if pos > 0 and toks[pos - 1][0] == "," :
                    pos -= 1
            else:
                break
        if not found:
            continue
        kind = "code_list" if len(found) > 1 else "code_name"
        for k, (a, b, canon, surface, _) in enumerate(reversed(found)):
            end = h.end() if k == len(found) - 1 else b
            alias = act_key(surface + " кодекс") != act_key(canon + " кодекс")
            nm = (canon + " кодекси") if alias else f"{surface} {sing}"
            out.append(Mention(a, end, nm, kind))
    return out


def scan_act_mentions(text: str) -> list[Mention]:
    """Raw mentions of laws / codes / the Constitution, sorted by start. Quoted titles win over
    code names and the Constitution inside them; only a title nested in another quoted title
    overlaps it."""
    s = _scan_cfg()
    p = s["p"]
    out = _outer_unquoted(text, p, _quoted_mentions(text, p))
    taken = [(m.start, m.end) for m in out]

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for m in _code_mentions(text, s):
        if free(m.start, m.end):
            out.append(m)
    taken = [(m.start, m.end) for m in out]
    for m in p["constitution"].finditer(text):
        if free(m.start(), m.end()):
            out.append(Mention(m.start(), m.end(), m.group("name"), "constitution"))
    return sorted(out, key=lambda x: (x.start, x.end, x.name))


# --- registry --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ActRef:
    key: str
    doc_id: str | None          # corpus Document vertex (None: external, ambiguous or unlinked)
    external: bool              # an EXT/<key> stub
    display: str
    candidates: tuple = ()      # feasible corpus doc_ids when ambiguous
    edition: str | None = None  # external only: number / adoption date telling an older edition or an
                                # omnibus law apart from a same-key act -> EXT/<key>#<edition>
    number: str | None = None   # the cited registration number, if any

    @property
    def node_id(self) -> str | None:
        if self.doc_id:
            return self.doc_id
        if not self.external:
            return None
        return EXT_PREFIX + self.key + (f"#{self.edition}" if self.edition else "")

    @property
    def ambiguous(self) -> bool:
        """Several feasible corpus acts: for the LLM ref_resolve pass."""
        return self.doc_id is None and not self.external and bool(self.candidates)

    @property
    def unlinked(self) -> bool:
        """A subject-less omnibus title with no feasible corpus act: no vertex, no queue entry."""
        return self.doc_id is None and not self.external and not self.candidates

    @property
    def status(self) -> str:
        return ("corpus" if self.doc_id else "external" if self.external
                else "ambiguous" if self.candidates else "unlinked")


@dataclass
class Act:
    key: str
    head: str                                         # doc_id of the act vertex (latest edition)
    title: str
    date: str | None
    versions: list[str] = field(default_factory=list)  # all doc_ids, oldest first (head last)
    dates: list[str] = field(default_factory=list)     # adoption dates of all versions


GENERIC_MAX_GAP_DAYS = 60     # Senate resolution / stenogram vs. chamber adoption of an omnibus law
# a committee conclusion can precede the adoption date in the law text by a few days
# (XULOSALAR-94be3010 2025-07-11 vs QONUNLAR-ece0cec3 2025-07-15); titled acts only
DATE_GRACE_DAYS = 30
# a cited adoption date ("2023 йил 3 июлда қабул қилинган … ЎРҚ-850") is the enactment date; the
# corpus law text carries the earlier Legislative Chamber date -> feasible if <= 365 days before
CITED_MAX_GAP_DAYS = 365

_MONTHS = {  # uz-Cyrl, uz-Latn, ru month stems -> number
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "май": 5, "мая": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6, "iyul": 7, "avgust": 8,
    "sentabr": 9, "sentyabr": 9, "oktabr": 10, "oktyabr": 10, "noyabr": 11, "dekabr": 12,
}
# Evidence that a record IS a law text (for records M1 typed otherwise): the adoption line in its
# own header (config/graph.yaml act_titles.adopted, case-insensitive) together with >= 3 own articles.
_ADOPTED = _title_rx("adopted")
_UZ_DATE = re.compile(r"(\d{4})\s*[-–]?\s*(?:йил|yil)\w*\s+(\d{1,2})\s*[-–]?\s*(\w+)", re.I)
_RU_DATE = re.compile(r"(\d{1,2})\s+(\w+)\s+(\d{4})", re.I)
_LAW_MIN_ARTICLES = 3


def _parse_date(s: str) -> str | None:
    m = _UZ_DATE.search(s)
    y, d, mon = (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)
    if not m:
        m = _RU_DATE.search(s)
        if not m:
            return None
        d, mon, y = m.group(1), m.group(2), m.group(3)
    mon = mon.lower()
    num = next((v for k, v in _MONTHS.items() if mon.startswith(k)), None)
    if not num:
        return None
    try:
        return _dt.date(int(y), num, int(d)).isoformat()
    except ValueError:
        return None


def _adoption(rec) -> re.Match | None:
    return _ADOPTED.search(rec.text[:3000])


def is_law_text(rec) -> bool:
    """meta.type == "law_text", or (for mistyped records) an own adoption header + >= 3 articles."""
    if rec.meta.type == "law_text":
        return True
    arts = sum(1 for n in rec.structure if n.type == "статья")
    return arts >= _LAW_MIN_ARTICLES and _adoption(rec) is not None


def law_date(rec) -> str | None:
    if rec.meta.date:
        return rec.meta.date
    m = _adoption(rec)
    return _parse_date(m.group("a") or m.group("b") or "") if m else None


def record_date(rec) -> str | None:
    """meta.date, else the first adoption date its text states ("Qonunchilik palatasi tomonidan
    2025-yil 17-iyunda qabul qilingan"), for undated records."""
    if rec.meta.date:
        return rec.meta.date
    m = _ADOPTED.search(rec.text)
    return _parse_date(m.group("a") or m.group("b") or "") if m else None


def _days(a: str, b: str) -> int:
    try:
        return (_dt.date.fromisoformat(b) - _dt.date.fromisoformat(a)).days
    except ValueError:
        return 10 ** 6


def _title_span(rec) -> Evidence:
    """Where the record's own title sits in its text (whitespace-tolerant); fallback: 1st paragraph."""
    text, title = rec.text, re.sub(r"\s+", " ", rec.meta.title or "").strip()
    words = title.split()
    head = text[:6000]
    for drop in range(0, max(1, len(words) // 2)):
        ws = words[drop:]
        if not ws:
            break
        m = re.search(r"\s+".join(re.escape(w) for w in ws), head)
        if m:
            return Evidence(rec.doc_id, m.start(), m.end(), m.group(0))
    end = text.find("\n\n")
    end = len(text) if end < 0 else end
    return Evidence(rec.doc_id, 0, end, text[:end])


def _articles(rec) -> set[str]:
    return {re.sub(r"\s+", " ", n.title).strip().lower() for n in rec.structure if n.type == "статья" and n.title}


def _n_articles(rec) -> int:
    return sum(1 for n in rec.structure if n.type == "статья")


def _same_act(a, b) -> bool:
    """Guard against two standing laws with one title that are not editions of each other."""
    x, y = _articles(a), _articles(b)
    if not x or not y:
        return True
    return len(x & y) / min(len(x), len(y)) >= 0.3


class ActRegistry:
    def __init__(self):
        self.acts: dict[str, list[Act]] = {}           # act_key -> acts (several only for amending laws)
        self.doc_act: dict[str, Act] = {}              # every corpus law doc_id -> its act
        self.aliases: dict[str, str] = {}              # extra key -> key ("Сув кодекси" -> "…тасдиқлаш…")
        self.version_edges: list[GraphEdge] = []
        self.ambiguous: list[dict] = []
        self._amb_seen: set = set()
        self._doc_nodes: dict[str, GraphNode] = {}
        self._ext_forms: dict[str, Counter] = defaultdict(Counter)
        self._n_art: dict[str, int] = {}
        self._status: dict[str, str] = {}
        self._ext_numbers: dict[tuple, set] = defaultdict(set)
        self.known_keys: frozenset = frozenset()     # keys of all quoted titles / code names (from_records)
        self.amending_split: frozenset = frozenset() # Ruling R12: amending titles shared by several laws

    # construction
    @classmethod
    def from_records(cls, records) -> "ActRegistry":
        reg = cls()
        laws = [r for r in records if r.meta.title and is_law_text(r)]
        ru = [r for r in laws if r.meta.lang == ["ru"]]
        groups: dict[str, list] = defaultdict(list)
        for r in laws:
            if r in ru:
                continue
            k = act_key(r.meta.title)
            if k:
                groups[k].append(r)
        for k in sorted(groups):
            recs = sorted(groups[k], key=lambda r: (law_date(r) or "", r.doc_id))
            if is_amending(k):
                clusters = [[r] for r in recs]         # distinct acts, never merged
            else:
                clusters = []
                for r in recs:                         # editions unless their articles disagree
                    for c in clusters:
                        if _same_act(c[-1], r):
                            c.append(r)
                            break
                    else:
                        clusters.append([r])
            for c in clusters:
                head = c[-1]
                act = Act(k, head.doc_id, head.meta.title, law_date(head), [r.doc_id for r in c],
                          sorted({law_date(r) for r in c if law_date(r)}))
                reg.acts.setdefault(k, []).append(act)
                for r in c:
                    reg._add_doc(r, act)
                for r in c[:-1]:
                    reg.version_edges.append(GraphEdge(
                        r.doc_id, head.doc_id, "REFERS_TO",
                        {"evidence": _title_span(r).to_json(), "relation": "version_of"}))
            m = _CODE_APPROVAL.match(k)
            if m and m.group(1) not in groups:
                reg.aliases[m.group(1)] = k
        # every quoted act title / code name of the corpus (order-independent recovery of unquoted titles)
        reg.known_keys = frozenset(reg.aliases.get(k, k) for r in records
                                   for k in (act_key(m.name) for m in scan_act_mentions(r.text)
                                             if m.kind in ("act_title", "code_name", "code_list")) if k)
        # Ruling R12: an external amending title that the corpus cites with >= 2 distinct registration
        # numbers or adoption dates names several laws ("…шунингдек … ўз кучини йўқотган деб топиш",
        # ЎРҚ–355 / 372 / 446): its mentions get an edition by number / date (EXT/<key>#<edition>).
        # Other amending titles keep one vertex, so dated and undated mentions of one law do not split.
        seen: dict[str, tuple[set, set]] = defaultdict(lambda: (set(), set()))
        for r in records:
            for m in scan_act_mentions(r.text):
                if m.kind != "act_title" or not (m.number or m.date):
                    continue
                k = act_key(m.name)
                k = reg.aliases.get(k, k)
                if k and is_amending(k) and not is_generic(k) and k not in reg.acts:
                    if m.number:
                        seen[k][0].add(m.number)
                    if m.date:
                        seen[k][1].add(m.date)
        reg.amending_split = frozenset(k for k, (ns, ds) in seen.items() if len(ns) > 1 or len(ds) > 1)
        for r in ru:                                    # Russian editions: translation_of the uz act
            target = reg._translation_target(r)
            if target is None:
                continue
            reg._add_doc(r, target)
            reg.version_edges.append(GraphEdge(
                r.doc_id, target.head, "REFERS_TO",
                {"evidence": _title_span(r).to_json(), "relation": "translation_of"}))
            k = act_key(r.meta.title)
            if k and k not in reg.acts:
                reg.aliases[k] = target.key
        return reg

    def _add_doc(self, r, act: Act) -> None:
        self.doc_act[r.doc_id] = act
        self._n_art[r.doc_id] = _n_articles(r)
        self._status[r.doc_id] = r.meta.status
        self._doc_nodes[r.doc_id] = GraphNode(
            r.doc_id, "document", DOC_TYPE, r.meta.title, r.doc_id, False,
            {"act_key": act.key, "date": law_date(r), "act_head": act.head})

    def _translation_target(self, r) -> Act | None:
        """The one uz act adopted the same day whose article count is within 10% of r's."""
        d, n = law_date(r), _n_articles(r)
        if not d or not n:
            return None
        hits = [a for v in self.acts.values() for a in v if a.date == d]
        close = [a for a in hits if abs(self._n_art.get(a.head, 0) - n) <= max(1, n // 10)]
        return close[0] if len(close) == 1 else None

    # resolution (pure: no registry state changes)
    def resolve(self, title_or_name: str, date_hint: str | None = None, *, number: str | None = None,
                cited_date: str | None = None, cited_kind: str | None = None,
                src_doc: str | None = None) -> ActRef:
        """`date_hint`: the citing record's date. `number` / `cited_date`: the cited act's own
        registration number and adoption date written in the text (Mention.number / .date).
        `src_doc`: the citing record, which never resolves to itself."""
        surface = title_or_name or ""
        key = act_key(surface)
        if not key:
            raise ValueError(f"no act key for {title_or_name!r}")
        key = self.aliases.get(key, key)
        m = _CODE_APPROVAL.match(key)
        disp = display_name(surface)
        if m and key not in self.acts:                  # "X кодексини тасдиқлаш тўғрисида" = the code X
            key = self.aliases.get(m.group(1), m.group(1))
            c = re.search(r"(?i)^(.*?(?:кодекс|kodeks))", surface)
            disp = display_name(c.group(1)) if c else disp
        acts = self.acts.get(key) or []
        generic = is_generic(key)
        cand = acts
        if src_doc and any(src_doc in a.versions for a in acts):
            # a law quoting its own title: itself (no edge, no stub) unless the text marks an older
            # edition by a number / adoption date ("2012 йил 6 январда қабул қилинган «Рақобат
            # тўғрисида»ги ЎРҚ–319-сонли Қонуни")
            cand = [a for a in cand if src_doc not in a.versions]
            if not (number or cited_date):
                return ActRef(key, None, False, disp, ())
        if number:                                      # a numbered citation names an ENACTED act; corpus
            cand = [a for a in cand if self._status.get(a.head) != "проект"]   # drafts carry no number
        hint, grace = date_hint, DATE_GRACE_DAYS
        if cited_date and cited_kind == "chamber":      # the Legislative Chamber's adoption date
            if generic:     # a subject-less title: only the exact Chamber date identifies the act
                cand = [a for a in cand if cited_date in a.dates]
            else:           # a titled act: Senate papers and the law text may differ by a few days
                cand = [a for a in cand if any(abs(_days(d, cited_date)) <= DATE_GRACE_DAYS for d in a.dates)]
            hint, grace = cited_date, DATE_GRACE_DAYS
        elif cited_date:                                # an enactment / other stated date: strict
            win = GENERIC_MAX_GAP_DAYS if generic else CITED_MAX_GAP_DAYS
            cand = [a for a in cand if any(0 <= _days(d, cited_date) <= win for d in a.dates)]
            hint, grace = cited_date, 0
        if cand:
            if len(cand) == 1 and not is_amending(key) and not cited_date:
                return ActRef(key, cand[0].head, False, cand[0].title)   # a standing law: any edition
            feasible, when = cand, {}
            if hint:
                for a in cand:          # the version date that fits the hint best
                    if generic:         # a subject-less title: adopted by the mention date, not long before
                        ok = [d for d in a.dates if 0 <= _days(d, hint) <= GENERIC_MAX_GAP_DAYS]
                    else:               # a titled act: adopted by the mention date (+ grace for early drafts)
                        ok = [d for d in a.dates if _days(d, hint) >= -grace]
                    if ok:
                        before = [d for d in ok if d <= hint]
                        when[a.head] = max(before) if before else min(ok)
                feasible = [a for a in cand if a.head in when]
                if len(feasible) > 1:                   # nearest date wins; a same-day tie stays
                    before = [when[a.head] for a in feasible if when[a.head] <= hint]
                    best = max(before) if before else min(when[a.head] for a in feasible)
                    feasible = [a for a in feasible if when[a.head] == best]
            if len(feasible) == 1:
                return ActRef(key, feasible[0].head, False, feasible[0].title)
            if feasible:
                return ActRef(key, None, False, disp, tuple(a.head for a in feasible))
        # no feasible corpus act. A same-key corpus act exists (or the title is subject-less): the
        # cited act is another edition / omnibus law, told apart by its number or adoption date
        # Ruling R12 (Task 5): distinct amending laws with one title ("«X»ги Қонунга ўзгартиш киритиш
        # ҳақида", "…шунингдек … ўз кучини йўқотган деб топиш") are told apart by number / date too
        edition = (number or cited_date) if (acts or generic or key in self.amending_split) else None
        if generic and not edition:                     # unlinked: no vertex, no queue entry
            return ActRef(key, None, False, disp, ())
        return ActRef(key, None, True, disp, (), edition, number)

    def note_mention(self, ref: ActRef, title: str | None = None, date_hint: str | None = None) -> None:
        """Record a resolved mention: external stubs count their display forms, ambiguous ones
        join the `ambiguous` queue (once per (title, date_hint))."""
        if ref.external:
            self._ext_forms[(ref.key, ref.edition)][ref.display] += 1
            if ref.number:
                self._ext_numbers[(ref.key, ref.edition)].add(ref.number)
        elif ref.ambiguous:
            tag = (title if title is not None else ref.display, date_hint)
            if tag not in self._amb_seen:
                self._amb_seen.add(tag)
                self.ambiguous.append({"title": tag[0], "key": ref.key, "date_hint": date_hint,
                                       "candidates": list(ref.candidates)})

    def unnote_mention(self, ref: ActRef) -> None:
        """Ruling R29 D: take back one note_mention(ref) of an external ref whose mention was then settled to a
        corpus act (ref_resolve fuzzy_title): its EXT vertex exists only while some mention stays external."""
        k = (ref.key, ref.edition)
        forms = self._ext_forms.get(k)
        if not forms or not ref.external or forms.get(ref.display, 0) <= 0:
            return
        forms[ref.display] -= 1
        if forms[ref.display] <= 0:
            del forms[ref.display]
        if not forms:
            del self._ext_forms[k]
            self._ext_numbers.pop(k, None)

    def resolve_mention(self, title_or_name, date_hint: str | None = None, *, src_doc: str | None = None,
                        number: str | None = None, cited_date: str | None = None,
                        cited_kind: str | None = None) -> ActRef:
        """resolve() + note_mention(); `title_or_name` may be a Mention (its number/date are used)."""
        if isinstance(title_or_name, Mention):
            m = title_or_name
            number, cited_date, cited_kind, title_or_name = m.number, m.date, m.date_kind, m.name
        ref = self.resolve(title_or_name, date_hint, number=number, cited_date=cited_date,
                           cited_kind=cited_kind, src_doc=src_doc)
        self.note_mention(ref, title_or_name, date_hint)
        return ref

    # graph output
    def external_nodes(self) -> list[GraphNode]:
        out = []
        for k, ed in sorted(self._ext_forms, key=lambda x: (x[0], x[1] or "")):
            forms = self._ext_forms[(k, ed)]
            best = sorted(forms.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            attrs = {"act_key": k, "mentions": sum(forms.values()), "amending": is_amending(k)}
            if ed:
                attrs["edition"] = ed
            nums = sorted(self._ext_numbers.get((k, ed), ()))
            if nums:
                attrs["numbers"] = nums
            out.append(GraphNode(EXT_PREFIX + k + (f"#{ed}" if ed else ""), "document", DOC_TYPE, best,
                                 None, True, attrs))
        return out

    def nodes(self) -> list[GraphNode]:
        return sorted(self._doc_nodes.values(), key=lambda n: n.node_id) + self.external_nodes()

    def edges(self) -> list[GraphEdge]:
        return sorted(self.version_edges, key=lambda e: (e.src, e.dst))

    def validate(self, schema: Schema | None = None) -> list[str]:
        schema = schema or Schema.load()
        nodes = {n.node_id: n for n in self.nodes()}
        errs = [m for n in nodes.values() for m in schema.validate_node(n)]
        return errs + [m for e in self.edges() for m in schema.validate_edge(e, nodes)]

    def stats(self) -> dict:
        acts = [a for v in self.acts.values() for a in v]
        return {"corpus_law_docs": len(self.doc_act), "corpus_acts": len(acts),
                "edition_groups": [a.versions for a in acts if len(a.versions) > 1],
                "same_key_distinct": {k: [a.head for a in v] for k, v in self.acts.items() if len(v) > 1},
                "external_acts": len(self._ext_forms), "ambiguous": len(self.ambiguous)}


def corpus_mentions(records, reg: ActRegistry) -> Counter:
    """Raw mention count per resolved node id over all records (external stubs are EXT/<key>;
    AMBIG/<key> and UNLINKED/<key> for the rest). Notes every mention in `reg`."""
    c: Counter = Counter()
    for r in records:
        for m, ref in resolve_record(r, reg):
            c[ref.node_id or f"{'AMBIG' if ref.ambiguous else 'UNLINKED'}/{ref.key}"] += 1
    return c


MAX_REF_CANDIDATES = 5       # ref_resolve (Task 4) gets at most this many candidates


def _ref_dates(reg: ActRegistry, ref: ActRef) -> list[str] | None:
    """Adoption dates of the act `ref` names (None: unknown, e.g. a #number edition stub)."""
    if ref.doc_id:
        act = reg.doc_act.get(ref.doc_id)
        return list(act.dates) if act and act.dates else None
    if ref.edition and _DATE_ISO.match(ref.edition):
        return [ref.edition]
    return None


_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# an undated omnibus mention follows a dated one of its record / agenda item when the record is at
# most this long after the act (a Senate conclusion can come > 60 days after the Chamber adoption:
# XULOSALAR-42616e31, 2025-07-08, on the law of 2025-05-06)
GENERIC_FOLLOW_MAX_DAYS = 180


def date_compatible(reg: ActRegistry, ref: ActRef, date_hint: str | None) -> bool:
    """Whether an undated mention in a record dated `date_hint` may name the act `ref` names:
    adopted by date_hint (+ DATE_GRACE_DAYS), and for a subject-less omnibus title at most
    GENERIC_FOLLOW_MAX_DAYS before it. Unknown dates on either side are compatible."""
    dates = _ref_dates(reg, ref)
    if not date_hint or not dates:
        return True
    if is_generic(ref.key):
        return any(-DATE_GRACE_DAYS <= _days(t, date_hint) <= GENERIC_FOLLOW_MAX_DAYS for t in dates)
    return any(_days(t, date_hint) >= -DATE_GRACE_DAYS for t in dates)


def _scopes(rec, ms) -> list[str]:
    """Consistency scope of each mention: its stenogram agenda item (вопрос), else the record."""
    qs = sorted((n.start, n.end, n.node_id) for n in rec.structure if n.type == "вопрос")
    out = []
    for m in ms:
        out.append(next((nid for a, b, nid in qs if a <= m.start < b), ""))
    return out


def omnibus_candidates(reg: ActRegistry, key: str, date_hint: str | None) -> tuple:
    """Corpus acts an undated subject-less omnibus mention may name (Ruling R7): adopted within
    [date_hint - GENERIC_MAX_GAP_DAYS, date_hint + DATE_GRACE_DAYS], nearest first, at most
    MAX_REF_CANDIDATES; () without a date hint or with more feasible acts than that."""
    if not date_hint:
        return ()
    scored = []
    for a in reg.acts.get(key) or []:
        gaps = [abs(_days(t, date_hint)) for t in a.dates
                if -DATE_GRACE_DAYS <= _days(t, date_hint) <= GENERIC_MAX_GAP_DAYS]
        if gaps:
            scored.append((min(gaps), a.head))
    if len(scored) > MAX_REF_CANDIDATES:
        return ()
    return tuple(h for _, h in sorted(scored))


def r7_undated(reg: ActRegistry, ref: ActRef, date_hint: str | None) -> ActRef:
    """Ruling R7 for an undated, unnumbered mention that no dated one settles: a subject-less
    omnibus title never takes the nearest-dated corpus act; it becomes ambiguous over the feasible
    corpus acts (omnibus_candidates), or unlinked. Other refs are returned unchanged."""
    if is_generic(ref.key) and (ref.doc_id or ref.ambiguous):
        return ActRef(ref.key, None, False, ref.display, omnibus_candidates(reg, ref.key, date_hint))
    return ref


def nested_flags(ms: list[Mention]) -> list[bool]:
    """Whether each mention lies strictly inside another mention's span: a title quoted in an outer title
    («…«A»ги Қонунига … ҳақида»); scan_act_mentions lets only such nested titles overlap."""
    return [any(o is not m and o.start <= m.start and m.end <= o.end and (o.start, o.end) != (m.start, m.end)
                for o in ms) for m in ms]


def follow_undated(ms: list[Mention], refs: list[ActRef], scopes: list[str], nested: list[bool],
                   compatible) -> list[ActRef | None]:
    """Same-law consistency (resolve_record): the ref each undated, unnumbered act_title mention takes from the
    dated act_title mentions of the same (scope, title key, nesting role), or None when it does not follow.
    It follows only when those dated mentions all name one node and `compatible(that ref)` holds; the returned
    ref keeps the mention's display."""
    dated: dict[tuple, set] = defaultdict(set)
    by_node: dict[tuple, ActRef] = {}
    for m, ref, sc, nz in zip(ms, refs, scopes, nested):
        if m.date and m.kind == "act_title":
            dated[(sc, ref.key, nz)].add(ref.node_id)
            by_node[(ref.key, ref.node_id)] = ref
    out: list[ActRef | None] = []
    for m, ref, sc, nz in zip(ms, refs, scopes, nested):
        nodes = dated.get((sc, ref.key, nz))
        if not m.date and not m.number and m.kind == "act_title" and nodes and len(nodes) == 1:
            (nid,) = nodes
            if nid is not None and compatible(by_node[(ref.key, nid)]):
                out.append(ref if ref.node_id == nid else replace(by_node[(ref.key, nid)], display=ref.display))
                continue
        out.append(None)
    return out


def resolve_record(rec, reg: ActRegistry) -> list[tuple[Mention, ActRef]]:
    """Resolve and note every mention of one record.

    Same-law consistency: when the dated mentions (Mention.date) of one title key in this record
    (in a stenogram: in one agenda item) all resolve to the same node, the undated, unnumbered
    mentions of that key there follow them (a Senate resolution names the law in its title without
    a date and again, dated, in its body) -- only when that node's adoption date is compatible
    with the record date (date_compatible). Consistency is keyed on (scope, title key, nesting role) —
    M3 Task 2: a title quoted inside an outer amending title never follows (nor anchors) a free mention.

    Ruling R7 (Task 4): an undated, unnumbered subject-less omnibus mention ("айрим қонун
    ҳужжатларига … киритиш тўғрисида") that no dated mention settles never takes the nearest-dated
    corpus act: it becomes ambiguous with the feasible corpus acts as candidates (for the LLM
    ref_resolve pass), or unlinked when none (or too many) are feasible."""
    d = record_date(rec)
    ms = scan_act_mentions(rec.text)
    refs = [reg.resolve(m.name, d, number=m.number, cited_date=m.date, cited_kind=m.date_kind,
                        src_doc=rec.doc_id) for m in ms]
    scope = _scopes(rec, ms)
    follow = follow_undated(ms, refs, scope, nested_flags(ms), lambda ref: date_compatible(reg, ref, d))
    out = []
    for m, ref, f in zip(ms, refs, follow):
        undated = not m.date and not m.number and m.kind == "act_title"
        if f is not None:
            ref = f
        elif undated:
            ref = r7_undated(reg, ref, d)
        reg.note_mention(ref, m.name, d)
        out.append((m, ref))
    return out


def main(argv=None) -> int:
    from graphrag.data.ingest import read_records
    recs = read_records(config.path("data/records"))
    reg = ActRegistry.from_records(recs)
    counts = corpus_mentions(recs, reg)
    st = reg.stats()
    print(f"law_text docs: {st['corpus_law_docs']}  corpus acts: {st['corpus_acts']}  "
          f"edition groups: {len(st['edition_groups'])}  external acts: {st['external_acts']}  "
          f"ambiguous mentions (distinct): {st['ambiguous']}")
    by_id = {a.head: a for v in reg.acts.values() for a in v}
    for g in st["edition_groups"]:
        print("  editions:", [(d, reg._doc_nodes[d].attrs["date"]) for d in g], "|", by_id[g[-1]].title)
    for k, v in st["same_key_distinct"].items():
        print(f"  same key, distinct acts ({len(v)}): {k}")
    errs = reg.validate()
    print("schema errors:", len(errs), errs[:5])
    ext = {n.node_id: n for n in reg.external_nodes()}
    n_top = int(argv[0]) if argv else 30
    print(f"\ntop-{n_top} external acts by raw mentions:")
    for nid, n in sorted(((k, v) for k, v in counts.items() if k in ext), key=lambda kv: (-kv[1], kv[0]))[:n_top]:
        print(f"{n:5d}  {ext[nid].title}  ->  {nid}")
    corp = sum(v for k, v in counts.items() if k in reg.doc_act)
    amb = sum(v for k, v in counts.items() if k.startswith("AMBIG/"))
    unl = sum(v for k, v in counts.items() if k.startswith("UNLINKED/"))
    print(f"\nmentions: total {sum(counts.values())}, corpus {corp}, ambiguous {amb}, unlinked {unl}, "
          f"external {sum(v for k, v in counts.items() if k in ext)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
