"""Structure parsing (spec §2.3): paragraphs → tree of structural nodes with char spans.

The document text is "\\n\\n".join(paragraph texts). Node ids are built from the path
(<doc_id>/ГЛ.3/СТ.21/Ч.2/П.3), never from a global counter, so a rebuild keeps ids.
Parsers by document kind:
  law        глава → статья → часть → пункт → подпункт   (+ преамбула, реквизиты)
  resolution преамбула → пункт → подпункт                (+ реквизиты)
  stenogram  заголовок, вопрос (повестки) → выступление
  prose      заголовок → блок                            (+ реквизиты)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from graphrag.data.extract import Para
from graphrag.data.normalize import canon_num
from graphrag.types import Node

CODE = {"глава": "ГЛ", "статья": "СТ", "часть": "Ч", "пункт": "П", "подпункт": "ПП",
        "преамбула": "ПРЕАМБ", "реквизиты": "РЕКВ", "заголовок": "ЗАГ", "блок": "БЛ",
        "вопрос": "ВОПР", "выступление": "ВЫСТ", "параграф": "§"}

_SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_ART = re.compile(rf"^(?P<num>\d+[{_SUPS}]*)\s*[-–]\s*(?:модда|modda)\.\s*(?P<rest>.*)$",
                  re.IGNORECASE | re.DOTALL)
_ART_RU = re.compile(rf"^Статья\s+(?P<num>\d+[{_SUPS}]*)\.\s*(?P<rest>.*)$", re.DOTALL)
_CH = re.compile(r"^(?P<num>\d+|[IVXLC]+)\s*[-–]\s*(?:боб|bob)\.?\s*(?P<rest>.*)$",
                 re.IGNORECASE | re.DOTALL)
_CH_RU = re.compile(r"^Глава\s+(?P<num>\d+|[IVXLC]+)\.?\s*(?P<rest>.*)$", re.DOTALL)
_SECT = re.compile(r"^(?:(?P<num>\d+)\s*[-–]\s*§|§\s*(?P<num2>\d+))\.?\s*(?P<rest>.*)$", re.DOTALL)
_POINT = re.compile(rf"^(?P<num>\d+[{_SUPS}]*)\)\s+")
_POINT_DOT = re.compile(r"^(?P<num>\d+)\.\s+(?=\S)")
_SUB = re.compile(r"^(?P<num>[а-яўқғҳa-z][¹²³]?)\)\s+")
_REQ = re.compile(r"президент|раис|prezident|rais|тошкент|toshkent|№|ўрқ|oʼrq|сқ-|sq-|йил|yil|ql_",
                  re.IGNORECASE)
# a place / office marker: reference lines ("(2020 йил 5 май)", "ЎРҚ-620-сон") alone are no requisites
_REQ_STRONG = re.compile(r"президент|раис|prezident|rais|тошкент|toshkent", re.IGNORECASE)
_ROLE_LINE = re.compile(r"^(?:(?:qoʼmita|қўмита|komissiya|комиссия|kengash|кенгаш)\s+)?(?:raisi|раиси)\s*[.:]?$",
                        re.IGNORECASE)
_IMPERATIVE_END = re.compile(r"(?:син|sin|сын)\.?$", re.IGNORECASE)   # "… юклансин."
_SIGNER = re.compile(r"(?<!\w)[А-ЯЁЎҚҒҲA-Z]\.\s?[А-ЯЁЎҚҒҲA-Z][а-яёўқғҳʼ'a-z]+")   # "Ш. Мирзиёев"
MIN_LAW_ARTICLES = 3
_QUOTE_RESET = 40
_QUOTE_CAP = 400
_SELF_CLOSED = re.compile(r"»[.;,:]*$")


@dataclass
class DocText:
    paras: list[str]
    offsets: list[tuple[int, int]]
    text: str
    pages: list[int | None]          # PDF page of each paragraph (None for docx/md)
    page_ends: list[int | None] = field(default_factory=list)   # last page (page_end or page)

    @classmethod
    def build(cls, paras: list[Para]) -> "DocText":
        texts, offs, pos = [p.text for p in paras], [], 0
        for t in texts:
            offs.append((pos, pos + len(t)))
            pos += len(t) + 2
        return cls(texts, offs, "\n\n".join(texts), [p.page for p in paras],
                   [p.page_end or p.page for p in paras])

    def span(self, a: int, b: int) -> tuple[int, int]:
        """Char span of paragraphs [a, b)."""
        return self.offsets[a][0], self.offsets[b - 1][1]


class _Tree:
    def __init__(self, doc_id: str, dt: DocText, title: str):
        self.doc_id, self.dt = doc_id, dt
        self.nodes = [Node(doc_id, "документ", title or "", None, None, 0, len(dt.text))]
        self._ids = {doc_id}

    def add(self, parent_id: str, type_: str, number: str, title: str, a: int, b: int) -> str:
        seg = f"{CODE[type_]}.{number}"
        nid, k = f"{parent_id}/{seg}", 2
        while nid in self._ids:
            nid, k = f"{parent_id}/{seg}~{k}", k + 1
        self._ids.add(nid)
        s, e = self.dt.span(a, b)
        self.nodes.append(Node(nid, type_, title, number, parent_id, s, e))
        return nid


def _num_key(num: str) -> tuple[int, int]:
    m = re.fullmatch(rf"(\d+)([{_SUPS}]*)", num)
    sup = m[2].translate(str.maketrans(_SUPS, "0123456789"))
    return int(m[1]), int(sup) if sup else 0


def _quote_closes(P: list[str], i: int, depth: int) -> bool:
    """Whether the quote opened at paragraph i (depth after it) is closed later in the text."""
    for t in P[i + 1:]:
        depth = max(0, depth + t.count("«") - t.count("»"))
        if depth == 0:
            return True
    return False


def _is_header(t: str) -> bool:
    return bool(_ART.match(t) or _ART_RU.match(t) or _CH.match(t) or _CH_RU.match(t))


def _own_flags(P: list[str]) -> list[bool]:
    """True where a paragraph is outside « » quotes (quoted text = inserted/cited norms).

    * A paragraph opening at depth 0 with « and ending with » (+ optional .;,:) closes itself,
      whatever its « / » balance (RU typography: «… «О санации банков»; — one » closes both).
    * A quote closed later is honoured however long it is (inserted chapters run to dozens of
      paragraphs), up to _QUOTE_CAP paragraphs; an unclosed quote is dropped after
      _QUOTE_RESET paragraphs.
    * A non-« article header numbered exactly (last own article + 1) ends the quote when the
      quote is unclosed, or was not opened by a quoted header («N-модда. / «N-боб. / «Статья N),
      or was opened by a higher-numbered quoted article («41-модда. … → own 2-модда.).
      Known limitation: in the last case an inner lower-numbered article that happens to equal
      last+1 inside a closed quote opened by a higher article ends the quote early.
    """
    out, depth, opened, closes, q_hdr, q_art, last = [], 0, 0, True, False, None, (0, 0)
    for i, t in enumerate(P):
        m = _ART.match(t) or _ART_RU.match(t)
        if depth and m and _num_key(m["num"]) == (last[0] + 1, 0) and \
                (not closes or not q_hdr or (q_art is not None and _num_key(m["num"]) < q_art)):
            depth = 0
        if depth and (i - opened > _QUOTE_CAP or (not closes and i - opened > _QUOTE_RESET)):
            depth = 0
        own = depth == 0 and not t.startswith("«")
        out.append(own)
        if own and m and _num_key(m["num"]) > last:
            last = _num_key(m["num"])
        if depth == 0 and _SELF_CLOSED.search(t) and t.startswith("«"):
            new = 0
        else:
            new = max(0, depth + t.count("«") - t.count("»"))
        if depth == 0 and new > 0:
            opened, closes = i, _quote_closes(P, i, new)
            inner = t[1:] if t.startswith("«") else ""
            q_hdr = _is_header(inner)
            qm = _ART.match(inner) or _ART_RU.match(inner)
            q_art = _num_key(qm["num"]) if qm else None
        depth = new
    return out


def _continues(prev: str, t: str) -> bool:
    return t[:1].islower() or t[:1] in "–—-•" or prev.rstrip().endswith((":", ";", ","))


def _is_point(t: str) -> bool:
    return bool(_POINT.match(t) or _POINT_DOT.match(t) or _SUB.match(t))


def _req_line(t: str) -> bool:
    """Short line that may belong to the trailing requisites block.

    Up to 7 words; or up to 12 words when it is a joined signature-table row ("cell | cell")
    or carries >= 2 distinct requisite markers ("Тошкент шаҳри, 2024 йил 19 январь, № СҚ-851").
    """
    if len(t) > 80 or _is_point(t):
        return False
    w = len(t.split())
    return w <= 7 or (w <= 12 and (" | " in t or len({m.lower() for m in _REQ.findall(t)}) >= 2))


def _requisites_start(P: list[str], lo: int) -> int | None:
    j = len(P)
    while j > lo and _req_line(P[j - 1]):
        j -= 1
    run = P[j:]
    hits = sum(bool(_REQ.search(t) or _SIGNER.search(t)) for t in run)
    signed = any(_SIGNER.search(t) for t in run)
    if len(run) == 1:                                # one "Раиси | Т. Нарбаева" row
        if _ROLE_LINE.match(run[0]):                 # a bare closing "Qoʼmita raisi" (xulosa, no signer)
            return j
        rest = _SIGNER.sub("", run[0]).rstrip()      # the signer's surname may end in -sin ("А. Ясин")
        ok = _SIGNER.search(run[0]) and _REQ.search(run[0]) and not _IMPERATIVE_END.search(rest)
        return j if ok else None
    strong = any(_REQ_STRONG.search(t) for t in run)
    return j if len(run) >= 2 and (signed or (hits >= 2 and strong)) else None


def _extend(part: dict, b: int) -> None:
    part["b"] = b
    if part["points"]:
        pt = part["points"][-1]
        pt["b"] = b
        if pt["subs"]:
            pt["subs"][-1]["b"] = b


def _article_body(tree: _Tree, aid: str, P: list[str], own: list[bool], a: int, b: int) -> None:
    # Two-level numbering: when the article has own "N." points, "N)" paragraphs after an
    # "N." point are its subpoints; a letter "а)" under such a subpoint stays inside it.
    # An "N)" before the article's first "N." point is body text when an "N." point with the same
    # number follows (M2 sweep): as a point it would collide with it (a "~2" id).
    dots = [j for j in range(a, b) if own[j] and _POINT_DOT.match(P[j]) and not _POINT.match(P[j])]
    two_level = bool(dots)
    dot_nums = {canon_num(_POINT_DOT.match(P[j])["num"]) for j in dots}
    parts: list[dict] = []
    for j in range(a, b):
        t = P[j]
        pm = (_POINT.match(t) or _POINT_DOT.match(t)) if own[j] else None
        if pm and two_level and _POINT.match(t) and j < dots[0] and canon_num(pm["num"]) in dot_nums:
            pm = None
        sm = _SUB.match(t) if own[j] else None
        cur = parts[-1]["points"][-1] if parts and parts[-1]["points"] else None
        if pm and two_level and _POINT.match(t) and cur and cur["dot"]:
            cur["subs"].append({"num": canon_num(pm["num"]), "a": j, "b": j, "numeric": True})
        elif pm:
            if not parts:
                parts.append({"a": j, "b": j, "points": []})
            parts[-1]["points"].append({"num": canon_num(pm["num"]), "a": j, "b": j, "subs": [],
                                        "dot": not _POINT.match(t)})
        elif sm and cur and cur["subs"] and cur["subs"][-1].get("numeric"):
            pass                                             # а) inside an "N)" subpoint
        elif sm and cur:
            cur["subs"].append({"num": canon_num(sm["num"]), "a": j, "b": j})
        elif parts and (not own[j] or _continues(P[j - 1], t)):
            pass
        else:
            parts.append({"a": j, "b": j, "points": []})
        _extend(parts[-1], j + 1)
    for k, part in enumerate(parts, 1):
        pid = tree.add(aid, "часть", str(k), "", part["a"], part["b"])
        for pt in part["points"]:
            ptid = tree.add(pid, "пункт", pt["num"], "", pt["a"], pt["b"])
            for sp in pt["subs"]:
                tree.add(ptid, "подпункт", sp["num"], "", sp["a"], sp["b"])


def _article(tree: _Tree, parent: str, P: list[str], own: list[bool], i: int, stop: int,
             num: str, rest: str) -> None:
    title, body_a = rest, i + 1
    if rest.endswith((".", ":", ";")) or len(rest) > 150:
        title, body_a = "", i                                # header line is body text
    elif rest and body_a < stop and own[body_a] and len(P[body_a]) < 80 and P[body_a][:1].islower():
        title, body_a = f"{rest} {P[body_a]}", body_a + 1    # wrapped title
    elif not rest and body_a < stop and len(P[body_a]) < 120 and not P[body_a].endswith((".", ":", ";")):
        title, body_a = P[body_a], body_a + 1                # title on its own line
    aid = tree.add(parent, "статья", num, title, i, stop)
    if body_a < stop:
        _article_body(tree, aid, P, own, body_a, stop)


def _wrapped_title(P: list[str], own: list[bool], i: int, title: str) -> str:
    """Join continuation lines of a chapter / § title: own, short, lower-case start, while the
    title so far has no terminal punctuation (same rule as the article-title wrap)."""
    j = i + 1
    while (title and j < len(P) and own[j] and len(P[j]) < 80 and P[j][:1].islower()
           and not title.endswith((".", ":", ";"))):
        title, j = f"{title} {P[j]}", j + 1
    return title


def parse_law(doc_id: str, dt: DocText, title: str) -> list[Node] | None:
    P, n = dt.paras, len(dt.paras)
    own = _own_flags(P)
    heads: list[tuple[int, str, str, str]] = []
    last = (0, 0)
    for i, t in enumerate(P):
        if not own[i]:
            continue
        m = _CH.match(t) or _CH_RU.match(t)
        if m:
            heads.append((i, "глава", canon_num(m["num"]), _wrapped_title(P, own, i, m["rest"].strip())))
            continue
        m = _SECT.match(t)
        if m:
            heads.append((i, "параграф", m["num"] or m["num2"],
                          _wrapped_title(P, own, i, m["rest"].strip())))
            continue
        m = _ART.match(t) or _ART_RU.match(t)
        if m and _num_key(m["num"]) > last:
            heads.append((i, "статья", canon_num(m["num"]), m["rest"].strip()))
            last = _num_key(m["num"])
    arts = [h for h in heads if h[1] == "статья"]
    if len(arts) < MIN_LAW_ARTICLES:
        return None
    tree = _Tree(doc_id, dt, title)
    if heads[0][0] > 0:
        tree.add(doc_id, "преамбула", "1", "", 0, heads[0][0])
    tail = _requisites_start(P, arts[-1][0] + 1)
    end = n if tail is None else tail
    parent_of: dict[int, str] = {}
    chapter, sect = doc_id, None
    for k, (i, kind, num, ttl) in enumerate(heads):
        if kind == "глава":
            nxt = next((h[0] for h in heads[k + 1:] if h[1] == "глава"), end)
            chapter, sect = tree.add(doc_id, "глава", num, ttl, i, nxt), None
        elif kind == "параграф":                     # container between глава and статья
            nxt = next((h[0] for h in heads[k + 1:] if h[1] in ("глава", "параграф")), end)
            sect = tree.add(chapter, "параграф", num, ttl, i, nxt)
        else:
            parent_of[i] = sect or chapter
    starts = [h[0] for h in heads] + [end]
    for i, _, num, rest in arts:
        stop = next(s for s in starts if s > i)
        _article(tree, parent_of.get(i, doc_id), P, own, i, stop, num, rest)
    if tail is not None:
        tree.add(doc_id, "реквизиты", "1", "", tail, n)
    return tree.nodes


# ── resolution (qaror / postanovlenie) ───────────────────────────────────────
_DECIDE = re.compile(r"ҚАРОРҚИЛАДИ|QARORQILADI|ПОСТАНОВЛЯЕТ|ҚАРОРҚИЛИНАДИ|QARORQILINADI",
                     re.IGNORECASE)


def _words(t: str) -> int:
    return len(t.split())


def parse_resolution(doc_id: str, dt: DocText, title: str, max_tokens: int = 384,
                     count_tokens: Callable[[str], int] = _words) -> list[Node]:
    """Resolution; a second document glued after it (committee QARORI + XULOSASI in one file)
    is split off at its prose head and packed as prose (заголовок / блок / реквизиты)."""
    P, n = dt.paras, len(dt.paras)
    dec = next((i for i, t in enumerate(P) if _DECIDE.search(re.sub(r"\s+", "", t))), None)
    if dec is None and any(_is_prose_head(t) for t in P[:40]):
        return parse_prose(doc_id, dt, title, max_tokens, count_tokens)   # a xulosa titled "QARORI"
    tree = _Tree(doc_id, dt, title)
    a = 0
    if dec is not None:
        tree.add(doc_id, "преамбула", "1", "", 0, dec + 1)
        a = dec + 1
    head = next((j for j in range(a, n) if _is_prose_head(P[j])), None) if dec is not None else None
    second = sig_end = None
    if head is None:
        tail = _requisites_start(P, a)
        end = n if tail is None else tail
    else:
        # the resolution's own signature: first short requisite line after the last item start
        last = max((j for j in range(a, head) if _is_point(P[j])), default=a)
        tail = next((j for j in range(last + 1, head)
                     if _req_line(P[j]) and (_REQ.search(P[j]) or _SIGNER.search(P[j]))), None)
        if tail is None:
            end = second = head
        else:
            sig_end = tail
            while sig_end < head and _req_line(P[sig_end]):
                sig_end += 1
            end, second = tail, sig_end
    body = range(a, end)
    top = _POINT_DOT if any(_POINT_DOT.match(P[j]) for j in body) else _POINT
    numbered = any(top.match(P[j]) for j in body)
    items: list[dict] = []
    for j in body:
        t = P[j]
        m = top.match(t)
        sm = None if m else (_SUB.match(t) or (_POINT.match(t) if top is _POINT_DOT else None))
        if m:
            items.append({"num": canon_num(m["num"]), "a": j, "b": j, "subs": []})
        elif sm and items:
            items[-1]["subs"].append({"num": canon_num(sm["num"]), "a": j, "b": j})
        elif not items:
            items.append({"num": "0" if numbered else "1", "a": j, "b": j, "subs": []})
        elif not numbered:
            items.append({"num": str(len(items) + 1), "a": j, "b": j, "subs": []})
        items[-1]["b"] = j + 1
        if items[-1]["subs"]:
            items[-1]["subs"][-1]["b"] = j + 1
    for it in items:
        iid = tree.add(doc_id, "пункт", it["num"], "", it["a"], it["b"])
        for sp in it["subs"]:
            tree.add(iid, "подпункт", sp["num"], "", sp["a"], sp["b"])
    if tail is not None:
        tree.add(doc_id, "реквизиты", "1", "", tail, n if second is None else sig_end)
    if second is not None:
        _pack_prose(tree, P, second, n, max_tokens, count_tokens)
    return tree.nodes


# ── stenogram ────────────────────────────────────────────────────────────────
_SPEAKER = re.compile(
    r"^(?P<name>Raisli[kq] [qQ]iluvchi|Раислик қилувчи|"
    r"[A-ZА-ЯЎҚҒҲ][\wʼ\-]+(?:\s+(?:[A-ZА-ЯЎҚҒҲ][\wʼ\-.]+|oʼgʼli|qizi|ўғли|қизи)){1,3})\s*:\s*")
# Role labels (≤ 6 words) ending in a role noun: "Toshkent shahar hokimligi vakili:", "Maʼruzachi:"
_ROLE = re.compile(
    r"^(?P<name>(?:[«“\"]?[A-ZА-ЯЎҚҒҲ][^\s:()]*\s+(?:[^\s:()]+\s+){0,4})?"   # "“Hududgaztaʼminot” AJ vakili:"
    r"(?i:vakili|masʼul|hokimi|oʼrinbosari|boshligʼi|raisi|maʼruzachi|mutaxassisi?|"
    r"вакили|масъул|ҳокими|ўринбосари|бошлиғи|раиси|маърузачи|мутахассиси?))\s*:\s*")
# closing remarks ("Kun tartibidagi barcha masalalarni koʼrib boʼldik", "kun tartibimizda qolgan
# masalalarimizni koʼrib chiqdik") do not open an item
_AGENDA = re.compile(r"(kun tartibi|кун тартиби)(?!\w*\s+(?:barcha|hamma|qolgan|барча|ҳамма|қолган)\b)"
                     r".{0,300}?(oʼtamiz|ўтамиз|masala|масала)", re.IGNORECASE | re.DOTALL)
# Sentence-initial "Navbatdagi masala…" / "Keyingi qonun…" in a chair turn also opens an agenda item.
_NEXT_ITEM = re.compile(r"(?:^|\n|[.!?]\s+)(?:Navbatdagi|Навбатдаги|Keyingi|Кейинги)\s+"
                        r"(?:masala|qonun|масала|қонун)")


def _is_chair(speaker: str) -> bool:
    return speaker.lower().startswith(("raisli", "раисли"))


_OPEN_Q, _CLOSE_Q = "«“", "»”"
_ACT_AFTER = re.compile(
    r"\s*(?:-?\s*(?:gi|ga|ning|ги|га|нинг)\b)?\s*"
    r"(?:(?:Oʼzbekiston|Ўзбекистон)\s+(?:Respublikasi|Республикаси)(?:ning|нинг)?\s+)?"
    r"(?:qonun|kodeks|qaror|farmon|konvensiya|shartnoma|bitim|"
    r"қонун|кодекс|қарор|фармон|конвенция|шартнома|битим)", re.IGNORECASE)
_ACT_END = re.compile(r"(?:toʼgʼrisida|haqida|тўғрисида|ҳақида)$", re.IGNORECASE)
# a quoted non-act item: "“… parlament soʼrovi yuborish” haqida", "“… hisoboti” toʼgʼrisida"
_TOPIC_AFTER = re.compile(r"\s*(?:toʼgʼrisida|haqida|тўғрисида|ҳақида)(?:gi|dagi|ги|даги)?(?!\w)", re.IGNORECASE)


def _outer_quotes(text: str, pos: int):
    """Outermost balanced «…» / “…” spans starting at or after pos: (content, end_index)."""
    i = pos
    while i < len(text):
        if text[i] in _OPEN_Q:
            depth, j = 1, i + 1
            while j < len(text) and depth:
                depth += (text[j] in _OPEN_Q) - (text[j] in _CLOSE_Q)
                j += 1
            if depth:                                  # a stray opener, unbalanced to the end of the
                i += 1                                 # window: skip it, later quotes may still balance
                continue
            yield text[i + 1:j - 1], j
            i = j
        else:
            i += 1


def _agenda_title(P: list[str], tr: dict) -> str | None:
    """None if the turn does not announce an agenda item; else the first act-like outermost
    quoted title in the trigger paragraph + the next paragraph of the turn ("" if none).
    Act-like: the quote is followed by an act word ("…”gi Oʼzbekiston Respublikasi Qonuni",
    "»ги Қонун") or the quoted text ends with toʼgʼrisida / haqida (тўғрисида / ҳақида)."""
    paras = P[tr["a"]:tr["b"]]
    text = "\n".join(paras)
    lab = _SPEAKER.match(text) or _ROLE.match(text)
    off = lab.end() if lab else 0                     # text after the label starts a sentence
    hits = [off + m.start() + len(m[0]) - len(m[0].lstrip(".!?\n\t "))   # the trigger word itself
            for m in (_AGENDA.search(text[off:]), _NEXT_ITEM.search(text[off:])) if m]
    if not hits:
        return None
    pos, k, start = min(hits), 0, 0                   # paragraph k of the turn holds the trigger
    while k + 1 < len(paras) and start + len(paras[k]) < pos:
        start, k = start + len(paras[k]) + 1, k + 1
    window = "\n".join(paras[k:k + 2])
    for content, end in _outer_quotes(window, pos - start):
        if _ACT_AFTER.match(window, end) or _ACT_END.search(content.rstrip(" .,")) or _TOPIC_AFTER.match(window, end):
            return re.sub(r"\s+", " ", content).strip()
    return ""


def parse_stenogram(doc_id: str, dt: DocText, title: str, agenda=None) -> list[Node] | None:
    """Turns grouped into agenda items (вопрос). `agenda(title, P, turns, is_chair)` (see
    graphrag.data.agenda.AgendaSegmenter) may return verified [(start turn, title)]; when it is
    None or returns None, the deterministic regex triggers (_agenda_title) segment the turns."""
    P = dt.paras
    turns: list[dict] = []
    for j, t in enumerate(P):
        m = _SPEAKER.match(t) or _ROLE.match(t)
        if m and len(m["name"]) <= 60:
            turns.append({"a": j, "b": j + 1, "speaker": m["name"]})
        elif turns:
            turns[-1]["b"] = j + 1
    if not turns:
        return None
    tree = _Tree(doc_id, dt, title)
    if turns[0]["a"] > 0:
        tree.add(doc_id, "заголовок", "1", "", 0, turns[0]["a"])
    groups: list[list[int]] = []
    titles: list[str] = []
    items = dict(agenda(title, P, turns, _is_chair) or []) if agenda is not None else {}
    for k, tr in enumerate(turns):
        if items:
            announce = items.get(k)
        else:
            announce = _agenda_title(P, tr) if _is_chair(tr["speaker"]) else None
        if announce is not None or not groups:
            groups.append([k])
            titles.append(announce or "")
        else:
            groups[-1].append(k)
    for gi, g in enumerate(groups, 1):
        qid = tree.add(doc_id, "вопрос", str(gi), titles[gi - 1], turns[g[0]]["a"], turns[g[-1]]["b"])
        for ti, k in enumerate(g, 1):
            tree.add(qid, "выступление", str(ti), turns[k]["speaker"], turns[k]["a"], turns[k]["b"])
    return tree.nodes


# ── prose (xulosa, reports, notes, anything unstructured) ────────────────────
_PROSE_HEAD = {"ХУЛОСА", "XULOSA", "ҲИСОБОТ", "HISOBOT", "АХБОРОТ", "AXBOROT", "МАЪЛУМОТ", "MAʼLUMOT"}
_PROSE_HEAD_CAPS = _PROSE_HEAD | {"ҚЎМИТАХУЛОСАСИ", "QOʼMITAXULOSASI"}
_CAPS_TAIL = re.compile(r"(?:^|\s)([A-ZА-ЯЁЎҚҒҲʼ](?:\s*[A-ZА-ЯЁЎҚҒҲʼ])*)\s*[:.]?$")


def _is_prose_head(t: str) -> bool:
    """"Х У Л О С А" line, or a line ending with an upper-case (often letter-spaced) marker:
    "Q Oʼ M I T A X U L O S A S I", "… Қонун юзасидан Х У Л О С А"."""
    if re.sub(r"\s+", "", t).rstrip(":.").upper() in _PROSE_HEAD:
        return True
    m = _CAPS_TAIL.search(t)
    return bool(m) and re.sub(r"\s+", "", m[1]) in _PROSE_HEAD_CAPS


def _pack_prose(tree: _Tree, P: list[str], lo: int, hi: int, max_tokens: int,
                count_tokens: Callable[[str], int]) -> None:
    """Paragraphs [lo, hi) as prose: optional заголовок (up to a head marker within the first
    40 paragraphs), token-packed блоки, trailing реквизиты. Ordinals continue those already
    in the tree, so a prose part after a resolution gets РЕКВ.2 etc."""
    doc_id = tree.doc_id
    nxt = lambda type_: str(1 + sum(nd.type == type_ and nd.parent_id == doc_id for nd in tree.nodes))
    head_end = next((i + 1 for i in range(lo, min(lo + 40, hi)) if _is_prose_head(P[i])), lo)
    if head_end > lo:
        tree.add(doc_id, "заголовок", nxt("заголовок"), "", lo, head_end)
    tail = _requisites_start(P[:hi], head_end)
    end = hi if tail is None else tail
    blocks: list[list[int]] = []
    cur_tok = 0
    for j in range(head_end, end):
        tk = count_tokens(P[j])
        if blocks and blocks[-1][1] == j and cur_tok + tk <= max_tokens:
            blocks[-1][1] = j + 1
            cur_tok += tk
        else:
            blocks.append([j, j + 1])
            cur_tok = tk
    k0 = int(nxt("блок"))
    for k, (a, b) in enumerate(blocks, k0):
        tree.add(doc_id, "блок", str(k), "", a, b)
    if tail is not None:
        tree.add(doc_id, "реквизиты", nxt("реквизиты"), "", tail, hi)


def parse_prose(doc_id: str, dt: DocText, title: str, max_tokens: int,
                count_tokens: Callable[[str], int]) -> list[Node]:
    tree = _Tree(doc_id, dt, title)
    _pack_prose(tree, dt.paras, 0, len(dt.paras), max_tokens, count_tokens)
    return tree.nodes


# ── dispatcher ───────────────────────────────────────────────────────────────
def parse_structure(doc_id: str, section: str, doc_type: str, paras: list[Para], title: str,
                    max_tokens: int, count_tokens: Callable[[str], int],
                    agenda=None) -> tuple[DocText, list[Node]]:
    dt = DocText.build(paras)
    nodes = None
    if section != "STENOGRAMMA":
        nodes = parse_law(doc_id, dt, title)
    if nodes is None and (section == "STENOGRAMMA" or doc_type == "stenogram"):
        nodes = parse_stenogram(doc_id, dt, title, agenda)
    if nodes is None and doc_type in ("resolution", "postanovlenie"):
        nodes = parse_resolution(doc_id, dt, title, max_tokens, count_tokens)
    if nodes is None:
        nodes = parse_prose(doc_id, dt, title, max_tokens, count_tokens)
    return dt, nodes
