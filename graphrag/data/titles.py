"""Document titles: post-processing of the vendored legacy parser's `title_raw` (Task 11b).

* Uzbek act titles split over several header paragraphs are joined until the title terminator
  (тўғрисида / ҳақида / toʼgʼrisida / haqida, or a closing quote + -ги/-gi + an act word):
  "Янги таҳрирдаги … Конституцияси" + "қабул қилинганлиги муносабати …" + "… киритиш тўғрисида".
* Russian laws: "ЗАКОН" / "Республики Узбекистан" / "О …" line(s) → "О …".
* Parliamentary inquiries: "П А Р Л А М Е Н Т  С Ў Р О В И" + "(… юзасидан)" → "Парламент сўрови (…)".
* Stenograms: a session title derived from the file stem ("Senatning 7-yalpi majlisi 28-29.05.25"),
  dated from the text header ("(2023-yil 3-4-may)") when the stem has no date.
Titles are built from the document's own paragraphs only (verbatim, whitespace-normalized).
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

_TERM = re.compile(r"(?:тўғрисида|ҳақида|toʼgʼrisida|to['‘’ʻ]g['‘’ʻ]risida|haqida)$", re.IGNORECASE)
_QUOTE_ACT = re.compile(r"[»”\"]\s*-?(?:ги|gi)\s+(?:Ўзбекистон\s+Республикаси\s+|Oʼzbekiston\s+Respublikasi\s+)?"
                        r"(?:Қонун|Кодекс|Qonun|Kodeks)(?:и|i)?$", re.IGNORECASE)
_STOP = re.compile(r"^(?:Қонунчилик палатаси|Qonunchilik palatasi|қабул қилинган\b|qabul qilingan\b|"
                   r"Сенат томонидан|Senat tomonidan|Принят|Одобрен|Глава|Статья|\d+\s*[-–]\s*(?:модда|modda|боб|bob))",
                   re.IGNORECASE)
MAX_TITLE_PARAS = 6
MAX_TITLE_CHARS = 500
_WS = re.compile(r"\s+")


def _clean(t: str) -> str:
    return _WS.sub(" ", t).strip()


def _terminated(t: str) -> bool:
    """Ends with the terminator word and every quote opened in it is closed ("«Давлат божи
    тўғрисида" + "»ги … Қонунига … ҳақида" is one title)."""
    if t.count("«") > t.count("»") or t.count("“") > t.count("”"):
        return False
    t = t.rstrip(" .,;:")
    return bool(_TERM.search(t) or _QUOTE_ACT.search(t))


def join_header_title(title: str | None, paras: list[str], head: int = 15) -> str | None:
    """`title` extended over the consecutive header paragraphs it starts in, up to the terminator
    (whole paragraphs are joined, so "тўғрисида" never matches inside "тўғрисидаги"). Unchanged
    when its start paragraph is not found, no terminator follows within MAX_TITLE_PARAS
    paragraphs, or the joined title would not extend it (it must start with the legacy title:
    "Ўзбекистон Республикасининг" / "ҚОНУНИ" label lines above the title are never pulled in)."""
    if not title:
        return title
    t0 = _clean(title)
    for s, p in enumerate(paras[:head]):
        p = _clean(p)
        if not p or not (t0.startswith(p[:40]) or p.startswith(t0[:40])):
            continue
        parts: list[str] = []
        for q in paras[s:s + MAX_TITLE_PARAS]:
            q = _clean(q)
            if parts and (not q or _STOP.match(q)):
                break
            parts.append(q)
            joined = " ".join(parts)
            if len(joined) > MAX_TITLE_CHARS:
                break
            if _terminated(joined):
                if len(joined) > len(t0) and joined.startswith(t0):
                    return joined                      # an extension of the legacy title, never a rewrite
                break
    return title


_RU_LAW = re.compile(r"^З\s*А\s*К\s*О\s*Н$", re.IGNORECASE)
_RU_REPUBLIC = re.compile(r"^Республики Узбекистан$", re.IGNORECASE)
_RU_TITLE = re.compile(r"^Об?\s")


def russian_law_title(paras: list[str], head: int = 12) -> str | None:
    """"ЗАКОН" / "Республики Узбекистан" / "О санации и ликвидации банков" → the "О …" line(s)."""
    for i, p in enumerate(paras[:head]):
        if not _RU_LAW.match(_clean(p)):
            continue
        j = i + 1
        if j < len(paras) and _RU_REPUBLIC.match(_clean(paras[j])):
            j += 1
        if j >= len(paras) or not _RU_TITLE.match(_clean(paras[j])):
            return None
        parts = [_clean(paras[j])]
        for q in paras[j + 1:j + 3]:
            q = _clean(q)
            if not q or _STOP.match(q) or not q[:1].islower():
                break
            parts.append(q)
        return " ".join(parts)
    return None


_INQUIRY = {"ПАРЛАМЕНТСЎРОВИ": "Парламент сўрови", "PARLAMENTSOʼROVI": "Parlament soʼrovi"}


def inquiry_title(paras: list[str], head: int = 12) -> str | None:
    """"П А Р Л А М Е Н Т  С Ў Р О В И" + "(… юзасидан)" → "Парламент сўрови (… юзасидан)"."""
    for i, p in enumerate(paras[:head]):
        name = _INQUIRY.get(re.sub(r"\s+", "", p).upper())
        if name:
            nxt = _clean(paras[i + 1]) if i + 1 < len(paras) else ""
            return f"{name} {nxt}" if nxt.startswith("(") and nxt.endswith(")") else name
    return None


_SESSION = re.compile(r"(\d+)\s*-\s*(?:yalpi|ялпи)", re.IGNORECASE)
_STEM_DATE = re.compile(r"(\d{1,2}(?:-\d{1,2})?\.\d{2}\.\d{2,4})")
_TEXT_DATE = re.compile(r"\((\d{4}-yil [^()]{2,40}|\d{4} йил [^()]{2,40})\)")


def session_title(relpath: str, paras: list[str], head: int = 15) -> str | None:
    """Stenogram session title from the file stem: "Oliy Majlis Senati  7-yalpi majlis 28-29.05.25"
    → "Senatning 7-yalpi majlisi 28-29.05.25"; "Стен. 40-ялпи мажлис" + header "(2023-yil 3-4-may)"
    → "Senatning 40-yalpi majlisi (2023-yil 3-4-may)"."""
    stem = PurePosixPath(relpath).stem
    m = _SESSION.search(stem)
    if not m:
        return None
    title = f"Senatning {m[1]}-yalpi majlisi"
    d = _STEM_DATE.search(stem[m.end():])
    if d:
        return f"{title} {d[1]}"
    for p in paras[:head]:
        d = _TEXT_DATE.search(p)
        if d:
            return f"{title} ({_clean(d[1])})"
    return title


def fix_title(title: str | None, paras: list[str], section: str, relpath: str) -> str | None:
    """The document title after the rules above (None when nothing applies)."""
    if section == "STENOGRAMMA":
        return session_title(relpath, paras) or title
    if title:
        return join_header_title(title, paras)
    return russian_law_title(paras) or inquiry_title(paras)
