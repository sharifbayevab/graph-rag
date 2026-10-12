"""Source file → paragraphs.

.docx is read from word/document.xml directly: superscript runs keep their digits as
Unicode superscripts (122¹ instead of docx2txt's "1221"), Word auto-numbering is
rendered from word/numbering.xml ("1.", "а)"), tables become one paragraph per row,
and headers/footers (page numbers) are never read. Other formats go through the
vendored legacy extractor and are split into paragraphs heuristically.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from graphrag.data.legacy.extract_text import file_text

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_CYR_LOWER = "абвгдежзиклмнопрстуфхцчшщэюя"
_LAT_LOWER = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Para:
    text: str
    list_level: int | None = None   # w:ilvl of an auto-numbered paragraph
    page: int | None = None         # 1-based PDF page (None when the format has no pages)
    page_end: int | None = None     # last page when a PDF paragraph continues onto later pages


def _roman(n: int) -> str:
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                 (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def _fmt_number(fmt: str, n: int) -> str:
    if fmt == "decimal":
        return str(n)
    if fmt == "lowerLetter":
        return _LAT_LOWER[(n - 1) % 26]
    if fmt == "upperLetter":
        return _LAT_LOWER[(n - 1) % 26].upper()
    if fmt == "russianLower":
        return _CYR_LOWER[(n - 1) % len(_CYR_LOWER)]
    if fmt == "russianUpper":
        return _CYR_LOWER[(n - 1) % len(_CYR_LOWER)].upper()
    if fmt == "lowerRoman":
        return _roman(n).lower()
    if fmt == "upperRoman":
        return _roman(n)
    if fmt == "bullet":
        return "–"
    return ""


class _Numbering:
    """(numId, ilvl) → (numFmt, lvlText, start) from numbering.xml, plus running counters."""

    def __init__(self, xml_bytes: bytes | None):
        self.levels: dict[tuple[str, int], tuple[str, str, int]] = {}
        self.counters: dict[str, list[int]] = {}
        if not xml_bytes:
            return
        root = ET.fromstring(xml_bytes)
        abstract: dict[str, dict[int, tuple[str, str, int]]] = {}
        for an in root.findall(f"{W}abstractNum"):
            lv = {}
            for lvl in an.findall(f"{W}lvl"):
                il = int(lvl.get(f"{W}ilvl"))
                fmt, txt, st = lvl.find(f"{W}numFmt"), lvl.find(f"{W}lvlText"), lvl.find(f"{W}start")
                lv[il] = (fmt.get(f"{W}val") if fmt is not None else "decimal",
                          txt.get(f"{W}val") if txt is not None else f"%{il + 1}.",
                          int(st.get(f"{W}val")) if st is not None else 1)
            abstract[an.get(f"{W}abstractNumId")] = lv
        for num in root.findall(f"{W}num"):
            a = num.find(f"{W}abstractNumId")
            if a is None:
                continue
            for il, spec in abstract.get(a.get(f"{W}val"), {}).items():
                self.levels[(num.get(f"{W}numId"), il)] = spec

    def label(self, num_id: str, ilvl: int) -> str:
        if (num_id, ilvl) not in self.levels:
            return ""
        cnt = self.counters.setdefault(num_id, [0] * 9)
        _, txt, start = self.levels[(num_id, ilvl)]
        cnt[ilvl] = start if cnt[ilvl] == 0 else cnt[ilvl] + 1
        for deeper in range(ilvl + 1, 9):
            cnt[deeper] = 0
        out = txt
        for lvl in range(ilvl, -1, -1):
            fmt = self.levels.get((num_id, lvl), ("decimal", "", 1))[0]
            out = out.replace(f"%{lvl + 1}", _fmt_number(fmt, max(cnt[lvl], 1)))
        return out.strip()


def _run_text(r) -> str:
    rpr = r.find(f"{W}rPr")
    va = rpr.find(f"{W}vertAlign") if rpr is not None else None
    sup = va is not None and va.get(f"{W}val") == "superscript"
    out = []
    for ch in r:
        if ch.tag == f"{W}t":
            t = ch.text or ""
            out.append(t.translate(_SUPERSCRIPT) if sup else t)
        elif ch.tag in (f"{W}tab", f"{W}br", f"{W}cr"):
            out.append(" ")
        elif ch.tag == f"{W}noBreakHyphen":
            out.append("-")
    return "".join(out)


def _para_text(p) -> str:
    return "".join(_run_text(r) for r in p.iter(f"{W}r"))


def _para(p, numbering: _Numbering) -> list[Para]:
    text = _para_text(p)
    level = None
    npr = p.find(f"{W}pPr/{W}numPr")
    if npr is not None:
        nid_el, il_el = npr.find(f"{W}numId"), npr.find(f"{W}ilvl")
        nid = nid_el.get(f"{W}val") if nid_el is not None else "0"
        if nid != "0":
            level = int(il_el.get(f"{W}val")) if il_el is not None else 0
            lab = numbering.label(nid, level)
            if lab and text.strip():
                text = f"{lab} {text.lstrip()}"
    return [Para(text, level)] if text.strip() else []


def read_docx(path: Path) -> list[Para]:
    with zipfile.ZipFile(path) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        names = set(z.namelist())
        numbering = _Numbering(z.read("word/numbering.xml") if "word/numbering.xml" in names else None)
    body = doc.find(f"{W}body")
    out: list[Para] = []
    for el in body:
        if el.tag == f"{W}p":
            out.extend(_para(el, numbering))
        elif el.tag == f"{W}tbl":
            for tr in el.iter(f"{W}tr"):
                cells = [" ".join(_para_text(p) for p in tc.iter(f"{W}p")).strip()
                         for tc in tr.findall(f"{W}tc")]
                row = " | ".join(c for c in cells if c)
                if row:
                    out.append(Para(row))
        elif el.tag == f"{W}sdt":
            for p in el.iter(f"{W}p"):
                out.extend(_para(p, numbering))
    return out


_PAGE_NUM = re.compile(r"^\d{1,3}$")
_DOT_LEADER = re.compile(r"(\.{4,}|…{2,})\s*\d+([-–]\d+)?$")
_SENT_END = re.compile(r"[.:;!?]$")
_PARA_START = re.compile("^[A-ZА-ЯЁЎҚҒҲ0-9«\"“]")


def paras_from_pdf_text(text: str) -> list[Para]:
    """Reflow hard-wrapped PDF lines; drop page numbers, TOC leaders, running headers."""
    lines = [ln.strip() for ln in text.splitlines()]
    freq = Counter(ln for ln in lines if ln)
    paras: list[Para] = []
    buf: list[str] = []

    def flush():
        if buf:
            paras.append(Para(" ".join(buf)))
            buf.clear()

    for s in lines:
        if not s:
            flush()
            continue
        if _PAGE_NUM.match(s) or _DOT_LEADER.search(s) or (freq[s] >= 5 and len(s) < 120):
            continue
        if buf and _SENT_END.search(buf[-1]) and _PARA_START.match(s):
            flush()
        buf.append(s)
    flush()
    return paras


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_TAGS = re.compile(r"</?(?:br|sup|sub)\s*/?>", re.IGNORECASE)
_MD_TABLE_RULE = re.compile(r"^\|?[\s:\-|]+\|?$")


def paras_from_md_text(text: str) -> list[Para]:
    out = []
    for line in text.splitlines():
        s = _MD_TAGS.sub(" ", _MD_IMAGE.sub("", line))
        s = re.sub(r"^#{1,6}\s*", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s and not _MD_TABLE_RULE.match(s):
            out.append(Para(s))
    return out


_VISION = None
PDF_ENGINE: dict[str, str] = {}      # str(path) → "opendataloader" | "pypdf" | "pypdf_fallback", this process


def _vision(v: dict):
    """One transcriber per process, so its API-call budget covers the whole run. API calls only
    with `pdf.vision.api: true` or GRL_VISION_API=1; otherwise cached answers only, and a miss
    raises VisionCacheMiss (`on_miss: skip`, set by build_corpus.py --allow-degraded, drops it)."""
    global _VISION
    if _VISION is None:
        import os
        from graphrag import config
        from graphrag.data.pdf import VisionTranscriber
        allow = bool(v.get("api")) or os.environ.get("GRL_VISION_API") == "1"
        _VISION = VisionTranscriber(v["model"], config.path(v["cache_dir"]),
                                    max_calls=int(v.get("max_calls_per_run", 600)),
                                    allow_api=allow, on_miss=v.get("on_miss", "raise"))
    return _VISION


def read_pdf_paragraphs(path: Path) -> list[Para]:
    from graphrag import config
    from graphrag.data import pdf

    pcfg = config.load("data")["pdf"]
    if pcfg["engine"] == "opendataloader":
        v = pcfg["vision"]
        try:
            out = pdf.read_pdf(path, _vision(v) if v["enabled"] else None)
            PDF_ENGINE[str(path)] = "opendataloader"
            return out
        except pdf.PdfEngineError as e:  # no Java / ODL failure → degrade, keep pages
            import logging
            logging.getLogger(__name__).warning("opendataloader failed on %s: %r; using pypdf", path, e)
    from pypdf import PdfReader
    import logging
    PDF_ENGINE[str(path)] = "pypdf_fallback" if pcfg["engine"] == "opendataloader" else "pypdf"
    logging.getLogger(__name__).warning(
        "pypdf fallback for %s: text is not filtered for visibility and may include invisible or "
        "off-page text; no tables, reading order or image transcription", path)
    pages = [pg.extract_text() or "" for pg in PdfReader(str(path)).pages]
    # running headers repeat across pages, not within one: count them document-wide
    freq = Counter(s for t in pages for s in {ln.strip() for ln in t.splitlines()} if s)
    out = []
    for i, t in enumerate(pages, 1):
        kept = "\n".join(ln for ln in t.splitlines() if not (freq[ln.strip()] >= 5 and len(ln.strip()) < 200))
        out += [Para(p.text, None, i) for p in paras_from_pdf_text(kept)]
    return out


def read_paragraphs(path: Path) -> list[Para]:
    ext = path.suffix.lower()
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf_paragraphs(path)
    text = file_text(path)
    if ext == ".md":
        return paras_from_md_text(text)
    # .txt and legacy .doc (textutil emits one line per paragraph)
    return [Para(ln.strip()) for ln in text.splitlines() if ln.strip()]
