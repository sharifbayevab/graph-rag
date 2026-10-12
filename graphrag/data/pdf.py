"""PDF → paragraphs with page numbers.

opendataloader-pdf (local, deterministic: XY-Cut++ reading order, tables, header/footer
filtering) produces elements; images with data are transcribed by gpt-4.1 vision and kept
only when two independent readings agree on every number. Vision answers are cached by
image hash so rebuilds are byte-identical and offline.

ODL JSON schema (opendataloader-pdf 2.5.11, captured from the two ombudsman reports):
  document  {"file name", "number of pages", ..., "kids": [element, ...]}  (array order = reading order)
  element   {"type", "id", "page number", "bounding box", ...}
    heading / paragraph / caption   text in "content"
    list      "list items": [list item]
    list item "content" (its own text) + "kids" (nested lists / continuation paragraphs)
    table     "rows": [{"type": "table row", "cells": [{"type": "table cell", "page number", "kids"}]}]
    text block "kids" (a layout group; №13 infographics)
    image     "source": path relative to the ODL output dir
    header / footer "kids" (only emitted with include_header_footer; dropped here anyway)
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from graphrag import config
from graphrag.data.extract import Para

log = logging.getLogger(__name__)

PROMPT_VERSION = 5          # policy: B always asked; numbers(A) == numbers(B) as multisets (v5: two views of B)
PROMPT_A_VERSION = 3        # bump when PROMPT_A changes (cached A answers are reused otherwise)
PROMPT_B_VERSION = 2
PAGE_PROMPT_VERSION = 2     # v2: no section/heading numbers (they are excluded from the check too)
PROMPT_A = ("This image is from an official report. If it is a chart, graph, diagram, infographic, table, "
            "labelled map or a scanned page that conveys facts or figures, transcribe VERBATIM all text and "
            "numbers visible in it, in the original language and script (Uzbek Cyrillic/Latin or Russian). "
            "Keep labels next to their numbers. Do not translate, summarize, comment or add anything. "
            "Otherwise - a photograph (even with visible signs, banners, badges or screens), a logo, emblem, "
            "seal, icon, ornament, book or document cover, or a screenshot of an app or website - "
            "answer exactly NONE.")
PROMPT_B = ("List every number visible in this image - every occurrence, including years, dates, "
            "percentages, article and item numbers - one per line in the form `NUMBER | LABEL`, where "
            "LABEL is the text that number belongs to, in the original language and script. Do not "
            "number or bullet the lines and add no commentary. If there are no numbers, answer NONE.")
PAGE_PROMPT = ("This is one page of an official report (Uzbek or Russian). For EVERY number printed on the "
               "page - every occurrence, including years, dates, percentages and numbers inside sentences, "
               "but NOT the section or heading numbers in front of headings (such as \"2.\" in \"2. ТИТУЛ\") "
               "and NOT page numbers - output one line `LABEL: VALUE`, where VALUE is the number exactly as "
               "printed (with its unit such as та, нафар or %) and LABEL is the short text on the page that "
               "this number describes, in the original language and script. Do not translate, do not number "
               "the lines, add no commentary.")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_THOUSANDS = re.compile(r"(?<=\d)[   ](?=\d{3}(?!\d))")   # "20 611" → "20611"
_TOK = re.compile(r"\w+")
_WS = re.compile(r"\s+")
_TOC_LEADER = re.compile(r"(\.{4,}|…{2,})\s*(\|\s*)?(\d+|[IVXLC]+\b)")   # "…… 4-8", "…… | 58", "…… V"
_LINE_HYPHEN = re.compile(r"(?<=\w)-[ \t]*\n\s*")        # "ижтимоий-\nҳуқуқий": a compound, keep "-"
_SOFT_HYPHEN = re.compile(r"\u00ad\s*\n?\s*")
_SENT_END = re.compile(r"([.:;!?…][\"»”’)]*|ди|дилар|ldi|dilar)$")   # Uzbek past-tense verb ends a sentence
_CONT_START = re.compile(r"[a-zа-яёўқғҳʼ‘(]")
_LOWER_START = re.compile(r"[a-zа-яёўқғҳʼ‘]")
_CLAUSE_END = re.compile(r"(ган|gan)$")
_PRED_END = re.compile(r"(?:моқда|moqda|эга|ega|мавжуд|mavjud)$")         # predicate without a full stop
_VALUE_END = re.compile(r"\d[\d\s.,–-]*\s*(?:та|нафар|фоиз|%|ёш|ta|nafar|foiz|yosh)?$")   # "… 201 та", "11-17 ёш"
_JOIN_GAP = 24.0            # pt: a same-page continuation starts at most this far below the piece before it
_PROSE_MIN = 200                                          # a split prose paragraph is at least this long
_NUM_ONLY = re.compile(r"^[\d\s.,%]+(?:\s*(?:та|нафар|%))?$")
_ATTACH_GAP = 1.0
_VALUE = re.compile(r"^[\d\s.,%+\-–]*\d[\d\s.,%]*(?:[-–][^\W\d_]+)?(?:\s+\S+){0,2}$")   # + "2024-йил", "11-модда"
_SECTION_NO = re.compile(r"^\s*\d+(?:\.\d+)*\.\s+")   # "2. ЖИСМОНИЙ …": heading numbering, not data
_DIGIT = re.compile(r"\d")           # a label that already carries a number is not a label for another
_LABEL_TYPES = {"paragraph", "heading", "caption", "list_item"}
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_NUL_TI = re.compile(r"(?<=[A-Za-z])\x00(?=[a-z])")      # unmapped "ti" ligature ("Interna\x00onal")
_SKIP_TYPES = {"header", "footer", "watermark", "page header", "page footer"}
_TEXT_TYPES = {"heading", "paragraph", "caption"}


class PdfEngineError(RuntimeError):
    """opendataloader-pdf / Java unavailable or failed: the caller may fall back to pypdf."""


class VisionBudgetExceeded(RuntimeError):
    """More vision API calls than `max_calls` would be made in one run."""


class VisionCacheMiss(RuntimeError):
    """A vision answer is not cached and API calls are not allowed (`pdf.vision.api: false`,
    no GRL_VISION_API=1): the build would not be reproducible offline."""


# ---------------------------------------------------------------- ODL JSON → elements

def _el(typ: str, page: int, text: str, image: str | None = None, bbox=None) -> dict:
    return {"type": typ, "page": page, "text": text, "image": image,
            "bbox": list(bbox) if bbox else None}


def _content(node: dict) -> str:
    """ODL runs with keep_line_breaks (its own de-hyphenation glues compounds such as
    "хотин-қизлар" into "хотинқизлар"); lines are re-joined here keeping the hyphen."""
    s = _NUL_TI.sub("ti", (node.get("content") or "").translate(_LIGATURES))
    s = _SOFT_HYPHEN.sub("", s)
    return _WS.sub(" ", _LINE_HYPHEN.sub("-", s)).strip()


def _walk(node: dict, page: int | None, out: list[dict]) -> None:
    typ = (node.get("type") or "").strip().lower()
    if typ in _SKIP_TYPES:
        return
    page = node.get("page number") or page
    if typ == "image":
        out.append(_el("image", page, "", node.get("source"), node.get("bounding box")))
        return
    if typ == "table":
        for row in node.get("rows") or []:
            _table_row(row, page, out)
        return
    if typ == "list":
        for item in node.get("list items") or []:
            _walk(item, page, out)
        return
    text = _content(node)
    if text:
        out.append(_el("list_item" if typ == "list item" else typ.replace(" ", "_") or "paragraph",
                       page, text, None, node.get("bounding box")))
    for kid in node.get("kids") or []:
        _walk(kid, page, out)
    for item in node.get("list items") or []:
        _walk(item, page, out)
    for row in node.get("rows") or []:
        _table_row(row, page, out)


def _table_row(row: dict, page: int | None, out: list[dict]) -> None:
    cells, images, row_page = [], [], None
    for cell in row.get("cells") or []:
        cpage = cell.get("page number") or page
        row_page = row_page or cpage
        sub: list[dict] = []
        for kid in cell.get("kids") or []:
            _walk(kid, cpage, sub)
        txt = " ".join(e["text"] for e in sub if e["type"] != "image" and e["text"])
        if txt:
            cells.append(txt)
        images += [e for e in sub if e["type"] == "image"]
    if cells:
        out.append(_el("table_row", row_page or page, " | ".join(cells)))
    out.extend(images)


def flatten_elements(doc_json) -> list[dict]:
    """Recursively walk ODL output → [{type, page, text, image, bbox}] in reading order.
    Tables yield one 'table_row' per row (non-empty cells joined with ' | '); lists one
    'list_item' per item (nested lists follow their parent item); text blocks are
    transparent; header/footer/watermark subtrees are dropped."""
    kids = doc_json.get("kids", []) if isinstance(doc_json, dict) else doc_json
    out: list[dict] = []
    for node in kids:
        _walk(node, None, out)
    out = [e for e in out if isinstance(e["page"], int)]
    out.sort(key=lambda e: e["page"])          # stable: ODL order within a page is kept
    return out


def _num_only(e: dict) -> bool:
    t = e["text"].strip()
    return e["type"] in _LABEL_TYPES and bool(_NUM_ONLY.match(t)) and any(c.isdigit() for c in t)


def _nearest_label(nb, cands: list[tuple[int, list]]) -> int | None:
    """Label for a number box: the closest box to its right (same line) or below (same column)
    within _ATTACH_GAP font heights; else any box that close; ties → nearest centre. On №13 all
    correct pairs are ≤ 0.75 font heights apart and the wrong candidates ≥ 1.5, hence 1.0.
    Boxes are PDF [left, bottom, right, top] (y grows upwards)."""
    l, b, r, t = nb
    fh = max(t - b, 1.0)
    cx, cy = (l + r) / 2, (b + t) / 2
    best = []
    for k, (cl, cb, cr, ct) in cands:
        cdist = ((cl + cr) / 2 - cx) ** 2 + ((cb + ct) / 2 - cy) ** 2
        gaps = []
        if cb < t and ct > b:
            gaps.append(cl - r)                               # right
        if cl < r and cr > l:
            gaps.append(b - ct)                               # below
        gaps = [g for g in gaps if -0.5 * fh <= g <= _ATTACH_GAP * fh]
        if gaps:
            best.append((0, max(min(gaps), 0), cdist, k))
            continue
        dx = max(cl - r, l - cr, 0)
        dy = max(cb - t, b - ct, 0)
        if (dx * dx + dy * dy) ** 0.5 <= _ATTACH_GAP * fh:
            best.append((1, (dx * dx + dy * dy) ** 0.5, cdist, k))
    return min(best)[3] if best else None


def attach_numbers(elements: list[dict]) -> list[dict]:
    """Infographics put a big number ("61,4%") and its label ("қиз бола") in separate boxes that
    reading order may pull apart. Each number-only element is prefixed to its nearest label on
    the same page (by bounding box) and removed; elements without a bbox are left alone."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(elements):
        by_page[e["page"]].append(i)
    target: dict[int, int] = {}
    for idxs in by_page.values():
        labels = [(k, elements[k]["bbox"]) for k in idxs
                  if elements[k]["type"] in _LABEL_TYPES and elements[k].get("bbox")
                  and elements[k]["text"].strip() and not _DIGIT.search(elements[k]["text"])]
        for i in idxs:
            if _num_only(elements[i]) and elements[i].get("bbox"):
                j = _nearest_label(elements[i]["bbox"], labels)
                if j is not None:
                    target[i] = j
    prefix: dict[int, list[str]] = defaultdict(list)
    for i in sorted(target):
        prefix[target[i]].append(elements[i]["text"].strip())
    out = []
    for i, e in enumerate(elements):
        if i in target:
            continue
        if i in prefix:
            e = {**e, "text": " ".join(prefix[i] + [e["text"].strip()]), "attached": True}
        out.append(e)
    return out


def _side_by_side(a, b) -> bool:
    """Two boxes on the same line at different x: separate labels, not one wrapped text."""
    return bool(a and b) and abs(a[3] - b[3]) <= 3 and abs(a[0] - b[0]) > 3 and (b[0] >= a[2] - 3 or a[0] >= b[2] - 3)


def _continues(prev: str, nxt: str) -> bool:
    """Is paragraph `nxt` the rest of the unfinished paragraph `prev`?
    Long prose (≥ _PROSE_MIN chars) continues with a lowercase word or "("; a short piece
    (an infographic label cut by a font change) only with a lowercase word, and not after an
    all-caps heading or a clause already closed by a participle ("… келиб тушган")."""
    if len(prev) >= _PROSE_MIN:
        return bool(_CONT_START.match(nxt))
    last = prev.split()[-1] if prev.split() else ""
    return bool(_LOWER_START.match(nxt)) and not last.isupper() and not _CLAUSE_END.search(prev)


_JOIN_TYPES = {"paragraph", "heading", "caption", "list_item"}


def _sentence_continues(prev: dict, e: dict, prev_text: str, t: str, prev_page: int) -> bool:
    """Second-chance join for a sentence split across ODL elements of different types (a
    paragraph continued by a caption, a prose line typed "heading", a list item continued by a
    paragraph): `prev_text` has no terminal punctuation and `t` starts lowercase, on the same or
    the next page. Never for table rows, "[расм…]" lines, number/label pairs from the bbox merge,
    all-caps headings or numbered headings; a short piece (< _PROSE_MIN) additionally not side by
    side with `t` and, when it closes a clause ("… келиб тушган"), only before a long prose paragraph."""
    if prev["type"] not in _JOIN_TYPES or e["type"] not in _JOIN_TYPES:
        return False
    if prev.get("attached") or e.get("attached") or prev_text.startswith("[расм"):
        return False
    if _SENT_END.search(prev_text) or not _LOWER_START.match(t) or e["page"] not in (prev_page, prev_page + 1):
        return False
    words = prev_text.split()
    if not words or words[-1].isupper() or _SECTION_NO.match(prev_text):
        return False
    if _PRED_END.search(prev_text) or _VALUE_END.search(prev_text):
        return False                                  # a finished statement / a value ends an infographic item
    a, b = prev.get("bbox"), e.get("bbox")
    if e["page"] == prev_page and a and b and not (-_JOIN_GAP < a[1] - b[3] <= _JOIN_GAP):
        return False                                  # not the next line: another list item / column
    if len(prev_text) < _PROSE_MIN:
        if e["page"] == prev_page and _side_by_side(a, b):
            return False
        if _CLAUSE_END.search(prev_text) and len(t) < _PROSE_MIN // 2:
            return False
    return True


def elements_to_paras(elements: list[dict], image_text: Callable[[dict], str | None]) -> list[Para]:
    elements = attach_numbers(elements)
    pages_of = Counter()
    for t in {(e["text"].strip(), e["page"]) for e in elements if e["text"].strip()}:
        pages_of[t[0]] += 1
    toc, text_els = Counter(), Counter()
    for e in elements:
        if e["type"] != "image" and e["text"].strip():
            text_els[e["page"]] += 1
            toc[e["page"]] += bool(_TOC_LEADER.search(e["text"]))
    toc_pages = {p for p, n in toc.items() if n * 2 >= text_els[p]}    # contents page: drop it whole
    out: list[Para] = []
    prose = None          # index in `out` of the last paragraph that may continue (images may follow it)
    prose_box = None      # bbox of the element that last extended out[prose]
    last = None           # (index in `out`, element) of the last text element, for _sentence_continues
    for e in elements:
        if e["type"] == "image":
            txt = image_text(e)
            if txt:
                out.append(Para(f"[расм] {txt}", None, e["page"]))
            continue
        t = e["text"].strip()
        if not t or (len(t) < 120 and pages_of[t] >= 5):
            continue                                  # running header that escaped ODL
        if _TOC_LEADER.search(t) or e["page"] in toc_pages:
            continue                                  # table of contents
        box = e.get("bbox")
        if (e["type"] == "paragraph" and prose is not None and _continues(out[prose].text, t)
                and not (len(out[prose].text) < _PROSE_MIN and e["page"] == (out[prose].page_end or out[prose].page)
                         and _side_by_side(prose_box, box))):
            cur = prose                               # split by a page break, a figure or a font change
        elif last is not None and _sentence_continues(last[1], e, out[last[0]].text, t,
                                                      out[last[0]].page_end or out[last[0]].page):
            cur = last[0]                             # sentence split across element types
        else:
            cur = None
        if cur is not None:
            q = out[cur]
            end = e["page"] if e["page"] != q.page else q.page_end
            out[cur] = Para(q.text + t if q.text.endswith("-") else f"{q.text} {t}", None, q.page, end)
        else:
            out.append(Para(t, None, e["page"]))
            cur = len(out) - 1
        prose = cur if e["type"] == "paragraph" and not _SENT_END.search(out[cur].text) else None
        prose_box = box
        last = (cur, e) if e["type"] in _JOIN_TYPES else None
    return out


# ---------------------------------------------------------------- vision

def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG IHDR chunk, None if `data` is not a PNG."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _numbers(s: str) -> Counter:
    """Multiset of normalized numbers ("20 611" = "20611", "31,8" = "31.8")."""
    return Counter(n.replace(",", ".") for n in _NUM.findall(_THOUSANDS.sub("", s)))


def _b_numbers(b: str) -> Counter:
    """Numbers of a `NUMBER | LABEL` listing, from the NUMBER fields only (labels may hold digits)."""
    if _is_none(b):
        return Counter()
    out = Counter()
    for line in b.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        out += _numbers(line.split(" | ", 1)[0] if " | " in line else line)
    return out


def _agree(a: str, b: str) -> bool:
    """Readings agree: same multiset of numbers; an A without digits against a B with digits
    (values dropped by the transcription, e.g. a choropleth map) does not agree. B is read two
    ways - its NUMBER fields, or every number in it (for "01 | 1666 таси …" listings)."""
    if _is_none(a):
        return False
    na = _numbers(a)
    return na == _b_numbers(b) or (bool(na) and na == _numbers(b))   # NUMBER fields, or all of B


def _is_none(s: str) -> bool:
    return not s or s.strip().strip(".").upper() == "NONE"


def _snap_label(label: str, page_texts: list[str], threshold: float = 0.8) -> str | None:
    """The contiguous span of one text-layer element that best matches `label` (character-level
    difflib ratio over word windows of the label's length ± 2), verbatim; None below `threshold`
    (a ratio equal to `threshold` is accepted)."""
    want = label.lower()
    n = len(label.split())
    best, best_key = None, (threshold, float("-inf"))
    for text in page_texts:
        words = text.split()
        for size in range(max(1, n - 2), n + 3):
            for i in range(0, max(len(words) - size, 0) + 1):
                cand = " ".join(words[i:i + size]).strip(" ,;:.")
                if not cand:
                    continue
                r = difflib.SequenceMatcher(None, want, cand.lower()).ratio()
                key = (r, -abs(size - n))
                if key > best_key:
                    best, best_key = cand, key
    return best


def _write_json(f: Path, rec: dict) -> None:
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, f)                                  # atomic: a crash never leaves half a record


class VisionTranscriber:
    """image bytes + page text → verified transcription or None.

    Call A transcribes; call B lists `NUMBER | LABEL` for every number. A is kept only if the
    multiset of numbers in A equals that of B's NUMBER fields, it has ≥ 3 tokens, and < `min_overlap`
    of its tokens are already in the page's text layer. Answers are cached in
    `cache_dir/<sha256>.json`; a cached A is reused while PROMPT_A is unchanged, B while PROMPT_B is.
    `page_pairs` reads a whole rendered page as `LABEL: VALUE` lines (see there).
    At most `max_calls` API calls per instance (one instance = one run).
    With `allow_api=False` a cache miss never calls the API: `on_miss="raise"` raises
    VisionCacheMiss, `on_miss="skip"` treats the image / page as unreadable (counted in
    stats["offline_miss"], nothing cached)."""

    def __init__(self, model: str, cache_dir: Path, client=None, min_overlap: float = 0.8,
                 max_calls: int = 600, allow_api: bool = True, on_miss: str = "raise"):
        self.model, self.cache_dir, self.min_overlap = model, Path(cache_dir), min_overlap
        self.max_calls = max_calls
        if on_miss not in ("raise", "skip"):
            raise ValueError(f"on_miss must be 'raise' or 'skip', got {on_miss!r}")
        self.allow_api, self.on_miss = allow_api, on_miss
        self._missed: set[str] = set()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    def _call(self, image: bytes, prompt: str) -> str:
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(max_retries=5)
        b64 = base64.b64encode(image).decode()
        r = self._client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        return (r.choices[0].message.content or "").strip()

    def _offline_miss(self, key: str) -> None:
        """Called before any API call when API calls are not allowed."""
        with self._lock:
            self._missed.add(key)
            self.stats["offline_miss"] = len(self._missed)
        if self.on_miss == "raise":
            raise VisionCacheMiss(
                f"vision answer for {key} is not in the cache ({self.cache_dir}) and API calls are off "
                f"(pdf.vision.api: false); set GRL_VISION_API=1 to fetch it, or pass --allow-degraded")

    def _reserve(self, n: int) -> None:
        with self._lock:
            if self.stats["api_calls"] + n > self.max_calls:
                raise VisionBudgetExceeded(
                    f"vision budget of {self.max_calls} API calls per run would be exceeded")
            self.stats["api_calls"] += n

    def _record(self, image: bytes) -> dict:
        key = hashlib.sha256(image).hexdigest()
        f = self.cache_dir / f"{key}.json"
        old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if old.get("model") != self.model:
            old = {}
        if old.get("prompt_version") == PROMPT_VERSION:
            self.stats["cache_hits"] += 1
            return old
        a_ok = old.get("prompt_a_version", old.get("prompt_version")) == PROMPT_A_VERSION and "a" in old
        b_ok = a_ok and old.get("prompt_b_version") == PROMPT_B_VERSION and "b" in old
        if not self.allow_api and not (b_ok or (a_ok and _is_none(old["a"]))):
            self._offline_miss(key)
            return {"a": "NONE", "b": "NONE", "accepted": False}   # skip mode: unreadable, not cached
        if not a_ok:
            self._reserve(2)
            self.stats["sent"] += 1
            a = self._call(image, PROMPT_A)
            if _is_none(a):
                with self._lock:
                    self.stats["api_calls"] -= 1       # B not needed: give the reserved call back
        else:
            a = old["a"]
            if not b_ok and not _is_none(a):
                self._reserve(1)
                self.stats["sent_b_only"] += 1
        if b_ok:
            b = old["b"]
        else:
            b = "NONE" if _is_none(a) else self._call(image, PROMPT_B)
        rec = {"model": self.model, "prompt_version": PROMPT_VERSION, "prompt_a_version": PROMPT_A_VERSION,
               "prompt_b_version": PROMPT_B_VERSION, "a": a, "b": b, "accepted": _agree(a, b)}
        _write_json(f, rec)
        return rec

    def prefetch(self, images: list[bytes], workers: int = 8) -> None:
        """Fill the cache for `images` concurrently (answers do not depend on order)."""
        todo = list({hashlib.sha256(b).hexdigest(): b for b in images}.values())
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(self._record, todo):
                pass
        self.stats["cache_hits"] = 0                   # prefetch hits are not the build's hits

    def __call__(self, image: bytes, page_text: str) -> str | None:
        self.stats["seen"] += 1
        rec = self._record(image)
        a, b = rec["a"], rec["b"]
        if _is_none(a):
            self.stats["none"] += 1
            return None
        if len(_TOK.findall(a)) < 3:
            self.stats["rejected_short"] += 1
            return None                                   # "X", "1", a stray glyph: no data
        if not _agree(a, b):
            self.stats["rejected_disagreement"] += 1
            log.info("vision: numbers disagree, dropped: A=%r B=%r", a[:200], b[:200])
            return None                                   # readings disagree on a number
        toks = _TOK.findall(a.lower())
        page = set(_TOK.findall(page_text.lower()))
        if toks and sum(t in page for t in toks) / len(toks) >= self.min_overlap:
            self.stats["rejected_overlap"] += 1
            return None                                   # already in the text layer
        self.stats["accepted"] += 1
        return _WS.sub(" ", a).strip()

    def page_unreadable(self, key: str) -> None:
        """A page answer that is not cached could not be fetched because the page could not be
        rendered (no pdftoppm / pdftoppm error): counted like an offline miss (a degraded,
        machine-dependent build), never raised."""
        with self._lock:
            self._missed.add(f"page-{key}")
            self.stats["offline_miss"] = len(self._missed)
            self.stats["page_render_failed"] += 1

    def page_cached(self, key: str) -> bool:
        f = self.cache_dir / f"page-{key}.json"
        if not f.exists():
            return False
        rec = json.loads(f.read_text(encoding="utf-8"))
        return rec.get("model") == self.model and rec.get("page_prompt_version") == PAGE_PROMPT_VERSION

    def page_pairs(self, page_png: bytes, key: str, page_texts: list[str]) -> list[str] | None:
        """A rendered page → `LABEL: VALUE` lines, accepted only if the multiset of numbers in the
        answer equals the multiset of numbers in the page's text layer (so every pair is anchored
        to numbers that are really printed there). Lines lacking a label or a value are not
        emitted. Cached as `page-<key>.json`."""
        f = self.cache_dir / f"page-{key}.json"
        rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if rec.get("model") != self.model or rec.get("page_prompt_version") != PAGE_PROMPT_VERSION:
            if not self.allow_api:
                self._offline_miss(f"page-{key}")
                return None
            self._reserve(1)
            self.stats["page_sent"] += 1
            ans = self._call(page_png, PAGE_PROMPT)
            rec = {"model": self.model, "page_prompt_version": PAGE_PROMPT_VERSION, "answer": ans,
                   "image_sha256": hashlib.sha256(page_png).hexdigest()}
            _write_json(f, rec)
        want = _numbers(" ".join(page_texts))
        got = _numbers(rec["answer"])
        if not want or got != want:
            self.stats["page_rejected"] += 1
            log.info("page vision rejected %s: missing %s extra %s", key,
                     dict(want - got), dict(got - want))
            return None
        self.stats["page_accepted"] += 1
        out = []
        for ln in rec["answer"].splitlines():
            if ":" not in ln:
                continue
            label, value = (_WS.sub(" ", x).strip() for x in ln.rsplit(":", 1))
            if _DIGIT.search(label) and not _DIGIT.search(value):
                label, value = value, label             # the model wrote "VALUE: LABEL"
            if not (label and _VALUE.match(value)):
                continue                                # value: a number plus at most two unit words
            span = _snap_label(label, page_texts)
            if span:
                out.append(f"{span}: {value}")        # label = verbatim text-layer span, never the model's
            else:
                self.stats["page_pairs_dropped"] += 1
        return out


# ---------------------------------------------------------------- ODL driver

def _convert(path: Path, out_dir: str, pcfg: dict) -> None:
    import opendataloader_pdf

    env_path = os.environ.get("PATH", "")
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        os.environ["PATH"] = f"{Path(java_home) / 'bin'}{os.pathsep}{env_path}"
    try:
        opendataloader_pdf.convert(
            input_path=[str(path)], output_dir=out_dir, format="json", quiet=True,
            image_output="external", image_format="png", reading_order="xycut",
            table_method=pcfg.get("table_method", "default"), include_header_footer=False,
            keep_line_breaks=True, space_ratio=str(pcfg.get("space_ratio", 0.08)))
    finally:
        os.environ["PATH"] = env_path


def insert_page_pairs(paras: list[Para], extra: dict[int, list[str]]) -> list[Para]:
    """Add `[расм-жадвал] LABEL: VALUE` paragraphs after the last paragraph of their page."""
    out = list(paras)
    for page in sorted(extra, reverse=True):
        at = next((i + 1 for i in range(len(out) - 1, -1, -1) if (out[i].page or 0) <= page), 0)
        out[at:at] = [Para(f"[расм-жадвал] {ln}", None, page) for ln in extra[page]]
    return out


def render_page(pdf: Path, page: int, dpi: int) -> bytes | None:
    """One page as PNG via poppler's pdftoppm (None when it is not installed)."""
    exe = shutil.which("pdftoppm")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="page-") as tmp:
        subprocess.run([exe, "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", "-singlefile",
                        str(pdf), f"{tmp}/p"], check=True, capture_output=True)
        return Path(f"{tmp}/p.png").read_bytes()


def read_page_pairs(vision, pdf: Path, page: int, dpi: int, key: str, texts: list[str]) -> list[str] | None:
    """Page-level vision for one page; renders only on a cache miss. With API calls off a miss goes
    straight to `vision.page_pairs` (offline miss: raised or counted, never silently skipped, and
    independent of pdftoppm). A render failure (no pdftoppm, pdftoppm error) skips the page with a
    warning instead of failing the build, and is counted as a degradation (`page_unreadable`).
    `vision` only needs `page_pairs`; `page_cached` / `allow_api` / `page_unreadable` are optional."""
    cached = getattr(vision, "page_cached", None)
    if cached is not None and cached(key):
        return vision.page_pairs(b"", key, texts)
    if not getattr(vision, "allow_api", True):
        return vision.page_pairs(b"", key, texts)       # records the offline miss
    try:
        png = render_page(pdf, page, dpi)
    except (subprocess.CalledProcessError, OSError) as e:
        png = None
        log.warning("page render failed for %s p.%d: %r; page vision skipped", pdf, page, e)
    if png is None:
        if not shutil.which("pdftoppm"):
            log.warning("pdftoppm not found: page vision skipped for %s p.%d", pdf, page)
        if hasattr(vision, "page_unreadable"):
            vision.page_unreadable(key)
        return None
    return vision.page_pairs(png, key, texts)


def read_pdf(path: Path, vision: Callable[[bytes, str], str | None] | None = None) -> list[Para]:
    """ODL local conversion in a temp dir; image elements → `vision(image_bytes, page_text)`.
    Pages whose infographic numbers stay unlabelled after `attach_numbers` (≥ page_min_numbers
    number-only boxes) are also read whole by `vision.page_pairs`.
    Raises PdfEngineError when Java / ODL fail (the source PDF is never modified)."""
    pcfg = config.load("data").get("pdf", {})
    vcfg = pcfg.get("vision", {})
    min_side = int(vcfg.get("min_side_px", 120))
    max_repeat = int(vcfg.get("max_repeat_pages", 5))
    stats = getattr(vision, "stats", Counter())
    with tempfile.TemporaryDirectory(prefix="odl-") as tmp:
        try:
            _convert(Path(path), tmp, pcfg)
            outs = sorted(Path(tmp).glob("*.json"))
            if len(outs) != 1:
                raise RuntimeError(f"expected one JSON output, got {len(outs)}")
            doc = json.loads(outs[0].read_text(encoding="utf-8"))
        except Exception as e:   # FileNotFoundError (no java), CalledProcessError, bad output
            raise PdfEngineError(f"opendataloader-pdf failed on {path}: {e!r}") from e
        elements = flatten_elements(doc)

        page_text: dict[int, list[str]] = defaultdict(list)
        for e in elements:
            if e["type"] != "image" and e["text"]:
                page_text[e["page"]].append(e["text"])
        blobs: dict[str, bytes] = {}
        pages_of_hash: dict[str, set[int]] = defaultdict(set)
        for e in elements:
            if e["type"] == "image" and e["image"] and (Path(tmp) / e["image"]).is_file():
                b = (Path(tmp) / e["image"]).read_bytes()
                blobs[e["image"]] = b
                pages_of_hash[hashlib.sha256(b).hexdigest()].add(e["page"])

        def eligible(b: bytes) -> bool:
            size = png_size(b)
            return not (size and min(size) < min_side) and \
                len(pages_of_hash[hashlib.sha256(b).hexdigest()]) < max_repeat

        if vision is not None and hasattr(vision, "prefetch"):
            vision.prefetch([b for b in blobs.values() if eligible(b)],
                            workers=int(vcfg.get("workers", 8)))

        def image_text(e: dict) -> str | None:
            stats["images"] += 1
            b = blobs.get(e["image"] or "")
            if vision is None or b is None:
                return None
            size = png_size(b)
            if size and min(size) < min_side:
                stats["skipped_small"] += 1
                return None
            if len(pages_of_hash[hashlib.sha256(b).hexdigest()]) >= max_repeat:
                stats["skipped_repeated"] += 1       # logo / ornament repeated through the report
                return None
            return vision(b, " ".join(page_text.get(e["page"], [])))

        paras = elements_to_paras(elements, image_text)

    extra: dict[int, list[str]] = {}
    min_nums = int(vcfg.get("page_min_numbers", 3))
    if vision is not None and hasattr(vision, "page_pairs") and vcfg.get("pages", True):
        loose = Counter(e["page"] for e in attach_numbers(elements) if _num_only(e))
        pdf_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        dpi = int(vcfg.get("page_dpi", 150))
        for page in sorted(p for p, n in loose.items() if n >= min_nums):
            stats["pages_candidates"] += 1
            key = f"{pdf_sha[:32]}-p{page}-{dpi}dpi"   # by source PDF + page: renderer-independent
            texts = [_SECTION_NO.sub("", e["text"]) if e["type"] == "heading" else e["text"]
                     for e in elements if e["page"] == page and e["type"] != "image" and e["text"]]
            pairs = read_page_pairs(vision, Path(path), page, dpi, key, texts)
            if pairs:
                extra[page] = pairs
    return insert_page_pairs(paras, extra)
