# graph/parse.py
"""
Deterministic, header-anchored field extraction for one logical document.

Precision rules (from adversarial review of the real corpus):
  - The document's OWN metadata (date, body, title) is read ONLY from the header
    block. The body cites many repealed/amended acts with their own dates/titles
    ("citation contamination"); those are handled later as relations, never here.
  - doc_type is derived from the header text, NOT from the folder (folders are
    mislabeled: a QONUNLAR file may actually be a xulosa or a Russian postanovlenie).
  - is_draft comes from a 'Лойиҳа'/'Loyiha' marker in the body/header, not filename.
  - All quote glyphs « » “ ” „ " are one class.
  - Date placeholders (___-sonli, '2025-yil - -sonli') => date_unfilled, not a guess.
  - doc_number is captured but treated as UNTRUSTED (template artifacts exist).
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List

from graphrag.data.legacy.normalize import skeleton_key, detect_language, collapse_letter_spacing
from graphrag.data.legacy.extract_text import header_block, find_type_headers, looks_multi_document

_QUOTE_OPEN = "«“„"
_QUOTE_CLOSE = "»”"
# Any quoted span (greedy-safe, non-nested).
_QUOTED = re.compile(r"[«“„\"]([^«»“”„\"]{3,300}?)[»”\"]")

# Relator words that follow a quoted act title (the doc is ABOUT that act).
_RELATOR = r"(?:ги|gi)?\s*(?:ўзбекистон\s+республикаси|o" + "ʼ" + r"?zbekiston)?\s*(?:қонуни|qonuni|кодекси|kodeksi)?\s*(?:юзасидан|yuzasidan|ҳақида|haqida|бўйича|bo" + "ʼ" + r"?yicha)"

_DRAFT = re.compile(r"(?:^|\n)\s*(лойиҳа|loyiha)\s*(?:\n|$)", re.IGNORECASE)

# Omnibus ("amendments to CERTAIN legislative acts") — ambiguous title, never auto-merge.
_RX_OMNIBUS = re.compile(
    r"айрим\s+қонун\s+ҳужжатлари|ayrim\s+qonun\s+hujjatlari", re.IGNORECASE)

# Explicit "consists of N chapters and M articles" — authoritative structural count
# (anchored on 'иборат/iborat' so cited article numbers from other codes don't match).
# Anchor on 'M moddadan iborat' (so cited article numbers from other codes don't match),
# then look back a short window for chapters (боб), sections (бўлим) and a preamble word.
_RX_ART_ONLY = re.compile(
    r"(\d+)\s*(?:та|ta)?\s*(?:модда|modda)\w*\s*(?:дан|dan)?\s+(?:иборат|iborat)", re.IGNORECASE)
_RX_BOB = re.compile(r"(\d+)\s*(?:та|ta)?\s*(?:боб|bob)\b", re.IGNORECASE)
_RX_BOLIM = re.compile(r"(\d+)\s*(?:та|ta)?\s*(?:бўлим|bo[‘'`ʻ]?lim)", re.IGNORECASE)
_RX_PREAMBLE = re.compile(r"муқаддима|преамбула|muqaddima|preambula", re.IGNORECASE)


def _structure(text: str):
    """(chapters, articles, sections, has_preamble) from the explicit
    'N bob/bo'lim va M moddadan iborat' statement (None where not stated)."""
    a = _RX_ART_ONLY.search(text)
    if not a:
        return (None, None, None, False)
    w = text[max(0, a.start() - 160):a.start()]
    bob = _RX_BOB.search(w)
    bolim = _RX_BOLIM.search(w)
    pre = bool(_RX_PREAMBLE.search(text[max(0, a.start() - 200):a.end()]))
    return (int(bob.group(1)) if bob else None, int(a.group(1)),
            int(bolim.group(1)) if bolim else None, pre)


# Standalone numbered "N-боб" / "N-модда" headings (for counting a law text's own structure).
_RX_BOB_NUM = re.compile(r"(?:^|\n)\s*(\d+)\s*[-–—]?\s*(?:боб|bob)\b", re.IGNORECASE)
_RX_ART_NUM = re.compile(r"(?:^|\n)\s*(\d+)\s*[-–—]?\s*(?:модда|modda)\b", re.IGNORECASE)


def _contiguous_from_one(nums):
    """Largest N such that 1..N are all present — the law's own article/chapter count, robust
    against a stray cited number (e.g. a '730-модда' reference from another code)."""
    present = set(nums)
    k = 0
    while (k + 1) in present:
        k += 1
    return k or None


def _law_text_structure(text: str):
    """(chapters, articles) for a self-contained law text, from its own sequential headings."""
    return (_contiguous_from_one(int(x) for x in _RX_BOB_NUM.findall(text)),
            _contiguous_from_one(int(x) for x in _RX_ART_NUM.findall(text)))
# A single code targeted by an amendment, named in the title.
_RX_CODE_TARGET = re.compile(
    r"([A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ’ʼ‘`\-]+(?:\s+[A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ’ʼ‘`\-]+){0,4}?\s*"
    r"(?:кодекси|кодексига|kodeksi|kodeksiga))", re.IGNORECASE)

# Adopting/issuing body markers.
_RX_SENATE = re.compile(r"\bсенат|\bsenat", re.IGNORECASE)
_RX_CHAMBER = re.compile(r"қонунчилик\s+палатаси|qonunchilik\s+palatasi", re.IGNORECASE)
_RX_COMMITTEE = re.compile(r"қ[ўу]\s*мита|qo" + "ʼ" + r"?mita|q[oo" + "ʼ" + r"]?mita", re.IGNORECASE)

# Document number (UNTRUSTED). Captures СҚ-/ЎРҚ-/PF-/PQ- styles and -sonli.
_RX_DOCNUM = re.compile(
    r"(?:№\s*)?((?:СҚ|ЎРҚ|ПҚ|ПФ|PF|PQ|QQ|O" + "ʼ" + r"RQ)\s*[–\-]\s*\d+[–\-]?[IVXLC]*)",
    re.IGNORECASE,
)

# Date placeholders -> unfilled.
_RX_PLACEHOLDER = re.compile(r"_{2,}|-\s*-\s*sonli|\.{3,}|—\s*—", re.IGNORECASE)

_MONTHS = {
    "январ": 1, "yanvar": 1, "феврал": 2, "fevral": 2, "март": 3, "mart": 3,
    "апрел": 4, "aprel": 4, "май": 5, "may": 5, "июн": 6, "iyun": 6,
    "июл": 7, "iyul": 7, "август": 8, "avgust": 8, "сентябр": 9, "sentyabr": 9,
    "sentabr": 9, "октябр": 10, "oktyabr": 10, "oktabr": 10, "ноябр": 11,
    "noyabr": 11, "декабр": 12, "dekabr": 12,
}
# YYYY (йил|yil) DD <month> ; day may be attached (31-yanvarda) or spaced.
_RX_DATE = re.compile(
    r"(\d{4})\s*[-\s]?\s*(?:йил|yil)\s+(\d{1,2})[-\s]+([A-Za-zА-Яа-яЁёЎўҚқҒғҲҳ]+)",
)


@dataclass
class ParsedDoc:
    doc_type: str = "unknown"
    is_draft: Optional[bool] = None
    language: str = "unknown"
    title_raw: Optional[str] = None
    title_norm: Optional[str] = None
    quoted_act_title_raw: Optional[str] = None
    quoted_act_title_norm: Optional[str] = None
    doc_date: Optional[date] = None
    doc_date_unfilled: bool = False
    issuing_body: str = "unknown"
    committee: Optional[str] = None
    doc_number: Optional[str] = None
    is_multi_document: bool = False
    is_omnibus: bool = False
    n_chapters: Optional[int] = None
    n_articles: Optional[int] = None
    n_sections: Optional[int] = None
    has_preamble: bool = False
    amend_target_raw: Optional[str] = None
    amend_target_norm: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# Degenerate "titles": boilerplate that carries no act identity. Creating an act
# from one of these would silently fuse many unrelated documents (review finding #1).
_DEGENERATE_TITLES = {
    "", "qonuni", "qonun", "qarori", "qaror", "xulosa", "qonun loyihasi",
    "loyiha", "ozbekiston respublikasining qonuni", "qonun matni",
    "respublikasi qonuni", "qonun hujjati",
}


def _is_degenerate(norm: Optional[str]) -> bool:
    if not norm:
        return True
    n = norm.strip()
    return n in _DEGENERATE_TITLES or len(n) < 8 or len(n.split()) < 2


def _month_num(word: str) -> Optional[int]:
    w = word.lower().strip("-—. ")
    for stem, num in _MONTHS.items():
        if w.startswith(stem):
            return num
    return None


def _parse_date(text: str) -> Optional[date]:
    m = _RX_DATE.search(text)
    if not m:
        return None
    year, day, month_word = int(m.group(1)), int(m.group(2)), m.group(3)
    mon = _month_num(month_word)
    if not mon or not (1 <= day <= 31) or not (1900 <= year <= 2100):
        return None
    try:
        return date(year, mon, day)
    except ValueError:
        return None


_RX_ARTICLE_LINE = re.compile(r"(?:^|\n)\s*\d+\s*[-–—]?\s*(?:модда|modda|статья|статьи)\b", re.IGNORECASE)


def _has_article_body(full: str) -> bool:
    """A real law_text carries the law's article body (several 'N-модда' lines); a short
    decree that merely quotes a law title (e.g. a Qaror adopting it) does not."""
    return len(_RX_ARTICLE_LINE.findall(full or "")) >= 3


def _detect_doc_type(section: str, header: str, full: str, filename: str) -> str:
    dt = _detect_doc_type_raw(section, header, full, filename)
    # guard against decrees/conclusions that quote a law title being mistyped as the law itself
    if dt == "law_text" and section != "STENOGRAMMA" and not _has_article_body(full):
        fn = filename.lower()
        if "хулоса" in fn or "xulosa" in fn:
            return "committee_conclusion"
        return "resolution"
    return dt


def _detect_doc_type_raw(section: str, header: str, full: str, filename: str) -> str:
    if section == "STENOGRAMMA":
        return "stenogram"
    low = full[:2500].lower()
    if "постановление" in low:
        return "postanovlenie"
    # first standalone type-header line in the document wins
    hits = find_type_headers(full[:6000])
    if hits:
        first = hits[0][0]
        if first == "committee_conclusion":
            return "committee_conclusion"
        if first == "resolution":
            return "resolution"
        if first == "law":
            return "law_text"
        if first in ("postanovlenie", "stenogram"):
            return first
    h = collapse_letter_spacing(header).lower()
    fn = filename.lower()
    # fallbacks via header keywords / filename
    if "маъруза" in h or "ma_ruza" in fn or "maruza" in fn or "ma'ruza" in fn:
        return "report"
    if "маълумотнома" in h or "маълумот" in h or "ma_lumot" in fn or "malumot" in fn:
        return "note"
    if "йўл харитаси" in h or "йўл харита" in h or "yol xaritasi" in fn or "yo`l xaritasi" in fn:
        return "roadmap"
    if re.search(r"\bилова\b|\bilova\b", h):
        return "annex"
    if re.search(r"\bқонуни\b|\bqonuni\b", h) or "қонун матни" in fn or "qonun matni" in fn:
        return "law_text"
    if "закон" in low and "республик" in low:
        return "law_text"  # Russian-language law text
    # last-resort fallback by filename keyword (lower confidence; filenames can lie)
    if "хулоса" in fn or "xulosa" in fn:
        return "committee_conclusion"
    if "қарор" in fn or "qaror" in fn or "сўров" in fn or "so`rov" in fn or "so‘rov" in fn:
        return "resolution"
    if "маълумот" in fn or "ma_lumot" in fn or "malumot" in fn:
        return "note"
    if "қонун" in fn or "qonun" in fn or "loyiha" in fn or "лойиҳа" in fn:
        return "law_text"
    return "unknown"


def _issuing_body(doc_type: str, header: str) -> tuple:
    committee = None
    if _RX_COMMITTEE.search(header):
        committee = _extract_committee(header)
    if doc_type == "committee_conclusion":
        return "committee", committee
    if _RX_SENATE.search(header):
        return "senate", committee
    if _RX_CHAMBER.search(header):
        return "legislative_chamber", committee
    if committee:
        return "committee", committee
    return "unknown", committee


def _extract_committee(header: str) -> Optional[str]:
    m = re.search(r"([А-ЯЁЎҚҒҲ][^\n]{0,80}?қ[ўу]митаси)", header)
    if m:
        return collapse_letter_spacing(m.group(1)).strip()
    m = re.search(r"([A-Z][^\n]{0,80}?qo" + "ʼ" + r"?mitasi)", header, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _best_quoted_title(header: str) -> Optional[str]:
    """Pick the quoted span that looks like an act title (longest with a law cue)."""
    cands = [c.strip() for c in _QUOTED.findall(header)]
    cands = [c for c in cands if len(c) >= 6]
    if not cands:
        return None
    cued = [c for c in cands if re.search(
        r"тўғрисида|to" + "ʼ" + r"?g" + "ʼ" + r"?risida|тасдиқлаш|tasdiqlash|кодекс|kodeks|конвенци|konvensi",
        c, re.IGNORECASE)]
    pool = cued or cands
    return max(pool, key=len)


def parse_document(path: str, section: str, text: str, filename: str) -> ParsedDoc:
    pd = ParsedDoc()
    head = header_block(text)
    pd.language = detect_language(text)
    pd.is_multi_document = looks_multi_document(text)
    if pd.is_multi_document:
        pd.notes.append("multi_document_file")

    pd.doc_type = _detect_doc_type(section, head, text, filename)
    pd.is_draft = bool(_DRAFT.search(head) or _DRAFT.search(text[:1500]))

    pd.issuing_body, pd.committee = _issuing_body(pd.doc_type, head)

    # quoted act title (what a xulosa/qaror points at)
    q = _best_quoted_title(head)
    if q:
        pd.quoted_act_title_raw = q
        pd.quoted_act_title_norm = skeleton_key(q)

    # own title: for xulosa/qaror the subject IS the quoted act; for laws derive from header
    if pd.doc_type in ("committee_conclusion", "resolution", "postanovlenie") and q:
        pd.title_raw = q
    else:
        pd.title_raw = _law_own_title(head) or q
    if pd.title_raw:
        tn = skeleton_key(pd.title_raw)
        # drop boilerplate-only "titles" to avoid silently fusing unrelated docs
        if _is_degenerate(tn):
            pd.title_raw, pd.title_norm = None, None
            pd.notes.append("title_unreliable_needs_llm")
        else:
            pd.title_norm = tn
    if pd.quoted_act_title_norm and _is_degenerate(pd.quoted_act_title_norm):
        pd.quoted_act_title_raw, pd.quoted_act_title_norm = None, None

    # date from the header; fall back to the top of the body, because header_block can
    # be degenerate (e.g. a stray page-number "10" before the real title block, where the
    # adoption date "…YYYY йил DD <month> қабул қилинган" actually sits).
    if _RX_PLACEHOLDER.search(head):
        pd.doc_date_unfilled = True
    pd.doc_date = _parse_date(head) or _parse_date(text[:2000])
    if pd.doc_date is None and not pd.doc_date_unfilled and _RX_PLACEHOLDER.search(text[:2500]):
        pd.doc_date_unfilled = True

    m = _RX_DOCNUM.search(head) or _RX_DOCNUM.search(text[:2500])
    if m:
        pd.doc_number = re.sub(r"\s+", "", m.group(1)).upper()

    # structural counts from the explicit statement (full text, top-k independent)
    pd.n_chapters, pd.n_articles, pd.n_sections, pd.has_preamble = _structure(text)
    # a self-contained law text rarely states "N moddadan iborat"; count its own sequential
    # 1..N headings instead (skip amendment/omnibus laws, whose bodies cite foreign articles).
    if (pd.doc_type == "law_text" and pd.n_articles is None
            and not pd.is_omnibus and not pd.amend_target_norm):
        bobs, arts = _law_text_structure(text)
        if arts:
            pd.n_articles = arts
        if bobs and pd.n_chapters is None:
            pd.n_chapters = bobs

    # amendment target: omnibus (many) vs a single named code in the title.
    # Omnibus is decided ONLY from the title (the head contains citations that
    # mention "айрим қонун ҳужжатлари" without the doc itself being omnibus).
    title_for_target = pd.title_raw or pd.quoted_act_title_raw or ""
    if _RX_OMNIBUS.search(title_for_target):
        pd.is_omnibus = True
    else:
        cm = _RX_CODE_TARGET.search(title_for_target)
        if cm:
            pd.amend_target_raw = collapse_letter_spacing(cm.group(1)).strip()
            pd.amend_target_norm = skeleton_key(pd.amend_target_raw, drop_boilerplate=False)

    return pd


def _law_own_title(header: str) -> Optional[str]:
    """
    For a law text/draft, the title is the descriptive block ending in
    '...тўғрисида' that precedes the 'қабул қилинган' adoption line.
    """
    flat = re.sub(r"\s+", " ", header).strip()
    # strip leading page number / draft marker / "Ўзбекистон Республикасининг ҚОНУНИ"
    flat = re.sub(r"^\s*\d+\s+", "", flat)
    flat = re.sub(r"^(лойиҳа|loyiha)\s+", "", flat, flags=re.IGNORECASE)
    flat = re.sub(r"^(ўзбекистон\s+республикасининг\s+)?(қонуни|qonuni)\s+",
                  "", flat, flags=re.IGNORECASE)
    # cut at adoption / body / first-article markers — title precedes them
    cut = re.split(
        r"қабул\s+қилинган|qabul\s+qilingan|қонунчилик\s+палатаси|"
        r"qonunchilik\s+palatasi|1\s*-\s*модда|1\s*-\s*modda|1\s*-\s*боб|1\s*-\s*bob",
        flat, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # the operative "тўғрисида" is the LAST one (avoids the embedded
    # "...жавобгарлик тўғрисидаги кодексига..." trap)
    ends = list(re.finditer(
        r"тўғрисида|to" + "ʼ" + r"?g" + "ʼ" + r"?risida", cut, re.IGNORECASE))
    if ends:
        return cut[: ends[-1].end()].strip()[:400]
    return cut[:300].strip() or None if 8 <= len(cut) <= 300 else None
