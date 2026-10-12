"""Normalization (spec §2.2), in the spec's order:
  1 encoding → 2 dates → 3 numbering → 4 script → 5 service noise.

Display text (what citations quote) gets steps 1 and 5 only. The internal
representation `text_norm` (lexical index, lemmatization, embedding) gets all five:
ISO dates, canonical numbers, Uzbek Cyrillic transliterated to Latin, lower case.
The original is kept in DocRecord.text / Chunk.text.
"""
from __future__ import annotations

import re
import unicodedata

from graphrag.data.extract import Para
from graphrag.data.legacy.normalize import _fold_apostrophes, cyr_to_lat, detect_language
from graphrag.data.legacy.parse import _month_num

# ── 1. encoding ──────────────────────────────────────────────────────────────
_DROP = dict.fromkeys(map(ord, "\x00\xad\u200b\u200c\u200d\ufeff"), None)
_SPACES = re.compile(r"[ \t\xa0\u2007\u2009\u202f\u3000]+")


def clean_encoding(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_DROP)
    s = _fold_apostrophes(s)
    return _SPACES.sub(" ", s).strip()


# ── 2. dates → ISO ───────────────────────────────────────────────────────────
_RX_DATE_UZ = re.compile(
    r"(\d{4})\s*[-\s]?\s*(?:йил|yil)\s+(\d{1,2})\s*[-\s]?\s*([A-Za-zА-Яа-яЎўҚқҒғҲҳʼ]+)", re.IGNORECASE)
_RX_DATE_DMY = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_RU_MONTHS = ["январ", "феврал", "март", "апрел", "ма[йяе]", "июн", "июл", "август",
              "сентябр", "октябр", "ноябр", "декабр"]
# a month word, not a prefix: stem + at most one case ending ("мая", "марта", "августе"; not "майора")
_RX_DATE_RU = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_RU_MONTHS) + r")[ьяеа]?(?![а-яё])\s+(\d{4})",
                         re.IGNORECASE)


def _iso(y: int, m: int | None, d: int) -> str | None:
    if not m or not (1 <= m <= 12) or not (1 <= d <= 31) or not (1900 <= y <= 2100):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def _ru_month(word: str) -> int | None:
    w = word.lower()
    return next((i + 1 for i, p in enumerate(_RU_MONTHS) if re.fullmatch(p, w)), None)


def iso_dates(s: str) -> str:
    s = _RX_DATE_UZ.sub(lambda m: _iso(int(m[1]), _month_num(m[3]), int(m[2])) or m[0], s)
    s = _RX_DATE_DMY.sub(lambda m: _iso(int(m[3]), int(m[2]), int(m[1])) or m[0], s)
    s = _RX_DATE_RU.sub(lambda m: _iso(int(m[3]), _ru_month(m[2]), int(m[1])) or m[0], s)
    return s


# ── 3. numbering → canonical ─────────────────────────────────────────────────
_SUP_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


_RX_ROMAN_VALID = re.compile(r"C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")   # 1..399, canonical form


def _is_roman(s: str) -> bool:
    """A well-formed numeral ("XIV"), not just roman letters ("IIII", "VX", "IC")."""
    return bool(s) and bool(_RX_ROMAN_VALID.fullmatch(s))


def _roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        total += -v if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v else v
    return total


def canon_num(raw: str) -> str:
    """'122¹' → '122.1'; 'IV' → '4'; 'а)' → 'a'; '(3)' → '3'."""
    s = raw.strip().strip("().[]§ ")
    m = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", s)
    if m:
        return f"{m[1]}.{m[2].translate(_SUP_TO_DIGIT)}"
    if _is_roman(s):
        return str(_roman_to_int(s))
    if len(s) == 1 and s.isalpha():
        return cyr_to_lat(s.lower())
    return s.lower()


_RX_SUP_NUM = re.compile(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
# the unit word is a whole word or an inflected one ("бобида", "бўлимининг", "Боб"), never a
# prefix of another word ("VI bobur", "II bobo")
_RX_ROMAN_UNIT = re.compile(
    r"\b([IVXLC]+)\s*[-–]?\s*(?=(?i:боб|bob|бўлим|boʼlim)"
    r"(?:(?i:и|га|да|дан|нинг|ни|лар|i|ga|da|dan|ning|ni|lar)[^\W\d_]*)?(?![^\W\d_]))")
# U+2011 (non-breaking hyphen) is folded too: it never follows a digit in the indexed corpus, so
# the index is unchanged. U+2012 ‒ / U+2212 − are folded only before a unit word ("21‒moddasi",
# composite modda#21): the 12 indexed spans ("432‒II-сонли") are not unit words and keep the dash,
# and the index has 0 "N‒<unit>" spans, so this is a query-side change only (M2 sweep, parked (a)).
_RX_NUM_DASH = re.compile(r"(\d)\s*[-–—\u2011]\s*(?=[^\W\d_])")
_RX_UNIT_DASH = re.compile(r"(\d)\s*[\u2012\u2212]\s*(?=(?i:modda|bob|band|qism|модда|боб|банд|қисм))")


def canon_numbering(s: str) -> str:
    s = _RX_SUP_NUM.sub(lambda m: f"{m[1]}.{m[2].translate(_SUP_TO_DIGIT)}", s)
    s = _RX_ROMAN_UNIT.sub(lambda m: f"{_roman_to_int(m[1])}-" if _is_roman(m[1]) else m[0], s)
    s = _RX_UNIT_DASH.sub(r"\1-", s)
    return _RX_NUM_DASH.sub(r"\1-", s)


# ── 4. script → common internal representation ───────────────────────────────
_RX_ASCII_APOS = re.compile(r"(?<=[^\W\d_])'(?=[^\W\d_])")


_RX_SEGMENT = re.compile(r"((?<=[.!?;:])\s+|\n+)")       # sentence / paragraph separators
_RX_CYR = re.compile(r"[Ѐ-ӿ]")
_RX_CYR_WORD = re.compile(r"[Ѐ-ӿ]+")
_UZ_ONLY = set("ўқғҳЎҚҒҲ")                                 # Uzbek-only Cyrillic letters
_RU_ONLY = set("ыщЫЩ")                                     # absent from Uzbek Cyrillic (ъ, э, ё are not)
_RU_FUNC = {"во", "на", "не", "что", "по", "для", "от", "из", "за", "это", "как", "также",
            "при", "или", "со", "об", "но", "же", "бы", "мы", "вы", "он", "она", "они",
            "его", "их", "все", "был", "была", "были", "уже", "только", "если", "чтобы", "этого",
            "того", "который", "которые", "года", "году", "прошу", "уважаемые", "коллеги", "закон",
            "закона", "статья", "статьи"}
_RX_RU_END = re.compile(r"(?:ого|его|ение|ения|ции|ать|ять|ить|ский|ская|ское|ные|ный|ная|ной|ых|ую|ает|яет|ует|ают|яют|ется|ится|ются|ться)$")


# Uzbek evidence besides the Uzbek-only letters: function words and inflectional endings that
# Russian words do not have ("ва", "бузилишлар", "Қонуннинг")
_UZ_FUNC = {"ва", "билан", "учун", "ҳамда", "ёки", "бўйича", "тўғрисида", "ҳақида", "ушбу", "мазкур",
            "ҳам", "эса", "деб", "бу", "шу", "бўлган", "этиш", "қилиш"}
_RX_UZ_END = re.compile(r"(?:нинг|лар|ларни|ларга|ларда|лари|ган|ганлар|моқда|ади|адиган|иш|ишга|ишни|ида|идан|ига|дан|даги)$")


# Endings Russian words share with Uzbek ("инвалида", "книга", "создан"): they only reinforce Uzbek evidence the
# segment already has (M2 Task 0 minor, deferred to M3 query tests by Ruling R29)
_RX_UZ_END_AMBIG = re.compile(r"(?:ида|идан|ига|дан)$")


def _evidence(words: list[str]) -> tuple[int, int]:
    """(Russian, Uzbek) evidence: the number of words carrying Russian-only letters, a Russian
    function word or ending, resp. Uzbek-only letters, an Uzbek function word or an Uzbek-only ending;
    the ambiguous endings (_RX_UZ_END_AMBIG) count for Uzbek only next to other Uzbek evidence."""
    ru = sum(1 for w in words if _RU_ONLY & set(w) or w in _RU_FUNC or (len(w) > 4 and _RX_RU_END.search(w)))
    strong = sum(1 for w in words if _UZ_ONLY & set(w) or w in _UZ_FUNC
                 or (len(w) > 4 and _RX_UZ_END.search(w) and not _RX_UZ_END_AMBIG.search(w)))
    weak = sum(1 for w in words if len(w) > 4 and _RX_UZ_END_AMBIG.search(w)
               and not (_UZ_ONLY & set(w) or w in _UZ_FUNC))
    return ru, strong + (weak if strong else 0)


_QUOTE_PAIRS = {"«": "»", "“": "”", "„": "“"}


def _strip_quoted(seg: str) -> str:
    """seg with every quoted span blanked, nested quotes included («A «B» C» -> the whole span). A quote
    that is never closed quotes nothing: an unbalanced « must not hide the rest of the segment. ASCII "
    pairs toggle at depth 0 only. Closing is checked before opening („…“: “ closes „)."""
    spans, stack, ascii_open = [], [], None
    for i, ch in enumerate(seg):
        if stack and ch == _QUOTE_PAIRS[seg[stack[-1]]]:
            j = stack.pop()
            if not stack:
                spans.append((j, i + 1))
        elif ch in _QUOTE_PAIRS:
            stack.append(i)
        elif ch == '"' and not stack:
            if ascii_open is None:
                ascii_open = i
            else:
                spans.append((ascii_open, i + 1))
                ascii_open = None
    out = list(seg)
    for a, b in spans:
        out[a:b] = " " * (b - a)
    return "".join(out)


def segment_is_russian(seg: str, default_ru: bool) -> bool:
    """Script routing of one sentence/paragraph segment by comparing evidence (M2 sweep; M1 vetoed
    Russian on any Uzbek-only letter): Russian iff it has more Russian-evidence words than
    Uzbek-evidence words, so a Russian sentence quoting «Ўзбекистон Республикаси …» stays
    Russian and an Uzbek row with a loanword ("Шизофрения … ва … бузилишлар") stays Uzbek.
    Words inside quotes («» “” "") are cited titles / names and count only when the whole segment
    is quoted. No Cyrillic, no evidence or a tie → default. One-word segments ("С." initials, letter-spaced heads):
    Uzbek-only letters → Uzbek, Russian-only letters → Russian, else default."""
    if not _RX_CYR.search(seg):
        return default_ru
    # single letters are no evidence: initials ("С."), letter-spaced heads ("Х У Л О С А"), codes
    words = [w.lower() for w in _RX_CYR_WORD.findall(seg) if len(w) > 1]
    if len(words) < 2:
        if _UZ_ONLY & set(seg):
            return False
        if _RU_ONLY & set(seg):
            return True
        return default_ru
    outside = [w.lower() for w in _RX_CYR_WORD.findall(_strip_quoted(seg)) if len(w) > 1]
    ru, uz = _evidence(outside or words)          # a wholly quoted segment counts its own words
    return ru > uz if ru != uz else default_ru


def to_internal(s: str, lang: str) -> str:
    """Internal representation: ISO dates, canonical numbers, Latin for Uzbek, lower case.

    Script is routed per sentence / paragraph segment, not per document: a Russian segment
    inside an Uzbek document keeps Cyrillic (Russian lemmatization, no token inflation), an
    Uzbek Cyrillic segment is transliterated; `lang` decides ambiguous segments."""
    s = canon_numbering(iso_dates(s))
    default_ru = lang == "ru"
    parts = _RX_SEGMENT.split(s)
    out = []
    for i, seg in enumerate(parts):
        if i % 2 == 0 and seg and not segment_is_russian(seg, default_ru):
            seg = cyr_to_lat(seg)
            if lang != "en":                               # English "don't" keeps its apostrophe
                seg = _RX_ASCII_APOS.sub("ʼ", seg)          # Uzbek Latin o'z → oʼz (internal only)
        out.append(seg)
    return "".join(out).lower()


def query_internal(text: str, lang: str | None = None) -> str:
    """A query in the index's internal representation: the same §2.2 steps as indexed text
    (encoding clean-up, then `to_internal`), so query and index share one normalizer."""
    return to_internal(clean_encoding(text), lang or detect_language(text))


# ── 5. service noise (page numbers / footers left inside the body) ───────────
_PAGE_ONLY = re.compile(r"^(?:\d{1,3}|-\s*\d{1,3}\s*-|стр\.?\s*\d+|\d+\s*/\s*\d+)$", re.IGNORECASE)


_FIELD_CODE = re.compile(r"\s*\bPAGE\s*\\\*\s*MERGEFORMAT\s*\d*", re.IGNORECASE)  # Word field residue
_FIELD_ONLY = re.compile(r"^(?:PAGE|NUMPAGES|SECTIONPAGES)(?:\s*\\\*\s*MERGEFORMAT)?\s*\d*$")  # bare field


def drop_noise(paras: list[Para]) -> list[Para]:
    out = []
    for p in paras:
        t = p.text
        if _FIELD_CODE.search(t):
            t = _SPACES.sub(" ", _FIELD_CODE.sub("", t)).strip()
            p = Para(t, p.list_level, p.page, p.page_end)
        if t and not _PAGE_ONLY.match(t) and not _FIELD_ONLY.match(t):
            out.append(p)
    return out


def normalize_paragraphs(paras: list[Para]) -> list[Para]:
    """Display-level normalization: encoding (step 1) + service noise (step 5)."""
    return drop_noise([Para(clean_encoding(p.text), p.list_level, p.page, p.page_end) for p in paras])
