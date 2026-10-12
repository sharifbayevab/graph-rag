# graph/normalize.py
"""
Text/title normalization for cross-script entity resolution.

Design notes (grounded in adversarial review of the real corpus):
  - Quotes appear as « » in only ~81/294 files and as “ ” „ in ~221; some mix.
    -> treat all of them as one class.
  - Apostrophe glyphs (‘ ’ ʻ ʼ ` ´ and the Cyrillic ў/ғ markers) are folded to a
    SINGLE canonical glyph but NOT deleted: in Uzbek Latin g‘ (ғ) ≠ g and o‘ (ў) ≠ o,
    so deleting them would cause false merges.
  - Transliteration is Cyrillic -> Latin ONLY (near-deterministic). Latin -> Cyrillic
    is ambiguous (sh/ch/yo/ng) and is never attempted.
  - Distinctions preserved: ў/у (oʼ/u), ғ/г (gʼ/g), қ/к (q/k), х/ҳ (x/h).
  - Header words are often letter-spaced ("Х У Л О С А", "Қ А Р О Р") -> collapsed.
"""
import re
import unicodedata

# Canonical apostrophe (Uzbek tutuq belgisi). Folded-to, never deleted.
APOS = "ʼ"
_APOS_GLYPHS = "‘’ʻʼ`´′‛"  # ‘ ’ ʻ ʼ ` ´ ′ ‛

# All quotation marks used around titles -> stripped for the key.
_QUOTES = "«»“”„‟″\"'"

# Uzbek Cyrillic -> Latin. Order matters: multi-char Cyrillic first is N/A here
# (all keys are single chars), but ў/ғ/қ/ҳ must map before plain г/к/х/у.
_CYR2LAT = {
    "ё": "yo", "ж": "j", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ю": "yu", "я": "ya", "ў": "o" + APOS, "ғ": "g" + APOS, "қ": "q",
    "ҳ": "h", "х": "x", "ъ": APOS, "ь": "",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "ы": "i",
    "э": "e",
}

# Uzbek-specific Cyrillic letters: their presence => uz-Cyrl (not Russian).
_UZ_CYR_MARKERS = set("ўқғҳ")
# Letters/words that mark Russian rather than Uzbek Cyrillic.
_RU_MARKERS = set("ыэёъ")
_RU_WORDS = ("постановление", "закон", "года", "республики", "решение", "статьи")

_CYR_RANGE = re.compile(r"[Ѐ-ӿ]")
_LAT_RANGE = re.compile(r"[A-Za-z]")


def _fold_apostrophes(s: str) -> str:
    for g in _APOS_GLYPHS:
        s = s.replace(g, APOS)
    return s


def cyr_to_lat(s: str) -> str:
    """Transliterate Cyrillic chars to Latin; leave Latin/other chars as-is."""
    out = []
    for ch in s:
        low = ch.lower()
        if low in _CYR2LAT:
            rep = _CYR2LAT[low]
            out.append(rep.upper() if ch.isupper() else rep)
        else:
            out.append(ch)
    return "".join(out)


def collapse_letter_spacing(s: str) -> str:
    """'Х У Л О С А' -> 'ХУЛОСА'; tolerant of multiple/odd spaces between letters."""
    # collapse single spaces sitting between two single letters, repeatedly
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = re.sub(r"(?<=\b\w)\s+(?=\w\b)", "", cur)
    return cur


def detect_language(text: str) -> str:
    """Return one of: uz-Cyrl, uz-Latn, ru, unknown."""
    sample = text[:4000]
    cyr = len(_CYR_RANGE.findall(sample))
    lat = len(_LAT_RANGE.findall(sample))
    low = sample.lower()
    if cyr == 0 and lat == 0:
        return "unknown"
    if cyr > lat:
        if _UZ_CYR_MARKERS & set(low):
            return "uz-Cyrl"
        ru_hits = sum(low.count(w) for w in _RU_WORDS) + len(_RU_MARKERS & set(low))
        return "ru" if ru_hits >= 2 else "uz-Cyrl"
    return "uz-Latn"


# Boilerplate prefix present in nearly every title; carries no discriminating signal.
_BOILER_PREFIX = re.compile(
    r"^(o" + APOS + r"?zbekiston respublikasi(ning)?\s+)", re.IGNORECASE
)


def skeleton_key(title: str, drop_boilerplate: bool = True) -> str:
    """
    Canonical, script-agnostic matching key for a title.

    Pipeline: NFKC -> fold apostrophes -> Cyrillic->Latin -> collapse letter-spacing
              -> strip quotes -> lowercase -> drop boilerplate prefix
              -> collapse whitespace/punctuation.
    """
    if not title:
        return ""
    s = unicodedata.normalize("NFKC", title)
    s = _fold_apostrophes(s)
    s = cyr_to_lat(s)
    s = collapse_letter_spacing(s)
    s = "".join(" " if c in _QUOTES else c for c in s)
    s = s.lower()
    # keep letters, digits, the canonical apostrophe and spaces; rest -> space
    s = re.sub(r"[^0-9a-z" + APOS + r"\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if drop_boilerplate:
        s = _BOILER_PREFIX.sub("", s).strip()
    return s


def target_key(name: str) -> str:
    """
    Stronger key for amendment/reference TARGETS (codes & standing laws), so the
    same code in Cyrillic/Latin and across case endings collapses to one node:
    e.g. 'Ўзбекистон Республикасининг Жиноят кодексига' and 'Jinoyat kodeksi'
    both -> 'jinoyat kodeks'.
    """
    k = skeleton_key(name, drop_boilerplate=True)
    # strip the edition-citation wrapper so versioned references collapse to the
    # canonical code identity: "...YYYY ... ОRQ-NNN ... (yangi tahrirda) tasdiqlangan X"
    # -> "X". The edition's date/number is preserved separately in target_editions.
    k = re.sub(r".*\btasdiqlangan\s+", "", k)
    # remove the "Republic of Uzbekistan" boilerplate wherever it appears (not just prefix)
    k = re.sub(r"o" + APOS + r"?zbekiston respublikasi(ning)?", " ", k)
    k = re.sub(r"kodeks\w*", "kodeks", k)            # kodeksiga/kodeksining/kodeksi -> kodeks
    k = re.sub(r"to" + APOS + r"?g" + APOS + r"?risida\w*", "togrisida", k)
    k = re.sub(r"\b(gi|qonuni|qonun|qonuniga|haqida)\b", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k
