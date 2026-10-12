"""Lemmatization shared by indexing and query processing (spec §2.6).

Input is `text_norm` (see normalize.to_internal): Uzbek is Latin, Russian stays Cyrillic.
  - Russian: pymorphy3 normal form.
  - Uzbek:   lexicon-guided suffix stripping. All stems reachable by stripping
             inflectional suffixes are generated; the SHORTEST one attested as a word in
             the corpus lexicon wins. Unknown words fall back to greedy rule stripping.
  - Composite tokens make article/chapter numbers and dates exact-matchable and
    language-neutral: "21-moddasi" / "статья 21" / "ст. 21" → "modda#21"; "2024-01-19" → "date#…".
    Lists and ranges expand ("10-12-moddalar", "10, 11 va 12-moddalari", "статьями 5 и 6").
    Official-gazette citations ("…axborotnomasi, 1995 yil, № 1, 3-modda", "№ 11, ст. 790")
    are bulletin entry numbers, not articles: they become "gaz#N" and never "modda#N".
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

from graphrag import config

log = logging.getLogger(__name__)

# Uzbek inflection order is stem + plural + possessive + case, so suffixes are stripped from
# the end in slot order: at most one case suffix, then possessive, then plural.
_CASE = ["gacha", "ning", "dagi", "dan", "ga", "ka", "qa", "da", "ni"]
_POSS = ["si", "i"]
_PLURAL = ["lar"]
_SLOTS = [sorted(x, key=len, reverse=True) for x in (_CASE, _POSS, _PLURAL)]
_UZ_SUFFIXES = sorted({*_CASE, *_POSS, *_PLURAL}, key=len, reverse=True)
_VOWELS = set("aeiou")
_MIN_STEM = 3

STOP = {
    # uz
    "va", "bilan", "uchun", "ushbu", "bu", "shu", "ham", "yoki", "hamda", "esa", "deb", "edi",
    "bo", "ekan", "lekin", "biroq", "agar", "qaysi", "nima",
    # ru
    "и", "в", "во", "на", "по", "с", "со", "о", "об", "к", "от", "до", "из", "за", "для",
    "не", "что", "как", "а", "но", "или", "это", "его", "ее", "её", "их",
}

_WORD = re.compile(r"\w+")
_CYR = re.compile(r"[а-яё]")
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_NUM = r"\d+(?:\.\d+)?"
_UNITS = r"(modda|bob|band|qism)"
_ISSUE = r"№\s*\d+(?:\s*[/\-–‒−]\s*\d+)?(?:\s*[-–]\s*son)?"
# gazette citations: "№ 1, 3-modda", "№ 8, 800, 803-moddalar", "№ 11–12, 5-modda"; ru "№ 11, ст. 790"
_UZ_GAZ = re.compile(_ISSUE + rf"\s*,?\s*((?:{_NUM}\s*,\s*)*{_NUM})-modda")
_RU_GAZ = re.compile(_ISSUE + rf"\s*,\s*ст\.\s*((?:{_NUM}\s*,\s*)*{_NUM})(?![\d.])")
_UZ_RANGE = re.compile(rf"(?<![\d.\-])({_NUM})\s*[-–]\s*({_NUM})-{_UNITS}")
_UZ_LIST = re.compile(rf"(?<![\d.\-])((?:{_NUM}\s*,\s*)*{_NUM}\s*(?:,|\bva\b|\bhamda\b|\byoki\b)\s*)"
                      rf"({_NUM})-{_UNITS}lar")
_UZ_UNIT = re.compile(rf"({_NUM})-{_UNITS}")
_RU_UNIT = re.compile(
    r"\b((?:стать(?:я|и|е|ю|ей|ёй|ям|ями|ях)|статей|глав(?:а|ы|е|у|ой|ою|ам|ами|ах)?"
    r"|пункт(?:а|у|ом|е|ы|ов|ам|ами|ах)?|част(?:ь|и|ью|ей|ям|ями|ях))(?!\w)|ст\.|ч\.|п\.)"
    rf"\s*({_NUM}(?:\s*(?:,|\bи\b|\bили\b|[-–])\s*{_NUM})*)(?![\d.\-]*\d)")
_RU_UNIT_MAP = {"ст": "modda", "гл": "bob", "пу": "band", "п.": "band", "ча": "qism", "ч.": "qism"}
_RANGE_CAP = 50


def _can_strip(w: str, suf: str) -> bool:
    """Suffix `suf` may be removed from `w`: min stem length; -ka only after k, -qa only after q."""
    if not w.endswith(suf) or len(w) - len(suf) < _MIN_STEM:
        return False
    base = w[: -len(suf)]
    if suf == "ka":
        return base.endswith("k")
    if suf == "qa":
        return base.endswith("q")
    return True


def _lig(w: str) -> str:
    return w[:-1] + "k" if w.endswith("lig") else w


def rule_stem(w: str) -> str:
    """Greedy fallback: per slot (case, possessive, plural) strip the longest matching suffix;
    possessive -si only after a vowel (a consonant + ʼ such as gʼ does not count, oʼ does)."""
    for slot in _SLOTS:
        for suf in slot:
            if not _can_strip(w, suf):
                continue
            if suf == "si" and not (w[-3] in _VOWELS or w[-4:-2] == "oʼ"):
                continue
            w = w[: -len(suf)]
            break
    return _lig(w)


class UzStemmer:
    def __init__(self, vocab: dict[str, int] | None = None, min_freq: int = 2):
        self.vocab = {w for w, c in (vocab or {}).items() if c >= min_freq}
        self._cache: dict[str, str] = {}

    def candidates(self, w: str) -> set[str]:
        """All stems reachable by stripping ≤1 suffix per slot (case, possessive, plural)."""
        seen = {w}
        for slot in _SLOTS:
            seen |= {x[: -len(suf)] for x in seen for suf in slot if _can_strip(x, suf)}
        return seen | {_lig(x) for x in seen}

    def stem(self, w: str) -> str:
        if w not in self._cache:
            attested = [c for c in self.candidates(w) if c in self.vocab]
            self._cache[w] = min(attested, key=lambda c: (len(c), c)) if attested else rule_stem(w)
        return self._cache[w]


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def ru_lemma(w: str) -> str:
    return _morph().parse(w)[0].normal_form


def build_vocab(texts: Iterable[str]) -> dict[str, int]:
    """Latin (Uzbek) word frequencies over text_norm strings; digits and Cyrillic excluded."""
    c: Counter[str] = Counter()
    for t in texts:
        for w in _WORD.findall(t.lower()):
            if not w.isdigit() and not _CYR.search(w) and "_" not in w:
                c[w] += 1
    return dict(sorted(c.items()))


@lru_cache(maxsize=1)
def get_stemmer() -> UzStemmer:
    """Stemmer over the corpus lexicon (config data.yaml `lexicon`), cached per process.

    A rebuilt lexicon is only picked up after a process restart or `get_stemmer.cache_clear()`.
    If the lexicon file is missing, a warning is logged and rule-only stemming is used.
    """
    lx = config.load("data")["lexicon"]
    p = config.path(lx["path"])
    if p.exists():
        vocab = json.loads(p.read_text(encoding="utf-8"))
    else:
        log.warning("lexicon %s not found; Uzbek stemming falls back to rules only", p)
        vocab = {}
    return UzStemmer(vocab, lx["min_freq"])


def _span(a: str, b: str) -> list[str]:
    """Range a..b: integer ranges up to _RANGE_CAP are expanded, otherwise endpoints only."""
    if a.isdigit() and b.isdigit() and 0 < int(b) - int(a) <= _RANGE_CAP:
        return [str(i) for i in range(int(a), int(b) + 1)]
    return [a, b]


def _nums(listed: str) -> list[str]:
    """'5, 6 и 7' / '5-7' / '10, 11 va' → numbers, ranges expanded."""
    out: list[str] = []
    for m in re.finditer(rf"({_NUM})(?:\s*[-–]\s*({_NUM}))?", listed):
        out += _span(m.group(1), m.group(2)) if m.group(2) else [m.group(1)]
    return out


def _composites(s: str) -> list[str]:
    """Composite tokens; each matched span is blanked so later patterns cannot re-read it."""
    out: list[str] = []
    buf = [s]

    def take(rx: re.Pattern, emit) -> None:
        def sub(m: re.Match) -> str:
            out.extend(emit(m))
            return " " * len(m.group(0))
        buf[0] = rx.sub(sub, buf[0])

    take(_ISO, lambda m: [f"date#{m.group(1)}"])
    take(_UZ_GAZ, lambda m: [f"gaz#{n}" for n in _nums(m.group(1))])
    take(_RU_GAZ, lambda m: [f"gaz#{n}" for n in _nums(m.group(1))])
    take(_UZ_RANGE, lambda m: [f"{m.group(3)}#{n}" for n in _span(m.group(1), m.group(2))])
    take(_UZ_LIST, lambda m: [f"{m.group(3)}#{n}" for n in _nums(m.group(1)) + [m.group(2)]])
    take(_UZ_UNIT, lambda m: [f"{m.group(2)}#{m.group(1)}"])
    take(_RU_UNIT, lambda m: [f"{_RU_UNIT_MAP[m.group(1)[:2]]}#{n}" for n in _nums(m.group(2))])
    return out


def tokens(text_norm: str, stemmer: UzStemmer | None = None) -> list[str]:
    st = stemmer or get_stemmer()
    s = text_norm.lower()
    out = _composites(s)
    for w in _WORD.findall(s):
        if w in STOP or "_" in w or (len(w) < 2 and not w.isdigit()):
            continue
        if w.isdigit():
            out.append(w)
        elif _CYR.search(w):
            out.append(ru_lemma(w))
        else:
            out.append(st.stem(w))
    return out
