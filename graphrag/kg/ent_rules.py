"""Entity typing and context rules of Task 6 fix round 2 (Ruling R16), all patterns / word lists as data
in config/graph.yaml `dictionaries` (types, clause_final, context2, aliases).

* `head_type(key, typ, lang)`: the type decided by the key's head noun (last lemma for Uzbek; first, or
  second after an adjective, for Russian): organ / position / procedure heads, conditional heads,
  category heads (-> legal_category); None = drop.
* `allowed(key, typ, lang)`: whether a type is compatible with the head (verification retypes).
* `is_clause(name, lang)`: a verb-final name (finite verb / participle) is a clause.
* `split_coordinated(name, typ)`: "A, B ва C вазирликлари" -> the headed conjuncts; a fused list is never
  one entity.
* `Context2`: REGULATES context of a mention (owner before "ҳузуридаги", «…» деган сўзлар, boilerplate
  sentences, bare modifier use).
"""
from __future__ import annotations

import re
from functools import lru_cache

from graphrag import config
from graphrag.kg.concepts import WORD_RX, norm_key

_C: dict | None = None
_CYR = re.compile(r"[а-яё]", re.I)


def _cfg() -> dict:
    global _C
    if _C is None:
        d = config.load("graph")["dictionaries"]
        t = d["types"]
        c2 = d["context2"]
        _C = {"organ": set(t["organ"]["heads"]), "organ_ru": set(t["organ"]["ru"]),
              "organ_generic": set(t["organ"].get("generic_first") or []),
              "position": set(t["position"]["heads"]), "position_ru": set(t["position"]["ru"]),
              "position_cond": {k: list(v) for k, v in (t["position"].get("conditional") or {}).items()},
              "procedure": set(t["procedure"]["heads"]), "procedure_ru": set(t["procedure"]["ru"]),
              "procedure_tails": [tuple(x.split()) for x in t["procedure"].get("tails") or []],
              "procedure_cond": set(t["procedure"].get("conditional") or []),
              "category": set(t["category"]["heads"]), "category_ru": set(t["category"]["ru"]),
              "clause": re.compile("(?i)" + d["clause_final"]),
              "owner_after": re.compile("(?i)" + c2["owner_after"].lstrip("^")),
              "quoted_after": re.compile("(?i)" + c2["quoted_words_after"].lstrip("^")),
              "boiler": [re.compile("(?i)" + x) for x in c2["boilerplate"]],
              "modifier_after": re.compile("(?i)" + c2["modifier_after"].lstrip("^")),
              "act_noun": re.compile("(?i)" + c2["act_owner_noun"]),
              "act_window": int(c2.get("act_owner_window", 300)), "act_words": int(c2.get("act_owner_words", 8)),
              "drafted": re.compile("(?i)" + c2["drafted_by"]),
              "quantity": set(t["category"].get("quantity") or []),
              "activity": [tuple(x.split()) for x in t["category"].get("activity") or []],
              "nominal": re.compile(t["category"].get("nominalisation") or r"(?!x)x"),
              "abstract": set(t["category"].get("abstract") or []),
              "obligation": set(t["category"].get("obligation") or []),
              "member_heads": set(t["category"].get("member_heads") or []),
              "member_owners": set(t["category"].get("member_owners") or []),
              "nonstate": [tuple(x.split()) for x in t["category"].get("nonstate_markers") or []],
              "nonstate_heads": set(t["category"].get("nonstate_heads") or []),
              "object_verbs": [tuple(x.split()) for x in t["procedure"].get("object_verbs") or []],
              "objects": set(t["procedure"].get("objects") or []),
              "aliases": d.get("aliases") or [],
              "adopt_final": re.compile("(?i)" + c2["adoption_final"]),
              "adopt_agent": re.compile("(?i)" + c2["adoption_agent_after"].lstrip("^")),
              "adopt_session": re.compile("(?i)" + c2["adoption_session_after"].lstrip("^")),
              "adopt_before": re.compile("(?i)" + c2["adoption_before"]),
              "adopt_before_names": re.compile("(?i)" + c2["adoption_before_names"].lstrip("^")),
              "adopt_header": re.compile("(?i)" + c2["adoption_header"]),
              "adopt_header_chars": int(c2.get("adoption_header_chars", 160))}
    return _C


_SENT_END = re.compile(r"[.!?](?=\s|$)|\n")


def adoption_line(text: str, a: int, b: int) -> str | None:
    """Ruling R20 (b): the mention text[a:b] stands in an adoption / approval / signature line (legislative
    history, no REGULATES): "final_agent" = the agent ("X томонидан") of a sentence ending in "… қабул
    қилинди / маъқулланди / имзоланди"; "ru_participle" = a chamber right after the short participle "Принят / Одобрен / Подписан";
    "header" = inside a short header paragraph "X томонидан <date> қабул қилинган / маъқулланган",
    "Принят Законодательной палатой …"; "session_agent" (Ruling R29 B) = a chamber in the genitive before its
    plenary session ("Сенатининг 25 ялпи мажлисида …") in such a sentence; the sentence predicate also covers the
    history forms "… рад этилган." and "… имзоланган эди." (a converb "қабул қилиниб, …" before them). None: not such
    a line."""
    C = _cfg()
    lo = text.rfind("\n\n", 0, a)
    lo = 0 if lo < 0 else lo + 2
    hi = text.find("\n\n", b)
    hi = len(text) if hi < 0 else hi
    para = text[lo:hi].strip()
    if len(para) <= C["adopt_header_chars"] and C["adopt_header"].search(para):
        return "header"
    if C["adopt_before"].search(text[max(lo, a - 40):a]) and C["adopt_before_names"].match(text, a):
        return "ru_participle"
    ends = [m.end() for m in _SENT_END.finditer(text, b, hi)]
    se = ends[0] if ends else hi
    if C["adopt_final"].search(text[b:se]) and C["adopt_agent"].match(text, b):
        return "final_agent"
    if C["adopt_final"].search(text[b:se]) and re.search(r"(?i)(?:нинг|ning)$", text[a:b]) \
            and C["adopt_session"].match(text, b):
        return "session_agent"                           # Ruling R29 B
    return None


def is_ru_key(key: str) -> bool:
    return bool(_CYR.search(key))


def _cond_ok(head: str, lem: list[str]) -> bool:
    C = _cfg()
    need = C["position_cond"].get(head)
    if need is None:
        return True
    rest = set(lem[:-1])
    for x in need:
        if x == "@organ" and rest & C["organ"]:
            return True
        if x == "@position" and rest & C["position"]:
            return True
        if x in rest:
            return True
    return False


@lru_cache(maxsize=100_000)
def _ru_pos(lemma: str) -> str:
    from graphrag.data.lemma import _morph
    p = _morph().parse(lemma)[0].tag.POS
    return p or ""


def _ru_head(lem: list[str]) -> str:
    for x in lem[:3]:
        if _ru_pos(x) not in ("ADJF", "ADJS", "PRTF", "PRTS", "NUMR"):
            return x
    return lem[0]


_WEAK_PROC = {"tartib", "taomil", "rejim"}      # "…тартибда …" is no evidence of a named procedure


def head_type(key: str, typ: str) -> str | None:
    """The type decided by the head of `key` (current type `typ` breaks ties for legal_category)."""
    C = _cfg()
    lem = key.split()
    if not lem:
        return None
    if is_ru_key(key):
        h = _ru_head(lem)
        if h in C["organ_ru"]:
            return "organ"
        if h in C["position_ru"]:
            return "position"
        if h in C["procedure_ru"]:
            return "procedure"
        if h in C["quantity"] or h in C["obligation"] or h in C["abstract"]:
            return None
        if h in C["category_ru"]:
            return "legal_category"
        return "legal_category" if typ == "legal_category" else None
    h = lem[-1]

    def has_seq(seqs):
        return any(tuple(lem[i:i + len(q)]) == q for q in seqs for i in range(len(lem) - len(q) + 1))
    if h in C["organ"]:
        if len(lem) <= 3 and lem[0] in C["organ_generic"]:
            return None                                   # "манфаатдор ташкилотлар", "бошқа орган"
        if h in C["nonstate_heads"] and has_seq(C["nonstate"]):
            return "legal_category"                       # R17: "Ўзбекистон фермерлари кенгаши", "Савдо-саноат палатаси"
        return "organ"
    if h in C["member_heads"] and set(lem[:-1]) & (C["member_owners"] | C["organ"]):
        return "position"                                 # R17: "Ҳукумат аъзолари", "Сенат аъзоси"
    if h in C["position"] or (h in C["position_cond"] and _cond_ok(h, lem)):
        return "position"
    if h in C["procedure"] or any(tuple(lem[-len(t):]) == t for t in C["procedure_tails"]):
        return "procedure"
    if h in C["procedure_cond"] and len(lem) >= 2 and (lem[-2] in C["procedure"] or
                                                     set(lem[-4:-1]) & (C["procedure"] - _WEAK_PROC)):
        return "procedure"                                # a named-procedure noun right before / near the verb
    if any(tuple(lem[-len(v):]) == v for v in C["object_verbs"]) and set(lem[:-1]) & C["objects"]:
        return "procedure"                                # R17: "шикоятни иш юритишга қабул қилиш", "… кўриб чиқиш"
    if h in C["quantity"]:
        return None
    if h in C["category"]:
        return "legal_category"
    # R17: activities, verbal nouns, abstract heads and obligations are no legal categories
    if h in C["abstract"] or h in C["obligation"] or any(tuple(lem[-len(a):]) == a for a in C["activity"]) \
            or (len(h) > 5 and C["nominal"].search(h)):
        return None
    return "legal_category" if typ == "legal_category" else None


def allowed(key: str, typ: str) -> bool:
    """A verification retype to `typ` must agree with the head lists."""
    h = head_type(key, typ)
    return h == typ


def is_clause(name: str, lang: str = "uz") -> bool:
    words = WORD_RX.findall(name)
    if not words:
        return False
    w = words[-1].lower().translate(str.maketrans({c: "ʼ" for c in "ʻʼ'`‘’"}))
    if _CYR.search(w) and lang == "ru":
        from graphrag.data.lemma import _morph
        return _morph().parse(w)[0].tag.POS in ("VERB", "INFN")
    return bool(_cfg()["clause"].search(w))


_TERRITORIAL = {"туман", "шаҳар", "вилоят", "туманлараро", "tuman", "shahar", "viloyat", "tumanlararo"}
_SEP = re.compile(r"\s*,\s*|\s+(?:ва|va|ҳамда|hamda|и|ёки|yoki)\s+")


def _top_level_seps(name: str) -> list[tuple[int, int]]:
    out, depth = [], 0
    for m in _SEP.finditer(name):
        pre = name[:m.start()]
        depth = pre.count("(") - pre.count(")") + pre.count("«") - pre.count("»") + pre.count("“") - pre.count("”")
        if depth == 0:
            out.append((m.start(), m.end()))
    return out


_PARTICIPLE = re.compile(r"(?:ган|gan|ётган|ayotgan|диган|digan|адиган|adigan|ладиган|ladigan)$")


def _words_nopar(x: str) -> list[str]:
    return [w for w in re.sub(r"\([^()]*\)", " ", x).split() if WORD_RX.search(w)]


def head_level_fused(name: str) -> bool:
    """R17: a coordination at the HEAD level ("юридик ва жисмоний шахслар", "шуъба ва қарам хўжалик
    жамиятлари", "акциядорлари ва кредиторлари") -- not one inside a modifier ("Жамиятдан чиқарилган ёки
    ундан чиқиб кетган жамият иштирокчиси", "приобретатели акций, активов, прав и (или) обязательств")."""
    seps = _top_level_seps(name)
    if not seps:
        return False
    a, b = seps[-1]
    before = _words_nopar(name[:a])
    after = _words_nopar(name[b:])
    if not before or not after:
        return False
    bw = before[-1].lower()
    if _CYR.search(bw) and re.search(r"[ыщ]", name.lower()):       # Russian: head-first
        return _ru_pos(bw) in ("ADJF", "PRTF") and len(after) <= 2
    if _PARTICIPLE.search(bw):
        return False
    first_single = seps[0][0] == len(name[:seps[0][0]]) and len(_words_nopar(name[:seps[0][0]])) == 1
    return len(after) <= 2 or (first_single and len(after) <= 3)


def distribute_head(name: str) -> list[tuple[str, int | None]] | None:
    """R17: "юридик ва жисмоний шахслар" -> [("юридик шахслар", None), ("жисмоний шахслар", offset)]: one
    top-level separator, a one-word first conjunct and a >= 2-word second one share its last word (the head).
    Offset = where the second conjunct starts in `name` (None: a composed name)."""
    seps = _top_level_seps(name)
    if len(seps) != 1:
        return None
    a, b = seps[0]
    first = _words_nopar(name[:a])
    second = name[b:].strip()
    sw = _words_nopar(second)
    if len(first) != 1 or len(sw) < 2:
        return None
    fw = first[0].lower()
    adjectival = bool(re.search(r"(?:ий|iy|ик|ik|ли|li|даги|dagi|ган|gan|вий|viy)$", fw)) or \
        (_CYR.search(fw) and _ru_pos(fw) in ("ADJF", "PRTF"))
    if not adjectival:
        return None                               # "кластер ва фермер хўжаликлари": two nouns, no shared head
    
    return [(f"{first[0]} {sw[-1]}", None), (second, b + (len(name[b:]) - len(name[b:].lstrip())))]


def list_like(name: str, typ: str) -> bool:
    """A name ending in a conjunction (or, for procedures / legal categories, a genitive) is a fragment; a
    legal_category coordinated at the head level is a fused list (R17)."""
    last = (WORD_RX.findall(name) or [""])[-1].lower()
    if last in ("ва", "va", "ёки", "yoki", "ҳамда", "hamda", "и"):
        return True
    if typ in ("procedure", "legal_category") and re.search(r"(?:нинг|ning)$", last):
        return True
    if typ in ("procedure", "legal_category"):
        if name.lstrip()[:1] in "()" or _oblique_end(name):
            return True                           # R17: a fragment ("(ёки) …", "… ваколатларга")
    if typ == "legal_category":
        return head_level_fused(name)
    return False


def oblique_short(name: str) -> bool:
    """An inflected short name ("фуқароларга"): dropped as a fragment before verification (the verified
    inputs stay unchanged), restored afterwards when its entity survives verification (R17)."""
    return len(name.split()) < 5 and name.lstrip()[:1] not in "()" and _oblique_end(name)


_OBLIQUE = re.compile(r"(?<=\w{3})(?:га|ga|да|da|дан|dan|ни|ni|ка|ka|қа|qa)$")


def _oblique_end(name: str) -> bool:
    """The last word carries a dative / locative / ablative / accusative case (lemma unchanged when cut)."""
    from graphrag.kg.concepts import word_lemmas
    w = (WORD_RX.findall(name) or [""])[-1]
    m = _OBLIQUE.search(w.lower())
    if not m:
        return False
    return word_lemmas(w, "uz") == word_lemmas(w[:m.start()], "uz")


def split_coordinated(name: str, typ: str) -> list[tuple[int, int]] | None:
    """None: not a coordinated list of entities. Else the (start, end) spans of the conjuncts that are
    entities of `typ` themselves (possibly empty: the fused name is dropped). A list is fused when >= 2
    conjuncts end with a head of `typ`, or a comma-separated conjunct is one plural word ("вилоятлар")."""
    if typ not in ("organ", "position", "procedure"):
        return None
    seps = _top_level_seps(name)
    if not seps:
        return None
    spans, pos = [], 0
    for a, b in seps:
        spans.append((pos, a))
        pos = b
    spans.append((pos, len(name)))
    headed = []
    single_plural = False
    for s, e in spans:
        part = name[s:e].strip()
        if not part:
            continue
        k = norm_key(part)
        last = part.split()[-1].lower()
        bare_plural = bool(re.search(r"(?:лар|lar)$", last))       # "демократик институтлар" = a modifier
        if k and head_type(k, typ) == typ and not (bare_plural and len(part.split()) > 1):
            headed.append((s, e))
        elif len(part.split()) == 1 and (bare_plural or last in _TERRITORIAL):
            single_plural = True
    if typ == "procedure":
        single_plural = False                     # "товарлар ва хизматлар экспорти тартиби" is one procedure
    elif re.search(r"(?:лари|lari)$", name.strip().split()[-1].lower()) and \
            any("," in name[a:b] for a, b in seps):
        single_plural = True                      # a shared plural head: "Сув хўжалиги, Энергетика вазирликлари"
    if len(headed) >= 2 or (single_plural and headed):
        # keep a headed conjunct unless it follows an undistributed plural modifier ("шаҳарлар давлат
        # ҳокимияти органлари": the modifier list shares its head) and starts in lower case
        keep = []
        for s, e in headed:
            part = name[s:e].strip()
            if single_plural and not part[:1].isupper():
                continue
            keep.append((s + (len(name[s:e]) - len(name[s:e].lstrip())), e))
        return keep
    return None


def alias_map() -> dict[str, str]:
    out = {}
    for a in _cfg()["aliases"]:
        ck = norm_key(a["canonical"])
        for n in a["names"]:
            out[norm_key(n)] = ck
    return out


class Context2:
    """Per record: spans of «…» деган сўзлар and boilerplate sentences; owner / modifier checks."""

    def __init__(self, text: str, doc_type: str = ""):
        C = _cfg()
        self.text = text
        self.quoted = []
        stack = []
        for i, ch in enumerate(text):
            if ch in "«“":
                stack.append(i)
            elif ch in "»”" and stack:
                a = stack.pop()
                if not stack and C["quoted_after"].match(text, i + 1):
                    self.quoted.append((a, i + 1))
        rxs = list(C["boiler"]) + ([C["drafted"]] if doc_type == "committee_conclusion" else [])
        self.boiler = sorted((m.start(), m.end()) for rx in rxs for m in rx.finditer(text))

    def reason(self, a: int, b: int) -> str | None:
        C = _cfg()
        if C["owner_after"].match(self.text, b):
            return "owner_only"
        if C["modifier_after"].match(self.text, b):
            return "modifier"
        if self._act_owner(a, b):
            return "act_owner"
        if any(x <= a and b <= y for x, y in self.quoted):
            return "quoted_words"
        for x, y in self.boiler:
            if self._in_sentence_before(a, b, x, y):
                return "boilerplate"
        return None

    _DATEISH = re.compile(r"(?i)^(?:\S*\d\S*|йил\w*|yil\w*|сонли|sonli|сон|son|ПФ|ПҚ|PF|PQ|ЎРҚ|O[ʻʼ'`‘’]?RQ)$")

    def _act_owner(self, a: int, b: int) -> bool:
        """A genitive mention owning an act named after it: "Президентининг … Фармони"."""
        C = _cfg()
        name = self.text[a:b]
        if not re.search(r"(?i)(?:нинг|ning)$", name):
            return False
        w = self.text[b:b + C["act_window"]]
        m = C["act_noun"].search(w)
        if not m:
            return False
        gap = w[:m.start()]
        if re.search(r"[.;!?]|\n\n", re.sub(r"[«“\"][^»”\"]*[»”\"]", " ", gap)):
            return False
        gap = re.sub(r"[«“\"][^»”\"]*[»”\"]\s*[-‐]?\s*(?:ги|gi|даги|dagi)?", " ", gap)
        words = [x for x in WORD_RX.findall(gap) if not self._DATEISH.match(x)
                 and not re.search(r"(?i)(?:даги|dagi)$", x)]
        return len(words) <= C["act_words"]

    def _in_sentence_before(self, a, b, x, y) -> bool:
        """[a, b) lies inside the boilerplate match, or in the same sentence before it."""
        if x <= a and b <= y:
            return True
        if b <= x:
            gap = self.text[b:x]
            return not re.search(r"[.;!?]|\n\n", gap) and len(gap) < 200
        return False


# --- fix round 4 (Ruling R18): dedup keys, restores, splits (post-verification, deterministic) --------

_DD: dict | None = None


def _dd() -> dict:
    global _DD
    if _DD is None:
        d = config.load("graph")["dictionaries"]
        x = d["dedup"]
        rs = x.get("restore") or {}
        _DD = {"translit": [(re.compile(a), b) for a, b in x.get("translit") or []],
               "token_map": [(re.compile(r"(?<!\S)" + re.escape(a) + r"(?!\S)"), b)
                             for a, b in sorted((x.get("token_map") or {}).items(), key=lambda t: -len(t[0]))],
               "light": [tuple(t.split()) for t in x.get("light_tails") or []],
               "generic": [tuple(t.split()) for t in x.get("generic_heads") or []],
               "verbal": re.compile(x.get("verbal_noun") or r"(?!x)x"),
               "keep_light": set(x.get("keep_before_light") or []),
               "rewrites": {t: [(re.compile(a), b) for a, b in v] for t, v in (x.get("rewrites") or {}).items()},
               "pairs": x.get("pairs") or [],
               "aliases": d.get("aliases") or [],
               "restore": rs,
               "split_owner": re.compile(x["split_owner_post"]) if x.get("split_owner_post") else None,
               "split_rest_last": set(x.get("split_owner_rest_last") or []),
               "named_inst": set(x.get("named_institution_heads") or []),
               "pair_map": None}
    return _DD


def fold_key(key: str, typ: str) -> str:
    """Transliteration folds + owner rewrites + token map (no stripping, no pairs)."""
    D = _dd()
    k = key
    for rx, rep in D["translit"]:
        k = rx.sub(rep, k)
    for rx, rep in D["rewrites"].get(typ, []) + D["rewrites"].get("*", []):
        k = rx.sub(rep, k)
    if typ == "procedure":
        for rx, rep in D["token_map"]:
            k = rx.sub(rep, k)
    return " ".join(k.split())


def strip_procedure(lem: list[str]) -> list[str]:
    """Strip light verbs ("амалга ошириш", "ўтказиш", "қилиш") after a named-procedure noun and generic heads
    ("тартиб(-таомил)") after a named-procedure noun or a >= 2-lemma verbal-noun name, repeatedly."""
    C, D = _cfg(), _dd()
    proc = {fold_key(h, "procedure") for h in C["procedure"]} | C["procedure"]
    changed = True
    while changed and len(lem) > 1:
        changed = False
        for tail in D["light"] + D["generic"]:
            n = len(tail)
            if len(lem) <= n or tuple(lem[-n:]) != tail:
                continue
            rest = lem[:-n]
            head = rest[-1]
            weak = head in _WEAK_PROC
            if tail in D["light"]:
                ok = (head in proc and not weak and head not in D["keep_light"]) or (weak and len(rest) >= 2)
            else:
                ok = (head in proc and not weak) or (len(rest) >= 2 and not weak and bool(D["verbal"].search(head)))
            if ok:
                lem = rest
                changed = True
                break
    return lem


def _pair_map() -> dict[tuple[str, str], str]:
    """(lang-agnostic) folded alias key -> folded canonical key, for every type (pairs never cross types:
    the caller groups by (type, key))."""
    D = _dd()
    if D["pair_map"] is None:
        from graphrag.kg.extract_ent import entity_key
        out = {}

        def base(name, lang):
            return entity_key(name, lang)
        for p in D["pairs"]:
            ck = base(p["canonical"], "uz")
            for lang in ("ru", "uz"):
                for n in p.get(lang) or []:
                    out[base(n, lang)] = ck
        for a in D["aliases"]:
            ck = base(a["canonical"], "uz")
            for n in a["names"]:
                for lang in ("ru", "uz"):
                    out.setdefault(base(n, lang), ck)
        D["pair_map"] = out
    return D["pair_map"]


def dedup_key(key: str, typ: str) -> str:
    """The R18 grouping key of an entity key of type `typ`."""
    pm = _pair_map()

    def norm(k):
        f = fold_key(k, typ)
        if typ == "procedure":
            f = " ".join(strip_procedure(f.split()))
        return f
    k0 = key
    for k in (k0, fold_key(k0, typ)):
        if k in pm:
            return norm(pm[k])
    f = norm(k0)
    # pairs are also matched after folding (a Russian name with "Республики Узбекистан" etc.)
    for a, c in pm.items():
        if norm(a) == f:
            return norm(c)
    return f


def restore_verify_drop(key: str, typ: str) -> bool:
    """R18 item 3: a verification drop overruled by a deterministic rule (deputies, state organ classes
    "… органлари", subdivisions "… вазирлигининг бўлинмалари")."""
    rs = _dd()["restore"]
    lem = key.split()
    if not lem:
        return False
    if typ == "position":
        return lem[-1] in set(rs.get("position_last") or []) or key in set(rs.get("position_keys") or [])
    if typ == "organ" and lem[-1] in set(rs.get("organ_last") or []) and len(lem) >= 3:
        if lem[0] in set(rs.get("organ_generic_first") or []):
            return False
        if set(lem[:-1]) & set(rs.get("organ_owner_heads") or []):
            return True
        return bool(set(lem[:-1]) & set(rs.get("organ_state_markers") or []))
    return False


def restore_head_drop(key: str) -> str | None:
    """R18 item 3: a head-typing drop restored with a type (legal_category keys)."""
    return "legal_category" if key in set(_dd()["restore"].get("legal_category_keys") or []) else None


def modifier_level_coordination(name: str) -> bool:
    """R18: "товар ва молия бозорлари иштирокчилари": a one-word first conjunct coordinated with the
    possessive modifier of a 3-word second conjunct (the head is shared by the whole name, not fused)."""
    seps = _top_level_seps(name)
    if len(seps) != 1:
        return False
    a, b = seps[0]
    first, after = _words_nopar(name[:a]), _words_nopar(name[b:])
    return len(first) == 1 and len(after) == 3 and \
        bool(re.search(r"(?:лари|lari|си|si|и|i)$", after[-2].lower())) and \
        bool(re.search(r"(?:лари|lari|си|si|и|i)$", after[-1].lower()))


def strict_adjective(w: str) -> bool:
    """An adjective by its suffix (-ий / -ик / -вий), not a "-лик" noun ("ҳайдовчилик")."""
    w = w.lower()
    return bool(re.search(r"(?:ий|iy|ик|ik)$", w)) and not re.search(r"(?:лик|lik)$", w)


_PLURAL_END = re.compile(r"(?:лар|lar)$")


def split_category_list(name: str) -> list[tuple[int, int]] | None:
    """R18 item 4: an Uzbek legal_category name "A ҳамда B" / "A ва B" whose conjuncts are each >= 2 words
    and each end in a nominative plural noun ("юридик шахслар ҳамда якка тартибдаги тадбиркорлар") is a
    fused list: the conjunct spans. None: not such a list."""
    if _CYR.search(name) and re.search(r"[ыщэё]", name.lower()):
        return None
    seps = _top_level_seps(name)
    if not seps:
        return None
    spans, pos = [], 0
    for a, b in seps:
        if re.search(r"(?i)ёки|yoki", name[a:b]):
            return None
        spans.append((pos, a))
        pos = b
    spans.append((pos, len(name)))
    out = []
    for s, e in spans:
        part = name[s:e]
        w = _words_nopar(part)
        if len(w) < 2 or not _PLURAL_END.search(w[-1].lower()) or _PARTICIPLE.search(w[-2].lower()) and len(w) == 2:
            return None
        lead = len(part) - len(part.lstrip())
        out.append((s + lead, e - (len(part) - len(part.rstrip()))))
    return out


def split_owner_post(name: str) -> tuple[str, str, int] | None:
    """R18 item 4: "<ombudsman>нинг <… органлари>" -> (owner post name, rest name, offset of rest)."""
    D = _dd()
    rx = D["split_owner"]
    if rx is None:
        return None
    m = rx.match(name)
    if not m:
        return None
    rest = m.group("rest")
    k = norm_key(rest)
    if not k or k.split()[-1] not in D["split_rest_last"]:
        return None
    return m.group("own"), rest, m.start("rest")


def named_institution(key: str, surface: str) -> bool:
    """R18 item 4: a capitalised name headed by a research / educational institution noun."""
    lem = key.split()
    last = (WORD_RX.findall(surface) or [""])[-1].lower()
    return bool(lem) and lem[-1] in _dd()["named_inst"] and surface[:1].isupper() and \
        not re.search(r"(?:лар|lar)\w*$", last)          # a plural ("Фуқаролик жамияти институтлари") is a class


def ru_nominative(title: str) -> str:
    """R18: a Russian inflected title in the nominative singular: the head noun (first noun) and the
    adjectives / participles agreeing with it before it ("потенциальных приобретателей" -> "потенциальный
    приобретатель", "кредиторов" -> "кредитор", "Акционеры банка" -> "Акционер банка"). Conservative: a
    title with punctuation, a non-agreeing modifier or a non-genitive dependent is kept as it is."""
    from graphrag.data.lemma import _morph
    if not _CYR.search(title) or re.search(r"[ўқғҳЎҚҒҲ]", title) or re.search(r"[,;:()–—«»\"]", title):
        return title
    words = list(re.finditer(r"[А-Яа-яЁё-]+", title))
    if not words or len(words) != len(title.split()) or any(m.group(0).lower() in ("и", "или", "а", "либо")
                                                            for m in words):
        return title
    morph = _morph()
    parses = [morph.parse(m.group(0))[0] for m in words]
    head = next((i for i, p in enumerate(parses) if p.tag.POS == "NOUN"), None)
    if head is None:
        return title
    hp = parses[head]
    if any(p.tag.POS not in ("ADJF", "PRTF", "NUMR") or p.tag.case != hp.tag.case for p in parses[:head]):
        return title
    if any(not any(q.tag.case == "gent" for q in morph.parse(m.group(0))) for m in words[head + 1:]):
        return title
    hi = hp.inflect({"nomn", "sing"}) or hp.inflect({"nomn"})
    if hi is None:
        return title
    out, pos = [], 0
    for i in range(head + 1):
        m, p = words[i], parses[i]
        if i == head:
            w = hi.word
        else:
            q = p.inflect({"nomn", "sing"} | ({hi.tag.gender} if hi.tag.gender else set()))
            if q is None:
                return title
            w = q.word
        if m.group(0)[:1].isupper():
            w = w[:1].upper() + w[1:]
        out.append(title[pos:m.start()] + w)
        pos = m.end()
    return "".join(out) + title[pos:]
