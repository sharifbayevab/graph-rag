"""Concepts ("понятия") and DEFINES ("определяет понятие") edges (M2 Task 6), plus the lemma keys
shared with graphrag.kg.extract_ent.

* `norm_key(name)`: the dedup key of a term / entity name: the lemmas of `graphrag.data.lemma.tokens`
  (internal Latin form for Uzbek, pymorphy3 normal forms for Russian; stop words and composite
  tokens dropped) of each word, joined by spaces. "Вазирлар Маҳкамасининг" and "Vazirlar Mahkamasi"
  -> "vazir mahkama".
* `extract_concepts(rec)`: in a law text, an article titled "Асосий тушунчалар" / "Asosiy
  tushunchalar" / "Основные понятия", or the node holding "…қуйидаги асосий тушунчалар
  қўлланилади:", is a definitions list; each following paragraph "term – definition;" gives a Concept
  (title = the term verbatim, attrs.definition = the definition verbatim) and a DEFINES edge from
  μ(paragraph span) with evidence = the "term – definition;" span. Patterns: graph.yaml
  patterns.definitions.
* `merge_concepts(nodes)`: one vertex per norm_key (CON/<norm_key>) across laws; attrs.definitions
  keeps the per-source definitions.
* `concept_mentions(records, concepts, defines)`: REGULATES norm -> concept where the norm's lemma
  sequence contains the concept's key (>= min_lemmas lemmas, or one lemma of >= min_single_len chars);
  a concept matching more than max_norms norms gets no edges (too generic, reported).

Norms (the only sources of REGULATES / DEFINES, controller ruling): elements of law texts (статья and
its descendants), top-level пункт (and подпункт) of resolutions, блок of committee conclusions.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import lru_cache

from graphrag import config
from graphrag.data.lemma import tokens
from graphrag.data.normalize import clean_encoding, detect_language, to_internal
from graphrag.kg.acts import is_law_text
from graphrag.kg.schema import Evidence, GraphEdge, GraphNode

CONCEPT_PREFIX = "CON/"
CONCEPT_TYPE = "понятие"
RESOLUTION_TYPES = ("resolution", "postanovlenie")
CONCLUSION_TYPES = ("committee_conclusion",)

WORD_RX = re.compile(r"[^\W_]+(?:[’ʼʻ'`‘-][^\W_]+)*")

# --- lemma keys -------------------------------------------------------------------------------------


def lang_of(rec) -> str:
    """"ru" for a Russian record, else "uz" (the to_internal routing default)."""
    return "ru" if rec.meta.lang and rec.meta.lang[0] == "ru" else "uz"


@lru_cache(maxsize=500_000)
def word_lemmas(word: str, lang: str) -> tuple[str, ...]:
    """Lemmas of one original word (usually one; "atrof-muhitni" -> atrof, muhit)."""
    norm = to_internal(clean_encoding(word), lang)
    return tuple(t for t in tokens(norm) if "#" not in t)


def lemma_spans(text: str, lang: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """(lemma, start, end) per lemma of every word of `text`; spans are offsets of the original word."""
    out = []
    for m in WORD_RX.finditer(text):
        for lem in word_lemmas(m.group(0), lang):
            out.append((lem, offset + m.start(), offset + m.end()))
    return out


def _guess_lang(name: str) -> str:
    return "ru" if detect_language(name) == "ru" else "uz"


def norm_key(name: str, lang: str | None = None) -> str:
    """Lemmatized Latin key of a name (stop words dropped), lemmas joined by single spaces."""
    lang = lang or _guess_lang(name)
    return " ".join(lem for lem, _, _ in lemma_spans(name, lang))


# --- norms ------------------------------------------------------------------------------------------


def norm_tops(rec) -> list:
    """The top-level norm nodes of a record (статья of a law text; top-level пункт of a resolution;
    блок of a committee conclusion). Their descendants are norms too; stenograms have none."""
    ns = {n.node_id: n for n in rec.structure}
    if is_law_text(rec):
        return [n for n in rec.structure if n.type == "статья"]
    if rec.meta.type in RESOLUTION_TYPES:
        return [n for n in rec.structure if n.type == "пункт" and n.parent_id and ns[n.parent_id].type == "документ"]
    if rec.meta.type in CONCLUSION_TYPES:
        return [n for n in rec.structure if n.type == "блок"]
    return []


def norm_ids(rec) -> set[str]:
    """All norm node ids of a record: the tops and their descendants."""
    tops = norm_tops(rec)
    ids = {n.node_id for n in tops}
    children = defaultdict(list)
    for n in rec.structure:
        if n.parent_id:
            children[n.parent_id].append(n.node_id)
    stack = list(ids)
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in ids:
                ids.add(c)
                stack.append(c)
    return ids


# --- definitions ------------------------------------------------------------------------------------

_P: dict | None = None


def _pats() -> dict:
    global _P
    if _P is None:
        cfg = config.load("graph")["patterns"]["definitions"]
        _P = {k: re.compile("(?i)" + v) for k, v in cfg.items() if isinstance(v, str)}
        _P["cfg"] = cfg
    return _P


def _paragraph_spans(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """Stripped paragraph spans of text[a:b] (paragraphs are separated by blank lines)."""
    out, pos = [], a
    for part in text[a:b].split("\n\n"):
        s, e = pos, pos + len(part)
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
        pos += len(part) + 2
    return out


def _top_level_dash(p: str, dash: re.Pattern) -> re.Match | None:
    """The first term/definition dash outside parentheses and quotes."""
    for m in dash.finditer(p):
        pre = p[:m.start()]
        if pre.count("(") > pre.count(")"):
            continue
        if pre.count("«") > pre.count("»") or pre.count("“") > pre.count("”"):
            continue
        return m
    return None


def parse_definition(p: str) -> dict | None:
    """{term, title, alias, definition, term_end, def_start, def_end} of a "term – definition;"
    paragraph (offsets into p), or None."""
    P = _pats()
    m = _top_level_dash(p, P["dash"])
    if m is None:
        return None
    term = p[:m.start()].rstrip()
    if not term or len(term) > int(P["cfg"].get("max_term_chars", 300)) or term.endswith(":"):
        return None
    ds, de = m.end(), len(p)
    while de > ds and p[de - 1] in ";.":
        de -= 1
    while de > ds and p[de - 1].isspace():
        de -= 1
    if de <= ds:
        return None
    title, alias = term, None
    am = P["alias"].search(term)
    if am and am.start() > 0:
        title, alias = term[:am.start()].rstrip(), am.group("alias").strip()
    return {"term": term, "title": title, "alias": alias, "definition": p[ds:de], "def_start": ds, "def_end": de}


def definition_paragraphs(rec) -> list[tuple[int, int]]:
    """Candidate paragraph spans of the record's definitions lists (law texts only)."""
    if not is_law_text(rec):
        return []
    P = _pats()
    ns = {n.node_id: n for n in rec.structure}
    spans: set[tuple[int, int]] = set()
    for n in rec.structure:
        if n.type == "статья" and P["article_title"].search((n.title or "").strip()):
            spans.update(_paragraph_spans(rec.text, n.start, n.end))
    for m in P["intro"].finditer(rec.text):
        holder = [n for n in rec.structure if n.start <= m.start() and m.end() <= n.end and n.parent_id]
        if not holder:
            continue
        deep = max(holder, key=lambda n: (n.node_id.count("/"), -(n.end - n.start)))
        art = deep
        while art is not None and art.type != "статья":
            art = ns.get(art.parent_id) if art.parent_id else None
        box = art or deep
        spans.update(s for s in _paragraph_spans(rec.text, m.end(), box.end))
    return sorted(spans)


def concept_id(key: str) -> str:
    return CONCEPT_PREFIX + key


def extract_concepts(rec, di=None) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Concept vertices (one per distinct key in this record) and DEFINES edges of one record."""
    from graphrag.kg.extract_ref import DocIndex
    di = di or DocIndex(rec)
    lang = lang_of(rec)
    nodes: dict[str, GraphNode] = {}
    evs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a, b in definition_paragraphs(rec):
        d = parse_definition(rec.text[a:b])
        if d is None:
            continue
        key = norm_key(d["title"], lang)
        if not key:
            continue
        cid = concept_id(key)
        src = di.node_at(a, b)
        if src is None or src.parent_id is None:
            continue
        entry = {"doc_id": rec.doc_id, "node_id": src.node_id, "term": d["term"], "definition": d["definition"]}
        if cid not in nodes:
            nodes[cid] = GraphNode(cid, "concept", CONCEPT_TYPE, d["title"], None, False,
                                   {"norm_key": key, "lang": lang, "definition": d["definition"],
                                    "definitions": [], "aliases": []})
        att = nodes[cid].attrs
        if entry not in att["definitions"]:
            att["definitions"].append(entry)
        if d["alias"] and d["alias"] not in att["aliases"]:
            att["aliases"].append(d["alias"])
        evs[(src.node_id, cid)].append(Evidence(rec.doc_id, a, b, rec.text[a:b]).to_json())
    edges = [_edge(src, cid, "DEFINES", ev) for (src, cid), ev in sorted(evs.items())]
    return [nodes[k] for k in sorted(nodes)], edges


def _edge(src: str, dst: str, typ: str, evs: list[dict]) -> GraphEdge:
    uniq = {(e["doc_id"], e["start"], e["end"]): e for e in evs}
    evs = [uniq[k] for k in sorted(uniq)]
    attrs = {"evidence": evs[0]}
    if len(evs) > 1:
        attrs["evidence_all"] = evs
    return GraphEdge(src, dst, typ, attrs)


def merge_concepts(nodes: list[GraphNode]) -> list[GraphNode]:
    """One vertex per concept id: title / definition of the first source (by doc_id, node_id);
    attrs.definitions = all per-source definitions, attrs.aliases = the union."""
    by: dict[str, list[GraphNode]] = defaultdict(list)
    for n in nodes:
        by[n.node_id].append(n)
    out = []
    for cid in sorted(by):
        defs = sorted({(d["doc_id"], d["node_id"], d["term"], d["definition"])
                       for n in by[cid] for d in n.attrs["definitions"]})
        first = min(by[cid], key=lambda n: (n.attrs["definitions"][0]["doc_id"], n.attrs["definitions"][0]["node_id"]))
        aliases = sorted({a for n in by[cid] for a in n.attrs.get("aliases", [])})
        attrs = dict(first.attrs)
        attrs["definitions"] = [{"doc_id": a, "node_id": b, "term": t, "definition": d} for a, b, t, d in defs]
        attrs["definition"] = attrs["definitions"][0]["definition"]
        attrs["aliases"] = aliases
        attrs["n_sources"] = len({d["doc_id"] for d in attrs["definitions"]})
        out.append(GraphNode(cid, "concept", CONCEPT_TYPE, first.title, None, False, attrs))
    return out


# --- concept mentions in norms ----------------------------------------------------------------------


def mention_eligible(key: str) -> bool:
    cfg = _pats()["cfg"]
    lem = key.split()
    return len(lem) >= int(cfg.get("min_lemmas", 2)) or (len(lem) == 1 and len(lem[0]) >= int(cfg.get("min_single_len", 6)))


_LOCAL_ALIAS = re.compile(r"(?i)\(\s*(?:бундан\s+буён\s+(?:матнда\s+)?|bundan\s+buyon\s+(?:matnda\s+)?|далее\s*[–—-]?\s*)"
                          r"(?P<a>[^()]{1,80}?)(?:\s+(?:деб|deb)\s+(?:юритилади|yuritiladi))?\s*\)")


def law_specific(concepts: list[GraphNode]) -> set[str]:
    """R16 rule 6: concepts whose meaning is fixed by their law: every definition names a specific body
    (it starts with a capitalised word: "ваколатли орган – Ўзбекистон Республикаси Президенти ҳузуридаги
    Ижтимоий ҳимоя миллий агентлиги"), or refers to such a concept of the same law ("суғурталовчи –
    ваколатли органнинг жамғармаси"). Their mention edges follow the single-lemma scope rule."""
    out = {c.node_id for c in concepts if c.attrs["definitions"]
           and all(d["definition"][:1].isupper() for d in c.attrs["definitions"])}
    # one step: a definition built on a multi-lemma seed of the same law ("ваколатли органнинг жамғармаси")
    seeds = [(tuple(c.attrs["norm_key"].split()), {d["doc_id"] for d in c.attrs["definitions"]})
             for c in concepts if c.node_id in out and len(c.attrs["norm_key"].split()) >= 2]
    extra = set()
    for c in concepts:
        if c.node_id in out:
            continue
        docs = {d["doc_id"] for d in c.attrs["definitions"]}
        for d in c.attrs["definitions"]:
            lem = tuple(x[0] for x in lemma_spans(d["definition"], c.attrs.get("lang", "uz")))
            if any(dd & docs and any(lem[i:i + len(k)] == k for i in range(len(lem) - len(k) + 1))
                   for k, dd in seeds):
                extra.add(c.node_id)
                break
    return out | extra


# an organ head in the possessive (izofa) form: "хизмати", "агентлигининг", "қўмитасига"
_POSS_HEAD = re.compile(r"(?:и|си|i|si)(?:нинг|ning|га|ga|да|da|дан|dan|ни|ni|даги|dagi)?$")


def _proper_start(text: str, a: int) -> bool:
    """The mention starts capitalised, or right after a capitalised word ("Давлат йўл ҳаракати …"): part of
    a proper organ name, not "воситачи хизматлари"."""
    if text[a:a + 1].isupper():
        return True
    prev = list(WORD_RX.finditer(text, max(0, a - 40), a))
    return bool(prev) and prev[-1].end() >= a - 1 and prev[-1].group(0)[:1].isupper()


def concept_mentions(records, concepts: list[GraphNode], defines: list[GraphEdge], docs=None,
                     cites: dict[str, set[str]] | None = None, organ_spans: dict[str, list] | None = None):
    """(REGULATES edges norm -> concept, stats). A mention = a contiguous run of the norm's lemmas
    equal to a concept key (longest match first, non-overlapping); src = μ(mention span); evidence =
    the original words. (src, concept) pairs that already have DEFINES are skipped.

    Fix round 1 (R15): a single-lemma concept ("жамғарма", "олувчи") only gets mentions in its defining
    law(s) or in top-level norms that cite a defining law (`cites`: defining doc_id -> top norm ids
    citing it; None = no restriction); the too-generic cutoff counts distinct top-level norms."""
    from graphrag.kg.extract_ref import DocIndex
    cfg = _pats()["cfg"]
    # R18: the concept only as the subject of legislation ("Рақобат тўғрисидаги қонунчилик …"), or inside an
    # organ name (an organ mention span, or an organ head word right after it: "… хавфсизлиги хизмати")
    dcfg = config.load("graph")["dictionaries"]
    leg_rx = re.compile("(?i)" + dcfg["concept_filters"]["legislation_after"].lstrip("^"))
    organ_heads = set(dcfg["types"]["organ"]["heads"])
    max_norms = int(cfg.get("max_norms", 200))
    keys: dict[str, dict[tuple, str]] = defaultdict(dict)       # lang -> key tuple -> concept id
    home: dict[str, set[str]] = {}                              # single-lemma concept -> defining doc_ids
    specific = law_specific(concepts)
    defs_of = {c.node_id: {d["doc_id"] for d in c.attrs["definitions"]} for c in concepts}
    for c in concepts:
        k = c.attrs["norm_key"]
        if mention_eligible(k):
            keys[c.attrs.get("lang", "uz")][tuple(k.split())] = c.node_id
            if len(k.split()) == 1 or c.node_id in specific:
                home[c.node_id] = {d["doc_id"] for d in c.attrs["definitions"]}
    first: dict[str, dict[str, list[tuple]]] = {}
    for lang, kk in keys.items():
        f = defaultdict(list)
        for t in kk:
            f[t[0]].append(t)
        for v in f.values():
            v.sort(key=lambda t: (-len(t), t))
        first[lang] = f
    has_def = {(e.src, e.dst) for e in defines}
    evs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    tops_of: dict[str, set] = defaultdict(set)
    restricted = Counter()
    for rec in records:
        lang = lang_of(rec)
        if lang not in first:
            continue
        f, kk = first[lang], keys[lang]
        tops = norm_tops(rec)
        if not tops:
            continue
        di = (docs or {}).get(rec.doc_id) or DocIndex(rec)
        # R17: never inside an act title; a term this record re-defines by its own alias ("Қишлоқ хўжалиги
        # суғуртаси жамғармаси (бундан буён матнда Жамғарма деб юритилади)") means that record's own thing
        from graphrag.kg.acts import scan_act_mentions
        act_spans = [(x.start, x.end) for x in scan_act_mentions(rec.text)]
        local = {norm_key(m.group("a").strip(), lang) for m in _LOCAL_ALIAS.finditer(rec.text)}
        for t in tops:
            sp = lemma_spans(rec.text[t.start:t.end], lang, t.start)
            i = 0
            while i < len(sp):
                hit = None
                for cand in f.get(sp[i][0], ()):
                    n = len(cand)
                    if i + n <= len(sp) and tuple(x[0] for x in sp[i:i + n]) == cand:
                        hit = cand
                        break
                if hit is None:
                    i += 1
                    continue
                a, b = sp[i][1], sp[i + len(hit) - 1][2]
                i += len(hit)
                src = di.node_at(a, b)
                cid = kk[hit]
                if src is None or (src.node_id, cid) in has_def:
                    continue
                if any(x <= a and b <= y for x, y in act_spans):
                    restricted["_in_act_title"] += 1
                    continue
                e2 = b
                while e2 < len(rec.text) and (rec.text[e2].isalnum() or rec.text[e2] in "ʼʻ'’-"):
                    e2 += 1
                if leg_rx.match(rec.text, e2):
                    restricted["_legislation_subject"] += 1
                    continue
                nxt = WORD_RX.match(rec.text, e2 + (len(rec.text[e2:]) - len(rec.text[e2:].lstrip(" "))))
                if any(x <= a and b <= y for x, y in (organ_spans or {}).get(rec.doc_id, ())) or \
                        (nxt and set(word_lemmas(nxt.group(0), lang)) & organ_heads and
                         _POSS_HEAD.search(nxt.group(0).lower()) and _proper_start(rec.text, a)):
                    restricted["_in_organ_name"] += 1
                    continue
                own_docs = home.get(cid) or defs_of.get(cid, set())
                if " ".join(hit) in local and rec.doc_id not in own_docs:
                    restricted["_local_alias"] += 1
                    continue
                if cid in home and cites is not None and rec.doc_id not in home[cid] \
                        and not any(t.node_id in cites.get(d, ()) for d in home[cid]):
                    restricted[cid] += 1
                    continue
                evs[(src.node_id, cid)].append(Evidence(rec.doc_id, a, b, rec.text[a:b]).to_json())
                tops_of[cid].add(t.node_id)
    generic = {cid: len(s) for cid, s in tops_of.items() if len(s) > max_norms}
    edges = [_edge(src, cid, "REGULATES", ev) for (src, cid), ev in sorted(evs.items()) if cid not in generic]
    stats = {"eligible_concepts": sum(len(v) for v in keys.values()),
             "concepts_with_mentions": len(tops_of) - len(generic),
             "too_generic": dict(sorted(generic.items(), key=lambda x: (-x[1], x[0]))),
             "single_lemma_cross_law_dropped": sum(v for k, v in restricted.items() if not k.startswith("_")),
             "in_act_title_dropped": restricted.get("_in_act_title", 0),
             "local_alias_dropped": restricted.get("_local_alias", 0),
             "legislation_subject_dropped": restricted.get("_legislation_subject", 0),
             "in_organ_name_dropped": restricted.get("_in_organ_name", 0),
             "law_specific_concepts": sorted(specific),
             "single_lemma_cross_law_by_concept": dict(sorted((k, v) for k, v in restricted.items() if not k.startswith("_"))),
             "max_norms": max_norms, "cutoff_counts": "top-level norms"}
    return edges, stats
