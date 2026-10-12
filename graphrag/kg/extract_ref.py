"""REFERS_TO ("ссылается") extraction (M2 Task 4).

For one DocRecord, `extract_refs(rec, registry, ctx)` emits validated REFERS_TO edges:

* internal   "ушбу Қонуннинг 5-моддаси", "ушбу модданинг иккинчи қисми" -> the record's own node;
* amended    inside an amendment article ("Жиноят кодексига қуйидаги ўзгартишлар киритилсин: 1)
             27-модданинг иккинчи қисми …") the instruction head's bare "N-модда" -> the amended
             act's element (real corpus node or EXT stub): the element the instruction amends;
* cited_in_amendment
             the same owner rules for a citation INSIDE the quoted new / replacement text of the
             instruction ("«… ушбу Кодекснинг 198⁸-моддасида …»", «101-моддасида» деган сўзлар,
             "ушбу модда" in a quoted new article) -> the amended act's element that the new text
             cites, not the element being amended (Ruling R20);
* element    "«X»ги Қонуннинг 5-моддаси", "Жиноят кодексининг 27-моддаси", "Конституциянинг
             93-моддаси" -> that act's element (real corpus node or EXT/<key>/СТ.N[/Ч.K[/П.M]] stub);
* act        every other act mention (graphrag.kg.acts.resolve_record) -> the act vertex;
* agenda     stenogram вопрос -> the act its title names (evidence: the title in the announcing turn);
* title      a Senate resolution / committee conclusion -> the act its meta.title names.

Every edge: evidence = exact rec.text[start:end] of the mention (Evidence dict), src = the deepest
node whose span contains the mention (node_at, μ of the span) -- the вопрос / document itself for
agenda / title links -- and attrs.relation = the class above; several mentions of one (src, dst)
become one edge with evidence_all. Self-links (to the source node or one of its ancestors) are
dropped; gazette citations ("Ахборотнома…, № 1, 3-модда") are never references.

Act resolution goes through ActRegistry.resolve_record (per-record consistency, Ruling R7).
Ambiguous acts (same-day ties; undated omnibus titles; typo titles close to a corpus act key) go to
the LLM pass `ref_resolve` (graphrag.kg.llm, cached and verified) when ctx has an LLM, else they are
queued in ctx.queue and left unlinked. An element of a corpus act that the act does not contain is
recorded in ctx.unresolved (no corpus node is invented); the act-level edge is kept.
"""
from __future__ import annotations

import bisect
import difflib
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from graphrag import config
from graphrag.data.legacy.normalize import skeleton_key
from graphrag.data.normalize import canon_num
from graphrag.data.structure import _own_flags
from graphrag.kg.acts import (r7_undated, CITED_MAX_GAP_DAYS, DATE_GRACE_DAYS, EXT_PREFIX, _days, _parse_date, ActRef, ActRegistry, _title_span, act_key, is_amending, is_generic,
                              is_law_text, omnibus_candidates, record_date, resolve_record, scan_act_mentions, _OUTER_CLOSE)
from graphrag.kg.schema import Evidence, GraphEdge, GraphNode, Schema

PASS = "ref_resolve"
CTX_CHARS = 600
BODY_PARA_CHARS = 250       # the first paragraph this long ends a qaror / xulosa heading block
CARRY_CHARS = 300          # a bare article after an owned one in the same paragraph keeps its act
REL_PRIORITY = ("title", "agenda", "internal", "amended", "cited_in_amendment", "element", "act")
SEC_TITLED = ("QARORLAR", "XULOSALAR")
# Ruling R24 (2): one REFERS_TO edge per (src, dst, relation class). REL_PRIORITY folds the relations of one
# class into its highest-priority label; a citation inside quoted new / replaced text (cited_in_amendment) is
# its own class and is never folded into the instruction head's `amended` edge (nor absorbs element / act).
REL_CLASSES = {"cited_in_amendment": "cited_in_amendment"}


def rel_class(rel: str) -> str:
    return REL_CLASSES.get(rel, "")

# --- patterns (config/graph.yaml patterns.refs) ---------------------------------------------------

_P: dict | None = None
ABBREV: dict[str, str] = {}


def _pats() -> dict:
    global _P
    if _P is None:
        cfg = (config.load("graph").get("patterns") or {})["refs"]
        ords: dict[str, int] = {}
        for n, forms in (cfg.get("ordinals") or {}).items():
            for f in forms:
                ords[f.lower()] = int(n)
        alt = "|".join(sorted((re.escape(f) for f in ords), key=len, reverse=True))
        ord_rx = rf"(?:(?:ўн|o[ʻʼ'`‘’]?n)\s+)?(?:{alt})"
        sub = {"{NUM}": cfg["num"], "{DASH}": cfg["dash"], "{SEP}": cfg["sep"], "{ORD}": ord_rx}

        def comp(v):
            for a, b in sub.items():
                v = v.replace(a, b)
            return re.compile("(?i)" + v)

        _P = {k: comp(v) for k, v in cfg.items() if isinstance(v, str) and k not in ("num", "dash", "sep")}
        ab = cfg.get("abbreviations") or {}
        ABBREV.clear()
        ABBREV.update({k.lower(): v for k, v in ab.items()})
        _P["abbrev"] = re.compile(r"(?i)(?<!\w)(?P<abbr>" + "|".join(re.escape(k) for k in sorted(ab, key=len, reverse=True))
                                  + r")(?:ning|нинг|da|да|ga|га|ni|ни)?\s*$") if ab else re.compile(r"(?!x)x")
        _P["_ords"] = ords
        _P["_num"] = re.compile(cfg["num"])
        _P["_ord"] = re.compile("(?i)" + ord_rx)
        _P["_dash"] = re.compile(cfg["dash"])
    return _P


_SEG: dict | None = None


def _segment_words() -> dict:
    """node type -> title word of EXT stub segments (graph.yaml ext_stub_segments)."""
    global _SEG
    if _SEG is None:
        segs = config.load("graph").get("ext_stub_segments") or {}
        _SEG = {v["type"]: v["word"] for v in segs.values()}
    return _SEG


def _ord_value(tok: str) -> int | None:
    p = _pats()
    t = re.sub(r"\s+", " ", tok.strip().lower())
    m = re.match(r"^(\d+)", t)
    if m:
        return int(m.group(1))
    ten = 0
    m2 = re.match(r"^(?:ўн|o[ʻʼ'`‘’]?n) (.+)$", t)
    if m2:
        ten, t = 10, m2.group(1)
    v = p["_ords"].get(t)
    return ten + v if v else None


def _parse_nums(s: str) -> list[str]:
    """"5, 6 ва 27" -> ["5", "6", "27"]; "12–14" -> 12..14 (ranges of <= 30); canonical numbers."""
    p = _pats()
    toks = [(m.group(0), m.start(), m.end()) for m in p["_num"].finditer(s)]
    out: list[str] = []
    for i, (t, a, b) in enumerate(toks):
        dash = i and p["_dash"].fullmatch(s[toks[i - 1][2]:a].strip() or "x")
        pv, cv = (canon_num(toks[i - 1][0]), canon_num(t)) if i else ("", "")
        if dash and t.isdecimal() and toks[i - 1][0].isdecimal() and 0 < int(t) - int(toks[i - 1][0]) <= 30:
            out += [str(k) for k in range(int(toks[i - 1][0]) + 1, int(t) + 1)]
        elif dash and re.fullmatch(r"\d+\.\d+", cv) and re.fullmatch(r"\d+(?:\.\d+)?", pv) \
                and pv.split(".")[0] == cv.split(".")[0] and 0 < int(cv.split(".")[1]) - int(pv.split(".")[1] if "." in pv else 0) <= 30:
            base, lo = cv.split(".")[0], int(pv.split(".")[1]) if "." in pv else 0
            out += [f"{base}.{k}" for k in range(lo + 1, int(cv.split(".")[1]) + 1)]   # "389¹ – 389⁵"
        elif dash and re.fullmatch(r"\d+\.\d+", cv) and pv.isdecimal() and 0 < int(cv.split(".")[0]) - int(pv) <= 30:
            b, k = cv.split(".")                                 # "383 – 385¹", "416 – 419³" (R29 A: 419¹, 419²)
            out += [str(j) for j in range(int(pv) + 1, int(b) + 1)] + [f"{b}.{j}" for j in range(1, int(k) + 1)]
        else:
            out.append(canon_num(t))
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def _parse_parts(s: str) -> list[str]:
    p = _pats()
    vals = []
    for m in re.finditer(rf"(?i)(?:{p['_ord'].pattern[4:]}|\d+)", s):
        v = _ord_value(m.group(0))
        if v:
            vals.append(str(v))
    return list(dict.fromkeys(vals))


# --- record helpers ---------------------------------------------------------------------------------

def structure_nodes(rec) -> list[GraphNode]:
    """Graph vertices of one record's structure: the документ node -> kind document, the rest ->
    kind element (type / title / number / doc_id)."""
    out = []
    for n in rec.structure:
        if n.parent_id is None:
            out.append(GraphNode(n.node_id, "document", n.type, rec.meta.title or n.title or "", rec.doc_id,
                                 False, {"doc_type": rec.meta.type}))
        else:
            out.append(GraphNode(n.node_id, "element", n.type, n.title or "", rec.doc_id, False,
                                 {"number": n.number}))
    return out


class DocIndex:
    """Spans and article numbers of one record's structure."""

    def __init__(self, rec):
        self.rec = rec
        self.nodes = {n.node_id: n for n in rec.structure}
        self.children: dict[str, list] = defaultdict(list)
        for n in rec.structure:
            if n.parent_id is not None:
                self.children[n.parent_id].append(n)
        self.arts: dict[str, list] = defaultdict(list)
        for n in rec.structure:
            if n.type == "статья" and n.number:
                self.arts[n.number].append(n)
        self._sorted = sorted(rec.structure, key=lambda n: (n.start, -n.end))
        self._starts = [n.start for n in self._sorted]

    def node_at(self, start: int, end: int):
        """The deepest node whose span contains [start, end) (μ of a mention span)."""
        best, depth = None, -1
        i = bisect.bisect_right(self._starts, start)
        for n in self._sorted[:i]:
            if n.start <= start and end <= n.end:
                d = n.node_id.count("/")
                if d > depth or (d == depth and n.end - n.start < best.end - best.start):
                    best, depth = n, d
        return best

    def ancestors(self, nid: str) -> list[str]:
        out = []
        n = self.nodes.get(nid)
        while n is not None and n.parent_id is not None:
            out.append(n.parent_id)
            n = self.nodes.get(n.parent_id)
        return out

    def enclosing(self, nid: str, typ: str):
        n = self.nodes.get(nid)
        while n is not None:
            if n.type == typ:
                return n
            n = self.nodes.get(n.parent_id) if n.parent_id else None
        return None

    def descendants(self, nid: str, typ: str, number: str) -> list:
        out, stack = [], list(self.children.get(nid, []))
        while stack:
            n = stack.pop()
            if n.type == typ and n.number == number:
                out.append(n)
            stack += self.children.get(n.node_id, [])
        return out

    def element(self, art: str, part: str | None, point: str | None) -> tuple[str | None, str]:
        """(node_id, status): status "ok" | "truncated" (a deeper level is missing) | "missing"."""
        arts = self.arts.get(art)
        if not arts:
            return None, "missing"
        a = arts[0]
        cur, status = a.node_id, "ok"
        if part:
            ps = [c for c in self.children.get(a.node_id, []) if c.type == "часть" and c.number == part]
            if ps:
                cur = ps[0].node_id
            else:
                return cur, "truncated"
        if point:
            pts = self.descendants(cur, "пункт", point)
            if len(pts) == 1:
                cur = pts[0].node_id
            else:
                status = "truncated"
        return cur, status


def _paragraphs(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for t in text.split("\n\n"):
        out.append((pos, pos + len(t)))
        pos += len(t) + 2
    return out


class QuoteMask:
    """Whether a position lies inside quoted text: a paragraph of an inserted / cited norm
    (structure._own_flags) or an inline «…» / “…” quote within its paragraph."""

    def __init__(self, text: str):
        self.text = text
        self.paras = _paragraphs(text)
        self.starts = [a for a, _ in self.paras]
        P = [text[a:b] for a, b in self.paras]
        try:
            self.own = _own_flags(P)
        except Exception:                               # pragma: no cover - defensive
            self.own = [True] * len(P)

    def para(self, pos: int) -> int:
        return max(0, bisect.bisect_right(self.starts, pos) - 1)

    def inside(self, pos: int) -> bool:
        i = self.para(pos)
        if not self.own[i]:
            return True
        a, _ = self.paras[i]
        depth = 0
        for ch in self.text[a:pos]:
            if ch in "«“":
                depth += 1
            elif ch in "»”" and depth:
                depth -= 1
        return depth > 0


# --- context over the corpus -----------------------------------------------------------------------

REF_PROMPT = """You resolve a citation of a law in a document of the Senate of the Oliy Majlis of \
Uzbekistan (Uzbek in Cyrillic or Latin script, sometimes Russian). You get the cited name as it \
stands in the text, the text around it, the citing document's type and date, the adoption date the \
text states for the cited law (if any), and a numbered list of candidate laws from the corpus (date \
of adoption by the Legislative Chamber, title, and the beginning of the law's preamble, which says \
what the law is about and what it changes).

Decide which candidate the citation names. Several candidates may share one title (omnibus laws \
"on amendments to certain legislative acts"): tell them apart by the subject matter, the codes and \
laws named in the context, and dates. Rules:
- A date decides only if exactly one candidate has that date; candidates with the same date and \
title can only be told apart by their subject.
- The subject in the context must match what the candidate's own preamble or title says; a subject \
that no candidate's preamble mentions means the citation names another law: answer null.
- A candidate adopted after the citing document's date (beyond a few weeks) cannot be meant.
- A title that differs from the candidate's title in its subject words (not only in spelling or \
word endings) names another law: answer null.
- If the context does not show clearly that one candidate is meant, answer null.

Answer as JSON {"choice": <candidate number or null>, "evidence": "<a short fragment copied \
character for character from the context (not from the candidates) that shows the decisive subject \
or date; empty string for null>"}. Give no other text."""
REF_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": ["integer", "null"]}, "evidence": {"type": "string"}},
    "required": ["choice", "evidence"], "additionalProperties": False}


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_STOP = ("qonun", "kodeks", "oʼzga", "qoʼsh", "kirit", "toʼgʼ", "haqid", "respu", "oʼzbe", "qabul", "qilin",
         "majli", "senat", "palat", "qoʼmi", "masal", "hujja", "ayrim", "loyih", "modda", "yanad", "takom",
         "munos", "bilan", "tomon", "kuni", "yilda", "kiriti", "tizim", "sohas")
_DATE_IN = re.compile(r"(\d{4})\s*[-–]?\s*(?:йил|yil)\w*\s+(\d{1,2})\s*[-–]?\s*([^\W\d]+)", re.I)


def _stems(text: str) -> set[str]:
    k = skeleton_key(text, False)
    return {w[:5] for w in re.findall(r"[a-zʼ]+", k) if len(w) >= 5 and not w[:5].startswith(_STOP)}


MIN_SUBJECT_HITS = 2          # distinctive subject stems the evidence must share with the chosen act
MAX_STEM_DF = 0.2             # a stem in more than this share of corpus-law titles+preambles is not distinctive


def _supports(ev: str, ch: int, cands: list[dict], df: dict | None = None, n_docs: int = 0) -> bool:
    """Deterministic check of an LLM choice: the evidence names the chosen candidate's date (and no
    other candidate has it), or >= MIN_SUBJECT_HITS distinctive subject stems (rare across the corpus
    laws, `df`) of its title / preamble that are not all shared by every other candidate."""
    chosen = cands[ch]
    dates = {_parse_date(m.group(0)) for m in _DATE_IN.finditer(ev)}
    if chosen["date"] and chosen["date"] in dates and sum(c["date"] == chosen["date"] for c in cands) == 1:
        return True
    ev_st = _stems(ev)
    if df is not None and n_docs:
        ev_st = {t for t in ev_st if df.get(t, 0) <= max(2.0, MAX_STEM_DF * n_docs)}
    own = _stems(chosen["title"] + " " + chosen["summary"])
    others = [_stems(c["title"] + " " + c["summary"]) for i, c in enumerate(cands) if i != ch]
    hits = {t for t in ev_st if t in own}
    if len(hits) < MIN_SUBJECT_HITS:
        return False
    return not others or any(not all(t in o for o in others) for t in hits)


@dataclass
class RefContext:
    registry: ActRegistry
    schema: Schema
    docs: dict = field(default_factory=dict)            # doc_id -> DocIndex
    llm: object | None = None                           # graphrag.kg.llm.KgLLM (None: queue only)
    stubs: dict = field(default_factory=dict)           # EXT element stubs node_id -> GraphNode
    stub_edges: dict = field(default_factory=dict)      # (parent, child) -> CONTAINS GraphEdge
    unresolved: list = field(default_factory=list)
    queue: list = field(default_factory=list)           # ref_resolve inputs not sent (no LLM)
    decisions: dict = field(default_factory=dict)       # group key -> ActRef | None
    llm_log: list = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    fuzzy: bool = True
    unres_seen: set = field(default_factory=set)
    recovered: list = field(default_factory=list)       # (doc_id, start, end, rule) of fix-round-1 local rules
    mention_cache: dict = field(default_factory=dict)   # doc_id -> resolve_record(rec) (noted once per record)
    violations: list = field(default_factory=list)      # Task 5: edges emitted against §3.4 (REPEALS without date)
    repealed_on: dict = field(default_factory=dict)     # Task 5: node_id -> latest repeal date
    unnoted: set = field(default_factory=set)           # R29 D: mention spans settled from EXT to a corpus act

    @classmethod
    def from_records(cls, records, registry: ActRegistry, llm=None, schema: Schema | None = None) -> "RefContext":
        ctx = cls(registry, schema or Schema.load(), llm=llm)
        for r in records:
            ctx.docs[r.doc_id] = DocIndex(r)
        ctx._corpus_keys = sorted(registry.acts)
        return ctx

    def record_mentions(self, rec) -> list:
        """resolve_record(rec) once per record: its registry notes (EXT display forms, mention
        counts) must not be counted twice when several extractors (Task 4, Task 5) read it."""
        if rec.doc_id not in self.mention_cache:
            self.mention_cache[rec.doc_id] = resolve_record(rec, self.registry)
        return self.mention_cache[rec.doc_id]

    # nodes known to this context (for Schema.validate_edge)
    def get(self, nid: str, default=None):
        if nid in self.stubs:
            return self.stubs[nid]
        if nid.startswith(EXT_PREFIX):
            return GraphNode(nid, "document", "документ", nid, None, True, {})
        doc = nid.split("/", 1)[0]
        di = self.docs.get(doc)
        if di is None or nid not in di.nodes:
            return default
        n = di.nodes[nid]
        return GraphNode(nid, "document" if n.parent_id is None else "element", n.type, n.title or "",
                         doc, False, {})

    def stub(self, act_nid: str, chain: list[tuple[str, str, str]]) -> str:
        """EXT element stub chain under act vertex `act_nid`: [(code, type, number)] -> deepest id."""
        parent = act_nid
        for code, typ, num in chain:
            nid = f"{parent}/{code}.{num}"
            if nid not in self.stubs:
                unit = _segment_words().get(typ, typ)
                self.stubs[nid] = GraphNode(nid, "element", typ, f"{num}-{unit}", None, True,
                                            {"number": num, "act": act_nid})
                self.stub_edges[(parent, nid)] = GraphEdge(parent, nid, "CONTAINS", {})
            parent = nid
        return parent

    def stub_nodes(self) -> list[GraphNode]:
        return sorted(self.stubs.values(), key=lambda n: n.node_id)

    def contains_edges(self) -> list[GraphEdge]:
        return [self.stub_edges[k] for k in sorted(self.stub_edges)]

    # --- ref_resolve --------------------------------------------------------------------------------
    def candidate_info(self, doc_id: str) -> dict:
        di = self.docs.get(doc_id)
        act = self.registry.doc_act.get(doc_id)
        title = _ws(act.title if act else (di.rec.meta.title if di else doc_id))
        date = act.date if act else None
        summ = ""
        if di is not None:
            pre = next((n for n in di.rec.structure if n.type == "преамбула"), None)
            body = di.rec.text[pre.start:pre.end] if pre else di.rec.text[:1500]
            m = re.search(r"(?:қабул қилинган|qabul qilingan)\s*", body)
            summ = _ws(body[m.end():] if m else body)[:700]
        return {"doc_id": doc_id, "date": date, "title": title, "summary": summ}

    def choose(self, group, rec, start: int, end: int, ref: ActRef, candidates: tuple, reason: str,
               stated: str | None = None) -> ActRef | None:
        """The candidate act the LLM picks for this mention group (cached per group), or None."""
        if group in self.decisions:
            return self.decisions[group]
        cands = [self.candidate_info(c) for c in candidates][:5]
        a, b = max(0, start - CTX_CHARS), min(len(rec.text), end + CTX_CHARS)
        context = rec.text[a:b]
        inp = {"document": {"doc_id": rec.doc_id, "type": rec.meta.type, "date": record_date(rec),
                            "title": _ws(rec.meta.title or "")[:300]},
               "citation": rec.text[start:end], "citation_stated_date": stated, "context": context,
               "candidates": [{"n": i, "date": c["date"], "title": c["title"], "preamble": c["summary"]}
                              for i, c in enumerate(cands)]}
        item = {"group": list(map(str, group)), "doc_id": rec.doc_id, "start": start, "end": end,
                "quote": rec.text[start:end], "reason": reason, "candidates": [c["doc_id"] for c in cands]}
        if self.llm is None:
            self.queue.append(item)
            self.decisions[group] = None
            return None
        ctx_norm = _ws(context)

        def verify(ans):
            if not isinstance(ans, dict) or "choice" not in ans:
                return False, "shape"
            ch, ev = ans.get("choice"), ans.get("evidence")
            if ch is None:
                return True
            if not (isinstance(ch, int) and not isinstance(ch, bool) and 0 <= ch < len(cands)):
                return False, "choice_not_candidate"
            if not isinstance(ev, str) or len(_ws(ev)) < 3 or _ws(ev) not in ctx_norm:
                return False, "evidence_not_in_context"
            if not _supports(ev, ch, cands, *self.stem_df()):
                return False, "evidence_not_distinctive"
            return True

        ans = self.llm.cached_json(PASS, inp, REF_SCHEMA, verify, prompt=REF_PROMPT)
        out = None
        if ans is not None and ans.get("choice") is not None:
            doc = cands[ans["choice"]]["doc_id"]
            act = self.registry.doc_act.get(doc)
            out = ActRef(ref.key, doc, False, act.title if act else ref.display)
        item["answer"] = ans
        item["raw"] = getattr(self.llm, "last_raw", None)
        item["verdict"] = ("rejected" if ans is None else "none" if ans.get("choice") is None else "chosen")
        item["chosen"] = out.doc_id if out else None
        self.llm_log.append(item)
        self.decisions[group] = out
        return out

    def stem_df(self) -> tuple[dict, int]:
        """Document frequency of subject stems over the corpus laws' titles + preambles."""
        if getattr(self, "_df", None) is None:
            df: Counter = Counter()
            heads = sorted({a.head for v in self.registry.acts.values() for a in v})
            for h in heads:
                c = self.candidate_info(h)
                df.update(_stems(c["title"] + " " + c["summary"]))
            self._df = (dict(df), len(heads))
        return self._df

    def fuzzy_candidates(self, key: str) -> tuple:
        """Corpus acts whose key is close to an external act_title key (a typo / variant spelling)."""
        if not self.fuzzy or not key:
            return ()
        close = difflib.get_close_matches(key, self._corpus_keys, n=3, cutoff=0.9)
        heads = []
        for k in close:
            if is_generic(k):
                continue
            heads += [a.head for a in self.registry.acts[k]]
        return tuple(dict.fromkeys(heads))[:5]


# --- amendment context --------------------------------------------------------------------------------

def _head_para(rec, n) -> tuple[int, int]:
    a = n.start
    b = rec.text.find("\n\n", a, n.end)
    return a, (n.end if b < 0 else b)


def amendment_context(rec, registry: ActRegistry, mentions=None) -> dict[str, ActRef]:
    """node_id -> the act its amendment instruction changes, for every статья / часть / пункт /
    подпункт whose head paragraph (outside quotes) names an act and has an amendment verb
    ("… кодексига қуйидаги ўзгартишлар киритилсин:", "… Қонуннинг 5-моддаси … билан тўлдирилсин"),
    and for all its descendants (a deeper instruction overrides). Only acts with a vertex
    (corpus or EXT) qualify; the mention closest before the verb in the dative wins, else the first."""
    p = _pats()
    if mentions is None:
        mentions = resolve_record(rec, registry)
    ms = sorted(mentions, key=lambda x: x[0].start)
    starts = [m.start for m, _ in ms]
    qm = QuoteMask(rec.text)
    gaz = _gazette_spans(rec.text)
    own: dict[str, ActRef] = {}
    for n in rec.structure:
        if n.parent_id is None or n.type not in ("статья", "часть", "пункт", "подпункт"):
            continue
        a, b = _head_para(rec, n)
        if qm.inside(a) and not qm.own[qm.para(a)]:
            continue
        head = rec.text[a:b]
        v = p["amend_verb"].search(head)
        if not v:
            continue
        stop = a + v.start()
        i, j = bisect.bisect_left(starts, a), bisect.bisect_left(starts, stop)
        cands = [(m, r) for m, r in ms[i:j] if m.end <= stop and r.node_id and not _in_spans(gaz, m.start)
                 and not qm.inside(m.start)]
        # nested titles: keep the outermost mention of each overlap group
        cands = [(m, r) for m, r in cands if not any(o.start <= m.start and m.end <= o.end and (o.start, o.end) != (m.start, m.end)
                                                     for o, _ in cands)]
        if not cands:
            continue
        dative = [(m, r) for m, r in cands
                  if re.search(r"(?i)(?:га|ga|ка|ka|ға|g[ʻʼ']a)\s*$", rec.text[m.start:m.end])]
        own[n.node_id] = (dative[-1] if dative else cands[0])[1]
    out: dict[str, ActRef] = {}
    by_id = {n.node_id: n for n in rec.structure}
    for n in rec.structure:                                  # structure is in document order, parents first
        if n.node_id in own:
            out[n.node_id] = own[n.node_id]
        elif n.parent_id in out:
            out[n.node_id] = out[n.parent_id]
    return {k: out[k] for k in out if k in by_id}


# --- extraction ---------------------------------------------------------------------------------------

@dataclass
class _Art:
    start: int
    end: int
    arts: list
    parts: list
    points: list
    kind: str            # plain | own_article | ru


def _gazette_spans(text: str) -> list[tuple[int, int]]:
    """Parentheses that cite an official gazette (their article numbers are gazette items)."""
    p = _pats()
    spans, stack = [], []
    for i, ch in enumerate(text):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            a = stack.pop()
            if not stack and p["gazette"].search(text, a, i):
                spans.append((a, i + 1))
    return spans


def _in_spans(spans, pos) -> bool:
    i = bisect.bisect_right([a for a, _ in spans], pos) - 1
    return i >= 0 and spans[i][0] <= pos < spans[i][1]


def _scan_articles(text: str, gaz) -> list[_Art]:
    p = _pats()
    out = []
    starts = [a for a, _ in gaz]

    def gz(pos):
        i = bisect.bisect_right(starts, pos) - 1
        return i >= 0 and gaz[i][0] <= pos < gaz[i][1]

    for m in p["article"].finditer(text):
        s = m.start()
        if gz(s) or p["gazette_before"].search(text, max(0, s - 40), s):
            continue
        ps = text.rfind("\n\n", 0, s)
        ps = 0 if ps < 0 else ps + 2
        if not text[ps:s].strip(' «“"') and (p["heading"].match(text, ps) or p["ru_heading"].match(text, ps)):
            out.append(_Art(s, m.end(), _parse_nums(m.group("nums")), [], [], "heading"))
            continue                                           # "5-модда." / «74¹-модда. heading
        arts = _parse_nums(m.group("nums"))
        end, parts, points = m.end(), [], []
        pm = p["part"].match(text, end)
        if pm:
            parts, end = _parse_parts(pm.group("parts")), pm.end()
        qm = p["point"].match(text, end)
        if qm:
            points, end = _parse_nums(qm.group("points")), qm.end()
        out.append(_Art(s, end, arts, parts, points, "plain"))
    for m in p["own_article"].finditer(text):
        end = m.end()
        pm = p["part"].match(text, end)
        parts, points = [], []
        if pm:
            parts, end = _parse_parts(pm.group("parts")), pm.end()
        qm = p["point"].match(text, end)
        if qm:
            points, end = _parse_nums(qm.group("points")), qm.end()
        if parts or points:
            out.append(_Art(m.start(), end, [], parts, points, "own_article"))
    for m in p["ru_article"].finditer(text):
        if gz(m.start()):
            continue
        ps = text.rfind("\n\n", 0, m.start())
        ps = 0 if ps < 0 else ps + 2
        if not text[ps:m.start()].strip(' «“"') and p["ru_heading"].match(text, ps):
            out.append(_Art(m.start(), m.end(), _parse_nums(m.group("nums")), [], [], "heading"))
            continue
        parts = [str(v) for v in [_ord_value(m.group("part"))] if v] if m.group("part") else []
        out.append(_Art(m.start(), m.end(), _parse_nums(m.group("nums")), parts, [], "ru"))
    out.sort(key=lambda x: (x.start, -x.end))
    res, last_end = [], -1                                   # drop overlaps (own_article inside plain etc.)
    for x in out:
        if x.start >= last_end:
            res.append(x)
            last_end = x.end
    return res


def _title_in(rec, title: str, a: int, b: int) -> tuple[int, int] | None:
    """Whitespace-tolerant first occurrence of `title` in rec.text[a:b]."""
    words = title.split()
    if not words:
        return None
    rx = re.compile(r"\s+".join(re.escape(w) for w in words))
    m = rx.search(rec.text, a, b)
    return (m.start(), m.end()) if m else None


class _Emitter:
    def __init__(self, rec, ctx: RefContext):
        self.rec, self.ctx = rec, ctx
        self.edges: dict[tuple, dict] = {}

    def add(self, src: str, dst: str | None, start: int, end: int, rel: str) -> bool:
        if not dst or not src:
            return False
        di = self.ctx.docs[self.rec.doc_id]
        if dst == src or dst in di.ancestors(src):
            self.ctx.stats["dropped_self_link"] += 1
            return False
        ev = Evidence(self.rec.doc_id, start, end, self.rec.text[start:end]).to_json()
        k = (src, dst, rel_class(rel))
        e = self.edges.get(k)
        if e is None:
            self.edges[k] = {"rel": rel, "ev": [ev]}
        else:
            if ev not in e["ev"]:
                e["ev"].append(ev)
            if REL_PRIORITY.index(rel) < REL_PRIORITY.index(e["rel"]):
                e["rel"] = rel
        return True

    def build(self) -> list[GraphEdge]:
        out = []
        for (src, dst, _), e in sorted(self.edges.items()):
            evs = sorted(e["ev"], key=lambda x: (x["start"], x["end"]))
            attrs = {"evidence": evs[0], "relation": e["rel"]}
            if len(evs) > 1:
                attrs["evidence_all"] = evs
            edge = GraphEdge(src, dst, "REFERS_TO", attrs)
            errs = self.ctx.schema.validate_edge(edge, self.ctx)
            for ev in evs:
                if self.rec.text[ev["start"]:ev["end"]] != ev["quote"]:
                    errs.append("evidence quote is not the text span")
            if errs:
                self.ctx.stats["schema_rejected"] += 1
                self.ctx.unresolved.append({"doc_id": self.rec.doc_id, "reason": "schema", "errors": errs[:3],
                                            "src": src, "dst": dst})
                continue
            out.append(edge)
        return out


def _unres(ctx, rec, start, end, reason, **kw):
    k = (rec.doc_id, start, end, reason)
    if k in ctx.unres_seen:                              # title / agenda mentions are settled twice
        return
    ctx.unres_seen.add(k)
    ctx.unresolved.append({"doc_id": rec.doc_id, "start": start, "end": end, "quote": rec.text[start:end],
                           "reason": reason, **kw})
    ctx.stats[f"unresolved_{reason}"] += 1


def _group(rec, di: DocIndex, pos: int, key: str) -> tuple:
    n = di.node_at(pos, pos + 1)
    q = di.enclosing(n.node_id, "вопрос") if n is not None else None
    return (rec.doc_id, q.node_id if q else rec.doc_id, key)


def _date_feasible(reg: ActRegistry, head: str, d: str | None, m) -> bool:
    """A corpus act adopted by the citing record's date (+ grace) and, when the mention states the
    cited act's adoption date, compatible with it (the resolve() rules)."""
    act = reg.doc_act.get(head)
    ok = list(act.dates) if act and act.dates else []
    if not ok:
        return True
    if d:
        ok = [t for t in ok if _days(t, d) >= -DATE_GRACE_DAYS]
    if m is not None and m.date:
        if m.date_kind == "chamber":
            ok = [t for t in ok if abs(_days(t, m.date)) <= DATE_GRACE_DAYS]
        else:
            ok = [t for t in ok if 0 <= _days(t, m.date) <= CITED_MAX_GAP_DAYS]
    return bool(ok)


def _settle(ctx: RefContext, rec, di, start, end, ref: ActRef, mention=None) -> ActRef | None:
    """An act with a vertex for this mention: corpus / external as resolved, ambiguous via ref_resolve
    (typo titles close to a corpus key too, when date-feasible), else None (unlinked; logged)."""
    stated = mention.date if mention is not None else None
    if ref.external and ref.node_id and not ref.edition:
        cands = ctx.fuzzy_candidates(ref.key) if ref.key not in ctx.registry.acts else ()
        own = ctx.registry.doc_act.get(rec.doc_id, _NoAct).versions
        cands = tuple(c for c in cands if c != rec.doc_id and c not in own
                      and _date_feasible(ctx.registry, c, record_date(rec), mention))
        if cands:
            got = ctx.choose(_group(rec, di, start, ref.key), rec, start, end, ref, cands, "fuzzy_title", stated)
            ctx.stats["fuzzy_title_" + ("linked" if got else "kept_external")] += 1
            if got and (tag := (rec.doc_id, start, end, ref.key, ref.edition)) not in ctx.unnoted:
                ctx.unnoted.add(tag)                     # R29 D: this mention is no longer an EXT mention
                ctx.registry.unnote_mention(ref)
            return got or ref
    if ref.node_id:
        return ref
    if ref.ambiguous:
        got = ctx.choose(_group(rec, di, start, ref.key), rec, start, end, ref, ref.candidates,
                         "omnibus_undated" if is_generic(ref.key) else "same_key_tie", stated)
        if got is None:
            _unres(ctx, rec, start, end, "ambiguous_unsettled", key=ref.key, candidates=list(ref.candidates))
        return got
    own = ctx.registry.doc_act.get(rec.doc_id)
    why = ("self_title" if own is not None and own.key == ref.key
           else "omnibus_no_feasible_act" if is_generic(ref.key) else "unlinked_act")
    _unres(ctx, rec, start, end, why, key=ref.key)
    return None


class _NoAct:
    versions: list = []


def _element_target(ctx: RefContext, rec, act: ActRef, art: str, part: str | None, point: str | None,
                    start: int, end: int) -> tuple[str | None, str]:
    """(node_id, status) of an element of `act`: a real node of a corpus act (missing -> None,
    "missing"), else an EXT stub chain."""
    if act.doc_id:
        di = ctx.docs.get(act.doc_id)
        if di is None:
            return None, "no_structure"
        nid, st = di.element(art, part, point)
        if st == "truncated":
            ctx.stats["element_truncated"] += 1
        return nid, st
    chain = [("СТ", "статья", art)]
    if part:
        chain.append(("Ч", "часть", part))
        if point:
            chain.append(("П", "пункт", point))
    return ctx.stub(act.node_id, chain), "stub"


def _targets(a: _Art):
    arts = a.arts or [None]
    for art in arts:
        parts = a.parts or [None]
        for part in parts:
            pts = a.points if (a.points and (part == parts[-1])) else [None]
            if len(arts) > 1 and art != arts[-1]:
                part, pts = None, [None]                         # "5, 6 ва 27-моддасининг иккинчи қисми"
            for pt in pts:
                yield art, part, pt


def subject_act(ctx: RefContext, rec, di: DocIndex, mentions) -> ActRef | None:
    """The act a qaror / xulosa record is about (its meta.title), with the title-span evidence."""
    res = _titled_act(ctx, rec, di, mentions)
    return res[0] if res else None


def _outer_mention(mentions, a: int, b: int):
    inside = [(m, r) for m, r in mentions if m.start < b and m.end > a]
    if not inside:
        return None
    return max(inside, key=lambda x: (x[0].end - x[0].start, -x[0].start))


def _title_like(title: str) -> bool:
    k = act_key(title)
    p = _pats()
    return bool(k) and bool(re.search(r"(?:toʼgʼrisida|kodeks|konstitusiya)$", k)) \
        and not p["title_block"].search(_ws(title)) and not p["title_block_any"].search(_ws(title))


def _act_for_title(ctx, rec, di, mentions, title: str, a: int, b: int, mode: str):
    """(ActRef, start, end) of the act a title at text span [a, b) names; mode "agenda" | "title".

    The record's own mention of it (resolve_record: per-record consistency) when one overlaps the
    span and has the title's act key (or its nested base law has); else registry.resolve(title)
    (Ruling R7 via r7_undated) when that is a corpus act or ambiguous; else, for an agenda item
    only, an EXT act when the title is law-like (patterns.refs agenda_law_after / agenda_law_title).
    An agenda title with a non-act word anywhere (agenda_block) names no act."""
    p = _pats()
    text = rec.text
    key = act_key(title)
    key = ctx.registry.aliases.get(key, key)
    om = _outer_mention(mentions, a, b)
    if om is not None:
        m, ref = om
        nested = any(r.key == key and m.start <= x.start and x.end <= m.end and (x.start, x.end) != (m.start, m.end)
                     for x, r in mentions)
        if ref.key == key or nested:     # its own mention, or the title is the base law nested in it
            return ref, m.start, m.end   # ("«X»ги Қонунига ўзгартиш киритиш ҳақида»ги Қонун": that law)
    if not key:
        return None
    if mode == "agenda":
        # law-like evidence: "…]gi qonun", "qonun “[…]”", a code / ratification / accession title
        lawlike = bool(p["agenda_law_after"].match(text, b) or p["agenda_law_before"].search(text, max(0, a - 40), a)
                       or p["agenda_law_title"].search(title))
        quoted = a > 0 and text[a - 1] in "«“„\"" and b < len(text) and text[b] in "»”\""
        if not lawlike and p["agenda_block"].search(_ws(title)):
            ctx.stats["agenda_blocked"] += 1
            return None
    d = record_date(rec)
    ref = r7_undated(ctx.registry, ctx.registry.resolve(title, d, src_doc=rec.doc_id), d)
    if ref.doc_id or ref.ambiguous or (ref.unlinked and is_generic(ref.key)):
        return ref, a, b                                 # an omnibus title with no feasible act: _settle logs it
    if mode != "agenda" or not ref.external or not _title_like(title):
        return None
    if not (lawlike or quoted):
        ctx.stats["agenda_not_lawlike"] += 1
        return None
    ctx.registry.note_mention(ref, title, d)
    return ref, a, b


def _outer_quoted(text: str, s: int, e: int, hend: int) -> tuple[int, int] | None:
    """The quoted title that encloses a nested quoted title at [s, e) in the heading text[:hend]:
    (open quote position, close quote position), or None."""
    stack = []
    for i in range(0, s):
        ch = text[i]
        if ch in "«“„":
            stack.append(i)
        elif ch in "»”" and stack:
            stack.pop()
    inner = s > 0 and text[s - 1] in "«“„" and bool(stack) and stack[-1] == s - 1
    if inner:
        stack = stack[:-1]
    if not stack:
        return None
    q0, depth = stack[-1], 2 if inner else 1
    for i in range(s, hend):
        ch = text[i]
        if ch in "«“„":
            depth += 1
        elif ch in "»”":
            depth -= 1
            if depth == 0:
                return (q0, i) if i >= e else None
    return None


def _heading_act(ctx, rec, di, mentions, hend: int):
    """Ruling R29 C: (act, start, end) of the heading's first quoted act (outermost act-title mention before
    `hend`) for a qaror / xulosa whose meta.title is not in its heading (a decree / law cited in the body:
    XULOSALAR-8d225faf, 161d951d, …), settled exactly like the record's own act mention (no new LLM input).
    None when that mention is the base law nested in an amending-law title whose opening quote is missing
    ("“X”gi … Qonuniga … kiritish haqida”gi Qonuni", XULOSALAR-46ff44bd / fa8ed64a: the subject is the outer law)."""
    text = rec.text
    ms = sorted(((m, r) for m, r in mentions if m.kind == "act_title" and m.end <= hend), key=lambda x: (x[0].start, -x[0].end))
    ms = [(m, r) for m, r in ms if not any(o.start <= m.start and m.end <= o.end and (o.start, o.end) != (m.start, m.end)
                                           for o, _ in ms)]
    if not ms:
        return None
    m, ref = ms[0]
    eop = text.find("\n\n", m.end, hend)
    oc = _OUTER_CLOSE.search(text, m.end, hend if eop < 0 else eop)
    if oc is not None and not (set("«“„") & set(text[m.end:oc.start()])):
        ctx.stats["title_heading_outer_unquoted"] += 1
        return None
    act = _settle(ctx, rec, di, m.start, m.end, ref, m)
    return (act, m.start, m.end) if act else None


def _titled_act(ctx, rec, di, mentions):
    """(act, start, end) for a qaror / xulosa: the act its heading's title names. No link for a
    parliamentary inquiry (patterns.refs inquiry in the heading) or when meta.title is only found
    after the heading (a law cited in the body). A title that is the base law nested in a longer
    quoted amending-law title in the heading resolves to that amending law."""
    if rec.meta.section not in SEC_TITLED or is_law_text(rec) or not rec.meta.title:
        return None
    p = _pats()
    text = rec.text
    hm = p["heading_end"].search(text)
    hend = hm.start() if hm else min(len(text), 1500)
    ev = _title_span(rec)
    # the heading window ends with the heading's title paragraph: the paragraph holding the title span
    # when it is in the heading, else the first long paragraph (a long title paragraph is inside it)
    paras = _paragraphs(text)
    if ev.start < hend and not (ev.start == 0 and ev.end == len(text)):
        wend = next((b for a, b in paras if a <= ev.end <= b + 2), hend)
    else:
        wend = next((b for a, b in paras if b - a >= BODY_PARA_CHARS), len(text))
    if p["inquiry"].search(text[:min(max(wend, 1), hend, len(text))]):   # R29 C: the window never passes hend
        ctx.stats["title_skipped_inquiry"] += 1
        return None
    if ev.start >= hend or (ev.start == 0 and ev.end == len(text)):
        got = _heading_act(ctx, rec, di, mentions, hend)
        ctx.stats["title_heading_fallback" if got else "title_skipped_not_in_heading"] += 1
        return got
    oq = _outer_quoted(text, ev.start, ev.end, hend)
    if oq is not None:
        name = text[oq[0] + 1:oq[1]]
        d = record_date(rec)
        ref = r7_undated(ctx.registry, ctx.registry.resolve(name, d, src_doc=rec.doc_id), d)
        if is_amending(ref.key) and (ref.node_id or ref.ambiguous):
            if ref.external:
                ctx.registry.note_mention(ref, name, d)
            act = _settle(ctx, rec, di, oq[0], oq[1] + 1, ref)
            if act:
                ctx.stats["title_outer_amending"] += 1
                return act, oq[0], oq[1] + 1
    got = _act_for_title(ctx, rec, di, mentions, rec.meta.title, ev.start, ev.end, "title")
    if not got:
        return None
    ref, s, e = got
    act = _settle(ctx, rec, di, s, e, ref)
    return (act, ev.start, ev.end) if act else None


def extract_refs(rec, registry: ActRegistry, ctx: RefContext | None = None) -> list[GraphEdge]:
    """Validated REFERS_TO edges of one record (see the module docstring). Stubs, unresolved
    references and the ref_resolve queue accumulate in `ctx` (a fresh one over [rec] if None)."""
    if ctx is None:
        ctx = RefContext.from_records([rec], registry)
    if rec.doc_id not in ctx.docs:
        ctx.docs[rec.doc_id] = DocIndex(rec)
    di = ctx.docs[rec.doc_id]
    p = _pats()
    text = rec.text
    em = _Emitter(rec, ctx)
    mentions = ctx.record_mentions(rec) if ctx.registry is registry else resolve_record(rec, registry)
    law = is_law_text(rec)
    amend = amendment_context(rec, registry, mentions)
    qm = QuoteMask(text)
    gaz = _gazette_spans(text)
    ms = sorted(mentions, key=lambda x: (x[0].start, x[0].end))
    m_ends = sorted(((m.end, i) for i, (m, _) in enumerate(ms)))
    end_keys = [e for e, _ in m_ends]

    def src_of(a, b):
        n = di.node_at(a, b)
        return n.node_id if n else rec.doc_id

    # 7. subject act (qaror / xulosa title, stenogram вопрос titles)
    subj_doc = _titled_act(ctx, rec, di, mentions)
    if subj_doc:
        act, s, e = subj_doc
        if em.add(rec.doc_id, act.node_id, s, e, "title"):
            ctx.stats["title_edges_raw"] += 1
    q_act: dict[str, ActRef] = {}
    if rec.meta.type == "stenogram":
        for n in rec.structure:
            if n.type != "вопрос" or not (n.title or "").strip():
                continue
            span = _title_in(rec, n.title, n.start, n.end)
            if span is None:
                _unres(ctx, rec, n.start, n.start, "agenda_title_not_in_text", node=n.node_id)
                continue
            got = _act_for_title(ctx, rec, di, mentions, n.title, span[0], span[1], "agenda")
            if not got:
                ctx.stats["agenda_not_an_act"] += 1
                continue
            ref, s, e = got
            act = _settle(ctx, rec, di, s, e, ref)
            if act is None:
                continue
            q_act[n.node_id] = act
            em.add(n.node_id, act.node_id, span[0], span[1], "agenda")

    def subject_for(pos) -> ActRef | None:
        if subj_doc:
            return subj_doc[0]
        n = di.node_at(pos, pos + 1)
        q = di.enclosing(n.node_id, "вопрос") if n else None
        return q_act.get(q.node_id) if q else None

    def subject_target(pos: int, word: str) -> ActRef | None:
        """The act an anaphoric "Қонуннинг / ушбу Кодекснинг / loyihaning" names in a non-law
        record: its subject act when that is a standing law; for an amending subject act and the
        word "Кодекс", the one code the subject's title names; else None."""
        subj = subject_for(pos)
        if subj is None:
            return None
        if not is_amending(subj.key):
            return subj
        code = bool(re.match(r"(?i)(?:кодекс|kodeks)", word))
        title = ctx.registry.doc_act[subj.doc_id].title if subj.doc_id in ctx.registry.doc_act else subj.display
        codes = {}
        for m in scan_act_mentions(title):          # the act the amending subject changes (by the word's kind)
            if (m.kind in ("code_name", "code_list")) if code else (m.kind == "act_title"):
                r = ctx.registry.resolve(m.name)
                if r.node_id and not is_generic(r.key) and not is_amending(r.key):
                    codes[r.node_id] = r
        if len(codes) != 1:
            return None
        (r,) = codes.values()
        if r.external:
            ctx.registry.note_mention(r, r.display, None)
        return r

    def abbrev_act(before: str) -> ActRef | None:
        m = p["abbrev"].search(before)
        if not m:
            return None
        name = ABBREV[m.group("abbr").lower()]
        r = ctx.registry.resolve(name)
        if r.external:
            ctx.registry.note_mention(r, name, None)
        return r if r.node_id else None

    def const_act() -> ActRef | None:
        r = ctx.registry.resolve("Конституция")
        if r.external:
            ctx.registry.note_mention(r, "Конституция", None)
        return r if r.node_id else None

    ext_keys = ctx.registry.known_keys                     # pre-scanned: independent of record order

    def known_title_act(words: list[str]) -> ActRef | None:
        """The longest suffix of `words` (+ "тўғрисида") that is a known act (corpus or a quoted EXT)."""
        for k in range(len(words), 0, -1):
            name = " ".join(words[-k:])
            key = act_key(name)
            key = ctx.registry.aliases.get(key, key)
            if key and not is_generic(key) and (key in ctx.registry.acts or key in ext_keys):
                d = record_date(rec)
                r = ctx.registry.resolve(name, d, src_doc=rec.doc_id)
                if r.node_id:
                    if r.external:
                        ctx.registry.note_mention(r, name, d)
                    return r
        return None

    def recent_act(pos: int, word: str) -> ActRef | None:
        """"Қонуннинг / ушбу Кодекснинг" with no subject act: the one law (code) named last within
        the previous 400 characters, with no other act mention after it."""
        code = bool(re.match(r"(?i)(?:кодекс|kodeks)", word))
        near_ms = [(m, r) for m, r in ms if pos - 400 <= m.start and m.end <= pos]
        if not near_ms:
            return None
        m, r = near_ms[-1]
        if (m.kind in ("code_name", "code_list")) != code or m.kind == "constitution" or m.kind == "code_list":
            return None
        if p["plural_head"].match(text, m.end) or (len(near_ms) > 1 and p["list_join"].match(
                text[near_ms[-2][0].end:m.start])):
            return None                                      # a list of laws: which one is meant is open
        return _settle(ctx, rec, di, m.start, m.end, r, m)

    def local_rules(a, src, para):
        """Fix round 1 (R8): owners for a bare article that the strict rules leave open."""
        lo = max(0, a.start - 200)
        before = text[lo:a.start]
        # an act mention a few plain words before ("kodeksning yangi kiritilayotgan 34¹-moddasiga")
        j = bisect.bisect_right(end_keys, a.start) - 1
        if j >= 0:
            m, ref = ms[m_ends[j][1]]
            gap = text[m.end:a.start]
            if m.end <= a.start and p["loose_gap"].match(gap) and not p["article"].search(gap) \
                    and not p["prim_gap"].search(gap):
                act = _settle(ctx, rec, di, m.start, m.end, ref, m)
                if act is not None:
                    return ("act", act), "loose_gap_mention"
        if not law:
            sl = p["subject_before_loose"].search(before)
            if sl and not p["article"].search(sl.group(0)) and not p["prim_gap"].search(sl.group(0)):
                t = subject_target(a.start, sl.group(0).split()[0])
                if t is not None:
                    return ("subject", t), "loose_gap_subject"
        ut = p["unquoted_title"].search(before)
        if ut:
            r = known_title_act(ut.group("t").split())
            if r is not None:
                return ("act", r), "unquoted_title"
        if not law and p["paren_bare"].search(before) and p["paren_close"].match(text, a.end):
            # "“X”gi Qonunda (83-modda)", "Jinoyat kodeksida … (167-modda)": the act named last in the sentence
            ss = max(text.rfind(". ", 0, a.start), text.rfind("\n", 0, a.start), 0)
            inside = [(m, r) for m, r in ms if ss <= m.start and m.end <= a.start]
            if inside and inside[-1][0].kind != "constitution":
                act = _settle(ctx, rec, di, inside[-1][0].start, inside[-1][0].end, inside[-1][1], inside[-1][0])
                if act is not None:
                    return ("act", act), "paren_sentence_act"
            subj = subject_for(a.start)
            if subj is not None and not is_amending(subj.key) and not p["ratify"].search(subj.display or ""):
                return ("subject", subj), None               # "… belgilanmoqda (4-modda)" in a speech on the law
        if a.kind == "ru":
            k = bisect.bisect_left([m.start for m, _ in ms], a.end)
            if k < len(ms) and ms[k][0].start - a.end <= 3 and not text[a.end:ms[k][0].start].strip():
                act = _settle(ctx, rec, di, ms[k][0].start, ms[k][0].end, ms[k][1], ms[k][0])
                if act is not None:
                    return ("act", act), "ru_act_after"
        # the Constitution named earlier in the same sentence, with no other act in between
        ss = max(text.rfind(". ", 0, a.start), text.rfind("\n", 0, a.start), 0)
        inside = [(m, r) for m, r in ms if ss <= m.start and m.end <= a.start]
        if inside and inside[-1][0].kind == "constitution":
            act = _settle(ctx, rec, di, inside[-1][0].start, inside[-1][0].end, inside[-1][1], inside[-1][0])
            if act is not None:
                return ("act", act), "constitution_sentence"
        # a section heading "II. Ер кодексида:" a few paragraphs up
        if not law:
            for k in range(para, max(-1, para - 8), -1):
                pa, pb = qm.paras[k]
                ptxt = text[pa:pb]
                if k < para and p["section_heading"].match(ptxt):
                    if ptxt.rstrip().endswith(":"):
                        hs = [(m, r) for m, r in ms if pa <= m.start and m.end <= pb]
                        if hs:
                            act = _settle(ctx, rec, di, hs[-1][0].start, hs[-1][0].end, hs[-1][1], hs[-1][0])
                            if act is not None:
                                return ("act", act), "section_heading"
                    break
        return None, None

    # article references
    consumed: set[int] = set()                               # indexes of act mentions owning an article
    arts = _scan_articles(text, gaz)
    last_art: dict[str, tuple] = {}                          # amendment node -> (act, article) last named in it
    prev = None                                              # (end, owner, paragraph)
    for a in arts:
        src = src_of(a.start, a.end)
        ctx_act, ctx_node = None, None
        for nid in [src] + di.ancestors(src):
            if nid in amend:
                ctx_act, ctx_node = amend[nid], nid
                break
        inq = qm.inside(a.start)
        para = qm.para(a.start)
        if a.kind == "heading":                              # «74¹-модда. … : the article "ушбу модда" names
            if inq and ctx_act is not None and a.arts:
                last_art[ctx_node] = (ctx_act, a.arts[0])
            continue
        owner = None
        before = text[max(0, a.start - 80):a.start]
        j = bisect.bisect_right(end_keys, a.start) - 1
        near = None                                          # an act mention right before (gap: space / one parenthesis)
        for jj in (j, j - 1):
            if jj < 0:
                continue
            mi = m_ends[jj][1]
            m, ref = ms[mi]
            if m.end <= a.start and p["act_gap"].match(text[m.end:a.start]):
                near = (mi, m, ref)
                break
        ob = p["own_act_before"].search(before)
        sb = p["subject_before"].search(before)
        rule = None                                          # a fix-round-1 local rule that decided the owner
        if a.kind == "own_article":
            if inq and ctx_act is not None:
                if ctx_node in last_art:
                    owner = ("amended_art",) + last_art[ctx_node]
            elif law or di.arts:
                art_node = di.enclosing(src, "статья")
                if art_node is not None:
                    owner = ("own_art", None, art_node.number)
            if owner is None:
                _unres(ctx, rec, a.start, a.end, "own_article_unknown", doc_type=rec.meta.type)
                prev = None
                continue
        elif a.kind == "ru" and p["ru_own_after"].match(text, a.end):
            owner = ("own", None)
        elif near is not None:
            mi, m, ref = near
            consumed.add(mi)
            act = _settle(ctx, rec, di, m.start, m.end, ref, m)
            if act is None:
                owner = ("dead", None)
            elif ctx_act is not None and not inq and act.node_id == ctx_act.node_id:
                owner = ("amended", act)                     # the amendment instruction's own head
            else:
                owner = ("act", act)
        elif ob:
            if inq and ctx_act is not None:
                owner = ("amended", ctx_act)
            elif law:
                owner = ("own", None)
            else:
                w = ob.group(0).split()[-1]
                t = subject_target(a.start, w)
                owner = ("subject", t) if t else None
                if t is not None and (sf := subject_for(a.start)) is not None and is_amending(sf.key) \
                        and not re.match(r"(?i)(?:кодекс|kodeks)", w):
                    rule = "subject_base_law"
                if owner is None and (t := recent_act(a.start, w)) is not None:
                    owner, rule = ("act", t), "recent_act"
        elif prev is not None and prev[0] <= a.start and (
                p["chain_gap"].match(text[prev[0]:a.start])
                or (prev[2] == para and p["chain_gap_loose"].match(text[prev[0]:a.start]))):
            owner = prev[1]
            if not p["chain_gap"].match(text[prev[0]:a.start]):
                rule = "chain_count"
        elif (ab := abbrev_act(before)) is not None:
            owner = ("act", ab)
        elif ctx_act is not None:
            owner = ("amended", ctx_act)
        elif p["basic_law"].search(before):
            owner, rule = ("act", const_act()), "basic_law"
        elif (ut := p["unquoted_title"].search(text, max(0, a.start - 200), a.start)) is not None \
                and (uta := known_title_act(ut.group("t").split())) is not None:
            owner, rule = ("act", uta), "unquoted_title"      # "xalqaro shartnomalari toʼgʼrisidagi qonunning 11-"
        elif not law and sb:
            w = sb.group(0).strip()
            t = subject_target(a.start, w)
            owner = ("subject", t) if t else None
            if t is not None and (sf := subject_for(a.start)) is not None and is_amending(sf.key) \
                    and not re.match(r"(?i)(?:кодекс|kodeks)", w):
                rule = "subject_base_law"                    # an amending subject: the one law it changes
            if owner is None and (t := recent_act(a.start, w)) is not None:
                owner, rule = ("act", t), "recent_act"
        elif p["owner_noun"].search(before):
            owner = None                                     # "konvensiyaning 12-moddasi": not our act, never carried
        else:
            owner, rule = local_rules(a, src, para)
            if owner is None and prev is not None and prev[2] == para and a.start - prev[0] <= CARRY_CHARS \
                    and prev[1][0] != "dead":
                owner = prev[1]                              # "Qonunning 43-moddasida … va 48-moddasida"
                ctx.stats["carried_in_paragraph"] += 1
        if owner is not None and owner[0] != "dead" and owner[1] is None and owner[0] not in ("own", "own_art"):
            owner = None
        if owner is None:
            _unres(ctx, rec, a.start, a.end, "no_act_context", doc_type=rec.meta.type)
            prev = None
            continue
        if rule:
            ctx.stats[f"rule_{rule}"] += 1
            ctx.recovered.append((rec.doc_id, a.start, a.end, rule))
        prev = (a.end, owner, para)
        kind = owner[0]
        if kind == "dead":
            continue
        if ctx_node is not None and not inq and kind in ("amended", "act") and a.arts and owner[1] is not None:
            last_art[ctx_node] = (owner[1], a.arts[-1])
        for art, part, pt in _targets(a):
            if kind in ("own", "own_art"):
                if kind == "own_art":
                    art = owner[2]
                nid, st = di.element(art, part, pt)
                if nid is None:
                    _unres(ctx, rec, a.start, a.end, "internal_missing_article", article=art)
                    continue
                em.add(src, nid, a.start, a.end, "internal")
                continue
            act = owner[1]
            if kind == "amended_art":
                art = owner[2]
            rel = {"act": "element", "amended": "amended", "amended_art": "amended", "subject": "element"}[kind]
            if rel == "amended" and inq:
                rel = "cited_in_amendment"                   # R20: cited inside the quoted new / replaced text
            if art is None:
                continue
            nid, st = _element_target(ctx, rec, act, art, part, pt, a.start, a.end)
            if nid is None:
                _unres(ctx, rec, a.start, a.end, "corpus_missing_article", act=act.node_id, article=art)
                em.add(src, act.node_id, a.start, a.end, "act")
                continue
            if em.add(src, nid, a.start, a.end, rel):
                ctx.stats[f"elem_{st}"] += 1

    # act mentions without an article
    for i, (m, ref) in enumerate(ms):
        if i in consumed:
            continue
        if _in_spans(gaz, m.start):
            continue
        act = _settle(ctx, rec, di, m.start, m.end, ref, m)
        if act is None:
            continue
        em.add(src_of(m.start, m.end), act.node_id, m.start, m.end, "act")
    return em.build()


# --- corpus run ---------------------------------------------------------------------------------------

def extract_all(records, registry: ActRegistry | None = None, llm=None):
    registry = registry or ActRegistry.from_records(records)
    ctx = RefContext.from_records(records, registry, llm=llm)
    edges = []
    for r in records:
        edges += extract_refs(r, registry, ctx)
    return edges, ctx


def edge_class(e: GraphEdge) -> str:
    rel = e.attrs.get("relation")
    if rel in ("element", "amended", "cited_in_amendment"):
        return f"{rel}:{'stub' if e.dst.startswith(EXT_PREFIX) else 'corpus'}"
    if rel == "act":
        return "act:" + ("external" if e.dst.startswith(EXT_PREFIX) else "corpus")
    return rel or "?"


def main(argv=None) -> int:
    import json
    from graphrag.data.ingest import read_records
    from graphrag.kg.llm import KgLLM
    argv = list(argv or [])
    use_llm = "--llm" in argv
    recs = read_records(config.path("data/records"))
    reg = ActRegistry.from_records(recs)
    llm = KgLLM() if use_llm else None
    edges, ctx = extract_all(recs, reg, llm)
    schema = ctx.schema
    nodes = {}
    for r in recs:
        for n in structure_nodes(r):
            nodes[n.node_id] = n
    for n in reg.nodes() + ctx.stub_nodes():
        nodes[n.node_id] = n
    all_edges = edges + reg.edges() + ctx.contains_edges()
    errs = [m for e in all_edges for m in schema.validate_edge(e, nodes)]
    errs += [m for n in ctx.stub_nodes() + reg.external_nodes() for m in schema.validate_node(n)]
    cls = Counter(edge_class(e) for e in edges)
    for e in reg.edges():
        cls[e.attrs["relation"]] += 1
    out = {"edges_refers_to": len(edges), "classes": dict(sorted(cls.items())),
           "stubs": len(ctx.stubs), "contains": len(ctx.stub_edges), "schema_errors": len(errs),
           "unresolved": dict(Counter(u["reason"] for u in ctx.unresolved).most_common()),
           "queue": len(ctx.queue), "stats": dict(sorted(ctx.stats.items())),
           "llm": llm.stats_json() if llm else None, "registry": {k: v for k, v in reg.stats().items()
                                                                   if k not in ("edition_groups", "same_key_distinct")}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if "--out" in argv:                                   # edges / unresolved / LLM log for review
        d = config.path(argv[argv.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "refs_edges.jsonl").write_text("".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                                    for e in sorted(edges, key=lambda e: (e.src, e.dst))), encoding="utf-8")
        (d / "refs_unresolved.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                                         for u in ctx.unresolved), encoding="utf-8")
        (d / "refs_llm.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                                  for u in (ctx.llm_log or ctx.queue)), encoding="utf-8")
        (d / "refs_recovered.jsonl").write_text("".join(json.dumps(list(r), ensure_ascii=False) + "\n"
                                                        for r in ctx.recovered), encoding="utf-8")
        (d / "refs_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    if errs:
        print("schema errors:", errs[:10])
    tgt = Counter()
    for e in edges:
        if e.dst.startswith(EXT_PREFIX):
            tgt[e.dst.split("/СТ.")[0]] += len(e.attrs.get("evidence_all") or [1])
    ext = {n.node_id: n.title for n in reg.external_nodes()}
    print("top-20 external targets (mentions):")
    for nid, c in tgt.most_common(20):
        print(f"{c:6d}  {ext.get(nid, '?')}  ->  {nid}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
