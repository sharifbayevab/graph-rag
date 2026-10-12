"""AMENDS ("уточняет/дополняет") and REPEALS ("отменяет") extraction (M2 Task 5).

`extract_amendments(rec, registry, ctx)` -> (edges, new stub nodes) for one record:

* AMENDS {evidence, op, date?} -- one per amendment operation of a law text. The act an
  instruction changes is `extract_ref.amendment_context` (Task 4; not re-derived here). Inside such
  a node the text outside quotes is cut into clauses (at ";", ":", paragraph ends, act-mention and
  gazette-parenthesis ends and after every operation verb). A clause ending in an operation verb
  (graph.yaml patterns.amend.ops: edit | add | delete | replace | renumber) is one operation:
    - src      = the deepest node containing the evidence span (μ of the verb formula);
    - evidence = the clause up to the end of its verb ("27-модданинг иккинчи қисми қуйидаги таҳрирда
                 баён этилсин"), an exact rec.text substring;
    - dst      = the element the clause names, completed by the header clauses above it in the same
                 node or its ancestors ("1) 9-модда:" / "24-модданинг:" -> "биринчи қисми …"); an add
                 names the new element ("122¹-модда билан тўлдирилсин" -> СТ.122.1); a clause that
                 names nothing targets the header's element, else the act itself. Numbered chapters /
                 sections / annexes / table positions (patterns.amend.units, Ruling R13) become EXT stub
                 chains ГЛ.N, ИЛОВА.N[/БЎЛ.M[/БЎЛ.K]][/ПОЗ.P | /П.P] (graph.yaml ext_stub_segments); for
                 a corpus act its own ГЛ node, else the act. Articles are resolved exactly like
                 Task 4 (`_element_target`): a real node of a corpus act (a missing article -> the act
                 vertex, as Task 4), else an EXT stub chain EXT/<key>/СТ.N[/Ч.K[/П.M]] via ctx.stub.
                 So an AMENDS edge has the same (src, dst) as the REFERS_TO{relation: amended} edge
                 of the same instruction (Ruling R11: Task 9 drops that REFERS_TO edge);
    - date     = the amending record's adoption date (document_date), absent when unknown.
  Several operations with one (src, dst) become one edge: op = the first, evidence_all + ops (one op
  per evidence, index-aligned; Ruling R12).
* REPEALS {evidence, date} -- "Қуйидагилар ўз кучини йўқотган деб топилсин:" (Latin, Russian
  "Признать утратившими силу:") + enumerated items: one edge per item (per element it names) to the
  act resolved by resolve_record (or its element); an inline "… ўз кучини йўқотган деб топилсин"
  repeals the act / element named in its clause. A REPEALS edge without a date is still emitted
  and listed in ctx.violations (§3.4; Task 9 fails on it). Targets get `repealed_on` (latest date):
  set in place on stubs, `apply_repealed_on` for other nodes -- never on an act that only stands for
  an unnamed sub-unit of it ("…Қонунининг тегишли бўлими").

Only law texts (acts.is_law_text) are scanned: in resolutions "… Қонунини … Президентига имзолаш
учун киритилсин" is not an amendment. A translation (a Russian edition with a translation_of edge) is
skipped: the original-language act carries the same operations. Deterministic; no LLM (an ambiguous act in a repeal item uses
only a ref_resolve decision already in ctx.decisions, else it is left unresolved).
"""
from __future__ import annotations

import bisect
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from graphrag import config
from graphrag.kg.acts import (_OUTER_CLOSE, EXT_PREFIX, ActRef, ActRegistry, _norm_number, _parse_date, _scan_cfg, is_amending, is_law_text,
                              record_date)
from graphrag.kg.extract_ref import (DocIndex, QuoteMask, RefContext, _Art, _element_target, _gazette_spans,
                                     _group, _ord_value, _parse_nums, _parse_parts, _pats, _scan_articles, _targets,
                                     _unres, amendment_context)
from graphrag.data.normalize import canon_num
from graphrag.kg.schema import Evidence, GraphEdge, GraphNode

ITEM_TYPES = ("пункт", "подпункт")

# --- patterns (config/graph.yaml patterns.amend / patterns.repeal) ---------------------------------

_AP: dict | None = None


def _apats() -> dict:
    global _AP
    if _AP is None:
        pats = config.load("graph")["patterns"]
        refs, am, rp = pats["refs"], pats["amend"], pats["repeal"]
        ord_rx = _pats()["_ord"].pattern[len("(?i)"):]
        sub = {"{NUM}": refs["num"], "{DASH}": refs["dash"], "{SEP}": refs["sep"], "{ORD}": ord_rx}

        def comp(v: str) -> re.Pattern:
            for a, b in sub.items():
                v = v.replace(a, b)
            return re.compile("(?i)" + v)

        sub["{UNUM}"] = am["unum"].replace("{ORD}", ord_rx)   # R29 A: "саккизинчи бўлим", "раздел восьмой"
        units = [(k, u["code"], u.get("rank"), u.get("default"), comp(u["rx"])) for k, u in am["units"].items()]
        segs = config.load("graph").get("ext_stub_segments") or {}
        _AP = {"units": units, "segments": {c: (v["type"], v["word"]) for c, v in segs.items()},
               "ops": [(op, comp(v)) for op, v in am["ops"].items()],
               "ops_ru": [(op, comp(v)) for op, v in am["ops_ru"].items()],
               **{k: comp(am[k]) for k in ("part", "point", "part_ru", "point_ru", "coarse", "enum")},
               **{f"repeal_{k}": comp(v) for k, v in rp.items()}}
    return _AP


# --- dates ------------------------------------------------------------------------------------------

def document_date(rec) -> str | None:
    """The amending / repealing record's adoption date: meta.date, else the adoption line of its
    text (acts.record_date), else the first date in its преамбула, then its реквизиты; else None."""
    d = record_date(rec)
    if d:
        return d
    for typ in ("преамбула", "реквизиты"):
        for n in rec.structure:
            if n.type == typ:
                d = _parse_date(rec.text[n.start:n.end])
                if d:
                    return d
    return None


# --- one record -------------------------------------------------------------------------------------

@dataclass
class _Clause:
    cs: int                 # clause span (raw) [cs, ce)
    ce: int
    term: str | None        # ":" / ";" / "\n" (paragraph end) / "v" (after a verb) / None
    es: int = 0             # evidence span (enumerator / leading space stripped)
    op: str | None = None
    vs: int = 0             # verb span
    ve: int = 0
    node: str | None = None
    ee: int = 0             # stripped end


@dataclass
class _El:
    """An element as named by a clause / accumulated from headers: articles, parts, points."""
    arts: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    points: list = field(default_factory=list)
    coarse: bool = False                              # an unnumbered chapter / section / annex word
    units: list = field(default_factory=list)         # Ruling R13: [(code, [numbers], rank)] ИЛОВА / БЎЛ / ГЛ / ПОЗ

    def empty(self) -> bool:
        return not (self.arts or self.parts or self.points or self.coarse or self.units)

    def then(self, e: "_El") -> "_El":
        """This (header) context refined by a lower element `e`."""
        if e.arts:
            return _El(list(e.arts), list(e.parts), list(e.points))
        if e.units:
            return _El([], [], list(e.points), False, _nest(self.units, e.units))
        if e.coarse:
            return _El([], [], [], True)
        if e.parts:
            return _El(list(self.arts), list(e.parts), list(e.points), self.coarse, list(self.units))
        if e.points:
            return _El(list(self.arts), list(self.parts), list(e.points), self.coarse, list(self.units))
        return _El(list(self.arts), list(self.parts), list(self.points), self.coarse, list(self.units))


def _nest(outer: list, inner: list) -> list:
    """Unit chain `inner` under the units of `outer` that rank above it ("4-илова:" + "I бўлимининг:" +
    "32-позицияси" -> ИЛОВА.4/БЎЛ.1/ПОЗ.32); a rankless unit (боб) stands alone."""
    if not inner or inner[0][2] is None:
        return list(inner)
    return [u for u in outer if u[2] is not None and u[2] < inner[0][2]] + list(inner)


def _unit_num(raw: str) -> str:
    """"XXIV¹" -> "24.1", "3²" -> "3.2", "IV" -> "4", "15" -> "15"."""
    m = re.fullmatch(r"([IVXLCivxlc]+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", raw.strip())
    if not m and not re.fullmatch(r"[IVXLCivxlc\d⁰¹²³⁴⁵⁶⁷⁸⁹]+", raw.strip()):
        v = _ord_value(raw)                              # an ordinal word: "саккизинчи" -> "8" (Ruling R29 A)
        return str(v) if v else canon_num(raw)
    if m:
        return f"{canon_num(m.group(1))}.{canon_num('1' + m.group(2)).split('.')[1]}"
    return canon_num(raw)


def _parse_list(s: str, ordinals: bool) -> list[str]:
    """Parts ("иккинчи ва учинчи", "учинчи — бешинчи" -> 3..5) or points ("1, 2-", "3–5-")."""
    if not ordinals:
        return _parse_nums(s)
    p = _pats()
    toks = [(m.start(), m.end(), _parse_parts(m.group(0))) for m in
            re.finditer(rf"(?i)(?:{p['_ord'].pattern[4:]}|\d+)", s)]
    out: list[str] = []
    for i, (a, b, v) in enumerate(toks):
        if not v:
            continue
        if i and out and p["_dash"].fullmatch(s[toks[i - 1][1]:a].strip() or "x") \
                and out[-1].isdecimal() and v[0].isdecimal() and 0 < int(v[0]) - int(out[-1]) <= 30:
            out += [str(k) for k in range(int(out[-1]) + 1, int(v[0]) + 1)]
        else:
            out += v
    return list(dict.fromkeys(out))


class _Doc:
    """Per-record scanning state."""

    def __init__(self, rec, ctx: RefContext):
        self.rec, self.ctx = rec, ctx
        self.text = rec.text
        if rec.doc_id not in ctx.docs:
            ctx.docs[rec.doc_id] = DocIndex(rec)
        self.di: DocIndex = ctx.docs[rec.doc_id]
        self.qm = QuoteMask(self.text)
        self.gaz = _gazette_spans(self.text)
        self.unmasked = 0
        self.q = self._mask()
        self.arts = [a for a in _scan_articles(self.text, self.gaz) if a.kind in ("plain", "ru")]
        self.art_starts = [a.start for a in self.arts]
        self.mentions = sorted(ctx.record_mentions(rec), key=lambda x: (x[0].start, x[0].end))

    def _mask(self) -> bytearray:
        """1 where a position is quoted (QuoteMask.inside) or inside a gazette parenthesis. A
        paragraph structure._own_flags takes for a quoted insertion because it starts with «, but
        whose quotes close and which then ends in an operation verb outside them ("«X» деган сўзлар
        «Y» деган сўзлар билан алмаштирилсин;" after "245-модданинг биринчи қисмидаги:"), is an
        instruction: inline quotes only, when the paragraph before it is not inside a quote."""
        text, qm = self.text, self.qm
        q = bytearray(len(text))
        verbs = [rx for _, rx in _apats()["ops"]]
        prev_own = True
        for i, (a, b) in enumerate(qm.paras):
            row = bytearray(b - a)
            depth = 0
            for j in range(a, b):
                if depth:
                    row[j - a] = 1
                ch = text[j]
                if ch in "«“":
                    depth += 1
                elif ch in "»”" and depth:
                    depth -= 1
            own = qm.own[i]
            if not own and prev_own and depth == 0 and text[a:b].lstrip()[:1] in "«“":
                free = "".join(" " if row[j] else text[a + j] for j in range(b - a))
                if any(rx.search(free) for rx in verbs):
                    own = True
                    self.unmasked += 1
            q[a:b] = row if own else b"\x01" * (b - a)
            prev_own = own
        for a, b in self.gaz:
            q[a:b] = b"\x01" * (b - a)
        return q

    def free(self, pos: int) -> bool:
        return 0 <= pos < len(self.q) and not self.q[pos]

    def node(self, a: int, b: int) -> str:
        n = self.di.node_at(a, b)
        return n.node_id if n else self.rec.doc_id

    # elements named in [a, b)
    def elements(self, a: int, b: int, op: str | None) -> _El:
        P = _apats()
        text = self.text
        items = []                                       # (start, end, kind, value)
        i = bisect.bisect_left(self.art_starts, a)
        while i < len(self.arts) and self.arts[i].start < b:
            x = self.arts[i]
            if self.free(x.start) and x.arts:
                items.append((x.start, x.end, "art", x))
            i += 1

        def covered(s, e):
            return any(s < it[1] and it[0] < e for it in items if it[2] == "art")

        for kind, rxs, ords in (("part", ("part", "part_ru"), True), ("point", ("point", "point_ru"), False)):
            for rk in rxs:
                for m in P[rk].finditer(text, a, b):
                    s = m.start()
                    if not self.free(s) or covered(s, m.end()):
                        continue
                    g = m.group(kind + "s") if m.groupdict().get(kind + "s") else (m.groupdict().get(kind + "s2") or "")
                    vals = _parse_list(g, ords)
                    if vals:
                        items.append((s, m.end(), kind, vals))
        units = []
        for key, code, rank, default, rx in P["units"]:
            for m in rx.finditer(text, a, b):
                if not self.free(m.start()) or covered(m.start(), m.end()) \
                        or any(m.start() < u[1] and u[0] < m.end() for u in units):
                    continue
                gd = m.groupdict()
                nums = _parse_nums(gd["nums"]) if gd.get("nums") else \
                    [_unit_num(gd.get("num") or gd["num2"])] if gd.get("num") or gd.get("num2") else \
                    ([default] if default else [])
                if nums:
                    units.append((m.start(), m.end(), "unit", (code, nums, rank), key.endswith("_ru")))
        items += units
        for m in P["coarse"].finditer(text, a, b):
            if self.free(m.start()) and not covered(m.start(), m.end()) \
                    and not any(m.start() < u[1] and u[0] < m.end() for u in units):
                items.append((m.start(), m.end(), "coarse", True))
        items.sort(key=lambda it: (it[0], it[1]))
        pick = (lambda xs: xs[-1]) if op == "add" else (lambda xs: xs[0])
        arts = [it for it in items if it[2] == "art"]
        if arts:
            it = pick(arts)
            x = it[3]
            el = _El(list(x.arts), list(x.parts), list(x.points))
            if not el.parts:
                ps = [p for p in items if p[2] == "part" and p[0] >= it[1]]
                if ps:
                    el.parts = pick(ps)[3]
                    it = pick(ps)
            if not el.points:
                pts = [p for p in items if p[2] == "point" and p[0] >= it[1]]
                if pts:
                    el.points = pick(pts)[3]
            return el
        ps = [it for it in items if it[2] == "part"]
        pts = [it for it in items if it[2] == "point"]
        us = [it for it in items if it[2] == "unit"]
        if us:                                           # "I бўлимининг 8-банди", "2-илованинг 98-позицияси"
            chain, end = [], us[0][1]
            for it in us:
                if it[4] and chain and it[3][2] is not None and all(u[2] is not None and u[2] > it[3][2] for u in chain):
                    chain = [it[3]] + chain              # Russian genitive: "подраздел 4 раздела II" (R29 A)
                else:
                    chain = _nest(chain, [it[3]])        # a lower rank refines, an equal / higher one restarts
                end = it[1]
            later = [p for p in pts if p[0] >= end]
            return _El([], [], list(pick(later)[3]) if later else [], False, chain)
        coarse = any(it[2] == "coarse" for it in items)
        if coarse:
            return _El([], [], [], True)
        if ps:
            it = pick(ps)
            later = [p for p in pts if p[0] >= it[1]]
            return _El([], list(it[3]), list(pick(later)[3]) if later else [])
        if pts:
            return _El([], [], list(pick(pts)[3]))
        return _El()


class _Out:
    """Merged edges of one record: (src, type, dst) -> evidences with their ops (index-aligned)."""

    def __init__(self):
        self.edges: dict[tuple, dict] = {}

    def add(self, src, typ, dst, ev: dict, op: str | None, date: str | None, stats: Counter,
            partial: bool = False):
        """`partial`: the dst act stands for a sub-unit it cannot name (no repealed_on on the act)."""
        k = (src, typ, dst)
        e = self.edges.get(k)
        if e is None:
            self.edges[k] = {"op": op, "ev": [ev], "ops": [op], "date": date, "partial": partial}
            return
        e["partial"] = e["partial"] and partial
        if ev in e["ev"]:
            return
        e["ev"].append(ev)
        e["ops"].append(op)
        if op != e["op"]:
            stats[f"{typ.lower()}_merged_op_conflict"] += 1
        stats[f"{typ.lower()}_merged_operations"] += 1


class _Nodes:
    """Node lookup for Schema.validate_edge: the context's records / stubs / EXT acts + the
    registry's corpus act vertices."""

    def __init__(self, ctx: RefContext, registry: ActRegistry):
        self.ctx, self.reg = ctx, registry

    def get(self, nid, default=None):
        n = self.ctx.get(nid)
        if n is None:
            n = self.reg._doc_nodes.get(nid)
        return n if n is not None else default


def _strip(doc: _Doc, a: int, b: int) -> tuple[int, int]:
    text = doc.text
    m = _apats()["enum"].match(text, a, b)
    if m:
        a = m.end()
    while a < b and (text[a].isspace() or text[a] in ",;:)"):
        a += 1
    while b > a and (text[b - 1].isspace() or text[b - 1] in ",;:."):
        b -= 1
    return a, b


def _verbs(doc: _Doc, a: int, b: int, key: str) -> list[tuple[int, int, str]]:
    """Unquoted operation verbs in [a, b): (start, end, op), overlaps resolved (longest first)."""
    P = _apats()
    found = []
    for op, rx in P[key]:
        for m in rx.finditer(doc.text, a, b):
            if doc.free(m.start()) and doc.free(m.end() - 1):
                found.append((m.start(), m.end(), op))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, last = [], -1
    for v in found:
        if v[0] >= last:
            out.append(v)
            last = v[1]
    return out


def _clauses(doc: _Doc, a: int, b: int, repeal_spans) -> list[_Clause]:
    """The clauses of [a, b) outside quotes / gazette parentheses (see the module docstring)."""
    text = doc.text
    cuts: list[tuple[int, int, str | None]] = []        # (clause end, next start, terminator)
    for i in range(a, b):
        if text[i] in ";:" and doc.free(i):
            cuts.append((i, i + 1, text[i]))
    for pa, pb in doc.qm.paras:
        if a < pb <= b:
            cuts.append((pb, pb + 2, "\n"))
    for m, _ in doc.mentions:
        if a < m.end < b and doc.free(m.start):
            cuts.append((m.end, m.end, None))
    for ga, gb in doc.gaz:
        if a < gb < b:
            cuts.append((gb, gb, None))
    rep = [(s, e) for s, e in repeal_spans if a <= s < b]
    verbs = [v for v in _verbs(doc, a, b, "ops") if not any(s <= v[0] < e for s, e in rep)]
    for vs, ve, op in verbs:
        cuts.append((ve, ve, "v"))
    vmap = {ve: (vs, op) for vs, ve, op in verbs}
    cuts.sort()
    out, cur = [], a
    for end, nxt, term in cuts:
        if end < cur:
            continue
        if end > cur:
            c = _Clause(cur, end, term)
            if end in vmap and vmap[end][0] >= cur:
                c.vs, c.op = vmap[end][0], vmap[end][1]
                c.ve = end
            out.append(c)
        cur = max(cur, nxt)
    if cur < b:
        out.append(_Clause(cur, b, None))
    res = []
    for c in out:
        if not any(doc.free(i) and not text[i].isspace() for i in range(c.cs, c.ce)):
            continue                                    # quoted text only
        if c.op is None:                                 # a Russian verb inside the clause
            rv = _verbs(doc, c.cs, c.ce, "ops_ru")
            if rv:
                c.vs, c.ve, c.op = rv[0]
        c.es, ee = _strip(doc, c.cs, c.ce)
        if c.es >= ee:
            continue
        c.ee = ee
        if c.op is not None and c.ve > ee:
            c.ee = c.ve
        res.append(c)
    return res


def extract_amendments(rec, registry: ActRegistry, ctx: RefContext | None = None
                       ) -> tuple[list[GraphEdge], list[GraphNode]]:
    """AMENDS and REPEALS edges of one record and the EXT element stubs this call created (see the
    module docstring). Unresolved items go to ctx.unresolved, REPEALS without a date to
    ctx.violations, repeal dates to ctx.repealed_on; CONTAINS edges of stubs: ctx.contains_edges()."""
    if ctx is None:
        ctx = RefContext.from_records([rec], registry)
    if not is_law_text(rec):
        return [], []
    tr = getattr(registry, "_translation_srcs", None)
    if tr is None or tr[0] != len(registry.version_edges):   # indexed once per registry (version_edges by record)
        tr = (len(registry.version_edges), frozenset(e.src for e in registry.version_edges
                                                    if e.attrs.get("relation") == "translation_of"))
        registry._translation_srcs = tr
    if rec.doc_id in tr[1]:
        ctx.stats["amend_skipped_translation"] += 1      # the original-language act carries these facts
        return [], []
    before = set(ctx.stubs)
    doc = _Doc(rec, ctx)
    ctx.stats["amend_unmasked_instruction_paras"] += doc.unmasked
    date = document_date(rec)
    out = _Out()
    st = ctx.stats
    P = _apats()
    text = rec.text

    repeal_list = [m for m in P["repeal_list"].finditer(text) if doc.free(m.start())]
    repeal_inline = [m for m in P["repeal_inline"].finditer(text) if doc.free(m.start())
                     and not any(r.start() <= m.start() < r.end() for r in repeal_list)]
    rep_spans = [(m.start(), m.end()) for m in repeal_list + repeal_inline]

    def ev(a, b) -> dict:
        return Evidence(rec.doc_id, a, b, text[a:b]).to_json()

    def targets(act: ActRef, el: _El, a: int, b: int) -> list[tuple[str, str]]:
        """[(dst, status)] for element `el` of `act`. Status "act_part" = the act stands for a sub-unit
        it cannot name (an unnumbered chapter / annex word, a corpus unit that does not exist), so a
        repeal must not mark the whole act repealed; "act" = nothing was named."""
        if el.units and not el.arts:
            return unit_targets(act, el)
        if el.coarse:
            return [(act.node_id, "act_part")]
        if not el.arts:
            return [(act.node_id, "act")]
        res = []
        for art, part, pt in _targets(_Art(a, b, el.arts, el.parts, el.points, "plain")):
            nid, stt = _element_target(ctx, rec, act, art, part, pt, a, b)
            if nid is None:                              # corpus act without that article: its vertex
                st["amend_corpus_missing_article"] += 1
                res.append((act.node_id, "missing"))
            else:
                res.append((nid, stt))
        return list(dict.fromkeys(res))

    def unit_targets(act: ActRef, el: _El) -> list[tuple[str, str]]:
        """Ruling R13: ГЛ / БЎЛ / ИЛОВА / ПОЗ [+ П] stub chains under an EXT act; for a corpus act its
        own chapter node (ГЛ) when it exists, else the act (never an invented corpus node)."""
        segs = P["segments"]
        chains = [[]]
        for code, nums, _ in el.units:
            chains = [c + [(code, n)] for c in chains for n in nums]
        chains = [c + [("П", pt)] for c in chains for pt in (el.points or [None]) if pt] \
            if el.points else chains
        res = []
        for c in chains:
            if act.doc_id:
                di = ctx.docs.get(act.doc_id)
                hit = None
                if di is not None and len(c) == 1 and c[0][0] == "ГЛ":
                    hit = next((n.node_id for n in di.rec.structure if n.type == segs["ГЛ"][0]
                                and n.number == c[0][1]), None)
                res.append((hit, "unit_corpus") if hit else (act.node_id, "act_part"))
            else:
                res.append((ctx.stub(act.node_id, [(code, segs[code][0], n) for code, n in c]), "unit_stub"))
        return list(dict.fromkeys(res))

    # --- AMENDS (and inline element repeals) in amendment instructions ---
    amend = amendment_context(rec, registry, doc.mentions)
    roots = [n for n in rec.structure if n.node_id in amend and n.parent_id not in amend]
    inline_done: set[int] = set()
    for root in roots:
        cls = _clauses(doc, root.start, root.end, rep_spans)
        headers: list[tuple[_Clause, _El]] = []
        last = None                                      # (clause end, target element) of the last operation
        for c in cls:
            c.node = doc.node(c.es, c.ee)
            anc = {c.node, *doc.di.ancestors(c.node)}
            rep = next((m for m in repeal_inline if c.cs <= m.start() < c.ce + 1), None)
            if c.op is None and rep is None:
                if c.term == ":":
                    headers.append((c, doc.elements(c.cs, c.ee, None)))
                continue
            if c.node not in amend:
                st["amend_verb_outside_context"] += 1
                continue
            act = amend[c.node]                          # a deeper instruction overrides its parent's act
            ctx_el = _El()
            for h, hel in headers:
                if h.ce <= c.cs and h.node in anc:
                    ctx_el = ctx_el.then(hel)
            own = doc.elements(c.cs, c.ce, c.op)
            if own.arts and c.op != "add":               # "9) 43-модданинг … алмаштирилсин: «…»; тўртинчи қисми …"
                headers.append((c, _El(list(own.arts))))  # the point's article for its later clauses
            if own.empty() and last is not None and last[0] == c.cs:
                el = last[1]                              # "… чиқарилсин ва «Y» … билан тўлдирилсин"
            else:
                el = ctx_el.then(own)
            if rep is not None:                          # "… учинчи қисми ўз кучини йўқотган деб топилсин"
                inline_done.add(rep.start())
                ee = rep.end()
                for dst, stt in targets(act, el, c.es, ee):
                    out.add(doc.node(c.es, ee), "REPEALS", dst, ev(c.es, ee), None, date, st, stt == "act_part")
                    st["repeals_inline_element"] += 1
                continue
            last = (c.ce, el)
            ve = c.ve if c.ve == c.ce else max(c.ve, c.ee)   # a Russian verb: the whole clause
            e = ev(c.es, ve)
            src = doc.node(c.es, ve)
            for dst, stt in targets(act, el, c.es, ve):
                out.add(src, "AMENDS", dst, e, c.op, date, st)
                st[f"amend_target_{stt}"] += 1
            st[f"amend_op_{c.op}"] += 1

    # --- REPEALS lists ---
    for m in repeal_list:
        n = doc.di.node_at(m.start(), m.end())
        if n is None:
            continue
        items = [c for c in doc.di.children.get(n.node_id, []) if c.type in ITEM_TYPES and c.start >= m.end()]
        spans = [(c.start, c.end) for c in items]
        if not spans:                                    # items not split by the structure parser
            for pa, pb in doc.qm.paras:
                if m.end() <= pa < n.end and re.match(r"\s*\d+\s*\)", text[pa:pb]):
                    spans.append((pa, pb))
        st["repeal_lists"] += 1
        for a, b in spans:
            _repeal_item(doc, a, b, m, date, out, targets)

    # --- inline repeals outside amendment instructions: "«X»ги Қонуни ўз кучини йўқотган деб топилсин" ---
    for m in repeal_inline:
        if m.start() in inline_done:
            continue
        pa = text.rfind("\n\n", 0, m.start())
        pa = 0 if pa < 0 else pa + 2
        cs = max([pa] + [i + 1 for i in range(pa, m.start()) if text[i] in ";:" and doc.free(i)])
        ms = [(x, r) for x, r in doc.mentions if cs <= x.start and x.end <= m.start() and doc.free(x.start)]
        ms = [(x, r) for x, r in ms if not any(o.start <= x.start and x.end <= o.end and (o.start, o.end) != (x.start, x.end)
                                                for o, _ in ms)]
        if not ms:
            continue
        x, r = ms[-1]
        act = _act(doc, x, r)
        es, ee = _strip(doc, cs, m.end())
        if act is None:
            _unres(ctx, rec, es, ee, "repeal_act_unresolved", key=r.key)
            continue
        el = doc.elements(x.end, m.start(), None)
        for dst, stt in targets(act, el, es, ee):
            out.add(doc.node(es, ee), "REPEALS", dst, ev(es, ee), None, date, st, stt == "act_part")
            st["repeals_inline"] += 1

    edges = _build(doc, registry, out)
    new = [ctx.stubs[k] for k in sorted(set(ctx.stubs) - before)]
    return edges, new


def _act(doc: _Doc, m, ref: ActRef) -> ActRef | None:
    """The act vertex of a mention: as resolved, or an ambiguous one settled by an existing
    ref_resolve decision (Task 4; no LLM call here)."""
    if ref.node_id:
        return ref
    if ref.ambiguous:
        got = doc.ctx.decisions.get(_group(doc.rec, doc.di, m.start, ref.key))
        if got is not None and got.node_id:
            return got
    return None


def _repeal_item(doc: _Doc, a: int, b: int, formula, date, out: _Out, targets) -> None:
    """One enumerated item of a repeal list: the act (or element) its head paragraph names."""
    P = _apats()
    text, rec, ctx = doc.text, doc.rec, doc.ctx
    hb = text.find("\n\n", a, b)
    hb = b if hb < 0 else hb
    es, ee = _strip(doc, a, hb)
    if es >= ee:
        return
    gz = [g for g in doc.gaz if es <= g[0] < ee]
    head_end = gz[0][0] if gz else ee                   # the repealed document's head noun ends here
    head = text[es:head_end].rstrip(" ;,.")
    if P["repeal_non_law_head"].search(head):
        _unres(ctx, rec, es, ee, "repeal_non_law_act")
        return
    ms = [(x, r) for x, r in doc.mentions if es <= x.start and x.end <= head_end and doc.free(x.start)]
    ms = [(x, r) for x, r in ms if not any(o.start <= x.start and x.end <= o.end and (o.start, o.end) != (x.start, x.end)
                                            for o, _ in ms)]
    if not ms:
        _unres(ctx, rec, es, ee, "repeal_no_act")
        return
    x, r = ms[-1]                                       # the mention whose head noun governs the item
    el_from = x.end
    oc = _OUTER_CLOSE.search(text, x.end, head_end)
    if oc is not None and doc.free(oc.start()):
        # «X»ги … Қонунига ўзгартиш киритиш ҳақида»ги ЎРҚ–733-сонли Қонуни: the outer title's opening
        # quote is missing, so the scanner saw only the base law X; the item repeals the outer law
        name = text[x.start:oc.start()]
        num = _scan_cfg()["p"]["act_number"].search(text, oc.end(), head_end)
        d = record_date(rec)
        r = ctx.registry.resolve(name, d, number=_norm_number(num.group(0)) if num else None,
                                 cited_date=x.date, cited_kind=x.date_kind, src_doc=rec.doc_id)
        # (Ruling R29 A: a numbered outer title is now one scanner mention (acts._outer_unquoted), so this
        # path is only a fallback; its edition follows the registry rule, as every other mention)
        if r.external:
            ctx.registry.note_mention(r, name, d)
        el_from = oc.end()
        ctx.stats["repeal_outer_title"] += 1
    act = _act(doc, x, r)
    if act is None:
        _unres(ctx, rec, es, ee, "repeal_act_unresolved", key=r.key, status=r.status)
        return
    el = doc.elements(el_from, ee, None)
    fev = Evidence(rec.doc_id, formula.start(), formula.end(), text[formula.start():formula.end()]).to_json()
    iev = Evidence(rec.doc_id, es, ee, text[es:ee]).to_json()
    src = doc.node(es, ee)
    for dst, stt in targets(act, el, es, ee):
        out.add(src, "REPEALS", dst, iev, None, date, ctx.stats, stt == "act_part")
        k = (src, "REPEALS", dst)
        if fev not in out.edges[k].setdefault("formula", []):
            out.edges[k]["formula"].append(fev)
        ctx.stats[f"repeal_target_{stt}"] += 1


def _build(doc: _Doc, registry: ActRegistry, out: _Out) -> list[GraphEdge]:
    ctx, rec = doc.ctx, doc.rec
    nodes = _Nodes(ctx, registry)
    res = []
    for (src, typ, dst), e in sorted(out.edges.items()):
        pairs = sorted(zip(e["ev"], e["ops"]), key=lambda x: (x[0]["start"], x[0]["end"]))
        evs = [x for x, _ in pairs]
        attrs = {"evidence": evs[0]}
        if typ == "AMENDS":
            attrs["op"] = pairs[0][1]
            if len(pairs) > 1:                           # Ruling R12: one op per evidence_all entry
                attrs["ops"] = [o for _, o in pairs]
        if e["date"]:
            attrs["date"] = e["date"]
        allev = sorted(evs + e.get("formula", []), key=lambda x: (x["start"], x["end"]))
        if len(allev) > 1:
            attrs["evidence_all"] = allev
        edge = GraphEdge(src, dst, typ, attrs)
        errs = ctx.schema.validate_edge(edge, nodes)
        for x in allev:
            if rec.text[x["start"]:x["end"]] != x["quote"]:
                errs.append("evidence quote is not the text span")
        undated = typ == "REPEALS" and not e["date"]
        if undated:
            errs = [m for m in errs if not m.endswith("missing required attr 'date'")]
        if errs:
            ctx.stats["amend_schema_rejected"] += 1
            ctx.unresolved.append({"doc_id": rec.doc_id, "reason": "schema", "errors": errs[:3], "src": src, "dst": dst})
            continue
        if undated:
            ctx.violations.append({"reason": "repeal_without_date", "doc_id": rec.doc_id, "src": src, "dst": dst,
                                   "evidence": evs[0]})
        elif typ == "REPEALS" and e["partial"]:
            ctx.stats["repeal_partial_act_no_repealed_on"] += 1   # a sub-unit repealed, not the act
        elif typ == "REPEALS":
            if e["date"] > ctx.repealed_on.get(dst, ""):
                ctx.repealed_on[dst] = e["date"]
            if dst in ctx.stubs:
                ctx.stubs[dst].attrs["repealed_on"] = ctx.repealed_on[dst]
        res.append(edge)
    return res


def apply_repealed_on(nodes: list[GraphNode], repealed_on: dict) -> list[GraphNode]:
    """`nodes` with attrs.repealed_on set on every repealed target (the latest repeal date)."""
    out = []
    for n in nodes:
        d = repealed_on.get(n.node_id)
        if d and n.attrs.get("repealed_on") != d:
            n = GraphNode(n.node_id, n.kind, n.type, n.title, n.doc_id, n.external, {**n.attrs, "repealed_on": d})
        out.append(n)
    return out


# --- corpus run ---------------------------------------------------------------------------------------

def extract_all(records, registry: ActRegistry | None = None, ctx: RefContext | None = None):
    registry = registry or ActRegistry.from_records(records)
    ctx = ctx or RefContext.from_records(records, registry)
    edges, nodes = [], []
    for r in records:
        e, n = extract_amendments(r, registry, ctx)
        edges += e
        nodes += n
    return edges, nodes, ctx


def _src_type(ctx: RefContext, nid: str) -> str:
    di = ctx.docs.get(nid.split("/", 1)[0])
    n = di.nodes.get(nid) if di else None
    return n.type if n else "?"


def main(argv=None) -> int:
    """python -m graphrag.kg.extract_amend [--out DIR]: Task 4 REFERS_TO first (same context, ref_resolve
    decisions from the cache, no API), then AMENDS / REPEALS; stats and review files."""
    import json
    from graphrag.data.ingest import read_records
    from graphrag.kg import extract_ref
    from graphrag.kg.llm import KgLLM
    argv = list(argv or [])
    recs = read_records(config.path("data/records"))
    reg = ActRegistry.from_records(recs)
    llm = KgLLM(allow_api=False) if "--llm" in argv else None   # Task 5 never calls the API
    refs, ctx = extract_ref.extract_all(recs, reg, llm)
    ctx.llm = None                                        # Task 5 never calls the LLM
    edges, new, ctx = extract_all(recs, reg, ctx)
    nodes = {}
    for r in recs:
        nodes.update({n.node_id: n for n in extract_ref.structure_nodes(r)})
    nodes.update({n.node_id: n for n in reg.nodes() + ctx.stub_nodes()})
    errs = [m for e in edges for m in ctx.schema.validate_edge(e, nodes)]
    viol = {(v["src"], v["dst"]) for v in ctx.violations}
    errs = [m for m in errs if not m.endswith("missing required attr 'date'")] + \
        [m for e in edges if (e.src, e.dst) in viol for m in ctx.schema.validate_edge(e, nodes)
         if m.endswith("missing required attr 'date'")]
    am = [e for e in edges if e.type == "AMENDS"]
    rp = [e for e in edges if e.type == "REPEALS"]
    amended = {(e.src, e.dst) for e in refs if e.attrs.get("relation") == "amended"}
    anyref = {(e.src, e.dst): e.attrs.get("relation") for e in refs}
    overlap = sum((e.src, e.dst) in amended for e in am)
    other = Counter(anyref[(e.src, e.dst)] for e in am if (e.src, e.dst) in anyref and (e.src, e.dst) not in amended)
    am_keys = {(e.src, e.dst) for e in am}
    amended_left = [k for k in sorted(amended) if k not in am_keys]
    ref_by_src: dict[str, set] = {}
    for e in refs:
        ref_by_src.setdefault(e.src, set()).add(e.dst)

    def twin_class(src, dst, others) -> str:
        if any(o.startswith(dst + "/") for o in others):
            return "other_side_deeper"          # e.g. REFERS_TO into a quoted new chapter / AMENDS at the act
        if any(dst.startswith(o + "/") for o in others):
            return "other_side_shallower"       # e.g. header "9-модда:" vs AMENDS СТ.9/Ч.1
        return "no_related_edge_from_src" if not others else "unrelated_targets_from_src"
    am_no_twin = Counter(twin_class(e.src, e.dst, ref_by_src.get(e.src, set())) for e in am
                         if (e.src, e.dst) not in amended)
    am_by_src: dict[str, set] = {}
    for e in am:
        am_by_src.setdefault(e.src, set()).add(e.dst)
    left_cls = Counter(twin_class(s, d, am_by_src.get(s, set())) for s, d in amended_left)
    stats = {"amends": len(am), "amends_by_op": dict(sorted(Counter(e.attrs["op"] for e in am).items())),
             "amends_with_date": sum(1 for e in am if e.attrs.get("date")),
             "amends_dst_stub": sum(e.dst.startswith(EXT_PREFIX) for e in am),
             "amends_dst_act": sum(1 for e in am if "/" not in e.dst.replace(EXT_PREFIX, "", 1)),
             "repeals": len(rp), "repeals_with_date": sum(1 for e in rp if e.attrs.get("date")),
             "violations": ctx.violations, "schema_errors": len(errs),
             "r11_amends_equal_refers_amended": overlap, "r11_refers_amended_total": len(amended),
             "r11_refers_amended_without_amends": len(amended_left),
             "amends_equal_refers_other_relation": dict(other),
             "amends_without_twin": dict(sorted(am_no_twin.items())),
             "refers_amended_without_twin": dict(sorted(left_cls.items())),
             "amends_by_src_type": dict(sorted(Counter(_src_type(ctx, e.src) for e in am).items())),
             "llm": llm.stats_json() if llm else None,
             "new_stub_nodes": len(new), "repealed_nodes": len(ctx.repealed_on),
             "unresolved": dict(Counter(u["reason"] for u in ctx.unresolved if u["reason"].startswith(("repeal", "schema"))).most_common()),
             "stats": {k: v for k, v in sorted(ctx.stats.items()) if k.startswith(("amend", "repeal"))}}
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if errs:
        print("schema errors:", errs[:10])
    if "--out" in argv:
        d = config.path(argv[argv.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "amend_edges.jsonl").write_text("".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                                     for e in sorted(edges, key=lambda e: (e.type, e.src, e.dst))), encoding="utf-8")
        (d / "amend_unresolved.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                                          for u in ctx.unresolved if u["reason"].startswith(("repeal", "schema"))), encoding="utf-8")
        (d / "amend_refers_amended_left.jsonl").write_text("".join(json.dumps(list(k), ensure_ascii=False) + "\n"
                                                                   for k in amended_left), encoding="utf-8")
        (d / "amend_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
