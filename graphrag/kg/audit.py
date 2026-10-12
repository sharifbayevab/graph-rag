"""Edge-precision audit of the M2 graph (Task 7): a stratified sample of extracted edges judged by a
cached gpt-4.1 judge (KgLLM pass "audit"), with planted known-wrong edges (Ruling R19) to measure the
judge's sensitivity, a manually labelled subset to measure its specificity / agreement, and
precision per stratum and per edge type with Wilson 95% CIs (raw and corrected for judge error).

Everything here is deterministic given the edges, the seed and the judge answers:

* strata (`stratum_of`): REFERS_TO by relation class (version_of / translation_of pooled as
  "version"), AMENDS by op, REPEALS, DEFINES, REGULATES by target kind (entity / concept).
  CONTAINS is structural and checked programmatically (`contains_violations`), RELATED is latent;
* sample (`sample_sizes`, `draw_sample`): `major` edges per stratum (all when the stratum is
  smaller or listed in `full`), drawn with random.Random(f"{seed}:{stratum}") from the population
  sorted by artifact.edge_sort_key;
* planting (`plant_stratum`): ceil(plant_share * n) corrupted copies of population edges NOT in the
  sample: swap_in_doc (the target of another edge of the same type from the same document),
  random_same_type (a random other target of the same node type from the stratum), shift_evidence
  (the evidence moved to a far paragraph of the same document, src = its deepest node), op_swap
  (AMENDS only). Planted edges are marked only in the private key; the judge sees the same fields;
* judge input (`render_item`): source document / node path (+ heads of ancestors outside the
  window) and the text ±`window` chars around the evidence (marked ⟦…⟧), edge type + attrs, the
  target (corpus node path + text, EXT act title, stub act title + element path, entity name /
  aliases, concept term / definition). Batches are per stratum, shuffled, `batch` items per call;
* metrics (`wilson`, `corrected`, `pooled`): per stratum k/n and Wilson CI; per type the strata
  pooled with their population weights (effective-n Wilson); corrected precision
  pi = (p + Se - 1) / (Se + Sp - 1) with Se = share of planted edges the judge rejects and Sp = share
  of manually-correct edges the judge accepts (Rogan-Gladen), clipped to [0, 1].
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from graphrag.kg import judge as J
from graphrag.kg.artifact import edge_sort_key  # the population / sample order (R31 (2): not an identity)
from graphrag.kg.schema import GraphEdge, GraphNode

ENT_PREFIX, CON_PREFIX, EXT_PREFIX = "ENT/", "CON/", "EXT/"
AUDITED_TYPES = ("REFERS_TO", "AMENDS", "REPEALS", "DEFINES", "REGULATES")
OP_SWAP = {"delete": "add", "add": "delete", "edit": "delete", "replace": "delete", "renumber": "delete"}
METHODS = {
    "REFERS_TO": ("swap_in_doc", "random_same_type", "shift_evidence"),
    "AMENDS": ("swap_in_doc", "op_swap", "random_same_type", "shift_evidence"),
    "REPEALS": ("swap_in_doc", "random_same_type", "shift_evidence"),
    "DEFINES": ("swap_in_doc", "random_same_type", "shift_evidence"),
    "REGULATES": ("random_same_type", "swap_in_doc", "shift_evidence"),
}
MANUAL_QUOTA = {"REFERS_TO": 8, "AMENDS": 6, "REPEALS": 4, "DEFINES": 4, "REGULATES": 8}
DEFAULTS = {"seed": 20260928, "major": 40, "full": ["REPEALS"], "plant_share": 0.125, "batch": 6,
            "window": 400, "quote_cap": 1500, "target_text_cap": 500}


# --- strata and sampling ------------------------------------------------------------------------------

def stratum_of(e: GraphEdge) -> str | None:
    if e.type == "REFERS_TO":
        rel = e.attrs.get("relation") or "?"
        return "REFERS_TO:" + ("version" if rel in ("version_of", "translation_of") else rel)
    if e.type == "AMENDS":
        return f"AMENDS:{e.attrs.get('op')}"
    if e.type == "REGULATES":
        return "REGULATES:" + ("concept" if e.dst.startswith(CON_PREFIX) else "entity")
    if e.type in ("REPEALS", "DEFINES"):
        return e.type
    return None


def type_of(stratum: str) -> str:
    return stratum.split(":", 1)[0]


def uid_of(stratum: str, e: GraphEdge, planted: bool = False) -> str:
    s = json.dumps([stratum, planted, e.to_json()], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def populations(edges) -> dict[str, list[GraphEdge]]:
    pops: dict[str, list[GraphEdge]] = defaultdict(list)
    for e in edges:
        s = stratum_of(e)
        if s is not None:
            pops[s].append(e)
    return {s: sorted(v, key=edge_sort_key) for s, v in sorted(pops.items())}


def sample_sizes(counts: dict[str, int], major: int = 40, full=("REPEALS",)) -> dict[str, int]:
    """n per stratum: all edges of a `full` stratum or of one with <= major edges, else `major`."""
    return {s: (N if (s in full or type_of(s) in full or N <= major) else major) for s, N in sorted(counts.items())}


def _rng(seed, *tag) -> random.Random:
    return random.Random(":".join(str(x) for x in (seed,) + tag))


def draw_sample(pops: dict[str, list[GraphEdge]], sizes: dict[str, int], seed) -> dict[str, list[GraphEdge]]:
    out = {}
    for s, pop in sorted(pops.items()):
        pop = sorted(pop, key=edge_sort_key)                   # independent of the input order
        n = min(sizes.get(s, 0), len(pop))
        idx = sorted(_rng(seed, s).sample(range(len(pop)), n))
        out[s] = [pop[i] for i in idx]
    return out


def n_planted(n: int, share: float) -> int:
    return max(1, math.ceil(share * n)) if n else 0


# --- the audited graph (texts, nodes, structure) ---------------------------------------------------------

@dataclass
class AuditGraph:
    """What rendering and planting need: record texts / titles / structure and all graph vertices."""
    texts: dict[str, str]
    titles: dict[str, str]                       # doc_id -> document title
    meta: dict[str, dict]                        # doc_id -> {type, date}
    struct: dict[str, Any]                       # node_id -> structure Node (start, end, parent_id, ...)
    nodes: dict[str, GraphNode]                  # every vertex (structure, acts, stubs, entities, concepts)
    edges: list[GraphEdge] = field(default_factory=list)   # every audited-type edge (existing-edge checks)

    def __post_init__(self):
        self.by_doc_nodes: dict[str, list] = defaultdict(list)
        for n in self.struct.values():
            self.by_doc_nodes[n.node_id.split("/", 1)[0]].append(n)
        for v in self.by_doc_nodes.values():
            v.sort(key=lambda n: (n.start, -n.end))
        self.out_targets: dict[str, set] = defaultdict(set)
        self.existing: set = set()
        for e in self.edges:
            self.out_targets[e.src].add(e.dst)
            self.existing.add((e.src, e.type, e.dst))

    @classmethod
    def from_records(cls, records, nodes: dict[str, GraphNode], edges=()) -> "AuditGraph":
        struct = {n.node_id: n for r in records for n in r.structure}
        return cls({r.doc_id: r.text for r in records},
                   {r.doc_id: (r.meta.title or next((n.title for n in r.structure if n.parent_id is None), "") or "")
                    for r in records},
                   {r.doc_id: {"type": r.meta.type, "date": r.meta.date} for r in records},
                   struct, nodes, list(edges))

    def node_at(self, doc: str, a: int, b: int):
        best = None
        for n in self.by_doc_nodes.get(doc, ()):
            if n.start > a:
                break
            if n.start <= a and b <= n.end:
                if best is None or n.node_id.count("/") > best.node_id.count("/") or \
                        (n.node_id.count("/") == best.node_id.count("/") and n.end - n.start < best.end - best.start):
                    best = n
        return best

    def ancestors(self, nid: str) -> list:
        out, n = [], self.struct.get(nid)
        while n is not None:
            out.append(n)
            n = self.struct.get(n.parent_id) if n.parent_id else None
        return out[::-1]                                  # document first


_cap = J.cap_middle                                     # the judge-view text cap (graphrag.kg.judge)


def _label(n) -> str:
    t = (n.title or "").strip().replace("\n", " ")
    num = f" {n.number}" if getattr(n, "number", None) else ""
    return f"{n.type}{num}" + (f": {t[:140]}" if t and n.parent_id is not None else "")


def _first_line(s: str, cap: int = 250) -> str:
    s = s.strip()
    return s.split("\n", 1)[0][:cap]


def render_target(g: AuditGraph, nid: str, cfg: dict) -> dict:
    n = g.nodes.get(nid)
    if nid.startswith(ENT_PREFIX):
        return {"kind": "entity", "entity_type": n.type if n else "?", "name": n.title if n else nid,
                "aliases": list((n.attrs.get("aliases") if n else None) or [])[:6]}
    if nid.startswith(CON_PREFIX):
        a = n.attrs if n else {}
        defs = a.get("definitions") or []
        return {"kind": "concept", "term": n.title if n else nid,
                "definition": _cap(a.get("definition") or (defs[0].get("definition") if defs else "") or "", 400),
                "defined_in": sorted({g.titles.get(d.get("doc_id"), d.get("doc_id")) for d in defs})[:3]}
    if nid in g.struct:
        doc = nid.split("/", 1)[0]
        s = g.struct[nid]
        out = {"kind": "corpus document" if s.parent_id is None else "corpus element",
               "document": g.titles.get(doc, doc), "document_date": (g.meta.get(doc) or {}).get("date")}
        if s.parent_id is not None:
            out["path"] = [_label(x) for x in g.ancestors(nid)[1:]]
            out["text"] = _cap(g.texts[doc][s.start:s.end], cfg["target_text_cap"])
        return out
    if nid.startswith(EXT_PREFIX):
        parts = nid.split("/")
        act_id = "/".join(parts[:2])
        act = g.nodes.get(act_id)
        head = {"act_title": act.title if act else parts[1]}
        if act:
            for k in ("numbers", "edition"):
                if act.attrs.get(k):
                    head[k] = act.attrs[k]
        if len(parts) == 2:
            return {"kind": "external act (not in the corpus)", **head}
        chain = []
        for i in range(3, len(parts) + 1):
            st = g.nodes.get("/".join(parts[:i]))
            chain.append(st.title if st else parts[i - 1])
        return {"kind": "element of an external act (stub)", **head, "element": " / ".join(chain)}
    if n is not None:                                     # a corpus act vertex that is not a record
        return {"kind": n.kind, "title": n.title}
    return {"kind": "unknown", "id": nid}


def render_item(g: AuditGraph, e: GraphEdge, cfg: dict) -> dict:
    """The judge's view of one edge (no planting marks)."""
    ev = e.attrs["evidence"]
    doc, a, b = ev["doc_id"], ev["start"], ev["end"]
    text = g.texts[doc]
    w = cfg["window"]
    lo, hi = max(0, a - w), min(len(text), b + w)
    ctx = text[lo:a] + "⟦" + _cap(text[a:b], cfg["quote_cap"]) + "⟧" + text[b:hi]
    anc = g.ancestors(e.src)
    src = {"document": g.titles.get(doc, doc), "document_type": (g.meta.get(doc) or {}).get("type"),
           "document_date": (g.meta.get(doc) or {}).get("date"),
           "node_path": [_label(x) for x in anc[1:]] or ["(the whole document)"]}
    heads = [f"{_label(x)} :: {_first_line(text[x.start:x.end])}" for x in anc[1:] if x.start < lo]
    if heads:
        src["ancestor_heads"] = heads
    attrs = {k: e.attrs[k] for k in ("relation", "op", "date") if e.attrs.get(k) is not None}
    return {"edge_type": e.type, "attrs": attrs, "source": src, "context": ctx,
            "evidence": _cap(ev["quote"], cfg["quote_cap"]), "target": render_target(g, e.dst, cfg)}


# --- planting (Ruling R19) --------------------------------------------------------------------------------

def _target_type(g: AuditGraph, nid: str) -> str:
    if nid in g.struct:
        return ("doc" if g.struct[nid].parent_id is None else "el:" + g.struct[nid].type)
    n = g.nodes.get(nid)
    if nid.startswith(EXT_PREFIX):
        return "ext:" + (n.type if n and "/" in nid[len(EXT_PREFIX):] else "act")
    return f"{n.kind}:{n.type}" if n else "?"


def _names(g: AuditGraph, e: GraphEdge) -> list[str]:
    """Strings whose presence in a shifted window would make the planted edge possibly correct."""
    out = [e.attrs["evidence"]["quote"]]
    n = g.nodes.get(e.dst)
    if n is not None and (e.dst.startswith(ENT_PREFIX) or e.dst.startswith(CON_PREFIX)):
        out += [n.title] + list(n.attrs.get("aliases") or [])[:10]
    return [s for s in out if s and len(s) >= 4]


def _stem_in(names: list[str], window: str) -> bool:
    w = window.casefold()
    for s in names:
        s = s.casefold().strip()
        if s[: max(5, min(len(s), 12))] in w:
            return True
    return False


def _swap_in_doc(g, e, rng, pool_by_doc, pool_type):
    doc = e.attrs["evidence"]["doc_id"]
    cands = sorted({d for d in pool_by_doc.get(doc, ()) if d != e.dst and d not in g.out_targets[e.src]
                    and _target_type(g, d) == _target_type(g, e.dst)})
    if not cands:
        cands = sorted({d for d in pool_by_doc.get(doc, ()) if d != e.dst and d not in g.out_targets[e.src]})
    quote = e.attrs["evidence"]["quote"]
    cands = [d for d in cands if not _dst_named(g, d, quote)]
    if not cands:
        return None
    return GraphEdge(e.src, rng.choice(cands), e.type, dict(e.attrs))


def _dst_named(g, d, quote) -> bool:
    n = g.nodes.get(d)
    return bool(n is not None and n.title and len(n.title) >= 4 and n.title.casefold() in quote.casefold())


def _random_same_type(g, e, rng, pool_by_doc, pool_type):
    tt = _target_type(g, e.dst)
    doc = e.attrs["evidence"]["doc_id"]
    cands = [d for d in pool_type.get(tt, ()) if d != e.dst and d not in g.out_targets[e.src]
             and not d.startswith(doc + "/") and d != doc]
    if tt.startswith("el:") and e.dst.split("/", 1)[0] == doc:     # internal: another node of the same doc
        cands = [d for d in pool_type.get(tt, ()) if d.startswith(doc + "/") and d != e.dst
                 and d not in g.out_targets[e.src]]
    quote = e.attrs["evidence"]["quote"]
    cands = [d for d in cands if not _dst_named(g, d, quote)]
    if not cands:
        return None
    return GraphEdge(e.src, rng.choice(sorted(cands)), e.type, dict(e.attrs))


def _shift_evidence(g, e, rng, pool_by_doc, pool_type, cfg=None):
    cfg = cfg or DEFAULTS
    ev = e.attrs["evidence"]
    doc, text = ev["doc_id"], g.texts[ev["doc_id"]]
    L = min(max(40, ev["end"] - ev["start"]), 400)
    paras = [(m.start(), m.end()) for m in re.finditer(r"[^\n]{30,}", text)]
    far = [(p, q) for p, q in paras if abs(p - ev["start"]) >= 1500]
    names = _names(g, e)
    rng.shuffle(far)
    for p, q in far[:60]:
        end = min(q, p + L)
        sp = text.rfind(" ", p, end)
        if end < q and sp > p + 20:
            end = sp
        n = g.node_at(doc, p, end)
        if n is None or n.parent_id is None or n.node_id == e.src:
            continue
        w = cfg["window"]
        if _stem_in(names, text[max(0, p - w):end + w]):
            continue
        attrs = {k: v for k, v in e.attrs.items() if k not in ("evidence", "evidence_all", "ops")}
        attrs["evidence"] = {"doc_id": doc, "start": p, "end": end, "quote": text[p:end]}
        return GraphEdge(n.node_id, e.dst, e.type, attrs)
    return None


def _op_swap(g, e, rng, pool_by_doc, pool_type):
    op = e.attrs.get("op")
    if op not in OP_SWAP:
        return None
    attrs = {k: v for k, v in e.attrs.items() if k not in ("evidence_all", "ops")}
    attrs["op"] = OP_SWAP[op]
    return GraphEdge(e.src, e.dst, e.type, attrs)


_PLANTERS = {"swap_in_doc": _swap_in_doc, "random_same_type": _random_same_type,
             "shift_evidence": _shift_evidence, "op_swap": _op_swap}


def plant_stratum(g: AuditGraph, stratum: str, pop: list[GraphEdge], sample: list[GraphEdge], k: int,
                  seed, cfg: dict | None = None) -> list[dict]:
    """k planted wrong edges of `stratum`: [{edge, method, base}] (deterministic)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    t = type_of(stratum)
    rng = _rng(seed, "plant", stratum)
    in_sample = {edge_sort_key(e) for e in sample}
    bases = [e for e in pop if edge_sort_key(e) not in in_sample] or list(pop)
    same_type = [e for e in g.edges if e.type == t] or pop
    pool_by_doc: dict[str, set] = defaultdict(set)
    for e in same_type:
        pool_by_doc[e.attrs["evidence"]["doc_id"]].add(e.dst)
    pool_type: dict[str, list] = defaultdict(list)
    for d in sorted({e.dst for e in pop}):
        pool_type[_target_type(g, d)].append(d)
    methods = list(METHODS[t])
    start = rng.randrange(len(methods))
    out, used = [], set()
    for i in range(k):
        order = methods[(start + i) % len(methods):] + methods[:(start + i) % len(methods)]
        for _ in range(50):
            base = rng.choice(bases)
            if edge_sort_key(base) in used and len(used) < len(bases):
                continue
            got = None
            for m in order:
                fn = _PLANTERS[m]
                pe = fn(g, base, rng, pool_by_doc, pool_type, cfg) if m == "shift_evidence" else \
                    fn(g, base, rng, pool_by_doc, pool_type)
                if pe is None:
                    continue
                if m == "op_swap" or ((pe.src, pe.type, pe.dst) not in g.existing
                                      and (pe.src, pe.dst) != (base.src, base.dst)):
                    got = (pe, m)
                    break
            if got:
                used.add(edge_sort_key(base))
                out.append({"edge": got[0], "method": got[1], "base": base})
                break
    return out


# --- judge ------------------------------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict auditor of a legal knowledge graph built from Uzbek (Cyrillic and Latin script) and Russian parliamentary documents (laws, Senate resolutions, committee conclusions, session stenograms). You receive a batch of extracted edges. For EACH edge decide whether it is CORRECT: the evidence text in its context really supports this edge from this source to this target, with these attributes.

Each item gives: the edge type and attrs; the source document title / type / date and the source node path (статья = article, часть = part, пункт = point, подпункт = sub-point, блок = block, вопрос = agenda item, выступление = speech), heads of ancestor nodes above the window, the text context with the evidence marked ⟦…⟧, and the target (a corpus node with its path and text; an external act not in the corpus with its normalised title; an element stub of an external act = act title + element path such as "11-модда / 5-қисм"; an entity with type and name; a concept with term and definition).

Rules per edge type:

REFERS_TO — the evidence is a mention in the source that refers to the target.
- relation "internal": a reference to an article / part / point of the SAME act ("ушбу Қонуннинг 5-моддаси", "статьей 36 настоящего Закона", a bare "5-модда" inside the act's own norms); the target must be exactly that element of the same document.
- relation "element": a reference to an article / part / point of ANOTHER act; the target must be that element (right number and level) of the right act (the act named in or right before the mention, or the act the document is about when the text says "Қонуннинг", "loyihaning", "мазкур Қонун" in a resolution / conclusion / stenogram about that act).
- relation "amended": an amendment instruction names the act or element being amended; the target must be that element / act of the act being amended (named in the instruction head, see ancestor heads).
- relation "act": a mention of an act (law, code, Constitution, other act); the target must be that act. An external act target is correct when its normalised title is the title of the act mentioned (inflection, script and abbreviation differences are fine); a mentioned registration number / date must not contradict the target's number / edition.
- relation "agenda": a stenogram / protocol agenda item names the bill / law under discussion; the target must be that act.
- relation "title": the document's own title names the act the document is about; the target must be that act.
- relation "version_of" / "translation_of": the source document is another version / a translation of the target document (same act).
- For element targets: a target that is the named element's nearest existing ancestor (the article when the text names its part) is acceptable; a different element, a wrong number or a wrong act is wrong.

AMENDS — the evidence is one amendment operation of an amending law; the target must be the element (or act) that the operation changes, in the act named by the instruction head (see ancestor heads / context), with the right numbers; a newly added element ("122¹-модда билан тўлдирилсин" → the new article 122¹) or the element supplemented is acceptable for op add. The op must match the verb: edit = "… таҳрирда баён этилсин" / "изложить" (restate) or "ўзгартирилсин"; add = "… билан тўлдирилсин" / "дополнить"; delete = "чиқариб ташлансин / чиқарилсин" / "исключить"; replace = "… деган сўзлар … билан алмаштирилсин" / "заменить"; renumber = "… деб ҳисоблансин" / "считать". date is the amending act's adoption date: do not reject for the date unless the context contradicts it.

REPEALS — the evidence declares an act or an element of an act invalid ("ўз кучини йўқотган деб топилсин", "признать утратившим силу", usually an enumerated item under "Қуйидагилар ўз кучини йўқотган деб топилсин:"); the target must be exactly the act (title, number, date consistent) or its element that the item names. date is the repealing act's date: do not reject for it unless contradicted.

DEFINES — the evidence is a "term – definition" paragraph of a law's basic-concepts article; correct only if the target concept's term is the term defined in THIS evidence (not a neighbouring definition) and the definition is this paragraph's definition.

REGULATES — the source is a norm (article / part / point of a law, a point of a resolution, a block of a committee conclusion) whose evidence is a mention of the target entity (a state body, an official post, a named legal procedure, a legal category) or of the target concept as used in that norm. Correct only if the evidence mentions exactly this entity / concept (the same real-world body / post / category; script, inflection, "Ўзбекистон Республикаси" / "Олий Мажлисининг" prefixes do not matter) AND the mention is NOT: inside a gazette citation "(… Ахборотномаси, YYYY йил, № …)"; inside a quoted act title; inside quoted replaced words («…» деган сўзлар); only the owner in "X ҳузуридаги Y" / "X huzuridagi Y" (the edge must go to Y); a modifier-only use ("Марказий банк курси" → not the bank itself); committee author / introductory boilerplate; a signature line. For a concept: the evidence uses the defined term with its defined meaning. Judge the identity of the target, not its entity_type label.

General: judge only from what is shown; if the evidence clearly supports the edge, mark it correct; if any requirement is violated (wrong target, wrong act, wrong element number, wrong op, the evidence does not mention the target), mark it incorrect; a borderline case is incorrect with low confidence. Write a short reason (quote the decisive words), then correct (true/false) and confidence in [0, 1]. Return one verdict per item id, every id exactly once."""


def judge_schema() -> dict:
    return J.verdict_schema({"reason": {"type": "string"}, "correct": {"type": "boolean"},
                             "confidence": {"type": "number"}})


def _verdict_ok(v: dict) -> bool:
    return isinstance(v.get("correct"), bool) and isinstance(v.get("confidence"), (int, float))


def verify_verdicts(ids: list[str]) -> Callable[[Any], tuple[bool, str | None]]:
    return J.verdict_checker(ids, _verdict_ok)


def make_batches(items: list[dict], size: int, seed) -> list[list[dict]]:
    """Per stratum: items shuffled (random.Random(seed:batch:stratum)) and cut into `size` chunks;
    judge ids e1..eK within a batch."""
    by: dict[str, list] = defaultdict(list)
    for it in items:
        by[it["stratum"]].append(it)
    out = []
    for s in sorted(by):
        out += J.shuffled_batches(by[s], size, f"{seed}:batch:{s}")
    return out


batch_input = J.batch_input


def llm_judge(llm, pass_name: str = "audit") -> Callable[[list[dict]], dict]:
    """judge(batch) -> {uid: verdict | None} via KgLLM.cached_json (cache, gate, budget, ledger)."""
    return J.make_judge(llm, pass_name, JUDGE_PROMPT, judge_schema(), verify_verdicts,
                        lambda v: {"correct": v["correct"], "reason": v["reason"],
                                   "confidence": min(1.0, max(0.0, float(v["confidence"])))})


# --- metrics ----------------------------------------------------------------------------------------------

def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes of n (n may be an effective, non-integer size)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def corrected(p: float | None, se: float | None, sp: float | None) -> float | None:
    """Rogan-Gladen: true precision from the judge-observed one, sensitivity Se (rejects a wrong
    edge) and specificity Sp (accepts a correct edge); clipped to [0, 1]."""
    if p is None or se is None or sp is None or se + sp - 1 <= 0:
        return None
    return min(1.0, max(0.0, (p + se - 1) / (se + sp - 1)))


def pooled(strata: dict[str, dict]) -> dict:
    """Strata {name: {N, n, k}} pooled with population weights: P = sum w_s p_s,
    var = sum w_s^2 p~(1-p~)/n_s (p~ = (k+0.5)/(n+1)), Wilson with n_eff = P~(1-P~)/var capped at the
    real sample size sum n_s (strata with k = n add almost no variance; an n_eff above n is anti-conservative)."""
    rows = [r for r in strata.values() if r["n"] > 0]
    N = sum(r["N"] for r in rows)
    if not rows or N == 0:
        return {"N": N, "precision": None, "ci": None, "n_eff": 0}
    P = sum(r["N"] / N * r["k"] / r["n"] for r in rows)
    var, Pt = 0.0, 0.0
    for r in rows:
        w = r["N"] / N
        pt = (r["k"] + 0.5) / (r["n"] + 1)
        Pt += w * pt
        var += w * w * pt * (1 - pt) / r["n"]
    n_real = sum(r["n"] for r in rows)
    n_eff = Pt * (1 - Pt) / var if var > 0 else n_real
    n_eff = min(n_eff, n_real)                            # never more information than edges looked at (R20 e)
    return {"N": N, "precision": P, "ci": list(wilson(P * n_eff, n_eff)), "n_eff": n_eff}


def contains_violations(records, stub_edges: list[GraphEdge], nodes: dict[str, GraphNode]) -> dict:
    """CONTAINS must go parent -> direct child: structure edges (Node.parent_id) need the child's id =
    parent id + one segment and the child's span inside the parent's; stub edges need the id rule and
    both endpoints in the graph."""
    bad: Counter = Counter()
    ex: list = []
    total = 0

    def viol(kind, s, d):
        bad[kind] += 1
        if len(ex) < 10:
            ex.append({"check": kind, "src": s, "dst": d})

    for r in records:
        by = {n.node_id: n for n in r.structure}
        for n in r.structure:
            if n.parent_id is None:
                continue
            total += 1
            p = by.get(n.parent_id)
            if p is None:
                viol("parent_missing", n.parent_id, n.node_id)
                continue
            rest = n.node_id[len(p.node_id) + 1:] if n.node_id.startswith(p.node_id + "/") else None
            if not rest or "/" in rest:
                viol("not_direct_child_id", p.node_id, n.node_id)
            if not (p.start <= n.start and n.end <= p.end):
                viol("span_outside_parent", p.node_id, n.node_id)
    for e in stub_edges:
        total += 1
        rest = e.dst[len(e.src) + 1:] if e.dst.startswith(e.src + "/") else None
        if not rest or "/" in rest:
            viol("not_direct_child_id", e.src, e.dst)
        for end in (e.src, e.dst):
            if end not in nodes:
                viol("endpoint_missing", e.src, e.dst)
    return {"total": total, "violations": sum(bad.values()), "by_check": dict(sorted(bad.items())), "examples": ex}


def manual_subset(sample: dict[str, list[GraphEdge]], seed, quota: dict | None = None) -> list[tuple[str, GraphEdge]]:
    """A fixed subset of the genuine sample for manual labelling: quota per edge type, spread over strata."""
    quota = quota or MANUAL_QUOTA
    out = []
    for t, q in sorted(quota.items()):
        pool = [(s, e) for s in sorted(sample) if type_of(s) == t for e in sample[s]]
        rng = _rng(seed, "manual", t)
        by_s: dict[str, list] = defaultdict(list)
        for s, e in pool:
            by_s[s].append(e)
        order = sorted(by_s)
        picks: list = []
        i = 0
        while len(picks) < min(q, len(pool)):
            s = order[i % len(order)]
            rest = [e for e in by_s[s] if (s, e) not in picks]
            if rest:
                picks.append((s, rest[rng.randrange(len(rest))]))
            i += 1
        out += picks
    return out


ADJ_CORRECT = {"strict": ("judge_error",), "lenient": ("judge_error", "borderline", "relation_label")}
# Rogan-Gladen needs one Se / Sp for every stratum; here Sp comes from 30 manual edges (CI ~0.74-0.97) and
# differs a lot by stratum (AMENDS:renumber 0.675 vs ~1.0 elsewhere), and Se from easy synthetic errors, so
# every stratum with p >= ~Sp clips to 1.0: the column is kept for reference only (Task 7 review, R20 e)
CORRECTED_STATUS = ("non-informative: Rogan-Gladen assumes one judge Se/Sp for all strata (violated: Sp by "
                    "stratum 0.675-1.0), Sp is estimated from 30 edges and Se from synthetic plants; values clip "
                    "to 1.0. Kept as secondary, never used for the gate; use precision_adjudicated_strict / the "
                    "rule-based rescore")


def corrected_ci(ci, se_ci, sp_ci) -> list | None:
    """Rogan-Gladen applied to the Wilson bounds over every corner of the Se / Sp CIs (a conservative
    box: [min corrected(lo), max corrected(hi)])."""
    if not ci or not se_ci or not sp_ci:
        return None
    lo = [corrected(ci[0], a, b) for a in se_ci for b in sp_ci]
    hi = [corrected(ci[1], a, b) for a in se_ci for b in sp_ci]
    lo, hi = [x for x in lo if x is not None], [x for x in hi if x is not None]
    return [min(lo), max(hi)] if lo and hi else None


def evaluate(items: list[dict], verdicts: dict[str, dict | None], counts: dict[str, int],
             manual: dict[str, dict] | None = None, adjudication: dict[str, dict] | None = None) -> dict:
    """Metrics from judged items [{uid, stratum, planted}], verdicts {uid: {correct,...}|None},
    population sizes per stratum, manual labels {uid: {correct}} and the manual adjudication of
    judge-rejected genuine edges {uid: {category}} (judge_error | true_error | borderline |
    relation_label). Per stratum / type: raw (judge) precision, Rogan-Gladen corrected precision
    (Se = planted, Sp = manual subset; CI = box over the Se / Sp CIs) and adjudicated precision
    (judge rejections replaced by the manual verdict; strict and lenient)."""
    manual, adjudication = manual or {}, adjudication or {}
    per: dict[str, dict] = {}
    for s in sorted(counts):
        per[s] = {"N": counts[s], "n": 0, "k": 0, "unjudged": 0, "planted": 0, "planted_rejected": 0,
                  "rejected_unadjudicated": 0, "k_adj_strict": 0, "k_adj_lenient": 0,
                  "adjudication": Counter()}
    for it in items:
        v = verdicts.get(it["uid"])
        r = per[it["stratum"]]
        if v is None:
            r["unjudged"] += 1
            continue
        if it["planted"]:
            r["planted"] += 1
            r["planted_rejected"] += int(not v["correct"])
            continue
        r["n"] += 1
        r["k"] += int(v["correct"])
        if v["correct"]:
            r["k_adj_strict"] += 1
            r["k_adj_lenient"] += 1
            continue
        cat = (adjudication.get(it["uid"]) or {}).get("category")
        if cat is None:
            r["rejected_unadjudicated"] += 1
            continue
        r["adjudication"][cat] += 1
        r["k_adj_strict"] += int(cat in ADJ_CORRECT["strict"])
        r["k_adj_lenient"] += int(cat in ADJ_CORRECT["lenient"])
    pl = sum(r["planted"] for r in per.values())
    plr = sum(r["planted_rejected"] for r in per.values())
    se = plr / pl if pl else None
    se_ci = list(wilson(plr, pl)) if pl else None
    agree = n_man = man_ok = man_ok_acc = man_bad = man_bad_rej = 0
    for uid, lab in sorted(manual.items()):
        v = verdicts.get(uid)
        if v is None:
            continue
        n_man += 1
        agree += int(v["correct"] == lab["correct"])
        if lab["correct"]:
            man_ok += 1
            man_ok_acc += int(v["correct"])
        else:
            man_bad += 1
            man_bad_rej += int(not v["correct"])
    sp = man_ok_acc / man_ok if man_ok else None
    sp_ci = list(wilson(man_ok_acc, man_ok)) if man_ok else None
    adj_all = sum((r["adjudication"] for r in per.values()), Counter())
    rej = sum(r["n"] - r["k"] for r in per.values())
    judge = {"sensitivity_planted": se, "planted": pl, "planted_rejected": plr, "sensitivity_planted_ci": se_ci,
             "manual_n": n_man, "manual_agreement": agree / n_man if n_man else None,
             "manual_agreement_ci": list(wilson(agree, n_man)) if n_man else None,
             "specificity_manual": sp, "specificity_manual_ci": sp_ci, "manual_correct": man_ok,
             "manual_correct_accepted": man_ok_acc, "manual_wrong": man_bad, "manual_wrong_rejected": man_bad_rej,
             "genuine_rejected": rej, "adjudication": dict(sorted(adj_all.items())),
             "rejections_overturned_share": (adj_all.get("judge_error", 0) / rej) if rej else None}
    for s, r in per.items():
        r["adjudication"] = dict(sorted(r["adjudication"].items()))
        n = r["n"]
        p = r["k"] / n if n else None
        r["precision"] = p
        r["ci"] = list(wilson(r["k"], n)) if n else None
        r["precision_corrected"] = corrected(p, se, sp)
        r["ci_corrected"] = corrected_ci(r["ci"], se_ci, sp_ci) if r["precision_corrected"] is not None else None
        for mode in ("strict", "lenient"):
            k = r[f"k_adj_{mode}"]
            r[f"precision_adjudicated_{mode}"] = k / n if n else None
            r[f"ci_adjudicated_{mode}"] = list(wilson(k, n)) if n else None
        r["sensitivity"] = r["planted_rejected"] / r["planted"] if r["planted"] else None
        r["below_gate"] = p is not None and p < 0.90
        r["below_gate_adjudicated_strict"] = n > 0 and r["precision_adjudicated_strict"] < 0.90
    types = {}
    for t in sorted({type_of(s) for s in per}):
        st = {s: r for s, r in per.items() if type_of(s) == t}
        pp = pooled(st)
        pc = corrected(pp["precision"], se, sp)
        pl_t = sum(r["planted"] for r in st.values())
        row = {**pp, "n": sum(r["n"] for r in st.values()), "k": sum(r["k"] for r in st.values()),
               "precision_corrected": pc, "corrected_informative": False,
               "ci_corrected": corrected_ci(pp["ci"], se_ci, sp_ci) if pc is not None else None,
               "planted": pl_t, "sensitivity": (sum(r["planted_rejected"] for r in st.values()) / pl_t) if pl_t else None,
               "gate_raw": pp["precision"] is not None and pp["precision"] >= 0.90,
               "gate_corrected": pc is not None and pc >= 0.90,
               "strata_below_0.90": sorted(s for s, r in st.items() if r["below_gate"])}
        for mode in ("strict", "lenient"):
            pa = pooled({s: {"N": r["N"], "n": r["n"], "k": r[f"k_adj_{mode}"]} for s, r in st.items()})
            row[f"precision_adjudicated_{mode}"] = pa["precision"]
            row[f"ci_adjudicated_{mode}"] = pa["ci"]
            row[f"gate_adjudicated_{mode}"] = pa["precision"] is not None and pa["precision"] >= 0.90
        row["strata_below_0.90_adjudicated_strict"] = sorted(s for s, r in st.items() if r["below_gate_adjudicated_strict"])
        types[t] = row
    return {"judge": judge, "strata": per, "types": types, "precision_corrected_status": CORRECTED_STATUS}


# --- rule-based checks of edge attributes (Ruling R20: the judge is weak on op / relation) ---------------

# AMENDS op from the instruction's verb, independent of the extractor's clause splitter: quoted text is
# masked; the LAST Uzbek verb formula decides (Uzbek verbs end the clause), else the FIRST Russian one.
_APO = "[ʻʼ'`‘’]"
OP_LEXICON = {
    "uz": [("replace", r"(?:алмаштирилсин|almashtirilsin)"),
           ("renumber", r"(?:деб|deb)\s+(?:ҳисоблансин|hisoblansin)"),
           ("edit", r"(?:баён|bayon)\s+(?:этилсин|қилинсин|etilsin|qilinsin)|(?:ўзгартирилсин|o" + _APO + r"zgartirilsin)"),
           ("add", r"(?:тўлдирилсин|to" + _APO + r"ldirilsin)"),
           ("delete", r"(?:чиқариб\s+ташлансин|chiqarib\s+tashlansin|чиқарилсин|chiqarilsin)")],
    "ru": [("replace", r"замени(?:ть|в)"), ("renumber", r"считать"), ("edit", r"изложить"),
           ("add", r"дополнить"), ("delete", r"исключить")],
}
_OP_RX = {lang: [(op, re.compile(r"(?i)(?<!\w)(?:" + rx + r")(?!\w)")) for op, rx in v] for lang, v in OP_LEXICON.items()}


def _mask_quotes(s: str) -> str:
    """Text inside «…» / “…” (nested) replaced by spaces (positions kept)."""
    out, depth = [], 0
    for ch in s:
        if ch in "«“":
            depth += 1
            out.append(" ")
        elif ch in "»”" and depth:
            depth -= 1
            out.append(" ")
        else:
            out.append(" " if depth else ch)
    return "".join(out)


def expected_op(quote: str) -> tuple[str | None, str]:
    """(op, how) the instruction verb implies: how = "uz_last" | "ru_first" | "no_verb"; "…_multi" when
    verbs of several ops stand unquoted in the clause (the decisive one is still returned)."""
    t = _mask_quotes(quote)
    hits = sorted((m.start(), op) for op, rx in _OP_RX["uz"] for m in rx.finditer(t))
    how = "uz_last"
    if not hits:
        hits = sorted((m.start(), op) for op, rx in _OP_RX["ru"] for m in rx.finditer(t))
        how = "ru_first"
        if not hits:
            return None, "no_verb"
        op = hits[0][1]
    else:
        op = hits[-1][1]
    return op, how + ("_multi" if len({o for _, o in hits}) > 1 else "")


def op_pairs(e: GraphEdge) -> list[tuple[dict, str]]:
    """(evidence, op) of an AMENDS edge: evidence_all aligned with ops (R12) when both have one entry per
    operation, else the primary evidence with the primary op."""
    ops = e.attrs.get("ops")
    evs = e.attrs.get("evidence_all")
    if ops and evs and len(ops) == len(evs):
        return list(zip(evs, ops))
    return [(e.attrs["evidence"], e.attrs.get("op"))]


def op_consistency(edges) -> dict:
    """Programmatic AMENDS op check over `edges`: every (evidence, op) pair against expected_op; a
    contradiction = a verb found and it implies another op."""
    n = match = 0
    how: Counter = Counter()
    conf: Counter = Counter()
    bad: list = []
    edges_bad = set()
    n_edges = unaligned = 0
    for e in edges:
        if e.type != "AMENDS":
            continue
        n_edges += 1
        unaligned += int(bool(e.attrs.get("ops")) and len(e.attrs["ops"]) != len(e.attrs.get("evidence_all") or ()))
        for ev, op in op_pairs(e):
            n += 1
            exp, h = expected_op(ev["quote"])
            how[h] += 1
            if exp is None:
                continue
            conf[f"{op}->{exp}"] += 1
            if exp == op:
                match += 1
            else:
                edges_bad.add((e.src, e.dst))
                bad.append({"src": e.src, "dst": e.dst, "op": op, "expected": exp, "how": h, "quote": ev["quote"]})
    verbs = n - how.get("no_verb", 0)
    return {"edges": n_edges, "ops_unaligned_edges": unaligned, "pairs": n, "with_verb": verbs, "match": match, "contradictions": len(bad),
            "contradicting_edges": len(edges_bad), "agreement": match / verbs if verbs else None,
            "how": dict(sorted(how.items())), "op_vs_expected": dict(sorted(conf.items())),
            "contradiction_list": sorted(bad, key=lambda x: (x["src"], x["dst"], x["quote"]))}


def in_node_quote(g: AuditGraph, e: GraphEdge) -> bool:
    """The evidence starts inside a quotation opened earlier in its SOURCE NODE («…» / “…” balance over the
    node text before it): a citation inside quoted new / replaced text (multi-paragraph quotes included)."""
    ev = e.attrs["evidence"]
    t = g.texts[ev["doc_id"]]
    n = g.struct.get(e.src)
    a = n.start if n is not None and n.start <= ev["start"] else t.rfind("\n", 0, ev["start"]) + 1
    pre = t[a:ev["start"]]
    return pre.count("«") > pre.count("»") or pre.count("“") > pre.count("”")


def _is_act_vertex(nid: str) -> bool:
    return (nid.startswith(EXT_PREFIX) and nid.count("/") == 1) or (not nid.startswith(EXT_PREFIX) and "/" not in nid)


def rule_relation(g: AuditGraph, e: GraphEdge) -> str | None:
    """The relation class a REFERS_TO edge should carry by structural rules (None: no rule applies):
    agenda / title / version by their source, internal = same document, act = an act vertex target, an
    element of another act: cited_in_amendment when the evidence stands in a quotation of its source node,
    amended when it does not and the source is an amendment instruction (edge labelled amended /
    cited_in_amendment), else element."""
    rel = e.attrs.get("relation")
    sdoc, ddoc = e.src.split("/", 1)[0], e.dst.split("/", 1)[0] if not e.dst.startswith(EXT_PREFIX) else None
    if rel in ("version_of", "translation_of"):
        return rel if "/" not in e.src and _is_act_vertex(e.dst) else "?"
    if rel == "agenda":
        return "agenda" if "/ВОПР." in e.src else None
    if rel == "title":
        return "title" if "/" not in e.src else None
    if ddoc == sdoc:
        return "internal"
    if _is_act_vertex(e.dst):
        return "act"
    if rel in ("amended", "cited_in_amendment"):
        return "cited_in_amendment" if in_node_quote(g, e) else "amended"
    return "element"


def rescore(items: list[dict], verdicts: dict, adjudication: dict, g: AuditGraph, edges, pops: dict,
            adoption_line=None) -> dict:
    """Rule-based re-score of the judged (frozen) genuine sample against the CURRENT edges, no LLM calls.
    Target correctness = the judge verdict with the manual adjudication of its rejections (strict:
    judge_error / relation_label = right target); label = rule_relation (REFERS_TO), expected_op (AMENDS),
    not an adoption line (REGULATES, `adoption_line(text, a, b)`). A sampled edge absent from the current
    graph is 'removed' (out of the population); present ones are re-stratified by their current attrs and
    pooled with the CURRENT population sizes."""
    cur = {(e.type, e.src, e.dst): e for e in edges}
    per: dict[str, dict] = {}
    status: Counter = Counter()
    removed, label_bad = [], []
    for it in sorted(items, key=lambda x: x["uid"]):
        if it["planted"]:
            continue
        v = verdicts.get(it["uid"])
        if v is None:
            continue
        old = it["edge"]
        cat = (adjudication.get(it["uid"]) or {}).get("category")
        tgt = bool(v["correct"]) or cat in ("judge_error", "relation_label")
        e = cur.get((old.type, old.src, old.dst))
        if e is None:
            status["removed"] += 1
            removed.append({"uid": it["uid"], "stratum": it["stratum"], "src": old.src, "dst": old.dst,
                            "target_correct": tgt, "adjudication": cat})
            continue
        s = stratum_of(e)
        lab, why = True, None
        if e.type == "REFERS_TO":
            want = rule_relation(g, e)
            lab = want is None or want == e.attrs.get("relation")
            why = f"{e.attrs.get('relation')}->{want}"
        elif e.type == "AMENDS":
            for ev, op in op_pairs(e):
                exp, _ = expected_op(ev["quote"])
                if exp is not None and exp != op:
                    lab, why = False, f"{op}->{exp}"
        elif e.type == "REGULATES" and adoption_line is not None:
            for ev in e.attrs.get("evidence_all") or [e.attrs["evidence"]]:
                if adoption_line(g.texts[ev["doc_id"]], ev["start"], ev["end"]):
                    lab, why = False, "adoption_line"
        status["relabelled" if s != it["stratum"] else "same_stratum"] += 1
        r = per.setdefault(s, {"N": len(pops.get(s, ())), "n": 0, "k_target": 0, "k_label": 0, "from": Counter()})
        r["n"] += 1
        r["k_target"] += int(tgt)
        r["k_label"] += int(tgt and lab)
        r["from"][it["stratum"]] += 1
        if not lab:
            label_bad.append({"uid": it["uid"], "stratum": s, "src": e.src, "dst": e.dst, "why": why})
    for s, r in per.items():
        r["from"] = dict(sorted(r["from"].items()))
        for k in ("target", "label"):
            r[f"precision_{k}"] = r[f"k_{k}"] / r["n"] if r["n"] else None
            r[f"ci_{k}"] = list(wilson(r[f"k_{k}"], r["n"])) if r["n"] else None
    types = {}
    for t in sorted({type_of(s) for s in pops} | {type_of(s) for s in per}):
        st = {s: r for s, r in per.items() if type_of(s) == t}
        unsampled = sorted(s for s in pops if type_of(s) == t and s not in st)
        row = {"N": sum(len(pops[s]) for s in pops if type_of(s) == t), "n": sum(r["n"] for r in st.values()),
               "strata_without_sample": {s: len(pops[s]) for s in unsampled}}
        for k in ("target", "label"):
            pp = pooled({s: {"N": r["N"], "n": r["n"], "k": r[f"k_{k}"]} for s, r in st.items()})
            row[f"precision_{k}"], row[f"ci_{k}"], row[f"n_eff_{k}"] = pp["precision"], pp["ci"], pp["n_eff"]
            row[f"gate_{k}"] = pp["precision"] is not None and pp["precision"] >= 0.90
        types[t] = row
    return {"status": dict(sorted(status.items())), "strata": dict(sorted(per.items())), "types": types,
            "removed": removed, "label_wrong": label_bad,
            "note": "precision_target = right target (judge + adjudication, strict); precision_label = right "
                    "target AND the rule-based attribute (REFERS_TO relation, AMENDS op, REGULATES not an "
                    "adoption line); CURRENT population weights, frozen judged sample re-stratified"}


def relation_rule_population(g: AuditGraph, edges) -> dict:
    """rule_relation over every REFERS_TO edge: agreement per relation and the mismatch classes."""
    n: Counter = Counter()
    bad: Counter = Counter()
    ex: list = []
    for e in edges:
        if e.type != "REFERS_TO":
            continue
        rel = e.attrs.get("relation")
        want = rule_relation(g, e)
        n[rel] += 1
        if want is not None and want != rel:
            bad[f"{rel}->{want}"] += 1
            if len(ex) < 20:
                ex.append({"src": e.src, "dst": e.dst, "relation": rel, "rule": want, "quote": e.attrs["evidence"]["quote"]})
    return {"edges": dict(sorted(n.items())), "mismatch": dict(sorted(bad.items())), "mismatch_total": sum(bad.values()),
            "examples": ex}
