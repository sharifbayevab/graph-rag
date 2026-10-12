"""Gold labels anchored on text (IMPLEMENTATION.md §7, §8.1; roadmap §5a item 5): (doc_id, start, end,
sha256(text[start:end])); chunk ids derived from anchors and re-derived by `remap` after re-chunking.
Q-real (eval/validation_set.jsonl, type == "content", 340, both scripts) is the separate REAL-question test set
(user decision П2): split "qreal_test". The document split (train / val / test, П3) for the synthetic dataset is
assigned here, before any generation."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path

from graphrag import config
from graphrag.data.legacy.normalize import detect_language
from graphrag.data.normalize import to_internal

VALIDATION = config.ROOT.parent / "eval" / "validation_set.jsonl"
_APOS = re.compile(r"[‘’ʼʻ`'´]")
_QUOT = re.compile(r"[«»“”„\"]")


@dataclass(frozen=True)
class Anchor:
    doc_id: str
    start: int
    end: int
    sha256: str


@dataclass
class GoldItem:
    qid: str
    section: str
    difficulty: str
    question_cyr: str
    question_lat: str | None
    gold_answer: str
    supporting_quote: str
    doc_id: str
    tier: str
    anchors: list[Anchor]
    core: list[str]
    support_anchors: list[Anchor] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    alt_equiv: list[str] = field(default_factory=list)     # equivalent restatement of the answer (any-of eval credit)
    alt_mention: list[str] = field(default_factory=list)   # the quote without the answer / unreviewed copies (leakage only)
    split: str = "qreal_test"
    review: str | None = None
    origin: str = "qreal"
    gold_answer_orig: str | None = None                    # validation-set answer when the review corrected it
    context: list[str] = field(default_factory=list)       # M3-R9: chunks that only name the cited act / session --
                                                           # not C*, not alt_equiv; neutral in eval (no recall, no FP)

    @property
    def gold(self) -> list[str]:
        return sorted(set(self.core) | set(self.support))

    @property
    def alt(self) -> list[str]:
        """Every non-gold chunk carrying the quote (D8 leakage exclusion): alt_equiv | alt_mention (Ruling M3-R5)."""
        return sorted(set(self.alt_equiv) | set(self.alt_mention))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GoldItem":
        d = dict(d)
        d["anchors"] = [Anchor(**a) for a in d["anchors"]]
        d["support_anchors"] = [Anchor(**a) for a in d.get("support_anchors", [])]
        if "alt" in d:                                      # pre-M3-R5 files: unclassified copies are mentions
            d["alt_mention"] = sorted(set(d.get("alt_mention", [])) | set(d.pop("alt")))
        return cls(**d)


def text_sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_map(s: str) -> tuple[str, list[int]]:
    """Whitespace / case / quote / apostrophe-normalized `s` and, per output character, its index in `s`."""
    out, pos = [], []
    for i, ch in enumerate(s):
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                pos.append(i)
            continue
        ch = "'" if _APOS.fullmatch(ch) else '"' if _QUOT.fullmatch(ch) else ch
        for c in ch.lower():
            out.append(c)
            pos.append(i)
    if out and out[-1] == " ":
        out.pop()
        pos.pop()
    return "".join(out), pos


def wsn(s: str) -> str:
    return norm_map(unicodedata.normalize("NFC", s))[0]


def find_spans(text: str, quote: str) -> list[tuple[int, int]]:
    n, pos = norm_map(text)
    q = wsn(quote)
    out, i = [], (n.find(q) if q else -1)
    while i >= 0:
        out.append((pos[i], pos[i + len(q) - 1] + 1))
        i = n.find(q, i + 1)
    return out


def chunks_overlapping(rec, start: int, end: int) -> list[str]:
    return sorted(c.chunk_id for c in rec.chunks if c.start < end and c.end > start)


def locate(rec, quote: str) -> tuple[list[tuple[int, int]], str]:
    """Spans of the quote in rec.text and the tier: exact; else on the internal form against chunk text_norm --
    internal (one chunk), internal-span2 (two adjacent); else fuzzy60. Non-exact tiers anchor whole chunk spans."""
    spans = find_spans(rec.text, quote)
    if spans:
        return spans, "exact"
    qi = wsn(to_internal(quote, detect_language(quote)))
    cs = sorted(rec.chunks, key=lambda c: c.start)
    nt = [wsn(c.text_norm) for c in cs]
    hit = [(c.start, c.end) for c, t in zip(cs, nt) if qi and qi in t]
    if hit:
        return hit, "internal"
    for a, b, ta, tb in zip(cs, cs[1:], nt, nt[1:]):
        if qi and qi in ta + " " + tb:
            return [(a.start, b.end)], "internal-span2"
    best = (0, None)
    for c, t in zip(cs, nt):
        m = SequenceMatcher(None, qi, t, autojunk=False).find_longest_match(0, len(qi), 0, len(t))
        if m.size > best[0]:
            best = (m.size, c)
    if qi and best[1] is not None and best[0] >= 0.6 * len(qi):
        return [(best[1].start, best[1].end)], "fuzzy60"
    return [], "unmapped"


def anchors_of(rec, spans) -> list[Anchor]:
    return [Anchor(rec.doc_id, s, e, text_sha(rec.text[s:e])) for s, e in spans]


def docs_by_file(records) -> dict:
    out = {}
    for r in records:
        for f in {unicodedata.normalize("NFC", x) for x in [r.source.file, *r.source.aliases]}:
            out[f] = r
    return out


def source_record(x: dict, by_file: dict):
    f = unicodedata.normalize("NFC", x["source_files"][0])
    if f in by_file:
        return by_file[f]
    b = f.rsplit("/", 1)[-1]
    cand = {r.doc_id: r for k, r in by_file.items() if k.rsplit("/", 1)[-1] == b}
    return next(iter(cand.values())) if len(cand) == 1 else None


def build_items(records, validation: Path = VALIDATION) -> tuple[list[GoldItem], dict]:
    by_file = docs_by_file(records)
    all_chunks = [(r.doc_id, c.chunk_id, wsn(c.text)) for r in records for c in r.chunks]
    xs = [json.loads(line) for line in open(validation, encoding="utf-8")]
    items, tiers = [], Counter()
    for x in (x for x in xs if x["type"] == "content"):
        rec = source_record(x, by_file)
        if rec is None:
            tiers["no-file"] += 1
            continue
        spans, tier = locate(rec, x["supporting_quote"])
        tiers[tier] += 1
        if not spans:
            continue
        core = sorted({c for s, e in spans for c in chunks_overlapping(rec, s, e)})
        q = wsn(x["supporting_quote"])
        # the quote verbatim in OTHER documents: unreviewed copies (a Senate decision citing the law, stenogram
        # boilerplate) -> alt_mention (leakage only); same-document equivalents come from the review (alt_equiv)
        alt = sorted(cid for d, cid, t in all_chunks if d != rec.doc_id and q and q in t)
        review = (f"tier:{tier}" if tier != "exact" else f"multi:{len(spans)}" if len(spans) > 1
                  else f"wide:{len(core)}" if len(core) > 2 else None)
        items.append(GoldItem(x["id"], x["section"], x["difficulty"], x["question_cyr"], x.get("question_lat"),
                              x["gold_answer"], x["supporting_quote"], rec.doc_id, tier, anchors_of(rec, spans), core,
                              alt_mention=alt, review=review))
    return sorted(items, key=lambda i: i.qid), {"tiers": dict(sorted(tiers.items()))}


def remap(items: list[GoldItem], records) -> tuple[list[GoldItem], list[str], list[str]]:
    recs = {r.doc_id: r for r in records}
    out, problems, notes = [], [], []
    for it in items:
        rec = recs.get(it.doc_id)
        if rec is None:
            problems.append(f"{it.qid}: document {it.doc_id} is gone")
            continue
        anchors = list(it.anchors)
        if any(text_sha(rec.text[a.start:a.end]) != a.sha256 for a in anchors):
            spans, tier = locate(rec, it.supporting_quote)
            if not spans:
                problems.append(f"{it.qid}: anchor text changed and the quote is not found")
                continue
            anchors = anchors_of(rec, spans)
            notes.append(f"{it.qid}: anchor relocated by quote ({tier})")
        sup = [a for a in it.support_anchors if text_sha(rec.text[a.start:a.end]) == a.sha256]
        if len(sup) != len(it.support_anchors):
            notes.append(f"{it.qid}: {len(it.support_anchors) - len(sup)} support anchors changed (re-run complete_gold)")
        core = sorted({c for a in anchors for c in chunks_overlapping(rec, a.start, a.end)})
        support = sorted({c for a in sup for c in chunks_overlapping(rec, a.start, a.end)} - set(core))
        out.append(replace(it, anchors=anchors, core=core, support_anchors=sup, support=support))
    return out, problems, notes


def linked_doc_pairs(path: str = "data/graph/related_stats.json") -> list[tuple[str, str]]:
    """Translation / edition pairs and near-duplicate documents (M2 Task 8 grouping): one split each."""
    g = json.loads(config.path(path).read_text(encoding="utf-8"))["groups"]
    pairs = [tuple(p) for p in g["linked"]]
    for nd in g["near_duplicates"]:
        pairs += [(nd["docs"][0], d) for d in nd["docs"][1:]]
    return sorted(pairs)


ACT_LINK_RELATIONS = ("title", "version_of", "translation_of")


def act_link_pairs(edges, doc_sections: dict[str, str]) -> list[tuple[str, str]]:
    """Ruling M3-R4: same-subject-act links (a QARORLAR decision / XULOSALAR conclusion titled with a corpus law,
    editions, translations) -> document pairs that must share a split. `edges`: dicts {src, dst, type, attrs} of the
    pinned graph artifact; vertices map to their document (prefix before '/'). Only REFERS_TO with relation in
    ACT_LINK_RELATIONS between two DIFFERENT corpus documents; EXT/ stubs, agenda / act / amendment citations and
    any link touching a STENOGRAMMA document never merge groups."""
    out = set()
    for e in edges:
        if e["type"] != "REFERS_TO" or (e.get("attrs") or {}).get("relation") not in ACT_LINK_RELATIONS:
            continue
        a, b = e["src"].split("/", 1)[0], e["dst"].split("/", 1)[0]
        if a == b or a not in doc_sections or b not in doc_sections:
            continue
        if "STENOGRAMMA" in (doc_sections[a], doc_sections[b]):
            continue
        out.add((min(a, b), max(a, b)))
    return sorted(out)


HEADING_VERTICES = ("", "/ЗАГ.1", "/ПРЕАМБ.1", "/П.1")
_MB = " munosabati bilan "


def _same_act_fuzzy(a: str, b: str, min_ratio: float) -> bool:
    """Drifted wording of ONE act title (draft vs adopted title, Qirgʼiz / Qirgʼiziston): ratio >= min_ratio, the same
    numbers, and -- for "… munosabati bilan <object>" titles -- the same amended object (a reform package amending
    several codes shares its preamble words but is one law per code)."""
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    if _MB in a and _MB in b and a.split(_MB, 1)[1] != b.split(_MB, 1)[1]:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= min_ratio


def subject_act_pairs(records, edges, hub_max: int = 6, fuzzy_min: float = 0.90) -> tuple[list[tuple[str, str]], dict]:
    """Ruling M3-R5: documents about the same subject act share one split group. A non-stenogram document's subject
    references are
      (a) its own act key (acts.act_key(meta.title)) and its title-level refs (REFERS_TO{title, version_of,
          translation_of}) -- to a corpus document or an EXT act key alike;
      (b) the FIRST non-hub quoted act (smallest evidence start) among the REFERS_TO{act} refs of its heading vertices
          (root, ЗАГ.1, ПРЕАМБ.1, П.1): always for decisions / conclusions / resolutions (meta.type != law_text, whose
          meta.title may be unreliable), for a law text only when it has no usable title-level ref.
    An EXT key equals a corpus QONUNLAR act when the keys are identical or drifted wordings (_same_act_fuzzy).
    Never merging: generic omnibus keys (acts.is_generic), EXT sub-vertices ("/"), and hub acts -- keys referenced
    (title-level or heading-level) by more than `hub_max` documents (the Constitution, the codes).
    Returns (doc pairs, audit {fuzzy, hubs, heading_act})."""
    from graphrag.kg.acts import act_key, is_generic
    recs = {r.doc_id: r for r in records}
    docs = sorted(d for d, r in recs.items() if r.meta.section != "STENOGRAMMA")
    dset = set(docs)
    title, heads = defaultdict(set), defaultdict(list)
    for e in edges:
        if e["type"] != "REFERS_TO":
            continue
        d = e["src"].split("/", 1)[0]
        if d not in dset:
            continue
        dst = e["dst"]
        if dst.startswith("EXT/"):
            t = "K:" + dst[4:]
            if "/" in dst[4:] or is_generic(dst[4:]):
                continue
        else:
            t = dst.split("/", 1)[0]
            if t not in dset or t == d:           # stenograms never merge; self-references are no link
                continue
            t = "D:" + t
        rel = (e.get("attrs") or {}).get("relation")
        if rel in ACT_LINK_RELATIONS:
            title[d].add(t)
        elif rel == "act" and e["src"][len(d):] in HEADING_VERTICES:
            heads[d].append((int(e["attrs"]["evidence"]["start"]), t))
    self_key = {d: act_key(recs[d].meta.title or "") for d in docs}
    alias = defaultdict(set)                      # "K:<key>" -> corpus documents that ARE this act
    for d, k in self_key.items():
        if k and not is_generic(k):
            alias["K:" + k].add(d)
    fuzzy = []
    for x in sorted({t for ts in title.values() for t in ts if t.startswith("K:")}):
        if any(recs[d].meta.section == "QONUNLAR" for d in alias.get(x, ())):
            continue
        for d in docs:
            k = self_key[d]
            if recs[d].meta.section == "QONUNLAR" and k and not is_generic(k) and _same_act_fuzzy(x[2:], k, fuzzy_min):
                fuzzy.append([x[2:], d])
    for x, d in fuzzy:
        alias["K:" + x].add(d)

    def canon(t: str) -> set[str]:
        return {t} | {"D:" + d for d in alias.get(t, ())} if t.startswith("K:") else {t}

    referrers = defaultdict(set)                  # hub test: title-level or heading-level referrers of a key
    for d in docs:
        for t in title[d] | {t for _, t in heads[d]}:
            for n in canon(t):
                referrers[n].add(d)
    hubs = sorted(n for n, ds in referrers.items() if len(ds) > hub_max)
    hub = set(hubs)
    refs, heading_act = {}, {}
    for d in docs:
        own = {"D:" + d} | (canon("K:" + self_key[d]) if self_key[d] and not is_generic(self_key[d]) else set())
        tl = {n for t in title[d] for n in canon(t)} - hub
        r = (own | tl) - hub
        # a decision / conclusion / resolution is ABOUT the first act its heading quotes (its meta.title may be wrong,
        # e.g. an amended law's name); a law text uses it only when it has no usable title-level ref of its own
        if recs[d].meta.type != "law_text" or not tl:
            for _, t in sorted(heads[d]):
                c = canon(t) - hub
                if c and not (canon(t) & hub):
                    r |= c
                    heading_act[d] = t
                    break
        refs[d] = r
    members = defaultdict(set)
    for d, ns in refs.items():
        for n in ns:
            members[n].add(d)
    pairs = set()
    for n, ds in members.items():
        ds = sorted(ds)
        pairs |= {(ds[0], x) for x in ds[1:]}
    return sorted(pairs), {"fuzzy": fuzzy, "hubs": [h[2:] for h in hubs], "heading_act": dict(sorted(heading_act.items()))}


def graph_subject_act_pairs(records, version: str | None = None) -> tuple[list[tuple[str, str]], dict]:
    version = version or config.load("train")["env"]["graph_version"]
    with open(config.path(f"data/graph/{version}/edges.jsonl"), encoding="utf-8") as f:
        return subject_act_pairs(records, [json.loads(line) for line in f if line.strip()])


def graph_act_link_pairs(doc_sections: dict[str, str], version: str | None = None) -> list[tuple[str, str]]:
    version = version or config.load("train")["env"]["graph_version"]
    with open(config.path(f"data/graph/{version}/edges.jsonl"), encoding="utf-8") as f:
        return act_link_pairs((json.loads(line) for line in f if line.strip()), doc_sections)


def curated_link_pairs(links, doc_ids) -> list[tuple[str, str]]:
    """Ruling M3-R7: train.yaml split.links -- curated same-act families ({docs: [...], why: "..."}) that the automatic
    rules miss (hub / generic-titled documents). Every entry needs >= 2 known documents and a rationale."""
    known = set(doc_ids)
    out = set()
    for ln in links or []:
        ds, why = list(ln.get("docs") or []), str(ln.get("why") or "").strip()
        if len(ds) < 2 or not why:
            raise ValueError(f"split.links entry needs >= 2 docs and a 'why': {ln}")
        bad = [d for d in ds if d not in known]
        if bad:
            raise ValueError(f"split.links names unknown documents {bad}")
        ds = sorted(ds)
        out |= {(ds[0], d) for d in ds[1:]}
    return sorted(out)


def strict_doc_groups(doc_ids, pairs) -> dict[str, str]:
    """graphrag.kg.semantic.doc_groups, but a pair naming an unknown document raises (Ruling M3-R5 (6)) instead of
    being skipped. (doc_groups itself is a graph-version input -- graphrag/kg is hashed into the pinned version --
    so the check lives here rather than changing it.)"""
    from graphrag.kg.semantic import doc_groups
    ids = list(doc_ids)
    known = set(ids)
    bad = sorted({d for p in pairs for d in p if d not in known})
    if bad:
        raise ValueError(f"split pairs name {len(bad)} unknown documents: {bad[:5]}")
    return doc_groups(ids, pairs)


def assign_splits(doc_sections: dict[str, str], doc_weight: dict[str, int], pairs, seed: int,
                  ratios: dict[str, float]) -> dict[str, str]:
    """doc_id -> split for EVERY document (П3): groups (linked / near-duplicate / same-subject-act documents together),
    heaviest first (ties: sha256(seed:group) order), go to the split with the least growth of the per-section weight
    deviation from `ratios` (weight = the document's fragment count, a proxy for how many pairs it can seed).
    Heaviest-first (M3-R5) keeps every section near its ratios once groups span sections."""
    names = list(ratios)
    group = strict_doc_groups(sorted(doc_sections), pairs)
    members = defaultdict(list)
    for d, g in group.items():
        members[g].append(d)
    for g, ds in members.items():                     # the group id is its smallest doc_id (doc_groups)
        assert g in ds, f"group {g} does not contain its representative"
    order = sorted(members, key=lambda g: (-sum(doc_weight.get(d, 0) for d in members[g]),
                                           hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()))
    # A group may span sections (a law, its Senate decision and its committee conclusion; M3-R4/R5): the whole group
    # keeps one split and goes to the split whose per-section weight deviation from the ratios grows least
    # (sum over the group's sections of the increase in ((have + w) - ratio * total)^2 / total).
    gw = {g: Counter() for g in members}
    for g, ds in members.items():
        for d in ds:
            gw[g][doc_sections[d]] += doc_weight.get(d, 0)
    total, have = Counter(), Counter()
    for g in order:
        total.update(gw[g])

    def cost(g, s):
        return sum((((have[(sec, s)] + w) - ratios[s] * total[sec]) ** 2 - (have[(sec, s)] - ratios[s] * total[sec]) ** 2)
                   / total[sec] for sec, w in gw[g].items() if total[sec] > 0)
    out: dict[str, str] = {}
    for g in order:
        best = min(names, key=lambda s: (round(cost(g, s), 9), names.index(s)))
        for sec, w in gw[g].items():
            have[(sec, best)] += w
        for d in members[g]:
            out[d] = best
    return dict(sorted(out.items()))


def read_gold(path) -> list[GoldItem]:
    return [GoldItem.from_json(json.loads(line)) for line in open(path, encoding="utf-8") if line.strip()]


def write_gold(path, items: list[GoldItem]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(i.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                           for i in sorted(items, key=lambda i: i.qid)), encoding="utf-8")
    tmp.replace(p)


def gold_rows(item: GoldItem, data, use_support: bool = True) -> frozenset:
    return frozenset(data.chunk_row[c] for c in (item.gold if use_support else item.core))
