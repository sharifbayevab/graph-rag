"""ds_bridge (user decision M3-R21 (1)): agentic multi-hop. For an ANCHOR content fragment, code collects <= 8
candidate neighbour fragments over the graph, with their texts:
- refers_to: REFERS_TO whose evidence lies in the anchor -> the cited corpus element; a cited EXTERNAL element -> the
  fragments of other documents that cite / amend the same element (procedural hub elements excluded);
- amends: AMENDS / REPEALS whose evidence lies in the anchor -> the amended corpus element, or the fragments of other
  documents about the same amended external element;
- concept: the anchor's evidence REGULATES a concept -> the fragment holding the DEFINES evidence of that concept
  ("term -> definition");
- related: RELATED edges of the anchor's vertex -> the most similar eligible fragment of the related vertex.
Every neighbour is an eligible content fragment (the caller's `ok` set: content, >= min tokens, no Q-real leak) of the
SAME split as the anchor, never the anchor's own vertex. gpt-4.1 picks the one neighbour genuinely needed (or none)
and composes one question: {neighbor, needed_fact {quote, value}, anchor_fact {quote, value}, question, answer,
reason}. Code snaps / checks both quotes as exact source substrings (graphrag.dataset.facts.snap_fact), then every
M3-R3 control runs on the 2-fragment pair as on any multi-hop candidate: generate.check_item_ex (typed value checks,
chain / redundant, leak, script, copy ...), then ds_verify (blind judge + per-fact ablation, plants, agreement),
ds_quality, finalize (dedup / leak / select). Deterministic: every order is sha256(seed:key)."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from graphrag.dataset.facts import Fact, guard_total, snap_fact, value_type
from graphrag.dataset.seeds import Seed, _h
from graphrag.kg.llm import _canon_input, prompt_digest

PASS = "ds_bridge"
SOURCES = ("refers_to", "amends", "concept", "related")
MAX_NEIGHBORS = 8
TEXT_CAP_ANCHOR = 2500
TEXT_CAP_NEIGHBOR = 1500
RELATION = {"refers_to": "the anchor cites this provision (or this fragment deals with the provision the anchor cites)",
            "amends": "the anchor amends / repeals this provision (or this fragment deals with the same amended provision)",
            "concept": "this fragment defines a term the anchor regulates",
            "related": "topically related fragment"}
# prompt_version 1 (M3-R21), with the M3-R22 generator rules of the strict re-pilot reading
PROMPT_BRIDGE = """You build MULTI-HOP question-answer pairs for a legal retrieval dataset over Uzbek parliamentary documents
(laws, Senate decisions, committee conclusions, plenary transcripts).
Each ITEM gives a target LANG, an ANCHOR fragment (act, where, text) and up to 8 NEIGHBOUR fragments (id, relation,
act, where, text) linked to it in a legal graph. Decide whether ONE neighbour is GENUINELY needed together with the
anchor to answer a natural question:
- a genuine pair: the question starts from the anchor (e.g. what the anchor refers to, amends or uses) and its answer
  is a fact that only the neighbour states (the cited article's term, the definition of the term, the old / new
  wording, the body named in the cited provision). Without the neighbour the question cannot be answered; without the
  anchor the reader would not know which provision is meant.
- if no neighbour gives such a pair (only topical similarity, restatements, the same fact in both), return
  neighbor "none" and empty strings elsewhere. "none" is a good answer when true.
For a chosen neighbour return:
- anchor_fact: {quote: a verbatim span (20-300 characters) of the ANCHOR text, value: the shortest span inside the
  quote that identifies the link (e.g. "64-моддасида", the term used)}; it may appear in the question;
- needed_fact: {quote: a verbatim span (20-300 characters) of the NEIGHBOUR text, value: the shortest span inside the
  quote that answers the question; a number keeps its unit}; the question must not reveal it in any form;
- question: ONE information need, self-contained for a reader who has never seen the fragments: name the anchor's act
  by its own title (`act`; for an amending law also the amended act and article), name the concrete subject; never
  join two questions; no pointers ("ушбу / мазкур / bu hujjat / данный документ"); do not copy more than 5
  consecutive words of a text; the interrogative must fit the answer type (threshold -> "up to what amount", term ->
  "within what period", body -> "which body", manner -> "on what terms");
- answer: 1-3 sentences stating the needed value (and, if not in the question, the anchor value); nothing unsupported;
- every proper name in the question and answer must occur in the fragments or their act titles;
- script: ENTIRELY in LANG -- uz-Latn: Latin letters only (oʻ gʻ with ʻ), transliterate Cyrillic letter by letter;
  uz-Cyrl: Cyrillic only (ў ғ қ ҳ, never ʻ / ʼ or Latin letters), transliterate Latin letter by letter; ru: natural,
  grammatical Russian in Cyrillic, act titles rendered in Russian (the original title may follow once in «»),
  standard legal terms (see GLOSSARY);
- reason: one sentence: why the neighbour is needed (or why none is).
Return {"items": [{"key", "neighbor", "anchor_fact", "needed_fact", "question", "answer", "reason"}]}, one per key."""
_FACT = {"type": "object", "additionalProperties": False, "required": ["quote", "value"],
         "properties": {"quote": {"type": "string"}, "value": {"type": "string"}}}
SCHEMA_BRIDGE = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["key", "neighbor", "anchor_fact", "needed_fact", "question", "answer",
                                            "reason"],
                               "properties": {"key": {"type": "string"}, "neighbor": {"type": "string"},
                                              "anchor_fact": _FACT, "needed_fact": _FACT,
                                              "question": {"type": "string"}, "answer": {"type": "string"},
                                              "reason": {"type": "string"}}}}}}
RETRY_NOTE = "The previous answer did not return exactly one item per key; return exactly the keys given."


def _doc(v: str) -> str:
    return v.split("/", 1)[0]


def _stub(v: str) -> bool:
    return _doc(v) in ("EXT", "CON", "ENT")


class BridgeIndex:
    """Neighbour candidates of anchor fragments over the graph (see the module docstring)."""

    def __init__(self, data, records, edges, ok: set[str], split_of: dict[str, str], seed,
                 referrers: dict | None = None, max_referrer_docs: int = 6):
        from graphrag.dataset.walks import SKIP_RELATIONS, _chunk_at, is_act_level
        self.data, self.ok, self.split_of, self.seed = data, ok, split_of, seed
        recs = {r.doc_id: r for r in records}
        self.parent = {c.chunk_id: c.parent for r in records for c in r.chunks}
        corpus_docs = set(recs)
        self.hub = {k for k, n in (referrers or {}).items() if n > int(max_referrer_docs)}
        self.at: dict[str, list] = defaultdict(list)          # chunk -> edges whose evidence lies in it
        self.by_target: dict[str, set] = defaultdict(set)     # element -> chunks whose evidence edges target it
        self.defines: dict[str, set] = defaultdict(set)       # concept -> chunks holding a DEFINES evidence
        self.related: dict[str, set] = defaultdict(set)       # vertex -> RELATED vertices
        self.incoming: dict[str, list] = defaultdict(list)    # corpus element -> (chunk, type) of AMENDS / REFERS_TO to it
        for e in sorted(edges, key=lambda e: (e.src, e.type, e.dst)):
            if e.type == "RELATED":
                self.related[e.src].add(e.dst)
                self.related[e.dst].add(e.src)
                continue
            if e.type not in ("REFERS_TO", "AMENDS", "REPEALS", "REGULATES", "DEFINES"):
                continue
            if e.type == "REFERS_TO" and (e.attrs or {}).get("relation") in SKIP_RELATIONS:
                continue
            ev = (e.attrs or {}).get("evidence") or {}
            if ev.get("doc_id") not in recs:
                continue
            c = _chunk_at(recs[ev["doc_id"]], ev["start"], ev["end"])
            if c is None:
                continue
            self.at[c].append((e.src, e.type, e.dst))
            if e.type == "DEFINES":
                self.defines[e.dst].add(c)
            elif e.type != "REGULATES" and not is_act_level(e.dst, corpus_docs):
                self.by_target[e.dst].add(c)
        self._act_level = lambda v: is_act_level(v, corpus_docs)
        self.recs = recs

    def _split(self, c: str) -> str | None:
        return self.split_of.get(_doc(c))

    def _sim(self, a: str, b: str) -> float:
        d = self.data
        return round(float(d.vecs64[d.chunk_row[b]] @ d.vecs64[d.chunk_row[a]]), 6)

    def _eligible(self, anchor: str, c: str) -> bool:
        return (c != anchor and c in self.ok and self._split(c) == self._split(anchor)
                and self.parent.get(c) != self.parent.get(anchor))

    def _best_chunk(self, v: str, anchor: str) -> str | None:
        d = self.data
        row = d.node_row.get(v)
        if row is None or _doc(v) not in self.recs:
            return None
        cand = sorted(d.chunk_ids[i] for i in d.mu_inv(row) if self._eligible(anchor, d.chunk_ids[i]))
        return min(cand, key=lambda x: (-self._sim(anchor, x), x)) if cand else None

    def candidates(self, anchor: str) -> dict[str, list[tuple[str, tuple]]]:
        """source -> [(neighbour chunk, hops)] (similarity order, one entry per chunk and source)."""
        out: dict[str, dict[str, tuple]] = {s: {} for s in SOURCES}

        def add(src, c, hops):
            if c and self._eligible(anchor, c) and c not in out[src]:
                out[src][c] = tuple(hops)
        for s, t, dst in self.at.get(anchor, ()):
            if t in ("REFERS_TO", "AMENDS", "REPEALS"):
                src = "refers_to" if t == "REFERS_TO" else "amends"
                if self._act_level(dst) or dst in self.hub:
                    continue
                if not _stub(dst):
                    add(src, self._best_chunk(dst, anchor), [(s, t, dst)])
                for c in sorted(self.by_target.get(dst, ())):     # other documents on the same element
                    if _doc(c) != _doc(anchor):
                        for s2, t2, d2 in self.at.get(c, ()):
                            if d2 == dst and t2 in ("REFERS_TO", "AMENDS", "REPEALS"):
                                add(src, c, [(s, t, dst), (s2, t2, dst)])
                                break
            elif t == "REGULATES" and dst.startswith("CON/"):
                for c in sorted(self.defines.get(dst, ())):
                    d_src = next((a for a, tt, b in self.at.get(c, ()) if tt == "DEFINES" and b == dst), None)
                    add("concept", c, [(s, t, dst), (d_src or self.parent[c], "DEFINES", dst)])
        v = self.parent.get(anchor)
        for u in sorted(self.related.get(v, ())):
            add("related", self._best_chunk(u, anchor), [(v, "RELATED", u)])
        return {s: sorted(m.items(), key=lambda x: (-self._sim(anchor, x[0]), x[0])) for s, m in out.items() if m}


def shortlist(cands: dict, first: str | None = None, k: int = MAX_NEIGHBORS) -> list[tuple[str, str, tuple]]:
    """<= k (source, chunk, hops): sources round-robin (the anchor's stratum `first`, then SOURCES order), each
    source in its similarity order; a chunk appears once (first source wins)."""
    order = ([first] if first in cands else []) + [s for s in SOURCES if s in cands and s != first]
    pos = {s: 0 for s in order}
    out, seen = [], set()
    while len(out) < k and any(pos[s] < len(cands[s]) for s in order):
        for s in order:
            while pos[s] < len(cands[s]) and cands[s][pos[s]][0] in seen:
                pos[s] += 1
            if pos[s] < len(cands[s]) and len(out) < k:
                c, hops = cands[s][pos[s]]
                out.append((s, c, hops))
                seen.add(c)
                pos[s] += 1
    return out


def sample_anchors(index: BridgeIndex, anchors, n: int, langs, seed, exclude=(), per_doc: int = 2,
                   sources=SOURCES) -> list[dict]:
    """M3-R21: `n` anchors stratified by bridge source (cell_counts over the sources that have anchors, equal
    shares), each stratum's anchors in sha256(seed:bridge|src|chunk) order; an anchor is used once, a document at
    most `per_doc` times; languages round-robin (config order) over the picks. Returns [{anchor, stratum, lang,
    split, candidates}]."""
    from graphrag.dataset.seeds import cell_counts
    by_src: dict[str, list] = defaultdict(list)
    cache = {}
    for a in sorted(set(anchors) - set(exclude)):
        cs = index.candidates(a)
        if cs:
            cache[a] = cs
            for s in cs:
                by_src[s].append(a)
    have = [s for s in sources if by_src.get(s)]
    want = cell_counts(n, {s: 1.0 for s in have})
    used, docs, out = set(), Counter(), []
    langs = list(langs)
    for s in have:
        got = 0
        for a in sorted(by_src[s], key=lambda a: _h(seed, f"bridge|{s}|{a}")):
            if got >= want[s]:
                break
            if a in used or docs[_doc(a)] >= per_doc:
                continue
            used.add(a)
            docs[_doc(a)] += 1
            out.append({"anchor": a, "stratum": s, "split": index._split(a), "candidates": cache[a]})
            got += 1
    for i, x in enumerate(out):
        x["lang"] = langs[i % len(langs)]
    return out


def _frag(chunks_by_id, index, cid, label, cap):
    c = chunks_by_id[cid]
    return {"id": label, "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent),
            "text": c.text[:cap]}


def bridge(picks: list[dict], chunks_by_id, index, llm, cfg: dict, doc_section: dict, glossary: dict | None = None,
           batch: int = 3, amended=None) -> tuple[list, dict]:
    """ds_bridge over `picks` (sample_anchors) in their order, `batch` anchors per call; one retry with a distinct
    cached input on a key mismatch. Returns (generate.Candidate list, stats) -- per stratum: anchors, chosen
    (a neighbour, not "none"), generate (every code check passed)."""
    from graphrag.dataset.generate import LANG_NAME, Candidate, cell_of, check_item_ex, exempt_words
    from graphrag.dataset.walks import walk_class
    digest = prompt_digest(PROMPT_BRIDGE, SCHEMA_BRIDGE)
    st = {s: Counter() for s in SOURCES}
    why = Counter()
    chosen_src = Counter()
    out = []
    for b0 in range(0, len(picks), int(batch)):
        grp = picks[b0:b0 + int(batch)]
        items, meta = [], {}
        for j, p in enumerate(grp):
            short = shortlist(p["candidates"], p["stratum"])
            nb = {f"N{i + 1}": (s, c, hops) for i, (s, c, hops) in enumerate(short)}
            key = f"k{j + 1}"
            it = {"key": key, "lang": p["lang"], "LANG": LANG_NAME[p["lang"]],
                  "anchor": _frag(chunks_by_id, index, p["anchor"], "A", TEXT_CAP_ANCHOR),
                  "neighbors": [dict(_frag(chunks_by_id, index, c, n, TEXT_CAP_NEIGHBOR), relation=RELATION[s])
                                for n, (s, c, _) in nb.items()]}
            if glossary:
                it["GLOSSARY"] = glossary
            items.append(it)
            meta[key] = (p, nb)
            st[p["stratum"]]["anchors"] += 1
        ans = ck = None
        for inp in ({"items": items}, {"items": items, "retry": RETRY_NOTE}):
            guard_total(llm, PASS, inp, PROMPT_BRIDGE, SCHEMA_BRIDGE)
            ans = llm.cached_json(PASS, inp, SCHEMA_BRIDGE,
                                  lambda a: (sorted(x["key"] for x in a["items"]) == sorted(meta), "keys"),
                                  prompt=PROMPT_BRIDGE)
            ck = llm.cache_key(PASS, _canon_input(inp), digest)
            if ans is not None:
                break
        if ans is None:
            for p, _ in meta.values():
                why[f"{p['stratum']}|bad_answer"] += 1
            continue
        for it in sorted(ans["items"], key=lambda x: x["key"]):
            p, nb = meta[it["key"]]
            s0 = p["stratum"]

            def rej(r):
                why[f"{s0}|{r}"] += 1
            if it["neighbor"] not in nb:
                rej("none" if it["neighbor"].strip().lower() in ("none", "") else "unknown_neighbor")
                continue
            src, nc, hops = nb[it["neighbor"]]
            st[s0]["chosen"] += 1
            chosen_src[f"{s0}->{src}"] += 1
            a = p["anchor"]
            fa = _frag(chunks_by_id, index, a, "F1", 10 ** 9)
            fn = _frag(chunks_by_id, index, nc, "F2", 10 ** 9)
            ga, ra = snap_fact({"statement": "", **it["anchor_fact"]}, chunks_by_id[a].text, exempt_words([fa]))
            gn, rn = snap_fact({"statement": "", **it["needed_fact"]}, chunks_by_id[nc].text, exempt_words([fn]))
            if ra or rn:
                rej(f"{'anchor' if ra else 'needed'}_{ra or rn}")
                continue
            docs = tuple(dict.fromkeys((_doc(a), _doc(nc))))
            chunks = (a, nc)
            seed = Seed(f"multi-{_h(cfg['seed'], '|'.join(chunks))[:10]}", "multi", p["lang"], chunks, docs,
                        p["split"], doc_section[docs[0]], tuple(tuple(h) for h in hops))
            f1 = Fact(f"{a}@bridge-{seed.seed_id}-0", a, "", ga["value"].strip(), ga["quote"], PASS, ck,
                      value_type(ga["value"]))
            f2 = Fact(f"{nc}@bridge-{seed.seed_id}-1", nc, "", gn["value"].strip(), gn["quote"], PASS, ck,
                      value_type(gn["value"]))
            frags = [fa, fn]
            if amended is not None:
                for f in frags:
                    f["amended"] = amended(f["id"] == "F1" and a or nc)
            item = {"key": it["key"], "question": it["question"], "answer": it["answer"], "used": ["F1.0", "F2.0"]}
            r, info = check_item_ex(item, seed, {"F1.0": f1, "F2.0": f2}, frags, cfg, path_chunks={a})
            if r:
                rej(r)
                continue
            if not any(f.chunk_id == nc for f in info["answer"]):
                rej("needed_not_in_answer")
                continue
            st[s0]["generate"] += 1
            used = list(dict.fromkeys(info["answer"] + info["path"]))
            out.append(Candidate(seed, it["question"].strip(), it["answer"].strip(), used, {
                "seed_id": seed.seed_id, "cell": cell_of(seed), "chunks": list(chunks), "walk": [list(h) for h in hops],
                "fact_ids": [f.fact_id for f in used], "answer_fact_ids": [f.fact_id for f in info["answer"]],
                "path_fact_ids": [f.fact_id for f in info["path"]], "value_checks": dict(info["status"]),
                "walk_class": walk_class(hops), "bridge_source": src, "stratum": s0,
                "facts": {"pass": PASS, "model": llm.model, "prompt_digest": digest,
                          "cache_keys": {"ds_bridge": ck}},
                "bridge": {"pass": PASS, "model": llm.model, "prompt_version": llm.passes[PASS].get("prompt_version"),
                           "prompt_digest": digest, "cache_key": ck, "item": it["key"], "reason": it["reason"],
                           "neighbor": nc, "shortlist": [[n, s, c] for n, (s, c, _) in nb.items()]},
                "generate": {"pass": PASS, "model": llm.model, "prompt_digest": digest, "cache_key": ck,
                             "item": it["key"]}}))
    return out, {"by_stratum": {s: dict(v) for s, v in st.items() if v}, "rejected": dict(sorted(why.items())),
                 "chosen_source": dict(sorted(chosen_src.items())), "batch": int(batch), "candidates": len(out)}
