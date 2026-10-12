"""M7 Q3 (compare): document pairs linked in the pinned graph, generator packets, code checks and ingest
(docs/m7/2026-10-02-hard-questions-design.md §2 Q3 and «Общие правила»; plan M7 part 1, Task 6).

Pairs (`pick_doc_pairs`). Internal documents by attrs.doc_type; four kinds, the tuple oriented as the kind name says:
  law_conclusion      (law_text, committee_conclusion)   -- a direct link: REFERS_TO / RELATED between the two
  law_resolution      (law_text, resolution)             -- documents (or their elements); a shared entity alone
  stenogram_law       (stenogram, law_text)              -- does not make a document «about» the law;
  laws_shared_entity  (law_text, law_text)               -- a direct link or a shared REGULATES entity / concept
                                                            regulated by at most DF_MAX documents (rarer = stronger).
R23: a pair linked by REFERS_TO relation translation_of / version_of is a twin and is dropped; inside such a family a
conclusion / resolution pairs only with the version of its own date (none -> dropped). R24: stenogram_law needs a
REFERS_TO (agenda / act) from the sitting to the law, the sitting not dated before the law, and a linked law element
that owns a chunk. attrs.doc_type `postanovlenie` counts as resolution.
Score = 2 x REFERS_TO + Σ RELATED weight + Σ 1 / (df - 1) over the shared entities + SAME_TITLE_BONUS when a law and
its conclusion / resolution carry the same title (the conclusion / resolution on that law). Dropped: documents with
fewer than MIN_CHUNKS chunks; documents whose name is ambiguous ((doc_type, folded title, date) shared by another
internal document -- the question could not say which one it means); pairs of two laws with the same folded title
(translated twins: nothing to compare). Selection: per kind, by score (ties by sha1(seed, pair)), round-robin over
KINDS so the kinds are even as far as supply allows, each document in at most MAX_PER_DOC pairs and in at most
MAX_PER_DOC_KIND pair of one kind (spread: no single law / stenogram / title family fills a kind). The order does not
depend on n, so a larger n extends a smaller one (the packet cap of 60 extends the first 40).

Packets (`build_packets`): per pair the two documents' titles / types / dates and chunk texts with ids, at most
PACKET_TOKENS corpus tokens per packet. Chunks are chosen deterministically (`select_chunks`): first those of the
elements the linking edges touch (μ⁻¹ of the element; strongest first), then the leading chunks in document order;
each document gets half the budget, a short document's leftover goes to the other. Only shown chunks may be gold.
Languages: per kind, the running largest-deficit rule over the old-set shares (prefix-stable).

Checks (`check_compare`, brief): 2 <= |gold| <= 5, no duplicates; every id exists; the gold documents are exactly the
two documents of the pair; leak_ok against the gold texts (with ids). `ingest_compare` adds: a known packet, at most
one item per packet, qtype / template / docs / lang as the packet says, a script check of the question
(`script_ok`), a non-empty question and answer, gold within the shown chunks; refusal lines are counted apart."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from grl.hard.checks import leak_ok
from grl.hard.templates import HardItem, title_key

KINDS = ("law_conclusion", "law_resolution", "laws_shared_entity", "stenogram_law")
KIND_TYPES = {"law_conclusion": ("law_text", "committee_conclusion"), "law_resolution": ("law_text", "resolution"),
              "laws_shared_entity": ("law_text", "law_text"), "stenogram_law": ("stenogram", "law_text")}
DIRECT_ONLY = {"law_conclusion", "law_resolution", "stenogram_law"}
DOC_TYPE_ALIASES = {"postanovlenie": "resolution"}
TWIN_RELATIONS = ("translation_of", "version_of")       # R23: REFERS_TO attrs.relation between two documents
STENOGRAM_RELATIONS = ("agenda", "act")                  # R24: the sitting's agenda item / act mention of the law
MAX_PER_DOC, MAX_PER_DOC_KIND, MIN_CHUNKS, DF_MAX = 2, 1, 2, 8
SAME_TITLE_BONUS = 3.0
PACKET_TOKENS = 12000
MIN_GOLD, MAX_GOLD = 2, 5


# --- R26 (plan M7 part 2, Task 6): approval-only resolutions -------------------------------------------------------

NON_CONTENT = ("ПРЕАМБ", "РЕКВ", "ЗАГ")          # preamble, requisites / signature, heading: never a resolution point
BOILERPLATE = re.compile(
    r"маъқуллан|ma[ʼʻ'‘’`]?qullan|одобр|тасдиқлан|tasdiqlan|утверд"            # approve / endorse the law
    r"|кучга\s+кир|kuchga\s+kir|вступает\s+в\s+силу"                          # this resolution enters into force
    r"|имзола[шн]|imzola[sn]|на\s+подпись", re.IGNORECASE)                     # forwarded for signature
MIN_POINT_WORDS = 6                              # a shorter «point» is a split signature line («Сенатининг Раиси»)
MAX_BOILER_WORDS = 25                            # an approve / in-force point, the quoted title removed
_TITLE_Q = re.compile(r"[“«\"„][^”»\"“]*[”»\"“]")


def _element_kind(chunk_id: str) -> str:
    rest = chunk_id.split("/", 1)[1] if "/" in chunk_id else chunk_id
    return re.split(r"[.#/]", rest, maxsplit=1)[0]


def substantive_points(chunks) -> list[str]:
    """R26: the chunk ids of a resolution's content points other than the approval template. `chunks`: [(chunk_id,
    text)] of the whole document. A chunk of the preamble / requisites / heading never counts; a point counts unless
    it is shorter than MIN_POINT_WORDS words, or it says approve / enters into force / forwarded for signature and
    has at most MAX_BOILER_WORDS words outside its quoted title."""
    out = []
    for cid, text in chunks:
        if _element_kind(cid) in NON_CONTENT:
            continue
        words = re.findall(r"\w+", text or "")
        if len(words) < MIN_POINT_WORDS:
            continue
        bare = re.findall(r"\w+", _TITLE_Q.sub(" ", text))
        if BOILERPLATE.search(text) and len(bare) <= MAX_BOILER_WORDS:
            continue
        out.append(cid)
    return out


def approval_only(chunks) -> bool:
    """R26: a resolution whose points only approve the law (+ entry into force / signature). No text -> False (the
    filter never drops a pair on missing data)."""
    chunks = list(chunks)
    return bool(chunks) and not substantive_points(chunks)


# --- the graph index --------------------------------------------------------------------------------------------

def _index(nodes, edges):
    by_id = {n["node_id"]: n for n in nodes}
    docs = {n["node_id"]: n for n in nodes if n["kind"] == "document" and not n.get("external")
            and (n.get("attrs") or {}).get("doc_type")}
    node_doc = {n["node_id"]: n["doc_id"] for n in nodes if n.get("doc_id") in docs}
    direct = defaultdict(list)          # (a, b) sorted -> [(type, elem_a | None, elem_b | None, weight, relation)]
    twins = set()                       # (a, b) sorted: translation_of / version_of (R23)
    parent: dict[str, str] = {}         # R23 families: union-find over the twin relations

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    reg = defaultdict(lambda: defaultdict(set))     # entity -> doc -> its elements regulating it
    for e in edges:
        t = e["type"]
        if t not in ("REFERS_TO", "RELATED", "REGULATES"):
            continue
        a = node_doc.get(e["src"])
        if a is None:
            continue
        if t == "REGULATES":
            if e["src"] != a:
                reg[e["dst"]][a].add(e["src"])
            continue
        b = node_doc.get(e["dst"])
        if b is None or b == a:
            continue
        ea = e["src"] if e["src"] != a else None
        eb = e["dst"] if e["dst"] != b else None
        w = float((e.get("attrs") or {}).get("weight") or 0.0) if t == "RELATED" else 1.0
        rel = (e.get("attrs") or {}).get("relation") if t == "REFERS_TO" else None
        src_side = "a" if a < b else "b"         # which sorted side the edge starts on
        key, rec = ((a, b), (t, ea, eb, w, rel, src_side)) if a < b else ((b, a), (t, eb, ea, w, rel, src_side))
        direct[key].append(rec)
        if rel in TWIN_RELATIONS:
            twins.add(key)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    shared = defaultdict(list)          # (a, b) sorted -> [(entity, df)]
    for ent in sorted(reg):
        ds = sorted(reg[ent])
        if not 2 <= len(ds) <= DF_MAX:
            continue
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                shared[(a, b)].append((ent, len(ds)))
    family = defaultdict(set)
    for x in list(parent):
        family[find(x)].add(x)
    fam = {x: sorted(family[find(x)]) for x in parent}
    return {"by_id": by_id, "docs": docs, "direct": direct, "reg": reg, "shared": shared, "twins": twins,
            "family": fam}


def doc_type(node) -> str:
    t = (node.get("attrs") or {}).get("doc_type")
    return DOC_TYPE_ALIASES.get(t, t)


def _dtype(ix, d) -> str:
    return doc_type(ix["docs"][d])


def _kind(ix, a, b) -> str | None:
    ta, tb = _dtype(ix, a), _dtype(ix, b)
    for k, (x, y) in KIND_TYPES.items():
        if (ta, tb) in ((x, y), (y, x)):
            return k
    return None


def _orient(ix, a, b, kind) -> tuple[str, str]:
    first = KIND_TYPES[kind][0]
    if kind == "laws_shared_entity":
        return (a, b) if a < b else (b, a)
    return (a, b) if _dtype(ix, a) == first else (b, a)


def _pair_info(ix, a, b, kind) -> dict:
    """Score and the linked elements of each side (element -> weight) of the pair (a, b) (any order)."""
    key = (a, b) if a < b else (b, a)
    elems = {key[0]: Counter(), key[1]: Counter()}
    refers = related = 0
    wsum = 0.0
    for t, ea, eb, w, *_ in ix["direct"].get(key, ()):
        if t == "REFERS_TO":
            refers += 1
        else:
            related += 1
            wsum += w
        for d, el in ((key[0], ea), (key[1], eb)):
            if el:
                elems[d][el] += w
    shared, ssum = [], 0.0
    for ent, df in ix["shared"].get(key, ()):
        shared.append(ent)
        ssum += 1.0 / (df - 1)
        for d in key:
            for el in ix["reg"][ent][d]:
                elems[d][el] += 1.0 / (df - 1)
    same = (kind != "laws_shared_entity"
            and title_key(ix["docs"][a]["title"]) == title_key(ix["docs"][b]["title"]))
    score = 2.0 * refers + wsum + ssum + (SAME_TITLE_BONUS if same else 0.0)
    return {"score": round(score, 6), "refers_to": refers, "related": related, "shared": shared,
            "same_title": same, "elems": {d: dict(sorted(c.items())) for d, c in elems.items()}}


def _tiebreak(seed: int, a: str, b: str) -> str:
    return hashlib.sha1(f"{seed}|{a}|{b}".encode("utf-8")).hexdigest()


def _r24(ix, s, law, info, dates: dict, mu_inv: dict | None) -> str | None:
    """R24 for a stenogram_law pair: a REFERS_TO (agenda / act) from the sitting to the law, the sitting not dated
    before the law, and a linked law element that owns a chunk (μ⁻¹; any linked element without mu_inv)."""
    key = (s, law) if s < law else (law, s)
    s_side = "a" if key[0] == s else "b"
    if not any(t == "REFERS_TO" and rel in STENOGRAM_RELATIONS and side == s_side
               for t, _, _, _, rel, side in ix["direct"].get(key, ())):
        return "sten_no_refers"
    ds, dl = dates.get(s), dates.get(law)
    if ds and dl and ds < dl:
        return "sten_before_law"
    els = info["elems"][law]
    linked = any(mu_inv.get(el) for el in els) if mu_inv is not None else bool(els)
    if not linked:
        return "sten_law_unlinked"
    return None


def _r23_family(ix, law, other, dates: dict) -> str | None:
    """R23: inside a version / translation family a conclusion or resolution pairs only with the version of its date."""
    fam = ix["family"].get(law)
    if not fam or len(fam) < 2:
        return None
    return None if dates.get(other) and dates.get(other) == dates.get(law) else "version_date_mismatch"


def pair_candidates(nodes, edges, rng_seed: int, *, n_chunks: dict | None = None, doc_dates: dict | None = None,
                    stats: Counter | None = None, mu_inv: dict | None = None,
                    why: dict | None = None, res_texts: dict | None = None) -> dict[str, list[tuple[str, str, dict]]]:
    """kind -> [(doc_a, doc_b, info)] sorted by (-score, sha1 tiebreak): every eligible pair (see module doc).
    `why` (out): sorted (a, b) -> the reason a linked pair is not eligible. `res_texts` (R26, part 2): doc ->
    [(chunk_id, text)] of the whole document; a law_resolution pair whose resolution is approval_only() is dropped
    (`approval_only`). None (part 1): no R26 filter."""
    st = stats if stats is not None else Counter()
    why = why if why is not None else {}
    dates = doc_dates or {}
    ix = _index(nodes, edges)
    ok = set()
    names = defaultdict(list)
    for d, n in sorted(ix["docs"].items()):
        if not (n.get("title") or "").strip():
            st["empty_title_docs"] += 1
            continue
        if n_chunks is not None and n_chunks.get(d, 0) < MIN_CHUNKS:
            st["short_docs"] += 1
            continue
        names[(doc_type(n), title_key(n["title"]), dates.get(d))].append(d)
    for ds in names.values():
        if len(ds) == 1:
            ok.add(ds[0])
        else:
            st["ambiguous_name_docs"] += len(ds)
    out = {k: [] for k in KINDS}
    for key in sorted(set(ix["direct"]) | set(ix["shared"])):
        a, b = key
        if key in ix["twins"]:
            why[key] = "twin_relation"                                  # R23 (reported before naming)
            st["twin_relation"] += 1
            continue
        if a not in ok or b not in ok:
            why[key] = "ambiguous_name"
            continue
        kind = _kind(ix, a, b)
        if kind is None:
            why[key] = "no_kind"
            continue
        reason = None
        if kind in DIRECT_ONLY and key not in ix["direct"]:
            reason = f"no_direct_link:{kind}"
        elif kind == "laws_shared_entity" and title_key(ix["docs"][a]["title"]) == title_key(ix["docs"][b]["title"]):
            reason = "twin_pairs"
        x, y = _orient(ix, a, b, kind)
        info = None
        if reason is None and kind in ("law_conclusion", "law_resolution"):
            reason = _r23_family(ix, x, y, dates)
        if reason is None and kind == "law_resolution" and res_texts is not None and approval_only(res_texts.get(y, ())):
            reason = "approval_only"                                    # R26 (part 2)
        if reason is None:
            info = _pair_info(ix, x, y, kind)
            if kind == "stenogram_law":
                reason = _r24(ix, x, y, info, dates, mu_inv)
        if reason:
            why[key] = reason
            st[reason] += 1
            continue
        out[kind].append((x, y, info))
    for k in KINDS:
        out[k].sort(key=lambda p: (-p[2]["score"], _tiebreak(rng_seed, p[0], p[1])))
        st[f"supply:{k}"] = len(out[k])
    return out


def pick_doc_pairs(nodes, edges, n: int, rng_seed: int, *, n_chunks: dict | None = None,
                   doc_dates: dict | None = None, stats: Counter | None = None, mu_inv: dict | None = None,
                   preset=(), exclude=(), quota: dict | None = None, exclude_pairs=(), res_texts: dict | None = None,
                   max_per_doc: int = MAX_PER_DOC, max_per_doc_kind: int = MAX_PER_DOC_KIND,
                   fixed_splits: dict | None = None) -> list[tuple[str, str, str]]:
    """(doc_a, doc_b, kind) x <= n: round-robin over KINDS, best score first, each document in <= max_per_doc pairs
    (and <= max_per_doc_kind per kind). R25 top-up: `preset` pairs (kept packets) count toward the caps and are not
    returned, `exclude` pairs (every existing packet) are never returned, `quota` caps the new pairs per kind.
    Part 2: `exclude_pairs` -- document pairs (any order, any kind; the Q3 probe) never returned and not counted
    toward the caps (st["probe_excluded:<kind>"]); `res_texts` -- the R26 filter (pair_candidates).
    `fixed_splits` (R53 pre-screen): doc -> split already fixed (old set + Q1 / Q2); a pair whose two documents have
    different fixed splits is skipped before selection (st["r32_prescreen:<kind>"]): R32 would drop its item."""
    st = stats if stats is not None else Counter()
    cands = pair_candidates(nodes, edges, rng_seed, n_chunks=n_chunks, doc_dates=doc_dates, stats=st, mu_inv=mu_inv,
                            res_texts=res_texts)
    if fixed_splits:
        for k in KINDS:
            keep = [c for c in cands[k] if not (c[0] in fixed_splits and c[1] in fixed_splits
                                                and fixed_splits[c[0]] != fixed_splits[c[1]])]
            st[f"r32_prescreen:{k}"] = len(cands[k]) - len(keep)
            cands[k] = keep
    banned = {frozenset(p) for p in exclude_pairs}
    if banned:
        for k in KINDS:
            keep = [c for c in cands[k] if frozenset(c[:2]) not in banned]
            st[f"probe_excluded:{k}"] = len(cands[k]) - len(keep)
            cands[k] = keep
    ptr = {k: 0 for k in KINDS}
    used, used_k = Counter(), Counter()
    for a, b, k in preset:
        for d in (a, b):
            used[d] += 1
            used_k[(k, d)] += 1
    skip = {(a, b, k) for a, b, k in list(preset) + list(exclude)}
    out = []
    while len(out) < n:
        progressed = False
        for k in KINDS:
            if len(out) >= n:
                break
            if quota is not None and sum(1 for p in out if p[2] == k) >= quota.get(k, 0):
                continue
            cs = cands[k]
            while ptr[k] < len(cs):
                a, b, _ = cs[ptr[k]]
                ptr[k] += 1
                if (a, b, k) in skip:
                    continue
                if (used[a] < max_per_doc and used[b] < max_per_doc
                        and used_k[(k, a)] < max_per_doc_kind and used_k[(k, b)] < max_per_doc_kind):
                    used[a] += 1
                    used[b] += 1
                    used_k[(k, a)] += 1
                    used_k[(k, b)] += 1
                    out.append((a, b, k))
                    progressed = True
                    break
                st[f"cap_skipped:{k}"] += 1
        if not progressed:
            break
    for k in KINDS:
        st[f"picked:{k}"] = sum(1 for p in out if p[2] == k)
    return out


# --- packets -----------------------------------------------------------------------------------------------------

def select_chunks(elem_weights: dict[str, dict[str, float]], doc_chunks: dict[str, list[tuple[str, int]]],
                  mu_inv: dict[str, list[str]], budget: int = PACKET_TOKENS):
    """(shown, linked): doc -> chunk ids in document order. Linked chunks (μ⁻¹ of the linked elements, strongest
    first) are taken before the leading chunks; each document gets budget // 2 tokens, a short one's rest goes to the
    other document."""
    docs = list(elem_weights)
    need = {d: sum(t for _, t in doc_chunks[d]) for d in docs}
    half = budget // len(docs)
    cap = {d: min(need[d], half) for d in docs}
    rest = budget - sum(cap.values())
    for d in docs:
        if need[d] > cap[d]:
            add = min(rest, need[d] - cap[d])
            cap[d] += add
            rest -= add
    shown, linked = {}, {}
    for d in docs:
        pos = {c: i for i, (c, _) in enumerate(doc_chunks[d])}
        tok = dict(doc_chunks[d])
        w = Counter()
        for el, x in elem_weights[d].items():
            for c in mu_inv.get(el, ()):
                if c in pos:
                    w[c] += x
        order = sorted(w, key=lambda c: (-w[c], pos[c])) + [c for c, _ in doc_chunks[d] if c not in w]
        take, used = [], 0
        for c in order:
            if used + tok[c] <= cap[d]:
                take.append(c)
                used += tok[c]
        shown[d] = sorted(take, key=pos.get)
        linked[d] = [c for c in shown[d] if c in w]
    return shown, linked


def assign_langs(pairs, shares: dict, prior=()) -> list[str]:
    """Per kind, the j-th pair gets the language with the largest deficit j * share - count (ties by name).
    `prior` [(kind, lang)]: already assigned packets (R25 top-up continues their running counts)."""
    tot = float(sum(shares.values()))
    frac = {k: v / tot for k, v in sorted(shares.items()) if v > 0}
    seen, cnt, out = Counter(), defaultdict(Counter), []
    for k, lang in prior:
        seen[k] += 1
        cnt[k][lang] += 1
    for _, _, k in pairs:
        seen[k] += 1
        j = seen[k]
        lang = max(sorted(frac), key=lambda l: (j * frac[l] - cnt[k][l], -sorted(frac).index(l)))
        cnt[k][lang] += 1
        out.append(lang)
    return out


def packet_qid(kind: str, a: str, b: str, lang: str) -> str:
    return "m7-compare-" + hashlib.sha1(f"{kind}|{a}|{b}|{lang}".encode("utf-8")).hexdigest()[:12]


def build_packets(pairs, nodes, edges, doc_chunks: dict, mu_inv: dict, chunk_texts: dict, doc_meta: dict,
                  shares: dict, budget: int = PACKET_TOKENS, start: int = 0, prior=()) -> list[dict]:
    """One generator packet per pair (module doc). doc_chunks: doc -> [(chunk_id, tokens)] in document order;
    doc_meta: doc -> record meta (date, lang, title). R25: packet ids from cmp-<start>; `prior` as in assign_langs;
    every packet records its `pair` key [doc_a, doc_b, kind]."""
    ix = _index(nodes, edges)
    langs = assign_langs(pairs, shares, prior)
    out = []
    for i, ((a, b, kind), lang) in enumerate(zip(pairs, langs), start):
        info = _pair_info(ix, a, b, kind)
        shown, linked = select_chunks({a: info["elems"][a], b: info["elems"][b]}, doc_chunks, mu_inv, budget)
        tok = {c: t for d in (a, b) for c, t in doc_chunks[d]}
        docs = []
        for d in (a, b):
            n, m = ix["docs"][d], doc_meta.get(d) or {}
            full = re.sub(r"\s+", " ", m.get("title") or "").strip()
            docs.append({"doc_id": d, "title": n["title"], "full_title": full if full and full != n["title"] else None,
                         "doc_type": doc_type(n), "date": m.get("date"), "source_lang": m.get("lang"),
                         "n_chunks_total": len(doc_chunks[d]), "n_chunks_shown": len(shown[d]),
                         "linked_chunks": linked[d],
                         "chunks": [{"chunk_id": c, "text": chunk_texts[c], "linked": c in linked[d]}
                                    for c in shown[d]]})
        ents = [ix["by_id"][e]["title"] if e in ix["by_id"] else e for e in info["shared"]]
        out.append({"packet_id": f"cmp-{i:03d}", "qid": packet_qid(kind, a, b, lang), "kind": kind, "lang": lang,
                    "pair": [a, b, kind],
                    "link": {"refers_to": info["refers_to"], "related": info["related"], "shared_entities": ents[:12],
                             "same_title": info["same_title"], "score": info["score"]},
                    "tokens": sum(tok[c] for d in (a, b) for c in shown[d]), "docs": docs})
    return out


# --- checks ------------------------------------------------------------------------------------------------------

def check_compare(item: HardItem, chunk_doc: dict[str, str], chunk_text: dict[str, str]) -> list[str]:
    """Reasons to refuse a Q3 item ([] = OK): gold_size, gold_duplicate, unknown_id, gold_docs, leak."""
    why = []
    gold = list(item.gold)
    if not MIN_GOLD <= len(gold) <= MAX_GOLD:
        why.append("gold_size")
    if len(set(gold)) != len(gold):
        why.append("gold_duplicate")
    known = [g for g in gold if g in chunk_doc]
    if len(known) != len(gold):
        why.append("unknown_id")
    gdocs = {chunk_doc[g] for g in known}
    pair = set(item.docs)
    if len(pair) != 2 or len(gdocs) < 2 or gdocs != pair:
        why.append("gold_docs")
    if not leak_ok(item.question, {g: chunk_text.get(g, "") for g in set(gold)}):
        why.append("leak")
    return why


_QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”|„[^“”]*[“”]")
_CYR = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")
_LAT = re.compile(r"[A-Za-z]")
_UZ_CYR = re.compile(r"[ЎўҚқҒғҲҳ]")
_RU_ONLY = re.compile(r"[ЫыЩщ]")


def script_ok(question: str, lang: str) -> bool:
    """Heuristic script / language check outside quoted spans (a quoted title may keep its own script):
    uz-Latn -- Latin letters outnumber Cyrillic; ru -- Cyrillic majority, no ў қ ғ ҳ; uz-Cyrl -- Cyrillic majority,
    no ы / щ, and at least one ў қ ғ ҳ -- all outside «» (a quoted Uzbek title does not make a question Uzbek)."""
    bare = _QUOTED.sub(" ", question)
    cyr, lat = len(_CYR.findall(bare)), len(_LAT.findall(bare))
    if lang == "uz-Latn":
        return lat > cyr
    if cyr <= lat:
        return False
    if lang == "ru":
        return not _UZ_CYR.search(bare)
    if lang == "uz-Cyrl":
        return not _RU_ONLY.search(bare) and bool(_UZ_CYR.search(bare))
    return False


def packet_key(p: dict) -> tuple[str, str, str]:
    """R25: the pair key (doc_a, doc_b, kind) of a packet -- its `pair` field, or (for cmp-000..039, written before
    the field existed) its two doc ids in order and its kind."""
    if p.get("pair"):
        a, b, k = p["pair"]
        return a, b, k
    return p["docs"][0]["doc_id"], p["docs"][1]["doc_id"], p["kind"]


def raw_packet_id(r) -> str | None:
    """The packet id of a raw line: `packet_id`, else `meta.packet_id` (used for every count, refusals included)."""
    if not isinstance(r, dict):
        return None
    pid = r.get("packet_id")
    if pid is None and isinstance(r.get("meta"), dict):
        pid = r["meta"].get("packet_id")
    return pid if isinstance(pid, str) else None


def _item_reasons(r: dict, p: dict, done: set, valid, chunk_doc, chunk_text):
    shown = {c["chunk_id"] for d in p["docs"] for c in d["chunks"]}
    pdocs = [d["doc_id"] for d in p["docs"]]
    q, ans = r.get("question"), r.get("answer")
    q = q.strip() if isinstance(q, str) else ""
    gold_raw = r.get("gold")
    gold = sorted(str(g) for g in gold_raw) if isinstance(gold_raw, list) else []
    item = HardItem(p["qid"], "compare", p["kind"], p["lang"], q, ans, gold, pdocs,
                    {"packet_id": p["packet_id"], "kind": p["kind"]})
    why = []
    if valid is not None and packet_key(p) not in valid:
        why.append("pair_invalid")                                          # R25
    if p["packet_id"] in done:
        why.append("duplicate_packet")
    if r.get("qtype", "compare") != "compare" or r.get("template") != p["kind"]:
        why.append("template_mismatch")
    docs = r.get("docs")
    if not (isinstance(docs, list) and all(isinstance(d, str) for d in docs) and sorted(docs) == sorted(pdocs)):
        why.append("docs_mismatch")
    if r.get("lang") != p["lang"]:
        why.append("lang_mismatch")
    if not q:
        why.append("empty_question")
    elif not script_ok(q, p["lang"]):
        why.append("script_mismatch")
    if not (ans.strip() if isinstance(ans, str) else ans not in (None, [], {})):
        why.append("empty_answer")
    if not isinstance(gold_raw, list):
        why.append("malformed_gold")
    if any(g not in shown for g in gold):
        why.append("gold_not_shown")
    why += check_compare(item, chunk_doc, chunk_text)
    return item, why


def ingest_compare(raws, packets: dict[str, dict], chunk_doc: dict[str, str], chunk_text: dict[str, str],
                   valid=None):
    """(accepted HardItems in packet order, stats Counter, rejections [(raw, reasons)], refusal_and_item packets). Lines are processed in packet
    order (stable within a packet; unknown packets last). One item per packet: the first acceptable line wins, later
    ones are `duplicate_packet`. `valid` (R25): the set of valid pair keys; an item whose packet key is not in it is
    `pair_invalid`. A malformed line is rejected (`malformed`), never fatal. Fourth return: the packets with both a
    refusal and an item line (flagged; st["refusal_and_item"] counts them)."""
    order = {pid: i for i, pid in enumerate(sorted(packets))}
    raws = sorted(raws, key=lambda r: order.get(raw_packet_id(r), len(order)))
    st, rej, items, done = Counter(), [], [], set()
    refused, itemed = set(), set()
    valid = {tuple(k) for k in valid} if valid is not None else None
    for r in raws:
        st["lines"] += 1
        pid = raw_packet_id(r)
        if isinstance(r, dict) and r.get("refusal") is not None:
            st["refusals"] += 1
            st[f"refusals:{(packets.get(pid) or {}).get('kind', '?')}"] += 1
            refused.add(pid)
            continue
        p = packets.get(pid)
        item = None
        if not isinstance(r, dict):
            why = ["malformed"]
        elif p is None:
            why = ["unknown_packet"]
        else:
            itemed.add(pid)
            try:
                item, why = _item_reasons(r, p, done, valid, chunk_doc, chunk_text)
            except (TypeError, ValueError, AttributeError, KeyError):
                why = ["malformed"]
        if why:
            st["rejected"] += 1
            for w in dict.fromkeys(why):
                st[f"rejected:{w}"] += 1
            rej.append((r, list(dict.fromkeys(why))))
            continue
        done.add(p["packet_id"])
        st["accepted"] += 1
        st[f"accepted:{p['kind']}"] += 1
        items.append(item)
    both = sorted(x for x in refused & itemed if x)
    st["refusal_and_item"] = len(both)
    st["packets_answered"] = len({x for x in refused | itemed if x in packets})
    return items, st, rej, both
