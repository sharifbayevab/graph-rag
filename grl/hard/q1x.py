"""M7 Task 6c (b): the Q1 expansion -- five new count templates with gold and answer computed from the pinned graph,
like grl.hard.templates.count_items (R6 visibility, R16 naming, leak check, probe ids excluded, per-root cap).

  count_parts_article            «Сколько частей в статье N закона Y?»   gold [STRUCT/<article>] (Task 6c rows):
                                 internal articles with >= 2 explicit part children and NO points (fix round 1, I1:
                                 in an amending article with amendment points the parser cuts "parts" at the quoted
                                 inserted text -- those parts belong to the amended act), the number unique in Y.
  count_points_article           «Сколько пунктов в статье N закона Y?»  same rows; >= 2 points (children of the
                                 article's parts) whose numbers are unique within the article -- a numbering that
                                 restarts per part makes «пункт N» ambiguous, so that article is not asked.
  count_codes_by_law             «Сколько кодексов изменил закон от <дата>?»  gold = the STRUCT/<code> rows of the
                                 codes (EXT roots `EXT/... kodeks`) the law's AMENDS edges reach, |gold| = answer in
                                 [1, 5]; every code's STRUCT text must print the law's group line (R6). The law has
                                 one date, unique among all amending sources of the graph.
  count_laws_amended_code_year   «Сколько законов внесли изменения в кодекс Y в G году?»  gold [STRUCT/<code>];
                                 answer = distinct amending sources of Y dated in G; every one of their group lines
                                 must be printed in Y's STRUCT text (R6; the 384-token cap may cut them).
  count_articles_deleted_by_law  «Сколько статей акта Y было исключено законом Z от <дата>?»  the AMENDS edges of Z with
                                 op delete whose dst is an article (статья) of Y AND whose evidence quote deletes the
                                 article itself (WHOLE_ARTICLE_DELETE: «N-модда чиқариб ташлансин», «N ва M-моддалар
                                 …»; the graph also tags word / paragraph deletions inside an article as an article
                                 delete -- «номидаги «…» деган сўзлар чиқариб ташлансин» -- and those edges are not
                                 counted); answer = distinct dst articles of the kept edges (the count comes from the
                                 edges, the quote only filters them); gold = the R33 evidence fragments of the kept
                                 edges, |gold| in [1, 5], every quote visible in its fragments. No STRUCT row carries this count (Y is an EXT act:
                                 no article rows, and the document row prints only per-law op totals), so this is the
                                 one template whose gold is corpus text -- the probe filter is not trivial for it.

Selection: templates.count_items' machinery (round-robin over templates, language quotas, least-used root first),
per-root cap Q1X_CAP = 6 over the five templates together (ruling R57: its own counter, the Task 4 Q1 items do not count
-- one act gets at most 4 old + 6 new Q1 items). The root of an item is docs[0]: the document asked about (the law Z
for count_codes_by_law). R44: a caller's pre_drop runs before selection. Probe (Task 6c): `probe_drop(cand, lang) ->
reason | None` removes a (candidate, language) rendering whose gold the dense@5 or fused@5 baseline fully retrieves,
before selection."""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from grl.env.structure import _article, _Resolver, article_structure, doc_structure, struct_id
from grl.hard.templates import (MAX_GOLD, MIN_AMENDS, OLD_SET_LANG_COUNTS, HardItem, _Cand, _amends_by_root,
                                _check_langs, _drop_excluded, _law_line_articles, _pre_drop, _r16_ok, _select,
                                _unique_date_laws, evidence_gold, root_names, src_kind)

Q1X_TEMPLATES = ("count_parts_article", "count_points_article", "count_codes_by_law",
                 "count_laws_amended_code_year", "count_articles_deleted_by_law")
Q1X_CAP = 6                      # R57: new Q1 items per root over the five templates (own counter)
MIN_PARTS, MIN_POINTS = 2, 2     # a 1-part article is an undivided text (the parser's single Ч.1), not «1 часть»
CODE_RE = re.compile(r"^EXT/[^/]+ kodeks$")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LINE = re.compile(r"^(\S+) (?:«.*» )?(\d{4}-\d{2}-\d{2}|): ")
_SUP = "¹²³⁴⁵⁶⁷⁸⁹⁰"
WHOLE_ARTICLE_DELETE = re.compile(rf"(?:^|\s)[\d{_SUP}]+(?:\s*(?:,|ва|va|–|—|-)\s*[\d{_SUP}]+)*-(?:модда|modda)"
                                  r"(?:лар|lar)?(?:си|si|и|i)?\s+(?:чиқариб\s+ташлансин|chiqarib\s+tashlansin)\s*[.;]?$")


def whole_article_delete(quote: str) -> bool:
    """The evidence quote deletes whole articles («44-модда чиқариб ташлансин», «78 – 81-моддалар …»), not words or a
    paragraph inside one."""
    return bool(WHOLE_ARTICLE_DELETE.search(" ".join((quote or "").split())))


def is_code(root: str) -> bool:
    return bool(CODE_RE.match(root))


def _year_sources_printed(text: str, year: str) -> set[str]:
    """Sources of the AMENDS group lines of `year` printed in a document STRUCT text (REPEALS lines «бекор: …» and the
    header are not group lines)."""
    out = set()
    for ln in text.split("\n"):
        m = _LINE.match(ln)
        if m and m.group(2)[:4] == year:
            out.add(m.group(1))
    return out


def _cand(stats, template, root, title, number, answer, gold, docs, meta, params, texts, exempt) -> _Cand:
    stats[f"accepted_candidates:{template}"] += 1
    return _Cand(template, root, title, number, str(answer), sorted(gold), docs, meta, params,
                 [texts[g] for g in sorted(gold)], exempt)


def article_candidates(nodes, edges, art_texts: dict, named, dropped, stats: Counter) -> list[_Cand]:
    arts = article_structure(nodes, edges)
    numbers = defaultdict(Counter)                      # doc -> article number -> articles (all internal articles)
    by_id = {n["node_id"]: n for n in nodes}
    roots = {i: n for i, n in by_id.items() if n["kind"] == "document"}
    root_of = _Resolver(roots, edges)
    for n in nodes:
        if n.get("type") == "статья" and not n.get("external"):
            d = root_of(n["node_id"])
            if d:
                numbers[d][str((n.get("attrs") or {}).get("number") or "")] += 1
    out = []
    for a in sorted(arts):
        info = arts[a]
        doc, sid = info["doc"], struct_id(a)
        text = art_texts.get(sid, "")
        if not info["doc_title"].strip():
            continue
        if info["n_parts"] >= MIN_PARTS and info["n_points"]:
            stats["parts_with_points"] += 1                 # I1: parts cut at quoted inserted text
        for template, n, ok in (("count_parts_article", info["n_parts"],
                                 info["n_parts"] >= MIN_PARTS and not info["n_points"]),
                                ("count_points_article", info["n_points"],
                                 info["n_points"] >= MIN_POINTS and info["points_unique"])):
            if not ok:
                if template == "count_points_article" and info["n_points"] >= MIN_POINTS:
                    stats["points_ambiguous"] += 1
                continue
            stats["candidates"] += 1
            if not _r16_ok(doc, named, dropped, template, stats):
                continue
            if numbers[doc][info["number"]] != 1 or not info["number"]:
                stats["article_number_ambiguous"] += 1
                stats[f"article_number_ambiguous:{template}"] += 1
                continue
            if f"қисмлар: {info['n_parts']}; бандлар: {info['n_points']}" not in text:
                stats["r6_dropped"] += 1
                stats[f"r6_dropped:{template}"] += 1
                continue
            params = {"article": info["number"], "doctype": "law" if info["doc_type"] == "law_text" else "doc"}
            out.append(_cand(stats, template, doc, info["doc_title"], named[doc], n, [sid], [doc],
                             {"article": info["number"]}, params, {sid: text}, {sid: text.split("\n", 1)[0]}))
    return out


def code_candidates(nodes, edges, struct_texts: dict, named, dropped, stats: Counter) -> list[_Cand]:
    s = doc_structure(nodes, edges)
    out = []
    # count_codes_by_law: law -> codes, dates over ALL its AMENDS edges
    dates_of, codes_of, laws_of_date = defaultdict(set), defaultdict(set), defaultdict(set)
    for r, info in s.items():
        for src, date, _, _ in info["amended_by"]:
            dates_of[src].add(date)
            if is_code(r):
                codes_of[src].add(r)
    for src, ds in dates_of.items():
        if len(ds) == 1:
            laws_of_date[next(iter(ds))].add(src)
    for z in sorted(codes_of):
        codes = sorted(codes_of[z])
        stats["candidates"] += 1
        t = "count_codes_by_law"
        ds = dates_of[z]
        d = next(iter(ds)) if len(ds) == 1 else None
        if d is None or not _DATE.fullmatch(d) or len(laws_of_date[d]) != 1:
            stats["law_date_not_unique"] += 1
            stats[f"law_date_not_unique:{t}"] += 1
            continue
        if not 1 <= len(codes) <= MAX_GOLD:
            stats["gold_out_of_range"] += 1
            stats[f"gold_out_of_range:{t}"] += 1
            continue
        texts = {struct_id(c): struct_texts.get(struct_id(c), "") for c in codes}
        if any(_law_line_articles(texts[struct_id(c)], z, d) is None for c in codes):
            stats["r6_dropped"] += 1
            stats[f"r6_dropped:{t}"] += 1
            continue
        out.append(_cand(stats, t, z, "", None, len(codes), list(texts), [z] + codes, {"law": z, "date": d},
                         {"date": d, "src": src_kind(z)}, texts, {g: x.split("\n", 1)[0] for g, x in texts.items()}))
    # count_laws_amended_code_year
    for r in sorted(s):
        info = s[r]
        if not is_code(r) or len(info["amended_by"]) < MIN_AMENDS or not info["title"].strip():
            continue
        sid = struct_id(r)
        text = struct_texts.get(sid, "")
        by_year = defaultdict(set)
        for src, date, _, _ in info["amended_by"]:
            by_year[date[:4]].add(src)
        for g in sorted(by_year):
            t = "count_laws_amended_code_year"
            if not re.fullmatch(r"\d{4}", g):
                continue
            stats["candidates"] += 1
            if not _r16_ok(r, named, dropped, t, stats):
                continue
            laws = by_year[g]
            if _year_sources_printed(text, g) != laws:
                stats["r6_dropped"] += 1
                stats[f"r6_dropped:{t}"] += 1
                continue
            kind = "law" if all(src_kind(z) == "law" for z in laws) else "doc"
            out.append(_cand(stats, t, r, info["title"], named[r], len(laws), [sid], [r], {"year": g},
                             {"year": g, "src": kind}, {sid: text}, {sid: text.split("\n", 1)[0]}))
    return out


def deleted_candidates(nodes, edges, mu_inv: dict, chunk_texts: dict, chunk_spans: dict, named, dropped,
                       stats: Counter) -> list[_Cand]:
    typ = {n["node_id"]: n.get("type") for n in nodes}
    titles = {n["node_id"]: n.get("title") or "" for n in nodes if n["kind"] == "document"}
    out, t = [], "count_articles_deleted_by_law"
    for r, recs in _amends_by_root(nodes, edges).items():
        if len(recs) < MIN_AMENDS or not titles.get(r, "").strip():
            continue
        for z, d in _unique_date_laws([x[:4] for x in recs]).items():
            dl = [x for x in recs if x[0] == z and x[2] == "delete" and typ.get(x[3]) == "статья"]
            if not dl:
                continue
            stats["candidates"] += 1
            whole = [x for x in dl if whole_article_delete(x[5])]
            stats["not_whole_article_edges"] += len(dl) - len(whole)
            if not whole:
                stats[f"not_whole_article:{t}"] += 1
                continue
            dl = whole
            if not _r16_ok(r, named, dropped, t, stats):
                continue
            golds = []
            for x in dl:
                span = x[6]
                ev = {"attrs": {"evidence": {"doc_id": span[0], "start": span[1], "end": span[2]}}} if span else {}
                golds.append(evidence_gold(ev, mu_inv.get(x[4], []), chunk_spans) if span else [])
            if any(not g for g in golds):
                stats["r33_no_quote"] += 1
                stats[f"r33_no_quote:{t}"] += 1
                continue
            if any(x[5] and " ".join(x[5].split()) not in " ".join(" ".join(chunk_texts.get(c, "") for c in g).split())
                   for x, g in zip(dl, golds)):
                stats["evidence_invisible"] += 1
                stats[f"evidence_invisible:{t}"] += 1
                continue
            gold = sorted({c for g in golds for c in g})
            if not 1 <= len(gold) <= MAX_GOLD:
                stats["gold_out_of_range"] += 1
                stats[f"gold_out_of_range:{t}"] += 1
                continue
            arts = {_article(x[3]) for x in dl}
            out.append(_cand(stats, t, r, titles[r], named[r], len(arts), gold, [r, z],
                             {"law": z, "date": d, "n_edges": len(dl)}, {"date": d, "src": src_kind(z)},
                             {c: chunk_texts.get(c, "") for c in gold}, {}))
    return out


def q1x_candidates(nodes, edges, *, struct_texts: dict, art_texts: dict, mu_inv: dict, chunk_texts: dict,
                   chunk_spans: dict, stats: Counter | None = None) -> list[_Cand]:
    stats = stats if stats is not None else Counter()
    named, dropped = root_names(nodes, doc_structure(nodes, edges))
    return (article_candidates(nodes, edges, art_texts, named, dropped, stats)
            + code_candidates(nodes, edges, struct_texts, named, dropped, stats)
            + deleted_candidates(nodes, edges, mu_inv, chunk_texts, chunk_spans, named, dropped, stats))


def _probe(cands: list[_Cand], probe_drop, stats: Counter) -> list[_Cand]:
    """Task 6c probe: drop the (candidate, language) renderings probe_drop refuses (dense@5 / fused@5 all@5 = 1);
    a candidate with no language left is dropped. Counted per template."""
    if probe_drop is None:
        return cands
    out = []
    for c in cands:
        had = bool(c.questions)
        for lang in sorted(c.questions):
            stats[f"probe_checked:{c.template}"] += 1
            why = probe_drop(c, lang)
            if why:
                stats[why] += 1
                stats[f"{why}:{c.template}"] += 1
                del c.questions[lang]
        if c.questions:
            out.append(c)
        elif had:
            stats[f"probe_dropped_candidates:{c.template}"] += 1
    return out


def q1x_items(nodes, edges, rng_seed: int, n: int, *, struct_texts: dict, art_texts: dict, mu_inv: dict,
              chunk_texts: dict, chunk_spans: dict, lang_shares: dict | None = None, max_per_root: int = Q1X_CAP,
              stats: Counter | None = None, exclude=None, pre_drop=None, probe_drop=None,
              cands: list | None = None) -> list[HardItem]:
    """The new Q1 items: candidates -> probe ids excluded -> R44 pre_drop -> leak check per language -> probe filter
    -> selection (one item per (template, root, meta); per-root cap over the five templates). `cands`: precomputed
    q1x_candidates (re-selection rounds reuse them; their questions are re-rendered here)."""
    st = Counter()
    cs = cands if cands is not None else q1x_candidates(nodes, edges, struct_texts=struct_texts,
                                                         art_texts=art_texts, mu_inv=mu_inv, chunk_texts=chunk_texts,
                                                         chunk_spans=chunk_spans, stats=st)
    cs = _pre_drop(_drop_excluded(cs, "count", exclude, st), pre_drop, st)
    for c in cs:
        c.questions = {}
        _check_langs(c, st)
    cs = _probe(cs, probe_drop, st)
    items = _select(cs, n, random.Random(rng_seed), lang_shares or OLD_SET_LANG_COUNTS, max_per_root,
                    lambda c: c.key, None, "count", Counter(), st)
    st["selected"] = len(items)
    if stats is not None:
        stats.update(st)
    return items


def make_q1x(nodes, edges, seed: int, n: int, *, fixed_splits: dict, chunk_doc: dict, max_rounds: int = 50, **kw):
    """R32 / R44 / R45 for the new items against `fixed_splits` (doc -> split of the old set and every existing M7
    row: grl.hard.splits.fixed_doc_splits): a candidate whose documents hold two fixed splits is dropped before
    selection; the selected rows are split by assign_splits with the fixed map as the «old» splits (inherited
    components keep their split, new ones go by deficit); a chain conflict among the selected rows drops those
    candidates and re-selects. Returns (rows = HardItem dicts + gold_docs + split, stats, rounds)."""
    from grl.hard.splits import assign_splits, gold_docs
    from grl.hard.templates import item_key
    chained: set = set()
    st0 = Counter()
    cands = q1x_candidates(nodes, edges, struct_texts=kw["struct_texts"], art_texts=kw["art_texts"],
                           mu_inv=kw["mu_inv"], chunk_texts=kw["chunk_texts"], chunk_spans=kw["chunk_spans"],
                           stats=st0)

    def pre_drop(c):
        docs = set(c.docs) | set(gold_docs(c.gold, chunk_doc))
        if len({fixed_splits[d] for d in docs if d in fixed_splits}) > 1:
            return "r32_conflict"
        if c.key in chained:
            return "r32_chain"
        return None

    for rounds in range(1, max_rounds + 1):
        st = Counter(st0)
        items = q1x_items(nodes, edges, seed, n, stats=st, pre_drop=pre_drop, cands=cands, **kw)
        rows = [{**it.to_json(), "gold_docs": gold_docs(it.gold, chunk_doc)} for it in items]
        split = assign_splits(rows, fixed_splits, seed)
        bad = [r for r in rows if split[r["qid"]] is None]
        if not bad:
            break
        chained |= {item_key(r) for r in bad}
    else:
        raise RuntimeError(f"R32 chain conflicts not resolved in {max_rounds} rounds")
    for r in rows:
        r["split"] = split[r["qid"]]
    return rows, st, rounds


# --- fix round 1, R59: alternative evidence of the part / point items ----------------------------------------------

ARTICLE_TEMPLATES = ("count_parts_article", "count_points_article")


def article_chunks(article: str, chunk_ids, chunk_node_ids) -> list[str]:
    """The corpus chunks whose μ vertex is the article or lies under it (`chunk_node_ids[i]` = the node id of chunk i),
    sorted: together they hold the article's text, every part and point."""
    pre = article + "/"
    return sorted(c for c, v in zip(chunk_ids, chunk_node_ids) if v == article or v.startswith(pre))


def contiguous(chunks, spans: dict) -> bool:
    """The chunks cover one contiguous stretch of their document: no other chunk of the document starts inside
    [min start, max end) of the set (spans: chunk id -> (doc, start, end), all chunks of the corpus)."""
    if not chunks:
        return False
    docs = {spans[c][0] for c in chunks}
    if len(docs) != 1:
        return False
    d = next(iter(docs))
    lo, hi = min(spans[c][1] for c in chunks), max(spans[c][2] for c in chunks)
    inside = {c for c, (dd, a, _) in spans.items() if dd == d and lo <= a < hi}
    return inside == set(chunks)


def annotate_alt(rows, chunk_ids, chunk_node_ids, spans: dict) -> dict:
    """R59: rows of the article templates get alt_gold (article_chunks of the gold article) and easy (the article is
    one chunk); every other row gets alt_gold [] and easy False. Training labels keep the STRUCT gold. A part / point
    item whose chunks are not one contiguous stretch is a loud error (the any-of set would not hold the article).
    Returns counts per template / easy."""
    by_t = Counter()
    for r in rows:
        if r["template"] in ARTICLE_TEMPLATES:
            art = r["gold"][0][len("STRUCT/"):]
            alt = article_chunks(art, chunk_ids, chunk_node_ids)
            if not contiguous(alt, spans):
                raise ValueError(f"{r['qid']}: the chunks of {art} are not one contiguous stretch: {alt[:3]}")
            r["alt_gold"], r["easy"] = alt, len(alt) == 1
        else:
            r["alt_gold"], r["easy"] = [], False
        art = r["template"] in ARTICLE_TEMPLATES
        by_t[f"{r['template']}|{'easy' if r['easy'] else 'non_easy'}" if art else r["template"]] += 1
    return dict(sorted(by_t.items()))
