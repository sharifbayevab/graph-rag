"""M7 structure fragments (docs/m7/2026-10-02-hard-questions-design.md §3.2): one virtual fragment per document /
external act, built deterministically from the pinned graph -- element counts by CONTAINS, amendment history by
AMENDS / REPEALS. Not in any index and not in μ⁻¹: only the `structure` operator reaches it (grl.model.variants).

Roots are every `kind == "document"` vertex: internal documents and external acts `EXT/<act_key>`. A vertex belongs
to the root found by walking CONTAINS upward; if that walk does not end at a root (an element with no CONTAINS
parent), the id prefix decides: `EXT/<act_key>/...` -> `EXT/<act_key>`, `<doc_id>/...` -> `<doc_id>`."""
from __future__ import annotations

from collections import Counter, defaultdict

PREFIX = "STRUCT/"
LEVELS = {"глава": "n_chapters", "статья": "n_articles", "часть": "n_parts", "пункт": "n_points"}


def struct_id(node_id: str) -> str:
    return PREFIX + node_id


def _prefix_root(node_id: str) -> str:
    parts = node_id.split("/")
    return "/".join(parts[:2]) if parts[0] == "EXT" else parts[0]


class _Resolver:
    def __init__(self, roots, edges):
        self.roots = roots
        self.parent = {}
        for e in edges:
            if e["type"] == "CONTAINS":
                self.parent.setdefault(e["dst"], e["src"])
        self.cache: dict[str, str | None] = {}

    def __call__(self, node_id: str) -> str | None:
        if node_id in self.cache:
            return self.cache[node_id]
        x, seen = node_id, set()
        while x not in self.roots and x in self.parent and x not in seen:
            seen.add(x)
            x = self.parent[x]
        if x not in self.roots:
            x = _prefix_root(node_id)
        r = x if x in self.roots else None
        self.cache[node_id] = r
        return r


def _article(node_id: str) -> str | None:
    for part in node_id.split("/"):
        if part.startswith("СТ."):
            return part[3:]
    return None


def doc_structure(nodes, edges) -> dict[str, dict]:
    roots = {n["node_id"]: n for n in nodes if n["kind"] == "document"}
    by_id = {n["node_id"]: n for n in nodes}
    root_of = _Resolver(roots, edges)
    out = {r: {"title": n.get("title") or "", "kind": "ext" if n.get("external") else "document",
               "doc_type": (n.get("attrs") or {}).get("doc_type") or "",
               **{v: 0 for v in LEVELS.values()}, "amended_by": [], "repealed_by": [],
               "amend_counts": {"by_year": {}, "by_op": {}}, "by_year_op": {}, "amended_articles": [],
               "src_titles": {}} for r, n in roots.items()}
    for n in nodes:
        lv = LEVELS.get(n.get("type"))
        if not lv or n["node_id"] in roots:
            continue
        r = root_of(n["node_id"])
        if r:
            out[r][lv] += 1
    years, ops, arts = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    year_ops = defaultdict(lambda: defaultdict(Counter))
    for e in edges:
        if e["type"] not in ("AMENDS", "REPEALS"):
            continue
        r = root_of(e["dst"])
        if r is None:
            continue
        src_doc = by_id.get(e["src"], {}).get("doc_id") or root_of(e["src"]) or _prefix_root(e["src"])
        a = e.get("attrs") or {}
        rec = (src_doc, a.get("date") or "", a.get("op") or ("repeal" if e["type"] == "REPEALS" else ""), e["dst"])
        out[r]["src_titles"][src_doc] = by_id.get(src_doc, {}).get("title") or ""
        if e["type"] == "REPEALS":
            out[r]["repealed_by"].append(rec)
            continue
        out[r]["amended_by"].append(rec)
        years[r][(a.get("date") or "")[:4]] += 1
        ops[r][a.get("op") or ""] += 1
        year_ops[r][(a.get("date") or "")[:4]][a.get("op") or ""] += 1
        art = _article(e["dst"])
        if art:
            arts[r].add(art)
    for r, info in out.items():
        info["amended_by"].sort(key=lambda x: (x[1], x[0], x[3]))
        info["repealed_by"].sort(key=lambda x: (x[1], x[0], x[3]))
        info["amend_counts"] = {"by_year": dict(sorted(years[r].items())), "by_op": dict(sorted(ops[r].items()))}
        info["by_year_op"] = {y: dict(sorted(year_ops[r][y].items())) for y in sorted(year_ops[r])}
        info["src_titles"] = dict(sorted(info["src_titles"].items()))
        info["amended_articles"] = sorted(arts[r], key=_art_key)
    return out


def _art_key(s: str):
    return (len(s), s)


def _short_title(title: str, count_tokens, n: int = 12) -> str:
    """Longest whitespace-word prefix of `title` that fits in n tokens."""
    out = ""
    for w in title.split():
        cand = (out + " " + w).strip()
        if count_tokens(cand) > n:
            break
        out = cand
    return out


def _groups(recs, titles, count_tokens, art_label: str) -> list[str]:
    """One line per (source doc, date): op counts (sorted) and the distinct articles touched."""
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for d, date, op, dst in recs:
        grouped[(d, date)].append((op, dst))
    lines = []
    for (d, date) in sorted(grouped, key=lambda k: (k[1], k[0])):
        items = grouped[(d, date)]
        opc = Counter(op for op, _ in items)
        arts = sorted({a for a in (_article(dst) for _, dst in items) if a}, key=_art_key)
        title = _short_title(titles.get(d, ""), count_tokens)
        line = f"{d} «{title}» {date}: " if title else f"{d} {date}: "
        line += ", ".join(f"{o} {c}" for o, c in sorted(opc.items()))
        if arts:
            line += f"; {art_label} " + ", ".join(arts)
        lines.append(line)
    return lines


def struct_text(info: dict, count_tokens, max_tokens: int = 384) -> str:
    year_ops = info.get("by_year_op") or {}
    by_year = "; ".join(
        f"{y}: {c}" + (" (" + ", ".join(f"{o} {k}" for o, k in year_ops[y].items()) + ")" if y in year_ops else "")
        for y, c in info["amend_counts"]["by_year"].items())
    head = [info["title"],
            f"тур: {info['doc_type']}" if info.get("doc_type") else "",
            # EXT roots hold only stub elements (the articles that were amended / referenced), not the act's structure
            "" if info["kind"] == "ext" else
            f"боблар: {info['n_chapters']}; моддалар: {info['n_articles']}; қисмлар: {info['n_parts']}; "
            f"бандлар: {info['n_points']}",
            f"ўзгартишлар: {len(info['amended_by'])}" + (f"; {by_year}" if by_year else ""),
            "; ".join(f"{o}: {c}" for o, c in info["amend_counts"]["by_op"].items()),
            f"бекор қилинган: {len(info['repealed_by'])}"]
    titles = info.get("src_titles") or {}
    tail = _groups(info["amended_by"], titles, count_tokens, "ўзгартирилган моддалар")
    tail += ["бекор: " + x for x in _groups(info["repealed_by"], titles, count_tokens, "бекор қилинган моддалар")]
    if info["amended_articles"]:
        tail.append("ўзгартирилган моддалар (жами): " + ", ".join(info["amended_articles"]))
    text = "\n".join(h for h in head if h)
    if count_tokens(text) > max_tokens:   # counts are never cut silently
        raise ValueError(f"structure header alone exceeds {max_tokens} tokens: {info['title'][:60]}")
    for line in tail:                      # counts first and always complete; lists cut in order at the token cap
        cand = text + "\n" + line
        if count_tokens(cand) > max_tokens:
            break
        text = cand
    return text


def build_struct_texts(nodes, edges, count_tokens, max_tokens: int = 384) -> dict[str, str]:
    s = doc_structure(nodes, edges)
    return {struct_id(r): struct_text(s[r], count_tokens, max_tokens) for r in sorted(s)}


# --- Task 6c: article-level structure fragments ---------------------------------------------------------------------
# STRUCT/<article node> for every INTERNAL article with >= 1 explicit part child (CONTAINS статья -> часть). The counts
# are read from the graph's CONTAINS children only (never from text); an external (EXT) article is a stub that holds
# only the elements some law amended / referenced, so its children are not the article's structure and it gets no
# row (controller ruling: rows only where the counts come from explicit graph structure).

ARTICLE, PART, POINT = "статья", "часть", "пункт"


def struct_doc(sid: str) -> str:
    """The document of a STRUCT id: STRUCT/<root> -> <root>, STRUCT/<doc>/<...>/СТ.N -> <doc> (EXT/<act>/... ->
    EXT/<act>); document ids hold no "/" (internal) or exactly one (EXT/<act_key>)."""
    if not sid.startswith(PREFIX):
        raise ValueError(f"not a STRUCT id: {sid!r}")
    return _prefix_root(sid[len(PREFIX):])


def _num_key(s: str):
    """Numeric order of element numbers ('2', '10', '3.1'); non-numeric parts after the numeric ones."""
    out = []
    for x in str(s).split("."):
        out.append((0, int(x), "") if x.isdigit() else (1, 0, x))
    return tuple(out)


def article_structure(nodes, edges) -> dict[str, dict]:
    """article node id -> {doc, doc_title, doc_type, number, title, parts [numbers], points {part: [numbers]},
    n_parts, n_points, points_unique, amended_by, repealed_by, src_titles} for every internal article with >= 1 part
    child. Amendment records (src doc, date, op, dst) are the AMENDS / REPEALS edges whose dst is the article or lies
    under it (CONTAINS)."""
    by_id = {n["node_id"]: n for n in nodes}
    roots = {i: n for i, n in by_id.items() if n["kind"] == "document"}
    root_of = _Resolver(roots, edges)
    kids, parent = defaultdict(list), {}
    for e in edges:
        if e["type"] == "CONTAINS":
            kids[e["src"]].append(e["dst"])
            parent.setdefault(e["dst"], e["src"])

    def typ(i):
        return by_id.get(i, {}).get("type")

    def num(i):
        return str((by_id[i].get("attrs") or {}).get("number") or "")

    out = {}
    for a, n in sorted(by_id.items()):
        if n.get("type") != ARTICLE or n.get("external") or a in roots:
            continue
        parts = sorted((p for p in kids[a] if typ(p) == PART), key=lambda p: (_num_key(num(p)), p))
        doc = root_of(a)
        if not parts or doc is None or roots[doc].get("external"):
            continue
        points = {num(p): [num(x) for x in sorted((x for x in kids[p] if typ(x) == POINT),
                                                   key=lambda x: (_num_key(num(x)), x))] for p in parts}
        points = {p: v for p, v in points.items() if v}
        flat = [x for v in points.values() for x in v]
        out[a] = {"doc": doc, "doc_title": roots[doc].get("title") or "",
                  "doc_type": (roots[doc].get("attrs") or {}).get("doc_type") or "",
                  "number": num(a), "title": " ".join((n.get("title") or "").split()),
                  "parts": [num(p) for p in parts], "points": points, "n_parts": len(parts), "n_points": len(flat),
                  "points_unique": len(set(flat)) == len(flat), "amended_by": [], "repealed_by": [], "src_titles": {}}
    for e in edges:
        if e["type"] not in ("AMENDS", "REPEALS"):
            continue
        x, seen = e["dst"], set()
        while x not in out and x in parent and x not in seen:
            seen.add(x)
            x = parent[x]
        if x not in out:
            continue
        a = e.get("attrs") or {}
        src_doc = by_id.get(e["src"], {}).get("doc_id") or root_of(e["src"]) or _prefix_root(e["src"])
        rec = (src_doc, a.get("date") or "", a.get("op") or ("repeal" if e["type"] == "REPEALS" else ""), e["dst"])
        out[x]["src_titles"][src_doc] = by_id.get(src_doc, {}).get("title") or ""
        out[x]["repealed_by" if e["type"] == "REPEALS" else "amended_by"].append(rec)
    for info in out.values():
        for k in ("amended_by", "repealed_by"):
            info[k].sort(key=lambda r: (r[1], r[0], r[3]))
        info["src_titles"] = dict(sorted(info["src_titles"].items()))
    return out


def article_struct_text(info: dict, count_tokens, max_tokens: int = 384) -> str:
    """Line 1 = the document title (the leak check's exempt prefix, as in struct_text); then the article number (and
    its heading), the part / point counts (never cut), the part numbers, the point numbers per part, and the amendment
    lines in struct_text's format (cut in order at the token cap)."""
    head = [info["doc_title"],
            f"{info['number']}-модда" + (f". {info['title']}" if info["title"] else ""),
            f"қисмлар: {info['n_parts']}; бандлар: {info['n_points']}"]
    titles = info.get("src_titles") or {}
    tail = ["қисм рақамлари: " + ", ".join(info["parts"])]
    if info["points"]:
        tail.append("банд рақамлари: " + "; ".join(f"{p}-қисм: " + ", ".join(v) for p, v in info["points"].items()))
    if info["amended_by"]:
        tail.append(f"ўзгартишлар: {len(info['amended_by'])}")
        tail += _groups(info["amended_by"], titles, count_tokens, "ўзгартирилган моддалар")
    if info["repealed_by"]:
        tail.append(f"бекор қилинган: {len(info['repealed_by'])}")
        tail += ["бекор: " + x for x in _groups(info["repealed_by"], titles, count_tokens, "бекор қилинган моддалар")]
    text = "\n".join(h for h in head if h)
    if count_tokens(text) > max_tokens:
        raise ValueError(f"article structure header alone exceeds {max_tokens} tokens: {info['doc_title'][:60]}")
    for line in tail:
        cand = text + "\n" + line
        if count_tokens(cand) > max_tokens:
            break
        text = cand
    return text


def build_article_struct_texts(nodes, edges, count_tokens, max_tokens: int = 384) -> dict[str, str]:
    s = article_structure(nodes, edges)
    return {struct_id(a): article_struct_text(s[a], count_tokens, max_tokens) for a in sorted(s)}
