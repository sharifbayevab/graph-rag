"""M7 code checks on hard questions (docs/m7/2026-10-02-hard-questions-design.md §2, «Общие правила»).

leak_ok -- the lexical-leak rule: a question is refused if it shares a run of >= n consecutive tokens with any gold
fragment, or contains a gold fragment id. Tokens are graphrag.dataset.facts.ctoks (the M3 canonical tokens: Uzbek
Cyrillic folded to Latin, apostrophes dropped, numbers / dates canonical), so a transliterated copy counts as a copy.
By default no span is exempt, act titles included (unlike the M3 copy rule, which exempts titles). Since R14b,
leak_ok(exempt=...) takes fragment id -> a prefix of its text (a STRUCT fragment's first line = the root's title):
gold n-grams lying wholly inside that prefix are ignored; n-grams reaching past it, ids, and every other fragment
are still checked.

doc_components -- connected components of «question documents ∪ gold documents» for the future 60/20/20 split by
documents (spec §2, «Сплиты»)."""
from __future__ import annotations

from graphrag.dataset.facts import ctoks

STRUCT_PREFIX = "STRUCT/"


def _ngrams(toks: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {toks[i:i + n] for i in range(len(toks) - n + 1)}


def max_shared_run(question: str, text: str) -> int:
    """Longest run of consecutive canonical tokens of `question` that also occurs in `text` (diagnostics)."""
    a, b = [t.s for t in ctoks(question)], [t.s for t in ctoks(text)]
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def leak_ok(question: str, gold_texts, n: int = 8, ids=None, exempt: dict | None = None) -> bool:
    """True iff `question` has no n-gram (n canonical tokens) in common with any gold text and contains no gold
    fragment id. `gold_texts`: a mapping fragment id -> text (the ids are then derived from its keys) or a list of
    texts; `ids`: the gold fragment ids -- every production caller passes them (or the mapping). A bare list without
    ids checks only the n-grams (the brief's signature leak_ok(question, gold_texts, n=8) is kept; ruling R20).
    `exempt` (R14b, mapping form only): fragment id -> a prefix of its text (a STRUCT fragment's first line = the
    named root's title); gold n-grams lying wholly inside that prefix are ignored, every other n-gram still counts."""
    skip = {}
    if isinstance(gold_texts, dict):
        ids = sorted(set(ids or ()) | set(gold_texts))
        for k, pre in (exempt or {}).items():
            if k in gold_texts and pre and gold_texts[k].startswith(pre):
                skip[k] = len(ctoks(pre))
        texts = [(k, gold_texts[k]) for k in sorted(gold_texts)]
    else:
        texts = [(None, t) for t in gold_texts]
    if any(i and i in question for i in ids or ()):
        return False
    q = tuple(t.s for t in ctoks(question))
    if len(q) < n:
        return True
    grams = _ngrams(q, n)
    for k, t in texts:
        toks = tuple(x.s for x in ctoks(t))
        lo = max(0, skip.get(k, 0) - n + 1)             # first n-gram that reaches past the exempt prefix
        if grams & {toks[i:i + n] for i in range(lo, len(toks) - n + 1)}:
            return False
    return True


def _doc_of(fragment_id: str) -> str:
    """The document of a fragment id: STRUCT/<root> -> <root>, STRUCT/<article> -> its document (Task 6c);
    <doc_id>/<path>#k -> <doc_id>."""
    if fragment_id.startswith(STRUCT_PREFIX):
        from grl.env.structure import struct_doc
        return struct_doc(fragment_id)
    return fragment_id.split("/", 1)[0].split("#", 1)[0]


def doc_components(items, edges=None) -> dict[str, int]:
    """qid -> component index. An item joins its question documents (item.docs) and the documents of its gold
    fragments; components are the connected sets over all items, numbered 0.. in order of their smallest document id.
    `edges` is accepted for the interface (Task 5 brief); documents are joined only through items -- an EXT act is its
    own document, so two laws amending the same act fall into one component only if some item names both."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    item_docs = {}
    for it in items:
        ds = sorted(set(it.docs) | {_doc_of(g) for g in it.gold})
        item_docs[it.qid] = ds
        for d in ds:
            find(d)
        for d in ds[1:]:
            union(ds[0], d)
    roots = sorted({find(d) for d in parent})
    index = {r: i for i, r in enumerate(roots)}
    return {q: index[find(ds[0])] for q, ds in item_docs.items()}
