"""Chunking (spec §2.4).

A chunk never crosses the boundary of its chunking unit (article, resolution item,
speech, block, preamble, requisites, header) nor of a point inside it; chunks never
overlap. A unit that fits `max_tokens` is one chunk. A larger unit is cut into atomic
segments — child elements (recursively) → paragraphs → sentences → word windows — and
consecutive segments are packed greedily up to `max_tokens`. The chunk's `parent` is the
deepest node whose span contains it (μ, spec §3.3).
"""
from __future__ import annotations

import bisect
import re
from collections import defaultdict
from typing import Callable

from graphrag.data.structure import _ART, _ART_RU, _CH, _CH_RU, _SECT, DocText
from graphrag.types import Chunk, Node

UNIT_TYPES = {"статья", "преамбула", "реквизиты", "заголовок", "блок", "выступление"}
SEALED_TYPES = {"пункт", "подпункт"}   # a packed chunk may contain or sit inside these, never cut them
ATTACH_MIN_TOKENS = 48        # default for config/data.yaml chunker.attach_min_tokens
BREADCRUMB_MAX_TOKENS = 96
ENCODER_MAX_TOKENS = 512      # encoder max_seq_length: prefix + index_text + 2 special tokens must fit
# Budget reserve for a passage prefix. bge-m3 uses none, but its tokenizer is the e5 XLM-R one
# (identical vocab/ids), so keeping the e5 reserve (3 tokens) leaves chunk ids and index_text
# unchanged and stays safe for an e5-style fine-tune (U8).
PASSAGE_PREFIX = "passage: "
_HEAD_RX = {"глава": (_CH, _CH_RU), "параграф": (_SECT,), "статья": (_ART, _ART_RU)}
_SENT = re.compile(r"(?<=[.!?;])\s+")
_WORDS = re.compile(r"\S+")


def is_unit(n: Node, by_id: dict[str, Node]) -> bool:
    if n.type in UNIT_TYPES:
        return True
    return n.type == "пункт" and n.parent_id is not None and by_id[n.parent_id].type == "документ"


class _Chunker:
    def __init__(self, dt: DocText, nodes: list[Node], max_tokens: int, count: Callable[[str], int],
                 to_norm: Callable[[str], str] = lambda t: t):
        self.dt, self.max, self.count, self.to_norm = dt, max_tokens, count, to_norm
        self.children: dict[str, list[Node]] = defaultdict(list)
        for n in nodes:
            if n.parent_id:
                self.children[n.parent_id].append(n)
        for v in self.children.values():
            v.sort(key=lambda n: (n.start, -n.end))
        self.starts = [s for s, _ in dt.offsets]

    def measure(self, s: int, e: int) -> int:
        """Tokens of a span: max over the display text and its internal form, so that both
        Chunk.text and Chunk.text_norm stay within max_tokens."""
        t = self.dt.text[s:e]
        return max(self.count(t), self.count(self.to_norm(t)))

    def fits(self, s: int, e: int) -> bool:
        return self.measure(s, e) <= self.max

    def paragraphs(self, s: int, e: int) -> list[tuple[int, int]]:
        i = max(bisect.bisect_right(self.starts, s) - 1, 0)
        out = []
        while i < len(self.dt.offsets) and self.dt.offsets[i][0] < e:
            a, b = max(self.dt.offsets[i][0], s), min(self.dt.offsets[i][1], e)
            if a < b:
                out.append((a, b))
            i += 1
        return out

    def windows(self, s: int, e: int) -> list[tuple[int, int]]:
        out, ws, last = [], None, None
        for m in _WORDS.finditer(self.dt.text, s, e):
            if not self.fits(m.start(), m.end()):        # one run alone exceeds max_tokens
                if ws is not None:
                    out.append((ws, last))
                out.extend(self.char_split(m.start(), m.end()))
                ws = last = None
                continue
            if ws is None:
                ws = m.start()
            elif not self.fits(ws, m.end()):
                out.append((ws, last))
                ws = m.start()
            last = m.end()
        if ws is not None:
            out.append((ws, last))
        return out

    def char_split(self, s: int, e: int) -> list[tuple[int, int]]:
        """Split a whitespace-free run into consecutive largest fitting prefixes (binary search)."""
        out = []
        while s < e:
            lo, hi = s + 1, e                             # at least one char per piece
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.fits(s, mid):
                    lo = mid
                else:
                    hi = mid - 1
            out.append((s, lo))
            s = lo
        return out

    def split_long(self, s: int, e: int) -> list[tuple[int, int]]:
        if self.fits(s, e):
            return [(s, e)]
        pieces, cur = [], s
        for m in _SENT.finditer(self.dt.text, s, e):
            pieces.append((cur, m.start()))
            cur = m.end()
        pieces.append((cur, e))
        out = []
        for a, b in pieces:
            if a < b:
                out.extend([(a, b)] if self.fits(a, b) else self.windows(a, b))
        return out

    def _flat(self, s: int, e: int) -> list[tuple[int, int]]:
        return [x for ps, pe in self.paragraphs(s, e) for x in self.split_long(ps, pe)]

    def segments(self, node: Node) -> list[tuple[int, int]]:
        if self.fits(node.start, node.end):
            return [(node.start, node.end)]
        kids = self.children.get(node.node_id, [])
        if not kids:
            return self._flat(node.start, node.end)
        segs, cur = [], node.start
        for k in kids:
            if k.start > cur:
                segs += self._flat(cur, k.start)
            segs += self.segments(k)
            cur = max(cur, k.end)
        if cur < node.end:
            segs += self._flat(cur, node.end)
        return segs

    def sealed(self, node: Node) -> list[tuple[int, int]]:
        """Spans of point / sub-point descendants of `node`."""
        out, stack = [], list(self.children.get(node.node_id, []))
        while stack:
            k = stack.pop()
            if k.type in SEALED_TYPES:
                out.append((k.start, k.end))
            stack.extend(self.children.get(k.node_id, []))
        return out

    def pack(self, segs: list[tuple[int, int]],
             sealed: list[tuple[int, int]] = ()) -> list[tuple[int, int]]:
        """Greedy packing; a merge is refused if the merged span would cut a sealed span
        (segments are element-atomic, but two fragments of different points must not join)."""
        out: list[tuple[int, int]] = []
        for s, e in segs:
            if out and self.fits(out[-1][0], e) and not self._cuts(out[-1][0], e, sealed):
                out[-1] = (out[-1][0], e)
            else:
                out.append((s, e))
        return out

    @staticmethod
    def _cuts(s: int, e: int, sealed) -> bool:
        return any(a < e and s < b and not (a <= s and e <= b) and not (s <= a and b <= e)
                   for a, b in sealed)

    def parent(self, unit: Node, s: int, e: int) -> Node:
        node = unit
        while True:
            nxt = next((k for k in self.children.get(node.node_id, [])
                        if k.start <= s and e <= k.end), None)
            if nxt is None:
                return node
            node = nxt

    def src(self, s: int, e: int) -> str:
        """'абз. 3-5', or 'с. 12, абз. 3-5' when paragraphs carry PDF pages (spec §2.1 example)."""
        a, b = bisect.bisect_right(self.starts, s), bisect.bisect_right(self.starts, e - 1)
        par = f"абз. {a}" if a == b else f"абз. {a}-{b}"
        pa = self.dt.pages[a - 1]
        pb = self.dt.page_ends[b - 1] if self.dt.page_ends else self.dt.pages[b - 1]
        if pa is None or pb is None:
            return par
        return f"{'с. ' + str(pa) if pa == pb else f'с. {pa}-{pb}'}, {par}"


def _cfg_attach_min_tokens() -> int:
    try:
        from graphrag import config
        return int(config.load("data").get("chunker", {}).get("attach_min_tokens", ATTACH_MIN_TOKENS))
    except (OSError, ValueError, TypeError, AttributeError):
        return ATTACH_MIN_TOKENS


def _attach(ck: _Chunker, spans: list[list], threshold: int) -> list[list]:
    """Merge tiny whole-turn chunks (< threshold tokens) with the adjacent whole-turn chunk of the
    same вопрос — the following turn first (chair intros precede the speech), else the previous —
    while the merged span fits max_tokens. Items: [start, end, unit, whole_turn, merged]."""
    out: list[list] = []
    run: list[list] = []

    def flush():
        i = 0
        while i < len(run):
            g = run[i]
            if ck.measure(g[0], g[1]) < threshold:
                if i + 1 < len(run) and ck.fits(g[0], run[i + 1][1]):
                    run[i:i + 2] = [[g[0], run[i + 1][1], g[2], True, True]]
                    continue
                if i > 0 and ck.fits(run[i - 1][0], g[1]):
                    run[i - 1:i + 1] = [[run[i - 1][0], g[1], run[i - 1][2], True, True]]
                    i -= 1
                    continue
            i += 1
        out.extend(run)
        run.clear()

    for it in spans:
        if not it[3]:
            flush()
            out.append(it)
            continue
        if run and run[-1][2].parent_id != it[2].parent_id:
            flush()
        run.append(it)
    flush()
    return out


def chunk_document(dt: DocText, nodes: list[Node], max_tokens: int,
                   count_tokens: Callable[[str], int], to_norm: Callable[[str], str],
                   attach_min_tokens: int | None = None) -> list[Chunk]:
    ck = _Chunker(dt, nodes, max_tokens, count_tokens, to_norm)
    by_id = {n.node_id: n for n in nodes}
    root = next(n for n in nodes if n.parent_id is None)
    threshold = _cfg_attach_min_tokens() if attach_min_tokens is None else attach_min_tokens
    units = sorted((n for n in nodes if is_unit(n, by_id)), key=lambda n: (n.start, n.end))
    spans: list[list] = []                      # [start, end, unit, whole_turn, merged]
    for u in units:
        packed = ck.pack(ck.segments(u), ck.sealed(u))
        whole = u.type == "выступление" and len(packed) == 1 and u.parent_id in by_id \
            and by_id[u.parent_id].type == "вопрос"
        spans += [[s, e, u, whole, False] for s, e in packed]
    if threshold > 0:
        spans = _attach(ck, spans, threshold)
    crumbs: dict[tuple[str, int], str] = {}
    chunks: list[Chunk] = []
    per_unit: dict[str, int] = defaultdict(int)
    for s, e, u, _, merged in spans:
        p = ck.parent(root, s, e) if merged else ck.parent(u, s, e)
        key = p.node_id if merged else u.node_id
        per_unit[key] += 1
        par = per_unit[key]
        text = dt.text[s:e]
        norm = to_norm(text)
        speakers: list[str] = []
        if merged:
            speakers = list(dict.fromkeys(k.title for k in ck.children.get(p.node_id, [])
                                          if k.type == "выступление" and s <= k.start and k.end <= e
                                          and k.title))
        cap = min(BREADCRUMB_MAX_TOKENS,
                  ENCODER_MAX_TOKENS - 2 - count_tokens(PASSAGE_PREFIX) - 1 - count_tokens(norm))
        if cap <= 0:
            crumb = ""
        elif merged:
            crumb = _breadcrumb(ck, by_id, p, count_tokens, to_norm, cap, extra=", ".join(speakers))
        else:
            if (u.node_id, cap) not in crumbs:
                crumbs[(u.node_id, cap)] = _breadcrumb(ck, by_id, u, count_tokens, to_norm, cap)
            crumb = crumbs[(u.node_id, cap)]
        while crumb and count_tokens(f"{PASSAGE_PREFIX}{crumb}\n{norm}") + 2 > ENCODER_MAX_TOKENS:
            crumb = crumb.split(None, 1)[1] if " " in crumb.strip() else ""   # tokenization is not additive
        chunks.append(Chunk(chunk_id=f"{p.node_id}#{par}", parent=p.node_id, par=par,
                            tokens=count_tokens(text), src=ck.src(s, e), text=text,
                            text_norm=norm, start=s, end=e,
                            index_text=f"{crumb}\n{norm}" if crumb else norm))
    return chunks


def _label(dt: DocText, n: Node) -> str:
    """Breadcrumb label of a node: its header marker + title for глава / § / статья
    ("3-боб. Умумий қоидалар", "21-модда. Мақсад"), title for документ / вопрос / выступление,
    "N." for a resolution item, "" otherwise (omitted)."""
    if n.type in _HEAD_RX:
        end = dt.text.find("\n\n", n.start, n.end)
        fp = dt.text[n.start:n.end if end < 0 else end]
        for rx in _HEAD_RX[n.type]:
            m = rx.match(fp)
            if m:
                return f"{fp[:m.start('rest')].strip()} {n.title}".strip()
        return n.title
    if n.type == "пункт":
        return f"{n.number}." if n.number else ""
    if n.type in ("документ", "вопрос", "выступление"):
        return n.title
    return ""


def _breadcrumb(ck: _Chunker, by_id: dict[str, Node], node: Node, count: Callable[[str], int],
                to_norm: Callable[[str], str], cap: int = BREADCRUMB_MAX_TOKENS, extra: str = "") -> str:
    """Titles from the document down to `node` (+ `extra`), " › "-joined and normalized; capped at
    `cap` tokens by dropping labels, then leading words, from the document side."""
    chain, n = [], node
    while n is not None:
        chain.append(n)
        n = by_id.get(n.parent_id) if n.parent_id else None
    labels = [lb for lb in (" ".join(_label(ck.dt, k).split()) for k in reversed(chain)) if lb]
    if extra:
        labels.append(extra)
    crumb = to_norm(" › ".join(labels))
    while len(labels) > 1 and count(crumb) > cap:
        labels.pop(0)
        crumb = to_norm(" › ".join(labels))
    if count(crumb) > cap:
        ws = crumb.split()
        lo, hi = 0, len(ws)                        # smallest number of leading words to drop
        while lo < hi:
            mid = (lo + hi) // 2
            if count(" ".join(ws[mid:])) <= cap:
                hi = mid
            else:
                lo = mid + 1
        crumb = " ".join(ws[lo:])
    return crumb
