"""y_ref.sources (IMPLEMENTATION.md §8.1: "<акт>, ст. N ч. M"): built by code from the vertices μ(C_ref), never by the
LLM, and resolvable back to exactly one graph vertex. Equal act titles are told apart by date, then doc_id."""
from __future__ import annotations

import re
from collections import Counter

ABBR = {"СТ": "ст.", "Ч": "ч.", "П": "п.", "ПП": "пп.", "ГЛ": "гл.", "§": "§", "ВОПР": "вопрос",
        "ВЫСТ": "выступление", "БЛ": "блок", "РЕКВ": "реквизиты", "ПРЕАМБ": "преамбула", "ЗАГ": "заголовок",
        "БЎЛ": "разд.", "ИЛОВА": "прил.", "ПОЗ": "поз."}
_SEG = re.compile(r"^(?P<k>[^.\d]+)\.(?P<n>.+)$")


def segment_text(seg: str) -> str:
    m = _SEG.match(seg)
    if not m or m["k"] not in ABBR:
        raise ValueError(f"unknown path segment {seg!r}")
    return f"{ABBR[m['k']]} {m['n']}"


class SourceIndex:
    def __init__(self, records):
        titles = {}
        for r in records:
            doc = next((n for n in r.structure if n.node_id == r.doc_id), None)
            titles[r.doc_id] = " ".join((r.meta.title or (doc.title if doc else "") or r.doc_id).split())
        dates = {r.doc_id: r.meta.date for r in records}
        c1 = Counter(titles.values())
        act = {d: (t if c1[t] == 1 else f"{t} ({dates[d] or d})") for d, t in titles.items()}
        c2 = Counter(act.values())
        self.act = {d: (a if c2[a] == 1 else f"{a} [{d}]") for d, a in act.items()}
        self._fmt: dict[str, str] = {}
        self._node: dict[str, str] = {}
        for r in records:
            for n in r.structure:
                segs = n.node_id.split("/")[1:]
                s = self.act[r.doc_id] + ("" if not segs else ", " + " ".join(segment_text(x) for x in segs))
                if s in self._node:
                    raise ValueError(f"source string collision {s!r}: {self._node[s]} / {n.node_id}")
                self._fmt[n.node_id], self._node[s] = s, n.node_id

    def format(self, node_id: str) -> str:
        return self._fmt[node_id]

    def resolve(self, s: str) -> str:
        return self._node[s]

    def sources_for(self, data, chunk_ids) -> list[str]:
        return sorted({self._fmt[data.node_ids[int(data.chunk_node[data.chunk_row[c]])]] for c in chunk_ids})
