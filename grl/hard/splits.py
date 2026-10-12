"""M7 part 2: splits of the new hard questions by connected document components (spec §2 «Сплиты»; ruling R32).

A question's documents are its question documents (`docs`) and the documents of its gold (`gold_docs`: STRUCT/<root>
-> root, a chunk id -> its record's doc_id; ruling P2, `gold_docs()` below). Questions sharing a document form one
component; a component goes to one split as a whole.

R32: (1) a component holding a document of the old set (data/gold/splits.json) takes that document's old split;
(2) a question that would join documents of different old splits is dropped (None); (3) the other components are cut
60 / 20 / 20 deterministically (seed).

(2) is applied greedily over the questions in qid order: a question is added to the union-find only if the component
it would form holds at most one old split; otherwise it is dropped and its documents are not joined. So a chain
D1 (train) - a - N - b - D2 (test) drops the later question, never the whole chain, and every kept component carries at
most one old split. The result does not depend on the order of `items`.

(3) R45: the new components (no old document) are walked in sorted order (by their smallest document) and each goes
to the split with the largest deficit, so the combined split (inherited + new) comes as close to `ratios` as it can.
The deficit is per question type (item "qtype"; one type if absent): target_t[s] = ratio[s] * (items of type t kept),
and a component holding k_t items of each type goes to argmax_s sum_t k_t * (target_t[s] - current_t[s]) (ties: train,
val, test). Components are global over the types, so a document never sits in two splits. `seed` is kept for the
interface; the walk is deterministic without it.

make_full (R44): R32 runs before selection -- a candidate whose own documents hold two old splits is dropped before
the cap / singles selection, so its slot is refilled. A conflict that only appears between selected items (a chain
through a shared new document) drops those candidates and the selection is re-run, until assign_splits drops
nothing."""
from __future__ import annotations

from collections import Counter, defaultdict

from grl.hard.checks import STRUCT_PREFIX

SPLITS = ("train", "val", "test")


def gold_docs(gold, chunk_doc: dict) -> list[str]:
    """Ruling P2: the documents of gold fragment ids -- STRUCT/<root> -> <root>, STRUCT/<article> -> its document
    (Task 6c, grl.env.structure.struct_doc), a chunk id -> chunk_doc[id] (the records' doc_id; KeyError for an
    unknown chunk, never a guess from the id string)."""
    from grl.env.structure import struct_doc
    return sorted({struct_doc(g) if g.startswith(STRUCT_PREFIX) else chunk_doc[g] for g in gold})


class _UF:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.old: dict[str, str | None] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = min(ra, rb), max(ra, rb)
            self.parent[hi] = lo
            self.old[lo] = self.old.get(lo) or self.old.get(hi)


def assign_splits(items, old_splits: dict[str, str], seed: int,
                  ratios=(0.6, 0.2, 0.2)) -> dict[str, str | None]:
    """qid -> "train" | "val" | "test", or None for a question dropped by R32 (2). `items`: dicts with "qid", "docs",
    "gold_docs"; `old_splits`: doc_id -> old split (data/gold/splits.json)."""
    if abs(sum(ratios) - 1.0) > 1e-9 or len(ratios) != len(SPLITS):
        raise ValueError(f"ratios must be 3 shares summing to 1, got {ratios}")
    bad = sorted({s for s in old_splits.values() if s not in SPLITS})
    if bad:
        raise ValueError(f"old splits outside {SPLITS}: {bad}")
    uf, out, kept = _UF(), {}, {}
    for it in sorted(items, key=lambda x: x["qid"]):
        q = it["qid"]
        if q in out:
            raise ValueError(f"duplicate qid {q}")
        ds = sorted(set(it["docs"]) | set(it["gold_docs"]))
        if not ds:
            raise ValueError(f"{q}: no documents")
        for d in ds:
            r = uf.find(d)
            if d in old_splits:
                uf.old[r] = uf.old.get(r) or old_splits[d]
        olds = {uf.old.get(uf.find(d)) for d in ds} | {old_splits.get(d) for d in ds}
        olds.discard(None)
        if len(olds) > 1:
            out[q] = None
            continue
        for d in ds[1:]:
            uf.union(ds[0], d)
        out[q], kept[q] = "", ds[0]
    comps = defaultdict(list)
    for q, d in kept.items():
        comps[uf.find(d)].append(q)
    qtype = {it["qid"]: it.get("qtype", "") for it in items}
    cur = defaultdict(Counter)                        # type -> split -> items placed (inherited first)
    new = []
    for root in sorted(comps):
        old = uf.old.get(root)
        if old:
            for q in comps[root]:
                out[q] = old
                cur[qtype[q]][old] += 1
        else:
            new.append(root)
    n_type = Counter(qtype[q] for q in kept)
    for root in new:                                  # sorted: comps is walked by its sorted roots
        k = Counter(qtype[q] for q in comps[root])
        split = max(SPLITS, key=lambda s: (sum(k[t] * (ratios[SPLITS.index(s)] * n_type[t] - cur[t][s]) for t in k),
                                           -SPLITS.index(s)))
        for q in comps[root]:
            out[q] = split
        for t, m in k.items():
            cur[t][split] += m
    return out


def make_full(nodes, edges, mu_inv, seed: int, n_count: int, n_version: int, *, old_splits: dict,
              chunk_doc: dict, max_rounds: int = 50, **kw):
    """R43 + R44 + R45: Q1 / Q2 (templates.full_items, per-type caps) with R32 applied before selection, then
    assign_splits. Returns (rows, count_stats, version_stats, rounds); rows are HardItem dicts + gold_docs + split
    (Q1 first, selection order). `kw`: struct_texts, chunk_texts, chunk_spans, lang_shares, exclude."""
    from grl.hard.templates import SINGLE_SHARE, full_items, item_key
    chained: set = set()

    def pre_drop(c):
        docs = set(c.docs) | set(gold_docs(c.gold, chunk_doc))
        if len({old_splits[d] for d in docs if d in old_splits}) > 1:
            return "r32_conflict"
        if c.key in chained:
            return "r32_chain"
        return None

    for rounds in range(1, max_rounds + 1):
        count, ver, sc, sv, _, _ = full_items(nodes, edges, mu_inv, seed, n_count, n_version, pre_drop=pre_drop, **kw)
        rows = [{**it.to_json(), "gold_docs": gold_docs(it.gold, chunk_doc)} for it in count + ver]
        split = assign_splits(rows, old_splits, seed)
        bad = [r for r in rows if split[r["qid"]] is None]
        if not bad:
            break
        chained |= {item_key(r) for r in bad}
    else:
        raise RuntimeError(f"R32 chain conflicts not resolved in {max_rounds} rounds")
    for r in rows:
        r["split"] = split[r["qid"]]
    vs = [r for r in rows if r["qtype"] == "version"]
    if sum(len(r["gold"]) == 1 for r in vs) > SINGLE_SHARE * len(vs):
        raise AssertionError("Q2 single share > 20 % after selection")      # R44: no post-selection drop left
    return rows, sc, sv, rounds


# --- Q3 (plan M7 part 2, Task 6): splits consistent with the old set AND the Q1 / Q2 splits -----------------------

def fixed_doc_splits(old_splits: dict[str, str], rows) -> dict[str, str]:
    """doc -> split that a new question must respect: the old set (data/gold/splits.json) plus every document (docs +
    gold_docs) of the already split Q1 / Q2 rows. A document with two splits (old vs Q1 / Q2, or two Q1 / Q2 rows) is a
    loud ValueError: the same document is always in the same split."""
    out = dict(old_splits)
    src = {d: "old" for d in old_splits}
    for r in rows:
        s = r.get("split")
        if s not in SPLITS:
            raise ValueError(f"{r.get('qid')}: split {s!r} not in {SPLITS}")
        for d in sorted(set(r["docs"]) | set(r["gold_docs"])):
            if d in out and out[d] != s:
                raise ValueError(f"document {d}: split {out[d]} ({src[d]}) vs {s} ({r.get('qid')})")
            out.setdefault(d, s)
            src.setdefault(d, r.get("qid"))
    return out


def assign_q3_splits(items, old_splits: dict[str, str], q12_rows, seed: int,
                     ratios=(0.6, 0.2, 0.2)) -> dict[str, str | None]:
    """R32 for Q3 against both the old set and the Q1 / Q2 splits: assign_splits with the fixed document splits of
    fixed_doc_splits(old, Q1 / Q2) as the «old» map -- a component touching such a document inherits its split, a
    question joining two fixed splits is dropped (None), the rest is cut by deficit (R45)."""
    return assign_splits(items, fixed_doc_splits(old_splits, q12_rows), seed, ratios)
