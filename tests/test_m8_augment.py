# tests/test_m8_augment.py -- M7 pilot 8, ruling R55: Q2 train-only rows under a per-act cap of 16 (template wording),
# R32 guards (train-only components, existing splits unchanged), no repeat of a Task 4 item / answer, marked rows.
from collections import Counter

import pytest

from grl.hard.augment import AUG_MARK, augment_q2, train_only_filter
from tests.test_m7_full_make import _chunk_doc, _make, _toy_busy_root


def _base(old):
    rows, *_ = _make(old, n_count=3, n_version=8, toy=_toy_busy_root)
    return rows


def test_added_rows_are_marked_train_and_new():
    nodes, edges, mu = _toy_busy_root()
    old = {"EXT/kodeks": "train"}
    base = _base(old)
    rows, rep = augment_q2(nodes, edges, mu, 1, 100, old_splits=old, chunk_doc=_chunk_doc(mu), q12_rows=base,
                           q3_rows=[], cap=16)
    assert rows and rep["n_added"] == len(rows)
    assert all(r["split"] == "train" and r["meta"]["augment"] == AUG_MARK for r in rows)
    assert not {r["qid"] for r in rows} & {r["qid"] for r in base}
    per = Counter(r["docs"][0] for r in base if r["qtype"] == "version") + Counter(r["docs"][0] for r in rows)
    assert max(per.values()) <= 16


def test_cap_counts_existing_rows():
    nodes, edges, mu = _toy_busy_root()
    old = {"EXT/kodeks": "train"}
    base = _base(old)
    have = Counter(r["docs"][0] for r in base if r["qtype"] == "version")["EXT/kodeks"]
    rows, _ = augment_q2(nodes, edges, mu, 1, 100, old_splits=old, chunk_doc=_chunk_doc(mu), q12_rows=base,
                         q3_rows=[], cap=have + 1)
    assert Counter(r["docs"][0] for r in rows)["EXT/kodeks"] <= 1


def test_val_component_gives_no_rows():
    """Every candidate touches EXT/kodeks; its old split val makes every candidate a val question: nothing added."""
    nodes, edges, mu = _toy_busy_root()
    old = {"EXT/kodeks": "val"}
    rows, rep = augment_q2(nodes, edges, mu, 1, 100, old_splits=old, chunk_doc=_chunk_doc(mu),
                           q12_rows=_base(old), q3_rows=[], cap=16)
    assert not [r for r in rows if r["docs"][0] == "EXT/kodeks"]
    assert rep["stats"].get("aug_val_or_test_component", 0) >= 1


def test_filter_reasons():
    class C:
        def __init__(self, docs, gold, key=("t", "R", "{}"), root="R", answer=()):
            self.docs, self.gold, self.key, self.root, self.answer = docs, gold, key, root, list(answer)
    cd = {"A#1": "A", "B#1": "B", "N#1": "N"}
    f = train_only_filter({"A": "train", "B": "test"}, cd, {("t", "X", "{}")}, {("R", '[["z"]]')})
    assert f(C(["A"], ["A#1"])) is None
    assert f(C(["A"], ["B#1"])) == "aug_r32_conflict"
    assert f(C(["B"], ["B#1"])) == "aug_val_or_test_component"
    assert f(C(["N"], ["N#1"])) == "aug_all_docs_new"
    assert f(C(["A"], ["A#1"], key=("t", "X", "{}"))) == "aug_existing_item"
    assert f(C(["A"], ["A#1"], answer=[["z"]])) == "aug_dup_existing_answer"


def test_split_of_existing_row_never_moves(monkeypatch):
    """A broken filter that lets a val-touching candidate through is caught by the R32 re-check."""
    import grl.hard.augment as aug
    nodes, edges, mu = _toy_busy_root()
    old = {"EXT/kodeks": "val"}
    monkeypatch.setattr(aug, "train_only_filter", lambda *a, **k: (lambda c: None))
    with pytest.raises(ValueError):
        aug.augment_q2(nodes, edges, mu, 1, 100, old_splits=old, chunk_doc=_chunk_doc(mu), q12_rows=_base(old),
                       q3_rows=[], cap=16)
