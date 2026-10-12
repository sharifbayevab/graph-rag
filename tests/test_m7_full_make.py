# tests/test_m7_full_make.py -- M7 part 2 Task 4: full Q1 / Q2 generation -- R33 evidence-quoted Q2 gold, per-type
# per-root caps (R43), probe exclusion by qid, R32 before selection (R44), greedy split balance (R45), the singles
# share as a selection fixpoint, one Q2 item per (root, answer) (R48), frozen language shares (R47).
from collections import Counter

import pytest

from grl.hard.splits import assign_splits, gold_docs
from grl.hard.templates import (LANGS, _qid, count_items, evidence_gold, full_items, version_candidates,
                                 version_items)
from tests.scripts_import import load_script
from tests.test_m7_templates import _toy


# --- the brief's tests -------------------------------------------------------------------------------------------

def test_old_split_wins():
    items = [{"qid": "a", "docs": ["D1"], "gold_docs": ["D1", "D2"]}]
    out = assign_splits(items, {"D2": "test"}, seed=1)
    assert out["a"] == "test"


def test_conflicting_old_splits_dropped():
    items = [{"qid": "a", "docs": ["D1"], "gold_docs": ["D2"]}]
    assert assign_splits(items, {"D1": "train", "D2": "test"}, seed=1)["a"] is None


def test_new_components_ratio_and_determinism():
    items = [{"qid": f"q{i}", "docs": [f"N{i}"], "gold_docs": []} for i in range(1000)]
    a, b = assign_splits(items, {}, seed=7), assign_splits(items, {}, seed=7)
    assert a == b
    share = sum(v == "train" for v in a.values()) / 1000
    assert 0.55 < share < 0.65


def test_evidence_gold_keeps_only_quoted_chunks():
    edge = {"attrs": {"evidence": {"doc_id": "Q", "start": 100, "end": 140}}}
    spans = {"Q#1": ("Q", 0, 90), "Q#2": ("Q", 90, 200), "Q#3": ("Q", 200, 300)}
    assert evidence_gold(edge, ["Q#1", "Q#2", "Q#3"], spans) == ["Q#2"]


# --- splits: chained conflicts, components, order independence ------------------------------------------------------

def test_chained_conflict_drops_one_and_keeps_components_pure():
    """a joins D1 (train) with N; b joins N with D2 (test): each is fine alone, together the component would hold
    two old splits -- the later one (by qid) is dropped, the other keeps the old split of its component."""
    items = [{"qid": "b", "docs": ["N"], "gold_docs": ["D2"]}, {"qid": "a", "docs": ["D1"], "gold_docs": ["N"]}]
    out = assign_splits(items, {"D1": "train", "D2": "test"}, seed=1)
    assert out == {"a": "train", "b": None}
    assert assign_splits(list(reversed(items)), {"D1": "train", "D2": "test"}, seed=1) == out


def test_component_shares_one_split():
    items = [{"qid": f"q{i}", "docs": [f"N{i // 3}"], "gold_docs": [f"G{i // 3}"]} for i in range(300)]
    out = assign_splits(items, {}, seed=3)
    for k in range(100):
        assert len({out[f"q{3 * k + j}"] for j in range(3)}) == 1
    assert set(out.values()) == {"train", "val", "test"}


def test_old_split_reaches_through_new_docs():
    """Review Focus 2: a new question whose component holds an old test document lands in test, never train."""
    items = [{"qid": "a", "docs": ["NEW1"], "gold_docs": ["NEW2"]},
             {"qid": "b", "docs": ["NEW2"], "gold_docs": ["OLD"]}]
    out = assign_splits(items, {"OLD": "test"}, seed=1)
    assert out == {"a": "test", "b": "test"}


def test_gold_docs_from_struct_and_chunk_ids():
    """Ruling P2: STRUCT/<root> -> root; a chunk id -> its record's doc_id (not parsed from the id)."""
    assert gold_docs(["STRUCT/EXT/kodeks", "Q-1/СТ.1#1"], {"Q-1/СТ.1#1": "Q-1"}) == ["EXT/kodeks", "Q-1"]
    with pytest.raises(KeyError):
        gold_docs(["X/СТ.1#1"], {})


# --- R33 in version_items ----------------------------------------------------------------------------------------

def _toy_r33():
    """The part-1 toy with evidence spans: every source element owns two chunks (#1 at [0, 50), #2 at [50, 100) of its
    document); each edge quotes inside #2, except Q-2/СТ.2 -> EXT/kodeks/СТ.62, whose span hits neither chunk."""
    nodes, edges, mu = _toy()
    spans = {}
    for el, cs in list(mu.items()):
        doc = el.split("/")[0]
        mu[el] = [el + "#1", el + "#2"]
        spans[el + "#1"], spans[el + "#2"] = (doc, 0, 50), (doc, 50, 100)
    out = []
    for e in edges:
        if e["type"] == "AMENDS":
            doc = e["src"].split("/")[0]
            ev = {"doc_id": doc, "start": 60, "end": 70, "quote": "q"}
            if e["src"] == "Q-2/СТ.2":
                ev = {"doc_id": doc, "start": 500, "end": 510, "quote": "q"}
            e = {**e, "attrs": {**e["attrs"], "evidence": ev}}
        out.append(e)
    return nodes, out, mu, spans


def test_version_gold_is_quoted_chunks_only():
    nodes, edges, mu, spans = _toy_r33()
    st = Counter()
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, chunk_spans=spans, stats=st)
    assert items
    for it in items:
        assert all(g.endswith("#2") for g in it.gold)                   # #1 holds no quote: never gold
    wl = {it.meta["law"]: it.gold for it in items if it.template == "what_law_changed"}
    assert wl.get("Q-1") == ["Q-1/ГЛ.1/СТ.1#2", "Q-1/ГЛ.1/СТ.2#2"]
    assert "Q-2" not in wl                                              # Q-2/СТ.2's span misses its chunks
    assert not [it for it in items if any(a[0] == "Q-2" and a[3] == "EXT/kodeks/СТ.62" for a in it.answer)]
    assert st["r33_no_quote"] >= 1


def test_version_without_spans_keeps_part1_gold():
    nodes, edges, mu, _ = _toy_r33()
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100)
    assert any(g.endswith("#1") for it in items for g in it.gold)


def test_r33_no_quote_counted_per_candidate():
    nodes, edges, mu, spans = _toy_r33()
    st = Counter()
    version_candidates(nodes, edges, mu, None, st, chunk_spans=spans)
    assert st["r33_no_quote"] == st["r33_no_quote:what_law_changed"] + st["r33_no_quote:which_laws_article"] \
        + st["r33_no_quote:which_law_part"] + st["r33_no_quote:which_deleted_year"]


# --- shared per-root cap, probe exclusion -------------------------------------------------------------------------

def _toy_busy_root():
    """The toy plus 12 laws, each amending a different article of EXT/kodeks (2 source elements, 2 chunks): many Q2
    candidates on one root, several Q1 candidates on it too."""
    from tests.test_m7_templates import _am, _c, _doc, _el
    nodes, edges, mu = _toy()
    for i in range(12):
        d, art = f"Z-{i}", f"EXT/kodeks/СТ.{100 + i}"
        nodes += [_doc(d, f"Law Z{i}"), _el(art, "статья", None, True)]
        edges.append(_c("EXT/kodeks", art))
        for k in (1, 2):
            s = f"{d}/СТ.{k}"
            nodes.append(_el(s, "статья", d))
            edges += [_c(d, s), _am(s, art, f"2024-0{k}-{i + 10}", "edit")]
            mu[s] = [s + "#1"]
    return nodes, edges, mu


def test_per_type_caps_r43():
    """R43: Q1 <= 4 per root within Q1; Q2 <= 8 per root within Q2 (not 4, not shared)."""
    nodes, edges, mu = _toy_busy_root()
    c, v, sc, sv, rc_c, rc_v = full_items(nodes, edges, mu, 1, 100, 100)
    assert max(Counter(it.docs[0] for it in c).values()) <= 4 and rc_c == Counter(it.docs[0] for it in c)
    per_v = Counter(it.docs[0] for it in v)
    assert per_v["EXT/kodeks"] == 8 and max(per_v.values()) <= 8 and rc_v == per_v
    assert sv["cap_blocked"] >= 1                                       # more than 8 Q2 candidates on the root


def test_q1_filling_a_root_does_not_block_q2():
    nodes, edges, mu = _toy_busy_root()
    c, v, *_ = full_items(nodes, edges, mu, 1, 100, 100, count_cap=3)
    assert Counter(it.docs[0] for it in c)["EXT/kodeks"] == 3           # Q1 fills its cap on the root ...
    alone = version_items(nodes, edges, mu, 1, 100, max_per_root=8)
    assert [it.qid for it in v] == [it.qid for it in alone]             # ... and Q2 is exactly what it is alone
    assert Counter(it.docs[0] for it in v)["EXT/kodeks"] == 8


def test_shared_counter_still_honoured_when_passed():
    """root_counts stays a plain hook: a caller passing one counter gets one cap (full_items never does, R43)."""
    nodes, edges, mu = _toy()
    rc = Counter()
    c = count_items(nodes, edges, rng_seed=1, n=100, root_counts=rc)
    v = version_items(nodes, edges, mu, rng_seed=1, n=100, root_counts=rc)
    assert max(Counter(it.docs[0] for it in c + v).values()) <= 4


def test_probe_qid_excluded_in_every_language():
    nodes, edges, mu = _toy()
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100)
    gone = items[0]
    st = Counter()
    again = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, exclude={gone.qid}, stats=st)
    assert gone.qid not in {it.qid for it in again}
    key = (gone.template, gone.docs[0], tuple(sorted(gone.meta.items())))
    assert not [it for it in again if (it.template, it.docs[0], tuple(sorted(it.meta.items()))) == key]
    assert st["probe_excluded"] == 1
    c = count_items(nodes, edges, rng_seed=1, n=100)
    st = Counter()
    c2 = count_items(nodes, edges, rng_seed=1, n=100, exclude={c[0].qid}, stats=st)
    assert c[0].qid not in {it.qid for it in c2} and st["probe_excluded"] == 1


def test_qid_does_not_depend_on_seed():
    nodes, edges, mu = _toy()
    def key(it):
        return it.template, it.docs[0], tuple(sorted(it.meta.items())), it.lang
    a = {key(it): it.qid for it in count_items(nodes, edges, rng_seed=1, n=100)}
    b = {key(it): it.qid for it in count_items(nodes, edges, rng_seed=9, n=100)}
    assert set(a) & set(b) and all(b[k] == q for k, q in a.items() if k in b)
    assert set(LANGS) == {"ru", "uz-Cyrl", "uz-Latn"} and callable(_qid)


# --- R44: R32 before selection; R45: greedy rebalance of new components -----------------------------------------

def _chunk_doc(mu):
    return {c: c.split("/")[0] for v in mu.values() for c in v}


def _make(old, n_count=100, n_version=100, toy=None, **kw):
    from grl.hard.splits import make_full
    nodes, edges, mu = (toy or _toy)()
    return make_full(nodes, edges, mu, 1, n_count, n_version, old_splits=old, chunk_doc=_chunk_doc(mu), **kw)


def test_r44_conflicting_candidate_filtered_and_slot_refilled():
    rows, *_ = _make({}, n_count=0, n_version=2, toy=_toy_busy_root)
    assert len(rows) == 2
    first = next(r for r in rows if len(set(r["gold_docs"]) | {r["docs"][0]}) >= 2)
    docs = sorted(set(first["docs"]) | set(first["gold_docs"]))
    old = {docs[0]: "train", docs[1]: "test"}                     # its own documents now hold two old splits
    rows2, sc, sv, rounds = _make(old, n_count=0, n_version=2, toy=_toy_busy_root)
    assert len(rows2) == 2 and first["qid"] not in {r["qid"] for r in rows2}     # the slot is refilled
    assert sv["r32_conflict"] >= 1 and all(r["split"] for r in rows2)


def test_r44_chain_conflict_reselected_until_clean():
    """what_law_changed Q-1 and Q-2 on EXT/kodeks are each fine (Q-1 train, Q-2 test), together they chain the two
    old splits through EXT/kodeks: make_full drops candidates and reselects; nothing kept has split None."""
    old = {"Q-1": "train", "Q-2": "test"}
    rows, sc, sv, rounds = _make(old)
    assert rows and rounds >= 2 and sc["r32_chain"] + sv["r32_chain"] >= 1
    for r in rows:
        assert r["split"] in ("train", "val", "test")
        olds = {old[d] for d in r["docs"] + r["gold_docs"] if d in old}
        assert not olds or olds == {r["split"]}
    sing = [r for r in rows if r["qtype"] == "version" and len(r["gold"]) == 1]
    assert len(sing) <= 0.2 * sum(r["qtype"] == "version" for r in rows)


def test_r45_new_components_fill_the_deficit():
    """60 % of the items inherit train: the new components go to val / test, combined within ±5 pp of 60/20/20."""
    items = [{"qid": f"o{i}", "docs": [f"O{i}"], "gold_docs": []} for i in range(60)]
    items += [{"qid": f"n{i}", "docs": [f"N{i}"], "gold_docs": []} for i in range(40)]
    out = assign_splits(items, {f"O{i}": "train" for i in range(60)}, seed=20261003)
    new = Counter(out[f"n{i}"] for i in range(40))
    assert new["val"] + new["test"] >= 36
    tot = Counter(out.values())
    for s, t in (("train", 60), ("val", 20), ("test", 20)):
        assert abs(tot[s] - t) <= 5


def test_r45_balance_is_per_type_components_global():
    """Q1 inherits train heavily, Q2 not at all: each type is balanced on its own; a component holding both types
    still has one split."""
    items = [{"qid": f"c{i}", "qtype": "count", "docs": [f"O{i}"], "gold_docs": []} for i in range(60)]
    items += [{"qid": f"d{i}", "qtype": "count", "docs": [f"N{i}"], "gold_docs": []} for i in range(40)]
    items += [{"qid": f"v{i}", "qtype": "version", "docs": [f"V{i}"], "gold_docs": []} for i in range(100)]
    items += [{"qid": "x1", "qtype": "count", "docs": ["S"], "gold_docs": []},
              {"qid": "x2", "qtype": "version", "docs": ["S"], "gold_docs": []}]
    out = assign_splits(items, {f"O{i}": "train" for i in range(60)}, seed=20261003)
    for t in ("count", "version"):
        c = Counter(out[it["qid"]] for it in items if it.get("qtype") == t)
        n = sum(c.values())
        for s, r in (("train", .6), ("val", .2), ("test", .2)):
            assert abs(c[s] / n - r) <= 0.05, (t, c)
    assert out["x1"] == out["x2"]


def _toy_singles(n_roots=8):
    """part-1 single-cap toy: each root has one part amended by one law (|gold| = 1) + 3 article amendments."""
    from tests.test_m7_templates import _am, _c, _doc, _el
    nodes, edges, mu = _toy()
    for i in range(n_roots):
        r = f"EXT/act{i}"
        nodes += [_doc(r, f"Act {i}", ext=True), _el(f"{r}/СТ.5", "статья", None, True),
                  _el(f"{r}/СТ.5/Ч.1", "часть", None, True), _doc(f"Z-{i}", f"Law Z{i}")]
        edges += [_c(r, f"{r}/СТ.5"), _c(f"{r}/СТ.5", f"{r}/СТ.5/Ч.1")]
        for k in range(4):
            s = f"Z-{i}/СТ.{k + 1}"
            nodes.append(_el(s, "статья", f"Z-{i}"))
            edges += [_c(f"Z-{i}", s), _am(s, f"{r}/СТ.5/Ч.1" if k == 0 else f"{r}/СТ.5", "2024-01-01",
                                           "edit" if k == 0 else "add")]
            mu[s] = [s + "#1"]
    return nodes, edges, mu


def test_singles_share_is_a_selection_fixpoint():
    """I1: n = 100 slots, far fewer fill: the share is met by the selection itself (floor(0.2 * selected)), not by a
    trim after it -- so a re-run with the same max_single selects exactly the same items."""
    from grl.hard import templates as T
    nodes, edges, mu = _toy_singles()
    st = Counter()
    items = version_items(nodes, edges, mu, rng_seed=2, n=100, stats=st)
    ones = sum(len(it.gold) == 1 for it in items)
    assert len(items) < 100 and ones <= T._share_floor(len(items)) and st["selected"] == len(items)
    assert st["singles_rounds"] >= 2                                   # floor(0.2 * 100) = 20 did not hold
    cands = T.version_candidates(nodes, edges, mu, None, Counter())
    for c in cands:
        T._check_langs(c, Counter())
    again = T._select(cands, 100, T.random.Random(2), T.OLD_SET_LANG_COUNTS, 4, lambda c: c.key,
                      T._share_floor(len(items)), "version")
    assert [it.qid for it in again] == [it.qid for it in items]        # nothing was removed after selection


def test_make_full_singles_assert_can_fail(monkeypatch):
    from grl.hard import splits as S
    from grl.hard.templates import HardItem
    def fake(*a, **k):
        v = [HardItem(f"m7-version-{i}", "version", "which_law_part", "ru", "q", [], [f"Q-1/СТ.{i}#1"], ["Q-1"], {})
             for i in range(5)]
        return [], v, Counter(), Counter(), Counter(), Counter()
    monkeypatch.setattr("grl.hard.templates.full_items", fake)
    with pytest.raises(AssertionError, match="single"):
        S.make_full([], [], {}, 1, 0, 5, old_splits={}, chunk_doc={f"Q-1/СТ.{i}#1": "Q-1" for i in range(5)})


def test_r48_one_item_per_root_and_answer():
    """what_law_changed Q-2 and which_laws_article 62 on EXT/kodeks have the same answer (Q-2's two edges): one is
    kept, the other counted as dup_answer and its slot refilled."""
    nodes, edges, mu = _toy()
    plain = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100)
    keys = [(it.docs[0], str(it.answer)) for it in plain]
    assert len(keys) > len(set(keys))                                  # the toy has a duplicate without R48
    st = Counter()
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, dedup_answers=True, stats=st)
    keys = [(it.docs[0], str(it.answer)) for it in items]
    assert len(keys) == len(set(keys)) and st["dup_answer"] >= 1
    c, v, *_ = full_items(nodes, edges, mu, 1, 0, 100)
    assert len({(it.docs[0], str(it.answer)) for it in v}) == len(v)    # full_items applies R48


def test_r47_lang_shares_frozen(tmp_path, monkeypatch):
    hp = load_script("hard_probe")
    monkeypatch.setattr(hp, "lang_shares", lambda: {"ru": 2, "uz-Cyrl": 3, "uz-Latn": 1})
    assert hp.frozen_lang_shares(tmp_path) == ({"ru": 2, "uz-Cyrl": 3, "uz-Latn": 1}, "computed")
    monkeypatch.setattr(hp, "lang_shares", lambda: {"ru": 99, "uz-Cyrl": 1, "uz-Latn": 1})   # the old set grew
    assert hp.frozen_lang_shares(tmp_path) == ({"ru": 2, "uz-Cyrl": 3, "uz-Latn": 1}, "frozen")
    (tmp_path / hp.LANG_SHARES_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="malformed"):
        hp.frozen_lang_shares(tmp_path)


def test_full_make_requires_exclude_file(tmp_path):
    hp = load_script("hard_probe")
    with pytest.raises(SystemExit, match="exclude"):
        hp.read_exclude(tmp_path / "ids.json")
    (tmp_path / "ids.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="exclude"):
        hp.read_exclude(tmp_path / "ids.json")
    (tmp_path / "ids.json").write_text('["m7-count-x"]', encoding="utf-8")
    assert hp.read_exclude(tmp_path / "ids.json") == {"m7-count-x"}
