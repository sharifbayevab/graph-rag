"""M7 pilot 7 (plan Task 7, Q1 / Q2): the joint old + hard struct ship -- hard rows as Pair rows, check_ship guards
(probe / Q-real / test rows refused, STRUCT gold only for variant struct), merged rankings, the spark allowlist of
data/m7/full. Toy data only (no stores, no model)."""
import hashlib
import json

import numpy as np
import pytest

from tests.scripts_import import load_script


def _hard(qid, qtype="count", split="train", gold=("STRUCT/D1",)):
    return {"qid": qid, "qtype": qtype, "template": f"{qtype}_t", "lang": "uz-Latn", "question": f"savol {qid}?",
            "answer": "3", "gold": list(gold), "docs": ["D1"], "gold_docs": ["D1"], "meta": {}, "split": split}


def _old(pid, split="train"):
    return {"id": pid, "split": split, "C_ref": ["D1/СТ.1#0"], "origin": "synthetic"}


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _ids(monkeypatch):
    import grl.hard.probe as pr
    import grl.train.data as td
    monkeypatch.setattr(td, "qreal_ids", lambda: frozenset({"qreal-1"}))
    monkeypatch.setattr(pr, "probe_ids", lambda: frozenset({"m7-probe"}))


def test_hard_pairs_keep_q1_q2_train_val_only(tmp_path):
    from graphrag.dataset.finalize import Pair
    from grl.hard.ship import HARD_ORIGIN, hard_pairs
    f = _write(tmp_path / "pj.jsonl", [_hard("m7-a"), _hard("m7-b", "version", "val", ("D1/СТ.2#0", "D1/СТ.3#0")),
                                       _hard("m7-t", split="test"), _hard("m7-c", "compare", "train")])
    rows, rep = hard_pairs(f)
    assert [r["id"] for r in rows] == ["m7-a", "m7-b"]
    assert rep["skipped"] == {"compare|train": 1, "count|test": 1}
    p = Pair.from_json(rows[1])                       # a valid §8.1 Pair
    assert p.C_ref == ["D1/СТ.2#0", "D1/СТ.3#0"] and p.origin == HARD_ORIGIN and p.pattern == "version"
    assert "qid" not in rows[0] and rows[0]["provenance"]["set"] == "hard"
    from grl.train.data import SplitGuardError
    with pytest.raises(SplitGuardError, match="probe"):
        hard_pairs(_write(tmp_path / "p.jsonl", [_hard("m7-probe")]))
    with pytest.raises(SplitGuardError, match="Q-real"):
        hard_pairs(_write(tmp_path / "q.jsonl", [_hard("qreal-1")]))
    with pytest.raises(SplitGuardError):
        hard_pairs(f, splits=("train", "test"))


def test_check_ship_refuses_probe_qreal_and_test_rows(tmp_path):
    from grl.hard.ship import hard_pair
    from grl.train.data import SplitGuardError
    from grl.train.pilot import check_ship
    for i, row in enumerate([hard_pair(_hard("m7-probe")), hard_pair(_hard("qreal-1")),
                             hard_pair(_hard("m7-t", split="test"))]):
        ship = tmp_path / f"s{i}"
        _write(ship / "pairs.jsonl", [row])
        with pytest.raises(SplitGuardError):
            check_ship(ship, {row["id"]}, None, "struct")


def test_struct_gold_only_for_variant_struct(tmp_path):
    from grl.hard.ship import hard_pair
    from grl.train.data import SplitGuardError
    from grl.train.pilot import check_ship
    ship = tmp_path / "ship"
    _write(ship / "pairs.jsonl", [_old("p1"), hard_pair(_hard("m7-a"))])
    check_ship(ship, {"p1", "m7-a"}, None, "struct")
    for v in (None, "full", "rerank"):
        with pytest.raises(SplitGuardError, match="STRUCT"):
            check_ship(ship, {"p1", "m7-a"}, None, v)
    ok = tmp_path / "ok"                               # chunk gold (Q2 / old): any variant
    _write(ok / "pairs.jsonl", [_old("p1"), hard_pair(_hard("m7-b", "version", gold=("D1/СТ.2#0",)))])
    check_ship(ok, {"p1", "m7-b"}, None, None)


def test_read_hard_pairs_and_build_rankings_family_hard(tmp_path):
    from grl.hard.ship import hard_pair, read_hard_pairs
    from grl.train.data import SplitGuardError
    f = _write(tmp_path / "h.jsonl", [hard_pair(_hard("m7-b")), hard_pair(_hard("m7-a", split="val"))])
    assert [p.id for p in read_hard_pairs(f)] == ["m7-b", "m7-a"]
    br = load_script("build_rankings")
    assert br.family_queries("hard", f) == [("m7-a|q", "savol m7-a?"), ("m7-b|q", "savol m7-b?")]
    old = {**hard_pair(_hard("x")), "origin": "synthetic"}
    for bad in ([old], [hard_pair(_hard("m7-t", split="test"))]):
        with pytest.raises(SplitGuardError):
            read_hard_pairs(_write(tmp_path / "b.jsonl", bad))


def _rank(root, keys, inputs):
    from graphrag.retrieval.rankings import ARRAYS, write_rankings
    n = len(keys)
    rng = np.random.default_rng(len(keys))
    arrays = {a: (rng.random((n, 4)).astype(np.float32) if a == "qvec" else
                  rng.integers(0, 9, (n, 3)).astype(np.int64) if a.endswith("_idx") else rng.random((n, 3)))
              for a in ARRAYS}
    write_rankings(root, [(k, f"t {k}") for k in keys], arrays, inputs)
    return root / "latest.json", arrays


def test_merge_rankings_copies_arrays_and_refuses_mismatch(tmp_path):
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.ship import MERGE_SAME, merge_rankings
    base = {k: "x" for k in MERGE_SAME}
    la, aa = _rank(tmp_path / "a", ["o2|q", "o1|q"], {**base, "family": "dataset", "candidates": "c1"})
    lb, ab = _rank(tmp_path / "b", ["h1|q"], {**base, "family": "hard", "candidates": "c2"})
    merge_rankings([la, lb], tmp_path / "m")
    m = load_rankings(tmp_path / "m" / "latest.json")
    assert m.keys() == ["h1|q", "o1|q", "o2|q"]
    assert np.array_equal(m.get("o2|q").qvec, aa["qvec"][0]) and np.array_equal(m.get("h1|q").lex, ab["lex_idx"][0])
    assert m.meta["inputs"]["family"] == "merged" and len(m.meta["inputs"]["merged_from"]) == 2
    lc, _ = _rank(tmp_path / "c", ["h2|q"], {**base, "encoder": "other"})
    with pytest.raises(ValueError, match="encoder"):
        merge_rankings([la, lc], tmp_path / "m2")
    with pytest.raises(ValueError, match="share"):
        merge_rankings([la, la], tmp_path / "m3")


def test_spark_ships_m7_full_with_the_manifest_variant(tmp_path):
    from grl.hard.ship import hard_pair
    sp = load_script("spark")
    d = "data/m7/full/abc"
    keep = [f"{d}/ship/{n}" for n in ("pairs.jsonl", "splits.json", "trajectories.jsonl", "rankings/latest.json")]
    drop = [f"{d}/ship/manifest.json", f"{d}/candidates.jsonl", f"{d}/rankings_hard/x.npy"]
    assert sp.select_files([], keep + drop, [f"{d}/ship"]) == sorted(keep)
    rows = [_old("p1"), hard_pair(_hard("m7-a"))]
    snap = tmp_path / d
    ship = snap / "ship"
    _write(snap / "candidates.jsonl", rows)
    _write(ship / "pairs.jsonl", rows)
    _write(ship / "trajectories.jsonl", [])
    _write(ship / "rankings/v1/queries.jsonl", [{"qkey": "p1|q", "text": "t"}, {"qkey": "m7-a|q", "text": "t"}])

    def manifest(variant):
        files = sorted(p for p in ship.rglob("*") if p.is_file() and p.name != "manifest.json")
        man = {"sha256": {str(p.relative_to(ship)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        if variant:
            man["variant"] = variant
        (ship / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    manifest("struct")
    assert sp.check_extra(tmp_path, f"{d}/ship") == f"{d}/ship"
    sp.check_rows(tmp_path, [f"{d}/ship/pairs.jsonl"], [f"{d}/ship"])
    manifest(None)
    with pytest.raises(SystemExit, match="ship check"):
        sp.check_extra(tmp_path, f"{d}/ship")
    _write(ship / "pairs.jsonl", rows + [hard_pair(_hard("m7-t", split="test"))])
    with pytest.raises(SystemExit, match="split"):
        sp.check_rows(tmp_path, [f"{d}/ship/pairs.jsonl"], [f"{d}/ship"])


def test_group_summary_splits_old_q1_q2():
    from types import SimpleNamespace as N
    from grl.train.evaluate import group_summary
    acts = [N(op="stop"), N(op="structure"), N(op="lexical")]
    items = [N(qkey="o|q", pattern="single", gold=frozenset({1})), N(qkey="c|q", pattern="count", gold=frozenset({9})),
             N(qkey="v|q", pattern="version", gold=frozenset({2, 3}))]
    eps = [N(qkey="o|q", actions=[2, 0]), N(qkey="c|q", actions=[1, 0]), N(qkey="v|q", actions=[2])]
    finals = [N(ctx_set={1}), N(ctx_set={9, 4}), N(ctx_set={2})]
    parts = [N(total=0.5), N(total=0.8), N(total=0.2)]
    g = group_summary(items, eps, finals, parts, acts)
    assert g["old"]["all_share"] == 1.0 and g["count"]["reward_mean"] == 0.8 and g["version"]["all_share"] == 0.0
    assert g["version"]["recall_mean"] == 0.5 and g["count"]["action_share"] == {"stop": 0.5, "structure": 0.5}
    assert group_summary(items[:1], eps[:1], finals[:1], parts[:1], acts) == {}
