"""M7 Task 7: the probe snapshot -- probe rows (split "probe"), their guards, the probe ship and its checks, the
STRUCT-free ranking check and the gold-in-top-k rates. Toy data only (no stores, no model)."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import grl.hard.probe as _pr
from tests.scripts_import import load_script

_REAL_PROBE_IDS = _pr.probe_ids                  # before the autouse fixture below replaces it


def _item(qid, qtype="count", gold=("STRUCT/D1",), split=None):
    r = {"qid": qid, "qtype": qtype, "template": f"{qtype}_t", "lang": "uz-Latn", "question": f"savol {qid}?",
         "answer": "3", "gold": list(gold), "docs": ["D1"], "meta": {}}
    if split is not None:
        r["split"] = split
    return r


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _no_real_ids(monkeypatch):
    import grl.hard.probe as pr
    import grl.train.data as td
    monkeypatch.setattr(td, "qreal_ids", lambda: frozenset({"qreal-1"}))
    monkeypatch.setattr(pr, "probe_ids", lambda: frozenset({"m7-a", "m7-b", "m7-c"}))


def test_probe_rows_get_split_probe_and_must_match_ids(tmp_path):
    from grl.hard.probe import PROBE_SPLIT, probe_rows
    a = _write(tmp_path / "count.jsonl", [_item("m7-b"), _item("m7-a")])
    b = _write(tmp_path / "version.jsonl", [_item("m7-c", "version", gold=("D1/СТ.1#0",))])
    rows = probe_rows([a, b], ["m7-a", "m7-b", "m7-c"])
    assert [r["qid"] for r in rows] == ["m7-a", "m7-b", "m7-c"] and {r["split"] for r in rows} == {PROBE_SPLIT}
    with pytest.raises(ValueError, match="ids.json"):
        probe_rows([a], ["m7-a", "m7-b", "m7-c"])                # a probe id with no row
    with pytest.raises(ValueError, match="duplicate"):
        probe_rows([a, a, b], ["m7-a", "m7-b", "m7-c"])
    c = _write(tmp_path / "x.jsonl", [_item("m7-a", split="train")])
    with pytest.raises(ValueError, match="split"):
        probe_rows([c], ["m7-a"])


def test_read_probe_rows_refuses_non_probe_and_qreal(tmp_path):
    from grl.hard.probe import read_probe_rows
    from grl.train.data import SplitGuardError
    ok = _write(tmp_path / "p.jsonl", [_item("m7-a", split="probe")])
    assert [r["qid"] for r in read_probe_rows(ok)] == ["m7-a"]
    for bad in ([_item("m7-a", split="test")], [_item("qreal-1", split="probe")], [_item("zz", split="probe")]):
        with pytest.raises(SplitGuardError):
            read_probe_rows(_write(tmp_path / "b.jsonl", bad))


def test_training_loaders_refuse_a_probe_file(tmp_path):
    from grl.train.data import SplitGuardError, check_variant_file, load_items, load_rank_items
    from grl.train.expert import expert_inputs
    p = _write(tmp_path / "pairs.jsonl", [_item("m7-a", split="probe")])
    data = SimpleNamespace(chunk_row={})
    for f in (lambda: load_items(p, data, ("train", "val")), lambda: load_rank_items(p, data, ("train",)),
              lambda: expert_inputs(p), lambda: check_variant_file(p, "struct", None, True)):
        with pytest.raises(SplitGuardError, match="probe"):
            f()


def test_load_probe_items_maps_struct_gold_past_the_base_rows(tmp_path):
    from grl.hard.probe import load_probe_items
    p = _write(tmp_path / "pairs.jsonl", [_item("m7-a", split="probe"),
                                          _item("m7-b", "version", gold=("c0", "c1"), split="probe")])
    data = SimpleNamespace(chunk_row={"c0": 0, "c1": 1, "STRUCT/D1": 2}, n_base_chunks=2)
    its = load_probe_items(p, data)
    assert [(i.pid, i.qkey, i.split, i.pattern, sorted(i.gold)) for i in its] == [
        ("m7-a", "m7-a|q", "probe", "count", [2]), ("m7-b", "m7-b|q", "probe", "version", [0, 1])]
    with pytest.raises(ValueError, match="STRUCT"):            # base EnvData has no STRUCT rows
        load_probe_items(p, SimpleNamespace(chunk_row={"c0": 0, "c1": 1}, n_base_chunks=2))


def _ship(tmp_path, rows, keys):
    ship = tmp_path / "ship"
    _write(ship / "pairs.jsonl", rows)
    (ship / "splits.json").write_text('{"probe": %d}\n' % len(rows), encoding="utf-8")
    _write(ship / "rankings/v1/queries.jsonl", [{"qkey": k, "text": "t"} for k in keys])
    (ship / "rankings/latest.json").write_text('{"dir": "x"}\n', encoding="utf-8")
    return ship


def test_check_probe_ship(tmp_path):
    from grl.hard.probe import check_probe_ship
    from grl.train.data import SplitGuardError
    allowed = {"m7-a", "m7-b"}
    rows = [_item("m7-a", split="probe"), _item("m7-b", split="probe")]
    check_probe_ship(_ship(tmp_path / "1", rows, ["m7-a|q", "m7-b|q"]), allowed)
    for i, (rs, ks) in enumerate([([_item("m7-a", split="train")], ["m7-a|q"]),     # not a probe row
                                  ([_item("m7-c", split="probe")], ["m7-c|q"]),      # not an allowed id
                                  (rows, ["m7-a|q", "old-1|q"]),                      # an old-set ranking key
                                  (rows, ["m7-a|cyr"])]):                             # a Q-real-shaped key
        with pytest.raises(SplitGuardError):
            check_probe_ship(_ship(tmp_path / f"b{i}", rs, ks), allowed)
    s = _ship(tmp_path / "te", rows, ["m7-a|q"])
    _write(s / "test_eval/test_eval.jsonl", rows)
    with pytest.raises(SplitGuardError, match="only"):
        check_probe_ship(s, allowed)
    s = _ship(tmp_path / "g", [_item("m7-a", gold=(), split="probe")], ["m7-a|q"])
    with pytest.raises(SplitGuardError, match="gold"):
        check_probe_ship(s, allowed)


def test_old_set_check_ship_refuses_probe_ids(tmp_path):
    from grl.train.data import SplitGuardError
    from grl.train.pilot import check_ship
    ship = tmp_path / "ship"
    _write(ship / "pairs.jsonl", [{"id": "m7-a", "split": "train"}])
    with pytest.raises(SplitGuardError, match="probe"):
        check_ship(ship, {"m7-a"})
    ship2 = tmp_path / "ship2"
    _write(ship2 / "pairs.jsonl", [{"id": "p1", "split": "train"}])
    _write(ship2 / "test_eval/test_eval.jsonl", [{"id": "m7-b", "split": "test", "provenance": {"test_eval": True}}])
    with pytest.raises(SplitGuardError, match="probe"):
        check_ship(ship2, {"p1"}, {"m7-b"})


class _Ranks:
    def __init__(self, d):
        self.d = d

    def keys(self):
        return sorted(self.d)

    def get(self, k):
        lex, vec = self.d[k]
        return SimpleNamespace(lex=np.asarray(lex), vec=np.asarray(vec), title=np.zeros(0, np.int64))


def test_check_rank_rows_refuses_struct_rows_and_empty_channels():
    from grl.hard.probe import check_rank_rows
    rep = check_rank_rows(_Ranks({"a|q": ([0, 1], [1, 0])}), ["a|q"], n_base=2)
    assert rep == {"n": 1, "empty_lex": 0, "empty_vec": 0, "rows_ge_n_base": 0, "max_row": 1}
    with pytest.raises(ValueError, match="n_base"):
        check_rank_rows(_Ranks({"a|q": ([0, 2], [1])}), ["a|q"], n_base=2)
    with pytest.raises(ValueError, match="empty"):
        check_rank_rows(_Ranks({"a|q": ([], [1])}), ["a|q"], n_base=2)
    with pytest.raises(ValueError, match="missing"):
        check_rank_rows(_Ranks({}), ["a|q"], n_base=2)


def test_gold_in_top_rates_per_type():
    from grl.hard.probe import gold_in_top
    rows = [_item("m7-a"), _item("m7-b", "version", gold=("c1", "c2")), _item("m7-c", "version", gold=("c0",))]
    ids = ["c0", "c1", "c2"]
    ranks = _Ranks({"m7-a|q": ([0, 1, 2], [0, 1, 2]), "m7-b|q": ([1], [0]), "m7-c|q": ([2], [0])})
    rep = gold_in_top(ranks, rows, ids, k=50)
    assert rep["count"] == {"n": 1, "lex_any": 0.0, "vec_any": 0.0, "lex_all": 0.0, "vec_all": 0.0,
                            "lex_recall": 0.0, "vec_recall": 0.0}
    assert rep["version"]["n"] == 2 and rep["version"]["lex_any"] == 0.5 and rep["version"]["vec_any"] == 0.5
    assert rep["version"]["lex_recall"] == 0.25 and rep["version"]["vec_recall"] == 0.5
    assert rep["version"]["vec_all"] == 0.5 and rep["version"]["lex_all"] == 0.0


def test_build_rankings_probe_queries(tmp_path):
    from grl.hard.probe import probe_queries
    rows = [_item("m7-b", split="probe"), _item("m7-a", split="probe")]
    assert probe_queries(rows) == [("m7-a|q", "savol m7-a?"), ("m7-b|q", "savol m7-b?")]


def test_spark_probe_ship_allowlist_rows_and_extra(tmp_path, monkeypatch):
    sp = load_script("spark")
    d = "data/m7/probe/abc"
    keep = [f"{d}/ship/{n}" for n in ("pairs.jsonl", "splits.json", "rankings/latest.json", "rankings/v1/qvec.npy")]
    drop = [f"{d}/ship/manifest.json", f"{d}/pairs.jsonl", f"{d}/parity/x.npy"]
    assert sp.select_files([], keep + drop, [f"{d}/ship"]) == sorted(keep)
    (tmp_path / d / "ship").mkdir(parents=True)
    f = tmp_path / d / "ship/pairs.jsonl"
    f.write_text(json.dumps(_item("m7-a", split="probe")) + "\n", encoding="utf-8")
    q = tmp_path / d / "ship/rankings/v1/queries.jsonl"          # ranking queries carry no split: fine
    _write(q, [{"qkey": "m7-a|q", "text": "t"}])
    sp.check_rows(tmp_path, [f"{d}/ship/pairs.jsonl", f"{d}/ship/rankings/v1/queries.jsonl"], [f"{d}/ship"])
    f.write_text(json.dumps({k: v for k, v in _item("m7-a").items()}) + "\n", encoding="utf-8")   # no split
    with pytest.raises(SystemExit, match="probe"):
        sp.check_rows(tmp_path, [f"{d}/ship/pairs.jsonl"], [f"{d}/ship"])
    f.write_text(json.dumps(_item("m7-a", split="train")) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="probe"):
        sp.check_rows(tmp_path, [f"{d}/ship/pairs.jsonl"], [f"{d}/ship"])
    # an old-set snapshot never carries probe rows
    o = "data/m5/pilot/x/ship"
    _write(tmp_path / o / "pairs.jsonl", [_item("m7-a", split="probe")])
    with pytest.raises(SystemExit):
        sp.check_rows(tmp_path, [f"{o}/pairs.jsonl"], [o])
    # check_extra: a probe ship dir whose manifest verifies and whose rows / keys pass check_probe_ship
    ship = _ship(tmp_path / d, [_item("m7-a", split="probe")], ["m7-a|q"])
    files = sorted(p for p in ship.rglob("*") if p.is_file())
    man = {str(p.relative_to(ship)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ship / "manifest.json").write_text(json.dumps({"sha256": man}), encoding="utf-8")
    assert sp.check_extra(tmp_path, f"{d}/ship") == f"{d}/ship"
    _write(ship / "rankings/v1/queries.jsonl", [{"qkey": "old-1|q", "text": "t"}])
    man["rankings/v1/queries.jsonl"] = hashlib.sha256((ship / "rankings/v1/queries.jsonl").read_bytes()).hexdigest()
    (ship / "manifest.json").write_text(json.dumps({"sha256": man}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ship check"):
        sp.check_extra(tmp_path, f"{d}/ship")


def test_build_rankings_reads_probe_candidates(tmp_path):
    br = load_script("build_rankings")
    p = _write(tmp_path / "pairs.jsonl", [_item("m7-b", split="probe"), _item("m7-a", split="probe")])
    assert br.family_queries("probe", p) == [("m7-a|q", "savol m7-a?"), ("m7-b|q", "savol m7-b?")]


def test_ce_scores_query_keys_probe_and_old_set(tmp_path):
    from grl.train.data import SplitGuardError
    ce = load_script("ce_scores")
    p = _write(tmp_path / "pairs.jsonl", [_item("m7-b", split="probe"), _item("m7-a", split="probe")])
    assert ce.query_keys(p, probe=True) == ["m7-a|q", "m7-b|q"]
    assert ce.query_keys(p, probe=True, limit=1) == ["m7-a|q"]
    with pytest.raises(SplitGuardError, match="probe"):          # the old-set path refuses a probe file
        ce.query_keys(p, probe=False)
    o = _write(tmp_path / "old.jsonl", [_item("m7-a", split="train")])
    with pytest.raises(SplitGuardError):                           # the probe path refuses non-probe rows
        ce.query_keys(o, probe=True)


def test_probe_ids_fail_hard_off_spark_and_fall_back_on_spark(tmp_path, monkeypatch):
    """R30: a missing ids.json is a SplitGuardError on the Mac (not a silent, cached empty set); on Spark
    (GRL_HOST=spark, exported by scripts/spark.py) it is the empty set and the guards fall back to split / qid."""
    from grl.train.data import SplitGuardError
    from grl.train.pilot import check_ship
    f = tmp_path / "ids.json"
    monkeypatch.setattr(_pr, "PROBE_IDS_PATH", str(f))
    monkeypatch.setattr(_pr, "probe_ids", _REAL_PROBE_IDS)
    monkeypatch.delenv(_pr.HOST_VAR, raising=False)
    with pytest.raises(SplitGuardError, match="missing"):
        _pr.probe_ids()
    ship = _write(tmp_path / "ship" / "pairs.jsonl", [{"id": "p1", "split": "train"}]).parent
    with pytest.raises(SplitGuardError, match="missing"):              # the old-set guard does not skip the ids
        check_ship(ship, {"p1"})
    with pytest.raises(SplitGuardError, match="missing"):
        _pr.refuse_probe_file(ship / "pairs.jsonl")
    monkeypatch.setenv(_pr.HOST_VAR, _pr.SPARK_HOST)
    assert _pr.probe_ids() == frozenset()
    check_ship(ship, {"p1"})                                            # Spark: split / id guards only, as before
    with pytest.raises(SplitGuardError, match="probe"):                 # split-based refusal still holds on Spark
        _pr.refuse_probe_file(_write(tmp_path / "p.jsonl", [_item("m7-a", split="probe")]))
    monkeypatch.delenv(_pr.HOST_VAR)
    f.write_text("[]", encoding="utf-8")
    with pytest.raises(SplitGuardError, match="no probe ids"):
        _pr.probe_ids()
    _pr._probe_ids.cache_clear()
    f.write_text(json.dumps(["m7-a"]), encoding="utf-8")
    assert _pr.probe_ids() == frozenset({"m7-a"})
    f.unlink()
    _pr._probe_ids.cache_clear()
    with pytest.raises(SplitGuardError, match="missing"):               # a miss is never cached as empty
        _pr.probe_ids()
