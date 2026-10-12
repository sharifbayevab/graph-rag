"""Pilot 5 (controller ruling): the verified source (verified minus reader-BAD / reader-C veto / superseded originals)
and the strict-consensus test-eval file -- shipped only on request, read only by eval / baselines --split test,
refused by every training / labelling loader."""
import json

import numpy as np
import pytest

from grl.train.data import (TEST_EVAL_MARK, TEST_EVAL_NAME, SplitGuardError, check_variant_file, load_items,
                            load_test_items)
from grl.train.expert import expert_inputs
from grl.train.pilot import (ADMITTED_BY, SOURCE_JSON, TEST_CANDIDATES, TEST_EVAL_DIR, TEST_RANKINGS, build_ship,
                             check_ship, verified_snapshot, verified_source, verify_ship)
from tests.m4_toy import TOY_RANKS, toy_env, toy_pairs
from tests.scripts_import import load_script


@pytest.fixture(autouse=True)
def _no_real_probe_ids(monkeypatch):
    # R31: check_ship reads grl.hard.probe.probe_ids (data/hard/probe/ids.json, required off Spark); patched as in
    # tests/test_m7_probe_snapshot.py so this file does not depend on the local ids.json
    import grl.hard.probe as pr
    monkeypatch.setattr(pr, "probe_ids", lambda: frozenset({"m7-a", "m7-b", "m7-c"}))


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _row(sid, split, review=None):
    prov = {"seed_id": sid, **({"review": review} if review else {})}
    return {"question": f"q {sid}", "answer": "a", "facts": [], "provenance": prov,
            "seed": {"seed_id": sid, "split": split, "chunks": [], "docs": [], "lang": "ru", "pattern": "single"}}


def _src(tmp_path):
    """s1 consensus train, s2 borderline val (kept), s3 A-BAD, s4 B-BAD, s5 C-veto, s6 superseded by s6-r1 (kept),
    s7 unread by A (kept), t1 consensus test, t2 non-consensus test (not TEST)."""
    from graphrag.dataset.opus import reader_id
    src = tmp_path / "fable"
    good = {"verdict": "GOOD", "consensus": True}
    split = {"s1": "train", "s2": "val", "s3": "train", "s4": "train", "s5": "val", "s6": "train", "s6-r1": "train",
             "s7": "train", "t1": "test", "t2": "test"}
    _w(src / "verified.jsonl", [_row(s, sp) for s, sp in split.items()])
    _w(src / "reviewed.jsonl", [_row("s1", "train", good), _row("t1", "test", good)])
    a = {"s1": "GOOD", "s2": "BORDERLINE", "s3": "BAD", "s4": "GOOD", "s5": "GOOD", "s6": "BORDERLINE",
         "s6-r1": "GOOD", "t1": "GOOD", "t2": "BORDERLINE"}
    b = {**a, "s3": "GOOD", "s4": "BAD", "s7": "GOOD"}
    _w(src / "review/verdicts/review-001.jsonl", [{"pair_id": reader_id(s), "verdict": v} for s, v in a.items()])
    (src / "review/verdicts/review-001.jsonl").open("a").write("not json\n")
    _w(src / "review2/verdicts/second-review-001.jsonl", [{"pair_id": reader_id(s), "verdict": v} for s, v in b.items()])
    _w(src / "review3/verdicts/third-review-001.jsonl", [{"pair_id": reader_id("s5"), "verdict": "BAD"}])
    _w(src / "review_rejections.jsonl", [{"seed_id": "s5", "reason": "reader_c_veto"},
                                         {"seed_id": "s2", "reason": "no_consensus"}])
    (src / "stats.json").write_text("{}", encoding="utf-8")
    return src


def test_verified_source_excludes_bad_veto_and_superseded(tmp_path):
    rows, rep = verified_source(_src(tmp_path))
    ids = [r["seed"]["seed_id"] for r in rows]
    assert ids == ["s1", "s2", "s6-r1", "s7", "t1"]
    assert rep["excluded_primary"] == {"superseded_by_repair": 1, "reader_a_bad": 1, "reader_b_bad": 1,
                                       "reader_c_veto": 1}
    assert rep["source_by"] == {"test|consensus": 1, "test|non_consensus": 1, "train|consensus": 1,
                                "train|non_consensus": 2, "val|non_consensus": 1}
    assert rep["test_consensus"] == 1 and rep["reader_unread"]["A"] == 1 and rep["reader_unparsable_lines"]["A"] == 1
    by = {r["seed"]["seed_id"]: r["provenance"]["review"] for r in rows}
    assert by["s1"] == {"verdict": "GOOD", "consensus": True} and by["t1"]["consensus"] is True
    assert by["s2"] == {"verdict": "GOOD", "admitted_by": ADMITTED_BY,
                        "readers": {"A": "BORDERLINE", "B": "BORDERLINE", "C": None}}
    assert by["s7"]["readers"]["A"] is None


def test_verified_snapshot_is_deterministic_and_records_the_report(tmp_path):
    src = _src(tmp_path)
    d = verified_snapshot(src, tmp_path / "out")
    assert verified_snapshot(src, tmp_path / "out") == d
    assert not list((tmp_path / "out").glob(".src-*"))
    assert (d / "verified.jsonl").read_bytes() == (d / "reviewed.jsonl").read_bytes()
    rep = json.loads((d / SOURCE_JSON).read_text(encoding="utf-8"))
    assert rep["excluded"] == 4 and rep["source_in_splits"] == 4


def _snap(tmp_path, test_rows, test_keys):
    from graphrag.retrieval.rankings import TOP, write_rankings
    d = tmp_path / "snap"
    d.mkdir(parents=True)
    _w(d / "candidates.jsonl", [{"id": "a", "split": "train"}, {"id": "b", "split": "val"}])
    _w(d / "trajectories.jsonl", [{"id": "a", "split": "train"}, {"id": "b", "split": "val"}])
    _w(d / TEST_CANDIDATES, test_rows)

    def ranks(root, keys):
        arr = {"qvec": np.ones((len(keys), 4), np.float32)}
        for k in ("lex", "vec", "title"):
            arr[f"{k}_idx"] = np.zeros((len(keys), TOP), np.int32)
            arr[f"{k}_score"] = np.zeros((len(keys), TOP), np.float32)
        write_rankings(root, [(k, k) for k in keys], arr, {"graph_version": "g"})
    ranks(d / "rankings", ["a|q", "b|q"])
    ranks(d / TEST_RANKINGS, test_keys)
    return d


def test_ship_test_eval_holds_only_marked_test_rows(tmp_path):
    from graphrag.retrieval.rankings import load_rankings
    d = _snap(tmp_path, [{"id": "t1", "split": "test"}], ["t1|q"])
    man = build_ship(d, {"train": 1, "val": 1}, test_eval=True)
    te = d / "ship" / TEST_EVAL_DIR
    rows = [json.loads(l) for l in (te / TEST_EVAL_NAME).read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "t1", "split": "test", "provenance": {TEST_EVAL_MARK: True}}]
    assert f"{TEST_EVAL_DIR}/{TEST_EVAL_NAME}" in man["sha256"] and verify_ship(d) == man
    assert load_rankings(te / "rankings" / "latest.json", graph_version="g").keys() == ["t1|q"]
    assert "t1" not in (d / "ship" / "trajectories.jsonl").read_text()
    with pytest.raises(SplitGuardError, match="no test ids"):
        check_ship(d / "ship", {"a", "b"})
    with pytest.raises(SplitGuardError, match="overlap"):
        check_ship(d / "ship", {"a", "b", "t1"}, {"t1"})
    (te / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SplitGuardError, match="holds only"):
        check_ship(d / "ship", {"a", "b"}, {"t1"})


@pytest.mark.parametrize("rows, keys", [
    ([{"id": "a", "split": "test"}], ["a|q"]),             # a train id as a test row (overlap with train / val)
    ([{"id": "t1", "split": "train"}], ["t1|q"]),          # not a test row
    ([{"id": "t1", "split": "test"}], ["t1|q", "b|q"]),    # a val key among the test rankings
])
def test_ship_test_eval_refuses_anything_else(tmp_path, rows, keys):
    d = _snap(tmp_path, rows, keys)
    with pytest.raises(SplitGuardError):
        build_ship(d, {"train": 1, "val": 1}, test_eval=True)
    assert not (d / "ship").exists()


def _test_file(tmp_path, name=TEST_EVAL_NAME, mark=True, split="test"):
    rows = [json.loads(l) for l in toy_pairs(tmp_path / "toy.jsonl").read_text(encoding="utf-8").splitlines()]
    t = dict(rows[-1], split=split, actions=[], action_equiv=[], action_masks=[])
    if mark:
        t["provenance"] = {TEST_EVAL_MARK: True}
    f = tmp_path / "te" / name
    _w(f, [t])
    return f


def test_test_eval_file_is_refused_by_every_training_loader(tmp_path):
    data = toy_env().data
    f = _test_file(tmp_path)
    items = load_test_items(f, data)
    assert [(i.pid, i.qkey, i.split, i.actions) for i in items] == [("p3", "p3|q", "test", ())]
    assert items[0].gold == frozenset({int(data.chunk_row["D1/СТ.1#1"])})
    renamed = _test_file(tmp_path, name="trajectories.jsonl")          # the mark alone is enough
    for path in (f, renamed):
        for load in (lambda p: load_items(p, data, ("train", "val")), expert_inputs,
                     lambda p: check_variant_file(p, "full", None, True)):
            with pytest.raises(SplitGuardError, match="test-eval"):
                load(path)


def test_load_test_items_reads_only_the_named_marked_file(tmp_path):
    data = toy_env().data
    with pytest.raises(SplitGuardError, match=TEST_EVAL_NAME):
        load_test_items(toy_pairs(tmp_path / "pairs.jsonl"), data)
    with pytest.raises(SplitGuardError, match="not a marked"):
        load_test_items(_test_file(tmp_path, mark=False), data)
    with pytest.raises(SplitGuardError, match="not a marked"):
        load_test_items(_test_file(tmp_path, split="val"), data)


def test_spark_ships_test_eval_only_on_request_and_only_test_rows(tmp_path):
    sp = load_script("spark")
    ship = "data/m5/pilot/abc/ship"
    keep = [f"{ship}/trajectories.jsonl", f"{ship}/rankings/latest.json"]
    te = [f"{ship}/test_eval/test_eval.jsonl", f"{ship}/test_eval/rankings/latest.json",
          f"{ship}/test_eval/rankings/v1/qvec.npy"]
    other = [f"{ship}/test_eval/notes.json", "data/m5/pilot/abc/test_candidates.jsonl",
             "data/m5/pilot/abc/test_rankings/latest.json"]
    assert sp.select_files([], keep + te + other, [ship]) == sorted(keep)
    assert sp.select_files([], keep + te + other, [ship], test_eval=True) == sorted(keep + te)
    assert sp.select_files([], te, [], test_eval=True) == []           # never outside an --extra ship dir
    _w(tmp_path / te[0], [{"id": "t1", "split": "test"}])
    _w(tmp_path / keep[0], [{"id": "a", "split": "train"}])
    sp.check_rows(tmp_path, [te[0], keep[0]], [ship])
    for bad in ([{"id": "a", "split": "train"}], [{"id": "x"}]):
        _w(tmp_path / te[0], [{"id": "t1", "split": "test"}] + bad)
        with pytest.raises(SystemExit):
            sp.check_rows(tmp_path, [te[0]], [ship])
    _w(tmp_path / keep[0], [{"id": "a", "split": "train"}, {"id": "t1", "split": "test"}])
    with pytest.raises(SystemExit, match="test"):
        sp.check_rows(tmp_path, [keep[0]], [ship])


def test_check_extra_verifies_the_test_eval_dir(tmp_path):
    sp = load_script("spark")
    root = tmp_path / "root"
    d = _snap(root / "data/m5/pilot", [{"id": "t1", "split": "test"}], ["t1|q"])
    d = d.rename(root / "data/m5/pilot/abc")
    build_ship(d, {"train": 1, "val": 1}, test_eval=True)
    assert sp.check_extra(root, "data/m5/pilot/abc/ship") == "data/m5/pilot/abc/ship"
    (d / TEST_CANDIDATES).unlink()
    with pytest.raises(SystemExit, match="ship check"):
        sp.check_extra(root, "data/m5/pilot/abc/ship")


def test_train_cli_reads_split_test_only_in_eval_and_baselines():
    tr = load_script("train")
    for argv in (["sft", "--pairs", "x", "--rankings", "y", "--run", "runs/r", "--split", "test"],
                 ["grpo", "--pairs", "x", "--rankings", "y", "--run", "runs/r", "--sft", "s", "--split", "test"],
                 ["grpo", "--pairs", "x", "--rankings", "y", "--run", "runs/r", "--sft", "s", "--per-query", "f"]):
        with pytest.raises(SystemExit):
            tr.main(argv)


def test_reference_rows_out_match_the_means(tmp_path):
    from graphrag.env.reward import Weights
    from grl.train.baselines import run_references
    env = toy_env()
    items = load_items(toy_pairs(tmp_path / "p.jsonl"), env.data, ("train", "val"))
    rows = {}
    out = run_references(["dense@3", "bm25@3"], env, TOY_RANKS, items, Weights(), rows_out=rows)
    assert sorted(rows) == ["bm25@3", "dense@3"]
    for n, rs in rows.items():
        assert [r["qkey"] for r in rs] == [i.qkey for i in items]
        assert out[n]["reward"] == pytest.approx(np.mean([r["reward"] for r in rs]), abs=1e-6)


def test_per_query_rows_match_the_greedy_summary():
    """train.py eval --per-query: one row per item; the mean of the rows' reward equals the summary's reward_mean."""
    from collections import namedtuple

    import torch

    from graphrag.env.reward import Weights
    from grl.model.policy import GraphPolicy
    from grl.model.variants import env_spec, model_config
    from grl.train.evaluate import evaluate_policy, per_query
    from grl.train.rollout import SerialVecEnv
    from tests.m4_toy import SMALL_CFG, TOY_CFG, TOY_START, toy_data, toy_graph, toy_init

    Item = namedtuple("Item", "qkey gold split pattern lang")
    items = [Item("toy|q", frozenset({3, 4}), "test", "single", "ru"), Item("toy|q2", frozenset({2}), "test", "single", "ru")]
    cls, kw = env_spec("full")
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, model_config(SMALL_CFG, "full"))
    vec = SerialVecEnv(**{**toy_init(), "env_cls": cls, "env_kw": kw})
    env, w = cls(toy_data(), TOY_CFG, TOY_START, **kw), Weights()
    s, eps, finals = evaluate_policy(pol, g, vec, TOY_RANKS, items, w, TOY_CFG["T_max"], "cpu", env.actions)
    rows = per_query(env, TOY_RANKS, items, eps, finals, w, env.actions)
    assert [r["qkey"] for r in rows] == ["toy|q", "toy|q2"]
    s1, eps1, fin1 = evaluate_policy(pol, g, vec, TOY_RANKS, items, w, TOY_CFG["T_max"], "cpu", env.actions, chunk=1)
    assert s1 == s and [e.actions for e in eps1] == [e.actions for e in eps]       # chunked = one pass
    assert np.mean([r["reward"] for r in rows]) == pytest.approx(s["reward_mean"], abs=1e-6)
    by = {e.qkey: e for e in eps}
    for r in rows:
        assert r["action_ids"] == list(by[r["qkey"]].actions) and len(r["actions"]) == len(r["action_ids"])
