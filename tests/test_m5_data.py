"""Training data: split guard, §8.1 pairs -> SFT examples by replay, pilot snapshot (IMPLEMENTATION.md §8.1, §9.1)."""
import hashlib
import json

import numpy as np
import pytest

from grl.train.data import (DataMismatch, SplitGuardError, guard_qkeys, guard_splits, load_items,
                            replay_examples, replay_item)
from grl.train.pilot import keep_splits, snapshot, subset_rankings
from tests.m4_toy import DKW, TOY_RANKS, toy_env, toy_pairs


@pytest.fixture(scope="module")
def pairs_file(tmp_path_factory):
    return toy_pairs(tmp_path_factory.mktemp("d") / "pairs.jsonl")


def test_split_guard_refuses_test_and_qreal():
    for bad in (["train", "test"], ["qreal_test"], ["dev"]):
        with pytest.raises(SplitGuardError):
            guard_splits(bad)
    with pytest.raises(SplitGuardError):
        guard_qkeys(["val-0001|cyr"])
    guard_splits(["train", "val"])
    guard_qkeys(["p1|q"])


def test_load_items_skips_test_pairs_and_needs_trajectories(pairs_file, tmp_path):
    data = toy_env().data
    tr, va = load_items(pairs_file, data, ("train",)), load_items(pairs_file, data, ("val",))
    assert [i.pid for i in tr] == ["p1"] and [i.pid for i in va] == ["p2"]
    assert tr[0].qkey == "p1|q" and tr[0].gold == frozenset({3, 4}) and tr[0].actions[-1] == 0
    with pytest.raises(SplitGuardError):
        load_items(pairs_file, data, ("train", "test"))
    row = json.loads(pairs_file.read_text(encoding="utf-8").splitlines()[0])
    row["actions"] = []
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(f, data, ("train",))


def test_replay_builds_examples_and_expert_contexts(pairs_file):
    env = toy_env()
    items = load_items(pairs_file, env.data, ("train", "val"))
    ex, expert = replay_examples(env, TOY_RANKS, items, DKW)
    assert len(ex) == sum(len(i.actions) for i in items)
    for e in ex:
        assert set(e.equiv) <= set(np.flatnonzero(e.desc.mask).tolist())
    assert expert["p1|q"] >= frozenset({3, 4}) and expert["p2|q"] >= frozenset({2})
    seen = []
    q, s = replay_item(env, TOY_RANKS, items[0], lambda t, qq, st: seen.append(t))
    assert s.reason == "stop" and frozenset(s.ctx_set) == expert[items[0].qkey]
    assert seen == list(range(len(items[0].actions))) and q.qkey == items[0].qkey


def test_replay_detects_a_tampered_mask(pairs_file):
    env = toy_env()
    it = load_items(pairs_file, env.data, ("train",))[0]
    bad = it.__class__(**{**it.__dict__, "masks": (tuple(it.masks[0][:-1]),) + it.masks[1:]})
    with pytest.raises(DataMismatch, match="p1"):
        replay_examples(env, TOY_RANKS, [bad], DKW)


def test_snapshot_keep_splits(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n, t in (("verified.jsonl", "a\n"), ("reviewed.jsonl", "b\n"), ("stats.json", "{}")):
        (src / n).write_text(t, encoding="utf-8")
    d = snapshot(src, tmp_path / "out")
    assert d.name == hashlib.sha256(b"b\n").hexdigest()[:12] and (d / "reviewed.jsonl").read_text() == "b\n"
    assert snapshot(src, tmp_path / "out") == d
    rows = [{"id": "a", "split": "train"}, {"id": "b", "split": "test"}, {"id": "c", "split": "val"}]
    f = d / "candidates.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert keep_splits(f, ("train", "val")) == {"train": 1, "val": 1, "dropped": 1}
    assert [json.loads(l)["id"] for l in f.read_text().splitlines()] == ["a", "c"]


def test_subset_rankings_keeps_only_the_given_keys(tmp_path):
    from graphrag.retrieval.rankings import TOP, load_rankings, write_rankings
    qs = [("a|q", "A"), ("b|q", "B"), ("t|q", "T")]
    arr = {"qvec": np.eye(3, 4, dtype=np.float32)}
    for n in ("lex", "vec", "title"):
        arr[f"{n}_idx"] = np.tile(np.arange(TOP, dtype=np.int32), (3, 1))
        arr[f"{n}_score"] = np.ones((3, TOP), np.float32)
    write_rankings(tmp_path / "src", qs, arr, {"graph_version": "g"})
    ver = subset_rankings(tmp_path / "src" / "latest.json", ["a|q", "b|q"], tmp_path / "dst")
    r = load_rankings(tmp_path / "dst" / "latest.json", graph_version="g")
    assert r.keys() == ["a|q", "b|q"] and r.meta["version"] == ver
    assert np.array_equal(r.get("b|q").qvec, arr["qvec"][1])


# --- fix round 1: class-checked replay, loud Q-real refusal, dev-port assert, atomic snapshot, ship/ subset ---

def _p1(pairs_file):
    env = toy_env()
    return env, load_items(pairs_file, env.data, ("train",))[0]


def _with(it, **kw):
    return it.__class__(**{**it.__dict__, **kw})


def test_replay_refuses_a_tampered_equiv_class(pairs_file):
    env, it = _p1(pairs_file)
    extra = next(a for a in it.masks[0] if a not in it.equiv[0])          # mask unchanged, class grows by one action
    bad = _with(it, equiv=(tuple(sorted(it.equiv[0] + (extra,))),) + it.equiv[1:])
    with pytest.raises(DataMismatch, match="p1 t=1: equiv"):
        replay_item(env, TOY_RANKS, bad)


def test_replay_refuses_an_action_outside_its_class(pairs_file):
    env, it = _p1(pairs_file)
    other = next(a for a in it.masks[0] if a not in it.equiv[0])
    with pytest.raises(DataMismatch, match="p1 t=1"):
        replay_item(env, TOY_RANKS, _with(it, actions=(other,) + it.actions[1:]))


def test_replay_refuses_an_early_stop_and_a_missing_stop(pairs_file):
    env, it = _p1(pairs_file)
    early = _with(it, actions=(0,) + it.actions, equiv=((0,),) + it.equiv, masks=(it.masks[0],) + it.masks)
    with pytest.raises(DataMismatch, match="p1 t=2: episode ended"):
        replay_item(env, TOY_RANKS, early)
    nostop = _with(it, actions=it.actions[:-1], equiv=it.equiv[:-1], masks=it.masks[:-1])
    with pytest.raises(DataMismatch, match="p1: replay ends"):
        replay_item(env, TOY_RANKS, nostop)


def test_replay_rederives_recorded_reward_parts(pairs_file):
    from dataclasses import asdict

    from graphrag.env.reward import Weights, reward
    env, it = _p1(pairs_file)
    q, s = replay_item(env, TOY_RANKS, it)
    ablation = Weights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)          # R_chunks-only run: other total
    parts = asdict(reward(env.data, q.depth, s.context, it.gold, ablation, env.limits.T_max))
    assert parts["total"] != asdict(reward(env.data, q.depth, s.context, it.gold, Weights(), env.limits.T_max))["total"]
    replay_item(env, TOY_RANKS, _with(it, reward=tuple(sorted(parts.items()))))   # total is never compared
    replay_item(env, TOY_RANKS, _with(it, reward=tuple(sorted({**parts, "coverage": parts["coverage"] + 1e-9}.items()))))
    for k in ("chunks", "coverage", "diversity", "coherence"):
        wrong = tuple(sorted({**parts, k: parts[k] + 0.01}.items()))
        with pytest.raises(DataMismatch, match=f"p1: reward.*{k}"):
            replay_item(env, TOY_RANKS, _with(it, reward=wrong))


@pytest.fixture
def fake_qreal(tmp_path, monkeypatch):
    import grl.train.data as D
    f = tmp_path / "qreal.jsonl"
    f.write_text(json.dumps({"qid": "val-9999", "question_cyr": "x"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(D, "QREAL_IDS_PATH", str(f))
    return "val-9999"


def test_qreal_is_refused_loudly_by_id_origin_and_split(pairs_file, tmp_path, fake_qreal):
    data = toy_env().data
    with pytest.raises(SplitGuardError):
        guard_qkeys([f"{fake_qreal}|q"])
    with pytest.raises(SplitGuardError):
        guard_qkeys([fake_qreal])
    rows = [json.loads(l) for l in pairs_file.read_text(encoding="utf-8").splitlines()]
    for change in ({"id": fake_qreal}, {"origin": "qreal"}, {"split": "qreal_test"}):
        f = tmp_path / "q.jsonl"                                              # the Q-real row is the TEST row p3
        f.write_text("".join(json.dumps({**r, **(change if r["id"] == "p3" else {})}) + "\n" for r in rows),
                     encoding="utf-8")
        with pytest.raises(SplitGuardError):
            load_items(f, data, ("train",))


def test_dev_postgres_port_is_required():
    from grl.train.pilot import dsn_port, require_dev_pg
    assert dsn_port("postgresql://u:p@localhost:5437/db") == 5437
    assert dsn_port("postgresql://localhost/db") == 5432
    assert dsn_port("host=localhost port=5436 dbname=x") == 5436
    require_dev_pg("postgresql://graphrag:graphrag@localhost:5437/graphrag")
    for bad in ("postgresql://u:p@localhost:5436/db", "postgresql://localhost/db", "host=h port=5436",
                "postgresql://u@h1:5437,h2:5436/db"):
        with pytest.raises(RuntimeError):
            require_dev_pg(bad)


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n, t in (("verified.jsonl", "a\n"), ("reviewed.jsonl", "b\n"), ("stats.json", "{}")):
        (src / n).write_text(t, encoding="utf-8")
    return src


def test_snapshot_is_atomic_and_reverified(tmp_path):
    src = _src(tmp_path)
    d = snapshot(src, tmp_path / "out")
    assert not list((tmp_path / "out").glob(".tmp-*"))
    (d / "stats.json").write_text('{"finalize": {}}', encoding="utf-8")      # M3 rewrites stats.json: tolerated
    assert snapshot(src, tmp_path / "out") == d
    (d / "verified.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        snapshot(src, tmp_path / "out")
    (d / "snapshot.json").unlink()                                             # a partial dir is refused
    with pytest.raises(ValueError, match="partial"):
        snapshot(src, tmp_path / "out")


def _pilot_dir(tmp_path, traj_rows, rank_keys):
    from graphrag.retrieval.rankings import TOP, write_rankings
    d = tmp_path / "snap"
    d.mkdir()
    for n in ("verified.jsonl", "reviewed.jsonl", "manual_sample.jsonl"):
        (d / n).write_text(json.dumps({"id": "t1", "split": "test"}) + "\n", encoding="utf-8")   # Mac-only files
    cands = [{"id": "a", "split": "train"}, {"id": "b", "split": "val"}]
    (d / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cands), encoding="utf-8")
    (d / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in traj_rows), encoding="utf-8")
    n = len(rank_keys)
    arr = {"qvec": np.ones((n, 4), np.float32)}
    for k in ("lex", "vec", "title"):
        arr[f"{k}_idx"] = np.zeros((n, TOP), np.int32)
        arr[f"{k}_score"] = np.zeros((n, TOP), np.float32)
    write_rankings(d / "rankings", [(k, k) for k in rank_keys], arr, {"graph_version": "g"})
    return d


def test_ship_holds_only_the_shippable_subset(tmp_path):
    from graphrag.retrieval.rankings import load_rankings
    from grl.train.pilot import build_ship, verify_ship
    d = _pilot_dir(tmp_path, [{"id": "a", "split": "train"}, {"id": "b", "split": "val"}], ["a|q", "b|q"])
    man = build_ship(d, {"train": 1, "val": 1})
    ship = d / "ship"
    names = {str(p.relative_to(ship)) for p in ship.rglob("*") if p.is_file()}
    assert {"trajectories.jsonl", "splits.json", "manifest.json", "rankings/latest.json"} <= names
    assert not names & {"verified.jsonl", "reviewed.jsonl", "manual_sample.jsonl", "candidates.jsonl", "stats.json"}
    assert set(man["sha256"]) == names - {"manifest.json"} and verify_ship(d) == man
    r = load_rankings(ship / "rankings" / "latest.json", graph_version="g")
    assert r.keys() == ["a|q", "b|q"] and r.dir.is_relative_to(ship)
    blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in ship.rglob("*.json*"))
    assert '"t1"' not in blob and "test" not in json.loads((ship / "splits.json").read_text())


@pytest.mark.parametrize("traj, keys", [
    ([{"id": "a", "split": "train"}, {"id": "t1", "split": "test"}], ["a|q"]),       # a test pair
    ([{"id": "a", "split": "train"}], ["a|q", "t1|q"]),                             # a test ranking key
    ([{"id": "a", "split": "train", "origin": "qreal"}], ["a|q"]),                  # Q-real origin
    ([{"id": "a", "split": "train"}], ["a|q", "val-0001|cyr"]),                     # Q-real key
    ([{"id": "a", "split": "train"}], ["a|q", "val-9999|q"]),                       # Q-real id
])
def test_ship_never_holds_test_or_qreal(tmp_path, fake_qreal, traj, keys):
    from grl.train.pilot import build_ship
    d = _pilot_dir(tmp_path, traj, keys)
    with pytest.raises(SplitGuardError):
        build_ship(d, {"train": 1, "val": 1})
    assert not (d / "ship").exists()


def test_pilot_script_removes_the_stray_rankings_report(tmp_path, monkeypatch):
    import importlib.util

    from graphrag import config
    spec = importlib.util.spec_from_file_location("m5_pilot_data", config.path("scripts/m5_pilot_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rep = tmp_path / "m3" / "rankings_report_dataset.json"
    monkeypatch.setattr(m, "STRAY_REPORT", str(rep))

    def fake_run(*args):
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text("new", encoding="utf-8")
    monkeypatch.setattr(m, "_run", fake_run)
    m._rankings("--family", "dataset")
    assert not rep.parent.exists()                                    # created by the run -> removed with it
    rep.parent.mkdir()
    rep.write_text("old", encoding="utf-8")
    m._rankings("--family", "dataset")
    assert rep.read_text(encoding="utf-8") == "old"                   # a pre-existing report is restored
