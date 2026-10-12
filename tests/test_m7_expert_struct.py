"""M7 part 2, Task 2 (rulings R34 / R35): the struct expert accepts limit-ended finals (reason k_max / l_max, no STOP
step, meta.expert.final_by = "limit") and does not prune zero-Δ bridge steps; replay and the SFT gate check both.
Toy: the struct toy of tests/test_m7_struct_env.py (V₀ = [D/СТ.1]; graph:AMENDS:1 is a zero-Δ bridge to EXT/k)."""
import pytest

from grl.train.expert import expert_search
from grl.train.fbeta import GRLWeights
from tests.test_m7_struct_env import make

W = GRLWeights(0.5, 0.2, 0.2, 0.1)


def _gold(env, ids):
    return frozenset(env.data.chunk_row[g] for g in ids)


@pytest.fixture
def toy_struct_kmax2():
    # K_max = 2, gold = {D/СТ.1#1, STRUCT/D}: both fit only by filling the budget (the 2nd add ends with k_max)
    env, q = make({"K_max": 2})
    return env, q, _gold(env, ["D/СТ.1#1", "STRUCT/D"]), W


@pytest.fixture
def toy_struct_bridge():
    # K_max = 2, gold = {STRUCT/EXT/k}: the only STOP-able state holding it is graph:AMENDS:1 (adds nothing, makes
    # EXT/k a structure root) -> structure:1; the fragment path CONTAINS -> structure:1 fills K_max
    env, q = make({"K_max": 2})
    return env, q, _gold(env, ["STRUCT/EXT/k"]), W


class _Ranks:
    def __init__(self, q):
        self.q = q

    def get(self, key):
        return self.q


def _pair(env, gold_ids, pid="toy"):
    from graphrag.dataset.finalize import Pair
    return Pair(pid, "q", "ru", "P1", "1", {"text": "", "sources": []}, list(gold_ids), [], [], ["D"], "train", "",
                pid)


def _struct_setup(k_max=2):
    env, q = make({"K_max": k_max})
    env.query = lambda rank: rank                      # the toy query is built by query_from_vector, not a ranking
    return env, _Ranks(q)


def _limit_pair(variant="struct"):
    from dataclasses import replace

    from graphrag.env.reward import Weights
    from grl.train.data import item_of
    from grl.train.expert import expert_pair
    env, ranks = _struct_setup()
    out, info = expert_pair(env, ranks, _pair(env, ["D/СТ.1#1", "STRUCT/D"]), W, Weights(), "struct", {}, depth=3,
                            prune_zero=False, limit_finals=True)
    assert info["final_by"] == "limit"
    if variant != "struct":
        meta = {**out.provenance["meta"], "variant": variant}
        out = replace(out, provenance={**out.provenance, "meta": meta})
    return env, ranks, item_of(out, env.data), out


@pytest.fixture
def limit_pair_row():
    env, ranks, it, _ = _limit_pair()
    return env, ranks, it


@pytest.fixture
def full_pair_row():
    env, ranks, it, _ = _limit_pair("full")
    return env, ranks, it


# ---- the brief's tests (assertions verbatim) ------------------------------------------------------------------------

def test_limit_final_reaches_k_max(toy_struct_kmax2):        # toy StructEnv, K_max=2, gold = two fragments reachable only by filling the budget
    env, q, gold, w = toy_struct_kmax2
    stop_only = expert_search(env, q, gold, w, depth=3, prune_zero=False)
    limit = expert_search(env, q, gold, w, depth=3, prune_zero=False, limit_finals=True)
    assert len(limit["final"].context) == 2 and limit["final"].reason == "k_max"
    assert limit["score"] > stop_only["score"]


def test_bridge_not_pruned(toy_struct_bridge):                # gold reachable only after a zero-Δ graph step (as in test_m7_probe_metrics)
    env, q, gold, w = toy_struct_bridge
    assert not set(gold) <= set(expert_search(env, q, gold, w, depth=3, prune_zero=True)["final"].context)
    assert set(gold) <= set(expert_search(env, q, gold, w, depth=3, prune_zero=False)["final"].context)


def test_replay_accepts_limit_final_only_for_struct(limit_pair_row, full_pair_row):
    import pytest
    from grl.train.data import replay_item, DataMismatch
    replay_item(*limit_pair_row)                             # struct + final_by=limit -> ok
    with pytest.raises(DataMismatch):
        replay_item(*full_pair_row)                          # full variant row whose trajectory lacks STOP -> refused


def test_sft_gate_refuses_pruned_struct_labels(tmp_path):
    import pytest
    from grl.train.data import check_struct_labels           # new: refuses meta.expert.prune_zero != False for struct
    with pytest.raises(SystemExit, match="prune_zero"):
        check_struct_labels([{"provenance": {"meta": {"variant": "struct", "expert": {"prune_zero": True}}}}])


# ---- details --------------------------------------------------------------------------------------------------------

def test_limit_final_has_no_stop_step_and_labels_cover_every_step():
    env, ranks, it, out = _limit_pair()
    assert it.final_by == "limit" and it.actions[-1] != 0 and 0 not in it.actions
    assert len(it.equiv) == len(it.actions) == len(it.masks) and [0] not in out.action_equiv
    assert out.trajectory[-1]["op"] != "stop"
    m = out.provenance["meta"]["expert"]
    assert m["final_by"] == "limit" and m["prune_zero"] is False and m["reached"] is True


def test_limit_labels_match_trajectory_labels_on_the_retrieval_steps():
    from graphrag.env.policies import trajectory_labels
    from grl.train.expert import limit_labels
    env, q = make({"K_max": 2})
    gold = _gold(env, ["D/СТ.1#1", "STRUCT/D"])
    s = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True)["final"]
    assert limit_labels(env, q, s) == trajectory_labels(env, q, s)
    stop = expert_search(env, q, gold, W, depth=3, prune_zero=False)["final"]
    with pytest.raises(ValueError):
        limit_labels(env, q, stop)                           # a STOP final is not a limit final


def test_replay_refuses_a_struct_limit_row_without_final_by():
    from dataclasses import replace

    from grl.train.data import DataMismatch, replay_item
    env, ranks, it, _ = _limit_pair()
    with pytest.raises(DataMismatch):
        replay_item(env, ranks, replace(it, final_by=None))  # no final_by -> STOP required
    with pytest.raises(DataMismatch):
        replay_item(env, ranks, replace(it, final_by="stop"))


def test_replay_refuses_a_limit_tag_on_a_stop_trajectory():
    from dataclasses import replace

    from graphrag.env.reward import Weights
    from grl.train.data import DataMismatch, item_of, replay_item
    from grl.train.expert import expert_pair
    env, ranks = _struct_setup(k_max=5)
    out, info = expert_pair(env, ranks, _pair(env, ["STRUCT/EXT/k"]), W, Weights(), "struct", {}, depth=3,
                            prune_zero=False, limit_finals=True)
    it = item_of(out, env.data)
    assert info["final_by"] == "stop" and it.final_by == "stop" and it.actions[-1] == 0
    replay_item(env, ranks, it)
    with pytest.raises(DataMismatch):
        replay_item(env, ranks, replace(it, final_by="limit"))


def test_exhausted_is_not_a_final():
    # a real search state that ends "exhausted": on the K_max 5 toy, gold = all three fragments, the oracle (which
    # counts exhausted states as finals) picks graph:CONTAINS:1 -> graph:AMENDS:1 -> structure:3, ended exhausted.
    # The limit-final expert never returns it, limit_labels refuses it and replay refuses it as a limit row.
    from dataclasses import replace

    from graphrag.env.policies import trajectory_labels
    from graphrag.env.reward import Weights, reward
    from grl.hard.oracle import diag_search
    from grl.train.data import LIMIT_FINAL_REASONS, DataMismatch, TrainItem, replay_item
    from grl.train.expert import limit_labels
    assert set(LIMIT_FINAL_REASONS) == {"k_max", "l_max"}
    env, ranks = _struct_setup(k_max=5)
    q = ranks.q
    gold = _gold(env, ["STRUCT/D", "STRUCT/EXT/k", "D/СТ.1#1"])
    o = diag_search(env, q, gold, W, 5, "noprune")
    ex = o["final"]
    assert ex.done and ex.reason == "exhausted" and 0 not in [h.action for h in ex.history]
    e = expert_search(env, q, gold, W, depth=5, prune_zero=False, limit_finals=True)
    assert e["final"].reason in ("stop", "k_max", "l_max") and e["final_by"] == "stop"
    with pytest.raises(ValueError):
        limit_labels(env, q, ex)
    equiv, masks = trajectory_labels(env, q, ex)
    parts = reward(env.data, q.depth, ex.context, gold, Weights(), env.limits.T_max)
    it = TrainItem("exh", "exh|q", "train", "P1", "ru", gold, tuple(h.action for h in ex.history),
                   tuple(tuple(c) for c in equiv), tuple(tuple(m) for m in masks),
                   tuple(sorted((k, float(v)) for k, v in parts.__dict__.items())), "struct", "{}", True, None,
                   float(o["score"]), 5, "limit", False)
    with pytest.raises(DataMismatch, match="exhausted"):
        replay_item(env, ranks, it)
    with pytest.raises(DataMismatch):
        replay_item(env, ranks, replace(it, final_by="stop"))


def test_check_struct_labels_accepts_unpruned_and_refuses_missing():
    from grl.train.data import check_struct_labels
    ok = {"provenance": {"meta": {"variant": "struct", "expert": {"prune_zero": False}}}}
    check_struct_labels([ok, ok])
    for bad in ({"provenance": {"meta": {"variant": "struct", "expert": {}}}},
                {"provenance": {"meta": {"variant": "struct"}}}, {"provenance": {}}):
        with pytest.raises(SystemExit, match="prune_zero"):
            check_struct_labels([ok, bad])


def test_check_variant_file_runs_the_struct_gate(tmp_path):
    import json
    from dataclasses import replace

    from grl.train.data import check_variant_file
    _, _, _, out = _limit_pair()
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(out.to_json(), ensure_ascii=False) + "\n")
    assert check_variant_file(good, "struct", {}, False, None, 2, 24) == "expert"
    meta = {**out.provenance["meta"], "expert": {**out.provenance["meta"]["expert"], "prune_zero": True}}
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(replace(out, provenance={**out.provenance, "meta": meta}).to_json(),
                              ensure_ascii=False) + "\n")
    with pytest.raises(SystemExit, match="prune_zero"):
        check_variant_file(bad, "struct", {}, False, None, 2, 24)


def test_variant_defaults():
    from grl.train.expert import expert_defaults
    assert expert_defaults("struct", None, None) == (False, True)
    assert expert_defaults("full", None, None) == (True, False)
    assert expert_defaults("fused_adaptive", None, None) == (True, False)
    assert expert_defaults("struct", True, None) == (True, True)          # an explicit flag wins
    assert expert_defaults("full", False, None) == (False, False)
    with pytest.raises(ValueError):
        expert_defaults("full", None, True)                              # replay accepts limit finals for struct only


def test_meta_without_limit_finals_is_unchanged():
    from grl.train.expert import expert_meta
    res = {"exact": True, "beam": None, "capped": False, "depth": 3, "score": 0.5, "n_states": 7}
    m = expert_meta("full", {}, res, True, W, True, 2, 24)
    assert "final_by" not in m["expert"]
    assert expert_meta("struct", {}, res, True, W, False, 2, 5, final_by="limit")["expert"]["final_by"] == "limit"


def test_sft_drop_keeps_a_one_step_limit_final():
    from grl.train.expert import sft_drop
    assert sft_drop(0.5, True, 1)                            # an immediate STOP
    assert not sft_drop(0.5, True, 1, "limit")               # one retrieval step that filled the budget


@pytest.mark.parametrize("kmax", [1, 2, 3, 5])
def test_unpruned_limit_expert_matches_oracle_noprune(kmax):
    # the struct expert (set key) and grl.hard.oracle noprune (ordered key, + exhausted finals) agree on R wherever
    # the oracle's final is a STOP / k_max / l_max final
    from grl.hard.oracle import diag_search
    golds = (["STRUCT/EXT/k"], ["D/СТ.1#1", "STRUCT/EXT/k"], ["STRUCT/D", "STRUCT/EXT/k", "D/СТ.1#1"], ["STRUCT/D"],
             ["D/СТ.1#1", "STRUCT/D"])
    for gold_ids in golds:
        env, q = make({"K_max": kmax})
        gold = _gold(env, gold_ids)
        for depth in (3, 5):
            o = diag_search(env, q, gold, W, depth, "noprune")
            e = expert_search(env, q, gold, W, depth=depth, prune_zero=False, limit_finals=True)
            if o["reason"] in ("stop", "k_max", "l_max"):
                assert e["score"] == pytest.approx(o["score"], abs=1e-12), (kmax, gold_ids, depth)
                assert e["final_by"] == ("stop" if o["reason"] == "stop" else "limit")
            assert e["score"] <= o["score"] + 1e-12


def test_run_expert_struct_defaults_end_to_end(tmp_path, monkeypatch):
    # train.py expert --variant struct on the M3 toy + STRUCT rows: unpruned, limit finals on, every row replays,
    # and the struct SFT gate accepts the file; --prune-zero labels are refused by it
    from dataclasses import replace

    from graphrag import config
    from graphrag.dataset.finalize import read_pairs
    from grl.train import expert as ex
    from grl.train.data import check_variant_file
    from tests.m4_toy import TOY_RANKS, toy_pairs
    import numpy as np

    from grl.env.struct_data import extend_with_struct
    from tests.m3_toy import toy_data
    from tests.test_m7_struct_policy import ROOT_VECS
    ver = config.load("train")["env"]["graph_version"]
    ids = {f"STRUCT/{r}": r for r in ROOT_VECS}
    data = extend_with_struct(replace(toy_data(), version=ver), {i: i for i in ids},
                              {i: np.array(ROOT_VECS[r], np.float32) for i, r in ids.items()}, {i: 40 for i in ids})
    monkeypatch.setattr("grl.model.variants.load_variant_data", lambda *a, **k: data)
    monkeypatch.setattr("graphrag.retrieval.rankings.load_rankings", lambda *a, **k: TOY_RANKS)
    cfg = config.load("train")
    cfg["env"]["K_max"] = 2
    src = toy_pairs(tmp_path / "m3.jsonl")
    out = tmp_path / "struct.jsonl"
    st = ex.run_expert(src, "unused", "struct", out, cfg, workers=1, depth=3)
    assert st["prune_zero"] is False and st["limit_finals"] is True
    assert st["max_states"] == 1_000_000 and st["max_rss_gb"] == 20.0 and st["rss_ceiling_gb"] is None
    for split in ("train", "val"):                                      # R40: peak n_states and RSS per worker
        assert st[split]["n_states_peak"] == st[split]["n_states_max"] and st[split]["capped_rss"] == 0
        assert st[split]["rss_growth_gb_max"] >= 0.0
        assert list(st[split]["rss_peak_gb_per_worker"].values())[0] > 0
    rows = read_pairs(out)
    metas = [p.provenance["meta"]["expert"] for p in rows]
    assert all(m["prune_zero"] is False and m["final_by"] in ("stop", "limit") for m in metas)
    for p, m in zip(rows, metas):
        assert (p.actions[-1] == 0) == (m["final_by"] == "stop")
    assert sorted(m["final_by"] for m in metas) == ["limit", "stop"]       # both kinds of final on the toy
    assert check_variant_file(out, "struct", {}, False, None, 2, 24) == "expert"
    pruned = tmp_path / "struct-pruned.jsonl"
    ex.run_expert(src, "unused", "struct", pruned, cfg, workers=1, depth=3, prune_zero=True)
    with pytest.raises(SystemExit, match="prune_zero"):
        check_variant_file(pruned, "struct", {}, False, None, 2, 24)


def test_sft_drop_still_drops_a_genuine_immediate_stop():
    # R39: only a one-step LIMIT final survives the length-1 rule; a one-action STOP row ([0]) is still dropped,
    # with or without final_by recorded, by sft_drop and by sft_train_filter
    from dataclasses import replace

    from grl.train.data import sft_train_filter
    from grl.train.expert import sft_drop
    assert sft_drop(0.5, True, 1, "stop") and sft_drop(0.5, True, 1, None)
    _, _, it, _ = _limit_pair()
    stop1 = replace(it, pid="stop1", actions=(0,), equiv=((0,),), masks=((0,),), final_by="stop")
    stop1_m3 = replace(stop1, pid="stop1b", final_by=None)
    one_limit = replace(it, pid="lim1", actions=it.actions[:1], final_by="limit", score=0.5, reach=True)
    kept, cnt = sft_train_filter([stop1, stop1_m3, one_limit])
    assert [i.pid for i in kept] == ["lim1"] and cnt["len1"] == 2


def test_struct_max_states_only_for_struct():
    # R38: expert.struct_max_states is the cap of variant struct; every other variant keeps expert.max_states
    from graphrag import config
    from grl.train.expert import expert_config
    cfg = config.load("train")
    assert cfg["expert"]["max_states"] == 20000 and cfg["expert"]["struct_max_states"] == 1_000_000
    assert expert_config(cfg)["max_states"] == 20000
    assert expert_config(cfg, "full")["max_states"] == 20000
    assert expert_config(cfg, "struct")["max_states"] == 1_000_000
    assert set(expert_config(cfg)) == set(expert_config(cfg, "full")) == {"depth", "max_states", "cap_beam"}
    sc = expert_config(cfg, "struct")
    assert sc["workers"] == cfg["expert"]["struct_workers"] == 4
    assert sc["max_rss_gb"] == float(cfg["expert"]["struct_max_rss_gb"]) == 20.0
    assert sc["rss_ceiling_gb"] is None and cfg["expert"]["struct_rss_ceiling_gb"] is None


def test_ladder_rungs_set_the_struct_cap():
    # I1: a ladder rung must change the cap struct actually reads (expert.struct_max_states, R38)
    from graphrag import config
    from grl.train.expert import expert_config
    from tests.scripts_import import load_script
    chk = load_script("m7_expert_struct_check")
    cfg = config.load("train")
    caps = [expert_config(chk.rung_cfg(cfg, ms), "struct")["max_states"] for ms in (20000, 200000)]
    assert caps == [20000, 200000]
    assert expert_config(chk.rung_cfg(cfg, 20000), "full")["max_states"] == cfg["expert"]["max_states"]
    assert cfg["expert"]["struct_max_states"] == 1_000_000                  # the loaded cfg is not mutated


def _rss_seq(monkeypatch, values):
    """grl.train.expert.rss_gb -> the given values in order (the last repeats); returns the call counter."""
    import grl.train.expert as ex
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        return float(values[min(calls["n"], len(values)) - 1])
    monkeypatch.setattr(ex, "rss_gb", fake)
    return calls


def test_rss_guard_ignores_rss_left_by_an_earlier_query(monkeypatch):
    # R42: the guard is per query (growth over the RSS at query start): a worker already at 100 GB does not flag the
    # next query under a 1 GB growth budget; only the optional absolute ceiling would
    env, q = make({"K_max": 5})
    gold = _gold(env, ["STRUCT/EXT/k"])
    _rss_seq(monkeypatch, [100.0])
    r = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True, max_rss_gb=1.0)
    assert r["capped"] is False and r["capped_reason"] is None and r["exact"] is True and r["rss_growth_gb"] == 0.0
    free = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True)
    assert r["score"] == free["score"] and free["rss_growth_gb"] is None             # guard off: no RSS read
    c = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True, max_rss_gb=1.0,
                      rss_ceiling_gb=50.0)
    assert c["capped"] is True and c["capped_reason"] == "rss_ceiling"


def test_rss_guard_trips_on_growth_within_a_query(monkeypatch):
    # R42: growth within the query trips it -> the flagged cap beam (capped, exact False, capped_reason "rss",
    # meta search beam<cap_beam>); the record still replays (expert_pair replays it)
    from graphrag.env.reward import Weights
    from grl.train.expert import expert_pair
    env, q = make({"K_max": 5})
    gold = _gold(env, ["STRUCT/EXT/k"])
    _rss_seq(monkeypatch, [10.0, 10.5, 13.0])                # rss0, level 0 (+0.5), level 1 (+3 > 2)
    r = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True, max_rss_gb=2.0, cap_beam=2)
    assert r["capped"] is True and r["capped_reason"] == "rss" and r["exact"] is False and r["beam"] == 2
    assert r["rss_growth_gb"] == 3.0
    env2, ranks = _struct_setup(k_max=5)
    _rss_seq(monkeypatch, [10.0, 10.5, 13.0])
    out, info = expert_pair(env2, ranks, _pair(env2, ["STRUCT/EXT/k"]), W, Weights(), "struct", {}, depth=3,
                            prune_zero=False, limit_finals=True, max_rss_gb=2.0)
    m = out.provenance["meta"]["expert"]
    assert m["capped"] is True and m["capped_reason"] == "rss" and m["search"] == "beam256"
    assert info["capped_reason"] == "rss" and info["rss_growth_gb"] == 3.0 and info["rss_peak_gb"] > 0


def test_rss_guard_mid_level_check(monkeypatch):
    # the in-level path (j % RSS_CHECK_EVERY): with RSS_CHECK_EVERY = 1 the 4th read is the check before the 2nd
    # parent of level 1 (reads: rss0, level 0, level 1, then mid-level) -- it trips there and stops reading
    import grl.train.expert as ex
    env, q = make({"K_max": 5})
    gold = _gold(env, ["STRUCT/EXT/k"])
    monkeypatch.setattr(ex, "RSS_CHECK_EVERY", 1)
    calls = _rss_seq(monkeypatch, [10.0, 10.0, 10.0, 15.0])
    full = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True)
    r = expert_search(env, q, gold, W, depth=3, prune_zero=False, limit_finals=True, max_rss_gb=2.0)
    assert calls["n"] == 4 and r["capped"] is True and r["capped_reason"] == "rss" and r["exact"] is False
    assert r["n_states"] < full["n_states"]                  # level 1 was cut after its first parent


def test_reproduction_counts_limit_finals():
    # R41: a limit-final row is reproduced by an episode ending k_max / l_max on the same set; a STOP row by STOP
    from types import SimpleNamespace as NS

    from grl.train.evaluate import reproduction
    lim, stop = NS(qkey="a", final_by="limit"), NS(qkey="b", final_by=None)
    exp = {"a": frozenset({1, 2}), "b": frozenset({3})}

    def f(reason, ctx):
        return NS(reason=reason, ctx_set=frozenset(ctx))
    assert reproduction([f("k_max", {1, 2}), f("stop", {3})], [lim, stop], exp) == 1.0
    assert reproduction([f("l_max", {1, 2}), f("k_max", {3})], [lim, stop], exp) == 0.5
    assert reproduction([f("stop", {1, 2}), f("stop", {3})], [lim, stop], exp) == 0.5
    assert reproduction([f("exhausted", {1, 2}), f("stop", {4})], [lim, stop], exp) == 0.0


def test_check_variant_items_runs_the_struct_gate():
    # the R35 gate also covers items handed to cmd_sft directly (TrainItem.prune_zero)
    from dataclasses import replace

    from grl.train.data import check_variant_items
    _, _, it, _ = _limit_pair()
    assert it.prune_zero is False
    assert check_variant_items([it], "struct", {}, False, None, 2, 24) == "expert"
    for bad in (True, None):
        with pytest.raises(SystemExit, match="prune_zero"):
            check_variant_items([it, replace(it, pid="x", prune_zero=bad)], "struct", {}, False, None, 2, 24)
