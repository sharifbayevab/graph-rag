"""M7 part 2, Task 1: pre-fixes from the part-1 final review -- struct oracle baseline rows skipped, grpo K_max
gate, hard_probe pairs flag check, narrow pool except in hard_probe_eval. (The cheap structure mask and the
struct_scores memo are tested in tests/test_m7_struct_env.py, next to its toy fixtures.)"""
import json
from concurrent.futures.process import BrokenProcessPool

import pytest

from tests.scripts_import import load_script

ORACLES = ("greedy_oracle", "beam_oracle")


def test_baselines_skip_oracles_for_struct():
    from grl.train.cli_checks import STRUCT_ORACLE_SKIP, baseline_names
    names, skipped = baseline_names("struct", [3, 5, 10])
    assert not set(ORACLES) & set(names)
    assert skipped == {n: STRUCT_ORACLE_SKIP for n in ORACLES} and STRUCT_ORACLE_SKIP == "lookahead ignores structure"
    assert {"bm25@5", "graph_heuristic", "hybrid_heuristic", "random@0"} <= set(names)


def test_baselines_keep_oracles_for_full():
    from grl.train.cli_checks import baseline_names
    names, skipped = baseline_names("full", [3, 5, 10])
    assert set(ORACLES) <= set(names) and skipped == {}
    assert names[:9] == [f"{k}@{n}" for k in ("bm25", "dense", "rrf") for n in (3, 5, 10)]


def test_baselines_default_k_grid_is_train_yaml():
    from graphrag import config
    from grl.train.cli_checks import baseline_names
    k = config.load("train")["env"]["k_grid"]
    assert baseline_names("full")[0] == baseline_names("full", k)[0]


def test_beam_lookahead_refused_on_struct_env():
    from grl.train.baselines import _no_lookahead
    from tests.test_m7_struct_env import make
    env, _ = make()
    assert _no_lookahead(env)                  # BeamSearch's lookahead (successor_additions) never sees `structure`


def test_train_uses_cli_checks():
    import grl.train.cli_checks as cc
    tr = load_script("train")
    assert tr.check_grpo_k_max is cc.check_grpo_k_max and tr.baseline_names is cc.baseline_names


def test_grpo_refuses_k_max_mismatch(tmp_path):
    from grl.train.cli_checks import check_grpo_k_max
    (tmp_path / "pi_ref.json").write_text('{"K_max": 5}')
    with pytest.raises(SystemExit, match="K_max"):
        check_grpo_k_max(tmp_path, 24)
    assert check_grpo_k_max(tmp_path, 5) == 5


def test_grpo_k_max_from_manifest(tmp_path):
    from grl.train.cli_checks import check_grpo_k_max
    (tmp_path / "pi_ref.json").write_text('{"path": "pi_ref.pt"}')
    (tmp_path / "manifest.json").write_text(json.dumps({"phase": "sft", "K_max": 5}))
    with pytest.raises(SystemExit, match="K_max"):
        check_grpo_k_max(tmp_path, 24)
    assert check_grpo_k_max(tmp_path, 5) == 5


def test_grpo_k_max_conflicting_records(tmp_path):
    from grl.train.cli_checks import check_grpo_k_max
    (tmp_path / "pi_ref.json").write_text('{"K_max": 5}')
    (tmp_path / "manifest.json").write_text('{"K_max": 24}')
    with pytest.raises(SystemExit, match="K_max"):
        check_grpo_k_max(tmp_path, 5)


def test_grpo_k_max_unrecorded(tmp_path):
    from grl.train.cli_checks import check_grpo_k_max
    (tmp_path / "pi_ref.json").write_text('{"path": "pi_ref.pt"}')
    with pytest.raises(SystemExit, match="K_max"):              # no record and no legacy value: refuse
        check_grpo_k_max(tmp_path, 5)
    assert check_grpo_k_max(tmp_path, 24, legacy=24) == 24      # pre-R12 SFT: built under train.yaml K_max
    with pytest.raises(SystemExit, match="K_max"):
        check_grpo_k_max(tmp_path, 5, legacy=24)


def test_pairs_exclude_kinds_needs_n_new():
    hp = load_script("hard_probe")
    with pytest.raises(SystemExit, match="--n-new"):
        hp.main(["pairs", "--topup", "--exclude-kinds", "law_resolution"])


def _jobs():
    return [("q1", "all5", 7, [], 10, None), ("q2", "all5", 7, [], 10, None)]


def _raise(e):
    def run():
        raise e
    return run


def _wrapped(cause):
    from graphrag.env.parallel import ParallelRunError
    try:
        try:
            raise cause
        except BaseException as c:
            raise ParallelRunError("parallel.run failed") from c
    except ParallelRunError as e:
        return e


@pytest.mark.parametrize("err", [BrokenProcessPool("worker died"), MemoryError(), "wrapped_broken", "wrapped_mem"])
def test_diag_pool_error_recorded_only_deep(err):
    ev = load_script("hard_probe_eval")
    e = {"wrapped_broken": lambda: _wrapped(BrokenProcessPool("x")),
         "wrapped_mem": lambda: _wrapped(MemoryError())}.get(err, lambda: err)() if isinstance(err, str) else err
    res = ev.run_diag_jobs(_raise(e), _jobs(), 7)
    assert [r["qid"] for r in res] == ["q1", "q2"] and all(r["error"].startswith("pool: ") for r in res)
    with pytest.raises(type(e)):
        ev.run_diag_jobs(_raise(e), _jobs(), 5)                 # depth 5: loud


@pytest.mark.parametrize("depth", [5, 7])
def test_diag_code_errors_are_loud(depth):
    ev = load_script("hard_probe_eval")
    for e in (ValueError("bug"), KeyError("k"), _wrapped(ValueError("bug in diag_one"))):
        with pytest.raises(type(e)):
            ev.run_diag_jobs(_raise(e), _jobs(), depth)


def test_diag_ok_passes_through():
    ev = load_script("hard_probe_eval")
    assert ev.run_diag_jobs(lambda: [{"qid": "q1"}], _jobs(), 5) == [{"qid": "q1"}]
