"""Task 9f / 10: the once-token gate of scripts/m7_final_eval.py (single use end to end, pre-checks before any
consumption), the frozen router, the pre-registered criteria (R64), the Spark step lists. No test file is read."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.scripts_import import load_script

TOK = "a" * 32


@pytest.fixture()
def fe(tmp_path, monkeypatch):
    m = load_script("m7_final_eval")
    frozen = {"once_token_sha256": m.token_sha(TOK), "criterion": m.CRITERION,
              "ranker": {"run": "runs/p5-rr-sft", "pi_ref_sha256": "0"},
              "seeds": {s: {"spark": f"runs/x-{s}.pt", "local": f"runs/spark/x-{s}.pt", "sha256": "0"} for s in m.SEEDS}}
    monkeypatch.setattr(m, "load_frozen", lambda: frozen)
    monkeypatch.setattr(m, "_git_head", lambda: "deadbeef")
    monkeypatch.setattr(m.config, "path", lambda p: tmp_path / p)
    return m, frozen, tmp_path


def _a(**kw):
    base = {"once_token": TOK, "split": "test", "dry": False}
    base.update(kw)
    return SimpleNamespace(**base)


def _used(tmp):
    d = tmp / "data/m7/final/used"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_refuses_without_or_with_a_wrong_token(fe):
    m, _, _ = fe
    for argv in (["open", "--split", "test"], ["spark", "--split", "test", "--name", "x"],
                 ["report", "--split", "val", "--dry", "--run", "r", "--out", "o"]):
        with pytest.raises(SystemExit):
            m.main(argv)
    with pytest.raises(SystemExit, match="does not match"):
        m.gate(_a(once_token="b" * 32), "open")


def test_consume_is_once_per_stage(fe):
    m, _, tmp = fe
    t = m.token_sha(TOK)
    rec = m.consume(t, "open")
    assert rec["stage"] == "open" and json.loads(m.marker_path(t, "open").read_text()) == rec
    with pytest.raises(SystemExit, match="already used"):
        m.consume(t, "open")


def test_gate_consumes_nothing(fe):
    m, _, tmp = fe
    m.gate(_a(), "open")
    m.gate(_a(split="val", dry=True), "report")
    with pytest.raises(SystemExit):
        m.gate(_a(split="val"), "open")                    # val without --dry
    with pytest.raises(SystemExit):
        m.gate(_a(dry=True), "open")                       # test with --dry
    assert _used(tmp) == []


def test_test_ship_is_computed_from_the_token(fe):
    m, _, _ = fe
    t = m.token_sha(TOK)
    assert m.test_ship_path(t) == m.test_ship_path(t)
    assert m.test_ship_path(t).startswith("data/m7/full/f9test-") and m.test_ship_path(t).endswith("/ship")
    assert m.test_ship_path(t) != m.test_ship_path(m.token_sha("b" * 32))


def test_open_typo_or_failed_precheck_consumes_nothing(fe, monkeypatch):
    m, _, tmp = fe
    monkeypatch.setattr(m, "verify_frozen", lambda f: (_ for _ in ()).throw(SystemExit("frozen artifacts changed")))
    import grl.train.pilot as pilot
    monkeypatch.setattr(pilot, "require_dev_pg", lambda dsn: None)
    monkeypatch.setattr(m.config, "QDRANT_URL", "http://localhost:6343")
    with pytest.raises(SystemExit, match="frozen artifacts"):
        m.cmd_open(_a())
    assert _used(tmp) == []
    monkeypatch.setattr(m, "verify_frozen", lambda f: None)
    monkeypatch.setattr(m, "_git_head", lambda: "deadbeef dirty")
    with pytest.raises(SystemExit, match="not a clean commit"):
        m.cmd_open(_a())
    assert _used(tmp) == []


def _spark_args(**kw):
    base = {"once_token": TOK, "split": "test", "dry": False, "dry_testpath": False, "test_ship": None,
            "name": "f9-test", "only": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_spark_preflight_refusals(fe):
    m, _, _ = fe
    t = m.token_sha(TOK)
    for kw, msg in (({"test_ship": "data/m7/full/f9test-typo/ship"}, "is not"), ({"name": "f9-test-2"}, "named"),
                    ({"dry": True}, "--dry")):
        with pytest.raises(SystemExit, match=msg):
            m.spark_preflight(_spark_args(**kw), m.load_frozen(), t)


def test_spark_test_runs_once_under_any_name(fe, monkeypatch):
    m, _, tmp = fe
    monkeypatch.setattr(m, "spark_steps", lambda *a, **k: [])

    def pre(a, frozen, tsha):
        return tmp / "runs" / a.name, "ship", {"split": "test"}
    monkeypatch.setattr(m, "spark_preflight", pre)
    assert m.cmd_spark(_spark_args(name="f9-test")) == 0
    with pytest.raises(SystemExit, match="already used"):
        m.cmd_spark(_spark_args(name="another-name"))
    assert not (tmp / "runs" / "another-name").exists()
    assert m.cmd_spark(_spark_args(name="dry-x", dry_testpath=True)) == 0    # dry-testpath consumes nothing


def _run_dir(m, tmp, name, once):
    d = tmp / "runs/spark" / name
    d.mkdir(parents=True)
    (d / "once.json").write_text(json.dumps(once))
    (d / "steps.json").write_text(json.dumps([{"step": "x", "rc": 0}]))
    for f in ["baselines.json", "ranker.jsonl"] + [f"agent-{s}-k{K}.jsonl" for s in m.SEEDS for K in m.K_EVAL]:
        (d / f).write_text("{}")
    return f"runs/spark/{name}"


def _report_args(run, **kw):
    base = {"once_token": TOK, "split": "test", "dry": False, "run": run, "out": "runs/f9-local/test_report.json",
            "test_ship": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_report_refuses_the_dry_testpath_run_and_consumes_nothing(fe, monkeypatch):
    """The real f9-dry-testpath2 provenance (split test, this token, DRY_SNAP ship) must never yield a test report."""
    m, _, tmp = fe
    monkeypatch.setattr(m, "verify_frozen", lambda f: None)
    t = m.token_sha(TOK)
    once = {"dry": False, "git": "", "split": "test", "test_ship": "data/m7/full/f9dry-val/ship",
            "time": "2026-10-05T00:05:29", "token_sha256": t}                  # = runs/spark/f9-dry-testpath2/once.json
    run = _run_dir(m, tmp, "f9-dry-testpath2", once)
    with pytest.raises(SystemExit, match="dry-testpath"):
        m.cmd_report(_report_args(run))
    assert _used(tmp) == []


def test_report_bound_to_the_opened_ship_and_the_one_run(fe, monkeypatch):
    m, _, tmp = fe
    monkeypatch.setattr(m, "verify_frozen", lambda f: None)
    t = m.token_sha(TOK)
    ship = m.test_ship_path(t)
    good = {"split": "test", "token_sha256": t, "test_ship": ship, "dry_testpath": False, "name": "f9-test",
            "marker": {"stage": "spark"}, "test_eval_sha256": {f: "h" for f in m.TEST_EVAL_FILES}}
    run = _run_dir(m, tmp, "f9-test", good)
    with pytest.raises(SystemExit, match="not opened"):                 # no open marker / state
        m.cmd_report(_report_args(run))
    m.consume(t, "open")
    m.save_state(t, {"open": {"stage": "open"}, "ship": ship, "ship_manifest_sha256": "h", "report": None})
    (tmp / ship).mkdir(parents=True)
    (tmp / ship / "manifest.json").write_text(json.dumps({"sha256": {f: "h" for f in m.TEST_EVAL_FILES}}))
    with pytest.raises(SystemExit, match="ship manifest"):            # the ship changed since the opening
        m.cmd_report(_report_args(run))
    with pytest.raises(SystemExit, match="not the opened ship"):
        m.cmd_report(_report_args(run, test_ship="data/m7/full/f9test-other/ship"))
    other = _run_dir(m, tmp, "f9-other", {**good, "test_ship": "data/m7/full/f9test-other/ship"})
    with pytest.raises(SystemExit, match="not the opened ship"):
        m.cmd_report(_report_args(other))
    h = m._sha(tmp / ship / "manifest.json")
    m.save_state(t, {"open": {"stage": "open"}, "ship": ship, "ship_manifest_sha256": h, "report": None})
    other_files = _run_dir(m, tmp, "f9-y", {**good, "test_eval_sha256": {f: "z" for f in m.TEST_EVAL_FILES}})
    with pytest.raises(SystemExit, match="test-eval files differ"):     # Spark ran on other test files
        m.cmd_report(_report_args(other_files))
    noname = _run_dir(m, tmp, "f9-x", {**good, "name": "f9-x"})
    with pytest.raises(SystemExit, match="one Spark test run"):
        m.cmd_report(_report_args(noname))
    assert _used(tmp) == [f"{t[:16]}-open.json"]                         # report never consumed


def test_spark_steps(fe):
    m, frozen, _ = fe
    val = m.spark_steps("val", "runs/d", frozen)
    test = m.spark_steps("test", "runs/t", frozen, "data/m7/full/f9test-x/ship")
    assert [n for n, _ in test][:3] == ["ce", "baselines", "ranker"]
    assert [n for n, _ in val][:2] == ["baselines", "ranker"]
    assert sum(n.startswith("agent-") for n, _ in val) == len(m.SEEDS) * len(m.K_EVAL)
    for _, argv in val:
        assert "test" not in argv and not any("test_eval" in x for x in argv)
    for n, argv in test:
        if n != "ce":
            assert argv[argv.index("--split") + 1] == "test"
        assert any(x.endswith("test_eval/test_eval.jsonl") for x in argv)
    agent = dict(test)["agent-s02-k24"]
    assert "runs/x-s02.pt" in agent and "env.K_max=24" in agent


def test_router_equals_sklearn():
    from sklearn.linear_model import LogisticRegression
    m = load_script("m7_final_eval")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = np.repeat([0, 1, 2], 20)
    X[y == 1] += 2
    X[y == 2] -= 2
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    r = {"coef": clf.coef_, "intercept": clf.intercept_, "classes": clf.classes_, "labels": ["old", "count", "version"]}
    assert m.route(r, X) == [r["labels"][i] for i in clf.predict(X)]


def _crit():
    rows = {f"h{i:02d}": {"pattern": "version", "provenance": {"m7": {"template": "what_law_changed"}}}
            for i in range(20)}
    rows |= {f"o{i:02d}": {"pattern": "single", "provenance": {}} for i in range(10)}
    keys = sorted(rows)
    M = {f"bl:{b}@5": {k: {"all5": 0.0, "anyof5": 0.0, "R": 0.0} for k in keys}
         for b in ("dense", "bm25", "fused", "ce+fused:0.6")}
    for i in range(5):
        M["bl:bm25@5"][f"h{i:02d}"]["all5"] = 1.0                       # best baseline on the hard set: .25
    return rows, keys, M


def test_criterion_pinned_numbers(fe):
    m = fe[0]
    rows, keys, M = _crit()
    M["sys"] = {k: {"all5": 1.0, "R": 0.65} for k in keys}                # hard all@5 1.0 vs .25
    M["sys24"] = {k: {"all5": 0.0, "R": 0.62 if k.startswith("o") else 0.0} for k in keys}
    c = m.criterion(rows, M, "sys", "sys24", keys)
    assert c["best_baseline"] == "bm25" and c["n_hard"] == 20 and c["n_old"] == 10
    assert c["diff"]["mean"] == 0.75 and abs(c["diff"]["se"] - 0.099340) < 1e-5 and c["a_met"] is True
    assert c["b_met"] is False and abs(c["r37_threshold"] - 0.6436) < 1e-12        # .62 < .6436
    M["sys24"] = {k: {"all5": 0.0, "R": 0.6436} for k in keys}
    assert m.criterion(rows, M, "sys", "sys24", keys)["b_met"] is True
    M["sys"] = {k: dict(M["bl:bm25@5"][k]) for k in keys}
    assert m.criterion(rows, M, "sys", "sys24", keys)["a_met"] is False


def test_r64_both_blocks_reported(fe):
    m = fe[0]
    rows, keys, M = _crit()
    good = {k: {"all5": 1.0, "anyof5": 1.0, "R": 0.7} for k in keys}
    bad = {k: {"all5": 0.25 if k < "h05" else 0.0, "anyof5": 0.0, "R": 0.4} for k in keys}
    for s in m.SEEDS:
        M[f"sys:{s}"], M[f"sys24:{s}"] = good, good
        M[f"agent:{s}"], M[f"agent24:{s}"] = bad, bad
    M["agent:mean"], M["agent24:mean"] = bad, bad
    c = m.criteria_r64(rows, M, keys)
    assert c["headline"] == "B_system" and c["B_system"]["pass"] is True and c["A_agent_spec"]["pass"] is False
    assert set(c["A_agent_spec"]["per_seed"]) == set(m.SEEDS)
    t = m.criteria_text(c)
    assert t.splitlines()[0].startswith("(B) HEADLINE deployed system: PASS")
    assert "(A) spec §1 / R37 verbatim, struct agent, seed mean: FAIL" in t and "supp. Q2-only" in t


def test_hard_test_rows_only_behind_the_open_marker(tmp_path):
    from grl.hard.ship import hard_test_pairs
    from grl.train.data import SplitGuardError
    with pytest.raises(SplitGuardError):
        hard_test_pairs(tmp_path / "none.jsonl", {})
    with pytest.raises(SplitGuardError):
        hard_test_pairs(tmp_path / "none.jsonl", {"stage": "open", "token_sha256": "x" * 64})        # no marker
    fake = {"stage": "open", "token_sha256": "x" * 64, "marker": str(tmp_path / ("x" * 16 + "-open.json"))}
    with pytest.raises(SplitGuardError):
        hard_test_pairs(tmp_path / "none.jsonl", fake)                                            # marker missing
    (tmp_path / ("x" * 16 + "-open.json")).write_text(json.dumps({"stage": "open", "token_sha256": "y"}))
    with pytest.raises(SplitGuardError):
        hard_test_pairs(tmp_path / "none.jsonl", fake)                                            # not this record
