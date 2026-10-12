"""Run logging (IMPLEMENTATION.md §8.6–8.7), checkpoints, manifest, determinism (§9.6)."""
import json

import pytest
import torch

from grl.train.log import (Checkpoints, comparable, RunLogger, alarms, hacking_alarm, read_rows, run_manifest,
                           set_determinism)

CFG = {"tmax_share": 0.5, "op_collapse_share": 0.9, "kl_max": 0.5, "stop_share_min": 0.5}


def test_logger_jsonl_csv_truncate_and_refuses_overwrite(tmp_path):
    d = tmp_path / "r"
    lg = RunLogger(d, {"seed": 1})
    lg.log("grpo", 1, reward_mean=0.5, op_hist={"lexical": 2})
    lg.log("grpo", 2, reward_mean=0.6, op_hist={"vector": 1})
    lg.log("grpo_val", 2, val_reward_mean=0.4)
    files = lg.export_csv()
    lg.close()
    rows = read_rows(d)
    assert [(r["phase"], r["step"]) for r in rows] == [("grpo", 1), ("grpo", 2), ("grpo_val", 2)]
    assert rows[0]["op_hist"] == {"lexical": 2} and json.loads((d / "manifest.json").read_text())["seed"] == 1
    header = (d / "grpo.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"step", "reward_mean", "op_hist.lexical", "op_hist.vector"} <= set(header) and len(files) == 2
    with pytest.raises(FileExistsError):
        RunLogger(d, {"seed": 1})
    lg2 = RunLogger(d, None, resume=True)
    lg2.truncate({"grpo": 1, "grpo_val": 1})
    lg2.close()
    assert [(r["phase"], r["step"]) for r in read_rows(d)] == [("grpo", 1)]


def test_alarms_fire_on_8_7_symptoms():
    ok = {"tmax_share": 0.1, "op_hist": {"lexical": 5, "vector": 4, "stop": 9}, "kl_ref": 0.01, "stop_share": 0.9}
    assert alarms(ok, CFG) == []
    bad = {"tmax_share": 0.8, "op_hist": {"lexical": 95, "stop": 5}, "kl_ref": 0.9, "stop_share": 0.2}
    assert [a.split(":")[0] for a in alarms(bad, CFG)] == ["t_max", "collapse", "kl", "stop"]
    assert hacking_alarm([0.3, 0.4, 0.5], [0.5, 0.45, 0.4]) and not hacking_alarm([0.3, 0.4, 0.5], [0.4, 0.45, 0.5])


def test_checkpoints_keep_top3_plus_last(tmp_path):
    ck = Checkpoints(tmp_path, keep_top=3)
    for i, m in enumerate([0.1, 0.5, None, 0.3, 0.4, 0.2]):
        ck.save(f"s{i}", {"step": torch.tensor(i)}, m)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["last.pt", "s1.pt", "s3.pt", "s4.pt"]
    assert ck.best().name == "s1.pt" and int(torch.load(ck.last_path(), weights_only=True)["step"]) == 5
    assert Checkpoints(tmp_path, keep_top=3).best().name == "s1.pt"               # index survives a restart


def test_manifest_and_determinism(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("a\n")
    m = run_manifest({"env": {"graph_version": "v"}, "a": 1}, {"pairs": str(f)}, {"phase": "sft"})
    assert m["graph_version"] == "v" and len(m["inputs"]["pairs"]) == 64 and m["phase"] == "sft"
    assert {"config_sha256", "versions", "machine", "git_head"} <= set(m)
    try:
        assert set_determinism(3, "strict")["mode"] == "strict" and torch.are_deterministic_algorithms_enabled()
        a = torch.rand(3)
        set_determinism(3, "strict")
        assert torch.equal(a, torch.rand(3))
    finally:
        torch.use_deterministic_algorithms(False)


def test_checkpoint_resave_same_name_does_not_duplicate(tmp_path):
    ck = Checkpoints(tmp_path, keep_top=2)
    ck.save("a", {"x": 1}, 0.5)
    ck.save("b", {"x": 2}, 0.4)
    ck.save("a", {"x": 3}, 0.6)                                                   # resumed run re-saves "a"
    ck.save("c", {"x": 4}, 0.1)
    assert [r["name"] for r in ck.index["top"]] == ["a", "b"] and not (tmp_path / "c.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_truncate_is_exact_and_plots(tmp_path):
    d = tmp_path / "r"
    lg = RunLogger(d, {"seed": 1})
    for i in range(1, 4):
        lg.log("grpo", i, reward_mean=i / 10, op_hist={"lexical": i}, stop_share=0.9)
    lg.close()
    lg = RunLogger(d, None, resume=True)
    lg.truncate({"grpo": 2})
    lg.log("grpo", 3, reward_mean=0.3, op_hist={"lexical": 3}, stop_share=0.9)
    lg.close()
    assert [r["step"] for r in read_rows(d)] == [1, 2, 3]
    assert json.loads((d / "manifest.json").read_text())["seed"] == 1             # resume keeps the original manifest
    from grl.train.log import plot_run
    files = plot_run(d)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert files == []
    else:
        assert {f.name for f in files} >= {"grpo_reward_mean.png", "grpo_op_hist.png"} and all(f.exists() for f in files)


def test_truncate_regenerates_csv_atomically_and_read_rows_tolerates_torn_tail(tmp_path):
    d = tmp_path / "r"
    lg = RunLogger(d, {"seed": 1})
    for i in (1, 2, 3):
        lg.log("grpo", i, reward_mean=i)
        lg.log("grpo_val", i, val_reward_mean=i)
    lg.export_csv()
    lg.close()
    lg = RunLogger(d, None, resume=True)
    lg.truncate({"grpo": 2, "grpo_val": 0})
    assert not list(d.glob("*.tmp"))
    lg.export_csv()
    lg.close()
    rows = read_rows(d)
    assert [(r["phase"], r["step"]) for r in rows] == [("grpo", 1), ("grpo", 2)]
    assert not (d / "grpo_val.csv").exists()
    assert len((d / "grpo.csv").read_text().splitlines()) == 1 + 2
    assert all("time" not in r for r in comparable(rows))
    f = d / "metrics.jsonl"
    good = f.read_text()
    f.write_text(good + '{"phase": "grpo", "ste')                       # torn tail
    assert read_rows(d) == rows
    f.write_text('{"phase": bad\n' + good)                              # corrupt middle line
    with pytest.raises(json.JSONDecodeError):
        read_rows(d)


def test_checkpoints_reconcile_orphans_and_reserve_last(tmp_path):
    ck = Checkpoints(tmp_path, keep_top=1)
    ck.save("a", {"x": 1}, 0.5)
    torch.save({"x": 9}, tmp_path / "orphan.pt")
    ck2 = Checkpoints(tmp_path, keep_top=1)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["a.pt", "last.pt"]
    with pytest.raises(ValueError):
        ck2.save("last", {"x": 1}, 0.9)


def _csv_rows(path):
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_resume_equals_uninterrupted_and_csv_matches_jsonl(tmp_path):
    from grl.train.log import _flat
    import time

    def run(d, n, k=None):
        lg = RunLogger(d, {"seed": 1})
        for ph in ("grpo", "grpo_val"):
            for i in range(1, n + 1):
                lg.log(ph, i, reward_mean=i / 10, op_hist={"lexical": i})
        lg.close()
        if k is not None:
            time.sleep(0.01)
            lg = RunLogger(d, None, resume=True)
            lg.truncate({"grpo": k, "grpo_val": 0})
            for i in range(k + 1, n + 1):
                lg.log("grpo", i, reward_mean=i / 10, op_hist={"lexical": i})
            lg.export_csv()
            lg.close()

    a, b = tmp_path / "a", tmp_path / "b"
    run(a, 4, k=2)                                           # interrupted: grpo_val fully dropped, grpo redone from 3
    lg = RunLogger(b, {"seed": 1})
    for i in range(1, 5):
        lg.log("grpo", i, reward_mean=i / 10, op_hist={"lexical": i})
    lg.export_csv()
    lg.close()
    assert comparable(read_rows(a)) == comparable(read_rows(b))
    assert [r["time"] for r in read_rows(a)] != [r["time"] for r in read_rows(b)]
    for d in (a, b):
        rows = read_rows(d)
        phases = {r["phase"] for r in rows}
        assert {c.stem for c in d.glob("*.csv")} == phases == {"grpo"}
        want = [{k: ("" if v is None else str(v)) for k, v in _flat(r).items()} for r in rows if r["phase"] == "grpo"]
        got = _csv_rows(d / "grpo.csv")
        assert [{k: v for k, v in g.items() if v != ""} for g in got] == [{k: v for k, v in w.items() if v != ""} for w in want]


def test_resume_repairs_torn_tail_before_appending(tmp_path):
    d = tmp_path / "r"
    lg = RunLogger(d, {"seed": 1})
    lg.log("grpo", 1, reward_mean=0.1)
    lg.close()
    with (d / "metrics.jsonl").open("a") as fh:
        fh.write('{"phase": "grpo", "ste')
    lg = RunLogger(d, None, resume=True)
    lg.log("grpo", 2, reward_mean=0.2)
    lg.close()
    assert [r["step"] for r in read_rows(d)] == [1, 2]
