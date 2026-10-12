"""Spark helpers without a network (plan Task 12): the main checkout's .env, target parsing, redaction, ssh argv
never carrying the password (D7 b: sshpass -e, no key probe), the sync file selection (no Q-real, no test split),
env fingerprints, run comparison."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.scripts_import import load_script


def test_repo_env_is_the_main_checkout_env(monkeypatch, tmp_path):
    sp = load_script("spark")
    monkeypatch.delenv("GRL_REPO_ENV", raising=False)
    assert sp.repo_env(str(tmp_path / "main" / ".git")) == tmp_path / "main" / ".env"   # a fake git-common-dir
    git = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=sp.ROOT,
                         capture_output=True, text=True)
    if git.returncode == 0:                                       # the Spark copy (tar, no .git) skips this part
        assert sp.repo_env() == Path(git.stdout.strip()).parent / ".env"   # in the worktree: the main checkout
    monkeypatch.setenv("GRL_REPO_ENV", str(tmp_path / "x.env"))
    assert sp.repo_env() == tmp_path / "x.env"


def test_parse_target_and_remote_dir():
    sp = load_script("spark")
    assert sp.parse_target("ssh -p 2222 u@1.2.3.4") == ("u@1.2.3.4", 2222)
    assert sp.parse_target("u@h") == ("u@h", None) and sp.parse_target("u@h:2200") == ("u@h", 2200)
    assert sp.parse_target("ssh -i ~/.ssh/k u@h -p 22") == ("u@h", 22)   # the token with '@' wins
    with pytest.raises(ValueError):
        sp.parse_target("   ")
    sp.check_remote_dir("abdurakhmon/grl/advanced-rag-v2")
    for bad in ("grl", "abdurakhmon/../x", "/abdurakhmon/x"):
        with pytest.raises(ValueError):
            sp.check_remote_dir(bad)


def test_redact_and_ssh_argv_never_carry_the_password(monkeypatch):
    sp = load_script("spark")
    assert sp.redact("x s3cret y u@h", ["s3cret", "u@h"]) == "x *** y ***"
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/sshpass")
    argv, env = sp.ssh_argv("u@h", 2222, "s3cret", use_key=False)
    assert argv[:2] == ["sshpass", "-e"] and "s3cret" not in " ".join(argv) and env["SSHPASS"] == "s3cret"
    assert "BatchMode=yes" not in argv and "PubkeyAuthentication=no" in argv and argv[-1] == "u@h"
    assert "NumberOfPasswordPrompts=1" in argv and "SSHPASS" not in os.environ
    argv, env = sp.ssh_argv("u@h", None, None, use_key=True)
    assert "BatchMode=yes" in argv and "SSHPASS" not in env and argv[-1] == "u@h"
    with pytest.raises(SystemExit):
        sp.ssh_argv("u@h", None, None, use_key=False)
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(SystemExit, match="brew install"):
        sp.ssh_argv("u@h", None, "s3cret", use_key=False)


def test_remote_password_auth_runs_no_probe_and_names_only_the_key(monkeypatch, tmp_path):
    sp = load_script("spark")
    f = tmp_path / ".env"
    f.write_text('SHH="ssh -p 2222 u@h"\nPASSWORD=s3cret\n', encoding="utf-8")
    monkeypatch.setenv("GRL_REPO_ENV", str(f))
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/sshpass")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no ssh probe under auth: password"))
    cfg = {"auth": "password", "env_target": "SHH", "env_password": "PASSWORD", "remote_dir": "abdurakhmon/grl/x"}
    r = sp.Remote(cfg)
    assert r.argv[:2] == ["sshpass", "-e"] and (r.target, r.port) == ("u@h", 2222)
    assert r.env["SSHPASS"] == "s3cret" and "SSHPASS" not in os.environ and "s3cret" not in " ".join(r.argv)
    f.write_text("SHH=u@h\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="PASSWORD"):
        sp.Remote(cfg)


def test_select_files_excludes_secrets_qreal_test_sets_and_caches():
    sp = load_script("spark")
    tracked = ["grl/model/gnn.py", ".env", "cache/dataset/x.json", "data/dataset/fable/reviewed.jsonl",
               "data/gold/qreal.jsonl", "data/gold/review.jsonl", "data/gold/alt_review.json",
               "data/gold/meta.json", "data/gold/splits.json", "data/m3/rankings_report_qreal.json",
               "data/m3/oracle_profile.json", "data/rankings/qreal/latest.json", "runs/a/x"]
    arts = ["data/graph/40d525a97cb9/pyg.pt", "data/records/QONUNLAR-1.json", "data/m5/pilot/abc/trajectories.jsonl"]
    assert sp.select_files(tracked, arts, ["data/m5/pilot/abc"]) == sorted(
        ["grl/model/gnn.py", "data/gold/splits.json", "data/m3/oracle_profile.json", *arts])


def test_extra_dirs_ship_only_the_allowlist_and_never_a_test_row(tmp_path):
    sp = load_script("spark")
    d = "data/m5/pilot/abc"
    keep = [f"{d}/{n}" for n in ("trajectories.jsonl", "pairs.jsonl", "splits.json", "rankings/latest.json",
                                 "rankings/v1/qvec.npy")]
    drop = [f"{d}/{n}" for n in ("verified.jsonl", "reviewed.jsonl", "manual_sample.jsonl", "candidates.jsonl",
                                 "stats.json", "snapshot.json", "fp_mac.json")]
    assert sp.select_files([], keep + drop, [d]) == sorted(keep)
    assert sp.select_files(["data/m5/final/xyz/stats.json", "data/m5/final/xyz/splits.json"], [], []) == \
        ["data/m5/final/xyz/splits.json"]                         # tracked snapshot files obey the allowlist too
    (tmp_path / d).mkdir(parents=True)
    f = tmp_path / d / "trajectories.jsonl"
    f.write_text('{"id": "a", "split": "train"}\n{"id": "b", "split": "val"}\n', encoding="utf-8")
    sp.check_rows(tmp_path, [f"{d}/trajectories.jsonl", f"{d}/splits.json"])
    f.write_text(f.read_text(encoding="utf-8") + '{"id": "c", "split": "test"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="test"):
        sp.check_rows(tmp_path, [f"{d}/trajectories.jsonl"])


def test_fingerprint_is_stable_and_compare_finds_changes(tmp_path):
    from graphrag.env.reward import Weights
    from grl.train.data import load_items
    from tests.m4_toy import TOY_RANKS, toy_env, toy_pairs
    cs = load_script("check_spark")
    env = toy_env()
    items = load_items(toy_pairs(tmp_path / "p.jsonl"), env.data, ("train", "val"))
    a = cs.fingerprint(env, TOY_RANKS, items, Weights())
    assert a == cs.fingerprint(toy_env(), TOY_RANKS, items, Weights()) and cs.compare(a, a) == []
    b = {k: dict(v) for k, v in a.items()}
    b["p1|q"]["reward"] += 1e-9
    assert cs.compare(a, b) == ["p1|q: reward"]


def test_max_abs_diff_of_metric_rows():
    cs = load_script("check_spark")
    ra = [{"phase": "sft", "step": 1, "time": 1.0, "loss": 0.5, "op_hist": {"a": 1}}]
    rb = [{"phase": "sft", "step": 1, "time": 9.0, "loss": 0.5 + 1e-7, "op_hist": {"a": 1}}]
    assert cs.max_abs_diff(ra, ra) == 0.0 and cs.max_abs_diff(ra, rb) == pytest.approx(1e-7)
    assert cs.max_abs_diff(ra, []) == float("inf")


def test_extra_must_be_a_verified_ship_dir(tmp_path):
    """--extra takes only a Task 8 ship/ dir under data/m5/{pilot,final}/<sha>/ whose manifest verifies; it keeps its
    ROOT-relative path (ship/rankings/latest.json "dir" is ROOT-relative)."""
    import hashlib
    import json
    sp = load_script("spark")
    ship = tmp_path / "data/m5/pilot/abc/ship"
    (ship / "rankings").mkdir(parents=True)
    (ship / "trajectories.jsonl").write_text('{"id": "a", "split": "train"}\n', encoding="utf-8")
    (ship / "rankings/latest.json").write_text('{"dir": "data/m5/pilot/abc/ship/rankings/v1"}\n', encoding="utf-8")
    man = {n: hashlib.sha256((ship / n).read_bytes()).hexdigest() for n in ("trajectories.jsonl", "rankings/latest.json")}
    (ship / "manifest.json").write_text(json.dumps({"sha256": man}), encoding="utf-8")
    (ship.parent / "candidates.jsonl").write_text('{"id": "a", "split": "train"}\n{"id": "t", "split": "test"}\n',
                                                  encoding="utf-8")
    assert sp.check_extra(tmp_path, "data/m5/pilot/abc/ship/") == "data/m5/pilot/abc/ship"
    assert sp.check_extra(tmp_path, str(ship)) == "data/m5/pilot/abc/ship"          # absolute -> ROOT-relative
    for bad in ("data/m5/pilot/abc", "data/dataset/fable", "data/m5/pilot/abc/ship/../ship2", "/etc"):
        with pytest.raises(SystemExit):
            sp.check_extra(tmp_path, bad)
    (ship / "trajectories.jsonl").write_text('{"id": "t", "split": "train"}\n', encoding="utf-8")   # a test-split id
    man["trajectories.jsonl"] = hashlib.sha256((ship / "trajectories.jsonl").read_bytes()).hexdigest()
    (ship / "manifest.json").write_text(json.dumps({"sha256": man}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ship check"):          # manifest fine, check_ship (Task 8) refuses
        sp.check_extra(tmp_path, "data/m5/pilot/abc/ship")
    (ship / "trajectories.jsonl").write_text('{"id": "a", "split": "val"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="manifest"):
        sp.check_extra(tmp_path, "data/m5/pilot/abc/ship")


def test_remote_commands_keep_caches_and_tmp_inside_abdurakhmon(monkeypatch, tmp_path):
    """pip / matplotlib / torch caches and TMPDIR stay under ~/abdurakhmon on the shared Spark account."""
    sp = load_script("spark")
    f = tmp_path / ".env"
    f.write_text("SHH=u@h\nPASSWORD=s3cret\n", encoding="utf-8")
    monkeypatch.setenv("GRL_REPO_ENV", str(f))
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/sshpass")
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: seen.append(argv) or subprocess.CompletedProcess(argv, 0, b"", b""))
    r = sp.Remote({"auth": "password", "env_target": "SHH", "env_password": "PASSWORD", "remote_dir": "abdurakhmon/grl/x"})
    assert r.sh("echo hi") == 0
    cmd = seen[0][-1]
    want = {"XDG_CACHE_HOME": ".cache", "PIP_CACHE_DIR": ".cache/pip", "MPLCONFIGDIR": ".cache/matplotlib",
            "CUDA_CACHE_PATH": ".cache/nv/ComputeCache", "TRITON_CACHE_DIR": ".cache/triton/cache",
            "TRITON_HOME": ".cache/triton", "HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch",
            "TMPDIR": "tmp"}
    for k, v in want.items():
        assert f'{k}="$HOME/abdurakhmon/{v}"' in cmd
    assert 'ONEDNN_JIT_PROFILE="0"' in cmd and 'DNNL_JIT_PROFILE="0"' in cmd   # no /tmp/perf-<pid>.map (oneDNN)
    assert cmd.startswith("export ") and cmd.endswith("echo hi") and "s3cret" not in " ".join(seen[0])
    setup = (sp.ROOT / "scripts/spark_setup.sh").read_text(encoding="utf-8")
    head = setup[: setup.index("PY=${PY:-python3.12}")]          # exported before python / pip ever run
    assert "ONEDNN_JIT_PROFILE=0" in head and "DNNL_JIT_PROFILE=0" in head
    for k, v in want.items():
        assert f'{k}="$C/{v[len(".cache/"):]}"' in head if v.startswith(".cache/") else f"{k}=" in head


def _remote(monkeypatch, tmp_path, target="u@h"):
    sp = load_script("spark")
    f = tmp_path / ".env"
    f.write_text(f'SHH="{target}"\nPASSWORD=s3cret\n', encoding="utf-8")
    monkeypatch.setenv("GRL_REPO_ENV", str(f))
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/sshpass")
    cfg = {"auth": "password", "env_target": "SHH", "env_password": "PASSWORD", "remote_dir": "abdurakhmon/grl/x"}
    return sp, cfg


def test_pull_redacts_remote_and_local_stderr(monkeypatch, tmp_path, capsys):
    sp, cfg = _remote(monkeypatch, tmp_path)
    r = sp.Remote(cfg)
    src = tmp_path / "remote_runs"
    (src / "run1").mkdir(parents=True)
    (src / "run1" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    # stands in for ssh: $0 = the source dir, the remote command becomes $1 and is ignored
    r.argv = ["sh", "-c", 'echo "remote says s3cret at u@h" >&2; tar -c -f - -C "$0" run1', str(src)]
    assert sp.cmd_pull(r, "run1", tmp_path / "out") == 0
    assert (tmp_path / "out" / "run1" / "metrics.jsonl").exists()
    err = capsys.readouterr().err
    assert "s3cret" not in err and "u@h" not in err and "remote says *** at ***" in err


def _no_network(monkeypatch, sp):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: pytest.fail(f"unexpected subprocess {argv[:1]}"))
    monkeypatch.setattr(sp.Remote, "sh", lambda self, cmd, stdin=None, check=True: seen.append(cmd) or 0)
    return seen


def test_name_of_only_dots_is_refused(monkeypatch, tmp_path):
    sp, _ = _remote(monkeypatch, tmp_path)
    seen = _no_network(monkeypatch, sp)
    for bad in (".", "..", "...", "a/b", ""):
        with pytest.raises(SystemExit, match="2"):               # argparse error, before any remote call
            sp.main(["status", "--name", bad])
    assert seen == []


def test_options_after_the_subcommand_are_parsed(monkeypatch, tmp_path):
    """`status --name ID`, `run --name ID -- ...`, `sync --extra DIR` (argparse.REMAINDER used to swallow them)."""
    sp, _ = _remote(monkeypatch, tmp_path)
    seen = _no_network(monkeypatch, sp)
    assert sp.main(["status", "--name", "run1"]) == 0 and "runs/run1.log" in seen[-1]
    monkeypatch.setattr(sp, "source_head", lambda root: "abc123")
    assert sp.main(["run", "--name", "r2", "--", "scripts/train.py", "sft", "--pairs", "a b"]) == 0
    assert "python scripts/train.py sft --pairs 'a b' >> runs/r2.log" in seen[-1] and "--name" not in seen[-1]
    assert 'echo "SOURCE_GIT_HEAD $(cat data/m5/SOURCE_GIT_HEAD' in seen[-1] and ") > runs/r2.log" in seen[-1]
    got = []
    monkeypatch.setattr(sp, "cmd_sync", lambda r, a, cfg: got.append(list(a.extra)) or 0)
    assert sp.main(["sync", "--extra", "data/m5/pilot/abc/ship"]) == 0 and got == [["data/m5/pilot/abc/ship"]]
    for bad in (["run", "--name", "r3"], ["status", "--name", "r1", "--", "x"]):
        with pytest.raises(SystemExit):
            sp.main(bad)


def test_malformed_port_names_only_the_key(monkeypatch, tmp_path):
    sp, cfg = _remote(monkeypatch, tmp_path, target="ssh -p 22x9secret u@h")
    with pytest.raises(SystemExit) as e:
        sp.Remote(cfg)
    assert "SHH" in str(e.value) and "22x9secret" not in str(e.value) and "u@h" not in str(e.value)
    for s in ("u@h:99999", "ssh -p 0 u@h", 'ssh "u@h'):
        with pytest.raises(ValueError) as e:
            sp.parse_target(s)
        assert "u@h" not in str(e.value) and "99999" not in str(e.value)


def test_extra_ship_dir_excludes_snapshot_root_files():
    sp = load_script("spark")
    ship = "data/m5/pilot/abc/ship"
    root_files = ["data/m5/pilot/abc/splits.json", "data/m5/pilot/abc/trajectories.jsonl",
                  "data/m5/final/xyz/splits.json"]
    shipped = [f"{ship}/trajectories.jsonl", f"{ship}/splits.json", f"{ship}/rankings/latest.json"]
    assert sp.select_files(root_files, shipped + [f"{ship}/manifest.json"], [ship]) == sorted(shipped)


def test_determinism_out_must_stay_under_runs():
    cs = load_script("check_spark")
    assert cs.check_out("runs/det") == (cs.ROOT / "runs/det").resolve()
    for bad in ("/tmp/det", "runs", "runs/../data", "../runs/x", "data/m5/x"):
        with pytest.raises(SystemExit):
            cs.check_out(bad)
    with pytest.raises(SystemExit):
        cs.main(["determinism", "--pairs", "p", "--rankings", "r", "--out", "/tmp/det"])


def test_fingerprint_digest_covers_dtype_and_shape():
    import numpy as np
    cs = load_script("check_spark")
    z32, z64 = np.zeros(2, np.float32), np.zeros(1, np.float64)      # the same 8 zero bytes
    assert z32.tobytes() == z64.tobytes() and cs.arrays_digest([z32]) != cs.arrays_digest([z64])
    a = np.arange(4, dtype=np.int64)
    assert cs.arrays_digest([a]) != cs.arrays_digest([a.reshape(2, 2)])
    assert cs.arrays_digest([a, a]) == cs.arrays_digest([a.copy(), a.copy()])


def test_verify_passes_its_own_marks_so_slow_runs_and_the_rest_stays_off(monkeypatch, tmp_path):
    """R30: a command-line -m replaces addopts' -m. addopts excludes slow; verify passes every exclusion but slow."""
    import tomllib
    sp, _ = _remote(monkeypatch, tmp_path)
    seen = _no_network(monkeypatch, sp)
    assert sp.main(["verify"]) == 0
    want = "-m 'not integration and not model and not corpus and not cuda' " + sp.TESTS
    assert want in seen[-1] and "slow" not in sp.VERIFY_MARKS
    assert 'GRL_HOST="spark"' in sp.REMOTE_ENV                  # grl.hard.probe.probe_ids: Spark has no ids.json
    addopts = tomllib.loads((sp.ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["addopts"]
    assert addopts == "-m 'not integration and not model and not corpus and not cuda and not slow'"
