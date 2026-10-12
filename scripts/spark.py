"""DGX Spark helper (plan Task 12; aarch64, NVIDIA GB10, work only in ~/abdurakhmon).
    PYTHONPATH=. python scripts/spark.py check | sync [--extra DIR ...] | setup | verify
    PYTHONPATH=. python scripts/spark.py run --name ID -- scripts/train.py sft --pairs ... --run runs/ID
    PYTHONPATH=. python scripts/spark.py status --name ID | pull --name ID
Credentials: the keys NAMED in train.yaml spark.env_target / spark.env_password of the MAIN checkout's repo-root .env
(repo_env(): parent of git's common dir, also from a worktree; GRL_REPO_ENV overrides); values are never printed,
logged or put in argv. Password via sshpass -e (SSHPASS in the ssh child's env only) -- decision D7 b."""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                    # advanced-rag-v2/
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402
from grl.hashing import sha256_file  # noqa: E402  -- stdlib only (no torch on this path)

EXCLUDE = (".env", "cache/", "data/dataset/", "data/gold/", "data/m3/rankings_report_qreal.json", "data/rankings/",
           "runs/", ".venv/")
ALWAYS = ("data/gold/splits.json",)                              # the only gold file Spark gets (the group split)
EXTRA_ALLOW = ("trajectories.jsonl", "pairs.jsonl", "splits.json")   # + rankings/** -- all a snapshot ships (§9.1)
TEST_EVAL_FILE = "test_eval/test_eval.jsonl"   # pilot 5: + test_eval/rankings/**, only with sync --test-eval
SNAP = re.compile(r"^data/(?:m5/(?:pilot|final)|m7/(?:probe|full))/[^/]+/(.+)$")
SHIP = re.compile(r"^data/(?:m5/(?:pilot|final)|m7/(?:probe|full))/[^/]+/ship$")   # --extra: a Task 8 / M7 ship/ dir only
PROBE_SNAP = "data/m7/probe/"                                    # M7 Task 7: rows of split "probe" only (§4)
PASSWORD_OPTS = ["-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive",
                 "-o", "NumberOfPasswordPrompts=1"]
TESTS = ("tests/test_m4_isolation.py tests/test_m4_graph.py tests/test_m4_gnn.py tests/test_m4_state.py "
         "tests/test_m4_policy.py tests/test_m4_rollout.py tests/test_m4_variants.py tests/test_m5_log.py "
         "tests/test_m5_data.py tests/test_m5_sft.py tests/test_m5_grpo.py tests/test_m5_calibrate.py "
         "tests/test_m5_spark.py tests/test_m3_mdp.py tests/test_m3_reward.py tests/test_m3_policies.py "
         "tests/test_m3_operators.py tests/test_m5_rerank.py tests/test_m5_pilot5.py tests/test_m5_grpo5b.py tests/test_m5_rank.py "
         "tests/test_m7_structure.py tests/test_m7_struct_data.py tests/test_m7_struct_env.py "
         "tests/test_m7_struct_policy.py tests/test_m7_templates.py tests/test_m7_compare.py tests/test_m7_probe_snapshot.py "
         "tests/test_m7_fixes2.py tests/test_m7_expert_struct.py tests/test_m7_full_make.py "
         "tests/test_m7_paraphrase_judge.py tests/test_m8_augment.py tests/test_m8_train.py "
         "tests/test_m7_q1x_struct.py tests/test_m7_q1x_templates.py tests/test_m7_amend_op.py")   # m7: incl. the slow real-data guard
VERIFY_MARKS = "not integration and not model and not corpus and not cuda"   # replaces addopts' -m: slow runs here
NAME = re.compile(r"^[A-Za-z0-9._-]+$")
CACHE_VARS = {                                                   # shared account: nothing outside ~/abdurakhmon
    "XDG_CACHE_HOME": ".cache", "PIP_CACHE_DIR": ".cache/pip", "MPLCONFIGDIR": ".cache/matplotlib",
    "CUDA_CACHE_PATH": ".cache/nv/ComputeCache",                 # GB10 PTX JIT (default ~/.nv/ComputeCache)
    "TRITON_HOME": ".cache/triton", "TRITON_CACHE_DIR": ".cache/triton/cache", "HF_HOME": ".cache/huggingface",
    "TORCH_HOME": ".cache/torch", "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor", "TMPDIR": "tmp"}
QUIET_VARS = {"ONEDNN_JIT_PROFILE": "0", "DNNL_JIT_PROFILE": "0"}   # oneDNN (in torch) wrote /tmp/perf-<pid>.map
HOST_VARS = {"GRL_HOST": "spark"}          # grl.hard.probe.probe_ids: no ids.json here, fall back to split / qid guards
REMOTE_ENV = ("export " + " ".join(f'{k}="$HOME/abdurakhmon/{v}"' for k, v in CACHE_VARS.items()) + " "
              + " ".join(f'{k}="{v}"' for k, v in {**QUIET_VARS, **HOST_VARS}.items()) + "; "
              + "mkdir -p " + " ".join(f'"${k}"' for k in CACHE_VARS) + "; ")   # in the command: after ~/.bashrc


def repo_env(git_common_dir: str | None = None) -> Path:
    """The repo-root .env of the MAIN checkout (a worktree has none and gets none)."""
    if os.environ.get("GRL_REPO_ENV"):
        return Path(os.environ["GRL_REPO_ENV"])
    if git_common_dir is None:
        git_common_dir = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                                        capture_output=True, text=True, check=True).stdout.strip()
    return Path(git_common_dir).parent / ".env"


def parse_target(s: str) -> tuple[str, int | None]:
    try:
        toks = shlex.split((s or "").strip())
    except ValueError:
        raise ValueError("unbalanced quotes in the ssh target") from None
    if toks and toks[0] == "ssh":
        toks = toks[1:]
    port, cands, i = None, [], 0
    while i < len(toks):
        if toks[i] == "-p" and i + 1 < len(toks):
            port, i = _port(toks[i + 1]), i + 2
            continue
        if not toks[i].startswith("-"):
            cands.append(toks[i])
        i += 1
    target = next((c for c in cands if "@" in c), cands[-1] if cands else None)   # user@host beats "-i key" paths
    if not target:
        raise ValueError("no ssh target in the .env value")
    if "@" in target and target.count(":") == 1 and target.rsplit(":", 1)[1].isdigit():
        target, p = target.rsplit(":", 1)
        port = port or _port(p)
    return target, port


def _port(p: str) -> int:
    if not p.isdigit() or not 0 < int(p) < 65536:
        raise ValueError("the ssh port is not a number in 1..65535")   # never echoes the value
    return int(p)


def check_remote_dir(d: str) -> str:
    if not d.startswith("abdurakhmon/") or ".." in d.split("/") or d.startswith("/"):
        raise ValueError("spark.remote_dir must stay inside ~/abdurakhmon (relative, no '..')")
    return d


def redact(text: str, secrets) -> str:
    for s in sorted((x for x in secrets if x), key=len, reverse=True):
        text = text.replace(s, "***")
    return text


def ssh_argv(target: str, port, password, use_key: bool):
    base = ["ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new"] + (["-p", str(port)] if port else [])
    env = {k: v for k, v in os.environ.items() if k != "SSHPASS"}
    if use_key:                                                  # spark.auth: key (D7 a, optional fallback)
        return ["ssh", "-o", "BatchMode=yes"] + base[1:] + [target], env
    if not password:
        raise SystemExit("spark.auth: password needs the Spark password in the repo-root .env (D7 b)")
    if not shutil.which("sshpass"):
        raise SystemExit("install sshpass (brew install hudochenkov/sshpass/sshpass) -- spark.auth: password, D7 b")
    env["SSHPASS"] = password                                    # the ssh child's env only, never argv
    return ["sshpass", "-e"] + base + PASSWORD_OPTS + [target], env


def _snapshot_rel(p: str, extra) -> str | None:
    for d in extra:
        d = d.rstrip("/") + "/"
        if p.startswith(d):
            return p[len(d):]
    m = SNAP.match(p)
    if m and extra:
        return ""                                                # --extra given: only its ship/ dirs, nothing else
    return m.group(1) if m else None


def _is_test_eval(rel: str) -> bool:
    return rel == TEST_EVAL_FILE or rel.startswith("test_eval/rankings/")


def _in_extra(p: str, extra) -> bool:
    return any(p.startswith(d.rstrip("/") + "/") for d in extra)


def select_files(tracked, artifacts, extra=(), test_eval: bool = False) -> list[str]:
    """test_eval (pilot 5, sync --test-eval): also the --extra ship dirs' test_eval/test_eval.jsonl + rankings/**."""
    def ok(p: str) -> bool:
        rel = _snapshot_rel(p, extra)
        if rel is not None:                                      # a pilot / final snapshot: allowlist only (F5)
            if test_eval and _in_extra(p, extra) and _is_test_eval(rel):
                return True
            return rel in EXTRA_ALLOW or rel.startswith("rankings/")
        return p in ALWAYS or (not p.startswith(EXCLUDE) and "/.env" not in p)
    return sorted({p for p in tracked if ok(p)} | {p for p in artifacts if ok(p)})


def check_extra(root, d: str) -> str:
    """--extra DIR must be a Task 8 ship/ dir (data/m5/{pilot,final}/<sha>/ship) whose manifest.json verifies; returns
    it ROOT-relative, so it lands at the same path on Spark (ship/rankings/latest.json "dir" is ROOT-relative)."""
    root = Path(root).resolve()
    p = Path(d) if Path(d).is_absolute() else root / d
    try:
        rel = Path(os.path.normpath(p)).relative_to(root).as_posix()
    except ValueError:
        raise SystemExit(f"--extra {d}: not under {root}") from None
    if ".." in Path(d).parts or not SHIP.match(rel) or not (root / rel).is_dir():
        raise SystemExit(f"--extra {d}: only a snapshot ship/ dir data/m5/{{pilot,final}}/<sha>/ship or "
                         f"data/m7/{{probe,full}}/<sha>/ship is shipped (F5)")
    from grl.train.data import ALLOWED_SPLITS, SplitGuardError
    from grl.train.pilot import TEST_CANDIDATES, _jsonl, check_ship, verify_ship  # numpy only, no torch
    snap = root / rel[: -len("/ship")]
    try:
        verify_ship(snap)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"--extra {rel}: manifest check failed: {e}") from None
    if rel.startswith(PROBE_SNAP):                               # M7 probe: probe rows / keys only
        from grl.hard.probe import check_probe_ship
        try:
            check_probe_ship(root / rel)
        except (SplitGuardError, FileNotFoundError, KeyError) as e:
            raise SystemExit(f"--extra {rel}: ship check failed: {e}") from None
        return rel
    try:                                                         # defence in depth: rows / ranking keys (Task 8)
        tc = snap / TEST_CANDIDATES
        variant = json.loads((root / rel / "manifest.json").read_text(encoding="utf-8")).get("variant")   # M7 joint
        check_ship(root / rel, {r["id"] for r in _jsonl(snap / "candidates.jsonl") if r["split"] in ALLOWED_SPLITS},
                   {r["id"] for r in _jsonl(tc) if r["split"] == "test"} if tc.exists() else None, variant)
    except (SplitGuardError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f"--extra {rel}: ship check failed: {e}") from None
    return rel


def check_rows(root, files, extra=()) -> None:
    """No row of a test split (or Q-real) ever leaves the Mac: every shipped snapshot JSONL row is train / val --
    except the pilot 5 test-eval file of an --extra ship dir, every row of which must be a test row (and nothing
    else); its loader (grl.train.data.load_test_items) and every training loader's refusal are the second guard.
    M7: every row of a probe snapshot (data/m7/probe/<sha>/ship) must be split "probe" (grl.hard.probe)."""
    for p in files:
        rel = _snapshot_rel(p, extra)
        if not p.endswith(".jsonl") or rel is None:
            continue
        strict = False                                           # every row must carry the wanted split
        if p.startswith(PROBE_SNAP):                             # M7 probe ship: every pairs row is a probe row
            want, strict = ("probe",), rel == "ship/pairs.jsonl" or rel == "pairs.jsonl"
        else:
            want = ("test",) if _in_extra(p, extra) and rel == TEST_EVAL_FILE else ("train", "val")
            strict = want == ("test",)
        bad = {}
        with open(Path(root) / p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line).get("split")
                    if (s is not None or strict) and s not in want:
                        bad[s] = bad.get(s, 0) + 1
        if bad:
            raise SystemExit(f"{p}: rows with split {sorted(bad)} ({sum(bad.values())}) -- never shipped to Spark (§9.1)")


class Remote:
    def __init__(self, cfg: dict):
        from dotenv import dotenv_values
        vals = dotenv_values(repo_env())
        kt, kp, auth = cfg["env_target"], cfg["env_password"], cfg.get("auth", "password")
        if auth not in ("password", "key"):
            raise SystemExit(f"spark.auth {auth!r} is not password | key (D7)")
        if not vals.get(kt):
            raise SystemExit(f"repo-root .env has no {kt} (Spark ssh target) -- decision D7")
        try:
            self.target, self.port = parse_target(vals[kt])
        except ValueError as e:
            raise SystemExit(f"repo-root .env {kt}: cannot parse the ssh target ({e})") from None
        self.password = vals.get(kp) or None
        if auth == "password" and not self.password:
            raise SystemExit(f"repo-root .env has no {kp} (Spark password; spark.auth: password) -- decision D7 b")
        self.secrets = [self.password, self.target, self.target.split("@")[-1]]
        self.dir = check_remote_dir(cfg["remote_dir"])
        self.argv, self.env = ssh_argv(self.target, self.port, self.password, use_key=auth == "key")   # no probe

    def sh(self, cmd: str, stdin=None, check: bool = True) -> int:
        r = subprocess.run(self.argv + [REMOTE_ENV + cmd], env=self.env, stdin=stdin, stderr=subprocess.PIPE, text=False)
        if r.stderr:
            sys.stderr.write(redact(r.stderr.decode("utf-8", "replace"), self.secrets))
        if check and r.returncode:
            raise SystemExit(f"remote command failed with exit {r.returncode}")
        return r.returncode

    def home(self, rel: str = "") -> str:
        return f'"$HOME/{self.dir}{("/" + rel) if rel else ""}"'


def artifact_files(version: str, extra) -> list[str]:
    pats = ["data/graph/latest.json", f"data/graph/{version}/*", "data/graph/chunk_vecs.npy",
            "data/graph/chunk_vecs_ids.json", "data/records/**/*", "data/lexicon/**/*", "data/gold/splits.json",
            "data/hard/struct_vecs.npz",                          # m7 structure-fragment vectors (once pulled)
            "data/hard/struct_art_vecs.npz"]                      # m7 Task 6c: article structure-fragment vectors
    pats += [f"{d.rstrip('/')}/**/*" for d in extra]
    out = set()
    for p in pats:
        out |= {str(f.relative_to(ROOT)) for f in ROOT.glob(p) if f.is_file()}
    return sorted(out)


def source_head(root) -> str:
    """git HEAD of the synced tree, + " dirty" when a tracked file differs from HEAD (sync tars the working tree, so
    the run then does not use exactly the committed code). Written to data/m5/SOURCE_GIT_HEAD and printed as the first
    line of every run log (fix round 1 of Task 6c: the run log records the code it ran)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    return head + (" dirty" if dirty else "")


RUN_HEAD = 'echo "SOURCE_GIT_HEAD $(cat data/m5/SOURCE_GIT_HEAD 2>/dev/null || echo unknown)"'


def cmd_sync(r: Remote, a, cfg) -> int:
    a.extra = [check_extra(ROOT, d) for d in a.extra]
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    arts = select_files([], artifact_files(config.load("train")["env"]["graph_version"], a.extra), a.extra,
                        a.test_eval)
    files = select_files(tracked, arts, a.extra, a.test_eval)
    gone = [p for p in files if not (ROOT / p).exists()]       # skip-worktree files absent from this worktree
    if gone:
        print(f"not in this checkout, skipped: {gone}")
        files = [p for p in files if p not in set(gone)]
    check_rows(ROOT, files, a.extra)                             # refuses before anything is written or sent
    m5 = ROOT / "data/m5"
    m5.mkdir(parents=True, exist_ok=True)
    (m5 / "SOURCE_GIT_HEAD").write_text(source_head(ROOT) + "\n", encoding="utf-8")
    (m5 / "spark_manifest.sha256").write_text("".join(f"{sha256_file(ROOT / p)}  {p}\n" for p in arts),
                                               encoding="utf-8")
    files += ["data/m5/SOURCE_GIT_HEAD", "data/m5/spark_manifest.sha256"]
    lst = m5 / "spark_files.txt"
    lst.write_text("\n".join(files) + "\n", encoding="utf-8")
    r.sh(f"mkdir -p {r.home()}")
    tar = subprocess.Popen(["tar", "-c", "-L", "--no-xattrs", "-f", "-", "-T", str(lst)], cwd=ROOT, stdout=subprocess.PIPE,
                           env={**os.environ, "COPYFILE_DISABLE": "1"})
    rc = r.sh(f"tar -x -f - -C {r.home()}", stdin=tar.stdout)
    tar.wait()
    print(f"synced {len(files)} files ({sum((ROOT / f).stat().st_size for f in files) / 1e6:.0f} MB), exit {rc}")
    return rc or tar.returncode


def cmd_pull(r: Remote, name: str, out: Path) -> int:
    """runs/<name> and / or runs/<name>.log from Spark into out/; every stderr line (remote and local tar) is redacted."""
    import threading
    out.mkdir(parents=True, exist_ok=True)
    pull = subprocess.Popen(r.argv + [REMOTE_ENV + f"cd {r.home('runs')} && tar -c -f - $(ls -d {name} {name}.log 2>/dev/null)"],
                            env=r.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    t = threading.Thread(target=lambda: err.append(pull.stderr.read()), daemon=True)   # drained: no pipe deadlock
    t.start()
    loc = subprocess.run(["tar", "-x", "-f", "-", "-C", str(out)], stdin=pull.stdout, stderr=subprocess.PIPE)
    pull.stdout.close()
    pull.wait()
    t.join()
    for e in (b"".join(err), loc.stderr):
        if e:
            sys.stderr.write(redact(e.decode("utf-8", "replace"), r.secrets))
    print(f"pulled runs/{name} -> {out / name}")
    return loc.returncode or pull.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "sync", "setup", "verify", "run", "status", "pull"])
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--test-eval", action="store_true", help="sync: also the --extra ship dirs' test_eval/ (pilot 5 "
                    "final report; never read by training)")
    ap.add_argument("--name")
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)      # argparse.REMAINDER would swallow --name / --extra
    a = ap.parse_args(argv[:cut])
    a.rest = argv[cut + 1:]
    if (a.cmd == "run") != bool(a.rest):
        ap.error("run needs `-- <python args>`; only run takes them")
    cfg = config.load("train")["spark"]
    if a.cmd in ("run", "status", "pull") and not (a.name and NAME.match(a.name) and set(a.name) != {"."}):
        ap.error("--name [A-Za-z0-9._-]+ is required")
    r = Remote(cfg)
    if a.cmd == "check":
        return r.sh('uname -m; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; '
                    'python3.12 --version; nproc; free -g | head -2; df -h "$HOME/abdurakhmon" | tail -1')
    if a.cmd == "sync":
        return cmd_sync(r, a, cfg)
    if a.cmd == "setup":
        return r.sh(f"cd {r.home()} && TORCH_VERSION={shlex.quote(cfg['torch_version'])} "
                    f"TORCH_INDEX={shlex.quote(cfg['torch_index'])} bash scripts/spark_setup.sh")
    if a.cmd == "verify":
        return r.sh(f"cd {r.home()} && sha256sum --quiet -c data/m5/spark_manifest.sha256 && "
                    f".venv/bin/python -m pytest -q -m {shlex.quote(VERIFY_MARKS)} {TESTS} && .venv/bin/python -m pytest -q -m cuda tests/test_m5_cuda.py")
    if a.cmd == "run":
        args = " ".join(shlex.quote(x) for x in a.rest)
        local = source_head(ROOT)
        synced = (ROOT / "data/m5/SOURCE_GIT_HEAD").read_text().strip() if (ROOT / "data/m5/SOURCE_GIT_HEAD").exists() \
            else "unknown"
        if local != synced:
            print(f"warning: local {local} != last sync {synced} (run spark.py sync to ship the current code)",
                  file=sys.stderr)
        return r.sh(f"cd {r.home()} && mkdir -p runs && ({RUN_HEAD}) > runs/{a.name}.log && "
                    f"(setsid nohup env PYTHONPATH=. .venv/bin/python {args} "
                    f">> runs/{a.name}.log 2>&1 < /dev/null &) && echo started {a.name}")
    if a.cmd == "status":
        return r.sh(f"cd {r.home()} && tail -n 20 runs/{a.name}.log; tail -n 2 runs/{a.name}/metrics.jsonl", check=False)
    return cmd_pull(r, a.name, ROOT / "runs" / "spark")


if __name__ == "__main__":
    sys.exit(main())
