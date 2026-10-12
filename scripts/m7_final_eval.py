"""M7 Task 9f / 10: the frozen final system (R62) and the ONE strict test evaluation, behind a once-token.

Frozen system (val-selected; the test was never read): router = LogisticRegression(class_weight=balanced, C 1) on the
bge-m3 question vectors, fit on the joint ship's TRAIN rows (as pilot 9); predicted
  old     -> the pilot-5 RL ranker (runs/p5-rr-sft, CE on): its no-STOP top-5 list for @k, its top-1 pick for R;
  count   -> (Q1 + Q1x) the struct agent theta* (articles + amend_op, 34 actions, K_max 5) of seed s01 (= pilot 9);
  version -> (Q2) the 6d-a rule «amendments of the named act» (grl.hard.amend_lookup; fallback = that agent's context).
The agents of seeds s02 / s03 (scripts/m7_runs.py) are reported beside it (per seed, mean ± sd), not selected among.
Q3 is not in the hard test (R54) and not in the ship (limitation).

Success criterion (spec §1, coded before the test is seen; CRITERION below):
  (a) hard set (Q1 + Q1x + Q2 rows of the evaluated split): all@5 of the frozen system minus all@5 of the best
      baseline -- the max of dense / BM25 / fused / ce+fused:0.6 @5 on that same hard set (ties -> ce+fused), i.e. the
      strongest comparator on the evaluated split -- paired over questions: mean > 2 SE (strict STRUCT gold, R59 strict);
  (b) old set (the strict old test of pilot 5 / the old val rows in the dry run): R of the frozen system with the
      agent at K_max 24 >= 0.6636 - 0.02 (R37; 0.6636 = the pilot-5 RL ranker on the strict old test).
  Verdict = (a) and (b). The same (a) / (b) are reported for the agent alone (R37 literally: the struct agent at
  K_max 24), per seed and for the seed mean -- research results, not the headline.

Stages (every stage refuses to run without --once-token; its sha256 is in docs/m7/final/frozen.json, the token in
docs/m7/final/once_token.txt; used (token, stage) markers in data/m7/final/used/):
  freeze                         (Mac, once) fit + save the router, record every frozen artifact's sha256, write the
                                 token. Refuses when frozen.json exists.
  open   --once-token T          (Mac, dev stores; consumes the token's `open` use) the test ship
                                 data/m7/full/f9test-<sha12>/ship with ship/test_eval/ = the hard Q1 / Q2 test rows
                                 of data/hard/full/p6c_hard.jsonl (their rankings built now) + the pilot-5 strict old
                                 test (136, its rankings), via grl.train.pilot.build_ship(test_eval=True).
         --split val --dry       (no use consumed) the whole `open` path on relabelled COPIES of --limit hard and
                                 old val rows -> data/m7/full/f9dry-val/ship (rankings equal to the ship's own), so
                                 the Spark test path can be proven on it before the opening.
  spark  --once-token T --split test --test-ship D --name N    (Spark, once: refuses when runs/N exists)
         --once-token T --split val --dry --name N             (Spark, dry: the ship's val rows)
                                 CE (test only), baselines, ranker --top5, the 3 agent seeds at K_max 5 and 24.
  report --once-token T --split test --run runs/spark/N [--test-ship D]   (Mac, once per token)
         --once-token T --split val --dry --run runs/spark/N             (Mac, dry: + reproduction of pilot 9)
                                 router, rule, systems, tables, paired SE by type and overall, the criterion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402

FINAL = "data/m7/final"                   # Mac-local artifacts (data/ is not in git; sha256 in artifacts.sha256)
FROZEN = "docs/m7/final/frozen.json"       # tracked: Spark gets it with the code (spark.py sync)
TOKEN_FILE = "docs/m7/final/once_token.txt"
USED_DIR = f"{FINAL}/used"                 # one O_EXCL marker per (token, stage): a rerun is refused
ROUTER = f"{FINAL}/router.npz"
DRY_SNAP = "data/m7/full/f9dry-val"        # `open --split val --dry`: relabelled VAL copies, never test rows
SPARK_USED = "runs/f9-final-used"          # Spark-side O_EXCL marker: ONE `spark --split test` per token, any name
TEST_RUN = "f9-test"                       # the name of that one Spark run
TEST_EVAL_FILES = ("test_eval/test_eval.jsonl", "test_eval/rankings/latest.json")
RANKER_CKPT = "pi_ref.pt"
SHIP = "data/m7/full/54aa9a000b4b/ship"
LABELS = "runs/p9-expert/labels.jsonl"     # Spark paths (runs/ there = runs/spark/ on the Mac)
VAL_CE = "runs/p9-ce/ce_scores.jsonl"
RANKER_RUN = "runs/p5-rr-sft"
HARD_SRC = "data/hard/full/p6c_hard.jsonl"
OLD_TEST_SNAP = "data/m5/pilot/083e116775b3"   # pilot 5: ship/test_eval/ = the strict old test (136) + rankings
TEST_ROOT = "data/m7/full"
SEEDS = ("s01", "s02", "s03")
FROZEN_SEED = "s01"
K_EVAL = (5, 24)
FLAGS = ["--set", "struct.articles=true", "--set", "struct.amend_op=true"]
CRITERION = {"hard_cells": ["Q1", "Q1x", "Q2"], "metric": "all5", "baselines_at": 5, "se_mult": 2.0,
             "r37_ref": 0.6636, "r37_tol": 0.02, "old_K_max": 24, "gold": "strict",
             "R64": {"headline": "B_system",
                     "A_agent_spec": "spec §1 / R37 verbatim on the struct AGENT: (a) on the seed mean (per question, "
                                     "the mean of the 3 seeds' all@5) and per seed; (b) the mean over seeds of the "
                                     "agent's old-set R at K_max 24 (per seed too); pass = (a) and (b) on the seed mean",
                     "B_system": "the deployed routed system (R62, agent s01); (b) is effectively the ranker's R on "
                                 "the old test minus the misroute loss",
                     "supplementary": ["Q2-only (system)", "R59 any-of (system, agent seed mean)", "per-seed systems"]}}
STAGES = ("open", "spark", "report")


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(path, obj) -> None:
    from scripts.m7_pilot8 import _write as w
    w(path, obj)


def _git_head() -> str:
    from scripts.spark import source_head
    return source_head(ROOT)


# --- the once-token ------------------------------------------------------------------------------------------------

def token_sha(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def load_frozen() -> dict:
    f = config.path(FROZEN)
    if not f.is_file():
        raise SystemExit(f"{FROZEN} missing: run `freeze` first (the system is not frozen)")
    return json.loads(f.read_text(encoding="utf-8"))


def check_token(token: str | None, frozen: dict) -> str:
    """The token's sha256 must equal the frozen one. Returns the sha."""
    if not token:
        raise SystemExit("refused: --once-token is required (data/m7/final/once_token.txt; see task-9f-report.md)")
    t = token_sha(token)
    if t != frozen.get("once_token_sha256"):
        raise SystemExit("refused: --once-token does not match the frozen system's token")
    return t


def consume(tsha: str, stage: str, used_dir=None) -> dict:
    """Atomically mark (token, stage) used; a second use is refused. Returns the ledger record."""
    d = Path(used_dir) if used_dir else config.path(USED_DIR)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "token_sha256": tsha, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _git_head()}
    try:
        fd = os.open(d / f"{tsha[:16]}-{stage}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise SystemExit(f"refused: the once-token was already used for `{stage}` (the strict test opens once)") \
            from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def gate(a, stage: str) -> tuple[dict, str]:
    """--once-token always; --split val needs --dry; --split test refuses --dry. Consumes NOTHING: each stage consumes
    its use only after all of its pre-checks passed (fix round 1, I2), immediately before the first test read."""
    frozen = load_frozen()
    tsha = check_token(getattr(a, "once_token", None), frozen)
    if a.split == "test":
        if a.dry:
            raise SystemExit("--dry goes with --split val only")
    elif a.split != "val" or not a.dry:
        raise SystemExit("--split val runs only as --dry (the pipeline proof); the test is --split test")
    return frozen, tsha


def marker_path(tsha: str, stage: str, used_dir=None) -> Path:
    return (Path(used_dir) if used_dir else config.path(USED_DIR)) / f"{tsha[:16]}-{stage}.json"


def test_ship_path(tsha: str) -> str:
    """The opened test ship -- a function of the token only (computed, never typed)."""
    return f"{TEST_ROOT}/f9test-{hashlib.sha256((tsha + ':f9test').encode()).hexdigest()[:12]}/ship"


def state_path(tsha: str) -> Path:
    return config.path(FINAL) / f"state-{tsha[:16]}.json"


def load_state(tsha: str) -> dict:
    f = state_path(tsha)
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def save_state(tsha: str, st: dict) -> None:
    _write(state_path(tsha), st)


def head_any() -> str:
    """git HEAD (+ dirty) on the Mac; on Spark (no .git) the synced data/m5/SOURCE_GIT_HEAD."""
    head = _git_head() or ""
    if not head:
        f = config.path("data/m5/SOURCE_GIT_HEAD")
        head = f.read_text().strip() if f.is_file() else ""
    return head


def require_clean_head() -> str:
    """Tracked files equal HEAD (Mac: git; Spark: the synced SOURCE_GIT_HEAD) -- the test runs committed code."""
    head = head_any()
    if not head or head.endswith("dirty"):
        raise SystemExit(f"refused: the code is not a clean commit ({head or 'unknown HEAD'}); commit / sync first")
    return head


# --- freeze (Mac) --------------------------------------------------------------------------------------------------

def fit_router(ship, C: float = 1.0):
    """pilot 9 / m7_pilot8.py router: LogisticRegression(class_weight=balanced) on the TRAIN rows' qvecs."""
    import numpy as np
    from graphrag.retrieval.rankings import load_rankings
    from scripts.m7_pilot8 import GROUPS, _group
    from sklearn.linear_model import LogisticRegression
    pairs = _jsonl(config.path(ship) / "pairs.jsonl")
    ranks = load_rankings(config.path(ship) / "rankings" / "latest.json")
    tr = [r for r in pairs if r["split"] == "train"]
    X = np.stack([np.asarray(ranks.get(f"{r['id']}|q").qvec, np.float32) for r in tr])
    y = np.array([GROUPS.index(_group(r["pattern"])) for r in tr])
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(X, y)
    return clf, {"n_train": len(tr), "train_acc": float((clf.predict(X) == y).mean())}


def route(router: dict, qvecs) -> list[str]:
    """The frozen router: argmax of X coef^T + b (= LogisticRegression.predict) -> group name."""
    import numpy as np
    X = np.asarray(qvecs, np.float64)
    s = X @ np.asarray(router["coef"], np.float64).T + np.asarray(router["intercept"], np.float64)
    return [str(router["labels"][int(router["classes"][i])]) for i in s.argmax(1)]


def load_router(path=None) -> dict:
    import numpy as np
    z = np.load(config.path(path or ROUTER), allow_pickle=False)
    return {"coef": z["coef"], "intercept": z["intercept"], "classes": z["classes"], "labels": list(z["labels"])}


def cmd_amend(a) -> int:
    """Fix round 1 (R64, M3): pre-register the two criteria in frozen.json (token unchanged), after checking the
    ranker checkpoint against its frozen sha256. Refused once anything was consumed with the token."""
    frozen = load_frozen()
    tsha = frozen["once_token_sha256"]
    used = sorted(p.name for p in config.path(USED_DIR).glob(f"{tsha[:16]}-*")) if config.path(USED_DIR).is_dir() else []
    if used or load_state(tsha):
        raise SystemExit(f"refused: the token was already used ({used}); amend only before the opening")
    rk = config.path(f"runs/spark/{frozen['ranker']['run'][len('runs/'):]}/{RANKER_CKPT}")
    if _sha(rk) != frozen["ranker"]["pi_ref_sha256"]:
        raise SystemExit(f"{rk}: differs from the frozen ranker sha256")
    frozen.setdefault("amendments", []).append({
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": _git_head(), "reason": a.reason,
        "criterion_before": frozen.get("criterion"), "ranker_ckpt_checked": str(rk.relative_to(config.ROOT))})
    frozen["criterion"] = CRITERION
    frozen["ranker"]["ckpt"] = RANKER_CKPT
    _write(config.path(FROZEN), frozen)
    print(json.dumps({"amended": FROZEN, "criterion": CRITERION["R64"]}, ensure_ascii=False, indent=1))
    return 0


def cmd_freeze(a) -> int:
    import numpy as np
    from scripts.m7_pilot8 import GROUPS
    if config.path(FROZEN).exists():
        raise SystemExit(f"{FROZEN} exists: the system is frozen (one token per freeze)")
    seeds = dict(s.split("=", 1) for s in a.seed_ckpt)
    if sorted(seeds) != list(SEEDS):
        raise SystemExit(f"--seed-ckpt for exactly {SEEDS}")
    ck = {}
    for s, p in seeds.items():                       # local pulled copy runs/spark/<...> -> Spark path runs/<...>
        if not p.startswith("runs/spark/"):
            raise SystemExit(f"{p}: a pulled checkpoint under runs/spark/")
        ck[s] = {"spark": "runs/" + p[len("runs/spark/"):], "local": p, "sha256": _sha(config.path(p)),
                 "note": dict(x.split("=", 1) for x in a.seed_note).get(s, "")}
    clf, info = fit_router(SHIP)
    rp = config.path(ROUTER)
    rp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(rp, coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_, labels=np.array(GROUPS))
    router = load_router()
    from graphrag.retrieval.rankings import load_rankings
    pairs = _jsonl(config.path(SHIP) / "pairs.jsonl")
    ranks = load_rankings(config.path(SHIP) / "rankings" / "latest.json")
    va = [r for r in pairs if r["split"] == "val"]
    Xv = np.stack([np.asarray(ranks.get(f"{r['id']}|q").qvec, np.float32) for r in va])
    mine = route(router, Xv)
    sk = [GROUPS[int(p)] for p in clf.predict(Xv)]
    p9 = json.loads(config.path(a.p9_router).read_text(encoding="utf-8"))["routes"]
    token = secrets.token_hex(16)
    frozen = {
        "R62": "Q2 branch = the 6d-a rule (val .409 vs agent .182); chosen on val, the test sealed",
        "system": {"old": "ranker_top5 (@k) / ranker_top1 (R)", "count": f"agent:{FROZEN_SEED}",
                   "version": f"rule:{FROZEN_SEED} (fallback agent:{FROZEN_SEED})"},
        "seeds": ck, "frozen_seed": FROZEN_SEED, "criterion": CRITERION,
        "router": {"path": ROUTER, "sha256": _sha(rp), "fit": info, "C": 1.0,
                   "val_routes_equal_sklearn": mine == sk,
                   "val_routes_equal_pilot9": mine == [p9[r["id"]] for r in va]},
        "ranker": {"run": RANKER_RUN, "pi_ref_sha256": json.loads(
            config.path("runs/spark/p5-rr-sft/pi_ref.json").read_text(encoding="utf-8"))["sha256"],
            "K_max": 5, "top5": True, "articles": True},
        "rule": {"code": "grl/hard/amend_lookup.py", "sha256": _sha(config.path("grl/hard/amend_lookup.py")), "k": 5},
        "ship": {"dir": SHIP, "manifest_sha256": _sha(config.path(SHIP) / "manifest.json")},
        "test_sources": {"hard": HARD_SRC, "hard_sha256": _sha(config.path(HARD_SRC)),
                         "old_test_manifest": f"{OLD_TEST_SNAP}/ship/manifest.json",
                         "old_test_manifest_sha256": _sha(config.path(OLD_TEST_SNAP) / "ship" / "manifest.json")},
        "agent_env": {"K_max": list(K_EVAL), "flags": FLAGS},
        "val_ce": VAL_CE, "labels": LABELS, "git": _git_head(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "once_token_sha256": token_sha(token)}
    if not (frozen["router"]["val_routes_equal_sklearn"] and frozen["router"]["val_routes_equal_pilot9"]):
        raise SystemExit(f"router refit differs from pilot 9 / sklearn: {frozen['router']}")
    _write(config.path(FROZEN), frozen)
    config.path(TOKEN_FILE).write_text(token + "\n", encoding="utf-8")
    print(json.dumps({"frozen": FROZEN, "token_file": TOKEN_FILE, "router": frozen["router"]}, indent=1))
    return 0


def verify_frozen(frozen: dict, local: bool = True) -> None:
    """The frozen files present here are unchanged (Mac: router, rule code, ship manifest, test sources, seeds)."""
    checks = [(frozen["router"]["path"], frozen["router"]["sha256"]),
              (frozen["rule"]["code"], frozen["rule"]["sha256"]),
              (f"{SHIP}/manifest.json", frozen["ship"]["manifest_sha256"])]
    if local:
        checks += [(v["local"], v["sha256"]) for v in frozen["seeds"].values()]
        checks += [(f"runs/spark/{frozen['ranker']['run'][len('runs/'):]}/{RANKER_CKPT}",
                    frozen["ranker"]["pi_ref_sha256"])]                                   # fix round 1, M3
    bad = [p for p, h in checks if not config.path(p).is_file() or _sha(config.path(p)) != h]
    if bad:
        raise SystemExit(f"frozen artifacts changed or missing: {bad}")
    if frozen.get("criterion") != CRITERION:
        raise SystemExit("the coded CRITERION differs from the pre-registered frozen one (freeze --amend-r64)")


def check_ranker_spark(frozen: dict) -> None:
    p = config.path(f"{frozen['ranker']['run']}/{RANKER_CKPT}")
    if not p.is_file() or _sha(p) != frozen["ranker"]["pi_ref_sha256"]:
        raise SystemExit(f"{p}: the ranker checkpoint differs from the frozen sha256")


# --- open (Mac) ----------------------------------------------------------------------------------------------------

def _rank(pairs_file: Path, out: Path, family: str) -> Path:
    from scripts.hard_probe import _run_rankings
    _run_rankings("--family", family, "--candidates", str(pairs_file), "--out", str(out))
    return out / "latest.json"


def build_test_snapshot(d: Path, hard: list[dict], old: list[dict], old_latest: Path, hard_family: str,
                        drop_ids=frozenset()) -> dict:
    """<d>/: the joint ship's train / val rows (minus `drop_ids`, the dry run's relabelled rows) and rankings, the
    hard test rows ranked now (`hard_family`), merged with the old test rankings (`old_latest`), the test candidates,
    then build_ship(test_eval=True) / verify_ship / check_ship and the key / gold-row checks."""
    import shutil
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    from grl.hard.ship import STRUCT_VARIANT, merge_rankings
    from grl.train.data import ALLOWED_SPLITS
    from grl.train.pilot import TEST_CANDIDATES, TEST_RANKINGS, build_ship, check_ship, verify_ship
    rows = sorted(hard + old, key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("hard and old test ids overlap")
    if d.exists():
        raise SystemExit(f"{d} exists")
    d.mkdir(parents=True)
    src = config.path(SHIP).parent
    tv = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                 for r in _jsonl(src / "pairs.jsonl") if r["id"] not in drop_ids)
    for n in ("pairs.jsonl", "candidates.jsonl"):
        (d / n).write_text(tv, encoding="utf-8")
    shutil.copyfile(src / "trajectories.jsonl", d / "trajectories.jsonl")
    if drop_ids:                                         # dry: the relabelled rows leave the train / val rankings
        from grl.train.pilot import subset_rankings
        subset_rankings(src / "rankings" / "latest.json",
                        [f"{r['id']}|q" for r in _jsonl(d / "pairs.jsonl")], d / "rankings")
    else:
        shutil.copytree(src / "rankings", d / "rankings")
    (d / "hard_test_pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                     for r in hard), encoding="utf-8")
    hl = _rank(d / "hard_test_pairs.jsonl", d / "rankings_hard_test", hard_family)
    merge_rankings([old_latest, hl], d / TEST_RANKINGS)
    (d / TEST_CANDIDATES).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                                     encoding="utf-8")
    tvrows = _jsonl(d / "pairs.jsonl")
    counts = {s: sum(1 for r in tvrows if r["split"] == s) for s in ALLOWED_SPLITS}
    build_ship(d, counts, test_eval=True, variant=STRUCT_VARIANT)
    verify_ship(d)
    check_ship(d / "ship", {r["id"] for r in tvrows}, {r["id"] for r in rows}, STRUCT_VARIANT)
    data = load_struct_env_data(articles=True)
    tr = load_rankings(d / "ship" / "test_eval" / "rankings" / "latest.json", graph_version=data.version)
    if sorted(tr.keys()) != sorted(f"{r['id']}|q" for r in rows):
        raise SystemExit("test rankings keys != the test ids")
    miss = [(r["id"], g) for r in rows for g in r["C_ref"] if g not in data.chunk_row]
    if miss:
        raise SystemExit(f"test gold ids that are no struct env row: {miss[:3]}")
    from scripts.m7_pilot9 import fine_group
    return {"dir": str(d.relative_to(config.ROOT)), "ship": str((d / "ship").relative_to(config.ROOT)),
            "n": len(rows), "n_old": len(old), "n_hard": len(hard),
            "by_group": {g: sum(fine_group(r) == g for r in rows) for g in ("old", "Q1", "Q1x", "Q2")},
            "manifest_sha256": _sha(d / "ship" / "manifest.json")}


def open_preflight(a, frozen: dict, tsha: str) -> Path:
    """Every check of `open --split test` that needs no test row; nothing is consumed or written before it passes."""
    verify_frozen(frozen)
    require_clean_head()
    if _sha(config.path(HARD_SRC)) != frozen["test_sources"]["hard_sha256"] or _sha(
            config.path(OLD_TEST_SNAP) / "ship" / "manifest.json") != frozen["test_sources"]["old_test_manifest_sha256"]:
        raise SystemExit("the test sources differ from the frozen ones")
    from grl.train.pilot import verify_ship
    verify_ship(config.path(OLD_TEST_SNAP))                     # file hashes vs its manifest (no row parsed)
    d = config.path(test_ship_path(tsha)).parent
    if d.exists() or marker_path(tsha, "open").exists() or load_state(tsha):
        raise SystemExit(f"refused: already opened ({d} / marker / state exists)")
    return d


def cmd_open(a) -> int:
    from grl.threads import pin_blas
    pin_blas(2, force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from grl.train.pilot import require_dev_pg
    require_dev_pg(config.PG_DSN)
    if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
        raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
    frozen, tsha = gate(a, "open")
    if a.split == "val":
        verify_frozen(frozen)
        return _open_dry(a)
    d = open_preflight(a, frozen, tsha)
    rec = consume(tsha, "open")                                 # <- the one use; the first test read follows
    rec["marker"] = str(marker_path(tsha, "open"))
    from grl.hard.ship import hard_test_pairs
    hard, hrep = hard_test_pairs(config.path(HARD_SRC), rec)    # <- the hard test rows are read here, once
    old_te = config.path(OLD_TEST_SNAP) / "ship" / "test_eval"
    old = _jsonl(old_te / "test_eval.jsonl")
    rep_ = build_test_snapshot(d, hard, old, old_te / "rankings" / "latest.json", "hard_test")
    rep_.update({"hard": hrep, "ledger": rec})
    _write(d / "open.json", rep_)
    save_state(tsha, {"token_sha256": tsha, "ship": rep_["ship"], "ship_manifest_sha256": rep_["manifest_sha256"],
                      "open": rec, "spark_run": TEST_RUN, "report": None})
    print(json.dumps(rep_, indent=1, ensure_ascii=False))
    print("next (exact):", "\n  spark.py sync --extra", rep_["ship"], "--test-eval",
          f"\n  spark.py run --name {TEST_RUN} -- scripts/rc.py scripts/m7_final_eval.py spark --split test "
          f"--once-token <T>")
    return 0


def _open_dry(a) -> int:
    """`open` on val rows, into DRY_SNAP (a test-eval-shaped ship whose "test" rows are relabelled VAL copies -- it
    proves the Spark test path, `spark --split test --test-ship DRY_SNAP/ship`, before the real opening): --limit hard val rows
    (both types) and as many old val rows, COPIES relabelled split "test" (dropped from that scratch ship's train /
    val), the hard ones ranked through the same path (--family hard_test), the old ones' rankings a subset of the
    ship's, then build_test_snapshot. Checks: the new per-query arrays equal the ship's own (qvec / scores 1e-5)."""
    import shutil
    import numpy as np
    from graphrag.retrieval.rankings import load_rankings
    from grl.train.pilot import subset_rankings
    hv = [r for r in _jsonl(config.path(SHIP).parent / "hard_pairs.jsonl") if r["split"] == "val"]
    hard = [r for t in ("count", "version") for r in [x for x in hv if x["pattern"] == t][:max(1, a.limit // 2)]]
    old = [r for r in _jsonl(config.path(SHIP) / "pairs.jsonl")
           if r["split"] == "val" and r.get("origin") != "m7_hard"][:a.limit]
    relab = [{**r, "split": "test"} for r in hard + old]
    root, snap = config.path("runs/f9-local/open-dry"), config.path(DRY_SNAP)
    for x in (root, snap):
        if x.exists():
            shutil.rmtree(x)
    root.mkdir(parents=True)
    old_latest = root / "old_rankings"
    subset_rankings(config.path(SHIP) / "rankings" / "latest.json", [f"{r['id']}|q" for r in old], old_latest)
    rep = build_test_snapshot(snap, relab[:len(hard)], relab[len(hard):], old_latest / "latest.json",
                              "hard_test", drop_ids=frozenset(r["id"] for r in relab))
    new = load_rankings(snap / "ship" / "test_eval" / "rankings" / "latest.json")
    ship = load_rankings(config.path(SHIP) / "rankings" / "latest.json")
    per = {}
    for r in relab:
        k = f"{r['id']}|q"
        x, y = new.get(k), ship.get(k)
        per[k] = {"lex": bool(np.array_equal(x.lex, y.lex)), "vec": bool(np.array_equal(x.vec, y.vec)),
                  "title": bool(np.array_equal(x.title, y.title)),
                  "qvec_maxdiff": float(np.abs(np.asarray(x.qvec) - np.asarray(y.qvec)).max()),
                  "vec_score_maxdiff": float(np.abs(np.asarray(x.vec_score) - np.asarray(y.vec_score)).max())}
    ok = all(v["lex"] and v["vec"] and v["title"] and v["qvec_maxdiff"] < 1e-5 and v["vec_score_maxdiff"] < 1e-5
             for v in per.values())
    rep.update({"all_equal": ok, "per_query": per})
    _write(config.path("runs/f9-local/open_dry.json"), rep)
    print(json.dumps({k: rep[k] for k in ("n", "n_old", "n_hard", "by_group", "all_equal")}))
    return 0 if ok else 1


# --- spark (Spark) -------------------------------------------------------------------------------------------------

def spark_steps(split: str, out: str, frozen: dict, test_ship: str | None = None) -> list[tuple[str, list[str]]]:
    """(name, python argv) of the Spark stage; every path Spark-relative."""
    seeds = {s: v["spark"] for s, v in frozen["seeds"].items()}
    if split == "val":
        rows, rk, ce, ev_pairs = None, f"{SHIP}/rankings/latest.json", VAL_CE, LABELS
        ship, sel = SHIP, []
    else:
        te = f"{test_ship}/test_eval"
        rows, rk, ce, ev_pairs = f"{te}/test_eval.jsonl", f"{te}/rankings/latest.json", f"{out}/ce/ce_scores.jsonl", \
            f"{te}/test_eval.jsonl"
        ship, sel = test_ship, ["--split", "test", "--rows", rows, "--rankings", rk]
    st = []
    if split == "test":
        st.append(("ce", ["scripts/ce_scores.py", "--pairs", f"{ship}/pairs.jsonl", "--rankings",
                          f"{ship}/rankings/latest.json", "--test-eval", rows, "--test-rankings", rk,
                          "--out", ce, "--set", "rerank.ce_n=50"]))
    st.append(("baselines", ["scripts/m7_pilot8.py", "baselines", "--ship", ship, *sel, "--ce-cache", ce,
                             "--out", f"{out}/baselines.json"]))
    st.append(("ranker", ["scripts/m7_pilot8.py", "ranker", "--ship", ship, *sel, "--run", RANKER_RUN,
                          "--ce-cache", ce, "--k-max", "5", "--articles", "--top5", "--out", f"{out}/ranker.jsonl"]))
    for s in SEEDS:
        for K in K_EVAL:
            st.append((f"agent-{s}-k{K}", ["scripts/train.py", "eval", "--variant", "struct", "--pairs", ev_pairs,
                                           "--rankings", rk, "--ckpt", seeds[s], "--split", split, "--per-query",
                                           f"{out}/agent-{s}-k{K}.jsonl", "--set", f"env.K_max={K}", *FLAGS]))
    return st


def spark_preflight(a, frozen: dict, tsha: str) -> tuple[Path, str | None, dict]:
    """Checks of the Spark stage before anything is consumed / created: split / flags, the ship (computed from the
    token for the test; DRY_SNAP for --dry-testpath), its manifest, the seed and ranker checkpoints, a clean synced
    HEAD, the run dir. Returns (run dir, test ship, once record)."""
    ship = None
    if a.split == "val":
        if not a.dry or a.dry_testpath or a.test_ship:
            raise SystemExit("--split val runs only as --dry (no --test-ship)")
    else:
        if a.dry:
            raise SystemExit("--dry goes with --split val only")
        want = f"{DRY_SNAP}/ship" if a.dry_testpath else test_ship_path(tsha)
        ship = a.test_ship or want
        if ship != want:
            raise SystemExit(f"refused: --test-ship {ship} is not {want}")
        if not a.dry_testpath and a.name != TEST_RUN:
            raise SystemExit(f"refused: the test run is named {TEST_RUN}")
        # R65 crash fix: spark.py sync ships no manifest.json (pairs / rankings / test_eval only), so the Spark side
        # records the sha256 of the shipped test-eval files; `report` checks them against the Mac ship manifest
        if not (config.path(ship) / "test_eval" / "test_eval.jsonl").is_file():
            raise SystemExit(f"refused: {ship} has no test_eval/ (spark.py sync --extra {ship} --test-eval)")
    bad = [s for s, v in frozen["seeds"].items() if not config.path(v["spark"]).is_file()
           or _sha(config.path(v["spark"])) != v["sha256"]]
    if bad:
        raise SystemExit(f"seed checkpoints differ from the frozen sha256: {bad}")
    check_ranker_spark(frozen)                                                      # M3
    head = require_clean_head() if a.split == "test" else head_any()
    out = config.path(f"runs/{a.name}")
    if a.split == "test" and out.exists():
        raise SystemExit(f"refused: {out} exists")
    once = {"token_sha256": tsha, "split": a.split, "dry": a.dry, "dry_testpath": bool(a.dry_testpath),
            "git": head, "test_ship": ship, "name": a.name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "test_eval_sha256": {f: _sha(config.path(ship) / f) for f in TEST_EVAL_FILES} if ship else None}
    return out, ship, once


def cmd_spark(a) -> int:
    frozen = load_frozen()
    tsha = check_token(a.once_token, frozen)
    out, ship, once = spark_preflight(a, frozen, tsha)
    if a.split == "test" and not a.dry_testpath:          # I1: one test run per token, under ANY name
        once["marker"] = consume(tsha, "spark", config.path(SPARK_USED))
    out.mkdir(parents=True, exist_ok=a.split == "val")
    _write(out / "once.json", once)
    log = []
    for name, argv in spark_steps(a.split, f"runs/{a.name}", frozen, ship):
        if a.only and name not in a.only:
            continue
        print("STEP", name, json.dumps(argv), flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, *argv], cwd=ROOT).returncode
        log.append({"step": name, "rc": rc, "seconds": round(time.time() - t0, 1)})
        print(f"STEP {name} RC {rc}", flush=True)
        _write(out / "steps.json", log)
        if rc:
            return rc
    return 0


# --- report (Mac) --------------------------------------------------------------------------------------------------

def _rule_contexts(rows, rankings, fallbacks: dict, k: int = 5) -> dict:
    """seed -> id -> (rule context rows, R): grl.hard.amend_lookup on the struct env (articles + amend_op, K_max 5);
    no qualifying root -> that seed's agent context."""
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    from grl.hard.amend_lookup import amend_lookup, build_amend_index, load_amends_pairs
    from grl.model.variants import StructEnv
    from grl.train.fbeta import grl_reward, grl_weights
    cfg = config.load("train")
    cfg["env"]["K_max"] = k
    sdata = load_struct_env_data(articles=True)
    ranks = load_rankings(rankings, graph_version=sdata.version)
    senv = StructEnv(sdata, cfg["env"], None, articles=True, amend_op=True)
    index = build_amend_index(sdata, senv._root_row, load_amends_pairs(sdata))
    w, L = grl_weights(cfg), int(cfg["env"]["L_max"])
    out = {s: {} for s in fallbacks}
    for r in rows:
        q = senv.query(ranks.get(f"{r['id']}|q"))
        gold = frozenset(int(sdata.chunk_row[g]) for g in r["C_ref"])
        for s, fb in fallbacks.items():
            ctx = amend_lookup(index, q, sdata.chunk_tokens, k=k, l_max=L, fallback=fb[r["id"]])
            R = grl_reward(sdata, q.depth, ctx, gold, w, senv.limits.T_max).total if ctx else 0.0
            out[s][r["id"]] = (ctx, R)
    return out


def criterion(rows: dict, M: dict, system: str, agent24: str, keys: list, metric: str | None = None) -> dict:
    """Spec §1 + R37 on the method `system` (all@5 on the hard rows vs the best baseline @5 there; R at K_max 24 on
    the old rows via `agent24`, a method of the same shape whose agent branch ran at K_max 24)."""
    from scripts.m7_pilot8 import mean_se
    from scripts.m7_pilot9 import BASELINES, fine_group
    hard = [k for k in keys if fine_group(rows[k]) in CRITERION["hard_cells"]]
    old = [k for k in keys if fine_group(rows[k]) == "old"]
    m = metric or CRITERION["metric"]               # "anyof5": the R59 any-of variant (supplementary, not §1)
    cand = {b: mean_se([M[f"bl:{b}@5"][k][m] for k in hard])["mean"] for b in BASELINES}
    best = max(cand, key=lambda b: (cand[b], b == "ce+fused:0.6"))
    d = mean_se([M[system][k][m] - M[f"bl:{best}@5"][k][m] for k in hard])
    a_ok = bool(d["n"]) and d["mean"] > CRITERION["se_mult"] * d["se"]
    r_old = mean_se([M[agent24][k]["R"] for k in old])
    thr = CRITERION["r37_ref"] - CRITERION["r37_tol"]
    b_ok = bool(r_old["n"]) and r_old["mean"] >= thr
    return {"metric": m, "n_hard": len(hard), "n_old": len(old), "baselines_hard_all5": cand, "best_baseline": best,
            "system_hard_all5": mean_se([M[system][k][m] for k in hard]), "diff": d,
            "a_met": a_ok, "a_margin_se": (d["mean"] / d["se"]) if d["se"] else None,
            "old_R_K24": r_old, "r37_threshold": thr, "b_met": b_ok, "met": a_ok and b_ok}


def _q2_only(rows, M, keys) -> dict:
    """Supplementary: the criterion's (a) restricted to Q2 (the only hard type the baselines can solve at all)."""
    from scripts.m7_pilot8 import mean_se
    from scripts.m7_pilot9 import BASELINES, fine_group
    q2 = [k for k in keys if fine_group(rows[k]) == "Q2"]
    cand = {b: mean_se([M[f"bl:{b}@5"][k]["all5"] for k in q2])["mean"] for b in BASELINES}
    best = max(cand, key=lambda b: (cand[b], b == "ce+fused:0.6"))
    d = mean_se([M[f"sys:{FROZEN_SEED}"][k]["all5"] - M[f"bl:{best}@5"][k]["all5"] for k in q2])
    return {"n": len(q2), "best_baseline": best, "baselines": cand, "diff": d,
            "above_2se": bool(d["n"]) and d["mean"] > CRITERION["se_mult"] * d["se"]}


def build_methods(rows: dict, keys: list, run: Path, routes: dict, rankings) -> dict:
    """method -> id -> metrics (scripts.m7_pilot9._metrics): baselines @1 / 2 / 5, ranker top-1 / top-5, the agents
    per seed and K_max, the rule per seed, the systems per seed (sys:<s>, agent branch at K_max 5; sys24:<s> at 24)."""
    from grl.env.struct_data import load_struct_env_data
    from scripts.m7_pilot9 import BASELINES, _metrics
    ids = load_struct_env_data(articles=True).chunk_ids
    bl = json.loads((run / "baselines.json").read_text(encoding="utf-8"))
    if sorted(bl["top"]) != sorted(keys):
        raise SystemExit("baselines do not cover exactly the evaluated rows")
    fused = {k: bl["top"][k]["fused"] for k in keys}
    M = {}

    def per(path):
        p = {x["qkey"][:-2]: x for x in _jsonl(path)}
        if set(p) != set(keys):
            raise SystemExit(f"{path}: does not cover exactly the evaluated rows")
        return p
    rk = per(run / "ranker.jsonl")
    M["ranker_top1"] = {k: _metrics([ids[int(c)] for c in rk[k]["context"]], rows[k], fused[k], rk[k]["R"],
                                    rk[k]["reason"]) for k in keys}
    M["ranker_top5"] = {k: {**_metrics([ids[int(c)] for c in rk[k]["context_top5"]], rows[k], fused[k], rk[k]["R"]),
                            "R": rk[k]["R"], "k": 1.0, "limit": 0.0} for k in keys}
    for b in BASELINES:
        for kk in (1, 2, 5):
            M[f"bl:{b}@{kk}"] = {}
            for k in keys:
                top = bl["top"][k][b]
                m, m10 = _metrics(top[:kk], rows[k], [], None), _metrics(top[:10], rows[k], [], None)
                M[f"bl:{b}@{kk}"][k] = {**m, "R": 0.0, "recall10": m10["recall10"], "all10": m10["all10"]}
    ag = {}
    for s in SEEDS:
        for K in K_EVAL:
            p = ag[(s, K)] = per(run / f"agent-{s}-k{K}.jsonl")
            M[f"agent{'' if K == 5 else K}:{s}"] = {
                k: _metrics([ids[int(c)] for c in p[k]["context"]], rows[k], fused[k], p[k]["reward"], p[k]["reason"])
                for k in keys}
    for fam in ("agent", "agent24"):                   # R64 (A): the per-question mean over the 3 seeds
        M[f"{fam}:mean"] = {k: {m: sum(M[f"{fam}:{s}"][k][m] for s in SEEDS) / len(SEEDS)
                                for m in M[f"{fam}:{SEEDS[0]}"][k]} for k in keys}
    rule = _rule_contexts([rows[k] for k in keys], rankings,
                          {s: {k: ag[(s, 5)][k]["context"] for k in keys} for s in SEEDS})
    for s in SEEDS:
        M[f"rule:{s}"] = {k: {**_metrics([ids[int(c)] for c in rule[s][k][0]], rows[k], fused[k], rule[s][k][1]),
                              "limit": 0.0} for k in keys}
        for tag, agent in (("sys", f"agent:{s}"), ("sys24", f"agent24:{s}")):
            src = {"old": "ranker_top5", "count": agent, "version": f"rule:{s}"}
            M[f"{tag}:{s}"] = {k: M[src[routes[k]]][k] for k in keys}
        src1 = {"old": "ranker_top1", "count": f"agent:{s}", "version": f"rule:{s}"}
        M[f"sys_top1:{s}"] = {k: M[src1[routes[k]]][k] for k in keys}
    return M


def summarise(rows, keys, M) -> dict:
    """Tables per method / cell, per-seed summaries (mean ± sd over seeds), paired SE by type and overall."""
    import statistics
    from scripts.m7_pilot9 import BASELINES, CELL_ORDER, METRICS, _cells, _paired, subgroups
    sub = {k: rows[k] for k in keys}
    table = {n: _cells(sub, keys, lambda k, m=m: m[k], METRICS) for n, m in M.items()}
    groups = ("all", "old", "hard", "Q1", "Q1x", "Q2")
    seeds = {}
    for fam in ("sys", "agent", "agent24", "rule"):
        seeds[fam] = {}
        for g in groups:
            if g not in table[f"{fam}:{SEEDS[0]}"]:
                continue
            seeds[fam][g] = {}
            for met in ("R", "all5", "recall5", "all10", "anyof5"):
                v = [table[f"{fam}:{s}"][g][met]["mean"] for s in SEEDS]
                seeds[fam][g][met] = {"per_seed": dict(zip(SEEDS, v)), "mean": statistics.mean(v),
                                      "sd": statistics.stdev(v) if len(v) > 1 else 0.0}
    fz = f"sys:{FROZEN_SEED}"
    best = {}
    for cell in CELL_ORDER:
        if cell not in table[fz]:
            continue
        cands = {b: table[f"bl:{b}@5"][cell]["all5"]["mean"] for b in BASELINES}
        bb = max(cands, key=lambda b: (cands[b], b == "ce+fused:0.6"))
        best[cell] = {"baseline": bb, "baseline_all5": cands[bb], "all_baselines_all5": cands,
                      "anyof_baselines": {b: table[f"bl:{b}@5"][cell]["anyof5"]["mean"] for b in BASELINES}}
    paired = {"frozen_minus_ranker_top5": _paired(sub, keys, lambda k: M[fz][k], lambda k: M["ranker_top5"][k]),
              "frozen_minus_agent": _paired(sub, keys, lambda k: M[fz][k], lambda k: M[f"agent:{FROZEN_SEED}"][k])}
    for b in BASELINES:
        paired[f"frozen_minus_bl:{b}@5"] = _paired(sub, keys, lambda k, b=b: M[fz][k], lambda k, b=b: M[f"bl:{b}@5"][k])
        paired[f"agent_minus_bl:{b}@5"] = _paired(sub, keys, lambda k, b=b: M[f"agent:{FROZEN_SEED}"][k],
                                                  lambda k, b=b: M[f"bl:{b}@5"][k])
    for s in SEEDS:
        paired[f"agent:{s}_minus_rule:{s}"] = _paired(sub, keys, lambda k, s=s: M[f"agent:{s}"][k],
                                                      lambda k, s=s: M[f"rule:{s}"][k])
    return {"table": table, "seeds": seeds, "best_baseline": best, "paired": paired,
            "n": {c: v["n"] for c, v in table[fz].items()}}


def _reproduce_p9(rows, keys, M, run: Path, a) -> dict:
    """Dry run: the val numbers of pilot 9 -- per-query equality of baselines, ranker, theta (s01) and the router,
    and the cell means of the frozen system against the pilot-9 recomputation (sys (c) with Q2 -> the rule)."""
    from scripts.m7_pilot8 import mean_se
    from scripts.m7_pilot9 import METRICS, _cells, fine_group
    p9 = json.loads(config.path(a.p9_tables).read_text(encoding="utf-8"))
    out = {}
    for mine, theirs in (("ranker_top1", "ranker_top1"), ("ranker_top5", "ranker_top5"), ("agent:s01", "theta"),
                         ("bl:ce+fused:0.6@5", "bl:ce+fused:0.6@5"), ("bl:dense@5", "bl:dense@5"),
                         ("bl:bm25@5", "bl:bm25@5"), ("bl:fused@5", "bl:fused@5")):
        t = p9["table"][theirs]
        mt = _cells({k: rows[k] for k in keys}, keys, lambda k: M[mine][k], METRICS)
        out[mine] = {c: {m: round(mt[c][m]["mean"] - t[c][m]["mean"], 6) for m in ("R", "all5", "recall5", "all10")
                         if m in t.get(c, {})} for c in ("all", "old", "Q1", "Q1x", "Q2") if c in t}
    out["max_abs_diff"] = max(abs(v) for d in out.values() for c in d.values() for v in c.values())
    # per-query: the agent s01 contexts vs pilot 9's theta file
    th = {x["qkey"][:-2]: x for x in _jsonl(config.path(a.p9_theta))}
    ag = {x["qkey"][:-2]: x for x in _jsonl(run / "agent-s01-k5.jsonl")}
    out["theta_contexts_equal"] = all(ag[k]["context"] == th[k]["context"] for k in keys)
    rk9 = {x["qkey"][:-2]: x for x in _jsonl(config.path(a.p9_ranker))}
    rk = {x["qkey"][:-2]: x for x in _jsonl(run / "ranker.jsonl")}
    out["ranker_equal"] = all(rk[k]["context"] == rk9[k]["context"] and rk[k]["context_top5"] == rk9[k]["context_top5"]
                              for k in keys)
    # the frozen system rebuilt from pilot 9's own files (routes, ranker top-5, theta, rule with fallback theta)
    routes = json.loads(config.path(a.p9_router).read_text(encoding="utf-8"))["routes"]
    q2 = json.loads(config.path(a.p9_q2rule).read_text(encoding="utf-8"))["per_item"]
    def p9sys(k, m):
        if routes[k] == "old":
            return {"all5": rk9[k]["all5_top5"], "R": rk9[k]["R"]}[m]
        if routes[k] == "count":
            return {"all5": float(th[k]["recall"] >= 1 - 1e-12), "R": th[k]["reward"]}[m]
        return q2[k]["rule"][m]
    out["frozen_vs_p9_files"] = {}
    for g in ("all", "old", "hard", "Q1", "Q1x", "Q2"):
        ks = [k for k in keys if g == "all" or fine_group(rows[k]) == g or (g == "hard" and fine_group(rows[k]) != "old")]
        out["frozen_vs_p9_files"][g] = {m: {"p9": mean_se([p9sys(k, m) for k in ks])["mean"],
                                            "mine": mean_se([M[f"sys:{FROZEN_SEED}"][k][m] for k in ks])["mean"],
                                            "per_query_mismatch": sum(abs(p9sys(k, m) - M[f"sys:{FROZEN_SEED}"][k][m])
                                                                      > 1e-6 for k in ks)} for m in ("all5", "R")}
    return out


def report_preflight(a, frozen: dict, tsha: str) -> tuple[Path, dict, dict]:
    """Checks of `report` before anything is consumed (I1 / I2): the run's provenance (once.json: this token, the
    split, not a dry-testpath run, the opened ship and its manifest sha, the one test run name), every step RC 0 and
    every per-method file present, the out file new, the state's open record and marker, a clean HEAD."""
    verify_frozen(frozen)
    run = config.path(a.run)
    if not (run / "once.json").is_file():
        raise SystemExit(f"refused: {run}/once.json missing (pull the Spark run)")
    once = json.loads((run / "once.json").read_text(encoding="utf-8"))
    if once.get("token_sha256") != tsha or once.get("split") != a.split:
        raise SystemExit(f"refused: {run} is not a `spark --split {a.split}` run of this token")
    steps = json.loads((run / "steps.json").read_text(encoding="utf-8")) if (run / "steps.json").is_file() else []
    want = ["baselines.json", "ranker.jsonl"] + [f"agent-{s}-k{K}.jsonl" for s in SEEDS for K in K_EVAL]
    miss = [f for f in want if not (run / f).is_file()]
    if miss or not steps or any(x["rc"] for x in steps):
        raise SystemExit(f"refused: {run} is incomplete (missing {miss}, steps {steps})")
    st = {}
    if a.split == "test":
        require_clean_head()
        st = load_state(tsha)
        ship = test_ship_path(tsha)
        if once.get("dry_testpath") or str(once.get("test_ship") or "").startswith(DRY_SNAP):
            raise SystemExit("refused: a dry-testpath run (relabelled val rows) is no test run")
        if not st.get("open") or not marker_path(tsha, "open").is_file():
            raise SystemExit("refused: the test was not opened with this token (no open marker / state)")
        if st.get("report") or marker_path(tsha, "report").exists():
            raise SystemExit("refused: the test report of this token exists")
        if (a.test_ship or ship) != ship or st["ship"] != ship or once.get("test_ship") != ship:
            raise SystemExit(f"refused: the run / --test-ship is not the opened ship {ship}")
        man = config.path(ship) / "manifest.json"
        if _sha(man) != st["ship_manifest_sha256"] or once.get("test_eval_sha256") != {
                f: json.loads(man.read_text(encoding="utf-8"))["sha256"].get(f) for f in TEST_EVAL_FILES}:
            raise SystemExit("refused: the ship manifest / the Spark run's test-eval files differ from the opened ship")
        if once.get("name") != TEST_RUN or not once.get("marker"):
            raise SystemExit(f"refused: not the one Spark test run ({TEST_RUN}, with its Spark marker)")
        if config.path(a.out).exists():
            raise SystemExit(f"refused: {a.out} exists")
    return run, once, st


def cmd_report(a) -> int:
    frozen, tsha = gate(a, "report")
    run, once, st = report_preflight(a, frozen, tsha)
    rec = consume(tsha, "report") if a.split == "test" else None    # <- the one use; test_eval is read next
    import numpy as np
    from graphrag.retrieval.rankings import load_rankings
    if a.split == "val":
        all_rows = {r["id"]: r for r in _jsonl(config.path(SHIP) / "pairs.jsonl")}
        keys = sorted(k for k, r in all_rows.items() if r["split"] == "val")
        rankings = config.path(SHIP) / "rankings" / "latest.json"
    else:
        te = config.path(test_ship_path(tsha)) / "test_eval"
        all_rows = {r["id"]: r for r in _jsonl(te / "test_eval.jsonl")}
        keys = sorted(all_rows)
        rankings = te / "rankings" / "latest.json"
    rows = {k: all_rows[k] for k in keys}
    ranks = load_rankings(rankings)
    routes = dict(zip(keys, route(load_router(), np.stack([np.asarray(ranks.get(f"{k}|q").qvec, np.float32)
                                                           for k in keys]))))
    from scripts.m7_pilot8 import _group
    M = build_methods(rows, keys, run, routes, rankings)
    out = summarise(rows, keys, M)
    out["criterion"] = criteria_r64(rows, M, keys)
    from scripts.m7_pilot8 import mean_se
    from scripts.m7_pilot9 import fine_group
    old = [k for k in keys if fine_group(rows[k]) == "old"]
    out["old_strict"] = {"ranker_R": mean_se([M["ranker_top1"][k]["R"] for k in old]),
                         "r37_ref": CRITERION["r37_ref"],
                         **{f"{fam}{K}:{s}": mean_se([M[f"{fam}{'' if K == 5 else K}:{s}"][k]["R"] for k in old])
                            for fam in ("agent", "sys") for K in K_EVAL for s in SEEDS},
                         "system_minus_ranker_R": mean_se([M[f"sys24:{FROZEN_SEED}"][k]["R"] - M["ranker_top1"][k]["R"]
                                                           for k in old])}
    labels = ("old", "count", "version")
    out["router"] = {"confusion": [[sum(1 for k in keys if _group(rows[k]["pattern"]) == t and routes[k] == p)
                                    for p in labels] for t in labels], "labels": list(labels),
                     "acc": sum(routes[k] == _group(rows[k]["pattern"]) for k in keys) / len(keys)}
    if a.split == "val":
        out["reproduction_p9"] = _reproduce_p9(rows, keys, M, run, a)
        p9r = json.loads(config.path(a.p9_router).read_text(encoding="utf-8"))["routes"]
        out["reproduction_p9"]["routes_equal"] = all(routes[k] == p9r[k] for k in keys)
    out.update({"split": a.split, "dry": a.dry, "run": a.run, "ledger": rec, "once": once, "frozen_git": frozen["git"],
                "git": _git_head(), "frozen_system": frozen["system"]})
    dst = config.path(a.out)
    _write(dst, out)
    if a.split == "test":
        save_state(tsha, {**st, "report": {"out": a.out, "sha256": _sha(dst), "ledger": rec, "run": a.run}})
    print(criteria_text(out["criterion"]))
    print(json.dumps({"split": a.split, "n": out["n"], "seeds_sys_all5": out["seeds"]["sys"]["all"]["all5"],
                      "seeds_sys_R": out["seeds"]["sys"]["all"]["R"],
                      **({"reproduction": {k: v for k, v in out["reproduction_p9"].items() if not isinstance(v, dict)}}
                         if a.split == "val" else {})}, ensure_ascii=False, indent=1))
    return 0


def criteria_r64(rows, M, keys) -> dict:
    """R64, pre-registered: (B) the deployed system = HEADLINE; (A) spec §1 / R37 verbatim on the struct agent (seed
    mean + per seed); supplementary: Q2-only, R59 any-of, per-seed systems. Every block carries its own pass / fail."""
    A = {"seed_mean": criterion(rows, M, "agent:mean", "agent24:mean", keys),
         "per_seed": {s: criterion(rows, M, f"agent:{s}", f"agent24:{s}", keys) for s in SEEDS}}
    A["pass"] = A["seed_mean"]["met"]
    B = criterion(rows, M, f"sys:{FROZEN_SEED}", f"sys24:{FROZEN_SEED}", keys)
    B["pass"] = B["met"]
    sup = {"system_q2_only": _q2_only(rows, M, keys),
           "system_anyof": criterion(rows, M, f"sys:{FROZEN_SEED}", f"sys24:{FROZEN_SEED}", keys, "anyof5"),
           "agent_seed_mean_anyof": criterion(rows, M, "agent:mean", "agent24:mean", keys, "anyof5"),
           "per_seed_system": {s: criterion(rows, M, f"sys:{s}", f"sys24:{s}", keys) for s in SEEDS}}
    return {"headline": "B_system", "B_system": B, "A_agent_spec": A, "supplementary": sup, "spec": CRITERION}


def criteria_text(c: dict) -> str:
    """The pre-registered report block (R64): pass / fail per criterion, no choice after the fact."""
    def ab(x):
        return (f"(a) Δall@5 {x['diff']['mean']:+.3f} ± {x['diff']['se']:.3f} vs {x['best_baseline']} -> "
                f"{'PASS' if x['a_met'] else 'FAIL'}; (b) old R@K24 {x['old_R_K24']['mean']:.4f} >= "
                f"{x['r37_threshold']:.4f} -> {'PASS' if x['b_met'] else 'FAIL'}")
    A, B, S = c["A_agent_spec"], c["B_system"], c["supplementary"]
    lines = [f"(B) HEADLINE deployed system: {'PASS' if B['pass'] else 'FAIL'} -- {ab(B)}",
             f"(A) spec §1 / R37 verbatim, struct agent, seed mean: {'PASS' if A['pass'] else 'FAIL'} -- "
             f"{ab(A['seed_mean'])}"]
    lines += [f"    agent {s}: {'PASS' if v['met'] else 'FAIL'} -- {ab(v)}" for s, v in A["per_seed"].items()]
    q = S["system_q2_only"]
    lines += [f"supp. Q2-only system: Δ {q['diff']['mean']:+.3f} ± {q['diff']['se']:.3f} vs {q['best_baseline']} -> "
              f"{'> 2 SE' if q['above_2se'] else '<= 2 SE'}",
              f"supp. any-of system: {ab(S['system_anyof'])}",
              f"supp. any-of agent seed mean: {ab(S['agent_seed_mean_anyof'])}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--seed-ckpt", action="append", required=True, help="sNN=runs/spark/<run>/checkpoints/<theta*>.pt")
    f.add_argument("--seed-note", action="append", default=[], help="sNN=why this checkpoint (theta* / SFT)")
    f.add_argument("--p9-router", default="runs/p9-local/router.json")
    am = sub.add_parser("amend-r64", help="pre-register the R64 criteria in frozen.json (before any use)")
    am.add_argument("--reason", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--once-token")
        p.add_argument("--split", required=True, choices=["val", "test"])
        p.add_argument("--dry", action="store_true")
        if name == "open":
            p.add_argument("--limit", type=int, default=8, help="--dry: val hard rows ranked again")
        if name in ("spark", "report"):
            p.add_argument("--test-ship", help="--split test: optional; must equal the ship computed from the token")
        if name == "spark":
            p.add_argument("--name", required=True)
            p.add_argument("--only", nargs="*", help="--dry only: a subset of the steps")
            p.add_argument("--dry-testpath", action="store_true", help="--split test on DRY_SNAP (relabelled val "
                           "copies): proves the test path; consumes nothing, its report is always refused")
        if name == "report":
            p.add_argument("--run", required=True, help="runs/spark/<name> (pulled)")
            p.add_argument("--out", required=True)
            p.add_argument("--p9-tables", default="runs/p9-local/tables.json")
            p.add_argument("--p9-theta", default="runs/spark/p9-ev-g-k5/pq.jsonl")
            p.add_argument("--p9-ranker", default="runs/spark/p9-ranker-k5/pq.jsonl")
            p.add_argument("--p9-router", default="runs/p9-local/router.json")
            p.add_argument("--p9-q2rule", default="runs/p9-local/q2rule_new.json")
    a = ap.parse_args(argv)
    if a.cmd not in ("freeze", "amend-r64") and not a.once_token:
        ap.error("refused: --once-token is required")
    if getattr(a, "only", None) and getattr(a, "split", None) == "test":
        ap.error("--only is a dry-run aid; the test stage runs every step")
    return {"freeze": cmd_freeze, "amend-r64": cmd_amend, "open": cmd_open, "spark": cmd_spark,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
