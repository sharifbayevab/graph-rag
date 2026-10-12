"""M5 pilot data (plan Task 8 / 13; Mac only: Postgres :5437 BM25 + bge-m3 on CPU for the query rankings).
    PYTHONPATH=. python scripts/m5_pilot_data.py [--source data/dataset/fable]
snapshot of the consensus-GOOD pairs -> finalize --no-manual-gate (the M3-R31 dual-reader consensus replaces the
manual gate for a PILOT only) -> keep train + val -> U9 rankings of those queries -> oracle trajectories (M3 Task 20
machinery, --no-select) -> ship/ (the only part Task 12 sends to Spark: trajectories, pairs if any, rankings/**,
splits.json + manifest). Writes data/m5/pilot/<sha12>/ and prints a summary. The M3 dataset directory is read only.
Refuses any Postgres but dev :5437. <sha12>/splits.json is written last: it is the restart gate (a crashed run
reruns every step on its next start).
Pilot 5 (controller ruling): --from-verified snapshots grl.train.pilot.verified_source instead of the consensus file
-- verified.jsonl minus reader-A / reader-B BAD, minus reader-C vetoes, minus originals superseded by an -r1 repair
(counts in <sha12>/source.json) -- plus the strict consensus test of reviewed.jsonl; --test-eval sets the finalized
test rows aside (test_candidates.jsonl, Mac only), builds their own rankings (test_rankings/) and ships them as
ship/test_eval/ (train.py eval / baselines --split test only; spark.py sync --test-eval)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402  -- stdlib only, before numpy

pin_blas()

from graphrag import config  # noqa: E402
from grl.train.pilot import (TEST_CANDIDATES, TEST_RANKINGS, build_ship, keep_splits, require_dev_pg,  # noqa: E402
                             set_aside, snapshot, verified_snapshot, verify_ship)

STRAY_REPORT = "data/m3/rankings_report_dataset.json"   # build_rankings side output, removed after use


def _run(*args) -> None:
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if r.returncode:
        raise SystemExit(f"{' '.join(args[:2])} exited {r.returncode}")


def _rankings(*args) -> None:
    """build_rankings, then its side report goes (a report that existed before is restored byte for byte)."""
    rep = config.path(STRAY_REPORT)
    before, had_dir = (rep.read_bytes() if rep.exists() else None), rep.parent.exists()
    try:
        _run("scripts/build_rankings.py", *args)
    finally:
        if before is not None:
            rep.write_bytes(before)
        else:
            rep.unlink(missing_ok=True)
            if not had_dir and rep.parent.exists() and not any(rep.parent.iterdir()):
                rep.parent.rmdir()


def main(argv=None) -> int:
    t = config.load("train")["m5_pilot"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=t["source_dir"])
    ap.add_argument("--from-verified", action="store_true", help="pilot 5 source (verified minus BAD / veto / "
                    "superseded) + the strict consensus test")
    ap.add_argument("--test-eval", action="store_true", help="pilot 5: ship/test_eval/ with the consensus test")
    a = ap.parse_args(argv)
    if a.test_eval and not a.from_verified:
        ap.error("--test-eval goes with --from-verified")
    require_dev_pg(config.PG_DSN)                     # before any DB work; subprocesses inherit the same GRL_PG_DSN
    mk = verified_snapshot if a.from_verified else snapshot
    d = mk(config.path(a.source), config.path(t["out_root"]))
    rel = d.relative_to(config.ROOT)
    if not (d / "splits.json").exists():
        _run("scripts/build_dataset.py", "finalize", "--dir", str(rel), "--no-manual-gate")
        if a.test_eval:
            print(json.dumps({"test_set_aside": set_aside(d / "candidates.jsonl", "test", d / TEST_CANDIDATES)}))
        counts = keep_splits(d / "candidates.jsonl", tuple(t["splits"]))
        _rankings("--family", "dataset", "--candidates", str(rel / "candidates.jsonl"), "--out", str(rel / "rankings"))
        if a.test_eval:
            _rankings("--family", "dataset", "--candidates", str(rel / TEST_CANDIDATES), "--out",
                      str(rel / TEST_RANKINGS))
        _run("scripts/build_dataset.py", "trajectories", "--dir", str(rel), "--rankings",
             str(rel / "rankings" / "latest.json"), "--no-select")
        build_ship(d, counts, test_eval=a.test_eval)
        tmp = d / "splits.json.tmp"
        tmp.write_text(json.dumps(counts, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, d / "splits.json")
    verify_ship(d)
    st = json.loads((d / "stats.json").read_text(encoding="utf-8")).get("trajectories", {})
    print(json.dumps({"dir": str(rel), "splits": json.loads((d / "splits.json").read_text(encoding="utf-8")),
                      "trajectories": {k: st.get(k) for k in ("candidates", "with_trajectory", "reasons",
                                                              "threshold_loss", "first_op")}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
