"""Mac <-> Spark parity (plan Task 12; §9.6; Review Focus 4).
    PYTHONPATH=. python scripts/check_spark.py fingerprint --pairs P --rankings R --out F
    PYTHONPATH=. python scripts/check_spark.py compare A B
    PYTHONPATH=. python scripts/check_spark.py determinism --pairs P --rankings R [--out runs/det] [--epochs 2] [--mode strict|warn] [--tol 1e-6]"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402  -- stdlib only, before numpy

pin_blas()

import numpy as np  # noqa: E402

from graphrag.env.reward import reward  # noqa: E402


def arrays_digest(arrs) -> str:
    """sha256 over dtype, shape and bytes of each array (equal bytes under another dtype / shape differ)."""
    h = hashlib.sha256()
    for arr in arrs:
        a = np.ascontiguousarray(arr)
        h.update(f"{a.dtype.str}{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def check_out(out: str) -> Path:
    """determinism --out: a relative path strictly inside runs/ (it is rmtree'd)."""
    runs = (ROOT / "runs").resolve()
    p = Path(out)
    if p.is_absolute() or ".." in p.parts:
        raise SystemExit(f"--out {out}: must be a relative path under runs/")
    d = (ROOT / p).resolve()
    if not d.is_relative_to(runs) or d == runs:
        raise SystemExit(f"--out {out}: must be under runs/")
    return d


def fingerprint(env, ranks, items, w) -> dict:
    """Per query: sha of the QueryCtx arrays + context / reason / R of the expert replay. The replay is Task 8's
    replay_item, so a mask drift on this host raises DataMismatch (pair id, step) instead of a silent difference."""
    from grl.train.data import replay_item
    out = {}
    for it in items:
        q, s = replay_item(env, ranks, it)
        out[it.qkey] = {"q": arrays_digest((q.chunk_qsim, q.node_sem, q.node_lex, q.v0, q.depth))[:16], "context": [int(c) for c in s.context], "reason": s.reason,
                        "reward": reward(env.data, q.depth, s.context, it.gold, w, env.limits.T_max).total}
    return out


def compare(a: dict, b: dict) -> list[str]:
    probs = [f"{k}: missing" for k in sorted(set(a) ^ set(b))]
    for k in sorted(set(a) & set(b)):
        probs += [f"{k}: {f}" for f in ("q", "context", "reason", "reward") if a[k][f] != b[k][f]]
    return probs


def max_abs_diff(ra: list, rb: list) -> float:
    if len(ra) != len(rb):
        return math.inf
    worst = 0.0
    for x, y in zip(ra, rb):
        if set(x) - {"time"} != set(y) - {"time"}:
            return math.inf
        for k in set(x) - {"time"}:
            u, v = x[k], y[k]
            if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
                worst = max(worst, abs(float(u) - float(v)))
            elif u != v:
                return math.inf
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fingerprint", "compare", "determinism"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pairs")
    ap.add_argument("--rankings")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mode", default="strict", choices=["strict", "warn"])
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        probs = compare(*(json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files))
        print(json.dumps({"mismatches": len(probs), "examples": probs[:20]}, ensure_ascii=False))
        return 1 if probs else 0
    if a.cmd == "fingerprint":
        from graphrag.env.data import load_env_data
        from graphrag.env.mdp import Env
        from graphrag.env.reward import weights_from_config
        from graphrag.retrieval.rankings import load_rankings
        from grl.train.data import load_items
        data = load_env_data()
        fp = fingerprint(Env(data), load_rankings(a.rankings, graph_version=data.version),
                         load_items(a.pairs, data, ("train", "val")), weights_from_config())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(fp, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{len(fp)} fingerprints -> {a.out}")
        return 0
    from grl.train.log import comparable, read_rows
    rows = []
    base = check_out(a.out or "runs/det")
    for tag in ("a", "b"):
        d = base / tag
        if d.exists():
            shutil.rmtree(d)
        cmd = [sys.executable, "scripts/train.py", "sft", "--pairs", a.pairs, "--rankings", a.rankings, "--run", str(d),
               "--set", f"sft.max_epochs={a.epochs}", "--set", f"run.deterministic={a.mode}"]
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if r.returncode:
            return r.returncode
        rows.append(read_rows(d))
    diff = max_abs_diff(*rows)
    same = [json.dumps(r, sort_keys=True) for r in comparable(rows[0])] == \
        [json.dumps(r, sort_keys=True) for r in comparable(rows[1])]    # byte-equal rows without `time`
    ok = same if a.mode == "strict" else diff <= a.tol
    print(json.dumps({"mode": a.mode, "rows": len(rows[0]), "identical": same, "max_abs_diff": diff, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
