"""M7 reward-weight sweep by the FINAL result (validation only; the test splits stay sealed): for every weight vector
w = (alpha F1, beta coverage, gamma diversity, delta coherence), all w_i >= 0.1, the full pilot-9 recipe is rerun under
--set reward.weights.*: expert relabel (struct, depth 5) -> per seed SFT from scratch -> GRPO -> greedy val eval of
the SFT (pi_ref.pt) and of GRPO's theta*. Runs are compared on reward-INDEPENDENT val metrics (all@5, recall@5 per
group), never on val R (each run has its own R). The current weights (0.5/0.2/0.2/0.1) reuse pilot 9 / task 9
(p9-expert, s01 = p9, s02 / s03 = f9) -- the same recipe and code path (scripts/m7_runs.py).

    PYTHONPATH=. python scripts/m7_weight_sweep.py jobs                  # the queue (never --split test)
    PYTHONPATH=. python scripts/m7_weight_sweep.py chain --point f1      # on Spark: expert -> seeds (sft, grpo, evals)
    PYTHONPATH=. python scripts/m7_weight_sweep.py pool --parallel 5     # on Spark: every point, N chains at a time
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.m7_runs import FLAGS, GRPO_SET, RANKINGS, SFT_SET, SHIP, _last_json, _sets  # noqa: E402

KEYS = ("alpha", "beta", "gamma", "delta")             # F1, coverage, diversity, coherence (graphrag.env.reward)
CURRENT = (0.5, 0.2, 0.2, 0.1)
POINTS = {                                              # stage 1: all w_i >= 0.1
    "eq": (0.25, 0.25, 0.25, 0.25),
    "f1": (0.7, 0.1, 0.1, 0.1),
    "cov": (0.1, 0.7, 0.1, 0.1),
    "div": (0.1, 0.1, 0.7, 0.1),
    "coh": (0.1, 0.1, 0.1, 0.7),
    "f6c2": (0.6, 0.2, 0.1, 0.1),
    "f4c4": (0.4, 0.4, 0.1, 0.1),
    "dc": (0.3, 0.2, 0.3, 0.2),
    "f4dc": (0.4, 0.2, 0.2, 0.2),
    # stage 2 (after stage 1: the edge diversity = coherence = 0.1 is best): the F1 / coverage split along that edge
    "f3c5": (0.3, 0.5, 0.1, 0.1),
    "f5c3": (0.5, 0.3, 0.1, 0.1),
    # check: the optimum of the Scheffé mixture model (scripts/plot_m7_ws_surface.py), not trained in stages 1-2
    "opt": (0.425, 0.35, 0.125, 0.1),
    # Scheffé {4,2} simplex-lattice completion (pseudo-component binary midpoints of the region w_i >= 0.1)
    "m_fd": (0.4, 0.1, 0.4, 0.1),
    "m_fh": (0.4, 0.1, 0.1, 0.4),
    "m_cd": (0.1, 0.4, 0.4, 0.1),
    "m_ch": (0.1, 0.4, 0.1, 0.4),
    "m_dh": (0.1, 0.1, 0.4, 0.4),
}
SEEDS = {"s01": 20261001, "s02": 20261002, "s03": 20261003}
STAGE1_SEEDS = ("s01", "s02")


def wset(w) -> list[str]:
    assert len(w) == 4 and abs(sum(w) - 1) < 1e-9 and min(w) >= 0.1 - 1e-9, w
    return _sets([f"reward.weights.{k}={v:g}" for k, v in zip(KEYS, w)])


def labels(point: str) -> str:
    return f"runs/ws-{point}-expert/labels.jsonl"


def steps(point: str, seeds=STAGE1_SEEDS, expert: bool = True) -> list[tuple[str, list[str]]]:
    w = wset(POINTS[point])
    base = ["--variant", "struct", "--pairs", labels(point), "--rankings", RANKINGS, *FLAGS, *w]
    out = []
    if expert:
        out.append(("expert", ["expert", "--variant", "struct", "--pairs", f"{SHIP}/pairs.jsonl", "--rankings", RANKINGS,
                               "--out", labels(point), "--depth", "5", *FLAGS, *w]))
    for sd in seeds:
        s = SEEDS[sd]
        sft, grpo = f"runs/ws-{point}-sft-{sd}", f"runs/ws-{point}-grpo-{sd}"
        out += [(f"sft:{sd}", ["sft", *base, "--run", sft, *_sets(SFT_SET + [f"sft.seed={s}"])]),
                (f"grpo:{sd}", ["grpo", *base, "--run", grpo, "--sft", sft, *_sets(GRPO_SET + [f"grpo.seed={s}"])]),
                (f"eval:{sd}", ["eval", *base, "--ckpt", "{best}", "--split", "val",
                                "--per-query", f"runs/ws-{point}-ev-{sd}/pq.jsonl"]),
                (f"evsft:{sd}", ["eval", *base, "--ckpt", f"{sft}/pi_ref.pt", "--split", "val",
                                 "--per-query", f"runs/ws-{point}-evsft-{sd}/pq.jsonl"])]
    for name, argv in out:
        if any(x == "--split" and y != "val" for x, y in zip(argv, argv[1:])):
            raise SystemExit(f"{point}/{name}: a step reads a split other than val")
    return out


def chain(point: str, seeds=STAGE1_SEEDS, expert: bool = True) -> int:
    best = None
    log = ROOT / f"runs/ws-{point}.chain.log"
    for name, argv in steps(point, seeds, expert):
        if name == "expert" and (ROOT / labels(point)).exists():
            print(f"[{point}] STEP expert SKIP (labels exist)", flush=True)
            continue
        argv = [best if x == "{best}" else x for x in argv]
        t0 = time.time()
        print(f"[{point}] STEP {name} {json.dumps(argv)}", flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"STEP {name} {json.dumps(argv)}\n")
            fh.flush()
            p = subprocess.Popen([sys.executable, "scripts/train.py", *argv], cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            lines = []
            for ln in p.stdout:
                fh.write(ln)
                fh.flush()
                lines.append(ln)
            rc = p.wait()
            fh.write(f"STEP {name} RC {rc} {time.time() - t0:.0f}s\n")
        print(f"[{point}] STEP {name} RC {rc} {time.time() - t0:.0f}s", flush=True)
        if rc:
            return rc
        if name.startswith("grpo"):
            best = _last_json(lines)["best"]
    return 0


MEM_GB = {"expert": 25, "gpu": 45}         # observed peaks: expert ~24 GB (4 workers), SFT / GRPO / eval up to ~45 GB
STATE = "runs/ws-sched-state.json"


def tasks(points, seeds) -> list[dict]:
    """The DAG: expert(point) -> per seed sft -> grpo -> (eval theta*, eval SFT). A point may carry its own seeds as
    "name:s01+s03" (else `seeds`)."""
    out = []
    for spec in points:
        pt, _, own = spec.partition(":")
        st = steps(pt, tuple(own.split("+")) if own else seeds)
        for name, argv in st:
            sd = name.split(":")[1] if ":" in name else None
            dep = {"expert": [], "sft": [f"{pt}/expert"], "grpo": [f"{pt}/sft:{sd}"], "eval": [f"{pt}/grpo:{sd}"],
                   "evsft": [f"{pt}/sft:{sd}"]}[name.split(":")[0]]
            run = next((y for x, y in zip(argv, argv[1:]) if x in ("--run", "--per-query", "--out")), None)
            out.append({"id": f"{pt}/{name}", "point": pt, "kind": name.split(":")[0], "argv": argv, "deps": dep,
                        "mem": MEM_GB["expert" if name == "expert" else "gpu"], "out": run})
    return out


def _external_expert(pt: str) -> bool:
    r = subprocess.run(["pgrep", "-f", f"train.py expert.*{labels(pt)}"], capture_output=True)
    return r.returncode == 0


def sched(points, seeds, budget: int) -> int:
    """Memory-budgeted scheduler (Spark has 121 GB unified memory shared by CPU and GPU): starts a ready task only if
    the running tasks' estimated peaks plus its own fit in `budget` GB; GPU tasks first (finish points), then experts.
    Done tasks are kept in STATE (restartable); a stale output dir of an unfinished task is moved aside."""
    sp = ROOT / STATE
    state = json.loads(sp.read_text()) if sp.exists() else {"done": {}, "best": {}}
    todo = tasks(points, seeds)
    points = [x.partition(":")[0] for x in points]
    for t in todo:                                                  # expert labels made outside the scheduler
        if t["kind"] == "expert" and (ROOT / labels(t["point"])).exists() and not _external_expert(t["point"]):
            state["done"].setdefault(t["id"], "labels-exist")
    running: dict[str, tuple] = {}
    failed: set[str] = set()
    ext_mem = lambda: sum(MEM_GB["expert"] for t in todo if t["kind"] == "expert" and t["id"] not in state["done"]  # noqa: E731
                          and t["id"] not in running and _external_expert(t["point"]))
    while True:
        for tid, (p, t, fh, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            dt = time.time() - t0
            print(f"TASK {tid} RC {rc} {dt:.0f}s", flush=True)
            del running[tid]
            if rc == 0:
                state["done"][tid] = time.strftime("%Y-%m-%d %H:%M:%S")
                if t["kind"] == "grpo":
                    log = (ROOT / f"runs/ws-{t['point']}.tasks" / f"{tid.split('/')[1].replace(':', '-')}.log")
                    state["best"][tid] = _last_json(log.read_text(encoding="utf-8").splitlines())["best"]
            else:
                failed.add(tid)
            sp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        blocked = set(failed)                                       # transitively: a failed ancestor blocks for good
        grew = True
        while grew:
            grew = False
            for t in todo:
                if t["id"] not in blocked and any(d in blocked for d in t["deps"]):
                    blocked.add(t["id"])
                    grew = True
        pending = [t for t in todo if t["id"] not in state["done"] and t["id"] not in running
                   and t["id"] not in blocked]
        if not pending and not running:
            break
        used = sum(t["mem"] for _, t, _, _ in running.values()) + ext_mem()
        ready = [t for t in pending if all(d in state["done"] for d in t["deps"])
                 and not (t["kind"] == "expert" and _external_expert(t["point"]))]
        ready.sort(key=lambda t: (t["kind"] == "expert", points.index(t["point"])))
        for t in ready:
            if used + t["mem"] > budget:
                continue
            argv = list(t["argv"])
            if t["kind"] == "eval":
                argv = [state["best"][f"{t['point']}/grpo:{t['id'].split(':')[1]}"] if x == "{best}" else x
                        for x in argv]
            if t["out"] and t["kind"] in ("sft", "grpo"):
                d = ROOT / t["out"]
                if d.exists():
                    d.rename(d.with_name(d.name + f".stale-{int(time.time())}"))
            ld = ROOT / f"runs/ws-{t['point']}.tasks"
            ld.mkdir(parents=True, exist_ok=True)
            fh = open(ld / f"{t['id'].split('/')[1].replace(':', '-')}.log", "w", encoding="utf-8")
            fh.write(f"ARGV {json.dumps(argv)}\n")
            fh.flush()
            p = subprocess.Popen([sys.executable, "scripts/train.py", *argv], cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT, text=True)
            running[t["id"]] = (p, t, fh, time.time())
            used += t["mem"]
            print(f"START {t['id']} (mem {used}/{budget} GB)", flush=True)
        time.sleep(15)
    print("SCHED DONE", json.dumps({"failed": sorted(failed)}), flush=True)
    return int(bool(failed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    c = sub.add_parser("chain")
    c.add_argument("--point", required=True, choices=sorted(POINTS))
    c.add_argument("--seeds", default=",".join(STAGE1_SEEDS))
    p = sub.add_parser("pool")
    p.add_argument("--points", default=",".join(POINTS))
    p.add_argument("--seeds", default=",".join(STAGE1_SEEDS))
    p.add_argument("--parallel", type=int, default=5)
    s = sub.add_parser("sched")
    s.add_argument("--points", default=",".join(POINTS))
    s.add_argument("--seeds", default=",".join(STAGE1_SEEDS))
    s.add_argument("--budget", type=int, default=100)
    a = ap.parse_args(argv)
    if a.cmd == "sched":
        return sched([x for x in a.points.split(",") if x], tuple(x for x in a.seeds.split(",") if x), a.budget)
    if a.cmd == "jobs":
        for pt in POINTS:
            print(json.dumps({"point": pt, "w": POINTS[pt], "steps": [[n, " ".join(x)] for n, x in steps(pt)]}))
        return 0
    seeds = tuple(s for s in a.seeds.split(",") if s)
    if a.cmd == "chain":
        return chain(a.point, seeds)
    pts = [x for x in a.points.split(",") if x]
    with ThreadPoolExecutor(a.parallel) as ex:
        rcs = dict(zip(pts, ex.map(lambda pt: chain(pt, seeds), pts)))
    print("POOL DONE", json.dumps(rcs), flush=True)
    return int(any(rcs.values()))


if __name__ == "__main__":
    sys.exit(main())
