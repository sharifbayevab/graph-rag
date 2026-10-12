"""M7 Task 8: the probe evaluation (docs/m7 design §4, §5 step 2) -- baselines @5 against the depth-5 struct oracle,
per question type (count = Q1, version = Q2, compare = Q3), and the go / no-go numbers.

Run (Spark, one job; the probe is 90 queries = one chunk of <= 128):
    PYTHONPATH=. python scripts/hard_probe_eval.py --ship data/m7/probe/<sha>/ship --variant struct \\
        --set env.K_max=5 --expert-depth 5 --ce-cache runs/m7-probe-ce/ce_scores.jsonl \\
        --ranker runs/p6-sft-st --ranker-sha <sha256 of its pi_ref.pt> --out runs/m7-probe-eval/out.json
Offline (Mac, from the pulled JSON): markdown tables on stdout, the figure with --plot:
    PYTHONPATH=. python scripts/hard_probe_eval.py --from runs/spark/m7-probe-eval/out.json \\
        --plot docs/m7/figures/probe_all_at5.png

Methods, all scored on their first 5 fragment ids (grl.hard.metrics.set_metrics):
- dense@5 / bm25@5 / fused@5 (w 0.7) / ce+fused:0.6@5 -- grl.train.rank.baseline_ranking over the pilot-6 RankEnv
  (pool = the first rerank.n = 50 rows of the fused ranking; dense / bm25 = the raw U9 rankings);
- rl_ranker@5 -- the pilot-6 RL ranker θ* = SFT static s01 (runs/p6-sft-st/pi_ref.pt), greedy, on the same pool with
  the probe CE cache as its ce feature (the recorded settings must match except the cache sha: other queries);
- oracle -- the struct-variant expert (grl.train.expert.expert_search, exact up to --expert-depth retrieval steps +
  STOP, train.yaml expert.max_states / cap_beam; --beam N for a beam) in StructEnv at env.K_max = 5; its final
  context is the oracle's top-5. It reads C*: an upper bound on the reward R within depth 5 (STOP-ended finals,
  K_max = 5), NOT an upper bound on all@5 (it maximises R and has no 5-fragment final), and not a method;
- --diagnostics (rulings R28 / R29, grl.hard.oracle; diagnostics, not methods), exact searches with limit-ended
  finals counted: oracle_kmax = the expert's search (zero-Δ pruned) maximising R; oracle_noprune = no zero-Δ pruning,
  max R (the cost of the expert's prune_zero); oracle_all5 = no pruning, max all@5 then recall then R -- the
  reachability upper bound of all@5 under the 31 actions at K_max = 5, depth 5; oracle_all5_d7 = the same at depth 7,
  kept only if every query finishes within 30 min of wall time (else reported as incomplete).
STRUCT gold (count items) is in no index, so every baseline and the ranker miss it by construction."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402  -- stdlib only, before numpy / torch

pin_blas()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml  # noqa: E402

from graphrag import config  # noqa: E402
from grl.hard.metrics import (TYPES, go_numbers, oracle_diagnostics, oracle_record, set_metrics,  # noqa: E402
                              summarize_methods, top_k_ids)

K = 5
BASELINES = ("dense", "bm25", "fused", "ce+fused:0.6")
RANKER = "rl_ranker"
ORACLE = "oracle"
METHODS = BASELINES + (RANKER, ORACLE)
LABELS = {"dense": "dense@5", "bm25": "BM25@5", "fused": "fused@5 (w 0,7)", "ce+fused:0.6": "ce+fused:0.6@5",
          RANKER: "RL-ранкер П6 @5", ORACLE: "оракул struct d5",
          "oracle_kmax": "диагн.: оракул R + финалы по K_max", "oracle_noprune": "диагн.: + без отсечения zero-Δ",
          "oracle_all5": "достижимость all@5 d5 (диагностика)",
          "oracle_all5_d7": "достижимость all@5 d7 (диагностика)"}
REACH = "oracle_all5"
TYPE_LABELS = {"count": "Q1 count", "version": "Q2 version", "compare": "Q3 compare", "pooled": "Q1–Q3"}
SET_KEYS = ("env.K_max", "expert.max_states", "expert.cap_beam", "struct.articles")   # Task 6c: article mode
CHUNK = 128


def _set(cfg: dict, kv: str) -> None:
    k, v = kv.split("=", 1)
    if k not in SET_KEYS:
        raise SystemExit(f"--set {k}: only {SET_KEYS} are read by the probe eval")
    a, b = k.split(".")
    if b not in cfg[a]:
        raise SystemExit(f"--set {k}: no such config key")
    cfg[a][b] = yaml.safe_load(v)


def _sha(p) -> str:
    from grl.hashing import sha256_file
    return sha256_file(p)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


ARTIFACTS = "docs/m7/artifacts.sha256"


def verify_ship_hashes(ship: Path, sha_file, rel: str | None = None) -> int:
    """Every file of the ship dir that docs/m7/artifacts.sha256 lists must match; pairs.jsonl and
    rankings/latest.json must be listed (spark.py sync ships no manifest.json, so this is the check on Spark;
    load_rankings checks the ranking arrays against latest.json). Returns the number of files checked."""
    sha_file = Path(sha_file)
    if rel is None:                                          # the ship path as artifacts.sha256 lists it
        rel = ship.resolve().relative_to(config.ROOT.resolve()).as_posix() if ship.is_absolute() else ship.as_posix()
    want = {}
    for line in sha_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            if p.startswith(rel + "/"):
                want[p[len(rel) + 1:]] = h
    missing = [n for n in ("pairs.jsonl", "rankings/latest.json") if n not in want]
    if missing:
        raise SystemExit(f"{sha_file}: no sha256 for {rel}/{missing}")
    n = 0
    for name, h in sorted(want.items()):
        f = ship / name
        if not f.exists():
            if name != "manifest.json":                      # only the manifest is not shipped by spark.py sync
                raise SystemExit(f"{f}: missing")
            continue
        if _sha(f) != h:
            raise SystemExit(f"{f}: sha256 differs from {sha_file}")
        n += 1
    return n


def load_ship(ship: Path) -> tuple[list[dict], int]:
    from grl.hard.probe import check_probe_ship, probe_ids, read_probe_rows
    from grl.train.pilot import verify_ship
    if (ship / "manifest.json").exists():                    # Mac; spark.py sync does not ship the manifest
        verify_ship(ship.parent)
    n_checked = verify_ship_hashes(ship, config.path(ARTIFACTS))
    rows = read_probe_rows(ship / "pairs.jsonl")
    check_probe_ship(ship, None if probe_ids() else {r["qid"] for r in rows})   # Spark: no ids.json (Mac only)
    if sorted({r["qtype"] for r in rows}) != sorted(TYPES):
        raise SystemExit(f"{ship}: probe types {sorted({r['qtype'] for r in rows})} != {sorted(TYPES)}")
    return rows, n_checked


DIAGS = (("oracle_kmax", "kmax", 5), ("oracle_noprune", "noprune", 5), ("oracle_all5", "all5", 5),
         ("oracle_all5_d7", "all5", 7))
DIAG_MAX_STATES = 2_000_000                                # per query; a guard, never a beam (d5 max seen: 257k)
DEEP_MAX_STATES = 600_000                                  # depth 7: memory guard on the shared Spark (R29 run 1:
DEEP_WORKERS = 4                                           # 8 workers x 2M states broke the pool)
DIAG_BUDGET_S = 1800.0                                     # R28: the depth-7 variant only if it fits 30 min
POOL_TOLERANT_DEPTH = 5                                    # a broken pool is recorded only deeper than this
POOL_ERRORS = (BrokenProcessPool, MemoryError)


def _is_pool_error(e: BaseException) -> bool:
    """A broken pool / out of memory, raised directly or wrapped by graphrag.env.parallel.run (ParallelRunError
    `from` the worker's exception)."""
    return isinstance(e, POOL_ERRORS) or isinstance(e.__cause__, POOL_ERRORS)


def run_diag_jobs(run, jobs, depth: int) -> list[dict]:
    """run() = the diagnostic pool run. A pool error (BrokenProcessPool / MemoryError, e.g. a worker OOM-killed) of a
    variant deeper than POOL_TOLERANT_DEPTH (depth 7: memory-bound) is recorded as an error row per job; anything
    else -- a code error, or any error at depth 5 -- is raised (final review of part 1, item 9)."""
    try:
        return run()
    except Exception as e:
        if not (depth > POOL_TOLERANT_DEPTH and _is_pool_error(e)):
            raise
        return [{"qid": j[0], "error": f"pool: {type(e).__name__}: {str(e)[:300]}", "seconds": 0.0} for j in jobs]


def run_oracle(rows, rankings: str, cfg: dict, depth: int, beam, diagnostics: bool = False) -> tuple[dict, dict]:
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.probe import check_rank_rows, load_probe_items
    from grl.model.variants import StructEnv, struct_data_for, struct_kw
    from grl.train.expert import expert_config, expert_search
    from grl.train.fbeta import grl_weights
    t0 = time.time()
    data = struct_data_for(cfg)                              # struct.articles (Task 6c fix round 1)
    ranks = load_rankings(rankings, graph_version=data.version)
    keys = [f"{r['qid']}|q" for r in rows]
    rank_check = check_rank_rows(ranks, keys, data.n_base_chunks)
    items = {it.pid: it for it in load_probe_items(Path(rankings).parents[1] / "pairs.jsonl", data)}
    env = StructEnv(data, cfg["env"], **struct_kw(cfg))
    w, ec = grl_weights(cfg), expert_config(cfg)
    load_s = round(time.time() - t0, 1)
    out = {}
    for i, r in enumerate(rows):
        it = items[r["qid"]]
        q = env.query(ranks.get(it.qkey))
        t1 = time.perf_counter()
        res = expert_search(env, q, it.gold, w, depth, True, beam, ec["max_states"], ec["cap_beam"])
        rec = oracle_record(env, res, [d for d in r["docs"] if d.startswith("EXT/")])
        rec["seconds"] = round(time.perf_counter() - t1, 4)
        rec["depth"] = int(res["depth"])
        out[r["qid"]] = rec
        if (i + 1) % 10 == 0:
            print(f"oracle {i + 1}/{len(rows)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    diag, dinfo = {}, {}
    if diagnostics:
        from functools import partial

        from graphrag.env.parallel import run as prun
        from grl.hard.oracle import diag_context, diag_one
        par = cfg.get("parallel") or {}
        workers = int(par.get("workers", 1))
        init = partial(diag_context, rankings, str(Path(rankings).parents[1] / "pairs.jsonl"), cfg["env"], cfg)
        for name, mode, dd in DIAGS:
            t2 = time.time()
            dl = t2 + DIAG_BUDGET_S if dd > depth else None      # R29: depth 7 only if it fits 30 min
            jobs = [(r["qid"], mode, dd, [d for d in r["docs"] if d.startswith("EXT/")], DIAG_MAX_STATES, dl)
                    for r in rows]
            deep = dd > depth                                   # depth 7: fewer workers, tighter state guard
            nw, cap = (DEEP_WORKERS, DEEP_MAX_STATES) if deep else (workers, DIAG_MAX_STATES)
            jobs = [j[:4] + (cap,) + j[5:] for j in jobs]
            res = run_diag_jobs(partial(prun, diag_one, jobs, nw, par.get("start_method"), init=init), jobs, dd)
            errs = [x for x in res if "error" in x]
            if not errs:
                diag[name] = {x["qid"]: x for x in res}
            dinfo[name] = {"mode": mode, "depth": dd, "n": len(res) - len(errs), "complete": not errs,
                           "errors": [x["error"] for x in errs][:5], "n_errors": len(errs),
                           "seconds": round(time.time() - t2, 1), "workers": nw, "max_states": cap,
                           "n_states_max": max((x["n_states"] for x in res if "error" not in x), default=0),
                           "seconds_max": max((x["seconds"] for x in res), default=0.0)}
            if errs and deep:                                   # partial: per-query rows for the report only
                ok = [x for x in res if "error" not in x]
                d5 = diag.get("oracle_all5", {})
                dinfo[name]["partial"] = [{"qid": x["qid"], "context": x["context"], "n_states": x["n_states"],
                                           "same_as_d5": d5.get(x["qid"], {}).get("context") == x["context"]}
                                          for x in ok]
            print(f"{name} {dinfo[name]}", file=sys.stderr, flush=True)
    info = {"K_max": int(env.limits.K_max), "T_max": int(env.limits.T_max), "L_max": int(env.limits.L_max),
            "n_actions": len(env.actions), **{k: ec[k] for k in ("max_states", "cap_beam")},
            "depth": int(depth), "beam": beam, "searched_depths": sorted({int(r["depth"]) for r in out.values()}),
            "n_base_chunks": int(data.n_base_chunks), "n_chunks": int(data.n_chunks), "rank_rows": rank_check,
            "load_s": load_s, "seconds": round(time.time() - t0, 1),
            "weights": {k: float(getattr(w, k)) for k in ("alpha", "beta", "gamma", "delta", "chunks_beta")},
            "diagnostics": dinfo}
    return out, info, diag


def run_ranked(rows, rankings: str, cfg: dict, ce_cache: str, ranker: Path, ranker_sha: str | None):
    """Baselines + the pilot-6 RL ranker on the pinned (STRUCT-free) EnvData: {qid: {method: [top-5 ids]}}, info."""
    import torch

    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.graph import build_graph
    from grl.model.rank import RankEnv
    from grl.model.rerank import RerankPolicy, rerank_settings
    from grl.model.variants import model_config
    from grl.train.log import pick_device
    from grl.train.rank import baseline_ranking
    from grl.train.rollout import SerialVecEnv, rollout
    t0 = time.time()
    rec = json.loads((ranker / "rerank_settings.json").read_text(encoding="utf-8"))
    meta = json.loads((ranker / "pi_ref.json").read_text(encoding="utf-8"))
    ckpt = ranker / meta["path"]
    sha = _sha(ckpt)
    if sha != meta["sha256"] or (ranker_sha and sha != ranker_sha):
        raise SystemExit(f"{ckpt}: sha256 {sha} != pi_ref.json {meta['sha256']} / --ranker-sha {ranker_sha}")
    if not rec.get("use_ce"):
        raise SystemExit(f"{ranker}: expected a CE-feature ranker (pilot 6 SFT static)")
    kw = {"n": int(rec["n"]), "fuse_w": float(rec["fuse_w"]), "norm_top_n": int(rec["norm_top_n"]),
          "k_max": int(rec["rank_k_max"]), "static": bool(rec["rank_static"]), "use_ce": True, "ce_cache": ce_cache}
    data = load_env_data()
    ranks = load_rankings(rankings, graph_version=data.version)
    env = RankEnv(data, cfg["env"], None, **kw)
    now = rerank_settings(env)
    diff = {k: (rec.get(k), now.get(k)) for k in sorted(set(rec) | set(now)) if k != "ce_sha256"
            and rec.get(k) != now.get(k)}
    if diff:
        raise SystemExit(f"rank env settings differ from the ranker's (trained, now): {diff}")
    out = {r["qid"]: {} for r in rows}
    for r in rows:
        q = env.query(ranks.get(f"{r['qid']}|q"))
        for name in BASELINES:
            out[r["qid"]][name] = top_k_ids([data.chunk_ids[int(c)] for c in baseline_ranking(env, q, name)], K)
        out[r["qid"]]["_pool_size"] = int(env.cand_rows(q).size)
    device = pick_device(cfg["run"]["device"])
    mcfg = model_config(cfg["model"], "rank")
    mcfg["rerank_head"] = dict((cfg.get("rerank") or {}).get("head") or {})
    graph = build_graph(mcfg["graph"]).to(device)
    pol = RerankPolicy(graph, mcfg).to(device)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    pol.eval()
    vec = SerialVecEnv(data, ranks, cfg["env"], None, RankEnv, kw, {})
    keys = [f"{r['qid']}|q" for r in rows]
    for b in range(0, len(keys), CHUNK):
        eps, finals = rollout(pol, graph, vec, ranks, keys[b:b + CHUNK], 1, greedy=True, device=device)
        for e, f in zip(eps, finals):
            out[e.qkey[:-2]][RANKER] = top_k_ids([data.chunk_ids[int(c)] for c in f.context], K)
    info = {"rank_env": {**kw, "ce_cache_sha256": now["ce_sha256"]}, "ranker": str(ranker), "ranker_sha256": sha,
            "ranker_settings": rec, "ce_sha_trained": rec.get("ce_sha256"), "device": str(device),
            "seconds": round(time.time() - t0, 1)}
    return out, info


def evaluate(a) -> int:
    cfg = config.load("train")
    for kv in a.set:
        _set(cfg, kv)
    if a.variant != "struct":
        raise SystemExit("--variant: the probe oracle is the struct variant")
    ship = Path(a.ship) if Path(a.ship).is_absolute() else config.path(a.ship)
    rows, n_checked = load_ship(ship)
    rankings = str(ship / "rankings" / "latest.json")
    t0 = time.time()
    ranked, rinfo = run_ranked(rows, rankings, cfg, a.ce_cache, config.path(a.ranker), a.ranker_sha)
    print(f"baselines + ranker {rinfo['seconds']}s", file=sys.stderr, flush=True)
    orc, oinfo, diag = run_oracle(rows, rankings, cfg, int(a.expert_depth), a.beam, a.diagnostics)
    per, mrows = [], []
    for r in rows:
        m = {name: {"top5": ranked[r["qid"]][name], **set_metrics(ranked[r["qid"]][name], r["gold"])}
             for name in BASELINES + (RANKER,)}
        o = orc[r["qid"]]
        m[ORACLE] = {"top5": o["context"], **set_metrics(o["context"], r["gold"])}
        for name, recs in diag.items():                     # R28 diagnostics: scored like the oracle
            d = recs[r["qid"]]
            d.update(set_metrics(d["context"], r["gold"]), qtype=r["qtype"])
            m[name] = {"top5": d["context"], **set_metrics(d["context"], r["gold"]), "path": d["path"],
                       "reason": d["reason"], "n_states": d["n_states"]}
        for name, v in m.items():
            mrows.append({"method": name, "qtype": r["qtype"], "qid": r["qid"], "recall": v["recall"], "all": v["all"]})
        o.update(set_metrics(o["context"], r["gold"]), qtype=r["qtype"])
        per.append({k: r[k] for k in ("qid", "qtype", "template", "lang", "question", "gold", "docs", "meta")}
                   | {"methods": m, "oracle": o, "pool_size": ranked[r["qid"]]["_pool_size"]})
    table = summarize_methods(mrows)
    out = {"k": K, "n": len(rows), "methods": list(METHODS) + list(diag), "diagnostics": list(diag), "table": table,
           "go": go_numbers(table, list(BASELINES) + [RANKER], ORACLE),
           "oracle_diag": oracle_diagnostics([q["oracle"] for q in per]),
           "diag_oracle_diag": {n: oracle_diagnostics(list(recs.values())) for n, recs in diag.items()},
           "per_query": per,
           "provenance": {"ship": str(Path(a.ship)), "ship_manifest_sha256": (_sha(ship / "manifest.json")
                                                   if (ship / "manifest.json").exists() else None),
                          "artifacts_checked": n_checked,
                          "pairs_sha256": _sha(ship / "pairs.jsonl"),
                          "rankings_version": json.loads(Path(rankings).read_text())["version"],
                          "ce_cache": a.ce_cache, "set": list(a.set), "variant": a.variant, "oracle": oinfo,
                          "ranked": rinfo, "seconds": round(time.time() - t0, 1),
                          "graph_version": cfg["env"]["graph_version"]}}
    from grl.train.log import run_manifest
    try:
        out["provenance"]["run"] = {k: v for k, v in run_manifest(cfg, {}, {}).items()
                                    if k in ("git_head", "git_dirty", "config_sha256", "machine", "versions")}
    except Exception as e:                                     # provenance only, never fatal
        out["provenance"]["run"] = {"error": repr(e)}
    _write_json(Path(a.out) if Path(a.out).is_absolute() else config.path(a.out), out)
    print(json.dumps({"go": out["go"], "seconds": out["provenance"]["seconds"],
                      "oracle_seconds": oinfo["seconds"]}, ensure_ascii=False))
    return 0


# ---- offline: tables + figure ------------------------------------------------------------------------------------

def _f(x, se=None) -> str:
    if x is None:
        return "—"
    s = f"{x:.3f}".replace(".", ",")
    return s if se is None else f"{s} ± {se:.3f}".replace(".", ",")


def tables(out: dict) -> str:
    t, lines = out["table"], []
    cols = list(TYPES) + ["pooled"]
    lines.append("| метод | " + " | ".join(f"{TYPE_LABELS[c]} recall@5 | {TYPE_LABELS[c]} all@5" for c in cols) + " |")
    lines.append("|---" * (1 + 2 * len(cols)) + "|")
    for m in out["methods"]:
        cells = [f"{_f(t[m][c]['recall'], t[m][c]['recall_se'])} | {_f(t[m][c]['all'], t[m][c]['all_se'])}"
                 for c in cols]
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(out: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, AQUA, YELLOW, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False})
    colors = dict(zip(METHODS, (BLUE, ORANGE, AQUA, YELLOW, MUTED, SURFACE)))
    t, cols = out["table"], list(TYPES) + ["pooled"]
    series = list(METHODS) + ([REACH] if REACH in t else [])
    colors[REACH] = SURFACE
    fig, ax = plt.subplots(figsize=(12, 4.8))
    n = len(series)
    wbar = 0.78 / n
    for j, m in enumerate(series):
        xs = [i + (j - (n - 1) / 2) * wbar for i in range(len(cols))]
        ys = [t[m][c]["all"] for c in cols]
        es = [t[m][c]["all_se"] for c in cols]
        orc, reach = m == ORACLE, m == REACH
        ax.bar(xs, ys, wbar * 0.92, yerr=es, color=colors[m], edgecolor=INK if (orc or reach) else SURFACE,
               linewidth=1.2 if (orc or reach) else 0.8, hatch="////" if orc else None, label=LABELS[m],
               linestyle="--" if reach else "-", error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        if orc or reach:                                      # oracle / reachability only: labels (no clutter)
            for x, y, e, c in zip(xs, ys, es, cols):
                if reach and abs(y - t[ORACLE][c]["all"]) < 1e-9:
                    continue                                  # equal to the oracle bar beside it: one label
                ax.text(x, y + e + 0.02, f"{y:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=7.5,
                        color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(-0.47, 0.51, "порог go 0,5", ha="left", va="bottom", color=INK2, fontsize=9)
    zero = [c for c in cols[:-1] if all(t[m][c]["all"] == 0 for m in METHODS if m != ORACLE)]
    for c in zero:                                            # all-zero baseline cluster: say why, not a blank
        x0 = cols.index(c) + (2 - (n - 1) / 2) * wbar
        ax.text(x0, 0.03, "базы: 0\n(STRUCT нет\nв индексах)", ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks(range(len(cols)), [TYPE_LABELS[c] + f"\n(n = {t[ORACLE][c]['n']})" for c in cols])
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("all@5 (все C* в top-5), ± SE")
    ax.set_title("Пробник M7: all@5 по методам и типам вопросов")
    ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor=colors[m], edgecolor=INK if m in (ORACLE, REACH) else colors[m],
                     hatch="////" if m == ORACLE else None, linestyle="--" if m == REACH else "-",
                     label="достижимость all@5, d5 (диагностика)" if m == REACH else LABELS[m]) for m in series]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)


def offline(a) -> int:
    out = json.loads(Path(a.from_json).read_text(encoding="utf-8"))
    print(tables(out))
    print(json.dumps({"go": out["go"], "oracle_diag": out["oracle_diag"],
                      "diag_oracle_diag": out.get("diag_oracle_diag"),
                      "diag_info": out.get("provenance", {}).get("oracle", {}).get("diagnostics")},
                     ensure_ascii=False, indent=1))
    if a.plot:
        plot(out, Path(a.plot))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ship", help="the probe ship dir data/m7/probe/<sha>/ship")
    ap.add_argument("--variant", default="struct")
    ap.add_argument("--set", action="append", default=[], help=f"one of {SET_KEYS}")
    ap.add_argument("--expert-depth", type=int, default=5)
    ap.add_argument("--beam", type=int, help="oracle: keep the best N live states per depth (default exact)")
    ap.add_argument("--ce-cache", default="runs/m7-probe-ce/ce_scores.jsonl")
    ap.add_argument("--ranker", default="runs/p6-sft-st", help="pilot-6 SFT static s01 run dir (pi_ref.pt)")
    ap.add_argument("--ranker-sha", help="expected sha256 of the ranker's pi_ref.pt (docs/m5/artifacts.sha256)")
    ap.add_argument("--diagnostics", action="store_true", help="R28 / R29: also the diagnostic oracles oracle_kmax / oracle_noprune / "
                    "oracle_all5 (depth 5) and oracle_all5_d7 (if it fits 30 min); not methods")
    ap.add_argument("--out", help="output JSON")
    ap.add_argument("--from", dest="from_json", help="offline: read this output JSON (tables, --plot)")
    ap.add_argument("--plot", help="offline: write the all@5 figure here")
    a = ap.parse_args(argv)
    if a.from_json:
        return offline(a)
    if not a.ship or not a.out:
        ap.error("--ship and --out are required (or --from for the offline tables / figure)")
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
