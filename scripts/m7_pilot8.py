"""M7 pilot 8 (validation only; the test splits stay sealed): within-episode accumulation, the pilot-5 RL ranker on
the joint val, the CE parity check, the type router and the report tables.

Spark (one job each):
    PYTHONPATH=. python scripts/m7_pilot8.py accum --pairs runs/p8-expert/labels.jsonl \\
        --rankings data/m7/full/<sha>/ship/rankings/latest.json --policy trained=runs/<sft>/checkpoints/<best>.pt \\
        --policy init=runs/<sft>/init_policy.pt --random-reps 10 --out runs/p8-accum/accum.json
  Greedy episodes of each policy on the val rows (variant struct, K_max 5) are replayed step by step in the struct
  env; after every step t = 0..T the context is scored: recall of C*, all (C* within the context) and the grl reward R
  of that context as if the episode had stopped there (an ended episode keeps its final context). The random policy
  draws uniformly among the valid actions (STOP included) at every step, --random-reps episodes per query (seeded).
  Means / SE per group (old, count = Q1, version = Q2) and per t; per-query finals of every policy.
    PYTHONPATH=. python scripts/m7_pilot8.py ranker --ship data/m7/full/<sha>/ship --run runs/p5-rr-sft \\
        --ce-cache runs/p7-ce/ce_scores.jsonl --k-max 5 --out runs/p8-ranker/k5.jsonl
  The pilot-5 RL ranker (variant rerank, its pi_ref = the pilot-5 candidate, val 0.7034) on every val row of the
  ship: the rerank settings must equal the run's (n, fuse_w, norm_top_n, L_max, use_ce) except the ce cache sha256 --
  the pilot-5 cache does not cover the new val rows; `ce-parity` checks the shared scores. The reward is computed on
  the STRUCT EnvData (its base rows are the base EnvData's rows: checked), so Q1 rows (STRUCT gold, in no index) are
  scored too; for old rows the base and struct rewards must agree. top-5 = the first 5 selected fragments.
Mac (offline):
    ce-parity --a runs/spark/p5-ce/ce_scores.jsonl --b runs/spark/p7-ce/ce_scores.jsonl --ship ... --out F
      (only the ship's val keys are read from either file)
    router --ship ... --agent PQ.jsonl --ranker PQ.jsonl --out F   (logistic regression on the bge-m3 question
      vectors of the ship rankings, fit on TRAIN rows only; val accuracy / confusion; combined system per group)"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas()

from graphrag import config  # noqa: E402

GROUPS = ("old", "count", "version")
T_PLOT = 6


def _group(pattern: str) -> str:
    return pattern if pattern in ("count", "version") else "old"


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if str(path).endswith(".jsonl"):
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in obj),
                       encoding="utf-8")
    else:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def mean_se(xs) -> dict:
    xs = [float(x) for x in xs]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "se": None}
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 6), "se": round(se, 6)}


def _train_cli():
    from tests.scripts_import import load_script
    return load_script("train")


# --- accum (Spark) -------------------------------------------------------------------------------------------------

def _trace(env, ranks, it, actions, w, T: int, data) -> list[dict]:
    """Per-step context metrics of one episode (replayed actions); row t = after t steps (t = 0: the start)."""
    from grl.train.fbeta import grl_reward
    q = env.query(ranks.get(it.qkey))
    s = env.reset(q)
    ctxs = [s]
    for a in actions:
        s = env.step(q, s, int(a))
        ctxs.append(s)
    out = []
    for t in range(T + 1):
        st = ctxs[min(t, len(ctxs) - 1)]
        C = set(int(c) for c in st.context)
        p = grl_reward(data, q.depth, list(st.context), it.gold, w, env.limits.T_max)
        out.append({"recall": len(C & it.gold) / len(it.gold), "all": float(it.gold <= C), "R": p.total})
    return out, ctxs[-1]


def _random_actions(env, ranks, it, dkw, rng) -> list[int]:
    from grl.model.observe import describe
    import numpy as np
    q = env.query(ranks.get(it.qkey))
    s, acts = env.reset(q), []
    while not s.done:
        d = describe(env, q, s, **dkw)
        valid = np.flatnonzero(np.asarray(d.mask))
        a = int(rng.choice(valid))
        acts.append(a)
        s = env.step(q, s, a)
    return acts


def _agg(traces: dict, items, T: int) -> dict:
    """traces: qkey -> list of per-episode traces (each T+1 rows). Per group / t: mean and SE over queries of the
    per-query mean (random: averaged over its episodes first)."""
    out = {}
    for g in GROUPS + ("all",):
        its = [i for i in items if g == "all" or _group(i.pattern) == g]
        if not its:
            continue
        out[g] = {}
        for key in ("recall", "all", "R"):
            out[g][key] = [mean_se([sum(tr[t][key] for tr in traces[i.qkey]) / len(traces[i.qkey]) for i in its])
                           for t in range(T + 1)]
    return out


def cmd_accum(a) -> int:
    import numpy as np
    import torch
    from grl.train.evaluate import evaluate_policy
    t0 = time.time()
    cli = _train_cli()
    ns = argparse.Namespace(cmd="eval", pairs=a.pairs, rankings=a.rankings, variant="struct", split="val",
                            set=[f"env.K_max={a.k_max}", *a.set], run=None, sft=None, ckpt=None, per_query=None)
    S = cli.setup(ns)
    items, env, data, w = S["val"], S["env"], S["data"], S["w"]
    T = a.t_plot
    res = {"K_max": int(env.limits.K_max), "n": len(items), "T": T, "policies": {}, "finals": {},
           "groups_n": dict(Counter(_group(i.pattern) for i in items))}
    from grl.model.policy import GraphPolicy
    for spec in a.policy:
        name, path = spec.split("=", 1)
        pol = GraphPolicy(S["graph"], S["mcfg"]).to(S["device"])
        st = torch.load(path, map_location="cpu", weights_only=True)
        pol.load_state_dict(st["model"] if "model" in st else st)
        s, eps, finals = evaluate_policy(pol, S["graph"], S["vec"], S["ranks"], items, w, env.limits.T_max,
                                         S["device"], env.actions)
        by = {e.qkey: (e, f) for e, f in zip(eps, finals)}
        traces, fin = {}, {}
        for it in items:
            e, f = by[it.qkey]
            tr, last = _trace(env, S["ranks"], it, e.actions, w, T, data)
            if frozenset(last.ctx_set) != frozenset(f.ctx_set):
                raise SystemExit(f"{name} {it.qkey}: the replayed context differs from the evaluation's")
            full, _ = _trace(env, S["ranks"], it, e.actions, w, len(e.actions), data)
            traces[it.qkey] = [tr]
            fin[it.qkey] = {"group": _group(it.pattern), "R": full[-1]["R"], "all": full[-1]["all"],
                            "recall": full[-1]["recall"], "len": len(e.actions)}
        res["policies"][name] = {"path": path, "accum": _agg(traces, items, T), "summary": {
            "reward_mean": s["reward_mean"], "by_group": s.get("by_group")}}
        res["finals"][name] = fin
        print(name, json.dumps({g: v["R"][-1] for g, v in res["policies"][name]["accum"].items()}), flush=True)
    rng = np.random.default_rng(a.seed)
    traces, fin = {}, {}
    for it in items:
        trs, fr = [], []
        for _ in range(a.random_reps):
            acts = _random_actions(env, S["ranks"], it, S["dkw"], rng)
            tr, _ = _trace(env, S["ranks"], it, acts, w, T, data)
            full, _ = _trace(env, S["ranks"], it, acts, w, len(acts), data)
            trs.append(tr)
            fr.append((full[-1]["R"], full[-1]["all"], full[-1]["recall"], len(acts)))
        traces[it.qkey] = trs
        fin[it.qkey] = {"group": _group(it.pattern), "R": float(np.mean([x[0] for x in fr])),
                        "all": float(np.mean([x[1] for x in fr])), "recall": float(np.mean([x[2] for x in fr])),
                        "len": float(np.mean([x[3] for x in fr]))}
    res["policies"]["random"] = {"path": None, "reps": a.random_reps, "seed": a.seed,
                                 "accum": _agg(traces, items, T)}
    res["finals"]["random"] = fin
    res["seconds"] = round(time.time() - t0, 1)
    _write(a.out, res)
    print(json.dumps({"n": len(items), "seconds": res["seconds"]}))
    if hasattr(S["vec"], "close"):
        S["vec"].close()
    return 0


def eval_rows(a) -> tuple[list[dict], Path]:
    """(rows, rankings latest.json) of `ranker` / `baselines`: the ship's val rows and rankings (default), or -- Task 9f,
    --split test, the final test only -- every row of a test-eval file (grl.train.data.load_test_items' guard: the
    file name, split test, the test-eval mark) with its own --rankings. Never train rows."""
    split = getattr(a, "split", "val") or "val"
    ship = config.path(a.ship)
    if split == "val":
        if getattr(a, "rows", None) or getattr(a, "rankings", None):
            raise SystemExit("--rows / --rankings go with --split test only")
        rows = [r for r in _jsonl(ship / "pairs.jsonl") if r["split"] == "val"]
        lt = ship / "rankings" / "latest.json"
    elif split == "test":
        from grl.train.data import TEST_EVAL_MARK, TEST_EVAL_NAME
        if not (a.rows and a.rankings) or Path(a.rows).name != TEST_EVAL_NAME:
            raise SystemExit(f"--split test reads only a {TEST_EVAL_NAME} file (--rows) with its --rankings")
        rows = _jsonl(config.path(a.rows))
        if any(r.get("split") != "test" or not (r.get("provenance") or {}).get(TEST_EVAL_MARK) for r in rows):
            raise SystemExit(f"{a.rows}: rows that are not marked test-eval rows")
        lt = config.path(a.rankings)
    else:
        raise SystemExit(f"--split {split}: val | test")
    if not rows:
        raise SystemExit(f"no {split} rows")
    return rows, lt


# --- ranker (Spark) ------------------------------------------------------------------------------------------------

class _NoStop:
    """Pilot 9: a RerankPolicy view whose STOP is masked while any candidate slot is valid, so a greedy rollout
    returns the ranker's own ordered top-K_max list (its first pick = the normal greedy pick when that is a slot)."""

    def __init__(self, policy):
        self.policy = policy

    def encode(self, g, qvec):
        return self.policy.encode(g, qvec)

    def log_probs(self, z, qz, qidx, descs):
        from grl.model.policy import masked_log_softmax
        lg, mask = self.policy.logits(z, qz, qidx, descs)
        mask = mask.clone()
        mask[mask[:, 1:].any(1), 0] = False
        return masked_log_softmax(lg, mask)


def cmd_ranker(a) -> int:
    import torch
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    from grl.model.graph import build_graph
    from grl.model.rerank import RerankPolicy, rerank_settings
    from grl.model.variants import StructEnv, load_variant_data, variant_env_kw
    from grl.train.data import guard_qkeys
    from grl.train.fbeta import grl_reward, grl_weights
    from grl.train.rollout import SerialVecEnv, real_init, rollout
    from grl.train.sft import load_reference
    t0 = time.time()
    cli = _train_cli()
    cfg = config.load("train")
    for kv in (f"rerank.use_ce=true", f"rerank.ce_cache={a.ce_cache}", f"env.K_max={a.k_max}"):
        cli._set(cfg, kv)
    run = Path(a.run)
    recorded = json.loads((run / "rerank_settings.json").read_text(encoding="utf-8"))
    cfg["rerank"]["n"] = int(recorded["n"])
    cfg["dynamic_k"]["fuse_w"] = float(recorded["fuse_w"])
    cfg["dynamic_k"]["norm_top_n"] = int(recorded["norm_top_n"])
    rows, lt = eval_rows(a)                                # 9f: val of the ship, or a marked test-eval file
    keys = [f"{r['id']}|q" for r in rows]
    guard_qkeys(keys)
    base = load_variant_data("rerank")
    sdata = load_struct_env_data(articles=a.articles)       # Task 6c: base-row check only
    nb = base.n_chunks if hasattr(base, "n_chunks") else len(base.chunk_ids)
    if list(sdata.chunk_ids[:nb]) != list(base.chunk_ids) or sdata.n_base_chunks != nb:
        raise SystemExit("the struct EnvData's base rows differ from the base EnvData")
    ranks = load_rankings(lt, graph_version=base.version)
    akw = variant_env_kw("rerank", cfg, observe=True)
    mcfg = cli._model_cfg(cfg, "rerank")
    dkw = {"preview": int(mcfg["state"]["preview"]), "cand_top": int(mcfg["state"]["cand_top"])}
    vec = SerialVecEnv(**real_init(str(lt), "rerank", dkw, akw, env_cfg=cfg["env"]))
    from grl.model.rerank import RerankEnv
    renv = RerankEnv(base, cfg["env"], None, **akw)
    now = rerank_settings(renv)
    diff = {k: (recorded.get(k), now.get(k)) for k in set(recorded) | set(now) if recorded.get(k) != now.get(k)}
    if set(diff) - {"ce_sha256"}:
        raise SystemExit(f"rerank settings differ beyond the ce cache: {diff}")
    graph = build_graph(mcfg["graph"])
    ref_meta = run / "pi_ref.json"
    pol = load_reference(graph, mcfg, ref_meta, policy_cls=RerankPolicy)
    eps, finals = rollout(pol, graph, vec, ranks, keys, 1, greedy=True)
    top5_lists = None
    if getattr(a, "top5", False):                      # pilot 9 fairness: the ranker's own top-5 list (STOP masked)
        top5_lists = {e.qkey: [int(c) for c in f.context]
                for e, f in zip(*rollout(_NoStop(pol), graph, vec, ranks, keys, 1, greedy=True))}
    senv = StructEnv(sdata, cfg["env"], None, articles=a.articles,          # fix 6d-b: the env flags from args
                     amend_op=bool(getattr(a, "amend_op", False)))
    w = grl_weights(cfg)
    out, maxdiff = [], 0.0
    by = {e.qkey: (e, f) for e, f in zip(eps, finals)}
    for r, k in zip(rows, keys):
        e, f = by[k]
        ctx = [int(c) for c in f.context]
        gold_s = frozenset(int(sdata.chunk_row[g]) for g in r["C_ref"])
        qs = senv.query(ranks.get(k))
        parts = grl_reward(sdata, qs.depth, ctx, gold_s, w, senv.limits.T_max)
        R = parts.total
        if _group(r["pattern"]) != "count":
            gold_b = frozenset(int(base.chunk_row[g]) for g in r["C_ref"])
            qb = renv.query(ranks.get(k))
            Rb = grl_reward(base, qb.depth, ctx, gold_b, w, renv.limits.T_max).total
            maxdiff = max(maxdiff, abs(R - Rb))
        top5 = set(ctx[:5])
        C = set(ctx)
        out.append({"qkey": k, "pattern": r["pattern"], "group": _group(r["pattern"]), "context": ctx,
                    "reason": f.reason, "R": R, "recall": len(C & gold_s) / len(gold_s),
                    "recall5": len(top5 & gold_s) / len(gold_s), "all5": float(gold_s <= top5),
                    "all": float(gold_s <= C), "n_ctx": len(ctx), "r_chunks": parts.chunks,
                    "r_coverage": parts.coverage, "r_diversity": parts.diversity, "r_coherence": parts.coherence,
                    "precision": len(C & gold_s) / len(C) if C else 0.0})
        if top5_lists is not None:
            t5 = top5_lists[k]
            if t5[:1] != ctx[:1]:
                raise SystemExit(f"{k}: the no-STOP top-1 {t5[:1]} differs from the greedy pick {ctx[:1]}")
            out[-1].update({"context_top5": t5, "recall5_top5": len(set(t5[:5]) & gold_s) / len(gold_s),
                            "all5_top5": float(gold_s <= set(t5[:5]))})
    if maxdiff > 1e-9:
        raise SystemExit(f"base vs struct reward differ on old / Q2 rows by {maxdiff}")
    _write(a.out, out)
    meta = {"run": str(run), "pi_ref_sha256": json.loads(ref_meta.read_text())["sha256"], "settings_recorded":
            recorded, "settings_now": now, "settings_diff": diff, "K_max": a.k_max, "n": len(out),
            "base_vs_struct_reward_maxdiff": maxdiff, "seconds": round(time.time() - t0, 1),
            "by_group": {g: {"R": mean_se([x["R"] for x in out if x["group"] == g]),
                             "all5": mean_se([x["all5"] for x in out if x["group"] == g])} for g in GROUPS}}
    _write(str(a.out) + ".meta.json", meta)
    print(json.dumps(meta["by_group"]))
    return 0


# --- baselines (Spark): top-20 of the fixed-k baselines on every val row --------------------------------------------

BASELINES = ("dense", "bm25", "fused", "ce+fused:0.6")
RANK_KW = {"n": 50, "fuse_w": 0.7, "norm_top_n": 50, "k_max": 20, "static": True, "use_ce": True}   # pilot 6 / 7 pool


def cmd_baselines(a) -> int:
    """The pilot-7 baselines (scripts/m7_pilot7.py: grl.train.rank.baseline_ranking over the pilot-6 RankEnv pool,
    base EnvData -- STRUCT rows are in no index) on EVERY val row of the ship, the first 20 chunk ids of each; fused
    is also the fill order of «agent + fused fill» @10."""
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.rank import RankEnv
    from grl.train.rank import baseline_ranking
    t0 = time.time()
    cfg = config.load("train")
    rows, lt = eval_rows(a)                                # 9f: val of the ship, or a marked test-eval file
    data = load_env_data()
    ranks = load_rankings(lt, graph_version=data.version)
    env = RankEnv(data, cfg["env"], None, **RANK_KW, ce_cache=a.ce_cache)
    out = {}
    for r in rows:
        q = env.query(ranks.get(f"{r['id']}|q"))
        out[r["id"]] = {n: [data.chunk_ids[int(c)] for c in baseline_ranking(env, q, n)] for n in BASELINES}
    _write(a.out, {"k": RANK_KW["k_max"], "n": len(rows), "baselines": list(BASELINES), "top": out,
                   "rank_env": {**RANK_KW, "ce_cache": a.ce_cache}, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({"n": len(rows), "seconds": round(time.time() - t0, 1)}))
    return 0


# --- ce-parity (Mac) -----------------------------------------------------------------------------------------------

def cmd_ce_parity(a) -> int:
    ship = config.path(a.ship)
    val = {f"{r['id']}|q" for r in _jsonl(ship / "pairs.jsonl") if r["split"] == "val"}

    def read(p):
        out = {}
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            k = line.split('"qkey": "', 1)[1].split('"', 1)[0] if '"qkey": "' in line else json.loads(line)["qkey"]
            if k in val:                                       # val keys only: nothing else is parsed
                r = json.loads(line)
                out[k] = dict(zip(r["rows"], r["scores"]))
        return out
    A, B = read(a.a), read(a.b)
    common = sorted(set(A) & set(B))
    diffs = [abs(A[k][c] - B[k][c]) for k in common for c in set(A[k]) & set(B[k])]
    rep = {"val_keys_a": len(A), "val_keys_b": len(B), "common_keys": len(common), "common_pairs": len(diffs),
           "max_abs_diff": max(diffs) if diffs else None,
           "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
           "b_covers_val": len(B) == len(val), "n_val": len(val)}
    _write(a.out, rep)
    print(json.dumps(rep))
    return 0


# --- router (Mac) --------------------------------------------------------------------------------------------------

def cmd_router(a) -> int:
    import numpy as np
    from graphrag.retrieval.rankings import load_rankings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    ship = config.path(a.ship)
    pairs = _jsonl(ship / "pairs.jsonl")
    if {r["split"] for r in pairs} - {"train", "val"}:
        raise SystemExit("ship holds rows outside train / val")
    ranks = load_rankings(ship / "rankings" / "latest.json")
    X = {r["id"]: np.asarray(ranks.get(f"{r['id']}|q").qvec, np.float32) for r in pairs}
    tr = [r for r in pairs if r["split"] == "train"]
    va = [r for r in pairs if r["split"] == "val"]
    lab = {g: i for i, g in enumerate(GROUPS)}
    Xtr = np.stack([X[r["id"]] for r in tr])
    ytr = np.array([lab[_group(r["pattern"])] for r in tr])
    clf = LogisticRegression(max_iter=2000, C=a.C, class_weight="balanced")
    clf.fit(Xtr, ytr)
    Xva = np.stack([X[r["id"]] for r in va])
    yva = np.array([lab[_group(r["pattern"])] for r in va])
    pred = clf.predict(Xva)
    cm = confusion_matrix(yva, pred, labels=[0, 1, 2]).tolist()
    acc = float((pred == yva).mean())
    train_acc = float((clf.predict(Xtr) == ytr).mean())
    route = {r["id"]: GROUPS[int(p)] for r, p in zip(va, pred)}
    agent = {r["qkey"][:-2]: r for r in _jsonl(a.agent)}
    ranker = {r["qkey"][:-2]: r for r in _jsonl(a.ranker)}
    if set(agent) < {r["id"] for r in va} or set(ranker) < {r["id"] for r in va}:
        raise SystemExit("per-query files do not cover the val rows")

    def ag(k):
        r = agent[k]
        return {"R": r["reward"], "all5": float(r["recall"] >= 1.0 - 1e-12), "recall5": r["recall"]}

    def rk(k):
        r = ranker[k]
        return {"R": r["R"], "all5": r["all5"], "recall5": r["recall5"]}
    systems = {"agent": lambda k: ag(k), "ranker": lambda k: rk(k),
               "router": lambda k: rk(k) if route[k] == "old" else ag(k),
               "oracle_router": lambda k: rk(k) if _group(next(r for r in va if r["id"] == k)["pattern"]) == "old"
               else ag(k)}
    table, per = {}, {}
    for name, f in systems.items():
        vals = {r["id"]: f(r["id"]) for r in va}
        per[name] = vals
        table[name] = {g: {m: mean_se([vals[r["id"]][m] for r in va if g == "all" or _group(r["pattern"]) == g])
                           for m in ("R", "all5", "recall5")} for g in GROUPS + ("all",)}
    paired = {}
    for other in ("agent", "ranker"):
        paired[f"router_minus_{other}"] = {
            g: {m: mean_se([per["router"][r["id"]][m] - per[other][r["id"]][m] for r in va
                            if g == "all" or _group(r["pattern"]) == g]) for m in ("R", "all5")}
            for g in GROUPS + ("all",)}
    out = {"classifier": {"model": "LogisticRegression(class_weight=balanced)", "C": a.C, "features":
                          "bge-m3 question vector (U9 rankings qvec, 1024-d)", "n_train": len(tr), "n_val": len(va),
                          "train_by_group": dict(Counter(_group(r["pattern"]) for r in tr)),
                          "train_acc": round(train_acc, 4), "val_acc": round(acc, 4), "labels": list(GROUPS),
                          "confusion_val": cm},
           "table": table, "paired": paired, "routes": route,
           "inputs": {"ship": a.ship, "agent": a.agent, "ranker": a.ranker}}
    _write(a.out, out)
    print(json.dumps({"val_acc": acc, "confusion": cm, "table": {k: {g: (v[g]["R"]["mean"], v[g]["all5"]["mean"])
                                                                     for g in v} for k, v in table.items()}},
                     ensure_ascii=False))
    return 0


# --- tables (Mac, offline): @5 / @10 (agent + fused fill) / dynamic k, R, k, limit share per group -----------------

def _fill10(sel: list[str], fused: list[str], struct_prefix: str) -> list[str]:
    """«agent + fused fill»: the agent's selected fragments in selection order (first 10), then the fused ranking
    order (STRUCT virtual rows never come from it; duplicates skipped) up to 10."""
    out = list(dict.fromkeys(sel))[:10]
    for c in fused:
        if len(out) >= 10:
            break
        if c not in out and not c.startswith(struct_prefix):
            out.append(c)
    return out


def _set_m(ids, gold) -> tuple[float, float]:
    S = set(ids)
    return len(S & gold) / len(gold), float(gold <= S)


def cmd_tables(a) -> int:
    import numpy as np
    from grl.env.struct_data import load_struct_env_data
    from grl.hard.probe import STRUCT_PREFIX
    ship = config.path(a.ship)
    val = {r["id"]: r for r in _jsonl(ship / "pairs.jsonl") if r["split"] == "val"}
    data = load_struct_env_data(articles=a.articles)        # Task 6c: article STRUCT gold ids are rows
    ids = data.chunk_ids
    bl = json.loads(Path(a.baselines).read_text(encoding="utf-8"))
    if sorted(bl["top"]) != sorted(val):
        raise SystemExit("baselines do not cover exactly the val rows")
    routes = json.loads(Path(a.router).read_text(encoding="utf-8"))["routes"] if a.router else None
    agents = {}
    for spec in a.agent:                                   # name=path[:kind]  kind: struct | ranker
        name, rest = spec.split("=", 1)
        path, kind = rest.rsplit(":", 1) if rest.endswith((":struct", ":ranker")) else (rest, "struct")
        per = {}
        for r in _jsonl(path):
            k = r["qkey"][:-2]
            if k not in val:
                raise SystemExit(f"{path}: {k} is not a val row")
            per[k] = {"ctx": [ids[int(c)] for c in r["context"]], "R": r["reward"] if kind == "struct" else r["R"],
                      "reason": r["reason"],
                      "parts": {p: r.get(p) for p in ("r_chunks", "r_coverage", "r_diversity", "r_coherence",
                                                       "precision")}}
        if set(per) != set(val):
            raise SystemExit(f"{path}: does not cover the val rows")
        agents[name] = per
    if routes is not None:
        a_s, a_r = a.router_struct, a.router_ranker
        agents["router"] = {k: agents[a_r][k] if routes[k] == "old" else agents[a_s][k] for k in val}

    def grp(k):
        return _group(val[k]["pattern"])
    slice126 = set()
    if a.old_traj:                                   # the 126 old val rows with a pilot-5 oracle trajectory
        slice126 = {r["id"] for r in _jsonl(config.path(a.old_traj)) if r["split"] == "val"} & set(val)
    rows = {}
    for name, per in agents.items():
        m = defaultdict(dict)
        for k, r in per.items():
            gold = set(val[k]["C_ref"])
            fused = bl["top"][k]["fused"]
            r5, a5 = _set_m(r["ctx"][:5], gold)
            r10, a10 = _set_m(_fill10(r["ctx"], fused, STRUCT_PREFIX), gold)
            rd, ad = _set_m(r["ctx"], gold)
            S, G = set(r["ctx"]), gold
            prec = len(S & G) / len(S) if S else 0.0
            f1 = 2 * prec * rd / (prec + rd) if prec + rd else 0.0
            f1_star = 2 * rd / (1 + rd) if rd else 0.0      # the same hits at precision 1 (count penalty removed)
            m[k] = {"R": r["R"], "set_precision": prec, "set_f1": f1, "set_f1_hits_only": f1_star,
                    "recall5": r5, "all5": a5, "recall10": r10, "all10": a10, "recall_dyn": rd,
                    "all_dyn": ad, "k": len(r["ctx"]), "limit": float(r["reason"] in ("k_max", "l_max")),
                    **{p: v for p, v in r["parts"].items() if v is not None}}
        rows[name] = m
    kmean = {name: float(np.mean([v["k"] for v in rows[name].values()])) for name in rows}
    ks = sorted({5, 10} | {max(1, int(round(kmean[n]))) for n in a.k_from})
    for b in bl["baselines"]:
        m = {}
        for k in val:
            gold = set(val[k]["C_ref"])
            top = bl["top"][k][b]
            m[k] = {}
            for kk in ks:
                rr, aa = _set_m(top[:kk], gold)
                m[k][f"recall{kk}"], m[k][f"all{kk}"] = rr, aa
        rows[f"bl:{b}"] = m
    table = {}
    for name, m in rows.items():
        keys = sorted(set.intersection(*(set(v) for v in m.values())))       # metrics every query has
        table[name] = {}
        for g in GROUPS + ("all",) + (("old126",) if slice126 else ()):
            ks_ = [k for k in val if (g == "all" or grp(k) == g or (g == "old126" and k in slice126))]
            t = {x: mean_se([m[k][x] for k in ks_]) for x in keys}
            if "k" in keys:
                t["k_median"] = float(np.median([m[k]["k"] for k in ks_]))
            table[name][g] = t
    out = {"table": table, "k_mean": kmean, "baseline_ks": ks, "n": {g: sum(grp(k) == g for k in val) for g in GROUPS},
           "fill": "agent + fused fill: selected fragments, then the pilot-7 fused ranking (RankEnv pool n 50, w 0.7), "
                   "STRUCT rows never from the fill, duplicates skipped, up to 10",
           "inputs": {"agents": a.agent, "baselines": a.baselines, "router": a.router}}
    _write(a.out, out)
    for name in table:
        print(name, {g: {x: table[name][g][x]["mean"] for x in ("R", "all5", "all10", "all_dyn", "k")
                         if x in table[name][g]} for g in GROUPS})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("accum")
    c.add_argument("--pairs", required=True)
    c.add_argument("--rankings", required=True)
    c.add_argument("--policy", action="append", required=True, help="name=checkpoint (repeatable)")
    c.add_argument("--random-reps", type=int, default=10)
    c.add_argument("--seed", type=int, default=20261004)
    c.add_argument("--k-max", type=int, default=5)
    c.add_argument("--t-plot", type=int, default=T_PLOT)
    c.add_argument("--set", action="append", default=[], help="pilot 9: extra train.yaml keys, e.g. "
                   "struct.articles=true / struct.amend_op=true")
    c.add_argument("--out", required=True)
    r = sub.add_parser("ranker")
    r.add_argument("--ship", required=True)
    r.add_argument("--run", default="runs/p5-rr-sft")
    r.add_argument("--ce-cache", required=True)
    r.add_argument("--k-max", type=int, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--articles", action="store_true", help="Task 6c: the article-mode struct EnvData")
    r.add_argument("--amend-op", action="store_true", help="Task 6d-b: the struct env with struct.amend_op")
    r.add_argument("--top5", action="store_true", help="pilot 9: also the ranker's top-5 list (greedy, STOP masked "
                   "while a slot is valid; K_max must be >= 5) -> context_top5 / recall5_top5 / all5_top5")
    b = sub.add_parser("baselines")
    b.add_argument("--ship", required=True)
    for sp in (r, b):                                  # Task 9f: the final test (m7_final_eval.py spark) only
        sp.add_argument("--split", default="val", choices=["val", "test"])
        sp.add_argument("--rows", help="--split test: the ship's test_eval/test_eval.jsonl")
        sp.add_argument("--rankings", help="--split test: its test_eval/rankings/latest.json")
    b.add_argument("--ce-cache", required=True)
    b.add_argument("--out", required=True)
    p = sub.add_parser("ce-parity")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--ship", required=True)
    p.add_argument("--out", required=True)
    o = sub.add_parser("router")
    o.add_argument("--ship", required=True)
    o.add_argument("--agent", required=True)
    o.add_argument("--ranker", required=True)
    o.add_argument("--C", type=float, default=1.0)
    o.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--ship", required=True)
    t.add_argument("--baselines", required=True)
    t.add_argument("--agent", action="append", required=True, help="name=pq.jsonl[:struct|:ranker]")
    t.add_argument("--router")
    t.add_argument("--articles", action="store_true", help="Task 6c: the article-mode struct EnvData")
    t.add_argument("--old-traj", default="data/m5/pilot/083e116775b3/ship/trajectories.jsonl")
    t.add_argument("--router-struct", default="p8")
    t.add_argument("--router-ranker", default="ranker_k5")
    t.add_argument("--k-from", nargs="*", default=["p8", "ranker_k24"], help="agents whose mean k sets a baseline k")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"tables": cmd_tables, "accum": cmd_accum, "ranker": cmd_ranker, "baselines": cmd_baselines, "ce-parity": cmd_ce_parity, "router": cmd_router}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
