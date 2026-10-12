"""M7 reward-weight calibration (docs/m7/reward-calibration-design.md; grl.train.calibrate_reward). VAL only.
    PYTHONPATH=. python scripts/m7_reward_calib.py candidates [--labels runs/.../labels.jsonl] [--ce C ...]
    GRL_HARD_API=1 PYTHONPATH=. python scripts/m7_reward_calib.py answer [--max-new N]
    GRL_HARD_API=1 PYTHONPATH=. python scripts/m7_reward_calib.py judge [--n 60]
    PYTHONPATH=. python scripts/m7_reward_calib.py fit
candidates: the pilot 7 joint struct ship (read-only), env struct at K_max 5; rows of val only (load_rank_items
refuses test-eval / probe / Q-real). answer / judge: the hard_llm client (cache/hard, its ledger, guard_total = the
900 cap over every pass) with own passes rc_answer / rc_judge (caps 400 / 60); without GRL_HARD_API=1 a cache miss
is an error. The OpenAI key is read from the main checkout's repo-root .env inside the process (never printed)."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402

pin_blas(2, force=True)

import numpy as np  # noqa: E402

from graphrag import config  # noqa: E402
from grl.train import calibrate_reward as cr  # noqa: E402

SHIP = "data/m7/full/b3c78c7712b5/ship"
OUT = "data/m7/calib"
CE_DEFAULT = ("runs/spark/p6-ce2/ce_scores.jsonl",)
QUOTAS = {"old": 60, "hard": 20}
SEED = 20261004
HARD_API_ENV = "GRL_HARD_API"
CAPS = {cr.PASS_ANSWER: 400, cr.PASS_JUDGE: 60}


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(p, rows) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + ".tmp")
    t.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    os.replace(t, p)


def _write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _set_of(row) -> str:
    return "hard" if row.get("origin") == "m7_hard" else "old"


# --- candidates ------------------------------------------------------------------------------------------------------

def cmd_candidates(a) -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from graphrag.retrieval.executor import Budget, take_fitting
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.rerank import load_ce_cache, rerank_candidates
    from grl.model.variants import env_spec, load_variant_data, with_scores
    from grl.train.baselines import topk_context
    from grl.train.data import load_items, load_rank_items, replay_item
    data = load_variant_data("struct")
    cfg = config.load("train")
    ecfg = dict(cfg["env"], K_max=5)
    cls, kw = env_spec("struct")
    env = cls(data, ecfg, None, **kw)
    ship = config.path(a.ship)
    ranks = load_rankings(ship / "rankings" / "latest.json", graph_version=data.version)
    items = {it.qkey: it for it in load_rank_items(ship / "pairs.jsonl", data, ("val",))}
    rows = {r["id"]: r for r in _jsonl(ship / "pairs.jsonl") if r["split"] == "val"}
    if {f"{i}|q" for i in rows} != set(items):
        raise SystemExit("val rows != loaded val items")
    ce = {}
    for p in a.ce:
        for k, v in load_ce_cache(p, data.version).items():
            ce.setdefault(k, v)
    expert, expert_meta = {}, {}
    for p in a.labels or ():
        for it in load_items(p, data, ("val",)):
            if it.qkey not in items:
                continue
            _, s = replay_item(env, ranks, it)
            expert[it.qkey] = [int(c) for c in s.context]
            expert_meta[it.qkey] = {"final_by": it.final_by, "reach": bool(it.reach), "src": str(p)}
    b0 = Budget(0, 0, env.limits.K_max, env.limits.L_max)
    out, notes = [], {"ce_missing": 0, "expert_missing": 0}
    for qk in sorted(items):
        it, r = items[qk], rows[qk[:-2]]
        rk = ranks.get(qk)
        q = with_scores(env.query(rk), rk)
        cand_rows, fused = rerank_candidates(q, cr.CAND_N, cr.FUSE_W, cr.NORM_TOP_N, data.chunk_tokens,
                                             env.limits.L_max)
        ctx = {"dense@3": list(topk_context(env, q, "dense", 3)), "dense@5": list(topk_context(env, q, "dense", 5)),
               "fused@5": [int(c) for c in take_fitting(cand_rows, set(), 5, b0, data.chunk_tokens)]}
        cef = cr.ce_fused_order(cand_rows, fused, ce[qk]) if qk in ce else None
        if cef is not None:
            ctx["ce+fused@5"] = [int(c) for c in take_fitting(cef[0], set(), 5, b0, data.chunk_tokens)]
        else:
            notes["ce_missing"] += 1
        alt_how = None
        if qk in expert:
            ctx["expert"] = expert[qk]
            ctx["expert_alt"], alt_how = cr.trunc_or_overfill(expert[qk], cand_rows.tolist())
        else:
            notes["expert_missing"] += 1
        gold = sorted(it.gold)
        cands = {}
        for name in cr.CANDIDATES:
            if name not in ctx or not ctx[name]:
                continue
            c = [int(x) for x in ctx[name]]
            cands[name] = {"rows": c, "ids": [data.chunk_ids[x] for x in c], "u_ret": cr.u_ret(c, gold),
                           "feats": {k: round(v, 9) for k, v in
                                     cr.context_features(data, q.depth, c, gold, env.limits.T_max).items()}}
        out.append({"id": r["id"], "qkey": qk, "set": _set_of(r), "pattern": r["pattern"], "lang": r["lang"],
                    "docs": r["docs"], "gold_rows": gold, "gold_ids": r["C_ref"], "cands": cands,
                    "expert_meta": expert_meta.get(qk), "expert_alt": alt_how})
    d = config.path(a.out)
    _write_jsonl(d / "candidates.jsonl", out)
    st = {"n": len(out), "by_set": {s: sum(1 for x in out if x["set"] == s) for s in ("old", "hard")},
          "ship": a.ship, "ce": list(a.ce), "labels": list(a.labels or ()), "K_max": env.limits.K_max,
          "graph_version": data.version, **notes,
          "mean_u_ret": {n: round(float(np.mean([x["cands"][n]["u_ret"] for x in out if n in x["cands"]])), 4)
                         for n in cr.CANDIDATES if any(n in x["cands"] for x in out)},
          "n_cands": {n: sum(1 for x in out if n in x["cands"]) for n in cr.CANDIDATES}}
    _write_json(d / "candidates.json", st)
    print(json.dumps(st, ensure_ascii=False))
    return 0


# --- LLM -------------------------------------------------------------------------------------------------------------

def _load_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    from dotenv import dotenv_values
    from scripts.spark import repo_env
    v = dotenv_values(repo_env()).get("OPENAI_API_KEY")
    if not v:
        raise SystemExit("repo-root .env has no OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = v


def calib_llm():
    """The hard_llm client (train.yaml hard_llm: gpt-4.1, cache/hard, ledger, max_calls_total 900) + own passes."""
    from graphrag.kg.llm import KgLLM
    h = dict(config.load("train")["hard_llm"])
    h["passes"] = {**h["passes"], **{p: {"prompt_version": 1, "max_calls": n} for p, n in CAPS.items()}}
    allow = os.environ.get(HARD_API_ENV) == "1"
    if allow:
        _load_openai_key()
    return KgLLM(cfg={"llm": h}, cache_root=h["cache_dir"], allow_api=allow)


def _texts(ids: set) -> dict:
    from graphrag.data.ingest import read_records
    from grl.env.struct_data import build_struct_inputs
    st = build_struct_inputs()[0]
    out = {i: st[i] for i in ids if i in st}
    for r in read_records(config.path(config.load("data")["records_dir"])):
        for c in r.chunks:
            if c.chunk_id in ids:
                out[c.chunk_id] = c.text
    miss = sorted(ids - set(out))
    if miss:
        raise SystemExit(f"fragments without text: {miss[:3]}")
    return out


def _ledger_total(llm) -> int:
    return sum(int(v) for p in llm.ledger().values() for v in p.values())


def cmd_answer(a) -> int:
    d = config.path(a.out)
    cands = _jsonl(d / "candidates.jsonl")
    rows = {r["id"]: r for r in _jsonl(config.path(a.ship) / "pairs.jsonl") if r["split"] == "val"}
    sub = cr.subsample(cands, QUOTAS, SEED)
    names = [n for n in cr.ANSWER_CANDIDATES if n in set(a.names or cr.ANSWER_CANDIDATES)]
    todo = [(x, n) for x in cands if x["id"] in sub for n in names if n in x["cands"]]
    texts = _texts({i for x, n in todo for i in x["cands"][n]["ids"]})
    llm = calib_llm()
    led0 = _ledger_total(llm)
    out, new = [], 0
    prev = {(r["id"], r["cand"]): r for r in _jsonl(d / "answers.jsonl")} if (d / "answers.jsonl").exists() else {}
    for x, n in todo:
        ids = sorted(x["cands"][n]["ids"])                    # canonical order: the same set -> the same cached call
        inp = cr.answer_input(rows[x["id"]]["q"], [texts[i] for i in ids])
        before = llm.total_api_calls()
        if a.max_new is not None and new >= a.max_new:
            if (x["id"], n) in prev:
                out.append(prev[(x["id"], n)])
            continue
        ans = cr.ask(llm, cr.PASS_ANSWER, inp, cr.ANSWER_PROMPT, cr.ANSWER_SCHEMA)
        new += llm.total_api_calls() - before
        out.append({"id": x["id"], "cand": n, "set": x["set"], "pattern": x["pattern"], "ids": ids, "answer": ans,
                    "u_ans_f1": round(cr.u_ans_f1(rows[x["id"]], ans), 6)})
    _write_jsonl(d / "answers.jsonl", sorted(out, key=lambda r: (r["id"], r["cand"])))
    st = {"subsample": len(sub), "answers": len(out), "api_calls": llm.total_api_calls(), "ledger_before": led0,
          "ledger_after": _ledger_total(llm), "stats": llm.stats_json(),
          "mean_u_ans_f1": {n: round(float(np.mean([r["u_ans_f1"] for r in out if r["cand"] == n])), 4)
                            for n in names if any(r["cand"] == n for r in out)}}
    _write_json(d / "answers.json", st)
    print(json.dumps(st, ensure_ascii=False))
    return 0


def cmd_judge(a) -> int:
    d = config.path(a.out)
    ans = _jsonl(d / "answers.jsonl")
    rows = {r["id"]: r for r in _jsonl(config.path(a.ship) / "pairs.jsonl") if r["split"] == "val"}
    pick = sorted(ans, key=lambda r: cr._u(SEED, f"judge|{r['id']}|{r['cand']}"))[:a.n]
    llm = calib_llm()
    out = []
    for r in pick:
        row = rows[r["id"]]
        a_ = r["answer"] or {}
        inp = {"question": row["q"], "reference": cr.reference_text(row),
               "answer": a_.get("answer", "") if a_.get("answerable") else "(no answer: the context lacks it)"}
        v = cr.ask(llm, cr.PASS_JUDGE, inp, cr.JUDGE_PROMPT, cr.JUDGE_SCHEMA)
        out.append({"id": r["id"], "cand": r["cand"], "u_ans_f1": r["u_ans_f1"],
                    "judge": None if v is None else v["score"] / 2, "why": None if v is None else v["why"]})
    _write_jsonl(d / "judge.jsonl", out)
    ok = [x for x in out if x["judge"] is not None]
    rho = cr.spearman([x["u_ans_f1"] for x in ok], [x["judge"] for x in ok])
    st = {"n": len(out), "judged": len(ok), "spearman": round(rho, 4), "use_judge": bool(rho < cr.SPEARMAN_MIN),
          "api_calls": llm.total_api_calls(), "ledger_after": _ledger_total(llm), "stats": llm.stats_json()}
    _write_json(d / "judge.json", st)
    print(json.dumps(st, ensure_ascii=False))
    return 0


# --- fit -------------------------------------------------------------------------------------------------------------

def _groups(cands, names_of, util, beta):
    keys = cr.feature_names(beta)
    out = []
    for x in cands:
        ns = names_of(x)
        if len(ns) < 2:
            continue
        F = np.array([[x["cands"][n]["feats"][k] for k in keys] for n in ns])
        U = np.array([util(x, n) for n in ns])
        out.append((x["id"], F, U, ns))
    return out


def _w_vec(weights: dict, beta: float) -> np.ndarray:
    keys = cr.feature_names(beta)
    bad = set(weights) - set(keys)
    if bad:
        raise ValueError(f"weights {bad} not in features of β={beta}")
    return np.array([weights.get(k, 0.0) for k in keys])


def evaluate(cands, names_of, util, folds, k: int = 5) -> dict:
    """Held-out τ per question for: current weights, fitted (β chosen on the training folds by τ), F_β alone, recall."""
    per = {}                                                    # model -> {qid: τ}
    chosen = []
    for f in range(k):
        test_ids = {q for q, g in folds.items() if g == f}
        best = None
        for beta in cr.BETAS:
            G = _groups(cands, names_of, util, beta)
            tr = [(F, U) for q, F, U, _ in G if q not in test_ids]
            w = cr.fit_simplex(tr)
            t = cr.mean_tau(tr, w)[0]
            if best is None or t > best[0] + 1e-12:
                best = (t, beta, w)
        _, beta, w = best
        chosen.append({"fold": f, "beta": beta, "w": dict(zip(cr.feature_names(beta), np.round(w, 4).tolist())),
                       "n_test": len(test_ids)})
        models = {"fitted": (beta, w), "current": (1.0, _w_vec(cr.CURRENT, 1.0)),
                  "recall": (1.0, _w_vec({"recall": 1.0}, 1.0))}
        for b in cr.BETAS:
            models[f"{cr.BETA_KEYS[b]}_only"] = (b, _w_vec({cr.BETA_KEYS[b]: 1.0}, b))
        for name, (b, wv) in models.items():
            for q, F, U, _ in _groups(cands, names_of, util, b):
                if q in test_ids:
                    per.setdefault(name, {})[q] = cr.kendall_tau(F @ wv, U)
    qids = sorted(per["current"])
    summary = {}
    for name, d in per.items():
        v = np.array([d[q] for q in qids])
        ok = v[~np.isnan(v)]
        summary[name] = {"tau": round(float(ok.mean()), 4), "se": round(float(ok.std(ddof=1) / math.sqrt(len(ok))), 4),
                         "n": int(len(ok))}
        if name != "current":
            ad = cr.adopt(v, np.array([per["current"][q] for q in qids]))
            summary[name]["vs_current"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in ad.items()}
    return {"heldout": summary, "folds": chosen}


def final_fit(cands, names_of, util) -> dict:
    best = None
    out = {}
    for beta in cr.BETAS:
        G = [(F, U) for _, F, U, _ in _groups(cands, names_of, util, beta)]
        w = cr.fit_simplex(G)
        t, se, _ = cr.mean_tau(G, w)
        out[cr.BETA_KEYS[beta]] = {"tau_train": round(t, 4), "loss": round(cr.pair_loss(G, w), 5),
                                   "w": dict(zip(cr.feature_names(beta), np.round(w, 4).tolist()))}
        if best is None or t > best[0] + 1e-12:
            best = (t, beta, w)
    return {"by_beta": out, "beta": best[1], "weights": dict(zip(cr.feature_names(best[1]),
                                                                 np.round(best[2], 4).tolist())),
            "tau_train": round(best[0], 4)}


def cmd_fit(a) -> int:
    d = config.path(a.out)
    cands = _jsonl(d / "candidates.jsonl")
    folds = cr.doc_folds([{"id": x["id"], "docs": x["docs"]} for x in cands], 5, SEED)
    rep = {"folds": {f: sum(1 for v in folds.values() if v == f) for f in range(5)}}
    # (B) U_ret on every val question, all candidates present
    uret = lambda x, n: x["cands"][n]["u_ret"]                  # noqa: E731
    all_names = lambda x: [n for n in cr.CANDIDATES if n in x["cands"]]      # noqa: E731
    rep["u_ret"] = {"n_q": len(cands), **evaluate(cands, all_names, uret, folds), "final": final_fit(cands, all_names,
                                                                                                     uret)}
    # (A) full U = 0.5 U_ret + 0.5 U_ans on the answered subsample
    if (d / "answers.jsonl").exists():
        ans = {(r["id"], r["cand"]): r for r in _jsonl(d / "answers.jsonl")}
        use_judge = False
        if (d / "judge.json").exists():
            use_judge = json.loads((d / "judge.json").read_text(encoding="utf-8"))["use_judge"]
        if use_judge:
            raise SystemExit("judge disagrees with token-F1 (Spearman < 0.6): U_ans needs judge scores for all answers")
        ufull = lambda x, n: 0.5 * x["cands"][n]["u_ret"] + 0.5 * ans[(x["id"], n)]["u_ans_f1"]   # noqa: E731
        sub_names = lambda x: [n for n in cr.ANSWER_CANDIDATES if n in x["cands"] and (x["id"], n) in ans]  # noqa
        sub = [x for x in cands if len(sub_names(x)) >= 2]
        rep["u_full"] = {"n_q": len(sub), "cands_per_q": round(float(np.mean([len(sub_names(x)) for x in sub])), 3),
                         **evaluate(sub, sub_names, ufull, folds), "final": final_fit(sub, sub_names, ufull)}
        # П5: quantile of the new R over contexts with U >= 0.5 on old-set (M3 train-like) questions
        fin = rep["u_full"]["final"]
        wv = _w_vec(fin["weights"], fin["beta"])
        keys = cr.feature_names(fin["beta"])
        R, Rc, U = [], [], []
        for x in sub:
            if x["set"] != "old":
                continue
            for n in sub_names(x):
                f = x["cands"][n]["feats"]
                R.append(float(np.array([f[k] for k in keys]) @ wv))
                Rc.append(cr.score(f, cr.CURRENT))
                U.append(ufull(x, n))
        rep["p5"] = {"new": {str(q): cr.p5_threshold(R, U, q) for q in (0.1, 0.2, 0.25)},
                     "current_at_0.55": {"good_pass": float(np.mean([r >= 0.55 for r, u in zip(Rc, U) if u >= 0.5])),
                                         "bad_pass": float(np.mean([r >= 0.55 for r, u in zip(Rc, U) if u < 0.5]))},
                     "current_q": {str(q): cr.p5_threshold(Rc, U, q) for q in (0.1, 0.2, 0.25)}, "n_ctx": len(R)}
        rep["u_ans_by_cand"] = {n: round(float(np.mean([r["u_ans_f1"] for k, r in ans.items() if k[1] == n])), 4)
                                for n in cr.ANSWER_CANDIDATES if any(k[1] == n for k in ans)}
    rep["u_ret_by_cand"] = {n: round(float(np.mean([x["cands"][n]["u_ret"] for x in cands if n in x["cands"]])), 4)
                            for n in cr.CANDIDATES if any(n in x["cands"] for x in cands)}
    _write_json(d / "fit.json", rep)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("candidates", "answer", "judge", "fit"):
        s = sub.add_parser(name)
        s.add_argument("--ship", default=SHIP)
        s.add_argument("--out", default=OUT)
        if name == "candidates":
            s.add_argument("--labels", nargs="*")
            s.add_argument("--ce", nargs="*", default=list(CE_DEFAULT))
        if name == "answer":
            s.add_argument("--max-new", type=int)
            s.add_argument("--names", nargs="*")
        if name == "judge":
            s.add_argument("--n", type=int, default=60)
    a = ap.parse_args(argv)
    return {"candidates": cmd_candidates, "answer": cmd_answer, "judge": cmd_judge, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
