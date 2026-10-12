"""RELATED research code of M2 Task 8 (Rulings R21-R23), moved out of the library module graphrag.kg.semantic
(Ruling R31 (2)): the graph build keeps only the adopted scoring (semantic.latent_edges at graph.yaml
related.theta_sim, graphrag.kg.build.related_edges); the dev-set statistics, the θ schedule and the RELATED judge
used to choose that θ live here and are used by scripts/build_related.py.

    auc / bootstrap_auc / logistic_fit / logistic_p / dev_theta    dev-set scoring (R22)
    class_removals / adopt_theta                                   boilerplate-class ablation, θ_a / θ_b (R23)
    JUDGE_PROMPT / judge_schema / verify_verdicts / sample_strata / make_batches / batch_input / llm_judge /
    precision                                                      the cached "audit_related" judge (R21)

The judge prompt, schema and batch input are byte-identical to the Task 8 ones (graphrag.kg.judge builds
them), so every cached answer in cache/kg/audit_related stays a hit."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.kg import judge as J  # noqa: E402
from graphrag.kg import semantic as S  # noqa: E402
from graphrag.kg.schema import GraphEdge  # noqa: E402


# --- dev-set scoring (R22) ---------------------------------------------------------------------------------

def auc(scores, labels) -> float | None:
    """ROC AUC (Mann-Whitney, ties count 1/2); -inf scores rank last."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return None
    g = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
    return float(g / (len(p) * len(n)))


def bootstrap_auc(scores, labels, B: int = 2000, seed: int = 0) -> list[float] | None:
    """95% percentile bootstrap interval of auc (resampling pairs; resamples with one class skipped)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        k = rng.integers(0, len(y), len(y))
        a = auc(s[k], y[k])
        if a is not None:
            out.append(a)
    return [round(float(x), 6) for x in np.quantile(out, [0.025, 0.975])] if out else None


def logistic_fit(s, y, iters: int = 100, ridge: float = 1e-6) -> tuple[float, float]:
    """P(same_topic | score) = sigmoid(a + b·score), Newton-Raphson (deterministic, tiny ridge)."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    A = np.stack([np.ones_like(s), s], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A @ w)))
        H = A.T @ (A * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        w = w - np.linalg.solve(H, A.T @ (p - t))
    return float(w[0]), float(w[1])


def logistic_p(w: tuple[float, float], s) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(w[0] + w[1] * np.asarray(s, dtype=np.float64))))


def dev_theta(near_weights: Callable[[float], np.ndarray], w: tuple[float, float], grid: list[float],
              gate: float, lo: float | None = None) -> dict:
    """Ruling R22: θ_dev = the smallest grid θ (>= lo) at which the precision of the near-θ edges estimated
    from the judged dev pairs -- the mean of logistic_p(w, weight) over the edges with θ < weight <= θ + band
    (near_weights(θ)) -- reaches `gate`. The fresh-pair judge run then confirms or rejects it."""
    table, th = [], None
    for t in grid:
        if lo is not None and t < lo - 1e-12:
            continue
        ws = near_weights(t)
        est = float(logistic_p(w, ws).mean()) if len(ws) else None
        table.append({"theta": t, "near_edges": int(len(ws)), "est_near_precision": None if est is None else round(est, 4)})
        if th is None and est is not None and est >= gate:
            th = t
    return {"theta_dev": th, "table": table}


# --- R23: class ablation, the θ_a / θ_b schedule ---------------------------------------------------------

def class_removals(reason_fn: Callable[[dict], dict], rules_spec: dict, classes: list[str], n_examples: int = 3) -> dict:
    """Ruling R23: for each boilerplate class in `classes`, the chunks it removes on its own -- eligible under
    `rules_spec` without all of `classes`, excluded once that one class is added (reason_fn(spec) ->
    {chunk_id: reason | None}). Examples: first / middle / last removed chunk id (sorted)."""
    base = {k: v for k, v in rules_spec.items() if k not in classes}
    r0 = reason_fn(base)
    out = {}
    for name in classes:
        r1 = reason_fn({**base, name: rules_spec[name]})
        ids = sorted(i for i, r in r1.items() if r is not None and r0.get(i) is None)
        pick = sorted({0, len(ids) // 2, len(ids) - 1}) if ids else []
        out[name] = {"removed": len(ids), "examples": [ids[k] for k in pick][:n_examples]}
    return out


def adopt_theta(rounds: list[dict], gate: float) -> tuple[float, str]:
    """Ruling R23: rounds = judged near-θ rounds in schedule order [{theta, near: {precision}}] (θ_a first).
    The first θ whose near precision reaches `gate` is adopted ("passed_theta_a" / "passed_theta_b");
    otherwise the judged θ with the best near precision, ties to the higher θ ("failed_best_judged")."""
    for k, r in enumerate(rounds):
        p = (r.get("near") or {}).get("precision")
        if p is not None and p >= gate:
            return r["theta"], "passed_theta_a" if k == 0 else "passed_theta_b"
    best = max(rounds, key=lambda r: (((r.get("near") or {}).get("precision")) or -1.0, r["theta"]))
    return best["theta"], "failed_best_judged"


# --- judge --------------------------------------------------------------------------------------------

JUDGE_PROMPT = """You audit latent "same topic" (RELATED) edges of a legal knowledge graph built from Uzbek (Cyrillic and Latin script) and Russian parliamentary documents (laws, Senate resolutions, committee conclusions, session stenograms, reports). Each edge joins two passages from DIFFERENT documents because their embeddings are similar. You receive a batch of edges; each item gives, for both ends, the node path (document section / title / structural path: статья = article, часть = part, пункт = point, блок = block, вопрос = agenda item, выступление = speech) and the passage text (possibly truncated).

For EACH edge decide same_topic: true only if both passages are substantively about the same specific subject matter — the same legal institution, procedure, regulated activity, body's competence, policy problem or the same act / provision (for example both deal with state social insurance benefits, or both with licensing of pharmacies, or one is a law norm and the other a speech or conclusion discussing that same norm or issue). A shared general area is enough only when the passages really discuss the same issue within it.

same_topic is false when the similarity comes only from: shared genre or boilerplate (procedural chair remarks, greetings, thanks, voting formulas, adoption / signature / entry-into-force formulas, gazette citations, generic "the law was approved" statements); the same document type or the same session; generic legal vocabulary (rights, obligations, bodies, responsibility) about different subjects; or different subjects that merely both mention the parliament, the Senate or a ministry.

Judge only from what is shown. Write a short reason first (name each passage's subject), then same_topic (true/false). Return one verdict per item id, every id exactly once."""


def judge_schema() -> dict:
    return J.verdict_schema({"reason": {"type": "string"}, "same_topic": {"type": "boolean"}})


def _verdict_ok(v: dict) -> bool:
    return isinstance(v.get("same_topic"), bool) and isinstance(v.get("reason"), str)


def verify_verdicts(ids: list[str]) -> Callable[[Any], tuple[bool, str | None]]:
    return J.verdict_checker(ids, _verdict_ok, exact_count=True)


_cap = J.cap_tail


def sample_strata(edges: list[GraphEdge], theta: float, cfg: dict, seed: int, tag: str = "",
                  exclude: set | frozenset = frozenset()) -> dict:
    """near: `near` random edges with θ < weight <= θ + near_band; top: `top` random edges of the top
    `top_share` (at least `top`) edges by weight. Deterministic: sorted by (src, dst), random.Random(seed:tag).
    Vertex pairs in `exclude` (R22: the judged dev pairs) are never sampled (they stay in the top ranking)."""
    j = {**S.DEFAULTS["judge"], **(cfg or {})}
    es = sorted(edges, key=lambda e: (e.src, e.dst))
    by_w = sorted(es, key=lambda e: (-e.attrs["weight"], e.src, e.dst))
    top_ids = {(e.src, e.dst) for e in by_w[:max(j["top"], int(math.ceil(j["top_share"] * len(es))))]}
    near_pool = [e for e in es if e.attrs["weight"] <= theta + j["near_band"] + 1e-12 and (e.src, e.dst) not in exclude]
    top_pool = [e for e in es if (e.src, e.dst) in top_ids and (e.src, e.dst) not in exclude]
    near = random.Random(f"{seed}:near:{tag}").sample(near_pool, min(j["near"], len(near_pool)))
    taken = {(e.src, e.dst) for e in near}
    top_pool = [e for e in top_pool if (e.src, e.dst) not in taken]
    top = random.Random(f"{seed}:top:{tag}").sample(top_pool, min(j["top"], len(top_pool)))
    out = {"near": near, "top": top, "near_pool": len(near_pool), "top_pool": len(top_pool),
           "top_min_weight": min((e.attrs["weight"] for e in top_pool), default=None)}
    if exclude:
        out["excluded_dev_pairs"] = sum(1 for e in es if (e.src, e.dst) in exclude)
    return out


def make_batches(items: list[dict], size: int, seed, tag: str = "") -> list[list[dict]]:
    """All items (strata mixed) shuffled with random.Random(seed:batch:tag) and cut into `size` chunks."""
    return J.shuffled_batches(items, size, f"{seed}:batch:{tag}")


batch_input = J.batch_input


def llm_judge(llm, pass_name: str = "audit_related") -> Callable[[list[dict]], dict]:
    """judge(batch) -> {uid: {same_topic, reason} | None} via KgLLM.cached_json (cache, gate, budget, ledger)."""
    return J.make_judge(llm, pass_name, JUDGE_PROMPT, judge_schema(), verify_verdicts,
                        lambda v: {"same_topic": v["same_topic"], "reason": v["reason"]})


def precision(verdicts: list[dict | None]) -> dict:
    from graphrag.kg.audit import wilson
    got = [v for v in verdicts if v is not None]
    k = sum(1 for v in got if v["same_topic"])
    n = len(got)
    return {"n": n, "k": k, "unjudged": len(verdicts) - n, "precision": (k / n) if n else None,
            "ci": list(wilson(k, n)) if n else None}
