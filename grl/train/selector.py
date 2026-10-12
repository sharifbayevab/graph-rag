"""Non-neural search selector (M5 pre-RL baseline): from per-query features available WITHOUT C* pick ONE action of
the variant env, execute it from reset, then STOP. Answers "is a learned per-query choice of the search channel /
threshold worth anything over the best fixed rule?" before RL.

Features (query_features; FEATURE_NAMES): read only the QueryCtx -- the U9 rankings and their scores, the fused
ranking, V0 and the query text. No gold argument exists anywhere on the feature path (guard-tested).
- per channel bm25 (U9 BM25), dense (cosine), cc (grl.model.variants.fused_scores, the env's fuse_w): raw top-1,
  top-2, gap; min-max normalised top-2 (minmax_scores, norm_top_n); fragments with normalised score ≥ 0.9 / ≥ 0.7;
- agreement: top-1 equal (bm25 / dense, dense / cc, bm25 / cc), overlap@3 / @10 of the BM25 and dense rankings;
- query: characters, words, Cyrillic share of letters, Uzbek-Cyrillic letters (ў қ ғ ҳ), Uzbek-Latin marks (oʻ gʻ),
  |V0|, |BM25 list|.
Labels (train only): "expert" = the variant expert's first action (grl.train.expert; meta-checked), "switch" = the
oracle_switch choice (the better of the first vector slot and the first lexical slot, ties -> vector, as the
baselines), "best1" = the best single action then STOP (ties -> lower id). Labels and evaluation read C*; features
never do.
Models (scikit-learn, pinned in requirements.lock; deterministic): "logreg" (standardised features, multinomial
L2 logistic regression, lbfgs) with C chosen by 5-fold CV on train (KFold shuffle seed 0) by the mean reward of the
executed choice; "tree" (DecisionTreeClassifier, random_state 0, min_samples_leaf 5) with max_depth chosen the same
way. A prediction is the most probable class whose action is valid at reset; none valid -> STOP."""
from __future__ import annotations

from collections import Counter

import numpy as np

from grl.model.variants import NORM_TOP_N, fused_scores, minmax_scores, search_scores, with_scores
from grl.train.data import guard_qkeys
from grl.train.evaluate import context_metrics

CHANNELS = ("bm25", "dense", "cc")
CH_FEATS = ("top1", "top2", "gap", "top2_norm", "n_ge_0.9", "n_ge_0.7")
FEATURE_NAMES = ([f"{c}_{f}" for c in CHANNELS for f in CH_FEATS]
                 + ["top1_eq_bm25_dense", "top1_eq_dense_cc", "top1_eq_bm25_cc", "overlap@3", "overlap@10",
                    "q_chars", "q_words", "q_cyr_share", "q_uz_cyr", "q_uz_lat", "v0_size", "lex_size"])
UZ_CYR = set("ўқғҳЎҚҒҲ")
UZ_LAT = ("oʻ", "gʻ", "o'", "g'", "oʼ", "gʼ", "o‘", "g‘")
LABELS = ("expert", "switch", "best1")
MODELS = {"logreg": (0.01, 0.1, 1.0, 10.0), "tree": (1, 2, 3, 4)}
CV_FOLDS = 5


def _channel(rows, scores, top_n: int) -> tuple[np.ndarray, list[float]]:
    rows = np.asarray(rows, np.int64)
    sc = np.asarray(scores, np.float64)
    if not sc.size:
        return rows, [0.0] * len(CH_FEATS)
    nm = minmax_scores(sc, top_n)
    t1 = float(sc[0])
    t2 = float(sc[1]) if sc.size > 1 else t1
    head = nm[:top_n]
    return rows, [t1, t2, t1 - t2, float(nm[1]) if nm.size > 1 else 1.0, float((head >= 0.9).sum()),
                  float((head >= 0.7).sum())]


def query_features(env, q) -> np.ndarray:
    """FEATURE_NAMES for one query; gold-free by construction (only q: rankings, scores, V0, text)."""
    top_n = int(getattr(env, "norm_top_n", NORM_TOP_N))
    w = float(getattr(env, "fuse_w", 0.7))
    lex, f_b = _channel(q.lex, search_scores(q, "lexical"), top_n)
    vec, f_d = _channel(q.vec, search_scores(q, "vector"), top_n)
    rows, fsc = fused_scores(q, w, top_n)
    cc, f_c = _channel(rows, fsc, top_n)

    def top1(r):
        return int(r[0]) if len(r) else -1

    def overlap(k):
        return len(set(lex[:k].tolist()) & set(vec[:k].tolist())) / k
    text = str(q.text or "")
    letters = [ch for ch in text if ch.isalpha()]
    cyr = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    low = text.lower()
    extra = [float(top1(lex) == top1(vec)), float(top1(vec) == top1(cc)), float(top1(lex) == top1(cc)),
             overlap(3), overlap(10), float(len(text)), float(len(text.split())),
             cyr / len(letters) if letters else 0.0, float(any(ch in UZ_CYR for ch in text)),
             float(any(m in low for m in UZ_LAT)), float(len(q.v0)), float(len(q.lex))]
    out = np.array(f_b + f_d + f_c + extra, np.float64)
    return np.where(np.isfinite(out), out, 0.0)


def features(env, ranks, items) -> tuple[np.ndarray, list[str]]:
    """Feature matrix of `items` -- reads only it.qkey (never it.gold)."""
    keys = [it.qkey for it in items]
    guard_qkeys(keys)
    rows = []
    for k in keys:
        r = ranks.get(k)
        rows.append(query_features(env, with_scores(env.query(r), r)))
    return np.stack(rows), keys


def action_table(env, ranks, items, w) -> list[list[dict | None]]:
    """[query][action] = context_metrics of `action` from reset then STOP (None when masked at reset; STOP = the
    empty context). READS C* (evaluation / labels only)."""
    out = []
    for it in items:
        q = env.query(ranks.get(it.qkey))
        s0 = env.reset(q)
        m = env.mask(q, s0)
        row = []
        for a in range(len(env.actions)):
            if not m[a]:
                row.append(None)
                continue
            ctx = () if a == 0 else env.step(q, s0, a).context
            row.append(context_metrics(env.data, q, ctx, it.gold, w, env.limits.T_max))
        out.append(row)
    return out


def slot_name(env, act) -> str:
    """A search slot under the variant's own semantics, named like the baselines: fused_adaptive lexical =
    cc<w>@<p>, adaptive lexical = bm25@<p>, adaptive vector = dense@<p> (threshold p, grl.model.variants.ADAPTIVE_P);
    fixed-k = bm25@k<k> / dense@k<k>. Graph / STOP actions: op[:r:m]."""
    if act.op not in ("lexical", "vector"):
        return f"{act.op}:{act.r}:{act.m}" if act.op == "graph" else act.op
    p_of = getattr(env, "p_of", None)
    if p_of is None:
        return f"{'bm25' if act.op == 'lexical' else 'dense'}@k{act.k}"
    p = p_of[act.k]
    if act.op == "vector":
        return f"dense@{p:g}"
    fw = getattr(env, "fuse_w", None)
    return f"cc{fw:g}@{p:g}" if fw is not None else f"bm25@{p:g}"


def first_slot(env, op: str) -> int:
    return next(i for i, a in enumerate(env.actions) if a.op == op)


def make_labels(kind: str, env, items, table) -> np.ndarray:
    if kind == "expert":
        if any(getattr(it, "variant", None) is None for it in items):
            raise ValueError("label 'expert' needs variant-expert trajectories (train.py expert)")
        return np.array([int(it.actions[0]) for it in items], np.int64)
    if kind == "switch":
        lex, vec = first_slot(env, "lexical"), first_slot(env, "vector")
        y = []
        for row in table:
            rl, rv = row[lex], row[vec]
            y.append(vec if rv is not None and (rl is None or rv["reward"] >= rl["reward"]) else
                     lex if rl is not None else 0)
        return np.array(y, np.int64)
    if kind == "best1":
        return np.array([max((a for a, r in enumerate(row) if r is not None), key=lambda a: (row[a]["reward"], -a))
                         for row in table], np.int64)
    raise ValueError(f"label {kind!r} is not one of {LABELS}")


class Selector:
    def __init__(self, model: str, param):
        if model not in MODELS:
            raise ValueError(f"selector model {model!r} is not one of {sorted(MODELS)}")
        self.model, self.param = model, param
        self.classes, self.est, self.const = None, None, None

    def fit(self, X, y):
        y = np.asarray(y, np.int64)
        self.classes = np.array(sorted(set(y.tolist())), np.int64)
        prior = Counter(y.tolist())
        self.order = sorted(self.classes.tolist(), key=lambda c: (-prior[c], c))      # fallback: majority first
        if len(self.classes) == 1:
            self.const = int(self.classes[0])
            return self
        if self.model == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.est = make_pipeline(StandardScaler(), LogisticRegression(C=float(self.param), max_iter=5000))
        else:
            from sklearn.tree import DecisionTreeClassifier
            self.est = DecisionTreeClassifier(max_depth=int(self.param), min_samples_leaf=5, random_state=0)
        self.est.fit(np.asarray(X, np.float64), y)
        return self

    def proba(self, X) -> np.ndarray:
        if self.const is not None:
            return np.ones((len(X), 1))
        return self.est.predict_proba(np.asarray(X, np.float64))

    def predict(self, X, valid) -> np.ndarray:
        """valid[i][a]: action a valid at reset of query i. Most probable valid class (ties -> majority order)."""
        P = self.proba(X)
        cls = [self.const] if self.const is not None else self.est.classes_.tolist()
        rank = {c: i for i, c in enumerate(self.order)}
        out = []
        for p, v in zip(P, valid):
            cand = [(float(pp), -rank[c], c) for pp, c in zip(p, cls) if v[c]]
            out.append(max(cand)[2] if cand else 0)
        return np.array(out, np.int64)


def summarize_choice(table, choice) -> dict:
    rows = [table[i][int(a)] for i, a in enumerate(choice)]
    keys = ("reward", "recall", "precision", "n_ctx", "length")
    out = {f"{k}_mean": round(float(np.mean([r[k] for r in rows])), 6) for k in keys} if rows else {}
    out["n"] = len(rows)
    out["choice_hist"] = {str(k): v for k, v in sorted(Counter(int(a) for a in choice).items())}
    return out


def _valid(table) -> list[list[bool]]:
    return [[r is not None for r in row] for row in table]


def cv_param(model: str, X, y, table, folds: int = CV_FOLDS) -> tuple:
    """The param of MODELS[model] with the best mean CV reward on train (ties -> the first = most regularised)."""
    from sklearn.model_selection import KFold
    n = len(y)
    if n < 2:                                             # nothing to cross-validate: the most regularised
        return MODELS[model][0], {}
    k = min(int(folds), n)
    scores = {}
    for param in MODELS[model]:
        got = []
        for tr, te in KFold(n_splits=k, shuffle=True, random_state=0).split(np.arange(n)):
            sel = Selector(model, param).fit(X[tr], y[tr])
            ch = sel.predict(X[te], [_valid(table)[i] for i in te])
            got += [table[i][int(a)]["reward"] for i, a in zip(te, ch)]
        scores[str(param)] = round(float(np.mean(got)), 6)
    best = max(MODELS[model], key=lambda p: (scores[str(p)], -MODELS[model].index(p)))
    return best, scores


def reference_rows(env, table, items, ranks=None) -> dict:
    """Fixed single actions (first lexical / vector slot, every slot), and the C*-reading oracle rows."""
    out = {}
    for a, act in enumerate(env.actions):
        if a and act.op in ("lexical", "vector"):
            ch = [a if row[a] is not None else 0 for row in table]
            out[f"always:{slot_name(env, act)}"] = summarize_choice(table, ch)
    sw = make_labels("switch", env, items, table)
    out["oracle:switch"] = {**summarize_choice(table, sw), "oracle": True}
    out["oracle:best1"] = {**summarize_choice(table, make_labels("best1", env, items, table)), "oracle": True}
    return out


def expert_rows(env, ranks, items, w) -> dict:
    """The variant expert's full trajectory on these items (replayed in env: an upper bound of the depth-limited
    policy; reads C*)."""
    from grl.train.data import replay_item
    rows = []
    for it in items:
        q, s = replay_item(env, ranks, it)
        rows.append(context_metrics(env.data, q, s.context, it.gold, w, env.limits.T_max))
    keys = ("reward", "recall", "precision", "n_ctx", "length")
    return {**{f"{k}_mean": round(float(np.mean([r[k] for r in rows])), 6) for k in keys}, "n": len(rows),
            "oracle": True}


def run_selector(env, ranks, train, val, w, label: str = "expert", model: str = "logreg") -> dict:
    if label not in LABELS:
        raise ValueError(f"label {label!r} is not one of {LABELS}")
    if getattr(env, "required", ()):
        raise ValueError("selector: one action then STOP is not valid under a `require` variant")
    Xtr, _ = features(env, ranks, train)                 # features first: no gold read on this path
    Xva, _ = features(env, ranks, val)
    ttr, tva = action_table(env, ranks, train, w), action_table(env, ranks, val, w)
    ytr, yva = make_labels(label, env, train, ttr), make_labels(label, env, val, tva)
    param, cv = cv_param(model, Xtr, ytr, ttr)
    sel = Selector(model, param).fit(Xtr, ytr)
    out = {"label": label, "model": model, "param": param, "cv_reward": cv, "features": list(FEATURE_NAMES),
           "classes": sel.classes.tolist(),
           "action_names": {str(i): slot_name(env, a) for i, a in enumerate(env.actions)}}
    for split, X, y, t, items in (("train", Xtr, ytr, ttr, train), ("val", Xva, yva, tva, val)):
        ch = sel.predict(X, _valid(t))
        out[split] = {**summarize_choice(t, ch), "label_acc": round(float(np.mean(ch == y)), 6),
                      "label_oracle": {**summarize_choice(t, [a if t[i][a] is not None else 0
                                                              for i, a in enumerate(y)]), "oracle": True},
                      "references": reference_rows(env, t, items)}
        if all(getattr(it, "variant", None) is not None for it in items):
            out[split]["references"]["oracle:expert"] = expert_rows(env, ranks, items, w)
    if model == "tree" and sel.est is not None:
        imp = sel.est.feature_importances_
        out["importances"] = {FEATURE_NAMES[i]: round(float(imp[i]), 6) for i in np.argsort(-imp)[:10] if imp[i] > 0}
    elif model == "logreg" and sel.est is not None:
        coef = np.abs(sel.est[-1].coef_).sum(0)
        out["importances"] = {FEATURE_NAMES[i]: round(float(coef[i]), 6) for i in np.argsort(-coef)[:10]}
    return out
