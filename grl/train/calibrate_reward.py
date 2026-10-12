"""M7 reward-weight calibration (docs/m7/reward-calibration-design.md; user decision: target recall@5 / all@5 +
gpt-4.1). VAL only (old val-182 + M7 val Q1 / Q2), never test, never Q-real.

Target utility of a (question q, context C):  U = 0.5·U_ret + 0.5·U_ans  (mix fixed in advance)
* U_ret = 0.5·recall@5 + 0.5·all@5 against C*, C cut to its first 5 fragments in addition order;
* U_ans = gpt-4.1 answers q from C only (one call, no self-assessment); the answer vs the reference answer by token-F1
  after script normalisation (Cyrillic -> Latin, apostrophes dropped, NFKC). Number-shaped answers (count / version
  questions) use the F1 of the number multisets (numbers of the question that are not in the reference are ignored,
  month names count as numbers). A gpt-4.1 judge on a random subsample checks it: Spearman(token-F1, judge) < 0.6 ->
  U_ans is the judge's score instead (design §Цель).

Candidate contexts per question: dense@3, dense@5, fused@5 (rerank candidate list, fuse_w 0.7), ce+fused@5
(a = CE_A over the same candidates), the struct expert final (pilot 7 labels) and its truncation to 1-2 fragments
(or an overfill to 8 with fused fragments when the final has one fragment).

Reward family: R = Σ w_i·f_i, w >= 0, Σw = 1, f = (F_β, recall, precision, coverage, diversity, coherence, n/8),
β from BETAS. Fit: pairwise logistic ranking loss inside each question (pairs U_i > U_j, weight U_i - U_j) by
exponentiated gradient (stays on the simplex). Metric: mean Kendall τ-b per question; 5-fold CV grouped by
document components; adoption iff held-out τ(new) - τ(current) > 2 SE (paired over questions)."""
from __future__ import annotations

import ast
import hashlib
import math
import re
import unicodedata
from collections import Counter

import numpy as np

from grl.train.fbeta import r_chunks_fbeta

BETAS = (0.5, 1.0, 2.0, 3.0)
BETA_KEYS = {0.5: "f0.5", 1.0: "f1", 2.0: "f2", 3.0: "f3"}
BASE_FEATURES = ("recall", "precision", "coverage", "diversity", "coherence", "n")
CURRENT = {"f1": 0.5, "coverage": 0.2, "diversity": 0.2, "coherence": 0.1}     # train.yaml reward.weights (ТЗ)
N_CAP = 8                         # n feature = |C| / 8 (the overfill length)
U_K = 5                           # recall@5 / all@5
CE_A = 0.5                        # ce+fused:<a>, the middle of rerank_baselines.CE_FUSED_A
FUSE_W, NORM_TOP_N, CAND_N = 0.7, 50, 20
CANDIDATES = ("dense@3", "dense@5", "fused@5", "ce+fused@5", "expert", "expert_alt")
ANSWER_CANDIDATES = ("dense@3", "dense@5", "fused@5", "ce+fused@5", "expert")   # U_ans subsample: 5 per question
SPEARMAN_MIN = 0.6


# --- U_ret and features ----------------------------------------------------------------------------------------------

def u_ret(context, gold) -> float:
    G = {int(g) for g in gold}
    if not G:
        raise ValueError("empty gold")
    first = {int(c) for c in list(context)[:U_K]}
    hit = len(first & G)
    return 0.5 * hit / len(G) + 0.5 * float(hit == len(G))


def chunk_features(context, gold) -> dict:
    C, G = {int(c) for c in context}, {int(g) for g in gold}
    tp = len(C & G)
    out = {BETA_KEYS[b]: r_chunks_fbeta(C, G, b) for b in BETAS}
    out.update(recall=tp / len(G), precision=tp / len(C) if C else 0.0, n=len(C) / N_CAP)
    return out


def context_features(data, depth, context, gold, t_max: int) -> dict:
    from graphrag.env.reward import chunk_depths, r_coherence, r_coverage, r_diversity
    C, G = {int(c) for c in context}, {int(g) for g in gold}
    return {**chunk_features(C, G), "coverage": r_coverage(data, C, G),
            "diversity": r_diversity(chunk_depths(data, depth, sorted(C), t_max), t_max),
            "coherence": r_coherence(data, C)}


def score(feats: dict, weights: dict) -> float:
    return float(sum(w * float(feats[k]) for k, w in weights.items()))


def feature_names(beta: float) -> tuple:
    return (BETA_KEYS[float(beta)],) + BASE_FEATURES


# --- candidates ------------------------------------------------------------------------------------------------------

def trunc_or_overfill(expert, fill, n_over: int = N_CAP) -> tuple[list, str]:
    """>= 3 fragments -> the first 2; 2 -> the first 1; 1 -> overfill to n_over with `fill` (fused order) fragments."""
    e = [int(c) for c in expert]
    if not e:
        return [], "empty"
    if len(e) >= 3:
        return e[:2], "trunc2"
    if len(e) == 2:
        return e[:1], "trunc1"
    return e + [int(c) for c in fill if int(c) not in e][:n_over - len(e)], f"over{n_over}"


def _mm(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.size < 2 or not float(x.max() - x.min()) > 1e-12:
        return np.ones_like(x) if x.size < 2 else np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def ce_fused_order(rows, fused, ce: dict, a: float = CE_A):
    """rows re-ordered by mm(a·mm(ce) + (1-a)·fused) desc, ties -> candidate position (rerank_baselines ce+fused:<a>);
    None when a candidate has no ce score."""
    rows = np.asarray(rows, np.int64)
    if any(int(r) not in ce for r in rows.tolist()):
        return None
    s = _mm(a * _mm([ce[int(r)] for r in rows.tolist()]) + (1 - a) * np.asarray(fused, np.float64))
    order = np.lexsort((np.arange(len(rows)), -s))
    return rows[order], s[order]


# --- answers: normalisation and F1 -----------------------------------------------------------------------------------

_CYR = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y",
        "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
        "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "yu",
        "я": "ya", "ў": "o", "қ": "q", "ғ": "g", "ҳ": "h"}
_APOS = "ʻʼ'`’‘ʹ′"
_MONTHS = ((r"yanvar\w*", 1), (r"fevral\w*", 2), (r"mart\w*", 3), (r"aprel\w*", 4), (r"ma(?:y|ya|yi|i|iy)\b", 5),
           (r"iyun\w*", 6), (r"iyul\w*", 7), (r"avgust\w*", 8), (r"sent(?:y)?abr\w*", 9), (r"okt(?:y)?abr\w*", 10),
           (r"noyabr\w*", 11), (r"dekabr\w*", 12))


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = "".join(_CYR.get(ch, ch) for ch in s)
    return "".join(ch for ch in s if ch not in _APOS)


def norm_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm_text(s))


def _f1(pred: list, ref: list) -> float:
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def token_f1(pred: str, ref: str) -> float:
    return _f1(norm_tokens(pred), norm_tokens(ref))


def numbers(s: str, months: bool = True) -> list[str]:
    t = norm_text(s)
    out = [str(int(n)) for n in re.findall(r"\d+", t)]
    if months:
        for rx, m in _MONTHS:
            out += [str(m)] * len(re.findall(r"\b" + rx, t))
    return out


def answer_kind(row: dict) -> str:
    if row["pattern"] == "count":
        return "count"
    if row["pattern"] == "version":
        tpl = (((row.get("provenance") or {}).get("m7") or {}).get("template") or "")
        return "version_law" if tpl.startswith("which_law") else "version"
    return "text"


_LABEL = re.compile(r"^[^\W\d_]+\.\s*(\S+)$")


def reference_numbers(kind: str, text: str) -> list[str]:
    if kind == "count":
        return numbers(text, months=False)
    edges = ast.literal_eval(text)
    if kind == "version_law":
        return sorted({str(int(x)) for e in edges for x in str(e[1]).split("-")})
    out = []
    for e in edges:
        for seg in str(e[3]).split("/"):
            m = _LABEL.match(seg.strip())
            if m:
                out += [str(int(n)) for n in re.findall(r"\d+", m.group(1))]
    return out


def number_f1(answer: str, ref: list, question: str) -> float:
    """F1 of number multisets; a number restated from the question is dropped from the answer, but never below its
    count in the reference (keep = min(c_ans, max(c_ref, c_ans - c_question)))."""
    cp, cr_, cq = Counter(numbers(answer)), Counter(ref), Counter(numbers(question))
    pred = [n for n, c in sorted(cp.items()) for _ in range(min(c, max(cr_[n], c - cq[n])))]
    return _f1(pred, list(ref))


def count_correct(answer: str, ref: list, question: str) -> float:
    """1 iff the first number of the answer that does not restate the question equals the reference count."""
    q = Counter(numbers(question, months=False))
    for n in numbers(answer, months=False):
        if q[n] > 0 and n not in ref:
            q[n] -= 1
            continue
        return float(n in ref)
    return 0.0


def u_ans_f1(row: dict, ans: dict) -> float:
    if not ans or not ans.get("answerable") or not (ans.get("answer") or "").strip():
        return 0.0
    kind = answer_kind(row)
    if kind == "text":
        return token_f1(ans["answer"], row["y_ref"]["text"])
    if kind == "count":
        return count_correct(ans["answer"], reference_numbers(kind, row["y_ref"]["text"]), row["q"])
    return number_f1(ans["answer"], reference_numbers(kind, row["y_ref"]["text"]), row["q"])


def reference_text(row: dict) -> str:
    """The reference answer as shown to the judge (version edges rendered readably)."""
    kind = answer_kind(row)
    if kind in ("text", "count"):
        return row["y_ref"]["text"]
    edges = ast.literal_eval(row["y_ref"]["text"])
    if kind == "version_law":
        return "; ".join(sorted({f"law of {e[1]}" for e in edges}))
    return "; ".join(f"{e[3].split('/', 2)[-1]} ({e[2]})" for e in edges)


# --- LLM -------------------------------------------------------------------------------------------------------------

PASS_ANSWER, PASS_JUDGE = "rc_answer", "rc_judge"

ANSWER_PROMPT = """You answer questions about Uzbek legislation and Senate (Oliy Majlis) documents. Use ONLY the \
numbered fragments in the input; do not use outside knowledge. Questions are in Russian, Uzbek (Cyrillic) or Uzbek \
(Latin).

Answer in the language and script of the question, concisely: at most two sentences. For a «how many» question give \
the number. For «which provisions / articles» list them with their article and part numbers. For «which law» give the \
law's date (and number if the fragments show it).
If the fragments do not contain the information needed, set answerable = false and answer = ""."""

ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answerable", "answer"],
                 "properties": {"answerable": {"type": "boolean"}, "answer": {"type": "string"}}}

JUDGE_PROMPT = """You grade an answer to a question about Uzbek legislation against the reference answer. Languages: \
Russian, Uzbek (Cyrillic or Latin); a different language or script is not an error.

score 2: the answer states the same facts as the reference (numbers, articles / parts, dates, the key content); \
extra correct detail is fine.
score 1: partly right: some of the reference facts are there, others are missing or wrong.
score 0: wrong, unrelated, or no answer.
why: a short reason in English (at most 20 words)."""

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["score", "why"],
                "properties": {"score": {"type": "integer", "enum": [0, 1, 2]}, "why": {"type": "string"}}}


def answer_input(question: str, texts: list[str]) -> dict:
    return {"question": question, "fragments": [{"n": i + 1, "text": t} for i, t in enumerate(texts)]}


def ask(llm, pass_name: str, inp: dict, prompt: str, schema: dict):
    from graphrag.dataset.facts import guard_total
    guard_total(llm, pass_name, inp, prompt, schema)
    return llm.cached_json(pass_name, inp, schema, prompt=prompt)


# --- ranking metrics and fit -----------------------------------------------------------------------------------------

def kendall_tau(x, y) -> float:
    """τ-b of R (x) vs U (y): nan when U is constant (no order to match); 0 when R is constant but U is not."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(x)
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(round(x[i] - x[j], 12)), np.sign(round(y[i] - y[j], 12))
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                c += 1
            else:
                d += 1
    n0 = n * (n - 1) // 2
    if n0 - ty == 0:
        return float("nan")
    if n0 - tx == 0:
        return 0.0
    return (c - d) / math.sqrt((n0 - tx) * (n0 - ty))


def mean_tau(groups, w) -> tuple[float, float, list]:
    """groups: [(F, U)]; per-question τ (nan dropped) -> mean, SE, list (nan kept for pairing)."""
    taus = [kendall_tau(F @ w, U) for F, U in groups]
    ok = np.array([t for t in taus if not math.isnan(t)])
    return float(ok.mean()), float(ok.std(ddof=1) / math.sqrt(len(ok))) if len(ok) > 1 else float("nan"), taus


def _pairs(groups, eps: float = 1e-9):
    D, W = [], []
    for F, U in groups:
        F, U = np.asarray(F, np.float64), np.asarray(U, np.float64)
        for i in range(len(U)):
            for j in range(len(U)):
                if U[i] > U[j] + eps:
                    D.append(F[i] - F[j])
                    W.append(U[i] - U[j])
    return np.array(D), np.array(W)


def fit_simplex(groups, temp: float = 20.0, iters: int = 3000, eta: float = 0.5) -> np.ndarray:
    """argmin_w Σ_pairs (U_i - U_j)·softplus(-temp·(R_i - R_j)) / Σ weights on the simplex, by exponentiated gradient
    from the uniform point (deterministic)."""
    D, W = _pairs(groups)
    d = len(groups[0][0][0])
    w = np.full(d, 1.0 / d)
    if not len(D):
        return w
    W = W / W.sum()
    for _ in range(iters):
        m = temp * (D @ w)
        g = -(W * temp * (1.0 / (1.0 + np.exp(np.clip(m, -50, 50))))) @ D
        w = w * np.exp(-eta * g / (np.abs(g).max() + 1e-12))
        w = w / w.sum()
    return w


def pair_loss(groups, w, temp: float = 20.0) -> float:
    D, W = _pairs(groups)
    if not len(D):
        return 0.0
    return float((W * np.logaddexp(0.0, -temp * (D @ w))).sum() / W.sum())


def _u(seed, s: str) -> str:
    return hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()


def doc_folds(questions, k: int = 5, seed=20261004) -> dict:
    """{qid: fold}: questions sharing a document (transitively) land in one fold; components in sha256 order, larger
    first, each to the currently smallest fold (lowest index on ties)."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q in questions:
        ds = list(q["docs"]) or ["q:" + q["id"]]
        for d in ds[1:]:
            a, b = find(ds[0]), find(d)
            if a != b:
                parent[max(a, b)] = min(a, b)
        find(ds[0])
    comp: dict = {}
    for q in questions:
        comp.setdefault(find((list(q["docs"]) or ["q:" + q["id"]])[0]), []).append(q["id"])
    order = sorted(comp.values(), key=lambda ids: (-len(ids), _u(seed, min(ids))))
    size, out = [0] * k, {}
    for ids in order:
        f = min(range(k), key=lambda i: (size[i], i))
        size[f] += len(ids)
        for i in ids:
            out[i] = f
    return out


def adopt(tau_new, tau_cur) -> dict:
    a, b = np.asarray(tau_new, np.float64), np.asarray(tau_cur, np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"n": int(ok.sum()), "delta": float(d.mean()), "se": se, "adopt": bool(d.mean() > 2 * se)}


def subsample(rows, quotas: dict, seed=20261004) -> list[str]:
    """Ids: per set its quota, split over patterns by largest remainder, sha256(seed:id) order inside a pattern."""
    out = []
    for st, n in sorted(quotas.items()):
        rs = [r for r in rows if r["set"] == st]
        by = Counter(r["pattern"] for r in rs)
        exact = {p: n * c / len(rs) for p, c in by.items()}
        alloc = {p: int(math.floor(v)) for p, v in exact.items()}
        for p in sorted(exact, key=lambda p: (-(exact[p] - alloc[p]), p))[:n - sum(alloc.values())]:
            alloc[p] += 1
        for p, m in sorted(alloc.items()):
            ids = sorted((r["id"] for r in rs if r["pattern"] == p), key=lambda i: _u(seed, i))
            out += ids[:m]
    return sorted(out)


def spearman(x, y) -> float:
    from grl.train.calibrate import correlation
    return correlation(x, y, "spearman")


def p5_threshold(R, U, q: float = 0.1, good: float = 0.5) -> dict:
    R, U = np.asarray(R, np.float64), np.asarray(U, np.float64)
    g = U >= good
    thr = float(np.quantile(R[g], q)) if g.any() else float("nan")
    bad = ~g
    return {"threshold": thr, "q": q, "n_good": int(g.sum()), "n_bad": int(bad.sum()),
            "bad_pass": float((R[bad] >= thr).mean()) if bad.any() else float("nan"),
            "good_pass": float((R[g] >= thr).mean()) if g.any() else float("nan")}
