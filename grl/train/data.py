"""Training data for M4/M5 (IMPLEMENTATION.md §8.1, §9.1; §12 п. 7). Only the train and val splits are ever
materialised: the test split of the dataset and Q-real are refused (SplitGuardError). Expert trajectories (M3
oracle, Task 20 machinery) are replayed from reset; the recorded mask must equal the replayed one (DataMismatch
otherwise -- e.g. a different BLAS on another host), giving the SFT examples (state, class of equivalent actions).
Fix round 1: the recorded class must equal the replayed M3 action class exactly (a context drift that keeps the mask
is caught too), the recorded reward parts are re-derived, and Q-real is refused by origin / split / id."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from graphrag.dataset.finalize import read_pairs
from grl.model.observe import StateDesc, describe


class SplitGuardError(RuntimeError):
    pass


class DataMismatch(RuntimeError):
    pass


ALLOWED_SPLITS = ("train", "val")
QREAL_SUFFIXES = ("|cyr", "|lat")
QREAL_SPLITS = ("qreal_test",)
QREAL_ORIGIN = "qreal"
QREAL_IDS_PATH = "data/gold/qreal.jsonl"      # Mac only (never shipped); absent on Spark -> suffix check only
REWARD_TOL = 1.5e-9                       # one 9-decimal rounding unit + 1 ulp (libm on another host)
REWARD_PARTS = ("chunks", "coverage", "diversity", "coherence")   # weight-independent; `total` is never compared
TEST_EVAL_NAME = "test_eval.jsonl"     # pilot 5: ship/test_eval/test_eval.jsonl, the strict consensus test
TEST_EVAL_MARK = "test_eval"           # provenance flag of its rows (grl.train.pilot.build_ship)
LIMIT_FINAL_REASONS = ("k_max", "l_max")   # R34: limit-ended expert finals (no STOP step); exhausted / t_max are not
LIMIT_FINAL_VARIANTS = ("struct",)         # R34: replay accepts final_by == "limit" only for these variants


@lru_cache(maxsize=4)
def _qreal_ids(path: str) -> frozenset:
    from graphrag import config
    f = config.path(path)
    if not f.exists():
        return frozenset()
    ids = set()
    for line in f.read_text(encoding="utf-8").splitlines():   # only the id field is kept (F17: no Q-real content)
        if line.strip():
            r = json.loads(line)
            ids.update(str(r[k]) for k in ("qid", "id") if r.get(k))
    return frozenset(ids)


def qreal_ids() -> frozenset:
    return _qreal_ids(QREAL_IDS_PATH)


def guard_splits(splits) -> None:
    bad = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad:
        raise SplitGuardError(f"M4/M5 read only {ALLOWED_SPLITS}; refused {bad} (test / Q-real are M6, §9.1)")


def guard_qkeys(keys) -> None:
    ids = qreal_ids()
    bad = [k for k in keys if str(k).endswith(QREAL_SUFFIXES) or str(k).split("|")[0] in ids]
    if bad:
        raise SplitGuardError(f"Q-real query keys are a test set (IMPLEMENTATION.md §9.1): {bad[:3]}")


@dataclass(frozen=True)
class TrainItem:
    pid: str
    qkey: str
    split: str
    pattern: str
    lang: str
    gold: frozenset
    actions: tuple
    equiv: tuple
    masks: tuple
    reward: tuple = ()               # recorded RewardParts as sorted (name, value) pairs; () when not recorded
    variant: str | None = None       # M5 variant expert (grl.train.expert): provenance.meta.variant; None = M3 oracle
    env_kw: str | None = None        # ... and its env kwargs (canonical JSON); None = M3 oracle
    reach: bool = True               # the trajectory reaches all of C_ref (always for M3 П5; recorded by the expert)
    reward_key: str | None = None    # the reward the expert optimised (canonical JSON of chunks_beta + weights)
    score: float | None = None       # the expert's grl reward of this trajectory (None for M3 rows)
    k_max: int | None = None         # env K_max the expert searched under (provenance.meta.expert.K_max; None = not
                                     # recorded: M3 rows and pre-R12 expert rows, built under the train.yaml K_max)
    final_by: str | None = None      # R34: provenance.meta.expert.final_by ("stop" | "limit"); None = not recorded
                                     # (M3 rows and expert rows searched without limit finals) = a STOP final
    prune_zero: bool | None = None   # R35: provenance.meta.expert.prune_zero (None = not recorded / M3 row)


def pair_meta(p) -> dict | None:
    """The variant-expert meta of a §8.1 pair (provenance.meta, written by grl.train.expert); None for M3 rows."""
    m = (p.provenance or {}).get("meta")
    return m if isinstance(m, dict) else None


def canonical_kw(kw) -> str:
    return json.dumps(kw or {}, sort_keys=True)


def reward_key(spec: dict | None) -> str | None:
    """Canonical {chunks_beta, weights} (grl.train.expert.reward_spec) -- the SFT gate compares it to the run's."""
    if not spec or "chunks_beta" not in spec or "weights" not in spec:
        return None
    return json.dumps({"chunks_beta": float(spec["chunks_beta"]),
                       "weights": {k: float(v) for k, v in sorted(spec["weights"].items())}}, sort_keys=True)


def refuse_qreal(path, p, ids) -> None:
    if p.origin == QREAL_ORIGIN or p.split in QREAL_SPLITS or p.id in ids:          # loud, whatever was asked for
        raise SplitGuardError(f"{path}: Q-real row {p.id} (origin {p.origin!r}, split {p.split!r}) refused")


def item_of(p, data) -> TrainItem:
    if not p.actions or len(p.action_equiv) != len(p.actions) or len(p.action_masks) != len(p.actions):
        raise ValueError(f"{p.id}: no complete expert trajectory (actions / action_equiv / action_masks)")
    meta = pair_meta(p)
    exp = (meta or {}).get("expert") or {}
    return TrainItem(p.id, f"{p.id}|q", p.split, p.pattern, p.lang,
                     frozenset(int(data.chunk_row[c]) for c in p.C_ref), tuple(int(a) for a in p.actions),
                     tuple(tuple(int(a) for a in e) for e in p.action_equiv),
                     tuple(tuple(int(a) for a in m) for m in p.action_masks),
                     tuple(sorted((k, float(v)) for k, v in (p.reward or {}).items())),
                     None if meta is None else str(meta.get("variant")),
                     None if meta is None else canonical_kw(meta.get("env_kw")),
                     True if meta is None else bool(exp.get("reached", True)),
                     None if meta is None else reward_key(exp),
                     None if meta is None or exp.get("score") is None else float(exp["score"]),
                     None if meta is None or exp.get("K_max") is None else int(exp["K_max"]),
                     None if meta is None or exp.get("final_by") is None else str(exp["final_by"]),
                     None if meta is None or exp.get("prune_zero") is None else bool(exp["prune_zero"]))


def is_test_eval(path, p=None) -> bool:
    from pathlib import Path
    return Path(path).name == TEST_EVAL_NAME or (p is not None and bool((p.provenance or {}).get(TEST_EVAL_MARK)))


def refuse_probe(path) -> None:
    """M7: a probe file (split "probe", grl.hard.probe) is never a training / labelling / selection input."""
    from grl.hard.probe import refuse_probe_file
    refuse_probe_file(path)


def refuse_test_eval(path, p=None) -> None:
    """Pilot 5: the test-eval file (by name, or any row carrying the mark) is never a training / selection input."""
    if is_test_eval(path, p):
        raise SplitGuardError(f"{path}: the consensus test-eval file is read only by train.py eval / baselines "
                              f"--split test (pilot 5 final report); never by training, labelling or selection")


def load_test_items(path, data) -> list[TrainItem]:
    """Pilot 5 final report: the strict consensus test rows of a ship/test_eval/test_eval.jsonl file -> items for
    greedy evaluation and the reference baselines (qkey + C*, no trajectory). Refuses any other file and any row
    that is not a marked test row; Q-real refused as everywhere."""
    from pathlib import Path
    if Path(path).name != TEST_EVAL_NAME:
        raise SplitGuardError(f"{path}: --split test reads only a {TEST_EVAL_NAME} file")
    ids, out = qreal_ids(), []
    for p in read_pairs(path):
        refuse_qreal(path, p, ids)
        if p.split != "test" or not (p.provenance or {}).get(TEST_EVAL_MARK):
            raise SplitGuardError(f"{path}: row {p.id} (split {p.split!r}) is not a marked consensus-test row")
        out.append(TrainItem(p.id, f"{p.id}|q", p.split, p.pattern, p.lang,
                             frozenset(int(data.chunk_row[c]) for c in p.C_ref), (), (), ()))
    guard_qkeys([i.qkey for i in out])
    return sorted(out, key=lambda i: i.pid)


def load_items(path, data, splits) -> list[TrainItem]:
    guard_splits(splits)
    refuse_test_eval(path)
    refuse_probe(path)
    ids = qreal_ids()
    out = []
    for p in read_pairs(path):
        refuse_qreal(path, p, ids)
        refuse_test_eval(path, p)
        if p.split not in splits:
            continue
        out.append(item_of(p, data))
    guard_qkeys([i.qkey for i in out])
    return sorted(out, key=lambda i: i.pid)


def load_rank_items(path, data, splits) -> list[TrainItem]:
    """Pilot 6 (rank variant only): train / val rows WITH OR WITHOUT an expert trajectory (the rank SFT labels come
    from C* directly, so the unfiltered population -- snapshot ship/pairs.jsonl -- can be used). The same guards as
    load_items: only train / val splits, the test-eval file and its marked rows refused, Q-real refused (origin /
    split / id), Q-real query keys refused. Items carry no trajectory (actions / equiv / masks empty)."""
    guard_splits(splits)
    refuse_test_eval(path)
    refuse_probe(path)
    ids = qreal_ids()
    out = []
    for p in read_pairs(path):
        refuse_qreal(path, p, ids)
        refuse_test_eval(path, p)
        if p.split not in splits:
            continue
        if not p.C_ref:
            raise ValueError(f"{path}: row {p.id} has an empty C_ref")
        out.append(TrainItem(p.id, f"{p.id}|q", p.split, p.pattern, p.lang,
                             frozenset(int(data.chunk_row[c]) for c in p.C_ref), (), (), ()))
    guard_qkeys([i.qkey for i in out])
    return sorted(out, key=lambda i: i.pid)


@dataclass
class SFTExample:
    qkey: str
    t: int
    desc: StateDesc
    equiv: tuple


def replay_item(env, ranks, it: TrainItem, visit=None):
    """Mask- and class-checked replay of one expert trajectory from reset (Review Focus 4). At every step the replayed
    mask must equal the recorded one, the recorded action must lie in its recorded class, and that class must equal
    the M3 action class (graphrag.env.policies.action_classes: same ΔC, same covered set) of the replayed state. The
    episode must end with STOP on the last recorded action -- or, for a row of a LIMIT_FINAL_VARIANTS variant whose
    meta.expert.final_by is "limit" (R34), end on its last recorded (non-STOP) action with reason k_max / l_max --
    and reach C_ref (a variant-expert row recorded as not
    reaching it, TrainItem.reach False, must replay to the same miss); the recorded weight-independent reward
    parts (chunks, coverage, diversity, coherence; never `total`, which depends on the weights an ablation may change)
    must be re-derived. visit(t, q, s) sees every live state before its
    action. Returns (q, final state); DataMismatch names the pair and the step."""
    from graphrag.env.policies import action_classes
    if it.final_by not in (None, "stop", "limit"):
        raise DataMismatch(f"{it.pid}: unknown meta.expert.final_by {it.final_by!r}")
    limit = it.final_by == "limit"
    if limit and it.variant not in LIMIT_FINAL_VARIANTS:
        raise DataMismatch(f"{it.pid}: a limit-ended final (final_by 'limit') is accepted only for variants "
                           f"{LIMIT_FINAL_VARIANTS}, not {it.variant!r}")
    q = env.query(ranks.get(it.qkey))
    s = env.reset(q)
    for t, a in enumerate(it.actions):
        if s.done:
            raise DataMismatch(f"{it.pid} t={t + 1}: episode ended ({s.reason!r}) before recorded action {a}")
        m = np.flatnonzero(env.mask(q, s)).tolist()
        if m != list(it.masks[t]):
            raise DataMismatch(f"{it.pid} t={t + 1}: replayed mask {m} != recorded {list(it.masks[t])}")
        if a not in it.equiv[t] or not set(it.equiv[t]) <= set(m):
            raise DataMismatch(f"{it.pid} t={t + 1}: action {a} / class {list(it.equiv[t])} vs mask {m}")
        cls = next((c for c in action_classes(env, q, s) if a in c), None)
        if cls is None or tuple(cls) != tuple(it.equiv[t]):
            raise DataMismatch(f"{it.pid} t={t + 1}: equiv -- replayed class {cls} != recorded {list(it.equiv[t])}")
        if visit is not None:
            visit(t, q, s)
        s = env.step(q, s, a)
    if limit:
        ended = s.done and s.reason in LIMIT_FINAL_REASONS and bool(it.actions) and it.actions[-1] != 0
    else:
        ended = s.done and s.reason == "stop" and bool(it.actions) and it.actions[-1] == 0
    if not ended or (it.gold <= s.ctx_set) != it.reach:
        tag = " (final_by limit)" if limit else ""
        raise DataMismatch(f"{it.pid}: replay ends with {s.reason!r}{tag}, C_ref reached: {it.gold <= s.ctx_set} "
                           f"(recorded {it.reach})")
    if it.reward:
        from graphrag.env.reward import Weights, reward
        got = reward(env.data, q.depth, s.context, it.gold, Weights(), env.limits.T_max)   # parts ignore the weights
        rec = dict(it.reward)
        diff = {k: (rec.get(k), getattr(got, k)) for k in REWARD_PARTS
                if k not in rec or not math.isclose(rec[k], getattr(got, k), rel_tol=0, abs_tol=REWARD_TOL)}
        if diff:
            raise DataMismatch(f"{it.pid}: reward -- recorded vs replayed {diff}")
    return q, s


def replay_examples(env, ranks, items, describe_kw: dict) -> tuple[list[SFTExample], dict]:
    out, expert = [], {}
    for it in items:
        def visit(t, q, s, it=it):
            out.append(SFTExample(it.qkey, t + 1, describe(env, q, s, **describe_kw), it.equiv[t]))
        _, s = replay_item(env, ranks, it, visit)
        expert[it.qkey] = frozenset(s.ctx_set)
    return out, expert


class VariantMismatch(ValueError):
    pass


def check_k_max(rows, k_max: int | None, k_max_legacy: int | None) -> None:
    """Ruling R12: every row's recorded K_max (rows[i][4]; None = not recorded = the train.yaml K_max `k_max_legacy`
    the M3 / pre-R12 labels were built under) must equal the run's env K_max (`--set env.K_max`). k_max None: no
    check."""
    if k_max is None:
        return
    rec = [(r[0], r[4] if len(r) > 4 and r[4] is not None else k_max_legacy) for r in rows]
    bad = [pid for pid, k in rec if k is None or int(k) != int(k_max)]
    if bad:
        got = sorted({str(k) for _, k in rec})
        raise VariantMismatch(f"{len(bad)} rows (e.g. {bad[:3]}) were labelled under env K_max {got}, the run uses "
                              f"K_max {k_max} -- rebuild the labels with train.py expert --set env.K_max={k_max}")


def check_variant(rows, variant: str, env_kw: dict | None, fixed_k_ok: bool, reward: dict | None = None,
                  k_max: int | None = None, k_max_legacy: int | None = None) -> str:
    """SFT label gate (M5 pre-RL). rows = (pid, meta variant | None, canonical env kwargs | None, reward key |
    None). Variant-expert rows (grl.train.expert) are accepted only when EVERY row was searched in exactly this
    variant with exactly these env kwargs (dynamic_k normalise / norm_top_n / fuse_w) and -- when `reward` (the run's
    grl.train.expert.reward_spec) is given -- for exactly this reward (chunks_beta and the four weights) -> "expert";
    M3 rows (no meta) only when the variant's search effects are the fixed-k M3 ones (fixed_k_ok: not adaptive /
    fused) -> "m3". A mix is refused."""
    rows = list(rows)
    if not rows:
        raise VariantMismatch("no train / val rows")
    check_k_max(rows, k_max, k_max_legacy)
    tagged = [r for r in rows if r[1] is not None]
    if not tagged:
        if not fixed_k_ok:
            raise VariantMismatch(f"the trajectories are M3 oracle labels with fixed-k search effects; build "
                                  f"variant-expert labels first: train.py expert --variant {variant}")
        return "m3"
    want = canonical_kw(env_kw)
    bad = [r[0] for r in rows if r[1] != variant or r[2] != want]
    if bad:
        got = sorted({(str(r[1]), str(r[2])) for r in rows})[:3]
        raise VariantMismatch(f"{len(bad)} rows (e.g. {bad[:3]}) were not labelled in variant {variant!r} with env "
                              f"kwargs {want}; found (variant, env_kw) {got}")
    if reward is not None:
        rk = reward_key(reward)
        bad = [r[0] for r in rows if r[3] != rk]
        if bad:
            got = sorted({str(r[3]) for r in rows})[:3]
            raise VariantMismatch(f"{len(bad)} rows (e.g. {bad[:3]}) were labelled by an expert that optimised a "
                                  f"different reward than this run's {rk} (reward.chunks_beta / reward.weights); "
                                  f"found {got} -- rebuild the labels with train.py expert under the run's --set")
    return "expert"


def check_variant_items(items, variant: str, env_kw: dict | None, fixed_k_ok: bool, reward: dict | None = None,
                        k_max: int | None = None, k_max_legacy: int | None = None) -> str:
    items = list(items)
    out = check_variant([(i.pid, i.variant, i.env_kw, i.reward_key, i.k_max) for i in items], variant, env_kw,
                        fixed_k_ok, reward, k_max, k_max_legacy)
    if variant in STRUCT_GATE_VARIANTS:                      # R35 also on the loaded items (cmd_sft)
        check_struct_labels([{"id": i.pid, "provenance": {"meta": {"variant": i.variant,
                                                                     "expert": {"prune_zero": i.prune_zero}}}}
                             for i in items])
    return out


def check_variant_file(path, variant: str, env_kw: dict | None, fixed_k_ok: bool, reward: dict | None = None,
                       k_max: int | None = None, k_max_legacy: int | None = None) -> str:
    """check_variant over the train / val rows of a pairs file (cheap: no env, no replay); variant struct: then
    check_struct_labels (R35) over the same rows."""
    rows, raw = [], []
    refuse_test_eval(path)
    refuse_probe(path)
    for p in read_pairs(path):
        refuse_test_eval(path, p)
        if p.split in ALLOWED_SPLITS:
            m = pair_meta(p)
            raw.append(p.to_json())
            rows.append((p.id, None if m is None else str(m.get("variant")),
                         None if m is None else canonical_kw(m.get("env_kw")),
                         None if m is None else reward_key(m.get("expert")),
                         None if m is None or (m.get("expert") or {}).get("K_max") is None
                         else int(m["expert"]["K_max"])))
    out = check_variant(rows, variant, env_kw, fixed_k_ok, reward, k_max, k_max_legacy)
    if variant in STRUCT_GATE_VARIANTS:
        check_struct_labels(raw)
    return out


STRUCT_GATE_VARIANTS = ("struct",)


def check_struct_labels(rows) -> None:
    """R35 SFT gate: every struct-expert label must come from the UNPRUNED search (provenance.meta.expert.prune_zero
    is False -- a zero-Δ step is a bridge in StructEnv, the pruned expert cannot reach gold behind it). rows: pair
    JSON dicts (Pair.to_json()). A row whose meta records no prune_zero is refused as well. SystemExit names the
    rows."""
    bad = []
    for r in rows:
        meta = (r.get("provenance") or {}).get("meta")
        exp = (meta.get("expert") if isinstance(meta, dict) else None) or {}
        if exp.get("prune_zero") is not False:
            bad.append((r.get("id"), exp.get("prune_zero")))
    if bad:
        raise SystemExit(f"struct SFT labels refused (R35): {len(bad)} rows (e.g. {bad[:3]}) have "
                         f"meta.expert.prune_zero != False -- rebuild them with train.py expert --variant struct "
                         f"(unpruned by default)")


def _stop1(i) -> bool:
    """An immediate STOP: one recorded action on a STOP-ended row (a one-step limit final is a retrieval, R34)."""
    return len(i.actions) == 1 and i.final_by != "limit"


def sft_train_filter(items, keep: bool = False) -> tuple[list, dict]:
    """Default SFT-train rule for variant-expert rows (grl.train.expert.sft_drop): drop rows whose expert score is
    <= 1e-9, that STOP immediately (one action) or that miss C*; keep=True (train.py sft --keep-unreached) keeps them.
    M3 rows are never dropped. Returns (kept, counts) -- the counts go to the run manifest, never silent."""
    from grl.train.expert import DROP_EPS
    n = len(items)
    ex = [i for i in items if i.variant is not None]
    cnt = {"n": n, "expert_rows": len(ex),
           "score0": sum(1 for i in ex if i.score is not None and i.score <= DROP_EPS),
           "len1": sum(1 for i in ex if _stop1(i)), "unreached": sum(1 for i in ex if not i.reach)}

    def drop(i):
        return i.variant is not None and ((i.score is not None and i.score <= DROP_EPS) or _stop1(i)
                                          or not i.reach)
    kept = list(items) if keep else [i for i in items if not drop(i)]
    cnt.update({"dropped": n - len(kept), "kept": len(kept), "keep_unreached": bool(keep),
                **{f"{k}_share": round(cnt[k] / n, 6) if n else 0.0 for k in ("score0", "len1", "unreached")},
                "reached_share": round(1 - cnt["unreached"] / n, 6) if n else 0.0})
    return kept, cnt
