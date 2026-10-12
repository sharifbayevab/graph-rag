"""Variant-aware expert (M5 pre-RL): the SFT teacher searched in the run's OWN env (fused_adaptive, adaptive, full,
...), not the M3 beam oracle over fixed k. Knows C* (a teacher and an upper bound, never a deployable policy).

Search (expert_search): exact enumeration over the variant env's actions up to `depth` retrieval steps, then STOP.
- Children of every live state: every valid non-STOP action (env.mask, so the variant's `allowed` filter applies);
  a child whose step added no fragment ("zero-Δ": a graph hop onto chunkless vertices) is pruned unless
  prune_zero=False.
- Duplicates are merged by (ctx_set, covered): the reward depends only on ctx_set, the mask on (ctx_set, length,
  covered), graph candidates on covered and Δ_V on the context SET (a max over it), so two states with the same
  signature have the same futures; the one kept is the first in breadth-first order (shorter, then lexicographically
  smaller action ids), the representative every tie-break below would pick.
- Finals: the STOP-completion of every live state (incl. reset) whose mask allows STOP; a state ended by a limit
  (k_max / l_max / exhausted) is not a §8.1 final (it does not end with STOP) -- unless limit_finals (ruling R34,
  variant struct): then a state ended by k_max / l_max is a final as it is (no STOP step, final_by "limit", ranked
  with its own step count); exhausted / t_max never are.
- RSS guard (R40 / R42, struct only): a query whose process RSS grows by more than expert.struct_max_rss_gb (or
  passes the optional absolute expert.struct_rss_ceiling_gb) continues as the cap beam. Such a label depends on the
  process (memory state, allocator, host), not only on the query: it is flagged per row (meta.expert.capped,
  capped_reason "rss" / "rss_ceiling", search "beam<cap_beam>") and counted in the stats (capped_rss).
- Variant defaults (expert_defaults, R35): struct searches with prune_zero=False and limit_finals=True (a zero-Δ step
  is a bridge in StructEnv: it grows `covered` and opens graph / structure candidates); every other variant keeps
  prune_zero=True, limit_finals=False (byte-identical labels).
- Best final: max grl reward (grl.train.fbeta.grl_reward, train.yaml reward.weights and chunks_beta; F1 by default),
  ties -> fewer steps, then the lexicographically smaller action-id sequence. Deterministic (no RNG).
- `beam` (optional): keep only the best `beam` live states per depth (same key on the state's own context); the
  result is then flagged exact = False.
Records (expert_pair): the §8.1 schema of M3 (graphrag.env.policies.trajectory_steps / trajectory_labels, computed in
the VARIANT env: classes of identical effect under the variant's search; a limit final: limit_labels, the same
classes over every retrieval step and no closing [0]), spec reward parts, and provenance.meta = {variant, env_kw,
expert: {..., final_by when limit_finals}} -- the SFT gate (grl.train.data.check_variant) reads it. Every
record is replayed by grl.train.data.replay_item in the same env before it is returned."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, replace
from functools import partial

import numpy as np

from graphrag.env.policies import trajectory_labels, trajectory_steps
from graphrag.env.reward import reward as spec_reward
from grl.train.data import (ALLOWED_SPLITS, LIMIT_FINAL_REASONS, LIMIT_FINAL_VARIANTS, SplitGuardError, guard_qkeys,
                            item_of, qreal_ids, refuse_qreal, replay_item)
from grl.train.fbeta import chunks_beta, grl_reward

EXPERT_DEPTH = 3
EXPERT_VERSION = 2                       # v2: per-query state cap (expert.max_states) -> beam
MAX_STATES = 20000                       # train.yaml expert.max_states default
STRUCT_MAX_STATES = 1_000_000            # train.yaml expert.struct_max_states default (R38: variant struct only)
STRUCT_WORKERS = 4                       # train.yaml expert.struct_workers default (R40; --workers overrides)
STRUCT_MAX_RSS_GB = 20.0                 # train.yaml expert.struct_max_rss_gb default (R42: per-query RSS GROWTH)
STRUCT_RSS_CEILING_GB = None             # train.yaml expert.struct_rss_ceiling_gb default (R42: absolute; off)
RSS_CHECK_EVERY = 256                    # RSS guard: parents expanded between two RSS reads inside a level


def rss_gb() -> float | None:
    """CURRENT resident set size of this process in GB: Linux /proc/self/statm; else psutil when it is importable
    (not a declared dependency); else None -- the RSS guard is then off (never the monotonic ru_maxrss peak, R42)."""
    import os
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return None


def peak_rss_gb() -> float:
    """Peak RSS of this process (resource.getrusage ru_maxrss: KB on Linux, bytes on macOS) in GB."""
    import resource
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9
CAP_BEAM = 256                           # train.yaml expert.cap_beam default
DROP_EPS = 1e-9                          # SFT train drop rule: expert score <= DROP_EPS


def _actions(s) -> tuple:
    return tuple(h.action for h in s.history)


def expert_search(env, q, gold: frozenset, w, depth: int = EXPERT_DEPTH, prune_zero: bool = True,
                  beam: int | None = None, max_states: int | None = None, cap_beam: int = CAP_BEAM,
                  limit_finals: bool = False, max_rss_gb: float | None = None,
                  rss_ceiling_gb: float | None = None) -> dict:
    """{"final", "score", "n_states", "exact", "depth", "beam", "capped", "capped_reason", "final_by",
    "rss_growth_gb"} (module docstring). RSS guard (R40 / R42, struct): rss0 = the CURRENT process RSS when the
    query starts; before every level and every RSS_CHECK_EVERY expanded parents the current RSS is read, and when
    it has grown by more than max_rss_gb over rss0 (capped_reason "rss") -- or, if rss_ceiling_gb is set, exceeds
    that absolute ceiling (capped_reason "rss_ceiling") -- the query stops expanding the current level and continues
    as the flagged cap beam (capped = True, exact = False; the state cap gives capped_reason "states"). Memory left
    over from an earlier query of the same worker therefore never flags the next one. rss_growth_gb: the largest
    growth seen (None when the guard is off or RSS is unreadable). depth is
    capped at T_max − 1 so a depth-limited state can still STOP. limit_finals (R34): a state ended by k_max / l_max
    is a final as it is, key (−R, steps, action ids) -- a STOP final counts its STOP step; final_by "limit" /
    "stop". Runtime cap (train.yaml expert.max_states): before a level is
    expanded, if the states generated so far plus the level's worst case (live states × non-STOP actions) would
    exceed max_states, this query switches to a beam of cap_beam live states for the rest of the search
    (capped = True; the row is then "beam<cap_beam>", never silently exact)."""
    depth = min(int(depth), env.limits.T_max - 1)
    if depth < 0:
        raise ValueError(f"expert depth {depth} < 0")
    if beam is not None and int(beam) < 1:
        raise ValueError(f"expert beam {beam} < 1")
    if int(cap_beam) < 1 or (max_states is not None and int(max_states) < 1):
        raise ValueError(f"expert cap: max_states {max_states}, cap_beam {cap_beam} must be >= 1")
    t_max, cache = env.limits.T_max, {}

    def R(s) -> float:
        r = cache.get(s.ctx_set)
        if r is None:
            r = cache[s.ctx_set] = float(grl_reward(env.data, q.depth, s.context, gold, w, t_max).total)
        return r

    def key(s, extra: int = 0):
        return (-R(s), len(s.history) + extra, _actions(s))

    best = None

    def offer(s):
        nonlocal best
        if s.done:
            if not (limit_finals and s.reason in LIMIT_FINAL_REASONS):
                return
            k = key(s, 0)                                    # R34: the limit-ended state as it is, no STOP step
        elif env.mask(q, s)[0]:
            k = key(s, 1)                                    # + STOP
        else:
            return
        if best is None or k < best[0]:
            best = (k, s)

    s0 = env.reset(q)
    offer(s0)
    level, seen, n, exact = [s0], {(s0.ctx_set, s0.covered)}, 0, True
    width, capped, fan = (None if beam is None else int(beam)), False, len(env.actions) - 1
    reason = None
    guard = max_rss_gb is not None or rss_ceiling_gb is not None
    rss0 = rss_gb() if guard else None
    growth = 0.0 if rss0 is not None else None

    def over_rss() -> str | None:
        """None, or the capped_reason of a tripped RSS guard (R42: growth over rss0, then the absolute ceiling)."""
        nonlocal growth
        if rss0 is None or beam is not None or capped:
            return None
        cur = rss_gb()
        if cur is None:
            return None
        growth = max(growth, cur - rss0)
        if max_rss_gb is not None and cur - rss0 > float(max_rss_gb):
            return "rss"
        if rss_ceiling_gb is not None and cur > float(rss_ceiling_gb):
            return "rss_ceiling"
        return None

    for _ in range(depth):
        if beam is None and not capped and max_states is not None and n + len(level) * fan > int(max_states):
            width, capped, reason = int(cap_beam), True, "states"
        r = over_rss()
        if r:
            width, capped, reason, exact = int(cap_beam), True, r, False
        if width is not None and len(level) > width:
            level = sorted(level, key=key)[:width]
            exact = False
        children = []
        for j, s in enumerate(level):
            r = over_rss() if j and j % RSS_CHECK_EVERY == 0 else None
            if r:                                            # R40 / R42: stop this level, continue as the cap beam
                width, capped, reason, exact = int(cap_beam), True, r, False
                break
            for a in np.flatnonzero(env.mask(q, s)).tolist():
                if a == 0:
                    continue
                c = env.step(q, s, a, check=False)
                if prune_zero and not c.history[-1].added:
                    continue
                sig = (c.ctx_set, c.covered)
                if sig in seen:
                    continue
                seen.add(sig)
                children.append(c)
        n += len(children)
        for c in children:
            offer(c)
        level = [c for c in children if not c.done]
        if not level:
            break
    if best is None:
        raise ValueError(f"{q.qkey}: no STOP-ended final within depth {depth} (STOP masked everywhere?)")
    by = "limit" if best[1].done else "stop"
    final = best[1] if by == "limit" else env.step(q, best[1], 0)
    return {"final": final, "score": R(final), "n_states": n, "exact": exact, "depth": depth, "beam": width,
            "capped": capped, "capped_reason": reason, "final_by": by,
            "rss_growth_gb": None if growth is None else round(max(growth, 0.0), 4)}


def limit_labels(env, q, final) -> tuple[list[list[int]], list[list[int]]]:
    """(action_equiv, action_masks) of a limit-ended final (R34): graphrag.env.policies.trajectory_labels (which has
    no STOP check; graphrag's _require_stop is in trajectory_json) -- the class and the mask of every recorded
    (retrieval) step; no closing [0]. Refuses a final that is not ended by k_max / l_max or that contains STOP, and
    one whose replay does not end the same way."""
    if not (final.done and final.reason in LIMIT_FINAL_REASONS and final.history
            and all(h.action != 0 for h in final.history)):
        raise ValueError(f"not a limit-ended final (reason {final.reason!r}, actions "
                         f"{[h.action for h in final.history]})")
    equiv, masks = trajectory_labels(env, q, final)      # no STOP check there (only trajectory_json has one)
    s = env.reset(q)
    for h in final.history:
        s = env.step(q, s, h.action, check=False)
    if s.context != final.context or s.reason != final.reason:
        raise ValueError(f"limit_labels: replay ends with {s.reason!r} / {len(s.context)} fragments, the final with "
                         f"{final.reason!r} / {len(final.context)}")
    return equiv, masks


def expert_defaults(variant: str, prune_zero: bool | None, limit_finals: bool | None) -> tuple[bool, bool]:
    """(prune_zero, limit_finals) of a run: an explicit value wins, else R35 -- struct (False, True), every other
    variant (True, False). limit_finals for a variant outside LIMIT_FINAL_VARIANTS is refused (replay_item would
    refuse its rows)."""
    struct = variant in LIMIT_FINAL_VARIANTS
    lf = struct if limit_finals is None else bool(limit_finals)
    if lf and not struct:
        raise ValueError(f"expert --variant {variant}: limit-ended finals are accepted for {LIMIT_FINAL_VARIANTS} "
                         f"only (R34)")
    return (not struct if prune_zero is None else bool(prune_zero)), lf


def reward_spec(w) -> dict:
    """The reward an expert optimised / a run trains on: chunks_beta + the four weights (SFT gate key)."""
    return {"chunks_beta": chunks_beta(w), "weights": {k: float(getattr(w, k)) for k in ("alpha", "beta", "gamma",
                                                                                        "delta")}}


def expert_meta(variant: str, env_kw: dict, res: dict, reached: bool, w, prune_zero: bool, n_steps: int,
                k_max: int | None = None, final_by: str | None = None) -> dict:
    """k_max: the env K_max searched under (ruling R12; the SFT gate compares it to the run's). final_by ("stop" /
    "limit", R34): recorded only by a search with limit_finals (None: no field, the pre-R34 bytes)."""
    return {"variant": variant, "env_kw": dict(env_kw),
            "expert": {"version": EXPERT_VERSION, **({} if k_max is None else {"K_max": int(k_max)}),
                       "search": "exact" if res["exact"] else f"beam{int(res['beam'])}",
                       "capped": bool(res.get("capped")), "depth": res["depth"], "prune_zero": bool(prune_zero),
                       "score": round(res["score"], 9), "reached": bool(reached), "len": int(n_steps),
                       "n_states": int(res["n_states"]), **reward_spec(w),
                       **({"capped_reason": res["capped_reason"]}
                          if str(res.get("capped_reason")).startswith("rss") else {}),
                       **({} if final_by is None else {"final_by": str(final_by)})}}


def expert_pair(env, ranks, p, w, spec_w, variant: str, env_kw: dict, depth: int = EXPERT_DEPTH,
                prune_zero: bool = True, beam: int | None = None, max_states: int | None = MAX_STATES,
                cap_beam: int = CAP_BEAM, limit_finals: bool = False, max_rss_gb: float | None = None,
                rss_ceiling_gb: float | None = None):
    """(pair with the variant-expert §8.1 trajectory, info). The record is replay-checked (replay_item: mask,
    equivalence class, STOP, reach flag, reward parts) in `env` before it is returned."""
    t0 = time.perf_counter()
    q = env.query(ranks.get(f"{p.id}|q"))
    gold = frozenset(int(env.data.chunk_row[c]) for c in p.C_ref)
    res = expert_search(env, q, gold, w, depth, prune_zero, beam, max_states, cap_beam, limit_finals, max_rss_gb,
                        rss_ceiling_gb)
    s = res["final"]
    equiv, masks = limit_labels(env, q, s) if res["final_by"] == "limit" else trajectory_labels(env, q, s)
    parts = spec_reward(env.data, q.depth, s.context, gold, spec_w, env.limits.T_max)
    reached = gold <= s.ctx_set
    prov = {**(p.provenance or {}), "meta": expert_meta(variant, env_kw, res, reached, w, prune_zero, len(s.history),
                                                              env.limits.K_max,
                                                              res["final_by"] if limit_finals else None)}
    out = replace(p, trajectory=trajectory_steps(env, s), actions=list(_actions(s)), action_equiv=equiv,
                  action_masks=masks, reward=asdict(parts), provenance=prov)
    _, s2 = replay_item(env, ranks, item_of(out, env.data))
    if s2.context != s.context:
        raise RuntimeError(f"{p.id}: expert replay differs from the search final")
    info = {"id": p.id, "split": p.split, "score": res["score"], "reached": reached, "n_states": res["n_states"],
            "exact": res["exact"], "len": len(s.history), "first": action_name(env.actions[s.history[0].action]),
            "ops": [env.actions[h.action].op for h in s.history], "n_ctx": len(s.context), "tokens": int(s.length),
            "recall": len(gold & s.ctx_set) / len(gold), "capped": bool(res["capped"]),
            "actions": list(_actions(s)), "seconds": round(time.perf_counter() - t0, 4)}
    if limit_finals:                                        # R34 / R40 (struct): final kind, cap reason, memory
        import os
        info.update(final_by=res["final_by"], capped_reason=res["capped_reason"], pid=os.getpid(),
                    rss_peak_gb=round(peak_rss_gb(), 3), rss_growth_gb=res["rss_growth_gb"])
    return out, info


def action_name(a) -> str:
    if a.op in ("structure", "amendments"):       # M7 struct variant (+ Task 6d-b amend_op): one slot per m
        return f"{a.op}:{a.m}"
    return f"{a.op}:{a.r}:{a.m}" if a.op == "graph" else f"{a.op}:{a.k}" if a.k is not None else a.op


def expert_context(rankings: str, variant: str, env_extra: dict, env_cfg: dict | None = None) -> dict:
    """graphrag.env.parallel.run `init` (spawn-safe: EnvData and rankings load once per worker; struct: + the STRUCT
    rows, grl.model.variants.load_variant_data). env_cfg: the run's train.yaml `env` incl. --set env.K_max (R12);
    None = the env reloads train.yaml."""
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.variants import data_kw, env_spec, load_variant_data
    data = load_variant_data(variant, **data_kw(env_extra))
    cls, kw = env_spec(variant)
    return {"env": cls(data, env_cfg, None, **kw, **env_extra),
            "ranks": load_rankings(rankings, graph_version=data.version)}


def expert_one(ctx, p):
    out, info = expert_pair(ctx["env"], ctx["ranks"], p, ctx["w"], ctx["spec_w"], ctx["variant"], ctx["env_kw"],
                            ctx["depth"], ctx["prune_zero"], ctx["beam"], ctx["max_states"], ctx["cap_beam"],
                            bool(ctx.get("limit_finals", False)), ctx.get("max_rss_gb"), ctx.get("rss_ceiling_gb"))
    return out.to_json(), info


def expert_inputs(path) -> list:
    """train + val pairs of a snapshot file (any trajectory content); Q-real rows refused loudly, test rows dropped."""
    from graphrag.dataset.finalize import read_pairs
    from grl.train.data import refuse_probe, refuse_test_eval
    refuse_test_eval(path)
    refuse_probe(path)
    ids, out = qreal_ids(), []
    for p in read_pairs(path):
        refuse_qreal(path, p, ids)
        refuse_test_eval(path, p)
        if p.split in ALLOWED_SPLITS:
            out.append(p)
        elif p.split != "test":
            raise SplitGuardError(f"{path}: row {p.id} has unknown split {p.split!r}")
    guard_qkeys([f"{p.id}|q" for p in out])
    return sorted(out, key=lambda p: p.id)


def _mean(xs) -> float:
    return round(float(np.mean(xs)), 6) if len(xs) else 0.0


def sft_drop(score, reached: bool, n_steps: int, final_by: str | None = None) -> bool:
    """The default SFT-train drop rule for an expert row: score <= DROP_EPS, an immediate STOP (length 1 on a STOP
    final; a one-step limit final is a retrieval, R34), or C* not reached (train.py sft --keep-unreached keeps
    them)."""
    return float(score) <= DROP_EPS or (int(n_steps) == 1 and final_by != "limit") or not bool(reached)


def expert_stats(infos) -> dict:
    out = {}
    for split in ALLOWED_SPLITS:
        rows = [i for i in infos if i["split"] == split]
        out[split] = {"n": len(rows), "score_mean": _mean([i["score"] for i in rows]),
                      "recall_mean": _mean([i["recall"] for i in rows]),
                      "reached_share": _mean([float(i["reached"]) for i in rows]),
                      "exact_share": _mean([float(i["exact"]) for i in rows]),
                      "capped_share": _mean([float(i["capped"]) for i in rows]),
                      "score0_share": _mean([float(i["score"] <= DROP_EPS) for i in rows]),
                      "len1_share": _mean([float(i["len"] == 1 and i.get("final_by") != "limit") for i in rows]),
                      "drop_share": _mean([float(sft_drop(i["score"], i["reached"], i["len"], i.get("final_by")))
                                           for i in rows]),
                      "seconds_mean": _mean([i["seconds"] for i in rows]),
                      "seconds_max": round(max([i["seconds"] for i in rows], default=0.0), 4),
                      "n_states_max": max([i["n_states"] for i in rows], default=0),
                      "ep_len_mean": _mean([i["len"] for i in rows]),
                      "n_ctx_mean": _mean([i["n_ctx"] for i in rows]),
                      "tokens_mean": _mean([i["tokens"] for i in rows]),
                      "n_states_mean": _mean([i["n_states"] for i in rows]),
                      "len_hist": {str(k): v for k, v in sorted(Counter(i["len"] for i in rows).items())},
                      "first_action": {str(k): v for k, v in sorted(Counter(i["first"] for i in rows).items())},
                      "first_op": dict(sorted(Counter(i["ops"][0] for i in rows).items())),
                      "op_hist": dict(sorted(Counter(o for i in rows for o in i["ops"]).items()))}
        if any("final_by" in i for i in rows):              # R34 searches only (pre-R34 stats bytes unchanged)
            out[split]["limit_final_share"] = _mean([float(i.get("final_by") == "limit") for i in rows])
            out[split]["n_states_median"] = float(np.median([i["n_states"] for i in rows])) if rows else 0.0
            out[split]["n_states_peak"] = max([i["n_states"] for i in rows], default=0)
            out[split]["capped_rss"] = sum(1 for i in rows if str(i.get("capped_reason")).startswith("rss"))
            out[split]["rss_growth_gb_max"] = max([i.get("rss_growth_gb") or 0.0 for i in rows], default=0.0)
            out[split]["rss_peak_gb_per_worker"] = rss_per_worker(rows)
    return out


def rss_per_worker(rows) -> dict:
    """R40: {worker pid: peak RSS GB} (ru_maxrss after each query of that worker; the max over its queries)."""
    out: dict = {}
    for i in rows:
        if "pid" in i:
            k = str(i["pid"])
            out[k] = max(out.get(k, 0.0), float(i["rss_peak_gb"]))
    return dict(sorted(out.items()))


def expert_config(cfg: dict, variant: str | None = None) -> dict:
    """train.yaml `expert:` (depth, max_states, cap_beam) with the module defaults. variant struct (R38): max_states
    = expert.struct_max_states (the unpruned search needs a larger cap); every other variant: expert.max_states.
    struct also gets workers (expert.struct_workers, the train.py expert default; --workers overrides) and
    max_rss_gb (expert.struct_max_rss_gb: the per-query RSS GROWTH guard of expert_search) and rss_ceiling_gb
    (expert.struct_rss_ceiling_gb: an optional absolute RSS ceiling, null = off), R40 / R42."""
    e = {"depth": EXPERT_DEPTH, "max_states": MAX_STATES, "cap_beam": CAP_BEAM, "struct_max_states": STRUCT_MAX_STATES,
         "struct_workers": STRUCT_WORKERS, "struct_max_rss_gb": STRUCT_MAX_RSS_GB,
         "struct_rss_ceiling_gb": STRUCT_RSS_CEILING_GB}
    e.update({k: v for k, v in (cfg.get("expert") or {}).items() if v is not None or k == "struct_rss_ceiling_gb"})
    struct = variant in LIMIT_FINAL_VARIANTS
    if struct:
        e["max_states"] = e["struct_max_states"]
    out = {"depth": int(e["depth"]), "max_states": None if e["max_states"] in (None, 0) else int(e["max_states"]),
           "cap_beam": int(e["cap_beam"])}
    if struct:                                          # R40: fewer workers and a per-query RSS guard
        out.update(workers=int(e["struct_workers"]), max_rss_gb=float(e["struct_max_rss_gb"]),
                   rss_ceiling_gb=None if e["struct_rss_ceiling_gb"] is None else float(e["struct_rss_ceiling_gb"]))
    return out


def _run_mode(pairs, rankings, variant, env_kw, cfg, workers, start_method, depth, prune_zero, beam, ec,
              limit_finals: bool = False):
    from graphrag.env.parallel import run
    from graphrag.env.reward import weights_from_config
    from grl.train.fbeta import grl_weights
    t0 = time.time()
    res = run(expert_one, pairs, int(workers), start_method,
              init=partial(expert_context, rankings, variant, env_kw, env_cfg=cfg["env"]), w=grl_weights(cfg),
              spec_w=weights_from_config(cfg), variant=variant, env_kw=env_kw, depth=int(depth),
              prune_zero=bool(prune_zero), beam=None if beam is None else int(beam), max_states=ec["max_states"],
              cap_beam=ec["cap_beam"], limit_finals=bool(limit_finals), max_rss_gb=ec.get("max_rss_gb"),
              rss_ceiling_gb=ec.get("rss_ceiling_gb"))
    return res, round(time.time() - t0, 1)


def run_expert(pairs_path, rankings: str, variant: str, out_path, cfg: dict, workers: int,
               start_method: str | None = None, depth: int | None = None, prune_zero: bool | None = None,
               beam: int | None = None, probe_n: int | None = None, limit_finals: bool | None = None) -> dict:
    """Variant-expert trajectories of every train + val pair -> out_path (write_pairs: sorted by id) and
    <out stem>.stats.json. Refuses `require` variants (the expert would have to search the hybrid mask's forced
    families; not needed for the adaptive / fused finals). prune_zero / limit_finals None: the variant default
    (expert_defaults, R35: struct unpruned with limit finals; others pruned, STOP finals only).
    probe_n: runtime / pruning probe instead -- the first probe_n TRAIN pairs (by id) are searched twice, with
    zero-Δ pruning and without (--keep-zero); out_path receives a JSON report (per mode: n_states mean / max,
    seconds total / mean / max per query, score_mean, capped share; the number of queries whose best trajectory or
    score differs between the modes), no trajectories are written."""
    from pathlib import Path

    from graphrag.dataset.finalize import Pair, write_pairs
    from grl.hashing import sha256_file
    from grl.model.variants import VARIANTS, variant_env_kw
    from grl.train.fbeta import grl_weights
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if VARIANTS[variant].get("require"):
        raise ValueError(f"expert --variant {variant}: `require` variants are not supported")
    prune_zero, limit_finals = expert_defaults(variant, prune_zero, limit_finals)
    ec = expert_config(cfg, variant)
    depth = ec["depth"] if depth is None else int(depth)
    env_kw = variant_env_kw(variant, cfg)          # rerank: the candidate-list kwargs (no ce: labels ignore it)
    pairs = expert_inputs(pairs_path)
    if not pairs:
        raise ValueError(f"{pairs_path}: no train / val pairs")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    common = {"variant": variant, "env_kw": env_kw, "K_max": int(cfg["env"]["K_max"]), "depth": depth, "beam": beam,
              "workers": int(workers),
              **{k: ec[k] for k in ("max_states", "cap_beam", "max_rss_gb", "rss_ceiling_gb") if k in ec},
              **reward_spec(grl_weights(cfg)),
              "inputs": {"pairs": str(pairs_path), "pairs_sha256": sha256_file(pairs_path),
                         "rankings": str(rankings)}, **({"limit_finals": True} if limit_finals else {})}
    if probe_n is not None:
        sub = [p for p in pairs if p.split == "train"][:int(probe_n)]
        if not sub:
            raise ValueError("probe: no train pairs")
        modes = {}
        for name, pz in (("prune_zero", True), ("keep_zero", False)):
            res, sec = _run_mode(sub, rankings, variant, env_kw, cfg, workers, start_method, depth, pz, beam, ec,
                                 limit_finals)
            infos = [i for _, i in res]
            modes[name] = {"seconds": sec, "rows": infos, **expert_stats(infos)["train"]}
        a, b = modes["prune_zero"]["rows"], modes["keep_zero"]["rows"]
        pq = ("score", "actions", "n_states", "seconds", "capped") + (("final_by",) if limit_finals else ())
        rep = {**common, "probe_n": len(sub), "modes": {k: {x: y for x, y in v.items() if x != "rows"}
                                                         for k, v in modes.items()},
               "n_best_differs": sum(x["actions"] != y["actions"] for x, y in zip(a, b)),
               "n_score_differs": sum(abs(x["score"] - y["score"]) > DROP_EPS for x, y in zip(a, b)),
               "keep_zero_better": sum(y["score"] > x["score"] + DROP_EPS for x, y in zip(a, b)),
               "per_query": [{"id": x["id"], "prune_zero": {k: x[k] for k in pq}, "keep_zero": {k: y[k] for k in pq}}
                             for x, y in zip(a, b)]}
        out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return rep
    res, sec = _run_mode(pairs, rankings, variant, env_kw, cfg, workers, start_method, depth, prune_zero, beam, ec,
                         limit_finals)
    write_pairs(out_path, [Pair.from_json(r) for r, _ in res])
    stats = {**common, "prune_zero": bool(prune_zero), "seconds": sec, "out_sha256": sha256_file(out_path),
             **expert_stats([i for _, i in res])}
    stats_path = out_path.with_name(out_path.stem + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return stats
