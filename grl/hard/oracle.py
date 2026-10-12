"""M7 Task 8, rulings R28 / R29: DIAGNOSTIC oracles for the probe (not methods; they read C*). They attribute the gap
between the R-max struct expert (grl.train.expert.expert_search) and the all@5 the environment can reach.

Search: exact breadth-first enumeration of the env's valid non-STOP actions up to `depth` retrieval steps. Unlike the
expert, a state ended by a limit (k_max, l_max, t_max) or by "exhausted" is a final as it is (no STOP step); a live
state is a final through its STOP completion when STOP is valid. Two switches:
- prune: drop a child whose step added no fragment (zero-Δ), as expert_search(prune_zero=True). In StructEnv such a
  step can still be a bridge (it selects vertices, grows `covered` and opens graph / structure candidates), so a
  pruned search is NOT a reachability bound (R29);
- merge key: (ctx_set, covered) as the expert, or (ordered context, covered). Both are exact: the redundancy term of
  Δ_V (graphrag.retrieval.executor.delta_v, struct_delta_v) is a max over the context SET, so the order of C changes
  no later step and the set key is already an exact merge (R30). The ordered key is a finer merge -- still exact, but
  it keeps more states (the depth-7 pool OOM was partly that); plan 2 can rerun the unpruned modes on the set key.
Modes (MODES):
- "kmax" (oracle_kmax): prune, (ctx_set, covered), max R -- the expert plus limit-ended finals;
- "noprune" (oracle_noprune): no prune, ordered key, max R -- what the expert's prune_zero costs;
- "all5" (oracle_all5): no prune, ordered key, max all@K (gold ⊆ context), then recall, then R -- the reachability
  upper bound of all@K / recall@K under the env's actions, K_max and depth.
Ties: then fewer steps, then the lexicographically smaller action ids. max_states is a hard guard (ValueError) and
`deadline` (time.monotonic()) another one (TimeoutError), so a returned result is always exact."""
from __future__ import annotations

import time

import numpy as np

from graphrag.env.mdp import InvalidAction
from grl.train.fbeta import grl_reward

MODES = {"kmax": {"prune": True, "ordered": False, "target": "R"},
         "noprune": {"prune": False, "ordered": True, "target": "R"},
         "all5": {"prune": False, "ordered": True, "target": "all"}}


def diag_search(env, q, gold: frozenset, w, depth: int, mode: str, max_states: int | None = 5_000_000,
                deadline: float | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"diag_search mode {mode!r} is not one of {sorted(MODES)}")
    if not gold:
        raise ValueError("diag_search needs a non-empty gold")
    spec = MODES[mode]
    depth = min(int(depth), env.limits.T_max - 1)
    t_max, cache = env.limits.T_max, {}

    def R(s) -> float:
        r = cache.get(s.ctx_set)
        if r is None:
            r = cache[s.ctx_set] = float(grl_reward(env.data, q.depth, s.context, gold, w, t_max).total)
        return r

    def key(s, n_steps: int):
        acts = tuple(h.action for h in s.history)
        if spec["target"] == "R":
            head = (-R(s),)
        else:
            hit = len(gold & s.ctx_set)
            head = (-float(hit == len(gold)), -hit / len(gold), -R(s))
        return head + (n_steps, acts)

    def sig(s):
        return (s.context if spec["ordered"] else s.ctx_set, s.covered)

    best = None

    def offer(s):
        nonlocal best
        if s.done:
            k = key(s, len(s.history))                       # limit / exhausted: final as it is
        elif env.mask(q, s)[0]:
            k = key(s, len(s.history) + 1)                   # + STOP
        else:
            return
        if best is None or k < best[0]:
            best = (k, s)

    s0 = env.reset(q)
    offer(s0)
    level, seen, n, n_zero = [s0], {sig(s0)}, 0, 0
    for _ in range(depth):
        children = []
        for s in level:
            for a in np.flatnonzero(env.mask(q, s)).tolist():
                if a == 0:
                    continue
                try:
                    c = env.step(q, s, a, check=False)
                except InvalidAction:                        # a masked-valid step with no effect: not a child
                    continue
                if not c.history[-1].added:
                    n_zero += 1
                    if spec["prune"]:
                        continue
                g = sig(c)
                if g in seen:
                    continue
                seen.add(g)
                children.append(c)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"{q.qkey}: diag_search past its deadline at depth {depth}")
        n += len(children)
        if max_states is not None and n > int(max_states):
            raise ValueError(f"{q.qkey}: diag_search over {max_states} states at depth {depth}")
        for c in children:
            offer(c)
        level = [c for c in children if not c.done]
        if not level:
            break
    if best is None:
        raise ValueError(f"{q.qkey}: no final within depth {depth}")
    s = best[1]
    final = s if s.done else env.step(q, s, 0)
    return {"final": final, "score": R(final), "n_states": n, "n_zero_steps": n_zero, "exact": True, "depth": depth,
            "beam": None, "capped": False, "reason": final.reason, "mode": mode}


# ---- parallel workers (graphrag.env.parallel.run; spawn-safe, EnvData once per worker) ---------------------------

def diag_context(rankings: str, pairs: str, env_cfg: dict, cfg: dict) -> dict:
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.probe import load_probe_items
    from grl.model.variants import StructEnv, struct_data_for, struct_kw
    from grl.train.fbeta import grl_weights
    data = struct_data_for(cfg)                      # struct.articles (Task 6c fix round 1)
    return {"env": StructEnv(data, env_cfg, **struct_kw(cfg)), "ranks": load_rankings(rankings, graph_version=data.version),
            "items": {it.pid: it for it in load_probe_items(pairs, data)}, "w": grl_weights(cfg)}


def diag_one(ctx, job):
    """job = (qid, mode, depth, ext_targets, max_states, deadline_wall): the oracle_record of one search, or
    {"qid", "error"} when a guard (states / deadline) stopped it."""
    from grl.hard.metrics import oracle_record
    qid, mode, depth, ext, max_states, deadline_wall = job
    env, it = ctx["env"], ctx["items"][qid]
    q = env.query(ctx["ranks"].get(it.qkey))
    t0 = time.perf_counter()
    dl = None if deadline_wall is None else time.monotonic() + max(0.0, deadline_wall - time.time())
    try:
        res = diag_search(env, q, it.gold, ctx["w"], depth, mode, max_states, dl)
    except (ValueError, TimeoutError) as e:
        return {"qid": qid, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 3)}
    rec = oracle_record(env, res, ext)
    rec.update(qid=qid, seconds=round(time.perf_counter() - t0, 4), depth=int(res["depth"]), reason=res["reason"],
               n_zero_steps=int(res["n_zero_steps"]))
    return rec
