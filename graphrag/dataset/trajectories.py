"""Expert trajectories (IMPLEMENTATION.md §8.1-8.2; П5; M3-R6): the oracle -- not the walk -- finds the trajectory;
the best final AMONG those that end with STOP and reach all of C_ref, then R >= min_reward. Every accepted trajectory
is replayed action by action from reset (same STOP, same context, C_ref reached, same R) before it is written, with
the per-step equivalence class and mask (Review Focus 3). `select` fills exact split × pattern × language cells with
<= max_per_doc_cell pairs per document and <= ceil(max_template_share · n) per question opening; shortfalls are
reported, never backfilled (M3-R3 пп. 1, 7)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, replace

from graphrag.dataset.finalize import template
from graphrag.dataset.seeds import cell_counts
from graphrag.env.policies import beam_oracle, d5_accept
from graphrag.env.reward import reward


class TrajectoryReplayError(RuntimeError):
    pass


def expert_trajectory(env, q, gold: frozenset, w, min_reward: float, width: int | None = None):
    s = beam_oracle(env, q, gold, w, width, accept=d5_accept(gold))
    if s is None:
        return None, None, "no_stop_full_recall"
    parts = reward(env.data, q.depth, s.context, gold, w, env.limits.T_max)
    return s, parts, ("ok" if parts.total >= min_reward else "below_threshold")


def replay(env, q, actions: list[int], gold: frozenset, w, min_reward: float):
    """M3-R3 п. 3: re-run `actions` from reset with the mask checked at every step; the episode must end with STOP on
    the last action, reach all of `gold` and give R >= min_reward. Returns (final state, RewardParts)."""
    s = env.reset(q)
    for i, a in enumerate(actions):
        if s.done:
            raise TrajectoryReplayError(f"episode ended ({s.reason}) before action {i}")
        try:
            s = env.step(q, s, int(a))
        except Exception as e:                                  # InvalidAction: masked / foreign action
            raise TrajectoryReplayError(f"action {i} ({a}) invalid on replay: {e}") from e
    if not (s.done and s.reason == "stop" and actions and actions[-1] == 0):
        raise TrajectoryReplayError(f"replay does not end with STOP (reason {s.reason!r})")
    if not gold <= s.ctx_set:
        raise TrajectoryReplayError(f"replay misses C_ref rows {sorted(gold - s.ctx_set)}")
    parts = reward(env.data, q.depth, s.context, gold, w, env.limits.T_max)
    if parts.total < min_reward:
        raise TrajectoryReplayError(f"replay R {parts.total} < {min_reward}")
    return s, parts


def traj_one(ctx, p):
    """parallel.run worker (Task 13): (pair with its §8.1 trajectory when "ok", reason, R | None, first ops, cell).
    ctx: env, ranks (query key "<pair id>|q"), w, thr, width."""
    from graphrag.env.policies import trajectory_labels, trajectory_steps
    env = ctx["env"]
    q = env.query(ctx["ranks"].get(f"{p.id}|q"))
    gold = frozenset(int(env.data.chunk_row[c]) for c in p.C_ref)
    cell = f"{p.split}|{p.pattern}|{p.lang}"
    s, parts, why = expert_trajectory(env, q, gold, ctx["w"], ctx["thr"], ctx.get("width"))
    if s is None:
        return p, why, None, [], cell
    ops = [env.actions[h.action].op for h in s.history]
    if why != "ok":
        return p, why, parts.total, ops, cell
    actions = [h.action for h in s.history]
    s2, parts2 = replay(env, q, actions, gold, ctx["w"], ctx["thr"])
    if s2.context != s.context or parts2 != parts:
        raise TrajectoryReplayError(f"{p.id}: replay differs from the oracle's final")
    equiv, masks = trajectory_labels(env, q, s)
    return (replace(p, trajectory=trajectory_steps(env, s), actions=actions, action_equiv=equiv, action_masks=masks,
                    reward=asdict(parts)), why, parts.total, ops, cell)


def final_needs(size: int, hops, ratios, langs) -> dict[str, int]:
    """The D7 final need per cell "pattern|split|lang", allocated pattern -> split -> language exactly as
    scripts/build_dataset.py cmd_seeds (Task 14) does, so seeds and selection agree on every cell (review I3)."""
    out = {}
    for pat in hops:
        for sp, m in cell_counts(round(size * hops[pat]), ratios).items():
            for lang, n in cell_counts(m, langs).items():
                out[f"{pat}|{sp}|{lang}"] = n
    return out


def select(pairs, size: int, ratios, hops, langs, seed, max_per_doc_cell: int,
           max_template_share: float, transfer: dict | None = None,
           report: dict | None = None) -> tuple[list, dict[str, int]]:
    """Exact cells (final_needs), <= max_per_doc_cell per document and <= ceil(share · n) per opening per cell, order
    sha256(seed:id), no backfill. With `transfer` (train.yaml dataset.transfer, ruling M3-R15): in transfer.splits the
    `from` pattern (amends) takes what passes and its shortfall moves to the `to` pattern (refers_internal) of the same
    split, same language first (graphrag.dataset.seeds.effective_needs); train stays exact. `report` (a dict) receives
    final / effective needs and the transfer actually applied. Returns (chosen sorted by id, shortfall per cell)."""
    from graphrag.dataset.seeds import effective_needs
    order = sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}:{p.id}".encode()).hexdigest())
    by_cell: dict[str, list] = {}
    for p in order:
        by_cell.setdefault(f"{p.pattern}|{p.split}|{p.lang}", []).append(p)

    def take(cell: str, want: int) -> list:
        cap_t = max(1, math.ceil(max_template_share * want))
        per_doc, per_t, out = Counter(), Counter(), []
        for p in by_cell.get(cell, []):
            if len(out) >= want:
                break
            if any(per_doc[d] >= max_per_doc_cell for d in p.docs) or per_t[template(p.q)] >= cap_t:
                continue
            out.append(p)
            per_doc.update(p.docs)
            per_t[template(p.q)] += 1
        return out

    def fit(cell: str, hi: int) -> int:
        """The largest w <= hi the cell can supply at the template cap of w itself (the cap depends on w)."""
        for w in range(hi, -1, -1):
            if len(take(cell, w)) == w:
                return w
        return 0

    final = final_needs(size, hops, ratios, langs)
    need, applied = dict(final), {}
    if transfer:
        sps, src, dst = tuple(transfer["splits"]), transfer["from"], transfer["to"]
        got = {}
        for k, n in final.items():
            pat, sp, lang = k.split("|")
            if sp in sps and pat == src:
                got[k] = fit(k, n)                                          # what passes, <= the final need
            elif sp in sps and pat == dst:
                got[k] = fit(k, n + sum(v for kk, v in final.items() if kk.startswith(f"{src}|{sp}|")))
            else:
                got[k] = n
        for _ in range(len(final) + 1):                   # re-review m3: settle needs feasible at their own cap
            need, applied, _ = effective_needs(final, got, sps, src, dst)
            bad = [k for k in need if k.split("|")[1] in sps and k.split("|")[0] == dst
                   and len(take(k, need[k])) < need[k]]
            if not bad:
                break
            for k in bad:
                got[k] = fit(k, need[k] - 1)
    chosen, short = [], {}
    for k in sorted(need):
        t = take(k, need[k])
        if len(t) < need[k]:
            short[k] = need[k] - len(t)
        chosen += t
    if report is not None:
        report.update({"final_need_by_cell": final, "effective_need_by_cell": need,
                       "transfer": {"rule": dict(transfer) if transfer else None, "applied": applied,
                                    "total": sum(applied.values())}})
    return sorted(chosen, key=lambda p: p.id), short


def traj_context(rankings: str | None = None) -> dict:
    """parallel.run `init` (spawn: EnvData and rankings load once per worker, not pickled): env, ranks, w."""
    from graphrag.env.data import load_env_data
    from graphrag.env.mdp import Env
    from graphrag.env.reward import weights_from_config
    from graphrag.retrieval.rankings import DATASET_LATEST, load_rankings
    data = load_env_data()
    return {"env": Env(data), "ranks": load_rankings(rankings or DATASET_LATEST, graph_version=data.version),
            "w": weights_from_config()}
