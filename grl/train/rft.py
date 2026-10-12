"""Task GRPO-X E1 -- rejection-sampling fine-tuning / expert iteration on the policy's own samples (grpo research
report §7 B1). Two gold-separated halves:
- sample_best: per TRAIN query one greedy episode and G sampled episodes of the policy (grl.train.rollout.rollout:
  the policy sees StateDesc only); the gold set enters only afterwards, through the env rewards of the finished
  episodes (exactly as GRPO's collect). The best sample (max R, then fewer steps, then the lower copy) is kept when
  R_best > R_greedy + margin. Only train rows are accepted (val selects models, test is sealed).
- rft_examples: the kept action sequences replayed from reset into SFT examples (state description + the class of
  equivalent actions, graphrag.env.policies.action_classes: same ΔC and covered set -- gold-free), checked against
  the recorded episode (every action allowed by the mask; the episode ends exactly at its last action)."""
from __future__ import annotations

import math

import numpy as np
import torch

from grl.train.data import SFTExample, SplitGuardError, guard_qkeys
from grl.train.evaluate import episode_parts
from grl.train.rollout import rollout

RFT_SPLITS = ("train",)


def _check_train(items) -> None:
    bad = sorted({it.split for it in items if it.split not in RFT_SPLITS})
    if bad:
        raise SplitGuardError(f"RFT samples train rows only; refused splits {bad} (val selects, test is sealed)")
    guard_qkeys([it.qkey for it in items])


def _best(parts, eps) -> int:
    """Index of the best episode: max R, then fewer steps, then the lower copy (deterministic)."""
    return min(range(len(eps)), key=lambda j: (-parts[j].total, len(eps[j].actions), eps[j].copy))


@torch.no_grad()
def sample_best(policy, graph, vec, ranks, items, G: int, gen, w, device, margin: float = 0.05,
                chunk: int = 16) -> list[dict]:
    """One row per train item: greedy R, sampled mean / best R, the best sample's actions and keep (R_best > R_greedy
    + margin). The vec env's rewards read its current episodes, so each rollout is scored before the next one."""
    _check_train(items)
    if G < 1:
        raise ValueError(f"RFT needs G >= 1 samples per query (got {G})")
    policy.eval()
    golds = {it.qkey: it.gold for it in items}
    out = []
    for b in range(0, len(items), int(chunk)):
        its = items[b:b + int(chunk)]
        qk = [it.qkey for it in its]
        ge, gf = rollout(policy, graph, vec, ranks, qk, 1, greedy=True, device=device)
        gp = episode_parts(vec, ge, gf, golds, w)
        se, sf = rollout(policy, graph, vec, ranks, qk, G, greedy=False, gen=gen, device=device)
        sp = episode_parts(vec, se, sf, golds, w)
        for i, it in enumerate(its):
            eps, parts = se[i * G:(i + 1) * G], sp[i * G:(i + 1) * G]
            j = _best(parts, eps)
            rb, rg = float(parts[j].total), float(gp[i].total)
            out.append({"id": it.pid, "qkey": it.qkey, "split": it.split, "pattern": it.pattern,
                        "actions": [int(a) for a in eps[j].actions], "greedy_actions": [int(a) for a in ge[i].actions],
                        "r_best": round(rb, 9), "r_greedy": round(rg, 9),
                        "r_mean": round(float(np.mean([p.total for p in parts])), 9),
                        "n_best": sum(1 for p in parts if math.isclose(p.total, rb, abs_tol=1e-12)),
                        "context": sorted(int(c) for c in sf[i * G + j].ctx_set),
                        "keep": rb > rg + float(margin)})
    return out


def rft_examples(env, ranks, rows, describe_kw: dict) -> list[SFTExample]:
    """SFT examples of the kept rows (row["keep"]) by replaying their actions from reset; the class of the taken action
    is its M3 action class at the replayed state. ValueError when a replay leaves the recorded episode."""
    from graphrag.env.policies import action_classes
    from grl.model.observe import describe
    out = []
    for r in rows:
        if not r.get("keep"):
            continue
        if r.get("split") not in RFT_SPLITS:
            raise SplitGuardError(f"RFT row {r.get('id')}: split {r.get('split')!r} is not train")
        q = env.query(ranks.get(r["qkey"]))
        s = env.reset(q)
        acts = [int(a) for a in r["actions"]]
        for t, a in enumerate(acts):
            if s.done:
                raise ValueError(f"RFT row {r['id']} t={t + 1}: episode ended ({s.reason!r}) before action {a}")
            if not bool(env.mask(q, s)[a]):
                raise ValueError(f"RFT row {r['id']} t={t + 1}: action {a} is masked in the replayed state")
            cls = next((c for c in action_classes(env, q, s) if a in c), None)
            if cls is None:
                raise ValueError(f"RFT row {r['id']} t={t + 1}: no action class holds {a}")
            out.append(SFTExample(r["qkey"], t + 1, describe(env, q, s, **describe_kw), tuple(int(x) for x in cls)))
            s = env.step(q, s, a)
        if not s.done:
            raise ValueError(f"RFT row {r['id']}: the replay did not end after its {len(acts)} actions")
        if "context" in r and sorted(int(c) for c in s.ctx_set) != list(r["context"]):
            raise ValueError(f"RFT row {r['id']}: replayed context differs from the sampled one")
    guard_qkeys([e.qkey for e in out])
    return out
