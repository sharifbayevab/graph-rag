"""Stratified end-to-end pilot (M3 Task 17, decisions П2/П4/П5, M3-R3, M3-R6, M3-R10, M3-R13/R15/R18): yield per
pattern, LLM calls per accepted pair per pass (ds_facts / ds_generate / ds_verify), the R distribution, and the
projected calls of the full run within the 2500 cap.

`project` is the per-accepted-pair projection of the brief (seeds / overshoot per pattern). It scales ds_facts per pair,
which over-counts (a fact block is paid once whatever the yield; pre-pilot re-review I3), so the П2 gate of
`scripts/build_dataset.py pilot-report` uses the block-based `graphrag.dataset.budget.project_calls` for the call total
and keeps `project` for the seeds / overshoot per pattern and as a reported upper bound."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from graphrag.dataset.seeds import cell_counts


def pilot_seeds(seeds, per_cell: int, seed) -> list:
    """`per_cell` seeds per (pattern, language), spread over the splits in proportion to the seeds available there
    (cell_counts), each split's pick in sha256(seed:pilot|seed_id) order; independent of the input order."""
    by = defaultdict(lambda: defaultdict(list))
    for s in sorted(seeds, key=lambda s: hashlib.sha256(f"{seed}:pilot|{s.seed_id}".encode()).hexdigest()):
        by[(s.pattern, s.lang)][s.split].append(s)
    out = []
    for cell, splits in sorted(by.items()):
        want = cell_counts(per_cell, {sp: len(v) for sp, v in sorted(splits.items())})
        for sp, n in want.items():
            out += splits[sp][:n]
    return sorted(out, key=lambda s: s.seed_id)


def pass_calls(before: dict, after: dict) -> dict[str, int]:
    """API calls per dataset pass between two ledger snapshots {pass: {date: calls}}."""
    tot = lambda led, p: sum(int(v) for v in (led.get(p) or {}).values())
    return {p: tot(after, p) - tot(before, p) for p in ("ds_facts", "ds_generate", "ds_verify")}


def project(per_accepted: dict[str, float], accepted_rate: dict[str, float], size: int, hops: dict, margin: float,
            spent: int) -> dict:
    """Seeds of pattern p = ceil(size · hops_p · margin / yield_p); calls = Σ_pass ceil(accepted_needed · per_accepted)
    + already spent (an upper bound for ds_facts, see the module docstring)."""
    need = round(size * margin)
    calls = {p: math.ceil(need * r) for p, r in per_accepted.items()}
    seeds = {p: (math.ceil(size * hops[p] * margin / accepted_rate[p]) if accepted_rate.get(p) else math.inf) for p in hops}
    over = {p: (round(seeds[p] / (size * hops[p]), 3) if math.isfinite(seeds[p]) else math.inf) for p in hops}
    return {"accepted_needed": need, "calls": calls, "seeds": seeds, "overshoot": over, "spent": spent,
            "total": spent + sum(calls.values())}


def gates(rep: dict, cfg: dict) -> list[str]:
    """Every user stop of the pilot (any → STOP and ask the user)."""
    out = []
    if rep["projection"]["total"] > cfg["max_calls_total"]:
        out.append(f"П2: projected {rep['projection']['total']} calls > {cfg['max_calls_total']} -- ask the user")
    for p, loss in sorted(rep["threshold_loss"].items()):
        if loss > cfg["max_threshold_loss"]:
            out.append(f"П5: R >= 0.55 rejects {loss:.0%} of STOP ∧ full-recall trajectories in {p} -- ask the user")
    out += [f"yield 0 for {p}" for p, y in sorted(rep["yield"].items()) if y == 0]
    if rep["shortage"]:
        out.append(f"П4: seeds shortage with the projected overshoot {rep['shortage']} -- ask the user")
    if not rep["verify_ok"]:
        out.append("M3-R3 п. 8: judge sensitivity below 0.90 for a plant kind")
    if not rep["oracle_ok"]:
        out.append("M3-R6: oracle config loses > 2 % accepted pairs even at width 128")
    return out
