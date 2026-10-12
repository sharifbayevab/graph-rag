"""LLM call projection for the dataset passes (decision П2, M3-R10; pre-pilot re-review I3). Deterministic where the
plan is deterministic, measured where it depends on model output:
- ds_facts: one call per fact block (graphrag.dataset.seeds.block_of) that the full run's seeds touch and the cache does
  not hold yet, × (1 + the pilot's retry rate). NOT per accepted pair: a block is paid once whatever the yield.
- ds_generate: ceil(usable seeds / gen_batch), usable = seeds whose every C_ref fragment has a fact (pilot rate).
- ds_verify: judge views of the projected candidates (1 view per single-hop, 1 + |C_ref| per multi-hop; candidates =
  seeds × the pilot's generate yield per pattern) plus plant views (groups × plant_share × views per plant), packed at
  the pilot's measured views per call."""
from __future__ import annotations

import math
from collections import Counter


def blocks_of(seeds, block_of: dict[str, str]) -> set[str]:
    return {block_of[c] for s in seeds for c in s.chunks}


def views_of(n_chunks: int) -> int:
    return 1 if n_chunks == 1 else 1 + n_chunks


def project_calls(full_seeds, pilot_seeds, block_of: dict[str, str], cached_blocks: set[str], rates: dict,
                  cfg: dict) -> dict:
    """rates (from the pilot): facts_retry (retries / blocks), usable (usable seeds / seeds), gen_yield {pattern:
    accepted / seeds}, views_per_call (views / calls), plant_views (views per plant). cfg: gen_batch, verify_batch,
    plant_share. full_seeds include the pilot seeds (forced); the pilot's calls are cached and not re-paid.
    Returns {pass: {"pilot", "full_new", "total"}} + "total"."""
    pilot_ids = {s.seed_id for s in pilot_seeds}
    rest = [s for s in full_seeds if s.seed_id not in pilot_ids]
    retry = 1.0 + float(rates.get("facts_retry", 0.0))
    pilot_blocks = blocks_of(pilot_seeds, block_of)
    new_blocks = blocks_of(full_seeds, block_of) - set(cached_blocks) - pilot_blocks

    def gen_calls(seeds) -> int:
        return math.ceil(len(seeds) * float(rates["usable"]) / int(cfg["gen_batch"]))

    def verify_calls(seeds) -> int:
        cands = Counter()
        views = 0.0
        for s in seeds:
            y = float(rates["gen_yield"].get(s.pattern, 0.0))
            cands[s.pattern] += y
            views += y * views_of(len(s.chunks))
        n = sum(cands.values())
        plants = n / int(cfg["verify_batch"]) * float(cfg["plant_share"])
        return math.ceil((views + plants * float(rates["plant_views"])) / float(rates["views_per_call"]))

    out = {"ds_facts": {"pilot": math.ceil(len(pilot_blocks) * retry),     # the pilot's own spend (paid once)
                        "full_new": math.ceil(len(new_blocks) * retry)},
           "ds_generate": {"pilot": gen_calls(pilot_seeds), "full_new": gen_calls(rest)},
           "ds_verify": {"pilot": rates.get("verify_pilot_calls"), "full_new": verify_calls(rest)}}
    if out["ds_verify"]["pilot"] is None:
        out["ds_verify"]["pilot"] = verify_calls(pilot_seeds)
    for v in out.values():
        v["total"] = v["pilot"] + v["full_new"]
    out["total"] = {k: sum(v[k] for v in out.values()) for k in ("pilot", "full_new", "total")}
    out["blocks"] = {"pilot": len(pilot_blocks), "full_new": len(new_blocks), "cached": len(set(cached_blocks))}
    return out
