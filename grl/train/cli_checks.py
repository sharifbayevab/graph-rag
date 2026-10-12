"""CLI helpers of scripts/train.py that tests import (scripts/ is not a package; ruling P1, as R2 in M7 part 1):
the baselines name list with the rows skipped per variant, and the grpo K_max gate against the SFT run."""
from __future__ import annotations

import json
from pathlib import Path

ORACLE_NAMES = ("greedy_oracle", "beam_oracle")
# M7 part 2 Task 1 (final review of part 1, item 4): BeamSearch's lookahead (successor_additions) enumerates the M3
# operators only, so on the struct variant its upper_bound row would be false; both oracle rows are skipped
STRUCT_ORACLE_SKIP = "lookahead ignores structure"
REQUIRED_ORACLE_SKIP = "oracles STOP freely: not under a `require` mask"


def baseline_names(variant: str, k_grid=None) -> tuple[list[str], dict[str, str]]:
    """(names, skipped) of `train.py baselines` (the default list: §9.3 + M5 + dynamic k): skipped = {name: reason}
    for the rows left out on this variant. k_grid: the run's env.k_grid (default: train.yaml's)."""
    from grl.model.variants import VARIANTS, is_struct
    from grl.train.baselines import DYNAMIC_NAMES
    if k_grid is None:
        from graphrag import config
        k_grid = config.load("train")["env"]["k_grid"]
    names = [f"{k}@{n}" for k in ("bm25", "dense", "rrf") for n in k_grid]
    names += ["graph_heuristic", "hybrid_heuristic", *ORACLE_NAMES, "random@0"]     # §9.3 + M5
    names += list(DYNAMIC_NAMES)                                                   # M5 dynamic k
    skipped: dict[str, str] = {}
    if VARIANTS[variant].get("require"):
        skipped = {n: REQUIRED_ORACLE_SKIP for n in ORACLE_NAMES}
    elif is_struct(variant):
        skipped = {n: STRUCT_ORACLE_SKIP for n in ORACLE_NAMES}
    return [n for n in names if n not in skipped], skipped


def _recorded_k_max(path: Path):
    if not path.is_file():
        return None
    v = json.loads(path.read_text(encoding="utf-8")).get("K_max")
    return None if v is None else int(v)


def check_grpo_k_max(sft_dir, k_max: int, legacy: int | None = None) -> int:
    """GRPO refuses an SFT run built under another K_max (final review of part 1, item 5; R12). The SFT K_max is
    read from <sft>/pi_ref.json and <sft>/manifest.json ("K_max"); both recorded and different -> refuse. Neither
    recorded: a pre-R12 run, built under train.yaml's K_max = `legacy` (None -> refuse). Returns the SFT K_max."""
    sft = Path(sft_dir)
    rec = {n: _recorded_k_max(sft / n) for n in ("pi_ref.json", "manifest.json")}
    got = {v for v in rec.values() if v is not None}
    if len(got) > 1:
        raise SystemExit(f"grpo --sft {sft}: K_max differs between pi_ref.json and manifest.json {rec}")
    if not got:
        if legacy is None:
            raise SystemExit(f"grpo --sft {sft}: no K_max recorded in pi_ref.json / manifest.json")
        got = {int(legacy)}
    sft_k = got.pop()
    if sft_k != int(k_max):
        raise SystemExit(f"grpo --sft {sft}: SFT K_max={sft_k}, this run K_max={int(k_max)} "
                         f"(pass --set env.K_max={sft_k}, or train SFT under K_max={int(k_max)})")
    return sft_k
