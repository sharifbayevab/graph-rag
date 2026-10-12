"""Recall-favouring reward option (M5 "dynamic k"): R_chunks as F-beta instead of F1, applied on the grl side of the
reward call path (rollout rewards, context metrics) -- graphrag.env.reward stays the M3 spec. train.yaml
`reward.chunks_beta` (default 1 = the spec F1; 2 = F2 weighs recall 4x precision); the other three parts and the
weights are unchanged:  R = α·F_β(C, C*) + β·R_coverage + γ·R_diversity + δ·R_coherence.
The M3 oracles (greedy / beam) still optimise the spec reward (they call graphrag.env.reward directly)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from graphrag.env.reward import RewardParts, Weights, reward, weights_from_config


@dataclass(frozen=True)
class GRLWeights(Weights):
    chunks_beta: float = 1.0


def check_beta(beta) -> float:
    b = float(beta)
    if not b > 0:
        raise ValueError(f"reward.chunks_beta must be > 0, got {beta}")
    return b


def grl_weights(cfg: dict) -> GRLWeights:
    return GRLWeights(**asdict(weights_from_config(cfg)), chunks_beta=check_beta(cfg["reward"].get("chunks_beta", 1)))


def chunks_beta(w) -> float:
    return float(getattr(w, "chunks_beta", 1.0))


def chunks_label(w) -> str:
    b = chunks_beta(w)
    return "F1" if b == 1.0 else f"F{b:g}"


def r_chunks_fbeta(C: set, G: set, beta: float) -> float:
    tp = len(C & G)
    if not tp:
        return 0.0
    p, r, b2 = tp / len(C), tp / len(G), float(beta) ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def grl_reward(data, depth, context, target, w, t_max: int) -> RewardParts:
    """graphrag.env.reward, with R_chunks replaced by F-beta when w.chunks_beta != 1 (identical object otherwise)."""
    p = reward(data, depth, context, target, w, t_max)
    b = chunks_beta(w)
    if b == 1.0:
        return p
    fb = round(r_chunks_fbeta({int(c) for c in context}, {int(g) for g in target}, check_beta(b)), 9)
    total = w.alpha * fb + w.beta * p.coverage + w.gamma * p.diversity + w.delta * p.coherence
    return RewardParts(fb, p.coverage, p.diversity, p.coherence, round(total, 9))
