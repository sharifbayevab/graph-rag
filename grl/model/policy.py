"""Policy head (IMPLEMENTATION.md §6.4): MLP 512 -> 256 -> softmax over the 28 actions (31 for the M7 struct
variant), invalid actions masked (U4).
No value head: GRPO's advantage is the group-normalised reward (§8.3)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from grl.model.gnn import GraphEncoder
from grl.model.graph import GraphTensors
from grl.model.state import StateEncoder


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any(-1).all()):
        raise ValueError("a state with no valid action (the env always allows STOP)")
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), -1)


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256), n_actions: int = 28, dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            d = int(h)
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def policy_dims(env) -> dict[str, int]:
    """{n_actions, n_types} of a GraphPolicy for `env`: its action count and the candidate types grl.model.observe
    describes (the edge types, + `structure` for a StructEnv, + `amendments` with amend_op). M3 / M5 envs: 28 / 7;
    struct: 31 / 8; struct with struct.amend_op (Task 6d-b): 34 / 9."""
    return {"n_actions": len(env.actions),
            "n_types": len(env.data.graph.edge_types) + int(getattr(env, "struct_scores", None) is not None)
            + int(bool(getattr(env, "amend_op", False)))}


class GraphPolicy(nn.Module):
    """n_actions / n_types: explicit arguments, else the model cfg's `n_actions` / `n_types` (scripts/train.py sets
    them from policy_dims(env), so load_reference(g, mcfg) rebuilds the same shapes), else the M3 28 / 7."""

    def __init__(self, g: GraphTensors, cfg: dict, n_actions: int | None = None, n_types: int | None = None):
        super().__init__()
        n_actions = int(cfg.get("n_actions", 28) if n_actions is None else n_actions)
        n_types = int(cfg.get("n_types", 7) if n_types is None else n_types)
        st = cfg["state"]
        self.encoder = GraphEncoder(g, cfg)
        self.state = StateEncoder(self.encoder.dim, int(st["pool_hidden"]), int(st["cand_dim"]), n_types, n_actions)
        self.head = PolicyHead(self.state.out_dim, tuple(cfg["policy"]["hidden"]), n_actions,
                               float(cfg.get("dropout", 0.0)))
        self.n_actions = n_actions

    def encode(self, g: GraphTensors, qvec: torch.Tensor):
        return self.encoder(g, qvec)

    def log_probs(self, z, qz, qidx, descs) -> torch.Tensor:
        x = self.state(z, qz, qidx, descs)
        mask = torch.from_numpy(np.stack([d.mask for d in descs])).to(x.device)
        return masked_log_softmax(self.head(x), mask)


def param_breakdown(p: GraphPolicy) -> dict[str, int]:
    out = {k: sum(t.numel() for t in getattr(p, k).parameters()) for k in ("encoder", "state", "head")}
    out["total"] = sum(out.values())
    return out
