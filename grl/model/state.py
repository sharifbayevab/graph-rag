"""State aggregation (IMPLEMENTATION.md §6.3): mean pooling over vertex sets (order- and size-free) + MLP 128, plus
the per-type candidate pools, search previews and the U3 / U4 scalars of grl.model.observe.StateDesc. An empty set
pools to the zero vector (g_1 = 0 on the first step). n_types / n_actions follow the env (grl.model.policy.policy_dims:
28 / 7 for the M3 actions, 31 / 8 for the M7 struct variant, whose 8th type is `structure`)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from grl.model.observe import N_SCALARS


def segment_mean(z: torch.Tensor, qidx, rows_list) -> torch.Tensor:
    """z [B, N, D]; rows_list[e] -- int64 node rows of episode e whose query is z[qidx[e]]. Returns [E, D]."""
    dev, E, D = z.device, len(rows_list), z.shape[-1]
    lens = torch.tensor([len(r) for r in rows_list], device=dev, dtype=torch.long)
    if int(lens.sum()) == 0:
        return z.new_zeros(E, D)
    rows = torch.from_numpy(np.concatenate([np.asarray(r, np.int64) for r in rows_list])).to(dev)
    seg = torch.repeat_interleave(torch.arange(E, device=dev), lens)
    q = torch.as_tensor(np.asarray(qidx, np.int64), device=dev)[seg]
    out = z.new_zeros(E, D).index_add_(0, seg, z[q, rows])
    return out / lens.clamp(min=1).unsqueeze(1).to(z.dtype)


class StateEncoder(nn.Module):
    def __init__(self, dim: int = 256, pool_hidden: int = 128, cand_dim: int = 32, n_types: int = 7,
                 n_actions: int = 28):
        super().__init__()
        self.n_types, self.n_actions = n_types, n_actions
        mlp = lambda o: nn.Sequential(nn.Linear(dim, o), nn.ReLU())     # noqa: E731
        self.pool_g, self.pool_f, self.pool_v0 = mlp(pool_hidden), mlp(pool_hidden), mlp(pool_hidden)
        self.cand = nn.Linear(dim, cand_dim)
        self.prev = nn.Linear(dim, pool_hidden)
        self.n_num = 2 * n_types + 2 + N_SCALARS + 2 * n_actions
        self.out_dim = dim + 3 * pool_hidden + n_types * cand_dim + 2 * pool_hidden + self.n_num

    def forward(self, z, qz, qidx, descs) -> torch.Tensor:
        bad = [(len(d.cand), len(d.mask)) for d in descs
               if len(d.cand) != self.n_types or len(d.mask) != self.n_actions]
        if bad:
            raise ValueError(f"state descriptions with (types, actions) {bad[0]} for an encoder of "
                             f"({self.n_types}, {self.n_actions}): build the policy with policy_dims(env)")
        dev = z.device
        q = qz[torch.as_tensor(np.asarray(qidx, np.int64), device=dev)]
        g = self.pool_g(segment_mean(z, qidx, [d.covered for d in descs]))
        f = self.pool_f(segment_mean(z, qidx, [d.frontier for d in descs]))
        v = self.pool_v0(segment_mean(z, qidx, [d.v0 for d in descs]))
        cand = torch.cat([self.cand(segment_mean(z, qidx, [d.cand[r] for d in descs])) for r in range(self.n_types)], 1)
        prev = torch.cat([self.prev(segment_mean(z, qidx, [d.preview[i] for d in descs])) for i in range(2)], 1)
        num = np.stack([np.concatenate([np.log1p(d.cand_n), d.cand_best, d.preview_sim, d.scalars,
                                        d.mask.astype(np.float32), d.hist]).astype(np.float32) for d in descs])
        return torch.cat([q, g, f, v, cand, prev, torch.from_numpy(num).to(dev, z.dtype)], 1)
