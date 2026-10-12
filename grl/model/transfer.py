"""28 -> 31 weight transfer of a GraphPolicy (docs/m7 design §3.3: "the 31-action head; the weights of the old 28
actions are carried over from the best pilot-5 SFT").

Only the policy head depends on n_actions / n_types (GraphEncoder: none; StateEncoder: its pools and the shared
candidate / preview projections do not, `cand` is one Linear applied per type). The head's input is the StateEncoder
concatenation (grl.model.state)
    q D | g, f, v0 3P | cand T·C | preview 2P | log1p(cand_n) T | cand_best T | preview_sim 2 | scalars 6 |
    mask A | hist A
so going from (A, T) = (28, 7) to (31, 8):
  - head.net.0.weight: zero columns for the new type's C pool columns (after the 7·C block), its cand_n and cand_best
    (after each 7-block) and the 3 new mask / hist entries (after each 28-block); every old column is copied;
  - head.net.<last>.weight / bias: rows 0..27 copied; the new row 28 + j (structure, m = m_grid[j]) = the mean of the
    graph rows with that m (rows 7 + 3·r + j, r over the 7 edge types; graphrag.env.mdp.action_space layout);
  - everything else is copied unchanged.
With the zero columns the old 28 logits equal the source model's on every state whose old features are unchanged
(tests/test_m7_struct_policy.py: a state with no structure candidate, and one with).

Task 6d-b: the same rule widens (31, 8) -> (34, 9) for the `amendments` operator (struct.amend_op): its type is the 9th
(after structure), its 3 actions are ids 31..33 and their initial rows are the mean of the 8 types' rows with that m
(the 7 graph types and structure, rows 7 + 3·r + j). widen_policy_state chains the steps, so a 28-action `full` source
reaches 34 through 31; the old logits are unchanged on every state with no candidate of the new types."""
from __future__ import annotations

import torch

from grl.model.observe import N_SCALARS

N_SEARCH = 7                     # stop + 3 lexical + 3 vector: the graph rows start here (action_space)


def _head_layers(sd: dict) -> list[int]:
    return sorted(int(k.split(".")[2]) for k in sd if k.startswith("head.net.") and k.endswith(".weight"))


def expand_policy_state(sd_28: dict, n_old: int = 28, n_new: int = 31, n_types_old: int = 7,
                        n_types_new: int = 8) -> dict:
    """A new state dict for GraphPolicy(g, cfg, n_actions=n_new, n_types=n_types_new) (module docstring)."""
    n_m = n_new - n_old
    if n_types_new != n_types_old + 1 or n_m < 1 or n_old - N_SEARCH != n_types_old * n_m:
        raise ValueError(f"transfer {n_old}/{n_types_old} -> {n_new}/{n_types_new}: expected one new type whose "
                         f"{n_m} actions mirror the m grid of the {n_types_old} graph types")
    layers = _head_layers(sd_28)
    if len(layers) < 2:
        raise ValueError(f"policy head with {len(layers)} Linear layer(s) (model.policy.hidden = ()): the input "
                         "columns and the output rows would be padded on the same matrix; the transfer needs at least "
                         "one hidden layer")
    first, last = f"head.net.{layers[0]}", f"head.net.{layers[-1]}"
    C, D = sd_28["state.cand.weight"].shape
    P = sd_28["state.prev.weight"].shape[0]
    w0 = sd_28[f"{first}.weight"]
    blocks = [D + 3 * P + n_types_old * C, 2 * P + n_types_old, n_types_old, 2 + N_SCALARS + n_old, n_old]
    if w0.shape[1] != sum(blocks) or sd_28[f"{last}.weight"].shape[0] != n_old:
        raise ValueError(f"state dict is not a {n_old}-action / {n_types_old}-type GraphPolicy: head input "
                         f"{w0.shape[1]} (want {sum(blocks)}), output {sd_28[f'{last}.weight'].shape[0]}")
    pads = [C, 1, 1, n_m, n_m]
    parts, pos = [], 0
    for n, z in zip(blocks, pads):
        parts += [w0[:, pos:pos + n], w0.new_zeros(w0.shape[0], z)]
        pos += n
    out = {k: v.clone() for k, v in sd_28.items()}
    out[f"{first}.weight"] = torch.cat(parts, 1)
    for suf in ("weight", "bias"):
        x = sd_28[f"{last}.{suf}"]
        new = [x[[N_SEARCH + n_m * r + j for r in range(n_types_old)]].mean(0, keepdim=True) for j in range(n_m)]
        out[f"{last}.{suf}"] = torch.cat([x] + new, 0)
    return out


STEPS = ((28, 7), (31, 8), (34, 9))      # the action / type counts on the transfer chain (M3, struct, + amendments)


def widen_policy_state(sd: dict, src: tuple[int, int], dst: tuple[int, int]) -> dict:
    """Widen a (n_actions, n_types) = src state dict to dst along STEPS, one expand_policy_state per step (src == dst:
    an unchanged copy). ValueError if src / dst are not on the chain or dst comes before src."""
    src, dst = tuple(int(x) for x in src), tuple(int(x) for x in dst)
    if src not in STEPS or dst not in STEPS or STEPS.index(dst) < STEPS.index(src):
        raise ValueError(f"no transfer {src} -> {dst} (actions / types along {list(STEPS)})")
    out = {k: v.clone() for k, v in sd.items()}
    for (na, ta), (nb, tb) in zip(STEPS[STEPS.index(src):STEPS.index(dst)], STEPS[STEPS.index(src) + 1:]):
        out = expand_policy_state(out, na, nb, ta, tb)
    return out
