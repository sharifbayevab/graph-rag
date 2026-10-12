"""Phase 2 -- group policy optimisation (IMPLEMENTATION.md §8.3–8.6; roadmap U5 exact KL, U11 step penalty).
One iteration: B distinct train queries, G_s = 4 sampled episodes each in the M3 env (GNN once per query, shared by
the group, §8.5), terminal reward from graphrag.env.reward (no generator, §7.2), group advantage
A_i = (r_i − r̄)/(σ_r + ε) (a group with σ = 0 gets A = 0), one on-policy step on
    (1/n) Σ_i (1/|τ_i|) Σ_t [ −A_i·log π_θ(a_t|s_t) + β_KL·KL(π_θ‖π_ref)(s_t) ],
KL exact over the masked actions. On-policy with one step per batch the importance ratio is 1 and PPO clipping is a
no-op, so it is omitted. The policy stays in eval mode (no dropout) so rollouts and the update see one distribution;
sampling is at temperature 1 (episodes keep no behaviour log-probs). D6 (a): the GNN trains too; grpo.freeze_gnn is
a diagnostic switch only.
Pilot 5b options (all default-off; with the defaults the update is the one above, bit for bit):
- grpo.ent_coef: entropy bonus -c*H(pi_theta(.|s_t)) per step, H exact over the masked actions;
- grpo.ppo_epochs K / grpo.clip_eps eps: K passes over the collected batch with the PPO clipped surrogate
  -min(rho*A, clip(rho, 1-eps, 1+eps)*A), rho = pi_theta / pi_old, pi_old = the policy at collection. Episodes record
  the behaviour log-probs (Episode.logp_b); the first pass checks it is on-policy (|log pi_theta - logp_b| <=
  ONPOLICY_TOL) and its detached log-probs are pi_old. The exact KL to pi_ref is unchanged. These keys are optional
  in train.yaml (GRPO_OPTIONAL; train.py --set accepts them) and enter the resume config only when not default.
- grpo.G_s (existing key) may be 8 or 16: the GNN still runs once per query, shared by the group.
Pilot 8 option (default-off): grpo.upweight_pattern P with grpo.upweight_share s in (0, 1): every batch takes
round(s * B) queries from the train rows of pattern P (their own shuffled queue) and the rest from the other rows, so
P is sampled at share s instead of its share of train (M7: P = "version", s = 0.5).
Task GRPO-X options (default-off; with the defaults every run is the one above, bit for bit):
- grpo.adv_std false: Dr. GRPO advantage A_i = r_i - r̄ (no division by the group σ; a σ = 0 group still gets 0);
- grpo.upweight_mix "P1=s1,P2=s2": like upweight_pattern for several patterns at once -- every batch takes
  round(s_j * B) queries of pattern P_j (each pattern its own shuffled queue) and the rest from the other rows
  (excludes upweight_pattern);
- grpo.replay_lambda λ > 0 with grpo.replay_patterns "P1,P2" and grpo.replay_per_iter n: expert replay (LUFFY /
  SRFT-style) -- every update adds λ · the SFT marginal loss (grl.train.sft.sft_batch) of n expert-labelled train
  queries of those patterns, drawn without replacement from their own seeded generator (seed + 1)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from grl.train.evaluate import episode_parts, extra_summary, summarize
from grl.train.log import alarms, hacking_alarm, slot_top_share
from grl.train.rollout import qvecs, rollout
from grl.train.sft import set_torch_rng_state, torch_rng_state


def group_advantages(r: torch.Tensor, eps: float) -> torch.Tensor:
    mean = r.mean(1, keepdim=True)
    std = r.std(1, unbiased=False, keepdim=True)
    return torch.where(std > 1e-12, (r - mean) / (std + eps), torch.zeros_like(r))


def group_centered(r: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO advantage r - r̄ per group (grpo.adv_std false); a group with σ = 0 gets exactly 0."""
    std = r.std(1, unbiased=False, keepdim=True)
    return torch.where(std > 1e-12, r - r.mean(1, keepdim=True), torch.zeros_like(r))


def parse_mix(s: str) -> list[tuple[str, float]]:
    """grpo.upweight_mix "P1=s1,P2=s2" -> [(P1, s1), (P2, s2)] (order kept; empty string -> [])."""
    out = []
    for part in [x.strip() for x in str(s or "").split(",") if x.strip()]:
        k, _, v = part.partition("=")
        if not k.strip() or not v.strip():
            raise ValueError(f"grpo.upweight_mix: {part!r} is not pattern=share")
        out.append((k.strip(), float(v)))
    if len({k for k, _ in out}) != len(out):
        raise ValueError(f"grpo.upweight_mix {s!r}: a pattern is listed twice")
    return out


def exact_kl(logp: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = logp.masked_fill(~mask, 0.0)                        # no -inf · 0 anywhere (finite gradients)
    lr = logp_ref.masked_fill(~mask, 0.0)
    return (lp.exp() * mask * (lp - lr)).sum(-1)


GRPO_OPTIONAL = {"ent_coef": 0.0, "ppo_epochs": 1, "clip_eps": 0.0, "upweight_pattern": "", "upweight_share": 0.0,
                 "adv_std": True, "upweight_mix": "", "replay_lambda": 0.0, "replay_patterns": "",
                 "replay_per_iter": 4}
ONPOLICY_TOL = 1e-3


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = logp.masked_fill(~mask, 0.0)                        # finite with masked (-inf) actions
    return -(lp.exp() * mask * lp).sum(-1)


def ppo_surrogate(lp_taken, old_lp, adv_step, clip_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step -min(rho*A, clip(rho, 1-eps, 1+eps)*A) (eps <= 0: -rho*A) and rho."""
    ratio = torch.exp(lp_taken - old_lp)
    if clip_eps > 0:
        return -torch.minimum(ratio * adv_step, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_step), ratio
    return -ratio * adv_step, ratio


def grpo_loss(lp_taken, adv_step, kl, ep_index, n_eps: int, beta: float, ent=None, ent_coef: float = 0.0,
              surrogate=None) -> torch.Tensor:
    per = (-adv_step * lp_taken if surrogate is None else surrogate) + beta * kl
    if ent_coef:
        per = per - ent_coef * ent
    lens = torch.bincount(ep_index, minlength=n_eps).clamp(min=1).to(per.dtype)
    return (torch.zeros(n_eps, dtype=per.dtype, device=per.device).index_add_(0, ep_index, per) / lens).mean()


def _check_temperature(cfg) -> None:
    if float(cfg["temperature"]) != 1.0:
        raise ValueError(f"grpo.temperature must be 1 (got {cfg['temperature']}): episodes keep no behaviour "
                         "log-probs, so the one-step update is on-policy only when sampling from π_θ itself")


def collect(policy, graph, vec, ranks, items, cfg, gen, w, t_max, device):
    _check_temperature(cfg)
    G = int(cfg["G_s"])
    qkeys = [it.qkey for it in items]
    policy.eval()
    eps, finals = rollout(policy, graph, vec, ranks, qkeys, G, greedy=False, gen=gen, device=device,
                          temperature=float(cfg["temperature"]))
    golds = {it.qkey: it.gold for it in items}
    parts = episode_parts(vec, eps, finals, golds, w)
    lam = float(cfg["step_penalty"])
    r = torch.tensor([p.total - lam * len(e.actions) / t_max for p, e in zip(parts, eps)], dtype=torch.float64)
    if cfg.get("adv_std", True):
        adv = group_advantages(r.view(len(qkeys), G), float(cfg["adv_eps"])).view(-1).float()
    else:                                                     # Task GRPO-X: Dr. GRPO (no σ division)
        adv = group_centered(r.view(len(qkeys), G)).view(-1).float()
    return eps, finals, parts, r, adv, golds


def update(policy, ref, graph, qv, eps, adv, cfg, opt, device, replay=None) -> dict:
    """replay (Task GRPO-X, default None): a callable policy -> (SFT loss tensor, step accuracy, n steps) of this
    iteration's expert-replay queries; its loss enters every pass with weight grpo.replay_lambda."""
    policy.eval()
    K = int(cfg.get("ppo_epochs", 1))
    clip_eps, ent_coef = float(cfg.get("clip_eps", 0.0)), float(cfg.get("ent_coef", 0.0))
    if K < 1:
        raise ValueError(f"grpo.ppo_epochs must be >= 1 (got {K})")
    ppo = K > 1 or clip_eps > 0
    steps = [(i, d, a) for i, e in enumerate(eps) for d, a in zip(e.descs, e.actions)]
    qidx = [eps[i].qi for i, _, _ in steps]
    descs = [d for _, d, _ in steps]
    out, old_lp = {}, None
    for k in range(K):
        if cfg.get("freeze_gnn"):
            with torch.no_grad():
                z, qz = policy.encode(graph, qv)
        else:
            z, qz = policy.encode(graph, qv)
        logp = policy.log_probs(z, qz, qidx, descs)
        if k == 0:
            with torch.no_grad():
                zr, qzr = ref.encode(graph, qv)
                lref = ref.log_probs(zr, qzr, qidx, descs)
            dev = logp.device
            mask = torch.from_numpy(np.stack([d.mask for d in descs])).to(dev)
            act = torch.tensor([a for _, _, a in steps], device=dev)
            ep = torch.tensor([i for i, _, _ in steps], device=dev)
        lp_taken = logp.gather(1, act[:, None]).squeeze(1)
        kl = exact_kl(logp, lref, mask)
        ent_t = masked_entropy(logp, mask) if ent_coef else None
        sur = None
        if ppo:
            if k == 0:
                old_lp = lp_taken.detach()
                if all(len(e.logp_b) == len(e.actions) for e in eps):
                    lb = torch.tensor([x for e in eps for x in e.logp_b], device=dev, dtype=old_lp.dtype)
                    gap = float((old_lp - lb).abs().max()) if lb.numel() else 0.0
                    if gap > ONPOLICY_TOL:
                        raise RuntimeError(f"PPO first pass is not on-policy: |log pi_theta - behaviour| = {gap:.2e}")
                    out["onpolicy_gap"] = gap
            sur, ratio = ppo_surrogate(lp_taken, old_lp, adv.to(dev)[ep], clip_eps)
        loss = grpo_loss(lp_taken, adv.to(dev)[ep], kl, ep, len(eps), float(cfg["beta_kl"]), ent_t, ent_coef, sur)
        if replay is not None:
            rl, racc, rn = replay(policy)
            loss = loss + float(cfg["replay_lambda"]) * rl
            if k == 0:
                out.update({"replay_loss": float(rl.detach()), "replay_step_acc": float(racc), "replay_steps": rn})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], float(cfg["grad_clip"]))
        opt.step()
        if k == 0:
            with torch.no_grad():
                lp = logp.masked_fill(~mask, 0.0)
                ent = -(lp.exp() * mask * lp).sum(-1)
            out.update({"loss": float(loss.detach()), "kl_ref": float(kl.mean()), "entropy": float(ent.mean()),
                        "grad_norm": float(gn), "n_steps": len(steps)})
        if ppo:
            with torch.no_grad():
                out["ratio_max"] = max(out.get("ratio_max", 0.0), float(ratio.max()))
                if clip_eps > 0:
                    out["clip_frac"] = float(((ratio - 1).abs() > clip_eps).float().mean())
    if ppo:
        out["loss_last"] = float(loss.detach())
    return out


@dataclass
class GRPOResult:
    best_path: Path | None
    best_metric: float
    iterations: int
    stopped_early: bool


def _next_batch(queue: list, B: int) -> tuple[list, list]:
    batch, rest, seen = [], [], set()
    for p in queue:
        if len(batch) < B and p not in seen:
            batch.append(p)
            seen.add(p)
        else:
            rest.append(p)
    return batch, rest


RESUME_KEYS = ("batch_queries", "G_s", "seed", "val_every", "freeze_gnn", "temperature")


def _resume_cfg(cfg) -> dict:
    out = {k: cfg.get(k) for k in RESUME_KEYS}
    out.update({k: cfg[k] for k, d in GRPO_OPTIONAL.items() if k in cfg and cfg[k] != d})   # old checkpoints unchanged
    return out


def _replay_setup(cfg, train_items, replay_ex):
    """(by_q of the replay queries, their sorted keys, n per iteration, λ) or None when grpo.replay_lambda is 0."""
    lam = float(cfg.get("replay_lambda") or 0.0)
    if not lam:
        return None
    if lam < 0:
        raise ValueError(f"grpo.replay_lambda must be >= 0 (got {lam})")
    pats = {p.strip() for p in str(cfg.get("replay_patterns") or "").split(",") if p.strip()}
    n = int(cfg.get("replay_per_iter", 4))
    if not pats or n < 1:
        raise ValueError(f"grpo.replay_lambda {lam}: needs grpo.replay_patterns and replay_per_iter >= 1")
    if replay_ex is None:
        raise ValueError("grpo.replay_lambda > 0 needs the expert replay examples (train.py grpo builds them)")
    want = {it.qkey for it in train_items if it.pattern in pats}
    by_q: dict = {}
    for e in replay_ex:
        if e.qkey in want:
            by_q.setdefault(e.qkey, []).append(e)
    if len(by_q) < n:
        raise ValueError(f"grpo.replay_patterns {sorted(pats)}: {len(by_q)} replay queries < replay_per_iter {n}")
    return by_q, sorted(by_q), n, lam


def run_grpo(policy, ref, graph, vec, ranks, train_items, cfg, w, t_max, logger, ckpt, device, evaluate, actions,
             alarm_cfg, resume=False, replay_ex=None) -> GRPOResult:
    _check_temperature(cfg)
    B, G, every, total = int(cfg["batch_queries"]), int(cfg["G_s"]), int(cfg["val_every"]), int(cfg["iterations"])
    patience = int(cfg["patience"])
    if patience < 1:
        raise ValueError(f"grpo.patience must be >= 1 (got {patience})")
    if B > len(train_items):
        raise ValueError(f"batch_queries {B} > {len(train_items)} train queries")
    if cfg.get("freeze_gnn"):
        policy.encoder.requires_grad_(False)
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=float(cfg["lr"]))
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    by_id = {it.pid: it for it in train_items}
    up_pat, up_share = str(cfg.get("upweight_pattern") or ""), float(cfg.get("upweight_share") or 0.0)
    up_items = [it for it in train_items if up_pat and it.pattern == up_pat]
    rest_items = [it for it in train_items if not (up_pat and it.pattern == up_pat)]
    n_up = int(round(up_share * B)) if up_pat else 0
    if up_pat and not (0 < up_share < 1 and 0 < n_up < B and len(up_items) >= n_up and len(rest_items) >= B - n_up):
        raise ValueError(f"grpo.upweight_pattern {up_pat!r} / upweight_share {up_share}: need 0 < share < 1, "
                         f"0 < round(share*B) < B and enough rows ({len(up_items)} of the pattern, B {B})")
    queue_up = []
    mix = parse_mix(cfg.get("upweight_mix") or "")           # Task GRPO-X: several upweighted patterns
    if mix:
        if up_pat:
            raise ValueError("grpo.upweight_mix and grpo.upweight_pattern exclude each other")
        mix_pats = {p for p, _ in mix}
        mix_items = {p: [it for it in train_items if it.pattern == p] for p, _ in mix}
        mix_n = {p: int(round(s * B)) for p, s in mix}
        rest_items = [it for it in train_items if it.pattern not in mix_pats]
        n_rest = B - sum(mix_n.values())
        if any(not (0 < s < 1) or mix_n[p] < 1 or len(mix_items[p]) < mix_n[p] for p, s in mix) or n_rest < 1 \
                or len(rest_items) < n_rest:
            raise ValueError(f"grpo.upweight_mix {cfg.get('upweight_mix')!r}: need 0 < share < 1, round(share*B) >= 1 "
                             f"per pattern, enough rows and >= 1 other query per batch (B {B})")
    queue_mix = {p: [] for p, _ in mix}
    rep_cfg = _replay_setup(cfg, train_items, replay_ex)
    rrng = np.random.default_rng(int(cfg["seed"]) + 1) if rep_cfg else None
    queue, start, best, bad = [], 0, -math.inf, 0
    train_hist, val_hist = [], []                             # §8.7 hacking alarm: train / val reward at validations
    if resume:
        if not ckpt.last_path().exists():
            raise FileNotFoundError(f"nothing to resume: {ckpt.last_path()} does not exist")
        st = torch.load(ckpt.last_path(), map_location="cpu", weights_only=True)
        if st["cfg"] != _resume_cfg(cfg):
            raise ValueError(f"--resume with a different grpo config: checkpoint {st['cfg']} vs now {_resume_cfg(cfg)}")
        policy.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        rng.bit_generator.state = json.loads(st["rng"])
        set_torch_rng_state(st["torch_rng"])
        queue, start, best, bad = list(st["queue"]), int(st["step"]), float(st["best"]), int(st["bad"])
        queue_up = list(st.get("queue_up", []))
        if mix:
            queue_mix = {p: list(v) for p, v in st["queue_mix"].items()}
        if rep_cfg:
            rrng.bit_generator.state = json.loads(st["replay_rng"])
        train_hist, val_hist = list(st["train_hist"]), list(st["val_hist"])
        logger.truncate({"grpo": start, "grpo_val": start})
    stopped, done = False, start
    for it in range(start, total):
        if bad >= patience:                                   # early-stopped (also when resumed after that save)
            stopped = True
            break
        if mix:                                               # Task GRPO-X: several patterns, fixed counts
            pids = []
            for p, _ in mix:
                if len(queue_mix[p]) < mix_n[p]:
                    queue_mix[p] += [mix_items[p][i].pid for i in rng.permutation(len(mix_items[p]))]
                got, queue_mix[p] = _next_batch(queue_mix[p], mix_n[p])
                pids += got
            if len(queue) < n_rest:
                queue += [rest_items[i].pid for i in rng.permutation(len(rest_items))]
            got, queue = _next_batch(queue, n_rest)
            pids += got
        elif n_up:                                            # pilot 8: pattern upweighted to n_up of B
            if len(queue_up) < n_up:
                queue_up += [up_items[i].pid for i in rng.permutation(len(up_items))]
            if len(queue) < B - n_up:
                queue += [rest_items[i].pid for i in rng.permutation(len(rest_items))]
            p_up, queue_up = _next_batch(queue_up, n_up)
            p_rest, queue = _next_batch(queue, B - n_up)
            pids = p_up + p_rest
        else:
            if len(queue) < B:
                queue += [train_items[i].pid for i in rng.permutation(len(train_items))]
            pids, queue = _next_batch(queue, B)
        batch = [by_id[p] for p in pids]
        eps, finals, parts, r, adv, golds = collect(policy, graph, vec, ranks, batch, cfg, gen, w, t_max, device)
        replay = None
        if rep_cfg:
            by_rq, rkeys, rn, _ = rep_cfg
            pick = [rkeys[i] for i in sorted(rrng.choice(len(rkeys), rn, replace=False).tolist())]

            def replay(pol, pick=pick, by_rq=by_rq):
                from grl.train.sft import sft_batch
                return sft_batch(pol, graph, ranks, pick, by_rq, "marginal", device)
        m = update(policy, ref, graph, qvecs(ranks, [b.qkey for b in batch], device), eps, adv, cfg, opt, device,
                   replay)
        m.update(summarize(eps, finals, parts, golds, actions, t_max))
        m.update(extra_summary(w, eps, finals, golds))        # pilot 6 rank: ranking metrics + first-pick stats
        m["reward_group_std"] = round(float(r.view(-1, G).std(1, unbiased=False).mean()), 6)
        m["adv_zero_share"] = round(float((adv == 0).float().mean()), 6)
        if mix:
            m["batch_mix_share"] = {p: round(sum(by_id[q].pattern == p for q in pids) / len(pids), 6) for p, _ in mix}
        if n_up:
            m["batch_up_share"] = round(sum(by_id[p].pattern == up_pat for p in pids) / len(pids), 6)
        if "expert_slot0_share" in alarm_cfg:                 # rerank: expert-relative collapse threshold
            m["slot_top_share"] = slot_top_share(m.get("op_hist"))
            m["expert_slot0_share"] = alarm_cfg["expert_slot0_share"]
            m["collapse_threshold"] = float(alarm_cfg["op_collapse_share"])
        m["alarms"] = alarms(m, alarm_cfg)
        logger.log("grpo", it + 1, **m)
        metric = None
        if (it + 1) % every == 0 or it + 1 == total:
            v = evaluate(policy, "val")
            metric = float(v["reward_mean"])
            train_hist.append(float(m["reward_mean"]))
            val_hist.append(metric)
            hack = hacking_alarm(train_hist, val_hist)
            va = ["hacking: train reward rose and val reward fell over the last 3 validations (§8.7)"] if hack else []
            logger.log("grpo_val", it + 1, alarm_hacking=hack, alarms=va, **{f"val_{k}": x for k, x in v.items()})
            if metric > best + 1e-9:
                best, bad = metric, 0
            else:
                bad += 1
        ckpt.save(f"grpo-{it + 1:06d}", {"model": policy.state_dict(), "opt": opt.state_dict(),
                                          "gen": gen.get_state(), "rng": json.dumps(rng.bit_generator.state),
                                          "torch_rng": torch_rng_state(), "queue": list(queue), "step": it + 1,
                                          **({"queue_up": list(queue_up)} if n_up else {}),
                                          **({"queue_mix": {p: list(v) for p, v in queue_mix.items()}} if mix else {}),
                                          **({"replay_rng": json.dumps(rrng.bit_generator.state)} if rep_cfg else {}),
                                          "best": best, "bad": bad, "train_hist": list(train_hist),
                                          "val_hist": list(val_hist), "cfg": _resume_cfg(cfg)}, metric)
        done = it + 1
        if bad >= patience:
            stopped = True
            break
    return GRPOResult(ckpt.best() if ckpt.has_best() else None, best, done, stopped)
