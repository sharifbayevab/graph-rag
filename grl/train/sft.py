"""Phase 1 -- supervised learning on the expert trajectories (IMPLEMENTATION.md §8.2; plan D3). The label of a step
is the class of equivalent actions (same ΔC and covered, M3 Task 9): `marginal` maximises the mass of the class,
`uniform` spreads a soft target over it, `hard` takes the smallest id. GNN once per query of a batch. Best epoch by
val greedy reward (§8.5), early stop by val loss; the best policy is frozen as π_ref (needed by the KL of §8.3).
M7 pilot 8 options (SFT_OPTIONAL; all default-off, so earlier runs train exactly as before):
- sft.early_stop "reward": the patience counter follows the val greedy reward (the selection metric) instead of the
  val loss (an improvement is > min_delta);
- sft.eval_epoch0: before any update, log an sft_val row at epoch 0 (val loss, val greedy metrics, the train loss in
  eval mode) and save the initial weights as <run>/init_policy.pt (the untrained / init-from policy);
- sft.train_eval_loss: every sft_val row also carries the train loss / step accuracy in eval mode (no dropout), so the
  train and val curves are measured the same way."""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from grl.model.policy import GraphPolicy
from grl.train.log import sha256_file
from grl.train.rollout import qvecs


def sft_loss(logp: torch.Tensor, equiv, kind: str = "marginal") -> torch.Tensor:
    if kind not in ("marginal", "uniform", "hard"):
        raise ValueError(f"sft.loss {kind!r} is not marginal | uniform | hard")
    rows = []
    for e, eq in enumerate(equiv):
        if kind == "hard":
            rows.append(-logp[e, min(eq)])
            continue
        v = logp[e, torch.as_tensor(list(eq), device=logp.device)]
        rows.append(-torch.logsumexp(v, 0) if kind == "marginal" else -v.mean())
    return torch.stack(rows).mean()


def step_accuracy(logp: torch.Tensor, equiv) -> float:
    pred = logp.argmax(-1).tolist()
    return float(np.mean([p in set(eq) for p, eq in zip(pred, equiv)]))


def _group(examples) -> dict:
    out: dict = {}
    for e in examples:
        out.setdefault(e.qkey, []).append(e)
    return out


def query_batches(keys, B: int, rng) -> list[list[str]]:
    keys = list(keys)
    order = rng.permutation(len(keys))
    return [[keys[i] for i in order[j:j + B]] for j in range(0, len(keys), B)]


def sft_batch(policy, graph, ranks, keys, by_q, kind, device):
    z, qz = policy.encode(graph, qvecs(ranks, keys, device))
    ex = [e for k in keys for e in by_q[k]]
    qidx = [i for i, k in enumerate(keys) for _ in by_q[k]]
    logp = policy.log_probs(z, qz, qidx, [e.desc for e in ex])
    eq = [e.equiv for e in ex]
    return sft_loss(logp, eq, kind), step_accuracy(logp, eq), len(ex)


@torch.no_grad()
def sft_eval(policy, graph, ranks, by_q, kind, device, B: int):
    keys = sorted(by_q)
    tot, acc, n = 0.0, 0.0, 0
    for j in range(0, len(keys), B):
        loss, a, k = sft_batch(policy, graph, ranks, keys[j:j + B], by_q, kind, device)
        tot, acc, n = tot + float(loss) * k, acc + a * k, n + k
    return (tot / n, acc / n) if n else (math.nan, math.nan)


@dataclass
class SFTResult:
    best_path: Path
    ref: dict
    best_metric: float
    epochs: int


def freeze_reference(policy: GraphPolicy) -> GraphPolicy:
    ref = copy.deepcopy(policy).eval()
    ref.requires_grad_(False)
    return ref


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_reference(policy: GraphPolicy, run_dir) -> dict:
    """pi_ref.pt, then pi_ref.json (its sha256); each file replaced atomically, so a kill leaves either the old pair,
    or a new .pt with the old .json -- which load_reference refuses (sha256), never a torn file."""
    run_dir = Path(run_dir)
    _atomic_write(run_dir / "pi_ref.pt", lambda t: torch.save(policy.state_dict(), t))
    meta = {"path": "pi_ref.pt", "sha256": sha256_file(run_dir / "pi_ref.pt")}
    text = json.dumps(meta, indent=1, sort_keys=True) + "\n"
    _atomic_write(run_dir / "pi_ref.json", lambda t: t.write_text(text, encoding="utf-8"))
    return meta


def load_reference(g, model_cfg: dict, meta_path, device="cpu", policy_cls=None) -> GraphPolicy:
    """policy_cls: the policy class of the run's variant (default GraphPolicy; rerank: grl.model.rerank.RerankPolicy)."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    f = meta_path.parent / meta["path"]
    if sha256_file(f) != meta["sha256"]:
        raise ValueError(f"{f}: sha256 differs from {meta_path} -- π_ref must stay frozen")
    pol = (policy_cls or GraphPolicy)(g, model_cfg)
    pol.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
    return freeze_reference(pol.to(device))


def torch_rng_state() -> dict:
    """Every torch generator a training step may draw from (dropout masks): CPU, all CUDA devices, MPS. Tensors and
    lists only, so a checkpoint stays weights_only-loadable."""
    st = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_torch_rng_state(st: dict) -> None:
    torch.set_rng_state(st["cpu"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])
    if "mps" in st and torch.backends.mps.is_available():
        torch.mps.set_rng_state(st["mps"].cpu())


SFT_OPTIONAL = {"early_stop": "loss", "eval_epoch0": False, "train_eval_loss": False}
INIT_POLICY = "init_policy.pt"


def run_sft(policy, graph, ranks, train_ex, val_ex, cfg, logger, ckpt, device, evaluate, resume=False) -> SFTResult:
    kind, B, patience = cfg["loss"], int(cfg["batch_queries"]), int(cfg["patience"])
    if patience < 1:
        raise ValueError(f"sft.patience must be >= 1 (got {patience})")
    early = cfg.get("early_stop", "loss")
    if early not in ("loss", "reward"):
        raise ValueError(f"sft.early_stop {early!r} is not loss | reward")
    ep0, tr_eval = bool(cfg.get("eval_epoch0", False)), bool(cfg.get("train_eval_loss", False))
    by_q, val_q = _group(train_ex), _group(val_ex)
    if not by_q or not val_q:
        raise ValueError(f"SFT needs train and val examples (got {len(train_ex)} train, {len(val_ex)} val): "
                         "the val loss drives early stopping and the val reward selects the checkpoint")
    opt = torch.optim.Adam(policy.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    best_loss, bad, step, start = math.inf, 0, 0, 0
    if resume:
        st = torch.load(ckpt.last_path(), map_location="cpu", weights_only=True)
        policy.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = json.loads(st["rng"])
        set_torch_rng_state(st["torch_rng"])                  # after model / optimizer: dropout masks continue exactly
        best_loss, bad, step, start = float(st["best_loss"]), int(st["bad"]), int(st["step"]), int(st["epoch"])
        logger.truncate({"sft": step, "sft_val": start, "sft_final": -1})   # a stale final row is recomputed
    best_r = -math.inf
    if resume:
        best_r = float(st.get("best_r", -math.inf))
    if ep0 and not resume:                                    # epoch 0: the policy before any update
        policy.eval()
        torch.save({"model": policy.state_dict()}, Path(logger.dir) / INIT_POLICY)
        vl, va = sft_eval(policy, graph, ranks, val_q, kind, device, B)
        tl, ta = sft_eval(policy, graph, ranks, by_q, kind, device, B)
        v = evaluate(policy, "val")
        logger.log("sft_val", 0, val_loss=vl, val_step_acc=va, train_loss_eval=tl, train_step_acc_eval=ta,
                   **{f"val_{k}": x for k, x in v.items()})
    epochs = start
    for epoch in range(start, int(cfg["max_epochs"])):
        if bad >= patience:                                   # early-stopped (also when resumed after that save)
            break
        policy.train()
        for keys in query_batches(sorted(by_q), B, rng):
            loss, acc, _ = sft_batch(policy, graph, ranks, keys, by_q, kind, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["grad_clip"]))
            opt.step()
            step += 1
            logger.log("sft", step, loss=float(loss.detach()), step_acc=acc, grad_norm=float(gn), epoch=epoch + 1)
        policy.eval()
        vl, va = sft_eval(policy, graph, ranks, val_q, kind, device, B)
        extra = {}
        if tr_eval:
            tl, ta = sft_eval(policy, graph, ranks, by_q, kind, device, B)
            extra = {"train_loss_eval": tl, "train_step_acc_eval": ta}
        v = evaluate(policy, "val")
        logger.log("sft_val", epoch + 1, val_loss=vl, val_step_acc=va, **extra,
                   **{f"val_{k}": x for k, x in v.items()})
        vr = float(v["reward_mean"])
        if early == "reward":
            if vr > best_r + float(cfg["min_delta"]):
                bad = 0
            else:
                bad += 1
            best_loss = min(best_loss, vl)
        elif vl < best_loss - float(cfg["min_delta"]):
            best_loss, bad = vl, 0
        else:
            bad += 1
        best_r = max(best_r, vr)
        epochs = epoch + 1
        ckpt.save(f"sft-{epochs:04d}", {"model": policy.state_dict(), "opt": opt.state_dict(),
                                        "rng": json.dumps(rng.bit_generator.state), "torch_rng": torch_rng_state(),
                                        "best_loss": best_loss, "bad": bad, "step": step, "epoch": epochs,
                                        **({"best_r": best_r} if early == "reward" else {})},
                  float(v["reward_mean"]))
    best = ckpt.best()
    policy.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)["model"])
    policy.eval()
    tl, ta = sft_eval(policy, graph, ranks, by_q, kind, device, B)
    tr = evaluate(policy, "train")
    logger.log("sft_final", 0, best=best.name, train_loss=tl, train_step_acc=ta,
               **{f"train_{k}": x for k, x in tr.items()})
    ref = save_reference(policy, logger.dir)
    return SFTResult(best, ref, float(ckpt.index["top"][0]["metric"]), epochs)
