"""Vectorised env over the M3 Env and policy rollouts (IMPLEMENTATION.md §6.5, §8.5).

SerialVecEnv runs episodes in-process; ProcVecEnv shards them over spawn workers by query (deterministic owner
i % W over the sorted keys), so results do not depend on the number of workers. One Env per query (its memo is
per query, M3 Task 13), shared by the G_s copies of that query. The policy side sees StateDesc only; the target set
reaches a worker only through rewards() after the episodes end. rollout() encodes every query ONCE (GNN outside the
loop over attempts, §8.5) and steps all episodes in lockstep; sampling and argmax run on CPU (seeded generator,
first maximum) so trajectories do not depend on the device of the network."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np
import torch

from graphrag.env.mdp import Env
from grl.model.observe import describe
from grl.train.fbeta import grl_reward
from grl.threads import pin_blas


class VecEnvError(RuntimeError):
    pass


@dataclass
class Episode:
    qkey: str
    qi: int
    copy: int
    descs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logp_b: list = field(default_factory=list)      # behaviour log pi(a_t|s_t) at collection (PPO epochs > 1)


class SerialVecEnv:
    def __init__(self, data, ranks, env_cfg=None, start_cfg=None, env_cls=None, env_kw=None, describe_kw=None):
        self.data, self.ranks = data, ranks
        self.env_cfg, self.start_cfg = env_cfg, start_cfg
        self.env_cls, self.env_kw = env_cls or Env, dict(env_kw or {})
        self.describe_kw = dict(describe_kw or {})
        self.items, self.states, self._env, self._q = [], [], {}, {}

    def reset(self, items):
        self.items, self._q = list(items), {}
        for k, _ in self.items:
            if k in self._q:
                continue
            if k not in self._env:                   # one Env per key, kept across resets (static _nb cache)
                self._env[k] = self.env_cls(self.data, self.env_cfg, self.start_cfg, **self.env_kw)
            self._q[k] = self._env[k].query(self.ranks.get(k))   # query() clears the per-query memo
        self.states = [self._env[k].reset(self._q[k]) for k, _ in self.items]
        return [self._desc(i) for i in range(len(self.items))]

    def _desc(self, i):
        s, k = self.states[i], self.items[i][0]
        return None if s.done else describe(self._env[k], self._q[k], s, **self.describe_kw)

    def step(self, actions):
        if len(actions) != len(self.items):
            raise ValueError(f"{len(actions)} actions for {len(self.items)} episodes")
        for i, a in enumerate(actions):
            if a is not None:
                k = self.items[i][0]
                self.states[i] = self._env[k].step(self._q[k], self.states[i], int(a))
        return [self._desc(i) for i in range(len(self.items))]

    def finals(self):
        return list(self.states)

    def rewards(self, golds, w):
        return [grl_reward(self.data, self._q[k].depth, s.context, golds[k], w, self._env[k].limits.T_max)
                for (k, _), s in zip(self.items, self.states)]

    def env_keys(self) -> list[str]:
        return sorted(self._env)

    def close(self):
        pass


def _worker(conn, init):
    try:
        venv = SerialVecEnv(**init())
        conn.send(("ok", None))
    except BaseException as e:                       # noqa: BLE001 -- reported to the parent, never swallowed
        conn.send(("err", f"init: {type(e).__name__}: {e}"))
        conn.close()
        return
    while True:
        try:
            cmd, args = conn.recv()
        except EOFError:
            return
        if cmd == "close":
            conn.close()
            return
        try:
            conn.send(("ok", getattr(venv, cmd)(*args)))
        except BaseException as e:                   # noqa: BLE001
            conn.send(("err", f"{cmd}: {type(e).__name__}: {e}"))


class ProcVecEnv:
    """Workers own queries for the whole life of the env: a key is assigned on first sight (round-robin over the
    sorted new keys of a reset) and never moves, so every Env and its static _nb cache lives in exactly one worker.
    Any worker error or timeout leaves the env broken: outstanding replies are drained (or unreadable) and every
    later call raises VecEnvError instead of returning stale data."""

    def __init__(self, init, workers: int, start_method: str = "spawn", timeout_s: float = 600.0):
        pin_blas(force=True)                         # inherited by the workers (spawned after this line)
        ctx = mp.get_context(start_method)
        self.timeout, self.conns, self.procs, self.shards, self.n = float(timeout_s), [], [], [], 0
        self.owner: dict[str, int] = {}              # key -> worker, stable across resets
        self._next, self._broken = 0, None
        try:
            for _ in range(max(1, int(workers))):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, init), daemon=True)
                p.start()
                b.close()
                self.conns.append(a)
                self.procs.append(p)
            self._collect(range(len(self.conns)))
        except BaseException:
            self.close()
            raise

    def _recv(self, w):
        if not self.conns[w].poll(self.timeout):
            raise VecEnvError(f"worker {w}: no answer in {self.timeout} s (alive={self.procs[w].is_alive()})")
        try:
            st, v = self.conns[w].recv()
        except (EOFError, OSError) as e:
            raise VecEnvError(f"worker {w} died") from e
        if st == "err":
            raise VecEnvError(f"worker {w}: {v}")
        return v

    def _collect(self, ws):
        """One reply from every worker in ws (all read even after an error, so none is left stale)."""
        out, err = {}, None
        for w in ws:
            try:
                out[w] = self._recv(w)
            except VecEnvError as e:
                err = err or e
        if err is not None:
            self._broken = str(err)
            raise err
        return out

    def _map(self, cmd, per_worker):
        if self._broken is not None:
            raise VecEnvError(f"vec env is broken after an earlier worker error ({self._broken})")
        live = [w for w, a in enumerate(per_worker) if a is not None]
        sent = []
        for w in live:
            try:
                self.conns[w].send((cmd, per_worker[w]))
            except (BrokenPipeError, OSError) as e:
                self._broken = f"worker {w}: send failed: {e}"
                try:
                    self._collect(sent)
                except VecEnvError:
                    pass
                raise VecEnvError(self._broken) from e
            sent.append(w)
        return self._collect(live)

    def _gather(self, res):
        out = [None] * self.n
        for w, vals in res.items():
            for j, v in zip(self.shards[w], vals):
                out[j] = v
        return out

    def reset(self, items):
        W = len(self.conns)
        for k in sorted({k for k, _ in items} - set(self.owner)):
            self.owner[k] = self._next % W
            self._next += 1
        self.n = len(items)
        self.shards = [[j for j, (k, _) in enumerate(items) if self.owner[k] == w] for w in range(W)]
        return self._gather(self._map("reset", [([items[j] for j in sh],) if sh else None for sh in self.shards]))

    def step(self, actions):
        return self._gather(self._map("step", [([actions[j] for j in sh],) if sh else None for sh in self.shards]))

    def finals(self):
        return self._gather(self._map("finals", [() if sh else None for sh in self.shards]))

    def rewards(self, golds, w):
        return self._gather(self._map("rewards", [(golds, w) if sh else None for sh in self.shards]))

    def env_keys(self) -> list[list[str]]:
        """Keys whose Env each worker holds (diagnostics / tests)."""
        res = self._map("env_keys", [()] * len(self.conns))
        return [res[w] for w in range(len(self.conns))]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", ()))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for c in self.conns:
            c.close()
        self._broken = self._broken or "closed"


def qvecs(ranks, qkeys, device) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.asarray(ranks.get(k).qvec, np.float32) for k in qkeys])).to(device)


@torch.no_grad()
def rollout(policy, graph, vec, ranks, qkeys, G: int, *, greedy: bool, gen=None, device="cpu", temperature=1.0):
    if len(set(qkeys)) != len(qkeys):
        raise ValueError("rollout() needs distinct query keys (the group is G copies of one key)")
    from grl.train.data import guard_qkeys
    guard_qkeys(qkeys)
    if not greedy:
        if not isinstance(gen, torch.Generator):
            raise ValueError("sampling rollout() needs an explicit torch.Generator (no global RNG)")
        if not float(temperature) > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
    items = [(k, i) for k in qkeys for i in range(G)]
    z, qz = policy.encode(graph, qvecs(ranks, qkeys, device))         # once per query, shared by the group
    pos = {k: i for i, k in enumerate(qkeys)}
    eps = [Episode(k, pos[k], i) for k, i in items]
    descs = vec.reset(items)
    while True:
        live = [e for e, d in enumerate(descs) if d is not None]
        if not live:
            break
        lp = policy.log_probs(z, qz, [eps[e].qi for e in live], [descs[e] for e in live]).float().cpu()
        if greedy:
            a = lp.argmax(-1)
        else:
            a = torch.multinomial(torch.softmax(lp / float(temperature), -1), 1, generator=gen).squeeze(1)
        acts = [None] * len(items)
        for j, e in enumerate(live):
            eps[e].descs.append(descs[e])
            eps[e].actions.append(int(a[j]))
            eps[e].logp_b.append(float(lp[j, int(a[j])]))
            acts[e] = int(a[j])
        descs = vec.step(acts)
    return eps, vec.finals()


def real_init(rankings: str, variant: str = "full", describe_kw: dict | None = None,
              env_extra: dict | None = None, env_cfg: dict | None = None) -> dict:
    """ProcVecEnv / SerialVecEnv context on the pinned artifacts (EnvData once per worker; spawn-safe). env_extra:
    extra env kwargs (AdaptiveEnv normalise / norm_top_n from train.yaml dynamic_k). env_cfg: the run's train.yaml `env`
    incl. --set env.K_max (ruling R12); None = the env reloads train.yaml."""
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.variants import data_kw, env_spec, load_variant_data
    data = load_variant_data(variant, **data_kw(env_extra))   # struct: + the STRUCT rows (6c: + article rows)
    cls, kw = env_spec(variant)
    return {"data": data, "ranks": load_rankings(rankings, graph_version=data.version), "env_cls": cls,
            "env_cfg": env_cfg,
            "env_kw": {**kw, **(env_extra or {})}, "describe_kw": dict(describe_kw or {})}
