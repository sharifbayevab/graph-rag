"""Order-preserving process pool over queries (M3 Task 13).

`run(fn, items, workers, start_method=None, **ctx)` returns `[fn(ctx, x) for x in items]` -- the same list, in the
same order, whatever the number of workers; `fn(ctx, item)` must be a module-level function (picklable by name).

Context. `ctx` is handed to every worker once (pool initializer), not per item:
- "fork": inherited copy-on-write, nothing is pickled;
- "spawn": pickled once per worker. For the heavy read-only context (EnvData, rankings, gold) pass
  `init=<module-level callable>` instead: each worker calls `init()` once and merges the returned dict into its
  context, so EnvData is loaded once per worker (the start-up cost is measured by scripts/profile_oracle.py).
  In the serial path (workers <= 1) `init()` runs once in-process, so both paths see the same context.

Start method (Task 13 decision): train.yaml `parallel.start_method`, "spawn" on this host. Forking a process that has
imported torch and used Accelerate's threaded BLAS is not fork-safe on macOS; spawn is the portable choice (and the
only one on Windows). "fork" stays available on Linux hosts.

Items go to the workers one at a time (chunksize 1) because per-query cost varies ~20x; results are collected in
item order. Failure is loud, never a hang (review I1): the pool is a concurrent.futures.ProcessPoolExecutor, so a
worker that dies hard (segfault, OOM kill, os._exit) or an `init` that raises breaks the pool (BrokenProcessPool)
instead of blocking forever / respawning workers; run() raises ParallelRunError naming the first failing item (its
index and repr) and chains the cause. An exception raised by `fn` is re-raised the same way.
Results do not depend on which worker computes an item: Env's memo is per query (cleared in Env.query) and every
oracle cache is per search."""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

_CTX: dict = {}


class ParallelRunError(RuntimeError):
    pass


def _set_ctx(ctx: dict) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    init = _CTX.get("init")
    if init is not None:
        _CTX.update(init())


def _call(fn, item):
    return fn(_CTX, item)


def default_start_method() -> str:
    from graphrag import config
    return str((config.load("train").get("parallel") or {}).get("start_method", "spawn"))


def _name(i: int, item) -> str:
    r = repr(item)
    return f"item {i} ({r[:200] + '...' if len(r) > 200 else r})"


def run(fn, items, workers: int, start_method: str | None = None, **ctx) -> list:
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        _set_ctx(ctx)
        return [fn(_CTX, x) for x in items]
    method = start_method or default_start_method()
    out = []
    with ProcessPoolExecutor(max_workers=min(int(workers), len(items)), mp_context=mp.get_context(method),
                             initializer=_set_ctx, initargs=(ctx,)) as ex:
        futs = [ex.submit(_call, fn, x) for x in items]
        for i, f in enumerate(futs):
            try:
                out.append(f.result())
            except BaseException as e:
                for g in futs[i + 1:]:
                    g.cancel()
                where = (f"items {i}..{len(items) - 1} (the pool broke: a worker died or init failed; items "
                         f"0..{i - 1} completed, the culprit is at or after {_name(i, items[i])})"
                         if isinstance(e, BrokenProcessPool) else _name(i, items[i]))
                raise ParallelRunError(f"parallel.run failed at {where} of {len(items)} "
                                       f"({type(e).__name__}: {e})") from e
    return out
