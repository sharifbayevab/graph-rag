"""Shared cached / verified LLM calls of the M2 graph passes (ref_resolve, entities, audit).

Mirrors the agenda / vision pattern (graphrag.data.agenda, graphrag.data.pdf):

* every answer is cached in `cache/kg/<pass>/<sha256>.json` (tracked in git), key =
  sha256(model, prompt_version, prompt_digest, input); prompt_digest = sha256(prompt + schema),
  so a changed prompt or schema never reuses an old answer; writes are atomic (tmp + os.replace);
* API calls happen only when GRL_KG_API=1 (or `llm.api: true` in graph.yaml). Otherwise a cache
  miss raises KgCacheMiss: a graph build is reproducible offline or fails loudly;
* budgets from graph.yaml `llm`: `passes.<pass>.max_calls` per pass is CUMULATIVE (Ruling R31 (5)): the
  calls already recorded for the pass in the ledger plus this run's calls not yet recorded there;
  `max_calls_total` caps all passes of one KgLLM instance (one run); exceeding either raises
  KgBudgetExceeded before the call;
* `verify_fn(answer) -> bool | (bool, reason)` runs on EVERY answer (cached or fresh); a rejected
  answer is kept in the cache (the rejection is reproducible) and returned as None;
* stats per pass: {hits, api_calls, rejected, offline_miss} (+ accepted);
* every API call is added to the cumulative ledger cache/kg/ledger.json {pass: {date: calls}}.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from graphrag import config

CACHE_ROOT = "cache/kg"
API_ENV = "GRL_KG_API"
LEDGER_FILE = "ledger.json"


class KgCacheMiss(RuntimeError):
    """No cached answer and API calls are off (set GRL_KG_API=1 to fetch)."""


class KgBudgetExceeded(RuntimeError):
    """The cumulative per-pass budget (ledger) or the total API call budget of this run would be exceeded."""


def prompt_digest(prompt: str, schema: dict) -> str:
    return hashlib.sha256((prompt + "\n" + json.dumps(schema, sort_keys=True, ensure_ascii=False))
                          .encode("utf-8")).hexdigest()


def _canon_input(inp: Any) -> str:
    return inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False, sort_keys=True)


def _write_json(f: Path, rec: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, f)


class KgLLM:
    def __init__(self, cfg: dict | None = None, cache_root: Path | str | None = None, client=None,
                 allow_api: bool | None = None):
        g = cfg if cfg is not None else config.load("graph")
        llm = g.get("llm") or {}
        self.model: str = llm.get("model", "gpt-4.1")
        self.max_total: int = int(llm.get("max_calls_total", 0))
        self.passes: dict = dict(llm.get("passes") or {})
        self.cache_root = config.path(str(cache_root or llm.get("cache_dir") or CACHE_ROOT))
        if allow_api is None:
            allow_api = bool(llm.get("api")) or os.environ.get(API_ENV) == "1"
        self.allow_api = allow_api
        self._client = client
        self._lock = threading.Lock()
        self.stats: dict[str, Counter] = defaultdict(Counter)
        self._ledgered: Counter = Counter()             # this run's calls already added to the ledger
        self.last_raw: str | None = None

    # --- cache ------------------------------------------------------------------------------------
    def _pass_cfg(self, pass_name: str) -> dict:
        if pass_name not in self.passes:
            raise KeyError(f"graph.yaml llm.passes has no pass {pass_name!r}")
        return self.passes[pass_name]

    def cache_key(self, pass_name: str, inp: str, digest: str) -> str:
        pv = self._pass_cfg(pass_name).get("prompt_version")
        return hashlib.sha256(f"{self.model}\n{pv}\n{digest}\n{inp}".encode("utf-8")).hexdigest()

    def cache_file(self, pass_name: str, key: str) -> Path:
        return self.cache_root / pass_name / f"{key}.json"

    def total_api_calls(self) -> int:
        return sum(c["api_calls"] for c in self.stats.values())

    def ledger(self) -> dict:
        f = self.cache_root / LEDGER_FILE
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def cumulative_calls(self, pass_name: str) -> int:
        """API calls ever made for `pass_name`: the ledger sum plus this run's calls not yet recorded there."""
        led = sum(int(v) for v in (self.ledger().get(pass_name) or {}).values())
        return led + self.stats[pass_name]["api_calls"] - self._ledgered[pass_name]

    # --- API --------------------------------------------------------------------------------------
    def _reserve(self, pass_name: str) -> None:
        with self._lock:
            cap = int(self._pass_cfg(pass_name).get("max_calls", 0))
            used = self.cumulative_calls(pass_name)
            if used + 1 > cap:
                raise KgBudgetExceeded(f"{pass_name}: cumulative budget of {cap} API calls would be exceeded "
                                       f"({used} used, ledger {LEDGER_FILE})")
            if self.total_api_calls() + 1 > self.max_total:
                raise KgBudgetExceeded(f"total LLM budget of {self.max_total} API calls per run would be exceeded")
            self.stats[pass_name]["api_calls"] += 1
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(max_retries=5)

    def _call(self, pass_name: str, prompt: str, schema: dict, user: str) -> str:
        self._reserve(pass_name)
        r = self._client.chat.completions.create(
            model=self.model, temperature=0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": pass_name, "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}])
        self._ledger(pass_name)
        return (r.choices[0].message.content or "").strip()

    def _ledger(self, pass_name: str) -> None:
        """Cumulative API spend: cache/kg/ledger.json {pass: {ISO date: calls}} (tracked, auditable)."""
        import datetime as _dt
        with self._lock:
            f = self.cache_root / LEDGER_FILE
            led = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            day = _dt.date.today().isoformat()
            led.setdefault(pass_name, {})
            led[pass_name][day] = int(led[pass_name].get(day, 0)) + 1
            _write_json(f, led)
            self._ledgered[pass_name] += 1

    # --- main entry -------------------------------------------------------------------------------
    def cached_json(self, pass_name: str, inp: Any, schema: dict,
                    verify_fn: Callable[[Any], Any] | None = None, *, prompt: str) -> Any | None:
        """The verified JSON answer for `inp` (str or JSON-able), or None when verify_fn rejects it
        (or the answer is not valid JSON). Raises KgCacheMiss offline, KgBudgetExceeded over budget."""
        st = self.stats[pass_name]
        user = _canon_input(inp)
        digest = prompt_digest(prompt, schema)
        key = self.cache_key(pass_name, user, digest)
        f = self.cache_file(pass_name, key)
        raw = None
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("model") == self.model and rec.get("prompt_digest") == digest \
                    and rec.get("prompt_version") == self._pass_cfg(pass_name).get("prompt_version"):
                raw = rec["answer"]
                st["hits"] += 1
        if raw is None:
            if not self.allow_api:
                st["offline_miss"] += 1
                raise KgCacheMiss(f"{pass_name}: answer {key[:12]} is not in the cache ({f.parent}) and API "
                                  f"calls are off; set {API_ENV}=1 to fetch it")
            raw = self._call(pass_name, prompt, schema, user)
            _write_json(f, {"model": self.model, "pass": pass_name,
                            "prompt_version": self._pass_cfg(pass_name).get("prompt_version"),
                            "prompt_digest": digest, "input": user, "answer": raw})
        self.last_raw = raw                                 # the raw answer (also when rejected)
        try:
            ans = json.loads(raw)
        except (TypeError, ValueError):
            st["rejected"] += 1
            st["rejected_bad_json"] += 1
            return None
        if verify_fn is not None:
            res = verify_fn(ans)
            ok, why = (res if isinstance(res, tuple) else (bool(res), None))
            if not ok:
                st["rejected"] += 1
                if why:
                    st[f"rejected_{why}"] += 1
                return None
        st["accepted"] += 1
        return ans

    def stats_json(self) -> dict:
        return {p: {k: c.get(k, 0) for k in ("hits", "api_calls", "rejected", "offline_miss", "accepted")}
                | {k: v for k, v in c.items() if k.startswith("rejected_")}
                for p, c in sorted(self.stats.items())}


_DEFAULT: KgLLM | None = None


def get_llm() -> KgLLM:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KgLLM()
    return _DEFAULT


def cached_json(pass_name: str, inp: Any, schema: dict, verify_fn=None, *, prompt: str):
    """Module-level shortcut on the process-wide KgLLM (graph.yaml config, env gate)."""
    return get_llm().cached_json(pass_name, inp, schema, verify_fn, prompt=prompt)
