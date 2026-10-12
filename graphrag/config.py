"""Config: YAML files in config/ plus service endpoints from the environment (.env)."""
import copy
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=None)
def _read(name: str) -> dict:
    with open(ROOT / "config" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_OVERRIDE: dict[str, dict] = {}


def load(name: str) -> dict:
    """config/<name>.yaml (or the active `overridden` config) as a fresh deep copy: callers may
    mutate the result without affecting any other reader."""
    return copy.deepcopy(_OVERRIDE.get(name) or _read(name))


@contextmanager
def overridden(name: str, cfg: dict):
    """Within the block every `load(name)` returns a copy of `cfg` (e.g. build_corpus.py --set),
    so deep readers (extract, pdf, chunker) see the run's overrides; restored on exit. Entering and
    leaving also reset the config-bound singletons (extract._VISION, ingest._AGENDA)."""
    prev = _OVERRIDE.get(name)
    _OVERRIDE[name] = copy.deepcopy(cfg)
    _reset_singletons()
    try:
        yield
    finally:
        if prev is None:
            _OVERRIDE.pop(name, None)
        else:
            _OVERRIDE[name] = prev
        _reset_singletons()


def _reset_singletons() -> None:
    """Drop process-wide objects built from the config (the vision transcriber and the agenda
    segmenters), so a run inside / after `overridden` never reuses another run's settings."""
    import sys
    ex, ing = sys.modules.get("graphrag.data.extract"), sys.modules.get("graphrag.data.ingest")
    if ex is not None:
        ex._VISION = None
    if ing is not None:
        ing._AGENDA.clear()


def path(p: str) -> Path:
    """Resolve a config path relative to the v2 package root."""
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q).resolve()


PG_DSN = os.environ.get("GRL_PG_DSN", "postgresql://graphrag:graphrag@localhost:5437/graphrag")
QDRANT_URL = os.environ.get("GRL_QDRANT_URL", "http://localhost:6343")
NEO4J_URI = os.environ.get("GRL_NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("GRL_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("GRL_NEO4J_PASSWORD", "graphrag-dev")
