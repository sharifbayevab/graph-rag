"""Guard for destructive store operations (build_indexes.py --reset): only the local v2 docker
services may be wiped, unless GRL_ALLOW_RESET=1 is set explicitly."""
from __future__ import annotations

import os
from urllib.parse import parse_qs, urlsplit

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
V2_PORTS = {"postgres": 5437, "qdrant": 6343, "neo4j": 7688}
# libpq honours these URI query parameters over the netloc ("…@localhost:5437/db?host=prod")
_LIBPQ_TARGET = ("host", "hostaddr", "port", "service")


def _problem(kind: str, url: str) -> str | None:
    try:
        u = urlsplit(url)
        scheme, host, port = u.scheme, u.hostname, u.port
    except ValueError:                       # e.g. a non-numeric port
        scheme = host = port = None
    if not scheme or host is None or port is None:
        return f"{kind}: cannot parse host:port from {url!r} (use a URL with an explicit port)"
    if host not in LOCAL_HOSTS or port != V2_PORTS[kind]:
        return f"{kind}: {host}:{port} is not the local v2 service (localhost:{V2_PORTS[kind]})"
    if kind == "postgres":
        q = parse_qs(u.query, keep_blank_values=True)
        for k in _LIBPQ_TARGET:
            for v in q.get(k, []):
                ok = (k in ("host", "hostaddr") and v in LOCAL_HOSTS) or (k == "port" and v == str(V2_PORTS[kind]))
                if not ok:
                    return f"{kind}: query parameter {k}={v!r} overrides {host}:{port} (not the local v2 service)"
    return None


def reset_problems(pg_dsn: str, qdrant_url: str, neo4j_uri: str, env=None) -> list[str]:
    """Reasons to refuse a reset; [] when every endpoint is a local v2 port or
    env GRL_ALLOW_RESET=1 overrides the check."""
    env = os.environ if env is None else env
    if env.get("GRL_ALLOW_RESET") == "1":
        return []
    return [p for p in (_problem("postgres", pg_dsn), _problem("qdrant", qdrant_url),
                        _problem("neo4j", neo4j_uri)) if p]


def neo4j_reset_problem(neo4j_uri: str, env=None) -> str | None:
    """Why GraphStore.reset_all must not wipe `neo4j_uri` (Ruling R31 (5): the guard lives inside the store,
    not only in build_indexes.py); None for the local v2 Neo4j (localhost:7688) or with GRL_ALLOW_RESET=1
    (the same override as reset_problems)."""
    env = os.environ if env is None else env
    if env.get("GRL_ALLOW_RESET") == "1":
        return None
    return _problem("neo4j", neo4j_uri)


def dev_neo4j_problem(neo4j_uri: str) -> str | None:
    """Why `neo4j_uri` is not the dev Neo4j (compose grl-neo4j, bolt localhost:7688); None when it is.
    The graph load (scripts/load_graph.py) and its integration tests write only there: no override
    (Ruling R27 (8))."""
    p = _problem("neo4j", neo4j_uri)
    if p is None and urlsplit(neo4j_uri).scheme not in ("bolt", "neo4j"):
        p = f"neo4j: scheme of {neo4j_uri!r} is not bolt:// or neo4j://"
    return p


def graph_write_problem(neo4j_uri: str, allowed) -> str | None:
    """Why GraphStore.load_graph / drop_versions must not write to `neo4j_uri`; None when it is a bolt:// or
    neo4j:// URI whose host:port is in `allowed` (config/graph.yaml store.write_endpoints, "host:port"
    strings). Ruling R28 (4): enforced inside the store, not only in the script; no env override."""
    try:
        u = urlsplit(neo4j_uri)
        scheme, host, port = u.scheme, u.hostname, u.port
    except ValueError:
        scheme = host = port = None
    if scheme not in ("bolt", "neo4j"):
        return f"neo4j: scheme of {neo4j_uri!r} is not bolt:// or neo4j://"
    if host is None or port is None:
        return f"neo4j: cannot parse host:port from {neo4j_uri!r} (use a URL with an explicit port)"
    ok = set()
    for a in allowed or ():
        h, _, p = str(a).rpartition(":")
        if h and p.isdigit():
            ok.add((h.strip("[]").lower(), int(p)))
    if (host.lower(), port) not in ok:
        return f"neo4j: {host}:{port} is not an allowlisted graph write endpoint {sorted(map(str, allowed or ()))}"
    return None
