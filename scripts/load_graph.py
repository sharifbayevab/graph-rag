"""Load the latest graph artifact (data/graph/latest.json) into the dev Neo4j as a new version of the
"main" graph (GraphStore.load_graph: UNWIND batches, full parity check, then version swap; Task 10, R27, R28).

Refuses any Neo4j endpoint other than the dev compose service (bolt localhost:7688), with no override.
The artifact files are checked against the sha256 recorded in latest.json. Writes a load report JSON
(counts by type in Neo4j vs the artifact, parity result, dangling-edge removals, timings) and exits 1
when the load or the parity check fails (the previous version then stays active, vertex props restored).
The store itself also refuses endpoints outside config/graph.yaml store.write_endpoints and takes a per-graph
load lock; --break-stale-lock breaks a lock older than store.lock_stale_s (a load that died)."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from graphrag import config
from graphrag.kg.artifact import EDGES_FILE, NODES_FILE, read_graph
from graphrag.kg.store import DEFAULT_GRAPH, GraphLoadError, GraphStore
from graphrag.stores.guard import dev_neo4j_problem


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", default="data/graph/latest.json")
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--report", default="data/graph/load_report.json")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=None, help="parity on N random vertices/edges (default: all)")
    ap.add_argument("--break-stale-lock", action="store_true")
    args = ap.parse_args()
    print(f"neo4j endpoint: {config.NEO4J_URI}", flush=True)
    problem = dev_neo4j_problem(config.NEO4J_URI)
    if problem:
        print(f"REFUSED: {problem} (the graph load writes only to the dev Neo4j)", file=sys.stderr)
        return 1
    latest = json.loads(config.path(args.latest).read_text(encoding="utf-8"))
    d, version = config.path(latest["dir"]), latest["version"]
    for name in (NODES_FILE, EDGES_FILE):
        if _sha(d / name) != latest["sha256"][name]:
            print(f"REFUSED: {d / name} sha256 differs from {args.latest}", file=sys.stderr)
            return 1
    t = time.time()
    nodes, edges = read_graph(d)
    t_read = time.time() - t
    print(f"artifact {version}: {len(nodes)} nodes, {len(edges)} edges (read {t_read:.1f}s)", flush=True)
    g = GraphStore()
    try:
        g.init_schema()
        try:
            report = g.load_graph(nodes, edges, version, graph=args.graph, batch=args.batch, sample=args.sample,
                                  break_stale_lock=args.break_stale_lock)
            rc = 0
        except GraphLoadError as e:
            report, rc = e.report, 1
    finally:
        g.close()
    report = {"artifact_dir": latest["dir"], "t_read_s": round(t_read, 2), **report}
    out = config.path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    par = report.get("parity") or {}
    print(f"load {'OK' if rc == 0 else 'FAILED'}: {report.get('error', '')}")
    print(f"parity ok={par.get('ok')} nodes {par.get('nodes_neo4j')}/{par.get('nodes_artifact')} "
          f"edges {par.get('edges_neo4j')}/{par.get('edges_artifact')} "
          f"{'full' if par.get('full') else 'sampled'} check of {par.get('checked_nodes')} vertices / "
          f"{par.get('checked_edges')} edges: mismatches nodes={par.get('node_mismatches')} "
          f"edges={par.get('edge_mismatches')}")
    for k, v in (par.get("edges_by_type") or {}).items():
        print(f"  {k}: neo4j {v['neo4j']} artifact {v['artifact']}")
    print(f"swap: {report.get('swap')}; dangling-edge removals: {len(report.get('dangling_edge_removals') or [])}")
    print(f"staged vertices {report.get('staged_vertices')}, cleared {report.get('staging_cleared')}, "
          f"purged staging {report.get('purged_staging')}; orphans of the previous version: {report.get('orphans')}")
    print(f"active version: {report.get('active_version')}; load time {report.get('t_total_s')}s; report {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
