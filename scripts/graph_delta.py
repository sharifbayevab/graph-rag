"""Difference of two graph artifacts by edge identity (graphrag.kg.artifact.edge_key) and node id.
    python scripts/graph_delta.py data/graph/<old> data/graph/<new> [--out docs/m3/graph_delta.json]"""
import argparse
import json
import sys

from graphrag.kg.artifact import edge_key, read_graph


def graph_delta(a_dir, b_dir) -> dict:
    na, ea = read_graph(a_dir)
    nb, eb = read_graph(b_dir)
    ka = {json.dumps(edge_key(e), ensure_ascii=False): e for e in ea}
    kb = {json.dumps(edge_key(e), ensure_ascii=False): e for e in eb}
    return {"edges_added": sorted(kb.keys() - ka.keys()), "edges_removed": sorted(ka.keys() - kb.keys()),
            "edges_attrs_changed": sorted(k for k in ka.keys() & kb.keys() if ka[k].attrs != kb[k].attrs),
            "nodes_added": sorted({n.node_id for n in nb} - {n.node_id for n in na}),
            "nodes_removed": sorted({n.node_id for n in na} - {n.node_id for n in nb})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    d = graph_delta(a.old, a.new)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: len(v) for k, v in d.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
