"""Step-2 acceptance (spec §11): a random chunk is found lexically, by vector, and via the
graph; its parent in Postgres equals its graph vertex; store counts agree with records.

Queries use the indexed field (data.yaml embed_field, index_text): the vector query embeds it,
the lexical query tokenizes it. A chunk whose text_norm is identical to other chunks' counts as
found if any of them is hit (identical requisites/procedural lines recur across documents)."""
import random
import sys
import time

import numpy as np

from graphrag import config
from graphrag.data.embed import Encoder
from graphrag.data.ingest import read_records
from graphrag.retrieval import lexical, vector
from graphrag.stores import pg
from graphrag.stores.sync import Stores

N, SEED, K = 200, 13, 10


def _pct(xs: list[float]) -> str:
    a = np.asarray(xs) * 1000
    return f"p50={np.percentile(a, 50):.1f}ms p95={np.percentile(a, 95):.1f}ms"


def main() -> int:
    cfg = config.load("data")
    field = cfg["embed_field"]
    records = read_records(config.path(cfg["records_dir"]))
    by_norm: dict[str, set[str]] = {}
    for r in records:
        for c in r.chunks:
            by_norm.setdefault(c.text_norm, set()).add(c.chunk_id)
    sample = random.Random(SEED).sample([(r, c) for r in records for c in r.chunks], N)
    st = Stores.open(Encoder())
    ok = {"lexical": 0, "vector": 0, "graph": 0}
    via_dup = {"lexical": 0, "vector": 0}            # found only through an identical-text twin
    t_lex: list[float] = []
    t_vec: list[float] = []
    t_emb: list[float] = []
    fails: list[str] = []
    for r, c in sample:
        same = by_norm[c.text_norm]
        qtext = getattr(c, field) or c.text_norm
        t = time.perf_counter()
        lex = {h[0] for h in lexical.search_terms(st.pg, lexical.query_terms(qtext, r.meta.lang[0]), K)}
        t_lex.append(time.perf_counter() - t)
        t = time.perf_counter()
        q = st.encoder.encode_queries([qtext])[0]
        t_emb.append(time.perf_counter() - t)
        t = time.perf_counter()
        vec = {h[0] for h in vector.search(st.qdrant, st.collection, q, K)}
        t_vec.append(time.perf_counter() - t)
        row = pg.chunk_rows(st.pg, [c.chunk_id]).get(c.chunk_id)
        node = st.graph.node(c.parent)
        path = st.graph.path_to_document(c.parent)
        f_lex, f_vec = bool(lex & same), bool(vec & same)
        ok["lexical"] += f_lex
        ok["vector"] += f_vec
        via_dup["lexical"] += f_lex and c.chunk_id not in lex
        via_dup["vector"] += f_vec and c.chunk_id not in vec
        g = bool(row and node and row["parent"] == node["node_id"] and path and path[0] == r.doc_id)
        ok["graph"] += g
        if not (f_lex and f_vec and g):
            fails.append(f"{c.chunk_id}: lex={f_lex} vec={f_vec} graph={g} tokens={c.tokens} "
                         f"dups={len(same)} text={c.text_norm[:80]!r}")
    n_chunks = sum(len(r.chunks) for r in records)
    n_nodes = sum(len(r.structure) for r in records)
    counts = pg.counts(st.pg)
    q_points = st.qdrant.count(st.collection, exact=True).count
    g_nodes = st.graph.count_nodes()
    st.close()
    rates = {k: v / N for k, v in ok.items()}
    print("rates:", rates)
    n_dup = sum(len(by_norm[c.text_norm]) > 1 for _, c in sample)
    print(f"sampled chunks with identical-text twins: {n_dup}; found only via a twin: {via_dup}")
    print(f"latency lexical {_pct(t_lex)}; vector search {_pct(t_vec)}; query embedding {_pct(t_emb)}")
    print(f"records docs={len(records)} pg={counts['documents']}; records chunks={n_chunks} "
          f"pg={counts['chunks']} qdrant={q_points}; records nodes={n_nodes} pg={counts['nodes']} "
          f"neo4j={g_nodes}")
    for f in fails:
        print("FAIL", f)
    consistent = (counts["chunks"] == n_chunks == q_points and counts["documents"] == len(records)
                  and counts["nodes"] == n_nodes == g_nodes)
    passed = rates["lexical"] >= 0.95 and rates["vector"] >= 0.95 and rates["graph"] == 1.0 and consistent
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
