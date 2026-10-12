"""Load data/records into Postgres + Qdrant + Neo4j via the single write path
(graphrag.stores.sync.upsert_document). Idempotent: re-running without --reset leaves the
same stores. --reset wipes the v2 stores first; it refuses unless every endpoint is a local v2
service (localhost:5437 / 6343 / 7688) or GRL_ALLOW_RESET=1 is set. In Neo4j the reset removes the graph
load bookkeeping too (GraphMeta active version / lock, GraphEvent), so reload the artifact with
scripts/load_graph.py afterwards (Ruling R28 (5))."""
import argparse
import sys
import time

from graphrag import config
from graphrag.data.embed import Encoder
from graphrag.data.ingest import read_records
from graphrag.stores import pg
from graphrag.stores.guard import reset_problems
from graphrag.stores.sync import Stores, upsert_document


class TimedEncoder:
    """Wraps the encoder to measure embedding throughput; encoding itself is unchanged."""

    def __init__(self, enc):
        self._enc, self.seconds, self.n = enc, 0.0, 0

    def __getattr__(self, name):
        return getattr(self._enc, name)

    def encode_passages(self, texts, *a, **kw):
        t = time.perf_counter()
        out = self._enc.encode_passages(texts, *a, **kw)
        self.seconds += time.perf_counter() - t
        self.n += len(texts)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    if args.reset:
        print(f"reset endpoints: postgres={config.PG_DSN.rsplit('@', 1)[-1]} "
              f"qdrant={config.QDRANT_URL} neo4j={config.NEO4J_URI}", flush=True)
        problems = reset_problems(config.PG_DSN, config.QDRANT_URL, config.NEO4J_URI)
        if problems:
            for p in problems:
                print(f"REFUSED --reset: {p}", file=sys.stderr)
            print("set GRL_ALLOW_RESET=1 to wipe these endpoints anyway", file=sys.stderr)
            return 1
    records = read_records(config.path(config.load("data")["records_dir"]))
    enc = TimedEncoder(Encoder())
    st = Stores.open(enc)
    if args.reset:
        with st.pg, st.pg.cursor() as cur:
            cur.execute("TRUNCATE postings, chunks, nodes, documents")
        st.graph.reset_all()                         # :Node + :GraphMeta + :GraphEvent (R28 (5))
        st.qdrant.delete_collection(st.collection)
        st.close()
        st = Stores.open(enc)                        # recreates the collection
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        upsert_document(st, rec)
        if i % 25 == 0 or i == len(records):
            print(f"{i}/{len(records)} {time.time() - t0:.0f}s", flush=True)
    t_load = time.time() - t0
    t = time.time()
    pg.analyze(st.pg)                                # planner stats + visibility map (index-only BM25)
    print(f"vacuum analyze: {time.time() - t:.1f}s")
    print(f"embedding: {enc.n} passages in {enc.seconds:.1f}s = {enc.n / max(enc.seconds, 1e-9):.1f}/s")
    print(f"load: {t_load:.1f}s; total: {time.time() - t_start:.1f}s")
    print(f"records: docs={len(records)} nodes={sum(len(r.structure) for r in records)} "
          f"chunks={sum(len(r.chunks) for r in records)}")
    print("postgres:", pg.counts(st.pg))
    print("qdrant points:", st.qdrant.count(st.collection, exact=True).count)
    print("neo4j nodes:", st.graph.count_nodes())
    st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
