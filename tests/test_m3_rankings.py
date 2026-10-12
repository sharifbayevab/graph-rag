import numpy as np
import pytest

from graphrag.retrieval.rankings import TitleIndex, dense_topk, load_rankings, rrf_rows, write_rankings


def test_dense_topk_is_exact_and_breaks_ties_by_row():
    v = np.array([[1, 0], [0.6, 0.8], [0.6, 0.8], [0, 1]], np.float64)
    rows, s = dense_topk(v, np.array([0.6, 0.8]), 3)
    assert rows.tolist() == [1, 2, 3] and s.tolist() == [1.0, 1.0, 0.8]


def test_title_index_bm25():
    ti = TitleIndex(["Солиқ кодекси", "Банклар тўғрисида", "", "Soliq kodeksi 5-modda"], 1.2, 0.75)
    out = ti.search("солиқ кодекси", 10)
    assert {r for r, _ in out} == {0, 3} and all(s > 0 for _, s in out)
    assert ti.search("мавжуд эмас", 10) == []


def test_rrf_rows_ties_by_row():
    assert rrf_rows(np.array([3, 1]), np.array([1, 3]), top=5).tolist() == [1, 3]


def test_rankings_roundtrip_padding_and_version_guard(tmp_path):
    q = [("a|cyr", "савол"), ("syn-1|q", "бошқа савол")]
    arr = {"qvec": np.eye(2, 4, dtype=np.float32),
           "lex_idx": np.array([[2, 0, -1], [1, -1, -1]], np.int32),
           "lex_score": np.array([[2, 1, np.nan], [3, np.nan, np.nan]], np.float32),
           "vec_idx": np.array([[0, 1, 2], [2, 1, 0]], np.int32),
           "vec_score": np.array([[.9, .8, .1], [.7, .6, .5]], np.float32),
           "title_idx": np.array([[-1, -1, -1], [4, -1, -1]], np.int32),
           "title_score": np.array([[np.nan] * 3, [1.5, np.nan, np.nan]], np.float32)}
    ver = write_rankings(tmp_path, q, arr, {"graph_version": "toy", "top": 3})
    r = load_rankings(tmp_path / "latest.json", graph_version="toy")
    assert r.keys() == ["a|cyr", "syn-1|q"] and r.get("a|cyr").lex.tolist() == [2, 0]
    assert r.get("a|cyr").title.tolist() == [] and r.get("syn-1|q").title.tolist() == [4]
    assert write_rankings(tmp_path / "again", q, arr, {"graph_version": "toy", "top": 3}) == ver
    for f in ("lex_idx.npy", "qvec.npy", "queries.jsonl", "meta.json"):
        assert (tmp_path / ver / f).read_bytes() == (tmp_path / "again" / ver / f).read_bytes()
    with pytest.raises(ValueError, match="graph"):
        load_rankings(tmp_path / "latest.json", graph_version="other")


@pytest.mark.integration
@pytest.mark.model
@pytest.mark.corpus
def test_artifact_equals_live_bm25_rebuilds_identically_and_cuts_the_ranking():
    """U9 is faithful (stored lexical rows = live BM25 top-200); a rebuild on 12 queries is byte-identical; the manual
    corpus lexical example: lexical_search(5) = the first 5 stored rows (their tokens fit L_max)."""
    from graphrag import config
    from graphrag.data.embed import Encoder
    from graphrag.env.data import load_env_data
    from graphrag.env.start import make_query
    from graphrag.kg.artifact import read_graph
    from graphrag.retrieval import lexical
    from graphrag.retrieval.executor import Budget, ranked_delta
    from graphrag.retrieval.rankings import QREAL_LATEST, TOP, build_rankings
    from graphrag.stores import pg
    data = load_env_data()
    ranks = load_rankings(QREAL_LATEST, graph_version=data.version)
    conn = pg.connect()
    try:
        for qk in ranks.keys()[:20]:
            r = ranks.get(qk)
            assert r.lex.tolist() == [data.chunk_row[c] for c, _ in lexical.search(conn, r.text, k=TOP)], qk
        qs = [(k, ranks.get(k).text) for k in ranks.keys()[:12]]
        nodes, _ = read_graph(config.path(f"data/graph/{data.version}"))
        bm = config.load("data")["bm25"]
        ti = TitleIndex([n.title for n in nodes], float(bm["k1"]), float(bm["b"]))
        enc = Encoder(device="cpu")
        a1, a2 = build_rankings(qs, data, conn, enc, ti, TOP), build_rankings(qs, data, conn, enc, ti, TOP)
        assert all(a1[k].tobytes() == a2[k].tobytes() for k in a1)
    finally:
        conn.close()
    rk = ranks.get(ranks.keys()[0])
    q = make_query(data, rk, config.load("data")["start"], 10)
    assert data.chunk_tokens[rk.lex[:5]].sum() <= 8000
    assert ranked_delta(q.lex, frozenset(), 5, Budget(0, 0, 24, 8000), data.chunk_tokens) == rk.lex[:5].tolist()
