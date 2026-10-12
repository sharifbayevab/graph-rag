# tests/test_m7_struct_data.py
import numpy as np
import pytest

from graphrag.env.data import EnvData
from grl.env.struct_data import extend_with_struct, struct_index_text, texts_digest


def toy():
    return EnvData.build(["D", "D/СТ.1"], ["D/СТ.1#1"], [("D", "CONTAINS", "D/СТ.1")], {"D/СТ.1#1": "D/СТ.1"},
                         {"D/СТ.1": ["D/СТ.1#1"]}, {"D/СТ.1#1": 10}, np.ones((1, 4)), np.ones((2, 4)), ["CONTAINS"])


def test_rows_appended_mu_unchanged():
    d = toy()
    e = extend_with_struct(d, {"STRUCT/D": "x"}, {"STRUCT/D": np.full(4, .5)}, {"STRUCT/D": 3})
    assert e.chunk_ids == ["D/СТ.1#1", "STRUCT/D"] and e.n_base_chunks == 1
    assert e.chunk_node[1] == e.node_row["D"] and e.struct_rows == {e.node_row["D"]: 1}
    assert np.array_equal(e.mu_ptr, d.mu_ptr) and np.array_equal(e.mu_idx, d.mu_idx)      # not reachable via graph ops
    assert e.chunk_tokens[1] == 3 and np.allclose(e.chunk_vecs[0], d.chunk_vecs[0])


def test_unknown_root_rejected():
    with pytest.raises(ValueError):
        extend_with_struct(toy(), {"STRUCT/X": "x"}, {"STRUCT/X": np.ones(4)}, {"STRUCT/X": 1})


def test_extend_keeps_input_and_version():
    d = toy()
    e = extend_with_struct(d, {"STRUCT/D": "x"}, {"STRUCT/D": np.full(4, .5)}, {"STRUCT/D": 3})
    assert d.n_chunks == 1 and d.chunk_vecs.shape == (1, 4) and e.version == d.version
    assert e.chunk_row["STRUCT/D"] == 1 and e.vecs64.shape == (2, 4) and e.chunk_vecs.dtype == np.float32


def test_index_text_is_corpus_format():
    # corpus index_text = to_internal(breadcrumb) + "\n" + to_internal(text); a document's breadcrumb = its title
    t = struct_index_text("Qonun A", "Qonun A\nбоблар: 1", "uz", lambda s: len(s.split()))
    crumb, body = t.split("\n", 1)
    assert crumb == "qonun a" and body.startswith("qonun a\n") and t == t.lower()


def test_index_text_crumb_capped_from_the_left():
    words = " ".join(f"w{i}" for i in range(200))
    t = struct_index_text(words, "body", "uz", lambda s: len(s.split()))
    crumb = t.split("\n", 1)[0]
    assert len(crumb.split()) == 96 and crumb.endswith("w199")             # chunker.BREADCRUMB_MAX_TOKENS


def test_texts_digest_order_free():
    assert texts_digest({"b": "2", "a": "1"}) == texts_digest({"a": "1", "b": "2"}) != texts_digest({"a": "1"})


# --- Spark verify only (Review Focus 1): real data, no STRUCT row in any ranking or in μ⁻¹ ------------------------

@pytest.mark.slow
def test_struct_rows_outside_rankings_and_mu():
    from graphrag import config
    from graphrag.env.start import make_query
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    e = load_struct_env_data()
    nb = e.n_base_chunks
    assert e.n_chunks - nb == len(e.struct_rows) == 989
    assert e.chunk_ids[nb:] == sorted(e.chunk_ids[nb:]) and all(c.startswith("STRUCT/") for c in e.chunk_ids[nb:])
    assert not any(c.startswith("STRUCT/") for c in e.chunk_ids[:nb])
    assert int(e.mu_idx.max()) < nb and len(e.mu_ptr) == e.n_nodes + 1
    ships = sorted(config.path("data/m5/pilot").glob("*/ship/rankings/latest.json"))
    assert ships, "no snapshot rankings on this host (sync --extra a data/m5/pilot/<sha>/ship dir)"
    ranks = load_rankings(ships[0], graph_version=e.version)
    keys = ranks.keys()[:50]
    assert len(keys) == 50
    start_cfg = config.load("data")["start"]
    for k in keys:
        q = make_query(e, ranks.get(k), start_cfg, 10)
        assert q.lex.size and q.vec.size and int(q.lex.max()) < nb and int(q.vec.max()) < nb, k
        assert q.chunk_qsim.shape == (e.n_chunks,) and np.isfinite(q.chunk_qsim).all()
