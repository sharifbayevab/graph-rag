# advanced-rag-v2/tests/test_chunker.py
from graphrag.data.chunker import chunk_document
from graphrag.data.extract import Para
from graphrag.data.mapping import mu, mu_inverse
from graphrag.data.structure import parse_structure
from tests.helpers import words
from tests.test_structure_law import LAW

D = "QONUNLAR-0000abcd"


def _build(paras, max_tokens):
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in paras], "t",
                                max_tokens, words)
    return dt, nodes, chunk_document(dt, nodes, max_tokens, words, str.lower)


def _invariants(dt, nodes, chunks, max_tokens):
    by_id = {n.node_id: n for n in nodes}
    prev_end = -1
    for c in sorted(chunks, key=lambda c: c.start):
        assert c.text == dt.text[c.start:c.end]                    # exact substring
        assert c.start >= prev_end                                 # no overlap
        prev_end = c.end
        assert c.tokens <= max_tokens
        p = by_id[c.parent]
        assert p.start <= c.start and c.end <= p.end               # inside μ(c)
    for n in nodes:
        if n.type in ("статья", "пункт"):                          # never crosses a boundary
            for c in chunks:
                inside = n.start <= c.start and c.end <= n.end
                contains = c.start <= n.start and n.end <= c.end
                disjoint = c.end <= n.start or c.start >= n.end
                assert inside or contains or disjoint, (n.node_id, c.chunk_id)


def test_short_articles_are_single_chunks():
    dt, nodes, chunks = _build(LAW, 384)
    _invariants(dt, nodes, chunks, 384)
    ids = [c.chunk_id for c in chunks]
    assert f"{D}/ГЛ.1/СТ.1#1" in ids and f"{D}/ГЛ.1/СТ.2#1" in ids
    assert f"{D}/ГЛ.1/СТ.1#2" not in ids


def test_long_article_splits_on_element_boundaries():
    dt, nodes, chunks = _build(LAW, 12)
    _invariants(dt, nodes, chunks, 12)
    a3 = [c for c in chunks if c.chunk_id.startswith(f"{D}/ГЛ.2/СТ.3")]
    assert len(a3) >= 2
    assert [c.par for c in a3] == list(range(1, len(a3) + 1))


def test_every_word_of_every_unit_is_covered():
    dt, nodes, chunks = _build(LAW, 12)
    by_id = {n.node_id: n for n in nodes}
    for n in nodes:
        if n.type == "статья":
            inside = [c.text for c in chunks if n.start <= c.start and c.end <= n.end]
            assert "".join("".join(inside).split()) == "".join(dt.text[n.start:n.end].split())


def test_oversized_paragraph_without_punctuation_uses_word_windows():
    big = " ".join(f"сўз{i}" for i in range(70))
    paras = ["1-модда. Биринчи", big, "2-модда. Иккинчи", "матн.", "3-модда. Учинчи", "матн."]
    dt, nodes, chunks = _build(paras, 25)
    _invariants(dt, nodes, chunks, 25)
    assert len([c for c in chunks if c.chunk_id.startswith(f"{D}/СТ.1")]) >= 3


def test_mu_and_mu_inverse():
    dt, nodes, chunks = _build(LAW, 12)
    m, inv = mu(chunks), mu_inverse(nodes, chunks)
    assert all(m[c.chunk_id] == c.parent for c in chunks)
    assert all(inv[n.node_id] for n in nodes)                  # every vertex has μ⁻¹ ≠ ∅
    for c in chunks:
        assert c.chunk_id in inv[c.parent] and c.chunk_id in inv[D]


def test_chunking_is_deterministic():
    assert _build(LAW, 12)[2] == _build(LAW, 12)[2]
