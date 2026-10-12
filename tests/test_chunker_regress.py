# Regression tests for chunker defects found beyond the brief fixtures.
from graphrag.data.chunker import chunk_document
from graphrag.data.extract import Para
from graphrag.data.structure import parse_structure
from tests.helpers import words

D = "QONUNLAR-0000abcd"


def test_fragments_of_two_split_points_are_never_packed_together():
    # Each point is over max_tokens and gets split; the brief's plain greedy pack joined the
    # tail of П.1 with the head of П.2 into one chunk, cutting both point boundaries.
    paras = ["1-модда. Биринчи", "Қуйидагилар:",
             "1) биринчи банд боши;", "биринчи банд давоми узун матн билан.",
             "2) иккинчи банд боши;", "иккинчи банд давоми узун матн билан.",
             "2-модда. Иккинчи матн.", "3-модда. Учинчи матн."]
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in paras], "t", 8, words)
    chunks = chunk_document(dt, nodes, 8, words, str.lower)
    pts = [n for n in nodes if n.type == "пункт"]
    assert len(pts) == 2
    for n in pts:
        for c in chunks:
            ok = (n.start <= c.start and c.end <= n.end) or (c.start <= n.start and n.end <= c.end) \
                 or c.end <= n.start or c.start >= n.end
            assert ok, (n.node_id, c.text)


# ── fix round 1 ──────────────────────────────────────────────────────────────
from graphrag.data.mapping import mu_inverse  # noqa: E402
from tests.test_structure_law import LAW  # noqa: E402

S = "STENOGRAMMA-0000abcd"
SPEECH = "Erkin Fayziyevich Gadoyev: " + " ".join(f"soʼz{i}" for i in range(30)) + "."
STENO = ["Oʼzbekiston Respublikasi Oliy Majlisi Senatining yalpi majlisi",
         "Raislik qiluvchi: Assalomu alaykum. Majlisni ochiq deb eʼlon qilaman.",
         "Raislik qiluvchi: Kun tartibidagi birinchi masalani koʼrib chiqishga oʼtamiz.",
         "Raislik qiluvchi: Marhamat.",
         SPEECH,
         "Raislik qiluvchi: Rahmat."]


def _steno(max_tokens, attach):
    dt, nodes = parse_structure(S, "STENOGRAMMA", "stenogram", [Para(t) for t in STENO], "Majlis",
                                max_tokens, words)
    return dt, nodes, chunk_document(dt, nodes, max_tokens, words, str.lower, attach_min_tokens=attach)


def _check(dt, nodes, chunks, max_tokens):
    by = {n.node_id: n for n in nodes}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    prev = -1
    for c in sorted(chunks, key=lambda c: c.start):
        assert c.text == dt.text[c.start:c.end] and c.start >= prev and c.tokens <= max_tokens
        prev = c.end
        assert c.chunk_id == f"{c.parent}#{c.par}"
        p = by[c.parent]
        assert p.start <= c.start and c.end <= p.end
        assert not any(k.parent_id == p.node_id and k.start <= c.start and c.end <= k.end for k in nodes)
    inv = mu_inverse(nodes, chunks)
    assert all(inv[n.node_id] for n in nodes)
    covered = "".join("".join(c.text for c in sorted(chunks, key=lambda c: c.start)).split())
    assert covered == "".join(dt.text.split())


def test_tiny_turns_attach_to_following_else_previous_turn_in_same_question():
    dt, nodes, chunks = _steno(40, 10)
    _check(dt, nodes, chunks, 40)
    q2 = [c for c in chunks if c.parent == f"{S}/ВОПР.2"]
    assert [c.chunk_id for c in q2] == [f"{S}/ВОПР.2#1", f"{S}/ВОПР.2#2"]
    assert q2[0].text.startswith("Raislik qiluvchi: Kun") and q2[0].text.endswith("Marhamat.")
    assert q2[1].text.startswith("Erkin") and q2[1].text.endswith("Rahmat.")   # tail → previous
    # different вопрос never merge: ВОПР.1's single tiny turn stays alone
    assert [c.parent for c in chunks if c.start < q2[0].start][-1] == f"{S}/ВОПР.1/ВЫСТ.1"
    assert _steno(40, 10)[2] == chunks                                               # deterministic


def test_attach_disabled_and_split_speech_fragments_do_not_attach():
    dt, nodes, chunks = _steno(40, 0)
    assert not any(c.parent.split("/")[-1].startswith("ВОПР") for c in chunks)   # nothing merged
    dt, nodes, chunks = _steno(20, 10)              # speech (32 words) is split into fragments
    _check(dt, nodes, chunks, 20)
    sp = next(n for n in nodes if n.title == "Erkin Fayziyevich Gadoyev")
    frags = [c for c in chunks if sp.start <= c.start and c.end <= sp.end]
    assert len(frags) >= 2 and all(c.parent == sp.node_id for c in frags)
    assert not any(c.start < sp.end < c.end or c.start < sp.start < c.end for c in chunks)


def test_index_text_carries_breadcrumb_and_text_is_unchanged():
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in LAW], "Санация тўғрисида",
                                384, words)
    chunks = chunk_document(dt, nodes, 384, words, str.lower)
    c = next(c for c in chunks if c.chunk_id == f"{D}/ГЛ.1/СТ.2#1")
    assert c.text == dt.text[c.start:c.end] and c.text.startswith("2-модда.")
    crumb, body = c.index_text.split("\n", 1)
    assert crumb == "санация тўғрисида › 1-боб. умумий қоидалар › 2-модда. асосий тушунчалар"
    assert body == c.text_norm
    c3 = next(c for c in chunks if c.chunk_id.startswith(f"{D}/ГЛ.2/СТ.3"))
    assert c3.index_text.split("\n", 1)[0].endswith("2-боб. санация тартиби › 3-модда.")
    dt, nodes, sc = _steno(40, 10)
    assert sc[-1].index_text.split("\n", 1)[0] == "majlis › erkin fayziyevich gadoyev, raislik qiluvchi"


def test_breadcrumb_is_capped_from_the_document_side():
    long_title = " ".join(f"ном{i}" for i in range(200))
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in LAW], long_title, 384, words)
    chunks = chunk_document(dt, nodes, 384, words, str.lower)
    for c in chunks:
        crumb = c.index_text.split("\n", 1)[0]
        assert words(crumb) <= 96
    c = next(c for c in chunks if c.chunk_id == f"{D}/ГЛ.1/СТ.2#1")
    assert c.index_text.startswith("1-боб. умумий қоидалар › 2-модда.")


def test_run_without_spaces_longer_than_max_is_split_by_characters():
    q = lambda s: len(s) // 4                       # noqa: E731
    paras = ["1-модда. Биринчи", "ж" * 400, "2-модда. Иккинчи", "матн.", "3-модда. Учинчи", "матн."]
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in paras], "t", 25, q)
    chunks = chunk_document(dt, nodes, 25, q, str.lower)
    assert all(c.tokens <= 25 for c in chunks)
    a1 = next(n for n in nodes if n.type == "статья" and n.number == "1")
    inside = [c.text for c in chunks if a1.start <= c.start and c.end <= a1.end]
    assert len(inside) >= 4 and "".join("".join(inside).split()) == "".join(dt.text[a1.start:a1.end].split())


# ── fix round 2 ──────────────────────────────────────────────────────────────
def test_text_norm_also_stays_within_max_tokens():
    # to_norm may inflate tokens (e.g. transliteration); packing must bound both forms.
    inflate = lambda s: s.lower().replace(" ", " x ")      # noqa: E731  ~2× words
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in LAW], "t", 12, words)
    chunks = chunk_document(dt, nodes, 12, words, inflate)
    assert chunks and all(words(c.text) <= 12 and words(c.text_norm) <= 12 for c in chunks)
    for n in nodes:
        if n.type == "статья":
            inside = [c.text for c in chunks if n.start <= c.start and c.end <= n.end]
            assert "".join("".join(inside).split()) == "".join(dt.text[n.start:n.end].split())


def test_passage_plus_index_text_fits_encoder_limit():
    # With 500-token chunks the 96-token breadcrumb cap alone would overflow 512.
    body = " ".join(f"сўз{i}" for i in range(495)) + "."
    paras = ["1-боб. " + " ".join(f"бобном{i}" for i in range(60)),
             "1-модда. Биринчи", body, "2-модда. Иккинчи матн.", "3-модда. Учинчи матн."]
    title = " ".join(f"ном{i}" for i in range(200))
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", [Para(t) for t in paras], title, 500, words)
    chunks = chunk_document(dt, nodes, 500, words, str.lower)
    for c in chunks:
        assert words("passage: " + c.index_text) + 2 <= 512, c.chunk_id
        assert c.index_text.endswith(c.text_norm)
    big = next(c for c in chunks if c.tokens > 400)
    assert big.index_text != big.text_norm                    # a (truncated) crumb survives
    small = next(c for c in chunks if "/СТ.2" in c.chunk_id)
    crumb = small.index_text.split("\n", 1)[0]               # short chunk: 96 cap, title dropped
    assert words(crumb) <= 96 and crumb.startswith("1-боб.") and crumb.endswith("› 2-модда.")
