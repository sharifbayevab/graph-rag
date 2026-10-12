"""Query/index normalization parity (final review I1) and the --reset endpoint guard (I2)."""
from types import SimpleNamespace

import numpy as np
import pytest

from graphrag.data.lemma import tokens
from graphrag.data.normalize import clean_encoding, query_internal, to_internal
from graphrag.retrieval import lexical, vector
from graphrag.stores.guard import reset_problems

VARIANTS = ["Oʻzbekiston", "O‘zbekiston", "O'zbekiston", "Oʼzbekiston"]  # U+02BB, U+2018, ASCII, U+02BC


def _indexed_terms(text: str, lang: str = "uz") -> list[str]:
    return sorted(set(tokens(to_internal(clean_encoding(text), lang))))   # the indexing pipeline


@pytest.mark.parametrize("q", VARIANTS)
def test_apostrophe_variants_match_indexed_form(q):
    want = _indexed_terms("Oʼzbekiston Respublikasi")
    assert lexical.query_terms(q + " Respublikasi") == want
    assert lexical.query_terms(q + " Respublikasi", "uz") == want


def test_all_apostrophe_variants_agree():
    assert len({tuple(lexical.query_terms(q)) for q in VARIANTS}) == 1
    assert len({query_internal(q, "uz") for q in VARIANTS}) == 1


def test_non_breaking_hyphen_article_number():
    assert "modda#21" in lexical.query_terms("21\u2011moddasi")
    assert lexical.query_terms("21\u2011moddasi") == lexical.query_terms("21-moddasi")


def test_query_internal_is_clean_encoding_then_to_internal():
    s = "  Oʻzbekiston\u00a0Respublikasi  21\u2011moddasi "
    assert query_internal(s, "uz") == to_internal(clean_encoding(s), "uz")


def test_vector_search_text_embeds_normalized_query():
    seen = {}

    class Enc:
        def encode_queries(self, texts):
            seen["texts"] = texts
            return np.ones((len(texts), 3), dtype=np.float32)

    class Cl:
        def query_points(self, name, query, limit, query_filter, with_payload):
            seen.update(name=name, limit=limit, flt=query_filter)
            return SimpleNamespace(points=[SimpleNamespace(payload={"chunk_id": "d/c1"}, score=0.5)])

    out = vector.search_text(Cl(), "coll", Enc(), "O‘zbekiston", k=3, exclude_doc="d0")
    assert out == [("d/c1", 0.5)]
    assert seen["texts"] == [query_internal("O‘zbekiston")] and seen["texts"][0] == "oʼzbekiston"
    assert seen["limit"] == 3 and seen["flt"] is not None


V2 = dict(pg_dsn="postgresql://graphrag:graphrag@localhost:5437/graphrag",
          qdrant_url="http://localhost:6343", neo4j_uri="bolt://127.0.0.1:7688")


def test_reset_guard_accepts_local_v2_endpoints():
    assert reset_problems(**V2, env={}) == []


@pytest.mark.parametrize("key,val", [
    ("pg_dsn", "postgresql://graphrag:graphrag@localhost:5436/graphrag"),   # prod pg port
    ("pg_dsn", "postgresql://graphrag:graphrag@10.0.0.5:5437/graphrag"),    # remote host
    ("pg_dsn", "dbname=graphrag host=localhost port=5437"),                 # unparseable → refuse
    ("qdrant_url", "http://localhost:6333"),                                # prod qdrant
    ("qdrant_url", "http://qdrant.example.com:6343"),
    ("neo4j_uri", "bolt://localhost:7687"),
    ("neo4j_uri", "neo4j://prod-host:7688"),
])
def test_reset_guard_refuses_non_v2_endpoints(key, val):
    probs = reset_problems(**{**V2, key: val}, env={})
    assert len(probs) == 1
    assert reset_problems(**{**V2, key: val}, env={"GRL_ALLOW_RESET": "1"}) == []
    assert reset_problems(**{**V2, key: val}, env={"GRL_ALLOW_RESET": "0"}) == probs


# --- M3 Task 0: Russian endings shared with Uzbek; nested guillemets ---------------------------------
from graphrag.data.normalize import _strip_quoted, segment_is_russian  # noqa: E402

RU_WITH_UZ_LOOKING_ENDINGS = ["Фонд создан для поддержки инвалида",
                              "пенсия инвалида назначается органом"]
NESTED_RU = ("Сенат одобрил закон «Ўзбекистон Республикасининг «Банклар тўғрисида»ги "
             "Қонунига ўзгартиш киритиш ҳақида»")


@pytest.mark.parametrize("q", RU_WITH_UZ_LOOKING_ENDINGS)
def test_russian_words_with_uzbek_looking_endings_stay_russian(q):
    """M2 Task 0 minor (R29 → M3): -ида/-ига/-дан alone are no Uzbek evidence ("инвалида", "создан")."""
    assert segment_is_russian(q, False)
    assert query_internal(q) == q.lower()             # Cyrillic kept: Russian lemmas, not transliteration


@pytest.mark.parametrize("q", ["Ушбу Қонуннинг 5-моддасига мувофиқ", "ушбу модда комиссиясида"])
def test_uzbek_segments_stay_uzbek(q):
    """The ambiguous endings still reinforce real Uzbek evidence (a function word, ўқғҳ)."""
    assert not segment_is_russian(q, False) and not segment_is_russian(q, True)


def test_nested_guillemets_are_quoted_as_a_whole():
    assert _strip_quoted("а «б «в» г» д").split() == ["а", "д"]
    assert _strip_quoted("а «б в").split() == ["а", "«б", "в"]        # unclosed: nothing is quoted
    assert _strip_quoted('x "y" „z“ w').split() == ["x", "w"]


def test_russian_sentence_quoting_a_nested_uzbek_title_stays_russian():
    assert segment_is_russian(NESTED_RU, False)
    assert query_internal(NESTED_RU).startswith("сенат одобрил закон «")
