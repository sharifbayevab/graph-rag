# advanced-rag-v2/tests/test_lemma.py
from graphrag.data.lemma import UzStemmer, build_vocab, rule_stem, ru_lemma, tokens

VOCAB = {"qonun": 5, "qonuni": 3, "modda": 9, "moddani": 2, "kodeks": 4,
         "vazirlik": 2, "sud": 3, "muvofiq": 4}


def test_lexicon_guided_stemming():
    st = UzStemmer(VOCAB)
    assert st.stem("qonuni") == "qonun"
    assert st.stem("qonunining") == "qonun"
    assert st.stem("qonunni") == "qonun"
    assert st.stem("moddasiga") == "modda"
    assert st.stem("moddani") == "modda"
    assert st.stem("kodeksiga") == "kodeks"
    assert st.stem("kodekslari") == "kodeks"
    assert st.stem("vazirligi") == "vazirlik"
    assert st.stem("sudga") == "sud"


def test_fallback_rule_stem_for_unknown_words():
    assert UzStemmer(VOCAB).stem("xyzlarga") == rule_stem("xyzlarga") == "xyz"


def test_ru_lemma():
    assert ru_lemma("статью") == "статья"
    assert ru_lemma("закона") == "закон"


def test_tokens_composites_dates_stopwords():
    st = UzStemmer(VOCAB)
    toks = tokens("2024-01-19 122.1-moddasiga va qonunga muvofiq", st)
    assert "date#2024-01-19" in toks
    assert "modda#122.1" in toks
    assert "modda" in toks and "qonun" in toks and "muvofiq" in toks
    assert "va" not in toks


def test_russian_units_map_to_language_neutral_tokens():
    toks = tokens("статью 21 закона", UzStemmer(VOCAB))
    assert "modda#21" in toks and "статья" in toks and "закон" in toks


def test_build_vocab_counts_latin_words_only():
    v = build_vocab(["qonun qonun modda", "закон 12"])
    assert v == {"modda": 1, "qonun": 2}


# ── fix round 1 ──────────────────────────────────────────────────────────────
def _units(text, lang="uz"):
    from graphrag.data.normalize import to_internal
    return [t for t in tokens(to_internal(text, lang), UzStemmer(VOCAB)) if "#" in t and not t.startswith("date#")]


def test_gazette_citations_are_not_articles():
    toks = tokens("1995 yil, № 1, 3-modda", UzStemmer(VOCAB))
    assert "gaz#3" in toks and "modda#3" not in toks
    assert "modda#3" in tokens("ushbu qonunning 3-moddasi", UzStemmer(VOCAB))
    assert _units("2008 yil, № 4, 187, 188-moddalar, № 11–12, 5-modda") == ["gaz#187", "gaz#188", "gaz#5"]
    assert _units("2019 г., № 11, ст. 790", "ru") == ["gaz#790"]


def test_russian_units_full_forms_and_abbreviations():
    assert _units("ч. 2 ст. 5", "ru") == ["qism#2", "modda#5"]
    assert _units("пункта 3 статьи 5", "ru") == ["band#3", "modda#5"]
    assert _units("п. 4 главы 2", "ru") == ["band#4", "bob#2"]
    assert _units("в статье 5.1", "ru") == ["modda#5.1"]
    assert _units("статьями 5 и 6", "ru") == ["modda#5", "modda#6"]


def test_russian_unit_false_hits():
    for s in ("достаточно 5 голосов", "участника 3", "статус 2", "главный 3"):
        assert _units(s, "ru") == [], s


def test_uzbek_unit_ranges_and_lists():
    assert _units("10-12-moddalar") == ["modda#10", "modda#11", "modda#12"]
    assert _units("10, 11 va 12-moddalari") == ["modda#10", "modda#11", "modda#12"]
    assert _units("43.1 hamda 43.2-moddalariga") == ["modda#43.1", "modda#43.2"]
    assert _units("70-701-moddasiga") == ["modda#70", "modda#701"]
    assert _units("5-moddasining 2-qismi") == ["modda#5", "qism#2"]


def test_suffix_slots_ka_qa_after_k_q_only():
    st = UzStemmer({**VOCAB, "boshqa": 9, "texnika": 5, "sutka": 4, "bosh": 9, "tex": 3, "sut": 3})
    assert st.stem("boshqa") == "boshqa"
    assert st.stem("texnika") == "texnika"
    assert st.stem("sutka") == "sutka"
    assert st.stem("sutkagacha") == "sutka"


def test_rule_stem_si_not_after_consonant_apostrophe():
    assert rule_stem("togʼsi") != "togʼ"  # -si not stripped after gʼ
    assert rule_stem("moddasi") == "modda"
