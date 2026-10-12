from graphrag.data.extract import Para
from graphrag.data.normalize import (canon_num, canon_numbering, clean_encoding, drop_noise,
                                     iso_dates, normalize_paragraphs, to_internal)


def test_clean_encoding():
    assert clean_encoding("1-модда.\xa0Асосий  тушунчалар\x00") == "1-модда. Асосий тушунчалар"
    assert clean_encoding("санация\xadлаш") == "санациялаш"
    assert clean_encoding("yo‘q\tva  yoʻq") == "yoʼq va yoʼq"


def test_iso_dates():
    assert iso_dates("2024 йил 19 январь куни") == "2024-01-19 куни"
    assert iso_dates("2025-yil 22-iyulda qabul") == "2025-07-22 qabul"
    assert iso_dates("19.01.2024 й.") == "2024-01-19 й."
    assert iso_dates("14 мая 2019 года") == "2019-05-14 года"
    assert iso_dates("2024 йил 5 та ҳужжат") == "2024 йил 5 та ҳужжат"   # not a month
    assert iso_dates("30 машин 2020") == "30 машин 2020"               # 'ма' alone is not May
    assert iso_dates("3 март 2020") == "2020-03-03"


def test_canon_num():
    assert canon_num("122¹") == "122.1"
    assert canon_num("IV") == "4"
    assert canon_num("а)") == "a"
    assert canon_num("(3)") == "3"
    assert canon_num("21") == "21"


def test_canon_numbering():
    assert canon_numbering("122¹-модда ва II боб, 5 – модда") == "122.1-модда ва 2-боб, 5-модда"
    assert canon_numbering("II бобига") == "2-бобига"
    assert canon_numbering("I bobning") == "1-bobning"


def test_to_internal_uzbek_cyrillic():
    assert to_internal("2024 йил 19 январда 122¹-моддаси", "uz-Cyrl") == "2024-01-19 122.1-moddasi"


def test_to_internal_russian_keeps_cyrillic():
    assert to_internal("Статья 21. Закон", "ru") == "статья 21. закон"


def test_ascii_apostrophe_internal_only():
    assert to_internal("o'zbekiston", "uz-Latn") == "oʼzbekiston"
    assert to_internal("Закон 'О труде'", "ru") == "закон 'о труде'"
    assert clean_encoding("the law's 'text'") == "the law's 'text'"


def test_drop_noise_and_normalize_paragraphs():
    paras = [Para("2"), Para("  "), Para("Матн\xa0бор"), Para("- 3 -")]
    assert drop_noise([Para("2"), Para("Матн")]) == [Para("Матн")]
    assert normalize_paragraphs(paras) == [Para("Матн бор")]


def test_word_field_code_residue_is_dropped_or_stripped():
    # 13 corpus docx carry literal "PAGE   \* MERGEFORMAT13" text (a Word field code) in w:t.
    from graphrag.data.extract import Para as _P
    from graphrag.data.normalize import normalize_paragraphs as _norm
    out = _norm([_P("Биринчи."), _P("PAGE   \\* MERGEFORMAT13"), _P("page \\* mergeformat"),
                 _P("Матн давоми PAGE \\* MERGEFORMAT2 охири.")])
    assert [p.text for p in out] == ["Биринчи.", "Матн давоми охири."]


def test_bare_word_field_paragraphs_are_dropped():
    # 7 QARORLAR chunks were a bare "PAGE" (field residue without "\* MERGEFORMAT").
    out = drop_noise([Para("PAGE"), Para("NUMPAGES 3"), Para("SECTIONPAGES \\* MERGEFORMAT2"),
                      Para("PAGE 12"), Para("PAGE маълумот"), Para("Матн")])
    assert [p.text for p in out] == ["PAGE маълумот", "Матн"]


def test_to_internal_routes_script_per_segment():
    # Russian passages inside Uzbek documents keep Cyrillic (Russian lemmatization, no token
    # inflation from transliteration); Uzbek Cyrillic is still transliterated.
    assert to_internal("Hurmatli senatorlar. Уважаемые коллеги, прошу голосовать.", "uz-Latn") == \
        "hurmatli senatorlar. уважаемые коллеги, прошу голосовать."
    assert to_internal("Ушбу Қонун расмий эълон қилинган кундан эътиборан кучга киради.", "uz-Latn") == \
        "ushbu qonun rasmiy eʼlon qilingan kundan eʼtiboran kuchga kiradi."
    assert to_internal("Қонун қабул қилинди.\n\nЗакон вступает в силу.", "uz-Cyrl") == \
        "qonun qabul qilindi.\n\nзакон вступает в силу."
    assert to_internal("Ассалому алайкум", "uz-Cyrl") == "assalomu alaykum"       # ambiguous → lang


def test_to_internal_initials_follow_document_language():
    assert to_internal("С. Иминов", "uz-Cyrl") == "s. iminov"


def test_to_internal_single_letters_are_no_russian_evidence():
    assert to_internal("Х У Л О С А", "uz-Latn") == "x u l o s a"
    assert to_internal("QL_95/Ж-М/С-Н", "uz-Latn") == "ql_95/j-m/s-n"
    assert to_internal("Закон вступает в силу со дня опубликования.", "uz-Cyrl") == \
        "закон вступает в силу со дня опубликования."
