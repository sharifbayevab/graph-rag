"""Proper-noun variants (orthography, script, Russian declension, uz<->ru gazetteer) and act titles cited in the
fragment text (copy / article exemptions)."""
from graphrag.dataset.generate import (article_without_act, exempt_words, foreign_names, max_copy_run, name_key,
                                       text_titles)

FR = [{"act": "Senatning 3-yalpi majlisi (2025-yil 23-, 24 – 25-yanvar)", "where": "x",
       "text": "Toshkent viloyati boʼyicha Obod qishloq dasturiga 17,4 mlrd soʼm ajratildi.",
       "speaker": "Anvar Mirzaaxmedovich Toʻychiyev"}]


def test_surname_spellings_and_declension_match():
    assert len({name_key(w) for w in ("Туйчиев", "Тўйчиев", "Tuychiyev", "Toʻychiyev", "Туйчиева", "Туйчиеву")}) == 1
    assert name_key("Гадоев") == name_key("Gadoyev")
    assert foreign_names("Сколько выделено, по словам сенатора Туйчиева?", FR, "ru") == []
    assert foreign_names("2025 йил 23–25 январдаги мажлисда сенатор Тўйчиев нима деди?", FR, "uz-Cyrl") == []


def test_russian_region_names_match_the_uzbek_form_by_gazetteer():
    assert foreign_names("Сколько выделено по Ташкентской области, по словам Туйчиева?", FR, "ru") == []


def test_invented_names_still_fail():
    assert foreign_names("Сколько выделено, по словам сенатора Каримова?", FR, "ru") == ["Каримова"]
    assert foreign_names("2025-yil 23–25-yanvardagi majlisda senator Karimov nima dedi?", FR, "uz-Latn") == ["Karimov"]
    assert foreign_names("Сколько выделено по Сурхандарьинской области и по словам Каримова?", FR, "ru") != []


def test_act_titles_cited_in_the_fragment_text_are_exempt():
    title = "Oʻzbekiston Respublikasining Maʼmuriy javobgarlik toʻgʻrisidagi kodeksiga oʻzgartish kiritish toʻgʻrisida"
    fr = [{"act": "Ayrim qonun hujjatlariga oʻzgartirish va qoʻshimchalar kiritish toʻgʻrisida", "where": "x",
           "text": f"1-modda. «{title}»gi Qonunning 5-moddasi quyidagi tahrirda bayon etilsin: jarima miqdori oshiriladi."}]
    assert text_titles(fr[0]["text"]) == [title]
    q = f"«{title}»gi Qonunning 5-moddasiga koʻra jarima qanday oʻzgaradi?"
    assert max_copy_run(q, fr[0]["text"], exempt_words(fr)) <= 7
    assert not article_without_act(q, fr)
    bare = [dict(fr[0], text="1-modda. 5-modda quyidagi tahrirda bayon etilsin: jarima miqdori oshiriladi.")]
    assert max_copy_run(q, bare[0]["text"], exempt_words(bare)) <= 7          # nothing to copy there either
    assert article_without_act("5-moddaga koʻra jarima qanday oʻzgaradi?", bare)


def test_superscript_digits_never_crash_the_tokenizer():
    from graphrag.dataset.facts import ctoks
    assert [t.s for t in ctoks("¹")] == [] and [t.s for t in ctoks("166¹-моддаси")][:1] == ["166.1"]
    assert name_key("¹") == "" or isinstance(name_key("¹"), str)


def test_uzbek_case_suffixes_and_uz_renderings_of_russian_titles():
    from graphrag.dataset.generate import wrong_act
    assert len({name_key(w) for w in ("Ҳамраевнинг", "Hamrayev", "Ҳамраевга", "Hamrayevning")}) == 1
    sp = [{"act": "Senatning 29-yalpi majlisi", "where": "x", "text": "Kredit platformasi orqali 2022 yilda kredit berildi.",
           "speaker": "Behzod Saylixonovich Hamrayev"}]
    assert foreign_names("2022 йил 4–5 августдаги ялпи мажлисда Беҳзод Сайлихонович Ҳамраевнинг айтишича қанча кредит?", sp,
                         "uz-Cyrl") == []
    ru = [{"act": "О санации и ликвидации банков", "where": "x", "text": "Статья 27. Промежуточный банк создается для передачи."}]
    q = "«Банкларни санация қилиш ва тугатиш тўғрисида»ги Қонуннинг 27-моддасига кўра оралиқ банк нима учун тузилади?"
    assert foreign_names(q, ru, "uz-Cyrl") == [] and not wrong_act(q, ru, "uz-Cyrl")
