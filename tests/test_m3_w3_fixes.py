"""Wave-3 code-check false positives: meta numbers, translated titles, one-word titles, quoted fund names, spelling
variants of values / names, abbreviations, Roman numerals, sitting day ranges, body names as values."""
from graphrag.dataset.facts import check_fact
from graphrag.dataset.generate import (amending_locus, article_without_act, foreign_names, lang_ok, loose_value_in,
                                       session_undated, wrong_act)


def test_meta_numbers_titles_and_one_word_acts_name_the_article():
    uz = [{"act": "Давлат мудофаа буюртмаси тўғрисида", "where": "x", "text": "15-модда. Ижрочи давлат сирини сақлайди."}]
    assert not article_without_act("Какое условие ставит статья 15 закона «О государственном оборонном заказе»?", uz)
    one = [{"act": "Pirotexnika buyumlari toʻgʻrisida", "where": "x", "text": "t", "amended_act": "Qurol toʼgʼrisida"}]
    assert not article_without_act("«Qurol toʻgʻrisida»gi Qonunning 17-moddasi qanday oʻzgardi?", one)
    assert article_without_act("17-moddasi qanday oʻzgardi?", one)


def test_quoted_fund_names_and_wh_words_are_no_act_titles():
    sp = [{"act": "Senatning 50-yalpi majlisi", "where": "x", "text": "Yoshlar daftari jamgʻarmasi mablagʻlari.",
           "discussed_act": "Andijon viloyati yoʻl xaritalari ijrosi toʻgʻrisida"}]
    assert not wrong_act("Ялпи мажлисда «Ёшлар дафтари» жамғармасидан қанча маблағ сарфланди?", sp, "uz-Cyrl")
    assert not wrong_act("Қонун қайси кодексларга ўзгартириш киритади?", sp, "uz-Cyrl")
    assert wrong_act("«Давлат божи тўғрисида»ги Қонунга кўра ким?", sp, "uz-Cyrl")
    rad = [{"act": "Senatning 36-yalpi majlisi", "where": "x", "text": "Qonunni rad etish haqidagi qaror.",
            "discussed_act": "Huquqbuzarliklar profilaktikasi toʼgʼrisida"}]
    assert not wrong_act("Qonunni rad etish haqidagi qaror loyihasiga necha kishi qarshi ovoz berdi?", rad, "uz-Latn")


def test_spelling_variants_of_values_and_names():
    assert loose_value_in("Korrupsiyaga qarshi kurashish agentligiga", "Коррупцияга қарши курашиш агентлигига юклатилмоқда")
    assert loose_value_in("mahalliy budjet hisobidan", "маҳаллий бюджет ҳисобидан молиялаштирилади")
    assert not loose_value_in("Davlat soliq qoʻmitasi", "Марказий банк")
    fr = [{"act": "Senatning 46-yalpi majlisi", "where": "x", "text": "Probatsiya xizmati prokurorni xabardor qiladi.",
           "discussed_act": "Probatsiya toʼgʼrisida", "speaker": "Qutbidin Nukritdinovich Burxonov"}]
    assert foreign_names("Что, по словам Кутбидина Нукритдиновича Бурханова, предусматривает закон «О пробации»?", fr, "ru") == []
    assert foreign_names("Что, по словам Каримова, предусматривает закон «О пробации»?", fr, "ru") == ["Каримова"]


def test_abbreviations_roman_numerals_and_sentence_initial_words():
    fr = [{"act": "Senatning 2-yalpi majlisi", "where": "x", "text": "Qoʻshilgan qiymat soligʻidan ozod qilinadi. BMT tavsiyalari."}]
    assert foreign_names("Мажлисда ҚҚС бўйича қандай имтиёз айтилди?", fr, "uz-Cyrl") == []
    assert foreign_names("Mirzatillo BMT qoʻmitalari haqida nima dedi?", fr, "uz-Latn") == ["Mirzatillo"]
    hul = [{"act": "Dori vositalari toʻgʻrisida", "where": "x", "text": "Xulosada Rossiya, Qozogʻiston tajribasi oʻrganildi."}]
    assert foreign_names("Хулосада Россия тажрибаси ўрганилгани қайд этилган.", hul, "uz-Cyrl") == []
    assert lang_ok("II–V хавфлилик синфидаги буюмларни ким белгилайди?", "uz-Cyrl")
    assert not lang_ok("II–V xavflilik синфидаги буюмларни ким белгилайди?", "uz-Cyrl")


def test_a_sitting_on_non_consecutive_days_written_as_a_range_and_body_names_as_values():
    s = [{"session": {"date": "2022-03-17", "title": "Senatning 24-yalpi majlisi (2022-yil 17-, 19-mart)"}}]
    assert not session_undated("2022-yil 17–19-martdagi Senat yalpi majlisida kim gapirdi?", s)
    assert session_undated("2022-yil 17–20-martdagi Senat yalpi majlisida kim gapirdi?", s)
    act = {"Ижтимоий", "шериклик", "тўғрисида"}
    assert check_fact({"value": "Олий Мажлиси палаталари", "quote": "тартиби Олий Мажлиси палаталари томонидан белгиланади"},
                      "Субсидиялар тартиби Олий Мажлиси палаталари томонидан белгиланади.", act) is None
    assert check_fact({"value": "3 yil", "quote": "ushbu holatlar 3 yil davomida boʼlmasa"},
                      "Agar ushbu holatlar 3 yil davomida boʼlmasa, tugatiladi.", {"2024-yil", "yil"}) is None


def test_amending_item_named_in_translation_through_meta():
    fr = [{"act": "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча киритиш тўғрисида", "where": "x",
           "text": "Давлат субсидияларини ажратиш тартиби белгиланади.", "amended_act": "Ижтимоий шериклик тўғрисида",
           "amended_unit": "ст. 19", "amended": []}]
    assert not amending_locus("Чьей компетенцией по статье 19 Закона «О социальном партнёрстве» являются субсидии?", fr)
    assert amending_locus("Чьей компетенцией по статье 19 являются субсидии?", fr)
