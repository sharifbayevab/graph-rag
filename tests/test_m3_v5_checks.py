"""Instructions v5 code checks: Russian purity, the amended provision of amending laws, sitting dates within the
session (Uzbek textual title ranges parsed)."""
from graphrag import config
from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import amending_locus, check_item_ex, ru_impure, session_iso, session_undated
from graphrag.dataset.seeds import Seed


def test_russian_texts_carry_no_pasted_uzbek():
    assert ru_impure("Не завершены по 23 проектам («23 та лойиҳа»).", True)              # figure gloss
    assert ru_impure("Комиссия признала «maqsadga muvofiq emas deb topdi va rad etdi shu bilan birga» это.", True)
    assert ru_impure("Разрешение выдаёт вилоят ҳокимлиги.", True)                       # Uzbek word outside «»
    assert ru_impure("Фонды «Obod qishloq» и «Obod mahalla»?", False)                    # two Uzbek spans
    assert not ru_impure("Лицензию выдаёт Центральный банк («Марказий банк»).", True)
    assert not ru_impure("Согласно Закону «Банклар ва банк фаолияти тўғрисида» кто выдаёт лицензию?", False)
    assert not ru_impure("Не завершены по 23 проектам первого сезона.", True)


def test_amending_law_article_needs_the_amended_act_named_in_the_text():
    fr = [{"act": "Айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида", "text": "«Товар белгиси эгаси …»",
           "amended": ["Товар белгилари тўғрисида"]}]
    assert amending_locus("Қонуннинг 2-моддаси билан киритилаётган нормага кўра гувоҳнома нима бўлади?", fr)
    assert not amending_locus("Товар белгиси эгаси инсофсиз рақобат қилса гувоҳнома нима бўлади?", fr)   # no article
    named = [dict(fr[0], text="«Товар белгилари тўғрисида»ги Қонуннинг 30-моддаси ўзгартирилсин")]
    assert not amending_locus("«Товар белгилари тўғрисида»ги Қонуннинг 30-моддасига кўра гувоҳнома нима бўлади?", named)
    assert not amending_locus("Божхона кодексининг 347-моддасига кўра муддат қанча?", fr)                 # a code
    own = [{"act": "Банклар ва банк фаолияти тўғрисида", "text": "t", "amended": []}]
    assert not amending_locus("Банклар тўғрисидаги Қонуннинг 5-моддасига кўра ким?", own)                # not amending


def test_uzbek_title_ranges_and_days_outside_the_sitting():
    assert session_iso({"date": "2024-12-18", "title": "Senatning 2-yalpi majlisi (2024-yil 18-19- va 21-dekabr)"}) == \
        ["2024-12-18", "2024-12-19", "2024-12-21"]
    assert session_iso({"date": "2025-01-23", "title": "Senatning 3-yalpi majlisi (2025-yil 23-, 24 – 25-yanvar)"}) == \
        ["2025-01-23", "2025-01-24", "2025-01-25"]
    assert session_iso({"date": "2022-03-17", "title": "Senatning 24-yalpi majlisi (2022-yil 17-, 19-mart)"}) == \
        ["2022-03-17", "2022-03-19"]
    s = [{"session": {"date": "2024-09-20", "title": "Senatning 57-yalpi majlisi (2024-yil 20-21-sentabr)"}}]
    assert not session_undated("Что сказал сенатор на заседании 20–21 сентября 2024 года?", s)
    assert session_undated("Что сказал сенатор на заседании 20–22 сентября 2024 года?", s)       # 22 outside
    assert session_undated("Что сказал сенатор на 57-м пленарном заседании Сената?", s)            # number alone


def test_the_item_check_reports_the_new_reasons():
    s = Seed("single-r", "single", "ru", ("D/П.1#1",), ("D",), "train", "QARORLAR")
    f = Fact("D/П.1#1@0", "D/П.1#1", "", "23 та лойиҳа", "23 та лойиҳа ижроси доирасидаги")
    fr = [{"id": "F1", "act": "Андижон вилояти йўл хариталари ижроси тўғрисида", "where": "x",
           "text": "1-мавсумдаги 23 та лойиҳа ижроси доирасидаги таъмирлаш ишлари якунланмаган."}]
    cfg = {"max_copy_words": 99, "max_lexical_overlap": 1.0}
    item = lambda a: {"question": "По скольким проектам первого сезона ремонт до сих пор не завершён?", "answer": a,
                      "used": ["F1.0"]}
    assert check_item_ex(item("По 23 проектам («23 та лойиҳа»)."), s, {"F1.0": f}, fr, cfg)[0] == "ru_impure"


def test_instructions_v5_rules_pinned():
    o = config.load("train")["dataset"]["opus"]
    t = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["generator"]["version"] >= 5 and o["generator_history"][o["generator"]["version"]] == o["generator"]["digest"]
    for x in ("ENTIRELY Russian", "name the AMENDED provision", "one of `session.dates`",
              "must name the LAW", "Three more reader-flagged defects"):
        assert x in t, x
