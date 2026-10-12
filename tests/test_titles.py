"""Task 11b: document titles joined over header paragraphs, Russian laws, inquiries, stenograms."""
from graphrag.data.titles import fix_title, join_header_title, russian_law_title, session_title


def test_uzbek_title_is_joined_until_the_terminator():
    P = ["Лойиҳа", "Ўзбекистон Республикасининг", "ҚОНУНИ",
         "Янги таҳрирдаги Ўзбекистон Республикаси Конституцияси",
         "қабул қилинганлиги муносабати билан Ўзбекистон Республикасининг айрим қонун ҳужжатларига",
         "ўзгартириш ва қўшимчалар киритиш тўғрисида",
         "Қонунчилик палатаси томонидан 2023 йил 14 ноябрда қабул қилинган"]
    assert join_header_title("Янги таҳрирдаги Ўзбекистон Республикаси Конституцияси", P) == (
        "Янги таҳрирдаги Ўзбекистон Республикаси Конституцияси қабул қилинганлиги муносабати билан "
        "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида")


def test_terminator_is_a_whole_word_and_label_lines_are_not_pulled_in():
    P = ["Ўзбекистон Республикасининг Маъмурий жавобгарлик", "тўғрисидаги кодексига ўзгартириш",
         "ва қўшимчалар киритиш ҳақида", "Қонунчилик палатаси томонидан 2023 йил 12 декабрда"]
    # the legacy title stopped inside "тўғрисидаги"
    assert join_header_title("Ўзбекистон Республикасининг Маъмурий жавобгарлик тўғрисида", P) == (
        "Ўзбекистон Республикасининг Маъмурий жавобгарлик тўғрисидаги кодексига ўзгартириш "
        "ва қўшимчалар киритиш ҳақида")
    Q = ["Ўзбекистон Республикасининг", "ҚОНУНИ", "Айрим қонун ҳужжатларига ўзгартириш киритиш тўғрисида"]
    t = "Айрим қонун ҳужжатларига ўзгартириш киритиш тўғрисида"
    assert join_header_title(t, Q) == t                  # complete already; "…ҚОНУНИ" never prepended
    assert join_header_title("Бошқа сарлавҳа", Q) == "Бошқа сарлавҳа"   # start paragraph not found
    R = ["«Давлат божи тўғрисида", "»ги Ўзбекистон Республикаси Қонунига қўшимча киритиш ҳақида"]
    assert join_header_title("«Давлат божи тўғрисида", R) == (
        "«Давлат божи тўғрисида »ги Ўзбекистон Республикаси Қонунига қўшимча киритиш ҳақида")


def test_russian_law_title_and_inquiry_title():
    P = ["Проект", "ЗАКОН", "Республики Узбекистан", "О санации и ликвидации банков",
         "Принят Законодательной палатой 13 февраля 2025 года", "Глава 1. Общие положения"]
    assert russian_law_title(P) == "О санации и ликвидации банков"
    assert fix_title(None, P, "QONUNLAR", "docs/QONUNLAR/1.4. Текст Закона.docx") == "О санации и ликвидации банков"
    Q = ["Лойиҳа", "Вазирлар Маҳкамасига", "П А Р Л А М Е Н Т С Ў Р О В И", "(яйловлар ҳолати юзасидан)", "Матн."]
    assert fix_title(None, Q, "BOSHQA", "docs/BOSHQA/x.docx") == "Парламент сўрови (яйловлар ҳолати юзасидан)"
    assert fix_title(None, ["Матн."], "BOSHQA", "docs/BOSHQA/x.docx") is None


def test_stenogram_session_titles():
    assert session_title("docs/STENOGRAMMA/Oliy Majlis Senati  7-yalpi majlis 28-29.05.25.docx", []) == \
        "Senatning 7-yalpi majlisi 28-29.05.25"
    head = ["Oʼzbekiston Respublikasi Oliy Majlisi Senatining qirqinchi yalpi majlisi (2023-yil 3-4-may)"]
    assert session_title("docs/STENOGRAMMA/Стен. 40-ялпи мажлис.docx", head) == \
        "Senatning 40-yalpi majlisi (2023-yil 3-4-may)"
    assert fix_title("Inson qadri uchun", head, "STENOGRAMMA", "docs/STENOGRAMMA/Стен.отчет_23-ялпи мажлис++.docx") == \
        "Senatning 23-yalpi majlisi (2023-yil 3-4-may)"
    assert fix_title("t", [], "STENOGRAMMA", "docs/STENOGRAMMA/s.md") == "t"      # no session number: unchanged
