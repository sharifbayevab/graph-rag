# Regression tests for defects found by running parse_law over the real QONUNLAR corpus.
from graphrag.data.extract import Para
from graphrag.data.structure import DocText, parse_law

D = "QONUNLAR-0000abcd"


def _ns(paras):
    dt = DocText.build([Para(t) for t in paras])
    nodes = parse_law(D, dt, "")
    return dt, {n.node_id: n for n in nodes}


def _arts(ns):
    return sorted(n.number for n in ns.values() if n.type == "статья")


def test_long_closed_quote_is_not_reset_after_40_paragraphs():
    # 3.3. Қонун матни(2).docx: an inserted chapter «43-боб. ... »; spans 46 paragraphs and
    # contains 372⁶-модда. — the 40-paragraph reset used to expose it as an own article.
    body = [f"инстанция матни {i}." for i in range(45)]
    paras = ["1-модда. Кодексга қуйидаги ўзгартишлар киритилсин:",
             "1) қуйидаги мазмундаги 43-боб билан тўлдирилсин:",
             "«43-боб. Умумий қоидалар"] + body + [
             "372⁶-модда. Ажримнинг қонуний кучга кириши",
             "Ажрим дарҳол қонуний кучга киради»;",
             "2-модда. Вазирлар Маҳкамаси:",
             "ҳукумат қарорларини мувофиқлаштирсин.",
             "3-модда. Ушбу Қонун расмий эълон қилинган кундан эътиборан кучга киради."]
    _, ns = _ns(paras)
    assert _arts(ns) == ["1", "2", "3"]
    assert f"{D}/ГЛ.43" not in ns


def test_unclosed_quoted_article_ends_at_next_own_article():
    # 8.2. Қонун матни.docx: «41-модда. ... never closes with »; own 2-модда. follows 10 paras later.
    paras = ["1-модда. Қонунга қуйидаги ўзгартишлар киритилсин:",
             "1) 41-модда қуйидаги таҳрирда баён этилсин:",
             "«41-модда. Фаолият билан шуғулланиш ҳуқуқи",
             "Биринчи қисм.",
             "Иккинчи қисм.",
             "2-модда. Бошқа Қонунга ўзгартиш киритилсин:",
             "«Дорихона мудири маълумотга эга бўлиши керак.»;",
             "3-модда. Вазирлар Маҳкамаси ижрони таъминласин.",
             "4-модда. Ушбу Қонун расмий эълон қилинган кундан эътиборан кучга киради."]
    _, ns = _ns(paras)
    assert _arts(ns) == ["1", "2", "3", "4"]


def test_requisites_two_line_president_signature():
    # Most corpus laws end with "Ўзбекистон Республикасининг" / "Президенти Ш. Мирзиёев" [/ "QL_..."].
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.",
             "3-модда. Ушбу Қонун расмий эълон қилинган кундан эътиборан кучга киради.",
             "Ўзбекистон Республикасининг", "Президенти Ш. Мирзиёев", "QL_1075/Б-М"]
    dt, ns = _ns(paras)
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end] == "Ўзбекистон Республикасининг\n\nПрезиденти Ш. Мирзиёев\n\nQL_1075/Б-М"
    assert ns[f"{D}/СТ.3"].end <= req.start


def test_requisites_one_line_signature_ru():
    paras = ["Статья 1. Первая", "Текст.", "Статья 2. Вторая", "Текст.",
             "Статья 3. Вступление в силу", "Настоящий Закон вступает в силу со дня опубликования.",
             "Президент", "Республики Узбекистан Ш. Мирзиёев"]
    dt, ns = _ns(paras)
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end].startswith("Президент")


def test_requisites_single_signature_line_plus_initial():
    # 5.1. Qonun matni.docx tail: one signature line followed by a stray "Ш."
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.", "3-модда. Учинчи.",
             "Ўзбекистон Республикасининг Президенти Ш. Мирзиёев", "Ш."]
    _, ns = _ns(paras)
    assert f"{D}/РЕКВ.1" in ns


def test_short_last_article_line_is_not_requisites():
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.", "3-модда. Учинчи модда",
             "Ушбу модда ўз кучини йўқотади.", "Бошқа гап."]
    _, ns = _ns(paras)
    assert f"{D}/РЕКВ.1" not in ns


# --- Fix round 1 -------------------------------------------------------------------------

def test_self_closing_nested_ru_quote_paragraph():
    # 1.4. Текст Закона.docx paras 896–899: «… «О санации и ликвидации банков»; — two «, one ».
    paras = ["1-модда. Первая.", "2-модда. Вторая.",
             "3-модда. Внести изменения:",
             "1) статью 15 дополнить частью девятой следующего содержания:",
             "«Особенности создания промежуточного банка устанавливаются Законом Республики "
             "Узбекистан «О санации и ликвидации банков»;",
             "2) абзац седьмой части первой статьи 54 изложить в следующей редакции:",
             "«снижение коэффициентов адекватности капитала на пятьдесят или более процентов»;",
             "3) абзац третий части первой статьи 55 исключить."]
    _, ns = _ns(paras)
    a3 = f"{D}/СТ.3/Ч.1"
    assert all(f"{a3}/П.{k}" in ns for k in (1, 2, 3))


def test_stray_open_quote_in_own_article_does_not_swallow_articles():
    # stray « in own 2-модда. and an unrelated » 60 paragraphs later
    paras = ["1-модда. Биринчи.", "2-модда. «Иккинчи модда матни."]
    for k in range(3, 62):
        paras += [f"{k}-модда. Модда {k}.", "матн."] if k != 32 else [f"{k}-модда. Модда {k}.", "матн»."]
    _, ns = _ns(paras)
    assert _arts(ns) == sorted(str(k) for k in range(1, 62))


def test_quote_capped_at_400_paragraphs():
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.", "3-модда. Учинчи:",
             "«43-боб. Қўшилган боб"] + [f"матн {i}." for i in range(420)] + ["охири»."] + \
            ["Ўз матн."]
    dt, _ = _ns(paras)
    # own paragraphs beyond the 400-paragraph cap are not part of a quote any more:
    from graphrag.data.structure import _own_flags
    own = _own_flags(dt.paras)
    assert not own[4] and own[4 + 401]


def test_tail_without_signature_is_not_requisites():
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.", "3-модда. Бекор қилинган ҳужжатлар",
             "Ушбу Қонун қабул қилиниши муносабати билан:",
             "Қуйидагилар ўз кучини йўқотган деб топилсин:",
             "2019 йил 9 январдаги ЎРҚ-512-сонли Қонун."]
    _, ns = _ns(paras)
    assert f"{D}/РЕКВ.1" not in ns


def test_two_level_numbering_dot_points_then_paren_subpoints():
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.",
             "3-модда. Ўзгартишлар",
             "1. Биринчи бандга:", "1) биринчи кичик банд;", "2) иккинчи кичик банд;",
             "2. Иккинчи бандга:", "1) яна кичик банд;", "а) ичидаги ҳарф."]
    _, ns = _ns(paras)
    a = f"{D}/СТ.3/Ч.1"
    for k in (f"{a}/П.1/ПП.1", f"{a}/П.1/ПП.2", f"{a}/П.2/ПП.1"):
        assert k in ns
    assert not any("~" in k for k in ns)
    assert not any(k.startswith(f"{a}/П.2/ПП.a") for k in ns)


def test_paren_points_only_stay_points():
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.", "3-модда. Учинчи:",
             "1) биринчи;", "а) ҳарф;", "2) иккинчи."]
    _, ns = _ns(paras)
    a = f"{D}/СТ.3/Ч.1"
    assert f"{a}/П.1" in ns and f"{a}/П.2" in ns and f"{a}/П.1/ПП.a" in ns


# --- Task 7 fix round 1 (law area) ---------------------------------------------------------

def test_wrapped_chapter_title_lines_join_the_chapter_title():
    # Қонун.docx: "5-боб. Рақобатни чеклашга йўл қўймаслик мақсадида" / "иқтисодий концентрацияни
    # ва савдоларни" / "тартибга солиш" / "26-модда. …"
    paras = ["1-модда. Биринчи.", "2-модда. Иккинчи.",
             "5-боб. Рақобатни чеклашга йўл қўймаслик мақсадида",
             "иқтисодий концентрацияни ва савдоларни", "тартибга солиш",
             "3-модда. Учинчи.", "Матн."]
    dt, ns = _ns(paras)
    ch = ns[f"{D}/ГЛ.5"]
    assert ch.title == ("Рақобатни чеклашга йўл қўймаслик мақсадида иқтисодий концентрацияни "
                        "ва савдоларни тартибга солиш")
    assert dt.text[ns[f"{D}/ГЛ.5/СТ.3"].start:].startswith("3-модда.")


def test_paragraph_section_containers_between_chapter_and_article():
    # 1.3. Қонун матни.docx / 1.4. Текст Закона.docx: "1-§." / "§ 1." sub-chapter headings.
    for s1, s2, wrap in [("1-§. Санация мақсадлари", "2-§. Санация қилиш бўйича ваколатлар",
                          "ва уларни амалга ошириш"),
                         ("§ 1. Цели санации", "§ 2. Полномочия по санации", "и их осуществление")]:
        paras = ["1-боб. Умумий қоидалар", "1-модда. Биринчи.",
                 "2-боб. Санация", s1, "2-модда. Иккинчи.", "3-модда. Учинчи.",
                 s2, wrap, "4-модда. Тўртинчи.",
                 "3-боб. Якун", "5-модда. Бешинчи."]
        dt, ns = _ns(paras)
        assert ns[f"{D}/ГЛ.2/§.1"].type == "параграф"
        assert f"{D}/ГЛ.2/§.1/СТ.2" in ns and f"{D}/ГЛ.2/§.1/СТ.3" in ns
        assert f"{D}/ГЛ.2/§.2/СТ.4" in ns and f"{D}/ГЛ.1/СТ.1" in ns and f"{D}/ГЛ.3/СТ.5" in ns
        assert ns[f"{D}/ГЛ.2/§.2"].title.endswith(" " + wrap)
        p1 = ns[f"{D}/ГЛ.2/§.1"]
        assert dt.text[p1.start:p1.end].endswith("3-модда. Учинчи.")
