# Regression tests for defects found by running the Task 7 parsers over the real corpus
# (QARORLAR / STENOGRAMMA / XULOSALAR / BOSHQA).
from graphrag.data.extract import Para
from graphrag.data.structure import DocText, parse_prose, parse_resolution, parse_stenogram
from tests.helpers import words

D = "X-0000abcd"


def _ns(nodes):
    return {n.node_id: n for n in nodes}


def _dt(P):
    return DocText.build([Para(t) for t in P])


def test_resolution_requisites_from_joined_table_rows():
    # QARORLAR/1. Қарор (1).docx: the signature table is extracted as "cell | cell" rows,
    # 8-9 words each (the plain 7-word limit rejected both).
    P = ["Сенат", "Қ А Р О Р Қ И Л А Д И :",
         "1. Қонун маъқуллансин.",
         "2. Ушбу Қарор қабул қилинган кундан эътиборан кучга киради.",
         "Ўзбекистон Республикаси Олий Мажлиси Сенатининг Раиси | Т. Нарбаева",
         "Тошкент шаҳри, 2024 йил 19 январь, № СҚ-851–IV."]
    dt = _dt(P)
    ns = _ns(parse_resolution(D, dt, "t"))
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end].startswith("Ўзбекистон Республикаси Олий Мажлиси")
    assert dt.text[ns[f"{D}/П.2"].start:ns[f"{D}/П.2"].end].endswith("кучга киради.")


def test_resolution_single_signed_row_is_requisites():
    # QARORLAR/1.2. Qaror.docx: one "title | signer" row after the last numbered item.
    P = ["Сенат", "ҚАРОР ҚИЛАДИ:", "1. Биринчи.", "2. Иккинчи.",
         "3. Ўзбекистон Республикаси Олий Мажлиси Сенатининг қўмитаси назорат қилсин.",
         "Ўзбекистон Республикаси Олий Мажлиси Сенатининг Раиси | Т. Нарбаева"]
    dt = _dt(P)
    ns = _ns(parse_resolution(D, dt, "t"))
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end] == P[-1]


def test_long_plain_last_line_is_still_not_requisites():
    P = ["Сенат", "ҚАРОР ҚИЛАДИ:", "1. Биринчи.",
         "Ушбу Қарор 2024 йил 1 январдан бошлаб амалга киритилсин ва ижро этилсин ҳамда",
         "Тошкент шаҳар ҳокимлиги ушбу Қарор ижросини таъминласин ва назорат қилиб борсин."]
    ns = _ns(parse_resolution(D, _dt(P), "t"))
    assert f"{D}/РЕКВ.1" not in ns


def test_prose_spaced_committee_conclusion_headers():
    # XULOSALAR/4. Қўмита хулосаси.docx, 5. Хулоса (7).docx, 7.5. ... Йўл ҳаракати.docx
    for head in ["Q Oʼ M I T A X U L O S A S I", "Қ Ў М И Т А Х У Л О С А С И",
                 "“Йўл ҳаракати хавфсизлиги тўғрисида”ги Қонун юзасидан Х У Л О С А"]:
        P = ["“Тўғрисида”ги Қонун", head, "Хулоса матни бир.", "Хулоса матни икки."]
        dt = _dt(P)
        ns = _ns(parse_prose(D, dt, "t", max_tokens=384, count_tokens=words))
        zag = ns[f"{D}/ЗАГ.1"]
        assert dt.text[zag.start:zag.end].endswith(head), head


def test_prose_decision_marker_and_body_sentence_are_not_headers():
    for t in ["Х У Л О С А Қ И Л А Д И :", "Қўмита қуйидаги хулоса."]:
        P = ["Кириш.", t, "Матн."]
        ns = _ns(parse_prose(D, _dt(P), "t", max_tokens=384, count_tokens=words))
        assert f"{D}/ЗАГ.1" not in ns, t


def test_stenogram_speaker_with_lowercase_patronymic():
    # STENOGRAMMA: "Azamat Hamid oʼgʼli Ziyo:", "Dilnozaxon Sobir qizi Kattaxanova:"
    P = ["Senat yalpi majlisi",
         "Raislik qiluvchi: Soʼz beriladi.",
         "Azamat Hamid oʼgʼli Ziyo: Hurmatli senatorlar.",
         "Dilnozaxon Sobir qizi Kattaxanova: Rahmat.",
         "Шукрулло Акмал ўғли Жаҳонгиров: Раҳмат."]
    ns = _ns(parse_stenogram(D, _dt(P), "t"))
    titles = [n.title for n in ns.values() if n.type == "выступление"]
    assert titles == ["Raislik qiluvchi", "Azamat Hamid oʼgʼli Ziyo",
                      "Dilnozaxon Sobir qizi Kattaxanova", "Шукрулло Акмал ўғли Жаҳонгиров"]


# --- Fix round 1 -------------------------------------------------------------------------

def test_single_row_ending_in_imperative_verb_is_not_requisites():
    P = ["Сенат", "ҚАРОР ҚИЛАДИ:", "1. Қонун маъқуллансин.", "2. Қарор кучга киради.",
         "Назорат Раис ўринбосари С. Сафоевга юклансин."]
    ns = _ns(parse_resolution(D, _dt(P), "t"))
    assert f"{D}/РЕКВ.1" not in ns


def test_stenogram_role_label_speakers_and_chair_typo():
    P = ["Senat yalpi majlisi",
         "Raislik qiluvchi: Soʼz beriladi.",
         "Toshkent shahar hokimligi vakili: Rahmat.",
         "Sirdaryo viloyati hokimi oʼrinbosari: Javob beraman.",
         "Maʼruzachi: Hurmatli senatorlar.",
         "Savol: bu qanday boʼladi?",
         "Misol uchun: bitta holat.",
         "Raisliq qiluvchi: Kun tartibidagi ikkinchi masalaga oʼtamiz."]
    nodes = parse_stenogram(D, _dt(P), "t")
    titles = [n.title for n in nodes if n.type == "выступление"]
    assert titles == ["Raislik qiluvchi", "Toshkent shahar hokimligi vakili",
                      "Sirdaryo viloyati hokimi oʼrinbosari", "Maʼruzachi", "Raisliq qiluvchi"]
    assert sum(n.type == "вопрос" for n in nodes) == 2      # typo chair still announces agenda


def test_stenogram_next_item_trigger_and_question_title():
    P = ["Senat yalpi majlisi",
         "Raislik qiluvchi: Kun tartibidagi birinchi masalaga oʼtamiz. “Banklar toʼgʼrisida”gi Qonun.",
         "Erkin Fayziyevich Gadoyev: Maʼruza.",
         "Raislik qiluvchi: “Oʼzbekiston 2030” strategiyasi. Qaror qabul qilindi. "
         "Keyingi masala “Organik mahsulotlar toʼgʼrisida”gi Qonun haqida.",
         "Anvar Mirzaaxmedovich Tuychiyev: Maʼruza.",
         "Raislik qiluvchi: Rahmat. Navbatdagi masala Oliy sud tarkibi haqida.",
         "Raislik qiluvchi: keyingi masala boʼyicha savollar bormi?"]      # not sentence-initial
    nodes = parse_stenogram(D, _dt(P), "t")
    qs = [n for n in nodes if n.type == "вопрос"]
    assert [q.title for q in qs] == ["Banklar toʼgʼrisida", "Organik mahsulotlar toʼgʼrisida", ""]
    assert sum(n.parent_id == qs[2].node_id for n in nodes) == 2


def test_resolution_followed_by_committee_conclusion_is_split():
    # XULOSALAR/6.Qo`mita xulosasi.docx: committee QARORI (items + signature) followed by the
    # committee XULOSASI (title lines, head, body, signature) in one file.
    P = ["Qoʼmitaning QARORI",
         "Q A R O R Q I L A D I:",
         "1. Qonun maʼqullansin.",
         "2. Ushbu masala boʼyicha yalpi majlisda maʼruza qilinsin.",
         "Qoʼmita raisi",
         "Xalqaro munosabatlar qoʼmitasining 2025-yil 17-iyundagi 12-sonli qaroriga ilova",
         "“Ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida”gi Qonun yuzasidan",
         "Q Oʼ M I T A X U L O S A S I",
         "Qonunchilik palatasi tomonidan qabul qilingan Qonun koʼrib chiqildi.",
         "Qonun Oʼzbekiston Respublikasi Konstitutsiyasiga va xalqaro shartnomalarga toʼliq muvofiq keladi.",
         "Qoʼmita raisi K. Burxanov"]
    dt = _dt(P)
    ns = _ns(parse_resolution(D, dt, "t", max_tokens=5, count_tokens=words))
    txt = lambda k: dt.text[ns[k].start:ns[k].end]
    assert txt(f"{D}/П.2").endswith("maʼruza qilinsin.")
    assert txt(f"{D}/РЕКВ.1") == "Qoʼmita raisi"
    assert txt(f"{D}/ЗАГ.1").startswith("Xalqaro munosabatlar") and txt(f"{D}/ЗАГ.1").endswith("S I")
    assert txt(f"{D}/БЛ.1").startswith("Qonunchilik palatasi") and f"{D}/БЛ.2" in ns
    assert txt(f"{D}/РЕКВ.2") == "Qoʼmita raisi K. Burxanov"
    assert not any("~" in k for k in ns)


def test_resolution_titled_qarori_without_decision_but_with_xulosa_head_is_prose():
    # XULOSALAR/3.3. Qo_mita xulosasi.docx: "Q A R O R I" header, no "QAROR QILADI", then "X U L O S A".
    P = ["Qoʼmitaning", "Q A R O R I", "2025-yil -sonli Toshkent sh.", "“Qonun” yuzasidan",
         "X U L O S A", "Qonun koʼrib chiqildi.", "Qonun maʼqullansin."]
    dt = _dt(P)
    ns = _ns(parse_resolution(D, dt, "t"))
    assert {n.type for n in ns.values()} <= {"документ", "заголовок", "блок", "реквизиты"}
    assert dt.text[ns[f"{D}/ЗАГ.1"].start:ns[f"{D}/ЗАГ.1"].end].endswith("X U L O S A")


# --- Fix round 2: agenda (вопрос) titles ------------------------------------------------------

def _q_titles(chair_turn_paras):
    P = ["Senat yalpi majlisi", "Raislik qiluvchi: Assalomu alaykum."] + chair_turn_paras + \
        ["Erkin Fayziyevich Gadoyev: Maʼruza."]
    nodes = parse_stenogram(D, _dt(P), "t")
    return [n.title for n in nodes if n.type == "вопрос"][1:]


def test_agenda_title_slogan_after_trigger_is_not_a_title():
    assert _q_titles(["Raislik qiluvchi: Rahmat. Navbatdagi masala Oliy sud tarkibi haqida.",
                      "Bu “Shon-sharaf” ordeni bilan bogʼliq emas.",
                      "“Yashil makon” loyihasi ham muhim."]) == [""]


def test_agenda_title_only_near_the_trigger():
    assert _q_titles(["Raislik qiluvchi: Rahmat. Navbatdagi masala Oliy sud tarkibi haqida.",
                      "Soʼz beriladi.",
                      "“Banklar toʼgʼrisida”gi Qonun ham bor."]) == [""]


def test_agenda_title_nested_quotes_keep_outer_title():
    t = ("Navbatdagi masala “Oʼzbekiston Respublikasining “Davlat boji toʼgʼrisida”gi Qonuniga "
         "oʼzgartirish kiritish haqida”gi Qonun.")
    assert _q_titles([f"Raislik qiluvchi: Rahmat. {t}"]) == [
        "Oʼzbekiston Respublikasining “Davlat boji toʼgʼrisida”gi Qonuniga oʼzgartirish kiritish haqida"]


def test_agenda_title_act_word_after_quote_latin_and_cyrillic():
    assert _q_titles(["Raislik qiluvchi: Keyingi masala “Organik mahsulotlar”gi Oʼzbekiston "
                      "Respublikasi Qonuni haqida."]) == ["Organik mahsulotlar"]
    assert _q_titles(["Раислик қилувчи: Навбатдаги масала «Ўсимликлар карантини тўғрисида»ги "
                      "Қонун ҳақида."]) == ["Ўсимликлар карантини тўғрисида"]
    assert _q_titles(["Раислик қилувчи: Навбатдаги масала «Маҳаллабай» тизими ҳақида."]) == [""]


def test_agenda_title_wrapped_across_the_next_paragraph():
    # Sten. 2-yalpi majlis: trigger opens a new paragraph of the chair turn, the title wraps.
    assert _q_titles(["Raislik qiluvchi: Qaror qabul qilindi. Rahmat.",
                      "Navbatdagi masala “Ipoteka krediti bozori munosabati bilan ayrim",
                      "qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida”gi Qonun haqida."]) == [
        "Ipoteka krediti bozori munosabati bilan ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida"]
