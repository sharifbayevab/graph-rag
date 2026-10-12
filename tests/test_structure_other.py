# advanced-rag-v2/tests/test_structure_other.py
from graphrag.data.extract import Para
from graphrag.data.structure import (DocText, parse_prose, parse_resolution, parse_stenogram,
                                     parse_structure)
from tests.helpers import words

D = "X-0000abcd"


def _ids(nodes):
    return {n.node_id: n for n in nodes}


def test_resolution():
    P = ["“Банклар тўғрисида”ги Қонун ҳақида",
         "Қонунчилик палатаси томонидан қабул қилинган Қонунни кўриб чиқиб, Сенат",
         "Қ  А  Р  О  Р     Қ  И  Л  А  Д  И :",
         "1. “Банклар тўғрисида”ги Қонун маъқуллансин.",
         "1) биринчи кичик банд;",
         "2. Ушбу Қарор қабул қилинган кундан эътиборан кучга киради.",
         "Олий Мажлиси Сенатининг Раиси", "Т. Нарбаева", "Тошкент шаҳри,",
         "2024 йил 19 январь,", "№ СҚ-851–IV."]
    dt = DocText.build([Para(t) for t in P])
    ns = _ids(parse_resolution(D, dt, "t"))
    assert dt.text[ns[f"{D}/ПРЕАМБ.1"].start:ns[f"{D}/ПРЕАМБ.1"].end].endswith("Қ  И  Л  А  Д  И :")
    assert f"{D}/П.1" in ns and f"{D}/П.1/ПП.1" in ns and f"{D}/П.2" in ns
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end].startswith("Олий Мажлиси")


def test_resolution_without_numbering_makes_ordinal_items():
    P = ["Сенат", "ҚАРОР ҚИЛАДИ:", "Қонун маъқуллансин.", "Қарор кучга киради."]
    ns = _ids(parse_resolution(D, DocText.build([Para(t) for t in P]), "t"))
    assert f"{D}/П.1" in ns and f"{D}/П.2" in ns


def test_stenogram_turns_grouped_by_agenda():
    P = ["Oʼzbekiston Respublikasi Oliy Majlisi Senatining yalpi majlisi",
         "Raislik qiluvchi: Assalomu alaykum. Majlisni ochiq deb eʼlon qilaman.",
         "(Davlat madhiyasi yangraydi)",
         "Raislik qiluvchi: Kun tartibidagi birinchi masalani koʼrib chiqishga oʼtamiz.",
         "Erkin Fayziyevich Gadoyev: Hurmatli senatorlar, qonun loyihasi haqida.",
         "Davomi matn.",
         "Raislik qiluvchi: Kun tartibidagi ikkinchi masalaga oʼtamiz."]
    ns = _ids(parse_stenogram(D, DocText.build([Para(t) for t in P]), "t"))
    assert f"{D}/ЗАГ.1" in ns
    assert f"{D}/ВОПР.1/ВЫСТ.1" in ns
    assert ns[f"{D}/ВОПР.2/ВЫСТ.2"].title == "Erkin Fayziyevich Gadoyev"
    assert f"{D}/ВОПР.3/ВЫСТ.1" in ns


def test_prose_blocks_pack_paragraphs():
    P = ["2024 йил “___” ____ -сонли", "Х У Л О С А",
         "бир икки уч тўрт беш", "олти етти саккиз тўққиз ўн", "ўн бир ўн икки ўн уч"]
    dt = DocText.build([Para(t) for t in P])
    ns = _ids(parse_prose(D, dt, "t", max_tokens=10, count_tokens=words))
    assert f"{D}/ЗАГ.1" in ns
    b1, b2 = ns[f"{D}/БЛ.1"], ns[f"{D}/БЛ.2"]
    assert dt.text[b1.start:b1.end] == "бир икки уч тўрт беш\n\nолти етти саккиз тўққиз ўн"
    assert dt.text[b2.start:b2.end] == "ўн бир ўн икки ўн уч"


def test_dispatch_qonunlar_without_articles_falls_back_to_prose():
    paras = [Para("Ратификация тўғрисида"), Para("Конвенция ратификация қилинсин.")]
    dt, nodes = parse_structure(D, "QONUNLAR", "law_text", paras, "t", 384, words)
    assert {n.type for n in nodes} <= {"документ", "заголовок", "блок", "реквизиты"}
    assert any(n.type == "блок" for n in nodes)


def test_dispatch_multi_document_file_is_valid_prose():
    paras = [Para(t) for t in ["ҚАРОРИ", "Қўмита қарор қилади.", "Х У Л О С А", "Хулоса матни."]]
    dt, nodes = parse_structure(D, "XULOSALAR", "committee_conclusion", paras, "t", 384, words)
    ids = [n.node_id for n in nodes]
    assert len(ids) == len(set(ids))
    assert all(0 <= n.start <= n.end <= len(dt.text) for n in nodes)
