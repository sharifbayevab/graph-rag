from graphrag.data.extract import Para
from graphrag.data.structure import DocText, parse_law

D = "QONUNLAR-0000abcd"
LAW = [
    "Ўзбекистон Республикасининг Қонуни",                                        # 0 преамбула
    "Банкларни санация қилиш тўғрисида",                                         # 1
    "1-боб. Умумий қоидалар",                                                    # 2
    "1-модда. Ушбу Қонуннинг мақсади",                                           # 3
    "Ушбу Қонуннинг мақсади муносабатларни тартибга солишдан иборат (Ахборотнома, 2019 йил, № 9,",  # 4
    "144-модда; 2020 йил, № 3, 47-модда).",                                      # 5 citation wrap
    "2-модда. Асосий тушунчалар",                                                # 6
    "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",                     # 7
    "санация – банкни тиклаш;",                                                  # 8
    "олувчи – банк.",                                                            # 9
    "Иккинчи қисм матни.",                                                       # 10
    "2-боб. Санация тартиби",                                                    # 11
    "3-модда. Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",                # 12 header = body
    "1) 27-модда биринчи қисмининг 1-банди қуйидаги таҳрирда баён этилсин:",     # 13
    "«1) агар иш судга юборилган бўлса;»;",                                      # 14 quoted
    "2) қуйидаги мазмундаги 122¹-модда билан тўлдирилсин:",                      # 15
    "«122¹-модда. Болани назоратсиз қолдириш",                                   # 16 quoted article
    "Болани назоратсиз қолдириш —",                                              # 17 inside quote
    "жарима солишга сабаб бўлади.»;",                                            # 18 closes quote
    "а) биринчи кичик банд;",                                                    # 19
    "Ўзбекистон Республикаси Президенти",                                        # 20 реквизиты
    "Ш. Мирзиёев",
    "Тошкент ш.,",
    "2024 йил 19 январь,",
    "№ ЎРҚ-900",
]


def _nodes(paras=LAW):
    dt = DocText.build([Para(t) for t in paras])
    return dt, {n.node_id: n for n in parse_law(D, dt, "Банкларни санация қилиш тўғрисида")}


def test_articles_and_chapters():
    _, ns = _nodes()
    arts = sorted(k for k, n in ns.items() if n.type == "статья")
    assert arts == [f"{D}/ГЛ.1/СТ.1", f"{D}/ГЛ.1/СТ.2", f"{D}/ГЛ.2/СТ.3"]
    assert ns[f"{D}/ГЛ.1"].title == "Умумий қоидалар"
    assert ns[f"{D}/ГЛ.1/СТ.1"].title == "Ушбу Қонуннинг мақсади"
    assert ns[f"{D}/ГЛ.2/СТ.3"].title == ""                    # header carried body text
    assert not any("122" in k or "144" in k for k in arts)       # quoted / citation not articles


def test_parts_points_subpoints():
    _, ns = _nodes()
    a2 = f"{D}/ГЛ.1/СТ.2"
    assert f"{a2}/Ч.1" in ns and f"{a2}/Ч.2" in ns and f"{a2}/Ч.3" not in ns
    a3 = f"{D}/ГЛ.2/СТ.3"
    assert f"{a3}/Ч.1/П.1" in ns and f"{a3}/Ч.1/П.2" in ns
    assert f"{a3}/Ч.1/П.2/ПП.a" in ns
    assert ns[f"{a3}/Ч.1/П.2"].number == "2"


def test_spans_nest_and_cover_text():
    dt, ns = _nodes()
    for n in ns.values():
        assert 0 <= n.start < n.end <= len(dt.text)
        if n.parent_id:
            p = ns[n.parent_id]
            assert p.start <= n.start and n.end <= p.end
    p2 = ns[f"{D}/ГЛ.2/СТ.3/Ч.1/П.2"]
    assert "жарима солишга" in dt.text[p2.start:p2.end]          # quoted block belongs to the point


def test_preamble_and_requisites():
    dt, ns = _nodes()
    pre, req = ns[f"{D}/ПРЕАМБ.1"], ns[f"{D}/РЕКВ.1"]
    assert dt.text[pre.start:pre.end].startswith("Ўзбекистон Республикасининг Қонуни")
    assert dt.text[req.start:req.end].startswith("Ўзбекистон Республикаси Президенти")
    assert dt.text[req.start:req.end].endswith("№ ЎРҚ-900")


def test_node_ids_are_unique_and_stable():
    _, a = _nodes()
    _, b = _nodes()
    assert list(a) == list(b) and len(a) == len(set(a))


def test_unclosed_quote_does_not_swallow_document():
    paras = ["1-модда. Биринчи", "«очиқ қолган кавычка"] + [f"матн {i}." for i in range(45)] + \
            ["2-модда. Иккинчи", "матн.", "3-модда. Учинчи", "матн."]
    _, ns = _nodes(paras)
    assert f"{D}/СТ.2" in ns and f"{D}/СТ.3" in ns


def test_too_few_articles_returns_none():
    dt = DocText.build([Para("Ратификация қилинсин."), Para("1-модда. Ягона модда")])
    assert parse_law(D, dt, "") is None
