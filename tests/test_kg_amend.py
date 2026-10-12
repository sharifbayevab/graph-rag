# tests/test_kg_amend.py -- M2 Task 5: AMENDS (op, date) and REPEALS (date)
from graphrag.kg.acts import ActRegistry
from graphrag.kg.extract_amend import extract_amendments
from graphrag.kg.extract_ref import RefContext, extract_refs, structure_nodes
from graphrag.kg.schema import Schema
from tests.test_kg_refs import LAW, _rec

TAIL = ["2-модда. Ижро", "Ижро этилсин.", "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]
AMEND = ["1-модда. Жиноят кодексига ўзгартиш ва қўшимчалар",
         "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартиш ва қўшимчалар киритилсин:",
         "1) 9-модда:",
         "биринчи қисми қуйидаги таҳрирда баён этилсин:",
         "«Янги матн»;",
         "қуйидаги мазмундаги учинчи қисм билан тўлдирилсин:",
         "«Учинчи қисм матни»;",
         "учинчи ва тўртинчи қисмлари тегишинча тўртинчи ва бешинчи қисмлар деб ҳисоблансин;",
         "2) қуйидаги мазмундаги 122¹-модда билан тўлдирилсин:",
         "«122¹-модда. Янги модда",
         "Янги модда матни»;",
         "3) 5-модда чиқариб ташлансин;",
         "4) 10-модданинг учинчи қисмидаги «жарима» деган сўзлар «жарима ёки ахлоқ тузатиш ишлари» деган сўзлар "
         "билан алмаштирилсин;",
         "5) 78 – 80-моддалар чиқариб ташлансин.",
         *TAIL]

REPEAL = ["1-модда. Айрим қонун ҳужжатларини ўз кучини йўқотган деб топиш",
          "Қуйидагилар ўз кучини йўқотган деб топилсин:",
          "1) Ўзбекистон Республикасининг 1997 йил 24 апрелда қабул қилинган «Табиий монополиялар тўғрисида»ги "
          "398–I-сонли Қонуни (Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, 1997 йил, № 4–5, 106-модда);",
          "2) Ўзбекистон Республикасининг 2014 йил 20 январда қабул қилинган «Ўзбекистон Республикасининг айрим "
          "қонун ҳужжатларига ўзгартиш ва қўшимчалар киритиш тўғрисида»ги ЎРҚ–365-сонли Қонунининг (Ўзбекистон "
          "Республикаси Олий Мажлиси палаталарининг Ахборотномаси, 2014 йил, № 1, 2-модда) 25-моддаси.",
          "2-модда. Ижро", "Ижро этилсин.", "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]


def _run(paras, date="2024-01-19", doc_id="QONUNLAR-aaaa0001", title="Банкларни санация қилиш тўғрисида"):
    rec = _rec(paras, doc_id=doc_id, title=title, date=date)
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    edges, nodes = extract_amendments(rec, reg, ctx)
    return rec, reg, ctx, edges, nodes


def _amends(edges):
    return {e.dst: e for e in edges if e.type == "AMENDS"}


def test_edit_of_part_27_2():
    rec, reg, ctx, edges, _ = _run(LAW)
    e = _amends(edges)["EXT/jinoyat kodeks/СТ.27/Ч.2"]
    assert e.attrs["op"] == "edit" and e.attrs["date"] == "2024-01-19"
    assert e.src == f"{rec.doc_id}/СТ.3/Ч.1/П.1"
    ev = e.attrs["evidence"]
    assert ev["quote"] == "27-модданинг иккинчи қисми қуйидаги таҳрирда баён этилсин"
    assert rec.text[ev["start"]:ev["end"]] == ev["quote"]


def test_add_of_new_article_122_1():
    rec, reg, ctx, edges, nodes = _run(AMEND)
    e = _amends(edges)["EXT/jinoyat kodeks/СТ.122.1"]
    assert e.attrs["op"] == "add" and e.src == f"{rec.doc_id}/СТ.1/Ч.1/П.2"
    stub = next(n for n in nodes if n.node_id == "EXT/jinoyat kodeks/СТ.122.1")
    assert stub.external and stub.type == "статья"


def test_delete():
    _, _, _, edges, _ = _run(AMEND)
    am = _amends(edges)
    assert am["EXT/jinoyat kodeks/СТ.5"].attrs["op"] == "delete"
    assert all(am[f"EXT/jinoyat kodeks/СТ.{n}"].attrs["op"] == "delete" for n in (78, 79, 80))   # "78 – 80-моддалар"


def test_replace():
    _, _, _, edges, _ = _run(AMEND)
    e = _amends(edges)["EXT/jinoyat kodeks/СТ.10/Ч.3"]
    assert e.attrs["op"] == "replace"
    assert e.attrs["evidence"]["quote"].startswith("10-модданинг учинчи қисмидаги «жарима»")
    assert e.attrs["evidence"]["quote"].endswith("билан алмаштирилсин")


def test_two_repeals_with_the_document_date():
    rec, reg, ctx, edges, nodes = _run(REPEAL, date="2023-02-14")
    rp = {e.dst: e for e in edges if e.type == "REPEALS"}
    assert set(rp) == {"EXT/tabiiy monopoliyalar toʼgʼrisida",
                       "EXT/ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida#oʼrq-365/СТ.25"}
    assert all(e.attrs["date"] == "2023-02-14" for e in rp.values())
    assert {e.src for e in rp.values()} == {f"{rec.doc_id}/СТ.1/Ч.1/П.1", f"{rec.doc_id}/СТ.1/Ч.1/П.2"}
    for e in rp.values():                         # the item + the list formula
        quotes = [x["quote"] for x in e.attrs["evidence_all"]]
        assert "Қуйидагилар ўз кучини йўқотган деб топилсин:" in quotes and e.attrs["evidence"]["quote"] in quotes
    assert ctx.repealed_on == {d: "2023-02-14" for d in rp} and ctx.violations == []
    stub = ctx.stubs["EXT/ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida#oʼrq-365/СТ.25"]
    assert stub.attrs["repealed_on"] == "2023-02-14"


def test_repeal_without_a_date_is_a_validation_violation():
    rec, reg, ctx, edges, _ = _run(REPEAL, date=None)
    rp = [e for e in edges if e.type == "REPEALS"]
    assert len(rp) == 2 and all("date" not in e.attrs for e in rp)       # still emitted ...
    assert {(v["src"], v["dst"]) for v in ctx.violations} == {(e.src, e.dst) for e in rp}   # ... and listed
    nodes = {n.node_id: n for n in structure_nodes(rec) + reg.nodes() + ctx.stub_nodes()}
    for e in rp:                                                        # Task 9 build fails on them
        assert any("missing required attr 'date'" in m for m in Schema.load().validate_edge(e, nodes))


# --- Task 5: more ------------------------------------------------------------------------------------

import json

from graphrag.kg.extract_amend import apply_repealed_on, document_date


def _validate(rec, reg, ctx, edges):
    nodes = {n.node_id: n for n in structure_nodes(rec) + reg.nodes() + ctx.stub_nodes()}
    return [m for e in edges + ctx.contains_edges() for m in Schema.load().validate_edge(e, nodes)]


def test_header_clauses_give_the_article_and_every_operation_is_one_edge():
    rec, reg, ctx, edges, _ = _run(AMEND)
    am = _amends(edges)
    p1 = f"{rec.doc_id}/СТ.1/Ч.1/П.1"
    assert am["EXT/jinoyat kodeks/СТ.9/Ч.1"].attrs["op"] == "edit" and am["EXT/jinoyat kodeks/СТ.9/Ч.1"].src == p1
    assert am["EXT/jinoyat kodeks/СТ.9/Ч.3"].attrs["op"] == "add"           # the new part (renumbered one merged)
    assert am["EXT/jinoyat kodeks/СТ.9/Ч.4"].attrs["op"] == "renumber"      # the old parts, not the new numbers
    assert "EXT/jinoyat kodeks/СТ.9/Ч.5" not in am
    merged = am["EXT/jinoyat kodeks/СТ.9/Ч.3"].attrs["evidence_all"]
    assert [x["quote"][:9] for x in merged] == ["қуйидаги ", "учинчи ва"]


def test_every_edge_validates_with_exact_evidence_and_deepest_source():
    for paras, date in ((LAW, "2024-01-19"), (AMEND, "2024-01-19"), (REPEAL, "2023-02-14")):
        rec, reg, ctx, edges, _ = _run(paras, date=date)
        assert edges and _validate(rec, reg, ctx, edges) == []
        di = ctx.docs[rec.doc_id]
        for e in edges:
            ev = e.attrs["evidence"]
            assert rec.text[ev["start"]:ev["end"]] == ev["quote"]
            assert e.src == di.node_at(ev["start"], ev["end"]).node_id


def test_amends_share_src_dst_with_the_refers_to_amended_edge_r11():
    rec = _rec(AMEND)
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    refs = extract_refs(rec, reg, ctx)
    edges, _ = extract_amendments(rec, reg, ctx)
    amended = {(e.src, e.dst) for e in refs if e.attrs["relation"] == "amended"}
    am = {(e.src, e.dst) for e in edges if e.type == "AMENDS"}
    for d in ("СТ.122.1", "СТ.5", "СТ.10/Ч.3", "СТ.78", "СТ.80"):
        assert any(k[1] == f"EXT/jinoyat kodeks/{d}" for k in am & amended)


def test_registry_notes_are_not_counted_twice():
    rec = _rec(AMEND)
    reg1 = ActRegistry.from_records([rec])
    extract_refs(rec, reg1, RefContext.from_records([rec], reg1))
    reg2 = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg2)
    extract_refs(rec, reg2, ctx)
    extract_amendments(rec, reg2, ctx)
    assert [n.to_json() for n in reg1.external_nodes()] == [n.to_json() for n in reg2.external_nodes()]


def test_instruction_paragraph_starting_with_a_quote():
    paras = ["1-модда. Жиноят кодексига ўзгартиш",
             "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",
             "1) 245-модданинг биринчи қисмидаги:",
             "«197⁷» рақами «197⁷, 197⁸» рақамлари билан алмаштирилсин;",
             "«назарда тутилган» деган сўзлар чиқариб ташлансин;",
             *TAIL]
    _, _, ctx, edges, _ = _run(paras)
    e = _amends(edges)["EXT/jinoyat kodeks/СТ.245/Ч.1"]
    assert e.attrs["op"] == "replace" and len(e.attrs["evidence_all"]) == 2
    assert ctx.stats["amend_unmasked_instruction_paras"] == 2


def test_chapter_stub_and_whole_text_operation_on_the_act():
    paras = ["1-модда. Жиноят кодексига ўзгартиш",
             "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",
             "1) қуйидаги мазмундаги 15¹-боб билан тўлдирилсин:", "«15¹-боб. Янги боб", "74²-модда. Матн»;",
             "2) Кодекснинг матнидаги «ички ишлар» деган сўзлар «ички ишлар ва миграция» деган сўзлар билан алмаштирилсин.",
             *TAIL]
    rec, _, _, edges, _ = _run(paras)
    ops = {(e.src.rsplit("/", 1)[-1], e.dst, e.attrs["op"]) for e in edges}
    assert ops == {("П.1", "EXT/jinoyat kodeks/ГЛ.15.1", "add"), ("П.2", "EXT/jinoyat kodeks", "replace")}   # R13
    assert not any(e.dst.endswith("/СТ.74.2") for e in edges)          # an article inside the quoted chapter


def test_corpus_act_element_is_the_real_node_and_a_missing_article_the_act():
    base = _rec(["1-модда. Мақсад", "Мақсад матни.", "2-модда. Тушунчалар", "Тушунчалар матни.",
                 "3-модда. Кучга кириш", "Қонун кучга киради."], doc_id="QONUNLAR-bbbb0001",
                title="Табиий монополиялар тўғрисида", date="2020-01-10")
    am = _rec(["1-модда. Қонунга ўзгартиш",
               "Ўзбекистон Республикасининг «Табиий монополиялар тўғрисида»ги Қонунига қуйидаги ўзгартишлар киритилсин:",
               "1) 2-модда қуйидаги таҳрирда баён этилсин:", "«Янги матн»;", "2) 7-модда чиқариб ташлансин.",
               *TAIL],
              doc_id="QONUNLAR-bbbb0002", title="«Табиий монополиялар тўғрисида»ги Қонунга ўзгартиш киритиш ҳақида",
              date="2024-05-01")
    reg = ActRegistry.from_records([base, am])
    ctx = RefContext.from_records([base, am], reg)
    edges, _ = extract_amendments(am, reg, ctx)
    got = {e.dst: e.attrs["op"] for e in edges}
    assert got == {"QONUNLAR-bbbb0001/СТ.2": "edit", "QONUNLAR-bbbb0001": "delete"}
    assert extract_amendments(base, reg, ctx) == ([], [])


def test_repeal_of_an_amending_law_whose_outer_quote_is_missing():
    paras = ["1-модда. Ўз кучини йўқотган деб топиш", "Қуйидагилар ўз кучини йўқотган деб топилсин:",
             "1) Ўзбекистон Республикасининг 2010 йил 14 декабрда қабул қилинган «Табиий монополиялар тўғрисида»ги "
             "Ўзбекистон Республикаси Қонунининг 4-моддасига ўзгартиш киритиш ҳақида»ги ЎРҚ–266-сонли Қонуни "
             "(Ўзбекистон Республикаси Олий Мажлиси палаталарининг Ахборотномаси, 2010 йил, № 12, 466-модда);",
             "2) Ўзбекистон Республикаси Олий Мажлисининг 1997 йил 24 апрелда қабул қилинган «Табиий монополиялар "
             "тўғрисида»ги Ўзбекистон Республикаси Қонунини амалга киритиш ҳақида»ги 399–I-сонли Қарори "
             "(Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, 1997 йил, № 4–5, 107-модда).",
             *TAIL]
    rec, reg, ctx, edges, _ = _run(paras, date="2023-02-14")
    # Ruling R29 A: one number for the key -> no edition (only a title the corpus cites with >= 2 numbers splits)
    assert [e.dst for e in edges] == ["EXT/tabiiy monopoliyalar toʼgʼrisida qonun 4 moddasiga oʼzgartirish kiritish toʼgʼrisida"]
    assert any(n.node_id == edges[0].dst for n in reg.external_nodes())             # the EXT act vertex exists
    assert [u["reason"] for u in ctx.unresolved] == ["repeal_non_law_act"]          # a Qaror has no act vertex


def test_inline_repeal_of_an_act():
    paras = ["1-модда. Ўз кучини йўқотган деб топиш",
             "Ўзбекистон Республикасининг 1997 йил 24 апрелда қабул қилинган «Табиий монополиялар тўғрисида»ги "
             "398–I-сонли Қонуни ўз кучини йўқотган деб топилсин.",
             *TAIL]
    rec, _, ctx, edges, _ = _run(paras, date="2023-02-14")
    assert [(e.type, e.dst, e.attrs["date"]) for e in edges] == [("REPEALS", "EXT/tabiiy monopoliyalar toʼgʼrisida", "2023-02-14")]
    assert edges[0].attrs["evidence"]["quote"].endswith("ўз кучини йўқотган деб топилсин")


def test_only_law_texts_are_scanned():
    text = ("1. «Бола ҳуқуқларининг кафолатлари тўғрисида»ги Ўзбекистон Республикаси Қонунининг 5-моддаси "
            "қуйидаги таҳрирда баён этилсин.")
    from tests.test_kg_refs import _manual
    rec = _manual("QARORLAR-cccc0001", "resolution", "QARORLAR", text, [("П.1", "пункт", "1", "", 0, len(text))])
    reg = ActRegistry.from_records([rec])
    assert extract_amendments(rec, reg) == ([], [])


def test_document_date_falls_back_to_the_adoption_line_and_the_preamble():
    from tests.test_kg_refs import _manual
    rec = _rec(["Қонунчилик палатаси томонидан 2025 йил 17 июнда қабул қилинган", "1-модда. Матн", "Матн.",
                "2-модда. Матн", "Матн.", "3-модда. Матн", "Матн."], date=None)
    assert document_date(rec) == "2025-06-17"
    text = "Тошкент ш., 2024 йил 5 март\n\n1-модда. Матн"
    rec2 = _manual("QONUNLAR-dddd0001", "law_text", "QONUNLAR", text, [("ПРЕАМБ.1", "преамбула", "1", "", 0, 27)],
                   date=None)
    assert document_date(rec2) == "2024-03-05"
    assert document_date(_manual("QONUNLAR-dddd0002", "law_text", "QONUNLAR", "1-модда. Матн", [], date=None)) is None


def test_apply_repealed_on_and_determinism():
    runs = []
    for _ in range(2):
        rec, reg, ctx, edges, nodes = _run(REPEAL, date="2023-02-14")
        runs.append(json.dumps([e.to_json() for e in edges] + [n.to_json() for n in nodes], ensure_ascii=False,
                               sort_keys=True))
    assert runs[0] == runs[1]
    ext = apply_repealed_on(reg.external_nodes(), ctx.repealed_on)
    assert next(n for n in ext if n.node_id == "EXT/tabiiy monopoliyalar toʼgʼrisida").attrs["repealed_on"] == "2023-02-14"


def test_a_translation_is_skipped():
    from graphrag.kg.schema import GraphEdge
    rec = _rec(AMEND)
    reg = ActRegistry.from_records([rec])
    assert extract_amendments(rec, reg)[0]
    reg.version_edges.append(GraphEdge(rec.doc_id, "QONUNLAR-ffff0001", "REFERS_TO", {"relation": "translation_of"}))
    assert extract_amendments(rec, reg) == ([], [])


# --- Task 5 fix round 1 (Rulings R12, R13) -----------------------------------------------------------

def test_merged_operations_keep_ops_aligned_with_evidence_all():
    _, _, _, edges, _ = _run(AMEND)
    e = _amends(edges)["EXT/jinoyat kodeks/СТ.9/Ч.3"]
    assert e.attrs["op"] == "add" and e.attrs["ops"] == ["add", "renumber"]
    assert len(e.attrs["ops"]) == len(e.attrs["evidence_all"])
    assert "ops" not in _amends(edges)["EXT/jinoyat kodeks/СТ.5"].attrs          # a single operation
    bad = GraphEdge(e.src, e.dst, "AMENDS", {**e.attrs, "ops": ["add", "move"]})
    assert any("op 'move'" in m for m in Schema.load().validate_edge(bad, {e.src: _N(e.src), e.dst: _N(e.dst)}))


def _N(nid):
    from graphrag.kg.schema import GraphNode
    return GraphNode(nid, "element", "статья", "", None, True, {})


from graphrag.kg.schema import GraphEdge  # noqa: E402


def test_units_become_ext_stub_chains():
    paras = ["1-модда. Қонунга ўзгартиш",
             "Ўзбекистон Республикасининг «Лицензиялаш тўғрисида»ги Қонунига қуйидаги ўзгартишлар киритилсин:",
             "1) 4-илова:", "I бўлимининг:", "32-позицияси чиқариб ташлансин;",
             "2) 4-илованинг II бўлими қуйидаги мазмундаги 116 ва 117-позициялар билан тўлдирилсин:", "«116. Матн»;",
             "3) XXIV¹ боб қуйидаги таҳрирда баён этилсин:", "«Матн»;",
             "4) IV бўлим қуйидаги мазмундаги 43-боб билан тўлдирилсин:", "«Матн»;",
             "5) илованинг 3-банди чиқариб ташлансин.", *TAIL]
    rec, reg, ctx, edges, nodes = _run(paras)
    act = "EXT/lisenziyalash toʼgʼrisida"
    got = {e.dst.replace(act, "", 1): e.attrs["op"] for e in edges}
    assert got == {"/ИЛОВА.4/БЎЛ.1/ПОЗ.32": "delete", "/ИЛОВА.4/БЎЛ.2/ПОЗ.116": "add", "/ИЛОВА.4/БЎЛ.2/ПОЗ.117": "add",
                   "/ГЛ.24.1": "edit", "/ГЛ.43": "add", "/ИЛОВА.1/П.3": "delete"}, got
    stub = ctx.stubs[act + "/ИЛОВА.4/БЎЛ.2"]
    assert stub.type == "раздел" and stub.title == "2-бўлим" and stub.external
    assert (act + "/ИЛОВА.4", act + "/ИЛОВА.4/БЎЛ.2") in {(e.src, e.dst) for e in ctx.contains_edges()}
    assert _validate(rec, reg, ctx, edges) == []


def test_a_corpus_chapter_is_its_node_and_never_invented():
    base = _rec(["1-боб. Умумий қоидалар", "1-модда. Мақсад", "Мақсад матни.", "2-модда. Тушунчалар", "Тушунчалар матни.",
                 "2-боб. Якуний қоидалар", "3-модда. Кучга кириш", "Қонун кучга киради."], doc_id="QONUNLAR-bbbb0001",
                title="Табиий монополиялар тўғрисида", date="2020-01-10")
    am = _rec(["1-модда. Қонунга ўзгартиш",
               "Ўзбекистон Республикасининг «Табиий монополиялар тўғрисида»ги Қонунига қуйидаги ўзгартишлар киритилсин:",
               "1) 2-бобнинг номи қуйидаги таҳрирда баён этилсин:", "«Янги ном»;",
               "2) 5-бобнинг номидаги «X» деган сўз чиқариб ташлансин.", *TAIL],
              doc_id="QONUNLAR-bbbb0002", title="«Табиий монополиялар тўғрисида»ги Қонунга ўзгартиш киритиш ҳақида",
              date="2024-05-01")
    reg = ActRegistry.from_records([base, am])
    ctx = RefContext.from_records([base, am], reg)
    edges, nodes = extract_amendments(am, reg, ctx)
    ch2 = next(n.node_id for n in base.structure if n.type == "глава" and n.number == "2")
    assert {e.dst: e.attrs["op"] for e in edges} == {ch2: "edit", "QONUNLAR-bbbb0001": "delete"}
    assert nodes == []


def test_a_repealed_section_is_its_stub_not_the_whole_act():
    paras = ["1-модда. Ўз кучини йўқотган деб топиш", "Қуйидагилар ўз кучини йўқотган деб топилсин:",
             "1) Ўзбекистон Республикасининг 1998 йил 29 августда қабул қилинган «Ўзбекистон Республикасининг айрим "
             "қонун ҳужжатларига ўзгартишлар ва қўшимчалар киритиш тўғрисида»ги 681–I-сонли Қонунининг (Ўзбекистон "
             "Республикаси Олий Мажлисининг Ахборотномаси, 1998 йил, № 9, 181-модда) VI бўлими;",
             "2) Ўзбекистон Республикасининг 1997 йил 25 апрелда қабул қилинган «Айрим қонун ҳужжатларига ўзгартишлар ва "
             "қўшимчалар киритиш тўғрисида»ги 421–I-сонли Қонуни (Ўзбекистон Республикаси Олий Мажлисининг "
             "Ахборотномаси, 1997 йил, № 4–5, 126-модда) I бўлимининг 8-банди.", *TAIL]
    rec, reg, ctx, edges, nodes = _run(paras, date="2025-01-31")
    omni = "EXT/ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida"
    assert {e.dst for e in edges} == {omni + "#681-i/БЎЛ.6", omni + "#421-i/БЎЛ.1/П.8"}
    assert set(ctx.repealed_on) == {e.dst for e in edges}                      # the stubs, not the acts
    assert ctx.stubs[omni + "#681-i/БЎЛ.6"].attrs["repealed_on"] == "2025-01-31"


def test_an_unnamed_sub_unit_repeal_does_not_repeal_the_act():
    item = ("1) Ўзбекистон Республикасининг 1997 йил 24 апрелда қабул қилинган «Табиий монополиялар тўғрисида»ги "
            "398–I-сонли Қонунининг (Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, 1997 йил, № 4–5, "
            "106-модда) {} бўлими.")
    act = "EXT/tabiiy monopoliyalar toʼgʼrisida"
    for unit, dst, repealed in (("тегишли", act, {}), ("II", act + "/БЎЛ.2", {act + "/БЎЛ.2": "2023-02-14"})):
        paras = ["1-модда. Ўз кучини йўқотган деб топиш", "Қуйидагилар ўз кучини йўқотган деб топилсин:",
                 item.format(unit), *TAIL]
        rec, reg, ctx, edges, _ = _run(paras, date="2023-02-14")
        assert [e.dst for e in edges] == [dst] and ctx.repealed_on == repealed       # never the whole act
        assert ctx.stats["repeal_partial_act_no_repealed_on"] == (1 if unit == "тегишли" else 0)


def test_superscripted_article_ranges_expand():
    from graphrag.kg.extract_ref import _parse_nums
    assert _parse_nums("389¹ – 389⁵") == ["389.1", "389.2", "389.3", "389.4", "389.5"]
    assert _parse_nums("383 – 385¹") == ["383", "384", "385", "385.1"]
    assert _parse_nums("12–14") == ["12", "13", "14"]


def test_amending_titles_cited_with_several_numbers_get_editions_r12():
    t = "«Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартиш ва қўшимчалар киритиш, шунингдек айрим қонун ҳужжатларини ўз кучини йўқотган деб топиш тўғрисида»ги"
    paras = ["1-модда. Ўз кучини йўқотган деб топиш", "Қуйидагилар ўз кучини йўқотган деб топилсин:",
             f"1) Ўзбекистон Республикасининг 2013 йил 7 октябрда қабул қилинган {t} ЎРҚ–355-сонли Қонунининг "
             "(Ахборотномаси, 2013 йил, № 10, 263-модда) 13-моддаси;",
             f"2) Ўзбекистон Республикасининг 2014 йил 14 майда қабул қилинган {t} ЎРҚ–372-сонли Қонунининг "
             "(Ахборотномаси, 2014 йил, № 5, 130-модда) 22-моддаси.", *TAIL]
    rec, reg, ctx, edges, _ = _run(paras, date="2023-02-14")
    assert sorted(e.dst.split("#", 1)[1] for e in edges) == ["oʼrq-355/СТ.13", "oʼrq-372/СТ.22"]
