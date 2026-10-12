# tests/test_kg_r20.py -- M2 Task 7 fix round 1 (Ruling R20): cited_in_amendment relation, REGULATES
# adoption-line filter, AMENDS op consistency check, депутат rules, audit metric corrections
import pytest

from graphrag.kg import audit as A
from graphrag.kg import ent_rules as R
from graphrag.kg.acts import ActRegistry
from graphrag.kg.extract_ent import entity_id, extract_entities
from graphrag.kg.extract_ref import extract_refs
from graphrag.kg.schema import GraphEdge, GraphNode, Schema
from tests.test_kg_refs import _manual, _rec

AMEND = ["1-модда. Мақсад", "Мақсад матни.", "2-модда. Жиноят кодексига ўзгартишлар",
         "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",
         "1) 27-модданинг иккинчи қисми қуйидаги таҳрирда баён этилсин:",
         "«Жарима ушбу Кодекснинг 44-моддасида назарда тутилган тартибда тайинланади»;",
         "2) қуйидаги мазмундаги 74¹-модда билан тўлдирилсин:",
         "«74¹-модда. Жазони қисқартириш",
         "Суд қарори билан.",
         "Ушбу Кодекснинг 73-моддасининг тўртинчи қисмида кўрсатилган шахс.»;",
         "3) 248-модданинг тўртинчи қисмидаги «101-моддасида» деган сўзлар «47 ва 101-моддаларида» деган сўзлар "
         "билан алмаштирилсин.",
         "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]


def _amend_edges():
    rec = _rec(AMEND, "QONUNLAR-aaaa2001", "Жиноят кодексига ўзгартишлар киритиш тўғрисида")
    return rec, {e.dst.split("/", 2)[-1]: e for e in extract_refs(rec, ActRegistry.from_records([rec]))
                 if e.dst.startswith("EXT/jinoyat kodeks/")}


# --- (a) cited_in_amendment ---------------------------------------------------------------------------

def test_r20_instruction_heads_stay_amended():
    _, by = _amend_edges()
    for el in ("СТ.27/Ч.2", "СТ.74.1", "СТ.248/Ч.4"):
        assert by[el].attrs["relation"] == "amended", el


def test_r20_citation_in_quoted_new_text_is_cited_in_amendment():
    rec, by = _amend_edges()
    # inline quoted new text, a multi-paragraph quoted new article, quoted replaced / replacement words
    for el in ("СТ.44", "СТ.73/Ч.4", "СТ.101", "СТ.47"):
        assert by[el].attrs["relation"] == "cited_in_amendment", el
    e = by["СТ.73/Ч.4"]
    assert e.src == f"{rec.doc_id}/СТ.2/Ч.1/П.2"             # same source as the instruction it stands in
    assert e.attrs["evidence"]["quote"] == "73-моддасининг тўртинчи қисмида"


def test_r20_relation_vocabulary_in_schema():
    s = Schema.load()
    assert "cited_in_amendment" in s.ref_relations and "amended" in s.ref_relations
    nodes = {"D1/СТ.1": GraphNode("D1/СТ.1", "element", "статья", "", "D1", False),
             "D2": GraphNode("D2", "document", "law_text", "", "D2", False)}
    ev = {"doc_id": "D1", "start": 0, "end": 1, "quote": "x"}
    ok = GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": ev, "relation": "cited_in_amendment"})
    bad = GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": ev, "relation": "cited"})
    assert s.validate_edge(ok, nodes) == []
    assert any("ref_relations" in m for m in s.validate_edge(bad, nodes))


def test_r20_rule_relation_uses_the_source_node_quote():
    rec, by = _amend_edges()
    nodes = {n.node_id: GraphNode(n.node_id, "element", n.type, n.title or "", rec.doc_id, False) for n in rec.structure}
    g = A.AuditGraph.from_records([rec], nodes)
    for e in by.values():
        assert A.rule_relation(g, e) == e.attrs["relation"]
    # the paragraph-scoped heuristic of the first report misses the multi-paragraph new article
    e = by["СТ.73/Ч.4"]
    t = rec.text
    pre = t[t.rfind("\n", 0, e.attrs["evidence"]["start"]) + 1:e.attrs["evidence"]["start"]]
    assert pre.count("«") == pre.count("»") and A.in_node_quote(g, e)


# --- (b) adoption / approval / signature lines --------------------------------------------------------

def _pos(t, s):
    i = t.index(s)
    return i, i + len(s)


def test_r20_adoption_line_rules():
    t = ("Бу муаммоларни бартараф этиш мақсадида Қонун Ички ишлар вазирлиги томонидан ишлаб чиқилди ва "
         "Қонунчилик палатаси томонидан қабул қилинди.")
    assert R.adoption_line(t, *_pos(t, "Қонунчилик палатаси")) == "final_agent"
    assert R.adoption_line(t, *_pos(t, "Ички ишлар вазирлиги")) == "final_agent"
    t = "Qonun Ichki ishlar vazirligi tomonidan ishlab chiqildi va Senat tomonidan maʼqullandi."
    assert R.adoption_line(t, *_pos(t, "Senat")) == "final_agent"
    t = "Принят Законодательной палатой 10 сентября 2024 года\nОдобрен Сенатом 5 октября 2024 года"
    assert R.adoption_line(t, *_pos(t, "Законодательной палатой")) == "header"
    assert R.adoption_line(t, *_pos(t, "Сенатом")) == "header"
    t = "Закон, принят Законодательной палатой и одобрен Сенатом, направляется Президенту для подписания в течение десяти дней."
    assert R.adoption_line(t, *_pos(t, "Законодательной палатой")) == "ru_participle"
    # norms about the procedure are regulation, not history
    for t, s in (("Қонун Сенат томонидан олтмиш кун ичида кўриб чиқилади ва маъқулланган тақдирда Президентга юборилади.", "Сенат"),
                 ("Бош вазир номзоди Қонунчилик палатаси томонидан маъқулланганидан кейин лавозимга тайинланади.", "Қонунчилик палатаси"),
                 ("Непроведение оценки не препятствует принятию Центральным банком решения о санации.", "Центральным банком"),
                 ("Решение, принятое Центральным банком, вступает в силу.", "Центральным банком")):
        assert R.adoption_line(t, *_pos(t, s)) is None, t


def test_r20_adoption_line_drops_regulates_edge_keeps_vertex():
    x = _manual("XULOSALAR-eeee2001", "committee_conclusion", "XULOSALAR",
                "Мудофаа масалалари қўмитаси\n\nБу муаммоларни бартараф этиш мақсадида Қонун Ички ишлар вазирлиги томонидан "
                "ишлаб чиқилди ва Қонунчилик палатаси томонидан қабул қилинди.\n\nСенат ушбу масалани кўриб чиқади.",
                [("БЛ.1", "блок", "1", "", 29, 260)])
    st = {}
    nodes, edges = extract_entities([x], stats=st)
    qp, sen = entity_id("organ", "qonunchilik palata"), entity_id("organ", "senat")
    assert not any(e.dst == qp for e in edges)
    assert any(e.dst == sen for e in edges)
    assert qp in {n.node_id for n in nodes}                   # a fixed dictionary name keeps its vertex
    assert [r["rule"] for r in st["r20"]["adoption_line_edges"] if r["dst"] == qp] == ["final_agent"]


# --- (c) AMENDS op consistency ---------------------------------------------------------------------------

def _am(quote, op, ops=None, quotes=None, n=[0]):
    n[0] += 1
    ev = {"doc_id": "D", "start": 0, "end": len(quote), "quote": quote}
    attrs = {"evidence": ev, "op": op}
    if ops:
        attrs["ops"] = ops
        attrs["evidence_all"] = [{"doc_id": "D", "start": i, "end": i + 1, "quote": q} for i, q in enumerate(quotes)]
    return GraphEdge("D/СТ.1", f"EXT/x/СТ.{n[0]}", "AMENDS", attrs)


def test_r20_expected_op_masks_quotes_and_takes_the_final_verb():
    assert A.expected_op("иккинчи қисми қуйидаги таҳрирда баён этилсин") == ("edit", "uz_last")
    assert A.expected_op("«чиқариб ташлансин» деган сўзлар «чиқарилсин» деган сўз билан алмаштирилсин") == ("replace", "uz_last")
    assert A.expected_op("«тўлдирилсин» деган сўздан кейин «X» деган сўз билан тўлдирилсин") == ("add", "uz_last")
    assert A.expected_op("учинчи қисми тўртинчи қисм деб ҳисоблансин") == ("renumber", "uz_last")
    assert A.expected_op("статью 5 дополнить частью третьей") == ("add", "ru_first")
    assert A.expected_op("ўн иккинчи қисми") == (None, "no_verb")


def test_r20_op_consistency_flags_contradictions():
    edges = [_am("5-модда чиқарилсин", "delete"), _am("5-модда чиқарилсин", "add"),
             _am("x", "add", ["add", "delete"], ["6-банд билан тўлдирилсин", "«ҳақли» деган сўз чиқариб ташлансин"]),
             _am("x", "add", ["add", "delete"], ["6-банд билан тўлдирилсин", "«ҳақли» деган сўз билан алмаштирилсин"])]
    r = A.op_consistency(edges)
    assert r["edges"] == 4 and r["pairs"] == 6 and r["contradictions"] == 2 and r["contradicting_edges"] == 2
    assert {(x["op"], x["expected"]) for x in r["contradiction_list"]} == {("add", "delete"), ("delete", "replace")}


# --- (d) депутат --------------------------------------------------------------------------------------

def test_r20_deputat_rules():
    law = _rec(["1-модда. Мақсад", "Олий Мажлиси Қонунчилик палатаси депутатлари қонунчилик ташаббуси ҳуқуқига эга.",
                "2-модда. Мандат", "Депутатлик мандати муддати тугагунга қадар сақланади.",
                "3-модда. Маҳаллий депутатлар", "Халқ депутати ўз сайловчилари олдида ҳисобот беради.",
                "4-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
               "QONUNLAR-eeee2002", "Депутат мақоми тўғрисида")
    st = {}
    nodes, edges = extract_entities([law], stats=st)
    dep, qpd = entity_id("position", "deputat"), entity_id("position", "qonunchilik palata deputat")
    by_art = {}
    for e in edges:
        by_art.setdefault(e.src.split("/")[1], set()).add(e.dst)
    assert qpd in by_art["СТ.1"] and dep not in by_art["СТ.1"]
    assert entity_id("organ", "qonunchilik palata") not in by_art["СТ.1"]   # the chamber is the post's owner
    (e,) = [e for e in edges if e.dst == qpd]
    assert e.attrs["evidence"]["quote"] == "Олий Мажлиси Қонунчилик палатаси депутатлари"
    assert dep not in by_art.get("СТ.2", set())                              # депутатлик = deputyship
    assert dep in by_art["СТ.3"]                                             # the generic post stays
    assert [x["name"] for x in st["r20"]["deputat"]["excluded"]] == ["Депутатлик"]


def test_r20_deputat_russian_owner_after():
    t = "Статья 1. Цель\n\nДепутаты Законодательной палаты Олий Мажлиса вносят законопроекты и запросы в установленном порядке."
    x = _manual("QONUNLAR-eeee2003", "law_text", "QONUNLAR", t, [("СТ.1", "статья", "1", "Цель", 0, len(t))],
                "О статусе депутата")
    nodes, edges = extract_entities([x])
    qpd = entity_id("position", "qonunchilik palata deputat")
    (e,) = [e for e in edges if e.dst == qpd]
    assert e.attrs["evidence"]["quote"] == "Депутаты Законодательной палаты Олий Мажлиса"
    assert not any(e.dst == entity_id("position", "deputat") for e in edges)


# --- (e) audit metrics --------------------------------------------------------------------------------------

def test_r20_pooled_n_eff_is_capped_at_the_real_sample():
    st = {"a": {"N": 1000, "n": 40, "k": 40}, "b": {"N": 1000, "n": 40, "k": 40}, "c": {"N": 10, "n": 10, "k": 5}}
    r = A.pooled(st)
    assert r["n_eff"] <= 90
    lo, hi = A.wilson(r["precision"] * 90, 90)
    assert r["ci"] == pytest.approx([lo, hi]) or r["n_eff"] < 90


def test_r20_corrected_is_flagged_non_informative():
    items = [{"uid": "g1", "stratum": "DEFINES", "planted": False}, {"uid": "p1", "stratum": "DEFINES", "planted": True}]
    res = A.evaluate(items, {"g1": {"correct": True}, "p1": {"correct": False}}, {"DEFINES": 10},
                     {"g1": {"correct": True}})
    assert res["precision_corrected_status"].startswith("non-informative")
    assert res["types"]["DEFINES"]["corrected_informative"] is False


def test_r20_rescore_restratifies_and_scores_labels():
    rec, by = _amend_edges()
    nodes = {n.node_id: GraphNode(n.node_id, "element", n.type, n.title or "", rec.doc_id, False) for n in rec.structure}
    g = A.AuditGraph.from_records([rec], nodes)
    old = [GraphEdge(e.src, e.dst, e.type, {**e.attrs, "relation": "amended"}) for e in by.values()]
    items = [{"uid": f"u{i}", "stratum": "REFERS_TO:amended", "planted": False, "edge": e} for i, e in enumerate(old)]
    verdicts = {it["uid"]: {"correct": True} for it in items}
    gone = GraphEdge("X/СТ.1", "ENT/organ/y", "REGULATES", {"evidence": {"doc_id": "X", "start": 0, "end": 1, "quote": "x"}})
    items.append({"uid": "r1", "stratum": "REGULATES:entity", "planted": False, "edge": gone})
    verdicts["r1"] = {"correct": False}
    cur = list(by.values())
    # a current edge whose label contradicts the rule (the head labelled cited_in_amendment)
    h = by["СТ.27/Ч.2"]
    cur = [e for e in cur if e is not h] + [GraphEdge(h.src, h.dst, h.type, {**h.attrs, "relation": "cited_in_amendment"})]
    pops = A.populations(cur)
    r = A.rescore(items, verdicts, {"r1": {"category": "true_error"}}, g, cur, pops)
    assert r["status"] == {"relabelled": 5, "removed": 1, "same_stratum": 2}
    assert r["removed"][0]["uid"] == "r1" and r["removed"][0]["target_correct"] is False
    c = r["strata"]["REFERS_TO:cited_in_amendment"]
    assert c["n"] == 5 and c["k_target"] == 5 and c["k_label"] == 4
    assert r["types"]["REFERS_TO"]["precision_target"] == 1.0 and r["types"]["REFERS_TO"]["precision_label"] < 1.0
