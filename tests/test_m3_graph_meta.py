"""Graph meta for naming: the law a speech discusses (agenda item) and the single act an amending item amends."""
from graphrag import config
from graphrag.dataset import opus
from graphrag.dataset.generate import amending_locus, foreign_names, wrong_act


class _E:
    def __init__(self, src, t, dst, attrs=None):
        self.src, self.type, self.dst, self.attrs = src, t, dst, attrs or {}


class _C:
    def __init__(self, cid, parent, start, text):
        self.chunk_id, self.parent, self.start, self.end, self.text = cid, parent, start, start + len(text), text


def _recs():
    N = lambda i, t: type("N", (), {"node_id": i, "title": t})()
    M = lambda sec, t: type("M", (), {"section": sec, "date": "2025-04-29", "title": t})()
    st = type("R", (), {"doc_id": "STENOGRAMMA-a", "meta": M("STENOGRAMMA", "Senatning 6-yalpi majlisi 29-30.04.25"),
                        "chunks": [_C("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#1", "STENOGRAMMA-a/ВОПР.1/ВЫСТ.2", 0, "x" * 50),
                                   _C("STENOGRAMMA-a/ВОПР.2/ВЫСТ.1#1", "STENOGRAMMA-a/ВОПР.2/ВЫСТ.1", 100, "y" * 50)],
                        "structure": [N("STENOGRAMMA-a", "Senatning 6-yalpi majlisi 29-30.04.25"),
                                      N("STENOGRAMMA-a/ВОПР.1", "banklarni sanatsiya qilish"),
                                      N("STENOGRAMMA-a/ВОПР.2", "Davlat mudofaa buyurtmasi toʼgʼrisida"),
                                      N("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2", "Erkin Gadoyev"),
                                      N("STENOGRAMMA-a/ВОПР.2/ВЫСТ.1", "Raislik qiluvchi")]})()
    q = type("R", (), {"doc_id": "QARORLAR-b", "meta": M("QARORLAR", "Айрим қонун ҳужжатларига ўзгартиш киритиш тўғрисида"),
                       "chunks": [_C("QARORLAR-b/П.1#1", "QARORLAR-b/П.1", 0, "11-модданинг бешинчи қисми чиқариб ташлансин"),
                                  _C("QARORLAR-b/П.2#1", "QARORLAR-b/П.2", 200, "икки жойда ўзгартирилсин")],
                       "structure": [N("QARORLAR-b", "Айрим қонун ҳужжатларига ўзгартиш киритиш тўғрисида")]})()
    law = type("R", (), {"doc_id": "QONUNLAR-c", "meta": M("QONUNLAR", "Banklarni sanatsiya qilish va tugatish toʻgʻrisida"),
                         "chunks": [], "structure": [N("QONUNLAR-c", "Banklarni sanatsiya qilish va tugatish toʻgʻrisida")]})()
    return [st, q, law]


def _edges():
    ev = lambda d, s, e, q="": {"evidence": {"doc_id": d, "start": s, "end": e, "quote": q}}
    ext = "EXT/ixtirolar foydali modellar va sanoat namunalar toʼgʼrisida"
    return [_E("STENOGRAMMA-a/ВОПР.1", "REFERS_TO", "QONUNLAR-c", {"relation": "agenda"}),
            _E("QARORLAR-b/П.1", "AMENDS", ext + "/СТ.11/Ч.5", ev("QARORLAR-b", 0, 10)),
            _E("QARORLAR-b/П.2", "AMENDS", ext + "/СТ.12", ev("QARORLAR-b", 200, 210)),
            _E("QARORLAR-b/П.2", "AMENDS", "EXT/reklama toʼgʼrisida/СТ.8", ev("QARORLAR-b", 205, 215)),
            _E("QARORLAR-b", "REFERS_TO", ext, {"relation": "act", **ev("QARORLAR-b", 0, 5,
               "«Ихтиролар, фойдали моделлар ва саноат намуналари тўғрисида»ги Қонунига")})]


def test_graph_meta_discussed_and_single_amended_target():
    from graphrag.dataset.sources import SourceIndex
    recs = _recs()
    of = opus.graph_meta(recs, _edges(), type("I", (), {"act": {r.doc_id: r.meta.title for r in recs}})())
    assert of("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#1") == {"discussed_act": "Banklarni sanatsiya qilish va tugatish toʻgʻrisida"}
    assert of("STENOGRAMMA-a/ВОПР.2/ВЫСТ.1#1") == {"discussed_act": "Davlat mudofaa buyurtmasi toʼgʼrisida"}   # structure
    assert of("QARORLAR-b/П.1#1") == {"amended_act": "Ихтиролар, фойдали моделлар ва саноат намуналари тўғрисида",
                                      "amended_unit": "ст. 11 ч. 5"}
    assert of("QARORLAR-b/П.2#1") == {}                                               # two targets: unset


def test_checks_accept_names_from_graph_meta_and_still_reject_others():
    fr = [{"act": "Айрим қонун ҳужжатларига ўзгартиш киритиш тўғрисида", "where": "x",
           "text": "11-модданинг бешинчи қисми чиқариб ташлансин", "amended": [],
           "amended_act": "Ихтиролар, фойдали моделлар ва саноат намуналари тўғрисида", "amended_unit": "ст. 11 ч. 5"}]
    q = "«Ихтиролар, фойдали моделлар ва саноат намуналари тўғрисида»ги Қонуннинг 11-моддасидан нима чиқарилди?"
    assert not amending_locus(q, fr) and not wrong_act(q, fr, "uz-Cyrl") and foreign_names(q, fr, "uz-Cyrl") == []
    no_meta = [{k: v for k, v in fr[0].items() if k not in ("amended_act", "amended_unit")}]
    assert amending_locus(q, no_meta) and wrong_act(q, no_meta, "uz-Cyrl")
    sp = [{"act": "Senatning 6-yalpi majlisi 29-30.04.25", "where": "x", "text": "Qonun loyihasi muhokama qilindi.",
           "discussed_act": "Banklarni sanatsiya qilish va tugatish toʻgʻrisida"}]
    assert not wrong_act("«Banklarni sanatsiya qilish va tugatish toʻgʻrisida»gi qonun muhokamasida nima dedi?", sp, "uz-Latn")
    assert wrong_act("«Davlat boji toʻgʻrisida»gi qonun muhokamasida nima dedi?", sp, "uz-Latn")


def test_reader_meta_carries_graph_meta():
    recs = _recs()
    g = lambda cid: {"discussed_act": "Banklarni sanatsiya qilish va tugatish toʻgʻrisida"}
    m = opus.reader_meta(recs, graph=g)("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#1")
    assert m["discussed_act"].startswith("Banklarni") and m["speaker"] == "Erkin Gadoyev"


def test_rubric_v3_and_instructions_v6_use_graph_meta():
    o = config.load("train")["dataset"]["opus"]
    r = config.path(o["reader"]["file"]).read_text(encoding="utf-8")
    g = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["reader"]["version"] == 3 and "discussed_act" in r and "amended_act" in r
    assert o["generator"]["version"] >= 6 and "`discussed_act`" in g and "`amended_act` + `amended_unit`" in g
