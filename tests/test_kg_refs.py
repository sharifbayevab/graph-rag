# tests/test_kg_refs.py
from graphrag.data.extract import Para
from graphrag.data.structure import parse_structure
from graphrag.data.chunker import chunk_document
from graphrag.kg.acts import ActRegistry
from graphrag.kg.extract_ref import extract_refs
from graphrag.types import DocRecord, Meta, Source
from tests.helpers import words


def _rec(paras, doc_id="QONUNLAR-aaaa0001", title="Банкларни санация қилиш тўғрисида", date="2024-01-19"):
    dt, nodes = parse_structure(doc_id, "QONUNLAR", "law_text", [Para(t) for t in paras], title, 384, words)
    ch = chunk_document(dt, nodes, 384, words, str.lower)
    return DocRecord(doc_id, Meta("law_text", date, "senate", ["uz-Cyrl"], "принят", "QONUNLAR", title),
                     Source("docs", f"docs/QONUNLAR/{doc_id}.docx", None, []), dt.text, nodes, ch)


LAW = ["1-модда. Мақсад", "Мақсад матни.", "2-модда. Тушунчалар",
       "Ушбу Қонуннинг 1-моддасида назарда тутилган мақсадлар учун.",
       "3-модда. Жиноят кодексига ўзгартиш",
       "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартишлар киритилсин:",
       "1) 27-модданинг иккинчи қисми қуйидаги таҳрирда баён этилсин:",
       "«Жарима…»;",
       "4-модда. «Банклар ва банк фаолияти тўғрисида»ги Қонуннинг 5-моддасига мувофиқ амалга оширилади."]


def test_internal_self_reference_resolves_to_own_article():
    rec = _rec(LAW)
    edges = extract_refs(rec, ActRegistry.from_records([rec]))
    own = [e for e in edges if e.dst == f"{rec.doc_id}/СТ.1"]
    assert own and own[0].src.startswith(f"{rec.doc_id}/СТ.2")
    ev = own[0].attrs["evidence"]
    assert rec.text[ev["start"]:ev["end"]] == ev["quote"]


def test_unqualified_article_in_amendment_context_targets_amended_code():
    rec = _rec(LAW)
    edges = extract_refs(rec, ActRegistry.from_records([rec]))
    tgt = {e.dst for e in edges}
    assert any(d.startswith("EXT/") and d.endswith("/СТ.27/Ч.2") for d in tgt)
    assert f"{rec.doc_id}/СТ.27" not in tgt


def test_quoted_external_law_with_article():
    rec = _rec(LAW)
    edges = extract_refs(rec, ActRegistry.from_records([rec]))
    assert any(e.dst.startswith("EXT/") and e.dst.endswith("/СТ.5") for e in edges)


# --- Task 4: extraction details, Ruling R7, kg/llm.py ---------------------------------------------

import json
from types import SimpleNamespace

import pytest

from graphrag.kg.acts import resolve_record
from graphrag.kg.extract_ref import (PASS, RefContext, amendment_context, extract_all, structure_nodes)
from graphrag.kg.llm import KgBudgetExceeded, KgCacheMiss, KgLLM
from graphrag.kg.schema import Schema
from graphrag.types import Node

OMNI = "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида"


def _manual(doc_id, typ, section, text, spans, title="", date="2025-05-20"):
    """A record with a hand-made structure: spans = [(node_id suffix, type, number, title, start, end)]."""
    nodes = [Node(doc_id, "документ", title, None, None, 0, len(text))]
    for suf, t, num, ttl, a, b in spans:
        parent = doc_id + ("/" + suf.rsplit("/", 1)[0] if "/" in suf else "")
        nodes.append(Node(f"{doc_id}/{suf}", t, ttl, num, parent, a, b))
    return DocRecord(doc_id, Meta(typ, date, "senate", ["uz-Cyrl"], "принят", section, title),
                     Source("docs", f"docs/{section}/{doc_id}.docx", None, []), text, nodes, [])


def _omni_law(doc_id, date, subject):
    return _rec(["1-модда. Жиноят кодексига", f"{subject} Ўзбекистон Республикасининг Жиноят кодексига "
                 "қуйидаги ўзгартиш киритилсин:", "1) 5-модда чиқарилсин;", "2-модда. Кучга кириш",
                 "Ушбу Қонун расмий эълон қилинган кундан кучга киради."], doc_id, OMNI, date)


def _validate_all(recs, reg, edges, ctx):
    nodes = {}
    for r in recs:
        nodes.update({n.node_id: n for n in structure_nodes(r)})
    nodes.update({n.node_id: n for n in reg.nodes() + ctx.stub_nodes()})
    sch = Schema.load()
    return [m for e in edges + ctx.contains_edges() for m in sch.validate_edge(e, nodes)]


def test_every_edge_has_exact_evidence_deepest_source_and_validates():
    rec = _rec(LAW)
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    edges = extract_refs(rec, reg, ctx)
    assert edges and _validate_all([rec], reg, edges, ctx) == []
    di = ctx.docs[rec.doc_id]
    for e in edges:
        ev = e.attrs["evidence"]
        assert rec.text[ev["start"]:ev["end"]] == ev["quote"]
        assert e.src == di.node_at(ev["start"], ev["end"]).node_id
        assert e.attrs["relation"] in ("internal", "amended", "element", "act")


def test_external_element_stub_chain_is_contained_in_the_act():
    rec = _rec(LAW)
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    extract_refs(rec, reg, ctx)
    stub = next(n for n in ctx.stub_nodes() if n.node_id.endswith("/СТ.27/Ч.2"))
    art = stub.node_id.rsplit("/", 1)[0]
    act = art.rsplit("/", 1)[0]
    assert stub.kind == "element" and stub.external and stub.type == "часть"
    assert ctx.stubs[art].type == "статья" and act == "EXT/jinoyat kodeks"
    assert {(e.src, e.dst) for e in ctx.contains_edges()} >= {(act, art), (art, stub.node_id)}


def test_amendment_context_marks_the_article_and_its_descendants():
    rec = _rec(LAW)
    ctxm = amendment_context(rec, ActRegistry.from_records([rec]))
    assert {k for k in ctxm} == {f"{rec.doc_id}/СТ.3/Ч.1", f"{rec.doc_id}/СТ.3/Ч.1/П.1"}
    assert all(v.node_id == "EXT/jinoyat kodeks" for v in ctxm.values())


def test_gazette_citations_and_own_headings_are_not_references():
    paras = ["1-модда. Ўзгартиш",
             "Ўзбекистон Республикасининг Жиноят кодекси (Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, "
             "1995 йил, № 1, 3-модда; 2024 йил, № 5, 460-модда) қуйидаги мазмундаги 74¹-модда билан тўлдирилсин:",
             "«74¹-модда. Жазони қисқартириш", "Ушбу Кодекс 73-моддасининг тўртинчи қисмида кўрсатилган шахс.",
             "Ушбу модданинг биринчи қисми қўлланилмайди».", "2-модда. Ижро", "Ижро этилсин.",
             "3-модда. Кучга кириш", "Қонун кучга киради."]
    rec = _rec(paras)
    reg = ActRegistry.from_records([rec])
    tgt = {e.dst for e in extract_refs(rec, reg)}
    assert "EXT/jinoyat kodeks/СТ.74.1" in tgt                      # the instruction
    assert "EXT/jinoyat kodeks/СТ.73/Ч.4" in tgt                    # "Ушбу Кодекс" inside the quoted new text
    assert "EXT/jinoyat kodeks/СТ.74.1/Ч.1" in tgt                  # "Ушбу модданинг" -> the quoted new article
    assert not any(d.endswith(("/СТ.3", "/СТ.460")) for d in tgt)   # gazette items
    assert not any(d.startswith(rec.doc_id) for d in tgt)          # no own heading / self references


def test_self_links_dropped_sibling_parts_kept():
    paras = ["1-модда. Мақсад", "Биринчи қисм.", "Ушбу модданинг биринчи қисмида назарда тутилган ҳолатлар.",
             "2-модда. Иккинчи", "Матн.", "3-модда. Учинчи", "Матн."]
    rec = _rec(paras)
    reg = ActRegistry.from_records([rec])
    edges = extract_refs(rec, reg)
    assert [(e.src, e.dst) for e in edges] == [(f"{rec.doc_id}/СТ.1/Ч.2", f"{rec.doc_id}/СТ.1/Ч.1")]
    ctx = RefContext.from_records([rec], reg)
    rec2 = _rec(["1-модда. Мақсад", "Ушбу модданинг биринчи қисми.", "2-модда. Б", "Матн.", "3-модда. В", "Матн."])
    ctx = RefContext.from_records([rec2], ActRegistry.from_records([rec2]))
    assert extract_refs(rec2, ctx.registry, ctx) == [] and ctx.stats["dropped_self_link"] == 1


def test_corpus_act_element_real_node_under_chapter_or_unresolved():
    code = _rec(["1-боб. Умумий қоидалар", "1-модда. Мақсад", "Матн.", "2-модда. Тушунча", "Биринчи.", "Иккинчи.",
                 "3-модда. Кучга кириш", "Матн."], "QONUNLAR-bbbb0001", "Банклар ва банк фаолияти тўғрисида",
                "2019-11-05")
    cite = _rec(["1-модда. Мақсад", "«Банклар ва банк фаолияти тўғрисида»ги Қонуннинг 2-моддаси иккинчи қисмига "
                 "ва 9-моддасига мувофиқ.", "2-модда. Б", "Матн.", "3-модда. В", "Матн."])
    reg = ActRegistry.from_records([code, cite])
    ctx = RefContext.from_records([code, cite], reg)
    edges = extract_refs(cite, reg, ctx)
    tgt = {e.dst: e.attrs["relation"] for e in edges}
    assert tgt.get("QONUNLAR-bbbb0001/ГЛ.1/СТ.2/Ч.2") == "element"
    assert tgt.get("QONUNLAR-bbbb0001") == "act"                   # СТ.9 does not exist: act edge kept
    assert not any("/СТ.9" in d for d in tgt)
    assert [u["article"] for u in ctx.unresolved if u["reason"] == "corpus_missing_article"] == ["9"]


def test_r7_undated_omnibus_mention_never_takes_the_nearest_corpus_act():
    a = _omni_law("QONUNLAR-cccc0001", "2025-05-06", "Жиноий жавобгарлик либераллаштирилади.")
    b = _omni_law("QONUNLAR-cccc0002", "2025-04-15", "Солиқ маъмурчилиги соддалаштирилади.")
    text = (f"Navbatdagi masala “{OMNI}”gi Qonun haqida. Soʼz qoʼmita raisiga.\n\n"
            "Ushbu qonun bilan jinoiy javobgarlik liberallashtirilmoqda.")
    st = _manual("STENOGRAMMA-dddd0001", "stenogram", "STENOGRAMMA", text,
                 [("ВОПР.1", "вопрос", "1", OMNI, 0, len(text)), ("ВОПР.1/ВЫСТ.1", "выступление", "1", "Raislik qiluvchi", 0, len(text))],
                 "Senat yalpi majlisi", "2025-05-20")
    reg = ActRegistry.from_records([a, b, st])
    (m, ref), = [x for x in resolve_record(st, reg) if x[0].kind == "act_title"]
    assert ref.doc_id is None and ref.ambiguous and set(ref.candidates) == {a.doc_id, b.doc_id}
    edges, ctx = extract_all([a, b, st], reg)                       # no LLM: queued, unlinked
    assert not [e for e in edges if e.src.startswith(st.doc_id)]
    assert len(ctx.queue) == 1 and ctx.queue[0]["reason"] == "omnibus_undated"

    class Pick:                                                     # a verified LLM answer picks the law
        def cached_json(self, pass_name, inp, schema, verify, prompt):
            assert pass_name == PASS and len(inp["candidates"]) == 2
            n = next(c["n"] for c in inp["candidates"] if "Жиноий" in c["preamble"])
            ans = {"choice": n, "evidence": "jinoiy javobgarlik liberallashtirilmoqda"}
            return ans if verify(ans) is True else None
    edges, ctx = extract_all([a, b, st], ActRegistry.from_records([a, b, st]), llm=Pick())
    got = {(e.src, e.dst, e.attrs["relation"]) for e in edges if e.src.startswith(st.doc_id)}
    assert (f"{st.doc_id}/ВОПР.1", a.doc_id, "agenda") in got
    assert (f"{st.doc_id}/ВОПР.1/ВЫСТ.1", a.doc_id, "act") in got


def test_ref_resolve_verify_rejects_unsupported_answers():
    a = _omni_law("QONUNLAR-cccc0001", "2025-05-06", "Жиноий.")
    text = f"Navbatdagi masala “{OMNI}”gi Qonun haqida."
    st = _manual("STENOGRAMMA-dddd0001", "stenogram", "STENOGRAMMA", text,
                 [("ВОПР.1", "вопрос", "1", OMNI, 0, len(text))], "S", "2025-05-20")
    seen = []

    class Bad:
        def cached_json(self, pass_name, inp, schema, verify, prompt):
            for ans in ({"choice": 0, "evidence": "not in the text at all"}, {"choice": 3, "evidence": "Navbatdagi"},
                        {"choice": 0, "evidence": ""}):
                seen.append(verify(ans))
            return None
    edges, ctx = extract_all([a, st], ActRegistry.from_records([a, st]), llm=Bad())
    assert [s[0] for s in seen] == [False, False, False]
    assert not edges or all(not e.src.startswith(st.doc_id) for e in edges)


def test_consistency_requires_a_compatible_date():
    old = _omni_law("QONUNLAR-cccc0003", "2023-01-31", "Эски.")
    text = (f"“{OMNI}”ги Қонуни ҳақида\n\nҚонунчилик палатаси томонидан 2023 йил 31 январда қабул қилинган “{OMNI}”ги "
            "Қонунини кўриб чиқиб")
    q = _manual("QARORLAR-eeee0001", "resolution", "QARORLAR", text, [], OMNI, "2025-06-01")
    reg = ActRegistry.from_records([old, q])
    refs = [r for _, r in resolve_record(q, reg)]
    assert refs[1].doc_id == old.doc_id                             # the dated mention
    assert refs[0].doc_id is None                                   # 2023 act, record of 2025: not followed


def test_qaror_title_and_agenda_links():
    law = _rec(["1-модда. Мақсад", "Матн.", "2-модда. Б", "Матн.", "3-модда. В", "Матн."], "QONUNLAR-ffff0001",
               "Геологик ўрганиш тўғрисида", "2025-03-01")
    text = "Сенатининг ҚАРОРИ\n\n“Геологик ўрганиш тўғрисида”ги Ўзбекистон Республикаси Қонуни ҳақида\n\n1. Маъқуллансин."
    q = _manual("QARORLAR-ffff0002", "resolution", "QARORLAR", text, [("П.1", "пункт", "1", "", text.index("1."), len(text))],
                "Геологик ўрганиш тўғрисида", "2025-03-20")
    reg = ActRegistry.from_records([law, q])
    edges = extract_refs(q, reg, RefContext.from_records([law, q], reg))
    title = [e for e in edges if e.attrs["relation"] == "title"]
    assert [(e.src, e.dst) for e in title] == [(q.doc_id, law.doc_id)]
    ev = title[0].attrs["evidence"]
    assert q.text[ev["start"]:ev["end"]] == ev["quote"] and "Геологик" in ev["quote"]


# kg/llm.py -----------------------------------------------------------------------------------------

class _FakeOpenAI:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


def _cfg(max_calls=2, total=10):
    return {"llm": {"model": "gpt-4.1", "max_calls_total": total,
                    "passes": {"ref_resolve": {"prompt_version": 1, "max_calls": max_calls},
                               "audit": {"prompt_version": 1, "max_calls": 5}}}}


SCH = {"type": "object", "properties": {"choice": {"type": ["integer", "null"]}}, "required": ["choice"],
       "additionalProperties": False}


def test_llm_cache_gate_budget_and_verify(tmp_path):
    fake = _FakeOpenAI({"choice": 1})
    on = KgLLM(_cfg(), tmp_path, client=fake, allow_api=True)
    assert on.cached_json("ref_resolve", {"q": 1}, SCH, lambda a: a["choice"] == 1, prompt="P") == {"choice": 1}
    assert on.cached_json("ref_resolve", {"q": 1}, SCH, lambda a: True, prompt="P") == {"choice": 1}
    assert fake.calls == 1 and on.stats["ref_resolve"]["hits"] == 1 and on.stats["ref_resolve"]["api_calls"] == 1
    files = list((tmp_path / "ref_resolve").glob("*.json"))
    assert len(files) == 1 and not list(tmp_path.rglob("*.tmp"))
    rec = json.loads(files[0].read_text())
    assert rec["model"] == "gpt-4.1" and rec["prompt_version"] == 1 and len(rec["prompt_digest"]) == 64
    # verify_fn on cached answers too; a rejected answer is counted and returned as None
    assert on.cached_json("ref_resolve", {"q": 1}, SCH, lambda a: (False, "why"), prompt="P") is None
    assert on.stats["ref_resolve"]["rejected"] == 1 and on.stats["ref_resolve"]["rejected_why"] == 1
    # offline: a hit works, a miss raises; a changed prompt is a miss
    off = KgLLM(_cfg(), tmp_path, client=fake, allow_api=False)
    assert off.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P") == {"choice": 1}
    with pytest.raises(KgCacheMiss):
        off.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P changed")
    assert off.stats["ref_resolve"]["offline_miss"] == 1 and fake.calls == 1
    # per-pass budget, then the total cap
    on.cached_json("ref_resolve", {"q": 2}, SCH, prompt="P")
    with pytest.raises(KgBudgetExceeded):
        on.cached_json("ref_resolve", {"q": 3}, SCH, prompt="P")
    tight = KgLLM(_cfg(max_calls=5, total=1), tmp_path / "t", client=fake, allow_api=True)
    tight.cached_json("audit", {"q": 1}, SCH, prompt="P")
    with pytest.raises(KgBudgetExceeded):
        tight.cached_json("ref_resolve", {"q": 9}, SCH, prompt="P")


def test_llm_env_gate_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv("GRL_KG_API", raising=False)
    assert KgLLM(_cfg(), tmp_path).allow_api is False
    monkeypatch.setenv("GRL_KG_API", "1")
    assert KgLLM(_cfg(), tmp_path).allow_api is True


def test_llm_ledger_counts_api_calls(tmp_path):
    fake = _FakeOpenAI({"choice": 1})
    on = KgLLM(_cfg(), tmp_path, client=fake, allow_api=True)
    on.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P")
    on.cached_json("ref_resolve", {"q": 1}, SCH, prompt="P")        # a hit: not counted
    on.cached_json("audit", {"q": 2}, SCH, prompt="P")
    led = json.loads((tmp_path / "ledger.json").read_text())
    assert sum(led["ref_resolve"].values()) == 1 and sum(led["audit"].values()) == 1


# --- fix round 1 (Ruling R8) --------------------------------------------------------------------

def _steno(items, date="2025-05-20", doc_id="STENOGRAMMA-99990001"):
    """A stenogram of agenda items [(title, speech text)]: one вопрос + one выступление each."""
    text, spans, pos = "", [], 0
    for k, (title, speech) in enumerate(items, 1):
        seg = speech + "\n\n"
        spans.append((f"ВОПР.{k}", "вопрос", str(k), title, pos, pos + len(speech)))
        spans.append((f"ВОПР.{k}/ВЫСТ.1", "выступление", "1", "Raislik qiluvchi", pos, pos + len(speech)))
        text += seg
        pos += len(seg)
    return _manual(doc_id, "stenogram", "STENOGRAMMA", text.rstrip("\n"), spans, "Senat yalpi majlisi", date)


def _dst_of(edges, rec, phrase):
    i = rec.text.index(phrase)
    return sorted({e.dst for e in edges for ev in (e.attrs.get("evidence_all") or [e.attrs["evidence"]])
                   if i <= ev["start"] < i + len(phrase)})


def test_r8_non_act_agenda_items_get_no_act_edge():
    st = _steno([("Oʼzbekiston Respublikasi Oliy Majlisi Senati Kengashining qarorlarini tasdiqlash toʼgʼrisida",
                  "Navbatdagi masala Oʼzbekiston Respublikasi Oliy Majlisi Senati Kengashining qarorlarini tasdiqlash toʼgʼrisida."),
                 ("Oʼzbekiston Respublikasi Oliy Sudi tarkibiga oʼzgartirishlar kiritish haqida",
                  "Navbatdagi masala Oʼzbekiston Respublikasi Oliy Sudi tarkibiga oʼzgartirishlar kiritish haqida."),
                 ("Energiyani tejash, undan oqilona foydalanish va energiya samaradorligini oshirish toʼgʼrisida",
                  "Navbatdagi masala “Energiyani tejash, undan oqilona foydalanish va energiya samaradorligini oshirish toʼgʼrisida”."),
                 ("Nobank kredit tashkilotlari va moliyalashtirish faoliyati toʼgʼrisida",
                  "Navbatdagi qonun Nobank kredit tashkilotlari va moliyalashtirish faoliyati toʼgʼrisida. Qonun qoʼmitada koʼrildi.")])
    reg = ActRegistry.from_records([st])
    agenda = {e.src.rsplit("/", 1)[-1]: e.dst for e in extract_refs(st, reg) if e.attrs["relation"] == "agenda"}
    assert set(agenda) == {"ВОПР.3", "ВОПР.4"}
    assert agenda["ВОПР.3"].startswith("EXT/energiyani tejash")


def test_r8_owner_noun_and_subject_precede_carry():
    law = "“Temir yoʼl transporti toʼgʼrisida”gi Qonun"
    speech = (f"Navbatdagi masala {law} haqida.\n\n"
              "Oʼzbekiston Respublikasi Konstitutsiyasining 19-moddasini taʼminlash maqsadida bepul yurish huquqi "
              "belgilanmoqda (21-modda).\n\n"
              "Mehnat kodeksining 5-moddasi va konvensiyaning 12-moddasini hisobga olish kerak.\n\n"
              "Maʼmuriy tartib-taomillar toʼgʼrisidagi kodeksning emas, muhokama qilinayotgan qonunning 43-moddasida "
              "rasmiy ogohlantirish va 48-moddasida hisobga qoʼyish belgilangan.")
    st = _steno([("Temir yoʼl transporti toʼgʼrisida", speech)])
    reg = ActRegistry.from_records([st])
    edges = extract_refs(st, reg)
    subj = "EXT/temir yoʼl transport toʼgʼrisida"
    assert _dst_of(edges, st, "(21-modda)") == [subj + "/СТ.21"]
    assert _dst_of(edges, st, "konvensiyaning 12-moddasini") == []
    assert _dst_of(edges, st, "qonunning 43-moddasida") == [subj + "/СТ.43"]
    assert _dst_of(edges, st, "va 48-moddasida") == [subj + "/СТ.48"]


def test_r8_title_skips_inquiry_and_prefers_outer_amending_title():
    text = ("Сенатининг Қ А Р О Р И\n\nПАРЛАМЕНТ СЎРОВИ\n\n“Банклар ва банк фаолияти тўғрисида”ги Қонун ижроси юзасидан"
            "\n\nкўриб чиқиб Қ А Р О Р Қ И Л А Д И :")
    q = _manual("QARORLAR-99990002", "resolution", "QARORLAR", text, [], "Банклар ва банк фаолияти тўғрисида")
    reg = ActRegistry.from_records([q])
    assert not [e for e in extract_refs(q, reg) if e.attrs["relation"] == "title"]
    text = ("Сенатининг Қ А Р О Р И\n\n“Ижро иши бўйича кафолат маблағларини киритиш институти\n\nжорий этилиши муносабати "
            "билан “Суд ҳужжатлари\n\nва бошқа органлар ҳужжатларини ижро этиш тўғрисида”ги\n\nЎзбекистон Республикаси "
            "Қонунига қўшимча ва ўзгартиришлар киритиш ҳақида”ги Ўзбекистон Республикаси Қонуни тўғрисида\n\nкўриб чиқиб")
    q = _manual("QARORLAR-99990003", "resolution", "QARORLAR", text, [], "Суд ҳужжатлари ва бошқа органлар ҳужжатларини ижро этиш тўғрисида")
    reg = ActRegistry.from_records([q])
    (t,) = [e for e in extract_refs(q, reg) if e.attrs["relation"] == "title"]
    assert "kafolat" in t.dst and "oʼzgartirish kiritish" in t.dst


def test_r8_amendment_head_element_is_amended_and_unresolved_is_distinct():
    rec = _rec(["1-модда. Ўзгартиш", "Ўзбекистон Республикасининг «Бола ҳуқуқларининг кафолатлари тўғрисида»ги Қонуннинг 14-моддаси қуйидаги "
                "мазмундаги иккинчи қисм билан тўлдирилсин:", "«Матн».", "2-модда. Б", "Матн.", "3-модда. В", "Матн."])
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    edges = extract_refs(rec, reg, ctx)
    (e,) = [e for e in edges if e.dst.endswith("/СТ.14")]
    assert e.attrs["relation"] == "amended"
    a = _omni_law("QONUNLAR-cccc0001", "2025-05-06", "Жиноий.")
    q = _manual("QARORLAR-99990004", "resolution", "QARORLAR", f"“{OMNI}”ги Қонуни ҳақида\n\nкўриб чиқиб", [], OMNI, "2025-05-20")
    reg = ActRegistry.from_records([a, q])
    ctx = RefContext.from_records([a, q], reg)
    extract_refs(q, reg, ctx)                                       # title + mention loop settle the same span
    keys = [(u["doc_id"], u.get("start"), u["reason"]) for u in ctx.unresolved]
    assert len(keys) == len(set(keys)) and keys


def test_r8_local_rules_recover_bare_articles():
    law = "“Maʼmuriy tartib-taomillar toʼgʼrisida”gi Qonun"
    speech = (f"Navbatdagi masala {law} haqida.\n\n"
              "Oʼzbekiston Respublikasining xalqaro shartnomalari toʼgʼrisidagi qonunning 11-moddasiga binoan ekspertiza.\n\n"
              "Maʼmuriy javobgarlik toʼgʼrisidagi kodeksning yangi kiritilayotgan 34¹-moddasiga taʼsir choralari.\n\n"
              "Maʼmuriy javobgarlik toʼgʼrisidagi kodeksning 4 ta moddasiga, xususan, 84-moddasiga oʼzgartish.\n\n"
              "Asosiy qonunimizning 57-moddasida belgilangan.\n\n"
              "Jinoyat kodeksida talon-toroj qilish (167-modda) uchun javobgarlik.")
    ext = _rec(["1-модда. Мақсад", "«Халқаро шартномалари тўғрисида»ги Қонун.", "2-модда. Б", "Матн.", "3-модда. В", "Матн."],
               "QONUNLAR-99990005", "Бошқа", "2020-01-01")
    st = _steno([("Maʼmuriy tartib-taomillar toʼgʼrisida", speech)])
    reg = ActRegistry.from_records([ext, st])
    ctx = RefContext.from_records([ext, st], reg)
    extract_refs(ext, reg, ctx)                                     # the treaty law is a quoted EXT act elsewhere
    edges = extract_refs(st, reg, ctx)
    assert _dst_of(edges, st, "11-moddasiga") == ["EXT/xalqaro shartnomalar toʼgʼrisida/СТ.11"]
    assert _dst_of(edges, st, "34¹-moddasiga") == ["EXT/maʼmuriy javobgarlik kodeks/СТ.34.1"]
    assert _dst_of(edges, st, "84-moddasiga") == ["EXT/maʼmuriy javobgarlik kodeks/СТ.84"]
    assert _dst_of(edges, st, "57-moddasida") == ["EXT/konstitusiya/СТ.57"]
    assert _dst_of(edges, st, "167-modda") == ["EXT/jinoyat kodeks/СТ.167"]
    rules = {r for *_, r in ctx.recovered}
    assert {"unquoted_title", "loose_gap_mention", "basic_law", "paren_sentence_act"} <= rules


def test_r8_section_heading_constitution_sentence_and_russian_act_after():
    text = ("Қонун лойиҳаси юзасидан\n\nII. Ер кодексида:\n\nбўш ерларни ҳисобга олиш (9-модда), ижара.\n\n"
            "Конституцияда белгиланган ҳуқуқ, жумладан 45-моддада ҳимоя қилинади.\n\n"
            "срок, предусмотренный статьей 150 Гражданского кодекса.")
    x = _manual("XULOSALAR-99990006", "committee_conclusion", "XULOSALAR", text, [], "Лойиҳа")
    reg = ActRegistry.from_records([x])
    edges = extract_refs(x, reg)
    assert _dst_of(edges, x, "9-модда") == ["EXT/er kodeks/СТ.9"]
    assert _dst_of(edges, x, "45-моддада") == ["EXT/konstitusiya/СТ.45"]
    assert _dst_of(edges, x, "статьей 150") == ["EXT/fuqarolik kodeks/СТ.150"]


# --- fix round 2 ----------------------------------------------------------------------------------

def test_r8b_omnibus_variants_with_repeated_words_are_generic():
    from graphrag.kg.acts import act_key, is_generic
    for t in ("Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish oʼzgartirish kiritish toʼgʼrisida",
              "Oʼzbekiston Respublikasining ayrim qonun hujjatlariga qoʼshimcha oʼzgartirish kiritish toʼgʼrisida",
              "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартишлар ва ўзгартиришлар киритиш тўғрисида"):
        assert is_generic(act_key(t)), t
    assert not is_generic(act_key("Ўзбекистон Республикасининг айрим қонун ҳужжатларига валюта муомаласи билан боғлиқ "
                                  "ўзгартиришлар киритиш тўғрисида"))
    st = _steno([("x", "Navbatdagi masala “Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish oʼzgartirish "
                       "kiritish toʼgʼrisida”gi Qonun haqida.")], date="2022-05-01")
    reg = ActRegistry.from_records([st])
    edges = extract_refs(st, reg)
    assert not any(e.dst.startswith("EXT/ayrim") for e in edges)       # never an EXT stub merging omnibus laws


def test_r8b_extraction_is_independent_of_record_order():
    a = _rec(["1-модда. Мақсад", "«Баҳолаш фаолияти тўғрисида»ги Қонун қўлланилади.", "2-модда. Б", "Матн.", "3-модда. В",
              "Матн."], "QONUNLAR-99990007", "Бошқа", "2020-01-01")
    st = _steno([("Z", "Amaldagi baholash faoliyati toʼgʼrisidagi qonunning 18-moddasida belgilangan.")])
    recs = [a, st]
    batch = extract_all(recs, ActRegistry.from_records(recs))[0]
    reg = ActRegistry.from_records(recs)
    alone = extract_refs(st, reg, RefContext.from_records(recs, reg))  # st first, before a is extracted
    as_json = lambda es: sorted(json.dumps(e.to_json(), sort_keys=True, ensure_ascii=False) for e in es)
    assert as_json(alone) == as_json([e for e in batch if e.src.startswith(st.doc_id)])
    assert any(e.dst == "EXT/baholash faoliyat toʼgʼrisida/СТ.18" for e in alone)


def test_r8b_inquiry_guard_only_in_heading_and_whole_word():
    law = _rec(["1-модда. Мақсад", "Матн.", "2-модда. Б", "Матн.", "3-модда. В", "Матн."], "QONUNLAR-99990008",
               "Геологик ўрганиш тўғрисида", "2025-03-01")
    body = "Қўмита " + "сўровномасига кўра ва бошқа маълумотлар асосида қонун таҳлил қилинди, " * 6
    text = f"“Геологик ўрганиш тўғрисида”ги Ўзбекистон Республикаси Қонуни юзасидан\n\n{body}\n\nкелиб тушди"
    x = _manual("XULOSALAR-99990009", "committee_conclusion", "XULOSALAR", text, [], "Геологик ўрганиш тўғрисида",
                "2025-03-20")
    reg = ActRegistry.from_records([law, x])
    edges = extract_refs(x, reg, RefContext.from_records([law, x], reg))
    assert [(e.src, e.dst) for e in edges if e.attrs["relation"] == "title"] == [(x.doc_id, law.doc_id)]


# --- fix round 3 ----------------------------------------------------------------------------------

def test_r8c_omnibus_variant_resolves_like_the_canonical_title():
    from graphrag.kg.acts import GENERIC_KEY, act_key, omnibus_candidates
    a = _omni_law("QONUNLAR-cccc0001", "2025-05-06", "Жиноий.")
    b = _omni_law("QONUNLAR-cccc0002", "2025-04-15", "Солиқ.")
    reg = ActRegistry.from_records([a, b])
    variant = "Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish oʼzgartirish kiritish toʼgʼrisida"
    assert act_key(variant) == act_key(OMNI) == GENERIC_KEY
    assert act_key("Ўзбекистон Республикасининг айрим қонун ҳужжатларига валюта муомаласи билан боғлиқ "
                   "ўзгартиришлар киритиш тўғрисида") != GENERIC_KEY
    assert omnibus_candidates(reg, act_key(variant), "2025-05-20") == omnibus_candidates(reg, GENERIC_KEY, "2025-05-20") \
        == (a.doc_id, b.doc_id)
    st = _steno([(variant, f"Navbatdagi masala {variant}.")], date="2025-05-20")
    reg = ActRegistry.from_records([a, b, st])
    edges, ctx = extract_all([a, b, st], reg)                       # no LLM: queued with both candidates
    assert not any(e.dst.startswith("EXT/") for e in edges if e.src.startswith(st.doc_id))
    assert [set(q["candidates"]) for q in ctx.queue] == [{a.doc_id, b.doc_id}]


def test_r8c_agenda_omnibus_without_feasible_act_is_an_unresolved_row():
    st = _steno([(OMNI, f"Navbatdagi masala {OMNI}.")], date="2022-05-01")
    reg = ActRegistry.from_records([st])
    ctx = RefContext.from_records([st], reg)
    assert extract_refs(st, reg, ctx) == []
    assert [u["reason"] for u in ctx.unresolved] == ["omnibus_no_feasible_act"]


def test_r8c_inquiry_in_a_long_heading_title_paragraph_and_inflected_forms():
    from graphrag.kg.extract_ref import _pats
    p = _pats()
    for w in ("сўрови", "сўровни", "сўровида", "soʼrovni", "soʼrovida", "soʻrovining"):
        assert p["inquiry"].search(f"парламент {w} юзасидан"), w
    assert not p["inquiry"].search("сўровномасига кўра")
    head = ("Ўзбекистон Республикаси Олий Мажлиси Сенатининг ҚАРОРИ Вазирлар Маҳкамасига қишлоқ хўжалигини ривожлантириш "
            "масалалари юзасидан парламент сўровини юбориш ҳақида " + "ва бошқа масалалар " * 12)
    text = head + "\n\n“Олий Мажлиснинг Сенати тўғрисида”ги Конституциявий Қонуннинг 12-моддасига мувофиқ\n\nкўриб чиқиб"
    q = _manual("QARORLAR-99990010", "resolution", "QARORLAR", text, [], "Олий Мажлиснинг Сенати тўғрисида")
    reg = ActRegistry.from_records([q])
    ctx = RefContext.from_records([q], reg)
    assert not [e for e in extract_refs(q, reg, ctx) if e.attrs["relation"] == "title"]
    assert ctx.stats["title_skipped_inquiry"] == 1
