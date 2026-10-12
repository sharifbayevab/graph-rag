# tests/test_kg_entities.py -- M2 Task 6: concepts (DEFINES), entities (dictionary + verified LLM), REGULATES
import json
from types import SimpleNamespace

from graphrag import config
from graphrag.kg.concepts import concept_mentions, extract_concepts, merge_concepts, norm_key
from graphrag.kg.extract_ent import build, dictionary_matches, entity_id, extract_entities, llm_units, verify_entity
from graphrag.kg.llm import KgLLM
from graphrag.kg.schema import Schema
from tests.test_kg_refs import _manual, _rec

DEFS = ["1-модда. Мақсад", "Ушбу Қонуннинг мақсади суғурта муносабатларини тартибга солишдан иборат.",
        "2-модда. Асосий тушунчалар",
        "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",
        "суғурталанган шахс – суғурта ҳодисаларидан ҳимояланган жисмоний шахс;",
        "суғурта стажи – суғурта бадалларини тўлаш амалга оширилган давр;",
        "транспорт воситаларининг тўхтаб туриш жойлари (бундан буён матнда тўхтаб туриш жойлари деб юритилади) – "
        "транспорт воситаларини сақлаш учун мўлжалланган жойлар.",
        "3-модда. Вазирлар Маҳкамасининг ваколатлари",
        "Вазирлар Маҳкамаси суғурталанган шахсларнинг ҳуқуқларини ҳимоя қилиш чораларини кўради.",
        "4-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]

OTHER = ["1-модда. Мақсад", "Суғурталанган шахсга нафақа Адлия вазирлиги томонидан белгиланган тартибда тўланади.",
         "2-модда. Тушунчалар",
         "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",
         "суғурталанган шахс – суғурта шартномаси бўйича ҳимояланган шахс;",
         "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]


def _law(paras=DEFS, doc_id="QONUNLAR-eeee0001", title="Давлат ижтимоий суғуртаси тўғрисида"):
    return _rec(paras, doc_id=doc_id, title=title)


def test_three_concepts_from_asosiy_tushunchalar_article():
    rec = _law()
    nodes, edges = extract_concepts(rec)
    titles = [n.title for n in nodes]
    assert sorted(titles) == sorted(["суғурталанган шахс", "суғурта стажи",
                                     "транспорт воситаларининг тўхтаб туриш жойлари"])
    by = {n.title: n for n in nodes}
    assert by["суғурта стажи"].attrs["definition"] == "суғурта бадалларини тўлаш амалга оширилган давр"
    assert by["транспорт воситаларининг тўхтаб туриш жойлари"].attrs["aliases"] == ["тўхтаб туриш жойлари"]
    assert all(e.type == "DEFINES" for e in edges) and len(edges) == 3
    for e in edges:
        ev = e.attrs["evidence"]
        assert rec.text[ev["start"]:ev["end"]] == ev["quote"]
        assert e.src.startswith(f"{rec.doc_id}/СТ.2")
    ev = next(e for e in edges if e.dst == by["суғурта стажи"].node_id).attrs["evidence"]
    assert ev["quote"] == "суғурта стажи – суғурта бадалларини тўлаш амалга оширилган давр;"


def test_same_term_from_two_laws_is_one_vertex_with_two_defines():
    a, b = _law(), _law(OTHER, "QONUNLAR-eeee0002", "Суғурта тўғрисида")
    na, ea = extract_concepts(a)
    nb, eb = extract_concepts(b)
    merged = merge_concepts(na + nb)
    c = [n for n in merged if n.title == "суғурталанган шахс"]
    assert len(c) == 1 and c[0].node_id == "CON/" + norm_key("суғурталанган шахс")
    assert [d["doc_id"] for d in c[0].attrs["definitions"]] == [a.doc_id, b.doc_id]
    assert c[0].attrs["definition"] == "суғурта ҳодисаларидан ҳимояланган жисмоний шахс"
    assert len([e for e in ea + eb if e.dst == c[0].node_id]) == 2


def test_norm_key_merges_scripts_and_inflections():
    assert norm_key("Вазирлар Маҳкамасининг") == norm_key("Vazirlar Mahkamasi") == "vazir mahkama"


def test_dictionary_vazirlar_mahkamasi_cyrillic_and_latin_is_one_vertex():
    law = _law()
    st = _manual("STENOGRAMMA-eeee0003", "stenogram", "STENOGRAMMA",
                 "Soʼz Vazirlar Mahkamasi vakiliga. Oʼzbekiston Respublikasi Vazirlar Mahkamasining qarori.",
                 [("ВОПР.1", "вопрос", "1", "Masala", 0, 20), ("ВОПР.1/ВЫСТ.1", "выступление", "1", "Raislik qiluvchi", 0, 20)])
    nodes, edges = extract_entities([law, st])
    vm = [n for n in nodes if n.node_id == entity_id("organ", "vazir mahkama")]
    assert len(vm) == 1 and vm[0].title == "Вазирлар Маҳкамаси" and vm[0].attrs["docs"] == 2
    assert not [n for n in nodes if n.type == "organ" and n.node_id != vm[0].node_id and "mahkama" in n.node_id]
    # REGULATES only from norms: the law's article 3, never the stenogram
    reg = [e for e in edges if e.dst == vm[0].node_id]
    assert reg and all(e.src.startswith(law.doc_id + "/СТ.3") for e in reg)
    assert not [e for e in edges if e.src.startswith(st.doc_id)]


def test_dictionary_head_rules():
    text = ("Ўзбекистон Республикаси Иқтисодиёт ва молия вазирлиги, Mening savolim Adliya vazirligiga. "
            "Soʼz Budjet va iqtisodiy islohotlar masalalari qoʼmitasi raisiga. Олий Мажлис Сенатининг Раиси.")
    got = {(m["type"], m["name"]) for m in dictionary_matches(text)}
    assert ("organ", "Иқтисодиёт ва молия вазирлиги") in got
    assert ("organ", "Adliya vazirligiga") in got
    assert ("organ", "Budjet va iqtisodiy islohotlar masalalari qoʼmitasi") in got
    assert ("position", "Budjet va iqtisodiy islohotlar masalalari qoʼmitasi raisiga") in got
    assert ("organ", "Олий Мажлис Сенатининг") in got


class _FakeOpenAI:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        inp = json.loads(kw["messages"][1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.fn(inp))))])


def _cfg():
    g = config.load("graph")
    g["llm"]["passes"]["entities"]["max_calls"] = 5
    return g


def _answer(inp):
    if "items" in inp:                                   # entity_verify pass (fix round 1): keep all
        return {"items": [{"id": x["id"], "keep": True, "type": x["type"], "reason": x["name"]} for x in inp["items"]]}
    out = []
    for u in inp["units"]:
        ents = []
        if "Адлия вазирлиги" in u["text"]:
            ents += [{"type": "organ", "name": "Адлия вазирлиги",                       # ok
                      "evidence": "нафақа  Адлия ВАЗИРЛИГИ томонидан"},                  # case / space tolerant
                     {"type": "legal_category", "name": "нафақа",
                      "evidence": "нафақа Молия вазирлиги томонидан"},                  # evidence not in the text
                     {"type": "procedure", "name": "тўлов тартиби",
                      "evidence": "белгиланган тартибда тўланади"},                    # name not in evidence
                     {"type": "person", "name": "Суғурталанган шахсга", "evidence": "Суғурталанган шахсга нафақа"}]
        out.append({"unit_id": u["unit_id"], "entities": ents})
    return {"units": out}


def test_llm_answer_is_verified_wrong_evidence_dropped(tmp_path):
    b = _law(OTHER, "QONUNLAR-eeee0002", "Суғурта тўғрисида")
    fake = _FakeOpenAI(_answer)
    llm = KgLLM(_cfg(), cache_root=tmp_path, client=fake, allow_api=True)
    stats = {}
    nodes, edges = extract_entities([b], llm, stats=stats)
    assert fake.calls == 2                                   # entities + entity_verify
    rej = stats["llm"]["rejected"]
    assert rej == {"bad_type": 1, "evidence_not_in_unit": 1, "name_not_in_evidence": 1}
    assert stats["llm"]["accepted"] == 1
    adl = entity_id("organ", norm_key("Адлия вазирлиги"))
    e = [e for e in edges if e.dst == adl]
    assert len(e) == 1 and e[0].src == f"{b.doc_id}/СТ.1/Ч.1"
    evs = e[0].attrs.get("evidence_all") or [e[0].attrs["evidence"]]
    # Ruling R29 B: the LLM mention's evidence is its name span (not the clause "нафақа Адлия вазирлиги томонидан"),
    # the same span as the dictionary mention -> one evidence
    assert {ev["quote"] for ev in evs} == {"Адлия вазирлиги"}
    assert all(b.text[ev["start"]:ev["end"]] == ev["quote"] for ev in evs)
    n = next(n for n in nodes if n.node_id == adl)
    assert n.attrs["sources"] == ["dictionary", "llm"]
    # offline re-run: answers from the cache, 0 calls, same result
    llm2 = KgLLM(_cfg(), cache_root=tmp_path, client=_FakeOpenAI(_answer), allow_api=False)
    nodes2, edges2 = extract_entities([b], llm2)
    assert [x.to_json() for x in edges2] == [x.to_json() for x in edges] and llm2.stats["entities"]["api_calls"] == 0


def test_verify_entity_rules():
    text = "Суғурталанган шахсга нафақа Адлия вазирлиги томонидан тўланади."
    types = Schema.load().entity_types
    assert verify_entity(text, {"type": "organ", "name": "адлия вазирлиги", "evidence": "Адлия  вазирлиги томонидан"}, types)[0]
    assert verify_entity(text, {"type": "organ", "name": "Adliya vazirligi", "evidence": "Adliya vazirligi"}, types)[1] \
        == "evidence_not_in_unit"
    assert verify_entity(text, {"type": "organ", "name": "Молия", "evidence": "Адлия вазирлиги"}, types)[1] \
        == "name_not_in_evidence"


def test_regulates_for_concept_mentioned_in_another_norm():
    rec = _law()
    nodes, defines = extract_concepts(rec)
    concepts = merge_concepts(nodes)
    edges, stats = concept_mentions([rec], concepts, defines)
    cid = "CON/" + norm_key("суғурталанган шахс")
    e = [e for e in edges if e.dst == cid]
    assert len(e) == 1 and e[0].src == f"{rec.doc_id}/СТ.3/Ч.1" and e[0].type == "REGULATES"
    assert e[0].attrs["evidence"]["quote"] == "суғурталанган шахсларнинг"
    assert not [x for x in edges if x.src.startswith(f"{rec.doc_id}/СТ.2")]    # the definitions part has DEFINES
    assert stats["too_generic"] == {}


def test_too_generic_concept_gets_no_mention_edges():
    rec = _law()
    g = config.load("graph")
    g["patterns"]["definitions"]["max_norms"] = 0
    import graphrag.kg.concepts as C
    with config.overridden("graph", g):
        C._P = None
        try:
            nodes, defines = extract_concepts(rec)
            edges, stats = concept_mentions([rec], merge_concepts(nodes), defines)
        finally:
            C._P = None
    assert edges == [] and "CON/" + norm_key("суғурталанган шахс") in stats["too_generic"]


def test_build_is_schema_valid_and_deterministic():
    recs = [_law(), _law(OTHER, "QONUNLAR-eeee0002", "Суғурта тўғрисида")]
    n1, e1, s1 = build(recs)
    n2, e2, s2 = build(list(reversed(recs)))
    assert s1["schema_errors"] == 0
    assert [x.to_json() for x in n1] == [x.to_json() for x in n2]
    assert [x.to_json() for x in e1] == [x.to_json() for x in e2]
    assert {e.type for e in e1} == {"DEFINES", "REGULATES"}


def test_llm_units_are_norms_only():
    law = _law()
    st = _manual("STENOGRAMMA-eeee0003", "stenogram", "STENOGRAMMA", "Soʼz Vazirlar Mahkamasi vakiliga.",
                 [("ВОПР.1", "вопрос", "1", "Masala", 0, 10)])
    units = llm_units([law, st])
    assert {u["doc_id"] for u in units} == {law.doc_id}
    assert [u["unit_id"] for u in units] == [f"{law.doc_id}/СТ.{i}" for i in range(1, 5)]


def test_llm_names_take_the_dictionary_key():
    from graphrag.kg.extract_ent import entity_key, llm_key
    for name, typ, key in [("Oliy Majlisining Senati", "organ", "senat"), ("Центральный банк", "organ", "markaziy bank"),
                           ("Олий Мажлиси Сенатининг Раиси", "position", "senat rais"),
                           ("Адлия вазирлиги", "organ", "adliya vazirlik")]:
        assert llm_key({"name": name, "type": typ, "key": entity_key(name)}) == key


# --- fix round 1 (Ruling R15) --------------------------------------------------------------------

def test_context_filter_gazette_act_title_and_author_heading():
    from graphrag.kg.extract_ent import _Context
    law = _law(["1-модда. Мақсад",
                "«Вазирлар Маҳкамаси тўғрисида»ги Қонун (Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, "
                "2003 йил, № 9, 12-модда) ва Адлия вазирлиги.",
                "2-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."])
    cx = _Context(law)
    t = law.text
    i = t.index("Вазирлар Маҳкамаси")
    assert cx.reason(i, i + 18) == "act_title"
    j = t.index("Олий Мажлисининг")
    assert cx.reason(j, j + 16) == "gazette"
    k = t.index("Адлия вазирлиги")
    assert cx.reason(k, k + 15) is None
    x = _manual("XULOSALAR-eeee0009", "committee_conclusion", "XULOSALAR",
                "Budjet masalalari qoʼmitasi\n\nX U L O S A Q I L A D I:\n\n1. Adliya vazirligi ijro etsin.",
                [("БЛ.1", "блок", "1", "", 49, 80)])
    cx = _Context(x)
    assert cx.reason(0, 10) == "author_heading" and cx.reason(52, 68) is None


def test_name_filter_and_head_retype():
    from graphrag.kg.extract_ent import head_type, name_reason
    assert name_reason("Tanzila Kamalovna, Ichki ishlar vazirligi vakili") == "patronymic"
    assert name_reason("мактабгача таълим стандарти ишлаб чиқилишини таъминлайди ҳамда уни тасдиқлайди") == "clause"   # R16
    assert name_reason("Adliya vazirligi") is None
    assert head_type(norm_key("юқори турувчи прокурор")) == "position"
    assert head_type(norm_key("Божхона қўмитаси")) == "organ"
    assert head_type(norm_key("юридик шахс")) is None


def test_ombudsman_keys_fold():
    from graphrag.kg.extract_ent import entity_key, llm_key
    k = "inson huquq boʼyicha vakil"
    assert entity_key("омбудсман") == k
    assert llm_key({"name": "Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари бўйича вакили",
                    "type": "position", "key": "x"}) == k


def test_reason_supported():
    from graphrag.kg.extract_ent import reason_supported
    snip = ["Суғурталанган шахсга нафақа Адлия вазирлиги томонидан тўланади."]
    assert reason_supported("It is a ministry: 'нафақа Адлия вазирлиги томонидан'", "X", snip)
    assert not reason_supported("'Адлия вазирлиги' is a body", "Адлия вазирлиги", snip)   # R16: the name alone never suffices
    assert not reason_supported("a ministry", "Адлия вазирлиги", snip)


def test_verify_pass_drops_and_retypes_only_with_supported_reasons(tmp_path):
    b = _law(OTHER, "QONUNLAR-eeee0002", "Суғурта тўғрисида")

    def ans(inp):
        if "items" in inp:
            out = []
            for x in inp["items"]:
                if x["name"] == "нафақа":                     # supported drop
                    out.append({"id": x["id"], "keep": False, "type": x["type"], "reason": "'Суғурталанган шахсга нафақа Адлия' - a payment"})
                else:                                         # unsupported retype: ignored
                    out.append({"id": x["id"], "keep": True, "type": "procedure", "reason": "looks like a process"})
            return {"items": out}
        return {"units": [{"unit_id": u["unit_id"], "entities":
                           [{"type": "legal_category", "name": "нафақа", "evidence": "нафақа Адлия вазирлиги"}]
                           if "нафақа" in u["text"] else []} for u in inp["units"]]}
    llm = KgLLM(_cfg(), cache_root=tmp_path, client=_FakeOpenAI(ans), allow_api=True)
    stats = {}
    nodes, edges = extract_entities([b], llm, stats=stats)
    ids = {n.node_id for n in nodes}
    assert entity_id("legal_category", norm_key("нафақа")) not in ids
    assert entity_id("organ", norm_key("Адлия вазирлиги")) in ids                 # unverified retype ignored
    assert stats["fix_round_1"]["9_verify"]["verdicts"] == {"drop": 1, "unverified": 1}


def test_isolated_head_rule_entity_dropped_and_concept_twin_mapped():
    law = _law()
    st = _manual("STENOGRAMMA-eeee0003", "stenogram", "STENOGRAMMA", "Soʼz Turizm qoʼmitasi vakiliga. Sizlarga senator.",
                 [("ВОПР.1", "вопрос", "1", "Masala", 0, 20)])
    cn, defines = extract_concepts(law)
    concepts = merge_concepts(cn)
    nodes, edges = extract_entities([law, st], concepts=concepts, defines=defines)
    ids = {n.node_id for n in nodes}
    assert not [i for i in ids if "turizm" in i]                                  # isolated head-rule: dropped
    assert entity_id("position", "senator") in ids                                # fixed dictionary name: kept
    # an LLM-like twin: an entity whose key is a concept key points at the concept
    from graphrag.kg import extract_ent as X
    assert "CON/" + norm_key("суғурталанган шахс") in {c.node_id for c in concepts}


def test_single_lemma_concept_mentions_only_in_own_or_citing_law():
    rec = _law(["1-модда. Тушунчалар", "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",
                "жамғарма – давлат мақсадли жамғармаси;",
                "2-модда. Жамғарма", "Жамғарма маблағлари бюджетдан ажратилади.",
                "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
               "QONUNLAR-eeee0005", "Жамғарма тўғрисида")
    other = _law(["1-модда. Мақсад", "Жамғарма маблағлари бошқа тартибда шакллантирилади.", "2-модда. Ижро",
                  "Ижро этилсин.", "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
                 "QONUNLAR-eeee0006", "Бошқа жамғарма тўғрисида")
    cn, defines = extract_concepts(rec)
    edges, stats = concept_mentions([rec, other], merge_concepts(cn), defines, cites={})
    assert {e.src.split("/")[0] for e in edges} == {rec.doc_id}
    assert stats["single_lemma_cross_law_dropped"] == 1


# --- fix round 2 (Ruling R16) --------------------------------------------------------------------

def test_r16_head_typing():
    from graphrag.kg.ent_rules import head_type
    k = norm_key
    assert head_type(k("давлат рўйхатидан ўтказиш"), "procedure") == "procedure"
    assert head_type(k("аккредитациядан ўтказиш"), "procedure") == "procedure"          # conditional head
    assert head_type(k("кўргазмали семинарлар ўтказиш"), "procedure") is None          # an activity
    assert head_type(k("инвестицияларни жалб қилиш"), "procedure") is None
    assert head_type(k("лицензия шартномаси"), "procedure") == "legal_category"         # a document
    assert head_type(k("ижтимоий ҳимоя органининг ходимлари"), "position") == "legal_category"
    assert head_type(k("Қонунчилик палатаси Спикери ўринбосарлари"), "position") == "position"
    assert head_type(k("манфаатдор ташкилотлар"), "organ") == "legal_category"
    assert head_type(k("бошқа орган"), "organ") is None
    assert head_type(k("махсус жамғармаси"), "organ") == "legal_category"               # a fund
    assert head_type(k("капитал воситаларининг баланс қиймати"), "legal_category") is None   # a quantity


def test_r16_coordinated_lists_and_clauses():
    from graphrag.kg.ent_rules import is_clause, split_coordinated
    n = "Қорақалпоғистон Республикаси Адлия вазирлиги, вилоятлар ва Тошкент шаҳар адлия бошқармалари"
    assert [n[a:b] for a, b in split_coordinated(n, "organ")] == \
        ["Қорақалпоғистон Республикаси Адлия вазирлиги", "Тошкент шаҳар адлия бошқармалари"]
    assert split_coordinated("Фуқаролик ишлари бўйича туманлараро, туман, шаҳар судлари", "organ") == []
    assert split_coordinated("Oliy taʼlim, fan va innovatsiyalar vazirligi", "organ") is None      # one ministry
    assert split_coordinated("Фан, таълим ва соғлиқни сақлаш масалалари қўмитаси раиси", "position") is None
    assert is_clause("давлат стандарти ишлаб чиқилишини таъминлайди ҳамда уни тасдиқлайди")
    assert not is_clause("давлат ижтимоий суғуртаси бўйича тўловлар")


def test_r16_regulates_context():
    from graphrag.kg.ent_rules import Context2
    t = ("Вазирлар Маҳкамаси ҳузуридаги Миграция агентлиги. «Молия вазирлиги» деган сўзлар алмаштирилсин. "
         "Олий Мажлисининг Қонунчилик палатаси томонидан 2025 йил 31 январда қабул қилинган Қонун. "
         "Президентининг 2025 йил 15 апрелдаги «Банклар тўғрисида»ги Фармони. Марказий банк курси бўйича.")
    c = Context2(t, "committee_conclusion")

    def why(s):
        i = t.index(s)
        return c.reason(i, i + len(s))
    assert why("Вазирлар Маҳкамаси") == "owner_only"
    assert why("Миграция агентлиги") is None
    assert why("Молия вазирлиги") == "quoted_words"
    assert why("Қонунчилик палатаси") == "boilerplate"
    assert why("Президентининг") == "act_owner"
    assert why("Марказий банк") == "modifier"


def test_r16_aliases_ombudsman_variants_and_nominative_titles():
    from collections import Counter
    from graphrag.kg.extract_ent import display_title, entity_key, dictionary_matches, match_key
    assert entity_key("Министерство экономики и финансов") == entity_key("Иқтисодиёт ва молия вазирлиги")
    t = "Bolalar ombudsmani va Biznes ombudsman"
    keys = {match_key(t, m) for m in dictionary_matches(t)}
    assert norm_key("Бола ҳуқуқлари бўйича вакил") in keys and "inson huquq boʼyicha vakil" not in keys
    assert display_title(Counter({"Бизнес омбудсманини": 2})) == "Бизнес омбудсмани"
    assert display_title(Counter({"нафақа": 1})) == "нафақа"                               # not a case suffix


def test_r16_verify_v2_rules(tmp_path):
    b = _law(OTHER, "QONUNLAR-eeee0002", "Суғурта тўғрисида")

    def ans(inp):
        if "items" in inp:
            out = []
            for x in inp["items"]:
                assert x["surface_forms"] and x["snippets"]
                if x["type"] == "legal_category":        # keep=false with another type = a retype; conflict -> drop
                    out.append({"id": x["id"], "keep": False, "type": "organ",
                                "reason": "the text says 'Суғурталанган шахсга нафақа Адлия'"})
                else:                                    # the name alone is no evidence
                    out.append({"id": x["id"], "keep": False, "type": x["type"], "reason": "'Адлия вазирлиги'"})
            return {"items": out}
        return {"units": [{"unit_id": u["unit_id"], "entities":
                           [{"type": "legal_category", "name": "нафақа", "evidence": "нафақа Адлия вазирлиги"}]
                           if "нафақа" in u["text"] else []} for u in inp["units"]]}
    llm = KgLLM(_cfg(), cache_root=tmp_path, client=_FakeOpenAI(ans), allow_api=True)
    stats = {}
    nodes, edges = extract_entities([b], llm, stats=stats)
    ids = {n.node_id for n in nodes}
    assert entity_id("organ", norm_key("Адлия вазирлиги")) in ids
    assert entity_id("legal_category", norm_key("нафақа")) in ids        # R17: a conflicting retype keeps the type
    assert stats["fix_round_1"]["9_verify"]["verdicts"] == {"retype_conflict_keep": 1, "unverified": 1}


def test_r16_law_specific_concept_scope():
    from graphrag.kg.concepts import law_specific
    rec = _law(["1-модда. Тушунчалар", "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",
                "ваколатли орган – Ўзбекистон Республикаси Президенти ҳузуридаги Ижтимоий ҳимоя миллий агентлиги;",
                "суғурталовчи – ваколатли органнинг жамғармаси;",
                "суғурта стажи – бадаллар тўланган давр.",
                "2-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
               "QONUNLAR-eeee0007", "Суғурта тўғрисида")
    cn, _ = extract_concepts(rec)
    spec = law_specific(merge_concepts(cn))
    assert "CON/" + norm_key("ваколатли орган") in spec and "CON/" + norm_key("суғурталовчи") in spec
    assert "CON/" + norm_key("суғурта стажи") not in spec


# --- fix round 3 (Ruling R17) --------------------------------------------------------------------

def test_r17_procedure_recall_heads():
    from graphrag.kg.ent_rules import head_type
    for n in ["Кассация шикоятини иш юритишга қабул қилиш", "жамиятдан чиқиш", "parlament soʼrovini yuborish",
              "медиация тартиб-таомилини амалга ошириш", "илмий натижаларни патентлаш",
              "тадбиркорлик субъектларида текшириш ўтказиш", "қонунчилик ташаббуси субъектларининг таклифларини кўриб чиқиш"]:
        assert head_type(norm_key(n), "procedure") == "procedure", n
    assert head_type(norm_key("кўргазмали семинарлар ўтказиш"), "procedure") is None


def test_r17_legal_category_rules():
    from graphrag.kg.ent_rules import head_type, list_like, distribute_head
    for n in ["ер участкасини бўш турган ер участкаси сифати", "иқтисодиёти тармоқлари", "мижозларнинг активларини ишончли бошқариш",
              "табиий газнинг улгуржи ва чакана савдоси", "мулкнинг бошқа шакллари", "atrof-muhit"]:
        assert head_type(norm_key(n), "legal_category") is None, n
    assert head_type(norm_key("Hukumat aʼzolari"), "legal_category") == "position"
    assert head_type(norm_key("Ўзбекистон фермерлари кенгаши"), "organ") == "legal_category"
    assert list_like("шуъба ва қарам хўжалик жамиятлари", "legal_category")
    assert not list_like("Жамиятдан чиқарилган ёки ундан чиқиб кетган жамият иштирокчиси", "legal_category")
    assert not list_like("потенциальные приобретатели акций, активов, прав и (или) обязательств", "legal_category")
    assert [x for x, _ in distribute_head("юридик ва жисмоний шахслар")] == ["юридик шахслар", "жисмоний шахслар"]
    assert distribute_head("кластер ва фермер хўжаликлари") is None


def test_r17_stem_aware_reason_masking():
    from graphrag.kg.extract_ent import reason_supported
    snip = ["Вазирлар Маҳкамасининг қарори билан тасдиқланган тартибда"]
    assert not reason_supported("'Вазирлар Маҳкамасининг' is named", "Вазирлар Маҳкамаси", snip)
    assert reason_supported("'қарори билан тасдиқланган тартибда' shows a rule", "Вазирлар Маҳкамаси", snip)


def test_r17_titles():
    from collections import Counter
    from graphrag.kg.extract_ent import display_title
    assert display_title(Counter({"Ҳукумати": 2})) == "Ҳукумат"
    assert display_title(Counter({"доверенности": 1})) == "доверенности"                 # never a Russian word
    assert display_title(Counter({"Президенти ҳузуридаги Хавфсизлик кенгаши котибияти": 1})) == "Хавфсизлик кенгаши котибияти"
    assert display_title(Counter({"Iqtisodiyot va moliya vaziri": 5, "Iqtisodiyot moliya vaziri": 1})) == "Iqtisodiyot va moliya vaziri"


# --- fix round 4 (Ruling R18) --------------------------------------------------------------------


def _dk(name, typ, lang=None):
    from graphrag.kg.ent_rules import dedup_key
    from graphrag.kg.extract_ent import entity_key
    return dedup_key(entity_key(name, lang), typ)


def test_r18_dedup_merges_variants():
    same = [("procedure", ["лицензиялаш", "лицензиялашни амалга ошириш"]),
            ("procedure", ["медиация", "медиация тартиб-таомилларини ўтказиш", "mediatsiya tartib-taomilini amalga oshirish"]),
            ("procedure", ["кассация тартиблари", "кассация"]),
            ("procedure", ["экспертиза", "экспертиза ўтказиш"]),
            ("procedure", ["давлат рўйхатидан ўтказиш", "давлат рўйхатидан ўтказилиши"]),
            ("procedure", ["банкни санация қилиш", "банкларни санация қилиш",
                           "Банкни санация қилиш тартиб-таомилларини амалга ошириш"]),
            ("procedure", ["жамиятдан чиқиш", "жамият иштирокчисининг жамиятдан чиқиши"]),
            ("organ", ["Korrupsiyaga qarshi kurashish agentligi", "Коррупцияга қарши курашиш агентлиги"]),
            ("legal_category", ["voyaga yetmagan shaxslar", "вояга етмаган шахслар"]),
            ("legal_category", ["tadbirkorlik subyektlari", "тадбиркорлик субъектлари"]),
            ("position", ["Қонунчилик палатаси Спикерининг ўринбосарлари", "Спикер ўринбосари"]),
            ("organ", ["xalq deputatlari mahalliy Kengashlari", "маҳаллий Кенгашлар"]),
            ("position", ["konsullik muassasasining mehnat migratsiyasi masalalari boʼyicha attashesi",
                          "Mehnat migratsiyasi masalalari boʼyicha attashe"]),
            ("position", ["Бола ҳуқуқлари бўйича вакили (Болалар омбудсмани)", "Бола ҳуқуқлари бўйича вакил"])]
    for typ, names in same:
        keys = {_dk(n, typ) for n in names}
        assert len(keys) == 1, (typ, names, keys)
    ru = [("procedure", "Ликвидация банка", "банкни тугатиш"), ("procedure", "санации банка", "банкни санация қилиш"),
          ("legal_category", "потенциальных приобретателей", "потенциал олувчи"),
          ("legal_category", "акционеров", "акциядорлар"), ("legal_category", "кредиторов", "кредиторлар"),
          ("organ", "Кабинету Министров Республики Узбекистан", "Вазирлар Маҳкамаси"),
          ("organ", "Агентством по гарантированию вкладов", "Омонатларни кафолатлаш агентлиги")]
    for typ, a, b in ru:
        assert _dk(a, typ, "ru") == _dk(b, typ, "uz"), (a, b)


def test_r18_dedup_must_not_merge():
    pairs = [("organ", "Қорақалпоғистон Республикаси Адлия вазирлиги", "Адлия вазирлиги"),     # regional vs national
             ("organ", "xalq deputatlari Kattaqoʼrgʼon shahar Kengashi", "xalq deputatlari Kattaqoʼrgʼon tuman Kengashi"),
             ("organ", "Давлат хавфсизлик хизмати", "Президенти давлат хавфсизлик хизмати"),
             # a Senate committee vs a Legislative Chamber committee
             ("organ", "Олий Мажлиси Сенатининг Суд-ҳуқуқ масалалари ва коррупцияга қарши курашиш қўмитаси",
              "korrupsiyaga qarshi kurashish va sud-huquq masalalari qoʼmitasi"),
             ("organ", "Agrar, suv xoʼjaligi masalalari va ekologiya qoʼmitasi", "Agrar va suv xoʼjaligi masalalari qoʼmitasi"),
             ("position", "Бош прокурор", "вилоят прокурори"),                                     # posts of different bodies
             ("position", "Қорақалпоғистон Республикаси адлия вазири", "Adliya vaziri"),
             ("position", "Олий судининг раиси", "Олий Мажлиси Сенатининг Раиси"),
             ("procedure", "банкни тугатиш", "тугатиш"),
             ("procedure", "мажбурий тиббий кўрикдан ўтказиш", "тиббий кўриклардан ўтказиш"),
             ("procedure", "давлат рўйхатидан ўтказиш", "давлат рўйхати"),
             ("legal_category", "юридик шахслар", "жисмоний шахслар")]
    for typ, a, b in pairs:
        assert _dk(a, typ) != _dk(b, typ), (a, b)


def test_r18_restore_rules_and_splits():
    from graphrag.kg import ent_rules as R
    assert R.restore_verify_drop(norm_key("депутат"), "position")
    assert R.restore_verify_drop(norm_key("Қонунчилик палатаси депутатлари"), "position")
    assert R.restore_verify_drop(norm_key("Давлат ўрмон хўжалиги органлари"), "organ")
    assert R.restore_verify_drop(norm_key("Фавқулодда вазиятлар вазирлигининг бўлинмалари"), "organ")
    assert R.restore_verify_drop(norm_key("республика ижро этувчи ҳокимият органлари"), "organ")
    assert not R.restore_verify_drop(norm_key("масъул давлат органлари"), "organ")
    assert not R.restore_verify_drop(norm_key("бош бухгалтер"), "legal_category")
    assert R.restore_head_drop(norm_key("Божхона ҳудуди")) == "legal_category"
    assert R.modifier_level_coordination("товар ва молия бозорлари иштирокчилари")
    assert not R.modifier_level_coordination("кластер ва фермер хўжаликлари")
    n = "юридик шахслар ҳамда якка тартибдаги тадбиркорлар"
    assert [n[a:b] for a, b in R.split_category_list(n)] == ["юридик шахслар", "якка тартибдаги тадбиркорлар"]
    assert R.split_category_list("Жамиятдан чиқарилган ёки ундан чиқиб кетган жамият иштирокчиси") is None
    own, rest, off = R.split_owner_post("Bolalar ombudsmanining hududiy davlat organlari")
    assert own == "Bolalar ombudsmanining" and rest == "hududiy davlat organlari"
    assert R.split_owner_post("Олий судининг Фуқаролик ишлари бўйича судлов ҳайъати") is None
    assert R.named_institution(norm_key("Davlat va huquq instituti"), "Davlat va huquq instituti")
    assert not R.named_institution(norm_key("илмий тадқиқот институтлари"), "илмий тадқиқот институтлари")
    assert not R.named_institution(norm_key("Фуқаролик жамияти институтлари"), "Фуқаролик жамияти институтлари")
    assert R.ru_nominative("потенциальных приобретателей") == "потенциальный приобретатель"
    assert R.ru_nominative("кредиторов") == "кредитор"
    assert R.ru_nominative("Бизнес омбудсмани") == "Бизнес омбудсмани"


def test_r18_list_intro_is_not_a_signature():
    law = _law(["1-модда. Мақсад", "Лицензиялашни амалга ошириш тартиби белгиланади.",
                "2-модда. Қонунчиликни ушбу Қонунга мувофиқлаштириш", "Ўзбекистон Республикаси Вазирлар Маҳкамаси:",
                "ҳукумат қарорларини ушбу Қонунга мувофиқлаштирсин;",
                "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
               "QONUNLAR-eeee0011", "Лицензиялаш тўғрисида")
    nodes, edges = extract_entities([law])
    vm = entity_id("organ", "vazir mahkama")
    assert any(e.dst == vm and e.src.startswith(f"{law.doc_id}/СТ.2") for e in edges)


def test_r18_concept_filters_legislation_subject_and_organ_name():
    rec = _law(["1-модда. Асосий тушунчалар", "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади:",
                "рақобат – хўжалик юритувчи субъектларнинг мусобақалашуви;",
                "йўл ҳаракати хавфсизлиги – йўл ҳаракати иштирокчиларининг ҳимояланганлик ҳолати;",
                "2-модда. Жавобгарлик", "Рақобат тўғрисидаги қонунчиликни бузганлик учун жавобгарлик белгиланади.",
                "3-модда. Хизмат", "Давлат йўл ҳаракати хавфсизлиги хизмати ваколатларни амалга оширади.",
                "4-модда. Мақсад", "Рақобат ривожлантирилади ва йўл ҳаракати хавфсизлиги таъминланади.",
                "5-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."],
               "QONUNLAR-eeee0012", "Рақобат тўғрисида")
    cn, defines = extract_concepts(rec)
    edges, stats = concept_mentions([rec], merge_concepts(cn), defines)
    srcs = {(e.src.split("/")[1], e.dst) for e in edges}
    rq, yh = "CON/" + norm_key("рақобат"), "CON/" + norm_key("йўл ҳаракати хавфсизлиги")
    assert ("СТ.4", rq) in srcs and ("СТ.4", yh) in srcs
    assert ("СТ.2", rq) not in srcs and ("СТ.3", yh) not in srcs
    assert stats["legislation_subject_dropped"] == 1 and stats["in_organ_name_dropped"] == 1
