import pytest

from graphrag.kg.acts import (ActRegistry, act_key, display_name, is_amending, is_generic,
                              scan_act_mentions)
from graphrag.kg.schema import Schema
from graphrag.types import DocRecord, Meta, Node, Source


def test_act_key_script_case_and_prefix_insensitive():
    variants = [
        "«Банкларни санация қилиш тўғрисида»ги Ўзбекистон Республикаси Қонуни",
        "“Banklarni sanatsiya qilish toʼgʼrisida”gi Qonun",
        "Банкларни санация қилиш тўғрисидаги Қонунга",
        "Ўзбекистон Республикасининг «Банкларни санация қилиш тўғрисида»ги Қонунининг",
    ]
    assert len({act_key(v) for v in variants}) == 1


def test_code_names_normalize():
    assert act_key("Жиноят кодексига") == act_key("Jinoyat kodeksi") == act_key("Жиноят кодексининг")
    assert act_key("Жиноят кодекси") != act_key("Жиноят-процессуал кодекси")


# Surface forms taken from the corpus records (mentions and document titles).
SAME = [
    ("Maʼmuriy javobgarlik toʼgʼrisidagi kodeksga", "Маъмурий жавобгарлик тўғрисидаги кодексига"),
    ("Maʼmuriy javobgarlik toʼgʼrisidagi kodeksining", "Ўзбекистон Республикасининг Маъмурий жавобгарлик тўғрисидаги кодексига"),
    ("Jinoyat kodeksining", "ЖИНОЯТ КОДЕКСИНИНГ"),
    ("Soliq kodeksiga", "Солиқ кодекси"),
    ("Jinoyat-protsessual kodeksiga", "Жиноят-процессуал кодексининг"),
    ("Jinoyat protsessual kodeksiga", "Жиноят-процессуал кодекси"),
    ("Fuqarolik-protsessual kodeksiga", "Фуқаролик процессуал кодексига"),
    ("Iqtisodiy-protsessual kodeksi", "Иқтисодий процессуал кодексига"),
    ("Maʼmuriy sud ishlarini yuritish toʼgʼrisidagi kodeksiga", "Маъмурий суд ишларини юритиш тўғрисидаги кодекси"),
    ("Yer kodeksiga", "Ер кодекси"),
    ("Budjet kodeksi", "Бюджет кодексининг"),
    ("Uy-joy kodeksida", "Уй-жой кодексининг"),
    ("Mehnat kodeksimizda", "Меҳнат кодексига"),
    ("Konstitutsiyasining", "Конституциясига"),
    ("Konstitutsiyada", "Конституция"),
    ("Davlat boji toʼgʼrisida", "Давлат божи тўғрисида"),
    ("Oʼzbekiston Respublikasining Davlat boji toʼgʼrisida", "«Давлат божи тўғрисида»ги Ўзбекистон Республикаси Қонунига"),
    ("Litsenziyalash, ruxsat berish va xabardor qilish tartib-taomillari toʼgʼrisida",
     "Лицензиялаш, рухсат бериш ва хабардор қилиш тартиб-таомиллари тўғрисида"),
    ("Banklar va bank faoliyati toʼgʼrisida", "Банклар ва банк фаолияти тўғрисида"),
    ("Aksiyadorlik jamiyatlari va aksiyadorlarning huquqlarini himoya qilish toʼgʼrisida",
     "Акциядорлик жамиятлари ва акциядорларнинг ҳуқуқларини ҳимоя қилиш тўғрисида"),
    ("Valutani tartibga solish toʼgʼrisida", "Valyutani tartibga solish toʼgʼrisida"),
    ("Xotin-qizlar va erkaklar uchun teng huquq hamda imkoniyatlar kafolatlari toʼgʼrisida",
     "Xotin-qizlar va erkaklar uchun teng huquq va imkoniyatlar kafolatlari toʼgʼrisida"),
    ("Oʼzbekiston Respublikasining xalqaro shartnomalari toʼgʼrisida", "Ўзбекистон Республикасининг халқаро шартномалари тўғрисида"),
    ("Ўзбекистон Республикаси Миллий гвардияси тўғрисида", "Milliy gvardiya toʼgʼrisida"),
    ("Oʼzbekiston Respublikasining referendumi toʼgʼrisida", "Oʼzbekiston Respublikasining referendum toʼgʼrisida"),
    ("Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари бўйича вакили (омбудсман) тўғрисида",
     "Oʼzbekiston Respublikasi Oliy Majlisining Inson huquqlari boʼyicha vakili toʼgʼrisida"),
    ("Олий таълимга оид малакаларни тан олиш тўғрисидаги глобал конвенцияни (Париж, 2019 йил 25 ноябрь) ратификация қилиш ҳақида",
     "Oliy taʼlimga oid malakalarni tan olish toʼgʼrisidagi global konventsiyani ratifikatsiya qilish haqida"),
    ("2023-yil uchun Oʼzbekiston Respublikasining Davlat budjeti toʼgʼrisida", "2023 йил учун Давлат бюджети тўғрисида"),
    ("Oʼzbekiston Respublikasining Jinoyat hamda Jinoyat-ijroiya kodekslariga oʼzgartish va qoʼshimchalar kiritish toʼgʼrisida",
     "Ўзбекистон Республикасининг Жиноят ҳамда Жиноят-ижроия кодексларига ўзгартиш ва қўшимчалар киритиш тўғрисида"),
    ("Oʼzbekiston Respublikasining Maʼmuriy javobgarlik toʼgʼrisidagi kodeksiga qoʼshimcha va oʼzgartirishlar kiritish haqida",
     "Ўзбекистон Республикасининг Маъмурий жавобгарлик тўғрисидаги кодексига қўшимча ва ўзгартиришлар киритиш ҳақида"),
    ("Aholiga ijtimoiy xizmatlar koʼrsatish tizimi yanada takomillashtirilishi munosabati bilan Oʼzbekiston Respublikasining "
     "ayrim qonun hujjatlariga qoʼshimcha va oʼzgartirishlar kiritish toʼgʼrisida",
     "Ўзбeкистон Рeспубликасининг ҚоНУНИ Аҳолига ижтимоий хизматлар кўрсатиш тизими янада такомиллаштирилиши муносабати "
     "билан Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартиришлар киритиш тўғрисида"),
    ("Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish va qoʼshimchalar kiritish toʼgʼrisida",
     "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартиришлар киритиш тўғрисида"),
    ("Davlat ijtimoiy sugʼurtasi toʼgʼrisida", "Давлат ижтимоий суғуртаси тўғрисида"),
    ("O'zbekiston Respublikasining Davlat boji to'g'risida", "Давлат божи тўғрисида"),     # ASCII apostrophes
    ("Respublikasining maʼmuriy javobgarlik toʼgʼrisidagi kodeksi", "Маъмурий жавобгарлик тўғрисидаги кодекси"),
]

DIFFERENT = [
    ("Жиноят кодекси", "Жиноят-процессуал кодекси"),
    ("Жиноят кодекси", "Жиноят-ижроия кодекси"),
    ("Фуқаролик кодекси", "Фуқаролик процессуал кодекси"),
    ("Маъмурий жавобгарлик тўғрисидаги кодекси", "Маъмурий суд ишларини юритиш тўғрисидаги кодекси"),
    ("Банкларни санация қилиш ва тугатиш тўғрисида", "«Банкларни санация қилиш тўғрисида»ги Қонун"),
    ("Davlat boji toʼgʼrisida", "Davlat budjeti toʼgʼrisida"),
    ("2023-yil uchun Davlat budjeti toʼgʼrisida", "2024-yil uchun Oʼzbekiston Respublikasining Davlat budjeti toʼgʼrisida"),
    ("Hayvonot dunyosini muhofaza qilish va undan foydalanish toʼgʼrisida",
     "Oʼsimlik dunyosini muhofaza qilish va undan foydalanish toʼgʼrisida"),
    ("Konstitutsiya", "Konstitutsiyaviy sud toʼgʼrisida"),
    ("Давлат божи тўғрисида", "«Давлат божи тўғрисида»ги Ўзбекистон Республикаси Қонунига қўшимча киритиш ҳақида"),
    ("Сув кодекси", "Солиқ кодекси"),
    ("Давлат ижтимоий суғуртаси тўғрисида", "Қишлоқ хўжалиги таваккалчиликларини суғурта қилиш тўғрисида"),
]


@pytest.mark.parametrize("a,b", SAME)
def test_act_key_same_act(a, b):
    assert act_key(a) == act_key(b) != ""


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_act_key_different_acts(a, b):
    assert act_key(a) != act_key(b)


def test_act_key_classifiers_and_display():
    omnibus = act_key("Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида")
    assert is_generic(omnibus) and is_amending(omnibus)
    assert is_amending(act_key("Ўзбекистон Республикасининг Жиноят кодексига ўзгартиш киритиш ҳақида"))
    assert not is_amending(act_key("Рақобат тўғрисида")) and not is_generic(act_key("Рақобат тўғрисида"))
    assert act_key("") == "" and act_key("   ") == ""
    assert display_name("Ўзбекистон Республикасининг Жиноят кодексига") == "Жиноят кодекси"
    assert display_name("Soliq kodeksining") == "Soliq kodeksi"
    assert display_name("Konstitutsiyasiga") == "Konstitutsiya"


def test_scan_act_mentions():
    text = ("Ўзбекистон Республикасининг 1993 йил 6 майда қабул қилинган «Сув ва сувдан фойдаланиш "
            "тўғрисида»ги 837–XII-сонли Қонуни ҳамда “Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi "
            "Qonuniga, Ўзбекистон Республикасининг Жиноят ҳамда Жиноят-ижроия кодексларига, "
            "Маъмурий жавобгарлик тўғрисидаги кодексининг 12-моддаси, Ushbu kodeks, Konstitutsiyaviy sud, "
            "Конституциясига мувофиқ")
    ms = scan_act_mentions(text)
    keys = [act_key(m.name) for m in ms]
    assert keys == [act_key("Сув ва сувдан фойдаланиш тўғрисида"), act_key("Davlat boji toʼgʼrisida"),
                    act_key("Жиноят кодекси"), act_key("Жиноят-ижроия кодекси"),
                    act_key("Маъмурий жавобгарлик тўғрисидаги кодекси"), act_key("Конституция")]
    assert [m.kind for m in ms] == ["act_title", "act_title", "code_list", "code_list", "code_name", "constitution"]
    assert all(text[m.start:m.end] for m in ms)


# --- ActRegistry on synthetic records --------------------------------------------------------------

def _rec(doc_id, title, date, body="", typ="law_text", articles=("Ушбу Қонуннинг мақсади",)):
    text = f"{title}\n\nҚонунчилик палатаси томонидан қабул қилинган\n\n{body}"
    structure = [Node(doc_id, "документ", title, None, None, 0, len(text))]
    for i, a in enumerate(articles, 1):
        structure.append(Node(f"{doc_id}/СТ.{i}", "статья", a, str(i), doc_id, 0, len(text)))
    return DocRecord(doc_id, Meta(typ, date, "legislative_chamber", ["uz-Cyrl"], "принят", "QONUNLAR", title),
                     Source("docs", f"{doc_id}.docx", None), text, structure, [])


def _registry():
    return ActRegistry.from_records([
        _rec("L-old", "Давлат ижтимоий суғуртаси тўғрисида", "2025-02-18"),
        _rec("L-new", "Давлат ижтимоий суғуртаси\n\nтўғрисида", "2025-10-14"),
        _rec("A-1", "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида", "2025-02-18"),
        _rec("A-2", "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартиришлар киритиш тўғрисида", "2025-07-01"),
        _rec("A-3", "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартиришлар киритиш тўғрисида", "2025-07-01"),
        _rec("W-1", "Ўзбекистон Республикасининг Сув кодексини тасдиқлаш тўғрисида", "2025-01-31"),
        _rec("R-1", "Давлат ижтимоий суғуртаси тўғрисида", "2025-10-20", typ="resolution"),
    ])


def test_registry_editions_one_vertex_with_version_of_edges():
    reg = _registry()
    ref = reg.resolve("Davlat ijtimoiy sugʼurtasi toʼgʼrisidagi Qonunga")
    assert ref.doc_id == "L-new" and not ref.external and ref.node_id == "L-new"
    assert reg.doc_act["L-old"] is reg.doc_act["L-new"]
    assert reg.doc_act["L-new"].versions == ["L-old", "L-new"]
    (e,) = [e for e in reg.edges() if e.attrs.get("relation") == "version_of"]
    assert (e.src, e.dst, e.type) == ("L-old", "L-new", "REFERS_TO")
    ev = e.attrs["evidence"]
    assert ev["doc_id"] == "L-old" and ev["quote"] == "Давлат ижтимоий суғуртаси тўғрисида" and ev["start"] == 0
    assert "R-1" not in reg.doc_act                          # only law_text documents are acts
    assert reg.validate() == []
    assert Schema.load().validate_edge(e, {n.node_id: n for n in reg.nodes()}) == []


def test_registry_external_stub_and_display():
    reg = _registry()
    a = reg.resolve_mention("Ўзбекистон Республикасининг Жиноят кодексига")
    b = reg.resolve_mention("Jinoyat kodeksining")
    c = reg.resolve_mention("Жиноят кодексининг")
    assert a.external and a.doc_id is None and a.node_id == "EXT/jinoyat kodeks" == b.node_id == c.node_id
    (stub,) = [n for n in reg.nodes() if n.node_id == "EXT/jinoyat kodeks"]
    assert stub.external and stub.kind == "document" and stub.doc_id is None
    assert stub.title == "Жиноят кодекси" and stub.attrs["mentions"] == 3   # most frequent display form
    assert reg.validate() == []


def test_registry_amending_laws_distinct_and_date_hint():
    reg = _registry()
    t = "Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish va qoʼshimchalar kiritish toʼgʼrisida"
    assert len(reg.acts[act_key(t)]) == 3                    # never merged as editions
    assert not [e for e in reg.edges() if e.src.startswith("A-")]
    assert reg.resolve_mention(t, date_hint="2025-03-01").doc_id == "A-1"
    amb = reg.resolve_mention(t, date_hint="2025-07-10")     # two acts adopted the same day
    assert amb.ambiguous and amb.node_id is None and set(amb.candidates) == {"A-2", "A-3"}
    far = reg.resolve_mention(t, date_hint="2025-12-31")     # omnibus title: nearest act is too far
    assert far.unlinked and far.node_id is None and far.status == "unlinked"
    assert reg.resolve_mention(t).ambiguous
    assert [a["date_hint"] for a in reg.ambiguous] == ["2025-07-10", None]


def test_registry_generic_external_title_is_not_a_stub():
    reg = ActRegistry.from_records([])
    ref = reg.resolve_mention("Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартишлар киритиш тўғрисида", "2020-01-01")
    assert ref.unlinked and not ref.ambiguous and ref.candidates == ()
    assert reg.external_nodes() == [] and reg.ambiguous == []


def test_registry_code_approval_law_answers_for_the_code():
    reg = _registry()
    assert reg.resolve("Сув кодексига").doc_id == "W-1"
    assert reg.resolve("Suv kodeksi").doc_id == "W-1"


def test_registry_empty_title_rejected():
    with pytest.raises(ValueError):
        _registry().resolve("  ")


@pytest.mark.corpus
def test_registry_on_corpus_records():
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.kg.acts import corpus_mentions
    recs = read_records(config.path("data/records"))
    reg = ActRegistry.from_records(recs)
    st = reg.stats()
    assert st["corpus_law_docs"] == 68 and st["corpus_acts"] == 66      # + 3 mistyped law texts (fix round 1)
    assert st["edition_groups"] == [["QONUNLAR-1c0f3967", "QONUNLAR-49a7f026"]]
    counts = corpus_mentions(recs, reg)
    assert reg.validate() == []
    for key in ("konstitusiya", "maʼmuriy javobgarlik kodeks", "jinoyat kodeks", "soliq kodeks"):
        assert counts["EXT/" + key] >= 100
    assert counts["QONUNLAR-475d92b7"] > 0                 # "Сув кодекси" -> the code-approval law


# --- fix round 1 ------------------------------------------------------------------------------------

def _keys(text):
    return [act_key(m.name) for m in scan_act_mentions(text)]


def test_act_title_numbered_citations():
    t = ("«Стандартлаштириш тўғрисида»ги ЎРҚ–800-сонли Қонуни ва «Олий Мажлис Сенатининг Регламенти "
         "тўғрисида»ги 523‒II-сонли Қонунига, «Мулкчилик тўғрисида»ги 152−XII-сонли Қонунига")
    assert _keys(t) == [act_key("Стандартлаштириш тўғрисида"), act_key("Олий Мажлис Сенатининг Регламенти тўғрисида"),
                        act_key("Мулкчилик тўғрисида")]


def test_act_title_every_list_member():
    t = ("“Aksiyadorlik jamiyatlari toʼgʼrisida”gi, “Auditorlik faoliyati toʼgʼrisida”gi va "
         "“Davlat boji toʼgʼrisida”gi qonunlarga oʼzgartish kiritiladi; “Siyosiy partiyalar toʼgʼrisida”gi va boshqa "
         "bir qator Oʼzbekiston Respublikasi qonunlariga")
    assert _keys(t) == [act_key(x) for x in ("Aksiyadorlik jamiyatlari toʼgʼrisida", "Auditorlik faoliyati toʼgʼrisida",
                                             "Davlat boji toʼgʼrisida", "Siyosiy partiyalar toʼgʼrisida")]
    paren = "Qonuniga (“Notariat toʼgʼrisida”gi, “Mediatsiya toʼgʼrisida”gi va “Davlat boji toʼgʼrisida”gi) oʼzgartirish"
    assert _keys(paren) == [act_key(x) for x in ("Notariat toʼgʼrisida", "Mediatsiya toʼgʼrisida", "Davlat boji toʼgʼrisida")]


def test_act_title_heads_and_blocks():
    assert _keys("“Oʼzbekiston Respublikasi Oliy Majlisining Senati toʼgʼrisida”gi Konstitutsiyaviy Qonunning 12-moddasi") == \
        [act_key("Oliy Majlisining Senati toʼgʼrisida")]
    assert _keys("«Сенати тўғрисида»ги Конституциявий қонунга") == [act_key("Сенати тўғрисида")]
    assert _keys("“Xizmat safarlari toʼgʼrisida”gi Prezident Farmoni qonun") == []      # a decree, not a law
    assert _keys("“Taʼlim toʼgʼrisida”gi qonunchilik") == []                            # "legislation"


def test_act_title_nested_and_multiline():
    t = ("“Oʼzbekiston Respublikasining “Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuniga oʼzgartirish "
         "kiritish haqida”gi Qonun toʼgʼrisida")
    ks = _keys(t)
    assert act_key("Davlat boji toʼgʼrisida") in ks
    assert act_key("“Davlat boji toʼgʼrisida”gi Qonuniga oʼzgartirish kiritish haqida") in ks
    multi = "“Ўзбекистон Республикасининг айрим қонун ҳужжатларига\n\nқўшимча ва ўзгартиришлар киритиш тўғрисида”ги\n\nЎзбекистон Республикаси Қонуни ҳақида"
    assert _keys(multi) == [act_key("айрим қонун ҳужжатларига ўзгартириш киритиш тўғрисида")]


def test_code_names_lexicon_only():
    assert _keys("Biroq kodeksda vakolatli. Bu yerda ikkita kodeks va Shu bois kodeksda. Intizom kodeksini") == []
    assert _keys("Bundan tashqari yer kodeksiga sugʼoriladigan") == [act_key("Ер кодекси")]
    assert _keys("xuddi shunday jinoyat protsessual kodeksida prokuror") == [act_key("Жиноят-процессуал кодекси")]
    assert _keys("Oʼzbekiston Respublikasi Qonuni Budjet kodeksi talablari") == [act_key("Бюджет кодекси")]
    ms = scan_act_mentions("statьей 259 Гражданского кодекса Республики")
    assert [act_key(m.name) for m in ms] == [act_key("Фуқаролик кодекси")] and ms[0].name == "Фуқаролик кодекси"
    t = "Maʼmuriy javobgarlik toʼgʼrisidagi hamda Jinoyat kodekslarining"
    ms = scan_act_mentions(t)
    assert [act_key(m.name) for m in ms] == [act_key("Маъмурий жавобгарлик тўғрисидаги кодекси"), act_key("Жиноят кодекси")]
    assert {m.kind for m in ms} == {"code_list"} and all(m.name.endswith("kodeksi") for m in ms)
    assert _keys("Iqtisodiy protsessual, Fuqarolik protsessual va Soliq kodekslariga") == \
        [act_key(x) for x in ("Иқтисодий процессуал кодекси", "Фуқаролик процессуал кодекси", "Солиқ кодекси")]
    assert all(m.name.endswith("кодекси") for m in scan_act_mentions("Жиноят ва Солиқ кодексларига"))


def test_constitution_possessive_forms():
    assert _keys("Konstitutsiyamizda, Конституциямизнинг ва Konstitusiyamizga") == [act_key("Конституция")] * 3
    assert act_key("Konstitutsiyamizda") == act_key("Конституция") == act_key("Konstitusiyasi")


def test_act_key_fold_hard_sign_and_tg_typos():
    assert act_key("Тадбиркорлик субъектлари ҳуқуқлари тўғрисида") == act_key("Tadbirkorlik subyektlari huquqlari toʼgʼrisida")
    assert act_key("Маданий мерос объектлари тўғрисида") == act_key("Madaniy meros obyektlari toʼgʼrisida")
    assert act_key("Reklama toʼgʼrsida") == act_key("Reklama toʼgʼrisada") == act_key("Реклама тўғрисида")


def test_resolve_is_pure():
    reg = _registry()
    reg.resolve("Jinoyat kodeksi")
    reg.resolve("Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida")
    assert reg.external_nodes() == [] and reg.ambiguous == []
    reg.note_mention(reg.resolve("Jinoyat kodeksi"), "Jinoyat kodeksi")
    assert [n.node_id for n in reg.external_nodes()] == ["EXT/jinoyat kodeks"]


def test_date_hint_drops_later_candidates():
    reg = ActRegistry.from_records([
        _rec("B-1", "«Божхона тарифи тўғрисида»ги Қонунга ўзгартириш киритиш ҳақида", "2023-05-01"),
        _rec("B-2", "«Божхона тарифи тўғрисида»ги Қонунга қўшимча киритиш ҳақида", "2025-05-01"),
    ])
    t = "“Bojxona tarifi toʼgʼrisida”gi Qonunga oʼzgartish kiritish haqida"
    assert reg.resolve(t, "2024-01-01").doc_id == "B-1"          # B-2 is later than the mention
    assert reg.resolve(t, "2025-04-20").doc_id == "B-1"          # adopted by the date beats the grace window
    only = ActRegistry.from_records([_rec("B-2", "«Божхона тарифи тўғрисида»ги Қонунга қўшимча киритиш ҳақида", "2025-05-01")])
    assert only.resolve(t, "2025-04-20").doc_id == "B-2"         # titled act: 30-day grace (early draft)
    ext = reg.resolve(t, "2020-01-01")                           # none feasible: another (external) act
    assert ext.external and ext.node_id == "EXT/" + act_key(t)
    assert reg.resolve(t).ambiguous and set(reg.resolve(t).candidates) == {"B-1", "B-2"}


def test_mistyped_law_texts_and_translation():
    body_ru = "\n\n".join(f"Статья {i}. Текст" for i in range(1, 5))
    ru = DocRecord("R-ru", Meta("resolution", None, "senate", ["ru"], "принят", "QONUNLAR", "О санации банков"),
                   Source("docs", "ru.docx", None),
                   f"Проект\n\nЗАКОН\n\nО санации банков\n\nПринят Законодательной палатой 13 февраля 2025 года\n\n{body_ru}",
                   [Node("R-ru", "документ", "О санации банков", None, None, 0, 10)]
                   + [Node(f"R-ru/{i}", "статья", f"С{i}", str(i), "R-ru", 0, 10) for i in range(1, 5)], [])
    note = _rec("N-1", "Банкларни санация қилиш тўғрисида", None, typ="note",
                articles=("Мақсад", "Асосий тушунчалар", "Қонунчилик"))
    note.text = note.text.replace("томонидан қабул", "томонидан 2025 йил 13 февралда қабул")
    uz = _rec("U-1", "Банкларни санация қилиш тўғрисида", "2025-02-13",
              articles=("Мақсад", "Асосий тушунчалар", "Қонунчилик", "Тартиб"))
    few = _rec("F-1", "Бошқа тўғрисида", "2025-01-01", typ="note", articles=("Бир", "Икки"))
    reg = ActRegistry.from_records([ru, note, uz, few])
    assert "N-1" in reg.doc_act and "F-1" not in reg.doc_act     # header + >= 3 articles
    assert reg.doc_act["N-1"] is reg.doc_act["U-1"]               # editions of one act
    (tr,) = [e for e in reg.edges() if e.attrs["relation"] == "translation_of"]
    assert tr.src == "R-ru" and tr.dst == reg.doc_act["U-1"].head
    assert reg.resolve("О санации банков").doc_id == reg.doc_act["U-1"].head
    assert reg.validate() == []


def test_code_approval_alias_external():
    ref = ActRegistry.from_records([]).resolve("Oʼzbekiston Respublikasining Tadbirkorlik kodeksini tasdiqlash toʼgʼrisida")
    assert ref.external and ref.node_id == "EXT/tadbirkorlik kodeks" and ref.display == "Tadbirkorlik kodeksi"


# --- fix round 2 ------------------------------------------------------------------------------------

def _mentions_resolved(reg, rec):
    from graphrag.kg.acts import record_date
    return [(rec.text[m.start:m.end], reg.resolve_mention(m, record_date(rec), src_doc=rec.doc_id))
            for m in scan_act_mentions(rec.text)]


OMNIBUS_311 = ("Ўзбекистон Республикасининг 2011 йил 21 декабрда қабул қилинган «Ўзбекистон Республикасининг айрим "
               "қонун ҳужжатларига ўзгартиш ва қўшимчалар киритиш тўғрисида»ги ЎРҚ–311-сонли Қонунининг")
RAQOBAT_319 = "Ўзбекистон Республикасининг 2012 йил 6 январда қабул қилинган «Рақобат тўғрисида»ги ЎРҚ–319-сонли Қонуни"
RAQOBAT_850 = "Ўзбекистон Республикасининг 2023 йил 3 июлда қабул қилинган «Рақобат тўғрисида»ги ЎРҚ−850-сонли Қонунининг"
MChJ_310 = ("Ўзбекистон Республикасининг 2001 йил 6 декабрда қабул қилинган «Масъулияти чекланган жамиятлар "
            "тўғрисида»ги 310–II-сонли Қонуни")


def _round2_registry():
    omni = _rec("O-1", "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида", "2025-05-06")
    raq = _rec("RQ", "Рақобат тўғрисида", "2023-02-14", body=RAQOBAT_319)
    mchj = _rec("MC", "Масъулияти чекланган жамиятлар тўғрисида", "2025-06-10", body=MChJ_310 + "\n\n" + OMNIBUS_311)
    mchj.meta.status = "проект"
    citing = _rec("CT", "Банкларни санация қилиш ва тугатиш тўғрисида", "2025-02-13",
                  body=RAQOBAT_850 + "\n\n" + "«Масъулияти чекланган жамиятлар тўғрисида»ги 310–II-сонли Қонуни")
    return ActRegistry.from_records([omni, raq, mchj, citing]), {"RQ": raq, "MC": mchj, "CT": citing}


def test_numbered_historical_citations_never_hit_corpus_drafts():
    reg, recs = _round2_registry()
    res = dict(_mentions_resolved(reg, recs["MC"]))
    omni = next(v for k, v in res.items() if "ЎРҚ–311" in k)
    assert omni.number == "oʼrq-311" and omni.external and omni.doc_id is None
    assert omni.node_id == "EXT/ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida#oʼrq-311"
    own = next(v for k, v in res.items() if "310–II" in k)            # self-link guard + older edition
    assert own.node_id == "EXT/masʼuliyati cheklangan jamiyatlar toʼgʼrisida#310-ii"
    other = dict(_mentions_resolved(reg, recs["CT"]))                 # numbered, undated: never the draft
    assert next(v for k, v in other.items() if "310–II" in k).node_id == own.node_id


def test_numbered_citation_date_and_self_link():
    reg, recs = _round2_registry()
    (_, old), = _mentions_resolved(reg, recs["RQ"])                   # the 2012 law quoted in the 2023 one
    assert old.node_id == "EXT/raqobat toʼgʼrisida#oʼrq-319" and old.edition == "oʼrq-319"
    new = next(v for k, v in _mentions_resolved(reg, recs["CT"]) if "850" in k)
    assert new.doc_id == "RQ"                                         # enacted 2023-07-03, chamber 2023-02-14
    assert reg.resolve("«Рақобат тўғрисида»ги Қонун", "2025-01-01", cited_date="2012-01-06").node_id == \
        "EXT/raqobat toʼgʼrisida#2012-01-06"
    assert reg.resolve("Рақобат тўғрисида", src_doc="RQ").unlinked  # a law naming itself: no edge, no stub
    stub = next(n for n in reg.external_nodes() if n.node_id.endswith("#oʼrq-319"))
    assert stub.attrs["edition"] == "oʼrq-319" and stub.attrs["numbers"] == ["oʼrq-319"]
    assert reg.validate() == []


def test_cited_chamber_date_of_an_older_edition():
    reg = _registry()                                                 # editions L-old 2025-02-18 -> L-new
    assert reg.resolve("Давлат ижтимоий суғуртаси тўғрисида", "2025-02-20", cited_date="2025-02-18").doc_id == "L-new"


def test_context_date_for_undated_records():
    from graphrag.kg.acts import record_date
    r = _rec("X-1", "Хулоса", None, typ="resolution", articles=())
    r.text = ("QARORI\n\nQonunchilik palatasi tomonidan 2025-yil 17-iyunda qabul qilingan “Oʼzbekiston "
              "Respublikasining ayrim qonun hujjatlariga oʼzgartirish va qoʼshimchalar kiritish toʼgʼrisida”gi Qonun")
    assert record_date(r) == "2025-06-17"
    (m,) = scan_act_mentions(r.text)
    assert m.date == "2025-06-17"


def test_code_span_is_tight():
    t = ("“2024-yil uchun Oʼzbekiston Respublikasining Davlat budjeti toʼgʼrisida”gi Oʼzbekiston Respublikasi "
         "Qonuni Budjet kodeksi talablari")
    ms = scan_act_mentions(t)
    assert [m.kind for m in ms] == ["act_title", "code_name"] and t[ms[1].start:ms[1].end] == "Budjet kodeksi"
    for t, span in (("билан тасдиқланган Ўзбекистон Республикасининг Солиқ кодекси", "Солиқ кодекси"),
                    ("Yangi tahrirdagi Soliq kodeksi", "Soliq kodeksi"), ("– Oʼzbekiston Respublikasi Mehnat kodeksi", "Mehnat kodeksi")):
        (m,) = scan_act_mentions(t)
        assert t[m.start:m.end] == span


# --- fix round 3 ------------------------------------------------------------------------------------

OMNI_TITLE = "Ўзбекистон Республикасининг айрим қонун ҳужжатларига қўшимча ва ўзгартиришлар киритиш тўғрисида"
CHAMBER_0513 = ("Ўзбекистон Республикаси Олий Мажлисининг Қонунчилик палатаси томонидан 2025 йил 13 май куни қабул "
                f"қилинган “{OMNI_TITLE}”ги Ўзбекистон Республикаси Қонунини кўриб чиқиб")
ENERGY = ("Yoqilgʼi-energetika sohasidagi huquqbuzarliklarga qarshi kurashish choralari takomillashtirilishi munosabati "
          "bilan Oʼzbekiston Respublikasining Jinoyat kodeksiga oʼzgartirish va qoʼshimchalar kiritish haqida")


def _round3_registry():
    return ActRegistry.from_records([
        _rec("D7", "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида", "2025-05-06"),
        _rec("EC", "Ёқилғи-энергетика соҳасидаги ҳуқуқбузарликларга қарши курашиш чоралари такомиллаштирилиши муносабати "
                   "билан Ўзбекистон Республикасининг Жиноят кодексига ўзгартириш ва қўшимчалар киритиш ҳақида", "2025-07-15"),
    ])


def test_chamber_date_kuni_form_and_exact_match_for_omnibus():
    reg = _round3_registry()
    (m,) = scan_act_mentions(CHAMBER_0513)
    assert (m.date, m.date_kind) == ("2025-05-13", "chamber")
    ref = reg.resolve(m.name, "2025-05-13", cited_date=m.date, cited_kind=m.date_kind)
    assert ref.external and ref.node_id.endswith("#2025-05-13")          # not D7 (2025-05-06)
    same = CHAMBER_0513.replace("13 май", "6 май")
    (m6,) = scan_act_mentions(same)
    assert reg.resolve(m6.name, "2025-05-20", cited_date=m6.date, cited_kind=m6.date_kind).doc_id == "D7"
    # an enactment ("stated") date keeps the 60-day window for an omnibus title
    stated = f"Ўзбекистон Республикасининг 2025 йил 20 майда қабул қилинган «{OMNI_TITLE}»ги ЎРҚ–1050-сонли Қонуни"
    (ms,) = scan_act_mentions(stated)
    assert ms.date_kind == "stated"
    assert reg.resolve(ms.name, cited_date=ms.date, cited_kind=ms.date_kind).doc_id == "D7"


def test_chamber_date_titled_act_keeps_grace():
    reg = _round3_registry()
    t = f"Oʼzbekiston Respublikasi Oliy Majlisining Qonunchilik palatasi tomonidan 2025-yil 11-iyul kuni qabul qilingan “{ENERGY}”gi Oʼzbekiston Respublikasi Qonuni"
    (m,) = scan_act_mentions(t)
    assert (m.date, m.date_kind) == ("2025-07-11", "chamber")
    assert reg.resolve(m.name, "2025-07-11", cited_date=m.date, cited_kind=m.date_kind).doc_id == "EC"


def test_same_law_consistency_within_a_record():
    from graphrag.kg.acts import resolve_record
    reg = _round3_registry()
    rec = _rec("Q-1", "Хулоса", "2025-05-13", typ="resolution", articles=())
    rec.text = f"Қ А Р О Р И\n\n“{OMNI_TITLE}”ги Ўзбекистон Республикаси Қонуни ҳақида\n\n{CHAMBER_0513}\n\n1. “{OMNI_TITLE}”ги Ўзбекистон Республикаси Қонуни маъқуллансин."
    refs = [ref for _, ref in resolve_record(rec, reg)]
    assert len(refs) == 3 and len({r.node_id for r in refs}) == 1 and refs[0].node_id.endswith("#2025-05-13")


from graphrag.kg.acts import ActRef, Mention, follow_undated, nested_flags


def _ref(key, doc):
    return ActRef(key, doc, False, key)


def test_undated_mention_follows_a_dated_one_only_in_the_same_nesting_role():
    """M2 Task 3 minor (R29 → M3): consistency keyed on the title key AND the nesting role."""
    dated = Mention(0, 60, "Банклар тўғрисида", "act_title", date="2019-03-05")
    outer = Mention(98, 200, "Банклар тўғрисида»ги Қонунига ўзгартиш киритиш ҳақида", "act_title")
    inner = Mention(100, 120, "Банклар тўғрисида", "act_title")
    free = Mention(300, 320, "Банклар тўғрисида", "act_title")
    ms = [dated, outer, inner, free]
    refs = [_ref("k", "D2019"), _ref("amend", None), _ref("k", "D1996"), _ref("k", "D1996")]
    nz = nested_flags(ms)
    assert nz == [False, False, True, False]
    out = follow_undated(ms, refs, ["", "", "", ""], nz, lambda ref: True)
    assert out[3] is not None and out[3].doc_id == "D2019" and out[3].display == "k"
    assert out[2] is None and out[0] is None and out[1] is None


def test_follow_respects_date_compatibility_and_uniqueness():
    a = Mention(0, 10, "X", "act_title", date="2019-01-01")
    b = Mention(20, 30, "X", "act_title", date="2020-01-01")
    u = Mention(40, 50, "X", "act_title")
    refs = [_ref("x", "D1"), _ref("x", "D2"), _ref("x", "D0")]
    assert follow_undated([a, b, u], refs, [""] * 3, [False] * 3, lambda r: True)[2] is None
    assert follow_undated([a, u], [refs[0], refs[2]], [""] * 2, [False] * 2, lambda r: False)[1] is None
