# tests/test_kg_sweep.py -- M2 pre-Task-11 sweep of deferred minors (Ruling R29, items A-G).
# Each test is built from the real corpus case it names (record / node in the comment).
from graphrag.kg.acts import ActRegistry, scan_act_mentions
from graphrag.kg.extract_amend import extract_amendments
from graphrag.kg.extract_ref import RefContext, extract_refs
from tests.test_kg_refs import _rec

TAIL = ["2-модда. Ижро", "Ижро этилсин.", "3-модда. Кучга кириш", "Ушбу Қонун расмий эълон қилинган кундан кучга киради."]
HEAD = ["1-модда. Ўзбекистон Республикасининг айрим қонун ҳужжатларини ўз кучини йўқотган деб топиш",
        "Қуйидагилар ўз кучини йўқотган деб топилсин:"]
GAZ = "(Ўзбекистон Республикаси Олий Мажлиси палаталарининг Ахборотномаси, {}, {}-модда)"

# QONUNLAR-eca1044a/ГЛ.8/СТ.68/Ч.1/П.24 (outer opening quote missing, ЎРҚ–911)
ITEM_911 = ("1) Ўзбекистон Республикасининг 2024 йил 21 февралда қабул қилинган «Тўловга қобилиятсизлик тўғрисида»ги "
            "Ўзбекистон Республикаси Қонуни қабул қилинганлиги муносабати билан Ўзбекистон Республикасининг айрим "
            "қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш ҳақида»ги ЎРҚ–911-сонли Қонунининг "
            + GAZ.format("2024 йил, № 2", 111) + " 12-моддаси;")
# STENOGRAMMA-5d7ede9b/ВОПР.10: the same law as an undated, unnumbered agenda title
AGENDA_911 = ("“Toʼlovga qobiliyatsizlik toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuni qabul qilinganligi munosabati "
              "bilan Oʼzbekiston Respublikasining ayrim qonun hujjatlariga oʼzgartirish va qoʼshimchalar kiritish toʼgʼrisida")
# QONUNLAR-f79c702c/ГЛ.9/СТ.45/Ч.1/П.4 (quoted, 815–I) and П.6 (outer quote missing, ЎРҚ–120): two laws, one title
ITEM_815 = ("1) Ўзбекистон Республикасининг 1999 йил 19 августда қабул қилинган «Ўзбекистон Республикасининг «Табиий "
            "монополиялар тўғрисида»ги Қонунига ўзгартишлар ва қўшимчалар киритиш ҳақида»ги 815–I-сонли Қонуни "
            + "(Ўзбекистон Республикаси Олий Мажлисининг Ахборотномаси, 1999 йил, № 9, 212-модда)" + ";")
ITEM_120 = ("2) Ўзбекистон Республикасининг 2007 йил 7 декабрда қабул қилинган «Табиий монополиялар тўғрисида»ги "
            "Ўзбекистон Республикаси Қонунига ўзгартиш ва қўшимчалар киритиш ҳақида»ги ЎРҚ–120-сонли Қонуни "
            + GAZ.format("2007 йил, № 12", 591) + ";")
# П.7: the outer title names an article of the base law ("Қонунининг 4-моддасига ўзгартиш киритиш ҳақида")
ITEM_266 = ("3) Ўзбекистон Республикасининг 2010 йил 14 декабрда қабул қилинган «Табиий монополиялар тўғрисида»ги "
            "Ўзбекистон Республикаси Қонунининг 4-моддасига ўзгартиш киритиш ҳақида»ги ЎРҚ–266-сонли Қонуни "
            + GAZ.format("2010 йил, № 12", 466) + ".")


def _law(items, doc_id="QONUNLAR-aaaa0911", date="2025-06-10"):
    rec = _rec(HEAD + items + TAIL, doc_id=doc_id, title="Масъулияти чекланган жамиятлар тўғрисида", date=date)
    return rec


def _edges(recs):
    reg = ActRegistry.from_records(recs)
    ctx = RefContext.from_records(recs, reg)
    refs = [e for r in recs for e in extract_refs(r, reg, ctx)]
    am = [e for r in recs for e in extract_amendments(r, reg, ctx)[0]]
    return reg, ctx, refs, am


def _by_item(edges, typ):
    out: dict[str, set] = {}
    for e in edges:
        if e.type == typ and "/П." in e.src:
            out.setdefault(e.src, set()).add(e.dst)
    return out


# ── A (Task 5): outer-title repeal items ──────────────────────────────────────────────────────────
def test_outer_title_without_opening_quote_is_one_mention_of_the_amending_law():
    text = ITEM_911
    ms = [m for m in scan_act_mentions(text) if m.kind == "act_title"]
    outer = max(ms, key=lambda m: m.end - m.start)
    assert outer.number == "oʼrq-911" and outer.date == "2024-02-21"
    assert text[outer.start:outer.end].endswith("ЎРҚ–911-сонли Қонунининг")
    assert "Тўловга қобилиятсизлик" in outer.name and outer.name.endswith("киритиш ҳақида")


def test_a_law_cited_once_by_number_and_once_undated_is_one_vertex():
    # ЎРҚ–911: the repeal item (numbered) and the agenda title (undated) must not split into two EXT vertices
    rec = _law([ITEM_911])
    reg, ctx, refs, am = _edges([rec])
    rep = [e for e in am if e.type == "REPEALS"]
    assert len(rep) == 1
    act = reg.resolve(AGENDA_911, "2024-03-01").node_id
    assert act and "#" not in act
    assert rep[0].dst == act + "/СТ.12"


def test_outer_title_repeal_items_refers_to_the_same_targets_as_repeals():
    # QONUNLAR-f79c702c П.4 / П.6 / П.7 and eca1044a П.24: REFERS_TO of an item = its REPEALS targets
    rec = _law([ITEM_815, ITEM_120, ITEM_266], doc_id="QONUNLAR-aaaa0120", date="2023-02-14")
    reg, ctx, refs, am = _edges([rec])
    rep, ref = _by_item(am, "REPEALS"), _by_item(refs, "REFERS_TO")
    assert len(rep) == 3
    base = "EXT/tabiiy monopoliyalar toʼgʼrisida"      # the base law nested in each outer title is cited too
    nested = {"П.1": {base}, "П.2": {base}, "П.3": {base + "/СТ.4"}}   # (Task 4 convention; П.3 names its СТ.4)
    for src, dsts in rep.items():
        assert ref.get(src) == dsts | nested[src.rsplit("/", 1)[1]], src
    tgt = sorted(d for v in rep.values() for d in v)
    # two laws with one title (815–I, ЎРҚ–120) are told apart; ЎРҚ–266 (one number) keeps one vertex
    assert tgt == ["EXT/tabiiy monopoliyalar toʼgʼrisida qonun 4 moddasiga oʼzgartirish kiritish toʼgʼrisida",
                   "EXT/tabiiy monopoliyalar toʼgʼrisida qonun oʼzgartirish kiritish toʼgʼrisida#815-i",
                   "EXT/tabiiy monopoliyalar toʼgʼrisida qonun oʼzgartirish kiritish toʼgʼrisida#oʼrq-120"]


def test_outer_title_item_article_is_the_amending_laws_article():
    # QONUNLAR-d0d0b450/…/ПП.8 and 475d92b7/…/П.22: "…ҳақида»ги ЎРҚ–721-сонли Қонунининг (…) 15-моддаси"
    item = ("1) Ўзбекистон Республикасининг 2021 йил 12 октябрда қабул қилинган «Лицензиялаш, рухсат бериш ва хабардор "
            "қилиш тартиб-таомиллари тўғрисида»ги Ўзбекистон Республикаси Қонуни қабул қилинганлиги муносабати билан "
            "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартиш ва қўшимчалар киритиш ҳақида»ги "
            "ЎРҚ–721-сонли Қонунининг " + GAZ.format("2021 йил, № 10", 968) + " 15-моддаси.")
    rec = _law([item], doc_id="QONUNLAR-aaaa0721", date="2023-01-31")
    reg, ctx, refs, am = _edges([rec])
    (rep,) = [e.dst for e in am if e.type == "REPEALS"]
    assert rep.endswith("/СТ.15") and "#" not in rep and "qabul qilinganligi munosabati" in rep
    base = "EXT/lisenziyalash ruxsat berish va xabardor qilish tartib taomillar toʼgʼrisida"
    assert {e.dst for e in refs if "/П." in e.src} == {rep, base}   # not the base law's СТ.15, not a code's


def _amend_run(paras, doc_id="QONUNLAR-aaaa0416", date="2024-01-19"):
    rec = _rec(paras + TAIL, doc_id=doc_id, title="Суд ҳокимияти тўғрисида", date=date)
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    refs = extract_refs(rec, reg, ctx)
    edges, _ = extract_amendments(rec, reg, ctx)
    return refs, [e for e in edges if e.type == "AMENDS"]


def test_mixed_range_ending_in_a_superscript_includes_every_member():
    # QONUNLAR-b82a9aa4 "28) 416 – 419³-моддалар қуйидаги таҳрирда баён этилсин:"
    from graphrag.kg.extract_ref import _parse_nums
    assert _parse_nums("416 – 419³") == ["416", "417", "418", "419", "419.1", "419.2", "419.3"]
    assert _parse_nums("383 – 385¹") == ["383", "384", "385", "385.1"]              # unchanged (Task 5)
    refs, am = _amend_run(["1-модда. Фуқаролик процессуал кодексига ўзгартиш",
                           "Ўзбекистон Республикасининг Фуқаролик процессуал кодексига қуйидаги ўзгартишлар киритилсин:",
                           "1) 416 – 419³-моддалар қуйидаги таҳрирда баён этилсин:", "«416-модда. Матн»;"])
    got = sorted(e.dst.rsplit("/", 1)[1] for e in am)
    assert got == ["СТ.416", "СТ.417", "СТ.418", "СТ.419", "СТ.419.1", "СТ.419.2", "СТ.419.3"]


def test_ordinal_section_word_is_a_unit():
    # QONUNLAR-7c19406d/СТ.1/Ч.1/П.2 "2) саккизинчи бўлим:" + term definitions of the Criminal Code's section VIII
    refs, am = _amend_run(["1-модда. Жиноят кодексига ўзгартиш",
                           "Ўзбекистон Республикасининг Жиноят кодексига қуйидаги ўзгартиришлар ва қўшимча киритилсин:",
                           "1) саккизинчи бўлим:",
                           "«Валюта операциялари» атамасининг ҳуқуқий маъноси қуйидаги таҳрирда баён этилсин:",
                           "«Валюта операциялари – …»;",
                           "қуйидаги мазмундаги атамалар билан тўлдирилсин:", "«Қимматбаҳо металлар – …»."],
                          doc_id="QONUNLAR-aaaa0008")
    assert [(e.dst, e.attrs.get("ops")) for e in am] == [("EXT/jinoyat kodeks/БЎЛ.8", ["edit", "add"])]


def test_russian_unit_words():
    # Russian instruction clauses name the unit before its number ("раздел восьмой", "главу XXIV¹", "приложение 2",
    # "подраздел 4 раздела II"); the corpus has no Russian amending law text yet (0 corpus edges), so the
    # clause scanner is checked directly
    from graphrag.kg.extract_amend import _Doc
    paras = ["1-модда. Матн", "раздел восьмой изложить в следующей редакции;", "главу XXIV¹ исключить;",
             "приложение 2 к Кодексу изложить в следующей редакции;", "подраздел 4 раздела II изложить в следующей редакции;",
             "восьмой раздел изложить в следующей редакции."]
    rec = _rec(paras + TAIL, doc_id="QONUNLAR-aaaa0009")
    reg = ActRegistry.from_records([rec])
    doc = _Doc(rec, RefContext.from_records([rec], reg))
    got = []
    for t in paras[1:]:
        a = rec.text.index(t)
        got.append([(c, n) for c, n, _ in doc.elements(a, a + len(t), "edit").units])
    assert got == [[("БЎЛ", ["8"])], [("ГЛ", ["24.1"])], [("ИЛОВА", ["2"])], [("БЎЛ", ["2"]), ("БЎЛ", ["4"])],
                   [("БЎЛ", ["8"])]]


# ── B (Task 6/7): REGULATES ───────────────────────────────────────────────────────────────────────
def test_llm_mention_naming_the_chamber_deputy_is_the_chamber_post():
    # QONUNLAR-3f9b4f02/СТ.11/Ч.1/П.24: LLM name "Қонунчилик палатаси депутати" (key stripped to "deputat")
    from graphrag.kg.extract_ent import owner_post, owner_rewrites
    orw = owner_rewrites()
    assert owner_post("Қонунчилик палатаси депутати", "position", "deputat", orw) == \
        ("position", "qonunchilik palata deputat")
    assert owner_post("Қонунчилик палатасининг депутати", "position", "deputat", orw) == \
        ("position", "qonunchilik palata deputat")                                   # СТ.5/Ч.1/П.1
    assert owner_post("Депутат", "position", "deputat", orw) is None                # a bare deputy stays generic
    assert owner_post("халқ депутатлари", "position", "deputat", orw) is None
    assert owner_post("Сенат аъзоси", "position", "senat aʼzo", orw) is None


def _line(text, name, nth=0):
    a = -1
    for _ in range(nth + 1):
        a = text.index(name, a + 1)
    return a, a + len(name)


def test_converb_and_pluperfect_history_lines_are_adoption_lines():
    from graphrag.kg.ent_rules import adoption_line
    # XULOSALAR-1d8fbb51/БЛ.2
    t1 = ("Ushbu Qonun dastlab Qonunchilik palatasi tomonidan 2025-yil 18-fevralda qabul qilinib, Senat tomonidan "
          "2025-yil 4-sentabrda rad etilgan. Oliy Majlis palatalarining Kelishuv komissiyasida Qonun qayta ishlab "
          "chiqildi va kelishildi.\n\nQonun 6 ta bob, 25 ta moddadan iborat.")
    assert adoption_line(t1, *_line(t1, "Qonunchilik palatasi"))
    assert adoption_line(t1, *_line(t1, "Senat"))
    # XULOSALAR-b7745577/БЛ.1
    t2 = ("Ўзбекистон Республикаси Олий Мажлиси Сенатининг 25 ялпи мажлисида “Қўриқлаш фаолияти тўғрисида”ги "
          "Ўзбекистон Республикаси Қонуни сенаторлар томонидан маъқулланиб, 2022 йил июнь ойида Давлатимиз Раҳбари "
          "томонидан имзоланган эди.")
    for name in ("Олий Мажлиси Сенатининг", "сенаторлар", "Давлатимиз Раҳбари"):
        assert adoption_line(t2, *_line(t2, name)), name


def test_norms_describing_the_procedure_stay_regulation():
    from graphrag.kg.ent_rules import adoption_line
    # present / conditional norms of QONUNLAR-3f9b4f02 (Task 7: participles inside norms are excluded on purpose)
    for t, name in (("Қонун Сенат томонидан маъқулланиб, Президентга имзолаш учун юборилади.", "Сенат"),
                    ("Сенат томонидан қонун рад этилган тақдирда, Қонунчилик палатаси келишув комиссиясини тузади.", "Сенат"),
                    ("Қонунчилик палатаси томонидан қабул қилинган қонун Сенатга юборилади.", "Қонунчилик палатаси"),
                    ("Сенатнинг ялпи мажлисида қонун маъқулланади.", "Сенатнинг")):
        assert adoption_line(t, *_line(t, name)) is None, t


# ── C (Task 4): title edge of a qaror / xulosa whose meta.title is a decree cited in the body ─────────
def _xulosa(doc_id, text, title, head_end, date=None):
    from tests.test_kg_refs import _manual
    return _manual(doc_id, "committee_conclusion", "XULOSALAR", text,
                   [("ЗАГ.1", "заголовок", "1", "", 0, head_end), ("БЛ.1", "блок", "1", "", head_end + 2, len(text))],
                   title=title, date=date)


def test_title_edge_falls_back_to_the_headings_first_quoted_act():
    # XULOSALAR-8d225faf: meta.title = the Presidential decree cited in the body; the heading names the law
    text = ("Loyiha\n\n2025-yil -_______ -sonli Toshkent sh.\n\n“Давлат мудофаа буюртмаси тўғрисида”ги Қонун юзасидан "
            "Х У Л О С А\n\nЎзбекистон Республикаси Олий Мажлисининг Қонунчилик палатаси томонидан 2024 йил 20 августда "
            "қабул қилинган “Давлат мудофаа буюртмаси тўғрисида”ги Ўзбекистон Республикаси Қонуни (бундан буён матнда "
            "Қонун деб юритилади) 2024 йил 22 август куни Ўзбекистон Республикаси Олий Мажлисининг Сенатига кўриб "
            "чиқиш учун келиб тушди.\n\nҚонун Ўзбекистон Республикаси Президентининг 2023 йил 31 январдаги “Ўзбекистон "
            "Республикаси Қуролли Кучларини қуроллантириш ва мудофаа саноати соҳасида давлат бошқарувини "
            "такомиллаштириш бўйича чора-тадбирлар тўғрисида”ги ПФ–16-сон Фармони ижросини таъминлаш мақсадида "
            "ишлаб чиқилган.")
    title = ("Ўзбекистон Республикаси Қуролли Кучларини қуроллантириш ва мудофаа саноати соҳасида давлат бошқарувини "
             "такомиллаштириш бўйича чора-тадбирлар тўғрисида")
    rec = _xulosa("XULOSALAR-aaaa8d22", text, title, text.index("\n\nЎзбекистон"), date="2024-08-20")
    reg = ActRegistry.from_records([rec])
    edges = extract_refs(rec, reg, RefContext.from_records([rec], reg))
    got = [(e.src, e.dst, e.attrs["evidence"]["quote"]) for e in edges if e.attrs.get("relation") == "title"]
    assert got == [(rec.doc_id, "EXT/davlat mudofaa buyurtma toʼgʼrisida", "“Давлат мудофаа буюртмаси тўғрисида”ги Қонун")]


def test_no_title_fallback_to_the_base_law_of_an_unquoted_outer_title():
    # XULOSALAR-fa8ed64a: "“Davlat boji toʼgʼrisida”gi … Qonuniga qoʼshimcha kiritish haqida”gi … Qonuni" -- the first
    # quoted title is the base law inside the (opening-quote-less) amending-law title: not the subject
    text = ("2023-yil 00-fevral | 00-sonli | Toshkent sh.\n\n“Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuniga "
            "qoʼshimcha kiritish haqida”gi Oʼzbekiston Respublikasi Qonuni yuzasidan\n\nX U L O S A\n\nOʼzbekiston "
            "Respublikasi Oliy Majlisining Qonunchilik palatasi tomonidan “Davlat boji toʼgʼrisida”gi Oʼzbekiston "
            "Respublikasi Qonuniga qoʼshimcha kiritish haqida”gi Qonuni (bundan buyon matnda Qonun deb yuritiladi) "
            "Oʼzbekiston Respublikasi Oliy Majlisining Senatiga koʼrib chiqish uchun kelib tushdi.\n\nQonun Oʼzbekiston "
            "Respublikasi Prezidentining 2022-yil 20-apreldagi “Davlat xizmatlari koʼrsatishni soddalashtirish "
            "chora-tadbirlari toʼgʼrisida”gi PF-113-son Farmoni bilan tasdiqlangan.")
    rec = _xulosa("XULOSALAR-aaaafa8e", text, "Davlat xizmatlari koʼrsatishni soddalashtirish chora-tadbirlari toʼgʼrisida",
                  text.index("\n\nOʼzbekiston Respublikasi Oliy"))
    reg = ActRegistry.from_records([rec])
    edges = extract_refs(rec, reg, RefContext.from_records([rec], reg))
    assert [e for e in edges if e.attrs.get("relation") == "title"] == []


def test_inquiry_window_stops_at_the_heading_end():
    # a body-only meta.title: the inquiry window is the first long paragraph, which lies after hend here; an
    # inquiry word in that body paragraph ("… deputat soʼrovi …") does not make the conclusion an inquiry
    body = ("Qonun Oʼzbekiston Respublikasi Prezidentining 2023-yil 1-martdagi “Tadbirkorlik muhitini yaxshilash "
            "chora-tadbirlari toʼgʼrisida”gi PF-30-son Farmoni ijrosini taʼminlash maqsadida ishlab chiqilgan boʼlib, "
            "unda deputat soʼrovi va senator soʼroviga javob berish muddatlari, raqobat organining vakolatlari hamda "
            "bozorda ustun mavqeni suisteʼmol qilishning oldini olish choralari aniq belgilab qoʼyilgan.")
    text = ("2023-yil 1-mart  63-sonli  Toshkent sh.\n\n“Raqobat toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuni "
            "yuzasidan\n\nX U L O S A\n\nOʼzbekiston Respublikasi Oliy Majlisining Qonunchilik palatasi tomonidan "
            "“Raqobat toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuni Senatiga koʼrib chiqish uchun kelib tushdi.\n\n" + body)
    rec = _xulosa("XULOSALAR-aaaa74af", text, "Tadbirkorlik muhitini yaxshilash chora-tadbirlari toʼgʼrisida",
                  text.index("\n\nOʼzbekiston"), date="2023-03-01")
    reg = ActRegistry.from_records([rec])
    ctx = RefContext.from_records([rec], reg)
    edges = extract_refs(rec, reg, ctx)
    assert ctx.stats.get("title_skipped_inquiry", 0) == 0
    assert [e.dst for e in edges if e.attrs.get("relation") == "title"] == ["EXT/raqobat toʼgʼrisida"]


# ── D (Task 9): no EXT vertex for a mention re-resolved to a corpus act ──────────────────────────────
def test_fuzzy_title_linked_to_a_corpus_act_leaves_no_external_vertex():
    # the phantom EXT/banklarni sanasiya qilish tugatish toʼgʼrisida of b08137fe08c1 (ref_resolve 5a2a2976…): a
    # variant title ("санация қилиш, тугатиш") whose ref_resolve answer is the corpus law QONUNLAR-07b69d6b
    from graphrag.kg.extract_ref import extract_all
    from tests.test_kg_refs import _manual
    law = _rec(["1-модда. Мақсад", "Банкларни санация қилиш ва тугатиш муносабатларини тартибга солиш.", "2-модда. Б",
                "Матн.", "3-модда. В", "Матн."], "QONUNLAR-aaaa07b6", "Банкларни санация қилиш ва тугатиш тўғрисида",
               "2025-03-01")
    text = ("Сенатининг ҚАРОРИ\n\n1. “Банкларни санация қилиш, тугатиш тўғрисида”ги Қонунни амалга ошириш юзасидан "
            "банкларни санация қилиш тартиби кучайтирилсин.")
    q = _manual("QARORLAR-aaaa5a2a", "resolution", "QARORLAR", text, [("П.1", "пункт", "1", "", text.index("1."), len(text))],
                "Банк тизими тўғрисида", "2025-04-01")

    class Pick:
        def cached_json(self, pass_name, inp, schema, verify, prompt):
            ans = {"choice": 0, "evidence": "банкларни санация қилиш тартиби"}
            return ans if verify(ans) is True else None
    reg = ActRegistry.from_records([law, q])
    edges, ctx = extract_all([law, q], reg, llm=Pick())
    assert {e.dst for e in edges if e.src.startswith(q.doc_id)} == {law.doc_id}
    assert [n.node_id for n in reg.external_nodes()] == []                        # no phantom EXT act


# ── E (Task 10): the version digest ignores non-graph config ────────────────────────────────────────
def test_version_digest_ignores_the_store_section_of_graph_yaml(tmp_path):
    from graphrag.kg import build as B
    real = (B.config.ROOT / "config/graph.yaml").read_text(encoding="utf-8")
    assert "\nstore:\n" in real                                   # the Neo4j write allowlist / lock (Task 10)
    (tmp_path / "config").mkdir()
    f = tmp_path / "config/graph.yaml"
    f.write_text(real, encoding="utf-8")
    d0 = B.config_digest(tmp_path)
    f.write_text(real.replace('write_endpoints: ["localhost:7688", "127.0.0.1:7688"]', 'write_endpoints: ["localhost:7699"]')
                 .replace("lock_stale_s: 600", "lock_stale_s: 60"), encoding="utf-8")
    assert B.config_digest(tmp_path) == d0                        # store: only -> same version
    f.write_text(real.replace("theta_sim: 0.65", "theta_sim: 0.66"), encoding="utf-8")
    assert B.config_digest(tmp_path) != d0                        # a graph setting -> new version
    f.write_text(real + "\nextra: 1\n", encoding="utf-8")
    assert B.config_digest(tmp_path) != d0                        # a section after store: still counts


# ── F (Task 8 code): one sample_strata path; the retired R21/R22 raise schedule refuses to run ─────────
def test_sample_strata_without_exclusions_equals_the_exclusion_path():
    import scripts.research_related as R
    from graphrag.kg.schema import GraphEdge
    es = [GraphEdge(f"D-{i:03d}", f"D-{i + 500:03d}", "RELATED", {"weight": round(0.65 + (i % 37) / 100, 6)})
          for i in range(300)]
    cfg = {"near": 30, "top": 30, "near_band": 0.02, "top_share": 0.05}
    a = R.sample_strata(es, 0.65, cfg, 20260928, tag="r23r0")
    b = R.sample_strata(es, 0.65, cfg, 20260928, tag="r23r0", exclude={("X-1", "X-2")})   # excludes no edge
    assert b.pop("excluded_dev_pairs") == 0
    assert a == b


def test_build_related_refuses_the_retired_raise_schedule(monkeypatch, tmp_path):
    import importlib
    import sys as _sys
    from graphrag import config as C
    _sys.path.insert(0, str(C.ROOT / "scripts"))
    br = importlib.import_module("build_related")
    real = C.load

    def fake(name, *a, **k):
        g = real(name, *a, **k)
        if name == "graph":
            g = {**g, "related": {**g["related"], "judge": {**g["related"]["judge"], "mode": "raise"}}}
        return g
    monkeypatch.setattr(C, "load", fake)
    import pytest
    with pytest.raises(SystemExit, match="mode"):
        br.main(["--no-judge", "--out", str(tmp_path)])     # refused before any work / output


# ── G (Task 1): eval_retrieval --out ────────────────────────────────────────────────────────────────
def test_eval_retrieval_out_flag(tmp_path):
    import importlib
    import sys as _sys
    from graphrag import config as C
    _sys.path.insert(0, str(C.ROOT / "scripts"))
    ev = importlib.import_module("eval_retrieval")
    assert ev.out_path(ev.parse_args([]), "grl_chunks__x") == C.path("data/eval/retrieval_grl_chunks__x.json")
    p = tmp_path / "m3" / "baseline.json"
    assert ev.out_path(ev.parse_args(["--out", str(p), "--limit", "5"]), "grl_chunks__x") == p
