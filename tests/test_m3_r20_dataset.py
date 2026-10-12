"""M3-R20 (user decision after research M3-R19): regression tests built from the Task 17 pilot's real outputs.
The research's false rejections must now pass; its genuine failures must still fail. Strings are taken (shortened)
from the cached ds_facts / ds_generate answers and the corpus fragments they came from."""
import json

import pytest

from graphrag.dataset.facts import Fact, check_fact, snap, snap_fact, value_type
from graphrag.dataset.generate import (article_without_act, check_item, check_item_ex, is_compound, redundant_fragment,
                                       value_status)
from graphrag.dataset.seeds import HOP, PATTERNS, Seed
from graphrag.dataset.walks import anchored, quality_walks, walk_class
from graphrag.kg.llm import KgLLM

CFG = {"max_copy_words": 6, "max_lexical_overlap": 0.6}


# ---- R1 ds_facts: snap-to-source -----------------------------------------------------------------------------------------
def test_inflected_value_is_snapped_to_the_source_span():
    text = "Комиссия субсидиялаш ҳажмларига доир таклифларни тайёрлайди ҳамда уларни Парламент комиссиясига киритади."
    f = {"statement": "s", "value": "Парламент комиссияси",
         "quote": "субсидиялаш ҳажмларига доир таклифларни тайёрлайди ҳамда уларни Парламент комиссиясига киритади."}
    assert check_fact(f, text, set()) == "value_not_in_quote"                  # the Task 17 rejection (277 such)
    g, r = snap_fact(f, text, set())
    assert r is None and g["value"] == "Парламент комиссиясига" and g["value"] in text and g["quote"] in text


def test_quote_retyped_over_mixed_script_source_is_snapped_verbatim():
    text = "Qonun jarimalarni ijro etish tizimini yanada soddalashtirishга хизмат қилади."
    f = {"statement": "s", "value": "jarimalarni ijro etish",
         "quote": "jarimalarni ijro etish tizimini yanada soddalashtirishga xizmat qiladi."}
    assert check_fact(f, text, set()) == "quote_not_verbatim"
    g, r = snap_fact(f, text, set())
    assert r is None and g["quote"] == "jarimalarni ijro etish tizimini yanada soddalashtirishга хизмат қилади"
    assert g["quote"] in text                                                  # stored = the source substring


def test_genuine_fact_failures_still_fail():
    text = ("Икки нафари Ҳиндистонга юборилди. Бошқалари эса Ишчи гуруҳ таркибида мазкур давлатларга хизмат "
            "сафарида бўлишди.")
    elided = {"statement": "s", "value": "Ишчи гуруҳ таркибида", "quote": "Икки нафари ... Ишчи гуруҳ таркибида"}
    assert snap_fact(elided, text, set())[1] == "quote_not_verbatim"          # "…" joins two spans
    t2 = "мансабдор шахсларга эса – ўн бараваригача миқдорда жарима солишга сабаб бўлади."
    reworded = {"statement": "s", "value": "базавий ҳисоблаш миқдорининг ўн бараваригача",
                "quote": "мансабдор шахсларга эса – ўн бараваригача миқдорда жарима"}
    assert snap_fact(reworded, t2, set())[1] == "value_not_in_quote"          # words that are not in the quote
    t3 = "Банк 15 кун ичида ҳисобот тақдим этади."
    assert snap_fact({"statement": "s", "value": "5 кун", "quote": "Банк 15 кун ичида ҳисобот"}, t3, set())[1] \
        == "value_not_in_quote"                                               # another number is never snapped
    assert snap("5 кун", t3, lemma=True) is None


def test_value_types():
    assert value_type("ўттиз кун") == "number" and value_type("2023-yil 10-fevral") == "number"
    assert value_type("Марказий банк") == "name" and value_type("мажбурийдир") == "term"
    assert value_type("нодавлат нотижорат ташкилотлари, оммавий ахборот воситалари ва кенг жамоатчилик билан доимий "
                      "ҳамкорликнинг йўлга қўйилишига алоҳида эътибор қаратсин") == "clause"


# ---- R2 ds_generate checks ----------------------------------------------------------------------------------------------
def _single(lang, cid="D/П.1#1", sec="STENOGRAMMA"):
    return Seed(f"single-{abs(hash((lang, cid))) % 10**10:010d}", "single", lang, (cid,), ("D",), "train", sec)


def _fact(cid, value, quote=None, i=0):
    return Fact(f"{cid}@{i}", cid, "s", value, quote or value, "", "", value_type(value))


def _frag(text, act="Senatning 2-yalpi majlisi", where="Senatning 2-yalpi majlisi (2024-yil 18-19- va 21-dekabr)"):
    return [{"id": "F1", "act": act, "where": where, "text": text}]


def test_ru_answer_translating_an_uzbek_number_value_passes_numbers_by_code_words_by_judge():
    t = "Oʼtgan uch yil ichida 104 mingta oila uy-joy bilan taʼminlandi."
    f = {"F1.0": _fact("D/П.1#1", "104 mingta oila")}
    it = {"key": "k1", "used": ["F1.0"],
          "question": "Сколько семей получили жильё за последние три года согласно обсуждению на 2-й пленарной сессии "
                      "Сената (18-19 и 21 декабря 2024 года)?",
          "answer": "За последние три года жильём были обеспечены 104 тысячи семей."}
    r, info = check_item_ex(it, _single("ru"), f, _frag(t), CFG)
    assert r is None and info["status"] == {"D/П.1#1@0": "num+judge"}          # Task 17: value_missing_in_answer
    wrong = dict(it, answer="За последние три года жильём были обеспечены 105 тысяч семей.")
    assert check_item(wrong, _single("ru"), f, _frag(t), CFG) in ("value_missing_in_answer", "unsupported_number")


def test_ru_answer_translating_a_term_goes_to_the_judge_quoting_it_is_code():
    t = "Тафтиш инстанцияси судининг ажримида баён қилинган кўрсатмалар мазкур ишни янгидан кўраётган суд учун мажбурийдир."
    f = {"F1.0": _fact("D/П.1#1", "мажбурийдир")}
    act = "Фуқаролик процессуал кодексига ўзгартириш ва қўшимчалар киритиш тўғрисида"
    it = {"key": "k1", "used": ["F1.0"],
          "question": "Имеют ли обязательную силу указания суда надзорной инстанции для суда, вновь рассматривающего "
                      "дело, по Гражданскому процессуальному кодексу?",
          "answer": "Указания, изложенные в определении суда надзорной инстанции, являются обязательными для суда."}
    r, info = check_item_ex(it, _single("ru", sec="QONUNLAR"), f, _frag(t, act, act), CFG)
    assert r is None and info["status"] == {"D/П.1#1@0": "judge"}
    q = dict(it, answer="Указания суда надзорной инстанции являются обязательными («мажбурийдир»).")
    assert check_item_ex(q, _single("ru", sec="QONUNLAR"), f, _frag(t, act, act), CFG)[1]["status"] == \
        {"D/П.1#1@0": "code"}


def test_a_list_value_joined_by_va_instead_of_a_comma_is_stated():
    f = _fact("D/П.1#1", "Transport vazirligi, Avtomobil yoʼllari qoʼmitasi")
    assert value_status(f, "Masʼul tashkilotlar Transport vazirligi va Avtomobil yoʻllari qoʻmitasi hisoblanadi.",
                        "uz-Latn", "uz-Latn") == "code"


def test_typed_value_not_stated_in_the_same_language_still_fails():
    f = _fact("D/П.1#1", "Transport vazirligi")
    assert value_status(f, "Masʼul tashkilot Qurilish vazirligi hisoblanadi.", "uz-Latn", "uz-Latn") is None


def test_clause_value_paraphrased_is_judge_checked():
    v = ("нодавлат нотижорат ташкилотлари, оммавий ахборот воситалари ва кенг жамоатчилик билан доимий "
         "ҳамкорликнинг йўлга қўйилишига алоҳида эътибор қаратсин")
    f = _fact("D/П.2#3", v)
    a = ("Омбудсман нодавлат нотижорат ташкилотлари, оммавий ахборот воситалари ва кенг жамоатчилик билан доимий "
         "ҳамкорликни йўлга қўйишга алоҳида эътибор қаратиши лозим.")
    assert value_status(f, a, "uz-Cyrl", "uz-Cyrl") == "judge"               # Task 17: value_missing_in_answer


def test_compound_questions_are_rejected():
    assert is_compound("В каком количестве статей состоит Закон, принятый 18 февраля 2025 года, и какие "
                       "законодательные акты он изменяет?")                     # pilot single-4e217bfe37
    assert is_compound("Банк тўловга қобилиятсиз деб қайси муддатда ҳисобланади ва бу ҳолат қайси муддат ичида юзага "
                       "келиши эҳтимоли юқори деб топилади?")                  # refers_internal-2480593f09
    assert not is_compound("Qaysi organ sanatsiya toʼgʼrisida qaror qabul qiladi?")
    assert not is_compound("Что изменили законы 2023 и 2025 годов в части 4 статьи 279 кодекса?")


def test_article_number_without_a_distinctive_act_title_is_rejected():
    omni = "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш тўғрисида (2025-06-17)"
    q = (f"«{omni}» 2025 йил 17 июнда қабул қилинган қонунга мувофиқ, 21-модданинг учинчи қисмида қайси органнинг номи "
         "бошқа орган номи билан алмаштирилди?")                               # pilot single-14b3f94dbd
    assert article_without_act(q, [{"act": omni}])
    named = "«Божхона кодекси»га кўра 382¹-моддада ҳуқуқ эгаси қайси муддатда хабардор қилинади?"
    assert not article_without_act(named, [{"act": omni}])
    act = "Банкларни санация қилиш ва тугатиш тўғрисида"
    assert not article_without_act(f"«{act}»ги Қонуннинг 64-моддасига кўра учинчи навбатда қайси талаблар "
                                   "қондирилади?", [{"act": act}])
    assert not article_without_act("Омбудсман кимлар билан ҳамкорлик қилиши лозим?", [{"act": omni}])


ACT = "Банкларни санация қилиш ва тугатиш тўғрисида"
TA = ("Санация қилинаётган банкнинг мажбуриятлари ушбу Қонуннинг 64-моддасида белгиланган кредиторларнинг "
      "талабларини қаноатлантириш навбатида олувчига ўтказилади.")
TB = ("64-модда. Кредиторларнинг талаблари қуйидаги навбатда қаноатлантирилади: учинчи навбатда – Омонатларни "
      "кафолатлаш агентлигига ўтган талаблар.")
BRIDGE = Seed("multi-0000000001", "multi", "uz-Cyrl", ("B/СТ.24/Ч.3#2", "B/СТ.64/Ч.3#2"), ("B",), "train", "QONUNLAR",
              (("B/СТ.24/Ч.3", "REFERS_TO", "B/СТ.64"),))
BF = {"F1.0": _fact("B/СТ.24/Ч.3#2", "кредиторларнинг талабларини қаноатлантириш навбатида"),
      "F2.0": _fact("B/СТ.64/Ч.3#2", "учинчи навбатда")}
BFR = [{"id": "F1", "act": ACT, "where": f"{ACT}, ст. 24", "text": TA},
       {"id": "F2", "act": ACT, "where": f"{ACT}, ст. 64", "text": TB}]


def test_bridge_path_fact_may_be_in_the_question_the_answer_fact_may_not():
    q = (f"«{ACT}»ги Қонунда банк мажбуриятлари кредиторларнинг талабларини қаноатлантириш навбатида ўтказилишини "
         "назарда тутган модда бўйича Омонатларни кафолатлаш агентлигининг талаблари нечанчи ўринда қондирилади?")
    it = {"key": "k1", "question": q, "answer": "Улар учинчи навбатда қаноатлантирилади.", "used": ["F1.0", "F2.0"]}
    r, info = check_item_ex(it, BRIDGE, BF, BFR, dict(CFG, max_copy_words=99, max_lexical_overlap=1.0))
    assert r is None and [f.fact_id for f in info["path"]] == ["B/СТ.24/Ч.3#2@0"]
    leak = dict(it, question=q.replace("нечанчи ўринда", "учинчи навбатда"))
    assert check_item(leak, BRIDGE, BF, BFR, dict(CFG, max_copy_words=99, max_lexical_overlap=1.0)) \
        == "answer_in_question"


def test_redundant_pairs_are_rejected_before_and_after_generation():
    law = "Якуний ҳужжат давлат органига келиб тушган кундан эътиборан бир ойгача бўлган муддатда кўриб чиқилади."
    concl = ("“Jamoatchilik nazorati toʼgʼrisida”gi Qonunga natijalar davlat organiga kelib tushgan kundan eʼtiboran "
             "bir oygacha boʼlgan muddatda koʼrib chiqilishi belgilanmoqda.")
    s = Seed("multi-0000000002", "multi", "ru", ("X/БЛ.6#1", "Q/СТ.7/Ч.1#1"), ("X", "Q"), "train", "XULOSALAR",
             (("X/БЛ.6", "REFERS_TO", "EXT/jamoatchilik nazorat/СТ.17"), ("Q/СТ.7/Ч.1", "REFERS_TO",
                                                                            "EXT/jamoatchilik nazorat/СТ.17")))
    facts = {"X/БЛ.6#1": [_fact("X/БЛ.6#1", "bir oygacha boʼlgan muddatda")],
             "Q/СТ.7/Ч.1#1": [_fact("Q/СТ.7/Ч.1#1", "бир ойгача бўлган муддатда")]}
    assert redundant_fragment(s, facts, {"X/БЛ.6#1": concl, "Q/СТ.7/Ч.1#1": law})   # pilot refers_cross-0f0e72f62c
    fl = {"F1.0": facts["X/БЛ.6#1"][0], "F2.0": facts["Q/СТ.7/Ч.1#1"][0]}
    fr = [{"id": "F1", "act": "Jamoatchilik nazorati", "where": "x", "text": concl},
          {"id": "F2", "act": "Жамоатчилик назорати", "where": "y", "text": law}]
    it = {"key": "k1", "used": ["F1.0", "F2.0"], "question": "В какой срок госорган рассматривает итоги общественного "
          "контроля по Закону «Жамоатчилик назорати тўғрисида»?",
          "answer": "В течение одного месяца («бир ойгача бўлган муддатда»)."}
    assert check_item(it, s, fl, fr, CFG) == "redundant"


def test_worded_date_month_in_the_question_is_supported():
    t = "Moyra Nabiyevna Toshova: Oʼtgan uch yil ichida 104 mingta oila uy-joy bilan taʼminlandi."
    it = {"key": "k1", "used": ["F1.0"], "question": "Сколько семей обеспечено жильём за три года, по словам Тошовой "
          "на заседании Сената 21 декабря 2024 года?", "answer": "104 тысячи семей."}
    assert check_item(it, _single("ru"), {"F1.0": _fact("D/П.1#1", "104 mingta oila")}, _frag(t), CFG) is None
    bad = dict(it, question=it["question"].replace("21 декабря", "22 декабря"))
    assert check_item(bad, _single("ru"), {"F1.0": _fact("D/П.1#1", "104 mingta oila")}, _frag(t), CFG) \
        == "unsupported_number"


def test_research_replay_multi_hop_glue_and_wrong_script_still_fail():
    """Genuine Task 17 failures that stay rejected: a question in the wrong script (uz-Latn target written in
    Cyrillic, pilot single-2c142bd127)."""
    t = "Мазкур тадбирга ушбу тумандаги 10 дан ортиқ МФЙлар жалб қилинди."
    it = {"key": "k1", "used": ["F1.0"], "question": "Косон туманидаги тадбирда қанча маҳалла иштирок этган?",
          "answer": "Тадбирда 10 дан ортиқ МФЙлар иштирок этган."}
    assert check_item(it, _single("uz-Latn"), {"F1.0": _fact("D/П.1#1", "10 дан ортиқ МФЙлар")},
                      _frag(t, "Ombudsman", "Ombudsman"), CFG) == "language"


# ---- R4 walks ------------------------------------------------------------------------------------------------------------
def test_walk_classes():
    assert walk_class((("B/СТ.24/Ч.3", "REFERS_TO", "B/СТ.64"),)) == "bridge"
    assert walk_class((("L/СТ.1/Ч.1/П.5", "AMENDS", "EXT/mjk/СТ.245/Ч.3"),
                       ("M/СТ.1/Ч.1/П.2", "AMENDS", "EXT/mjk/СТ.245/Ч.3"))) == "aggregation"
    assert walk_class((("S/ВОПР.5/ВЫСТ.2", "REFERS_TO", "EXT/soliq kodeks"),
                       ("T/ВОПР.5/ВЫСТ.2", "REFERS_TO", "EXT/soliq kodeks"))) == "act"      # topic-only link


def test_anchoring_drops_sibling_padding():
    hops = (("L/СТ.1/Ч.1/П.41", "REFERS_TO", "EXT/bojxona kodeks/СТ.382"),
            ("EXT/bojxona kodeks/СТ.382", "CONTAINS", "EXT/bojxona kodeks/СТ.382/Ч.2"),
            ("L/СТ.1/Ч.1/П.40", "AMENDS", "EXT/bojxona kodeks/СТ.382/Ч.2"))
    text = {"L/СТ.1/Ч.1/П.41#42": "41) 382¹ ва 382²-моддалар билан тўлдирилсин",
            "L/СТ.1/Ч.1#41": "39) 370-модда қуйидаги таҳрирда баён этилсин: дастлабки қарор"}   # pilot amends-249681791b
    assert not anchored(tuple(text), hops, text.get)
    text["L/СТ.1/Ч.1#41"] += "; 40) 382-модданинг иккинчи қисми"
    assert anchored(tuple(text), hops, text.get)


def test_quality_walks_drop_act_level_and_procedural_hub_provisions():
    w_ok = (("B/СТ.24/Ч.3#2", "B/СТ.64/Ч.3#2"), ("B",), (("B/СТ.24/Ч.3", "REFERS_TO", "B/СТ.64"),))
    w_act = (("S/ВЫСТ.2#1", "T/ВЫСТ.2#1"), ("S", "T"), (("S/ВЫСТ.2", "REFERS_TO", "EXT/x"),
                                                         ("T/ВЫСТ.2", "REFERS_TO", "EXT/x")))
    w_hub = (("X/БЛ.4#1", "Y/БЛ.3#1"), ("X", "Y"), (("X/БЛ.4", "REFERS_TO", "EXT/senat/СТ.23/Ч.2"),
                                                     ("Y/БЛ.3", "REFERS_TO", "EXT/senat/СТ.23/Ч.2")))
    text = {"B/СТ.24/Ч.3#2": TA, "B/СТ.64/Ч.3#2": TB, "S/ВЫСТ.2#1": "a", "T/ВЫСТ.2#1": "b",
            "X/БЛ.4#1": "23-моддаси", "Y/БЛ.3#1": "23-моддаси"}
    out, rep = quality_walks({"refers_internal": [w_ok], "refers_cross": [w_act, w_hub]}, text.get,
                             {"EXT/senat/СТ.23/Ч.2": 40})
    assert out == [w_ok] and rep["refers_cross|aggregation-hub|dropped"] == 1 and rep["refers_cross|act|dropped"] == 1


def test_one_multi_pattern_and_quotas():
    from graphrag import config
    ds = config.load("train")["dataset"]
    assert PATTERNS == ("single", "multi") and HOP["multi"] == "multi"
    assert ds["hops"] == {"single": 0.90, "multi": 0.10} and ds["langs"] == {"uz-Cyrl": 0.45, "uz-Latn": 0.25, "ru": 0.30}
    assert ds["gen_batch"] == 6 and "transfer" not in ds


# ---- R3 agreement pass ---------------------------------------------------------------------------------------------------
def _client(fn):
    calls = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    body = json.dumps(fn(json.loads(kw["messages"][1]["content"])))
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})
    return Client(), calls


def test_agreement_pass_rejects_disagreeing_translations_and_replays_offline(tmp_path):
    from graphrag.dataset.generate import Candidate
    from graphrag.dataset.verify import agree_candidates, needs_agreement
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 10, "passes": {"ds_verify": {"prompt_version": 4,
                                                                                         "max_calls": 10}}}}

    def cand(i, st):
        s = _single("ru", f"D/П.{i}#1")
        f = _fact(s.chunks[0], "мажбурийдир")
        return Candidate(s, f"Вопрос {i}?", f"Ответ {i}.", [f], {"value_checks": {f.fact_id: st}})
    cs = [cand(1, "judge"), cand(2, "code"), cand(3, "num+judge")]
    assert [needs_agreement(c) for c in cs] == [True, False, True]
    cl, calls = _client(lambda u: {"items": [{"key": x["key"], "agrees": x["answer"] != "Ответ 3.", "reason": "r"}
                                             for x in u["items"]]})
    kept, st = agree_candidates(cs, KgLLM(cfg, cache_root=tmp_path, client=cl, allow_api=True), 10)
    assert [c.answer for c in kept] == ["Ответ 1.", "Ответ 2."] and len(calls) == 1
    assert st["rejected"] == {"single|disagree": 1} and st["checked"] == 2
    assert "Ответ 1." in calls[0]["messages"][1]["content"]                  # the answer is shown only here
    off = KgLLM(cfg, cache_root=tmp_path, allow_api=False)
    kept2, _ = agree_candidates([cand(1, "judge"), cand(2, "code"), cand(3, "num+judge")], off, 10)
    assert [c.answer for c in kept2] == ["Ответ 1.", "Ответ 2."] and off.total_api_calls() == 0


def test_agreement_pass_respects_the_whole_ledger_cap(tmp_path):
    from graphrag.dataset.generate import Candidate
    from graphrag.dataset.verify import agree_candidates
    from graphrag.kg.llm import KgBudgetExceeded
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 5, "passes": {"ds_verify": {"prompt_version": 4,
                                                                                        "max_calls": 10}}}}
    (tmp_path / "ledger.json").write_text(json.dumps({"ds_facts": {"d": 5}}), encoding="utf-8")
    s = _single("ru")
    c = Candidate(s, "В?", "О.", [_fact(s.chunks[0], "мажбурийдир")], {"value_checks": {f"{s.chunks[0]}@0": "judge"}})
    cl, calls = _client(lambda u: {"items": []})
    with pytest.raises(KgBudgetExceeded):
        agree_candidates([c], KgLLM(cfg, cache_root=tmp_path, client=cl, allow_api=True), 10)
    assert calls == []


# ---- replay of the real Task 17 ds_facts cache (0 calls) -----------------------------------------------------------------
@pytest.mark.corpus
def test_task17_facts_cache_replays_with_snapping_and_rescues_no_fact_seeds():
    from collections import defaultdict
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.dataset.facts import extract
    from graphrag.dataset.seeds import block_of
    from graphrag.dataset.sources import SourceIndex
    t = config.load("train")
    recs = read_records(config.path(config.load("data")["records_dir"]))
    chunks = {c.chunk_id: c for r in recs for c in r.chunks}
    blk = block_of(recs, 8)
    bc = defaultdict(list)
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            bc[blk[c.chunk_id]].append(c.chunk_id)
    seeds = [Seed.from_json(json.loads(x)) for x in open(config.path("data/dataset/pilot/seeds.jsonl"))]
    llm = KgLLM(cfg={"llm": t["dataset_llm"]}, cache_root=t["dataset_llm"]["cache_dir"], allow_api=False)
    facts, st = extract({blk[c] for s in seeds for c in s.chunks}, bc, chunks, SourceIndex(recs), llm, 6)
    assert llm.total_api_calls() == 0 and st["blocks"] == 140
    assert st["facts"] == 2391 and st["rejected"]["snapped"] == 292          # Task 17: 2103 facts
    assert sum(not all(facts.get(c) for c in s.chunks) for s in seeds) == 15  # Task 17: 23 no_facts seeds
    from graphrag.dataset.facts import ctoks, find_seq
    from graphrag.eval.gold import wsn
    for fs in facts.values():                                                 # every stored quote / value is verbatim
        for f in fs:
            assert wsn(f.quote) in wsn(chunks[f.chunk_id].text) and find_seq(ctoks(f.quote), ctoks(f.value), lemma=False)
