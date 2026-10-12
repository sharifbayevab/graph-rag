"""Fable multi diagnosis (M3-R29 follow-up): cross-language evidence in the M3-R28 ablation, long clause values,
Cyrillic number words, speaker names, short Russian quoted titles."""
from graphrag.dataset.facts import Fact, numbers
from graphrag.dataset.generate import Candidate, foreign_names, ru_span, wrong_act
from graphrag.dataset.seeds import Seed
from graphrag.dataset.verify import judge_ok, value_evidence

UZ1 = "Ипак қуртини боқиш фаолиятидан олинган даромадлар 2040 йил 1 январга қадар солиқдан озод этилади ва бошқа имтиёзлар берилади."
UZ2 = "Уй шароитида пилла етиштирувчи жисмоний шахслар 2030 йил 1 январга қадар ижтимоий солиқдан озод қилинади."


def _ru_pair():
    s = Seed("multi-ru", "multi", "ru", ("D/П.1#1", "E/П.2#1"), ("D", "E"), "train", "QONUNLAR")
    return Candidate(s, "До какой даты действуют обе налоговые льготы для шелководов?", "…",
                     [Fact("D/П.1#1@0", "D/П.1#1", "", "2040 йил 1 январга қадар", "q"),
                      Fact("E/П.2#1@0", "E/П.2#1", "", "2030 йил 1 январга қадар", "q")], {})


def test_a_russian_answer_states_uzbek_dates_by_their_numbers():
    f = Fact("x", "x", "", "2040-yil 1-yanvarga qadar", "q")
    assert value_evidence(f, "до 1 января 2040 года", True) is True
    assert value_evidence(f, "до 1 января 2030 года", True) is False
    assert value_evidence(Fact("x", "x", "", "Миграция агентлиги", "q"), "Агентство по миграции", True) is None
    assert value_evidence(Fact("x", "x", "", "Миграция агентлиги", "q"), "бошқа орган", False) is False


def test_cross_language_partial_drop_answers_are_read_by_evidence_not_raw_flags():
    t = {"F1": UZ1, "F2": UZ2}
    full = {"answerable": True, "complete": True, "answer": "Первая — до 1 января 2040 года, вторая — до 1 января 2030 года.",
            "evidence": [{"id": "F1", "quote": "2040 йил 1 январга қадар"}, {"id": "F2", "quote": "2030 йил 1 январга қадар"}]}
    half1 = {"answerable": True, "complete": True, "answer": "до 1 января 2030 года", "evidence": []}
    half2 = {"answerable": True, "complete": True, "answer": "до 1 января 2040 года", "evidence": []}
    ok, rep = judge_ok(_ru_pair(), {"full": full, "drop:F1": half1, "drop:F2": half2}, t)
    assert ok and rep["ablation"] == {"drop:F1": True, "drop:F2": True}     # before: raw "answerable" -> rejected
    both = dict(half1, answer="до 1 января 2040 и 2030 года")
    assert not judge_ok(_ru_pair(), {"full": full, "drop:F1": both, "drop:F2": half2}, t)[0]


def test_a_long_clause_value_counts_in_a_tight_over_cap_quote_but_not_in_a_whole_fragment():
    head = "Қўмита қуйидагиларни белгилайди. "
    val = ("хусусий бандлик агентлигининг таъсисчилари ва раҳбарлари таркибида фирибгарлик, одам савдоси, терроризм "
           "билан боғлиқ жиноятлар учун судланганлиги тугалланмаган шахслар мавжуд эмаслиги ва бундай шахслар билан "
           "шартномалар тузилмаслиги талаби")
    tail = " Бу талаблар " + "барча агентликларга ва уларнинг филиалларига бирдек татбиқ этилади, " * 8
    text = head + val + "." + tail
    s = Seed("single-c", "single", "uz-Cyrl", ("D/СТ.1#1",), ("D",), "train", "QONUNLAR")
    c = Candidate(s, "?", "…", [Fact("D/СТ.1#1@0", "D/СТ.1#1", "", val, "q", vtype="clause")], {})
    cap = (300, 0.5, 200)
    tight = {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": head + val + "."}]}}
    whole = {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": text.strip()}]}}
    assert len(head + val) > 200 and judge_ok(c, tight, {"F1": text}, quote_cap=cap)[0]
    assert not judge_ok(c, whole, {"F1": text}, quote_cap=cap)[0]


def test_cyrillic_number_words_parse_like_latin_ones():
    assert numbers("етти кун") == numbers("yetti kun") == {"7"}
    assert numbers("етмиш беш") == {"75"} and numbers("етти юз") == {"700"}


def test_speaker_from_meta_is_a_known_name():
    fr = [{"act": "Senatning 6-yalpi majlisi 29-30.04.25", "where": "x", "text": "Qonun loyihasi haqida gapirdi.",
           "speaker": "Erkin Fayziyevich Gadoyev", "session": {"title": "Senatning 6-yalpi majlisi 29-30.04.25"}}]
    assert foreign_names("2025-yil 29-apreldagi majlisda senator Erkin Gadoyev nima dedi?", fr, "uz-Latn") == []
    assert foreign_names("2025-yil 29-apreldagi majlisda senator Karimov nima dedi?", fr, "uz-Latn") == ["Karimov"]


def test_short_russian_titles_in_quotes_are_translations():
    assert ru_span("О банках") and ru_span("О СМИ") and ru_span("О статусе депутата")
    assert not ru_span("Давлат божи") and not ru_span("Банклар тўғрисида") and not ru_span("Moliya")
    fr = [{"act": "Банклар ва банк фаолияти тўғрисида", "where": "x", "text": "Банк лицензияси бекор қилинади."}]
    assert foreign_names("Что предусматривает Закон «О банках и банковской деятельности»?", fr, "ru") == []
    assert not wrong_act("Что предусматривает Закон «О банках и банковской деятельности»?", fr, "ru")
