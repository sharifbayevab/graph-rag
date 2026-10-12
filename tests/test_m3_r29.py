"""M3-R29: instructions v3 checks -- blanked-sentence near-copy, act identity through the fragment text, dated
plenary sessions; packet session / speaker meta; Opus second-reader sample."""
from graphrag import config
from graphrag.dataset import opus
from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import check_item_ex, near_copy_ratio, session_days, session_undated, wrong_act
from graphrag.dataset.seeds import Seed

TXT = "Qonunchilik palatasi tomonidan 2025-yil 18-fevralda qabul qilingan Qonun Senatga 2025-yil 27-fevral kuni koʻrib chiqish uchun kelib tushgan."
FR = [{"id": "F1", "act": "Davlat ijtimoiy sugʻurtasi toʻgʻrisida", "where": "Davlat ijtimoiy sugʻurtasi toʻgʻrisida, blok 1",
       "text": TXT}]


def test_blanked_sentence_scores_high_and_a_paraphrase_low():
    assert near_copy_ratio("Qonunchilik palatasi qabul qilgan Qonun Senatga koʻrib chiqish uchun qachon kelib tushgan?", FR) >= 0.8
    assert near_copy_ratio("Senat bu qonunni qachondan boshlab koʻrib chiqa boshladi?", FR) < 0.5
    cfg = config.load("train")["dataset"]["near_copy"]
    assert cfg["max_ratio"] == 0.7222 and cfg["n"] == 680


def _check(q, a, frags, cfg=None):
    s = Seed("single-x", "single", "uz-Latn", ("D/БЛ.1#1",), ("D",), "train", "XULOSALAR")
    f = Fact("D/БЛ.1#1@0", "D/БЛ.1#1", "", "2025-yil 27-fevral", "2025-yil 27-fevral kuni koʻrib chiqish")
    return check_item_ex({"question": q, "answer": a, "used": ["F1.0"]}, s, {"F1.0": f}, frags,
                         cfg or {"max_copy_words": 99, "max_lexical_overlap": 1.0,
                                 "near_copy": {"max_ratio": 0.7222}})[0]


def test_near_copy_gate_in_the_item_check():
    a = "Qonun Senatga 2025-yil 27-fevral kuni kelib tushgan."
    assert _check("Qonunchilik palatasi qabul qilgan Qonun Senatga koʻrib chiqish uchun qachon kelib tushgan?", a, FR) == "near_copy"
    assert _check("Senat bu qonunni qachondan boshlab koʻrib chiqa boshladi?", a, FR) is None


def test_amended_act_counts_only_if_the_fragment_text_names_it():
    fr = [{"act": "Айрим қонун ҳужжатларига ўзгартиш киритиш тўғрисида", "where": "x",
           "text": "Божхона кодексининг 347-моддаси ўзгартирилсин", "amended": ["Божхона кодекси"]}]
    hidden = [dict(fr[0], text="ушбу Кодекснинг 347-моддаси ўзгартирилсин")]
    q = "Божхона кодекси 347-моддасига кўра муддат қанча?"
    assert not wrong_act(q, fr, "uz-Cyrl") and wrong_act(q, hidden, "uz-Cyrl")
    own = [{"act": "Банклар ва банк фаолияти тўғрисида", "where": "x", "text": "t", "amended": []}]
    assert not wrong_act("Банклар ва банк фаолияти тўғрисидаги Қонунга кўра ким?", own, "uz-Cyrl")
    assert wrong_act("Давлат божи тўғрисидаги Қонунга кўра ким?", own, "uz-Cyrl")
    assert not wrong_act("Санация қилиш тўғрисида қарор ким қабул қилади?", own, "uz-Cyrl")   # "a decision", no title


def test_stenogram_question_must_carry_the_session_date():
    s = {"date": "2025-04-29", "title": "Senatning 6-yalpi majlisi 29-30.04.25"}
    assert session_days(s) == ("2025", "4", {"29", "30"})
    fr = [{"session": s}]
    assert not session_undated("2025-yil 30-apreldagi yalpi majlisda senator Gadoyev nima dedi?", fr)
    assert not session_undated("Что сказал сенатор на пленарном заседании 29 апреля 2025 года?", fr)
    assert session_undated("Senat yalpi majlisida senator Gadoyev nima dedi?", fr)
    assert session_undated("2025-yil 12-maydagi majlisda nima deyildi?", fr)                  # another day
    assert not session_undated("Nima dedi?", [{"act": "x"}])                                  # not a transcript


def test_session_info_from_record_meta_and_speech_vertex():
    N = lambda i, t: type("N", (), {"node_id": i, "title": t})()
    M = lambda sec: type("M", (), {"section": sec, "date": "2025-04-29", "title": "Senatning 6-yalpi majlisi 29-30.04.25"})()
    recs = [type("R", (), {"doc_id": "STENOGRAMMA-a", "meta": M("STENOGRAMMA"),
                           "structure": [N("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2", "Erkin Gadoyev")]})(),
            type("R", (), {"doc_id": "QONUNLAR-b", "meta": M("QONUNLAR"), "structure": []})()]
    of = opus.session_info(recs)
    assert of("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#3") == {"session": {"date": "2025-04-29",
                                                              "title": "Senatning 6-yalpi majlisi 29-30.04.25",
                                                              "dates": ["2025-04-29", "2025-04-30"],
                                                              "range": "2025-04-29..2025-04-30"},
                                                  "speaker": "Erkin Gadoyev"}
    assert of("QONUNLAR-b/СТ.1#1") is None


def test_second_reader_sample_is_deterministic_and_about_the_fraction():
    from graphrag.dataset.generate import Candidate
    cs = [Candidate(Seed(f"single-{i:05d}", "single", "ru", (f"D/П.{i}#1",), ("D",), "train", "QONUNLAR"), f"В{i}?", "О.",
                    [], {"generate": {"role": "opus-generator"}}) for i in range(1000)]
    a, b = opus.second_sample(cs, 0.2, 7), opus.second_sample(list(reversed(cs)), 0.2, 7)
    assert {c.seed.seed_id for c in a} == {c.seed.seed_id for c in b} and 150 < len(a) < 250
    frag = lambda cid, lab: {"id": lab, "act": "A", "where": "W", "text": "T"}
    pk = opus.make_review_packets(a, frag, {"path": "r", "version": 1, "digest": "d", "text": "t"}, 40, 7, {},
                                  role=opus.ROLE_SECOND, prefix="second-", out_dir="v")
    assert pk[0]["packet_id"] == "second-review-001" and pk[0]["role"] == "strict-reader-2"
    again = opus.make_review_packets(a, frag, {"path": "r", "version": 1, "digest": "d", "text": "t"}, 40, 7,
                                     {p["packet_id"]: [] for p in pk}, role=opus.ROLE_SECOND, prefix="second-")
    assert again[0]["packet_id"] == f"second-review-{len(pk) + 1:03d}"


def test_instructions_v3_model_neutral_and_pinned():
    o = config.load("train")["dataset"]["opus"]
    t = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["generator"]["version"] >= 3 and o["generator_history"][o["generator"]["version"]] == o["generator"]["digest"]
    for s in ("apply to ANY generator", "no blanked-sentence copies", "named in the fragment", "Plenary transcripts",
              "session date", "Render every Latin-script name"):
        assert s in t, s
