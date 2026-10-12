"""M3-R30 / M3-R31: ds_verify v6 quoting rules, dual-reader consensus, reader packet meta, instructions v4."""
import json

from graphrag import config
from graphrag.dataset import opus
from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import Candidate, session_undated
from graphrag.dataset.seeds import Seed
from graphrag.dataset.verify import PROMPT_VER, judge_ok, quote_ok


def test_v6_prompt_one_verbatim_quote_per_value_and_elisions_never_count():
    assert "ONE quote per needed value" in PROMPT_VER and "at most 300 characters" in PROMPT_VER
    assert "never shortened with" in PROMPT_VER and "fragment's own script" in PROMPT_VER
    assert config.load("train")["dataset_llm"]["passes"]["ds_verify"]["prompt_version"] == 6
    t = "суд шикоят келиб тушган кундан эътиборан беш кунлик муддатда уни кассация инстанцияси судига юбориши шарт"
    assert quote_ok("беш кунлик муддатда", t) and not quote_ok("суд шикоят … беш кунлик муддатда", t)
    assert not quote_ok("суд шикоят ... кассация инстанцияси", t)
    s = Seed("single-x", "single", "ru", ("D/1",), ("D",), "train", "QONUNLAR")
    c = Candidate(s, "?", "…", [Fact("D/1@0", "D/1", "", "беш кунлик муддатда", "q")], {})
    assert not judge_ok(c, {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": "шикоят … беш кунлик муддатда"}]}},
                        {"F1": t})[0]


def _c(i, split="train", lang="ru"):
    return Candidate(Seed(f"single-{i:03d}", "single", lang, (f"D/П.{i}#1",), ("D",), split, "QONUNLAR"), f"В{i}?",
                     "О.", [], {"generate": {"role": "opus-generator"}})


def test_consensus_needs_both_readers_good_and_logs_disagreements():
    cs = [_c(i) for i in range(5)]
    pid = [opus.pair_id(c.seed.seed_id) for c in cs]
    v = lambda verdict, *crit: {"verdict": verdict, "criteria": list(crit)}
    a = {pid[0]: v("GOOD"), pid[1]: v("GOOD"), pid[2]: v("BAD", "c1"), pid[3]: v("GOOD")}
    b = {pid[0]: v("GOOD"), pid[1]: v("BORDERLINE", "c5"), pid[2]: v("BAD", "c4"), pid[4]: v("GOOD")}
    good, rej, st = opus.consensus(cs, a, b)
    assert [c.seed.seed_id for c in good] == ["single-000"] and good[0].provenance["review"]["consensus"]
    assert st["pending"] == 2 and {x["pair_id"] for x in st["pending_pairs"]} == {pid[3], pid[4]}
    assert st["agreement"] == {"BAD|BAD": 1, "GOOD|BORDERLINE": 1, "GOOD|GOOD": 1}
    assert st["disagreements_by_cell"] == {"train|single|ru": {"GOOD|BORDERLINE": 1}}
    assert st["disagreements_by_defect"] == {"B|c5": 1} and {r["reason"] for r in rej} == {"no_consensus"}


def test_ingest_verdicts_accepts_any_committed_rubric_version_and_reports_details():
    c = _c(1)
    rub = {"path": "r", "version": 1, "digest": "d1", "text": "t"}
    rub["digest"] = opus.sha256_text("t")
    pk = {"packet_id": "review-001", "rubric": rub, "role": "strict-reader", "items": [{"pair_id": opus.pair_id(c.seed.seed_id)}]}
    line = json.dumps({"pair_id": opus.pair_id(c.seed.seed_id), "verdict": "BORDERLINE",
                       "reasons": [{"criterion": "c7", "note": "x"}]})
    _, _, st = opus.ingest_verdicts([pk], {"review-001": line}, [c], {rub["digest"], "other"})
    d = st["verdict_of"][opus.pair_id(c.seed.seed_id)]
    assert d["verdict"] == "BORDERLINE" and d["criteria"] == ["c7"] and d["role"] == "strict-reader"


def test_reader_items_carry_record_meta_and_never_a_model_or_notes():
    N = lambda i, t: type("N", (), {"node_id": i, "title": t})()
    M = type("M", (), {"section": "STENOGRAMMA", "date": "2025-04-29", "title": "Senatning 6-yalpi majlisi 29-30.04.25"})()
    recs = [type("R", (), {"doc_id": "STENOGRAMMA-a", "meta": M,
                           "structure": [N("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2", "Erkin Gadoyev")]})()]
    meta = opus.reader_meta(recs)("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#1")
    assert meta["session"]["dates"] == ["2025-04-29", "2025-04-30"] and meta["speaker"] == "Erkin Gadoyev"
    assert meta["doc_type"] == "plenary transcript" and meta["date"] == "2025-04-29"
    s = Seed("single-s", "single", "uz-Latn", ("STENOGRAMMA-a/ВОПР.1/ВЫСТ.2#1",), ("STENOGRAMMA-a",), "train",
             "STENOGRAMMA")
    c = Candidate(s, "Q?", "A.", [], {"generate": {"role": "opus-generator", "model": "claude-fable-5-1"},
                                      "opus": {"notes": "SECRET-NOTE", "agent": "fable-7"}})
    frag = lambda cid, lab: {"id": lab, "act": "Senatning 6-yalpi majlisi", "where": "W", "text": "T"}
    rub = {"path": "config/dataset_reader_rubric.md", "version": 2, "digest": "d", "text": "t"}
    for role, prefix in ((opus.ROLE_READER, ""), (opus.ROLE_SECOND, "second-")):
        pk = opus.make_review_packets([c], frag, rub, 40, 0, {}, role=role, prefix=prefix,
                                      meta_of=opus.reader_meta(recs))
        blob = json.dumps(pk, ensure_ascii=False).lower()
        assert "speaker" in blob and "2025-04-30" in blob
        for bad in ("opus", "fable", "claude", "gpt", "secret-note", "notes"):
            assert bad not in blob, bad


def test_rubric_v2_meta_rule_and_instructions_v4():
    o = config.load("train")["dataset"]["opus"]
    r = config.path(o["reader"]["file"]).read_text(encoding="utf-8")
    assert o["reader"]["version"] >= 2 and o["reader_history"][o["reader"]["version"]] == o["reader"]["digest"]
    assert "opus" not in o["reader"]["file"] and "OR in the packet META" in r and "full range" in r
    g = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["generator"]["version"] >= 4 and o["generator_history"][o["generator"]["version"]] == o["generator"]["digest"]
    assert "MULTI-DAY sitting" in g and "ONLY from the packet's `speaker`" in g and g.count("BAD (ru over") == 2


def test_date_ranges_with_month_names_satisfy_the_session_check():
    s = [{"session": {"date": "2025-04-29", "title": "Senatning 6-yalpi majlisi 29-30.04.25"}}]
    assert not session_undated("2025 йил 29–30 апрелдаги ялпи мажлисда ким гапирди?", s)
    assert not session_undated("Кто выступил на заседании 29–30 апреля 2025 года?", s)
    assert session_undated("2025 йил 29–30 майдаги мажлисда ким гапирди?", s)


def test_two_digit_session_years_are_normalised_and_both_forms_pass():
    from graphrag.dataset.facts import Fact as F
    from graphrag.dataset.generate import check_item_ex, session_iso, session_numbers
    ses = {"date": "2025-05-28", "title": "Senatning 7-yalpi majlisi 28-29.05.25"}
    assert session_iso(ses) == ["2025-05-28", "2025-05-29"]
    fr = [{"id": "F1", "act": "Senatning 7-yalpi majlisi 28-29.05.25", "where": "Senatning 7-yalpi majlisi 28-29.05.25",
           "text": "Senator qonun loyihasi 30 kun ichida qayta koʻrib chiqilishini aytdi.", "session": ses}]
    assert {"2025", "5", "28", "29"} <= session_numbers(fr)
    s = Seed("single-t", "single", "uz-Latn", ("S/ВЫСТ.1#1",), ("S",), "train", "STENOGRAMMA")
    f = F("S/ВЫСТ.1#1@0", "S/ВЫСТ.1#1", "", "30 kun", "30 kun ichida qayta koʻrib chiqilishini")
    cfg = {"max_copy_words": 99, "max_lexical_overlap": 1.0}
    ok = lambda q: check_item_ex({"question": q, "answer": "30 kun ichida.", "used": ["F1.0"]}, s, {"F1.0": f}, fr, cfg)[0]
    assert ok("2025-yil 28–29-maydagi Senat yalpi majlisida loyiha qancha muddatda qayta koʻriladi?") is None
    assert ok("Senatning 28-29.05.25 dagi yalpi majlisida loyiha qancha muddatda qayta koʻriladi?") is None
    assert ok("Senat yalpi majlisida loyiha qancha muddatda qayta koʻriladi?") == "session_undated"
    assert ok("2024-yil 28–29-maydagi Senat yalpi majlisida loyiha qancha muddatda qayta koʻriladi?") in (
        "unsupported_number", "session_undated")
    N = lambda i, t: type("N", (), {"node_id": i, "title": t})()
    M = type("M", (), {"section": "STENOGRAMMA", "date": "2025-05-28", "title": ses["title"]})()
    recs = [type("R", (), {"doc_id": "S", "meta": M, "structure": [N("S/ВЫСТ.1", "Raislik qiluvchi")]})()]
    got = opus.session_info(recs)("S/ВЫСТ.1#1")["session"]
    assert got["dates"] == ["2025-05-28", "2025-05-29"] and got["range"] == "2025-05-28..2025-05-29"
