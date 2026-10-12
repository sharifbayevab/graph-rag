"""M2 Task 0: one-time sweep of deferred M1 minors (final-review triage "defer-M2" + ledger "Parked").

Each test names its triage source (ledger task / parked item)."""
import copy
from pathlib import Path

import pytest

from graphrag import config
from graphrag.data import pdf as pdfmod
from graphrag.data.agenda import tidy_title
from graphrag.data.extract import Para
from graphrag.data.ingest import _has_download_suffix
from graphrag.data.lemma import tokens
from graphrag.data.normalize import (canon_num, canon_numbering, iso_dates, query_internal,
                                     segment_is_russian, to_internal)
from graphrag.data.pdf import VisionCacheMiss, VisionTranscriber
from graphrag.data.structure import DocText, parse_law, parse_prose, parse_stenogram
from graphrag.retrieval import lexical
from graphrag.stores.guard import reset_problems
from tests.helpers import words

D = "X-0000abcd"


def _ids(nodes):
    return {n.node_id: n for n in nodes}


class _NoCalls:
    def __getattr__(self, name):
        raise AssertionError("API must not be called")


# ── (a) parked: query-side ‒ / − before a unit word ───────────────────────────────────────────
@pytest.mark.parametrize("dash", ["‒", "−"])
def test_figure_and_minus_dash_before_unit_word_keeps_composite(dash):
    assert "modda#21" in lexical.query_terms(f"21{dash}moddasi")
    assert lexical.query_terms(f"21{dash}moddasi") == lexical.query_terms("21-moddasi")
    assert "modda#21" in lexical.query_terms(f"21{dash}моддаси", "uz-Cyrl")
    assert "bob#3" in lexical.query_terms(f"3 {dash} bob")


def test_figure_dash_before_non_unit_word_is_unchanged_in_the_index():
    # the 12 indexed spans ("432‒II-сонли") keep their dash: text_norm of the index is unchanged
    assert "409‒i-sonli" in to_internal("409‒I-сонли Қонуни", "uz-Cyrl")
    assert "152−xii-sonli" in to_internal("152−XII-сонли Қонунига", "uz-Cyrl")


# ── (b) parked + T1: build_corpus.main must not mutate the lru_cached config ──────────────────
def test_config_load_returns_an_independent_copy():
    a = config.load("data")
    a["max_tokens"] = 1
    a["pdf"]["vision"]["on_miss"] = "skip"
    b = config.load("data")
    assert b["max_tokens"] == 384 and "on_miss" not in b["pdf"]["vision"]


def test_build_corpus_main_overrides_are_scoped_to_the_run(monkeypatch, tmp_path):
    import scripts.build_corpus as bc
    before = copy.deepcopy(config.load("data"))
    seen = {}

    def fake_build(cfg):
        seen["cfg"] = cfg
        seen["inner"] = config.load("data")            # deep readers (extract, pdf) see the overrides
        return [], []
    monkeypatch.setattr(bc, "build_corpus", fake_build)
    monkeypatch.setattr(bc, "_misses", lambda cfg: (0, 0))
    monkeypatch.setattr(bc, "write_records", lambda *a: None)
    monkeypatch.setattr(bc, "write_lexicon", lambda *a: None)
    assert bc.main(["--allow-degraded", "--set", "max_tokens=12",
                    "--set", f"records_dir={tmp_path}"]) == 0
    assert seen["cfg"]["max_tokens"] == 12 and seen["cfg"]["pdf"]["vision"]["on_miss"] == "skip"
    assert seen["inner"]["max_tokens"] == 12 and seen["inner"]["pdf"]["vision"]["on_miss"] == "skip"
    assert config.load("data") == before                # nothing leaks into later tests / runs


# ── (c) parked: libpq ?host= / ?port= query parameters ────────────────────────────────────────
V2 = dict(pg_dsn="postgresql://graphrag:graphrag@localhost:5437/graphrag",
          qdrant_url="http://localhost:6343", neo4j_uri="bolt://127.0.0.1:7688")


@pytest.mark.parametrize("dsn", [
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?host=10.0.0.5",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?port=5436",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?sslmode=disable&host=prod-db",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?hostaddr=10.0.0.5",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?service=prod",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?host=localhost,10.0.0.5",
    "postgresql://graphrag:graphrag@localhost:5437,10.0.0.5:5436/graphrag",
])
def test_reset_guard_refuses_libpq_query_host_port(dsn):
    probs = reset_problems(**{**V2, "pg_dsn": dsn}, env={})
    assert len(probs) == 1 and probs[0].startswith("postgres")
    assert reset_problems(**{**V2, "pg_dsn": dsn}, env={"GRL_ALLOW_RESET": "1"}) == []


@pytest.mark.parametrize("dsn", [
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?sslmode=disable",
    "postgresql://graphrag:graphrag@localhost:5437/graphrag?host=localhost&port=5437",
])
def test_reset_guard_accepts_harmless_or_matching_query_params(dsn):
    assert reset_problems(**{**V2, "pg_dsn": dsn}, env={}) == []


# ── (d) parked: page-vision miss without pdftoppm is a degradation ────────────────────────────
def test_offline_page_vision_miss_without_pdftoppm_is_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(pdfmod.shutil, "which", lambda name: None)       # no poppler
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=False, on_miss="skip")
    assert pdfmod.read_page_pairs(vt, Path("x.pdf"), 6, 150, "k1", ["жами 1064 та"]) is None
    assert vt.stats["offline_miss"] == 1 and vt.stats["api_calls"] == 0
    strict = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=False)
    with pytest.raises(VisionCacheMiss):
        pdfmod.read_page_pairs(strict, Path("x.pdf"), 6, 150, "k1", ["жами 1064 та"])


def test_page_render_failure_with_api_on_is_counted_not_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(pdfmod, "render_page", lambda *a: None)
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=True)
    assert pdfmod.read_page_pairs(vt, Path("x.pdf"), 6, 150, "k2", ["жами 1064 та"]) is None
    assert vt.stats["offline_miss"] == 1 and vt.stats["page_render_failed"] == 1


# ── T10: duck-typed vision object (page_pairs without page_cached) ─────────────────────────────
def test_read_page_pairs_accepts_vision_without_page_cached(monkeypatch):
    class V:
        def page_pairs(self, png, key, texts):
            return ["жами: 1064 та"] if png == b"png" else None
    monkeypatch.setattr(pdfmod, "render_page", lambda *a: b"png")
    assert pdfmod.read_page_pairs(V(), Path("x.pdf"), 6, 150, "k3", ["жами 1064 та"]) == ["жами: 1064 та"]


# ── T10: snap ratio exactly at the threshold; "2024-йил" / "11-модда" values ──────────────────
def test_snap_label_accepts_ratio_equal_to_threshold():
    assert pdfmod._snap_label("abcde", ["abcdx"]) == "abcdx"                # ratio 0.8 exactly
    assert pdfmod._snap_label("abcde", ["abxyz"]) is None


@pytest.mark.parametrize("v", ["2024-йил", "11-модда", "2024-yil", "11–моддаси", "5-банд"])
def test_page_pair_value_accepts_number_with_hyphenated_unit(v):
    assert pdfmod._VALUE.match(v)


def test_page_pair_value_still_rejects_sentences():
    assert not pdfmod._VALUE.match("35 та шахсий қабуллари ташкил этилган бўлиб")
    assert not pdfmod._VALUE.match("2024-йилда жами мурожаатлар сони ошди")


# ── T9: dedup suffix is a browser "(N)", not a year ───────────────────────────────────────────
def test_download_suffix_is_not_a_year():
    assert _has_download_suffix("docs/QONUNLAR/3. Қонун матни (1).docx")
    assert _has_download_suffix("docs/X/x(12).pdf")
    assert not _has_download_suffix("docs/BOSHQA/Hisobot (2020).docx")
    assert not _has_download_suffix("docs/BOSHQA/Hisobot (123).docx")


# ── T4: dates, roman numerals, apostrophes ────────────────────────────────────────────────────
def test_ru_date_needs_a_month_word_not_a_prefix():
    assert iso_dates("1 майора 2020") == "1 майора 2020"
    assert iso_dates("5 мартовский 2020") == "5 мартовский 2020"
    assert iso_dates("14 мая 2019 года") == "2019-05-14 года"
    assert iso_dates("1 января 2020") == "2020-01-01"
    assert iso_dates("3 августе 2021") == "2021-08-03"
    assert iso_dates("9 май 2022") == "2022-05-09"


def test_invalid_roman_numerals_are_not_converted():
    assert canon_num("XIV") == "14" and canon_num("XL") == "40" and canon_num("IX") == "9"
    assert canon_num("IIII") == "iiii"
    assert canon_num("VX") == "vx"
    assert canon_numbering("IC боб") == "IC боб"
    assert canon_numbering("XIV боб") == "14-боб"


def test_roman_unit_needs_a_word_boundary_and_ignores_case_of_the_unit():
    assert canon_numbering("VI bobur keldi") == "VI bobur keldi"
    assert canon_numbering("II bobo") == "II bobo"
    assert canon_numbering("XXIII бобида") == "23-бобида"
    assert canon_numbering("II бўлимининг") == "2-бўлимининг"
    assert canon_numbering("II Боб") == "2-Боб"
    assert canon_numbering("IV БОБ. УМУМИЙ") == "4-БОБ. УМУМИЙ"
    assert "bob#2" in tokens(to_internal("II Боб", "uz-Cyrl"))


def test_ascii_apostrophe_folding_is_uzbek_only():
    assert to_internal("don't stop", "en") == "don't stop"
    assert to_internal("o'zbek tili", "uz-Latn") == "oʼzbek tili"


# ── (f) T8: Russian sentence quoting an Uzbek name; Uzbek segment with Russian loanwords ──────
def test_russian_sentence_quoting_uzbek_name_stays_russian():
    s = ("В частности, сегодня в целях поддержки донорского движения учрежден нагрудный знак "
         "«Ўзбекистон Республикаси фахрий донори», который вручается гражданам.")
    assert segment_is_russian(s, False) is True
    assert "ўзбекистон" in to_internal(s, "uz-Latn")                   # kept Cyrillic, not transliterated


def test_uzbek_segment_with_russian_loanword_is_uzbek():
    assert segment_is_russian("| Шизофрения, шизотипик ва васвасавий бузилишлар", False) is False
    assert "shizofreniya" in to_internal("| Шизофрения, шизотипик ва васвасавий бузилишлар", "uz-Cyrl")


def test_evidence_routing_keeps_plain_cases():
    assert segment_is_russian("Ушбу Қонун расмий эълон қилинган кундан эътиборан кучга киради.", False) is False
    assert segment_is_russian("Закон вступает в силу со дня его официального опубликования.", False) is True
    assert segment_is_russian("С.", False) is False and segment_is_russian("С.", True) is True
    assert segment_is_russian("Ҳурматли сенаторлар, закон очень важный.", False) is False


# ── T6: two-level article, reference lines ────────────────────────────────────────────────────
def test_numbered_paren_point_before_first_dot_point_does_not_collide():
    P = ["1-модда. Мақсад", "Ушбу Қонуннинг мақсади.",
         "2-модда. Тушунчалар", "1) олдинги матн;", "1. биринчи банд:", "1) кичик банд;",
         "2. иккинчи банд.", "3-модда. Кучга кириш", "Ушбу Қонун кучга киради."]
    ns = _ids(parse_law(D, DocText.build([Para(t) for t in P]), "t"))
    assert not [i for i in ns if "~" in i]
    art = f"{D}/СТ.2/Ч.1"
    assert f"{art}/П.1" in ns and f"{art}/П.1/ПП.1" in ns and f"{art}/П.2" in ns


def test_two_reference_lines_without_signer_or_place_are_not_requisites():
    P = ["1-модда. А", "Матн.", "2-модда. Б", "Матн.", "3-модда. В", "Ушбу Қонун кучга киради.",
         "(2020 йил 5 май)", "ЎРҚ-620-сон"]
    ns = _ids(parse_law(D, DocText.build([Para(t) for t in P]), "t"))
    assert not any(n.type == "реквизиты" for n in ns.values())
    P2 = P[:6] + ["Тошкент ш.,", "2020 йил 5 май,", "ЎРҚ-620-сон"]
    ns2 = _ids(parse_law(D, DocText.build([Para(t) for t in P2]), "t"))
    assert f"{D}/РЕКВ.1" in ns2


# ── T7: xulosa closing role line, -sin surnames, regex agenda path ────────────────────────────
@pytest.mark.parametrize("closing", ["Qoʼmita raisi", "Қўмита раиси"])
def test_xulosa_bare_committee_chair_line_is_requisites(closing):
    P = ["X U L O S A", "Qoʼmita qonunni koʼrib chiqdi.", "X U L O S A Q I L A D I:",
         "“A toʼgʼrisida”gi Qonun Senat yalpi majlisida koʼrib chiqilsin va maʼqullansin.", closing]
    dt = DocText.build([Para(t) for t in P])
    ns = _ids(parse_prose(D, dt, "t", 384, words))
    req = ns[f"{D}/РЕКВ.1"]
    assert dt.text[req.start:req.end] == closing


def test_single_requisites_row_with_surname_ending_in_sin():
    P = ["X U L O S A", "Qoʼmita qonunni koʼrib chiqdi va uni Senat tomonidan maʼqullash maqsadga muvofiq deb hisoblaydi.", "Қўмита раиси А. Ясин"]
    ns = _ids(parse_prose(D, DocText.build([Para(t) for t in P]), "t", 384, words))
    assert f"{D}/РЕКВ.1" in ns
    P2 = ["X U L O S A", "Qoʼmita qonunni koʼrib chiqdi va uni Senat tomonidan maʼqullash maqsadga muvofiq deb hisoblaydi.", "Раис А. Каримовга юклансин."]
    ns2 = _ids(parse_prose(D, DocText.build([Para(t) for t in P2]), "t", 384, words))
    assert not any(n.type == "реквизиты" for n in ns2.values())      # imperative: still body text


def _steno(P):
    return _ids(parse_stenogram(D, DocText.build([Para(t) for t in P]), "t"))


def test_regex_agenda_ignores_closing_remarks():
    P = ["Senatning yalpi majlisi",
         "Raislik qiluvchi: Kun tartibidagi birinchi masalani koʼrib chiqishga oʼtamiz.",
         "Ali Valiyev: Hurmatli senatorlar.",
         "Raislik qiluvchi: Qaror qabul qilindi. Kun tartibidagi barcha masalalarni koʼrib boʼldik.",
         "Raislik qiluvchi: Kun tartibimizda qolgan masalalarimizni koʼrib chiqdik."]
    ns = _steno(P)
    assert f"{D}/ВОПР.1" in ns and f"{D}/ВОПР.2" not in ns
    assert f"{D}/ВОПР.1/ВЫСТ.4" in ns


def test_regex_agenda_titles_quoted_non_act_item_and_skips_stray_quote():
    P = ["Senatning yalpi majlisi",
         "Raislik qiluvchi: Navbatdagi masala “Vazirlar Mahkamasiga parlament soʼrovi yuborish” haqida.",
         "Ali Valiyev: Hurmatli senatorlar.",
         "Raislik qiluvchi: Navbatdagi masala “Oliy Majlis “Banklar toʼgʼrisida”gi Qonun.",
         "Ali Valiyev: Hurmatli senatorlar."]
    ns = _steno(P)
    assert ns[f"{D}/ВОПР.1"].title == "Vazirlar Mahkamasiga parlament soʼrovi yuborish"
    assert ns[f"{D}/ВОПР.2"].title == "Banklar toʼgʼrisida"


# ── (e) T11b: _TRAIL apostrophe variants and further tails ────────────────────────────────────
@pytest.mark.parametrize("ap", ["ʼ", "‘", "'", "ʻ", "’"])
def test_tidy_title_trail_accepts_apostrophe_variants(ap):
    t = f"Davlat mudofaa buyurtmasi to{ap}g{ap}risidagi qonun"
    assert tidy_title(t, t) == f"Davlat mudofaa buyurtmasi to{ap}g{ap}risida"


def test_tidy_title_trims_bare_qonun_and_accusative_draft():
    t = "Issiqxona gazlarining chiqarilishini cheklash toʼgʼrisida qonun"
    assert tidy_title(t, t) == "Issiqxona gazlarining chiqarilishini cheklash toʼgʼrisida"
    t2 = "Senatning ayrim qoʼmitalari faoliyatini takomillashtirish toʼgʼrisidagi qaror loyihasini"
    assert tidy_title(t2, t2) == "Senatning ayrim qoʼmitalari faoliyatini takomillashtirish toʼgʼrisidagi qaror loyihasi"
    t3 = "Kengashning qarorlarini tasdiqlash toʼgʼrisidagi qaror loyihasi"
    assert tidy_title(t3, t3) == t3                                       # M1 rule kept


def test_query_internal_folds_query_dashes_like_the_index():
    assert query_internal("21‒moddasi", "uz-Latn") == query_internal("21-moddasi", "uz-Latn")


# ── T7 (ruling R3): quoted organisation name before a role label ──────────────────────────────
def test_quoted_organisation_role_label_is_a_speaker():
    P = ["Senatning yalpi majlisi",
         "Raislik qiluvchi: Kun tartibidagi birinchi masalani koʼrib chiqishga oʼtamiz.",
         "“Hududgaztaʼminot” AJ vakili: Assalomu alaykum hurmatli senatorlar.",
         "«Ўзбекнефтгаз» АЖ вакили: Ассалому алайкум."]
    ns = _steno(P)
    assert ns[f"{D}/ВОПР.1/ВЫСТ.2"].title == "“Hududgaztaʼminot” AJ vakili"
    assert ns[f"{D}/ВОПР.1/ВЫСТ.3"].title == "«Ўзбекнефтгаз» АЖ вакили"


# ── review fix round: quoted words are no routing evidence; ties → default; singletons reset ──
def test_quoted_russian_title_in_uzbek_line_is_no_evidence():
    s = ("“Uzbekistan's Experience with Fighting Incidence of Torture” “The Korean Post” нашри, "
         "шунингдек “Защита прав женщин – приоритетное направление деятельности Омбудсмана” "
         "мақолалардир.")
    assert segment_is_russian(s, False) is False
    s2 = ("В частности, сегодня в целях поддержки донорского движения учрежден нагрудный знак "
          "«Ўзбекистон Республикаси фахрий донори», который вручается гражданам.")
    assert segment_is_russian(s2, False) is True


def test_evidence_tie_keeps_the_default():
    tie = "закона бузилишлар"           # 1 ru function word vs 1 uz ending
    assert segment_is_russian(tie, False) is False and segment_is_russian(tie, True) is True


def test_overridden_resets_config_bound_singletons(tmp_path):
    from graphrag.data import extract, ingest
    v = {"model": "gpt-4.1", "cache_dir": str(tmp_path / "v"), "api": False}
    a = {"enabled": True, "model": "gpt-4.1", "cache_dir": str(tmp_path / "a"), "api": False}
    cfg = config.load("data")
    cfg["pdf"]["vision"]["on_miss"] = "skip"
    with config.overridden("data", cfg):
        vt = extract._vision({**v, "on_miss": "skip"})
        seg = ingest.agenda_segmenter({"agenda": a})
        assert extract._VISION is vt and seg is not None
    assert extract._VISION is None                         # next run builds its own, from its config
    assert ingest.agenda_segmenter({"agenda": a}) is not seg
    assert extract._vision(v).on_miss == "raise"
    extract._VISION = None
    ingest._AGENDA.clear()
