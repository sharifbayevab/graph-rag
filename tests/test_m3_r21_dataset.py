"""M3-R21 / M3-R22 (user decisions 2026-09-29): calibrated copy thresholds, proportional per-document seed cap,
ds_bridge agentic multi-hop, generator checks of the strict re-pilot reading, and the ds_quality rubric judge."""
import json

import pytest

from graphrag.dataset.generate import calibrate_copy, copy_limits


# ---- (a) copy / paraphrase calibration -----------------------------------------------------------------------------------
def test_copy_limits_by_language_family_fall_back_to_legacy():
    cfg = {"max_copy_words": 6, "max_lexical_overlap": 0.6,
           "copy_calibration": {"families": {"uz": {"max_copy_words": 7, "max_lexical_overlap": 0.8462}}}}
    assert copy_limits(cfg, "uz-Cyrl") == (7, 0.8462) and copy_limits(cfg, "uz-Latn") == (7, 0.8462)
    assert copy_limits(cfg, "ru") == (6, 0.6)                              # no Russian Q-real question
    assert copy_limits({"max_copy_words": 6, "max_lexical_overlap": 0.6}, "uz-Cyrl") == (6, 0.6)


def test_calibration_takes_the_nearest_rank_p90_and_exempts_titles():
    fr = [{"act": "Банклар тўғрисида", "where": "Банклар тўғрисида, ст. 5",
           "text": "Марказий банк тижорат банкларига лицензияни ўттиз кун ичида беради ёки рад этади."}]
    qs = [("uz", "Лицензия қанча муддатда берилади?", fr)] * 9 + \
         [("uz", "Марказий банк тижорат банкларига лицензияни қанча кунда беради?", fr)]
    out = calibrate_copy(qs, 0.90)["uz"]
    assert out["n"] == 10 and out["max_copy_words"] == 0                  # 9 of 10 at run 0: P90 = the 9th value
    big = calibrate_copy(qs[:8] + qs[-1:] * 2, 0.90)["uz"]
    assert big["max_copy_words"] == 5 and big["run_hist"] == {"0": 8, "5": 2}
    t = calibrate_copy([("uz", "Банклар тўғрисида Қонунга кўра лицензия кимга берилади?", fr)])["uz"]
    assert t["max_copy_words"] == 0                                       # the act title adds nothing


def test_config_holds_the_measured_uz_calibration():
    from graphrag import config
    cal = config.load("train")["dataset"]["copy_calibration"]
    uz = cal["families"]["uz"]
    assert cal["quantile"] == 0.90 and "ru" not in cal["families"]
    assert uz["n"] == 680 and uz["max_copy_words"] == 7 and uz["max_lexical_overlap"] == uz["overlap_quantiles"]["p90"]
    assert sum(uz["run_hist"].values()) == 680


# ---- (b) proportional per-document seed cap ------------------------------------------------------------------------------
def test_doc_cap_is_max_3_or_2_percent_of_the_documents_fragments():
    from graphrag.dataset.seeds import doc_cap
    rec = lambda d, n: type("R", (), {"doc_id": d, "chunks": [None] * n})()
    cap = doc_cap([rec("small", 40), rec("mid", 151), rec("big", 1000)], 3, 0.02)
    assert (cap("small"), cap("mid"), cap("big"), cap("unknown")) == (3, 4, 20, 3)


def test_sample_seeds_honours_a_per_document_cap_function():
    from graphrag.dataset.seeds import sample_seeds
    pool = {"single": [((f"{d}/П.{i}#1",), (d,), ()) for d in ("A", "B") for i in range(10)]}
    splits, sec = {"A": "train", "B": "train"}, {"A": "QONUNLAR", "B": "QONUNLAR"}
    seeds, short = sample_seeds(pool, splits, sec, {"single": 12}, {"train": 1.0}, {"ru": 1.0}, {"QONUNLAR": 1.0},
                                lambda d: {"A": 8, "B": 2}[d], 0)
    from collections import Counter
    assert Counter(s.docs[0] for s in seeds) == {"A": 8, "B": 2} and short == {"single|train|ru": 2}


# ---- (c) ds_bridge ------------------------------------------------------------------------------------------------------
import numpy as np

from graphrag.kg.llm import KgLLM


class _E:
    def __init__(self, src, t, dst, attrs=None):
        self.src, self.type, self.dst, self.attrs = src, t, dst, attrs or {}


class _C:
    def __init__(self, cid, parent, start, text):
        self.chunk_id, self.parent, self.start, self.end, self.text, self.tokens = cid, parent, start, start + len(text), \
            text, 50


class _R:
    def __init__(self, doc, chunks, title):
        self.doc_id, self.chunks = doc, chunks
        self.meta = type("M", (), {"section": "QONUNLAR", "title": title, "date": None})()
        N = type("N", (), {})
        self.structure = []
        for v, t in [(doc, title)] + [(c.parent, "") for c in chunks]:
            n = N()
            n.node_id, n.title, n.type = v, t, "модда"
            self.structure.append(n)


ACT_B = "Банкларни санация қилиш ва тугатиш тўғрисида"
T_A = ("24-модда. Санация қилинаётган банкнинг мажбуриятлари ушбу Қонуннинг 64-моддасида белгиланган кредиторларнинг "
       "талабларини қаноатлантириш навбатида олувчига ўтказилади.")
T_B = ("64-модда. Кредиторларнинг талаблари қуйидаги навбатда қаноатлантирилади: учинчи навбатда – Омонатларни "
       "кафолатлаш агентлигига ўтган талаблар.")
T_C = "Ушбу Қонунда қуйидаги асосий тушунчалар қўлланилади: санация – банкнинг тўлов қобилиятини тиклаш чоралари."
T_D = "Банк тизимидаги ислоҳотлар ҳақида сенатор маъруза қилди ва депутатлар фикр билдирди, кўп саволлар берилди."


def _toy():
    """Doc B: СТ.24 (anchor: REFERS_TO СТ.64, REGULATES CON/sanatsiya), СТ.64, СТ.3 (DEFINES CON/sanatsiya);
    doc S (same split) RELATED to СТ.24; doc X in another split, RELATED too (must never be offered)."""
    cs = [_C("B/СТ.24#1", "B/СТ.24", 0, T_A), _C("B/СТ.64#1", "B/СТ.64", 1000, T_B), _C("B/СТ.3#1", "B/СТ.3", 2000, T_C)]
    recs = [_R("B", cs, ACT_B), _R("S", [_C("S/ВЫСТ.1#1", "S/ВЫСТ.1", 0, T_D)], "Сенат мажлиси"),
            _R("X", [_C("X/БЛ.1#1", "X/БЛ.1", 0, T_D + " x")], "Бошқа")]
    ev = lambda s, e: {"evidence": {"doc_id": "B", "start": s, "end": e}}
    edges = [_E("B/СТ.24", "REFERS_TO", "B/СТ.64", ev(60, 80)), _E("B/СТ.24", "REGULATES", "CON/sanatsiya", ev(10, 20)),
             _E("B/СТ.3", "DEFINES", "CON/sanatsiya", ev(2060, 2090)), _E("B/СТ.24", "RELATED", "S/ВЫСТ.1"),
             _E("X/БЛ.1", "RELATED", "B/СТ.24")]
    ids = [c.chunk_id for r in recs for c in r.chunks]
    nodes = [c.parent for r in recs for c in r.chunks]

    class Data:
        chunk_ids = ids
        chunk_row = {c: i for i, c in enumerate(ids)}
        node_row = {v: i for i, v in enumerate(nodes)}
        vecs64 = np.eye(len(ids))

        @staticmethod
        def mu_inv(row):
            return np.array([row])
    splits = {"B": "train", "S": "train", "X": "test"}
    from graphrag.dataset.bridge import BridgeIndex
    return BridgeIndex(Data(), recs, edges, set(ids), splits, 0), recs, {c.chunk_id: c for r in recs for c in r.chunks}


def test_bridge_candidates_by_source_same_split_only():
    bx, _, _ = _toy()
    cs = bx.candidates("B/СТ.24#1")
    assert [c for c, _ in cs["refers_to"]] == ["B/СТ.64#1"]
    assert [c for c, _ in cs["concept"]] == ["B/СТ.3#1"]
    assert cs["concept"][0][1] == (("B/СТ.24", "REGULATES", "CON/sanatsiya"), ("B/СТ.3", "DEFINES", "CON/sanatsiya"))
    assert [c for c, _ in cs["related"]] == ["S/ВЫСТ.1#1"]              # X/БЛ.1 is in another split
    from graphrag.dataset.bridge import sample_anchors, shortlist
    assert [s for s, _, _ in shortlist(cs, "related")] == ["related", "refers_to", "concept"]
    picks = sample_anchors(bx, ["B/СТ.24#1", "B/СТ.64#1"], 3, ["uz-Cyrl", "ru"], 0)
    assert {p["anchor"] for p in picks} == {"B/СТ.24#1"}                  # an anchor is used once


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


BCFG = {"llm": {"model": "gpt-4.1", "max_calls_total": 50, "passes": {"ds_bridge": {"prompt_version": 1, "max_calls": 10},
                                                                       "ds_quality": {"prompt_version": 2, "max_calls": 10}}}}
DSCFG = {"seed": 0, "max_copy_words": 99, "max_lexical_overlap": 1.0}


def _answer(u):
    out = []
    for it in u["items"]:
        nb = next(n["id"] for n in it["neighbors"] if n["text"].startswith("64-модда"))
        out.append({"key": it["key"], "neighbor": nb,
                    "anchor_fact": {"quote": "ушбу Қонуннинг 64-моддасида белгиланган кредиторларнинг", "value": "64-моддасида"},
                    "needed_fact": {"quote": "учинчи навбатда – Омонатларни кафолатлаш агентлигига ўтган талаблар",
                                    "value": "учинчи навбатда"},
                    "question": f"«{ACT_B}»ги Қонуннинг 24-моддаси ҳавола қилган 64-моддага кўра Омонатларни "
                                "кафолатлаш агентлигига ўтган талаблар нечанчи ўринда қондирилади?",
                    "answer": "Улар учинчи навбатда қондирилади.", "reason": "the cited article states the order"})
    return {"items": out}


def test_ds_bridge_builds_a_checked_two_fragment_pair_and_replays_offline(tmp_path):
    from graphrag.dataset.bridge import bridge, sample_anchors
    from graphrag.dataset.sources import SourceIndex
    bx, recs, chunks = _toy()
    picks = sample_anchors(bx, ["B/СТ.24#1"], 1, ["uz-Cyrl"], 0)
    idx = SourceIndex(recs)
    cl, calls = _client(_answer)
    cands, st = bridge(picks, chunks, idx, KgLLM(BCFG, cache_root=tmp_path, client=cl, allow_api=True), DSCFG,
                       {"B": "QONUNLAR", "S": "STENOGRAMMA", "X": "BOSHQA"})
    assert len(calls) == 1 and len(cands) == 1, st
    c = cands[0]
    assert c.seed.chunks == ("B/СТ.24#1", "B/СТ.64#1") and c.seed.pattern == "multi"
    assert [f.value for f in c.facts] == ["учинчи навбатда", "64-моддасида"] or \
        sorted(f.value for f in c.facts) == ["64-моддасида", "учинчи навбатда"]
    assert all(f.quote in chunks[f.chunk_id].text for f in c.facts)       # exact source substrings
    assert c.provenance["bridge_source"] == "refers_to" and st["by_stratum"]
    off = KgLLM(BCFG, cache_root=tmp_path, allow_api=False)
    c2, _ = bridge(picks, chunks, idx, off, DSCFG, {"B": "QONUNLAR", "S": "STENOGRAMMA", "X": "BOSHQA"})
    assert off.total_api_calls() == 0 and [x.question for x in c2] == [c.question]


def test_ds_bridge_rejects_none_invented_quotes_and_leaks(tmp_path):
    from graphrag.dataset.bridge import bridge, sample_anchors
    from graphrag.dataset.sources import SourceIndex
    bx, recs, chunks = _toy()
    picks = sample_anchors(bx, ["B/СТ.24#1"], 1, ["uz-Cyrl"], 0)
    sec = {"B": "QONUNLAR", "S": "STENOGRAMMA", "X": "BOSHQA"}
    for mut, why in ((lambda x: dict(x, neighbor="none"), "none"),
                     (lambda x: dict(x, needed_fact={"quote": "бешинчи навбатда – бошқа талаблар ва мажбуриятлар",
                                                     "value": "бешинчи навбатда"}), "needed_quote_not_verbatim"),
                     (lambda x: dict(x, question=x["question"].replace("нечанчи ўринда", "учинчи навбатда")),
                      "answer_in_question")):
        cl, _ = _client(lambda u: {"items": [mut(x) for x in _answer(u)["items"]]})
        d = tmp_path / why
        cands, st = bridge(picks, chunks, SourceIndex(recs), KgLLM(BCFG, cache_root=d, client=cl, allow_api=True),
                           DSCFG, sec)
        assert cands == [] and any(k.endswith("|" + why) for k in st["rejected"]), (why, st)


def test_ds_bridge_respects_the_whole_ledger_cap(tmp_path):
    from graphrag.dataset.bridge import bridge, sample_anchors
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.llm import KgBudgetExceeded
    bx, recs, chunks = _toy()
    (tmp_path / "ledger.json").write_text(json.dumps({"ds_facts": {"d": 50}}), encoding="utf-8")
    cl, calls = _client(_answer)
    with pytest.raises(KgBudgetExceeded):
        bridge(sample_anchors(bx, ["B/СТ.24#1"], 1, ["uz-Cyrl"], 0), chunks, SourceIndex(recs),
               KgLLM(BCFG, cache_root=tmp_path, client=cl, allow_api=True), DSCFG, {"B": "QONUNLAR"})
    assert calls == []


# ---- (d) M3-R22 generator checks and ds_quality -------------------------------------------------------------------------
def test_names_outside_the_fragment_are_rejected():
    from graphrag.dataset.generate import foreign_names
    fr = [{"act": "Ombudsman", "where": "Ombudsman", "text": "Мазкур тадбирга ушбу тумандаги 10 дан ортиқ МФЙлар жалб қилинди."}]
    assert foreign_names("Tadbirda Koson tumanidagi nechta MFY ishtirok etgan?", fr, "uz-Latn") == ["Koson"]
    assert foreign_names("Tadbirga nechta MFY jalb qilingan?", fr, "uz-Latn") == []
    ru = [{"act": "Senatning 2-yalpi majlisi", "where": "x", "text": "Moyra Toshova: 104 mingta oila uy-joy oldi."}]
    assert foreign_names("Сколько семей получили жильё, по словам сенатора Тошовой?", ru, "ru") == []   # translit
    assert foreign_names("Сколько семей получили жильё, по словам сенатора Каримовой?", ru, "ru") == ["Каримовой"]


def test_uz_cyrl_must_be_pure_cyrillic():
    from graphrag.dataset.generate import lang_ok
    assert lang_ok("Ўзбекистонда нечта маҳалла бор?", "uz-Cyrl")
    assert not lang_ok("Оʻзбекистонда нечта маҳалла бор?", "uz-Cyrl")          # Latin ʻ (re-pilot #10)
    assert not lang_ok("«Inson qadri» йилида нечта маҳалла бор?", "uz-Cyrl")  # Latin inside quotes


def test_the_act_named_must_be_the_fragments_own_or_the_amended_act():
    from graphrag.dataset.generate import wrong_act
    fr = [{"act": "Божхона кодексига ўзгартиш киритиш тўғрисида", "where": "x", "text": "t", "amended": ["bojxona kodeks"]}]
    assert not wrong_act("«Божхона кодексига ўзгартиш киритиш тўғрисида»ги Қонунга кўра муддат қанча?", fr, "uz-Cyrl")
    assert not wrong_act("Божхона кодекси 347-моддасига кўра муддат қанча?", fr, "uz-Cyrl")
    assert wrong_act("«Паррандачиликни янада қўллаб-қувватлаш чора-тадбирлари тўғрисида»ги қарорга кўра ким?", fr,
                     "uz-Cyrl")
    assert wrong_act("Солиқ кодекси 12-моддасига кўра муддат қанча?", fr, "uz-Cyrl")


def test_ds_quality_passes_only_good_and_replays_offline(tmp_path):
    from graphrag.dataset.generate import Candidate
    from graphrag.dataset.quality import calibration, judge_quality
    from graphrag.dataset.seeds import Seed
    cs = [Candidate(Seed(f"single-{i:010d}", "single", "ru", (f"D/П.{i}#1",), ("D",), "train", "QONUNLAR"), f"В{i}?",
                    f"О{i}.", [], {}) for i in range(8)]
    frag_of = lambda c, lab: {"id": lab, "act": "A", "where": "W", "text": "T " + c}
    verdicts = ["GOOD", "BORDERLINE", "BAD", "GOOD", "GOOD", "BAD", "BORDERLINE", "GOOD"]
    seen = []

    def fn(u):
        seen.append(u)
        ck = {c: {"ok": True, "note": ""} for c in ("c1", "c2", "c3", "c4", "c5", "c6", "c7")}
        return {"items": [{"key": x["key"], "verdict": verdicts[int(x["question"][1:-1])], "checks": ck, "reason": "r"}
                          for x in u["items"]]}
    cl, calls = _client(fn)
    kept, st = judge_quality(cs, frag_of, KgLLM(BCFG, cache_root=tmp_path, client=cl, allow_api=True), 6)
    assert len(calls) == 2 and [c.question for c in kept] == ["В0?", "В3?", "В4?", "В7?"]
    assert all(set(x) == {"key", "lang", "question", "answer", "fragments"} for x in seen[0]["items"])   # blind
    from graphrag.dataset.quality import final_verdict
    bad_check = {c: {"ok": c != "c5", "note": ""} for c in ("c1", "c2", "c3", "c4", "c5", "c6", "c7")}
    assert final_verdict({"verdict": "GOOD", "checks": bad_check}) == ("BORDERLINE", [5])   # GOOD only by code
    off = KgLLM(BCFG, cache_root=tmp_path, allow_api=False)
    kept2, _ = judge_quality(cs, frag_of, off, 6)
    assert off.total_api_calls() == 0 and len(kept2) == 4
    cal = calibration({"a": "GOOD", "b": "BAD", "c": "BORDERLINE"}, {"a": "GOOD", "b": "GOOD", "c": "BAD"})
    assert cal["exact"] == 1 and cal["binary"] == 2 and cal["bad_accepted"] == ["b"]
