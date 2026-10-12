import json

from graphrag.dataset.facts import Fact, check_fact, extract_block
from graphrag.dataset.generate import (Candidate, check_item, exempt_words, gen_batches, lang_ok, lexical_overlap,
                                       max_copy_run)
from graphrag.dataset.seeds import Seed
from graphrag.kg.llm import KgLLM

ACT = "Банклар ва банк фаолияти тўғрисида"
T1 = "Ушбу Қонуннинг 5-моддасида назарда тутилган ваколатлар Марказий банк томонидан амалга оширилади."
T2 = "5-модда. Ваколатли орган банкларни санация қилиш тўғрисида қарор ўттиз кун ичида қабул қилади."   # M3-R20: F1 not redundant
FR = [{"id": "F1", "act": ACT, "where": f"{ACT}, ст. 3", "text": T1},
      {"id": "F2", "act": ACT, "where": f"{ACT}, ст. 5", "text": T2}]
CFG = {"max_copy_words": 6, "max_lexical_overlap": 0.6}


def test_fact_checks():
    aw = exempt_words(FR)
    good = {"statement": "Санация ҳақидаги қарор 30 кун ичида қабул қилинади", "value": "ўттиз кун",
            "quote": "санация қилиш тўғрисида қарор ўттиз кун ичида"}
    assert check_fact(good, T2, aw) is None
    assert check_fact(dict(good, quote="бошқа матн бу ерда ёзилган эмас эди"), T2, aw) == "quote_not_verbatim"
    assert check_fact(dict(good, value="олтмиш кун"), T2, aw) == "value_not_in_quote"
    assert check_fact(dict(good, value="Банклар ва банк фаолияти"), T2, aw) in ("name_only", "value_not_in_quote")
    assert check_fact(dict(good, quote="ўттиз кун"), T2, aw) == "short_quote"


def test_extract_block_caches_per_block_and_drops_bad_facts(tmp_path):
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    body = json.dumps({"chunks": [{"id": "C1", "facts": [
                        {"statement": "Санация ҳақидаги қарор 30 кун ичида", "value": "ўттиз кун",
                         "quote": "санация қилиш тўғрисида қарор ўттиз кун ичида"},
                        {"statement": "уйдирма", "value": "йўқ", "quote": "бу фрагментда йўқ бўлган узун матн"}]}]})
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 2, "passes": {"ds_facts": {"prompt_version": 1, "max_calls": 2}}}}
    ch = {"D/СТ.5#1": type("Ch", (), {"text": T2, "parent": "D/СТ.5"})()}
    ix = type("Ix", (), {"act": {"D": ACT}, "format": staticmethod(lambda n: f"{ACT}, ст. 5")})()
    facts, why = extract_block(KgLLM(cfg, cache_root=tmp_path, client=Client(), allow_api=True), "D#B0",
                               ["D/СТ.5#1"], ch, ix, 6)
    assert [f.value for f in facts] == ["ўттиз кун"] and why["quote_not_verbatim"] == 1
    assert facts[0].fact_id == "D/СТ.5#1@0" and facts[0].chunk_id == "D/СТ.5#1"
    off = KgLLM(cfg, cache_root=tmp_path, allow_api=False)
    assert extract_block(off, "D#B0", ["D/СТ.5#1"], ch, ix, 6)[0] == facts and off.stats["ds_facts"]["api_calls"] == 0


def _seed(i, pattern="refers_internal", chunks=("D/СТ.3#1", "D/СТ.5#1")):
    return Seed(f"{pattern}-{i:010d}", pattern, "uz-Cyrl", chunks, ("D",), "train", "QONUNLAR",
                (("D/СТ.3", "REFERS_TO", "D/СТ.5"),) if pattern != "single" else ())


FACTS = {"F1.0": Fact("D/СТ.3#1@0", "D/СТ.3#1", "3-моддадаги ваколатлар 5-моддада", "Марказий банк",
                      "ваколатлар Марказий банк томонидан амалга оширилади"),
         "F2.0": Fact("D/СТ.5#1@0", "D/СТ.5#1", "қарор 30 кун ичида", "ўттиз кун",
                      "санация қилиш тўғрисида қарор ўттиз кун ичида")}


def test_item_must_use_every_chunk_state_every_value_and_not_leak_it():
    s = _seed(1)
    good = {"key": "k1", "question": "Кредит ташкилотини соғломлаштириш ҳақида қайси орган қанча муддатда ҳал қилади?",
            "answer": "Марказий банк; қарор ўттиз кун ичида қабул қилинади.", "used": ["F1.0", "F2.0"]}
    assert check_item(good, s, FACTS, FR, CFG) is None
    assert check_item(dict(good, used=["F2.0"]), s, FACTS, FR, CFG) == "chain_incomplete"
    # M3-R20: a listed fact the answer does not state is dropped from `used`; its fragment is then uncovered
    assert check_item(dict(good, answer="Марказий банк қарор қабул қилади."), s, FACTS, FR, CFG) == "chain_incomplete"
    assert check_item(dict(good, answer="Қарор қабул қилинади."), s, FACTS, FR, CFG) == "value_missing_in_answer"
    leak = dict(good, question="Кредит ташкилотини соғломлаштириш ҳақида қайси орган ўттиз кун ичида ҳал қилади?")
    assert check_item(leak, s, FACTS, FR, CFG) == "answer_in_question"
    # M3-R20: in a multi-hop item a fact in the question is a PATH fact (it identifies the provision), not a leak
    path = dict(good, question="Марказий банк соғломлаштириш ҳақида қанча муддатда қарор қабул қилади?",
                answer="Қарор ўттиз кун ичида қабул қилинади.")
    assert check_item(path, s, FACTS, FR, CFG) is None
    num = dict(good, answer="Марказий банк; қарор ўттиз кун ичида, 45 кунгача қабул қилинади.")
    assert check_item(num, s, FACTS, FR, CFG) == "unsupported_number"
    assert check_item(dict(good, used=["F9.9"]), s, FACTS, FR, CFG) == "unknown_fact"


def test_copy_rule_exempts_act_titles_and_overlap_measures_paraphrase():
    q = f"«{ACT}»ги Қонунга кўра кредит ташкилотини ким соғломлаштиради?"
    t = f"«{ACT}»ги Қонуннинг 5-моддасига мувофиқ Марказий банк ваколатли."
    assert max_copy_run(q, t) >= 5 and max_copy_run(q, t, exempt_words(FR)) <= 1
    assert lexical_overlap("Ваколатли орган банкларни санация қилиш тўғрисида қарор қабул қиладими?", [T2], ACT) > 0.8
    assert lang_ok("Qaysi organ sanatsiya toʼgʼrisida qaror qabul qiladi?", "uz-Latn")
    assert lang_ok("Какой орган принимает решение о санации банка?", "ru")
    assert not lang_ok("Какой орган принимает решение о санации банка?", "uz-Cyrl")


def test_pilot_batches_are_kept_intact_in_the_full_run():
    seeds = [_seed(i) for i in range(10)]
    pilot = {seeds[7].seed_id, seeds[2].seed_id, seeds[5].seed_id, seeds[9].seed_id}
    p_batches = gen_batches([s for s in seeds if s.seed_id in pilot], 3, ())
    full = gen_batches(seeds, 3, pilot)
    assert full[:len(p_batches)] == p_batches and sum(len(b) for b in full) == 10




# ---- M3-R3 controls beyond the brief's Step 1 (Task 15 scope: пп. 4, 5, 11, 12, 14) and the M3-R16 fix round --------
import unicodedata

import pytest

from graphrag.dataset.facts import (DEFAULT_MAX_FACTS, PROMPT_FACTS, SCHEMA_FACTS, facts_prompt, numbers, value_in)
from graphrag.dataset.generate import PROMPT_GEN, SCHEMA_GEN, generate
from graphrag.kg.llm import KgBudgetExceeded, KgCacheMiss, prompt_digest

T3 = f"«{ACT}»ги Қонун 2019 йилда қабул қилинган ва банклар фаолиятини тартибга солади."
DCFG = {"llm": {"model": "gpt-4.1", "max_calls_total": 50,
                "passes": {"ds_facts": {"prompt_version": 2, "max_calls": 20},
                           "ds_generate": {"prompt_version": 2, "max_calls": 20}}}}
GOOD_F = {"statement": "s", "value": "ўттиз кун", "quote": "санация қилиш тўғрисида қарор ўттиз кун ичида"}
GQ = "Кредит ташкилотини соғломлаштириш ҳақида қайси орган қанча муддатда ҳал қилади?"
GA = "Марказий банк; қарор ўттиз кун ичида қабул қилинади."
G = {"key": "k1", "question": GQ, "answer": GA, "used": ["F1.0", "F2.0"]}
QL = "Kredit tashkilotini sogʻlomlashtirish haqida qaysi organ qancha muddatda hal qiladi?"


def _client(answer_for):
    """A fake OpenAI client: answer_for(system_prompt, user_json) -> dict. Records every call."""
    calls = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    body = json.dumps(answer_for(kw["messages"][0]["content"], json.loads(kw["messages"][1]["content"])))
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})
    return Client(), calls


def _ix():
    return type("Ix", (), {"act": {"D": ACT}, "format": staticmethod(lambda n: f"{ACT}, ст. {n.split('.')[-1]}")})()


def _chunks():
    return {"D/СТ.3#1": type("Ch", (), {"text": T1, "parent": "D/СТ.3"})(),
            "D/СТ.5#1": type("Ch", (), {"text": T2, "parent": "D/СТ.5"})(),
            "D/СТ.7#1": type("Ch", (), {"text": T3, "parent": "D/СТ.7"})()}


def _s(lang, i=1):
    return Seed(f"refers_internal-{i:010d}", "refers_internal", lang, ("D/СТ.3#1", "D/СТ.5#1"), ("D",), "train",
                "QONUNLAR", (("D/СТ.3", "REFERS_TO", "D/СТ.5"),))


# -- ds_facts checks ------------------------------------------------------------------------------------------------------
def test_fact_whose_value_is_only_the_act_name_is_rejected():
    aw = exempt_words([{"act": ACT, "where": f"{ACT}, ст. 7"}])
    f = {"statement": "Қонун номи", "value": "Банклар ва банк фаолияти", "quote": f"«{ACT}»ги Қонун 2019 йилда"}
    assert check_fact(f, T3, aw) == "name_only"
    assert check_fact(dict(f, value="2019 йилда"), T3, aw) is None


def test_probe_4b_title_with_gi_qonun_suffix_is_name_only():
    aw = exempt_words([{"act": ACT, "where": f"{ACT}, ст. 7"}])
    assert check_fact({"statement": "s", "value": f"{ACT}»ги Қонун", "quote": f"«{ACT}»ги Қонун 2019 йилда"}, T3, aw) \
        == "name_only"


def test_probe_4c_number_value_equal_to_the_article_number_is_not_name_only():
    aw = exempt_words([{"act": ACT, "where": f"{ACT}, ст. 7"}])
    text = "7-модда. Банк 7 кун ичида жавоб беради"
    assert check_fact({"statement": "s", "value": "7", "quote": text}, text, aw) is None
    assert check_fact({"statement": "s", "value": "7 кун", "quote": text}, text, aw) is None


def test_probe_4d_latin_title_with_okina_is_name_only():
    act = "Oʻzbekiston Respublikasining Soliq kodeksi"
    text = f"{act} qabul qilindi"
    f = {"statement": "s", "value": act, "quote": text}
    assert check_fact(f, text, exempt_words([{"act": act, "where": act}])) == "name_only"
    assert check_fact(dict(f, value="O'zbekiston Respublikasining Soliq kodeksi"), text,
                      exempt_words([{"act": act, "where": act}])) == "name_only"      # ASCII apostrophe variant


def test_quote_must_be_verbatim_whitespace_case_and_unicode_form_aside():
    aw = exempt_words(FR)
    f = {"statement": "s", "value": "ўттиз кун", "quote": "САНАЦИЯ  қилиш\nтўғрисида қарор ўттиз кун ичида"}
    assert check_fact(f, T2, aw) is None                               # whitespace / case only
    nfd = dict(f, quote=unicodedata.normalize("NFD", f["quote"]), value=unicodedata.normalize("NFD", "ўттиз кун"))
    assert nfd["quote"] != f["quote"] and check_fact(nfd, T2, aw) is None   # m6: NFC before comparing
    assert check_fact(dict(f, quote="санация қилиш тўғрисида қарор 30 кун ичида", value="30 кун"), T2, aw) \
        == "quote_not_verbatim"                                        # a paraphrase is not a quote


def test_probe_12a_value_matches_whole_tokens_only():
    tx = "Банк 15 кун ичида ҳисобот тақдим этади ва тўлайди."
    assert check_fact({"statement": "s", "value": "5", "quote": "Банк 15 кун ичида ҳисобот"}, tx, set()) \
        == "value_not_in_quote"
    assert check_fact({"statement": "s", "value": "5 кун", "quote": "Банк 15 кун ичида ҳисобот"}, tx, set()) \
        == "value_not_in_quote"
    assert check_fact({"statement": "s", "value": "15 кун", "quote": "Банк 15 кун ичида ҳисобот"}, tx, set()) is None


def test_canonical_value_matching():
    assert value_in("ўттиз кун", "қарор 30 кун ичида") and value_in("30 кун", "qaror oʻttiz kun ichida")
    assert value_in("Марказий банк", "Markaziy bankka murojaat") and value_in("Марказий банк", "«Марказий банк»")
    assert not value_in("5 кун", "15 кун ичида") and not value_in("банк", "банкрот деб топилади")
    assert numbers("ўттиз беш кун") == {"35"} and numbers("тридцати дней") == {"30"} and numbers("бир ой") == set()
    assert numbers("ikki yuz oʻttiz") == {"230"} and numbers("он сказал") == set()


def test_extract_block_retries_a_bad_id_answer_once_then_rejects_and_caps_facts(tmp_path):
    good = GOOD_F
    bodies = iter([{"chunks": [{"id": "C9", "facts": [good]}]}, {"chunks": [{"id": "C1", "facts": [good]}]},
                   {"chunks": [{"id": "C1", "facts": [good]}, {"id": "C1", "facts": [good]}]}, {"chunks": []},
                   {"chunks": [{"id": "C1", "facts": [good] * 4}]}])
    cl, calls = _client(lambda sp, u: next(bodies))
    llm = KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True)
    ch = _chunks()
    fs, why = extract_block(llm, "D#B0", ["D/СТ.5#1"], ch, _ix(), 6)   # bad ids, then a good retry
    assert [f.value for f in fs] == ["ўттиз кун"] and why == {"retried_bad_answer": 1} and len(calls) == 2
    assert "retry" in json.loads(calls[1]["messages"][1]["content"])
    fs, why = extract_block(llm, "D#B1", ["D/СТ.5#1"], ch, _ix(), 6)   # repeated ids, then missing ids
    assert fs == [] and why == {"retried_bad_answer": 1, "bad_answer": 1}
    fs, why = extract_block(llm, "D#B2", ["D/СТ.5#1"], ch, _ix(), 2)
    assert [f.fact_id for f in fs] == ["D/СТ.5#1@0", "D/СТ.5#1@1"] and why["too_many"] == 2
    assert llm.stats_json()["ds_facts"]["rejected_ids"] == 3
    off = KgLLM(DCFG, cache_root=tmp_path, allow_api=False)            # m7: rejected answers are re-validated,
    assert extract_block(off, "D#B0", ["D/СТ.5#1"], ch, _ix(), 6)[0] == \
        extract_block(llm, "D#B0", ["D/СТ.5#1"], ch, _ix(), 6)[0]      # never reused; the replay needs 0 calls
    assert extract_block(off, "D#B1", ["D/СТ.5#1"], ch, _ix(), 6)[0] == [] and off.stats["ds_facts"]["api_calls"] == 0


def test_facts_carry_block_and_cache_key_of_the_accepted_answer(tmp_path):
    cl, _ = _client(lambda sp, u: {"chunks": [{"id": "C1", "facts": [GOOD_F]}]})
    llm = KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True)
    fs, _ = extract_block(llm, "D#B0", ["D/СТ.5#1"], _chunks(), _ix(), 6)
    assert fs[0].block == "D#B0" and (tmp_path / "ds_facts" / f"{fs[0].cache_key}.json").exists()


def test_fact_cap_comes_from_config_and_enters_the_prompt_digest(tmp_path):
    from graphrag import config
    assert int(config.load("train")["dataset"]["max_facts_per_chunk"]) == DEFAULT_MAX_FACTS
    assert "at most 6 facts" in PROMPT_FACTS and "at most 4 facts" in facts_prompt(4)
    seen = []
    cl, _ = _client(lambda sp, u: seen.append(sp) or {"chunks": [{"id": "C1", "facts": []}]})
    extract_block(KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True), "D#B0", ["D/СТ.5#1"], _chunks(), _ix(), 4)
    assert seen == [facts_prompt(4)]


def test_prompt_digest_and_version_enter_the_cache_key(tmp_path):
    cl, calls = _client(lambda sp, u: {"chunks": [{"id": "C1", "facts": []}]})
    KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True).cached_json(
        "ds_facts", {"x": 1}, SCHEMA_FACTS, prompt=PROMPT_FACTS)
    off = KgLLM(DCFG, cache_root=tmp_path, allow_api=False)
    assert off.cached_json("ds_facts", {"x": 1}, SCHEMA_FACTS, prompt=PROMPT_FACTS) is not None
    with pytest.raises(KgCacheMiss):                                   # an edited prompt never reuses an answer
        off.cached_json("ds_facts", {"x": 1}, SCHEMA_FACTS, prompt=PROMPT_FACTS + " ")
    bumped = {"llm": {**DCFG["llm"], "passes": {"ds_facts": {"prompt_version": 3, "max_calls": 20}}}}
    with pytest.raises(KgCacheMiss):
        KgLLM(bumped, cache_root=tmp_path, allow_api=False).cached_json(
            "ds_facts", {"x": 1}, SCHEMA_FACTS, prompt=PROMPT_FACTS)


# Pinned prompt digests (prompt_version 2, M3-R16): the pilot's cache is replayed by the full run only if the prompts are
# byte-identical. Changing a prompt must bump train.yaml dataset_llm.passes.<pass>.prompt_version and this pin.
PINNED = {"ds_facts": "049e6f9c4f1c63444f5a128dea8879f892064ff899006c75c37e8233c553c1ef", "ds_generate": "c90631e25d0d6e7ec671e765995bf206a8eb1a0a5172b0b1ca419e087bebe802"}   # ds_generate v5 (M3-R22; v4 f827f038 M3-R20); ds_facts v2 unchanged (cache reused)


def test_prompts_are_versioned_and_pinned():
    from graphrag import config
    passes = config.load("train")["dataset_llm"]["passes"]
    assert passes["ds_facts"]["prompt_version"] == 2 and passes["ds_generate"]["prompt_version"] == 5
    assert prompt_digest(PROMPT_FACTS, SCHEMA_FACTS) == PINNED["ds_facts"]
    assert prompt_digest(PROMPT_GEN, SCHEMA_GEN) == PINNED["ds_generate"]
    for rule in ("ONE information need", "in any form", "digits or words", 'end with "?"', "vary the form",
                 "bridge", "aggregation", "PATH fact", "act identity", "self-contained", "GLOSSARY", "interrogative must fit", "attribute correctly", "TEXT",
                 "transliterate Cyrillic values", "«"):
        assert rule in PROMPT_GEN
    for rule in ("keeps its", "never a bare number", "single generic"):
        assert rule in PROMPT_FACTS


def test_dataset_budget_config_gate_and_cumulative_ledger(tmp_path, monkeypatch):
    import importlib.util
    from graphrag import config
    t = config.load("train")
    dl = t["dataset_llm"]
    assert dl["cache_dir"] == "cache/dataset" and dl["model"] == "gpt-4.1"
    assert set(dl["passes"]) >= {"ds_facts", "ds_generate", "ds_verify"}
    assert sum(int(dl["passes"][p]["max_calls"]) for p in ("ds_facts", "ds_generate", "ds_verify")) <= 2500
    assert int(dl["max_calls_total"]) == 2500
    spec = importlib.util.spec_from_file_location("build_dataset_t15", config.ROOT / "scripts" / "build_dataset.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    assert {"facts", "generate"} <= set(bd.COMMANDS)
    monkeypatch.delenv("GRL_DATASET_API", raising=False)
    monkeypatch.setenv("GRL_KG_API", "1")                              # the KG gate never opens the dataset passes
    llm = bd._llm(t)
    assert llm.allow_api is False and str(llm.cache_root).endswith("cache/dataset")
    monkeypatch.setenv("GRL_DATASET_API", "1")
    assert bd._llm(t).allow_api is True
    # the pass caps are cumulative over the ledger (every earlier run counts)
    (tmp_path / "ledger.json").write_text(json.dumps({"ds_facts": {"2026-09-28": 20}}), encoding="utf-8")
    cl, calls = _client(lambda sp, u: {"chunks": []})
    with pytest.raises(KgBudgetExceeded):
        KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True).cached_json(
            "ds_facts", {"y": 1}, SCHEMA_FACTS, prompt=PROMPT_FACTS)
    assert calls == []


def test_probe_B4_total_cap_is_cumulative_over_the_ledger_across_runs(tmp_path):
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 3,
                   "passes": {"ds_facts": {"prompt_version": 2, "max_calls": 100},
                              "ds_generate": {"prompt_version": 2, "max_calls": 100}}}}
    cl, calls = _client(lambda sp, u: {"chunks": [{"id": "C1", "facts": []}]})
    done = 0
    for run in range(3):                                               # 3 runs x 3 blocks, total cap 3
        x = KgLLM(cfg, cache_root=tmp_path, client=cl, allow_api=True)
        try:
            for i in range(3):
                extract_block(x, f"D#B{run}{i}", ["D/СТ.5#1"], _chunks(), _ix(), 6)
                done += 1
        except KgBudgetExceeded:
            pass
    assert done == 3 and len(calls) == 3
    # the cap spans passes: a ds_generate call is refused too, while cached blocks still replay
    with pytest.raises(KgBudgetExceeded):
        generate([_seed(1)], {"D/СТ.3#1": [FACTS["F1.0"]], "D/СТ.5#1": [FACTS["F2.0"]]}, _chunks(), _ix(),
                 KgLLM(cfg, cache_root=tmp_path, client=cl, allow_api=True), dict(CFG, gen_batch=3))
    assert extract_block(KgLLM(cfg, cache_root=tmp_path, allow_api=False), "D#B00", ["D/СТ.5#1"], _chunks(), _ix(),
                         6) == ([], {})


# -- ds_generate checks ---------------------------------------------------------------------------------------------------
def test_probe_6e_value_leaked_as_digits_or_words():
    s = _seed(1)
    assert check_item(dict(G, question="Кредит ташкилотини соғломлаштириш ҳақида ким 30 кун ичида ҳал қилади?"),
                      s, FACTS, FR, CFG) == "answer_in_question"
    F30 = dict(FACTS, **{"F2.0": Fact("D/СТ.5#1@0", "D/СТ.5#1", "x", "30 кун", "санация 30 кун ичида қарор қабул")})
    assert check_item(dict(G, question="Кредит ташкилотини ким ўттиз кунда соғломлаштиради?",
                           answer="Марказий банк; қарор 30 кун ичида қабул қилинади."), s, F30, FR, CFG) \
        == "answer_in_question"
    assert check_item(dict(G, question="Кредит ташкилотини ким қанча кунда соғломлаштиради, 30 дами?",
                           answer="Марказий банк; қарор 30 кун ичида қабул қилинади."), s, F30, FR, CFG) \
        == "answer_in_question"                                        # a value's number alone leaks too


def test_value_leak_is_caught_across_scripts():
    s = _seed(3)
    for q in ("Qaysi organ kredit tashkilotini oʻttiz kun ichida sogʻlomlashtiradi?",
              "Kim kredit tashkilotini o'ttiz kun ichida sogʻlomlashtiradi?",
              "Кредит ташкилотини соғломлаштириш ҳақида ким oʻttiz kun ичида ҳал қилади?"):
        assert check_item(dict(G, question=q), s, FACTS, FR, CFG) == "answer_in_question"


def test_probe_7b_invented_number_in_the_question():
    assert check_item(dict(G, question="Кредит ташкилотини соғломлаштириш ҳақида 2021 йилдан қайси орган қанча муддатда "
                                       "ҳал қилади?"), _seed(1), FACTS, FR, CFG) == "unsupported_number"
    assert check_item(dict(G, question=f"Кредит ташкилотини соғломлаштириш ҳақида «{ACT}»ги Қонуннинг 5-моддасига кўра "
                                       "қайси орган қанча муддатда ҳал қилади?"), _seed(1), FACTS, FR, CFG) is None


LONG2 = "Молиявий соғломлаштириш режасини тасдиқлаш ва унинг ижросини назорат қилиш тартиби белгиланади."
FRL2 = [dict(FR[0]), dict(FR[1], text=LONG2 + " " + T2)]
CQ2 = ("Кредит ташкилотлари учун молиявий соғломлаштириш режасини тасдиқлаш ва унинг ижросини назорат қилиш тартиби "
       "қайси орган қанча муддатда?")


def test_probe_8b_title_word_inside_a_copied_span_does_not_reset_the_run():
    assert max_copy_run(CQ2, FRL2[1]["text"], exempt_words(FRL2)) >= 9
    assert check_item(dict(G, question=CQ2), _seed(1), FACTS, FRL2, CFG) in ("copied", "not_paraphrased")


def test_probe_8b_prime_copy_rule_alone():
    assert check_item(dict(G, question=CQ2), _seed(1), FACTS, FRL2, dict(CFG, max_lexical_overlap=1.0)) == "copied"


def test_copy_rule_counts_a_transliterated_copy_and_exempts_only_whole_titles():
    lat = "Kredit tashkilotlari uchun moliyaviy sogʻlomlashtirish rejasini tasdiqlash va uning ijrosini nazorat qilish kim?"
    assert max_copy_run(lat, LONG2, exempt_words(FRL2)) >= 9
    assert max_copy_run(f"«{ACT}»ги Қонунга кўра?", f"«{ACT}»ги Қонуннинг", exempt_words(FR)) <= 1


def test_copy_and_overlap_limits_reject_items():
    s = _seed(4)
    base = {"key": "k1", "answer": GA, "used": ["F1.0", "F2.0"]}
    copied = dict(base, question="Ваколатлар томонидан амалга оширилади деган орган банкларни санация қилиш тўғрисида "
                                 "қарор қанча вақтда қабул қилади?")
    assert check_item(copied, s, FACTS, FR, CFG) in ("copied", "not_paraphrased")
    assert check_item(copied, s, FACTS, FR, dict(CFG, max_copy_words=99, max_lexical_overlap=1.0)) is None
    near = dict(base, question="Банкларни санация қилиш ҳақида қарорни қайси ваколатли орган қабул қилади?")
    assert check_item(near, s, FACTS, FR, dict(CFG, max_copy_words=99)) == "not_paraphrased"


F5 = {"F1.0": FACTS["F1.0"], "F2.0": Fact("D/СТ.5#1@0", "D/СТ.5#1", "x", "5 кун", "санация 5 кун ичида ...")}


def test_probe_12b_answer_15_kun_does_not_satisfy_value_5_kun():
    assert check_item(dict(G, answer="Марказий банк; қарор 15 кун ичида қабул қилинади."), _seed(1), F5, FR, CFG) \
        == "chain_incomplete"                                          # M3-R20: "5 кун" is not stated -> dropped


def test_probe_12c_15_kun_in_the_question_is_not_a_leak_of_5_kun():
    r = check_item(dict(G, question="Кредит ташкилоти 15 кун ичида соғломлаштирилмаса ким ҳал қилади?",
                        answer="Марказий банк; қарор 5 кун ичида қабул қилинади."), _seed(1), F5, FR, CFG)
    assert r != "answer_in_question" and r == "unsupported_number"     # 15 is in no fact: an invented number


def test_probe_12d_value_bank_leaks_only_by_lemma_not_by_substring():
    """Probe 12d under M3-R16: 'банк' is not leaked by a word that merely contains it ('банкрот', M1 lemma 'bankrot');
    'банкларни' has the M1 lemma 'bank', so the ruling ("unless the lemma equals") makes it a leak."""
    Fb = {"F1.0": Fact("D/СТ.3#1@0", "D/СТ.3#1", "x", "банк", "ваколатлар Марказий банк томонидан"), "F2.0": FACTS["F2.0"]}
    one = Seed("single-0000000012", "single", "uz-Cyrl", ("D/СТ.3#1",), ("D",), "train", "QONUNLAR")
    g1 = {"key": "k1", "answer": "Ваколатлар банк томонидан амалга оширилади.", "used": ["F1.0"]}
    q = "Кредит ташкилоти банкрот бўлса, ваколатларни ким амалга оширади?"
    assert check_item(dict(g1, question=q), one, Fb, FR[:1], CFG) != "answer_in_question"
    q2 = "Кредит ташкилотлари ва банкларни соғломлаштириш ваколатини ким амалга оширади?"
    assert check_item(dict(g1, question=q2), one, Fb, FR[:1], CFG) == "answer_in_question"


def test_answer_may_inflect_a_value_with_the_same_lemma():
    assert check_item(dict(G, answer="Марказий банкка; қарор ўттиз кун ичида қабул қилинади."), _seed(1), FACTS, FR,
                      CFG) is None


def test_m3_r16_items_are_wholly_in_the_target_script():
    sl = _s("uz-Latn")
    ok = dict(G, question=QL, answer="Markaziy bank hal qiladi; qaror oʻttiz kun ichida qabul qilinadi.")
    assert check_item(ok, sl, FACTS, FR, CFG) is None                  # values transliterated Cyrl -> Latn
    mixed = dict(ok, answer="Марказий банк hal qiladi; qaror ўттиз кун ichida qabul qilinadi.")
    assert check_item(mixed, sl, FACTS, FR, CFG) == "language"         # original-script values are not allowed
    ru = _s("ru", 2)
    rq = f"В какой срок решает об оздоровлении кредитной организации орган, указанный в статье 3 Закона «{ACT}»?"
    ra = "Решение принимает Центральный банк («Марказий банк») в течение тридцати дней («ўттиз кун»)."
    assert check_item({"key": "k1", "question": rq, "answer": ra, "used": ["F1.0", "F2.0"]}, ru, FACTS, FR, CFG) is None
    assert check_item({"key": "k1", "question": rq, "answer": ra.replace("Центральный", "Central"),
                       "used": ["F1.0", "F2.0"]}, ru, FACTS, FR, CFG) == "language"   # Latin outside «»
    uzc = dict(G, answer="Markaziy bank; қарор ўттиз кун ичида қабул қилинади.")
    assert check_item(uzc, _seed(1), FACTS, FR, CFG) == "language"     # uz-Cyrl: no Latin outside «»


def test_probe_L3_uz_latn_answer_of_cyrillic_values_and_title_plus_one_latin_word():
    assert check_item(dict(G, question=QL, answer=f"Марказий банк, ўттиз кун, «{ACT}» boʻyicha."), _s("uz-Latn"), FACTS,
                      FR, CFG) == "language"


def test_probe_L4_uz_latn_question_with_a_cyrillic_title():
    assert check_item(dict(G, question=f"«{ACT}»ги Қонунга кўра qaysi organ qancha muddatda hal qiladi?",
                           answer="Markaziy bank; qaror oʻttiz kun ichida qabul qilinadi."), _s("uz-Latn"), FACTS, FR,
                      CFG) == "language"


def _gen_answer(sp, u):
    items = []
    for it in u["items"]:
        labs = [f["facts"][0]["label"] for f in it["fragments"]]
        if it["pattern"] == "single":                                  # a leaking item: rejected, logged per cell
            items.append({"key": it["key"], "question": "Марказий банк нима қилади ва қачон, айтинг?",
                          "answer": "Марказий банк ваколатли.", "used": labs})
        else:
            items.append({"key": it["key"], "question": GQ, "answer": GA, "used": labs})
    return {"items": items}


def test_generate_end_to_end_provenance_and_rejections_per_cell(tmp_path):
    ch = _chunks()
    fl = KgLLM(DCFG, cache_root=tmp_path, client=_client(lambda sp, u: {"chunks": [
        {"id": "C1", "facts": [{"statement": "ваколат", "value": "Марказий банк",
                                "quote": "ваколатлар Марказий банк томонидан амалга оширилади"}]},
        {"id": "C2", "facts": [GOOD_F]}]})[0], allow_api=True)
    fs, _ = extract_block(fl, "D#B0", ["D/СТ.3#1", "D/СТ.5#1"], ch, _ix(), 6)
    facts = {"D/СТ.3#1": [fs[0]], "D/СТ.5#1": [fs[1]], "D/СТ.7#1": []}
    seeds = [_seed(1), _seed(2), Seed("single-0000000003", "single", "uz-Cyrl", ("D/СТ.3#1",), ("D",), "test",
                                     "QONUNLAR"),
             Seed("single-0000000004", "single", "ru", ("D/СТ.7#1",), ("D",), "val", "QONUNLAR")]
    cl, calls = _client(_gen_answer)
    llm = KgLLM(DCFG, cache_root=tmp_path, client=cl, allow_api=True)
    cfg = dict(CFG, gen_batch=3, max_facts_per_chunk=6)
    cands, st = generate(seeds, facts, ch, _ix(), llm, cfg)
    assert [c.seed.seed_id for c in cands] == [seeds[0].seed_id, seeds[1].seed_id] and len(calls) == 1
    user = calls[0]["messages"][1]["content"]
    assert T1 in user and T2 in user                                   # M3-R20: the generator sees the fragment text
    assert st["rejected"] == {"single|answer_in_question": 1, "single|no_facts": 1}
    assert st["rejected_by_cell"] == {"test|single|uz-Cyrl": {"answer_in_question": 1},
                                      "val|single|ru": {"no_facts": 1}}
    assert st["accept_rate"] == {"refers_internal": 1.0, "single": 0.0}
    for c in cands:
        p = c.provenance
        assert p["seed_id"] == c.seed.seed_id and p["chunks"] == list(c.seed.chunks) and p["cell"] == "train|refers_internal|uz-Cyrl"
        assert p["fact_ids"] == ["D/СТ.3#1@0", "D/СТ.5#1@0"] and {f.chunk_id for f in c.facts} == set(c.seed.chunks)
        assert p["generate"]["prompt_digest"] == prompt_digest(PROMPT_GEN, SCHEMA_GEN)
        assert (tmp_path / "ds_generate" / f"{p['generate']['cache_key']}.json").exists()
        assert p["generate"]["prompt_version"] == 2 and p["generate"]["model"] == "gpt-4.1"
        pf = p["facts"]                                                # m1: ds_facts block, key, model, version
        assert pf["prompt_digest"] == prompt_digest(PROMPT_FACTS, SCHEMA_FACTS) and pf["prompt_version"] == 2
        assert pf["model"] == "gpt-4.1" and list(pf["cache_keys"]) == ["D#B0"]
        assert (tmp_path / "ds_facts" / f"{pf['cache_keys']['D#B0']}.json").exists()
        assert set(p["copy"]) == {"max_run", "overlap"}
    off = KgLLM(DCFG, cache_root=tmp_path, allow_api=False)             # offline replay: same candidates, 0 calls
    again, st2 = generate(seeds, facts, ch, _ix(), off, cfg)
    assert [(c.question, c.provenance) for c in again] == [(c.question, c.provenance) for c in cands] and st2 == st
    assert off.stats["ds_generate"]["api_calls"] == 0


def test_generate_retries_a_bad_key_batch_once(tmp_path):
    answers = iter([{"items": []}, None])
    def ans(sp, u):
        a = next(answers)
        return a if a is not None else _gen_answer(sp, u)
    cl, calls = _client(ans)
    facts = {"D/СТ.3#1": [FACTS["F1.0"]], "D/СТ.5#1": [FACTS["F2.0"]]}
    cands, st = generate([_seed(1)], facts, _chunks(), _ix(), KgLLM(DCFG, cache_root=tmp_path, client=cl,
                                                                     allow_api=True), dict(CFG, gen_batch=3))
    assert len(cands) == 1 and len(calls) == 2 and "retry" in json.loads(calls[1]["messages"][1]["content"])
