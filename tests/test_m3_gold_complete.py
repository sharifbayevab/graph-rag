import json

from graphrag.eval.gold_complete import SCHEMA, complete_one, verify
from graphrag.kg.llm import KgLLM


def test_verify_accepts_only_verbatim_evidence_from_a_listed_candidate():
    c = {"K1": "Ўзбекистон Республикасининг Конституциявий Қонуни 12-модда. Сенатнинг ваколатлари"}
    assert verify({"core_supports": True, "needed": [{"id": "K1", "evidence": "Конституциявий  Қонуни 12-модда"}]}, c) == (True, None)
    assert verify({"core_supports": True, "needed": [{"id": "K9", "evidence": "x" * 30}]}, c) == (False, "unknown_id")
    assert verify({"core_supports": True, "needed": [{"id": "K1", "evidence": "12-модда"}]}, c) == (False, "short_evidence")
    assert verify({"core_supports": True, "needed": [{"id": "K1", "evidence": "Олий суднинг ваколатлари ҳақида"}]}, c) == \
        (False, "evidence_not_in_candidate")


def test_complete_one_uses_the_cache_offline_and_maps_ids(tmp_path):
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    body = json.dumps({"core_supports": True, "needed": [
                        {"id": "K2", "evidence": "Сенатнинг ваколатлари қуйидагилардан иборат"}]})
                    msg = type("M", (), {"content": body})
                    return type("R", (), {"choices": [type("C", (), {"message": msg})]})
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 5, "passes": {"gold_complete": {"prompt_version": 1, "max_calls": 5}}}}
    core = {"G1": "12-моддага мувофиқ Сенат"}
    cands = {"K1": "бошқа матн бу ерда", "K2": "Сенатнинг ваколатлари қуйидагилардан иборат: қонунлар қабул қилиш"}
    ids = {"K1": "D#3", "K2": "D#4"}
    assert complete_one(KgLLM(cfg, cache_root=tmp_path, client=Client(), allow_api=True), "савол", "жавоб", core, cands, ids) == \
        (["D#4"], True, None)
    off = KgLLM(cfg, cache_root=tmp_path, allow_api=False)
    assert complete_one(off, "савол", "жавоб", core, cands, ids) == (["D#4"], True, None)
    assert off.stats["gold_complete"]["api_calls"] == 0 and SCHEMA["additionalProperties"] is False
