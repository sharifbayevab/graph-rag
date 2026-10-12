"""M3-R28 (strengthens п. 6): evidence-based ablation -- a drop view is answerable only if the judge's answer carries
the removed fragment's fact (typed values by code, clause values by the judge's `complete` flag)."""
from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import Candidate
from graphrag.dataset.seeds import Seed
from graphrag.dataset.verify import judge_ok, sensitivity_gate
from tests.test_m3_dataset_verify import _Judge, _run

T1 = "248-модданинг тўртинчи қисмидаги «101-моддасида» деган сўзлар «47⁶ ва 101-моддаларида» деган сўзлар билан алмаштирилсин."
T2 = "иккинчи хатбошиси «100-моддасида (ёритиш учун электр энергиясидан фойдаланишга оид қисмида)» деган сўзлар билан тўлдирилсин."


def _agg(v1="47⁶ ва 101-моддаларида", v2="100-моддасида", vt=""):
    s = Seed("multi-agg", "multi", "ru", ("D/П.4#1", "E/П.7#1"), ("D", "E"), "train", "QONUNLAR")
    return Candidate(s, "Какие ссылки добавили в часть четвёртую статьи 248 оба закона?", "…",
                     [Fact("D/П.4#1@0", "D/П.4#1", "", v1, "q", vtype=vt), Fact("E/П.7#1@0", "E/П.7#1", "", v2, "q", vtype=vt)], {})


FULL = {"answerable": True, "complete": True, "answer": "47⁶ ва 101-моддаларида; 100-моддасида",
        "evidence": [{"id": "F1", "quote": "«47⁶ ва 101-моддаларида»"}, {"id": "F2", "quote": "«100-моддасида"}]}
TX = {"F1": T1, "F2": T2}


def test_aggregation_partial_answer_on_a_drop_view_is_unanswerable():
    part1 = {"answerable": True, "complete": False, "answer": "47⁶ ва 101-моддаларида", "evidence": []}
    part2 = {"answerable": True, "complete": True, "answer": "100-моддасида", "evidence": []}   # claims complete
    ok, rep = judge_ok(_agg(), {"full": FULL, "drop:F2": part1, "drop:F1": part2}, TX)
    assert ok and rep["ablation"] == {"drop:F1": True, "drop:F2": True}        # the removed value is in neither


def test_genuinely_single_fragment_answerable_is_still_rejected():
    both = {"answerable": True, "complete": True, "answer": "47⁶ ва 101-моддаларида; 100-моддасида", "evidence": []}
    ok, rep = judge_ok(_agg(), {"full": FULL, "drop:F1": both, "drop:F2": {"answerable": False}}, TX)
    assert not ok and rep["ablation"]["drop:F1"] is False                      # F2 alone gave F1's value too
    clause = _agg("«47⁶ ва 101-моддаларида» деган сўзлар билан алмаштирилсин",
                  "«100-моддасида (ёритиш учун электр энергиясидан фойдаланишга оид қисмида)» деган сўзлар", "clause")
    fullc = dict(FULL, answer="…", evidence=[{"id": "F1", "quote": T1[:120]}, {"id": "F2", "quote": T2[:120]}])
    done = {"answerable": True, "complete": True, "answer": "«47⁶ ва 101-моддаларида», «100-моддасида»", "evidence": []}
    half = dict(done, complete=False)
    assert not judge_ok(clause, {"full": fullc, "drop:F1": done, "drop:F2": half}, TX)[0]       # complete: answerable
    assert judge_ok(clause, {"full": fullc, "drop:F1": half, "drop:F2": half}, TX)[0]           # incomplete: not
    assert not judge_ok(_agg(), {"full": FULL, "drop:F1": None, "drop:F2": {"answerable": False}}, TX)[0]  # missing


def test_a_lazy_judge_writing_no_values_keeps_raw_verdicts_and_fails(tmp_path):
    lazy = {"answerable": True, "complete": True, "answer": "x", "evidence": FULL["evidence"]}
    ok, rep = judge_ok(_agg(), {"full": lazy, "drop:F1": lazy, "drop:F2": lazy}, TX)
    assert not ok and rep["ablation"] == {"drop:F1": False, "drop:F2": False}  # no evidence mode: raw answerable
    kept, st, *_ = _run(tmp_path, "yes")                                         # answers "x" everywhere
    assert all(c.seed.pattern == "single" for c in kept) and sensitivity_gate(st, 0.9)


def test_a_clause_value_keeps_its_numbers_code_checked():
    clause = _agg("«47⁶ ва 101-моддаларида» деган сўзлар билан алмаштирилсин",
                  "«100-моддасида (ёритиш учун электр энергиясидан фойдаланишга оид қисмида)» деган сўзлар", "clause")
    fullc = dict(FULL, evidence=[{"id": "F1", "quote": T1[:120]}, {"id": "F2", "quote": T2[:120]}])
    claims = {"answerable": True, "complete": True, "answer": "47⁶ ва 101-моддаларида", "evidence": []}
    ok, rep = judge_ok(clause, {"full": fullc, "drop:F2": claims, "drop:F1": {"answerable": False}}, TX)
    assert ok and rep["ablation"]["drop:F2"] is True              # "complete" but without 100: not answerable


def test_list_joiners_do_not_hide_a_stated_value():
    full = dict(FULL, answer="«47⁶ и 101-моддаларида»; 100-моддасида")          # ru "и" for uz "ва"
    part = {"answerable": True, "complete": True, "answer": "только «47⁶ и 101-моддаларида»", "evidence": []}
    ok, rep = judge_ok(_agg(), {"full": full, "drop:F1": {"answerable": False}, "drop:F2": part}, TX)
    assert ok and rep["ablation"]["drop:F2"] is True
    both = dict(part, answer="«47⁶ и 101-моддаларида», 100-моддасида")
    assert not judge_ok(_agg(), {"full": full, "drop:F1": {"answerable": False}, "drop:F2": both}, TX)[0]
