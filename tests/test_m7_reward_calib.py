"""M7 reward-weight calibration (docs/m7/reward-calibration-design.md): pure parts of grl.train.calibrate_reward."""
import math

import numpy as np
import pytest

from grl.train import calibrate_reward as cr


# --- U_ret -----------------------------------------------------------------------------------------------------------

def test_u_ret_recall_and_all_at_5_on_first_five_in_order():
    gold = {1, 2}
    assert cr.u_ret([1, 2, 3], gold) == 1.0
    assert cr.u_ret([1, 9, 8], gold) == pytest.approx(0.25)           # recall .5, all 0
    assert cr.u_ret([3, 4, 5, 6, 7, 1, 2], gold) == 0.0               # gold only past the first five
    assert cr.u_ret([3, 4, 5, 6, 1, 2], gold) == pytest.approx(0.25)  # 1 in the first five, 2 is the 6th
    assert cr.u_ret([], gold) == 0.0


# --- features --------------------------------------------------------------------------------------------------------

def test_fbeta_family_and_counts():
    f = cr.chunk_features([1, 2, 3, 4, 5], {1})
    assert f["recall"] == 1.0 and f["precision"] == pytest.approx(0.2)
    assert f["f1"] == pytest.approx(1 / 3)
    assert f["f2"] == pytest.approx(5 * 0.2 / (4 * 0.2 + 1))
    assert f["f3"] > f["f2"] > f["f1"] > f["f0.5"]
    assert f["n"] == pytest.approx(5 / cr.N_CAP)
    z = cr.chunk_features([7, 8], {1})
    assert all(z[k] == 0.0 for k in ("f0.5", "f1", "f2", "f3", "recall", "precision"))


def test_current_weights_reproduce_the_spec_reward():
    feats = {"f1": 0.4, "coverage": 0.5, "diversity": 0.0, "coherence": 1.0, "recall": 1, "precision": 0.2, "n": 0.6,
             "f0.5": 0.1, "f2": 0.6, "f3": 0.7}
    assert cr.score(feats, cr.CURRENT) == pytest.approx(0.5 * 0.4 + 0.2 * 0.5 + 0.1 * 1.0)


# --- candidates ------------------------------------------------------------------------------------------------------

def test_trunc_or_overfill():
    assert cr.trunc_or_overfill([5, 6, 7], [1, 2]) == ([5, 6], "trunc2")
    assert cr.trunc_or_overfill([5, 6], [1, 2]) == ([5], "trunc1")
    out, how = cr.trunc_or_overfill([5], [5, 1, 2, 3, 4, 6, 7, 8, 9])
    assert how == "over8" and out == [5, 1, 2, 3, 4, 6, 7, 8]
    assert cr.trunc_or_overfill([], [1]) == ([], "empty")


def test_ce_fused_order_convex_and_ties_to_fused_position():
    rows = np.array([10, 11, 12])
    fused = np.array([1.0, 0.5, 0.0])
    ce = {10: 0.0, 11: 5.0, 12: 0.0}
    r, _ = cr.ce_fused_order(rows, fused, ce, a=0.5)
    assert r.tolist() == [11, 10, 12]
    assert cr.ce_fused_order(rows, fused, {10: 1.0}, a=0.5) is None   # a candidate without a ce score


# --- answers ---------------------------------------------------------------------------------------------------------

def test_script_normalisation_cyrillic_latin_apostrophes():
    a = cr.norm_tokens("Ўзбекистон Республикаси қонуни")
    b = cr.norm_tokens("Oʻzbekiston Respublikasi qonuni")
    assert a == b == ["ozbekiston", "respublikasi", "qonuni"]
    assert cr.norm_tokens("Шуҳрат, ҒАЛАБА!") == cr.norm_tokens("Shuhrat galaba")


def test_token_f1():
    assert cr.token_f1("бир икки уч", "бир икки уч") == 1.0
    assert cr.token_f1("bir ikki", "бир икки уч тўрт") == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert cr.token_f1("", "a") == 0.0


def test_version_reference_numbers_and_number_f1():
    ref = "[['QONUNLAR-44d353fc', '2025-04-15', 'edit', 'EXT/nodavlat toʼgʼrisida/СТ.22/Ч.4'], " \
          "['QONUNLAR-44d353fc', '2025-04-15', 'edit', 'EXT/nodavlat toʼgʼrisida/СТ.8/Ч.1']]"
    assert sorted(cr.reference_numbers("version", ref)) == ["1", "22", "4", "8"]
    q = "2025-yil 15-apreldagi qonun qaysi qoidalarni yangi tahrirda bayon qilgan?"
    assert cr.number_f1("2025-yil 15-aprel: 22-moddaning 4-qismi va 8-moddaning 1-qismi", ["22", "4", "8", "1"], q) == 1.0
    assert cr.number_f1("22-moddaning 4-qismi", ["22", "4", "8", "1"], q) == pytest.approx(2 * 0.5 / 1.5)
    assert cr.number_f1("39 ta", ["39"], "2025 yilda nechta?") == 1.0
    law = "[['QONUNLAR-44d353fc', '2025-04-15', 'add', 'EXT/soliq kodeks/СТ.22/Ч.5']]"
    assert sorted(cr.reference_numbers("version_law", law)) == ["15", "2025", "4"]
    assert cr.number_f1("2025 йил 15 апрелдаги қонун", ["2025", "4", "15"], "22-моддаси 5-қисмини") == 1.0
    assert cr.answer_kind({"pattern": "version", "provenance": {"m7": {"template": "which_law_part"}}}) == "version_law"
    assert cr.answer_kind({"pattern": "single", "provenance": {}}) == "text"
    assert cr.number_f1("2025 yilda 40 ta", ["39"], "2025 yilda nechta?") == 0.0


def test_count_answer_is_the_first_new_number():
    q = "2025 йилда «Ҳомийлик тўғрисида»га нечта ўзгартиш киритилган?"
    assert cr.count_correct("2025 йилда жами 8 та ўзгартиш (1, 5, 9-моддалар)", ["8"], q) == 1.0
    assert cr.count_correct("2025 йилда жами 9 та ўзгартиш (8-модда)", ["8"], q) == 0.0
    row = {"pattern": "count", "q": q, "y_ref": {"text": "8"}, "provenance": {}}
    assert cr.u_ans_f1(row, {"answerable": True, "answer": "8 та"}) == 1.0


def test_u_ans_f1_no_answer_is_zero():
    row = {"pattern": "single", "q": "?", "y_ref": {"text": "бир"}}
    assert cr.u_ans_f1(row, {"answerable": False, "answer": "бир"}) == 0.0
    assert cr.u_ans_f1(row, {"answerable": True, "answer": "бир"}) == 1.0


# --- ranking metrics and fit -----------------------------------------------------------------------------------------

def test_kendall_tau_b():
    assert cr.kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cr.kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(cr.kendall_tau([1, 2, 3], [1, 1, 1]))           # U constant: undefined
    assert cr.kendall_tau([1, 1, 1], [1, 2, 3]) == 0.0                 # R constant, U not: no information -> 0
    assert cr.kendall_tau([1, 2, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(3 * 2))


def test_fit_recovers_a_simplex_direction():
    rng = np.random.default_rng(0)
    true = np.array([0.7, 0.0, 0.3])
    groups = []
    for _ in range(60):
        F = rng.random((6, 3))
        groups.append((F, F @ true))
    w = cr.fit_simplex(groups)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert np.abs(w - true).sum() < 0.25
    taus = [cr.kendall_tau(F @ w, U) for F, U in groups]
    assert np.mean(taus) > 0.9


def test_doc_folds_keep_documents_together_and_are_deterministic():
    qs = [{"id": f"q{i}", "docs": [f"d{i % 7}"] + ([f"d{(i + 1) % 7}"] if i == 3 else [])} for i in range(40)]
    f1, f2 = cr.doc_folds(qs, k=5, seed=1), cr.doc_folds(qs, k=5, seed=1)
    assert f1 == f2 and set(f1.values()) <= set(range(5))
    by_doc = {}
    for q in qs:
        for d in q["docs"]:
            by_doc.setdefault(d, set()).add(f1[q["id"]])
    assert all(len(v) == 1 for v in by_doc.values())


def test_adoption_rule_two_se_of_the_paired_difference():
    cur = np.array([0.1, 0.2, 0.3, 0.2] * 10)
    better = cur + 0.2
    assert cr.adopt(better, cur)["adopt"] is True
    noisy = cur + np.array([0.3, -0.3, 0.3, -0.3] * 10) + 0.01
    assert cr.adopt(noisy, cur)["adopt"] is False


def test_subsample_stratified_deterministic():
    rows = [{"id": f"o{i}", "set": "old", "pattern": "single"} for i in range(30)] + \
           [{"id": f"h{i}", "set": "hard", "pattern": "count" if i % 2 else "version"} for i in range(10)]
    s = cr.subsample(rows, {"old": 6, "hard": 4}, seed=3)
    assert s == cr.subsample(rows, {"old": 6, "hard": 4}, seed=3)
    assert sum(r.startswith("o") for r in s) == 6 and sum(r.startswith("h") for r in s) == 4


def test_p5_threshold_quantile_of_good_contexts():
    R = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.1])
    U = np.array([0.6, 0.5, 0.9, 1.0, 0.7, 0.0])
    t = cr.p5_threshold(R, U, q=0.0)
    assert t["threshold"] == pytest.approx(0.2) and t["n_good"] == 5
