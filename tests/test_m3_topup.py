"""Cumulative judge-plant sizing: a judge kind short of min_kind_n valid plants gets a top-up round of its own."""
from graphrag.dataset.verify import JUDGE_KINDS, plan_topup, verify_candidates
from graphrag.kg.llm import KgLLM
from tests.test_m3_dataset_verify import CAP, VCFG, _Judge, _frag_of, _world


def _run(tmp_path, topup):
    text, cs, need = _world(60)
    j = _Judge(need)
    llm = KgLLM(VCFG, cache_root=tmp_path, client=j.client, allow_api=True)
    return verify_candidates(cs, _frag_of(text), llm, 2, 0.05, 7, lambda c: "U#1", min_kind_n=10, views_per_call=5,
                             neighbor=lambda c, i: f"N:{c.seed.chunks[i]}", quote_cap=CAP, topup=topup)


def test_short_kinds_are_topped_up_to_the_minimum(tmp_path):
    _, st0 = _run(tmp_path / "a", False)
    assert st0["kinds_below_min_n"]                                       # 5 % plant share: kinds short
    kept, st = _run(tmp_path / "b", True)
    assert st["kinds_below_min_n"] == [] and all(st["planted"][k] >= 10 for k in JUDGE_KINDS)
    assert st["topup"]["plants"] > 0 and set(st["topup"]["short_before"]) == set(st0["kinds_below_min_n"])
    assert [c.seed.seed_id for c in kept] == [c.seed.seed_id for c in _run(tmp_path / "c", False)[0]]   # same pairs


def test_topup_never_repeats_a_base_kind_variant():
    text, cs, _ = _world(12)
    gt = plan_topup(cs, {"question_swap": 4, "neighbor_swap": 6}, [], 7, lambda c: "U#1",
                    lambda c, i: f"N:{c.seed.chunks[i]}", {})
    keys = [(p["kind"], p["cand"].seed.seed_id, p.get("variant", 0)) for g in gt for p in g["plants"]]
    assert len(keys) == len(set(keys)) and {k for k, _, _ in keys} == {"question_swap", "neighbor_swap"}
