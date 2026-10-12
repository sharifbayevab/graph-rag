from collections import Counter

from graphrag.dataset.pilot import gates, pass_calls, pilot_seeds, project
from graphrag.dataset.seeds import Seed

PATS = ("single", "refers_internal", "refers_cross", "amends")
LANGS = ("uz-Cyrl", "uz-Latn", "ru")


def test_pilot_is_stratified():
    seeds = [Seed(f"{p}-{l}-{i:04d}", p, l, (f"D#{p}{l}{i}",), ("D",), ["train", "val", "test"][i % 3], "QONUNLAR")
             for p in PATS for l in LANGS for i in range(30)]
    pick = pilot_seeds(seeds, 8, 7)
    assert Counter((s.pattern, s.lang) for s in pick) == {(p, l): 8 for p in PATS for l in LANGS}
    assert {s.split for s in pick} == {"train", "val", "test"} and pick == pilot_seeds(list(reversed(seeds)), 8, 7)


def test_pass_calls_and_projection():
    before = {"ds_facts": {"2026-09-28": 3}}
    after = {"ds_facts": {"2026-09-28": 53}, "ds_generate": {"2026-09-28": 32}, "ds_verify": {"2026-09-28": 48}}
    assert pass_calls(before, after) == {"ds_facts": 50, "ds_generate": 32, "ds_verify": 48}
    pr = project({"ds_facts": 0.8, "ds_generate": 0.5, "ds_verify": 0.8},
                 {"single": 0.8, "refers_internal": 0.5, "refers_cross": 0.5, "amends": 0.4}, 1000,
                 {"single": 0.55, "refers_internal": 0.10, "refers_cross": 0.15, "amends": 0.20}, 1.15, 133)
    assert pr["accepted_needed"] == 1150 and pr["calls"]["ds_generate"] == 575 and pr["total"] == 133 + 920 + 575 + 920
    rep = {"projection": pr, "threshold_loss": {"single": 0.1}, "yield": {"single": 0.8}, "shortage": {},
           "verify_ok": True, "oracle_ok": True}
    g0 = gates(rep, {"max_calls_total": 2500, "max_threshold_loss": 0.20})
    assert len(g0) == 1 and "П2" in g0[0]                                        # 2548 > 2500
    rep["threshold_loss"]["amends"] = 0.3
    g = gates(rep, {"max_calls_total": 2500, "max_threshold_loss": 0.20})
    assert any("П5" in x and "amends" in x for x in g)


def test_pilot_split_spread_follows_availability_and_gates_all_kinds():
    seeds = [Seed(f"a-{i:03d}", "amends", "ru", (f"c{i}",), ("D",), "train" if i < 20 else "test", "QONUNLAR")
             for i in range(22)]
    pick = pilot_seeds(seeds, 8, 1)
    assert len(pick) == 8 and Counter(s.split for s in pick)["train"] >= 6
    rep = {"projection": {"total": 10}, "threshold_loss": {}, "yield": {"amends": 0.0}, "shortage": {"x": 1},
           "verify_ok": False, "oracle_ok": False}
    g = gates(rep, {"max_calls_total": 2500, "max_threshold_loss": 0.20})
    assert len(g) == 4 and any("yield 0" in x for x in g) and any("П4" in x for x in g)
