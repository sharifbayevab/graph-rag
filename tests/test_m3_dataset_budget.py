from graphrag.dataset.budget import blocks_of, project_calls, views_of
from graphrag.dataset.seeds import Seed


def _s(i, pattern, chunks):
    return Seed(f"{pattern}-{i:010d}", pattern, "ru", tuple(chunks), ("D",), "train", "QONUNLAR")


BLK = {f"D/П.{k}#1": f"D#B{k // 8}" for k in range(80)}


def test_facts_are_projected_by_new_blocks_not_per_accepted_pair():
    pilot = [_s(1, "single", ["D/П.0#1"]), _s(2, "amends", ["D/П.1#1", "D/П.9#1"])]           # blocks B0, B1
    full = pilot + [_s(3, "single", ["D/П.2#1"]),                                              # B0: cached by pilot
                    _s(4, "amends", ["D/П.17#1", "D/П.33#1"]), _s(5, "single", ["D/П.70#1"])]   # B2, B4, B8
    rates = {"facts_retry": 0.5, "usable": 1.0, "gen_yield": {"single": 0.5, "amends": 0.25}, "views_per_call": 5.0,
             "plant_views": 1.5, "verify_pilot_calls": 7}
    cfg = {"gen_batch": 3, "verify_batch": 2, "plant_share": 0.15}
    p = project_calls(full, pilot, BLK, {"D#B8"}, rates, cfg)
    assert p["blocks"] == {"pilot": 2, "full_new": 2, "cached": 1}
    assert p["ds_facts"] == {"pilot": 3, "full_new": 3, "total": 6}           # ceil(2 × 1.5) each; B8 already cached
    after = project_calls(full, pilot, BLK, {"D#B8", "D#B0", "D#B1"}, rates, cfg)   # after the pilot ran
    assert after["ds_facts"] == p["ds_facts"]                                  # the pilot's blocks are not re-paid
    rates2 = dict(rates, gen_yield={"single": 0.01, "amends": 0.01})
    assert project_calls(full, pilot, BLK, {"D#B8"}, rates2, cfg)["ds_facts"] == p["ds_facts"]   # yield-independent
    assert p["ds_generate"] == {"pilot": 1, "full_new": 1, "total": 2}
    # verify rest: cands 0.5 + 0.25 + 0.5 = 1.25; views 0.5·1 + 0.25·3 + 0.5·1 = 1.75; plants 1.25/2·.15 = .09375
    assert p["ds_verify"] == {"pilot": 7, "full_new": 1, "total": 8}
    assert p["total"]["total"] == 16
    assert views_of(1) == 1 and views_of(3) == 4 and blocks_of(pilot, BLK) == {"D#B0", "D#B1"}
