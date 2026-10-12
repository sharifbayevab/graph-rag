"""M7 Task 8: probe metrics @5 (set_metrics, mean ± SE per type, the spec §5 go / no-go numbers) and the oracle
episode diagnostics (structure / AMENDS use, EXT reached, steps). Ruling R2: grl/hard/metrics.py, not scripts/."""
import math

import pytest

from grl.hard.metrics import (TYPES, go_numbers, mean_se, oracle_diagnostics, oracle_record, set_metrics,
                              summarize_methods, top_k_ids)


def test_set_metrics():
    assert set_metrics(["a", "b", "x"], ["a", "b"]) == {"recall": 1.0, "all": True}
    assert set_metrics(["a", "x"], ["a", "b"]) == {"recall": 0.5, "all": False}


def test_set_metrics_edge_cases():
    assert set_metrics([], ["a"]) == {"recall": 0.0, "all": False}
    assert set_metrics(["a", "a"], ["a"]) == {"recall": 1.0, "all": True}       # duplicates count once
    with pytest.raises(ValueError):
        set_metrics(["a"], [])


def test_top_k_ids_keeps_order_and_drops_repeats():
    assert top_k_ids(["c", "a", "c", "b", "d", "e", "f"], 5) == ["c", "a", "b", "d", "e"]
    assert top_k_ids(["a"], 5) == ["a"]


def test_mean_se():
    m, se = mean_se([1.0, 0.0, 1.0, 0.0])
    assert m == 0.5 and se == pytest.approx(math.sqrt(1 / 3) / 2)               # sd(ddof=1) / sqrt(n)
    assert mean_se([0.7]) == (0.7, 0.0)
    assert mean_se([]) == (None, None)


def _rows(method, qtype, alls, recalls=None):
    recalls = recalls or [1.0 if a else 0.0 for a in alls]
    return [{"method": method, "qtype": qtype, "qid": f"{qtype}-{i}", "all": a, "recall": r}
            for i, (a, r) in enumerate(zip(alls, recalls))]


def test_summarize_methods_per_type_and_pooled():
    rows = (_rows("dense", "count", [False, False]) + _rows("dense", "version", [True, False], [1.0, 0.5])
            + _rows("dense", "compare", [True, True]))
    t = summarize_methods(rows)
    assert set(t["dense"]) == set(TYPES) | {"pooled"}
    assert t["dense"]["count"]["all"] == 0.0 and t["dense"]["count"]["n"] == 2
    assert t["dense"]["version"]["recall"] == 0.75 and t["dense"]["version"]["all"] == 0.5
    assert t["dense"]["version"]["all_se"] == pytest.approx(0.5)
    assert t["dense"]["pooled"]["n"] == 6 and t["dense"]["pooled"]["all"] == pytest.approx(0.5)


def test_summarize_refuses_a_method_with_repeated_qids():
    with pytest.raises(ValueError):
        summarize_methods(_rows("dense", "count", [True]) * 2)


def _table(per):
    return {m: {t: {"all": v, "all_se": 0.0, "recall": v, "recall_se": 0.0, "n": 30} for t, v in zip(TYPES, vals)}
            for m, vals in per.items()}


def test_go_numbers_rule():
    # count: best base 0.1, version: best 0.4, compare: best 0.8 -> 2 types below 0.5 -> rule says GO
    t = _table({"dense": (0.0, 0.3, 0.8), "bm25": (0.1, 0.4, 0.6), "oracle": (0.9, 0.4, 0.95)})
    g = go_numbers(t, ["dense", "bm25"], "oracle")
    assert g["best_baseline_all"] == {"count": 0.1, "version": 0.4, "compare": 0.8}
    assert g["best_baseline_name"] == {"count": "bm25", "version": "bm25", "compare": "dense"}
    assert g["types_below"] == ["count", "version"] and g["rule_go"] is True
    assert g["oracle_all"] == {"count": 0.9, "version": 0.4, "compare": 0.95}
    assert g["oracle_below"] == ["version"]


def test_go_numbers_no_go_when_one_type_below():
    t = _table({"dense": (0.2, 0.6, 0.8), "oracle": (0.9, 0.9, 0.9)})
    g = go_numbers(t, ["dense"], "oracle")
    assert g["types_below"] == ["count"] and g["rule_go"] is False and g["oracle_below"] == []


def test_go_numbers_threshold_is_strict():
    t = _table({"dense": (0.5, 0.5, 0.5), "oracle": (0.5, 0.5, 0.5)})
    g = go_numbers(t, ["dense"], "oracle")
    assert g["types_below"] == [] and g["oracle_below"] == []


# ---- oracle episodes on the M7 toy (tests/test_m7_struct_env: D/СТ.1 -AMENDS-> EXT/k/СТ.61) ----------------------

def test_oracle_record_reads_the_path_on_the_struct_toy():
    from grl.train.expert import expert_search
    from grl.train.fbeta import GRLWeights
    from tests.test_m7_struct_env import make
    env, q = make()
    gold = frozenset({env.data.chunk_row["STRUCT/EXT/k"]})
    res = expert_search(env, q, gold, GRLWeights(0.5, 0.2, 0.2, 0.1), depth=5)
    rec = oracle_record(env, res, ["EXT/k"])
    assert "STRUCT/EXT/k" in rec["context"]
    assert rec["used_structure"] is True
    assert rec["ext_covered"] is True and rec["ext_touched"] is True
    assert rec["n_steps"] == len(rec["path"]) and rec["path"][-1]["action"] != "stop"
    assert all(set(p) == {"action", "added"} for p in rec["path"])
    assert rec["used_amends"] == any(p["action"].startswith("graph:AMENDS") for p in rec["path"])


def test_oracle_record_without_ext_target():
    from grl.train.expert import expert_search
    from grl.train.fbeta import GRLWeights
    from tests.test_m7_struct_env import make
    env, q = make()
    gold = frozenset({env.data.chunk_row["D/СТ.1#1"]})
    rec = oracle_record(env, expert_search(env, q, gold, GRLWeights(0.5, 0.2, 0.2, 0.1), depth=5), [])
    assert rec["ext_covered"] is None and rec["ext_touched"] is None
    assert "D/СТ.1#1" in rec["context"]


def test_oracle_diagnostics_shares():
    recs = [{"qtype": "version", "used_structure": True, "used_amends": True, "ext_covered": True,
             "ext_touched": True, "n_steps": 3, "all": True, "exact": True, "capped": False, "seconds": 1.0},
            {"qtype": "version", "used_structure": False, "used_amends": False, "ext_covered": False,
             "ext_touched": True, "n_steps": 1, "all": False, "exact": True, "capped": False, "seconds": 2.0},
            {"qtype": "count", "used_structure": True, "used_amends": False, "ext_covered": None,
             "ext_touched": None, "n_steps": 2, "all": True, "exact": False, "capped": True, "seconds": 3.0}]
    d = oracle_diagnostics(recs)
    assert d["version"]["structure_share"] == 0.5 and d["version"]["amends_share"] == 0.5
    assert d["version"]["ext_covered_share"] == 0.5 and d["version"]["ext_touched_share"] == 1.0
    assert d["version"]["steps_mean"] == 2.0 and d["version"]["all"] == 0.5 and d["version"]["n"] == 2
    assert d["count"]["ext_covered_share"] is None and d["count"]["capped_share"] == 1.0
    assert d["pooled"]["n"] == 3 and d["pooled"]["exact_share"] == pytest.approx(2 / 3)


# ---- scripts/hard_probe_eval.py: CLI guards and the offline tables / figure ---------------------------------------

def _fake_out():
    from tests.scripts_import import load_script
    ev = load_script("hard_probe_eval")
    rows = []
    for m in ev.METHODS:
        for t in TYPES:
            alls = [True, False, m == ev.ORACLE]
            rows += [{"method": m, "qtype": t, "qid": f"{t}-{i}", "all": a, "recall": float(a)}
                     for i, a in enumerate(alls)]
    table = summarize_methods(rows)
    return ev, {"methods": list(ev.METHODS), "table": table,
                "go": go_numbers(table, list(ev.BASELINES) + [ev.RANKER], ev.ORACLE), "oracle_diag": {}}


def test_eval_set_accepts_only_the_probe_keys():
    from tests.scripts_import import load_script
    ev = load_script("hard_probe_eval")
    cfg = {"env": {"K_max": 24}, "expert": {"max_states": 20000, "cap_beam": 256}, "reward": {}}
    ev._set(cfg, "env.K_max=5")
    assert cfg["env"]["K_max"] == 5
    with pytest.raises(SystemExit):
        ev._set(cfg, "env.T_max=3")
    with pytest.raises(SystemExit):
        ev._set(cfg, "reward.chunks_beta=2")


def test_eval_needs_ship_and_out_or_from():
    from tests.scripts_import import load_script
    ev = load_script("hard_probe_eval")
    with pytest.raises(SystemExit):
        ev.main(["--ship", "x"])


def test_offline_tables_and_figure(tmp_path):
    ev, out = _fake_out()
    md = ev.tables(out)
    lines = md.splitlines()
    assert len(lines) == 2 + len(ev.METHODS) and lines[2].startswith("| dense@5 |")
    assert "0,333 ± 0,333" in lines[2] and "1,000 ± 0,000" not in lines[2]
    assert lines[-1].startswith("| оракул struct d5 |") and "0,667 ± 0,333" in lines[-1]
    png = tmp_path / "f.png"
    ev.plot(out, png)
    assert png.stat().st_size > 10000


def test_verify_ship_hashes(tmp_path):
    import hashlib
    from tests.scripts_import import load_script
    ev = load_script("hard_probe_eval")
    ship = tmp_path / "ship"
    (ship / "rankings").mkdir(parents=True)
    (ship / "pairs.jsonl").write_text("{}\n")
    (ship / "rankings" / "latest.json").write_text("{}")
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    shas = tmp_path / "a.sha256"
    shas.write_text(f"{h(ship / 'pairs.jsonl')}  data/s/ship/pairs.jsonl\n"
                    f"{h(ship / 'rankings' / 'latest.json')}  data/s/ship/rankings/latest.json\n"
                    f"{'0' * 64}  data/s/ship/manifest.json\n"            # listed, not shipped: skipped
                    f"{'0' * 64}  data/other/ship/pairs.jsonl\n")
    assert ev.verify_ship_hashes(ship, shas, "data/s/ship") == 2
    (ship / "pairs.jsonl").write_text("{\"x\": 1}\n")
    with pytest.raises(SystemExit):
        ev.verify_ship_hashes(ship, shas, "data/s/ship")
    with pytest.raises(SystemExit):                                          # nothing listed for this ship
        ev.verify_ship_hashes(ship, shas, "data/none/ship")


# ---- R28 diagnostic oracles (grl.hard.oracle.diag_search) on the M7 struct toy ------------------------------------

def _w():
    from grl.train.fbeta import GRLWeights
    return GRLWeights(0.5, 0.2, 0.2, 0.1)


def test_kmax_oracle_counts_limit_ended_finals():
    # K_max = 1: every fragment-adding step ends the episode with reason k_max, so the STOP-only expert can only
    # STOP at reset (empty context); oracle_kmax keeps the k_max-ended state that holds the gold chunk
    from grl.hard.oracle import diag_search
    from grl.train.expert import expert_search
    from tests.test_m7_struct_env import make
    env, q = make({"K_max": 1})
    gold = frozenset({env.data.chunk_row["D/СТ.1#1"]})
    ex = expert_search(env, q, gold, _w(), depth=5)
    assert ex["final"].context == ()
    k = diag_search(env, q, gold, _w(), 5, "kmax")
    assert k["final"].context == (env.data.chunk_row["D/СТ.1#1"],) and k["reason"] == "k_max"
    assert k["score"] > ex["score"] and k["exact"] is True
    rec = oracle_record(env, k, [])
    assert rec["context"] == ["D/СТ.1#1"] and rec["path"][-1]["action"] != "stop" and rec["n_steps"] == 1


def test_all5_oracle_maximises_all_then_recall_and_dominates():
    from grl.hard.oracle import diag_search
    from grl.train.expert import expert_search
    from tests.test_m7_struct_env import make
    env, q = make()
    for gold_ids in (["STRUCT/EXT/k"], ["D/СТ.1#1", "STRUCT/EXT/k"], ["STRUCT/D", "STRUCT/EXT/k", "D/СТ.1#1"]):
        gold = frozenset(env.data.chunk_row[g] for g in gold_ids)
        ex = expert_search(env, q, gold, _w(), depth=5)
        k = diag_search(env, q, gold, _w(), 5, "kmax")
        a = diag_search(env, q, gold, _w(), 5, "all5")
        m = {n: set_metrics([env.data.chunk_ids[c] for c in r["final"].context], gold_ids)
             for n, r in (("ex", ex), ("k", k), ("a", a))}
        assert k["score"] >= ex["score"] - 1e-12                      # superset of the expert's finals
        assert (m["a"]["all"], m["a"]["recall"]) >= (m["k"]["all"], m["k"]["recall"])
        assert (m["a"]["all"], m["a"]["recall"]) >= (m["ex"]["all"], m["ex"]["recall"])
        assert m["a"]["all"] is True                                  # every toy gold is reachable within K_max 5


def test_all5_prefers_all_over_reward():
    # K_max = 2 and gold = both STRUCT rows + the chunk (3 > K_max): all@K is impossible, so all5 maximises recall
    # (2 of 3), whatever R prefers
    from grl.hard.oracle import diag_search
    from tests.test_m7_struct_env import make
    env, q = make({"K_max": 2})
    gold_ids = ["STRUCT/D", "STRUCT/EXT/k", "D/СТ.1#1"]
    gold = frozenset(env.data.chunk_row[g] for g in gold_ids)
    a = diag_search(env, q, gold, _w(), 5, "all5")
    assert len(gold & a["final"].ctx_set) == 2


def test_diag_search_guards():
    from grl.hard.oracle import diag_search
    from tests.test_m7_struct_env import make
    env, q = make()
    gold = frozenset({env.data.chunk_row["D/СТ.1#1"]})
    with pytest.raises(ValueError):
        diag_search(env, q, gold, _w(), 5, "beam")
    with pytest.raises(ValueError):
        diag_search(env, q, frozenset(), _w(), 5, "kmax")
    with pytest.raises(ValueError):
        diag_search(env, q, gold, _w(), 5, "kmax", max_states=1)


def test_go_numbers_ties_and_empty():
    t = _table({"dense": (0.0, 0.3, 0.8), "bm25": (0.0, 0.3, 0.6), "oracle": (1.0, 1.0, 1.0)})
    g = go_numbers(t, ["dense", "bm25"], "oracle")
    assert g["best_baseline_name"] == {"count": "tie(all 0)", "version": ["dense", "bm25"], "compare": "dense"}
    with pytest.raises(ValueError):
        go_numbers(t, ["nope"], "oracle")


def test_ship_hashes_only_the_manifest_may_be_missing(tmp_path):
    import hashlib
    from tests.scripts_import import load_script
    ev = load_script("hard_probe_eval")
    ship = tmp_path / "ship"
    (ship / "rankings").mkdir(parents=True)
    for n in ("pairs.jsonl", "rankings/latest.json"):
        (ship / n).write_text("{}\n")
    h = hashlib.sha256(b"{}\n").hexdigest()
    shas = tmp_path / "a.sha256"
    shas.write_text(f"{h}  s/ship/pairs.jsonl\n{h}  s/ship/rankings/latest.json\n{'0' * 64}  s/ship/splits.json\n")
    with pytest.raises(SystemExit):                                          # a listed non-manifest file is gone
        ev.verify_ship_hashes(ship, shas, "s/ship")


def test_figure_with_the_reachability_series(tmp_path):
    ev, out = _fake_out()
    rows = [{"method": "oracle_all5", "qtype": t, "qid": f"{t}-{i}", "all": True, "recall": 1.0}
            for t in TYPES for i in range(3)]
    out["table"].update(summarize_methods(rows))
    out["methods"].append("oracle_all5")
    assert "достижимость all@5 d5" in ev.tables(out)
    ev.plot(out, tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 10000


# ---- R29: zero-Δ bridges and an independent brute force ----------------------------------------------------------

def test_reachability_takes_a_zero_delta_bridge_the_pruned_search_cannot():
    # K_max = 1, gold = STRUCT/EXT/k: at reset the structure candidates are the roots of V₀ ∪ F₀ = {D} only, and any
    # fragment-adding step fills K_max. Only graph:AMENDS:1 (selects the chunkless EXT/k/СТ.61, adds nothing) makes
    # EXT/k a candidate root, then structure:1 adds STRUCT/EXT/k
    from grl.hard.oracle import diag_search
    from tests.test_m7_struct_env import amends_m1, make
    env, q = make({"K_max": 1})
    gold = frozenset({env.data.chunk_row["STRUCT/EXT/k"]})
    pruned = diag_search(env, q, gold, _w(), 5, "kmax")
    assert gold.isdisjoint(pruned["final"].ctx_set)
    for mode in ("all5", "noprune"):
        r = diag_search(env, q, gold, _w(), 5, mode)
        assert gold <= r["final"].ctx_set, mode
        assert r["final"].history[0].action == amends_m1(env) and r["final"].history[0].added == ()
        assert r["n_zero_steps"] >= 1
    rec = oracle_record(env, diag_search(env, q, gold, _w(), 5, "all5"), ["EXT/k"])
    assert [p["action"] for p in rec["path"]] == ["graph:AMENDS:1", "structure:1"] and rec["ext_covered"] is True


def _brute(env, q, gold, w, depth):
    """Every action sequence up to `depth` steps (no merging, no pruning); finals = done states or STOP-valid live
    states. Returns (max (all, recall), max R)."""
    import numpy as np
    from graphrag.env.mdp import InvalidAction
    from grl.train.fbeta import grl_reward
    best_a, best_r = None, None

    def visit(s, d):
        nonlocal best_a, best_r
        if s.done or env.mask(q, s)[0]:
            h = len(gold & s.ctx_set)
            a = (h == len(gold), h / len(gold))
            r = float(grl_reward(env.data, q.depth, s.context, gold, w, env.limits.T_max).total)
            best_a = a if best_a is None or a > best_a else best_a
            best_r = r if best_r is None or r > best_r else best_r
        if s.done or d == depth:
            return
        for act in np.flatnonzero(env.mask(q, s)).tolist():
            if act:
                try:
                    visit(env.step(q, s, act, check=False), d + 1)
                except InvalidAction:
                    pass
    visit(env.reset(q), 0)
    return best_a, best_r


@pytest.mark.parametrize("kmax", [1, 2, 5])
def test_diag_search_matches_an_independent_brute_force(kmax):
    from grl.hard.oracle import diag_search
    from tests.test_m7_struct_env import make
    env, q = make({"K_max": kmax})
    golds = (["STRUCT/EXT/k"], ["D/СТ.1#1", "STRUCT/EXT/k"], ["STRUCT/D", "STRUCT/EXT/k", "D/СТ.1#1"], ["STRUCT/D"])
    for gold_ids in golds:
        gold = frozenset(env.data.chunk_row[g] for g in gold_ids)
        ba, br = _brute(env, q, gold, _w(), 4)
        a = diag_search(env, q, gold, _w(), 4, "all5")["final"]
        h = len(gold & a.ctx_set)
        assert (h == len(gold), h / len(gold)) == ba, (kmax, gold_ids)
        r = diag_search(env, q, gold, _w(), 4, "noprune")
        assert r["score"] == pytest.approx(br), (kmax, gold_ids)
