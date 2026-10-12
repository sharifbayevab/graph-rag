# tests/test_m7_compare_full.py -- M7 part 2, Task 6: full Q3 -- probe-excluded pairs, the approval-only resolution
# filter (R26), the ds_hard compare judge with plants, A/B/C reader consensus, splits against Q1 / Q2. Fake LLM only.
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphrag.kg.llm import KgLLM
from grl.hard import judge as J
from grl.hard import readers as R
from grl.hard.compare import approval_only, pair_candidates, pick_doc_pairs
from grl.hard.splits import assign_q3_splits, fixed_doc_splits
from tests.test_m7_compare import DATES, _toy

FIX = Path(__file__).parent / "fixtures" / "m7_resolutions_approval.json"


# --- R26: approval-only resolutions ------------------------------------------------------------------------------

def test_approval_heuristic_on_ten_labelled_probe_resolutions():
    rows = json.loads(FIX.read_text(encoding="utf-8"))["resolutions"]
    assert len(rows) == 10 and Counter(r["label"] for r in rows) == {"approval_only": 5, "substantive": 5}
    got = {r["doc_id"]: approval_only([tuple(c) for c in r["chunks"]]) for r in rows}
    assert got == {r["doc_id"]: r["label"] == "approval_only" for r in rows}          # 10 / 10


def test_approval_heuristic_signature_fragments_and_languages():
    sig = [("Q/ПРЕАМБ.1#1", "Лойиҳа Ўзбекистон Республикаси Олий Мажлиси Сенатининг Қ А Р О Р И"),
           ("Q/П.1#1", "“Рақобат тўғрисида”ги Ўзбекистон Республикаси Қонуни маъқуллансин."),
           ("Q/П.2#1", "Ушбу Қарор қабул қилинган кундан эътиборан кучга киради."),
           ("Q/П.3#1", "Ўзбекистон Республикаси"), ("Q/П.4#1", "Олий Мажлиси Сенатининг Раиси")]
    assert approval_only(sig)                                       # a signature line split into «points» is not content
    ru = [("Q/П.1#1", "1. Одобрить Закон Республики Узбекистан «О конкуренции»."),
          ("Q/П.2#1", "2. Настоящее постановление вступает в силу со дня его принятия.")]
    assert approval_only(ru)
    ru2 = ru + [("Q/П.3#1", "3. Кабинету Министров в двухмесячный срок привести решения правительства в соответствие "
                            "с настоящим Законом и проинформировать Сенат о проделанной работе.")]
    assert not approval_only(ru2)
    lat = [("Q/П.1#1", "1. “Raqobat toʻgʻrisida”gi Oʻzbekiston Respublikasi Qonuni maʼqullansin."),
           ("Q/П.2#1", "2. Ushbu Qaror qabul qilingan kundan eʼtiboran kuchga kiradi.")]
    assert approval_only(lat)
    assert not approval_only([])                                    # no text: never filtered on missing data


def test_pair_candidates_drops_approval_only_resolution():
    nodes, edges, nc = _toy()
    appr = [("R1/П.1#1", "1. “Qonun 2” Ўзбекистон Республикаси Қонуни маъқуллансин."),
            ("R1/П.2#1", "2. Ушбу Қарор қабул қилинган кундан эътиборан кучга киради.")]
    st, why = Counter(), {}
    c = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=DATES, stats=st, why=why, res_texts={"R1": appr})
    assert c["law_resolution"] == [] and st["approval_only"] == 1 and why[("L2", "R1")] == "approval_only"
    subst = appr + [("R1/П.3#1", "3. Вазирлар Маҳкамаси икки ой муддатда ҳукумат қарорларини ушбу Қонунга "
                                 "мувофиқлаштирсин ва Сенатга ахборот киритсин.")]
    c = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=DATES, res_texts={"R1": subst})
    assert [(a, b) for a, b, _ in c["law_resolution"]] == [("L2", "R1")]
    c = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=DATES)                 # off without texts (part 1)
    assert [(a, b) for a, b, _ in c["law_resolution"]] == [("L2", "R1")]


# --- probe exclusion ---------------------------------------------------------------------------------------------

def test_probe_pair_is_excluded_any_order_any_kind():
    nodes, edges, nc = _toy()
    full = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    assert ("L2", "R1", "law_resolution") in full
    got = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, exclude_pairs=[("R1", "L2")])
    assert not any({a, b} == {"L2", "R1"} for a, b, _ in got)
    assert len(got) == len(full) - 1                                # the rest is unchanged
    st = Counter()
    pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, exclude_pairs=[("R1", "L2")], stats=st)
    assert st["probe_excluded:law_resolution"] == 1


def test_caps_are_parameters():
    nodes, edges, nc = _toy()
    one = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    two = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, max_per_doc_kind=2)
    assert sum(k == "law_conclusion" for *_, k in one) == 1 and sum(k == "law_conclusion" for *_, k in two) == 2
    assert max(Counter(d for a, b, _ in two for d in (a, b)).values()) <= 2


def test_load_probe_pairs_union_and_loud_errors(tmp_path):
    from scripts.hard_probe import load_probe_pairs
    vp = tmp_path / "valid.json"
    vp.write_text(json.dumps({"existing": {"packets": {"cmp-000": {"pair": ["A", "B", "law_conclusion"]}}},
                              "pairs": [["A", "B", "law_conclusion"], ["C", "D", "law_resolution"]]}))
    pd = tmp_path / "packets"
    pd.mkdir()
    (pd / "cmp-001.json").write_text(json.dumps({"packet_id": "cmp-001", "kind": "stenogram_law",
                                                 "pair": ["S", "L", "stenogram_law"], "docs": []}))
    got = load_probe_pairs(vp, pd)
    assert got == {frozenset(("A", "B")), frozenset(("S", "L"))}     # probe packets, not every valid pair
    with pytest.raises(SystemExit):
        load_probe_pairs(tmp_path / "missing.json", pd)
    with pytest.raises(SystemExit):
        load_probe_pairs(vp, tmp_path / "empty")


# --- ds_hard compare judge ---------------------------------------------------------------------------------------

def _cmp_items(n):
    items, packets = [], {}
    for i in range(n):
        pid = f"cmp-{i:03d}"
        a, b = f"L{i}", f"C{i}"
        packets[pid] = {"packet_id": pid, "kind": "law_conclusion", "lang": "ru", "docs": [
            {"doc_id": a, "title": f"Закон {i}", "doc_type": "law_text", "date": "2025-01-01",
             "chunks": [{"chunk_id": f"{a}#1", "text": f"норма {i} закона: срок {i} дней"}]},
            {"doc_id": b, "title": f"Заключение {i}", "doc_type": "committee_conclusion", "date": "2025-02-01",
             "chunks": [{"chunk_id": f"{b}#1", "text": f"комитет {i} оценил срок {i} дней"}]}]}
        items.append({"qid": f"q{i:03d}", "qtype": "compare", "template": "law_conclusion", "lang": "ru",
                      "question": f"Как комитет {i} оценил срок закона {i}?", "answer": f"ANS{i}: срок {i} дней",
                      "gold": [f"{a}#1", f"{b}#1"], "docs": [a, b], "meta": {"packet_id": pid}})
    return items, packets


class FakeCmpClient:
    """Judges like a careful reader would on the toy: needs_both iff fragments of both documents are shown,
    supported iff the answer's ANS<i> tag matches the fragments' number. `blind` says yes to everything."""

    def __init__(self, blind=False):
        self.calls, self.blind = 0, blind
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        inp = json.loads(kw["messages"][1]["content"])
        out = []
        for q in inp["items"]:
            docs = {f["doc"] for f in q["fragments"]}
            tag = q["answer"].split(":")[0][3:]
            ok_sup = all(f" {tag} " in f" {f['text']} " for f in q["fragments"])
            out.append({"key": q["key"], "needs_both": self.blind or len(docs) == 2,
                        "supported": self.blind or (ok_sup and len(docs) == 2), "why": "toy"})
        ans = {"verdicts": out}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(ans)))])


def _llm(tmp_path, client, total=900):
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": total, "cache_dir": str(tmp_path),
                   "passes": {"ds_hard": {"prompt_version": 1, "max_calls": 900}}}}
    return KgLLM(cfg=cfg, cache_root=tmp_path, client=client, allow_api=True)


def test_cmp_plants_doc_drop_and_answer_swap():
    items, packets = _cmp_items(40)
    plants = J.plan_cmp_plants(items, packets, seed=5, plant_share=0.15, min_n=6)
    by = Counter(p["kind"] for p in plants)
    assert by == {"doc_drop": 6, "answer_swap": 6}
    for p in plants:
        base = next(it for it in items if it["qid"] == p["qid"])
        if p["kind"] == "doc_drop":
            assert p["answer"] == base["answer"] and len({f["doc"] for f in p["fragments"]}) == 1
        else:
            assert p["answer"] != base["answer"] and len({f["doc"] for f in p["fragments"]}) == 2
    assert J.plan_cmp_plants(items, packets, seed=5, plant_share=0.15, min_n=6) == plants     # deterministic


def test_judge_compare_gate_and_verdicts(tmp_path):
    items, packets = _cmp_items(40)
    res = J.judge_compare(items, packets, _llm(tmp_path / "a", FakeCmpClient()), plant_share=0.15, seed=5, min_n=6)
    assert set(res["verdicts"]) == {it["qid"] for it in items}
    assert all(v["needs_both"] and v["supported"] and v["ok"] for v in res["verdicts"].values())
    g = J.gate(res["plants"], min_n=6, kinds=J.CMP_PLANT_KINDS)
    assert g["ok"] and g["by_kind"]["doc_drop"]["sensitivity"] == 1.0 and g["by_kind"]["answer_swap"]["judged"] == 6
    blind = J.judge_compare(items, packets, _llm(tmp_path / "b", FakeCmpClient(blind=True)), seed=5, min_n=6)
    assert not J.gate(blind["plants"], min_n=6, kinds=J.CMP_PLANT_KINDS)["ok"]
    g30 = J.gate(res["plants"], kinds=J.CMP_PLANT_KINDS)                         # default 30 per kind: too few
    assert not g30["ok"]


def test_judge_compare_plants_never_share_a_call_with_their_base(tmp_path):
    items, packets = _cmp_items(40)
    calls = J.pack_calls(J.cmp_call_items(items, packets, J.plan_cmp_plants(items, packets, 5, 0.15, 6)),
                         per_call=J.PER_CALL_CMP, seed=5)
    for c in calls:
        assert len(c) <= J.PER_CALL_CMP
        assert len({x["qid"] for x in c}) == len(c)


# --- readers + consensus -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,c,want", [("GOOD", "GOOD", None, "accepted"), ("GOOD", "GOOD", "GOOD", "accepted"),
                                        ("GOOD", "GOOD", "BAD", "rejected"), ("GOOD", "BAD", None, "rejected"),
                                        ("BAD", "GOOD", None, "rejected"), ("GOOD", None, None, "pending"),
                                        (None, "BAD", None, "rejected")])
def test_consensus_rule(a, b, c, want):
    assert R.consensus(a, b, c) == want


def test_reader_packets_gold_only_and_size():
    items, packets = _cmp_items(25)
    for p in packets.values():                       # a shown but non-gold chunk must not reach the readers
        p["docs"][0]["chunks"].append({"chunk_id": p["docs"][0]["doc_id"] + "#2", "text": "лишний фрагмент"})
    rp = R.reader_packets(items, packets, n=10, seed=3)
    assert [len(p["items"]) for p in rp] == [10, 10, 5]
    seen = [x for p in rp for x in p["items"]]
    assert sorted(x["pair_id"] for x in seen) == sorted(it["qid"] for it in items)
    assert all(len(x["fragments"]) == 2 and all(f["id"].endswith("#1") for f in x["fragments"]) for x in seen)
    assert {d["key"] for x in seen for d in x["documents"]} == {"A", "B"}
    assert R.reader_packets(items, packets, n=10, seed=3) == rp


def test_ingest_verdicts_and_consensus_rows():
    items, _ = _cmp_items(4)
    q = [it["qid"] for it in items]
    va = [{"pair_id": q[0], "verdict": "GOOD", "reasons": []}, {"pair_id": q[1], "verdict": "GOOD", "reasons": []},
          {"pair_id": q[2], "verdict": "GOOD", "reasons": [], "reader": "opus-a2"}]
    vb = [{"pair_id": q[0], "verdict": "GOOD", "reasons": []}, {"pair_id": q[1], "verdict": "GOOD", "reasons": []},
          {"pair_id": q[2], "verdict": "BAD", "reasons": [{"criterion": "q2", "note": "x"}]},
          {"pair_id": q[3], "verdict": "GOOD", "reasons": []}]
    vc = [{"pair_id": q[1], "verdict": "BAD", "reasons": [{"criterion": "q1", "note": "one doc enough"}]},
          "not json", {"pair_id": q[0], "verdict": "MAYBE"}]
    rows, st = R.reader_consensus(items, va, vb, vc, reader_a="fable")
    status = {r["qid"]: r["status"] for r in rows}
    assert status == {q[0]: "accepted", q[1]: "rejected", q[2]: "rejected", q[3]: "pending"}
    a_model = {r["qid"]: r["readers"]["A"]["reader"] for r in rows if r["readers"]["A"]}
    assert a_model == {q[0]: "fable", q[1]: "fable", q[2]: "opus-a2"}
    assert st["malformed:C"] == 2 and st["accepted"] == 1 and st["pending"] == 1


# --- splits against the old set and Q1 / Q2 ----------------------------------------------------------------------

def test_q3_splits_follow_old_and_q1q2():
    old = {"OLD-T": "train", "OLD-X": "test"}
    q12 = [{"qid": "c1", "docs": ["EXT/k"], "gold_docs": ["EXT/k"], "split": "val"},
           {"qid": "v1", "docs": ["EXT/k", "LAW-V"], "gold_docs": ["LAW-V"], "split": "val"}]
    fixed = fixed_doc_splits(old, q12)
    assert fixed["LAW-V"] == "val" and fixed["OLD-T"] == "train"
    q3 = [{"qid": "x1", "qtype": "compare", "docs": ["LAW-V", "C1"], "gold_docs": ["LAW-V", "C1"]},
          {"qid": "x2", "qtype": "compare", "docs": ["OLD-T", "C2"], "gold_docs": ["OLD-T", "C2"]},
          {"qid": "x3", "qtype": "compare", "docs": ["OLD-T", "LAW-V"], "gold_docs": ["OLD-T", "LAW-V"]},
          {"qid": "x4", "qtype": "compare", "docs": ["N1", "N2"], "gold_docs": ["N1", "N2"]}]
    sp = assign_q3_splits(q3, old, q12, seed=1)
    assert sp["x1"] == "val" and sp["x2"] == "train" and sp["x3"] is None and sp["x4"] in ("train", "val", "test")
    with pytest.raises(ValueError):                                 # Q1 / Q2 inconsistent with the old set: loud
        fixed_doc_splits({"LAW-V": "test"}, q12)


def test_topup_with_raised_caps_extends_the_first_selection():
    nodes, edges, nc = _toy()
    first = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, exclude_pairs=[("R1", "L2")])
    more = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, exclude_pairs=[("R1", "L2")],
                          preset=first, exclude=first, max_per_doc=3, max_per_doc_kind=3)
    assert more and not set(more) & set(first) and not any({a, b} == {"L2", "R1"} for a, b, _ in more)
    per = Counter(d for a, b, _ in first + more for d in (a, b))
    assert max(per.values()) <= 3


def test_full_instructions_adapt_the_probe_v2():
    from graphrag import config
    from scripts.hard_probe import PROBE_INSTRUCTIONS, full_compare_instructions
    txt = full_compare_instructions(config.path(PROBE_INSTRUCTIONS).read_text(encoding="utf-8"), "/w/packets",
                                    "/w/compare_raw_<k>.jsonl")
    assert "version 3" in txt and "(R26)" in txt and "/w/packets/cmp-NNN.json" in txt and "/w/compare_raw_<k>.jsonl" in txt
    assert "There is no judge" not in txt and "data/hard/probe/compare_packets" not in txt
    with pytest.raises(SystemExit):
        full_compare_instructions("no anchors here", "/w", "/w/x")


def test_r53_prompt_v2_and_root_disjoint_donors():
    assert J.CMP_PROMPT_VERSION == 2 and "answer_about" in J.CMP_SCHEMA["properties"]["verdicts"]["items"]["required"]
    assert "MUST be false" in J.CMP_PROMPT
    items, packets = _cmp_items(40)
    roots = {it["docs"][0]: {"EXT/gpk"} for it in items[:20]}            # items 0..19 share the amended act
    pl = J.plan_cmp_plants(items, packets, seed=5, plant_share=0.15, min_n=6, roots=roots)
    by = {it["qid"]: it for it in items}
    for p in [p for p in pl if p["kind"] == "answer_swap"]:
        a, d = by[p["qid"]], by[p["donor"]]
        assert not set(a["docs"]) & set(d["docs"])
        assert not (J.item_roots(a, roots) & J.item_roots(d, roots))


def test_r53_prescreen_skips_pairs_with_two_fixed_splits():
    nodes, edges, nc = _toy()
    st = Counter()
    got = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, stats=st,
                         fixed_splits={"L2": "train", "R1": "test", "L4": "train"})
    assert not any({a, b} == {"L2", "R1"} for a, b, _ in got) and st["r32_prescreen:law_resolution"] == 1
    assert any({a, b} == {"L2", "L4"} for a, b, _ in got)                 # same split: kept


def test_r54_rubric_v2_identifier_rule():
    assert R.RUBRIC_VERSION == 2 and "version 2, ruling R54" in R.RUBRIC_V2
    assert "A date is never required" in R.RUBRIC_V2 and "entailed by the gold fragments alone" in R.RUBRIC_V2
    assert "its date or number where the text gives them" not in R.RUBRIC_V2          # the v1 contradiction is gone
    assert "q1 **needs both documents**" in R.RUBRIC_V2                                # q1 unchanged (strict)


def test_r54_repair_packets_carry_reasons_verbatim():
    items, packets = _cmp_items(12)
    note = "«Қонунчилик палатаси томонидан қабул қилинган» not in gold"
    vb = [{"pair_id": it["qid"], "verdict": "BAD", "reasons": [{"criterion": "q2", "note": note}]} for it in items]
    vc = [{"pair_id": items[0]["qid"], "verdict": "GOOD", "reasons": []}]
    rows, _ = R.reader_consensus(items, [], vb, vc)
    rp = R.repair_packets(items, packets, {r["qid"]: r["readers"] for r in rows}, "/w/packets", n=10)
    assert [len(p["items"]) for p in rp] == [10, 2] and rp[0]["packet_id"] == "rep-000"
    x = next(i for p in rp for i in p["items"] if i["qid"] == items[0]["qid"])
    assert x["reader_verdicts"]["B"]["reasons"][0]["note"] == note and x["reader_verdicts"]["C"]["verdict"] == "GOOD"
    assert "A" not in x["reader_verdicts"] and x["source_packet"] == "/w/packets/cmp-000.json"
    assert [g["text"] for g in x["gold"]] == ["норма 0 закона: срок 0 дней", "комитет 0 оценил срок 0 дней"]
    assert {d["title"] for d in x["documents"]} == {"Закон 0", "Заключение 0"}
    txt = R.repair_instructions("/w/rep", "/w/rubric.md", "/w/raw_<k>.jsonl", ["rep-000", "rep-001"])
    assert "/w/rep/rep-000.json" in txt and "rep-001.json" in txt and '"refusal"' in txt


def test_r54_final_q3_train_val_only_below_thresholds():
    def rows(n_train, n_test):
        return ([{"qid": f"t{i}", "split": "train", "template": "law_conclusion", "lang": "ru"} for i in range(n_train)]
                + [{"qid": f"x{i}", "split": "test", "template": "law_conclusion", "lang": "ru"} for i in range(n_test)])
    kept, st = R.q3_final(rows(24, 5))
    assert len(kept) == 24 and st["excluded_test"] == [f"x{i}" for i in range(5)] and not st["in_hard_test_report"]
    kept, st = R.q3_final(rows(25, 10))
    assert len(kept) == 35 and st["test_reportable"] and st["excluded_test"] == []
    kept, st = R.q3_final(rows(10, 12))                      # < 30 total: train / val only even with 12 test
    assert len(kept) == 10 and not st["test_reportable"] and st["test_diagnostic"]
