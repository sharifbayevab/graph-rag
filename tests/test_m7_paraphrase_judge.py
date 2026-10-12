"""M7 part 2, Task 5: Q1 / Q2 paraphrase packets, the anchor-checked ingest, the ds_hard judge with plants and its
budget (grl.hard.paraphrase, grl.hard.judge). Fake clients only: no API call."""
import json
from types import SimpleNamespace

import pytest

from graphrag.kg.llm import KgBudgetExceeded, KgLLM
from grl.hard import judge as J
from grl.hard import paraphrase as P

COUNT = {"qid": "m7-count-aaa", "qtype": "count", "template": "count_amends_year", "lang": "uz-Latn",
         "question": "2023-yilda «Davlat boji toʼgʼrisida»ning alohida elementlariga (moddalar, qismlar, bandlar) "
                     "nechta oʻzgartish kiritilgan?",
         "answer": "4", "gold": ["STRUCT/EXT/davlat boji"], "docs": ["EXT/davlat boji"], "meta": {"year": "2023"},
         "gold_docs": ["EXT/davlat boji"], "split": "train"}
VERSION = {"qid": "m7-version-bbb", "qtype": "version", "template": "what_law_changed", "lang": "ru",
           "question": "Что изменил в акте «Soliq kodeksi» закон от 15 апреля 2025 года (вид изменения: дополнение)?",
           "answer": [["QONUNLAR-1", "2025-04-15", "add", "EXT/soliq kodeks/СТ.22/Ч.5"],
                      ["QONUNLAR-1", "2025-04-15", "add", "EXT/soliq kodeks/СТ.31"]],
           "gold": ["QONUNLAR-1/СТ.8/Ч.1#2"], "docs": ["EXT/soliq kodeks", "QONUNLAR-1"],
           "meta": {"law": "QONUNLAR-1", "date": "2025-04-15", "year": "2025", "op": "add", "n_edges": 2},
           "gold_docs": ["QONUNLAR-1"], "split": "val"}
GOLD = {"m7-count-aaa": {"texts": {"STRUCT/EXT/davlat boji": "Davlat boji toʼgʼrisida\nmoddalar: 30; "
                                   "oʻzgartishlar yillar boʻyicha: 2023: 4; 2025: 6"},
                         "exempt": {"STRUCT/EXT/davlat boji": "Davlat boji toʼgʼrisida"}},
        "m7-version-bbb": {"texts": {"QONUNLAR-1/СТ.8/Ч.1#2": "статью 22 дополнить частью пятой следующего "
                                     "содержания налоговые органы вправе запрашивать сведения у коммерческих банков"},
                           "exempt": {}}}
ITEMS = [COUNT, VERSION]


def _ingest(q, item=COUNT):
    return P.para_ingest([{"qid": item["qid"], "question": q}], ITEMS, gold=GOLD)


# --- ingest ------------------------------------------------------------------------------------------------------

def test_normal_paraphrase_is_accepted():
    acc, rej = _ingest("«Davlat boji toʼgʼrisida» hujjatining moddalari, qismlari va bandlariga 2023-yil davomida "
                       "qancha oʻzgartish kiritildi?")
    assert rej == [] and len(acc) == 1
    assert acc[0]["template_question"] == COUNT["question"] and acc[0]["round"] == 0
    acc, rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 апреля 2025 г.?", VERSION)
    assert rej == [] and len(acc) == 1


def test_year_anchor_lost_is_rejected():
    acc, rej = _ingest("«Davlat boji toʼgʼrisida» hujjatining moddalari va bandlariga jami qancha oʻzgartish "
                       "kiritilgan?")
    assert acc == [] and "anchor_number" in rej[0]["reasons"]


def test_answer_number_in_question_is_rejected():
    acc, rej = _ingest("2023-yilda «Davlat boji toʼgʼrisida»ga 4 tadan ortiq oʻzgartish kiritilganmi?")
    assert acc == [] and "answer_in_question" in rej[0]["reasons"]
    acc, rej = _ingest("Что изменил закон от 15 апреля 2025 года в статье 31 «Soliq kodeksi» (дополнения)?", VERSION)
    assert acc == [] and "answer_in_question" in rej[0]["reasons"]          # a Q2 answer element's number


def test_other_anchor_and_form_rejections():
    rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 мая 2025 года?", VERSION)[1]
    assert "anchor_date" in rej[0]["reasons"]
    rej = _ingest("Что поменял в «Soliq kodeksi» закон, принятый 15 апреля 2025 года?", VERSION)[1]
    assert "anchor_op" in rej[0]["reasons"]
    rej = _ingest("Какие дополнения внёс в «Bojxona kodeksi» закон, принятый 15 апреля 2025 года?", VERSION)[1]
    assert "anchor_act" in rej[0]["reasons"]
    rej = _ingest("«Soliq kodeksi»ga 2025-yil 15-apreldagi qonun qanday qoʻshimchalar kiritgan?", VERSION)[1]
    assert "lang" in rej[0]["reasons"]
    rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон от 15 апреля 2025 года " + "очень " * 60, VERSION)[1]
    assert "length" in rej[0]["reasons"]
    rej = _ingest("Какие дополнения от 15 апреля 2025 года: налоговые органы вправе запрашивать сведения у "
                  "коммерческих банков согласно «Soliq kodeksi»?", VERSION)[1]
    assert "leak" in rej[0]["reasons"]
    rej = P.para_ingest([{"qid": "nope", "question": "x"}, {"qid": COUNT["qid"], "question": ""}], ITEMS, gold=GOLD)[1]
    assert rej[0]["reasons"] == ["unknown_qid"] and "empty" in rej[1]["reasons"]


def test_templates_pass_their_own_anchor_checks():
    for it in ITEMS:
        a = P.anchors(it)
        assert P.anchor_problems(it["question"], it, a) == []
        assert P.detect_lang(it["question"]) == it["lang"]


def test_packets_group_by_language_and_carry_anchors():
    items = [dict(COUNT, qid=f"q{i}") for i in range(50)] + [dict(VERSION, qid=f"v{i}") for i in range(10)]
    ps = P.para_packets(items, n_per_packet=40)
    assert [p["lang"] for p in ps].count("uz-Latn") == 1 and [p["lang"] for p in ps].count("ru") == 1
    assert sorted(x["qid"] for p in ps for x in p["items"]) == sorted(x["qid"] for x in items)
    x = next(x for p in ps for x in p["items"] if x["qid"] == "v0")
    assert x["anchors"]["law_date"] == "2025-04-15" and x["anchors"]["change_kind"] == "дополнение"
    assert x["answer"] == VERSION["answer"] and x["question"] == VERSION["question"]
    assert P.para_packets(items, n_per_packet=40) == ps                          # deterministic


# --- plants ------------------------------------------------------------------------------------------------------

def _pairs(n):
    out = []
    for i in range(n):
        it = dict(VERSION, qid=f"v{i:03d}")
        tag = "абвгдежзиклмнопрст"[i % 18] * (1 + i // 18)             # distinct texts -> distinct calls
        out.append(J.make_pair(it, f"Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 апреля 2025 года? "
                                   f"{tag}", P.anchors(it)))
    return out


TITLES = ["Bojxona kodeksi", "Soliq kodeksi", "Yer kodeksi", "Oila kodeksi", "Сув кодекси"]


def test_plants_are_deterministic_and_corrupt_the_anchor():
    pairs = _pairs(40)
    a = J.plan_plants(pairs, seed=7, plant_share=0.15, min_n=5, titles=TITLES)
    assert a == J.plan_plants(pairs, seed=7, plant_share=0.15, min_n=5, titles=TITLES)
    kinds = {p["kind"] for p in a}
    assert kinds == set(J.PLANT_KINDS)
    for p in a:
        base = next(x for x in pairs if x["qid"] == p["qid"])
        assert p["paraphrase"] != base["paraphrase"]
        if p["kind"] == "act_swap":
            assert "«Soliq kodeksi»" not in p["paraphrase"] and "kodeksi»" in p["paraphrase"]
        if p["kind"] == "number_swap":                     # the year or the day of the law date changed
            assert "2025-04-15" not in P.scan(p["paraphrase"])[0]
        if p["kind"] == "op_swap":
            assert "дополн" not in p["paraphrase"]


def test_plant_never_shares_a_call_with_its_base():
    pairs = _pairs(40)
    plants = J.plan_plants(pairs, seed=7, plant_share=0.5, min_n=5, titles=TITLES)
    calls = J.pack_calls(J.call_items(pairs, plants), per_call=10, seed=7)
    assert all(len(c) <= 10 for c in calls)
    assert sum(len(c) for c in calls) == len(pairs) + len(plants)
    for c in calls:
        qids = [it["qid"] for it in c]
        assert len(qids) == len(set(qids))                   # never a plant beside its base (nor two plants of one base)
    assert calls == J.pack_calls(J.call_items(pairs, plants), per_call=10, seed=7)


# --- gate --------------------------------------------------------------------------------------------------------

def _plant_rows(n, caught):
    return [{"who": f"p:{k}:{i}", "kind": k, "same": i >= caught} for k in J.PLANT_KINDS for i in range(n)]


def test_gate_fails_below_0_90_and_below_30_plants():
    g = J.gate(_plant_rows(40, 34))                          # 0.85
    assert not g["ok"] and any("sensitivity" in r for r in g["reasons"])
    assert J.gate(_plant_rows(40, 36))["ok"]                  # 0.90
    g = J.gate(_plant_rows(29, 29))
    assert not g["ok"] and any("plants" in r for r in g["reasons"])


# --- budget ------------------------------------------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        inp = json.loads(kw["messages"][1]["content"])
        ans = {"verdicts": [{"key": p["key"], "same": True, "why": "ok"} for p in inp["pairs"]]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(ans)))])


def _llm(tmp_path, client, total=3):
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": total, "cache_dir": str(tmp_path),
                   "passes": {"ds_hard": {"prompt_version": 1, "max_calls": 900}}}}
    return KgLLM(cfg=cfg, cache_root=tmp_path, client=client, allow_api=True)


def test_budget_cap_and_cache_reuse(tmp_path):
    pairs = _pairs(40)
    calls = J.pack_calls(J.call_items(pairs, []), per_call=10, seed=1)
    assert len(calls) == 4
    c1 = FakeClient()
    with pytest.raises(KgBudgetExceeded):
        J.judge_calls(calls, _llm(tmp_path, c1))
    assert c1.calls == 3
    assert sum(v for p in json.loads((tmp_path / "ledger.json").read_text()).values() for v in p.values()) == 3
    c2 = FakeClient()
    llm2 = _llm(tmp_path, c2, total=900)
    out = J.judge_calls(calls[:3], llm2)                      # the three saved answers: cache only
    assert c2.calls == 0 and llm2.stats["ds_hard"]["hits"] == 3 and len(out) == 30
    with pytest.raises(KgBudgetExceeded):                     # the cumulative ledger (3) + 1 > 3
        J.judge_calls(calls, _llm(tmp_path, FakeClient()))


def test_judge_paraphrase_maps_verdicts(tmp_path):
    pairs = _pairs(40)
    res = J.judge_paraphrase(pairs, _llm(tmp_path, FakeClient(), total=900), plant_share=0.15, seed=3,
                             titles=TITLES, min_n=5)
    assert set(res["verdicts"]) == {p["qid"] for p in pairs}
    assert all(v["same"] is True for v in res["verdicts"].values())
    assert res["plants"] and all(p["same"] is True for p in res["plants"])      # the fake judge misses every plant
    assert not J.gate(res["plants"], min_n=5)["ok"]


# --- the para-judge command: gate, retry round, final output ---------------------------------------------------------

class TruthClient(FakeClient):
    """same = True iff the paraphrase is one of the real ones and its qid is not in `bad`."""
    def __init__(self, real: dict, bad=()):
        super().__init__()
        self.real, self.bad = real, set(bad)

    def create(self, **kw):
        self.calls += 1
        inp = json.loads(kw["messages"][1]["content"])
        ans = {"verdicts": [{"key": p["key"], "same": p["paraphrase"] in self.real and self.real[p["paraphrase"]]
                             not in self.bad, "why": "x"} for p in inp["pairs"]]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(ans)))])


def _jl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_para_judge_command_retry_round_and_final(tmp_path, monkeypatch):
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(VERSION, qid=f"v{i:03d}") for i in range(42)]
    _jl(tmp_path / "items.jsonl", items)
    texts = {it["qid"]: f"Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 апреля 2025 года? "
                        f"{'абвгдежзиклмнопрст'[i % 18] * (1 + i // 18)}" for i, it in enumerate(items)}
    ok = [{"qid": q, "round": 0, "question": t} for q, t in texts.items() if q != "v041"]
    rej = [{"qid": "v041", "round": 0, "reasons": ["anchor_date"], "raw": {"qid": "v041", "question": "x"}}]
    _jl(tmp_path / "para_ok.jsonl", ok)
    _jl(tmp_path / "para_rejections.jsonl", rej)
    (tmp_path / "para_ingest.json").write_text(json.dumps({"round1_files": []}))
    (tmp_path / "packets").mkdir()
    client = TruthClient({t: q for q, t in texts.items()}, bad={"v040"})
    monkeypatch.setattr(hp, "hard_llm", lambda: _llm(tmp_path / "cache", client, total=900))
    monkeypatch.setattr(hp, "_graph_titles", lambda: TITLES)
    args = ["para-judge", "--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path), "--packets",
            str(tmp_path / "packets"), "--retry-instructions", str(tmp_path / "retry.md")]
    assert hp.main(args) == 3                                              # two items need a second round
    retry = json.loads((tmp_path / "packets" / "retry1-001.json").read_text())
    assert sorted(x["qid"] for x in retry["items"]) == ["v040", "v041"]
    prev = {x["qid"]: x["previous"]["problem"] for x in retry["items"]}
    assert prev["v040"].startswith("judge") and prev["v041"] == "code check: anchor_date"
    assert json.loads((tmp_path / "para_gate.json").read_text())["gate"]["ok"]
    n_calls = client.calls
    t40 = texts["v040"].replace("Какие", "Какого рода")
    client.real[t40] = "v040x"                                             # the judge accepts the new wording
    _jl(tmp_path / "para_ok.jsonl", ok + [{"qid": "v040", "round": 1, "question": t40}])
    _jl(tmp_path / "para_rejections.jsonl", rej + [{"qid": "v041", "round": 1, "reasons": ["leak"], "raw": {}}])
    (tmp_path / "para_ingest.json").write_text(json.dumps({"round1_files": ["para_raw_r1_1.jsonl"]}))
    assert hp.main(args) == 0
    assert client.calls == n_calls + 1                                     # round 0 replayed from the cache
    rows = [json.loads(x) for x in (tmp_path / "para_judged.jsonl").read_text().splitlines()]
    assert len(rows) == 41 and "v041" not in {r["qid"] for r in rows}
    r40 = next(r for r in rows if r["qid"] == "v040")
    assert r40["question"] == t40 and r40["meta"]["template_question"] == VERSION["question"]
    assert r40["meta"]["para"]["round"] == 1
    g = json.loads((tmp_path / "para_gate.json").read_text())
    assert g["final"] and g["retry"]["dropped"] == {"v041": "code check: leak"}


def test_para_judge_gate_failure_is_a_stop(tmp_path, monkeypatch):
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(VERSION, qid=f"v{i:03d}") for i in range(40)]
    _jl(tmp_path / "items.jsonl", items)
    _jl(tmp_path / "para_ok.jsonl", [{"qid": it["qid"], "round": 0, "question": "Какие дополнения внёс в «Soliq "
                                      f"kodeksi» закон, принятый 15 апреля 2025 года? {i}я"} for i, it in enumerate(items)])
    _jl(tmp_path / "para_rejections.jsonl", [])
    (tmp_path / "para_ingest.json").write_text("{}")
    monkeypatch.setattr(hp, "hard_llm", lambda: _llm(tmp_path / "cache", FakeClient(), total=900))   # misses all
    monkeypatch.setattr(hp, "_graph_titles", lambda: TITLES)
    assert hp.main(["para-judge", "--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path),
                    "--packets", str(tmp_path / "packets")]) == 2
    g = json.loads((tmp_path / "para_gate.json").read_text())
    assert not g["gate"]["ok"] and not g["final"] and not (tmp_path / "para_judged.jsonl").exists()


# --- fix round (review I1 + minors) ---------------------------------------------------------------------------------

def test_new_number_unchanged_op_changed_and_day_forms():
    rej = _ingest("Сколько изменений, согласно 3 разделам, внёс в «Soliq kodeksi» закон от 15 апреля 2025 года "
                  "(дополнения)?", VERSION)[1]
    assert rej[0]["reasons"] == ["new_number"]
    rej = _ingest(VERSION["question"], VERSION)[1]
    assert rej[0]["reasons"] == ["unchanged"]
    rej = _ingest("Какие дополнения и исключения внёс в «Soliq kodeksi» закон от 15 апреля 2025 года?", VERSION)[1]
    assert rej[0]["reasons"] == ["op_changed"]                 # a broadened scope: another kind of change
    acc, rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон от 15-го апреля 2025 года?", VERSION)
    assert rej == [] and len(acc) == 1                         # «15-го апреля»
    rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон от 16 апреля 2025 года?", VERSION)[1]
    assert rej[0]["reasons"] == ["anchor_date"]                # not also answer_in_question / new_number
    assert P.scan("2023-yilda raqamli iqtisodiyot")[0] == set()

def _ft(s):
    from graphrag.dataset.facts import ctoks
    return [t.s for t in ctoks(s)]


def test_stems_need_context_and_whole_words():
    for s in ("рақами ўзгартирилган", "raqamini oʻzgartirish", "рақамлари ўзгартирилди", "qayta raqamlangan",
              "перенумерованы"):
        assert P.has_op(_ft(s), "renumber"), s
    for s in ("qonun raqami", "raqamli iqtisodiyot", "ЗРУ рақами билан"):
        assert not P.has_op(_ft(s), "renumber"), s
    assert P.has_op(_ft("дополнения"), "add") and P.has_op(_ft("дополнен"), "add")
    assert not P.has_op(_ft("дополнительно"), "add") and not P.has_op(_ft("исключительно"), "delete")
    assert P.has_op(_ft("исключены"), "delete")
    acc, rej = _ingest("Какие дополнения внёс в «Soliq kodeksi» закон от 15 апреля 2025 года, если говорить "
                       "исключительно о нём?", VERSION)
    assert rej == [] and len(acc) == 1                        # «исключительно» is not an op_changed


def test_day_suffix_year_window():
    assert P.scan("2024-yil 17-chi iyun kuni")[0] == {"2024-06-17"}
    assert P.scan("2024-yilning 17-chi iyunida")[0] == {"2024-06-17"}


def test_r51_count_times_refused_from_round_2():
    q = "2023-yilda «Davlat boji toʼgʼrisida»ning moddalari, qismlari va bandlariga necha marta oʻzgartish kiritilgan?"
    assert P.asks_times(q) and P.asks_times("Сколько раз в 2023 году меняли «X»?") and \
        P.asks_times("2023 йилда «X»га неча марта ўзгартиш киритилган?")
    assert not P.asks_times("Сколько изменений внесено 3 марта 2023 года?")
    raw = [{"qid": COUNT["qid"], "question": q, "round": 2}]
    acc, rej = P.para_ingest(raw, ITEMS, gold=GOLD, retry_qids={2: {COUNT["qid"]}})
    assert acc == [] and rej[0]["reasons"] == ["count_times"]
    acc, rej = P.para_ingest([dict(raw[0], round=0)], ITEMS, gold=GOLD)      # rounds 0 / 1: frozen judge plan
    assert rej == [] and len(acc) == 1
    rej = P.para_ingest([dict(raw[0], round=2)], ITEMS, gold=GOLD, retry_qids={2: set()})[1]
    assert rej[0]["reasons"] == ["not_retry"]


def test_act_swap_titles_and_number_rotation():
    assert not J._act_like("Oʻzbekiston Respublikasining ayrim qonun hujjatlariga oʻzgartish kiritish toʼgʼrisida")
    assert not J._act_like("О внесении изменений в Кодекс")
    assert not J._act_like("“Soliq kodeksi”") and J._act_like("Soliq kodeksi")
    it = dict(COUNT, template="which_law_part", question="«Soliq kodeksi»нинг 22-моддаси 5-қисмига 2023 йилда қайси "
              "қонун ўзгартиш киритган?", lang="uz-Cyrl", meta={"article": "22", "part": "5", "year": "2023"})
    first = set()
    for i in range(30):
        pl = J.make_plant("number_swap", dict(J.make_pair(it, it["question"], P.anchors(it)), qid=f"q{i}"), 5)
        first.add("year" if "2023" not in pl["paraphrase"] else "other")
    assert first == {"year", "other"}                           # the swapped number rotates by seed


def test_judge_offline_cache_miss_names_the_gate(tmp_path, monkeypatch):
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(VERSION, qid="v000")]
    _jl(tmp_path / "items.jsonl", items)
    _jl(tmp_path / "para_ok.jsonl", [{"qid": "v000", "round": 0, "question": "Какие дополнения внёс в «Soliq "
                                      "kodeksi» закон, принятый 15 апреля 2025 года?"}])
    _jl(tmp_path / "para_rejections.jsonl", [])
    (tmp_path / "para_ingest.json").write_text("{}")
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 900,
                   "passes": {"ds_hard": {"prompt_version": 1, "max_calls": 900}}}}
    monkeypatch.setattr(hp, "hard_llm", lambda: KgLLM(cfg=cfg, cache_root=tmp_path / "c", client=FakeClient(),
                                                       allow_api=False))
    monkeypatch.setattr(hp, "_graph_titles", lambda: TITLES)
    with pytest.raises(SystemExit, match="GRL_HARD_API=1"):
        hp.main(["para-judge", "--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path),
                 "--packets", str(tmp_path / "packets")])


def test_duplicate_line_of_an_accepted_item_is_not_retried(tmp_path, monkeypatch):
    """I1: a duplicate (or not_retry) rejection of a qid whose first round-0 line was accepted never sends it to a
    retry; the run is final with the accepted paraphrase."""
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(VERSION, qid=f"v{i:03d}") for i in range(40)]
    _jl(tmp_path / "items.jsonl", items)
    texts = {it["qid"]: f"Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 апреля 2025 года? "
                        f"{'абвгдежзиклмнопрст'[i % 18] * (1 + i // 18)}" for i, it in enumerate(items)}
    _jl(tmp_path / "para_ok.jsonl", [{"qid": q, "round": 0, "question": t} for q, t in texts.items()])
    _jl(tmp_path / "para_rejections.jsonl", [{"qid": "v003", "round": 0, "reasons": ["duplicate"],
                                              "raw": {"qid": "v003", "question": "x"}},
                                             {"qid": "v004", "round": 1, "reasons": ["not_retry"], "raw": {}}])
    (tmp_path / "para_ingest.json").write_text("{}")
    client = TruthClient({t: q for q, t in texts.items()})
    monkeypatch.setattr(hp, "hard_llm", lambda: _llm(tmp_path / "cache", client, total=900))
    monkeypatch.setattr(hp, "_graph_titles", lambda: TITLES)
    assert hp.main(["para-judge", "--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path),
                    "--packets", str(tmp_path / "packets")]) == 0
    assert not list((tmp_path / "packets").glob("retry1-*.json")) if (tmp_path / "packets").exists() else True
    rows = {json.loads(x)["qid"]: json.loads(x) for x in (tmp_path / "para_judged.jsonl").read_text().splitlines()}
    assert len(rows) == 40 and rows["v003"]["question"] == texts["v003"]


# --- fix round 2: R51 forced round 2, artifacts.sha256 in place ------------------------------------------------------

def test_r51_forced_round_2_command_flow(tmp_path, monkeypatch):
    """A final para_judged row asking «necha marta» -> para-force-retry writes retry2 packets; para-judge refuses to
    finish while that row is the accepted one, then takes the round-2 paraphrase (or drops the item)."""
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(VERSION, qid=f"v{i:03d}") for i in range(40)] + [dict(COUNT, qid="c000"), dict(COUNT, qid="c001")]
    _jl(tmp_path / "items.jsonl", items)
    texts = {it["qid"]: f"Какие дополнения внёс в «Soliq kodeksi» закон, принятый 15 апреля 2025 года? "
                        f"{'абвгдежзиклмнопрст'[i % 18] * (1 + i // 18)}" for i, it in enumerate(items[:40])}
    texts["c000"] = "2023-yilda «Davlat boji toʼgʼrisida»ga necha marta oʻzgartish kiritilgan?"
    texts["c001"] = "2023-yil davomida «Davlat boji toʼgʼrisida»ning necha marta oʻzgartirilgan?"
    ok = [{"qid": q, "round": 0, "question": t} for q, t in texts.items()]
    _jl(tmp_path / "para_ok.jsonl", ok)
    _jl(tmp_path / "para_rejections.jsonl", [])
    (tmp_path / "para_ingest.json").write_text("{}")
    client = TruthClient({t: q for q, t in texts.items()})
    monkeypatch.setattr(hp, "hard_llm", lambda: _llm(tmp_path / "cache", client, total=900))
    monkeypatch.setattr(hp, "_graph_titles", lambda: TITLES)
    base = ["--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path), "--packets", str(tmp_path / "packets")]
    with pytest.raises(SystemExit, match="R51"):
        hp.main(["para-judge"] + base)                          # never final with a «how many times» row
    # the pre-R51 final file (as written by the first final run)
    _jl(tmp_path / "para_judged.jsonl", [dict(it, question=texts[it["qid"]]) for it in items])
    (tmp_path / "packets").mkdir()
    assert hp.main(["para-force-retry"] + base + ["--retry-instructions", str(tmp_path / "r2.md")]) == 0
    r2 = json.loads((tmp_path / "packets" / "retry2-001.json").read_text())
    assert sorted(x["qid"] for x in r2["items"]) == ["c000", "c001"]
    assert all(x["previous"]["problem"] == hp.R51_PROBLEM for x in r2["items"])
    assert hp.main(["para-judge"] + base) == 3                  # retry2 pending
    good = "2023-yilda «Davlat boji toʼgʼrisida»ning moddalari, qismlari va bandlariga jami qancha oʻzgartish kiritilgan?"
    client.real[good] = "c000"
    _jl(tmp_path / "para_ok.jsonl", ok + [{"qid": "c000", "round": 2, "question": good}])
    _jl(tmp_path / "para_rejections.jsonl", [{"qid": "c001", "round": 2, "reasons": ["count_times"], "raw": {}}])
    (tmp_path / "para_ingest.json").write_text(json.dumps({"round2_files": ["para_raw_r2_1.jsonl"]}))
    assert hp.main(["para-judge"] + base) == 0
    rows = {json.loads(x)["qid"]: json.loads(x) for x in (tmp_path / "para_judged.jsonl").read_text().splitlines()}
    assert rows["c000"]["question"] == good and rows["c000"]["meta"]["para"]["round"] == 2 and "c001" not in rows
    g = json.loads((tmp_path / "para_gate.json").read_text())
    assert g["retry2"]["dropped"] == {"c001": "code check: count_times"} and g["accepted"]["round2"] == 1


def test_update_hashes_replaces_in_place_and_dedupes(tmp_path):
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    f = tmp_path / "sums"
    f.write_text("00  a.txt\n11  b.txt\n22  a.txt\n")
    rep = hp.update_hashes(f, [], root=tmp_path)
    lines = f.read_text().splitlines()
    assert [l.split()[1] for l in lines] == ["a.txt", "b.txt"] and rep["deduped"] == ["a.txt"]
    assert lines[1] == "11  b.txt" and lines[0].split()[0] != "00"
    (tmp_path / "b.txt").write_text("B2")
    (tmp_path / "c.txt").write_text("C")
    hp.update_hashes(f, ["b.txt", "c.txt"], root=tmp_path)
    lines = f.read_text().splitlines()
    assert [l.split()[1] for l in lines] == ["a.txt", "b.txt", "c.txt"] and lines[1].split()[0] != "11"


# --- fix round 3: «неча бор», retry3, renumber verb of the number -----------------------------------------------------

def test_necha_bor_is_times_but_nechta_qancha_bor_are_not():
    assert P.asks_times("«Jinoyat kodeksi»нинг элементларига 2023 йилда жами неча бор ўзгартиш киритилган?")
    assert P.asks_times("2023-yilda «X»ga necha bor oʻzgartish kiritilgan?")
    for q in ("«X» қонуни таркибида нечта модда бор?", "«X» qonunida jami qancha modda bor?",
              "Jami qancha modda bor «X» qonunida?"):
        assert not P.asks_times(q), q                          # «how many are there» (3 valid count rows)
    q = "«Jinoyat kodeksi»нинг элементларига 2023 йилда неча бор ўзгартиш киритилган?"
    assert not P.asks_times(q, 2) and P.asks_times(q, 3)       # round 2 keeps its frozen plan; checked from round 3


def test_renumber_verb_belongs_to_the_number():
    assert not P.has_op(_ft("ЗРУ рақами билан ўзгартирилган"), "renumber")
    assert not P.has_op(_ft("qonun raqami bilan oʻzgartirilgan"), "renumber")
    assert P.has_op(_ft("элементларнинг рақами ўзгартирилган"), "renumber")
    assert P.has_op(_ft("qayta raqamlangan"), "renumber")


def test_force_retry_round_3_after_round_2(tmp_path, monkeypatch):
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    items = [dict(COUNT, qid="c000"), dict(COUNT, qid="c001", lang="uz-Cyrl")]
    _jl(tmp_path / "items.jsonl", items)
    (tmp_path / "packets").mkdir()
    (tmp_path / "packets" / "retry2-001.json").write_text(json.dumps({"items": [{"qid": "c000"}]}))
    _jl(tmp_path / "para_judged.jsonl", [
        dict(items[0], question="2023-yilda «Davlat boji toʼgʼrisida»ning bandlariga jami qancha oʻzgartish kiritilgan?"),
        dict(items[1], question="«Davlat boji toʼgʼrisida»нинг элементларига 2023 йилда неча бор ўзгартиш киритилган?")])
    base = ["--in", str(tmp_path / "items.jsonl"), "--out", str(tmp_path), "--packets", str(tmp_path / "packets")]
    assert hp.main(["para-force-retry"] + base + ["--retry-instructions", str(tmp_path / "r3.md")]) == 0
    r3 = json.loads((tmp_path / "packets" / "retry3-001.json").read_text())
    assert [x["qid"] for x in r3["items"]] == ["c001"] and r3["items"][0]["previous"]["problem"] == hp.R51_PROBLEM
    assert "para_raw_r3_" in (tmp_path / "r3.md").read_text()
    with pytest.raises(SystemExit, match="retry3"):
        hp.main(["para-force-retry"] + base + ["--round", "3"])          # never overwrites
