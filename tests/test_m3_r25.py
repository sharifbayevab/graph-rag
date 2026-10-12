"""M3-R25 (controller ruling 2026-09-29): verify fixes A (script-insensitive quotes), B (cap floor 200, typed values
exempt), C (plants in their own calls on accepted bases only, needed slot, replacement validity, contested misses);
ingest value cap; duplicate content across documents; generator instructions v2."""
import json

from graphrag import config
from graphrag.dataset import opus
from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import Candidate
from graphrag.dataset.seeds import Seed, content_signature, cross_split_twin, dup_content
from graphrag.dataset.verify import JUDGE_KINDS, judge_ok, neighbor_picker, quote_ok, quote_share, verify_candidates
from graphrag.kg.llm import KgLLM
from tests.test_m3_dataset_verify import CAP, VCFG, _Judge, _frag_of, _world
from tests.test_m3_r23_opus import _ingest, _row, _single_packet

LAT = "Qonun rasmiy e'lon qilingan kundan e'tiboran ikki oy muddatda kuchga kiradi va barcha idoralar ijro etadi."


def _c(value, vtype="", chunk="D/СТ.1#1"):
    s = Seed("single-x", "single", "ru", (chunk,), ("D",), "train", "QONUNLAR")
    return Candidate(s, "Когда?", "a", [Fact(f"{chunk}@0", chunk, "", value, "q", vtype=vtype)], {})


def _v(q):
    return {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": q}]}}


# ---- A ----------------------------------------------------------------------------------------------------------------
def test_a_script_rewritten_quote_counts_but_a_joined_one_does_not():
    cyr = "икки ой муддатда кучга киради"                     # the judge re-scripted a Latin span
    assert quote_ok(cyr, LAT.lower(), CAP) and quote_ok("ikki oy muddatda кучга киради", LAT.lower(), CAP)
    assert not quote_ok("qonun rasmiy … kuchga kiradi", LAT.lower(), CAP)             # "…" join: not a span
    assert judge_ok(_c("ikki oy muddatda"), _v(cyr), {"F1": LAT})[0]


# ---- B ----------------------------------------------------------------------------------------------------------------
def test_b_floor_200_and_typed_values_exempt_from_the_cap():
    txt = ("Вазирлик икки ой муддатда соғлиқни сақлаш муассасаларида санитария ҳолатини ўрганиб чиқсин ва натижалари "
           "бўйича таклифлар киритсин. ") * 3
    cap = (300, 0.5, 200)
    short = txt[:330]
    q = short[:190]                                                                   # 58 % of a 330-char fragment
    assert quote_ok(q, short, cap) and not quote_ok(q, short, (300, 0.5, 80))
    assert quote_ok(q[:150], txt[:200], cap)                                          # 75 % of a short fragment
    whole = txt.strip()
    assert not quote_ok(whole, whole, cap)                                            # > 300 chars
    assert judge_ok(_c("икки ой муддатда"), _v(whole), {"F1": whole}, quote_cap=cap)[0]           # typed: exempt
    clause = "соғлиқни сақлаш муассасаларида санитария ҳолатини ўрганиб чиқсин ва натижалари бўйича таклифлар киритсин"
    assert not judge_ok(_c(clause, "clause"), _v(whole), {"F1": whole}, quote_cap=cap)[0]       # clause: capped


def test_config_quote_cap_floor_is_200():
    assert config.load("train")["dataset"]["verify_quote_cap"] == [300, 0.5, 200]


# ---- C ----------------------------------------------------------------------------------------------------------------
def test_c_replacement_sharing_the_removed_quote_is_no_plant_and_a_second_option_exists():
    Ch = lambda t: type("Ch", (), {"tokens": 50, "text": t})()
    order = {"D": ["D/1", "D/2", "D/3"]}
    q = "ёнғин хавфсизлиги талабларини бузган шахсларга жарима солинади"
    ch = {"D/1": Ch("Ёнғин хавфсизлиги талабларини бузган ташкилотлар текширилади ва жарима солинади"),
          "D/2": Ch(q + " беш баравар миқдорда."), "D/3": Ch("Қўмита ҳисоботини ҳар йили Сенатга тақдим этади.")}
    c = Candidate(Seed("single-y", "single", "ru", ("D/2",), ("D",), "train", "QONUNLAR"), "?", "a",
                  [Fact("D/2@0", "D/2", "", "беш баравар", q)], {})
    assert quote_share([q], ch["D/1"].text) > 0.3 and quote_share([q], ch["D/3"].text) < 0.3
    pick = neighbor_picker(order, ch, 7, 30)
    assert pick(c, 0) == "D/3" and pick(c, 0, 1) is None


def _run(tmp_path, judge, target=12, n=60):
    text, cs, need = _world(n)
    llm = KgLLM(VCFG, cache_root=tmp_path, client=judge.client, allow_api=True)
    kept, st = verify_candidates(cs, _frag_of(text), llm, 2, 0.15, 7, lambda c: "U#1", min_kind_n=5,
                                 views_per_call=5, neighbor=lambda c, i: f"N:{c.seed.chunks[i]}",
                                 pilot_ids={c.seed.seed_id for c in cs}, quote_cap=CAP, pilot_target=target)
    return kept, st, llm, cs, need


def test_c_plants_have_their_own_calls_on_accepted_bases_and_never_move_candidate_calls(tmp_path):
    text, cs, need = _world(60)
    j = _Judge(need)
    kept, st, llm, cs, _ = _run(tmp_path, j)
    assert 0 < st["calls_plants"] < st["calls"]
    n_cand_calls = st["calls"] - st["calls_plants"]
    j2 = _Judge(need)
    kept2, st2, llm2, _, _ = _run(tmp_path, j2, target=8)          # another plant plan: candidate calls all cached
    assert llm2.stats["ds_verify"]["hits"] >= n_cand_calls and [c.seed.seed_id for c in kept2] == \
        [c.seed.seed_id for c in kept]
    assert all(st["pilot"]["planted"][k] >= 5 for k in JUDGE_KINDS)


def test_c_no_plant_on_a_rejected_base_and_contested_misses(tmp_path):
    text, cs, need = _world(60)
    bad = {c.question for c in cs[:30]}
    need2 = {q: v for q, v in need.items() if q not in bad}          # the judge rejects the first 30 pairs

    class Nb(_Judge):                                                # answers neighbor_swap views from the pad
        def answer(self, inp):
            out = super().answer(inp)
            for v, o in zip(inp["views"], out["views"]):
                nb = [f for f in v["fragments"] if f["text"].startswith("Қоида")]
                if nb and not o["answerable"]:
                    o.update(answerable=True, answer="x", evidence=[{"id": nb[0]["id"], "quote": nb[0]["text"][:30]}])
            return out
    kept, st, *_ = _run(tmp_path, Nb(need2))
    ids = {c.seed.seed_id for c in cs[:30]}
    assert not ids & {c.seed.seed_id for c in kept}
    assert st["contested_by_kind"].get("neighbor_swap", 0) > 0 and "neighbor_swap" not in st["sensitivity_by_kind"]
    assert all(x["replacement"].startswith("N:") for x in st["contested"])


# ---- ingest cap, duplicates, instructions ------------------------------------------------------------------------------
def test_ingest_rejects_a_value_over_the_quote_cap():
    pk, chunks = _single_packet()
    q = "учинчи навбатда – Омонатларни кафолатлаш агентлигига ўтган талаблар"
    import graphrag.dataset.opus as o
    cfg = {"seed": 0, "max_copy_words": 99, "max_lexical_overlap": 1.0, "verify_quote_cap": [20, 0.5, 200]}
    _, rej, _ = o.ingest_generated(pk, _row(used_facts=[{"fragment": "F1", "quote": q, "value": q}]), chunks, cfg,
                                   {"B": "QONUNLAR"}, "g", "m")
    assert [r["reason"] for r in rej] == ["value_over_cap"]


def test_generated_under_older_instructions_is_recorded():
    pk, chunks = _single_packet()
    old = {"version": 1, "digest": "f" * 64}
    cands, _, _ = opus.ingest_generated(pk, _row(), chunks, {"seed": 0, "max_copy_words": 99, "max_lexical_overlap": 1.0},
                                        {"B": "QONUNLAR"}, "g", "m", old)
    p = cands[0].provenance
    assert p["opus"]["instructions"]["digest"] == "f" * 64 and p["generate"]["prompt_version"] == 1
    assert p["opus"]["packet_instructions"]["digest"] == pk["instructions"]["digest"]


def _recs():
    C = lambda cid, t: type("C", (), {"chunk_id": cid, "text": t})()
    R = lambda d, cs: type("R", (), {"doc_id": d, "chunks": cs})()
    same = "4. Соғлиқни сақлаш вазирлиги икки ой муддатда шифохоналар фаолиятини ўрганиб чиқсин."
    return [R("A", [C("A/П.4#1", same), C("A/П.5#1", "5. Назорат қўмитага юклансин ва ҳисобот берилсин.")]),
            R("B", [C("B/П.4#1", same), C("B/П.6#1", "Бошқа матн бўлиб, у фақат шу ҳужжатда учрайди, такрорланмайди.")]),
            R("T", [C("T/П.1#1", same)])]


def test_duplicate_content_one_seed_per_text_and_no_cross_split_twin():
    dup = dup_content(_recs(), 40)
    assert set(dup) == {"A/П.4#1", "B/П.4#1", "T/П.1#1"} and dup["A/П.4#1"]["twins"] == ["B/П.4#1", "T/П.1#1"]
    assert content_signature(["A/П.4#1"], dup) == content_signature(["B/П.4#1"], dup)
    split = {"A": "train", "B": "train", "T": "test"}
    assert cross_split_twin(["A/П.4#1"], dup, split) and not cross_split_twin(["A/П.5#1"], dup, split)
    mk = lambda sid, ch: Candidate(Seed(sid, "single", "ru", (ch,), (ch.split("/")[0],), "train", "QARORLAR"), "?", "a",
                                   [], {"opus": {"packet_id": "p", "line": 1}})
    dup2 = dup_content(_recs()[:2], 40)                              # without the test twin
    kept, rej = opus.dedupe_content([mk("single-2", "B/П.4#1"), mk("single-1", "A/П.4#1"), mk("single-3", "A/П.5#1")],
                                    dup2, split)
    assert [c.seed.seed_id for c in kept] == ["single-1", "single-3"] and [r["reason"] for r in rej] == ["duplicate_content"]
    assert kept[0].provenance["duplicate_twins"] == {"A/П.4#1": ["B/П.4#1"]}
    kept, rej = opus.dedupe_content([mk("single-1", "A/П.4#1")], dup, split)
    assert kept == [] and rej[0]["reason"] == "duplicate_cross_split"


def test_generator_instructions_v2_rules_and_history():
    o = config.load("train")["dataset"]["opus"]
    assert o["generator"]["version"] >= 3 and o["generator_history"][o["generator"]["version"]] == o["generator"]["digest"]
    assert o["generator_history"][1] == "f2130e2054bdbdbae10d9778923d7ebac09c4017e6f280994b730f30fd44c7ee"
    t = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    for s in ("never restate", "never name the neighbour's topic", "Mandatory multi self-test", "No definition padding",
              "at most 300 characters", "one fragment suffices"):
        assert s in t, s


def test_d_a_path_fact_stated_by_the_question_need_not_be_quoted():
    s = Seed("multi-z", "multi", "uz-Cyrl", ("D/СТ.24#1", "D/СТ.64#1"), ("D",), "train", "QONUNLAR")
    fa = Fact("D/СТ.24#1@0", "D/СТ.24#1", "", "64-моддасида", "ушбу Қонуннинг 64-моддасида белгиланган")
    fb = Fact("D/СТ.64#1@0", "D/СТ.64#1", "", "учинчи навбатда", "учинчи навбатда – агентлик талаблари")
    t = {"F1": "Мажбуриятлар ушбу Қонуннинг 64-моддасида белгиланган навбатда ўтказилади.",
         "F2": "Талаблар қуйидагича: учинчи навбатда – агентлик талаблари қаноатлантирилади."}
    v = {"full": {"answerable": True, "evidence": [{"id": "F2", "quote": "учинчи навбатда – агентлик талаблари"}]},
         "drop:F1": {"answerable": False}, "drop:F2": {"answerable": False}}
    mk = lambda q, prov: Candidate(s, q, "Учинчи навбатда.", [fa, fb], prov)
    q = "Қонуннинг 24-моддаси ҳавола қилган 64-моддасида агентлик талаблари нечанчи навбатда қондирилади?"
    assert judge_ok(mk(q, {"path_fact_ids": [fa.fact_id]}), v, t)[0]              # path fact: exempt
    assert judge_ok(mk(q, {}), v, t)[0]                                            # stated by the question: exempt
    ok, rep = judge_ok(mk("Агентлик талаблари нечанчи навбатда қондирилади?", {}), v, t)
    assert not ok and rep["facts_recovered"] == [False, True]                      # an answer fact: must be quoted
    v2 = dict(v, **{"drop:F1": {"answerable": True}})
    assert not judge_ok(mk(q, {"path_fact_ids": [fa.fact_id]}), v2, t)[0]          # the ablation still decides
