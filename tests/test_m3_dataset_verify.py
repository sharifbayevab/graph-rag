from graphrag.dataset.facts import Fact
from graphrag.dataset.generate import Candidate
from graphrag.dataset.seeds import Seed
from graphrag.dataset.verify import PLANT_KINDS, judge_ok, plan_batches, views

T1 = "Ушбу Қонуннинг 5-моддасида назарда тутилган ваколатлар Марказий банк томонидан амалга оширилади."
T2 = "5-модда. Марказий банк банкларни санация қилиш тўғрисида қарор ўттиз кун ичида қабул қилади."


def _cand(i=1, doc="D", multi=True):
    ch = (f"{doc}/СТ.3#{i}", f"{doc}/СТ.5#{i}") if multi else (f"{doc}/СТ.5#{i}",)
    s = Seed(f"{'refers_internal' if multi else 'single'}-{i:010d}", "refers_internal" if multi else "single", "ru",
             ch, (doc,), "train", "QONUNLAR", (("D/СТ.3", "REFERS_TO", "D/СТ.5"),) if multi else ())
    fs = [Fact(f"{ch[0]}@0", ch[0], "", "Марказий банк", "ваколатлар Марказий банк томонидан амалга")]
    if multi:
        fs.append(Fact(f"{ch[1]}@0", ch[1], "", "ўттиз кун", "санация қилиш тўғрисида қарор ўттиз кун ичида"))
    return Candidate(s, f"Кто и в какой срок решает об оздоровлении банка {i}?", "Марказий банк; ўттиз кун.", fs, {})


def test_views_full_plus_one_ablation_per_fact_fragment():
    assert views(_cand()) == [("full", ["F1", "F2"]), ("drop:F1", ["F2"]), ("drop:F2", ["F1"])]
    assert views(_cand(multi=False)) == [("full", ["F1"])]


def test_every_fact_must_be_recovered_and_every_ablation_must_fail():
    c = _cand()
    texts = {"F1": T1, "F2": T2}
    full = {"answerable": True, "answer": "Марказий банк, ўттиз кун ичида", "evidence": [
        {"id": "F1", "quote": "ваколатлар Марказий банк томонидан"}, {"id": "F2", "quote": "қарор ўттиз кун ичида"}]}
    no = {"answerable": False, "answer": "", "evidence": []}
    ok, rep = judge_ok(c, {"full": full, "drop:F1": no, "drop:F2": no}, texts)
    assert ok and rep["facts_recovered"] == [True, True] and rep["ablation"] == {"drop:F1": True, "drop:F2": True}
    ok, rep = judge_ok(c, {"full": full, "drop:F1": no, "drop:F2": dict(full, evidence=full["evidence"][:1])}, texts)
    assert not ok and rep["ablation"]["drop:F2"] is False                       # still answerable without F2
    fake = dict(full, evidence=[full["evidence"][0], {"id": "F2", "quote": "ўттиз кун ичида ҳал этилади албатта"}])
    ok, rep = judge_ok(c, {"full": fake, "drop:F1": no, "drop:F2": no}, texts)
    assert not ok and rep["facts_recovered"] == [True, False]                  # non-verbatim quote does not count


def test_plants_cover_the_three_kinds_deterministically():
    cands = [_cand(i, doc=f"D{i % 4}") for i in range(60)]
    b1 = plan_batches(cands, 2, 0.6, 7, lambda c: "OTHER#1")
    assert b1 == plan_batches(cands, 2, 0.6, 7, lambda c: "OTHER#1")
    kinds = {p["kind"] for b in b1 for p in b["plants"]}
    assert kinds <= set(PLANT_KINDS) and len(kinds) >= 2
    assert sum(len(b["cands"]) for b in b1) == 60
    for b in b1:
        for p in b["plants"]:
            if p["kind"] == "chunk_swap":
                assert "OTHER#1" in p["chunks"]


# --- added over the brief (Task 16 hard requirements; fix round M3-R17) --------------------------------------------
import json  # noqa: E402

import pytest  # noqa: E402

from graphrag.dataset.verify import (CODE_KINDS, JUDGE_KINDS, PASS, PROMPT_VER, SCHEMA_VER, neighbor_picker,  # noqa: E402
                                     pack_calls, quote_ok, sensitivity_gate, unrelated_picker, verify_candidates)
from graphrag.kg.llm import KgBudgetExceeded, KgCacheMiss, KgLLM, _canon_input, prompt_digest  # noqa: E402

VCFG = {"llm": {"model": "gpt-4.1", "max_calls_total": 2500,
                "passes": {"ds_verify": {"prompt_version": 3, "max_calls": 1000}}}}
UNREL = "Сенат мажлисида кун тартиби тасдиқланди ва қўмиталар ҳисоботлари эшитилди, овоз бериш ўтказилди."
CAP = (300, 0.5, 80)


def _world(n=120):
    """n candidates over 4 documents, every third single-hop; fragment text carries each fact value inside its quote;
    N:<chunk> is an adjacent fragment of the same document without the value."""
    text, cands, need = {"U#1": UNREL}, [], {}
    for i in range(n):
        d = f"D{i % 4}"
        a, b = f"{d}/СТ.3#{i}", f"{d}/СТ.5#{i}"
        text[a] = f"Ушбу {i}-сон қоидада назарда тутилган ваколатлар Марказий банк{i} томонидан амалга оширилади ва назорат қилинади."
        text[b] = f"{i}-модда. Банкларни санация қилиш тўғрисида қарор {i + 10} кун ичида қабул қилинади ва эълон қилинади."
        for x in (a, b):
            text[f"N:{x}"] = f"Қоида {i} бўйича ҳисобот ва маълумотлар тартиби белгиланади, ижро назорат қилинади."
        single = i % 3 == 0
        ch = (b,) if single else (a, b)
        pat = "single" if single else "refers_internal"
        s = Seed(f"{pat}-{i:010d}", pat, ["ru", "uz-Cyrl"][i % 2], ch, (d,), ["train", "test"][(i // 2) % 2],
                 "QONUNLAR", () if single else ((f"{d}/СТ.3", "REFERS_TO", f"{d}/СТ.5"),))
        fa = Fact(f"{a}@0", a, f"ваколат {i}", f"Марказий банк{i}", f"ваколатлар Марказий банк{i} томонидан")
        fb = Fact(f"{b}@0", b, f"муддат {i}", f"{i + 10} кун", f"қарор {i + 10} кун ичида")
        fs = [fb] if single else [fa, fb]
        q = f"Кто и в какой срок решает об оздоровлении банка по правилу {i}?"
        cands.append(Candidate(s, q, f"ОТВЕТ-ГЕНЕРАТОРА-{i}: Марказий банк{i}; {i + 10} кун.", fs, {"seed_id": s.seed_id}))
        need[q] = [(text[f.chunk_id], f.quote) for f in fs]
    return text, cands, need


def _frag_of(text):
    return lambda cid, label: {"id": label, "act": "Қонун", "where": "Қонун", "text": text[cid]}


class _Judge:
    """Fake gpt-4.1 judges.
    truthful: answerable iff every needed fragment is shown, quoting the exact spans;
    yes: always answerable, quotes every fragment whole;
    blind: never reads which fragments are shown -- answerable unless a view looks reduced by structure (an id gap or
      fewer fragments than another view of the same question in the call), quoting the known spans;
    size: answerable iff it sees >= 2 fragments (\"multi-hop needs two\"), quoting the known spans."""

    def __init__(self, need, mode="truthful"):
        self.need, self.mode, self.requests = need, mode, []
        judge = self

        class _C:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        judge.requests.append(kw["messages"])
                        body = json.dumps(judge.answer(json.loads(kw["messages"][1]["content"])))
                        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})
        self.client = _C()

    def answer(self, inp):
        out = []
        most = {}
        for v in inp["views"]:
            most[v["question"]] = max(most.get(v["question"], 0), len(v["fragments"]))
        for v in inp["views"]:
            shown = {f["text"]: f["id"] for f in v["fragments"]}
            req = self.need.get(v["question"], [])
            ids = sorted(int(f["id"][1:]) for f in v["fragments"])
            if self.mode == "yes":
                out.append({"key": v["key"], "answerable": True, "answer": "x",
                            "evidence": [{"id": f["id"], "quote": f["text"]} for f in v["fragments"]]})
                continue
            if self.mode == "truthful":
                ok = bool(req) and all(t in shown for t, _ in req)
            elif self.mode == "blind":
                ok = len(v["fragments"]) == most[v["question"]] and ids == list(range(1, len(ids) + 1))
            else:
                ok = len(v["fragments"]) >= 2
            ev = [{"id": shown.get(t, v["fragments"][0]["id"]), "quote": q} for t, q in req] if ok else []
            out.append({"key": v["key"], "answerable": ok, "answer": "x" if ok else "", "evidence": ev})
        return {"views": out}


def _run(tmp_path, mode="truthful", n=120, share=1.0, api=True, judge=None, pilot=(), cands=None):
    text, cs, need = _world(n)
    cs = cands if cands is not None else cs
    j = judge or _Judge(need, mode)
    llm = KgLLM(VCFG, cache_root=tmp_path, client=j.client if api else None, allow_api=api)
    kept, st = verify_candidates(cs, _frag_of(text), llm, 2, share, 7, lambda c: "U#1", min_kind_n=5,
                                 views_per_call=5, neighbor=lambda c, i: f"N:{c.seed.chunks[i]}", pilot_ids=pilot,
                                 quote_cap=CAP)
    return kept, st, j, llm, cs


def test_judge_never_sees_the_generators_answer_or_facts(tmp_path):
    kept, st, j, _, cands = _run(tmp_path)
    assert j.requests
    for msgs in j.requests:
        assert msgs[0]["content"] == PROMPT_VER and "same question" not in PROMPT_VER
        inp = json.loads(msgs[1]["content"])
        assert set(inp) == {"views"}
        for v in inp["views"]:
            assert set(v) == {"key", "question", "fragments"}
            assert all(set(f) == {"id", "act", "where", "text"} for f in v["fragments"])
            assert v["key"].startswith("V") and "|" not in v["key"]     # opaque: a plant is not told apart by its key
        raw = msgs[1]["content"]
        assert "ОТВЕТ-ГЕНЕРАТОРА" not in raw and "statement" not in raw and "facts" not in raw
        assert not any(f.statement in raw for c in cands for f in c.facts)


def test_calls_hold_one_view_per_question_contiguous_ids_and_no_plant_beside_its_base(tmp_path):
    kept, st, j, _, cands = _run(tmp_path)
    own = {c.question: c.seed.seed_id for c in cands}
    for msgs in j.requests:
        inp = json.loads(msgs[1]["content"])
        qs = [v["question"] for v in inp["views"]]
        assert len(qs) == len(set(qs)) and len(qs) <= 5                  # never two views of one question in a call
        for v in inp["views"]:
            assert [f["id"] for f in v["fragments"]] == [f"F{k + 1}" for k in range(len(v["fragments"]))]
    text, cs, _ = _world(120)
    from graphrag.dataset.verify import plan_batches
    groups = plan_batches(cs, 2, 1.0, 7, lambda c: "U#1", neighbor=lambda c, i: f"N:{c.seed.chunks[i]}")
    for call in pack_calls(groups, 5, 7):
        fams = [x for it in call for x in it["fam"]]
        assert len(fams) == len(set(fams))                               # a pair, its plants and donors never share a call
    assert st["calls"] <= st["views"] / 5 * 1.3 and own


def test_seeded_fragment_order_differs_from_walk_order(tmp_path):
    kept, st, j, _, _ = _run(tmp_path)
    firsts = {json.loads(m[1]["content"])["views"][k]["fragments"][0]["text"].split()[0]
              for m in j.requests for k in range(len(json.loads(m[1]["content"])["views"]))
              if len(json.loads(m[1]["content"])["views"][k]["fragments"]) == 2}
    assert len(firsts) > 1                                               # "Ушбу…" (F1 of the walk) is not always first


def test_truthful_judge_keeps_every_pair_and_catches_every_judge_kind(tmp_path):
    kept, st, _, llm, cands = _run(tmp_path)
    assert len(kept) == len(cands) and st["rejected"] == {}
    assert set(st["sensitivity_by_kind"]) == set(JUDGE_KINDS)
    assert all(v == 1.0 for v in st["sensitivity_by_kind"].values()) and sensitivity_gate(st, 0.9) == []
    assert set(st["code_sensitivity_by_kind"]) == set(CODE_KINDS)
    for c in kept:
        v = c.provenance["verify"]
        assert v["prompt_digest"] == prompt_digest(PROMPT_VER, SCHEMA_VER) and v["prompt_version"] == 3
        assert all(llm.cache_file(PASS, k).exists() for k in v["cache_keys"].values())
        assert v["cache_key"] == v["cache_keys"]["full"]
        n = len(c.seed.chunks)
        assert v["report"]["facts_recovered"] == [True] * len(c.facts)
        assert v["report"]["ablation"] == ({} if n == 1 else {"drop:F1": True, "drop:F2": True})


def test_yes_judge_fails_every_judge_kind(tmp_path):
    kept, st, *_ = _run(tmp_path, "yes")
    # M3-R25 B: a typed value counts inside any verbatim quote, so whole-fragment quotes pass a single pair; every
    # multi-hop pair still fails the ablation and the plants (on the accepted singles) refuse the judge
    assert kept and all(c.seed.pattern == "single" for c in kept)
    # it quotes the replacement fragment too, so each neighbor_swap miss is contested (adjudicated, not scored) and
    # the kind falls below min n -- a STOP as well
    assert set(st["sensitivity_by_kind"]) == {"question_swap", "padded_single"} and st["contested_by_kind"]["neighbor_swap"]
    assert all(v < 0.9 for v in st["sensitivity_by_kind"].values()) and st["kinds_below_min_n"] == ["neighbor_swap"]
    assert {r.split(":")[0] for r in sensitivity_gate(st, 0.9)} == {"question_swap", "padded_single",
                                                                    "too few plants of kinds ['neighbor_swap']"}
    assert sum(sum(v.values()) for v in st["rejected_by_cell"].values()) == 80
    assert set(st["rejected_by_cell"]) == {f"{sp}|refers_internal|{l}" for sp in ("train", "test")
                                           for l in ("ru", "uz-Cyrl")}


def test_content_blind_judge_fails_the_ablations(tmp_path):
    """Review I1: a judge that reads no fragment text and only uses view structure cannot pass an ablation."""
    kept, st, *_ = _run(tmp_path, "blind")
    assert all(c.seed.pattern == "single" for c in kept)
    assert st["rejected"] == {"refers_internal|ablation": 80}
    assert sensitivity_gate(st, 0.9)                                     # its verdicts are refused as well


def test_fragment_count_judge_is_caught_by_padded_single(tmp_path):
    """A judge answering 'answerable iff >= 2 fragments' passes 2-hop ablations; the plants catch it. M3-R25: plants sit
    on accepted bases only -- such a judge rejects every single pair, so no padded_single can be planted (a STOP by
    itself) and question_swap on the multi bases catches it."""
    kept, st, *_ = _run(tmp_path, "size")
    assert all(c.seed.pattern != "single" for c in kept)
    assert "padded_single" in st["kinds_below_min_n"] and st["sensitivity_by_kind"]["question_swap"] < 0.9
    assert any("padded_single" in r for r in sensitivity_gate(st, 0.9))


def test_too_few_plants_of_a_kind_is_a_stop():
    st = {"sensitivity_by_kind": {"question_swap": 1.0}, "kinds_below_min_n": ["neighbor_swap", "padded_single"]}
    assert len(sensitivity_gate(st, 0.9)) == 1 and "padded_single" in sensitivity_gate(st, 0.9)[0]


def test_missing_ablation_verdict_rejects():
    c = _cand()
    full = {"answerable": True, "answer": "Марказий банк, ўттиз кун ичида", "evidence": [
        {"id": "F1", "quote": "ваколатлар Марказий банк томонидан"}, {"id": "F2", "quote": "қарор ўттиз кун ичида"}]}
    ok, rep = judge_ok(c, {"full": full, "drop:F1": {"answerable": False, "evidence": []}}, {"F1": T1, "F2": T2})
    assert not ok and rep["ablation"] == {"drop:F1": True, "drop:F2": False}


def test_quote_from_another_fragment_does_not_recover_a_fact():
    c = _cand()
    full = {"answerable": True, "answer": "…", "evidence": [
        {"id": "F1", "quote": "ваколатлар Марказий банк томонидан"}, {"id": "F1", "quote": "Марказий банк"}]}
    no = {"answerable": False, "evidence": []}
    texts = {"F1": T1 + " қарор ўттиз кун ичида", "F2": T2}
    ok, rep = judge_ok(c, {"full": dict(full, evidence=full["evidence"] + [{"id": "F1", "quote": "қарор ўттиз кун ичида"}]),
                           "drop:F1": no, "drop:F2": no}, texts)
    assert not ok and rep["facts_recovered"] == [True, False]                # F2's value quoted from F1 only


def test_quote_length_cap():
    t = "а" * 100 + " қарор ўттиз кун ичида " + "б" * 300
    assert quote_ok("қарор ўттиз кун ичида", t, CAP)
    assert not quote_ok(t, t, CAP)                                          # the whole fragment
    short = "қарор ўттиз кун ичида қабул қилинади"
    assert quote_ok(short, short, CAP)                                      # a short fragment may be quoted whole


def test_offline_replay_is_identical_with_zero_calls(tmp_path):
    kept, st, *_ = _run(tmp_path)
    kept2, st2, _, llm2, _ = _run(tmp_path, api=False)
    assert [c.provenance for c in kept2] == [c.provenance for c in kept] and st2 == st
    assert llm2.stats[PASS]["api_calls"] == 0


def test_pilot_first_calls_are_replayed_by_the_full_run(tmp_path):
    """Review I4: the full run re-uses every pilot call from the cache."""
    text, cs, need = _world(120)
    pilot = {c.seed.seed_id for c in cs[:30]}
    _, st_p, _, llm_p, _ = _run(tmp_path, pilot=pilot, cands=cs[:30])
    kept, st, _, llm, _ = _run(tmp_path, pilot=pilot)
    assert llm.stats[PASS]["hits"] == st_p["calls"] and llm.stats[PASS]["api_calls"] == st["calls"] - st_p["calls"]


def test_verify_respects_the_whole_ledger_total(tmp_path):
    """M3-R3 п. 14 (Task 15 fix guard_total): 2500 over cache/dataset/ledger.json, checked before each uncached call."""
    (tmp_path / "ledger.json").write_text(json.dumps({"ds_facts": {"2026-09-29": 1600}, "ds_generate": {"2026-09-29": 900}}))
    text, cs, need = _world(6)
    j = _Judge(need)
    llm = KgLLM(VCFG, cache_root=tmp_path, client=j.client, allow_api=True)
    with pytest.raises(KgBudgetExceeded):
        verify_candidates(cs, _frag_of(text), llm, 2, 1.0, 7, lambda c: "U#1")
    assert j.requests == []


def test_prompt_is_versioned_digest_pinned_and_in_the_cache_key(tmp_path):
    assert prompt_digest(PROMPT_VER, SCHEMA_VER) == PINNED_VERIFY
    _run(tmp_path, n=6)
    bumped = {"llm": {**VCFG["llm"], "passes": {"ds_verify": {"prompt_version": 4, "max_calls": 1000}}}}
    text, cands, _ = _world(6)
    with pytest.raises(KgCacheMiss):
        verify_candidates(cands, _frag_of(text), KgLLM(bumped, cache_root=tmp_path, allow_api=False), 2, 1.0, 7,
                          lambda c: "U#1")
    off = KgLLM(VCFG, cache_root=tmp_path, allow_api=False)
    assert off.cache_key(PASS, _canon_input({}), prompt_digest(PROMPT_VER + " ", SCHEMA_VER)) != \
        off.cache_key(PASS, _canon_input({}), prompt_digest(PROMPT_VER, SCHEMA_VER))


PINNED_VERIFY = "e74ed2067c3f0eeb1593cb4fe6a74ac35e7219f1189426ffcf9ea85a1f7b495f"   # ds_verify v6 (M3-R30: one verbatim quote per value)


def test_plant_details():
    cands = [_cand(i, doc=f"D{i % 4}") for i in range(60)] + [_cand(100 + i, doc=f"D{i % 4}", multi=False) for i in range(20)]
    from graphrag.dataset.verify import plan_batches
    kinds = set()
    for b in plan_batches(cands, 2, 1.0, 7, lambda c: "OTHER#1", neighbor=lambda c, i: "NB#1"):
        assert len(b["plants"]) == 1
        p = b["plants"][0]
        c = p["cand"]
        kinds.add(p["kind"])
        assert c in b["cands"]
        if p["kind"] == "question_swap":                         # another pair of the SAME document, disjoint C_ref
            other = next(o for o in cands if o.question == p["question"])
            assert other is not c and other.seed.docs[0] == c.seed.docs[0] and not set(other.seed.chunks) & set(c.seed.chunks)
        if p["kind"] == "value_swap":
            assert all(v != f.value for v, f in zip(p["values"], c.facts))
        if p["kind"] == "padded_single":
            assert len(c.seed.chunks) == 1 and sorted(p["chunks"]) == sorted([c.seed.chunks[0], "OTHER#1"])
        if p["kind"] == "neighbor_swap":
            assert "NB#1" in p["chunks"] and len(p["chunks"]) == len(c.seed.chunks)
    assert kinds == {"chunk_swap", "value_swap", "question_swap", "padded_single", "neighbor_swap"}


def test_unrelated_fragment_same_split_other_document_per_candidate():
    chunks = {f"{d}/П.{k}#1": type("Ch", (), {"tokens": 50})() for d in ("A", "B", "C", "T") for k in range(5)}
    splits = {"A": "train", "B": "train", "C": "train", "T": "test"}
    pick = unrelated_picker(chunks, splits, 7, 30, section_of={"A": "QONUNLAR", "B": "QONUNLAR", "C": "BOSHQA", "T": "QONUNLAR"})
    got = set()
    for i in range(10):
        c = _cand(i, doc="A")
        x = pick(c)
        assert x.split("/")[0] == "B"                            # same split, other document, same section first
        got.add(x)
    assert len(got) > 1                                          # not one fragment for every plant


def test_neighbor_is_adjacent_same_document_and_lacks_the_value():
    Ch = lambda t, n=50: type("Ch", (), {"tokens": n, "text": t})()
    order = {"D": ["D/СТ.2#1", "D/СТ.3#1", "D/СТ.4#1", "D/СТ.5#1", "D/СТ.6#1"]}
    ch = {"D/СТ.2#1": Ch("бошқа матн"), "D/СТ.3#1": Ch(T1), "D/СТ.4#1": Ch("Марказий банк ҳисобот беради"),
          "D/СТ.5#1": Ch(T2), "D/СТ.6#1": Ch("кичик", 5)}
    pick = neighbor_picker(order, ch, 7, 30)
    c = _cand(1)
    c = Candidate(c.seed.__class__(**{**c.seed.__dict__, "chunks": ("D/СТ.3#1", "D/СТ.5#1")}), c.question, c.answer,
                  [Fact("D/СТ.3#1@0", "D/СТ.3#1", "", "Марказий банк", "x"), Fact("D/СТ.5#1@0", "D/СТ.5#1", "", "ўттиз кун", "y")], {})
    assert pick(c, 0) == "D/СТ.2#1"                              # СТ.4 carries "Марказий банк"
    assert pick(c, 1) == "D/СТ.4#1"                              # СТ.6 is too short; СТ.4 lacks "ўттиз кун"


def test_value_must_match_whole_words_not_a_substring():
    """Task 15 concern: value "5" must not be recovered from a quote that only says "15"."""
    s = Seed("single-0000000001", "single", "ru", ("D/СТ.5#1",), ("D",), "train", "QONUNLAR")
    c = Candidate(s, "Сколько дней дается?", "5", [Fact("D/СТ.5#1@0", "D/СТ.5#1", "", "5 кун", "қарор 15 кун ичида")], {})
    t = {"F1": "Қарор 15 кун ичида қабул қилинади, 5 кун эмас."}
    v = lambda q: {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": q}]}}
    assert judge_ok(c, v("Қарор 15 кун ичида"), t)[0] is False
    assert judge_ok(c, v("5 кун эмас"), t)[0] is True


# --- pre-pilot fix (M3-R18 round, re-review I1): decimals and grouped numbers are single tokens -------------------
from graphrag.dataset.facts import check_fact, ctoks, numbers, value_in  # noqa: E402


@pytest.mark.parametrize("text,toks", [("0,5 фоиз", ["0.5", "foiz"]), ("1.5 баравар", ["1.5", "baravar"]),
                                       ("1 500 сўм", ["1500", "som"]), ("1 500 сўм", ["1500", "som"]),
                                       ("1 500 сўм", ["1500", "som"]), ("12,5%", ["12.5"]),
                                       ("1 000 000 сўм", ["1000000", "som"]), ("3, 5 ва 7", ["3", "5", "va", "7"]),
                                       ("5-модда", ["5", "modda"])])
def test_decimals_and_grouped_thousands_are_one_number_token(text, toks):
    assert [t.s for t in ctoks(text)] == toks


def test_a_value_never_matches_inside_a_decimal_or_a_grouped_number():
    assert not value_in("5 фоиз", "жарима 0,5 фоиз миқдорида")
    assert not value_in("500", "жарима 1 500 сўм")
    assert not value_in("5 баравар", "1.5 баравар оширилади")
    assert value_in("0,5 фоиз", "жарима 0,5 фоиз миқдорида") and value_in("1 500 сўм", "жарима 1 500 сўм")
    assert numbers("12,5% ва 1 500") == {"12.5", "1500"}
    f = {"statement": "s", "value": "5 фоиз", "quote": "жарима ставкаси 0,5 фоиз миқдорида белгиланади"}
    assert check_fact(f, "Жарима ставкаси 0,5 фоиз миқдорида белгиланади.", set()) is not None


def test_judge_recovery_rejects_a_value_inside_a_decimal():
    s = Seed("single-0000000002", "single", "ru", ("D/СТ.5#1",), ("D",), "train", "QONUNLAR")
    c = Candidate(s, "Какой размер пени?", "5 фоиз", [Fact("D/СТ.5#1@0", "D/СТ.5#1", "", "5 фоиз", "0,5 фоиз")], {})
    t = {"F1": "Пеня ставкаси 0,5 фоиз миқдорида белгиланади ва 1 500 сўмдан кам бўлмайди."}
    v = lambda q: {"full": {"answerable": True, "evidence": [{"id": "F1", "quote": q}]}}
    assert judge_ok(c, v("Пеня ставкаси 0,5 фоиз миқдорида"), t)[0] is False
    c2 = Candidate(s, "Какой минимум?", "500", [Fact("D/СТ.5#1@1", "D/СТ.5#1", "", "500 сўм", "1 500 сўмдан")], {})
    assert judge_ok(c2, v("1 500 сўмдан кам бўлмайди"), t)[0] is False


# --- M3-R18 (a): pilot plants >= min_kind_n per judge kind, frozen pilot plan replayed by the full run ------------
def _run18(tmp_path, cands, pilot, share, api=True, frozen=None, target=12):
    text, _, need = _world(240)
    j = _Judge(need)
    llm = KgLLM(VCFG, cache_root=tmp_path, client=j.client if api else None, allow_api=api)
    kept, st = verify_candidates(cands, _frag_of(text), llm, 2, share, 7, lambda c: "U#1", min_kind_n=10,
                                 views_per_call=5, neighbor=lambda c, i: f"N:{c.seed.chunks[i]}", pilot_ids=pilot,
                                 quote_cap=CAP, pilot_target=target, frozen_plan=frozen)
    return kept, st, j, llm


def test_pilot_reaches_min_kind_n_per_judge_kind_with_several_plants_per_base(tmp_path):
    _, cs, _ = _world(240)
    pilot = cs[:72]                                                        # ~ the Task 17 pilot: ~72 candidates
    kept, st, _, _ = _run18(tmp_path, pilot, {c.seed.seed_id for c in pilot}, 0.15)
    assert all(st["pilot"]["planted"][k] >= 10 for k in JUDGE_KINDS) and st["pilot"]["kinds_below_min_n"] == []
    assert sensitivity_gate(st["pilot"], 0.9) == [] and st["pilot"]["plant_share"] > 1.0   # > 1 plant per group
    from graphrag.dataset.verify import plan_batches
    groups = plan_batches(pilot, 2, 0.15, 7, lambda c: "U#1", {c.seed.seed_id for c in pilot},
                          lambda c, i: f"N:{c.seed.chunks[i]}", 12)
    per_base = __import__("collections").defaultdict(list)
    for g in groups:
        for p in g["plants"]:
            per_base[p["cand"].seed.seed_id].append(p["kind"])
    assert any(len(v) > 1 for v in per_base.values())
    assert all(len(v) == len(set(v)) for v in per_base.values())          # several plants per base: distinct kinds


def test_full_run_replays_every_pilot_verify_call_and_checks_the_frozen_plan(tmp_path):
    _, cs, _ = _world(240)
    pid = {c.seed.seed_id for c in cs[:72]}
    _, st_p, _, _ = _run18(tmp_path, cs[:72], pid, 0.15)
    kept, st, j, llm = _run18(tmp_path, cs, pid, 0.15, frozen=st_p["plan_digest_pilot"])
    assert llm.stats[PASS]["hits"] == st_p["calls"]                        # 100 % of the pilot's calls reused
    assert llm.stats[PASS]["api_calls"] == st["calls"] - st_p["calls"]
    assert st["plan_digest_pilot"] == st_p["plan_digest_pilot"]
    assert st["pilot"] == st_p["pilot"]                                    # the gate on the pilot is the same
    rest_groups = (len(cs) - 72) // 2
    assert sum(st["planted_total"].values()) - st_p["pilot"]["plants"] <= rest_groups * 0.3   # main run ~ 0.15
    with pytest.raises(ValueError, match="frozen pilot plan"):
        _run18(tmp_path, cs, pid, 0.15, api=False, frozen=st_p["plan_digest_pilot"], target=11)
