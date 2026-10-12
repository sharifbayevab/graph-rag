"""M3-R23 (user decision 2026-09-29): Opus generator packets, ingest of the generator JSONL through every code check,
gpt-4.1 verify per packet part (pilot first), reviewer packets without generator notes, reader verdict ingest."""
import json

import pytest

from graphrag import config
from graphrag.dataset import opus
from graphrag.dataset.facts import Fact
from graphrag.dataset.seeds import Seed
from graphrag.dataset.sources import SourceIndex
from graphrag.kg.llm import KgLLM
from tests.test_m3_r21_dataset import ACT_B, _client, _toy

SECTIONS = {"B": "QONUNLAR", "S": "STENOGRAMMA", "X": "BOSHQA"}
CFG = {"seed": 0, "max_copy_words": 99, "max_lexical_overlap": 1.0}
Q_OK = (f"«{ACT_B}»ги Қонунга кўра Омонатларни кафолатлаш агентлигига ўтган талаблар кредиторлар талаблари орасида "
        "нечанчи ўринда қондирилади?")
A_OK = "Улар учинчи навбатда қаноатлантирилади."
QUOTE_B = "учинчи навбатда – Омонатларни кафолатлаш агентлигига ўтган талаблар"


def _instr():
    o = config.load("train")["dataset"]["opus"]["generator"]
    return opus.instructions(config.path(o["file"]), o["version"], rel=o["file"])


def _rubric():
    o = config.load("train")["dataset"]["opus"]["reader"]
    return opus.instructions(config.path(o["file"]), o["version"], rel=o["file"])


def _single_packet(facts=None):
    bx, recs, chunks = _toy()
    idx = SourceIndex(recs)
    s = Seed("single-b64", "single", "uz-Cyrl", ("B/СТ.64#1",), ("B",), "train", "QONUNLAR")
    e = opus.single_entry(s, chunks, idx, facts or {})
    pk = opus.packet_doc("pilot-single-train-001", [e], _instr(), "gen/pilot-single-train-001.jsonl")
    return pk, chunks


def _row(**kw):
    r = {"seed_id": "single-b64", "question": Q_OK, "answer": A_OK, "lang": "uz-Cyrl",
         "used_facts": [{"fragment": "F1", "quote": QUOTE_B, "value": "учинчи навбатда"}],
         "neighbor_id": None, "sources": ["F1"], "notes": "SECRET-NOTE the order of claims"}
    r.update(kw)
    return json.dumps(r, ensure_ascii=False)


def _ingest(pk, chunks, text):
    return opus.ingest_generated(pk, text, chunks, CFG, SECTIONS, "gen/x.jsonl", "claude-opus-test")


# ---- versioned instruction files ---------------------------------------------------------------------------------------
def test_instruction_files_match_their_configured_digest_and_hold_rules_rubric_glossary():
    o = config.load("train")["dataset"]
    for which in ("generator", "reader"):
        c = o["opus"][which]
        d = opus.instructions(config.path(c["file"]), c["version"])
        assert d["digest"] == c["digest"], f"{c['file']} edited: bump version + digest in config dataset.opus"
    gen, rd = _instr()["text"], _rubric()["text"]
    for term, ru in o["legal_glossary"].items():
        assert f"| {term} | {ru} |" in gen and f"| {term} | {ru} |" in rd
    for c in ("c1", "c2", "c3", "c4", "c5", "c6", "c7"):
        assert f"- {c} **" in gen and f"- {c} **" in rd
    uz = o["copy_calibration"]["families"]["uz"]
    assert f"**{uz['max_copy_words']}**" in gen and f"**{uz['max_lexical_overlap']}**" in gen
    assert f"**{o['max_copy_words']}** for ru" in gen and "**0.60** for ru" in gen and o["max_lexical_overlap"] == 0.6
    assert '"skip": true' in gen and "notes" in gen and "notes" not in rd


# ---- generator packets -------------------------------------------------------------------------------------------------
def test_packets_are_split_pure_deterministic_and_carry_texts_facts_neighbours():
    from graphrag.dataset.bridge import sample_anchors
    bx, recs, chunks = _toy()
    idx = SourceIndex(recs)
    f = Fact("B/СТ.64#1@0", "B/СТ.64#1", "st", "учинчи навбатда", QUOTE_B, "B#B0", "k0", "term")
    seeds = [Seed(f"single-{i}", "single", "ru", ("B/СТ.64#1",), ("B",), sp, "QONUNLAR")
             for i, sp in enumerate(["train"] * 5 + ["val"] * 3)]
    es = [opus.single_entry(s, chunks, idx, {"B/СТ.64#1": [f]}) for s in seeds]
    g1 = opus.make_packets(es, 2, "", 0)
    g2 = opus.make_packets(list(reversed(es)), 2, "", 0)
    assert [(p, [e["seed_id"] for e in x]) for p, x in g1] == [(p, [e["seed_id"] for e in x]) for p, x in g2]
    assert [p for p, _ in g1] == ["single-train-001", "single-train-002", "single-train-003", "single-val-001",
                                  "single-val-002"]
    assert all(len({e["cell"]["split"] for e in x}) == 1 for _, x in g1)
    assert sorted(len(x) for _, x in g1 if "train" in _) == [1, 2, 2]
    e = es[0]
    assert e["fragments"][0]["text"] == chunks["B/СТ.64#1"].text and e["fragments"][0]["act"] == ACT_B
    assert e["facts"][0]["fact_id"] == "F1.0" and e["facts"][0]["src"]["cache_key"] == "k0" and e["facts_cached"]
    with pytest.raises(ValueError):
        opus.packet_doc("mixed", [es[0], es[-1]], _instr(), "x")
    pick = sample_anchors(bx, ["B/СТ.24#1"], 1, ["uz-Cyrl"], 0)[0]
    m = opus.multi_entry(pick, chunks, idx, {}, 0)
    assert m["anchor"]["text"] == chunks["B/СТ.24#1"].text and 1 <= len(m["neighbors"]) <= 8
    nb = {x["chunk_id"]: x for x in m["neighbors"]}
    assert nb["B/СТ.64#1"]["source"] == "refers_to" and nb["B/СТ.64#1"]["edges"] == ["REFERS_TO"]
    assert m["cell"] == {"split": "train", "pattern": "multi", "lang": "uz-Cyrl"}
    pk = opus.packet_doc("p", [m], _instr(), "gen/p.jsonl")
    assert pk["instructions"]["digest"] == opus.sha256_text(pk["instructions"]["text"]) and pk["role"] == "opus-generator"
    order = opus.wave_order(["m-1", "s-1", "s-2", "s-3", "s-4"], lambda p: p[0])
    assert order.index("m-1") in (1, 2) and sorted(order) == ["m-1", "s-1", "s-2", "s-3", "s-4"]


# ---- ingest-gen ---------------------------------------------------------------------------------------------------------
def test_ingest_accepts_a_checked_pair_with_provenance_and_replays_byte_identically():
    pk, chunks = _single_packet()
    text = _row() + "\n"
    cands, rej, st = _ingest(pk, chunks, text)
    assert rej == [] and len(cands) == 1, rej
    c = cands[0]
    assert c.seed.seed_id == "single-b64" and c.facts[0].quote in chunks["B/СТ.64#1"].text
    p = c.provenance
    assert p["generate"]["role"] == "opus-generator" and p["generate"]["prompt_digest"] == _instr()["digest"]
    assert p["opus"]["packet_id"] == "pilot-single-train-001" and p["opus"]["line"] == 1
    assert p["opus"]["row_sha256"] == opus.sha256_text(text.strip()) and p["generate"]["cache_key"] == p["opus"]["row_sha256"]
    assert p["facts"]["cache_keys"] == {"opus-generator": p["opus"]["row_sha256"]}
    again, _, _ = _ingest(pk, chunks, text)
    dump = lambda cs: json.dumps([[x.seed.to_json(), x.question, x.answer, [f.__dict__ for f in x.facts], x.provenance]
                                  for x in cs], ensure_ascii=False, sort_keys=True)
    assert dump(again) == dump(cands)


def test_ingest_uses_cached_facts_by_id_and_keeps_their_ds_facts_keys():
    f = Fact("B/СТ.64#1@0", "B/СТ.64#1", "st", "учинчи навбатда", QUOTE_B, "B#B0", "k0", "term")
    pk, chunks = _single_packet({"B/СТ.64#1": [f]})
    cands, rej, _ = _ingest(pk, chunks, _row(used_facts=[{"fact_id": "F1.0", "value": "учинчи навбатда"}]))
    assert rej == [] and cands[0].facts == [f] and cands[0].provenance["facts"]["cache_keys"] == {"B#B0": "k0"}
    _, rej, _ = _ingest(pk, chunks, _row(used_facts=[{"fact_id": "F1.0", "value": "бешинчи навбатда"}]))
    assert [r["reason"] for r in rej] == ["fact_value_mismatch"]


@pytest.mark.parametrize("row,reason", [
    (_row(question=Q_OK.replace("нечанчи ўринда", "учинчи навбатда")), "answer_in_question"),
    (_row(used_facts=[{"fragment": "F1", "quote": "бешинчи навбатда – бошқа талаблар ва мажбуриятлар",
                       "value": "бешинчи навбатда"}]), "fact_quote_not_verbatim"),
    (_row(lang="ru"), "lang_mismatch"),
    (_row(used_facts=[{"fragment": "N1", "quote": QUOTE_B, "value": "учинчи навбатда"}]), "fact_wrong_fragment"),
    (_row(sources=["F1", "F2"]), "sources_mismatch"),
    (_row(neighbor_id="N1"), "sources_mismatch"),
    (_row(question=Q_OK.replace("нечанчи ўринда", "нечанчи ўринда ва ким томонидан")), "compound"),
    (_row(answer="Улар учинчи навбатда Koson туманида қаноатлантирилади."), "language"),
    (_row(answer="Улар учинчи навбатда Самарқандда қаноатлантирилади."), "foreign_name"),
    (_row(question="Ушбу ҳужжатга кўра агентликка ўтган талаблар нечанчи ўринда қондирилади?"), "deictic"),
    (_row(used_facts=[{"fragment": "F1", "quote": QUOTE_B, "value": "Омонатларни кафолатлаш агентлигига"}]),
     "value_missing_in_answer"),
    (json.dumps({"seed_id": "single-b64", "skip": True, "reason": "filler"}), "skip"),
    (json.dumps({"seed_id": "nope", "question": "x"}), "unknown_seed"),
    ("{not json", "bad_json"),
    (_row(used_facts="F1.0"), "schema"),
])
def test_ingest_rejects_every_code_check_failure_with_its_reason(row, reason):
    pk, chunks = _single_packet()
    cands, rej, _ = _ingest(pk, chunks, row + "\n")
    assert cands == [] and reason in [r["reason"] for r in rej], rej
    r = next(r for r in rej if r["reason"] == reason)
    assert r["packet_id"] == "pilot-single-train-001"
    if reason not in ("unknown_seed", "bad_json"):
        assert r["cell"] == "train|single|uz-Cyrl"


def test_ingest_logs_missing_and_duplicate_rows_and_pending_packets():
    pk, chunks = _single_packet()
    _, rej, st = _ingest(pk, chunks, "")
    assert [r["reason"] for r in rej] == ["missing_output"]
    cands, rej, _ = _ingest(pk, chunks, _row() + "\n" + _row() + "\n")
    assert len(cands) == 1 and [r["reason"] for r in rej] == ["duplicate_seed"]
    _, rej, st = _ingest(pk, chunks, None)
    assert rej == [] and st["pending"]
    bad = json.loads(json.dumps(pk))
    bad["seeds"][0]["fragments"][0]["text"] += " x"
    with pytest.raises(ValueError, match="stale"):
        _ingest(bad, chunks, _row())
    bad = json.loads(json.dumps(pk))
    bad["instructions"]["text"] += " "
    with pytest.raises(ValueError, match="digest"):
        _ingest(bad, chunks, _row())


def _multi_packet():
    from graphrag.dataset.bridge import sample_anchors
    bx, recs, chunks = _toy()
    pick = sample_anchors(bx, ["B/СТ.24#1"], 1, ["uz-Cyrl"], 0)[0]
    e = opus.multi_entry(pick, chunks, SourceIndex(recs), {}, 0)
    nid = next(x["id"] for x in e["neighbors"] if x["chunk_id"] == "B/СТ.64#1")
    other = next(x["id"] for x in e["neighbors"] if x["chunk_id"] != "B/СТ.64#1")
    return opus.packet_doc("pilot-multi-train-001", [e], _instr(), "gen/m.jsonl"), chunks, e["seed_id"], nid, other


def _mrow(sid, nid, **kw):
    r = {"seed_id": sid, "lang": "uz-Cyrl", "neighbor_id": nid, "sources": ["A", nid], "notes": "n",
         "question": f"«{ACT_B}»ги Қонуннинг 24-моддаси ҳавола қилган 64-моддага кўра Омонатларни кафолатлаш "
                     "агентлигига ўтган талаблар нечанчи ўринда қондирилади?",
         "answer": "Улар учинчи навбатда қондирилади.",
         "used_facts": [{"fragment": "A", "quote": "ушбу Қонуннинг 64-моддасида белгиланган кредиторларнинг",
                         "value": "64-моддасида"},
                        {"fragment": nid, "quote": QUOTE_B, "value": "учинчи навбатда"}]}
    r.update(kw)
    return json.dumps(r, ensure_ascii=False)


def test_ingest_multi_builds_the_anchor_neighbour_seed_and_checks_the_chain():
    pk, chunks, sid, nid, other = _multi_packet()
    cands, rej, _ = _ingest(pk, chunks, _mrow(sid, nid))
    assert rej == [] and len(cands) == 1, rej
    c = cands[0]
    assert c.seed.chunks == ("B/СТ.24#1", "B/СТ.64#1") and c.seed.pattern == "multi" and c.seed.walk
    assert c.provenance["bridge_source"] == "refers_to" and c.provenance["opus"]["packet_seed_id"] == sid
    assert all(f.quote in chunks[f.chunk_id].text for f in c.facts)
    for row, why in ((_mrow(sid, other, sources=["A", other]), "fact_quote_not_verbatim"),
                     (_mrow(sid, nid, sources=["F1"]), "sources_mismatch"),
                     (_mrow(sid, "N9", sources=["A", "N9"]), "unknown_neighbor"),
                     (_mrow(sid, nid, used_facts=json.loads(_mrow(sid, nid))["used_facts"][1:]), "chain_incomplete")):
        cs, rj, _ = _ingest(pk, chunks, row)
        assert cs == [] and why in [r["reason"] for r in rj], (why, rj)


# ---- verify: named parts (one per packet) ------------------------------------------------------------------------------
def _cand(i, part_doc="B"):
    from graphrag.dataset.generate import Candidate
    s = Seed(f"single-{part_doc}{i:03d}", "single", "ru", (f"{part_doc}/П.{i}#1",), (part_doc,), "train", "QONUNLAR")
    return Candidate(s, f"Вопрос {part_doc}{i}?", "Ответ.", [Fact(f"{s.chunks[0]}@0", s.chunks[0], "", f"значение {i}",
                                                                  "q", "", "", "term")], {})


def test_verify_plans_each_named_part_on_its_own():
    from graphrag.dataset.verify import pack_calls, plan_batches
    a = [_cand(i, "A") for i in range(6)]
    b = [_cand(i, "B") for i in range(5)]
    parts = {c.seed.seed_id: "packet:a" for c in a} | {c.seed.seed_id: "packet:b" for c in b}
    un = lambda c: "Z/П.1#1"
    sig = lambda calls, who: [[(it["who"], it["view"]) for it in call] for call in calls
                              if any(it["who"].startswith(who) for it in call)]
    both = pack_calls(plan_batches(a + b, 2, 1.0, 0, un, parts=parts), 5, 0)
    only = pack_calls(plan_batches(a, 2, 1.0, 0, un, parts=parts), 5, 0)
    assert all(len({it["part"] for it in call}) == 1 for call in both)
    assert [c for c in both if c[0]["part"] == "packet:a"] == [c for c in only if c[0]["part"] == "packet:a"]
    legacy = pack_calls(plan_batches(a + b, 2, 1.0, 0, un), 5, 0)
    assert {it["part"] for c in legacy for it in c} == {"rest"}


def test_verify_replays_ingested_candidates_offline_with_zero_calls(tmp_path):
    from graphrag.dataset.verify import verify_candidates
    pk, chunks = _single_packet()
    cands, _, _ = _ingest(pk, chunks, _row())

    def judge(u):
        return {"views": [{"key": v["key"], "answerable": True, "answer": "учинчи",
                           "evidence": [{"id": f["id"], "quote": QUOTE_B} for f in v["fragments"]]}
                          for v in u["views"]]}
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 50,
                   "passes": {"ds_verify": {"prompt_version": 4, "max_calls": 10}}}}
    frag_of = lambda c, lab: {"id": lab, "act": ACT_B, "where": ACT_B, "text": chunks[c].text}
    cl, calls = _client(judge)
    parts = {c.seed.seed_id: "packet:" + c.provenance["opus"]["packet_id"] for c in cands}
    kept, st = verify_candidates(cands, frag_of, KgLLM(cfg, cache_root=tmp_path, client=cl, allow_api=True), 2, 0.0, 0,
                                 lambda c: "B/СТ.3#1", parts=parts)
    assert len(calls) == 1 and len(kept) == 1, st
    again, _, _ = _ingest(pk, chunks, _row())
    off = KgLLM(cfg, cache_root=tmp_path, allow_api=False)
    kept2, _ = verify_candidates(again, frag_of, off, 2, 0.0, 0, lambda c: "B/СТ.3#1", parts=parts)
    assert off.total_api_calls() == 0
    assert json.dumps([k.provenance for k in kept2], sort_keys=True, ensure_ascii=False) == \
        json.dumps([k.provenance for k in kept], sort_keys=True, ensure_ascii=False)


# ---- reviewer packets / verdicts ----------------------------------------------------------------------------------------
def _verified():
    pk, chunks = _single_packet()
    c = _ingest(pk, chunks, _row())[0][0]
    c.provenance["verify"] = {"verdicts": {"full": {"answer": "JUDGE-ANSWER"}}}
    return c, chunks


def test_review_packets_hold_pair_fragments_rubric_and_never_generator_notes():
    c, chunks = _verified()
    frag_of = lambda cid, lab: {"id": lab, "act": ACT_B, "where": ACT_B + ", ст. 64", "text": chunks[cid].text}
    pk = opus.make_review_packets([c], frag_of, _rubric(), 40, 0, {}, pilot_ids={c.seed.seed_id}, out_dir="v")
    assert [p["packet_id"] for p in pk] == ["review-pilot-001"]
    blob = json.dumps(pk, ensure_ascii=False)
    for secret in ("SECRET-NOTE", "used_facts", "notes", "JUDGE-ANSWER", "row_sha256", QUOTE_B[:40] + "\"", "cell"):
        assert secret not in blob.replace(chunks["B/СТ.64#1"].text, ""), secret
    it = pk[0]["items"][0]
    assert set(it) == {"pair_id", "lang", "pattern", "question", "answer", "fragments"}
    assert set(it["fragments"][0]) == {"id", "kind", "act", "where", "text"} and pk[0]["role"] == "strict-reader"
    assert pk[0]["rubric"]["digest"] == _rubric()["digest"]
    from graphrag.dataset.finalize import _h
    assert it["pair_id"] == f"syn-{_h('pair', c.seed.seed_id)[:10]}"
    assert opus.make_review_packets([c], frag_of, _rubric(), 40, 0, {"review-pilot-001": [it["pair_id"]]}) == []


def test_ingest_verdicts_keeps_only_good_and_logs_reasons_per_cell():
    from graphrag.dataset.generate import Candidate
    c, chunks = _verified()
    cs = [c] + [Candidate(Seed(f"single-z{i}", "single", "ru", (f"B/П.{i}#1",), ("B",), "val", "QONUNLAR"), f"В{i}?",
                          "О.", [], {"generate": {"role": "opus-generator"}, "opus": {"agent": "gen-1"}})
                for i in range(4)]
    ids = [opus.pair_id(x.seed.seed_id) for x in cs]
    rub = _rubric()
    pk = {"packet_id": "review-001", "rubric": rub, "items": [{"pair_id": p} for p in ids]}
    pend = {"packet_id": "review-002", "rubric": rub, "items": []}
    lines = [{"pair_id": ids[0], "verdict": "GOOD", "reasons": []},
             {"pair_id": ids[1], "verdict": "BAD", "reasons": [{"criterion": "c4", "note": "wrong act"},
                                                                {"criterion": "c3", "note": "which draft"}]},
             {"pair_id": ids[2], "verdict": "BORDERLINE", "reasons": [{"criterion": "c7", "note": "grammar"}]},
             {"pair_id": ids[3], "verdict": "GOOD", "reasons": [], "agent": "gen-1"},
             {"pair_id": "syn-unknown", "verdict": "GOOD", "reasons": []}]
    text = "\n".join(json.dumps(x) for x in lines)
    good, rej, st = opus.ingest_verdicts([pk, pend], {"review-001": text, "review-002": None}, cs, rub["digest"])
    assert [g.seed.seed_id for g in good] == ["single-b64"] and good[0].provenance["review"]["verdict"] == "GOOD"
    why = {r["reason"] for r in rej}
    assert {"bad", "borderline", "same_agent", "unknown_pair", "missing_verdict"} <= why
    assert st["rejected_by_reason"] == {"BAD|c3": 1, "BAD|c4": 1, "BORDERLINE|c7": 1}
    assert st["rejected_by_cell"]["val|single|ru"] == {"BAD|c3": 1, "BAD|c4": 1, "BORDERLINE|c7": 1}
    assert st["pending_packets"] == ["review-002"] and st["good"] == 1
    with pytest.raises(ValueError):
        opus.ingest_verdicts([pk], {"review-001": text}, cs, "0" * 64)


def test_reader_gate_and_generator_provenance(tmp_path):
    assert opus.reader_gate(40, 40, 0.90, 0.75)["ok"] and not opus.reader_gate(8, 10, 0.90, 0.75)["ok"]
    assert not opus.reader_gate(0, 0, 0.90, 0.75)["ok"]
    line = _row()
    (tmp_path / "g.jsonl").write_text(line + "\n", encoding="utf-8")
    prov = {"opus": {"gen_file": "g.jsonl", "line": 1, "row_sha256": opus.sha256_text(line)}}
    assert opus.gen_provenance_problem(prov, tmp_path) is None
    (tmp_path / "g.jsonl").write_text(line.replace("SECRET", "OTHER") + "\n", encoding="utf-8")
    assert "changed" in opus.gen_provenance_problem(prov, tmp_path)
