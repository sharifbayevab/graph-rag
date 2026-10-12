"""M3-R32 repair loop: eligibility, repair packets, repaired pairs as new candidates, opaque reader ids."""
import json

from graphrag import config
from graphrag.dataset import opus
from tests.test_m3_r23_opus import CFG, SECTIONS, _ingest, _instr, _row, _single_packet


def _orig():
    pk, chunks = _single_packet()
    c = _ingest(pk, chunks, _row())[0][0]
    return pk, chunks, c


def test_ids_repaired_pair_and_opaque_reader_id():
    assert opus.pair_id("single-abc-r1") == opus.pair_id("single-abc") + "-r1"
    r = opus.reader_id("single-abc-r1")
    assert r.startswith("syn-") and len(r) == len(opus.pair_id("single-abc")) and "r1" not in r[-3:]
    assert opus.reader_id("single-abc") == opus.pair_id("single-abc")


def test_eligibility_borderline_without_bad_and_one_attempt():
    _, _, c = _orig()
    rid = opus.reader_id(c.seed.seed_id)
    v = lambda x, *k: {rid: {"verdict": x, "criteria": list(k), "reasons": [{"criterion": q, "note": "n"} for q in k]}}
    assert len(opus.repair_eligible([c], v("GOOD"), v("BORDERLINE", "c7"), set())) == 1
    assert len(opus.repair_eligible([c], v("BORDERLINE", "c5"), v("BORDERLINE", "c7"), set())) == 1
    assert opus.repair_eligible([c], v("BAD", "c1"), v("BORDERLINE", "c7"), set()) == []
    assert opus.repair_eligible([c], v("GOOD"), v("GOOD"), set()) == []
    assert opus.repair_eligible([c], v("GOOD"), {}, set()) == []                          # one reader pending
    assert opus.repair_eligible([c], v("GOOD"), v("BORDERLINE"), {c.seed.seed_id}) == []  # attempted before


def _repair_packet():
    pk, chunks, c = _orig()
    rid = opus.reader_id(c.seed.seed_id)
    a = {"verdict": "GOOD", "criteria": [], "reasons": []}
    b = {"verdict": "BORDERLINE", "criteria": ["c7"], "reasons": [{"criterion": "c7", "note": "awkward wording"}]}
    items = opus.repair_eligible([c], {rid: a}, {rid: b}, set())
    o = config.load("train")["dataset"]["opus"]["repair"]
    rins = opus.instructions(config.path(o["file"]), o["version"], rel=o["file"])
    rp = opus.make_repair_packets(items, {pk["packet_id"]: pk}, lambda x: json.loads(_row()), rins, _instr(), 25, 0,
                                  {}, "repair/gen")
    return rp[0], chunks, c, rins


def test_repair_packet_holds_pair_reasons_seed_and_both_instructions():
    rp, _, c, rins = _repair_packet()
    it = rp["items"][0]
    assert rp["packet_id"] == "repair-001" and rp["role"] == "repair-agent"
    assert it["original_seed_id"] == c.seed.seed_id and it["pair"]["question"] == c.question
    assert it["reasons"]["reader_2"][0]["note"] == "awkward wording" and it["seed"]["fragments"][0]["text"]
    assert rp["repair_instructions"]["digest"] == rins["digest"] == config.load("train")["dataset"]["opus"]["repair"]["digest"]
    assert "skip" in rins["text"] and "Fix ONLY the defects" in rins["text"]
    assert rp["generator_instructions"]["digest"] == _instr()["digest"]


def test_ingest_repair_makes_a_new_checked_candidate_with_repaired_from():
    rp, chunks, c, rins = _repair_packet()
    sid = rp["items"][0]["seed_id"]
    q2 = c.question.replace("нечанчи ўринда", "қайси навбатда")
    text = _row(seed_id=sid, question=q2) + "\n"
    out, rej, _ = opus.ingest_repairs(rp, text, chunks, CFG, SECTIONS, "repair/gen/repair-001.jsonl", "m",
                                      {c.seed.seed_id: c})
    assert rej == [] and len(out) == 1, rej
    n = out[0]
    assert n.seed.seed_id == c.seed.seed_id + "-r1" and opus.pair_id(n.seed.seed_id) == opus.pair_id(c.seed.seed_id) + "-r1"
    rf = n.provenance["repaired_from"]
    assert rf["seed_id"] == c.seed.seed_id and rf["repair_instructions"]["digest"] == rins["digest"]
    assert rf["verdicts"] == {"reader_1": "GOOD", "reader_2": "BORDERLINE"}
    assert n.provenance["generate"]["role"] == opus.ROLE_GEN and opus.wave_of(n) == "repair"
    # all code checks still run: a leaking repair is rejected
    bad = _row(seed_id=sid, question=c.question.replace("нечанчи ўринда", "учинчи навбатда"))
    out, rej, _ = opus.ingest_repairs(rp, bad, chunks, CFG, SECTIONS, "g", "m", {c.seed.seed_id: c})
    assert out == [] and "answer_in_question" in [r["reason"] for r in rej]
    skip = json.dumps({"seed_id": sid, "skip": True, "reason": "cannot fix"})
    out, rej, _ = opus.ingest_repairs(rp, skip, chunks, CFG, SECTIONS, "g", "m", {c.seed.seed_id: c})
    assert out == [] and [r["reason"] for r in rej] == ["skip"]


def test_reader_packets_of_a_repaired_pair_never_reveal_the_repair():
    rp, chunks, c, _ = _repair_packet()
    sid = rp["items"][0]["seed_id"]
    out, _, _ = opus.ingest_repairs(rp, _row(seed_id=sid, question=c.question.replace("нечанчи ўринда", "қайси навбатда")),
                                    chunks, CFG, SECTIONS, "g", "m", {c.seed.seed_id: c})
    frag = lambda cid, lab: {"id": lab, "act": "A", "where": "W", "text": chunks[cid].text}
    pk = opus.make_review_packets(out, frag, {"path": "r", "version": 2, "digest": "d", "text": "t"}, 40, 0, {})
    blob = json.dumps(pk, ensure_ascii=False).lower()
    for bad in ("repair", "-r1", "awkward wording", "reader_1", c.question.lower()):
        assert bad not in blob, bad
    v = {"verdict": "GOOD", "criteria": []}
    good, _, _ = opus.consensus(out, {opus.reader_id(out[0].seed.seed_id): v}, {opus.reader_id(out[0].seed.seed_id): v})
    assert good == out
