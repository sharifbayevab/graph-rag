# tests/test_m7_compare.py -- M7 Task 6: Q3 (compare) doc pairs from graph links, packet chunks, code checks, ingest
from collections import Counter

from grl.hard.compare import (KINDS, assign_langs, build_packets, check_compare, ingest_compare, packet_key,
                              pair_candidates, pick_doc_pairs, script_ok, select_chunks)
from grl.hard.templates import HardItem


def it(gold, docs):
    return HardItem("c1", "compare", "law_conclusion", "ru", "Чем отличается ...?", "...", gold, docs)


def test_needs_two_docs():
    cd = {"A#1": "A", "A#2": "A", "B#1": "B"}
    ct = {k: "текст" for k in cd}
    assert check_compare(it(["A#1", "A#2"], ["A", "B"]), cd, ct)          # one doc only -> rejected
    assert check_compare(it(["A#1", "B#1"], ["A", "B"]), cd, ct) == []
    assert check_compare(it(["A#1"] * 6, ["A", "B"]), cd, ct)             # > 5 -> rejected


def test_check_compare_reasons():
    cd = {"A#1": "A", "A#2": "A", "B#1": "B", "C#1": "C"}
    ct = {k: "текст" for k in cd}
    ct["A#1"] = "закон устанавливает порядок выдачи лицензий органами местной власти на всей территории страны"
    assert "unknown_id" in check_compare(it(["A#1", "B#9"], ["A", "B"]), cd, ct)
    assert "gold_docs" in check_compare(it(["A#1", "C#1"], ["A", "B"]), cd, ct)      # C is not in the pair
    assert "gold_docs" in check_compare(it(["A#1", "B#1", "C#1"], ["A", "B"]), cd, ct)
    assert "gold_size" in check_compare(it(["A#1"], ["A", "B"]), cd, ct)
    assert "gold_duplicate" in check_compare(it(["A#1", "A#1", "B#1"], ["A", "B"]), cd, ct)
    leak = HardItem("c2", "compare", "law_conclusion", "ru", "Почему порядок выдачи лицензий органами местной власти на всей территории?",
                    "x", ["A#1", "B#1"], ["A", "B"])
    assert "leak" in check_compare(leak, cd, ct)
    ids = HardItem("c3", "compare", "law_conclusion", "ru", "Что в B#1 и A#1?", "x", ["A#1", "B#1"], ["A", "B"])
    assert "leak" in check_compare(ids, cd, ct)


# --- pairs -------------------------------------------------------------------------------------------------------

def _doc(nid, dt, title):
    return {"node_id": nid, "kind": "document", "external": False, "doc_id": nid, "title": title, "type": "документ",
            "attrs": {"doc_type": dt}}


def _el(nid, doc):
    return {"node_id": nid, "kind": "element", "external": False, "doc_id": doc, "title": "", "type": "статья",
            "attrs": {}}


def _ent(nid):
    return {"node_id": nid, "kind": "entity", "external": False, "doc_id": None, "title": nid, "type": "organ",
            "attrs": {}}


def _e(t, a, b, w=None, rel=None):
    at = {"weight": w} if w else {}
    if rel:
        at["relation"] = rel
    return {"src": a, "dst": b, "type": t, "attrs": at}


def _toy():
    """Laws L1..L4, conclusions C1..C3, resolutions R1, R2, stenogram S1. Links: C1->L1, C2->L1, C3->L1 (REFERS_TO;
    L1 is capped at 2 pairs), R1~L2 (RELATED), S1->L3 (REFERS_TO), L2/L4 regulate ENT/x (shared), L3 twin L3b."""
    nodes = [_doc(f"L{i}", "law_text", f"Qonun {i}") for i in (1, 2, 3, 4)]
    nodes += [_doc("L3b", "law_text", "Qonun 3"), _doc("C1", "committee_conclusion", "Xulosa 1"),
              _doc("C2", "committee_conclusion", "Xulosa 2"), _doc("C3", "committee_conclusion", "Xulosa 3"),
              _doc("R1", "resolution", "Qaror 1"), _doc("R2", "resolution", "Qaror 2"),
              _doc("S1", "stenogram", "Majlis 1"), _ent("ENT/x"), _ent("ENT/y")]
    for d in ("L1", "L2", "L3", "L3b", "L4", "C1", "C2", "C3", "R1", "R2", "S1"):
        nodes += [_el(f"{d}/СТ.1", d), _el(f"{d}/СТ.2", d)]
    edges = [_e("REFERS_TO", "C1/СТ.1", "L1"), _e("REFERS_TO", "C2/СТ.2", "L1/СТ.2"), _e("REFERS_TO", "C3/СТ.1", "L1"),
             _e("RELATED", "R1/СТ.1", "L2/СТ.2", 0.8), _e("REFERS_TO", "S1/СТ.2", "L3/СТ.1", rel="act"),
             _e("REGULATES", "L2/СТ.1", "ENT/x"), _e("REGULATES", "L4/СТ.2", "ENT/x"),
             _e("REGULATES", "L3/СТ.1", "ENT/y"), _e("REGULATES", "L3b/СТ.1", "ENT/y"),
             _e("REGULATES", "R2/СТ.1", "ENT/x")]                 # a shared entity does not make law_resolution
    n_chunks = {n["node_id"]: 3 for n in nodes if n["kind"] == "document"}
    return nodes, edges, n_chunks


DATES = {"L3": "2024-01-01", "L3b": "2024-02-02"}             # the twins are named apart by date


def test_pick_doc_pairs_kinds_cap_and_twins():
    nodes, edges, nc = _toy()
    pairs = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    kinds = {k for _, _, k in pairs}
    assert kinds <= set(KINDS)
    got = {(a, b, k) for a, b, k in pairs}
    assert ("L2", "R1", "law_resolution") in got
    assert ("S1", "L3", "stenogram_law") in got
    assert ("L2", "L4", "laws_shared_entity") in got
    assert not any({a, b} == {"L3", "L3b"} for a, b, _ in pairs)          # same title -> twin, not compared
    assert not any("R2" in (a, b) for a, b, _ in pairs)                     # law_resolution needs a direct link
    lc = [p for p in pairs if p[2] == "law_conclusion"]
    assert len(lc) == 1 and lc[0][0] == "L1"                                # L1 in at most 1 pair per kind
    from collections import Counter
    assert max(Counter(d for a, b, _ in pairs for d in (a, b)).values()) <= 2


def test_pick_doc_pairs_overall_cap():
    nodes, edges, nc = _toy()
    edges += [_e("RELATED", "R2/СТ.2", "L1/СТ.1", 0.9), _e("REFERS_TO", "S1/СТ.1", "L1", rel="agenda"),
              _e("RELATED", "S1/СТ.2", "L1/СТ.2", 0.7)]                        # L1: 3 kinds eligible
    pairs = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    from collections import Counter
    per = Counter(d for a, b, _ in pairs for d in (a, b))
    assert per["L1"] == 2 and max(per.values()) <= 2
    assert max(Counter((k, d) for a, b, k in pairs for d in (a, b)).values()) == 1


def test_pick_doc_pairs_even_spread_prefix_and_determinism():
    nodes, edges, nc = _toy()
    full = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    assert pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES) == full
    four = pick_doc_pairs(nodes, edges, 4, 7, n_chunks=nc, doc_dates=DATES)
    assert four == full[:4]
    assert len({k for _, _, k in four}) == 4                                # one of each kind first


def test_pick_doc_pairs_skips_short_docs():
    nodes, edges, nc = _toy()
    nc["R1"] = 1
    assert not any("R1" in (a, b) for a, b, _ in pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES))


def test_pick_doc_pairs_drops_ambiguous_names():
    nodes, edges, nc = _toy()
    st = __import__("collections").Counter()
    pairs = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, stats=st)      # L3 / L3b: same title, no date
    assert not any("L3" in (a, b) for a, b, _ in pairs) and st["ambiguous_name_docs"] == 2


def test_select_chunks_linked_first_and_budget():
    doc_chunks = {"A": [(f"A/СТ.{i}#1", 100) for i in range(1, 11)], "B": [("B/СТ.1#1", 100), ("B/СТ.2#1", 100)]}
    mu = {f"A/СТ.{i}": [f"A/СТ.{i}#1"] for i in range(1, 11)}
    shown, linked = select_chunks({"A": {"A/СТ.9": 1.0}, "B": {}}, doc_chunks, mu, budget=600)
    assert shown["B"] == ["B/СТ.1#1", "B/СТ.2#1"]                           # short doc fully, its leftover to A
    assert "A/СТ.9#1" in shown["A"] and len(shown["A"]) == 4                # 600 - 200 = 400 tokens -> 4 chunks
    assert shown["A"] == ["A/СТ.1#1", "A/СТ.2#1", "A/СТ.3#1", "A/СТ.9#1"]   # document order; linked one kept
    assert linked["A"] == ["A/СТ.9#1"]


def test_assign_langs_per_kind_shares():
    pairs = [("a", "b", k) for _ in range(10) for k in KINDS]
    langs = assign_langs(pairs, {"ru": 3, "uz-Cyrl": 4, "uz-Latn": 3})
    from collections import Counter
    for k in KINDS:
        c = Counter(l for p, l in zip(pairs, langs) if p[2] == k)
        assert c == Counter({"uz-Cyrl": 4, "ru": 3, "uz-Latn": 3})
    assert assign_langs(pairs[:8], {"ru": 3, "uz-Cyrl": 4, "uz-Latn": 3}) == langs[:8]   # prefix-stable


# --- ingest ------------------------------------------------------------------------------------------------------

def _packets():
    p = {"packet_id": "cmp-000", "qid": "m7-compare-x0", "kind": "law_conclusion", "lang": "uz-Latn",
         "docs": [{"doc_id": "A", "chunks": [{"chunk_id": "A#1", "text": "a"}, {"chunk_id": "A#2", "text": "b"}]},
                  {"doc_id": "B", "chunks": [{"chunk_id": "B#1", "text": "c"}]}]}
    return {"cmp-000": p}


def _raw(**kw):
    r = {"packet_id": "cmp-000", "qid": "m7-compare-x0", "qtype": "compare", "template": "law_conclusion",
         "lang": "uz-Latn", "question": "Qonun va xulosa qanday farq qiladi?", "answer": "Farqi bor.",
         "gold": ["B#1", "A#1"], "docs": ["A", "B"]}
    r.update(kw)
    return r


def test_ingest_accepts_and_rejects():
    cd = {"A#1": "A", "A#2": "A", "A#3": "A", "B#1": "B"}
    ct = {k: "matn" for k in cd}
    raws = [_raw(), _raw(packet_id="cmp-001"), {"packet_id": "cmp-000", "refusal": "no real comparison"}]
    items, st, rej, _ = ingest_compare(raws, _packets(), cd, ct)
    assert [x.gold for x in items] == [["A#1", "B#1"]] and items[0].qid == "m7-compare-x0"
    assert items[0].meta["packet_id"] == "cmp-000"
    assert st["accepted"] == 1 and st["refusals"] == 1 and st["rejected:unknown_packet"] == 1
    for kw, why in ((dict(gold=["A#3", "B#1"]), "gold_not_shown"), (dict(lang="ru"), "lang_mismatch"),
                    (dict(answer=" "), "empty_answer"), (dict(template="law_resolution"), "template_mismatch"),
                    (dict(docs=["A", "C"]), "docs_mismatch"),
                    (dict(question="Закон и заключение чем отличаются?"), "script_mismatch")):
        _, st, rej, _ = ingest_compare([_raw(**kw)], _packets(), cd, ct)
        assert st["accepted"] == 0 and st[f"rejected:{why}"] == 1, (kw, st)
    alt = _raw()
    alt["meta"] = {"packet_id": alt.pop("packet_id")}                       # packet id given only in meta
    assert ingest_compare([alt], _packets(), cd, ct)[1]["accepted"] == 1
    _, st, _, _ = ingest_compare([_raw(), _raw()], _packets(), cd, ct)
    assert st["accepted"] == 1 and st["rejected:duplicate_packet"] == 1


def test_script_ok():
    assert script_ok("Qonun va xulosa qanday farq qiladi?", "uz-Latn")
    assert not script_ok("Қонун ва хулоса қандай фарқ қилади?", "uz-Latn")
    assert script_ok("Қонун ва хулоса қандай фарқ қилади?", "uz-Cyrl")
    assert not script_ok("Қонун ва хулоса қандай фарқ қилади?", "ru")       # Uzbek letters outside quotes
    assert script_ok("Чем отличается закон «Йўл ҳаракати тўғрисида» от заключения?", "ru")  # quoted title exempt
    assert not script_ok("Чем отличается закон от заключения комитета?", "uz-Cyrl")      # ы / щ / no Uzbek letter


def test_build_packets_shape():
    nodes, edges, nc = _toy()
    pairs = pick_doc_pairs(nodes, edges, 4, 7, n_chunks=nc, doc_dates=DATES)
    doc_chunks = {d: [(f"{d}/СТ.{i}#1", 50) for i in (1, 2)] + [(f"{d}#3", 50)] for d in nc}
    mu = {f"{d}/СТ.{i}": [f"{d}/СТ.{i}#1"] for d in nc for i in (1, 2)}
    texts = {c: "t " + c for d in doc_chunks for c, _ in doc_chunks[d]}
    meta = {d: {"date": "2024-01-01", "lang": ["uz-Cyrl"]} for d in nc}
    pk = build_packets(pairs, nodes, edges, doc_chunks, mu, texts, meta, {"ru": 1, "uz-Cyrl": 1, "uz-Latn": 1})
    assert [p["kind"] for p in pk] == [k for _, _, k in pairs]
    p = pk[0]
    assert p["packet_id"] == "cmp-000" and p["qid"].startswith("m7-compare-")
    assert [d["doc_id"] for d in p["docs"]] == list(pairs[0][:2])
    for d in p["docs"]:
        assert {"title", "doc_type", "date", "n_chunks_total", "chunks", "linked_chunks"} <= set(d)
        assert all(set(c) == {"chunk_id", "text", "linked"} for c in d["chunks"])
    assert p["tokens"] <= 12000 and p["lang"] in {"ru", "uz-Cyrl", "uz-Latn"}
    assert len({q["qid"] for q in pk}) == len(pk)


# --- fix round 1: R23 / R24 / R25 and minors ------------------------------------------------------------------------

def _doc_d(nid, dt, title):
    return _doc(nid, dt, title)


def _fam():
    """R23: laws V1 (2025-02-18) and V2 (2025-10-14) are versions of one law; CV (2025-02-18) and RV (2025-05-05) refer
    to both. T1 and its Russian text T2 (typed resolution) are linked by translation_of and REFERS_TO."""
    nodes = [_doc("V1", "law_text", "Qonun V"), _doc("V2", "law_text", "Qonun V (yangi)"),
             _doc("CV", "committee_conclusion", "Xulosa V"), _doc("RV", "resolution", "Qaror V"),
             _doc("T1", "law_text", "Qonun T"), _doc("T2", "resolution", "Закон Т")]
    for d in ("V1", "V2", "CV", "RV", "T1", "T2"):
        nodes += [_el(f"{d}/СТ.1", d), _el(f"{d}/СТ.2", d)]
    edges = [_e("REFERS_TO", "V1", "V2", rel="version_of"), _e("REFERS_TO", "T2", "T1", rel="translation_of"),
             _e("REFERS_TO", "T2/СТ.1", "T1", rel="act"),
             _e("REFERS_TO", "CV/СТ.1", "V1", rel="act"), _e("REFERS_TO", "CV/СТ.2", "V2", rel="act"),
             _e("REFERS_TO", "RV/СТ.1", "V2", rel="act"), _e("REFERS_TO", "RV/СТ.2", "V1", rel="act")]
    dates = {"V1": "2025-02-18", "V2": "2025-10-14", "CV": "2025-02-18", "RV": "2025-05-05"}
    return nodes, edges, {n["node_id"]: 3 for n in nodes if n["kind"] == "document"}, dates


def test_r23_translation_and_version_family():
    nodes, edges, nc, dates = _fam()
    why, st = {}, Counter()
    c = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=dates, why=why, stats=st)
    got = {(a, b, k) for k, xs in c.items() for a, b, _ in xs}
    assert got == {("V1", "CV", "law_conclusion")}                          # the version of the conclusion's date
    assert why[("T1", "T2")] == "twin_relation"                            # translation_of -> twin, dropped
    assert why[("CV", "V2")] == "version_date_mismatch"
    assert why[("RV", "V1")] == why[("RV", "V2")] == "version_date_mismatch"  # no version of its date -> dropped


def _sten():
    """R24: law L (2024-05-01). S1 (2024-06-01) has an agenda REFERS_TO plus a RELATED to L/СТ.1 -> valid; S2 is
    dated before the law; S3 has only RELATED; S4 has only the document-level agenda edge (no linked law chunk)."""
    nodes = [_doc("L", "law_text", "Qonun L")] + [_doc(f"S{i}", "stenogram", f"Majlis {i}") for i in (1, 2, 3, 4)]
    for d in ("L", "S1", "S2", "S3", "S4"):
        nodes += [_el(f"{d}/СТ.1", d), _el(f"{d}/СТ.2", d)]
    edges = [_e("REFERS_TO", "S1/СТ.1", "L", rel="agenda"), _e("RELATED", "S1/СТ.2", "L/СТ.1", 0.7),
             _e("REFERS_TO", "S2/СТ.1", "L", rel="agenda"), _e("RELATED", "S2/СТ.2", "L/СТ.1", 0.7),
             _e("RELATED", "S3/СТ.2", "L/СТ.1", 0.9),
             _e("REFERS_TO", "S4/СТ.1", "L", rel="agenda")]
    dates = {"L": "2024-05-01", "S1": "2024-06-01", "S2": "2023-01-01", "S3": "2024-06-01", "S4": "2024-06-01"}
    return nodes, edges, {n["node_id"]: 3 for n in nodes if n["kind"] == "document"}, dates


def test_r24_stenogram_rules():
    nodes, edges, nc, dates = _sten()
    why = {}
    c = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=dates, why=why)
    assert [(a, b) for a, b, _ in c["stenogram_law"]] == [("S1", "L")]
    assert why[("L", "S2")] == "sten_before_law"
    assert why[("L", "S3")] == "sten_no_refers"                             # RELATED alone is not "discussed"
    assert why[("L", "S4")] == "sten_law_unlinked"
    why2 = {}                                                               # with μ⁻¹: the linked element owns no chunk
    c2 = pair_candidates(nodes, edges, 7, n_chunks=nc, doc_dates=dates, why=why2, mu_inv={"L/СТ.2": ["L/СТ.2#1"]})
    assert c2["stenogram_law"] == [] and why2[("L", "S1")] == "sten_law_unlinked"


def test_postanovlenie_is_resolution():
    nodes = [_doc("L", "law_text", "Qonun"), _doc("P", "postanovlenie", "Qaror"), _el("P/П.1", "P"), _el("L/СТ.1", "L")]
    edges = [_e("REFERS_TO", "P/П.1", "L", rel="act")]
    c = pair_candidates(nodes, edges, 7)
    assert [(a, b) for a, b, _ in c["law_resolution"]] == [("L", "P")]


def test_r25_topup_preset_exclude_quota():
    nodes, edges, nc = _toy()
    full = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES)
    kept, lost = full[1:], full[:1]                                         # the law_conclusion pair became invalid
    assert lost[0][2] == "law_conclusion" and lost[0][0] == "L1"
    q = {k: 0 for k in KINDS} | {"law_conclusion": 1}
    new = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, preset=kept, exclude=full, quota=q)
    assert len(new) == 1 and new[0][0] == "L1" and new[0][2] == "law_conclusion" and new[0] not in full
    assert pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, preset=full, exclude=full) == []
    both = pick_doc_pairs(nodes, edges, 20, 7, n_chunks=nc, doc_dates=DATES, preset=full[:1], exclude=full)
    assert ("L1", "C1", "law_conclusion") not in both and ("L1", "C2", "law_conclusion") not in both   # per-kind cap


def test_r25_packets_start_prior_and_pair_key():
    nodes, edges, nc = _toy()
    pairs = pick_doc_pairs(nodes, edges, 2, 7, n_chunks=nc, doc_dates=DATES)
    doc_chunks = {d: [(f"{d}/СТ.{i}#1", 50) for i in (1, 2)] for d in nc}
    mu = {f"{d}/СТ.{i}": [f"{d}/СТ.{i}#1"] for d in nc for i in (1, 2)}
    texts = {c: "t" for d in doc_chunks for c, _ in doc_chunks[d]}
    sh = {"ru": 1, "uz-Cyrl": 1, "uz-Latn": 1}
    pk = build_packets(pairs, nodes, edges, doc_chunks, mu, texts, {}, sh, start=40,
                       prior=[(pairs[0][2], "ru")])
    assert [p["packet_id"] for p in pk] == ["cmp-040", "cmp-041"]
    assert pk[0]["lang"] != "ru"                                            # continues the kind's running counts
    assert [tuple(p["pair"]) for p in pk] == pairs and [packet_key(p) for p in pk] == pairs
    old = {k: v for k, v in pk[0].items() if k != "pair"}                  # a cmp-000..039 packet: no `pair` field
    assert packet_key(old) == pairs[0]


def test_ingest_pair_invalid_order_malformed_and_both():
    cd = {"A#1": "A", "A#2": "A", "B#1": "B"}
    ct = {k: "matn" for k in cd}
    pk = _packets()
    p1 = {**pk["cmp-000"], "packet_id": "cmp-001", "qid": "m7-compare-x1"}
    pk["cmp-001"] = p1
    raws = [_raw(packet_id="cmp-001"), _raw()]
    items, st, _, _ = ingest_compare(raws, pk, cd, ct)
    assert [x.meta["packet_id"] for x in items] == ["cmp-000", "cmp-001"]   # packet order, not file order
    items, st, _, _ = ingest_compare(raws, pk, cd, ct, valid=[["A", "B", "law_conclusion"]])
    assert st["accepted"] == 2
    items, st, _, _ = ingest_compare(raws, pk, cd, ct, valid=[])
    assert st["accepted"] == 0 and st["rejected:pair_invalid"] == 2
    bad = [_raw(docs=[None, "A"]), _raw(packet_id="cmp-001", gold=None), "not a dict", None]
    items, st, rej, _ = ingest_compare(bad, pk, cd, ct)                     # never aborts
    assert st["accepted"] == 0 and st["rejected"] == 4
    assert st["rejected:docs_mismatch"] == 1 and st["rejected:malformed_gold"] == 1 and st["rejected:malformed"] == 2
    _, st, _, both = ingest_compare([_raw(), {"meta": {"packet_id": "cmp-000"}, "refusal": "x"}], pk, cd, ct)
    assert both == ["cmp-000"] and st["refusal_and_item"] == 1 and st["refusals:law_conclusion"] == 1
    assert st["packets_answered"] == 1


def test_script_ok_uz_cyrl_letters_outside_quotes():
    assert not script_ok("Чем отличается закон «Йўл ҳаракати тўғрисида» от заключения?", "uz-Cyrl")
    assert script_ok("«Йўл ҳаракати тўғрисида»ги қонун ва хулоса қандай фарқ қилади?", "uz-Cyrl")


# --- fix round 2: ingest CLI (atomic outputs, globs, R27) ----------------------------------------------------------

def test_cli_ingest_outputs_non_object_lines_and_atomic(tmp_path):
    import json

    import pytest
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    cd = {"A#1": "A", "A#2": "A", "B#1": "B"}
    ct = {k: "matn" for k in cd}
    (tmp_path / "ids.json").write_text('["m7-count-1"]\n', encoding="utf-8")
    raws = [_raw(), "not json", None, 7, ["x"]]
    files, summary, kept, _ = hp.ingest_outputs(raws, _packets(), cd, ct, [["A", "B", "law_conclusion"]], tmp_path,
                                                30)
    assert len(kept) == 1 and summary["stats"]["rejected:malformed"] == 4
    rej = [json.loads(x) for x in files[tmp_path / "compare_rejections.jsonl"].splitlines()]
    assert [r["packet_id"] for r in rej] == [None] * 4
    assert not (tmp_path / "compare.jsonl").exists()                        # nothing written before atomic_write
    hp.atomic_write(files)
    assert json.loads((tmp_path / "ids.json").read_text()) == ["m7-count-1", "m7-compare-x0"]
    assert not list(tmp_path.glob("*.tmp"))
    bad = {tmp_path / "compare.jsonl": "new\n", tmp_path / "missing_dir" / "x.json": "y"}
    before = (tmp_path / "compare.jsonl").read_text()
    with pytest.raises(FileNotFoundError):
        hp.atomic_write(bad)
    assert (tmp_path / "compare.jsonl").read_text() == before and not list(tmp_path.glob("*.tmp"))


def test_cli_raw_glob_without_match_and_missing_valid(tmp_path):
    import pytest
    from tests.scripts_import import load_script
    hp = load_script("hard_probe")
    with pytest.raises(SystemExit, match="no file matches"):
        hp._raw_lines([str(tmp_path / "compare_raw_*.jsonl")])
    (tmp_path / "r1.jsonl").write_text('{"packet_id": "cmp-000"}\nnot json\n', encoding="utf-8")
    assert hp._raw_lines([str(tmp_path / "r*.jsonl")]) == [{"packet_id": "cmp-000"}, "not json"]
    with pytest.raises(SystemExit, match="R27"):
        hp.load_valid(str(tmp_path / "compare_valid_pairs.json"))
