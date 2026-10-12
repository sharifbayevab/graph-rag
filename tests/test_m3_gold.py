import pytest

from graphrag.data.normalize import to_internal
from graphrag.eval.gold import (Anchor, GoldItem, assign_splits, chunks_overlapping, find_spans, locate, norm_map,
                                remap, text_sha)
from graphrag.types import Chunk, DocRecord, Meta, Node, Source

TEXT = ("1-модда. Мақсад\nУшбу Қонуннинг мақсади «банклар» фаолиятини тартибга солиш.\n"
        "2-модда. Тушунчалар\nСанация – банкни соғломлаштириш чоралари.")


def _rec(text: str, cuts: list[int], doc_id: str = "QONUNLAR-0000aaaa") -> DocRecord:
    node = Node(doc_id, "документ", "t", None, None, 0, len(text))
    chunks = [Chunk(f"{doc_id}#{i + 1}", doc_id, i + 1, 10, f"абз. {i + 1}", text[a:b], to_internal(text[a:b], "uz"),
                    a, b, to_internal(text[a:b], "uz")) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
    return DocRecord(doc_id, Meta("law_text", None, "senate", ["uz-Cyrl"], "принят", "QONUNLAR", "t"),
                     Source("docs", f"docs/QONUNLAR/{doc_id}.docx", None, []), text, [node], chunks)


def _item(qid: str, rec: DocRecord, anchors: list[Anchor], quote: str) -> GoldItem:
    core = sorted({c for a in anchors for c in chunks_overlapping(rec, a.start, a.end)})
    return GoldItem(qid, "QONUNLAR", "easy", "?", None, "a", quote, rec.doc_id, "exact", anchors, core)


def test_norm_map_positions_point_into_the_original():
    n, pos = norm_map("A  «b»\n c’d")
    assert n == 'a "b" c\'d' and len(pos) == len(n) and pos[0] == 0 and pos[3] == 4 and pos[-1] == 10


def test_quote_found_across_spaces_newlines_and_quote_styles():
    s = TEXT.index("Қонуннинг")
    assert find_spans(TEXT, "Қонуннинг  мақсади \n“банклар”") == [(s, TEXT.index("банклар»") + len("банклар»"))]


def test_locate_tiers():
    rec = _rec(TEXT, [0, TEXT.index("2-модда"), len(TEXT)])
    assert locate(rec, "банкни соғломлаштириш")[1] == "exact"
    assert locate(rec, "бутунлай бошқа гап йўқ") == ([], "unmapped")


def test_anchor_spanning_two_chunks_gives_both_core_chunks():
    rec = _rec(TEXT, [0, TEXT.index("фаолиятини"), len(TEXT)])
    (s, e), = find_spans(TEXT, "«банклар» фаолиятини")
    assert chunks_overlapping(rec, s, e) == [f"{rec.doc_id}#1", f"{rec.doc_id}#2"]


def test_remap_after_rechunking_and_after_text_change():
    """Review Focus 3."""
    old = _rec(TEXT, [0, TEXT.index("2-модда"), len(TEXT)])
    (s, e), = find_spans(TEXT, "Санация – банкни")
    it = _item("q1", old, [Anchor(old.doc_id, s, e, text_sha(TEXT[s:e]))], "Санация – банкни")
    assert it.core == [f"{old.doc_id}#2"]
    items, problems, notes = remap([it], [_rec(TEXT, [0, 20, TEXT.index("Санация"), len(TEXT)])])
    assert problems == [] and notes == [] and items[0].core == [f"{old.doc_id}#3"]
    shifted = "Кириш. " + TEXT
    items, problems, notes = remap([it], [_rec(shifted, [0, len(shifted)])])
    assert problems == [] and notes == ["q1: anchor relocated by quote (exact)"]
    assert shifted[items[0].anchors[0].start:items[0].anchors[0].end] == TEXT[s:e]
    items, problems, _ = remap([it], [_rec("бошқа матн", [0, 10])])
    assert items == [] and problems == ["q1: anchor text changed and the quote is not found"]


def test_document_splits_group_linked_documents_and_are_deterministic():
    ids = [f"QONUNLAR-{i:08x}" for i in range(20)]
    sections = {d: "QONUNLAR" for d in ids}
    weight = {d: 10 + i for i, d in enumerate(ids)}
    pairs = [(ids[0], ids[1]), (ids[2], ids[3])]
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    a = assign_splits(sections, weight, pairs, 7, ratios)
    assert a[ids[0]] == a[ids[1]] and a[ids[2]] == a[ids[3]]
    assert set(a) == set(ids) and set(a.values()) == {"train", "val", "test"}
    assert a == assign_splits(dict(reversed(list(sections.items()))), weight, pairs, 7, ratios)
    tr = sum(weight[d] for d in ids if a[d] == "train") / sum(weight.values())
    assert 0.6 < tr < 0.8


@pytest.mark.corpus
def test_corpus_gold_is_anchored_test_only_and_splits_cover_every_document():
    import json
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.eval.gold import read_gold
    recs = {r.doc_id: r for r in read_records(config.path(config.load("data")["records_dir"]))}
    items = read_gold(config.path("data/gold/qreal.jsonl"))
    assert len(items) >= 330 and len({i.qid for i in items}) == len(items)
    for it in items:
        r = recs[it.doc_id]
        assert all(text_sha(r.text[a.start:a.end]) == a.sha256 for a in it.anchors + it.support_anchors), it.qid
        assert it.core == sorted({c for a in it.anchors for c in chunks_overlapping(r, a.start, a.end)})
        assert it.split == "qreal_test"
    splits = json.loads(config.path("data/gold/splits.json").read_text())
    assert set(splits) == set(recs) and set(splits.values()) == {"train", "val", "test"}


def test_group_spanning_sections_keeps_one_split_under_its_dominant_section():
    ids = [f"QARORLAR-{i:08x}" for i in range(12)] + ["QONUNLAR-0000ffff"]
    sections = {d: d.split("-")[0] for d in ids}
    weight = {d: 5 for d in ids}
    weight["QONUNLAR-0000ffff"] = 3
    pairs = [(ids[0], ids[-1]), (ids[1], ids[-1])]
    a = assign_splits(sections, weight, pairs, 7, {"train": 0.70, "val": 0.15, "test": 0.15})
    assert a[ids[0]] == a[ids[1]] == a[ids[-1]] and set(a) == set(ids)


def test_act_links_put_a_law_its_qaror_and_its_xulosa_in_one_split():
    """Ruling M3-R4: same-subject-act title links merge split groups; stenogram / agenda / stub links do not."""
    from graphrag.eval.gold import act_link_pairs
    law, qaror, xulosa, steno = "QONUNLAR-00000001", "QARORLAR-00000002", "XULOSALAR-00000003", "STENOGRAMMA-00000004"
    others = [f"{s}-{i:08x}" for s in ("QONUNLAR", "QARORLAR", "XULOSALAR") for i in range(16, 40)]
    sections = {d: d.split("-")[0] for d in [law, qaror, xulosa, steno] + others}
    ref = lambda s, d, rel: {"src": s, "dst": d, "type": "REFERS_TO", "attrs": {"relation": rel}}
    edges = [ref(qaror, law, "title"), ref(xulosa + "/БЛ.1", law, "title"), ref(steno + "/ВОПР.1", law, "title"),
             ref(steno, others[0], "agenda"), ref(qaror, others[1], "act"), ref(xulosa, "EXT/some law", "title"),
             {"src": law, "dst": law + "/СТ.1", "type": "CONTAINS", "attrs": {}}]
    pairs = act_link_pairs(edges, sections)
    assert pairs == sorted(tuple(sorted(p)) for p in [(law, qaror), (law, xulosa)])
    weight = {d: 5 for d in sections}
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    for seed in range(20):
        a = assign_splits(sections, weight, pairs, seed, ratios)
        assert a[law] == a[qaror] == a[xulosa]


@pytest.mark.corpus
def test_corpus_every_act_linked_pair_shares_a_split():
    import json
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.eval.gold import graph_act_link_pairs, linked_doc_pairs
    recs = read_records(config.path(config.load("data")["records_dir"]))
    splits = json.loads(config.path("data/gold/splits.json").read_text())
    act = graph_act_link_pairs({r.doc_id: r.meta.section for r in recs})
    assert len(act) >= 100
    for a, b in act + linked_doc_pairs():
        assert splits[a] == splits[b], (a, b)


def _srec(doc_id: str, title: str, typ: str = "resolution") -> DocRecord:
    sec = doc_id.split("-")[0]
    return DocRecord(doc_id, Meta(typ, None, "senate", ["uz-Cyrl"], "принят", sec, title),
                     Source("docs", f"docs/{sec}/{doc_id}.docx", None, []), "x", [], [])


def _ref(src, dst, rel, start=0):
    return {"src": src, "dst": dst, "type": "REFERS_TO", "attrs": {"relation": rel, "evidence": {"start": start}}}


def test_subject_act_pairs_group_by_shared_act_key_but_not_hubs_generic_or_stenograms():
    """Ruling M3-R5 (1): a Senate decision and a committee conclusion on the same EXT treaty share a group; a law text
    with no title edge joins via its first heading act; a conclusion with a wrong meta.title joins via its heading act;
    hub acts (> 6 referrers), generic omnibus keys and stenograms never merge."""
    from graphrag.eval.gold import strict_doc_groups, subject_act_pairs
    treaty = "EXT/eron islom respublikasi bilan shartnomani ratifikasiya qilish toʼgʼrisida"
    q, x = "QARORLAR-00000001", "XULOSALAR-00000002"
    law, qlaw, xwrong = "QONUNLAR-00000003", "QARORLAR-00000004", "XULOSALAR-00000005"
    steno = "STENOGRAMMA-00000006"
    hub_users = [f"QARORLAR-000001{i:02d}" for i in range(7)]
    recs = [_srec(q, "t1"), _srec(x, "t2"), _srec(law, "Qamoqda saqlash toʼgʼrisida qonunga qoʼshimcha", "law_text"),
            _srec(qlaw, "t4"), _srec(xwrong, "Dori vositalari toʼgʼrisida", "committee_conclusion"),
            _srec(steno, "s", "stenogram")] + [_srec(d, f"h{i}") for i, d in enumerate(hub_users)]
    edges = [_ref(q, treaty, "title"), _ref(x + "/ЗАГ.1", treaty, "act", 5),
             _ref(law + "/ПРЕАМБ.1", "EXT/jinoyat ishini yuritish chogʼida qamoqda saqlash toʼgʼrisida", "act", 3),
             _ref(qlaw, "EXT/jinoyat ishini yuritish chogʼida qamoqda saqlash toʼgʼrisida", "title"),
             _ref(xwrong, "EXT/dori vositalari toʼgʼrisida", "title"), _ref(xwrong + "/ЗАГ.1", qlaw, "act", 1),
             _ref(steno, treaty, "title"), _ref(steno + "/ВОПР.1", law, "act"),
             _ref(q, "EXT/ayrim qonun hujjatlariga oʼzgartirish kiritish toʼgʼrisida", "title"),
             _ref(q, "EXT/konstitusiya/СТ.93", "title")]
    edges += [_ref(d, "EXT/soliq kodeks", "title") for d in hub_users]
    pairs, audit = subject_act_pairs(recs, edges)
    g = strict_doc_groups(sorted(r.doc_id for r in recs), pairs)
    assert g[q] == g[x] and g[law] == g[qlaw] == g[xwrong]
    assert g[q] != g[law] and g[steno] == steno
    assert len({g[d] for d in hub_users}) == len(hub_users) and "soliq kodeks" in audit["hubs"]


def test_same_act_fuzzy_needs_the_same_amended_object_and_numbers():
    from graphrag.eval.gold import _same_act_fuzzy
    a = "sud qarorlarining qonuniyligi asosliligi va adolatliligini tekshirish instituti takomillashtirilishi munosabati bilan maʼmuriy sud ishlarini yuritish kodeks oʼzgartirish kiritish toʼgʼrisida"
    b = "sud qarorlarining qonuniyligi va asosliligini tekshirish tartibi takomillashtirilishi munosabati bilan maʼmuriy sud ishlarini yuritish kodeks oʼzgartirish kiritish toʼgʼrisida"
    c = b.replace("maʼmuriy sud ishlarini yuritish", "fuqarolik prosessual")
    assert _same_act_fuzzy(a, b, 0.9) and not _same_act_fuzzy(a, c, 0.9)
    assert not _same_act_fuzzy("2022 yil uchun davlat budjet toʼgʼrisida", "2023 yil uchun davlat budjet toʼgʼrisida", 0.9)


def test_strict_doc_groups_raises_on_unknown_documents():
    from graphrag.eval.gold import strict_doc_groups
    with pytest.raises(ValueError):
        strict_doc_groups(["A", "B"], [("A", "C")])
    with pytest.raises(ValueError):
        assign_splits({"A": "S", "B": "S"}, {"A": 1, "B": 1}, [("A", "Z")], 1, {"train": 0.7, "val": 0.15, "test": 0.15})


def test_alt_is_equiv_plus_mention_and_legacy_alt_reads_as_mention():
    rec = _rec(TEXT, [0, TEXT.index("2-модда"), len(TEXT)])
    (s, e), = find_spans(TEXT, "банкни соғломлаштириш")
    it = _item("q1", rec, [Anchor(rec.doc_id, s, e, text_sha(TEXT[s:e]))], "банкни соғломлаштириш")
    it.alt_equiv, it.alt_mention = ["D#2"], ["D#9", "D#2"]
    assert it.alt == ["D#2", "D#9"]
    d = it.to_json()
    d.pop("alt_mention")
    d.pop("alt_equiv")
    d["alt"] = ["D#5"]
    back = GoldItem.from_json(d)
    assert back.alt_mention == ["D#5"] and back.alt_equiv == [] and back.gold_answer_orig is None


REVIEWER_FAMILIES = {   # Task 10 review I1 / Ruling M3-R5 (2): same-act families that must share one split
    "3-state border treaty": ["QONUNLAR-751ddc60", "QARORLAR-270d0a44", "XULOSALAR-f40adff8"],
    "Turkic civil protection": ["QARORLAR-a3b8345a", "XULOSALAR-de32686e"],
    "ILO C95": ["QARORLAR-5e36ae04", "XULOSALAR-cf4c3f3c"],
    "Vienna convention": ["QARORLAR-3f2d299d", "XULOSALAR-e5580e91"],
    "Iran prisoner transfer": ["QARORLAR-473927eb", "XULOSALAR-921bc41a"],
    "qamoqda saqlash": ["QARORLAR-babbe5ed", "QONUNLAR-9a6c144b", "XULOSALAR-46ff44bd"],
    "tibbiy yordam": ["XULOSALAR-4bf3df74", "QONUNLAR-e8e5812a", "QARORLAR-b882d75c"],
    "ijtimoiy sugʼurta": ["QONUNLAR-2e52426e", "QONUNLAR-49a7f026", "QONUNLAR-1c0f3967"],
    "admin-court procedure code": ["QONUNLAR-cc7ed263", "XULOSALAR-e8e5efdd"],
}


@pytest.mark.corpus
def test_corpus_same_act_families_share_a_split_and_gold_review_fields():
    import json
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.eval.gold import graph_subject_act_pairs, read_gold
    recs = read_records(config.path(config.load("data")["records_dir"]))
    splits = json.loads(config.path("data/gold/splits.json").read_text())
    for name, ds in REVIEWER_FAMILIES.items():
        assert len({splits[d] for d in ds}) == 1, (name, {d: splits[d] for d in ds})
    pairs, audit = graph_subject_act_pairs(recs)
    assert len(pairs) >= 150 and {"konstitusiya", "soliq kodeks"} <= set(audit["hubs"])
    assert all(splits[a] == splits[b] for a, b in pairs)
    w = {r.doc_id: len(r.chunks) for r in recs}
    meta = json.loads(config.path("data/gold/meta.json").read_text())
    assert meta["groups"]["largest"]["weight"] <= 0.10 * sum(w.values())
    assert meta["tiers"] == {"exact": 335, "manual": 5}
    items = {i.qid: i for i in read_gold(config.path("data/gold/qreal.jsonl"))}
    assert {q for q, i in items.items() if i.gold_answer_orig} == {"val-0329", "val-0347", "val-0459"}
    assert "197⁷" in items["val-0347"].gold_answer and "197-1" in items["val-0347"].gold_answer_orig
    for it in items.values():
        assert not (set(it.alt_equiv) & set(it.alt_mention)) and not (set(it.alt) & set(it.gold))
        assert not any(c.startswith("STENOGRAMMA-") and "/ЗАГ.1#" in c and not c.startswith(it.doc_id)
                       for c in it.alt_equiv), it.qid


def test_curated_link_pairs_need_known_docs_and_a_rationale():
    from graphrag.eval.gold import curated_link_pairs
    ids = ["A-1", "B-2", "C-3"]
    assert curated_link_pairs([{"docs": ["C-3", "A-1", "B-2"], "why": "same act"}], ids) == [("A-1", "B-2"), ("A-1", "C-3")]
    for bad in ([{"docs": ["A-1"], "why": "x"}], [{"docs": ["A-1", "B-2"]}], [{"docs": ["A-1", "Z-9"], "why": "x"}]):
        with pytest.raises(ValueError):
            curated_link_pairs(bad, ids)


CURATED_FAMILIES = {   # Ruling M3-R7 (1): train.yaml split.links, pinned
    "davlat boji amendment (LC 2022-10-25)": ["QONUNLAR-00207a4c", "QARORLAR-c820a7ab", "XULOSALAR-fa8ed64a"],
    "NGO / sponsorship omnibus (LC 2025-04-15)": ["QONUNLAR-44d353fc", "XULOSALAR-5071eecf"],
    "crime-prevention omnibus (LC 2025-04-15)": ["QONUNLAR-05d09e0e", "QONUNLAR-545c1718", "QONUNLAR-b6eaa3d8"],
}


@pytest.mark.corpus
def test_corpus_curated_families_and_alt_review():
    import json
    from graphrag import config
    from graphrag.eval.gold import read_gold
    links = config.load("train")["split"]["links"]
    assert sorted(sorted(ln["docs"]) for ln in links) == sorted(sorted(ds) for ds in CURATED_FAMILIES.values())
    splits = json.loads(config.path("data/gold/splits.json").read_text())
    for name, ds in CURATED_FAMILIES.items():
        assert len({splits[d] for d in ds}) == 1, (name, {d: splits[d] for d in ds})
    review = json.loads(config.path("data/gold/alt_review.json").read_text())
    items = {i.qid: i for i in read_gold(config.path("data/gold/qreal.jsonl"))}
    for q, a in review.items():
        assert a["note"].strip() and set(a["equiv"]) <= set(items[q].alt_equiv) | set(items[q].support), q
    assert all(i.qid in review for i in items.values() if i.alt_mention)
    assert sum(bool(a["equiv"]) for a in review.values()) >= 20


ADJUDICATED_SUPPORT = {   # Ruling M3-R8 (5 keeps of 79, two annotators 30/30)
    "val-0178": "STENOGRAMMA-7f70748f/ЗАГ.1#1", "val-0270": "QARORLAR-b882d75c/П.2#1",
    "val-0324": "STENOGRAMMA-f19466c7/ВОПР.1/ВЫСТ.3#2", "val-0420": "STENOGRAMMA-435fae6d/ВОПР.1/ВЫСТ.2#3",
    "val-0453": "XULOSALAR-20ed9856/РЕКВ.1#1",
}


@pytest.mark.corpus
def test_corpus_adjudicated_support_and_context():
    import json
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.eval.gold import read_gold
    recs = {r.doc_id: r for r in read_records(config.path(config.load("data")["records_dir"]))}
    items = {i.qid: i for i in read_gold(config.path("data/gold/qreal.jsonl"))}
    dec = json.loads(config.path("data/gold/support_decisions.json").read_text())
    assert {q: i.support for q, i in items.items() if i.support} == {q: [c] for q, c in ADJUDICATED_SUPPORT.items()}
    for q, cid in ADJUDICATED_SUPPORT.items():
        (a,) = items[q].support_anchors
        assert recs[a.doc_id].text[a.start:a.end] == dec[q][cid]["evidence"]
    rejected = {(q, c) for q, v in dec.items() for c, x in v.items() if not x["keep"]}
    assert not any((q, c) in rejected for q, i in items.items() for c in i.gold)
    border = {(q, c) for q, v in dec.items() for c, x in v.items() if x["borderline"]}
    assert {(q, c) for q, i in items.items() for c in i.context} == border and len(border) == 12
    assert all(not (set(i.context) & (set(i.gold) | set(i.alt_equiv))) for i in items.values())
