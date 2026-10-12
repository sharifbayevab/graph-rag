from collections import Counter

import pytest

from graphrag.dataset.seeds import Seed, allocate, cell_counts, sample_seeds
from graphrag.dataset.sources import SourceIndex, segment_text
from graphrag.dataset.walks import WALK_EDGES, classify
from graphrag.types import DocRecord, Meta, Node, Source


def _rec(doc_id, title, date, nodes):
    return DocRecord(doc_id, Meta("law_text", date, "senate", ["uz-Cyrl"], "принят", "QONUNLAR", title),
                     Source("docs", f"docs/{doc_id}.docx", None, []), "x" * 100,
                     [Node(doc_id, "документ", title, None, None, 0, 100)] +
                     [Node(n, "статья", "", None, doc_id, 0, 100) for n in nodes], [])


def test_segment_text_and_source_index_bijection():
    assert segment_text("СТ.122.1") == "ст. 122.1" and segment_text("ПП.a") == "пп. a" and segment_text("§.1") == "§ 1"
    with pytest.raises(ValueError):
        segment_text("ZZ.1")
    a = _rec("QONUNLAR-a", "Банклар тўғрисида", "2019-01-01", ["QONUNLAR-a/СТ.21", "QONUNLAR-a/СТ.21/Ч.2"])
    b = _rec("QONUNLAR-b", "Банклар тўғрисида", "2023-05-05", ["QONUNLAR-b/СТ.21"])
    ix = SourceIndex([a, b])
    assert ix.format("QONUNLAR-a/СТ.21/Ч.2") == "Банклар тўғрисида (2019-01-01), ст. 21 ч. 2"
    for n in ("QONUNLAR-a/СТ.21", "QONUNLAR-a/СТ.21/Ч.2", "QONUNLAR-b/СТ.21", "QONUNLAR-a"):
        assert ix.resolve(ix.format(n)) == n


def test_walk_edges_are_the_verified_types_and_patterns_follow_the_edges():
    assert WALK_EDGES == ("CONTAINS", "REFERS_TO", "AMENDS", "REPEALS", "DEFINES")
    doc = lambda n: n.split("/", 1)[0]
    assert classify((("A/СТ.1", "REFERS_TO", "A/СТ.5"),), doc) == "refers_internal"
    assert classify((("A/СТ.1", "REFERS_TO", "B/СТ.2"),), doc) == "refers_cross"
    assert classify((("A/СТ.1/Ч.1", "AMENDS", "EXT/k/СТ.9"), ("C/СТ.3", "REFERS_TO", "EXT/k/СТ.9")), doc) == "amends"
    assert classify((("A/СТ.1", "REPEALS", "EXT/k"),), doc) == "amends"
    assert classify((("A", "CONTAINS", "A/СТ.1"), ("A/СТ.1", "DEFINES", "CON/x")), doc) is None


def test_cells_are_exact_per_split_pattern_and_language_and_shortage_is_reported():
    assert cell_counts(7, {"train": 0.7, "val": 0.15, "test": 0.15}) == {"train": 5, "val": 1, "test": 1}
    seq = allocate(10, {"uz-Cyrl": 0.45, "uz-Latn": 0.25, "ru": 0.30})
    assert Counter(seq) == {"uz-Cyrl": 5, "uz-Latn": 2, "ru": 3}
    splits = {f"D{i}": ("train" if i < 8 else "test") for i in range(10)}
    sections = {d: "QONUNLAR" for d in splits}
    pool = {"single": [((f"D{i}#{j}",), (f"D{i}",), ()) for i in range(10) for j in range(10)],
            "amends": [((f"D{i}#50", f"D{i}#51"), (f"D{i}",), ((f"D{i}/СТ.1", "AMENDS", "EXT/k/СТ.9"),)) for i in range(8)],
            "refers_internal": [], "refers_cross": []}
    quotas = {"single": 20, "amends": 6, "refers_internal": 0, "refers_cross": 0}
    ratios = {"train": 0.5, "test": 0.5}
    langs = {"ru": 0.5, "uz-Cyrl": 0.5}
    s, short = sample_seeds(pool, splits, sections, quotas, ratios, langs, {"QONUNLAR": 1.0}, 99, 7)
    got = Counter((x.pattern, x.split, x.lang) for x in s)
    assert got[("single", "train", "ru")] == 5 and got[("single", "test", "uz-Cyrl")] == 5
    assert short == {"amends|test|ru": 2, "amends|test|uz-Cyrl": 1}                    # no backfill from train
    assert all(x.walk and x.walk[0][1] == "AMENDS" for x in s if x.pattern == "amends")
    assert s == sample_seeds(pool, splits, sections, quotas, ratios, langs, {"QONUNLAR": 1.0}, 99, 7)[0]
    assert Seed.from_json(s[0].to_json()) == s[0]


def test_forced_seeds_come_first_and_prefer_orders_singles():
    splits = {"D1": "train"}
    pool = {"single": [((f"D1#{j}",), ("D1",), ()) for j in range(10)]}
    q = {"single": 3, "refers_internal": 0, "refers_cross": 0, "amends": 0}
    first, _ = sample_seeds(pool, splits, {"D1": "QONUNLAR"}, q, {"train": 1.0}, {"ru": 1.0}, {"QONUNLAR": 1.0}, 99, 7,
                            prefer=lambda ch: ch[0] in ("D1#8", "D1#9"))
    assert {"D1#8", "D1#9"} <= {x.chunks[0] for x in first}
    again, _ = sample_seeds(pool, splits, {"D1": "QONUNLAR"}, dict(q, single=5), {"train": 1.0}, {"ru": 1.0},
                            {"QONUNLAR": 1.0}, 99, 7, forced=[first[0]])
    assert first[0] in again and len(again) == 5


@pytest.mark.corpus
def test_corpus_seeds_are_grounded_split_safe_and_exclude_qreal():
    import json
    from graphrag import config
    from graphrag.data.ingest import read_records
    from graphrag.dataset.seeds import qreal_exclusion
    from graphrag.eval.gold import linked_doc_pairs, read_gold
    recs = read_records(config.path(config.load("data")["records_dir"]))
    chunks = {c.chunk_id for r in recs for c in r.chunks}
    ex = qreal_exclusion(read_gold(config.path("data/gold/qreal.jsonl")), recs, linked_doc_pairs())
    splits = json.loads(config.path("data/gold/splits.json").read_text())
    seeds = [Seed.from_json(json.loads(l)) for l in config.path("data/dataset/seeds.jsonl").read_text().splitlines()]
    for s in seeds:
        assert set(s.chunks) <= chunks and not (set(s.chunks) & ex) and {splits[d] for d in s.docs} == {s.split}
        if s.pattern != "single":
            assert 1 <= len(s.walk) <= 3 and all(t in WALK_EDGES for _, t, _ in s.walk) and 2 <= len(s.chunks) <= 4
            assert classify(s.walk, lambda n: n.split("/", 1)[0]) is not None
            if s.pattern == "multi":                      # M3-R20: one multi pattern of bridge / aggregation walks
                from graphrag.dataset.walks import walk_class
                assert walk_class(s.walk) in ("bridge", "aggregation")
            else:
                assert classify(s.walk, lambda n: n.split("/", 1)[0]) == s.pattern


def _walk_toy():
    """A/СТ.1 -REFERS_TO-> EXT/k <-REFERS_TO- {A/СТ.5, A/СТ.6, B/СТ.2}; A/СТ.1 -REFERS_TO-> A/СТ.6 (internal);
    one chunk per article, chunk = whole article span."""
    import numpy as np
    from graphrag.env.data import EnvData
    from graphrag.kg.schema import GraphEdge
    from graphrag.types import Chunk
    arts = {"A": ["A/СТ.1", "A/СТ.5", "A/СТ.6"], "B": ["B/СТ.2"]}
    recs = []
    for d, ns in arts.items():
        r = _rec(d, f"Акт {d}", "2020-01-01", ns)
        r.structure = [r.structure[0]] + [type(r.structure[0])(n, "статья", "", None, d, 10 * i, 10 * i + 10)
                                          for i, n in enumerate(ns)]
        r.chunks = [Chunk(f"{n}#1", n, 1, 40, "", "t", "t", 10 * i, 10 * i + 10) for i, n in enumerate(ns)]
        recs.append(r)
    ev = lambda d, i: {"evidence": {"doc_id": d, "start": 10 * i + 1, "end": 10 * i + 3}}
    edges = [GraphEdge("A/СТ.1", "EXT/k", "REFERS_TO", ev("A", 0)), GraphEdge("A/СТ.5", "EXT/k", "REFERS_TO", ev("A", 1)),
             GraphEdge("A/СТ.6", "EXT/k", "REFERS_TO", ev("A", 2)), GraphEdge("B/СТ.2", "EXT/k", "REFERS_TO", ev("B", 0))]
    edges += [GraphEdge("A", n, "CONTAINS") for n in arts["A"]] + [GraphEdge("B", "B/СТ.2", "CONTAINS")]
    nodes = sorted(["A", "B", "EXT/k", *arts["A"], *arts["B"]])
    chunks = sorted(c.chunk_id for r in recs for c in r.chunks)
    mu = {n: [f"{n}#1"] for ns in arts.values() for n in ns}
    mu.update({d: [f"{n}#1" for n in ns] for d, ns in arts.items()})
    vecs = np.eye(4, dtype=np.float32)
    data = EnvData.build(nodes, chunks, [(e.src, e.type, e.dst) for e in edges], {c: c.split("#")[0] for c in chunks},
                         mu, {c: 40 for c in chunks}, vecs, np.zeros((len(nodes), 4), np.float32),
                         ["CONTAINS", "REFERS_TO", "AMENDS", "REPEALS", "REGULATES", "DEFINES", "RELATED"])
    return data, recs, edges, set(chunks)


def test_walk_is_a_seeded_uniform_pick_among_all_valid_walks_m3_r11():
    from graphrag.dataset.walks import enumerate_walks
    data, recs, edges, ok = _walk_toy()
    anchor = ("A/СТ.1", "REFERS_TO", "EXT/k")
    picks = set()
    for seed in range(40):
        ws = enumerate_walks(data, recs, edges, ok, seed=seed)
        w = next(w for p in ws.values() for w in p if w.hops[0] == anchor)
        assert w.nodes[0] == "A/СТ.1" and w.chunks[0] == "A/СТ.1#1" and 2 <= len(w.chunks) <= 4
        picks.add(w.hops)
        assert ws == enumerate_walks(data, recs, edges, ok, seed=seed)                 # deterministic
    assert len(picks) >= 3                                                           # not always the first valid walk
    assert all(len(h) >= 2 for h in picks)                                           # EXT/k is never an end
    assert any(len(h) == 3 for h in picks)          # first-valid BFS would never pass a valid 2-hop walk for a 3-hop one


def test_walks_never_cross_splits_and_contains_passes_bear_no_fact():
    from graphrag.dataset.walks import enumerate_walks
    data, recs, edges, ok = _walk_toy()
    split_of = {"A": "train", "B": "test"}
    for seed in range(20):
        st = {}
        for ws in enumerate_walks(data, recs, edges, ok, seed=seed, split_of=split_of, stats=st).values():
            for w in ws:
                assert {split_of[n.split("/", 1)[0]] for n in w.nodes if n.split("/", 1)[0] in split_of} == {"train"} \
                    or {split_of[n.split("/", 1)[0]] for n in w.nodes if n.split("/", 1)[0] in split_of} == {"test"}
                for i, v in enumerate(w.nodes[1:-1], start=1):
                    if w.hops[i - 1][1] == "CONTAINS" and w.hops[i][1] == "CONTAINS":
                        assert f"{v}#1" not in w.chunks
        assert st["anchor_edges"] == 4
    assert all(w.docs == ("B",) or "B" not in w.docs
               for ws in enumerate_walks(data, recs, edges, ok, split_of=split_of).values() for w in ws)


def test_qreal_exclusion_covers_context_chunks_m3_r9():
    from graphrag.dataset.seeds import qreal_exclusion
    from graphrag.eval.gold import GoldItem
    from graphrag.types import Chunk
    a = _rec("QONUNLAR-a", "Акт", "2019-01-01", ["QONUNLAR-a/СТ.1", "QONUNLAR-a/СТ.2"])
    a.chunks = [Chunk("QONUNLAR-a/СТ.1#1", "QONUNLAR-a/СТ.1", 1, 40, "", "t", "t", 0, 10),
                Chunk("QONUNLAR-a/СТ.2#1", "QONUNLAR-a/СТ.2", 1, 40, "", "t", "t", 10, 20),
                Chunk("QONUNLAR-a#1", "QONUNLAR-a", 1, 40, "", "t", "t", 20, 30)]
    it = GoldItem("q1", "QONUNLAR", "easy", "?", None, "ans", "quote", "QONUNLAR-a", "A", [], ["QONUNLAR-a/СТ.1#1"],
                  context=["QONUNLAR-a#1"])
    assert qreal_exclusion([it], [a], []) == {"QONUNLAR-a/СТ.1#1", "QONUNLAR-a#1"}


def test_m3_r12_overlap_rule_same_act_via_ext_and_round_robin_languages():
    from graphrag.dataset.seeds import _overlap_ok
    doc = lambda n: n.split("/", 1)[0]
    assert classify((("A/СТ.1", "REFERS_TO", "EXT/k"), ("A/СТ.9", "REFERS_TO", "EXT/k")), doc) == "refers_internal"
    assert classify((("A/СТ.1", "REFERS_TO", "EXT/k"), ("B/СТ.9", "REFERS_TO", "EXT/k")), doc) == "refers_cross"
    held = {"x": [{"x", "y"}], "y": [{"x", "y"}], "s": [{"s"}]}
    assert _overlap_ok(("x", "z"), held)                     # one shared fragment
    assert not _overlap_ok(("x", "y", "z"), held)            # two shared = another seed's full C_ref
    assert not _overlap_ok(("x",), held) and not _overlap_ok(("s", "q"), held)
    splits = {"D1": "train"}
    pool = {"refers_internal": [((f"D1/СТ.{j}#1", "D1/СТ.0#1"), ("D1",), (("D1/СТ.0", "REFERS_TO", f"D1/СТ.{j}"),))
                                for j in range(1, 7)]}
    q = {"single": 0, "refers_internal": 6, "refers_cross": 0, "amends": 0}
    s, short = sample_seeds(pool, splits, {"D1": "QONUNLAR"}, q, {"train": 1.0}, {"ru": 0.5, "uz-Cyrl": 0.5},
                            {"QONUNLAR": 1.0}, 99, 7)
    assert short == {} and Counter(x.lang for x in s) == {"ru": 3, "uz-Cyrl": 3}


def test_m3_r14_hub_bridge_chunk_cap_and_need_proportional_languages():
    from graphrag.dataset.walks import enumerate_walks, hub_vertices, is_act_level
    from graphrag.kg.schema import GraphEdge
    assert is_act_level("EXT/k", {"A"}) and not is_act_level("EXT/k/СТ.9", {"A"}) and is_act_level("A", {"A"})
    edges = [GraphEdge(f"D{i}/СТ.1", "EXT/k", "REFERS_TO") for i in range(8)] + [GraphEdge("D0/СТ.2", "EXT/m/СТ.1", "AMENDS")]
    hubs, audit = hub_vertices(edges, {f"D{i}" for i in range(8)}, 6, 0.0)
    assert hubs == {"EXT/k"} and audit["by_referrers"] == 1                    # 8 referrer docs > 6; EXT/m has 1
    data, recs, tedges, ok = _walk_toy()                                        # A/СТ.1 -> EXT/k <- ... : EXT/k bridges
    for seed in range(10):
        ws = enumerate_walks(data, recs, tedges, ok, seed=seed, hubs={"EXT/k"})
        assert all("EXT/k" not in w.nodes[1:-1] for p in ws.values() for w in p)
    splits = {"D1": "train"}
    pool = {"refers_internal": [((f"D1/СТ.{j}#1", "D1/СТ.0#1"), ("D1",), (("D1/СТ.0", "REFERS_TO", f"D1/СТ.{j}"),))
                                for j in range(1, 9)]}
    q = {"single": 0, "refers_internal": 8, "refers_cross": 0, "amends": 0}
    s, short = sample_seeds(pool, splits, {"D1": "QONUNLAR"}, q, {"train": 1.0}, {"ru": 0.75, "uz-Cyrl": 0.25},
                            {"QONUNLAR": 1.0}, 99, 7, max_per_chunk=3)
    assert len(s) == 3 and Counter(x.lang for x in s) == {"ru": 2, "uz-Cyrl": 1}   # D1/СТ.0#1 capped at 3 seeds
    assert short == {"refers_internal|train|ru": 4, "refers_internal|train|uz-Cyrl": 1}   # need 6/2: 2/6 vs 1/2 filled


def test_m3_r14_fact_bearing_chunks_are_content_chunks():
    from graphrag.dataset.seeds import content_chunks
    from graphrag.types import Chunk, Node
    body = ("Qonun bilan tadbirkorlik subyektlariga soliq imtiyozlari berish tartibi belgilanmoqda, jumladan kichik "
            "korxonalar uchun daromad soligʼi stavkasi uch yil davomida ikki baravar kamaytiriladi va hisobot "
            "topshirish muddati choraklik tartibdan yillik tartibga oʼtkaziladi hamda elektron shaklda amalga oshiriladi.")
    r = _rec("STENOGRAMMA-x", "Majlis", "2024-01-01", [])
    r.meta.section = "STENOGRAMMA"
    r.structure = [r.structure[0], Node("STENOGRAMMA-x/ВОПР.1", "вопрос", "", "1", "STENOGRAMMA-x", 0, 900),
                   Node("STENOGRAMMA-x/РЕКВ.1", "реквизиты", "", "1", "STENOGRAMMA-x", 900, 950)]
    mk = lambda cid, p, t: Chunk(cid, p, 1, 40, "", t, t, 0, len(t))
    r.chunks = [mk("STENOGRAMMA-x/ВОПР.1#1", "STENOGRAMMA-x/ВОПР.1", body),
                mk("STENOGRAMMA-x/ВОПР.1#2", "STENOGRAMMA-x/ВОПР.1", "Raislik qiluvchi: Rahmat. Marhamat, savollar boʼlsa."),
                mk("STENOGRAMMA-x/РЕКВ.1#1", "STENOGRAMMA-x/РЕКВ.1", body)]
    assert content_chunks([r]) == {"STENOGRAMMA-x/ВОПР.1#1"}


def test_m3_r15_amends_shortfall_moves_to_refers_internal_of_the_same_split():
    from graphrag.dataset.seeds import effective_needs
    final = {"amends|test|ru": 9, "amends|test|uz-Cyrl": 14, "refers_internal|test|ru": 4, "refers_internal|test|uz-Cyrl": 7,
             "amends|train|ru": 5, "refers_internal|train|ru": 3}
    got = {"amends|test|ru": 8, "amends|test|uz-Cyrl": 10, "refers_internal|test|ru": 20, "refers_internal|test|uz-Cyrl": 8,
           "amends|train|ru": 4, "refers_internal|train|ru": 9}
    eff, transfer, below = effective_needs(final, got, ("test",))
    assert eff["amends|test|ru"] == 8 and eff["amends|test|uz-Cyrl"] == 10          # amends needs what it got
    # uz-Cyrl moves 4 but refers_internal uz-Cyrl has room for 1 only (8 - 7): 3 spill to ru's surplus
    assert transfer == {"test|ru": 1 + 3, "test|uz-Cyrl": 1}
    assert eff["refers_internal|test|ru"] == 8 and eff["refers_internal|test|uz-Cyrl"] == 8
    assert sum(eff[k] for k in eff if "|test|" in k) == sum(final[k] for k in final if "|test|" in k)   # 45 % kept
    assert below == {"amends|train|ru": 1}                                            # train: no transfer
    eff2, _, below2 = effective_needs(final, dict(got, **{"refers_internal|test|ru": 5}), ("test",))
    assert below2 == {"amends|train|ru": 1, "refers_internal|test|ru": 1, "refers_internal|test|uz-Cyrl": 2}   # -> stop
    assert sum(eff2[k] for k in eff2 if "|test|" in k) == sum(final[k] for k in final if "|test|" in k)
