"""M7 Task 6c (b): the five Q1-expansion count templates (grl.hard.q1x) on a toy graph."""
from collections import Counter

import pytest

from grl.env.structure import build_article_struct_texts, build_struct_texts
from grl.hard.paraphrase import anchors, asks_times, detect_lang
from grl.hard.q1x import Q1X_TEMPLATES, make_q1x, q1x_candidates, q1x_items
from grl.hard.templates import _check_langs, _numbers, render


def toks(s):
    return len(s.split())


def _n(i, typ, num="", kind="element", title="", doc_type=None):
    ext = i.startswith("EXT/")
    doc = "/".join(i.split("/")[:2]) if ext else i.split("/")[0]
    attrs = {"number": num} if num else {}
    if doc_type:
        attrs["doc_type"] = doc_type
    return {"node_id": i, "kind": kind, "type": typ, "external": ext, "title": title,
            "doc_id": None if ext else doc, "attrs": attrs}


A = "QONUNLAR-a"                      # a law with articles
Z1, Z2, Z3 = "QONUNLAR-z1", "QONUNLAR-z2", "QONUNLAR-z3"
JK, SK, AB = "EXT/jinoyat kodeks", "EXT/soliq kodeks", "EXT/aholi bandlig"
NODES = [_n(A, "документ", kind="document", title="Mehnat muhofazasi toʼgʼrisida", doc_type="law_text"),
         *[_n(f"{A}/СТ.{k}", "статья", str(k)) for k in (1, 2, 3)],
         *[_n(f"{A}/СТ.1/Ч.{p}", "часть", str(p)) for p in (1, 2, 3)],
         _n(f"{A}/СТ.1/Ч.1/П.1", "пункт", "1"), _n(f"{A}/СТ.1/Ч.1/П.2", "пункт", "2"),
         _n(f"{A}/СТ.2/Ч.1", "часть", "1"),
         _n(f"{A}/СТ.3/Ч.1", "часть", "1"), _n(f"{A}/СТ.3/Ч.2", "часть", "2"),
         _n(f"{A}/СТ.3/Ч.1/П.1", "пункт", "1"), _n(f"{A}/СТ.3/Ч.2/П.1", "пункт", "1"),
         _n(f"{A}/СТ.4", "статья", "4"), _n(f"{A}/СТ.4/Ч.1", "часть", "1"), _n(f"{A}/СТ.4/Ч.2", "часть", "2"),
         _n(JK, "документ", kind="document", title="Jinoyat kodeksi"),
         _n(SK, "документ", kind="document", title="Soliq kodeksi"),
         _n(AB, "документ", kind="document", title="Aholi bandligi toʼgʼrisida"),
         *[_n(f"{JK}/СТ.{k}", "статья", str(k)) for k in (5, 6, 7)],
         _n(f"{SK}/СТ.9", "статья", "9"),
         *[_n(f"{AB}/СТ.{k}", "статья", str(k)) for k in (3, 4, 8)]]
for z in (Z1, Z2, Z3):
    NODES += [_n(z, "документ", kind="document", title=f"Qonun {z[-2:]}", doc_type="law_text"),
              _n(f"{z}/СТ.1", "статья", "1"), _n(f"{z}/СТ.1/Ч.1", "часть", "1"),
              *[_n(f"{z}/СТ.1/Ч.1/П.{k}", "пункт", str(k)) for k in (1, 2, 3, 4)]]
EDGES = []
for n in NODES:
    i = n["node_id"]
    if n["kind"] == "element":
        EDGES.append({"src": i.rsplit("/", 1)[0], "dst": i, "type": "CONTAINS", "attrs": {}})


def amend(z, k, dst, date, op, quote=None):
    ev = {"doc_id": z, "start": 100 * k, "end": 100 * k + 50, "quote": quote} if quote else {}
    EDGES.append({"src": f"{z}/СТ.1/Ч.1/П.{k}", "dst": dst, "type": "AMENDS",
                  "attrs": {"date": date, "op": op, **({"evidence": ev} if ev else {})}})


amend(Z1, 1, f"{JK}/СТ.5", "2024-03-01", "edit")
amend(Z1, 2, f"{SK}/СТ.9", "2024-03-01", "replace")
amend(Z1, 3, f"{AB}/СТ.3", "2024-03-01", "delete", "3-модда чиқариб ташлансин")
amend(Z1, 4, f"{AB}/СТ.4", "2024-03-01", "delete", "4-модда чиқариб ташлансин")
amend(Z2, 1, f"{JK}/СТ.6", "2024-05-05", "add")
amend(Z2, 2, f"{AB}/СТ.8", "2024-05-05", "edit")
amend(Z3, 1, f"{JK}/СТ.7", "2023-02-02", "edit")
amend(Z3, 2, f"{AB}/СТ.8", "2023-02-02", "delete", "номидаги «ва жавобгарлиги» деган сўзлар чиқариб ташлансин")
amend(Z1, 4, f"{AB}/СТ.8", "2024-03-01", "delete", "биринчи хатбошисидаги «x» деган сўзлар чиқариб ташлансин")
CHUNKS = {f"{z}/СТ.1/Ч.1/П.{k}#1": (z, 100 * k, 100 * k + 60) for z in (Z1, Z2, Z3) for k in (1, 2, 3, 4)}
CHUNK_TEXTS = {c: f"{c.split('П.')[1][0]}-модда чиқариб ташлансин ва бошқалар" for c in CHUNKS}
MU = {c.split("#")[0]: [c] for c in CHUNKS}


def inputs(drop_line=None):
    texts = build_struct_texts(NODES, EDGES, toks)
    if drop_line:
        texts = {k: "\n".join(x for x in v.split("\n") if not x.startswith(drop_line)) for k, v in texts.items()}
    return dict(struct_texts=texts, art_texts=build_article_struct_texts(NODES, EDGES, toks), mu_inv=MU,
                chunk_texts=CHUNK_TEXTS, chunk_spans=CHUNKS)


def by_template(cs):
    out = {}
    for c in cs:
        out.setdefault(c.template, []).append(c)
    return out


def test_candidates_and_answers():
    st = Counter()
    cs = by_template(q1x_candidates(NODES, EDGES, stats=st, **inputs()))
    parts = {c.meta["article"]: c for c in cs["count_parts_article"]}
    assert sorted(parts) == ["4"] and parts["4"].answer == "2"     # СТ.2: 1 part; СТ.1 / СТ.3: points (I1)
    assert parts["4"].gold == [f"STRUCT/{A}/СТ.4"] and parts["4"].docs == [A]
    pts = [c for c in cs["count_points_article"] if c.root == A]
    assert [(c.meta["article"], c.answer) for c in pts] == [("1", "2")] and st["points_ambiguous"] == 1   # СТ.3 restarts
    assert {c.root: c.answer for c in cs["count_points_article"] if c.root != A} == {Z1: "4", Z2: "4", Z3: "4"}
    codes = {c.root: c for c in cs["count_codes_by_law"]}
    assert codes[Z1].answer == "2" and codes[Z1].gold == [f"STRUCT/{JK}", f"STRUCT/{SK}"]
    assert codes[Z1].docs == [Z1, JK, SK] and codes[Z1].meta == {"law": Z1, "date": "2024-03-01"}
    years = {(c.root, c.meta["year"]): c.answer for c in cs["count_laws_amended_code_year"]}
    assert years == {(JK, "2024"): "2", (JK, "2023"): "1"}                     # SK: < 3 amendments
    dels = cs["count_articles_deleted_by_law"]
    assert len(dels) == 1 and dels[0].answer == "2" and dels[0].root == AB and dels[0].docs == [AB, Z1]
    assert dels[0].gold == sorted([f"{Z1}/СТ.1/Ч.1/П.3#1", f"{Z1}/СТ.1/Ч.1/П.4#1"])
    # Z1's word deletion in СТ.8 is not counted (answer stays 2); Z3's only "article delete" is a word deletion
    assert st["not_whole_article:count_articles_deleted_by_law"] == 1 and st["not_whole_article_edges"] == 2


def test_whole_article_delete():
    from grl.hard.q1x import whole_article_delete as w
    for q in ("44-модда чиқариб ташлансин", "78 – 81-моддалар чиқариб ташлансин", "31² ва 32-моддалар чиқариб ташлансин",
              "236, 237 ва 238-моддалар чиқариб ташлансин", "399¹ – 399⁴-моддалар чиқариб ташлансин",
              "Кодекснинг 25¹-моддаси чиқариб ташлансин.", "12-modda chiqarib tashlansin"):
        assert w(q), q
    for q in ("номидаги «ва жавобгарлиги» деган сўзлар чиқариб ташлансин", "ўн саккизинчи хатбошиси чиқариб ташлансин",
              "31-модданинг учинчи хатбошиси чиқариб ташлансин", "5-бандининг ўзбекча матнидаги «ҳақли» деган сўз "
              "чиқариб ташлансин", "", None):
        assert not w(q), q


def test_r6_codes_need_the_law_line():
    st = Counter()
    cs = by_template(q1x_candidates(NODES, EDGES, stats=st, **inputs(drop_line=Z1)))
    assert Z1 not in {c.root for c in cs.get("count_codes_by_law", [])}
    assert ("EXT/jinoyat kodeks", "2024") not in {(c.root, c.meta["year"]) for c in cs["count_laws_amended_code_year"]}
    assert st["r6_dropped:count_codes_by_law"] >= 1 and st["r6_dropped:count_laws_amended_code_year"] >= 1


def test_renderings_three_languages_no_times():
    cs = q1x_candidates(NODES, EDGES, **inputs())
    st = Counter()
    for c in cs:
        _check_langs(c, st)
    seen = set()
    for c in cs:
        for lang, q in c.questions.items():
            seen.add(c.template)
            assert detect_lang(q) == lang, q
            assert not asks_times(q), q
            assert c.answer not in _numbers(q)
    assert seen == set(Q1X_TEMPLATES)
    q = render("count_parts_article", "ru", "Qonun", {"article": "5", "doctype": "law"})
    assert q == "Сколько частей в статье 5 закона «Qonun»?"
    q = render("count_codes_by_law", "uz-Latn", "", {"date": "2024-03-01", "src": "law"})
    assert q == "2024-yil 1-martdagi qonun bilan nechta kodeksga oʻzgartish kiritilgan?"


def test_anchors_of_new_templates():
    cs = q1x_candidates(NODES, EDGES, **inputs())
    for c in cs:
        _check_langs(c, Counter())
    for c in cs:
        for lang, q in c.questions.items():
            item = {"qid": "x", "question": q, "lang": lang, "template": c.template, "meta": c.meta,
                    "answer": c.answer}
            an = anchors(item)
            if c.template == "count_codes_by_law":
                assert an["title"] == "" and an["dates"] == [c.meta["date"]]
            else:
                assert an["title"]
            if c.template == "count_articles_deleted_by_law":
                assert an["op"] == "delete"


def test_items_cap_and_probe_drop():
    st = Counter()
    items = q1x_items(NODES, EDGES, 1, 50, max_per_root=1, stats=st, **inputs())
    assert max(Counter(it.docs[0] for it in items).values()) == 1
    st2 = Counter()
    items2 = q1x_items(NODES, EDGES, 1, 50, stats=st2, probe_drop=lambda c, lang: "probe_all5"
                       if c.template == "count_articles_deleted_by_law" else None, **inputs())
    assert "count_articles_deleted_by_law" not in {it.template for it in items2}
    assert st2["probe_all5:count_articles_deleted_by_law"] == st2["probe_checked:count_articles_deleted_by_law"] > 0


def test_make_q1x_splits_inherit_and_conflict():
    fixed = {JK: "val", AB: "test", Z1: "train"}
    rows, st, rounds = make_q1x(NODES, EDGES, 3, 50, fixed_splits=fixed, chunk_doc={c: v[0] for c, v in CHUNKS.items()},
                                **inputs())
    by = {(r["template"], r["docs"][0], tuple(sorted(r["meta"].items()))): r for r in rows}
    assert all(r["split"] == "val" for r in rows if JK in r["docs"] + r["gold_docs"])
    assert not any(r["template"] == "count_codes_by_law" and r["docs"][0] == Z1 for r in rows)   # train + val docs
    assert st["r32_conflict"] >= 1
    for r in rows:
        own = {fixed[d] for d in r["docs"] + r["gold_docs"] if d in fixed}
        assert not own or own == {r["split"]}
    assert rounds >= 1 and by


def test_probe_counts_only_probe_drops():
    from grl.hard.q1x import _probe
    cs = q1x_candidates(NODES, EDGES, **inputs())
    for c in cs:
        _check_langs(c, Counter())
    empty = [c for c in cs if not c.questions]
    st = Counter()
    kept = _probe(cs, lambda c, lang: None, st)
    assert len(kept) == len(cs) - len(empty) and not any(k.startswith("probe_dropped") for k in st)
    st = Counter()
    kept = _probe(cs, lambda c, lang: "probe_all5", st)
    assert kept == [] and sum(v for k, v in st.items() if k.startswith("probe_dropped_candidates:")) == \
        len(cs) - len(empty)


def test_q1x_instructions_notes():
    from grl.hard.paraphrase import instructions
    assert "count_parts_article" in instructions("d", "o", prefix="para6c", notes=True)
    assert "count_parts_article" not in instructions("d", "o")


def test_i1_no_part_count_for_articles_with_points():
    st = Counter()
    cs = by_template(q1x_candidates(NODES, EDGES, stats=st, **inputs()))
    # СТ.1 (3 parts, points under Ч.1) and СТ.3 (2 parts, points) are no part-count items any more (fix round 1, I1)
    assert [c.meta["article"] for c in cs["count_parts_article"] if c.root == A] == ["4"]
    assert st["parts_with_points"] == 2


# --- fix round 1, R59 -------------------------------------------------------------------------------------------

def test_anyof_metrics():
    from grl.hard.metrics import anyof_metrics
    assert anyof_metrics(["a", "b"], ["STRUCT/x"], ["a", "b"]) == {"strict_all": False, "strict_recall": 0.0,
                                                                  "anyof_all": True}
    assert anyof_metrics(["a"], ["STRUCT/x"], ["a", "b"])["anyof_all"] is False
    assert anyof_metrics(["STRUCT/x"], ["STRUCT/x"], ["a"])["anyof_all"] is True
    assert anyof_metrics(["a"], ["STRUCT/x"], None)["anyof_all"] is False         # no alt set: any-of = strict


def test_article_chunks_contiguous_and_annotate():
    from grl.hard.q1x import annotate_alt, article_chunks, contiguous
    cids = ["D/СТ.1#1", "D/СТ.1/Ч.2#2", "D/СТ.10#1", "D/СТ.2#1"]
    nodes = ["D/СТ.1", "D/СТ.1/Ч.2", "D/СТ.10", "D/СТ.2"]
    assert article_chunks("D/СТ.1", cids, nodes) == ["D/СТ.1#1", "D/СТ.1/Ч.2#2"]        # not СТ.10 (prefix)
    spans = {"D/СТ.1#1": ("D", 0, 10), "D/СТ.1/Ч.2#2": ("D", 10, 20), "D/СТ.10#1": ("D", 40, 50),
             "D/СТ.2#1": ("D", 20, 30)}
    assert contiguous(["D/СТ.1#1", "D/СТ.1/Ч.2#2"], spans)
    assert not contiguous(["D/СТ.1#1", "D/СТ.2#1"], spans)                               # Ч.2 lies between
    assert not contiguous([], spans)
    rows = [{"qid": "a", "template": "count_parts_article", "gold": ["STRUCT/D/СТ.1"]},
            {"qid": "b", "template": "count_points_article", "gold": ["STRUCT/D/СТ.2"]},
            {"qid": "c", "template": "count_codes_by_law", "gold": ["STRUCT/EXT/k"]}]
    rep = annotate_alt(rows, cids, nodes, spans)
    assert rows[0]["alt_gold"] == ["D/СТ.1#1", "D/СТ.1/Ч.2#2"] and not rows[0]["easy"]
    assert rows[1]["alt_gold"] == ["D/СТ.2#1"] and rows[1]["easy"]
    assert rows[2]["alt_gold"] == [] and not rows[2]["easy"]
    assert rep == {"count_codes_by_law": 1, "count_parts_article|non_easy": 1, "count_points_article|easy": 1}
    with pytest.raises(ValueError):
        annotate_alt([{"qid": "x", "template": "count_parts_article", "gold": ["STRUCT/D/СТ.9"]}], cids, nodes, spans)


def test_hard_pair_carries_alt_gold_only_when_set():
    from grl.hard.ship import hard_pair
    base = {"qid": "q", "question": "?", "lang": "ru", "qtype": "count", "answer": "2", "gold": ["STRUCT/D/СТ.1"],
            "docs": ["D"], "split": "train", "template": "count_parts_article", "meta": {"article": "1"}}
    assert "alt_gold" not in hard_pair(base)["provenance"]["m7"]                 # pilot 7 / 8 rows: unchanged
    p = hard_pair({**base, "alt_gold": ["D/СТ.1#1"], "easy": True})
    assert p["provenance"]["m7"]["alt_gold"] == ["D/СТ.1#1"] and p["provenance"]["m7"]["easy"] is True
    assert p["C_ref"] == ["STRUCT/D/СТ.1"]                                          # training labels keep STRUCT


# --- Phase B: judge plants ----------------------------------------------------------------------------------------

def _pair(template, lang, para, qid="q1", question="orig", meta=None):
    from grl.hard.judge import make_pair
    item = {"qid": qid, "question": question, "lang": lang, "template": template, "meta": meta or {}}
    return make_pair(item, para, {"numbers": [], "dates": [], "title_key": [], "op": None})


def test_object_swap_outside_the_title_only():
    from grl.hard.judge import make_plant
    p = _pair("count_parts_article", "ru", "Из скольких частей состоит статья 5 закона «О частях и пунктах»?")
    pl = make_plant("object_swap", p, 1)
    assert pl["paraphrase"] == "Из скольких пунктов состоит статья 5 закона «О частях и пунктах»?"
    p = _pair("count_codes_by_law", "uz-Latn", "2024-yil 1-martdagi qonun nechta kodeksni oʻzgartirgan?")
    assert "nechta qonunni" in make_plant("object_swap", p, 1)["paraphrase"]
    p = _pair("count_points_article", "uz-Cyrl", "«Аҳоли бандлиги тўғрисида» қонунининг 3-моддасида нечта банд бор?")
    assert make_plant("object_swap", p, 1)["paraphrase"].endswith("нечта қисм бор?")
    assert make_plant("object_swap", _pair("which_laws_article", "ru", "Какие законы?"), 1) is None


def test_topup_fills_a_kind_and_is_marked():
    from grl.hard.judge import plan_plants
    pairs = [_pair("count_parts_article", "ru", f"Сколько частей в статье {k}?", qid=f"a{k}") for k in range(3)]
    top = [_pair("count_parts_article", "ru", f"Сколько частей в статье {k}?", qid=f"t{k}") for k in range(5)]
    pl = plan_plants(pairs, 1, min_n=4, kinds=("object_swap",), topup=top)
    assert len(pl) == 4 and sum(1 for p in pl if p.get("topup")) == 1
    assert {p["qid"] for p in pl if not p.get("topup")} == {"a0", "a1", "a2"}
    assert plan_plants(pairs, 1, min_n=4, kinds=("object_swap",)) == [p for p in pl if not p.get("topup")]
