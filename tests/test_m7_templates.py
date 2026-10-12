# tests/test_m7_templates.py -- M7 Task 5: Q1 (count) / Q2 (version) templates, gold from the graph, leak check
import re

import pytest

from grl.hard.checks import doc_components, leak_ok
from grl.hard.templates import count_items, lang_quotas, version_items
from grl.env.structure import build_struct_texts

LANGS = {"uz-Cyrl", "uz-Latn", "ru"}


def toks(s):
    return len(s.split())


def _el(nid, typ, doc, ext=False):
    return {"node_id": nid, "kind": "element", "type": typ, "doc_id": doc, "external": ext, "title": ""}


def _doc(nid, title, ext=False):
    return {"node_id": nid, "kind": "document", "external": ext, "title": title, "type": "документ",
            "doc_id": None if ext else nid, "attrs": {"doc_type": "law_text"}}


def _c(a, b):
    return {"src": a, "dst": b, "type": "CONTAINS", "attrs": {}}


def _am(src, dst, date, op):
    return {"src": src, "dst": dst, "type": "AMENDS", "attrs": {"date": date, "op": op}}


def _toy(n_laws_on_61: int = 0):
    """The Task 1 toy (document Q-1, external act EXT/kodeks) grown past the filters: Q-1 has 2 chapters and 3
    articles; EXT/kodeks has 4 amendments from two laws (Q-1 on 2023-05-02, Q-2 on 2025-01-10). μ⁻¹: every source
    element owns one chunk; EXT elements own none (as in the real graph)."""
    nodes = [_doc("Q-1", "Qonun A"), _el("Q-1/ГЛ.1", "глава", "Q-1"), _el("Q-1/ГЛ.2", "глава", "Q-1"),
             _el("Q-1/ГЛ.1/СТ.1", "статья", "Q-1"), _el("Q-1/ГЛ.1/СТ.2", "статья", "Q-1"),
             _el("Q-1/ГЛ.2/СТ.3", "статья", "Q-1"), _el("Q-1/ГЛ.1/СТ.2/Ч.1", "часть", "Q-1"),
             _doc("Q-2", "Qonun B"), _el("Q-2/СТ.1", "статья", "Q-2"), _el("Q-2/СТ.2", "статья", "Q-2"),
             _doc("EXT/kodeks", "Kodeks", ext=True), _el("EXT/kodeks/СТ.61", "статья", None, True),
             _el("EXT/kodeks/СТ.62", "статья", None, True), _el("EXT/kodeks/СТ.62/Ч.2", "часть", None, True)]
    edges = [_c("Q-1", "Q-1/ГЛ.1"), _c("Q-1", "Q-1/ГЛ.2"), _c("Q-1/ГЛ.1", "Q-1/ГЛ.1/СТ.1"),
             _c("Q-1/ГЛ.1", "Q-1/ГЛ.1/СТ.2"), _c("Q-1/ГЛ.2", "Q-1/ГЛ.2/СТ.3"), _c("Q-1/ГЛ.1/СТ.2", "Q-1/ГЛ.1/СТ.2/Ч.1"),
             _c("Q-2", "Q-2/СТ.1"), _c("Q-2", "Q-2/СТ.2"), _c("EXT/kodeks", "EXT/kodeks/СТ.61"),
             _c("EXT/kodeks", "EXT/kodeks/СТ.62"), _c("EXT/kodeks/СТ.62", "EXT/kodeks/СТ.62/Ч.2"),
             _am("Q-1/ГЛ.1/СТ.1", "EXT/kodeks/СТ.61", "2023-05-02", "replace"),
             _am("Q-1/ГЛ.1/СТ.2", "EXT/kodeks/СТ.61", "2023-05-02", "delete"),
             _am("Q-2/СТ.1", "EXT/kodeks/СТ.62/Ч.2", "2025-01-10", "delete"),
             _am("Q-2/СТ.2", "EXT/kodeks/СТ.62", "2025-01-10", "edit")]
    for i in range(n_laws_on_61):        # law L-i amends EXT/kodeks/СТ.61 (same year, same op) from its article 1
        d = f"L-{i}"
        nodes += [_doc(d, f"Qonun L{i}"), _el(f"{d}/СТ.1", "статья", d)]
        edges += [_c(d, f"{d}/СТ.1"), _am(f"{d}/СТ.1", "EXT/kodeks/СТ.61", "2024-03-0" + str(i % 9 + 1), "edit")]
    mu = {n["node_id"]: [n["node_id"] + "#1"] for n in nodes
          if n["kind"] == "element" and not n["external"]}
    return nodes, edges, mu


@pytest.fixture
def toy_graph():
    return _toy()


@pytest.fixture
def toy_graph_many_amends():
    return _toy(n_laws_on_61=9)


# --- the brief's tests -------------------------------------------------------------------------------------------

def test_count_answer_matches_graph(toy_graph):
    nodes, edges, mu = toy_graph
    items = count_items(nodes, edges, rng_seed=1, n=10)
    assert items
    for it in items:
        assert it.gold[0].startswith("STRUCT/") and it.answer.isdigit()
    by_lang = {}
    for it in items:                                        # Review Focus 3: same template+root -> same answer in any lang
        by_lang.setdefault((it.template, it.docs[0]), set()).add(it.answer)
    assert all(len(v) == 1 for v in by_lang.values())


def test_version_gold_size_and_reject_large(toy_graph_many_amends):   # Review Focus 4
    nodes, edges, mu = toy_graph_many_amends                # one article amended by 9 laws in one year, one op
    items = version_items(nodes, edges, mu, rng_seed=1, n=50)
    assert all(1 <= len(it.gold) <= 5 for it in items)
    assert not any(it.template == "which_laws_article" and len(it.gold) > 5 for it in items)


def test_leak():
    assert not leak_ok("а б в г д е ж з и к", ["x а б в г д е ж з и y"], n=8)
    assert leak_ok("сколько статей в законе", ["совсем другой текст фрагмента здесь"], n=8)


# --- added: exact answers, narrowing, filters, R6, leak, components ---------------------------------------------

def _by(items, template, **meta):
    return [it for it in items if it.template == template and all(it.meta.get(k) == v for k, v in meta.items())]


def test_count_exact_values(toy_graph):
    nodes, edges, _ = toy_graph
    items = count_items(nodes, edges, rng_seed=3, n=100)
    got = {(it.template, it.docs[0]): it.answer for it in items}
    assert got[("count_articles", "Q-1")] == "3" and got[("count_chapters", "Q-1")] == "2"
    assert ("count_articles", "Q-2") not in got                     # 2 articles < 3: filtered
    yr = {it.meta["year"]: it.answer for it in _by(items, "count_amends_year")}
    dl = {it.meta["year"]: it.answer for it in _by(items, "count_deleted_year")}
    assert len(yr) == 1 and set(yr.items()) <= {("2023", "2"), ("2025", "2")}    # one per (template, root)
    assert len(dl) == 1 and set(dl.items()) <= {("2023", "1"), ("2025", "1")}
    law = {it.meta["law"]: it.answer for it in _by(items, "count_articles_by_law")}
    assert len(law) == 1 and set(law.items()) <= {("Q-1", "1"), ("Q-2", "1")}
    assert {it.template for it in items} == {"count_articles", "count_chapters", "count_amends_year",
                                             "count_deleted_year", "count_articles_by_law"}
    for it in items:
        assert it.gold == ["STRUCT/" + it.docs[0]] and it.qtype == "count" and it.lang in LANGS
        assert it.answer not in re.findall(r"\d+", it.question)     # the number is never in the question


def test_count_one_per_template_root_and_root_cap(toy_graph):
    nodes, edges, _ = toy_graph
    items = count_items(nodes, edges, rng_seed=5, n=100, max_per_root=4)
    keys = [(it.template, it.docs[0]) for it in items]
    assert len(keys) == len(set(keys))
    from collections import Counter
    assert max(Counter(it.docs[0] for it in items).values()) <= 4


def test_count_deterministic(toy_graph):
    nodes, edges, _ = toy_graph
    a = count_items(nodes, edges, rng_seed=7, n=6)
    b = count_items(list(reversed(nodes)), list(reversed(edges)), rng_seed=7, n=6)
    assert a == b


def test_r6_drops_invisible_law_line(toy_graph):
    """R6: an item is kept only if its fact is in the STRUCT text. Truncate EXT/kodeks to its header: the per-law
    lines are gone, so count_articles_by_law on it is dropped; the header counts still give items."""
    nodes, edges, _ = toy_graph
    texts = build_struct_texts(nodes, edges, toks)
    head = texts["STRUCT/EXT/kodeks"].split("\nQ-1")[0]
    texts = {**texts, "STRUCT/EXT/kodeks": head}
    stats = {}
    items = count_items(nodes, edges, rng_seed=1, n=100, struct_texts=texts, stats=stats)
    assert not _by(items, "count_articles_by_law")
    assert stats["r6_dropped"] >= 2 and _by(items, "count_amends_year")


def test_count_long_title_exempt_r14b():
    """R14b: the STRUCT text starts with the root's title; that line is exempt, so a 9-word title is askable."""
    long_title = "qonun bank soliq bojxona yer suv havo tuproq ormon"   # 9 words, no number words
    nodes, edges, _ = _toy()
    nodes = [dict(n, title=long_title) if n["node_id"] == "Q-1" else n for n in nodes]
    stats = {}
    items = count_items(nodes, edges, rng_seed=1, n=100, stats=stats)
    q1 = [it for it in items if it.docs[0] == "Q-1"]
    assert q1 and all(long_title in it.question for it in q1)


def test_leak_r14b_title_only_vs_body():
    title = "qonun bank soliq bojxona yer suv havo tuproq ormon"
    q = f"«{title}» qonunida nechta modda bor?"
    frag = {"STRUCT/Q-1": f"{title}\nтур: law_text\nбоблар: 2; моддалар: 3"}
    assert not leak_ok(q, frag)                                          # no exemption: the title is a 9-gram
    assert leak_ok(q, frag, exempt={"STRUCT/Q-1": title})                 # title-only overlap passes
    body = {"STRUCT/Q-1": f"{title}\nтур: law_text\nсм. {title} ва бошқалар"}
    assert not leak_ok(q, body, exempt={"STRUCT/Q-1": title})             # the same words in the body still fail
    assert not leak_ok(q, [frag["STRUCT/Q-1"]], exempt={"STRUCT/Q-1": title})   # a plain list has no exemption


def test_version_values_and_gold(toy_graph):
    nodes, edges, mu = toy_graph
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100)
    wl = _by(items, "what_law_changed")
    assert {it.meta["law"]: tuple(it.gold) for it in wl} == {"Q-1": ("Q-1/ГЛ.1/СТ.1#1", "Q-1/ГЛ.1/СТ.2#1"),
                                                             "Q-2": ("Q-2/СТ.1#1", "Q-2/СТ.2#1")}
    art = _by(items, "which_laws_article")
    assert [tuple(it.gold) for it in art if it.meta["article"] == "61"] == [("Q-1/ГЛ.1/СТ.1#1", "Q-1/ГЛ.1/СТ.2#1")]
    part = _by(items, "which_law_part")
    assert [(it.meta["article"], it.meta["part"], tuple(it.gold)) for it in part] == [("62", "2", ("Q-2/СТ.1#1",))]
    for it in items:
        assert it.qtype == "version" and it.gold == sorted(set(it.gold))
        assert all(isinstance(a, list) and len(a) == 4 for a in it.answer)
        assert 2 <= len(it.gold) <= 5 or it.template == "which_law_part"


def test_version_narrow_by_year():
    """СТ.61: 2 chunks in 2023 + 9 in 2024 (11 > 5) -> narrowed to 2023 (2 chunks); 2024 (9, one op) is refused."""
    nodes, edges, mu = _toy(n_laws_on_61=9)
    items = _by(version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100), "which_laws_article",
                article="61")
    assert [(it.meta.get("year"), it.meta.get("op"), len(it.gold)) for it in items] == [("2023", None, 2)]
    assert "2023" in items[0].question


def test_version_narrow_by_op():
    """6 laws amend СТ.61 in 2024: 3 'edit', 3 'add' -> per-year 6 > 5 -> narrowed by op, 3 chunks each."""
    nodes, edges, mu = _toy()
    for i in range(6):
        d = f"L-{i}"
        nodes += [_doc(d, f"Qonun L{i}"), _el(f"{d}/СТ.1", "статья", d)]
        edges += [_c(d, f"{d}/СТ.1"), _am(f"{d}/СТ.1", "EXT/kodeks/СТ.61", f"2024-03-0{i + 1}", "edit" if i < 3 else "add")]
        mu[f"{d}/СТ.1"] = [f"{d}/СТ.1#1"]
    items = _by(version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100), "which_laws_article",
                article="61")
    got = sorted((it.meta.get("year"), it.meta.get("op") or "", len(it.gold)) for it in items)
    assert got == [("2023", "", 2), ("2024", "add", 3), ("2024", "edit", 3)]
    assert {it.meta["op"] for it in items if it.meta.get("op")} == {"add", "edit"}


def test_version_single_gold_cap():
    """|gold| = 1 only for which_law_part and at most 20 % of the Q2 items."""
    nodes, edges, mu = _toy()
    for i in range(8):                   # 8 roots, each: one part amended by one law (|gold| = 1) + 3 more amendments
        r = f"EXT/act{i}"
        nodes += [_doc(r, f"Act {i}", ext=True), _el(f"{r}/СТ.5", "статья", None, True),
                  _el(f"{r}/СТ.5/Ч.1", "часть", None, True), _doc(f"Z-{i}", f"Law Z{i}")]
        edges += [_c(r, f"{r}/СТ.5"), _c(f"{r}/СТ.5", f"{r}/СТ.5/Ч.1")]
        for k in range(4):
            s = f"Z-{i}/СТ.{k + 1}"
            nodes.append(_el(s, "статья", f"Z-{i}"))
            edges += [_c(f"Z-{i}", s), _am(s, f"{r}/СТ.5/Ч.1" if k == 0 else f"{r}/СТ.5", "2024-01-01",
                                           "edit" if k == 0 else "add")]
            mu[s] = [s + "#1"]
    items = version_items(nodes, edges, mu, rng_seed=2, n=20)
    ones = [it for it in items if len(it.gold) == 1]
    assert all(it.template == "which_law_part" for it in ones)
    assert len(ones) <= 0.2 * len(items)


def test_version_requires_chunks_and_leak(toy_graph):
    nodes, edges, mu = toy_graph
    mu = {k: v for k, v in mu.items() if k != "Q-2/СТ.1"}               # one Q-2 source without a fragment
    items = version_items(nodes, edges, mu, rng_seed=1, n=100)
    assert not [it for it in items if "Q-2/СТ.1#1" in it.gold or any(a[0] == "Q-2" and a[3].endswith("Ч.2")
                                                                      for a in it.answer)]
    texts = {c: "Kodeks" for v in mu.values() for c in v}
    stats = {}
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, chunk_texts=texts, stats=stats)
    assert items and stats.get("leak_rejected", 0) == 0
    long = "ab cd ef gh ij kl mn op qr"
    nodes2 = [dict(n, title=long) if n["node_id"] == "EXT/kodeks" else n for n in nodes]
    texts = {c: f"x {long} y" for c in texts}
    stats = {}
    assert version_items(nodes2, edges, mu, rng_seed=1, n=100, chunk_texts=texts, stats=stats) == []
    assert stats["leak_rejected"] > 0


def test_languages_follow_shares(toy_graph):
    nodes, edges, mu = toy_graph
    assert lang_quotas(30, {"uz-Cyrl": 528, "ru": 385, "uz-Latn": 328}) == {"uz-Cyrl": 13, "ru": 9, "uz-Latn": 8}
    items = count_items(nodes, edges, rng_seed=1, n=3, lang_shares={"ru": 1.0})
    assert {it.lang for it in items} == {"ru"}
    it = items[0]
    assert any(ch in it.question for ch in "абвгдеёжзийклмнопрстуфхцчшщыэюя")


def test_leak_id_and_transliteration():
    assert not leak_ok("see Q-1/СТ.1#1 please", ["text"], ids=["Q-1/СТ.1#1"])
    cyr = "Ўзбекистон Республикасининг айрим қонун ҳужжатларига ўзгартириш ва қўшимчалар киритиш"
    lat = "Oʻzbekiston Respublikasining ayrim qonun hujjatlariga oʻzgartirish va qoʻshimchalar kiritish"
    assert not leak_ok(f"nima {lat}?", [cyr])                         # a transliterated copy is a copy
    assert leak_ok("bir ikki uch tort besh olti yetti", ["bir ikki uch tort besh olti yetti"])   # 7 < 8


def test_doc_components(toy_graph):
    nodes, edges, mu = toy_graph
    items = version_items(nodes, edges, mu, rng_seed=1, n=100) + count_items(nodes, edges, rng_seed=1, n=100)
    comp = doc_components(items, edges)
    assert set(comp) == {it.qid for it in items}
    wl = {it.meta["law"]: it.qid for it in _by(items, "what_law_changed")}
    kod = [it.qid for it in items if it.docs[0] == "EXT/kodeks"]
    assert len({comp[q] for q in kod} | {comp[q] for q in wl.values()}) == 1    # Q-1, Q-2 joined through EXT/kodeks


# --- fix round 1: R16 (naming), R17 (delete / edit wording), source wording, empty buckets ------------------------

def test_r16_shared_title_disambiguated_or_dropped():
    """A third root shares Kodeks's title (folded: Cyrillic «Кодекс» = Latin «Kodeks»). The EXT act with a graph number
    is named with it; the internal resolution (no number in the graph, not a law -- R16b does not apply) is dropped;
    Q-1 keeps its plain title."""
    nodes, edges, mu = _toy()
    nodes = [dict(n, attrs={"numbers": ["oʼrq-642"]}) if n["node_id"] == "EXT/kodeks" else n for n in nodes]
    nodes += [dict(_doc("Q-9", "Кодекс"), attrs={"doc_type": "resolution"}), _el("Q-9/СТ.1", "статья", "Q-9"), _el("Q-9/СТ.2", "статья", "Q-9"),
              _el("Q-9/СТ.3", "статья", "Q-9"), _el("Q-9/СТ.4", "статья", "Q-9")]
    edges += [_c("Q-9", f"Q-9/СТ.{i}") for i in range(1, 5)]
    st = {}
    items = count_items(nodes, edges, rng_seed=1, n=100, stats=st) + \
        version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100)
    assert not [it for it in items if it.docs[0] == "Q-9"] and st["r16_title_dropped"] >= 1
    kod = [it for it in items if it.docs[0] == "EXT/kodeks"]
    assert kod and all(any(x in it.question for x in ("ЗРУ-642", "ЎРҚ-642", "OʻRQ-642")) for it in kod)
    assert all("ЗРУ" not in it.question and "ЎРҚ" not in it.question for it in items if it.docs[0] == "Q-1")


def test_r16_twin_dedupe():
    """Q-3 is a translated twin of Q-1 (same doc_type and 2 / 3 / 1 / 0 structure): only Q-1 (smaller id) is kept."""
    nodes, edges, _ = _toy()
    nodes += [_doc("Q-3", "Закон А"), _el("Q-3/ГЛ.1", "глава", "Q-3"), _el("Q-3/ГЛ.2", "глава", "Q-3"),
              _el("Q-3/ГЛ.1/СТ.1", "статья", "Q-3"), _el("Q-3/ГЛ.1/СТ.2", "статья", "Q-3"),
              _el("Q-3/ГЛ.2/СТ.3", "статья", "Q-3"), _el("Q-3/ГЛ.1/СТ.2/Ч.1", "часть", "Q-3")]
    edges += [_c("Q-3", "Q-3/ГЛ.1"), _c("Q-3", "Q-3/ГЛ.2"), _c("Q-3/ГЛ.1", "Q-3/ГЛ.1/СТ.1"),
              _c("Q-3/ГЛ.1", "Q-3/ГЛ.1/СТ.2"), _c("Q-3/ГЛ.2", "Q-3/ГЛ.2/СТ.3"), _c("Q-3/ГЛ.1/СТ.2", "Q-3/ГЛ.1/СТ.2/Ч.1")]
    st = {}
    items = count_items(nodes, edges, rng_seed=1, n=100, stats=st)
    roots = {it.docs[0] for it in items if it.template in ("count_articles", "count_chapters")}
    assert roots == {"Q-1"} and st["r16_twin_dropped"] == 2


def test_r17_wording():
    nodes, edges, mu = _toy()
    edges += [_am("Q-1/ГЛ.2/СТ.3", "EXT/kodeks/СТ.62", "2023-05-02", "delete")]    # 2023: 2 delete fragments
    for lang in ("ru", "uz-Cyrl", "uz-Latn"):
        sh = {lang: 1}
        items = count_items(nodes, edges, rng_seed=1, n=100, lang_shares=sh) + \
            version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, lang_shares=sh)
        q = {it.template: it.question for it in items}
        assert {"count_deleted_year", "count_amends_year", "which_deleted_year"} <= set(q)
        for t in ("count_deleted_year", "which_deleted_year", "count_amends_year"):
            assert not re.search(r"норм|norma|нормa", q[t]) and not re.search(r"\bраз\b", q[t])
        if lang == "ru":
            assert q["count_deleted_year"].startswith("Сколько исключений")
            assert q["which_deleted_year"].startswith("Что было исключено из акта")
            assert "отдельных элементов" in q["count_amends_year"]
        elif lang == "uz-Cyrl":
            assert "чиқариб ташлаш тарзида" in q["count_deleted_year"] and "дан нималар" in q["which_deleted_year"]
            assert "алоҳида элементларига" in q["count_amends_year"]
        else:
            assert "chiqarib tashlash tarzida" in q["count_deleted_year"] and "dan nimalar" in q["which_deleted_year"]
            assert "alohida elementlariga" in q["count_amends_year"]


def test_source_wording_and_singular():
    """Q-1 / Q-2 are not QONUNLAR -> «документ»; rename Q-2 to QONUNLAR-2 -> «закон». A single source -> singular."""
    nodes, edges, mu = _toy()
    ru = {"ru": 1}
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, lang_shares=ru)
    wl = {it.meta["law"]: it.question for it in _by(items, "what_law_changed")}
    assert "документ от" in wl["Q-1"] and "закон" not in wl["Q-1"]
    part = _by(items, "which_law_part")[0].question
    assert part.startswith("Какой документ и когда изменил")
    art = [it.question for it in _by(items, "which_laws_article") if it.meta["article"] == "61"][0]
    assert art.startswith("Какой документ вносил")                   # one source document (Q-1), two fragments

    def ren(x):
        return x.replace("Q-2", "QONUNLAR-2") if isinstance(x, str) else x
    nodes2 = [{k: ren(v) for k, v in n.items()} for n in nodes]
    edges2 = [{k: ren(v) for k, v in e.items()} for e in edges]
    mu2 = {ren(k): [ren(c) for c in v] for k, v in mu.items()}
    items = version_items(nodes2, edges2, mu2, rng_seed=1, n=100, max_per_root=100, lang_shares=ru)
    wl = {it.meta["law"]: it.question for it in _by(items, "what_law_changed")}
    assert "закон от" in wl["QONUNLAR-2"]
    lat = count_items(nodes2, edges2, rng_seed=1, n=100, lang_shares={"uz-Latn": 1})
    by_law = {it.meta["law"]: it.question for it in _by(lat, "count_articles_by_law")}
    assert all(("qonun bilan" in q) == (z.startswith("QONUNLAR")) for z, q in by_law.items())


def test_plural_sources():
    nodes, edges, mu = _toy()
    edges += [_am("Q-2/СТ.1", "EXT/kodeks/СТ.61", "2025-01-10", "edit")]
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, lang_shares={"ru": 1})
    art = [it for it in _by(items, "which_laws_article") if it.meta["article"] == "61"][0]
    assert art.question.startswith("Какие документы вносили")


def test_narrowing_refuses_empty_buckets():
    """7 undated amendments of СТ.61 (> 5 fragments): the "" year bucket is refused, not asked about."""
    nodes, edges, mu = _toy()
    for i in range(7):
        d = f"L-{i}"
        nodes += [_doc(d, f"Qonun L{i}"), _el(f"{d}/СТ.1", "статья", d)]
        edges += [_c(d, f"{d}/СТ.1"), _am(f"{d}/СТ.1", "EXT/kodeks/СТ.61", "", "edit")]
        mu[f"{d}/СТ.1"] = [f"{d}/СТ.1#1"]
    st = {}
    items = version_items(nodes, edges, mu, rng_seed=1, n=100, max_per_root=100, stats=st)
    art = [it for it in _by(items, "which_laws_article") if it.meta["article"] == "61"]
    assert [it.meta.get("year") for it in art] == ["2023"] and st["empty_bucket_refused"] >= 1


def test_leak_ids_from_mapping():
    assert not leak_ok("see Q-1/СТ.1#1", {"Q-1/СТ.1#1": "text"})
    assert leak_ok("see nothing", {"Q-1/СТ.1#1": "text"})


# --- fix round 2: R16b -------------------------------------------------------------------------------------------

def _law_doc(nid, title, doc_type="law_text", n_art=3):
    ns = [dict(_doc(nid, title), attrs={"doc_type": doc_type})] + \
        [_el(f"{nid}/СТ.{i}", "статья", nid) for i in range(1, n_art + 1)]
    return ns, [_c(nid, f"{nid}/СТ.{i}") for i in range(1, n_art + 1)]


def test_r16b_law_vs_resolution_and_two_laws():
    nodes, edges, _ = _toy()
    for nid, t, dt, k in (("Q-5", "Raqobat toʼgʼrisida", "law_text", 4), ("R-5", "Рақобат тўғрисида", "resolution", 0),
                          ("Q-6", "Sugurta toʼgʼrisida", "law_text", 3), ("Q-7", "Sugurta toʼgʼrisida", "law_text", 5)):
        ns, es = _law_doc(nid, t, dt, k)
        nodes += ns
        edges += es
    st = {}
    items = count_items(nodes, edges, rng_seed=1, n=100, stats=st, lang_shares={"ru": 1})
    by_root = {}
    for it in items:
        by_root.setdefault(it.docs[0], []).append(it)
    assert [it.answer for it in by_root["Q-5"] if it.template == "count_articles"] == ["4"]
    assert all("в законе «Raqobat" in it.question for it in by_root["Q-5"])    # the type word names the root
    assert "Q-6" not in by_root and "Q-7" not in by_root and st["r16_title_dropped"] >= 2
    ns, es = _law_doc("Q-8", "Kodeks", "law_text", 4)          # a law_text root sharing an EXT act's title
    items = count_items(nodes + ns, edges + es, rng_seed=1, n=100)
    assert not [it for it in items if it.docs[0] == "Q-8"]     # an EXT act is a law too: still ambiguous
