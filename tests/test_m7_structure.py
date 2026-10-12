# tests/test_m7_structure.py
from grl.env.structure import build_struct_texts, doc_structure, struct_id, struct_text

def toks(s):                      # whitespace counter for tests; production passes the corpus tokenizer
    return len(s.split())

NODES = [
    {"node_id": "Q-1", "kind": "document", "external": False, "title": "Qonun A", "type": "document", "doc_id": "Q-1"},
    {"node_id": "Q-1/ГЛ.1", "kind": "element", "type": "глава", "doc_id": "Q-1", "external": False, "title": ""},
    {"node_id": "Q-1/ГЛ.1/СТ.1", "kind": "element", "type": "статья", "doc_id": "Q-1", "external": False, "title": ""},
    {"node_id": "Q-1/ГЛ.1/СТ.2", "kind": "element", "type": "статья", "doc_id": "Q-1", "external": False, "title": ""},
    {"node_id": "Q-1/ГЛ.1/СТ.2/Ч.1", "kind": "element", "type": "часть", "doc_id": "Q-1", "external": False, "title": ""},
    {"node_id": "EXT/kodeks", "kind": "document", "external": True, "title": "Kodeks", "type": "document", "doc_id": "EXT/kodeks"},
    {"node_id": "EXT/kodeks/СТ.61", "kind": "element", "external": True, "type": "статья", "doc_id": "EXT/kodeks", "title": ""},
]
EDGES = [
    {"src": "Q-1", "dst": "Q-1/ГЛ.1", "type": "CONTAINS", "attrs": {}},
    {"src": "Q-1/ГЛ.1", "dst": "Q-1/ГЛ.1/СТ.1", "type": "CONTAINS", "attrs": {}},
    {"src": "Q-1/ГЛ.1", "dst": "Q-1/ГЛ.1/СТ.2", "type": "CONTAINS", "attrs": {}},
    {"src": "Q-1/ГЛ.1/СТ.2", "dst": "Q-1/ГЛ.1/СТ.2/Ч.1", "type": "CONTAINS", "attrs": {}},
    {"src": "Q-1/ГЛ.1/СТ.1", "dst": "EXT/kodeks/СТ.61", "type": "AMENDS", "attrs": {"date": "2025-01-10", "op": "replace"}},
    {"src": "Q-1/ГЛ.1/СТ.2", "dst": "EXT/kodeks/СТ.61", "type": "AMENDS", "attrs": {"date": "2023-05-02", "op": "delete"}},
]

def test_counts_document_and_ext():
    s = doc_structure(NODES, EDGES)
    assert (s["Q-1"]["n_chapters"], s["Q-1"]["n_articles"], s["Q-1"]["n_parts"]) == (1, 2, 1)
    assert s["EXT/kodeks"]["amend_counts"] == {"by_year": {"2023": 1, "2025": 1}, "by_op": {"delete": 1, "replace": 1}}
    assert s["EXT/kodeks"]["amended_articles"] == ["61"]
    assert [a[0] for a in s["EXT/kodeks"]["amended_by"]] == ["Q-1", "Q-1"]       # sorted by (date, src)

def test_text_deterministic_and_has_counts():
    t1 = build_struct_texts(NODES, EDGES, toks)
    t2 = build_struct_texts(list(reversed(NODES)), list(reversed(EDGES)), toks)
    assert t1 == t2 and set(t1) == {struct_id("Q-1"), struct_id("EXT/kodeks")}
    assert "2" in t1[struct_id("Q-1")] and "Qonun A" in t1[struct_id("Q-1")]

def test_long_list_truncated_counts_kept():
    nodes = NODES[:6] + [{"node_id": f"EXT/kodeks/СТ.{i}", "kind": "element", "external": True, "type": "статья",
                          "doc_id": "EXT/kodeks", "title": ""} for i in range(500)]
    edges = [{"src": "Q-1/ГЛ.1/СТ.1", "dst": f"EXT/kodeks/СТ.{i}", "type": "AMENDS",
              "attrs": {"date": "2025-01-10", "op": "edit"}} for i in range(500)]
    t = build_struct_texts(nodes, edges, toks, max_tokens=384)[struct_id("EXT/kodeks")]
    assert toks(t) <= 384 and "500" in t

def test_empty_document_has_text():
    t = struct_text(doc_structure(NODES[:1], [])["Q-1"], toks)
    assert "Qonun A" in t and toks(t) <= 384


# --- fix round 1 -------------------------------------------------------------------------------------------------

def test_repeals_path():
    nodes = NODES + [{"node_id": "Q-2", "kind": "document", "external": False, "title": "Qonun B", "type": "document",
                      "doc_id": "Q-2"},
                     {"node_id": "Q-2/СТ.3", "kind": "element", "type": "статья", "doc_id": "Q-2", "external": False,
                      "title": ""}]
    edges = EDGES + [{"src": "Q-2", "dst": "Q-2/СТ.3", "type": "CONTAINS", "attrs": {}},
                     {"src": "Q-2/СТ.3", "dst": "EXT/kodeks/СТ.61", "type": "REPEALS", "attrs": {"date": "2025-03-01"}}]
    s = doc_structure(nodes, edges)["EXT/kodeks"]
    assert [r[:2] for r in s["repealed_by"]] == [("Q-2", "2025-03-01")]
    assert s["amend_counts"]["by_year"] == {"2023": 1, "2025": 1}            # repeals are not amendments
    t = build_struct_texts(nodes, edges, toks)[struct_id("EXT/kodeks")]
    assert "бекор қилинган: 1" in t
    assert "бекор: Q-2 «Qonun B» 2025-03-01: repeal 1; бекор қилинган моддалар 61" in t.splitlines()


def test_nested_ext_element_resolved_by_contains_walk():
    # id prefix "X" is not a root: only the CONTAINS walk X/СТ.9 -> EXT/kodeks/ГЛ.2 -> EXT/kodeks finds it
    nodes = NODES + [{"node_id": "EXT/kodeks/ГЛ.2", "kind": "element", "external": True, "type": "глава",
                      "doc_id": None, "title": ""},
                     {"node_id": "X/СТ.9", "kind": "element", "external": True, "type": "статья", "doc_id": None,
                      "title": ""}]
    edges = EDGES + [{"src": "EXT/kodeks", "dst": "EXT/kodeks/ГЛ.2", "type": "CONTAINS", "attrs": {}},
                     {"src": "EXT/kodeks/ГЛ.2", "dst": "X/СТ.9", "type": "CONTAINS", "attrs": {}},
                     {"src": "Q-1/ГЛ.1/СТ.1", "dst": "X/СТ.9", "type": "AMENDS", "attrs": {"date": "2024-02-02",
                                                                                         "op": "edit"}}]
    s = doc_structure(nodes, edges)["EXT/kodeks"]
    assert s["amend_counts"]["by_year"] == {"2023": 1, "2024": 1, "2025": 1}
    assert s["amended_articles"] == ["9", "61"]


def test_header_overflow_raises():
    import pytest
    long = [{**NODES[0], "title": " ".join(["so'z"] * 400)}]
    with pytest.raises(ValueError):
        build_struct_texts(long, [], toks, max_tokens=384)


def test_long_list_exact_total_line():
    nodes = NODES[:6] + [{"node_id": f"EXT/kodeks/СТ.{i}", "kind": "element", "external": True, "type": "статья",
                          "doc_id": "EXT/kodeks", "title": ""} for i in range(500)]
    edges = [{"src": "Q-1/ГЛ.1/СТ.1", "dst": f"EXT/kodeks/СТ.{i}", "type": "AMENDS",
              "attrs": {"date": "2025-01-10", "op": "edit"}} for i in range(500)]
    lines = build_struct_texts(nodes, edges, toks, max_tokens=384)[struct_id("EXT/kodeks")].splitlines()
    assert "ўзгартишлар: 500; 2025: 500 (edit 500)" in lines and "edit: 500" in lines


def test_grouped_tail_one_line_per_law_and_date():
    edges = EDGES + [{"src": "Q-1/ГЛ.1/СТ.2", "dst": "EXT/kodeks/СТ.61", "type": "AMENDS",
                      "attrs": {"date": "2025-01-10", "op": "edit"}},
                     {"src": "Q-1/ГЛ.1/СТ.2", "dst": "EXT/kodeks/СТ.61", "type": "AMENDS",
                      "attrs": {"date": "2025-01-10", "op": "edit"}}]
    lines = build_struct_texts(NODES, edges, toks)[struct_id("EXT/kodeks")].splitlines()
    grp = [l for l in lines if l.startswith("Q-1 ") and "2025-01-10" in l]
    assert grp == ["Q-1 «Qonun A» 2025-01-10: edit 2, replace 1; ўзгартирилган моддалар 61"]
    assert "2025: 3 (edit 2, replace 1)" in lines[1]
    assert lines[-1] == "ўзгартирилган моддалар (жами): 61"
    assert lines.index(grp[0]) < len(lines) - 1


def test_ext_fragment_has_no_element_counts_line():
    t = build_struct_texts(NODES, EDGES, toks)
    assert "боблар:" not in t[struct_id("EXT/kodeks")]
    assert "боблар: 1; моддалар: 2; қисмлар: 1; бандлар: 0" in t[struct_id("Q-1")]
