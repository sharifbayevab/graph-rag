import copy
import json

import pytest

from graphrag import config
from graphrag.kg.artifact import SchemaMismatch, read_graph, read_schema_meta, write_graph
from graphrag.kg.schema import EDGE_TYPES, Evidence, GraphEdge, GraphNode, Schema


def _nodes():
    return [GraphNode("D1", "document", "документ", "Қонун", "D1", False, {}),
            GraphNode("D1/СТ.1", "element", "статья", "Мақсад", "D1", False, {}),
            GraphNode("CON/sanatsiya", "concept", "понятие", "санация", None, False, {})]


def test_schema_validates_edge_endpoints_and_attrs():
    s = Schema.load()
    nodes = {n.node_id: n for n in _nodes()}
    ok = GraphEdge("D1/СТ.1", "CON/sanatsiya", "DEFINES",
                   {"evidence": Evidence("D1", 0, 5, "санац").__dict__})
    assert s.validate_edge(ok, nodes) == []
    bad = GraphEdge("CON/sanatsiya", "D1", "CONTAINS", {})
    assert s.validate_edge(bad, nodes)                       # concept cannot CONTAIN
    no_date = GraphEdge("D1/СТ.1", "D1", "REPEALS", {"evidence": ok.attrs["evidence"]})
    assert any("date" in e for e in s.validate_edge(no_date, nodes))


def test_unknown_types_rejected():
    s = Schema.load()
    assert s.validate_node(GraphNode("X", "blob", "?", "", None, False, {}))
    assert s.validate_edge(GraphEdge("a", "b", "LIKES", {}), {})


def test_artifact_roundtrip_is_deterministic(tmp_path):
    nodes = _nodes()
    edges = [GraphEdge("D1", "D1/СТ.1", "CONTAINS", {})]
    write_graph(tmp_path / "a", list(reversed(nodes)), edges)
    write_graph(tmp_path / "b", nodes, edges)
    for f in ("nodes.jsonl", "edges.jsonl"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    n2, e2 = read_graph(tmp_path / "a")
    assert sorted(n.node_id for n in n2) == sorted(n.node_id for n in nodes) and e2 == edges


# --- additional coverage ---------------------------------------------------------------------

_EV = Evidence("D1", 3, 9, "тексти").to_json()


def test_edge_types_order_and_ids():
    s = Schema.load()
    assert EDGE_TYPES == ["CONTAINS", "REFERS_TO", "AMENDS", "REPEALS", "REGULATES", "DEFINES",
                          "RELATED"]
    assert s.edge_types == EDGE_TYPES and s.edge_type_id["RELATED"] == 6


def test_edge_attr_checks():
    s = Schema.load()
    nodes = {n.node_id: n for n in _nodes()}
    d2 = GraphNode("D2", "document", "документ", "Кодекс", "D2", True, {})
    nodes["D2"] = d2
    assert s.validate_edge(GraphEdge("D1", "D1/СТ.1", "CONTAINS", {}), nodes) == []
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "REPEALS",
                                     {"evidence": _EV, "date": "2024-01-31"}), nodes) == []
    bad_date = s.validate_edge(GraphEdge("D1/СТ.1", "D2", "REPEALS",
                                         {"evidence": _EV, "date": "2024-02-30"}), nodes)
    assert any("date" in e for e in bad_date)
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "AMENDS",
                                     {"evidence": _EV, "date": "2024-01-31", "op": "add"}), nodes) == []
    assert any("op" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "AMENDS", {"evidence": _EV, "date": "2024-01-31", "op": "zap"}),
        nodes))
    assert any("evidence" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": {"doc_id": "D1", "start": 0}}), nodes))
    assert any("evidence" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {}), nodes))
    assert s.validate_edge(GraphEdge("D1", "D2", "RELATED", {"weight": 0.83}), nodes) == []
    for w in (1.5, -0.1, "0.5", True, float("nan")):
        assert s.validate_edge(GraphEdge("D1", "D2", "RELATED", {"weight": w}), nodes)
    # endpoints: missing node, wrong kind (DEFINES must target a concept), self-loop
    assert s.validate_edge(GraphEdge("D1", "NOPE", "RELATED", {"weight": 0.5}), nodes)
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "DEFINES", {"evidence": _EV}), nodes)
    assert s.validate_edge(GraphEdge("D1", "D1", "RELATED", {"weight": 0.5}), nodes)


def test_validate_node():
    s = Schema.load()
    assert all(s.validate_node(n) == [] for n in _nodes())
    assert s.validate_node(GraphNode("E/x", "entity", "organ", "Сенат", None, False, {})) == []
    assert s.validate_node(GraphNode("E/y", "entity", "robot", "?", None, False, {}))
    assert s.validate_node(GraphNode("D9", "document", "", "?", "D9", False, {}))


def test_json_roundtrip_of_records():
    n = GraphNode("D1/СТ.1", "element", "статья", "Мақсад", "D1", False, {"level": 2})
    e = GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": _EV})
    assert GraphNode.from_json(json.loads(json.dumps(n.to_json()))) == n
    assert GraphEdge.from_json(json.loads(json.dumps(e.to_json()))) == e
    assert Evidence.from_json(_EV) == Evidence("D1", 3, 9, "тексти")


def test_artifact_canonical_order_bytes_and_extras(tmp_path):
    nodes = _nodes()
    edges = [GraphEdge("D1/СТ.1", "CON/sanatsiya", "DEFINES", {"evidence": _EV}),
             GraphEdge("D1", "D1/СТ.1", "RELATED", {"weight": 0.4}),
             GraphEdge("D1", "D1/СТ.1", "CONTAINS", {})]
    d = write_graph(tmp_path / "g", nodes, list(reversed(edges)), extra_files={"stats.json": b"{}"})
    lines = (d / "edges.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["type"] for x in lines] == ["CONTAINS", "RELATED", "DEFINES"]
    assert "Қонун" in (d / "nodes.jsonl").read_text(encoding="utf-8")        # ensure_ascii=False
    assert (d / "stats.json").read_bytes() == b"{}"
    assert not list(d.glob("*.tmp"))
    n2, e2 = read_graph(d)
    assert [n.node_id for n in n2] == sorted(n.node_id for n in nodes)
    assert [e.type for e in e2] == ["CONTAINS", "RELATED", "DEFINES"]


def test_artifact_rejects_duplicates_and_bad_extra_names(tmp_path):
    with pytest.raises(ValueError):
        write_graph(tmp_path / "x", _nodes() + _nodes()[:1], [])
    with pytest.raises(ValueError):
        write_graph(tmp_path / "y", _nodes(), [], extra_files={"../evil": b""})


# --- fix round 1 ------------------------------------------------------------------------------

def _nodes2():
    nodes = {n.node_id: n for n in _nodes()}
    nodes["D2"] = GraphNode("D2", "document", "документ", "Кодекс", "D2", True, {})
    return nodes


def test_required_vs_optional_attrs():
    s, nodes = Schema.load(), _nodes2()
    # AMENDS: date is optional now (R4), op required
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "AMENDS", {"evidence": _EV, "op": "edit"}), nodes) == []
    assert any("op" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "AMENDS", {"evidence": _EV}), nodes))
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "AMENDS",
                                     {"evidence": _EV, "op": "edit", "date": None}), nodes) == []
    # optional attrs type-checked when present
    assert any("date" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "AMENDS", {"evidence": _EV, "op": "edit", "date": "2024-02-30"}), nodes))
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "REFERS_TO",
                                     {"evidence": _EV, "relation": "version_of",
                                      "evidence_all": [_EV, _EV]}), nodes) == []
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": _EV, "relation": 3}), nodes)
    assert s.validate_edge(GraphEdge("D1/СТ.1", "D2", "REFERS_TO",
                                     {"evidence": _EV, "evidence_all": [{"doc_id": "D1"}]}), nodes)
    assert s.validate_edge(GraphEdge("D1", "D2", "RELATED", {"weight": 0.2, "attached": True}), nodes) == []
    assert s.validate_edge(GraphEdge("D1", "D2", "RELATED", {"weight": 0.2, "attached": "yes"}), nodes)
    assert s.validate_edge(GraphEdge("D1", "D2", "RELATED", {"weight": None}), nodes)


def test_every_known_attr_is_type_checked_and_unknown_attrs_rejected():
    s, nodes = Schema.load(), _nodes2()
    assert any("date" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": _EV, "date": "garbage"}), nodes))
    assert any("weight" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D1", "CONTAINS", {"weight": 7}), nodes))
    assert any("unknown attr" in e for e in s.validate_edge(
        GraphEdge("D1/СТ.1", "D2", "REFERS_TO", {"evidence": _EV, "evidnce": _EV}), nodes))


def test_evidence_instance_accepted():
    s, nodes = Schema.load(), _nodes2()
    ev = Evidence("D1", 0, 5, "санац")
    assert s.validate_edge(GraphEdge("D1/СТ.1", "CON/sanatsiya", "DEFINES", {"evidence": ev}), nodes) == []
    assert s.validate_edge(GraphEdge("D1/СТ.1", "CON/sanatsiya", "DEFINES",
                                     {"evidence": Evidence("D1", 5, 2, "x")}), nodes)


def _cfg():
    return copy.deepcopy(config.load("graph"))


def test_schema_init_guards():
    cfg = _cfg()
    cfg["edge_types"]["RELATED"]["optional"].append("colour")
    with pytest.raises(ValueError, match="checker"):
        Schema(cfg)
    cfg = _cfg()
    cfg["edge_types"]["RELATED"]["id"] = 0
    with pytest.raises(ValueError, match="ids"):
        Schema(cfg)
    cfg = _cfg()
    cfg["edge_types"]["RELATED"]["id"] = 9
    with pytest.raises(ValueError, match="ids"):
        Schema(cfg)
    cfg = _cfg()
    del cfg["edge_types"]["CONTAINS"]["id"]
    with pytest.raises(ValueError, match="id"):
        Schema(cfg)


def test_edge_types_ordered_by_pinned_id_not_yaml_order():
    cfg = _cfg()
    cfg["edge_types"] = dict(reversed(list(cfg["edge_types"].items())))
    s = Schema(cfg)
    assert s.edge_types == EDGE_TYPES and s.edge_type_id == {t: i for i, t in enumerate(EDGE_TYPES)}


def test_vertex_types_are_data():
    cfg = _cfg()
    cfg["vertex_types"]["concept"] = ["понятие"]
    s = Schema(cfg)
    assert s.validate_node(GraphNode("C/x", "concept", "понятие", "x", None, False, {})) == []
    assert s.validate_node(GraphNode("C/y", "concept", "термин", "y", None, False, {}))
    assert Schema.load().entity_types == {"organ", "position", "procedure", "legal_category"}


def test_external_passthrough_and_non_bool_flagged():
    n = GraphNode.from_json({"node_id": "D3", "kind": "document", "type": "документ", "title": "",
                             "doc_id": None, "external": "no", "attrs": {}})
    assert n.external == "no"
    assert any("external" in e for e in Schema.load().validate_node(n))


def test_write_graph_rejects_duplicate_edges(tmp_path):
    edges = [GraphEdge("D1", "D1/СТ.1", "RELATED", {"weight": 0.4}),
             GraphEdge("D1", "D1/СТ.1", "RELATED", {"weight": 0.9})]
    with pytest.raises(ValueError, match="duplicate edge"):
        write_graph(tmp_path / "g", _nodes(), edges)
    # same endpoints, different type: fine
    write_graph(tmp_path / "h", _nodes(), [edges[0], GraphEdge("D1", "D1/СТ.1", "CONTAINS", {})])


def test_artifact_records_schema_and_detects_mismatch(tmp_path):
    d = write_graph(tmp_path / "g", _nodes(), [GraphEdge("D1", "D1/СТ.1", "CONTAINS", {})])
    assert read_schema_meta(d)["edge_types"] == EDGE_TYPES
    with pytest.raises(ValueError):
        write_graph(tmp_path / "x", _nodes(), [], extra_files={"schema.json": b"{}"})
    cfg = _cfg()
    cfg["edge_types"]["CONTAINS"]["id"], cfg["edge_types"]["RELATED"]["id"] = 6, 0
    with pytest.raises(SchemaMismatch):
        read_graph(d, schema=Schema(cfg))
    assert read_graph(d, schema=Schema(cfg), check_schema=False)[1][0].type == "CONTAINS"


def test_evidence_instance_serialized_in_artifact(tmp_path):
    e = GraphEdge("D1/СТ.1", "CON/sanatsiya", "DEFINES", {"evidence": Evidence("D1", 0, 5, "санац")})
    d = write_graph(tmp_path / "g", _nodes(), [e])
    _, e2 = read_graph(d)
    assert e2[0].attrs["evidence"] == Evidence("D1", 0, 5, "санац").to_json()
