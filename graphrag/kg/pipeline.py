"""THE edge-regeneration path of the M2 graph (Ruling R31 (2); M2 final review I2).

    extract_all(records, cfg=None, llm_factory=None) -> Extracted
        Tasks 3-6 extractors on one context: ActRegistry -> REFERS_TO (+ registry version / translation edges)
        -> AMENDS / REPEALS (the same amendment context) -> entities / concepts (DEFINES / REGULATES). Every LLM
        decision comes from cache/kg (KgLLM(allow_api=False)): a miss raises graphrag.kg.llm.KgCacheMiss.
    r11_filter(edges) -> (kept, dropped)
        Ruling R11 / R24 (2), implemented once: a REFERS_TO {relation: amended} whose (src, dst) is the pair of
        an AMENDS edge is dropped (AMENDS supersedes it); cited_in_amendment citations are kept.

Used by the graph build (graphrag.kg.build.assemble / explicit_for_related), the Task 7 audit
(scripts/audit_graph_edges.regenerate), the Task 8 RELATED exclusion set (scripts/build_related.py via
regenerate) and the tests -- each edge type has exactly one generation path."""
from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.kg.schema import GraphEdge, GraphNode


@dataclass
class Extracted:
    nodes: dict[str, GraphNode]              # every non-structure vertex the extractors made (+ registry docs)
    edges: list[GraphEdge]                   # REFERS_TO (+ registry version edges), AMENDS, REPEALS, DEFINES, REGULATES
    stub_contains: list[GraphEdge]
    repealed_on: dict[str, str]
    llm: dict
    stats: dict = field(default_factory=dict)
    entity_stats: dict = field(default_factory=dict)   # extract_ent.build stats (the audit reports its r20 part)


def extract_all(records, cfg: dict | None = None, llm_factory=None) -> Extracted:
    """The Task 4 / 5 / 6 extractors on `records` (sorted by source file), LLM answers from the cache only.
    `cfg` is accepted for the uniform call signature of Ruling R31 (2); the extractors read config/graph.yaml
    themselves (config.overridden for a different one)."""
    from graphrag.kg import extract_amend, extract_ent, extract_ref
    from graphrag.kg.acts import ActRegistry
    from graphrag.kg.llm import KgLLM
    make = llm_factory or (lambda: KgLLM(cfg, allow_api=False) if cfg is not None else KgLLM(allow_api=False))
    recs = sorted(records, key=lambda r: r.source.file)
    reg = ActRegistry.from_records(recs)
    llm_ref = make()
    refs, ctx = extract_ref.extract_all(recs, reg, llm_ref)
    ctx.llm = None
    am, new, ctx = extract_amend.extract_all(recs, reg, ctx)
    llm_ent = make()
    ent_nodes, ent_edges, ent_stats = extract_ent.build(recs, llm_ent)
    nodes: dict[str, GraphNode] = {}
    for n in reg.nodes() + ctx.stub_nodes() + list(new) + list(ent_nodes):
        nodes.setdefault(n.node_id, n)
    edges = refs + reg.edges() + am + ent_edges
    llm = {"ref_resolve": llm_ref.stats_json() if hasattr(llm_ref, "stats_json") else None,
           "entities": llm_ent.stats_json() if hasattr(llm_ent, "stats_json") else None}
    return Extracted(nodes, edges, ctx.contains_edges(), dict(ctx.repealed_on), llm,
                     {"violations_repeal_without_date": len(ctx.violations),
                      "entity_schema_errors": ent_stats.get("schema_errors", 0)},
                     ent_stats)


def r11_filter(edges: list[GraphEdge]) -> tuple[list[GraphEdge], list[GraphEdge]]:
    """(kept, dropped), order preserved: dropped = REFERS_TO {relation: amended} with the (src, dst) of an
    AMENDS edge in `edges`."""
    amk = {(e.src, e.dst) for e in edges if e.type == "AMENDS"}
    kept, dropped = [], []
    for e in edges:
        if e.type == "REFERS_TO" and e.attrs.get("relation") == "amended" and (e.src, e.dst) in amk:
            dropped.append(e)
        else:
            kept.append(e)
    return kept, dropped
