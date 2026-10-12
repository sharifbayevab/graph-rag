"""M2 acceptance on the real corpus (plan Task 11, IMPLEMENTATION.md §11 step 3, Ruling R30).

The artifact under test is data/graph/latest.json (M2 accepted 1984bbfb852b; M3-prep, Ruling R31, re-pinned it as
26a7118adb6b; M3 Task 2 as VERSION below -- its delta vs 26a7118adb6b is docs/m3/graph_delta.json):
§3.4 validation, all seven edge types, the RELATED share cap, AMENDS recall against the amendment
instructions of the corpus, the per-type precision gate of the committed audit replay (and that the replay
audited this very graph), byte-identical offline rebuild with 0 API calls, Neo4j parity, and depth-3 Cypher
chains "article <- amending act's element <- its article <- the amending act" verified against the corpus.

Marker corpus (reads data/records, the LLM caches and the vector caches; never calls an API). The Neo4j
tests are also marked integration (dev Neo4j bolt :7688 only; read-only)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

import pytest

from graphrag import config

pytestmark = pytest.mark.corpus

VERSION = "40d525a97cb9"   # M3 Task 2: rebuild after digest isolation (Task 1 + M3-R2), nesting-role consistency
                           # (Task 2) and the query-side fixes (Task 0)
PREV_VERSION = "26a7118adb6b"           # Ruling R31 (content identical to M2's 1984bbfb852b)
M2_VERSION = "1984bbfb852b"
GATE = 0.90                       # precision gate per edge type (plan Task 7, §3.4 acceptance table)
AUDITED = ("AMENDS", "DEFINES", "REFERS_TO", "REGULATES", "REPEALS")
ARTIFACT_FILES = ("nodes.jsonl", "edges.jsonl", "schema.json", "x.npy", "pyg.pt", "stats.json")

# three depth-3 chains of the M2 report (docs/m2/report.md §7): amending act, its article / part, target
CHAINS = [
    # (amending act, its article, its part (AMENDS src), op, amended article (EXT stub), act words, number)
    ("QONUNLAR-a325e7c6", "QONUNLAR-a325e7c6/СТ.1", "QONUNLAR-a325e7c6/СТ.1/Ч.1", "add",
     "EXT/jinoyat kodeks/СТ.122.1", "Жиноят кодекси", "122¹"),
    ("QONUNLAR-68773ee2", "QONUNLAR-68773ee2/СТ.1", "QONUNLAR-68773ee2/СТ.1/Ч.1", "replace",
     "EXT/maʼmuriy javobgarlik kodeks/СТ.219", "Маъмурий жавобгарлик тўғрисидаги кодекси", "219"),
    ("QONUNLAR-dae564b2", "QONUNLAR-dae564b2/СТ.1", "QONUNLAR-dae564b2/СТ.1/Ч.1", "edit",
     "EXT/isteʼmol kredit toʼgʼrisida/СТ.15", "«Истеъмол кредити тўғрисида»ги", "15"),
]

CHAIN_QUERY = """
MATCH path = (x:Element {type: 'статья'})<-[r:AMENDS|REPEALS]-(p:Element)
             <-[:CONTAINS]-(s:Element {type: 'статья'})<-[:CONTAINS]-(d:Document)
RETURN d.node_id AS act, s.node_id AS article, p.node_id AS element, type(r) AS rel, r.op AS op,
       r.date AS date, r.evidence AS evidence, x.node_id AS target, length(path) AS depth
ORDER BY act, target
"""


# --- fixtures ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latest():
    return json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def art_dir(latest):
    return config.path(latest["dir"])


@pytest.fixture(scope="module")
def artifact(art_dir):
    from graphrag.kg.artifact import read_graph
    return read_graph(art_dir)


@pytest.fixture(scope="module")
def stats(art_dir):
    return json.loads((art_dir / "stats.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    from graphrag.data.ingest import read_records
    return read_records(config.path("data/records"))


@pytest.fixture(scope="module")
def audit():
    return json.loads(config.path("data/graph/audit.json").read_text(encoding="utf-8"))


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- §3.4 validation, edge types, latent share --------------------------------------------------------

def test_artifact_files_match_latest(latest, art_dir):
    assert latest["version"] == VERSION
    assert {f: _sha(art_dir / f) for f in ARTIFACT_FILES} == latest["sha256"]


def test_artifact_is_pinned_and_differs_from_m2_only_by_the_classified_delta(latest, art_dir):
    import scripts.graph_delta as GD
    import scripts.restore_artifacts as RA
    man = RA.read_manifest(config.path("docs/m3/artifacts.sha256"))     # "<sha256>  <path>" lines
    for f in ARTIFACT_FILES:
        assert man[f"data/graph/{VERSION}/{f}"] == latest["sha256"][f], f
    want = json.loads(config.path("docs/m3/graph_delta.json").read_text(encoding="utf-8"))
    assert GD.graph_delta(config.path(f"data/graph/{PREV_VERSION}"), art_dir) == want


def test_validate_graph_is_empty(artifact, records, stats):
    from graphrag.kg import build as B
    from graphrag.kg.features import RecordIndex
    from graphrag.kg.schema import Schema
    cfg = config.load("graph")
    nodes, edges = artifact
    g = B.Graph(nodes=nodes, edges=edges, rindex=RecordIndex(records), schema=Schema(cfg), cfg=cfg)
    assert B.validate(g) == []
    assert stats["validation"] == {"examples": [], "violations": 0}


def test_all_seven_edge_types_present(artifact, stats):
    from graphrag.kg.schema import Schema
    _, edges = artifact
    by_type = Counter(e.type for e in edges)
    declared = set(Schema(config.load("graph")).edge_types)
    assert len(declared) == 7 and set(by_type) == declared
    assert all(by_type[t] > 0 for t in declared)
    assert dict(by_type) == stats["edges_by_type"]


def test_related_share_at_most_the_cap(artifact, stats):
    _, edges = artifact
    cap = float(config.load("graph")["related"]["max_latent_share"])
    related = [e for e in edges if e.type == "RELATED"]
    latent = [e for e in related if not e.attrs.get("attached")]
    share = len(latent) / len(edges)
    assert cap <= 0.30 and share <= cap
    assert round(share, 6) == stats["related"]["share_without_attached"]
    assert stats["related_source"]["matches_adopted"] is True


# --- AMENDS recall ------------------------------------------------------------------------------------

def amendment_instructions(records):
    """The amendment instructions of the corpus (the "пункты-изменения" of plan Task 11), defined explicitly:
    every unquoted operation verb formula (config/graph.yaml patterns.amend.ops, uz, plus ops_ru) inside the
    amendment-context nodes (extract_ref.amendment_context) of the law texts that are not translations, outside
    repeal formulas -- the Task 5 operation-clause count (1726 in task-5-report.md). Also the numbered `N)`
    пункт / подпункт nodes inside amendment-context articles (944 in the Task 5 reconciliation).
    Returns (instructions [(doc_id, verb start, verb end, op)], points [(doc_id, node_id)])."""
    from graphrag.kg import extract_amend as X
    from graphrag.kg import extract_ref
    from graphrag.kg.acts import ActRegistry, is_law_text
    from graphrag.kg.llm import KgLLM
    reg = ActRegistry.from_records(records)
    _, ctx = extract_ref.extract_all(records, reg, KgLLM(allow_api=False))
    ctx.llm = None
    translations = {e.src for e in reg.version_edges if e.attrs.get("relation") == "translation_of"}
    P = X._apats()
    instr, points = [], []
    for rec in records:
        if not is_law_text(rec) or rec.doc_id in translations:
            continue
        doc = X._Doc(rec, ctx)
        text = rec.text
        rep = [(m.start(), m.end()) for k in ("repeal_list", "repeal_inline")
               for m in P[k].finditer(text) if doc.free(m.start())]
        amend = extract_ref.amendment_context(rec, reg, doc.mentions)
        for root in (n for n in rec.structure if n.node_id in amend and n.parent_id not in amend):
            instr += [(rec.doc_id, vs, ve, op) for vs, ve, op in X._verbs(doc, root.start, root.end, "ops")
                      if not any(s <= vs < e for s, e in rep)]
            instr += [(rec.doc_id, vs, ve, op) for vs, ve, op in X._verbs(doc, root.start, root.end, "ops_ru")]
        points += [(rec.doc_id, n.node_id) for n in rec.structure
                   if n.node_id in amend and n.type in X.ITEM_TYPES and re.match(r"\s*\d+\s*\)", text[n.start:n.end])]
    return instr, points


def test_amends_count_covers_the_amendment_instructions(artifact, records):
    _, edges = artifact
    instr, points = amendment_instructions(records)
    amends = [e for e in edges if e.type == "AMENDS"]
    assert len(instr) > 1000 and len(points) > 900
    assert len(amends) >= 0.8 * len(instr)                               # the plan's check
    # instruction-level recall: the verb formula lies inside an AMENDS evidence span (evidence or evidence_all)
    spans: dict[str, list] = {}
    for e in amends:
        for ev in [e.attrs["evidence"], *(e.attrs.get("evidence_all") or [])]:
            spans.setdefault(ev["doc_id"], []).append((ev["start"], ev["end"]))
    covered = sum(1 for d, vs, ve, _ in instr if any(a <= vs and ve <= b for a, b in spans.get(d, ())))
    assert covered / len(instr) >= 0.8
    # every numbered instruction point is the source (or an ancestor of the source) of an AMENDS / REPEALS edge,
    # except the one Қарори repeal item (a non-law act without a vertex, task-5-report.md reconciliation)
    srcs = {e.src for e in edges if e.type in ("AMENDS", "REPEALS")}
    anc = {s.rsplit("/", k)[0] for s in srcs for k in range(s.count("/") + 1)}
    uncovered = [nid for _, nid in points if nid not in anc]
    assert len(uncovered) <= 1


# --- precision gate of the committed audit replay (Ruling R30) -----------------------------------------

def test_audit_precision_per_type_at_least_the_gate(audit, artifact):
    _, edges = artifact
    by_type = Counter(e.type for e in edges)
    assert audit["provenance"]["mode"] == "frozen_replay" and audit["unjudged"] == 0
    assert audit["llm"]["audit"]["api_calls"] == 0
    types = audit["rescore"]["types"]
    assert sorted(types) == sorted(AUDITED)
    for t in AUDITED:
        r = types[t]
        assert r["N"] == by_type[t], t                                 # current population = this artifact
        assert r["precision_target"] >= GATE and r["gate_target"], t
        assert r["precision_label"] >= GATE and r["gate_label"], t
    assert audit["contains"]["violations"] == 0 and audit["contains"]["total"] == by_type["CONTAINS"]
    assert audit["op_check"]["population"]["contradictions"] == 0
    rel = json.loads(config.path("data/graph/related_stats.json").read_text(encoding="utf-8"))
    assert rel["gate_status"] == "passed_theta_b"                  # RELATED: θ_b near-θ judge ≥ 0.80 (R23)


def test_audit_replay_audited_this_graph_version(audit, artifact, stats):
    """The replay's commit holds exactly the code / config the version digest of this artifact was built from,
    its tree was clean, and the audited populations it hashed (provenance.edges_sha256) are what the current
    extractors produce offline -- the same edge identities as the artifact's audited types."""
    import scripts.audit_graph_edges as AG
    from graphrag.kg import audit as A
    from graphrag.kg import build as B
    from graphrag.kg.artifact import edge_key
    prov = audit["provenance"]
    assert prov["code_dirty"] == []
    commit = prov["commit"]
    vin = stats["version_inputs"]
    root = config.path(".")
    for path, digest in {**vin["code"], **vin["config"]}.items():
        blob = subprocess.run(["git", "show", f"{commit}:./{path}"], cwd=root, capture_output=True, check=True).stdout
        name = path.rsplit("/", 1)[-1]
        if path.startswith("config/") and name in B.VERSION_CONFIG_EXCLUDE:
            blob = B._without_sections(blob.decode("utf-8"), B.VERSION_CONFIG_EXCLUDE[name]).encode("utf-8")
        assert hashlib.sha256(blob).hexdigest() == digest, path
    frozen = gzip.decompress(config.path(prov["frozen_sample"]).read_bytes())
    assert hashlib.sha256(frozen).hexdigest() == prov["frozen_sample_sha256"]     # the committed judged sample
    _, _, reg_edges, _, _ = AG.regenerate()
    pops = A.populations(reg_edges)
    digest = hashlib.sha256("".join(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                                    for s in sorted(pops) for e in pops[s]).encode("utf-8")).hexdigest()
    assert digest == prov["edges_sha256"]
    _, edges = artifact
    art_pops = A.populations(edges)
    assert {s: len(v) for s, v in art_pops.items()} == audit["current_populations"]
    assert {edge_key(e) for v in art_pops.values() for e in v} == \
        {edge_key(e) for v in pops.values() for e in v}


# --- determinism: an offline rebuild is byte-identical, 0 API calls -----------------------------------------

def test_offline_rebuild_is_byte_identical(tmp_path, latest, art_dir):
    env = {k: v for k, v in os.environ.items() if k not in ("GRL_KG_API", "OPENAI_API_KEY")}
    env["HF_HUB_OFFLINE"] = "1"
    out = tmp_path / "graph"
    p = subprocess.run([sys.executable, "scripts/build_graph.py", "--out", str(out)], cwd=config.path("."),
                       env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-2000:]
    run = json.loads(p.stdout)
    assert run["version"] == latest["version"]
    assert run["llm_offline"] == {"api_calls": 0, "misses": 0} and run["validation"]["violations"] == 0
    new = out / latest["version"]
    for f in ARTIFACT_FILES:
        assert (new / f).read_bytes() == (art_dir / f).read_bytes(), f


# --- Neo4j (dev bolt :7688, read-only) -------------------------------------------------------------------

@pytest.fixture(scope="module")
def neo4j():
    from graphrag.kg.store import GraphStore
    from graphrag.stores.guard import dev_neo4j_problem
    assert dev_neo4j_problem(config.NEO4J_URI) is None, "acceptance reads only the dev Neo4j (bolt :7688)"
    g = GraphStore()
    yield g
    g.close()


@pytest.mark.integration
def test_neo4j_full_parity_with_the_artifact(neo4j, artifact, latest):
    nodes, edges = artifact
    assert neo4j.active_version() == latest["version"]
    par = neo4j.parity(nodes, edges, latest["version"])            # full check (sample=None)
    assert par["ok"], {k: par.get(k) for k in ("count_diffs", "node_mismatches", "edge_mismatches")}
    assert par["checked_nodes"] == len(nodes) and par["checked_edges"] == len(edges)
    assert par["node_mismatches"] == 0 and par["edge_mismatches"] == 0
    by_type = Counter(e.type for e in edges)
    assert {t: v["neo4j"] for t, v in par["edges_by_type"].items()} == dict(by_type)
    assert all(v == {} for v in par["count_diffs"].values())


@pytest.mark.integration
def test_depth3_amendment_chains_in_neo4j_match_the_corpus(neo4j, records, artifact):
    from graphrag.data.normalize import canon_num
    with neo4j.driver.session() as s:
        rows = [dict(r) for r in s.run(CHAIN_QUERY)]
    assert len(rows) >= 3 and all(r["depth"] == 3 for r in rows)
    got = {(r["act"], r["target"]): r for r in rows}
    recs = {r.doc_id: r for r in records}
    _, edges = artifact
    contains = {(e.src, e.dst) for e in edges if e.type == "CONTAINS"}
    for act, article, element, op, target, act_words, number in CHAINS:
        r = got[(act, target)]
        assert (r["article"], r["element"], r["rel"], r["op"]) == (article, element, "AMENDS", op)
        rec = recs[act]
        ev = json.loads(r["evidence"])
        node = {n.node_id: n for n in rec.structure}[element]
        assert ev["doc_id"] == act and rec.text[ev["start"]:ev["end"]] == ev["quote"]
        assert node.start <= ev["start"] and ev["end"] <= node.end
        assert node.parent_id == article and {n.node_id: n for n in rec.structure}[article].parent_id == act
        head = rec.text[node.start:ev["start"]]                          # the act is named before the verb
        assert act_words in head
        assert number + "-модда" in ev["quote"]                          # the article the instruction names
        assert target.rsplit("/", 1)[1] == "СТ." + canon_num(number)
        assert r["date"] == rec.meta.date                                # the amending law's adoption date
        assert (target.rsplit("/", 1)[0], target) in contains             # the EXT act contains the article
