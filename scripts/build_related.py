"""M2 Task 8: RELATED latent edges (cross-document ANN, calibrated θ, capped degree; Ruling R21).

    python scripts/build_related.py [--out data/graph] [--refresh-vectors] [--no-judge] [--device cpu|mps]
                                    [--manual-sheet FILE]
    GRL_KG_API=1 python scripts/build_related.py        # the single live judge run (answers cached)

Vectors: bge-m3 chunk vectors read from the v2 Qdrant (GRL_QDRANT_URL, default :6343; the prod :6333 is
refused) and cached to data/graph/chunk_vecs.npy + chunk_vecs_ids.json (chunk-id row order, collection,
sha256 of the embedded texts); the cache is reused when its ids and text digest match data/records.
Explicit edges (the exclusion set and the share denominator) are regenerated offline exactly like the
Task 7 audit (scripts/audit_graph_edges.regenerate: extractor LLM answers from cache/kg, 0 calls) plus the
structural CONTAINS edges of the records. Judge: KgLLM pass "audit_related" (cache/kg/audit_related).
Manual labels of the first `judge.manual` near-θ items: cache/kg/audit_related_manual.json, made before
the live run. Outputs (--out): related_stats.json, related_edges.jsonl, related_items.jsonl."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graphrag import config  # noqa: E402
from graphrag.kg import semantic as S  # noqa: E402
import research_related as R  # noqa: E402 -- Task 8 research helpers (Ruling R31 (2))

MANUAL_FILE = "cache/kg/audit_related_manual.json"
DEV_FILE = "cache/kg/audit_related_dev.jsonl"          # dev set: R22 = the 120 R21 pairs; R23 + the 120 R22 fresh pairs
COLLECTION = "grl_chunks__baai_bge_m3"


def _git(*args) -> str:
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001 -- provenance only
        return "?"


def text_digest(chunks) -> str:
    field = config.load("data")["embed_field"]
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(c.chunk_id.encode("utf-8") + b"\0" + (getattr(c, field) or c.text_norm).encode("utf-8") + b"\n")
    return h.hexdigest()


def load_vectors(chunks, out: Path, refresh: bool) -> tuple[list[str], np.ndarray, dict]:
    ids = sorted(c.chunk_id for c in chunks)
    dig = text_digest(chunks)
    npy, meta_f = out / "chunk_vecs.npy", out / "chunk_vecs_ids.json"
    if not refresh and npy.exists() and meta_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("ids") == ids and meta.get("text_sha256") == dig:
            X = np.load(npy)
            return ids, X, {k: v for k, v in meta.items() if k != "ids"} | {"source": "cache"}
    url = config.QDRANT_URL
    if ":6333" in url:
        raise SystemExit(f"refusing the production Qdrant {url}; the v2 store is :6343")
    from graphrag.retrieval.vector import client
    cl = client()
    got, off = {}, None
    while True:
        pts, off = cl.scroll(COLLECTION, limit=1024, offset=off, with_vectors=True, with_payload=["chunk_id"])
        for p in pts:
            got[p.payload["chunk_id"]] = np.asarray(p.vector, dtype=np.float32)
        if off is None:
            break
    missing = [i for i in ids if i not in got]
    if missing or len(got) != len(ids):
        raise SystemExit(f"Qdrant {COLLECTION}: {len(missing)} chunks missing, {len(got)} points for {len(ids)} chunks")
    X = np.stack([got[i] for i in ids]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1)
    meta = {"model": config.load("model")["encoder"]["name"], "collection": COLLECTION, "qdrant": url,
            "dim": int(X.shape[1]), "rows": len(ids), "text_sha256": dig,
            "embed_field": config.load("data")["embed_field"], "norm_min": float(norms.min()),
            "norm_max": float(norms.max()), "npy_sha256": None}
    out.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    meta["npy_sha256"] = hashlib.sha256(npy.read_bytes()).hexdigest()
    meta_f.write_text(json.dumps({**meta, "ids": ids}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return ids, X, meta | {"source": "qdrant"}


def load_topic_vectors(recs, ids: list[str], out: Path, speakers: set, drop_crumb, device: str | None,
                       refresh: bool) -> tuple[np.ndarray, dict]:
    """R22 (v2): bge-m3 vectors of semantic.topic_text (breadcrumb without session / speaker parts, speaker
    labels removed), cached to chunk_vecs_topic.npy + chunk_vecs_topic_ids.json (row order = sorted chunk
    ids, sha256 of the texts); re-encoded only when ids or text digest differ (bge-m3 ~9 min on MPS)."""
    sec = {c.chunk_id: r.meta.section for r in recs for c in r.chunks}
    by_id = {c.chunk_id: c for r in recs for c in r.chunks}
    texts = [S.topic_text(by_id[i].index_text, by_id[i].text_norm,
                          speakers if sec[i] == "STENOGRAMMA" else frozenset(), drop_crumb) for i in ids]
    h = hashlib.sha256()
    for i, t in zip(ids, texts):
        h.update(i.encode("utf-8") + b"\0" + t.encode("utf-8") + b"\n")
    dig = h.hexdigest()
    npy, meta_f = out / "chunk_vecs_topic.npy", out / "chunk_vecs_topic_ids.json"
    if not refresh and npy.exists() and meta_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("ids") == ids and meta.get("text_sha256") == dig:
            return np.load(npy), {k: v for k, v in meta.items() if k != "ids"} | {"source": "cache"}
    from graphrag.data.embed import Encoder
    enc = Encoder(device=device if device and device != "cpu" else None)
    X = enc.encode_passages(texts, progress=True)
    out.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    meta = {"model": enc.model_name, "rows": len(ids), "dim": int(X.shape[1]), "text_sha256": dig,
            "text": "semantic.topic_text", "npy_sha256": hashlib.sha256(npy.read_bytes()).hexdigest()}
    meta_f.write_text(json.dumps({**meta, "ids": ids}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    return X, meta | {"source": "encoded"}


def breadcrumb(struct: dict, nid: str) -> str:
    parts, x = [], struct.get(nid)
    while x is not None and x.parent_id is not None:
        lab = x.type + (f" {x.number}" if x.number else "")
        if x.title and x.title != x.number:
            lab += f" «{R._cap(x.title, 80)}»"
        parts.append(lab)
        x = struct.get(x.parent_id)
    return " / ".join(reversed(parts))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    t0 = time.time()
    g = config.load("graph")
    cfg = {**S.DEFAULTS, **(g.get("related") or {})}
    jcfg = {**S.DEFAULTS["judge"], **(cfg.get("judge") or {})}
    if jcfg.get("mode") != "ab":                          # R29 F: the R21 / R22 raise schedule is retired
        raise SystemExit(f"related.judge.mode must be 'ab' (Ruling R23 schedule), not {jcfg.get('mode')!r}")
    out = config.path(opt("--out", "data/graph"))
    from audit_graph_edges import regenerate
    recs, nodes, edges, stub_contains, ext_stats = regenerate()
    chunks = [c for r in recs for c in r.chunks]
    ids, X, vmeta = load_vectors(chunks, out, "--refresh-vectors" in argv)
    by_id = {c.chunk_id: c for c in chunks}
    doc_of, ptype, struct, parent_of, title = {}, {}, {}, {}, {}
    for r in recs:
        title[r.doc_id] = (r.meta.section, r.meta.type, r.meta.title or "")
        for n in r.structure:
            struct[n.node_id] = n
            parent_of[n.node_id] = n.parent_id
        for c in r.chunks:
            doc_of[c.chunk_id] = r.doc_id
            ptype[c.chunk_id] = struct[c.parent].type
    rules = S.compile_sentence_rules(cfg.get("sentence_rules"))
    speakers = S.speaker_names(recs, cfg)
    names = {w for x in speakers for w in x.split()}
    sec_of = {c.chunk_id: r.meta.section for r in recs for c in r.chunks}
    reasons = {i: S.chunk_reason(by_id[i], ptype[i], sec_of[i], cfg, rules, speakers, names) for i in ids}
    r23_classes = list(cfg.get("r23_classes") or [])
    class_removal = None
    if r23_classes:                                       # R23: chunks each new class removes on its own
        def reason_fn(spec):
            rs = S.compile_sentence_rules(spec)
            return {i: S.chunk_reason(by_id[i], ptype[i], sec_of[i], cfg, rs, speakers, names) for i in ids}
        class_removal = R.class_removals(reason_fn, cfg.get("sentence_rules") or {}, r23_classes)
    Xs, smeta = X, {"source": "index_text (Qdrant)"}
    if cfg.get("score_vectors") == "topic":
        Xs, smeta = load_topic_vectors(recs, ids, out, speakers, cfg.get("topic_drop_crumb") or [],
                                       opt("--encode-device"), "--refresh-topic-vectors" in argv)
    eligible = np.array([reasons[i] is None for i in ids])
    # explicit graph: structural CONTAINS + stub CONTAINS + extracted edges (REFERS_TO incl. version / translation,
    # AMENDS, REPEALS, REGULATES, DEFINES)
    n_struct_contains = sum(1 for r in recs for n in r.structure if n.parent_id is not None)
    explicit_counts = dict(sorted(Counter(e.type for e in edges).items()))
    explicit_counts["CONTAINS"] = n_struct_contains + len(stub_contains)
    n_explicit = sum(explicit_counts.values())
    linked = [(e.src, e.dst) for e in edges if e.type == "REFERS_TO"
              and e.attrs.get("relation") in ("version_of", "translation_of")]
    groups = S.doc_groups(sorted(title), linked)
    multi = Counter(groups.values())
    explicit = S.explicit_pairs(edges)
    grid = S.theta_grid(cfg["theta_grid"])
    bins = S.grid_bins(grid)
    device = opt("--device", "cpu")
    t1 = time.time()
    idx = S.build_index(ids, X, [by_id[i].parent for i in ids], [doc_of[i] for i in ids], groups, eligible,
                        explicit, parent_of, int(cfg["k_ann"]), int(cfg["k_rel"]), int(cfg["block"]), bins, device,
                        dedup=cfg.get("dedup"), template=cfg.get("template"), score_X=Xs,
                        anisotropy_pcs=cfg.get("anisotropy_pcs"), mutual=bool(cfg.get("mutual_knn")))
    t_ann = time.time() - t1
    assert idx.chunk_ids == ids                           # rows already in sorted chunk-id order
    for cid in idx.templates:
        reasons[cid] = "template"
    eligible = idx.eligible
    null = S.null_quantile(idx.score_X, idx.group, idx.eligible, int(cfg["null_pairs"]), float(cfg["null_quantile"]),
                           int(cfg["seed"]))
    n_vert = len({by_id[i].parent for i, ok in zip(ids, eligible) if ok})
    cache: dict = {}

    def run(theta):
        if theta not in cache:
            cache[theta] = idx.result(theta)
        return cache[theta]

    def evaluate(theta):
        kept, _ = run(theta)
        return len(kept), S.degree_stats(S.degrees(kept))["median"]

    cal = S.calibrate_theta(evaluate, n_explicit, grid, float(cfg["max_latent_share"]), int(cfg["k_rel"]),
                            null["value"])
    # R22: dev set = the pairs judged in the R21 run; score = max over eligible chunk pairs of the two
    # vertices (score_X rows, rounded like the ANN); pairs without eligible chunks at an end are dropped
    rowpos = {c: k for k, c in enumerate(idx.chunk_ids)}
    rows_of: dict = {}
    for k, cid in enumerate(idx.chunk_ids):
        if idx.eligible[k]:
            rows_of.setdefault(by_id[cid].parent, []).append(k)
    Z64 = idx.score_X.astype(np.float64)
    df = config.path(DEV_FILE)
    dev = [json.loads(line) for line in df.read_text(encoding="utf-8").splitlines()] if df.exists() else []
    dev_s = []
    for it in dev:
        a, b = rows_of.get(it["src"], []), rows_of.get(it["dst"], [])
        dev_s.append(float(S._round(Z64[a] @ Z64[b].T).max()) if a and b else -np.inf)
    dev_s = np.array(dev_s)
    dev_y = np.array([bool(it["same_topic"]) for it in dev])
    kept_m = np.isfinite(dev_s)
    dev_info = {"file": DEV_FILE, "pairs": len(dev), "positives": int(dev_y.sum()),
                "dropped_no_eligible_chunk": int((~kept_m).sum()),
                "dropped_positives": int((~kept_m & dev_y).sum()),
                "auc": R.auc(dev_s, dev_y), "auc_ci95_bootstrap": R.bootstrap_auc(dev_s, dev_y, 2000, int(cfg["seed"])),
                "auc_kept": R.auc(dev_s[kept_m], dev_y[kept_m]),
                # R29 F (Task 8 review M1): both AUCs are reported -- `auc` ranks the pairs without an eligible chunk
                # (score -inf) last, so removing a dev negative's chunk raises it by construction; `auc_kept` is over
                # the pairs with eligible chunks at both ends only
                "auc_definitions": {"auc": "all dev pairs; a pair without an eligible chunk at an end scores -inf",
                                    "auc_kept": "dev pairs with eligible chunks at both ends"}}
    theta_start = cal["theta"]
    if len(dev) and kept_m.sum() >= 10:
        w = R.logistic_fit(dev_s[kept_m], dev_y[kept_m])
        dev_info["logistic"] = [round(w[0], 6), round(w[1], 6)]

        def near_weights(t):
            kept_t, _ = run(t)
            ws = np.array([v[0] for v in kept_t.values()])
            return ws[ws <= t + float(jcfg["near_band"]) + 1e-12] if len(ws) else ws

        dt = R.dev_theta(near_weights, w, grid, float(jcfg["gate"]), lo=cal["theta"])
        dev_info.update(dt)
        if cfg.get("dev_theta") and dt["theta_dev"] is not None:
            theta_start = max(theta_start, dt["theta_dev"])
    dev_pairs = {(it["src"], it["dst"]) for it in dev}

    def summarize(theta):
        kept, cand = run(theta)
        deg = S.degrees(kept)
        L = len(kept)
        ws = np.array([t[0] for t in kept.values()]) if kept else np.zeros(0)
        return {"theta": theta, "edges": L, "latent_share": round(L / (n_explicit + L), 6),
                "candidates_before_cap": len(cand.best), "dropped_by_k_rel_cap": len(cand.best) - L,
                "degree": S.degree_stats(deg, n_vert),
                "weight": {"min": float(ws.min()), "p50": float(np.median(ws)), "max": float(ws.max())} if L else None,
                "excluded": {"vertex_pairs": dict(cand.excluded), "chunk_pairs_in_ann": dict(cand.excluded_chunk_pairs),
                             "same_document_chunk_pairs_above_theta": S.count_above(idx.ann.same_doc_hist, grid, theta),
                             "translation_version_chunk_pairs_above_theta": S.count_above(idx.ann.group_hist, grid, theta),
                             "near_duplicate_doc_chunk_pairs_above_theta": S.count_above(idx.ann.dup_hist, grid, theta)},
                "doc_pairs": len({S.pair(doc_of[idx.chunk_ids[t[1]]], doc_of[idx.chunk_ids[t[2]]]) for t in kept.values()}),
                "sections": dict(sorted(Counter("~".join(sorted((title[doc_of[idx.chunk_ids[t[1]]]][0],
                                                                  title[doc_of[idx.chunk_ids[t[2]]]][0])))
                                                for t in kept.values()).items()))}

    def summarize_short(t):
        kept_t, _ = run(t)
        L = len(kept_t)
        d = S.degree_stats(S.degrees(kept_t))
        ws = np.array([v[0] for v in kept_t.values()]) if L else np.zeros(0)
        near = ws[ws <= t + float(jcfg["near_band"]) + 1e-12]
        wl = dev_info.get("logistic")
        return {"theta": t, "edges": L, "latent_share": round(L / (n_explicit + L), 6), "degree_median": d["median"],
                "degree_p99": d["p99"], "degree_max": d["max"],
                "dev_est_near": round(float(R.logistic_p(wl, near).mean()), 4) if wl and len(near) else None}

    def view_end(ci):
        cid = idx.chunk_ids[ci]
        c = by_id[cid]
        sec, typ, ttl = title[doc_of[cid]]
        return {"path": f"{sec} · {typ} · «{R._cap(ttl, 160)}» · {breadcrumb(struct, c.parent)}",
                "text": R._cap(c.text, int(jcfg["text_cap"]))}

    def items_for(kept, strata: dict, tag: str) -> list[dict]:
        its = []
        for s in ("near", "top"):
            for e in strata[s]:
                t = kept[(e.src, e.dst)]
                its.append({"uid": f"{tag}:{s}:{e.src}|{e.dst}", "stratum": s, "round": tag, "src": e.src,
                            "dst": e.dst, "weight": e.attrs["weight"], "chunks": [idx.chunk_ids[t[1]], idx.chunk_ids[t[2]]],
                            "view": {"a": view_end(t[1]), "b": view_end(t[2])}})
        return its

    seed = int(cfg["seed"])
    theta = theta_start
    jtag = str(jcfg.get("tag") or "")
    rounds, all_items, verdicts = [], [], {}
    judge = None
    llm = None
    if "--no-judge" not in argv:
        from graphrag.kg.llm import KgLLM
        llm = KgLLM()
        judge = R.llm_judge(llm)
    mf = config.path(jcfg.get("manual_file") or MANUAL_FILE)
    manual = json.loads(mf.read_text(encoding="utf-8"))["labels"] if mf.exists() else {}
    # Ruling R23 schedule (see graph.yaml related.judge; the R21 / R22 raise loop was removed in the R29 sweep)
    tb = float(jcfg["theta_b"])
    est_tab = {r["theta"]: r["est_near_precision"] for r in dev_info.get("table", [])}
    tb_est = next((r["theta"] for r in dev_info.get("table", []) if r["est_near_precision"] is not None
                   and r["est_near_precision"] >= float(jcfg["theta_b_est"])), None)
    theta_b_info = {"theta_b": tb, "dev_est_at_theta_b": est_tab.get(tb), "theta_b_est_target": float(jcfg["theta_b_est"]),
                    "smallest_theta_with_dev_est_at_target": tb_est}
    budget, planned, judged = int(jcfg["run_budget"]), 0, set()
    excl = set(dev_pairs) if jcfg.get("exclude_dev") else set()
    plan = [("r0", theta_start), ("r1", tb)]

    def judge_items(its, rtag):
        for b in R.make_batches(its, int(jcfg["batch"]), seed, tag=rtag):
            verdicts.update(judge(b))

    for k, (suffix, th) in enumerate(plan):
        kept, _ = run(th)
        rtag = f"{jtag}{suffix}"
        strata = R.sample_strata(S.to_edges(kept), th, {**jcfg, "top": 0}, seed, tag=rtag, exclude=excl | judged)
        strata["top"] = []
        its = items_for(kept, strata, rtag)
        all_items += its
        nb = -(-len(its) // int(jcfg["batch"]))
        info = {"round": k, "tag": rtag, "theta": th, "near_pool": strata["near_pool"], "items": len(its),
                "excluded_dev_pairs": strata.get("excluded_dev_pairs", 0), "planned_calls": nb,
                "sample_seeds": {"near": f"{seed}:near:{rtag}", "batch": f"{seed}:batch:{rtag}"}}
        if k == 0 and opt("--manual-sheet"):
            near_its = its[:int(jcfg["manual"])]
            Path(opt("--manual-sheet")).write_text("".join(json.dumps({"uid": it["uid"], **it["view"]}, ensure_ascii=False)
                                                           + "\n" for it in near_its), encoding="utf-8")
        if judge is None:
            rounds.append(info)
            break
        if planned + nb > budget:
            info["skipped"] = "run_budget"
            rounds.append(info)
            break
        planned += nb
        judge_items(its, rtag)
        judged |= {(it["src"], it["dst"]) for it in its}
        info["near"] = R.precision([verdicts.get(it["uid"]) for it in its])
        info["gate"] = info["near"]["precision"] is not None and info["near"]["precision"] >= float(jcfg["gate"])
        rounds.append(info)
        if info["gate"]:
            break
    judged_rounds = [r for r in rounds if "near" in r]
    if judged_rounds:
        theta, adopt_status = R.adopt_theta(judged_rounds, float(jcfg["gate"]))
        kept, _ = run(theta)
        rtag = f"{jtag}top"
        strata = R.sample_strata(S.to_edges(kept), theta, {**jcfg, "near": 0}, seed, tag=rtag, exclude=excl | judged)
        its = items_for(kept, {"near": [], "top": strata["top"]}, rtag)
        nb = -(-len(its) // int(jcfg["batch"]))
        info = {"round": len(rounds), "tag": rtag, "theta": theta, "stratum": "top", "top_pool": strata["top_pool"],
                "top_min_weight": strata["top_min_weight"], "items": len(its), "planned_calls": nb,
                "sample_seeds": {"top": f"{seed}:top:{rtag}", "batch": f"{seed}:batch:{rtag}"}}
        if planned + nb <= budget:
            planned += nb
            all_items += its
            judge_items(its, rtag)
            info["top"] = R.precision([verdicts.get(it["uid"]) for it in its])
        else:
            info["skipped"] = "run_budget"
        rounds.append(info)
        theta_b_info.update({"adopt_status": adopt_status, "planned_calls": planned, "run_budget": budget})
    else:
        theta = theta_start
    final_theta = theta
    gate_status = "not_judged" if judge is None or theta_b_info.get("adopt_status") is None else theta_b_info["adopt_status"]
    final = summarize(final_theta)
    edges_final = S.to_edges(run(final_theta)[0])
    # schema check of the final edges
    from graphrag.kg.extract_ref import structure_nodes
    from graphrag.kg.schema import Schema
    sch = Schema.load()
    gnodes = {n.node_id: n for r in recs for n in structure_nodes(r)}
    schema_errors = [m for e in edges_final for m in sch.validate_edge(e, gnodes)]
    und = all(e.src < e.dst for e in edges_final)
    man_uids = [it["uid"] for it in all_items if it["round"] == f"{jtag}r0" and it["stratum"] == "near"][:int(jcfg["manual"])]
    man_cmp = [{"uid": u, "manual": manual[u]["same_topic"], "judge": (verdicts.get(u) or {}).get("same_topic")}
               for u in man_uids if u in manual]
    stats = {
        "provenance": {"commit": _git("rev-parse", "HEAD"),
                       "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                       for f in ("graphrag/kg/semantic.py", "scripts/build_related.py",
                                                 "scripts/audit_graph_edges.py", "config/graph.yaml")},
                       "code_dirty": _git("status", "--porcelain", "--", "graphrag", "config", "scripts").splitlines(),
                       "config": {k: v for k, v in cfg.items()}, "vectors": vmeta,
                       "extractors_offline_llm": ext_stats, "device": device,
                       "determinism": "float64 dot products (float32 vectors) rounded to 1e-6 before ranking / thresholds; ties by chunk id "
                                      "(row order = sorted chunk ids) and vertex id"},
        "chunks": {"total": len(ids), "eligible": int(eligible.sum()), "vertices_with_eligible_chunks": n_vert,
                   "excluded_boilerplate": dict(sorted(Counter(r for r in reasons.values() if r).items()))},
        "groups": {"linked_doc_pairs": len(linked), "linked": sorted(map(list, linked)),
                   "linked_multi_doc_groups": sum(1 for v in multi.values() if v > 1),
                   "near_duplicate_doc_pairs": len(idx.near_duplicates), "near_duplicates": idx.near_duplicates,
                   "final_multi_doc_groups": sum(1 for v in Counter(dict(zip(idx.doc, idx.group.tolist())).values()).values()
                                                 if v > 1)},
        "explicit": {"edges_by_type": explicit_counts, "total": n_explicit, "exclusion_pairs": len(explicit)},
        "null": null,
        "calibration": {k: v for k, v in cal.items() if k != "table"}, "grid_table": cal["table"],
        "r22": {"score_vectors": smeta, "anisotropy": idx.anisotropy, "mutual_knn": bool(cfg.get("mutual_knn")),
                "sentence_rule_classes": [r.name for r in rules], "dev": dev_info, "theta_start": theta_start,
                "theta_start_binding": "dev" if theta_start != cal["theta"] else cal["binding"]},
        "r23": {"classes": r23_classes, "class_removals": class_removal, "schedule": theta_b_info,
                "frontier": [summarize_short(t) for t in S.theta_grid(cfg.get("frontier") or {"start": 0.54, "stop": 0.70, "step": 0.01})]}
                if r23_classes else None,
        "calibrated": summarize(cal["theta"]),
        "theta": final_theta, "theta_raised_by_judge": final_theta != cal["theta"], "gate_status": gate_status,
        "final": final,
        "judge": {"model": llm.model if llm else None, "pass": "audit_related", "rounds": rounds,
                  "llm": llm.stats_json() if llm else None,
                  "definitions": {"near": f"random {jcfg['near']} edges with θ < weight <= θ + {jcfg['near_band']}",
                                  "top": f"random {jcfg['top']} of the top max({jcfg['top']}, {jcfg['top_share']}·E) by weight"},
                  "manual_agreement": {"n": len(man_cmp), "agree": sum(1 for x in man_cmp if x["manual"] == x["judge"]),
                                       "manual_same_topic": sum(1 for x in man_cmp if x["manual"]), "items": man_cmp}},
        "schema_errors": len(schema_errors), "schema_error_examples": schema_errors[:5], "undirected_src_lt_dst": und,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "related_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    kept = run(final_theta)[0]
    (out / "related_edges.jsonl").write_text("".join(
        json.dumps({**e.to_json(), "chunks": [idx.chunk_ids[kept[(e.src, e.dst)][1]], idx.chunk_ids[kept[(e.src, e.dst)][2]]]},
                   ensure_ascii=False, sort_keys=True) + "\n" for e in edges_final), encoding="utf-8")
    (out / "related_items.jsonl").write_text("".join(
        json.dumps({**it, "verdict": verdicts.get(it["uid"])}, ensure_ascii=False, sort_keys=True) + "\n"
        for it in all_items), encoding="utf-8")
    print(json.dumps({"theta": final_theta, "gate_status": gate_status, "calibration": stats["calibration"], "null": null,
                      "r22": {k: v for k, v in stats["r22"].items() if k != "dev"} | {"dev": {k: v for k, v in dev_info.items() if k != "table"}},
                      "final": {k: final[k] for k in ("edges", "latent_share", "candidates_before_cap", "excluded")},
                      "degree": {k: final["degree"][k] for k in ("vertices", "median", "p99", "max")},
                      "chunks": stats["chunks"], "groups": {k: v for k, v in stats["groups"].items() if k not in ("near_duplicates", "linked")}, "explicit": stats["explicit"],
                      "rounds": rounds, "llm": stats["judge"]["llm"], "manual": stats["judge"]["manual_agreement"],
                      "schema_errors": len(schema_errors), "ann_seconds": round(t_ann, 1), "seconds": round(time.time() - t0, 1)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
