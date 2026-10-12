"""M7 Task 6c: the Q1 expansion (grl.hard.q1x) -- Phase A: candidates -> probe -> selection -> R32 splits.

    PYTHONPATH=. python scripts/hard_q1x.py make --seed 20261004 --n 250 --exclude data/hard/probe/ids.json \\
        --out data/hard/full/ --workers 2
    -> <out>/q1x_raw.jsonl (HardItem + gold_docs + split; template wording), <out>/q1x_stats.json,
       <out>/q1x_probe/ (rankings of the probed renderings + probe.jsonl, Mac only)

Inputs as Task 4's full-make: document STRUCT texts + the Task 6c article STRUCT texts (grl.env.struct_data,
corpus tokenizer, data.yaml max_tokens), μ⁻¹ (pyg.pt), fragment texts / spans (records), the frozen language counts
<out>/lang_shares.json (R47), the probe qids (excluded in every language).

Splits (R32 / R44 / R45): the fixed document splits are the old set (data/gold/splits.json) + every existing M7 row
(count_raw, version_raw, q2_aug_train, compare_consensus_r1: grl.hard.splits.fixed_doc_splits); a candidate touching
two fixed splits is dropped before selection, a new item touching a fixed document inherits its split, the rest go by
per-type deficit; no document ends in two splits (checked again over everything).

Probe (mandatory, Task 6c): every (candidate, language) rendering must have all@5 = 0 for dense@5 and fused@5. A
STRUCT id is in no index, so a rendering whose gold holds a STRUCT id has all@5 = 0 for every ranking baseline by
construction (asserted: no ranking row is a STRUCT row -- grl.hard.probe.check_rank_rows on the probed rankings);
the renderings with corpus-fragment gold (count_articles_deleted_by_law) are ranked through the U9 path (dev pg 5437 /
qdrant 6343, CPU encoder, <= --workers threads) and dropped when all their gold is in dense@5 (q.vec) or fused@5 (the
pilot-6 pool: rerank_candidates(n = rerank.n, dynamic_k.fuse_w / norm_top_n, L_max)). Done on the candidates before
selection and splitting, so no test-split question is ever evaluated."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402

OLD_SPLITS = "data/gold/splits.json"
FULL = "data/hard/full"
FIXED_ROWS = (f"{FULL}/count_raw.jsonl", f"{FULL}/version_raw.jsonl", f"{FULL}/q2_aug_train.jsonl",
              f"{FULL}/compare_consensus_r1.jsonl")
RAW, STATS, PROBE_DIR = "q1x_raw.jsonl", "q1x_stats.json", "q1x_probe"
K = 5


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _dist(rows, f) -> dict:
    return dict(sorted(Counter(f(r) for r in rows).items()))


def load_inputs():
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    from grl.env.struct_data import build_struct_inputs
    texts, _, _, version = build_struct_inputs()
    art, _, _, _ = build_struct_inputs(articles=True)
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    d = config.path(latest["dir"])
    nodes, edges = _jsonl(d / "nodes.jsonl"), _jsonl(d / "edges.jsonl")
    data = load_env_data(version)
    mu = {}
    for v, nid in enumerate(data.node_ids):
        rows = data.mu_inv(v)
        if len(rows):
            mu[nid] = [data.chunk_ids[int(i)] for i in rows]
    recs = read_records(config.path(config.load("data")["records_dir"]))
    chunk_texts = {c.chunk_id: c.text for r in recs for c in r.chunks}
    spans = {c.chunk_id: (r.doc_id, int(c.start), int(c.end)) for r in recs for c in r.chunks}
    chunk_doc = {c.chunk_id: r.doc_id for r in recs for c in r.chunks}
    return {"version": version, "nodes": nodes, "edges": edges, "texts": texts, "art": art, "mu": mu,
            "chunk_texts": chunk_texts, "spans": spans, "chunk_doc": chunk_doc, "data": data}


def base_top5(data, ranks, keys, cfg) -> dict[str, dict[str, list[str]]]:
    """key -> {"dense": first 5 distinct chunk ids of q.vec, "fused": first 5 of the pilot-6 fused pool}."""
    from graphrag.env.start import make_query
    from grl.hard.metrics import top_k_ids
    from grl.model.rerank import rerank_candidates
    from grl.model.variants import rerank_kw, with_scores
    rk = rerank_kw(cfg)
    L_max = int(cfg["env"]["L_max"])
    start_cfg = config.load("data")["start"]
    out = {}
    for k in keys:
        q = with_scores(make_query(data, ranks.get(k), start_cfg, int(cfg["env"]["T_max"])), ranks.get(k))
        dense = top_k_ids([data.chunk_ids[int(c)] for c in q.vec], K)
        rows, _ = rerank_candidates(q, rk["n"], rk["fuse_w"], rk["norm_top_n"], data.chunk_tokens, L_max)
        fused = top_k_ids([data.chunk_ids[int(c)] for c in rows], K)
        out[k] = {"dense": dense, "fused": fused}
    return out


def probe_filter(inp, cands, out_dir: Path, workers: int):
    """(probe_drop(c, lang), report). Renders every candidate's languages first (leak-checked), ranks the
    corpus-gold renderings, and refuses a rendering whose gold is wholly inside dense@5 or fused@5."""
    import os
    from grl.hard.checks import STRUCT_PREFIX
    from grl.hard.templates import _check_langs, _qid
    from grl.threads import pin_blas
    pin_blas(workers, force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    st = Counter()
    for c in cands:
        c.questions = {}
        _check_langs(c, st)
    corpus = [(c, lang) for c in cands if not any(g.startswith(STRUCT_PREFIX) for g in c.gold)
              for lang in sorted(c.questions)]
    mixed = [c for c in cands if any(g.startswith(STRUCT_PREFIX) for g in c.gold)
             and not all(g.startswith(STRUCT_PREFIX) for g in c.gold)]
    if mixed:
        raise SystemExit(f"gold mixing STRUCT and corpus ids (the probe rule has no case for it): {mixed[0].key}")
    queries = sorted({(f"{_qid('count', c, lang)}|q", c.questions[lang]) for c, lang in corpus})
    rep = {"struct_gold_renderings": sum(len(c.questions) for c in cands
                                         if all(g.startswith(STRUCT_PREFIX) for g in c.gold)),
           "corpus_gold_renderings": len(corpus), "k": K}
    hits = {}
    if queries:
        from graphrag.retrieval.rankings import load_rankings
        from grl.hard.probe import check_rank_rows
        from grl.train.pilot import require_dev_pg
        require_dev_pg(config.PG_DSN)
        if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
            raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
        rdir = out_dir / "rankings"
        if not (rdir / "latest.json").exists():
            build_rankings_dir(inp["data"], queries, rdir)
        ranks = load_rankings(rdir / "latest.json", graph_version=inp["version"])
        if sorted(ranks.keys()) != [k for k, _ in queries]:
            raise SystemExit(f"{rdir}: ranking keys differ from the probed renderings (stale dir? delete it)")
        rep["rankings"] = json.loads((rdir / "latest.json").read_text(encoding="utf-8"))["version"]
        rep["rank_rows"] = check_rank_rows(ranks, [k for k, _ in queries], inp["data"].n_chunks)
        cfg = config.load("train")
        hits = base_top5(inp["data"], ranks, [k for k, _ in queries], cfg)
        from grl.model.variants import rerank_kw
        rep["base_settings"] = {"dense": "q.vec[:5]", "fused": "grl.model.rerank.rerank_candidates[:5]",
                                **rerank_kw(cfg), "L_max": int(cfg["env"]["L_max"])}
    rows = []
    for c, lang in corpus:
        k = f"{_qid('count', c, lang)}|q"
        g = set(c.gold)
        d_all, f_all = g <= set(hits[k]["dense"]), g <= set(hits[k]["fused"])
        rows.append({"qkey": k, "template": c.template, "lang": lang, "gold": sorted(g),
                     "dense5": hits[k]["dense"], "fused5": hits[k]["fused"],
                     "dense_all5": d_all, "fused_all5": f_all,
                     "dense_hits": len(g & set(hits[k]["dense"])), "fused_hits": len(g & set(hits[k]["fused"]))})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                for r in rows), encoding="utf-8")
    fail = {r["qkey"] for r in rows if r["dense_all5"] or r["fused_all5"]}
    rep["corpus_gold_failed"] = len(fail)
    rep["corpus_gold_failed_by_template"] = _dist([r for r in rows if r["qkey"] in fail], lambda r: r["template"])
    rep["corpus_gold_any_hit"] = sum(1 for r in rows if r["dense_hits"] or r["fused_hits"])

    def probe_drop(c, lang):
        return "probe_all5" if f"{_qid('count', c, lang)}|q" in fail else None
    return probe_drop, rep


def build_rankings_dir(data, queries, root: Path) -> str:
    """U9 rankings of `queries` written to root/ (build_rankings.py's path and inputs, family "q1x_probe")."""
    from graphrag.data.embed import Encoder
    from graphrag.kg.artifact import read_graph
    from graphrag.kg.build import files_digest
    from graphrag.retrieval.rankings import TOP, TitleIndex, build_rankings, write_rankings
    from graphrag.stores import pg
    nodes, _ = read_graph(config.path(f"data/graph/{data.version}"))
    assert [n.node_id for n in nodes] == data.node_ids
    bm = config.load("data")["bm25"]
    titles = TitleIndex([n.title for n in nodes], float(bm["k1"]), float(bm["b"]))
    enc = Encoder(device="cpu")
    conn = pg.connect()
    try:
        arrays = build_rankings(queries, data, conn, enc, titles, TOP)
    finally:
        conn.close()
    d = config.load("data")
    tcfg = config.load("train")
    inputs = {"graph_version": data.version, "top": TOP, "bm25": bm,
              "records_manifest": _sha256(config.path(f"{d['records_dir']}/manifest.jsonl")),
              "lexicon": _sha256(config.path(d["lexicon"]["path"])),
              "encoder": {"name": enc.model_name, "revision": config.load("model")["encoder"].get("revision"),
                          "device": "cpu"},
              "family": "q1x_probe", "gold": _sha256(config.path(tcfg["gold"]["file"])),
              "candidates": hashlib.sha256("".join(f"{k}\t{t}\n" for k, t in queries).encode("utf-8")).hexdigest(),
              "code": files_digest(config.ROOT, ("graphrag/retrieval/*.py", "graphrag/data/normalize.py",
                                                 "graphrag/data/lemma.py", "graphrag/data/embed.py"))}
    return write_rankings(root, queries, arrays, inputs)


def fixed_splits() -> tuple[dict, list[dict]]:
    from grl.hard.splits import fixed_doc_splits
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    rows, seen = [], set()
    for p in FIXED_ROWS:
        for r in _jsonl(config.path(p)):
            if r["qid"] not in seen:
                seen.add(r["qid"])
                rows.append(r)
    return fixed_doc_splits(old, rows), rows


def chunk_node_ids(data) -> list[str]:
    return [data.node_ids[int(v)] for v in data.chunk_node]


RAW_R0 = "q1x_raw_r0.jsonl"            # the Phase A rows (the qids the para6c paraphrases answer), kept verbatim


def cmd_refilter(a) -> int:
    """Fix round 1 (I1 + R59) without re-selection: the Phase A rows whose candidate the current rules still produce
    (same key, answer and gold) are kept, the others dropped -- no refill, so every kept row keeps its qid, language
    and paraphrase; the paraphrase lines of dropped qids are refused at ingest (unknown_qid). Then alt_gold / easy
    (R59). Writes <out>/q1x_raw.jsonl (the r0 rows copied once to q1x_raw_r0.jsonl) and q1x_stats.json["fix1"]."""
    import shutil
    from grl.hard.q1x import annotate_alt, q1x_candidates
    from grl.hard.templates import item_key
    out = config.path(a.out)
    r0 = out / RAW_R0
    if not r0.exists():
        shutil.copyfile(out / RAW, r0)
    rows = _jsonl(r0)
    inp = load_inputs()
    st = Counter()
    cands = q1x_candidates(inp["nodes"], inp["edges"], struct_texts=inp["texts"], art_texts=inp["art"],
                           mu_inv=inp["mu"], chunk_texts=inp["chunk_texts"], chunk_spans=inp["spans"], stats=st)
    by_key = {c.key: c for c in cands}
    keep, drop = [], []
    for r in rows:
        c = by_key.get(item_key(r))
        (keep if c is not None and c.answer == r["answer"] and c.gold == r["gold"] else drop).append(r)
    alt = annotate_alt(keep, inp["data"].chunk_ids, chunk_node_ids(inp["data"]), inp["spans"])
    (out / RAW).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    stats = json.loads((out / STATS).read_text(encoding="utf-8"))
    stats["fix1"] = {"rule": "I1: no count_parts_article for an article with points; no refill (paraphrases kept)",
                     "r0_sha256": _sha256(r0), "kept": len(keep), "dropped": [
                         {"qid": r["qid"], "template": r["template"], "split": r["split"], "answer": r["answer"],
                          "gold": r["gold"]} for r in drop],
                     "by_template": _dist(keep, lambda r: r["template"]), "by_split": _dist(keep, lambda r: r["split"]),
                     "alt_gold": alt, "parts_with_points_candidates": st.get("parts_with_points", 0)}
    (out / STATS).write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats["fix1"], ensure_ascii=False))
    return 0


COVER_DIR = "q1x_cover"
PHASE_B = "q1x"                        # Phase B dir: para_ok / judge / reprobe / final rows


FINAL = "q1x_final.jsonl"


def cmd_final(a) -> int:
    """Phase B steps 3-4: the judged rows (q1x/para_judged.jsonl) minus the re-probe failures -> <out>/q1x_final.jsonl,
    R32 re-checked over everything (old splits + every existing M7 row + the final rows: one split per document; every
    row's split reproduced by assign_splits against the fixed map; the train / val / test counts recorded)."""
    from grl.hard.splits import SPLITS, assign_splits, fixed_doc_splits
    out = config.path(a.out)
    rows = _jsonl(out / PHASE_B / "para_judged.jsonl")
    fail = set(json.loads((out / PHASE_B / "reprobe.json").read_text(encoding="utf-8"))["failed"])
    gate = json.loads((out / PHASE_B / "para_gate.json").read_text(encoding="utf-8"))
    if not gate.get("final") or not gate["gate"]["ok"]:
        raise SystemExit("the judge pass is not final / the gate did not pass")
    final = [r for r in rows if r["qid"] not in fail]
    fixed, fixed_rows = fixed_splits()
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    fixed_doc_splits(old, fixed_rows + final)                     # ValueError: a document in two splits
    again = assign_splits(final, fixed, 20261004)
    moved = [r["qid"] for r in final if again[r["qid"]] != r["split"] or r["split"] not in SPLITS]
    if moved:
        raise SystemExit(f"R32: splits not reproduced for {moved[:3]}")
    (out / FINAL).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final), encoding="utf-8")
    stats = json.loads((out / STATS).read_text(encoding="utf-8"))
    stats["final"] = {"n": len(final), "reprobe_dropped": sorted(fail), "judged": len(rows),
                      "by_template": _dist(final, lambda r: r["template"]),
                      "by_template_lang": _dist(final, lambda r: f"{r['template']}|{r['lang']}"),
                      "by_split": _dist(final, lambda r: r["split"]),
                      "by_template_split": _dist(final, lambda r: f"{r['template']}|{r['split']}"),
                      "gate": {k: {x: v[x] for x in ("judged", "sensitivity")} for k, v in gate["gate"]["by_kind"].items()},
                      "topup_plants": gate.get("topup_plants"), "ledger_total": gate["spend"]["ledger_total"],
                      "r32": "fixed_doc_splits over old + every M7 row + final: OK; assign_splits reproduces all"}
    (out / STATS).write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats["final"], ensure_ascii=False))
    return 0


def cmd_reprobe(a) -> int:
    """Phase B step 2: the Phase A probe again on the accepted PARAPHRASES of the corpus-gold items
    (count_articles_deleted_by_law): all@5 = 0 for dense@5 and fused@5 or the item is dropped. Dev stores, CPU,
    <= --workers threads; a construction-time probe (no model). Writes <out>/q1x/reprobe/ (rankings, probe.jsonl)
    and <out>/q1x/reprobe.json {checked, failed: [qid]}."""
    import os
    from grl.hard.checks import STRUCT_PREFIX
    from grl.threads import pin_blas
    pin_blas(a.workers, force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.probe import check_rank_rows
    from grl.train.pilot import require_dev_pg
    require_dev_pg(config.PG_DSN)
    if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
        raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
    out = config.path(a.out)
    items = {r["qid"]: r for r in _jsonl(out / RAW)}
    ok = [r for r in _jsonl(out / PHASE_B / "para_ok.jsonl") if r["qid"] in items]
    corpus = [r for r in ok if not any(g.startswith(STRUCT_PREFIX) for g in items[r["qid"]]["gold"])]
    data = load_env_data()
    queries = sorted((f"{r['qid']}|q", r["question"]) for r in corpus)
    rdir = out / PHASE_B / "reprobe" / "rankings"
    if not (rdir / "latest.json").exists():
        build_rankings_dir(data, queries, rdir)
    ranks = load_rankings(rdir / "latest.json", graph_version=data.version)
    if sorted(ranks.keys()) != [k for k, _ in queries]:
        raise SystemExit(f"{rdir}: keys differ from the paraphrases (stale dir? delete it)")
    check_rank_rows(ranks, [k for k, _ in queries], data.n_chunks)
    hits = base_top5(data, ranks, [k for k, _ in queries], config.load("train"))
    rows = []
    for r in corpus:
        g, h = set(items[r["qid"]]["gold"]), hits[f"{r['qid']}|q"]
        rows.append({"qid": r["qid"], "template": r["template"], "round": r["round"], "gold": sorted(g),
                     "dense5": h["dense"], "fused5": h["fused"], "dense_all5": g <= set(h["dense"]),
                     "fused_all5": g <= set(h["fused"]), "dense_hits": len(g & set(h["dense"])),
                     "fused_hits": len(g & set(h["fused"]))})
    (out / PHASE_B / "reprobe" / "probe.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    rep = {"rule": "paraphrase: all@5 = 0 for dense@5 and fused@5 (corpus-gold items)", "checked": len(rows),
           "rankings": json.loads((rdir / "latest.json").read_text(encoding="utf-8"))["version"],
           "failed": sorted(x["qid"] for x in rows if x["dense_all5"] or x["fused_all5"]),
           "any_hit": sum(1 for x in rows if x["dense_hits"] or x["fused_hits"])}
    (out / PHASE_B / "reprobe.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_coverage(a) -> int:
    """Fix round 1, I3 (reviewer's evidence-coverage probe): for every part / point item (its template question, as
    the Phase A probe) rank through the U9 path (dev stores, CPU, <= --workers threads) and flag
    dense_answerable = alt_gold ⊆ dense@5 ∪ fused@5 -- the baselines then retrieve every chunk of the article, so a
    reader can count its parts / points although the STRUCT gold id is unreachable. A construction-time probe like
    Phase A's (no model is evaluated). Writes the flag into <out>/q1x_raw.jsonl, <out>/q1x_cover/ (rankings,
    cover.jsonl) and q1x_stats.json["coverage"] (per template x easy / non-easy x split)."""
    import os
    from grl.hard.q1x import ARTICLE_TEMPLATES
    from grl.threads import pin_blas
    pin_blas(a.workers, force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from graphrag.env.data import load_env_data
    from graphrag.retrieval.rankings import load_rankings
    from grl.hard.probe import check_rank_rows
    from grl.train.pilot import require_dev_pg
    require_dev_pg(config.PG_DSN)
    if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
        raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
    out = config.path(a.out)
    rows = _jsonl(out / RAW)
    art = [r for r in rows if r["template"] in ARTICLE_TEMPLATES]
    if any(not r.get("alt_gold") for r in art):
        raise SystemExit("run refilter first (alt_gold missing)")
    data = load_env_data()
    queries = sorted((f"{r['qid']}|q", r["question"]) for r in art)
    rdir = out / COVER_DIR / "rankings"
    if not (rdir / "latest.json").exists():
        build_rankings_dir(data, queries, rdir)
    ranks = load_rankings(rdir / "latest.json", graph_version=data.version)
    if sorted(ranks.keys()) != [k for k, _ in queries]:
        raise SystemExit(f"{rdir}: keys differ from the article items (stale dir? delete it)")
    check_rank_rows(ranks, [k for k, _ in queries], data.n_chunks)
    hits = base_top5(data, ranks, [k for k, _ in queries], config.load("train"))
    recs = []
    for r in rows:
        if r["template"] not in ARTICLE_TEMPLATES:
            r["dense_answerable"] = None
            continue
        h = hits[f"{r['qid']}|q"]
        alt = set(r["alt_gold"])
        r["dense_answerable"] = alt <= set(h["dense"]) | set(h["fused"])
        recs.append({"qid": r["qid"], "template": r["template"], "split": r["split"], "easy": r["easy"],
                     "n_alt": len(alt), "dense_all_alt": alt <= set(h["dense"]), "fused_all_alt": alt <= set(h["fused"]),
                     "union_all_alt": r["dense_answerable"], "dense_hits": len(alt & set(h["dense"])),
                     "fused_hits": len(alt & set(h["fused"]))})
    (out / COVER_DIR / "cover.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n"
                                                        for x in recs), encoding="utf-8")
    groups = {}
    for x in recs:
        for k in (f"{x['template']}|ALL|ALL", f"{x['template']}|{'easy' if x['easy'] else 'non_easy'}|ALL",
                  f"{x['template']}|ALL|{x['split']}"):
            groups.setdefault(k, []).append(x)
    rep = {k: {"n": len(v), **{m: round(sum(x[m] for x in v) / len(v), 4)
                                for m in ("dense_all_alt", "fused_all_alt", "union_all_alt")}}
           for k, v in sorted(groups.items())}
    (out / RAW).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = json.loads((out / STATS).read_text(encoding="utf-8"))
    stats["coverage"] = {"rule": "dense_answerable = alt_gold ⊆ dense@5 ∪ fused@5 (template question, U9 rankings)",
                         "rankings": json.loads((rdir / "latest.json").read_text(encoding="utf-8"))["version"],
                         "by_group": rep}
    (out / STATS).write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_make(a) -> int:
    from grl.hard.q1x import Q1X_CAP, Q1X_TEMPLATES, annotate_alt, make_q1x, q1x_candidates
    from grl.hard.splits import SPLITS, assign_splits, fixed_doc_splits
    sys.path.insert(0, str(ROOT / "scripts"))
    from hard_probe import frozen_lang_shares, read_exclude
    out = config.path(a.out)
    shares, shares_src = frozen_lang_shares(out)
    if shares_src != "frozen":
        raise SystemExit(f"{out}/lang_shares.json missing: Task 6c uses Task 4's frozen language counts (R47)")
    exclude = read_exclude(config.path(a.exclude))
    fixed, fixed_rows = fixed_splits()
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    inp = load_inputs()
    st0 = Counter()
    kw = dict(struct_texts=inp["texts"], art_texts=inp["art"], mu_inv=inp["mu"], chunk_texts=inp["chunk_texts"],
              chunk_spans=inp["spans"])
    cands = q1x_candidates(inp["nodes"], inp["edges"], stats=st0, **kw)
    probe_drop, prep = probe_filter(inp, cands, out / PROBE_DIR, a.workers)
    print("probe:", json.dumps({k: v for k, v in prep.items() if k != "rank_rows"}, ensure_ascii=False))
    rows, st, rounds = make_q1x(inp["nodes"], inp["edges"], a.seed, a.n, fixed_splits=fixed,
                                chunk_doc=inp["chunk_doc"], lang_shares=shares, exclude=exclude,
                                probe_drop=probe_drop, max_per_root=Q1X_CAP, **kw)
    # --- checks ---------------------------------------------------------------------------------------------------
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)) or set(qids) & exclude or set(qids) & {r["qid"] for r in fixed_rows}:
        raise SystemExit("duplicate / probe / existing qids among the new rows")
    if max(Counter(r["docs"][0] for r in rows).values(), default=0) > Q1X_CAP:
        raise SystemExit("per-root cap broken")
    fixed_doc_splits(old, fixed_rows + rows)                        # ValueError: a document in two splits
    if any(r["split"] not in SPLITS for r in rows):
        raise SystemExit("unsplit rows")
    for r in rows:
        own = {fixed[d] for d in r["docs"] + r["gold_docs"] if d in fixed}
        if own and own != {r["split"]}:
            raise SystemExit(f"{r['qid']}: split {r['split']} vs fixed {own}")
    again = assign_splits(rows, fixed, a.seed)
    if any(again[r["qid"]] != r["split"] for r in rows):
        raise SystemExit("splits not reproducible")
    # --- report ---------------------------------------------------------------------------------------------------
    probe_by_t = {t: {"checked_renderings": st.get(f"probe_checked:{t}", 0),
                      "dropped_renderings": st.get(f"probe_all5:{t}", 0),
                      "dropped_candidates": st.get(f"probe_dropped_candidates:{t}", 0)} for t in Q1X_TEMPLATES}
    rep = {"graph_version": inp["version"], "seed": a.seed, "n_slots": a.n, "lang_shares": shares,
           "exclude": a.exclude, "n_exclude": len(exclude), "cap_per_root": Q1X_CAP,
           "cap_rule": "R57: <= 6 new Q1 items per root (docs[0]) over the five Task 6c templates; own counter",
           "inputs_sha256": {OLD_SPLITS: _sha256(config.path(OLD_SPLITS)),
                             **{p: _sha256(config.path(p)) for p in FIXED_ROWS},
                             "data/hard/struct_art_vecs.npz": _sha256(config.path("data/hard/struct_art_vecs.npz")),
                             f"{a.out.rstrip('/')}/lang_shares.json": _sha256(out / "lang_shares.json")},
           "r32_rounds": rounds, "selected": len(rows),
           "by_template": _dist(rows, lambda r: r["template"]), "by_lang": _dist(rows, lambda r: r["lang"]),
           "by_split": _dist(rows, lambda r: r["split"]),
           "by_template_split": _dist(rows, lambda r: f"{r['template']}|{r['split']}"),
           "gold_sizes": _dist(rows, lambda r: len(r["gold"])), "roots": len({r["docs"][0] for r in rows}),
           "max_per_root": max(Counter(r["docs"][0] for r in rows).values(), default=0),
           "answers_by_template": {t: _dist([r for r in rows if r["template"] == t], lambda r: int(r["answer"]))
                                   for t in Q1X_TEMPLATES},
           "inherited_split": sum(1 for r in rows if any(d in fixed for d in r["docs"] + r["gold_docs"])),
           "probe": {**prep, "by_template": probe_by_t,
                     "rule": "all@5 = 0 for dense@5 and fused@5; STRUCT gold: 0 by construction (no index)"},
           "stats": dict(sorted(st.items()))}
    print(json.dumps({k: rep[k] for k in ("selected", "by_template", "by_lang", "by_split", "gold_sizes", "roots",
                                          "max_per_root", "r32_rounds", "inherited_split")}, ensure_ascii=False))
    rep["alt_gold"] = annotate_alt(rows, inp["data"].chunk_ids, chunk_node_ids(inp["data"]), inp["spans"])
    out.mkdir(parents=True, exist_ok=True)
    (out / RAW).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / STATS).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out / RAW, out / STATS)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="Phase A: candidates -> probe -> selection -> splits")
    m.add_argument("--seed", type=int, default=20261004)
    m.add_argument("--n", type=int, default=250)
    m.add_argument("--exclude", default="data/hard/probe/ids.json")
    m.add_argument("--out", default=f"{FULL}/")
    m.add_argument("--workers", type=int, default=2)
    rf = sub.add_parser("refilter", help="fix round 1: I1 + R59 on the Phase A rows, no re-selection")
    rf.add_argument("--out", default=f"{FULL}/")
    rf.add_argument("--workers", type=int, default=2)
    cv = sub.add_parser("coverage", help="fix round 1, I3: evidence-coverage probe of the part / point items")
    cv.add_argument("--out", default=f"{FULL}/")
    cv.add_argument("--workers", type=int, default=2)
    rp = sub.add_parser("reprobe", help="Phase B: the probe on the accepted paraphrases of corpus-gold items")
    rp.add_argument("--out", default=f"{FULL}/")
    rp.add_argument("--workers", type=int, default=2)
    fn = sub.add_parser("final", help="Phase B: judged rows - re-probe failures -> q1x_final.jsonl + R32 re-check")
    fn.add_argument("--out", default=f"{FULL}/")
    fn.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)
    if a.workers > 2:
        raise SystemExit("--workers <= 2 on the Mac")
    return {"make": cmd_make, "refilter": cmd_refilter, "coverage": cmd_coverage, "reprobe": cmd_reprobe, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
