"""M3 synthetic dataset (decisions П1–П5, M3-R3, M3-R10): 1000 single-turn §8.1 pairs from graph walks + atomic facts.
    python scripts/build_dataset.py seeds [--forced data/dataset/pilot/seeds.jsonl]   # Task 14 / 18
    python scripts/build_dataset.py pilot-seeds                                        # Task 17
    GRL_DATASET_API=1 python scripts/build_dataset.py facts|generate|verify [--dir D]   # Task 17 / 18
    python scripts/build_dataset.py finalize [--dir D] [--no-manual-gate]              # Task 17 / 18
    python scripts/build_dataset.py trajectories [--dir D] [--rankings L] [--no-select] [--width W]   # Task 17 / 20
    python scripts/build_dataset.py pilot-report                                       # Task 17
  M3-R23 (Opus generator + gpt-4.1 verify + independent Opus reader), --dir data/dataset/opus (pilot dir derived):
    python scripts/build_dataset.py pilot-opus|packets --dir data/dataset/opus          # generator work packets
    python scripts/build_dataset.py ingest-gen --dir data/dataset/opus                  # code checks -> candidates_raw
    GRL_DATASET_API=1 python scripts/build_dataset.py verify --dir data/dataset/opus    # blind judge + ablation + plants
    python scripts/build_dataset.py review-packets|ingest-review --dir data/dataset/opus   # strict reader, GOOD only
    python scripts/build_dataset.py finalize|trajectories --dir data/dataset/opus"""
import argparse
import json
import os
import sys

from graphrag import config

OUT = "data/dataset"
PILOT_DIR = "data/dataset/pilot"
ARGS = None


def _jsonl_write(path, rows) -> None:
    p = config.path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _jsonl_read(path) -> list[dict]:
    return [json.loads(l) for l in config.path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _stats_update(section: str, value) -> None:
    f = config.path(f"{OUT}/stats.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    s = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    s[section] = value
    f.write_text(json.dumps(s, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _chunks():
    from graphrag.data.ingest import read_records
    recs = read_records(config.path(config.load("data")["records_dir"]))
    return recs, {c.chunk_id: c for r in recs for c in r.chunks}


def _pool(tcfg):
    from graphrag.dataset.seeds import content_chunks, qreal_exclusion, single_pool
    from graphrag.dataset.walks import enumerate_walks, hub_vertices
    from graphrag.env.data import load_env_data
    from graphrag.eval.gold import linked_doc_pairs, read_gold
    from graphrag.kg.artifact import read_graph
    ds = tcfg["dataset"]
    data = load_env_data()
    recs, chunks = _chunks()
    _, edges = read_graph(config.path(f"data/graph/{data.version}"))
    splits = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    items = read_gold(config.path(tcfg["gold"]["file"]))
    excluded = qreal_exclusion(items, recs, linked_doc_pairs())
    content = content_chunks(recs)                          # M3-R14 (2)
    ok = {c for c, ch in chunks.items() if ch.tokens >= int(ds["min_tokens"]) and c not in excluded and c in content}
    hub = ds["hub"]
    hubs, hub_audit = hub_vertices(edges, {r.doc_id for r in recs}, int(hub["max_referrer_docs"]),
                                   float(hub["top_degree_share"]))                           # M3-R14 (1)
    gold = {c for it in items for c in it.gold}
    alt = {c for it in items for c in it.alt} - gold
    ctx = {c for it in items for c in it.context} - gold - alt
    long_enough = {c for c, ch in chunks.items() if ch.tokens >= int(ds["min_tokens"])}
    wstats: dict = {"qreal_exclusion": {"items": len(items), "total": len(excluded), "gold": len(gold), "alt": len(alt),
                                        "context": len(ctx), "linked_same_path": len(excluded - gold - alt - ctx),
                                        "removed_from_eligible": len(excluded & long_enough)},
                    "eligible_chunks": len(ok), "chunks": len(chunks),
                    "non_content_chunks": len(set(chunks) - content),
                    "non_content_removed_from_eligible": len((long_enough - excluded) - content), "hubs": hub_audit}
    walks = enumerate_walks(data, recs, edges, ok, int(ds["max_hops"]), int(ds["max_branch"]), ds["seed"],
                            split_of=splits, stats=wstats, hubs=hubs)
    # M3-R20: one `multi` pattern -- bridge + aggregation walks, anchored, no procedural-hub provision
    from graphrag.dataset.walks import element_referrers, quality_walks
    multi, qrep = quality_walks(walks, lambda c: chunks[c].text, element_referrers(edges), int(hub["max_referrer_docs"]))
    wstats["quality_walks"] = qrep
    pool = {"single": single_pool(recs, ok), "multi": multi}
    return recs, splits, excluded, pool, wstats


def _final_need(tcfg) -> dict[str, int]:
    """Final need per cell pattern|split|lang (size × hops × split ratio × language share; M3-R20 quotas)."""
    from graphrag.dataset.seeds import PATTERNS, cell_counts
    ds = tcfg["dataset"]
    final = {}
    for p in PATTERNS:
        for sp, m in cell_counts(round(ds["size"] * ds["hops"][p]), tcfg["split"]["ratios"]).items():
            for lang, n in cell_counts(m, ds["langs"]).items():
                final[f"{p}|{sp}|{lang}"] = n
    return final


def _sample(tcfg, pool, splits, recs, overshoot: dict, forced) -> tuple[list, dict]:
    """The M3-R20 sampling (no M3-R15 transfer): multi first, then single preferring fact blocks the multi seeds
    already need. Returns (seeds, overshoot shortage per cell)."""
    from graphrag.dataset.seeds import PATTERNS, block_of, doc_cap, sample_seeds
    ds = tcfg["dataset"]
    dc = ds.get("seed_doc_cap")
    cap = doc_cap(recs, int(dc["floor"]), float(dc["share"])) if dc else int(ds["max_seeds_per_doc_cell"])
    quotas = {p: round(ds["size"] * ds["hops"][p] * float(overshoot[p])) for p in PATTERNS}
    sections = {r.doc_id: r.meta.section for r in recs}
    multi, _ = sample_seeds({"multi": pool["multi"]}, splits, sections, {"multi": quotas["multi"]},
                            tcfg["split"]["ratios"], ds["langs"], ds["single_sections"],
                            cap, ds["seed"], max_per_chunk=int(ds["max_seeds_per_chunk"]),
                            forced=[f for f in forced if f.pattern != "single"])
    blk = block_of(recs, int(ds["facts_block"]))
    needed = {blk[c] for s in multi for c in s.chunks} | {blk[c] for f in forced for c in f.chunks}
    return sample_seeds(pool, splits, sections, quotas, tcfg["split"]["ratios"], ds["langs"], ds["single_sections"],
                        cap, ds["seed"], max_per_chunk=int(ds["max_seeds_per_chunk"]),
                        forced=forced, prefer=lambda ch: blk[ch[0]] in needed)


def cmd_seeds(tcfg) -> int:
    from collections import Counter
    from graphrag.dataset.seeds import PATTERNS, Seed, block_of
    from graphrag.dataset.walks import walk_class
    ds = tcfg["dataset"]
    recs, splits, excluded, pool, wstats = _pool(tcfg)
    quotas = {p: round(ds["size"] * ds["hops"][p] * float(ds["overshoot"][p])) for p in PATTERNS}
    forced = [Seed.from_json(r) for r in _jsonl_read(ARGS.forced)] if ARGS.forced else []
    seeds, short = _sample(tcfg, pool, splits, recs, ds["overshoot"], forced)
    blk = block_of(recs, int(ds["facts_block"]))
    _jsonl_write(f"{OUT}/seeds.jsonl", [s.to_json() for s in seeds])
    final = _final_need(tcfg)
    got = Counter(f"{s.pattern}|{s.split}|{s.lang}" for s in seeds)
    margin = {k: round(got[k] / n, 4) if n else None for k, n in final.items()}
    below_final = {k: n - got[k] for k, n in final.items() if got[k] < n}
    sections = {r.doc_id: r.meta.section for r in recs}
    pool_cell = Counter(f"{p}|{splits[x[1][0]]}" for p, v in pool.items() for x in v)
    pool_sec = Counter(f"single|{splits[x[1][0]]}|{sections[x[1][0]]}" for x in pool["single"])
    need_split = Counter()
    for k, n in final.items():
        p, sp, _ = k.split("|")
        need_split[f"{p}|{sp}"] += n
    st = {"pool": {p: len(v) for p, v in pool.items()}, "quotas": quotas, "seeds": len(seeds), "forced": len(forced),
          "overshoot_shortage": short, "excluded_qreal_fragments": len(excluded),
          "pool_vs_final_need": {k: {"pool": pool_cell[k], "final_need": need_split[k],
                                     "pool_over_need": round(pool_cell[k] / need_split[k], 3) if need_split[k] else None}
                                 for k in sorted(need_split)},
          "single_pool_by_section": dict(sorted(pool_sec.items())),
          "facts_blocks": len({blk[c] for s in seeds for c in s.chunks}),
          "by_cell": dict(sorted(got.items())), "final_need_by_cell": final, "margin_by_cell": margin,
          "below_final": below_final,
          "walk_class_by_cell": dict(sorted(Counter(f"{s.split}|{walk_class(s.walk)}" for s in seeds if s.walk).items()))}
    from graphrag.dataset.seeds import doc_cap
    dc = ds.get("seed_doc_cap")
    st.update(_seed_report(pool, seeds, splits, wstats, quotas, tcfg,
                           doc_cap(recs, int(dc["floor"]), float(dc["share"])) if dc else None))
    _stats_update("seeds", st)
    print(json.dumps({k: st[k] for k in ("pool", "quotas", "seeds", "forced", "pool_vs_final_need", "below_final",
                                         "overshoot_shortage", "facts_blocks", "walk_class_by_cell")}, ensure_ascii=False))
    if below_final:
        print(f"STOP (П4, M3-R3 п. 1): cells below their final need {below_final}; report to the user")
        return 1
    return 0


def _seed_report(pool, seeds, splits, wstats, quotas, tcfg, cap=None) -> dict:
    """Report-only numbers (Task 14): pool per split × pattern, demand per cell, walk lengths and edge mix,
    per-document concentration. Nothing here feeds the sampling."""
    from collections import Counter
    from graphrag.dataset.seeds import PATTERNS, cell_counts
    ds = tcfg["dataset"]
    cap = cap if cap is not None else int(ds["max_seeds_per_doc_cell"])
    pool_cell = Counter((p, splits[x[1][0]]) for p, v in pool.items() for x in v)
    demand = {}
    for p in PATTERNS:
        for sp, m in cell_counts(quotas[p], tcfg["split"]["ratios"]).items():
            for lang, n in cell_counts(m, ds["langs"]).items():
                demand[f"{p}|{sp}|{lang}"] = n
    multi = [s for s in seeds if s.walk]
    per_doc = Counter((s.pattern, s.split, s.lang, d) for s in seeds for d in s.docs)
    return {"walk_stats": wstats,
            "pool_by_split": {f"{p}|{sp}": n for (p, sp), n in sorted(pool_cell.items())},
            "demand_by_cell": demand,
            "pool_hops_hist": {p: dict(sorted(Counter(len(x[2]) for x in v).items())) for p, v in sorted(pool.items())
                               if p != "single"},
            "pool_chunks_hist": {p: dict(sorted(Counter(len(x[0]) for x in v).items())) for p, v in sorted(pool.items())
                                 if p != "single"},
            "seed_hops_hist_by_pattern": {p: dict(sorted(Counter(len(s.walk) for s in multi if s.pattern == p).items()))
                                          for p in PATTERNS if p != "single"},
            "seed_chunks_hist": dict(sorted(Counter(len(s.chunks) for s in multi).items())),
            "seed_edge_mix": dict(sorted(Counter(t for s in multi for _, t, _ in s.walk).items())),
            "pool_edge_mix": dict(sorted(Counter(t for p, v in pool.items() for x in v for _, t, _ in x[2]).items())),
            "max_seeds_per_doc_cell_seen": max(per_doc.values(), default=0),
            "docs_at_cap": sum(1 for (_, _, _, d), v in per_doc.items() if v >= (cap(d) if callable(cap) else cap)),
            "max_seeds_per_doc": max(Counter(d for s in seeds for d in s.docs).values(), default=0),
            "docs_used": len({d for s in seeds for d in s.docs})}


def _llm(tcfg):
    from graphrag.kg.llm import KgLLM
    return KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"],
                 allow_api=os.environ.get("GRL_DATASET_API") == "1")


def _seeds():
    from graphrag.dataset.seeds import Seed
    return [Seed.from_json(r) for r in _jsonl_read(f"{OUT}/seeds.jsonl")]


def _pilot_ids():
    out = set()
    for name in ("seeds.jsonl", "bridge_seeds.jsonl"):          # M3-R21: ds_bridge pairs of the pilot count as pilot
        p = config.path(f"{PILOT_DIR}/{name}")
        if p.exists():
            out |= {json.loads(l)["seed_id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    for pid in _pilot_packets():               # a packet dir (M3-R23): the pilot packets' single seeds are pilot
        f = config.path(f"{OUT}/packets/{pid}.json")
        if f.exists():
            out |= {e["seed_id"] for e in json.loads(f.read_text(encoding="utf-8"))["seeds"] if "seed" in e}
    return out


def _bridge_index(tcfg, recs, chunks):
    """M3-R21: graphrag.dataset.bridge.BridgeIndex over the eligible content fragments (the _pool rule without walks)."""
    from graphrag.dataset.bridge import BridgeIndex
    from graphrag.dataset.seeds import content_chunks, qreal_exclusion
    from graphrag.dataset.walks import element_referrers
    from graphrag.env.data import load_env_data
    from graphrag.eval.gold import linked_doc_pairs, read_gold
    from graphrag.kg.artifact import read_graph
    ds = tcfg["dataset"]
    data = load_env_data()
    _, edges = read_graph(config.path(f"data/graph/{data.version}"))
    splits = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    excluded = qreal_exclusion(read_gold(config.path(tcfg["gold"]["file"])), recs, linked_doc_pairs())
    content = content_chunks(recs)
    ok = {c for c, ch in chunks.items() if ch.tokens >= int(ds["min_tokens"]) and c not in excluded and c in content}
    return BridgeIndex(data, recs, edges, ok, splits, ds["seed"], element_referrers(edges),
                       int(ds["hub"]["max_referrer_docs"]))


def _amended(bx, index):
    """M3-R22: chunk -> names of the acts it amends / repeals / cites (EXT act keys, corpus act titles)."""
    def names(cid):
        out = []
        for _, t, dst in bx.at.get(cid, ()):
            if t in ("AMENDS", "REPEALS", "REFERS_TO"):
                d = dst.split("/")
                out.append(d[1] if d[0] == "EXT" and len(d) > 1 else index.act.get(d[0], ""))
        return sorted({n for n in out if n})
    return names


def cmd_facts(tcfg) -> int:
    from collections import defaultdict
    from graphrag.dataset.facts import extract, write_facts
    from graphrag.dataset.seeds import block_of
    from graphrag.dataset.sources import SourceIndex
    ds = tcfg["dataset"]
    recs, chunks = _chunks()
    blk = block_of(recs, int(ds["facts_block"]))
    block_chunks = defaultdict(list)
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            block_chunks[blk[c.chunk_id]].append(c.chunk_id)
    needed = {blk[c] for s in _seeds() for c in s.chunks}
    llm = _llm(tcfg)
    facts, st = extract(needed, block_chunks, chunks, SourceIndex(recs), llm, int(ds["max_facts_per_chunk"]))
    write_facts(config.path(f"{OUT}/facts.jsonl"), facts)
    _stats_update("facts", {**st, "llm": llm.stats_json()})
    print(json.dumps(st, ensure_ascii=False))
    return 0


def _cand_rows(cands):
    from dataclasses import asdict
    return [{"seed": c.seed.to_json(), "question": c.question, "answer": c.answer,
             "facts": [asdict(f) for f in c.facts], "provenance": c.provenance} for c in cands]


def _rows_cands(rows):
    from graphrag.dataset.facts import Fact
    from graphrag.dataset.generate import Candidate
    from graphrag.dataset.seeds import Seed
    return [Candidate(Seed.from_json(r["seed"]), r["question"], r["answer"], [Fact(**f) for f in r["facts"]],
                      r["provenance"]) for r in rows]


def cmd_generate(tcfg) -> int:
    from graphrag.dataset.facts import read_facts
    from graphrag.dataset.generate import generate
    from graphrag.dataset.sources import SourceIndex
    recs, chunks = _chunks()
    llm = _llm(tcfg)
    index = SourceIndex(recs)
    bx = _bridge_index(tcfg, recs, chunks)
    cands, st = generate(_seeds(), read_facts(config.path(f"{OUT}/facts.jsonl")), chunks, index, llm,
                         tcfg["dataset"], pilot_ids=_pilot_ids(), amended=_amended(bx, index))
    _jsonl_write(f"{OUT}/candidates_raw.jsonl", _cand_rows(cands))
    _stats_update("generate", {**st, "llm": llm.stats_json()})
    print(json.dumps(st, ensure_ascii=False))
    return 0


def _frag_of(recs, chunks):
    from graphrag.dataset.sources import SourceIndex
    index = SourceIndex(recs)

    def frag_of(cid, label):
        c = chunks[cid]
        return {"id": label, "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent), "text": c.text}
    return frag_of, index


def cmd_verify(tcfg) -> int:
    from graphrag.dataset.seeds import content_chunks
    from graphrag.dataset.verify import neighbor_picker, sensitivity_gate, unrelated_picker, verify_candidates
    ds = tcfg["dataset"]
    recs, chunks = _chunks()
    frag_of, _ = _frag_of(recs, chunks)
    splits = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    content = content_chunks(recs)
    unrelated = unrelated_picker(chunks, splits, ds["seed"], int(ds["min_tokens"]),
                                 section_of={r.doc_id: r.meta.section for r in recs}, allowed=content)
    order = {r.doc_id: [c.chunk_id for c in sorted(r.chunks, key=lambda c: c.start)] for r in recs}
    neighbor = neighbor_picker(order, chunks, ds["seed"], int(ds["min_tokens"]), allowed=content)
    llm = _llm(tcfg)
    cands = _rows_cands(_jsonl_read(f"{OUT}/candidates_raw.jsonl"))
    if config.path(f"{OUT}/bridge_raw.jsonl").exists():            # M3-R21: ds_bridge candidates
        cands += _rows_cands(_jsonl_read(f"{OUT}/bridge_raw.jsonl"))
    if config.path(f"{OUT}/repaired_raw.jsonl").exists():          # M3-R32: repaired pairs, verified from scratch
        cands += _rows_cands(_jsonl_read(f"{OUT}/repaired_raw.jsonl"))
    sub = ARGS.pattern                          # M3-R28: re-verify one pattern only (own outputs, no plan file)
    if sub:
        cands = [c for c in cands if c.seed.pattern == sub]
    pilot = _pilot_ids()
    # M3-R18 (a): the pilot's verify plan is frozen -- the pilot run writes its digest, the full run must reproduce it
    # per dataset dir, keyed by the dir and its pilot candidates: another dir's plan, or a plan over other pilot
    # candidates (a regenerated pilot), is never enforced
    plan_file = config.path(f"{PILOT_DIR}/verify_plan.json")
    is_pilot_run = bool(cands) and all(c.seed.seed_id in pilot for c in cands)
    import hashlib
    pkey = hashlib.sha256(json.dumps(sorted([c.seed.seed_id, c.question] for c in cands if c.seed.seed_id in pilot),
                                     ensure_ascii=False).encode()).hexdigest()
    frozen = None
    if not is_pilot_run and pilot and plan_file.exists() and not sub:
        pf = json.loads(plan_file.read_text(encoding="utf-8"))
        if pf.get("dir") == OUT and pf.get("pilot_candidates") == pkey:
            frozen = pf["plan_digest_pilot"]
        else:
            print(f"NOTE: {plan_file} belongs to another dir / pilot candidate set -- not enforced")
    parts = {c.seed.seed_id: f"packet:{c.provenance['opus']['packet_id']}" for c in cands if c.provenance.get("opus")}
    kept, st = verify_candidates(cands, frag_of, llm,
                                 int(ds["verify_batch"]), 0.0 if ARGS.no_plants else float(ds["plant_share"]),
                                 ds["seed"], unrelated,
                                 min_kind_n=int(ds["verify_min_kind_n"]), views_per_call=int(ds["verify_views_per_call"]),
                                 neighbor=neighbor, pilot_ids=pilot, quote_cap=tuple(ds["verify_quote_cap"]),
                                 pilot_target=None if ARGS.no_plants else int(ds["plant_target_pilot"]),
                                 frozen_plan=frozen,
                                 agree_per_call=int(ds["verify_agree_per_call"]), parts=parts or None,
                                 topup=not ARGS.no_plants)
    if is_pilot_run and not sub and not ARGS.no_plants:
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps({"plan_digest_pilot": st["plan_digest_pilot"], "calls": st["calls"],
                                         "pilot_target": int(ds["plant_target_pilot"]), "dir": OUT,
                                         "pilot_candidates": pkey}, indent=1) + "\n",
                             encoding="utf-8")
    # gate: CUMULATIVE (M3-R3 п. 8, every judge kind >= min_kind_n valid plants, sensitivity >= floor); the pilot
    # part's own numbers are reported only (the pilot gates are retired, M3-R32). A failed gate writes NO verified
    # rows: they go to verified{tag}_blocked.jsonl and the previous verified file stays as it was.
    pilot_gate = [f"pilot: {r}" for r in sensitivity_gate(st["pilot"], float(ds["min_sensitivity_kind"]))] if pilot else []
    stop = sensitivity_gate(st, float(ds["min_sensitivity_kind"]))
    tag = (f"_{sub}" if sub else "") + ("_noplants" if ARGS.no_plants else "")
    _jsonl_write(f"{OUT}/verified{tag}{'_blocked' if stop else ''}.jsonl", _cand_rows(kept))
    _stats_update(f"verify{tag}", {**st, "gate": stop, "pilot_gate_report": pilot_gate, "llm": llm.stats_json()})
    print(json.dumps(st, ensure_ascii=False))
    if stop:
        print(f"STOP (M3-R3 п. 8): judge verdicts not accepted -- {'; '.join(stop)}")
        return 1
    return 0


def cmd_finalize(tcfg) -> int:
    from collections import Counter, defaultdict
    from graphrag.data.embed import Encoder
    from graphrag.dataset.finalize import (dedup_and_leak, facts_cache_keys, manual_sample, provenance_missing,
                                           question_vectors, to_pair, write_pairs)
    from graphrag.dataset.seeds import block_of, qreal_exclusion
    from graphrag.env.data import load_env_data
    from graphrag.eval.gold import linked_doc_pairs, read_gold
    from graphrag.kg.audit import wilson
    from graphrag.kg.llm import prompt_digest
    ds = tcfg["dataset"]
    recs, chunks = _chunks()
    data = load_env_data()
    _, index = _frag_of(recs, chunks)
    splits = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    cands = _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))
    from graphrag.dataset.opus import ROLE_GEN, gen_provenance_problem
    opus_run = any((c.provenance.get("generate") or {}).get("role") == ROLE_GEN for c in cands)
    unreviewed = 0
    if opus_run:                   # M3-R23: only the independent reader's GOOD pairs (ingest-review) are finalized
        f = config.path(f"{OUT}/reviewed.jsonl")
        if not f.exists():
            print(f"STOP (M3-R23): {f} missing -- run review-packets / ingest-review first")
            return 1
        ver = {c.seed.seed_id: c.question for c in cands}
        good = [c for c in _rows_cands(_jsonl_read(f)) if ver.get(c.seed.seed_id) == c.question
                and (c.provenance.get("review") or {}).get("verdict") == "GOOD"]
        unreviewed = len(cands) - len(good)
        bad_prov = {c.seed.seed_id: m for c in good if (m := gen_provenance_problem(c.provenance, config.path(".")))}
        if bad_prov:
            print(f"STOP (M3-R3 п. 11): generator provenance {dict(list(bad_prov.items())[:5])}")
            return 1
        cands = good
    real = read_gold(config.path(tcfg["gold"]["file"]))
    enc = Encoder(device="cpu")
    qv = question_vectors(enc, [c.question for c in cands])
    real_qv = question_vectors(enc, [t for i in real for t in (i.question_cyr, i.question_lat) if t])
    kept, dropped = dedup_and_leak(cands, qv, real_qv, qreal_exclusion(real, recs, linked_doc_pairs()), ds, ds["seed"],
                                   split_of=splits)
    by_cell = defaultdict(Counter)
    for c in cands:
        if c.seed.seed_id in dropped:
            by_cell[f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}"][dropped[c.seed.seed_id]] += 1
    sample = manual_sample(kept, int(ds["manual_sample"]), ds["seed"])
    _jsonl_write(f"{OUT}/manual_sample.jsonl",
                 [{"seed_id": c.seed.seed_id, "cell": f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}",
                   "question": c.question, "answer": c.answer, "walk": c.seed.walk,
                   "facts": [{"value": f.value, "quote": f.quote, "chunk_id": f.chunk_id} for f in c.facts],
                   "sources": index.sources_for(data, sorted(c.seed.chunks)),
                   "fragments": {cid: chunks[cid].text for cid in c.seed.chunks}} for c in sample])
    manual = None
    bad = set()
    if ARGS.no_manual_gate:
        print("NOTE: --no-manual-gate -- the independent manual review (M3-R3 п. 9) was SKIPPED; not a final dataset")
    else:
        f = config.path(f"{OUT}/manual_decisions.json")
        dec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        missing = [c.seed.seed_id for c in sample if c.seed.seed_id not in dec.get("items", {})]
        if missing or not dec.get("reviewer") or dec.get("reviewer") == dec.get("implementer"):
            print(f"{len(missing)} sampled pairs undecided or no independent reviewer recorded in {f}")
            return 1
        good = sum(bool(dec["items"][c.seed.seed_id]["good"]) for c in sample)
        lo, _ = wilson(good, len(sample))
        bad = {c.seed.seed_id for c in sample if not dec["items"][c.seed.seed_id]["good"]}
        manual = {"n": len(sample), "good": good, "share": round(good / len(sample), 4), "wilson_lb": round(lo, 4),
                  "reviewer": dec["reviewer"]}
    kept = [c for c in kept if c.seed.seed_id not in bad]
    # provenance (M3-R3 п. 11; Task 15 concern): the ds_facts cache files the pair's facts came from
    from graphrag.dataset.facts import PASS as FACTS_PASS
    blk = block_of(recs, int(ds["facts_block"]))
    block_chunks = defaultdict(list)
    for r in recs:
        for ch in sorted(r.chunks, key=lambda ch: ch.start):
            block_chunks[blk[ch.chunk_id]].append(ch.chunk_id)
    llm = _llm(tcfg)
    for c in kept:
        # Task 15 fix records the keys of the accepted ds_facts answers (incl. a retry); recompute only if absent
        keys = (c.provenance.get("facts") or {}).get("cache_keys") or facts_cache_keys(c, blk, block_chunks, chunks,
                                                                                       index, llm)
        # M3-R23: an Opus-quoted fact is keyed by its generator line (checked above); ds_bridge by its own cache
        absent = [b for b, k in keys.items() if b != ROLE_GEN
                  and not llm.cache_file("ds_bridge" if b == "ds_bridge" else FACTS_PASS, k).exists()]
        if absent:
            print(f"STOP (M3-R3 п. 11): {c.seed.seed_id}: ds_facts cache file missing for blocks {absent}")
            return 1
        c.provenance.setdefault("facts", {})["cache_keys"] = keys
        c.provenance["facts"]["prompt_version"] = llm.passes[FACTS_PASS].get("prompt_version")
    pairs = [to_pair(c, index, data, chunks) for c in kept]
    for p in pairs:
        want = {data.node_ids[int(data.chunk_node[data.chunk_row[c]])] for c in p.C_ref}
        assert {index.resolve(s) for s in p.y_ref["sources"]} == want, p.id
    incomplete = {p.id: m for p in pairs if (m := provenance_missing(p))}
    st = {"verified": len(cands) + unreviewed, "not_reader_good": unreviewed, "kept": len(pairs), "dropped": dict(sorted(Counter(dropped.values()).items())),
          "dropped_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
          "by_cell": dict(sorted(Counter(f"{p.pattern}|{p.split}|{p.lang}" for p in pairs).items())), "manual": manual,
          "provenance_incomplete": len(incomplete)}
    _stats_update("finalize", st)
    print(json.dumps(st, ensure_ascii=False))
    if incomplete:
        print(f"STOP (M3-R3 п. 11): incomplete provenance {dict(list(incomplete.items())[:5])}")
        return 1
    if manual and (manual["share"] < float(ds["manual_min_share"]) or manual["wilson_lb"] < float(ds["manual_min_wilson"])):
        print(f"STOP (M3-R3 п. 9): manual {manual}")
        return 1
    write_pairs(config.path(f"{OUT}/candidates.jsonl"), pairs)       # only after every gate (review m4)
    return 0


def cmd_trajectories(tcfg) -> int:
    from collections import Counter, defaultdict
    from functools import partial
    import numpy as np
    from graphrag.dataset.finalize import read_pairs, write_pairs
    from graphrag.dataset.trajectories import select, traj_context, traj_one
    from graphrag.env.parallel import run
    ds, par = tcfg["dataset"], tcfg["parallel"]
    cands = read_pairs(config.path(f"{OUT}/candidates.jsonl"))
    thr = float(ds["traj_min_reward"])
    try:                         # review m6: an init failure in a spawn worker only shows as BrokenProcessPool
        traj_context(ARGS.rankings)
    except Exception as e:
        print(f"STOP: trajectory context cannot load ({type(e).__name__}: {e}) -- dataset rankings (Task 19) built?")
        return 1
    res = run(traj_one, cands, int(par["workers"]), par["start_method"], init=partial(traj_context, ARGS.rankings),
              thr=thr, width=ARGS.width)
    why = Counter(f"{p.pattern}|{w}" for p, w, _, _, _ in res)
    by_cell = defaultdict(Counter)
    r_by, op_by = defaultdict(list), defaultdict(Counter)
    for p, w, r, ops, cell in res:
        by_cell[cell][w] += 1
        if r is not None:
            r_by[p.pattern].append(r)
            op_by[p.pattern].update(ops[:1])
    ok = [p for p, w, _, _, _ in res if w == "ok"]
    loss = {k: round(float(np.mean(np.array(v) < thr)), 4) for k, v in sorted(r_by.items())}
    st = {"candidates": len(cands), "width": ARGS.width, "reasons": dict(sorted(why.items())),
          "reasons_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
          "with_trajectory": len(ok),
          "R_quantiles": {k: [round(float(x), 4) for x in np.percentile(v, [5, 25, 50, 75, 95])]
                          for k, v in sorted(r_by.items())},
          "threshold_loss": loss, "first_op": {k: dict(v) for k, v in sorted(op_by.items())}}
    _jsonl_write(f"{OUT}/trajectories{'' if ARGS.width is None else f'_w{ARGS.width}'}.jsonl", [p.to_json() for p in ok])
    if ARGS.no_select:
        _stats_update("trajectories" if ARGS.width is None else f"trajectories_w{ARGS.width}", st)
        print(json.dumps(st, ensure_ascii=False))
        return 0
    ver = json.loads(config.path(f"{OUT}/stats.json").read_text(encoding="utf-8")).get("verify") or {}
    if not ver or ver.get("gate"):             # M3-R18 (a): cumulative judge sensitivity (pilot + main) before select
        print(f"STOP (M3-R3 п. 8): no accepted ds_verify gate in {OUT}/stats.json: {ver.get('gate', 'missing')}")
        return 1
    sel = {}
    chosen, short = select(ok, int(ds["size"]), tcfg["split"]["ratios"], ds["hops"], ds["langs"], ds["seed"],
                           int(ds["max_per_doc_cell"]), float(ds["max_template_share"]), transfer=ds.get("transfer"),
                           report=sel)                                        # M3-R20: no transfer configured
    st.update(chosen=len(chosen), shortfall=short, selection=sel,
              chosen_by_cell=dict(sorted(Counter(f"{p.pattern}|{p.split}|{p.lang}" for p in chosen).items())))
    _stats_update("trajectories", st)
    print(json.dumps(st, ensure_ascii=False))
    over = {k: v for k, v in loss.items() if v > float(ds["max_threshold_loss"])}
    if over:
        print(f"STOP (П5): R >= {thr} rejects more than {ds['max_threshold_loss']} of STOP ∧ full-recall "
              f"trajectories in {over}; ask the user -- the threshold is not changed without a decision")
        return 1
    if short:
        print(f"STOP (П4, M3-R3 п. 1): shortfall {short}; report to the user -- no backfill")
        return 1
    write_pairs(config.path(f"{OUT}/pairs.jsonl"), chosen)
    return 0


def cmd_project(tcfg) -> int:
    """Pre-pilot re-review I3: LLM calls per pass for the pilot + the full run (graphrag.dataset.budget), from the
    pilot's measured rates ({PILOT_DIR}/stats.json) and the ds_facts cache (blocks already paid)."""
    from graphrag.dataset.seeds import Seed
    full =[Seed.from_json(r) for r in _jsonl_read(f"{OUT}/seeds.jsonl")]
    out = _project_calls(tcfg, full)
    _stats_update("projection", out)
    print(json.dumps(out, ensure_ascii=False))
    cap = int(tcfg["dataset_llm"]["max_calls_total"])
    if out["projection"]["total"]["total"] > cap:
        print(f"STOP (П2): projected {out['projection']['total']['total']} calls > {cap}; ask the user")
        return 1
    return 0


def _project_calls(tcfg, full) -> dict:
    """graphrag.dataset.budget.project_calls for the seeds `full` (pilot seeds included, their calls cached) with the
    pilot's measured rates from {PILOT_DIR}/stats.json."""
    from graphrag.dataset.budget import project_calls
    from graphrag.dataset.facts import PASS as FACTS_PASS, SCHEMA_FACTS, facts_prompt
    from graphrag.dataset.seeds import Seed, block_of
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.llm import _canon_input, prompt_digest
    ds = tcfg["dataset"]
    recs, chunks = _chunks()
    blk = block_of(recs, int(ds["facts_block"]))
    index = SourceIndex(recs)
    pilot = [Seed.from_json(r) for r in _jsonl_read(f"{PILOT_DIR}/seeds.jsonl")]
    ps = json.loads(config.path(f"{PILOT_DIR}/stats.json").read_text(encoding="utf-8"))
    llm = _llm(tcfg)
    digest = prompt_digest(facts_prompt(int(ds["max_facts_per_chunk"])), SCHEMA_FACTS)
    members: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            members.setdefault(blk[c.chunk_id], []).append(c)
    cached = set()
    for b in sorted({blk[c] for s in full for c in s.chunks}):
        frags = [{"id": f"C{i + 1}", "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent),
                  "text": c.text} for i, c in enumerate(members[b])]
        if llm.cache_file(FACTS_PASS, llm.cache_key(FACTS_PASS, _canon_input({"block": b, "fragments": frags}),
                                                    digest)).exists():
            cached.add(b)
    f, g, v = ps["facts"], ps["generate"], ps["verify"]
    no_facts = sum(n for k, n in g["rejected"].items() if k.endswith("|no_facts"))
    rates = {"facts_retry": f["rejected"].get("retried_bad_answer", 0) / max(1, f["blocks"]),
             "usable": 1 - no_facts / max(1, g["seeds"]), "gen_yield": g["accept_rate"],
             "views_per_call": v["views"] / max(1, v["calls"]),
             "plant_views": v["views_plants"] / max(1, v["plants"]), "verify_pilot_calls": v["calls"]}
    cfg = {"gen_batch": ds["gen_batch"], "verify_batch": ds["verify_batch"], "plant_share": ds["plant_share"]}
    return {"rates": rates, "projection": project_calls(full, pilot, blk, cached, rates, cfg)}


def cmd_pilot_seeds(tcfg) -> int:
    """Task 17: the stratified pilot -- pilot_per_cell seeds per (pattern, language) from data/dataset/seeds.jsonl, and
    the dataset ledger before any pilot call (calls per pass = ledger after - before)."""
    from graphrag.dataset.pilot import pilot_seeds
    from graphrag.dataset.seeds import Seed
    from graphrag.kg.llm import KgLLM
    ds = tcfg["dataset"]
    pick = pilot_seeds([Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")],
                       int(ds["pilot_per_cell"]), ds["seed"])
    _jsonl_write(f"{PILOT_DIR}/seeds.jsonl", [s.to_json() for s in pick])
    led = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False).ledger()
    before = config.path(f"{PILOT_DIR}/ledger_before.json")
    if not before.exists():                   # re-running pilot-seeds after the live pass must not move the baseline
        before.write_text(json.dumps(led, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_seeds": len(pick), "ledger_before": json.loads(before.read_text(encoding="utf-8"))}))
    return 0


def cmd_repilot_seeds(tcfg) -> int:
    """M3-R20 re-pilot seeds, drawn ONLY from fact blocks the ds_facts cache already holds (0 ds_facts calls):
    the Task 17 pilot's single seeds, new singles (a language round-robin, sections in turn) and filtered multi walks
    (bridge / aggregation, every fragment with a fact, no redundant fragment), languages round-robin. Writes
    {PILOT_DIR}/seeds.jsonl and the ledger baseline (once). The full run forces these seeds (seeds --forced)."""
    import hashlib
    from collections import Counter
    from graphrag.dataset.facts import extract
    from graphrag.dataset.generate import redundant_fragment
    from graphrag.dataset.seeds import Seed, _h, block_of
    from graphrag.dataset.sources import SourceIndex
    from graphrag.dataset.walks import walk_class
    from graphrag.kg.llm import KgLLM
    ds, rp = tcfg["dataset"], tcfg["dataset"]["repilot"]
    recs, splits, _, pool, _ = _pool(tcfg)
    chunks = {c.chunk_id: c for r in recs for c in r.chunks}
    sections = {r.doc_id: r.meta.section for r in recs}
    blk = block_of(recs, int(ds["facts_block"]))
    block_chunks: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            block_chunks.setdefault(blk[c.chunk_id], []).append(c.chunk_id)
    old = [Seed.from_json(r) for r in _jsonl_read("data/dataset/pilot/seeds.jsonl")]
    off = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False)
    cached = {blk[c] for s in old for c in s.chunks}
    facts, _ = extract(cached, block_chunks, chunks, SourceIndex(recs), off, int(ds["max_facts_per_chunk"]))
    if off.total_api_calls():
        raise RuntimeError("re-pilot seeds must not call the API")
    h = lambda k: hashlib.sha256(f"{ds['seed']}:repilot|{k}".encode()).hexdigest()
    langs = list(ds["langs"])
    used_chunks, used_docs = set(), Counter()
    out = []
    for s in sorted((s for s in old if s.pattern == "single"), key=lambda s: s.seed_id)[:int(rp["single_old"])]:
        out.append(s)
        used_chunks |= set(s.chunks)
        used_docs[s.docs[0]] += 1
    def mk(pat, lang, x):
        ch, docs, walk = x
        return Seed(f"{pat}-{_h(ds['seed'], '|'.join(ch))[:10]}", pat, lang, tuple(ch), tuple(docs), splits[docs[0]],
                    sections[docs[0]], tuple(walk))
    singles = sorted((x for x in pool["single"] if blk[x[0][0]] in cached and facts.get(x[0][0])
                      and x[0][0] not in used_chunks), key=lambda x: h("|".join(x[0])))
    secs = sorted({sections[x[1][0]] for x in singles})
    n, i = 0, 0
    while n < int(rp["single_new"]) and singles:
        sec = secs[i % len(secs)]
        i += 1
        x = next((x for x in singles if sections[x[1][0]] == sec and used_docs[x[1][0]] < 2), None)
        if x is None:
            if i > 10 * int(rp["single_new"]):
                break
            continue
        singles.remove(x)
        out.append(mk("single", langs[n % len(langs)], x))
        used_chunks.add(x[0][0])
        used_docs[x[1][0]] += 1
        n += 1
    multi = [x for x in pool["multi"] if all(blk[c] in cached and facts.get(c) for c in x[0])
             and not redundant_fragment(Seed("x", "multi", "ru", tuple(x[0]), tuple(x[1]), "", ""), facts,
                                        {c: chunks[c].text for c in x[0]})]
    k, spare = 0, 0
    for cls in ("bridge", "aggregation"):                 # a bridge shortfall is filled by aggregation walks
        cand = sorted((x for x in multi if walk_class(x[2]) == cls), key=lambda x: h("|".join(x[0])))
        got, want = 0, int(rp[cls]) + spare
        for x in cand:
            if got >= want:
                break
            if set(x[0]) & used_chunks or any(used_docs[d] >= 6 for d in x[1]):
                continue
            out.append(mk("multi", langs[k % len(langs)], x))
            used_chunks |= set(x[0])
            used_docs.update(x[1])
            got += 1
            k += 1
        spare = want - got
    out = sorted(out, key=lambda s: s.seed_id)
    _jsonl_write(f"{PILOT_DIR}/seeds.jsonl", [s.to_json() for s in out])
    before = config.path(f"{PILOT_DIR}/ledger_before.json")
    if not before.exists():
        before.write_text(json.dumps(off.ledger(), sort_keys=True) + "\n", encoding="utf-8")
    rep = {"seeds": len(out), "cached_blocks": len(cached), "multi_pool_in_cache": len(multi),
           "by_cell": dict(sorted(Counter(f"{s.pattern}|{s.lang}" for s in out).items())),
           "by_class": dict(Counter(walk_class(s.walk) for s in out if s.walk)),
           "by_split": dict(sorted(Counter(f"{s.pattern}|{s.split}" for s in out).items())),
           "single_by_section": dict(sorted(Counter(s.section for s in out if s.pattern == "single").items()))}
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_repilot_report(tcfg) -> int:
    """M3-R20 re-pilot report (generate + verify stages; trajectories need Task 19 rankings): live calls per pass
    (ledger diff), yield per pattern × language per stage, judge sensitivity, and the full-run projection from the
    measured rates -- block-based ds_facts (graphrag.dataset.budget), ds_generate / ds_verify per seed, projected
    accepted per cell (seeds × verified yield of the pattern × language) vs the final need. Writes
    {PILOT_DIR}/report.json; exit 1 on a user stop (П2 total, П4 cell, M3-R3 п. 8 sensitivity)."""
    import math
    from collections import Counter
    from graphrag.dataset.budget import blocks_of
    from graphrag.dataset.facts import PASS as FACTS_PASS, SCHEMA_FACTS, facts_prompt
    from graphrag.dataset.pilot import pass_calls
    from graphrag.dataset.seeds import Seed, block_of
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.audit import wilson
    from graphrag.kg.llm import KgLLM, _canon_input, prompt_digest
    ds = tcfg["dataset"]
    P = PILOT_DIR
    st = json.loads(config.path(f"{P}/stats.json").read_text(encoding="utf-8"))
    pseeds = [Seed.from_json(r) for r in _jsonl_read(f"{P}/seeds.jsonl")]
    before = json.loads(config.path(f"{P}/ledger_before.json").read_text(encoding="utf-8"))
    off = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False)
    after = off.ledger()
    calls = pass_calls(before, after)
    spent = sum(int(v) for p in after.values() for v in p.values())
    cell = lambda s: f"{s.pattern}|{s.lang}"
    n_cell = Counter(cell(s) for s in pseeds)
    gen = Counter(cell(Seed.from_json(r["seed"])) for r in _jsonl_read(f"{P}/candidates_raw.jsonl"))
    ver = Counter(cell(Seed.from_json(r["seed"])) for r in _jsonl_read(f"{P}/verified.jsonl"))
    yields = {k: {"seeds": n, "generate": gen[k], "verify": ver[k], "rate": round(ver[k] / n, 4),
                  "wilson": [round(x, 4) for x in wilson(ver[k], n)]} for k, n in sorted(n_cell.items())}
    by_pat = Counter()
    for k, v in ver.items():
        by_pat[k.split("|")[0]] += v
    n_pat = Counter(s.pattern for s in pseeds)
    rate = {p: by_pat[p] / n_pat[p] for p in n_pat}
    rate_w = {p: [round(x, 4) for x in wilson(by_pat[p], n_pat[p])] for p in n_pat}
    g_rate = {p: sum(v for k, v in gen.items() if k.startswith(p + "|")) / n_pat[p] for p in n_pat}
    # full run: every seed of data/dataset/seeds.jsonl (pilot seeds forced into it, their calls cached)
    full = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    recs, chunks = _chunks()
    blk = block_of(recs, int(ds["facts_block"]))
    index = SourceIndex(recs)
    digest = prompt_digest(facts_prompt(int(ds["max_facts_per_chunk"])), SCHEMA_FACTS)
    members: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            members.setdefault(blk[c.chunk_id], []).append(c)
    new_blocks = 0
    for b in sorted(blocks_of(full, blk)):
        frags = [{"id": f"C{i + 1}", "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent),
                  "text": c.text} for i, c in enumerate(members[b])]
        if not off.cache_file(FACTS_PASS, off.cache_key(FACTS_PASS, _canon_input({"block": b, "fragments": frags}),
                                                        digest)).exists():
            new_blocks += 1
    pid = {s.seed_id for s in pseeds}
    rest = [s for s in full if s.seed_id not in pid]
    retry = 1 + st["facts"]["rejected"].get("retried_bad_answer", 0) / max(1, st["facts"]["blocks"])
    v = st["verify"]
    views_per_call = v["views"] / max(1, v["calls"])
    usable = 1 - sum(n for k, n in st["generate"]["rejected"].items() if k.endswith(("|no_facts", "|redundant"))) \
        / max(1, st["generate"]["seeds"])
    n_gen = math.ceil(len(rest) * usable / int(ds["gen_batch"]))
    cand = sum(g_rate[s.pattern] for s in rest if s.pattern in g_rate)
    views = sum(g_rate[s.pattern] * (1 if len(s.chunks) == 1 else 1 + len(s.chunks)) for s in rest if s.pattern in g_rate)
    plants = cand / int(ds["verify_batch"]) * float(ds["plant_share"])
    plant_views = v["views_plants"] / max(1, v["plants"])
    n_ver = math.ceil((views + plants * plant_views) / views_per_call)
    n_agree = math.ceil(cand * ((v.get("agree") or {}).get("checked", 0) / max(1, v["checked"])) / int(ds["verify_agree_per_call"]))
    new = {"ds_facts": math.ceil(new_blocks * retry), "ds_generate": n_gen, "ds_verify": n_ver + n_agree}
    total = spent + sum(new.values())
    # projected accepted per cell (pattern × split × language) vs the final need; CI from the pattern's Wilson bounds
    got = Counter(f"{s.pattern}|{s.split}|{s.lang}" for s in full)
    final = _final_need(tcfg)
    proj = {k: {"seeds": got[k], "need": n, "point": math.floor(got[k] * rate.get(k.split("|")[0], 0)),
                "low": math.floor(got[k] * rate_w.get(k.split("|")[0], [0, 0])[0])} for k, n in sorted(final.items())}
    short = {k: x for k, x in proj.items() if x["point"] < x["need"]}
    sens = v.get("sensitivity_by_kind") or {}
    rep = {"calls": calls, "spent_total": spent, "accepted": sum(ver.values()),
           "calls_per_accepted": round(sum(calls.values()) / max(1, sum(ver.values())), 3),
           "yield_by_pattern": {p: {"rate": round(rate[p], 4), "wilson": rate_w[p], "generate_rate": round(g_rate[p], 4)}
                                for p in sorted(rate)},
           "yield_by_pattern_lang": yields, "generate_rejected": st["generate"]["rejected"],
           "verify": {k: v.get(k) for k in ("rejected", "sensitivity_by_kind", "sensitivity_wilson_by_kind", "planted",
                                            "kinds_below_min_n", "code_sensitivity_by_kind", "calls", "views", "agree")},
           "projection": {"full_seeds": len(full), "new_blocks": new_blocks, "full_new": new, "total": total,
                          "cap": int(tcfg["dataset_llm"]["max_calls_total"])},
           "projected_accepted_by_cell": proj, "cells_below_need": sorted(short)}
    gates = []
    if total > int(tcfg["dataset_llm"]["max_calls_total"]):
        gates.append(f"П2: projected {total} calls > {tcfg['dataset_llm']['max_calls_total']}")
    if short:
        gates.append(f"П4: projected accepted below the final need in {len(short)} cells")
    gates += [f"M3-R3 п. 8: {k} sensitivity {x} < {ds['min_sensitivity_kind']}" for k, x in sorted(sens.items())
              if x < float(ds["min_sensitivity_kind"])]
    rep["gates"] = gates
    config.path(f"{P}/report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 1 if gates else 0


def cmd_pilot_report(tcfg) -> int:
    """Task 17: calls per accepted pair per pass (ledger), yield per pattern (× language) per stage, the R
    distribution, judge sensitivity, oracle width loss vs w128, the full-run projection and the user gates."""
    import math
    from collections import Counter
    from graphrag.dataset.pilot import gates, pass_calls, project
    from graphrag.dataset.seeds import PATTERNS, Seed
    from graphrag.dataset.trajectories import final_needs
    from graphrag.kg.llm import KgLLM
    ds = tcfg["dataset"]
    P = PILOT_DIR
    st = json.loads(config.path(f"{P}/stats.json").read_text(encoding="utf-8"))
    pseeds = [Seed.from_json(r) for r in _jsonl_read(f"{P}/seeds.jsonl")]
    before = json.loads(config.path(f"{P}/ledger_before.json").read_text(encoding="utf-8"))
    after = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False).ledger()
    calls = pass_calls(before, after)
    cell = lambda s: f"{s.pattern}|{s.lang}"
    n_seed = Counter(s.pattern for s in pseeds)
    n_cell = Counter(cell(s) for s in pseeds)
    stage = {}
    for name, f in (("generate", "candidates_raw.jsonl"), ("verify", "verified.jsonl")):
        stage[name] = Counter()
        for r in _jsonl_read(f"{P}/{f}"):
            s = Seed.from_json(r["seed"])
            stage[name][cell(s)] += 1
    stage["finalize"] = Counter(f"{r['pattern']}|{r['lang']}" for r in _jsonl_read(f"{P}/candidates.jsonl"))
    stage["trajectory"] = Counter(f"{r['pattern']}|{r['lang']}" for r in _jsonl_read(f"{P}/trajectories.jsonl"))
    ok = Counter()
    for k, v in stage["trajectory"].items():
        ok[k.split("|")[0]] += v
    acc = sum(ok.values())
    rate = {p: round(ok[p] / n_seed[p], 4) if n_seed[p] else 0.0 for p in PATTERNS}
    prev = {"generate": n_cell, "verify": stage["generate"], "finalize": stage["verify"], "trajectory": stage["finalize"]}
    yields = {k: {name: {"n": stage[name][k], "of": prev[name][k],
                         "rate": round(stage[name][k] / prev[name][k], 4) if prev[name][k] else None}
                  for name in ("generate", "verify", "finalize", "trajectory")}
              | {"end_to_end": round(stage["trajectory"][k] / n_cell[k], 4) if n_cell[k] else None}
              for k in sorted(n_cell)}
    spent = sum(int(v) for p in after.values() for v in p.values())
    naive = project({p: c / max(acc, 1) for p, c in calls.items()}, rate, int(ds["size"]), ds["hops"],
                    float(ds["target_margin"]), spent)
    # re-sample the full run with the projected overshoot (target_margin / pilot yield), pilot seeds forced (Task 18)
    recs, splits, _, pool, _ = _pool(tcfg)
    over = {p: (naive["overshoot"][p] if math.isfinite(naive["overshoot"][p]) else float(ds["overshoot"][p]))
            for p in PATTERNS}
    full, over_short = _sample(tcfg, pool, splits, recs, over, pseeds)
    got = Counter(f"{s.pattern}|{s.split}|{s.lang}" for s in full)
    proj_acc = {k: math.floor(n * rate[k.split("|")[0]]) for k, n in got.items()}
    final = final_needs(int(ds["size"]), ds["hops"], tcfg["split"]["ratios"], ds["langs"])
    eff, transfer = dict(final), {}                     # M3-R20: no transfer (one multi pattern)
    below = {k: n - proj_acc.get(k, 0) for k, n in final.items() if proj_acc.get(k, 0) < n}
    # M3-R13: below = a cell whose projected ACCEPTED count is under its effective final need (the overshoot is a
    # yield assumption, so an overshoot shortage alone is reported, not a stop)
    pc = _project_calls(tcfg, full)
    new = {p: v["full_new"] for p, v in pc["projection"].items() if p.startswith("ds_")}
    corrected = {"spent": spent, "spent_by_pass": {p: sum(int(x) for x in (after.get(p) or {}).values())
                                                   for p in ("ds_facts", "ds_generate", "ds_verify")},
                 "full_new": new, "by_pass_total": {}, "rates": pc["rates"], "blocks": pc["projection"]["blocks"],
                 "full_seeds": len(full), "full_seeds_by_pattern": dict(Counter(s.pattern for s in full)),
                 "overshoot": over, "total": spent + sum(new.values())}
    corrected["by_pass_total"] = {p: corrected["spent_by_pass"][p] + new[p] for p in new}
    tj, t128 = st["trajectories"], st.get("trajectories_w128") or {}
    ids32 = {r["id"] for r in _jsonl_read(f"{P}/trajectories.jsonl")}
    ids128 = {r["id"] for r in _jsonl_read(f"{P}/trajectories_w128.jsonl")} if t128 else set()
    loss = round(len(ids128 - ids32) / len(ids128), 4) if ids128 else None
    ver = st["verify"]
    rep = {"calls": calls, "accepted": acc,
           "calls_per_accepted": {p: round(c / max(acc, 1), 4) for p, c in calls.items()},
           "yield": rate, "yield_by_pattern_lang": yields, "projection": corrected, "projection_per_pair_upper": naive,
           "shortage": below, "overshoot_shortage": over_short, "projected_accepted_by_cell": dict(sorted(proj_acc.items())),
           "effective_final_need_by_cell": eff, "projected_transfer": transfer,
           "threshold_loss": tj["threshold_loss"], "R_quantiles": tj["R_quantiles"], "first_op": tj["first_op"],
           "trajectory_reasons": tj["reasons"], "facts": st["facts"], "generate_rejected": st["generate"]["rejected"],
           "verify": {k: ver.get(k) for k in ("sensitivity_by_kind", "sensitivity_wilson_by_kind", "planted",
                                              "code_sensitivity_by_kind", "rejected", "kinds_below_min_n", "pilot",
                                              "calls", "views", "gate")},
           "verify_ok": not ver["gate"] and not ver["kinds_below_min_n"] and all(
               v >= float(ds["min_sensitivity_kind"]) for v in ver["sensitivity_by_kind"].values()),
           "oracle": {"width": tcfg["oracle"]["beam_width"], "accepted": len(ids32), "accepted_w128": len(ids128),
                      "lost_vs_w128": sorted(ids128 - ids32), "gained_vs_w128": sorted(ids32 - ids128),
                      "candidates": tj["candidates"]},
           "oracle_loss_vs_w128": loss, "oracle_ok": loss is not None and loss <= 0.02,
           "finalize_dropped": st["finalize"]["dropped"]}
    rep["gates"] = gates(rep, {"max_calls_total": int(tcfg["dataset_llm"]["max_calls_total"]),
                               "max_threshold_loss": float(ds["max_threshold_loss"])})
    if tj["candidates"] < 60:
        rep["gates"].append(f"M3-R6 п. 4: oracle loss measured on {tj['candidates']} < 60 pilot candidates")
    config.path(f"{P}/report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 1 if rep["gates"] else 0


def cmd_calibrate_copy(tcfg) -> int:
    """M3-R21 (2): copy-run / lexical-overlap distributions of the 340 Q-real questions (Cyrillic and Latin forms)
    against their gold fragments (core ∪ support), per language family; threshold = P90. Writes
    {OUT}/copy_calibration.json; exit 1 if config dataset.copy_calibration differs (paste the printed block)."""
    from graphrag.dataset.generate import calibrate_copy
    from graphrag.dataset.sources import SourceIndex
    from graphrag.data.normalize import segment_is_russian
    recs, chunks = _chunks()
    index = SourceIndex(recs)
    qs = []
    for r in _jsonl_read(tcfg["gold"]["file"]):
        ids = list(dict.fromkeys(r["core"] + r["support"]))
        frags = [{"act": index.act[chunks[c].parent.split("/", 1)[0]], "where": index.format(chunks[c].parent),
                  "text": chunks[c].text} for c in ids]
        for q in (r["question_cyr"], r.get("question_lat")):
            if q:
                qs.append(("ru" if segment_is_russian(q, False) else "uz", q, frags))
    fams = calibrate_copy(qs, 0.90)
    out = {"source": tcfg["gold"]["file"], "quantile": 0.90, "families": fams}
    _jsonl_write(f"{OUT}/copy_calibration.jsonl", [out])
    print(json.dumps(out, ensure_ascii=False))
    have = (tcfg["dataset"].get("copy_calibration") or {}).get("families") or {}
    for fam, d in fams.items():
        h = have.get(fam) or {}
        if (h.get("max_copy_words"), h.get("max_lexical_overlap"), h.get("n")) != \
                (d["max_copy_words"], d["max_lexical_overlap"], d["n"]):
            print(f"config dataset.copy_calibration.families.{fam} differs from the measurement")
            return 1
    return 0


def _used_chunks() -> set[str]:
    out = set()
    for d in ("data/dataset/pilot", "data/dataset/pilot2"):
        for name in ("seeds.jsonl", "bridge_seeds.jsonl"):
            p = config.path(f"{d}/{name}")
            if p.exists():
                out |= {c for r in _jsonl_read(p) for c in r["chunks"]}
    return out


def cmd_pilot3_seeds(tcfg) -> int:
    """M3-R21 pilot 3 seeds: `bridge_anchors` ds_bridge anchors stratified by bridge source (bridge.sample_anchors)
    over content fragments not used by the earlier pilots, and fresh Uzbek single seeds (`pilot3.single` per
    language) from data/dataset/seeds.jsonl (rebuilt with the M3-R21 doc cap) -- not in an earlier pilot, preferring
    fact blocks the ds_facts cache holds (0 ds_facts calls), sections in turn, <= 1 per document. Writes
    {OUT}/anchors.jsonl, {OUT}/seeds.jsonl, {OUT}/ledger_before.json (once)."""
    import hashlib
    from collections import Counter
    from graphrag.dataset.bridge import sample_anchors
    from graphrag.dataset.facts import PASS as FACTS_PASS, SCHEMA_FACTS, facts_prompt
    from graphrag.dataset.seeds import Seed, block_of
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.llm import KgLLM, _canon_input, prompt_digest
    ds, p3 = tcfg["dataset"], tcfg["dataset"]["pilot3"]
    recs, chunks = _chunks()
    bx = _bridge_index(tcfg, recs, chunks)
    used = _used_chunks()
    picks = sample_anchors(bx, sorted(bx.ok), int(p3["bridge_anchors"]), list(ds["langs"]), ds["seed"], exclude=used)
    _jsonl_write(f"{OUT}/anchors.jsonl", [{k: (v if k != "candidates" else {s: [[c, [list(h) for h in hops]]
                                                                             for c, hops in x] for s, x in v.items()})
                                           for k, v in p.items()} for p in picks])
    off = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False)
    blk = block_of(recs, int(ds["facts_block"]))
    index = SourceIndex(recs)
    digest = prompt_digest(facts_prompt(int(ds["max_facts_per_chunk"])), SCHEMA_FACTS)
    members: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            members.setdefault(blk[c.chunk_id], []).append(c)

    def cached(b):
        frags = [{"id": f"C{i + 1}", "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent),
                  "text": c.text} for i, c in enumerate(members[b])]
        return off.cache_file(FACTS_PASS, off.cache_key(FACTS_PASS, _canon_input({"block": b, "fragments": frags}),
                                                        digest)).exists()
    prior = set()
    for d in ("data/dataset/pilot", "data/dataset/pilot2"):
        prior |= {r["seed_id"] for r in _jsonl_read(f"{d}/seeds.jsonl")}
    h = lambda k: hashlib.sha256(f"{ds['seed']}:pilot3|{k}".encode()).hexdigest()
    from graphrag.dataset.seeds import Seed as _S
    full = [_S.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    pool = [s for s in full if s.pattern == "single" and s.lang != "ru" and s.seed_id not in prior
            and not set(s.chunks) & used]
    out, docs = [], Counter()
    for lang in ("uz-Cyrl", "uz-Latn"):
        mine = sorted((s for s in pool if s.lang == lang), key=lambda s: (not cached(blk[s.chunks[0]]), h(s.seed_id)))
        secs = sorted({s.section for s in mine})
        n, i = 0, 0
        while n < int(p3["single_per_lang"]) and i < 50 * len(secs):
            sec = secs[i % len(secs)]
            i += 1
            s = next((s for s in mine if s.section == sec and docs[s.docs[0]] < 1 and s not in out), None)
            if s is None:
                continue
            out.append(s)
            docs[s.docs[0]] += 1
            n += 1
    out = sorted(out, key=lambda s: s.seed_id)
    _jsonl_write(f"{OUT}/seeds.jsonl", [s.to_json() for s in out])
    before = config.path(f"{OUT}/ledger_before.json")
    if not before.exists():
        before.write_text(json.dumps(off.ledger(), sort_keys=True) + "\n", encoding="utf-8")
    rep = {"anchors": len(picks), "anchors_by_stratum": dict(Counter(p["stratum"] for p in picks)),
           "anchors_by_lang": dict(Counter(p["lang"] for p in picks)),
           "anchors_by_split": dict(Counter(p["split"] for p in picks)),
           "singles": len(out), "singles_by_lang": dict(Counter(s.lang for s in out)),
           "singles_by_section": dict(Counter(s.section for s in out)),
           "single_blocks_uncached": len({blk[s.chunks[0]] for s in out if not cached(blk[s.chunks[0]])})}
    _stats_update("pilot3_seeds", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def _read_picks():
    return [{**p, "candidates": {s: [(c, tuple(tuple(h) for h in hops)) for c, hops in x]
                                 for s, x in p["candidates"].items()}} for p in _jsonl_read(f"{OUT}/anchors.jsonl")]


def cmd_bridge(tcfg) -> int:
    """M3-R21 ds_bridge over {OUT}/anchors.jsonl -> {OUT}/bridge_raw.jsonl (+ bridge_seeds.jsonl, stats "bridge")."""
    from graphrag.dataset.bridge import bridge
    from graphrag.dataset.sources import SourceIndex
    ds = tcfg["dataset"]
    recs, chunks = _chunks()
    index = SourceIndex(recs)
    bx = _bridge_index(tcfg, recs, chunks)
    llm = _llm(tcfg)
    cands, st = bridge(_read_picks(), chunks, index, llm, ds, {r.doc_id: r.meta.section for r in recs},
                       glossary=ds.get("legal_glossary"), batch=int(ds["bridge_batch"]), amended=_amended(bx, index))
    _jsonl_write(f"{OUT}/bridge_raw.jsonl", _cand_rows(cands))
    _jsonl_write(f"{OUT}/bridge_seeds.jsonl", [c.seed.to_json() for c in cands])
    _stats_update("bridge", {**st, "llm": llm.stats_json()})
    print(json.dumps(st, ensure_ascii=False))
    return 0


def cmd_quality(tcfg) -> int:
    """M3-R22 ds_quality over {OUT}/verified.jsonl -> {OUT}/quality.jsonl (GOOD only)."""
    from graphrag.dataset.quality import judge_quality
    recs, chunks = _chunks()
    frag_of, _ = _frag_of(recs, chunks)
    llm = _llm(tcfg)
    kept, st = judge_quality(_rows_cands(_jsonl_read(f"{OUT}/verified.jsonl")), frag_of, llm,
                             int(tcfg["dataset"]["quality_batch"]), _pilot_ids())
    _jsonl_write(f"{OUT}/quality.jsonl", _cand_rows(kept))
    _stats_update("quality", {**st, "llm": llm.stats_json()})
    print(json.dumps(st, ensure_ascii=False))
    return 0


def _reading_labels(path) -> dict[str, str]:
    import re
    out = {}
    for line in config.path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \d+ \| (single-\w+|multi-\w+) \| \S+ \| (GOOD|BORDERLINE|BAD) \|", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cmd_quality_calibrate(tcfg) -> int:
    """M3-R22 (3): ds_quality on the 17 re-pilot accepted pairs (data/dataset/pilot2/verified.jsonl) vs the strict
    reader's labels (repilot-reading.md). Target: exact agreement >= 15/17 and 0 of the reader's BAD accepted.
    Writes {OUT}/quality_calibration.json; exit 1 below target."""
    from graphrag.dataset.quality import calibration, judge_quality
    q = tcfg["dataset"]["quality_calibration"]
    labels = _reading_labels(q["labels"])
    recs, chunks = _chunks()
    frag_of, _ = _frag_of(recs, chunks)
    cands = [c for c in _rows_cands(_jsonl_read(q["pairs"])) if c.seed.seed_id in labels]
    llm = _llm(tcfg)
    _, st = judge_quality(cands, frag_of, llm, int(tcfg["dataset"]["quality_batch"]))
    got = {c.seed.seed_id: c.provenance["quality"]["verdict"] for c in cands if "quality" in c.provenance}
    cal = calibration(labels, got)
    rep = {"calibration": cal, "stats": st, "llm": llm.stats_json(),
           "items": {c.seed.seed_id: {"reader": labels[c.seed.seed_id], **c.provenance.get("quality", {})}
                     for c in cands}}
    config.path(f"{OUT}/quality_calibration.json").parent.mkdir(parents=True, exist_ok=True)
    config.path(f"{OUT}/quality_calibration.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(cal, ensure_ascii=False))
    ok = cal["exact"] >= int(q["min_exact"]) and not cal["bad_accepted"]
    return 0 if ok else 1


def cmd_pilot3_report(tcfg) -> int:
    """M3-R21/R22 pilot 3 report over {OUT}: live calls per pass (ledger diff), ds_bridge yield per stratum (anchors ->
    neighbour chosen -> generate -> verified -> ds_quality GOOD), Uzbek single yield, judge sensitivity, the full-run
    projection (calls within the 2500 cap; accepted per cell vs the final need) and the STOP gates. Writes
    {OUT}/report.json and {OUT}/accepted_pairs.jsonl (GOOD pairs with fragment texts, for the independent reader)."""
    import math
    from collections import Counter
    from graphrag.dataset.facts import PASS as FACTS_PASS, SCHEMA_FACTS, facts_prompt
    from graphrag.dataset.pilot import pass_calls
    from graphrag.dataset.seeds import Seed, block_of, cell_counts
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.audit import wilson
    from graphrag.kg.llm import KgLLM, _canon_input, prompt_digest
    ds = tcfg["dataset"]
    st = json.loads(config.path(f"{OUT}/stats.json").read_text(encoding="utf-8"))
    before = json.loads(config.path(f"{OUT}/ledger_before.json").read_text(encoding="utf-8"))
    off = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False)
    after = off.ledger()
    passes = ("ds_facts", "ds_generate", "ds_verify", "ds_bridge", "ds_quality")
    tot = lambda led, p: sum(int(v) for v in (led.get(p) or {}).values())
    calls = {p: tot(after, p) - tot(before, p) for p in passes}
    spent = sum(tot(after, p) for p in after)
    rows = lambda f: _jsonl_read(f"{OUT}/{f}") if config.path(f"{OUT}/{f}").exists() else []
    gen, ver, good = rows("candidates_raw.jsonl") + rows("bridge_raw.jsonl"), rows("verified.jsonl"), rows("quality.jsonl")
    sid = lambda r: r["seed"]["seed_id"]
    vset, gset = {sid(r) for r in ver}, {sid(r) for r in good}
    br = st.get("bridge") or {}
    by_str = {}
    for s, v in (br.get("by_stratum") or {}).items():
        mine = [r for r in gen if r["provenance"].get("stratum") == s]
        by_str[s] = {"anchors": v.get("anchors", 0), "chosen": v.get("chosen", 0), "generate": len(mine),
                     "verified": sum(sid(r) in vset for r in mine), "good": sum(sid(r) in gset for r in mine)}
    b_anchors = sum(x["anchors"] for x in by_str.values())
    b_good = sum(x["good"] for x in by_str.values())
    singles = [Seed.from_json(r) for r in rows("seeds.jsonl")]
    by_lang = {}
    for lang in sorted({s.lang for s in singles}):
        ids = {s.seed_id for s in singles if s.lang == lang}
        g = [r for r in gen if sid(r) in ids]
        by_lang[lang] = {"seeds": len(ids), "generate": len(g), "verified": sum(sid(r) in vset for r in g),
                         "good": sum(sid(r) in gset for r in g),
                         "wilson": [round(x, 4) for x in wilson(sum(sid(r) in gset for r in g), len(ids))]}
    v = st.get("verify") or {}
    q = st.get("quality") or {}
    sens = v.get("sensitivity_by_kind") or {}
    # ---- projection -------------------------------------------------------------------------------------------------
    cal = json.loads(config.path(f"{OUT}/quality_calibration.json").read_text(encoding="utf-8")) \
        if config.path(f"{OUT}/quality_calibration.json").exists() else {}
    ru_items = [x for k, x in (cal.get("items") or {}).items()]
    p2 = json.loads(config.path("data/dataset/pilot2/report.json").read_text(encoding="utf-8"))
    ru2 = p2["yield_by_pattern_lang"]["single|ru"]
    ru_ver = [k for k, x in (cal.get("items") or {}).items()]
    p2v = {r["seed"]["seed_id"]: r["seed"]["lang"] for r in _jsonl_read("data/dataset/pilot2/verified.jsonl")}
    ru_good = sum(1 for k, x in (cal.get("items") or {}).items() if p2v.get(k) == "ru" and x.get("verdict") == "GOOD")
    y_single = {l: x["good"] / x["seeds"] for l, x in by_lang.items() if x["seeds"]}
    y_single["ru"] = ru_good / max(1, ru2["seeds"])           # estimate: pilot2 ru (ds_generate v4) x ds_quality GOOD
    y_multi = b_good / max(1, b_anchors)
    full = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    prior = set()
    for d in ("data/dataset/pilot", "data/dataset/pilot2", OUT):
        if config.path(f"{d}/seeds.jsonl").exists():
            prior |= {r["seed_id"] for r in _jsonl_read(f"{d}/seeds.jsonl")}
    rest = [s for s in full if s.pattern == "single" and s.seed_id not in prior]
    recs, chunks = _chunks()
    blk = block_of(recs, int(ds["facts_block"]))
    index = SourceIndex(recs)
    digest = prompt_digest(facts_prompt(int(ds["max_facts_per_chunk"])), SCHEMA_FACTS)
    members: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            members.setdefault(blk[c.chunk_id], []).append(c)
    new_blocks = 0
    for b in sorted({blk[c] for s in rest for c in s.chunks}):
        frags = [{"id": f"C{i + 1}", "act": index.act[c.parent.split("/", 1)[0]], "where": index.format(c.parent),
                  "text": c.text} for i, c in enumerate(members[b])]
        if not off.cache_file(FACTS_PASS, off.cache_key(FACTS_PASS, _canon_input({"block": b, "fragments": frags}),
                                                        digest)).exists():
            new_blocks += 1
    n_single_seeds = len(singles) or 1
    g_rate_s = sum(x["generate"] for x in by_lang.values()) / n_single_seeds
    v_rate_s = sum(x["verified"] for x in by_lang.values()) / n_single_seeds
    anchors_needed = math.ceil(round(ds["size"] * ds["hops"]["multi"]) * float(ds["target_margin"]) / y_multi) \
        if y_multi else math.inf
    g_rate_m = sum(x["generate"] for x in by_str.values()) / max(1, b_anchors)
    v_rate_m = sum(x["verified"] for x in by_str.values()) / max(1, b_anchors)
    vpc = float(ds["verify_views_per_call"])
    cand_s, cand_m = len(rest) * g_rate_s, (anchors_needed if math.isfinite(anchors_needed) else 0) * g_rate_m
    plants = (cand_s + cand_m) / int(ds["verify_batch"]) * float(ds["plant_share"]) * 2.0
    new = {"ds_facts": new_blocks, "ds_generate": math.ceil(len(rest) * 0.94 / int(ds["gen_batch"])),
           "ds_bridge": math.ceil(anchors_needed / int(ds["bridge_batch"])) if math.isfinite(anchors_needed) else None,
           "ds_verify": math.ceil((cand_s + 3 * cand_m + plants) / vpc)
           + math.ceil((cand_s + cand_m) * 0.8 / int(ds["verify_agree_per_call"])),
           "ds_quality": math.ceil((len(rest) * v_rate_s + (anchors_needed if math.isfinite(anchors_needed) else 0)
                                    * v_rate_m) / int(ds["quality_batch"]))}
    total = spent + sum(x for x in new.values() if x is not None) if new["ds_bridge"] is not None else math.inf
    final = _final_need(tcfg)
    got = Counter(f"{s.pattern}|{s.split}|{s.lang}" for s in full if s.pattern == "single")
    proj_single = {k: {"seeds": got[k], "need": n, "point": math.floor(got[k] * y_single.get(k.split("|")[2], 0))}
                   for k, n in sorted(final.items()) if k.startswith("single")}
    bx = _bridge_index(tcfg, recs, chunks)
    used = _used_chunks() | {c for r in rows("anchors.jsonl") for c in [r["anchor"]]}
    pool_split = Counter(bx._split(a) for a in sorted(bx.ok - used) if bx.candidates(a))
    need_m = {sp: m for sp, m in cell_counts(round(ds["size"] * ds["hops"]["multi"]), tcfg["split"]["ratios"]).items()}
    proj_multi = {sp: {"anchor_pool": pool_split[sp], "need": need_m[sp],
                       "point": math.floor(pool_split[sp] * y_multi)} for sp in sorted(need_m)}
    multi_final = sum(min(x["point"], x["need"]) for x in proj_multi.values())
    gates = []
    if total > int(tcfg["dataset_llm"]["max_calls_total"]):
        gates.append(f"projected total {total} > {tcfg['dataset_llm']['max_calls_total']}")
    if multi_final < round(ds["size"] * ds["hops"]["multi"]):
        gates.append(f"multi projected {multi_final} < {round(ds['size'] * ds['hops']['multi'])} final")
    short = sorted(k for k, x in proj_single.items() if x["point"] < x["need"])
    if short:
        gates.append(f"single cells projected below need: {len(short)}")
    gates += [f"judge {k} {x} < {ds['min_sensitivity_kind']}" for k, x in sorted(sens.items())
              if x < float(ds["min_sensitivity_kind"])]
    gates += [f"judge kind {k} below min n" for k in v.get("kinds_below_min_n", [])]
    rep = {"calls": calls, "calls_total_pilot3": sum(calls.values()), "spent_total": spent,
           "bridge": {"batch": br.get("batch"), "by_stratum": by_str, "rejected": br.get("rejected"),
                      "chosen_source": br.get("chosen_source"), "yield_good_per_anchor": round(y_multi, 4),
                      "wilson": [round(x, 4) for x in wilson(b_good, b_anchors)]},
           "single_uz": by_lang, "generate_rejected": (st.get("generate") or {}).get("rejected"),
           "verify": {k: v.get(k) for k in ("rejected", "sensitivity_by_kind", "sensitivity_wilson_by_kind", "planted",
                                            "kinds_below_min_n", "calls", "views", "agree")},
           "quality": {k: q.get(k) for k in ("verdicts", "rejected", "calls")},
           "projection": {"yield_single_by_lang": {k: round(x, 4) for k, x in y_single.items()},
                          "yield_single_ru_basis": "pilot2 ru verified x ds_quality GOOD on them (calibration run)",
                          "rest_single_seeds": len(rest), "new_fact_blocks": new_blocks,
                          "anchors_needed": anchors_needed, "full_new": new, "total": total,
                          "single_by_cell": proj_single, "multi_by_split": proj_multi, "multi_final": multi_final},
           "gates": gates, "reader_gate": "PENDING independent strict reader (GOOD >= .90, Wilson LB >= .75)"}
    config.path(f"{OUT}/report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    frag = lambda c: {"chunk_id": c, "act": index.act[chunks[c].parent.split("/", 1)[0]],
                      "where": index.format(chunks[c].parent), "text": chunks[c].text}
    _jsonl_write(f"{OUT}/accepted_pairs.jsonl",
                 [{"seed_id": sid(r), "pattern": r["seed"]["pattern"], "lang": r["seed"]["lang"],
                   "split": r["seed"]["split"], "question": r["question"], "answer": r["answer"],
                   "bridge_source": r["provenance"].get("bridge_source"), "walk": r["seed"]["walk"],
                   "facts": [{"value": f["value"], "quote": f["quote"], "chunk_id": f["chunk_id"]} for f in r["facts"]],
                   "fragments": [frag(c) for c in r["seed"]["chunks"]]} for r in good])
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 1 if gates else 0


# ---- M3-R23: Opus generator packets, ingest, reviewer packets ----------------------------------------------------------
def _opus_cfg(tcfg) -> dict:
    return tcfg["dataset"]["opus"]


def _instr(tcfg, which: str) -> dict:
    """The versioned generator / reader instruction file; its sha256 must equal the configured digest (an edit
    needs a new version + digest in config dataset.opus)."""
    from graphrag.dataset.opus import instructions
    o = _opus_cfg(tcfg)[which]
    d = instructions(config.path(o["file"]), o["version"], rel=o["file"])
    if d["digest"] != o["digest"]:
        raise SystemExit(f"{o['file']}: sha256 {d['digest']} != config dataset.opus.{which}.digest {o['digest']} -- "
                         f"bump version and digest")
    return d


def _cached_facts(tcfg, recs, chunks, cids) -> dict:
    """chunk_id -> [Fact] for the fragments whose ds_facts block the cache holds (offline, 0 calls; an uncached
    block is simply absent -- the Opus generator then quotes the text itself)."""
    from graphrag.dataset.facts import extract_block
    from graphrag.dataset.seeds import block_of
    from graphrag.dataset.sources import SourceIndex
    from graphrag.kg.llm import KgCacheMiss, KgLLM
    ds = tcfg["dataset"]
    blk = block_of(recs, int(ds["facts_block"]))
    members: dict = {}
    for r in recs:
        for c in sorted(r.chunks, key=lambda c: c.start):
            members.setdefault(blk[c.chunk_id], []).append(c.chunk_id)
    off = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False)
    index = SourceIndex(recs)
    out = {}
    for b in sorted({blk[c] for c in cids}):
        try:
            fs, _ = extract_block(off, b, members[b], chunks, index, int(ds["max_facts_per_chunk"]))
        except KgCacheMiss:
            continue
        for cid in members[b]:
            out[cid] = [f for f in fs if f.chunk_id == cid]
    if off.total_api_calls():
        raise RuntimeError("packets must not call the API")
    return out


def _packet_files():
    d = config.path(f"{OUT}/packets")
    return sorted(p for p in d.glob("*.json") if p.name != "index.json") if d.exists() else []


def _write_packets(tcfg, groups, instr) -> list[str]:
    from graphrag.dataset.opus import packet_doc
    ids = []
    for pid, entries in groups:
        doc = packet_doc(pid, entries, instr, f"{OUT}/gen/{pid}.jsonl")
        doc["output"]["relative_to"] = "advanced-rag-v2/"
        f = config.path(f"{OUT}/packets/{pid}.json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        ids.append(pid)
    return ids


def _write_index(tcfg) -> dict:
    """{OUT}/packets/index.json: every packet (pattern, split, seeds, languages, instruction digest) and the dispatch
    order -- pilot packets first, then the rest interleaved over pattern x split (graphrag.dataset.opus.wave_order)."""
    from graphrag.dataset.opus import wave_order
    meta = {}
    for f in _packet_files():
        d = json.loads(f.read_text(encoding="utf-8"))
        meta[d["packet_id"]] = {"pattern": d["pattern"], "split": d["split"], "n_seeds": d["n_seeds"],
                                "langs": d["langs"], "instructions_digest": d["instructions"]["digest"],
                                "file": f"{OUT}/packets/{f.name}", "output": d["output"]["path"]}
    pilot = sorted(p for p in meta if p.startswith("pilot-"))
    rest = wave_order([p for p in meta if p not in pilot], lambda p: (meta[p]["pattern"], meta[p]["split"]))
    idx = {"packets": meta, "order": pilot + rest}
    config.path(f"{OUT}/packets/index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1, sort_keys=True)
                                                        + "\n", encoding="utf-8")
    return idx


def _dup(tcfg, recs):
    from graphrag.dataset.seeds import dup_content
    return dup_content(recs, int(_opus_cfg(tcfg).get("dup_min_chars", 40)))


_GRAPH_META = {}


def _graph_meta(recs, index):
    """graph-derived fragment meta (discussed_act / amended_act / amended_unit; opus.graph_meta), cached per run."""
    if "fn" not in _GRAPH_META:
        from graphrag.dataset.opus import graph_meta
        from graphrag.env.data import load_env_data
        from graphrag.kg.artifact import read_graph
        _, edges = read_graph(config.path(f"data/graph/{load_env_data().version}"))
        _GRAPH_META["fn"] = graph_meta(recs, edges, index)
    return _GRAPH_META["fn"]


def _frag_meta(recs, index):
    """Packet fragment meta: plenary session / speaker + graph meta (the discussed law, the amended act)."""
    from graphrag.dataset.opus import packet_meta, session_info
    return packet_meta(session_info(recs), _graph_meta(recs, index))


def _opus_context(tcfg):
    from graphrag.dataset.sources import SourceIndex
    recs, chunks = _chunks()
    index = SourceIndex(recs)
    bx = _bridge_index(tcfg, recs, chunks)
    return recs, chunks, index, bx, _amended(bx, index)


def cmd_pilot_opus(tcfg) -> int:
    """M3-R23 pilot preset: one single packet (`pilot.single` train seeds of data/dataset/seeds.jsonl, languages
    balanced, sections in turn, <= 1 per document, cached fact blocks first, no fragment of an earlier pilot) and one
    multi packet (`pilot.multi` train anchors, bridge sources balanced -- bridge.sample_anchors, languages round-robin).
    Writes {OUT}/packets/pilot-*.json + index, {PILOT_DIR}/seeds.jsonl (single seeds), anchors.jsonl, packets.json,
    ledger_before.json (once). 0 API calls."""
    import hashlib
    from collections import Counter
    from graphrag.dataset.bridge import sample_anchors
    from graphrag.dataset.opus import make_packets, multi_entry, single_entry
    from graphrag.dataset.seeds import Seed, block_of, cell_counts
    from graphrag.kg.llm import KgLLM
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    sp, n_single, n_multi = o["pilot"]["split"], int(o["pilot"]["single"]), int(o["pilot"]["multi"])
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    used = _used_chunks()
    dup = _dup(tcfg, recs)
    reuse = config.path(f"{PILOT_DIR}/seeds.jsonl").exists() and config.path(f"{PILOT_DIR}/anchors.jsonl").exists()
    full = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    pool = [s for s in full if s.pattern == "single" and s.split == sp and not set(s.chunks) & used]
    if reuse:          # M3-R25: rebuild the packets over the committed pilot selection (same seeds and anchors)
        pool = [Seed.from_json(r) for r in _jsonl_read(f"{PILOT_DIR}/seeds.jsonl")]
    facts = _cached_facts(tcfg, recs, chunks, [c for s in pool for c in s.chunks])
    h = lambda k: hashlib.sha256(f"{ds['seed']}:opus-pilot|{k}".encode()).hexdigest()
    langs = list(ds["langs"])
    want = cell_counts(n_single, {l: 1.0 for l in langs})
    out, docs = [], Counter()
    for lang in ([] if reuse else langs):
        mine = sorted((s for s in pool if s.lang == lang), key=lambda s: (s.chunks[0] not in facts, h(s.seed_id)))
        secs = sorted({s.section for s in mine})
        n, i = 0, 0
        while n < want[lang] and secs and i < 50 * len(secs):
            sec = secs[i % len(secs)]
            i += 1
            s = next((s for s in mine if s.section == sec and docs[s.docs[0]] < 1 and s not in out), None)
            if s is None:
                continue
            out.append(s)
            docs[s.docs[0]] += 1
            n += 1
    singles = sorted(pool if reuse else out, key=lambda s: s.seed_id)
    used_now = used | {c for s in singles for c in s.chunks}
    anchors = sorted(a for a in bx.ok if bx._split(a) == sp)
    if reuse:
        picks = [{**a, "candidates": bx.candidates(a["anchor"])} for a in _jsonl_read(f"{PILOT_DIR}/anchors.jsonl")]
    else:
        picks = sample_anchors(bx, anchors, n_multi, langs, ds["seed"], exclude=used_now)
    mfacts = _cached_facts(tcfg, recs, chunks, [c for p in picks for c in [p["anchor"]] +
                                                [x for v in p["candidates"].values() for x, _ in v]])
    se = [single_entry(s, chunks, index, facts, amended) for s in singles]
    me = [multi_entry(p, chunks, index, mfacts, ds["seed"], amended, dup) for p in picks]
    groups = make_packets(se, max(1, len(se)), "pilot-", ds["seed"]) + make_packets(me, max(1, len(me)), "pilot-",
                                                                                   ds["seed"])
    ids = _write_packets(tcfg, groups, instr)
    _write_index(tcfg)
    _jsonl_write(f"{PILOT_DIR}/seeds.jsonl", [s.to_json() for s in singles])
    _jsonl_write(f"{PILOT_DIR}/anchors.jsonl", [{"anchor": p["anchor"], "stratum": p["stratum"], "lang": p["lang"],
                                                 "split": p["split"]} for p in picks])
    config.path(f"{PILOT_DIR}/packets.json").write_text(json.dumps({"packets": ids}, indent=1) + "\n", encoding="utf-8")
    before = config.path(f"{PILOT_DIR}/ledger_before.json")
    if not before.exists():
        led = KgLLM(cfg={"llm": tcfg["dataset_llm"]}, cache_root=tcfg["dataset_llm"]["cache_dir"], allow_api=False).ledger()
        before.write_text(json.dumps(led, sort_keys=True) + "\n", encoding="utf-8")
    rep = {"packets": ids, "instructions": {k: instr[k] for k in ("path", "version", "digest")},
           "reused_selection": reuse, "single_with_duplicate_twin": sorted(s.seed_id for s in singles if s.chunks[0] in dup),
           "anchors_with_duplicate_twin": sorted(p["anchor"] for p in picks if p["anchor"] in dup),
           "single": len(singles), "single_by_lang": dict(Counter(s.lang for s in singles)),
           "single_by_section": dict(sorted(Counter(s.section for s in singles).items())),
           "single_facts_cached": sum(s.chunks[0] in facts for s in singles),
           "multi": len(picks), "multi_by_stratum": dict(Counter(p["stratum"] for p in picks)),
           "multi_by_lang": dict(Counter(p["lang"] for p in picks)),
           "multi_neighbors": sum(len(e["neighbors"]) for e in me)}
    _stats_update("pilot_opus", rep)
    print(json.dumps(rep, ensure_ascii=False))
    short = len(singles) < n_single or len(picks) < n_multi
    if short:
        print(f"STOP (П4): pilot short -- single {len(singles)}/{n_single}, multi {len(picks)}/{n_multi}")
    return 1 if short else 0


def _pilot_packets() -> list[str]:
    """The dir's pilot packets: {PILOT_DIR}/packets.json, else every {OUT}/packets/pilot-*.json (a copied packet dir,
    e.g. the Fable head-to-head)."""
    f = config.path(f"{PILOT_DIR}/packets.json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["packets"]
    d = config.path(f"{OUT}/packets")
    return sorted(p.stem for p in d.glob("pilot-*.json")) if d.exists() else []


def cmd_packets(tcfg) -> int:
    """M3-R23 full-run generator packets (pilot packets kept, their seeds / anchors excluded): every single seed of
    data/dataset/seeds.jsonl in split-pure packets of `packet_size`, and per split ceil(final multi need x
    multi_overshoot) ds_bridge anchors (sources balanced; languages by the D7 shares) in packets of
    `multi_packet_size`. Deterministic (sha256 orders). 0 API calls. Exit 1 if a split has fewer anchors than its
    final multi need (fewer than the overshoot is only reported)."""
    import math
    from collections import Counter
    from graphrag.dataset.bridge import sample_anchors
    from graphrag.dataset.opus import make_packets, multi_entry, single_entry
    from graphrag.dataset.seeds import Seed, allocate, cell_counts
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    pilot_seeds, pilot_anchors = set(), Counter()
    used_anchor = set()
    for pid in _pilot_packets():
        d = json.loads(config.path(f"{OUT}/packets/{pid}.json").read_text(encoding="utf-8"))
        for e in d["seeds"]:
            pilot_seeds.add(e["seed_id"])
            if "anchor" in e:
                used_anchor.add(e["anchor"]["chunk_id"])
                pilot_anchors[e["cell"]["split"]] += 1
    from graphrag.dataset.seeds import content_signature, cross_split_twin
    dup = _dup(tcfg, recs)
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    taken, pilot_keys = set(), set()
    for pid in _pilot_packets():               # the pilot's contents are taken first
        d = json.loads(config.path(f"{OUT}/packets/{pid}.json").read_text(encoding="utf-8"))
        taken |= {content_signature([e["fragments"][0]["chunk_id"]], dup) for e in d["seeds"] if "fragments" in e}
        pilot_keys |= {dup[e["anchor"]["chunk_id"]]["key"] for e in d["seeds"] if "anchor" in e
                       and e["anchor"]["chunk_id"] in dup}
    full = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    singles, dup_drop = [], Counter()
    for s in sorted((s for s in full if s.pattern == "single" and s.seed_id not in pilot_seeds), key=lambda s: s.seed_id):
        sig = content_signature(s.chunks, dup)            # M3-R25: one seed per duplicate content, no cross-split twin
        if cross_split_twin(s.chunks, dup, split_of):
            dup_drop["single|cross_split"] += 1
        elif sig in taken:
            dup_drop["single|same_content"] += 1
        else:
            taken.add(sig)
            singles.append(s)
    need = cell_counts(round(ds["size"] * ds["hops"]["multi"]), tcfg["split"]["ratios"])
    picks, short = [], {}
    for sp in sorted(need):
        n = max(0, math.ceil(need[sp] * float(o["multi_overshoot"])) - pilot_anchors[sp])
        # an anchor: no cross-split twin, one anchor per duplicate text (the smallest chunk id, unless the pilot has it)
        rep_of = {}
        for x in sorted(a for a in bx.ok if a in dup):
            rep_of.setdefault(dup[x]["key"], x)
        bad = {a for a in bx.ok if a in dup and (cross_split_twin([a], dup, split_of) or rep_of[dup[a]["key"]] != a
                                                  or dup[a]["key"] in pilot_keys)}
        dup_drop[f"anchor|{sp}"] += len({a for a in bad if bx._split(a) == sp})
        got = sample_anchors(bx, sorted(a for a in bx.ok if bx._split(a) == sp), n, list(ds["langs"]), ds["seed"],
                             exclude=used_anchor | bad)
        for p, lang in zip(got, allocate(len(got), ds["langs"])):
            p["lang"] = lang
        picks += got
        if len(got) < n:                     # below the overshoot (a yield assumption, M3-R13): reported
            short[sp] = {"anchors": len(got) + pilot_anchors[sp], "want": n + pilot_anchors[sp],
                         "final_need": need[sp]}
    facts = _cached_facts(tcfg, recs, chunks, [c for s in singles for c in s.chunks] +
                          [c for p in picks for c in [p["anchor"]] + [x for v in p["candidates"].values() for x, _ in v]])
    se = [single_entry(s, chunks, index, facts, amended) for s in singles]
    me = [multi_entry(p, chunks, index, facts, ds["seed"], amended, dup) for p in picks]
    groups = make_packets(se, int(o["packet_size"]), "", ds["seed"]) + \
        make_packets(me, int(o["multi_packet_size"]), "", ds["seed"])
    ids = _write_packets(tcfg, groups, instr)
    idx = _write_index(tcfg)
    rep = {"packets": len(ids), "pilot_packets": _pilot_packets(), "single_seeds": len(se), "multi_anchors": len(me),
           "by_cell": dict(sorted(Counter(f"{e['cell']['split']}|{e['cell']['pattern']}|{e['cell']['lang']}"
                                          for e in se + me).items())),
           "multi_by_stratum": dict(sorted(Counter(f"{p['split']}|{p['stratum']}" for p in picks).items())),
           "facts_cached_seeds": sum(e["facts_cached"] is True for e in se), "anchor_shortfall": short,
           "duplicate_content_excluded": dict(sorted(dup_drop.items())),
           "instructions": {k: instr[k] for k in ("path", "version", "digest")}, "order_head": idx["order"][:10]}
    _stats_update("packets", rep)
    print(json.dumps(rep, ensure_ascii=False))
    below = {sp: x for sp, x in short.items() if x["anchors"] < x["final_need"]}
    if short:
        print(f"NOTE: multi anchors below the overshoot {short} (anchors per final pair = multi_overshoot)")
    if below:
        print(f"STOP (П4): multi anchors below the final need {below}; report to the user")
        return 1
    return 0


def cmd_ingest_gen(tcfg) -> int:
    """M3-R23: every packet's generator JSONL ({OUT}/gen/<packet>.jsonl, committed) through all code checks ->
    {OUT}/candidates_raw.jsonl (input of verify), {OUT}/gen_rejections.jsonl (per reason / cell, skips included),
    {PILOT_DIR}/bridge_seeds.jsonl (multi seeds of the pilot packets: pilot-first in verify). Pure code, 0 calls,
    byte-identical on replay. A packet without output is pending."""
    from collections import Counter
    from graphrag.dataset.opus import ingest_generated, summarize
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    recs, chunks = _chunks()
    sections = {r.doc_id: r.meta.section for r in recs}
    cands, rej, pending, seeds_cell, digests = [], [], [], Counter(), Counter()
    pilot = set(_pilot_packets())
    pilot_multi = []
    for f in _packet_files():
        pk = json.loads(f.read_text(encoding="utf-8"))
        gp = f"{OUT}/gen/{pk['packet_id']}.jsonl"
        g = config.path(gp)
        text = g.read_text(encoding="utf-8") if g.exists() else None
        meta = config.path(f"{OUT}/gen/{pk['packet_id']}.meta.json")
        under = None
        if meta.exists():                      # M3-R25: an output written under earlier instructions keeps them
            under = json.loads(meta.read_text(encoding="utf-8"))["instructions"]
            hist = {int(k): v for k, v in (o.get("generator_history") or {}).items()}
            if hist.get(int(under["version"])) != under["digest"]:
                print(f"STOP: {meta}: instructions v{under['version']} {under['digest'][:12]} not in generator_history")
                return 1
        cs, rj, st = ingest_generated(pk, text, chunks, ds, sections, gp, ARGS.agent_model or o["agent_model"], under)
        if st["pending"]:
            pending.append(pk["packet_id"])
            continue
        digests[(under or pk["instructions"])["digest"]] += 1
        for e in pk["seeds"]:
            seeds_cell[f"{e['cell']['split']}|{e['cell']['pattern']}|{e['cell']['lang']}"] += 1
        cands += cs
        rej += rj
        if pk["packet_id"] in pilot:
            pilot_multi += [c.seed for c in cs if c.seed.pattern == "multi"]
    twice = [k for k, n in Counter(c.seed.seed_id for c in cands).items() if n > 1]
    if twice:
        print(f"STOP: one seed generated in two packets {twice[:5]}")
        return 1
    from graphrag.dataset.opus import dedupe_content
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    cands, drj = dedupe_content(cands, _dup(tcfg, recs), split_of)          # M3-R25: one pair per duplicate content
    rej += drj
    cands = sorted(cands, key=lambda c: c.seed.seed_id)
    _jsonl_write(f"{OUT}/candidates_raw.jsonl", _cand_rows(cands))
    _jsonl_write(f"{OUT}/gen_rejections.jsonl", sorted(rej, key=lambda r: (r["packet_id"], r["line"] or 0,
                                                                          r["seed_id"] or "", r["reason"])))
    if pilot:
        _jsonl_write(f"{PILOT_DIR}/bridge_seeds.jsonl", [s.to_json() for s in sorted(pilot_multi,
                                                                                     key=lambda s: s.seed_id)])
    st = {"packets": len(_packet_files()), "pending": pending, "seeds": sum(seeds_cell.values()),
          "accepted": len(cands), "skipped": sum(r["reason"] == "skip" for r in rej),
          "instructions_digests": dict(digests), **summarize(rej, cands, dict(seeds_cell))}
    _stats_update("ingest_gen", st)
    print(json.dumps(st, ensure_ascii=False))
    return 0


def _review_files(root: str = "review"):
    d = config.path(f"{OUT}/{root}/packets")
    return sorted(d.glob("*.json")) if d.exists() else []


def cmd_review_packets(tcfg) -> int:
    """M3-R23 / M3-R31: reviewer packets for the verified pairs ({OUT}/verified.jsonl) -- pair + fragments + record
    META (doc type, act title, date, session dates, speaker) + READER RUBRIC (config/dataset_reader_rubric.md); never
    generator notes / facts / verdicts, never a model name. Every pair gets TWO packets, one per independent reader:
    {OUT}/review/ (reader A) and {OUT}/review2/ (reader B, prefix second-); each root only for pairs not yet in one of
    its packets. Pilot pairs first. --sample-frac f (legacy M3-R29): reader B over a deterministic f-sample only."""
    from graphrag.dataset.opus import ROLE_READER, ROLE_SECOND, make_review_packets, reader_meta, second_sample
    from graphrag.dataset.sources import SourceIndex
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    rubric = _instr(tcfg, "reader")
    recs, chunks = _chunks()
    frag_of, _ = _frag_of(recs, chunks)
    idx_ = SourceIndex(recs)
    meta_of = reader_meta(recs, idx_, graph=_graph_meta(recs, idx_))
    cands = _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))
    roots = [("review2", ROLE_SECOND, "second-", second_sample(cands, ARGS.sample_frac, ds["seed"]))] \
        if ARGS.sample_frac else [("review", ROLE_READER, "", cands), ("review2", ROLE_SECOND, "second-", cands)]
    rep = {"verified": len(cands), "rubric": {k: rubric[k] for k in ("path", "version", "digest")}}
    for root, role, prefix, cs in roots:
        existing = {}
        for f in _review_files(root):
            d = json.loads(f.read_text(encoding="utf-8"))
            existing[d["packet_id"]] = [it["pair_id"] for it in d["items"]]
        new = make_review_packets(cs, frag_of, rubric, int(o["review_packet_size"]), ds["seed"], existing,
                                  _pilot_ids(), f"{OUT}/{root}/verdicts", role=role, prefix=prefix, meta_of=meta_of)
        for p in new:
            p["output"]["relative_to"] = "advanced-rag-v2/"
            f = config.path(f"{OUT}/{root}/packets/{p['packet_id']}.json")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(p, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        rep[root] = {"existing_packets": len(existing), "new_packets": [p["packet_id"] for p in new],
                     "new_pairs": sum(p["n_items"] for p in new)}
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def _read_root(root: str, cands, digests):
    from graphrag.dataset.opus import ingest_verdicts
    packets, texts = [], {}
    for f in _review_files(root):
        d = json.loads(f.read_text(encoding="utf-8"))
        packets.append(d)
        v = config.path(f"{OUT}/{root}/verdicts/{d['packet_id']}.jsonl")
        texts[d["packet_id"]] = v.read_text(encoding="utf-8") if v.exists() else None
    import copy
    return ingest_verdicts(packets, texts, copy.deepcopy(cands), digests)


def cmd_ingest_review(tcfg) -> int:
    """M3-R31 / M3-R32: both readers' verdicts ({OUT}/review = reader A, {OUT}/review2 = reader B) -> dual-reader
    consensus: {OUT}/reviewed.jsonl = pairs BOTH read GOOD (input of finalize); {OUT}/review_rejections.jsonl
    (invalid rows of either reader + no-consensus pairs); disagreements per cell / defect, and consensus per generator
    wave and per generator-instruction version in stats "review". A pair missing a verdict is pending; an original
    pair with a repaired successor (...-r1) is superseded. No STOP (the M3-R22 (4) pilot gate is retired)."""
    from collections import Counter, defaultdict
    from graphrag.dataset.opus import REPAIR_SUFFIX, consensus, reader_id, wave_of
    o = _opus_cfg(tcfg)
    _instr(tcfg, "reader")
    digests = set((o.get("reader_history") or {}).values()) | {o["reader"]["digest"]}
    cands = _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))
    _, rej_a, st_a = _read_root("review", cands, digests)
    _, rej_b, st_b = _read_root("review2", cands, digests)
    va, vb = st_a.pop("verdict_of"), st_b.pop("verdict_of")
    repaired = {c.seed.seed_id[:-len(REPAIR_SUFFIX)] for c in cands if c.seed.seed_id.endswith(REPAIR_SUFFIX)}
    good, rej_c, st = consensus(cands, va, vb)
    good = [c for c in good if c.seed.seed_id not in repaired]
    _jsonl_write(f"{OUT}/reviewed.jsonl", _cand_rows(good))
    rows = [dict(r, reader="A") for r in rej_a if r["reason"] not in ("bad", "borderline")] + \
        [dict(r, reader="B") for r in rej_b if r["reason"] not in ("bad", "borderline")] + rej_c
    _jsonl_write(f"{OUT}/review_rejections.jsonl", sorted(rows, key=lambda r: (r.get("pair_id") or "", r["reason"],
                                                                               r.get("reader", ""))))
    by_wave, by_ver = defaultdict(Counter), defaultdict(Counter)
    for c in cands:
        a, b = va.get(reader_id(c.seed.seed_id)), vb.get(reader_id(c.seed.seed_id))
        w = wave_of(c)
        ver = str(((c.provenance.get("opus") or {}).get("instructions") or {}).get("version"))
        for d, k in ((by_wave, f"{w}|{c.seed.pattern}"), (by_ver, f"v{ver}|{c.seed.pattern}")):
            d[k]["verified"] += 1
            if a is None or b is None:
                d[k]["pending"] += 1
                continue
            d[k]["read_by_both"] += 1
            d[k]["consensus_good"] += a["verdict"] == "GOOD" and b["verdict"] == "GOOD"
            d[k][f"{a['verdict']}|{b['verdict']}"] += 1
    rate = lambda d: {k: {**dict(v), "good_rate": round(v["consensus_good"] / v["read_by_both"], 4)
                          if v["read_by_both"] else None} for k, v in sorted(d.items())}
    out = {"consensus": {k: v for k, v in st.items() if k != "pending_pairs"}, "reader_A": st_a, "reader_B": st_b,
           "superseded_by_repair": len(repaired), "by_wave": rate(by_wave), "by_instructions": rate(by_ver)}
    _stats_update("review", out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


def _gen_row(c) -> dict:
    o = c.provenance.get("opus") or {}
    lines = config.path(o["gen_file"]).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[int(o["line"]) - 1])


def cmd_repair_packets(tcfg) -> int:
    """M3-R32: repair packets ({OUT}/repair/packets/repair-NNN.json, `repair_packet_size` items) for the verified
    pairs BOTH readers read with at least one BORDERLINE and no BAD, not consensus GOOD, not attempted before: the
    pair as generated, both readers' reasons, the seed entry, config/dataset_repair.md + the current generator
    instructions. Outputs go to {OUT}/repair/gen/<id>.jsonl. 0 API calls."""
    from collections import Counter
    from graphrag.dataset.opus import make_repair_packets, repair_eligible
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    o_rep = o["repair"]
    from graphrag.dataset.opus import instructions
    rins = instructions(config.path(o_rep["file"]), o_rep["version"], rel=o_rep["file"])
    if rins["digest"] != o_rep["digest"]:
        raise SystemExit(f"{o_rep['file']}: digest differs from config dataset.opus.repair -- bump version + digest")
    gins = _instr(tcfg, "generator")
    o2 = _opus_cfg(tcfg)
    digests = set((o2.get("reader_history") or {}).values()) | {o2["reader"]["digest"]}
    cands = _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))
    _, _, st_a = _read_root("review", cands, digests)
    _, _, st_b = _read_root("review2", cands, digests)
    existing, attempted = {}, set()
    for f in sorted(config.path(f"{OUT}/repair/packets").glob("*.json")) if config.path(f"{OUT}/repair/packets").exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        existing[d["packet_id"]] = [it["original_seed_id"] for it in d["items"]]
        attempted |= set(existing[d["packet_id"]])
    items = repair_eligible(cands, st_a["verdict_of"], st_b["verdict_of"], attempted)
    gen_packets = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in _packet_files()}
    from graphrag.dataset.sources import SourceIndex
    recs_, _ = _chunks()
    new = make_repair_packets(items, gen_packets, _gen_row, rins, gins, int(o["repair_packet_size"]), ds["seed"],
                              existing, f"{OUT}/repair/gen", meta_fn=_frag_meta(recs_, SourceIndex(recs_)))
    for p in new:
        p["output"]["relative_to"] = "advanced-rag-v2/"
        f = config.path(f"{OUT}/repair/packets/{p['packet_id']}.json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(p, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    rep = {"eligible": len(items), "packets": [p["packet_id"] for p in new],
           "by_cell": dict(sorted(Counter(f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}" for c, _, _ in items).items())),
           "by_verdicts": dict(Counter(f"{a['verdict']}|{b['verdict']}" for _, a, b in items)),
           "by_criterion": dict(sorted(Counter(k for _, a, b in items for k in set(a["criteria"]) | set(b["criteria"])
                                               ).items())),
           "repair_instructions": {k: rins[k] for k in ("path", "version", "digest")}}
    _stats_update("repair_packets", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_ingest_repair(tcfg) -> int:
    """M3-R32: every repair packet's output ({OUT}/repair/gen/<id>.jsonl) -> NEW candidates (seed <orig>-r1, pair
    syn-...-r1, provenance repaired_from + repair-instruction digest) through ALL code checks -> {OUT}/repaired_raw.jsonl
    (read by verify with the generator candidates; then fresh dual blind review) and {OUT}/repair_rejections.jsonl.
    Pure code, 0 calls."""
    from collections import Counter
    from graphrag.dataset.opus import ingest_repairs
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    recs, chunks = _chunks()
    sections = {r.doc_id: r.meta.section for r in recs}
    originals = {c.seed.seed_id: c for c in _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))}
    from graphrag.dataset.sources import SourceIndex
    meta_fn = _frag_meta(recs, SourceIndex(recs))
    out, rej, pending = [], [], []
    d = config.path(f"{OUT}/repair/packets")
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        pk = json.loads(f.read_text(encoding="utf-8"))
        gp = f"{OUT}/repair/gen/{pk['packet_id']}.jsonl"
        g = config.path(gp)
        if not g.exists():
            pending.append(pk["packet_id"])
            continue
        cs, rj, _ = ingest_repairs(pk, g.read_text(encoding="utf-8"), chunks, ds, sections, gp,
                                   ARGS.agent_model or o["agent_model"], originals, meta_fn=meta_fn)
        out += cs
        rej += rj
    out = sorted(out, key=lambda c: c.seed.seed_id)
    _jsonl_write(f"{OUT}/repaired_raw.jsonl", _cand_rows(out))
    _jsonl_write(f"{OUT}/repair_rejections.jsonl", sorted(rej, key=lambda r: (r["packet_id"], r["line"] or 0,
                                                                             r["seed_id"] or "", r["reason"])))
    st = {"pending": pending, "repaired": len(out), "rejected": dict(sorted(Counter(r["reason"] for r in rej).items()))}
    _stats_update("ingest_repair", st)
    print(json.dumps(st, ensure_ascii=False))
    return 0


def cmd_calibrate_near_copy(tcfg) -> int:
    """M3-R29: blanked-sentence near-copy threshold = P90 of the Q-real questions (Cyrillic + Latin forms) against
    their gold fragments' sentences (generate.near_copy_ratio), plus an audit on the strict-read pilot pairs
    (data/dataset/*/review: verdict x caught). Writes {OUT}/near_copy_calibration.json; exit 1 if config
    dataset.near_copy differs from the measurement."""
    from collections import Counter
    from graphrag.dataset.generate import calibrate_near_copy, near_copy_ratio
    from graphrag.dataset.sources import SourceIndex
    recs, chunks = _chunks()
    index = SourceIndex(recs)
    qs = []
    for r in _jsonl_read(tcfg["gold"]["file"]):
        ids = list(dict.fromkeys(r["core"] + r["support"]))
        frags = [{"act": index.act[chunks[c].parent.split("/", 1)[0]], "where": index.format(chunks[c].parent),
                  "text": chunks[c].text} for c in ids]
        for q in (r["question_cyr"], r.get("question_lat")):
            if q:
                qs.append(("uz", q, frags))
    fams = calibrate_near_copy(qs, 0.90)
    thr = fams["uz"]["max_ratio"]
    audit = {}
    for d in sorted(config.path("data/dataset").glob("*/review/packets/*.json")):
        pk = json.loads(d.read_text(encoding="utf-8"))
        vf = d.parent.parent / "verdicts" / (d.stem + ".jsonl")
        if not vf.exists():
            continue
        ver = {x["pair_id"]: x for x in _jsonl_read(vf)}
        rows = []
        for it in pk["items"]:
            v = ver.get(it["pair_id"])
            if v is None:
                continue
            r = near_copy_ratio(it["question"], it["fragments"])
            rows.append({"pair_id": it["pair_id"], "verdict": v["verdict"], "ratio": r, "caught": r >= thr,
                         "c5": any((x or {}).get("criterion") == "c5" for x in v.get("reasons", []))})
        audit[f"{d.parent.parent.parent.name}/{d.stem}"] = {
            "n": len(rows), "caught_by_verdict": dict(Counter(f"{x['verdict']}|{x['caught']}" for x in rows)),
            "c5_caught": sum(x["c5"] and x["caught"] for x in rows), "c5": sum(x["c5"] for x in rows),
            "good_wrongly_rejected": sorted(x["pair_id"] for x in rows if x["verdict"] == "GOOD" and x["caught"]),
            "rows": rows}
    out = {"source": tcfg["gold"]["file"], "quantile": 0.90, "families": fams, "threshold": thr, "audit": audit}
    config.path(f"{OUT}/near_copy_calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                            sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "audit"} | {"audit": {k: {kk: vv for kk, vv in v.items()
                                                                                   if kk != "rows"}
                                                                               for k, v in audit.items()}},
                     ensure_ascii=False))
    have = tcfg["dataset"].get("near_copy") or {}
    if (have.get("max_ratio"), have.get("n")) != (thr, fams["uz"]["n"]):
        print("config dataset.near_copy differs from the measurement")
        return 1
    return 0


def _dir_used(dirs=("data/dataset/opus", "data/dataset/fable")) -> tuple[set, set]:
    """Seed ids and fragment ids already in any generator packet of the Opus / Fable dirs (M3-R29: fresh seeds)."""
    ids, cids = set(), set()
    for d in dirs:
        for f in sorted(config.path(d).glob("packets/*.json")):
            if f.name == "index.json":
                continue
            for e in json.loads(f.read_text(encoding="utf-8"))["seeds"]:
                ids.add(e["seed_id"])
                for fr in e.get("fragments", []) + ([e["anchor"]] if "anchor" in e else []):
                    cids.add(fr["chunk_id"])
    return ids, cids


def cmd_mini_pilot(tcfg) -> int:
    """M3-R29 mini-pilot: `mini_pilot.single` fresh train singles of data/dataset/seeds.jsonl (languages balanced,
    sections in turn, <= 1 per document, cached fact blocks first; no seed / fragment of any earlier pilot or packet;
    one seed per duplicate content, no cross-split twin) -> {OUT}/packets/mini-single-<split>-001.json with the
    current generator instructions. 0 API calls."""
    import hashlib
    from collections import Counter
    from graphrag.dataset.opus import make_packets, session_info, single_entry
    from graphrag.dataset.seeds import Seed, cell_counts, content_signature, cross_split_twin
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    mp = o["mini_pilot"]
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    dup = _dup(tcfg, recs)
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    used_ids, used_c = _dir_used()
    used_c |= _used_chunks()
    taken = {content_signature([c], dup) for c in used_c}
    pool = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    pool = [s for s in pool if s.pattern == "single" and s.split == mp["split"] and s.seed_id not in used_ids
            and not set(s.chunks) & used_c and not cross_split_twin(s.chunks, dup, split_of)
            and content_signature(s.chunks, dup) not in taken]
    facts = _cached_facts(tcfg, recs, chunks, [c for s in pool for c in s.chunks])
    h = lambda k: hashlib.sha256(f"{ds['seed']}:mini-pilot|{k}".encode()).hexdigest()
    langs = list(ds["langs"])
    want = cell_counts(int(mp["single"]), {l: 1.0 for l in langs})
    out, docs = [], Counter()
    for lang in langs:
        mine = sorted((s for s in pool if s.lang == lang), key=lambda s: (s.chunks[0] not in facts, h(s.seed_id)))
        secs = sorted({s.section for s in mine})
        n, i = 0, 0
        while n < want[lang] and secs and i < 50 * len(secs):
            sec = secs[i % len(secs)]
            i += 1
            s = next((s for s in mine if s.section == sec and docs[s.docs[0]] < 1 and s not in out), None)
            if s is None:
                continue
            out.append(s)
            docs[s.docs[0]] += 1
            n += 1
    singles = sorted(out, key=lambda s: s.seed_id)
    sess = _frag_meta(recs, index)
    se = [single_entry(s, chunks, index, facts, amended, sess) for s in singles]
    ids = _write_packets(tcfg, make_packets(se, max(1, len(se)), "mini-", ds["seed"]), instr)
    _write_index(tcfg)
    rep = {"packets": ids, "instructions": {k: instr[k] for k in ("path", "version", "digest")},
           "single": len(singles), "by_lang": dict(Counter(s.lang for s in singles)),
           "by_section": dict(sorted(Counter(s.section for s in singles).items())),
           "facts_cached": sum(s.chunks[0] in facts for s in singles),
           "stenogram_with_session": sum(1 for e in se if e["fragments"][0].get("session"))}
    _stats_update("mini_pilot", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if len(singles) == int(mp["single"]) else 1


def cmd_multi_packets(tcfg) -> int:
    """M3-R29: multi packets over EVERY usable anchor of every split -- an eligible content fragment with a
    `multi_all.sources` neighbour (refers_to / amends / concept), no cross-split twin, one anchor per duplicate text,
    not already an anchor of a packet in the Opus / Fable dirs; <= multi_all.per_doc per document (direct links
    first, then sha256 order); languages by the D7 shares; split-pure packets of multi_all.packet_size -> {OUT}/packets/
    multi-<split>-NNN.json with the current generator instructions. 0 API calls."""
    from collections import Counter
    from graphrag.dataset.bridge import SOURCES
    from graphrag.dataset.opus import make_packets, multi_entry, session_info
    from graphrag.dataset.seeds import allocate, cross_split_twin
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    ma = o["multi_all"]
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    dup = _dup(tcfg, recs)
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    _, used_c = _dir_used()
    used_keys = {dup[c]["key"] for c in used_c if c in dup}
    want_src = [s for s in SOURCES if s in ma["sources"]]
    rank = {s: i for i, s in enumerate(want_src)}
    cand, drop = [], Counter()
    for a in sorted(bx.ok):
        cs = {s: v for s, v in bx.candidates(a).items() if s in rank}
        if not cs:
            continue
        if a in used_c:
            drop["already_in_a_packet"] += 1
            continue
        if a in dup and (cross_split_twin([a], dup, split_of) or dup[a]["key"] in used_keys):
            drop["duplicate"] += 1
            continue
        best = min(cs, key=rank.get)
        cand.append((rank[best], _hash(ds["seed"], a), a, best, cs))
    per_doc, seen_key, picks = Counter(), set(), []
    for _, _, a, best, cs in sorted(cand):
        d = a.split("/", 1)[0]
        if per_doc[d] >= int(ma["per_doc"]):
            drop["per_doc_cap"] += 1
            continue
        if a in dup:
            if dup[a]["key"] in seen_key:
                drop["duplicate"] += 1
                continue
            seen_key.add(dup[a]["key"])
        per_doc[d] += 1
        picks.append({"anchor": a, "stratum": best, "split": bx._split(a), "candidates": cs})
    by_split: dict = {}
    for p in sorted(picks, key=lambda p: _hash(ds["seed"], "lang|" + p["anchor"])):
        by_split.setdefault(p["split"], []).append(p)
    for sp, ps in by_split.items():
        for p, lang in zip(ps, allocate(len(ps), ds["langs"])):
            p["lang"] = lang
    facts = _cached_facts(tcfg, recs, chunks, [c for p in picks for c in [p["anchor"]] +
                                               [x for v in p["candidates"].values() for x, _ in v]])
    sess = _frag_meta(recs, index)
    me = [multi_entry(p, chunks, index, facts, ds["seed"], amended, dup, sess) for p in picks]
    me = [e for e in me if e["neighbors"]]
    ids = _write_packets(tcfg, make_packets(me, int(ma["packet_size"]), "", ds["seed"]), instr)
    _write_index(tcfg)
    rep = {"packets": ids, "anchors": len(me), "by_split": dict(Counter(e["cell"]["split"] for e in me)),
           "by_cell": dict(sorted(Counter(f"{e['cell']['split']}|{e['cell']['lang']}" for e in me).items())),
           "by_stratum": dict(sorted(Counter(f"{e['cell']['split']}|{e['stratum']}" for e in me).items())),
           "excluded": dict(drop), "sources": want_src, "per_doc": int(ma["per_doc"]),
           "instructions": {k: instr[k] for k in ("path", "version", "digest")}}
    _stats_update("multi_packets", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def _hash(seed, key: str) -> str:
    import hashlib
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def cmd_pool_report(tcfg) -> int:
    """M3-R29: the single seed pool of data/dataset/seeds.jsonl after the dedup / split rules vs the single need per
    split x language cell (1000 pairs upper bound), and its coverage at 1.5 x need / yield for yield 0.50 and 0.25.
    Writes data/dataset/single_pool_report.json."""
    import math
    from collections import Counter
    from graphrag.dataset.seeds import Seed, cell_counts, content_signature, cross_split_twin
    ds = tcfg["dataset"]
    recs, _ = _chunks()
    dup = _dup(tcfg, recs)
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    seeds = [Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")]
    ok, seen, drop, sec = Counter(), set(), Counter(), Counter()
    for s in sorted((s for s in seeds if s.pattern == "single"), key=lambda s: s.seed_id):
        sig = content_signature(s.chunks, dup)
        if cross_split_twin(s.chunks, dup, split_of):
            drop["cross_split_twin"] += 1
        elif sig in seen:
            drop["same_content"] += 1
        else:
            seen.add(sig)
            ok[f"{s.split}|{s.lang}"] += 1
            sec[f"{s.split}|{s.section}"] += 1
    need = {f"{sp}|{lang}": n for sp, m in cell_counts(int(ds["size"]), tcfg["split"]["ratios"]).items()
            for lang, n in cell_counts(m, ds["langs"]).items()}
    cells = {k: {"pool": ok[k], "need": n, "target_50": math.ceil(1.5 * n / 0.5), "target_25": math.ceil(1.5 * n / 0.25),
                 "covers_50": ok[k] >= 1.5 * n / 0.5, "covers_25": ok[k] >= 1.5 * n / 0.25}
             for k, n in sorted(need.items())}
    rep = {"seed_doc_cap": ds.get("seed_doc_cap"), "overshoot": ds["overshoot"], "single_after_dedup": sum(ok.values()),
           "dedup_dropped": dict(drop), "by_split_section": dict(sorted(sec.items())), "cells": cells,
           "all_cover_50": all(x["covers_50"] for x in cells.values()),
           "all_cover_25": all(x["covers_25"] for x in cells.values())}
    config.path("data/dataset/single_pool_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1,
                                                                               sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_refresh_packets(tcfg) -> int:
    """M3-R31: re-embed the CURRENT generator instructions (and refresh session / speaker meta, neighbour shortlists
    and cached facts) in the {OUT} packets that have no generator output yet -- same seeds / anchors, same ids.
    --packets a,b,... restricts the set. 0 API calls."""
    from graphrag.dataset.opus import multi_entry, packet_doc, session_info, single_entry
    from graphrag.dataset.seeds import Seed
    ds = tcfg["dataset"]
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    dup = _dup(tcfg, recs)
    sess = _frag_meta(recs, index)
    want = set(ARGS.packets.split(",")) if ARGS.packets else None
    done = []
    for f in _packet_files():
        pk = json.loads(f.read_text(encoding="utf-8"))
        pid = pk["packet_id"]
        if (want is not None and pid not in want) or config.path(f"{OUT}/gen/{pid}.jsonl").exists():
            continue
        cids = [c for e in pk["seeds"] for c in ([e["anchor"]["chunk_id"]] if "anchor" in e else e["seed"]["chunks"])]
        picks = []
        for e in pk["seeds"]:
            if "anchor" in e:
                a = e["anchor"]["chunk_id"]
                cs = {s2: v for s2, v in bx.candidates(a).items() if s2 in {n["source"] for n in e["neighbors"]}}
                picks.append({"anchor": a, "stratum": e["stratum"], "lang": e["cell"]["lang"], "split": e["cell"]["split"],
                              "candidates": cs})
                cids += [x for v in cs.values() for x, _ in v]
        facts = _cached_facts(tcfg, recs, chunks, cids)
        entries = [multi_entry(p, chunks, index, facts, ds["seed"], amended, dup, sess) for p in picks] if picks else \
            [single_entry(Seed.from_json(e["seed"]), chunks, index, facts, amended, sess) for e in pk["seeds"]]
        if [e["seed_id"] for e in entries] != [e["seed_id"] for e in pk["seeds"]]:
            raise SystemExit(f"{pid}: refreshed seeds differ from the packet's")
        doc = packet_doc(pid, entries, instr, pk["output"]["path"])
        doc["output"]["relative_to"] = "advanced-rag-v2/"
        f.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        done.append(pid)
    _write_index(tcfg)
    print(json.dumps({"refreshed": done, "instructions": {k: instr[k] for k in ("path", "version", "digest")}}))
    return 0


_TITLE_SKIP = __import__("re").compile(r"(?i)garbled|letterhead|title is|no usable title|cannot be named|stray sentence|"
                                       r"slogan|epigraph|act field is")


def _bad_docs(recs) -> set:
    """Documents that cannot be named in a self-contained question: an unusable act title (seeds.bad_title; transcripts
    are named by their sitting) or >= 2 generator skips citing the title (skip log, every Opus / Fable dir)."""
    from collections import Counter
    from graphrag.dataset.seeds import bad_title
    out = {r.doc_id for r in recs if r.meta.section != "STENOGRAMMA" and bad_title(r.meta.title or "")}
    n = Counter()
    for d in ("data/dataset/opus", "data/dataset/fable"):
        for g in sorted(config.path(d).glob("gen/*.jsonl")):
            p = config.path(f"{d}/packets/{g.stem}.json")
            if not p.exists():
                continue
            ent = {e["seed_id"]: e for e in json.loads(p.read_text(encoding="utf-8"))["seeds"] if "fragments" in e}
            for line in g.read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line.strip() else {}
                e = ent.get(r.get("seed_id"))
                if r.get("skip") and e and _TITLE_SKIP.search(r.get("reason") or ""):
                    n[e["fragments"][0]["chunk_id"].split("/", 1)[0]] += 1
    return out | {d for d, k in n.items() if k >= 2 and not d.startswith("STENOGRAMMA")}


def _fresh_singles(tcfg, recs, split=None):
    """Single seeds of data/dataset/seeds.jsonl not yet in any Opus / Fable packet nor an earlier pilot, one per
    duplicate content, no cross-split twin (M3-R25), in sha256 order."""
    from graphrag.dataset.seeds import Seed, content_signature, cross_split_twin
    dup = _dup(tcfg, recs)
    split_of = json.loads(config.path("data/gold/splits.json").read_text(encoding="utf-8"))
    used_ids, used_c = _dir_used()
    used_c |= _used_chunks()
    taken = {content_signature([c], dup) for c in used_c}
    out = []
    from graphrag.dataset.generate import _AMENDING
    from graphrag.dataset.sources import SourceIndex
    index = SourceIndex(recs)
    gm = _graph_meta(recs, index)

    from graphrag.dataset.opus import session_info
    from graphrag.dataset.seeds import bad_title, low_content
    sess = session_info(recs)
    chunks_ = {c.chunk_id: c for r in recs for c in r.chunks}
    bad_docs = _bad_docs(recs)

    def low(s) -> int:
        """Priority tier (lower first; nothing dropped but unnameable documents): 0 good; 1 a speech whose discussed
        law the graph does not know / an amending item without a single AMENDS target (wave-2 skips); 2 a chunk
        generators keep skipping (amending header only, headless list tail, chair filler; seeds.low_content)."""
        c = s.chunks[0]
        m = gm(c)
        if low_content(chunks_[c].text, (sess(c) or {}).get("speaker")):
            return 2
        if s.section == "STENOGRAMMA":
            return int("discussed_act" not in m)
        return int(bool(_AMENDING.search(index.act.get(c.split("/", 1)[0], ""))) and "amended_act" not in m)
    for s in sorted((Seed.from_json(r) for r in _jsonl_read("data/dataset/seeds.jsonl")
                     if not set(r["docs"]) & bad_docs),
                    key=lambda s: (low(s), _hash(tcfg["dataset"]["seed"], "fresh|" + s.seed_id))):
        if s.pattern != "single" or (split and s.split != split) or s.seed_id in used_ids or set(s.chunks) & used_c:
            continue
        sig = content_signature(s.chunks, dup)
        if cross_split_twin(s.chunks, dup, split_of) or sig in taken:
            continue
        taken.add(sig)
        out.append(s)
    return out


def _packeted_cells() -> "Counter":
    """Single seeds already in generator packets of {OUT}, per split|lang cell."""
    from collections import Counter
    out = Counter()
    for f in _packet_files():
        for e in json.loads(f.read_text(encoding="utf-8"))["seeds"]:
            if "fragments" in e:
                out[f"{e['cell']['split']}|{e['cell']['lang']}"] += 1
    return out


def cmd_wave_packets(tcfg) -> int:
    """M3-R31 main run: --wave W and either --split S --n-packets N, or --splits train:3,val:1,test:1 -> single
    packets of `packet_size` fresh seeds per split (split-pure). Languages: in proportion to the cell's REMAINING
    seed need of {OUT}/main_plan.json (seeds_needed - seeds already in packets), else the D7 shares; <= 4 seeds per
    document per wave; sha256 order. -> {OUT}/packets/w<W>-single-<split>-NNN.json with the current instructions."""
    from collections import Counter
    from graphrag.dataset.opus import make_packets, session_info, single_entry
    from graphrag.dataset.seeds import allocate
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    size = int(o["packet_size"])
    plan_f = config.path(f"{OUT}/main_plan.json")
    plan = json.loads(plan_f.read_text(encoding="utf-8"))["cells"] if plan_f.exists() else {}
    have = _packeted_cells()
    if ARGS.splits == "auto":                    # packets per split by the plan's remaining seeds (largest remainder)
        from graphrag.dataset.seeds import cell_counts
        by_sp = Counter()
        for k, v in plan.items():
            by_sp[k.split("|")[0]] += v.get("seeds_remaining") or 0
        splits = [(sp, n) for sp, n in cell_counts(int(ARGS.n_packets), {sp: v for sp, v in by_sp.items() if v}).items() if n]
    else:
        splits = [(x.split(":")[0], int(x.split(":")[1])) for x in ARGS.splits.split(",")] if ARGS.splits \
            else [(ARGS.split, int(ARGS.n_packets))]
    sess = _frag_meta(recs, index)
    ids, rep = [], {"by_split": {}}
    ok = True
    for sp, k in splits:
        n = k * size
        rem = {l: (plan.get(f"{sp}|{l}") or {}).get("seeds_remaining") for l in ds["langs"]}
        if any(v is None for v in rem.values()):                 # legacy plan: seeds_needed - seeds already packeted
            rem = {l: max(0, (plan.get(f"{sp}|{l}") or {}).get("seeds_needed", 0) - have[f"{sp}|{l}"]) for l in ds["langs"]}
        shares = rem if sum(rem.values()) else dict(ds["langs"])
        want = Counter(allocate(n, shares))
        pool = _fresh_singles(tcfg, recs, sp)
        by_lang: dict = {}
        for x in pool:
            by_lang.setdefault(x.lang, []).append(x)
        pick, docs = [], Counter()
        for lang, m in sorted(want.items()):
            got = 0
            for x in by_lang.get(lang, []):
                if got >= m:
                    break
                if docs[x.docs[0]] >= 4:
                    continue
                pick.append(x)
                docs[x.docs[0]] += 1
                got += 1
        facts = _cached_facts(tcfg, recs, chunks, [c for x in pick for c in x.chunks])
        se = [single_entry(x, chunks, index, facts, amended, sess) for x in pick]
        ids += _write_packets(tcfg, make_packets(se, size, f"w{ARGS.wave}-", ds["seed"]), instr)
        rep["by_split"][sp] = {"seeds": len(pick), "by_lang": dict(Counter(x.lang for x in pick)),
                               "remaining_need_before": rem, "fresh_pool_left": len(pool) - len(pick)}
        ok &= len(pick) == n
    _write_index(tcfg)
    rep.update(packets=ids, instructions={k: instr[k] for k in ("path", "version", "digest")})
    _stats_update(f"wave{ARGS.wave}_packets", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if ok else 1


def cmd_plan_main(tcfg) -> int:
    """Main-run plan from the MEASURED numbers ({OUT}): per split x language cell, need = cell share of
    (size - --multi-expected); have = consensus-GOOD singles (reviewed.jsonl); expected from pairs verified but not yet
    read by both readers and from packeted seeds not yet generated; the rest in fresh seeds at the measured GOOD per
    seed of the main waves (w*: verified pairs per generated seed x (consensus GOOD per read pair + BORDERLINE-no-BAD
    share x repaired share x --repair-good)); fresh pool per cell; waves of 5 x packet_size; ds_verify calls at
    --calls-per-seed from the ledger (--ledger). Writes {OUT}/main_plan.json."""
    import math
    from collections import Counter
    from graphrag.dataset.opus import reader_id, wave_of
    from graphrag.dataset.seeds import cell_counts
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    recs, _ = _chunks()
    cell = lambda c: f"{c.seed.split}|{c.seed.lang}"
    st = json.loads(config.path(f"{OUT}/stats.json").read_text(encoding="utf-8"))
    ver = _rows_cands(_jsonl_read(f"{OUT}/verified.jsonl"))
    digests = set((o.get("reader_history") or {}).values()) | {o["reader"]["digest"]}
    _, _, sa = _read_root("review", ver, digests)
    _, _, sb = _read_root("review2", ver, digests)
    va, vb = sa["verdict_of"], sb["verdict_of"]
    good = [c for c in _rows_cands(_jsonl_read(f"{OUT}/reviewed.jsonl"))] if config.path(f"{OUT}/reviewed.jsonl").exists() else []
    have = Counter(cell(c) for c in good if c.seed.pattern == "single")
    # measured rates on the main waves (w*), singles
    main = [c for c in ver if c.seed.pattern == "single" and wave_of(c).startswith("w")]
    read = [c for c in main if reader_id(c.seed.seed_id) in va and reader_id(c.seed.seed_id) in vb]
    vv = lambda c: (va[reader_id(c.seed.seed_id)]["verdict"], vb[reader_id(c.seed.seed_id)]["verdict"])
    n_good = sum(vv(c) == ("GOOD", "GOOD") for c in read)
    n_elig = sum("BAD" not in vv(c) and "BORDERLINE" in vv(c) for c in read)
    rdir = config.path(f"{OUT}/repair/packets")
    rep_items = sum(len(json.loads(f.read_text(encoding="utf-8"))["items"]) for f in rdir.glob("*.json")) \
        if rdir.exists() else 0
    rep_ok = len(_jsonl_read(f"{OUT}/repaired_raw.jsonl")) if config.path(f"{OUT}/repaired_raw.jsonl").exists() else 0
    repaired_share = rep_ok / rep_items if rep_items else 0.0          # cumulative over every repair packet
    p_pair = (n_good + n_elig * repaired_share * float(ARGS.repair_good)) / len(read) if read else 0.0
    gen_seeds = sum(len(json.loads(f.read_text(encoding="utf-8"))["seeds"]) for f in _packet_files()
                    if f.stem.startswith("w") and config.path(f"{OUT}/gen/{f.stem}.jsonl").exists())
    pairs_per_seed = len(main) / gen_seeds if gen_seeds else 0.0
    rate_note = {"pairs_per_seed_basis": "verified main-wave pairs / generated main-wave seeds"}
    if ARGS.rate_wave:               # the newest instructions' wave: its code-accepted pairs per seed x the judge keep rate
        raw = _rows_cands(_jsonl_read(f"{OUT}/candidates_raw.jsonl"))
        pre = ARGS.rate_wave + "-"
        seeds_w = sum(len(json.loads(f.read_text(encoding="utf-8"))["seeds"]) for f in _packet_files()
                      if f.stem.startswith(pre) and config.path(f"{OUT}/gen/{f.stem}.jsonl").exists())
        gen_rows = sum(1 for f in _packet_files() if f.stem.startswith(pre)
                       for l in (config.path(f"{OUT}/gen/{f.stem}.jsonl").read_text(encoding="utf-8").splitlines()
                                 if config.path(f"{OUT}/gen/{f.stem}.jsonl").exists() else [])
                       if l.strip() and not json.loads(l).get("skip"))
        code_w = sum(1 for c in raw if c.seed.pattern == "single" and wave_of(c) == ARGS.rate_wave)
        verified_ids = {c.seed.seed_id for c in ver}
        old = [c for c in raw if c.seed.pattern == "single" and wave_of(c) in ("w1", "w2")]
        keep = sum(c.seed.seed_id in verified_ids for c in old) / len(old) if old else 0.0
        pairs_per_seed = (code_w / seeds_w) * keep if seeds_w else pairs_per_seed
        rate_note = {"pairs_per_seed_basis": f"{ARGS.rate_wave}: generated {gen_rows}/{seeds_w} seeds, code-accepted "
                     f"{code_w}; judge keep {round(keep, 4)} (w1+w2)", "generation_rate": round(gen_rows / seeds_w, 4)
                     if seeds_w else None, "code_accept_per_generated": round(code_w / gen_rows, 4) if gen_rows else None}
    p_seed = pairs_per_seed * p_pair
    pending = Counter(cell(c) for c in ver if c.seed.pattern == "single"
                      and not (reader_id(c.seed.seed_id) in va and reader_id(c.seed.seed_id) in vb))
    # candidates of packets no verify run has seen yet (none of their pairs in verified.jsonl): expected pairs =
    # candidates x the judge keep rate
    raw_all = _rows_cands(_jsonl_read(f"{OUT}/candidates_raw.jsonl"))
    vids = {c.seed.seed_id for c in ver}
    seen_pk = {(c.provenance.get("opus") or {}).get("packet_id") for c in ver}
    unver = Counter(cell(c) for c in raw_all if c.seed.pattern == "single"
                    and (c.provenance.get("opus") or {}).get("packet_id") not in seen_pk and c.seed.seed_id not in vids)
    old = [c for c in raw_all if c.seed.pattern == "single" and wave_of(c) in ("w1", "w2")]
    judge_keep = sum(c.seed.seed_id in vids for c in old) / len(old) if old else 0.8
    inflight = Counter()
    for f in _packet_files():
        if config.path(f"{OUT}/gen/{f.stem}.jsonl").exists():
            continue
        for e in json.loads(f.read_text(encoding="utf-8"))["seeds"]:
            if "fragments" in e:
                inflight[f"{e['cell']['split']}|{e['cell']['lang']}"] += 1
    fresh = Counter(f"{s.split}|{s.lang}" for s in _fresh_singles(tcfg, recs))
    multi = int(ARGS.multi_expected)
    cells = {}
    for sp, m in cell_counts(int(ds["size"]) - multi, tcfg["split"]["ratios"]).items():
        for lang, k in cell_counts(m, ds["langs"]).items():
            key = f"{sp}|{lang}"
            exp = have[key] + pending[key] * p_pair + unver[key] * judge_keep * p_pair + inflight[key] * p_seed
            rem = max(0, math.ceil((k - exp) / p_seed)) if p_seed else None
            cells[key] = {"need": k, "have_good": have[key], "pending_pairs": pending[key], "unverified_candidates": unver[key],
                          "inflight_seeds": inflight[key],
                          "expected_good": round(exp, 1), "seeds_remaining": rem, "fresh_pool": fresh[key],
                          "pool_ok": fresh[key] >= (rem or 0)}
    rem = sum(x["seeds_remaining"] or 0 for x in cells.values())
    wave = 5 * int(o["packet_size"])
    cps = float(ARGS.calls_per_seed)
    led = int(ARGS.ledger) if ARGS.ledger else None
    inflight_n = sum(inflight.values())
    calls = math.ceil((rem + inflight_n + sum(unver.values()) / max(0.01, pairs_per_seed or 0.01) * 0) * cps) \
        + math.ceil(sum(unver.values()) * cps / 0.7)                   # unverified candidates still need verify calls
    plan = {"rates": {"read_pairs_main": len(read), "consensus_good_per_read_pair": round(n_good / len(read), 4) if read else None,
                      "borderline_no_bad_share": round(n_elig / len(read), 4) if read else None,
                      "repaired_share": round(repaired_share, 4), "repair_good": float(ARGS.repair_good),
                      "good_per_read_pair_with_repair": round(p_pair, 4), "verified_pairs_per_seed": round(pairs_per_seed, 4),
                      "good_per_seed": round(p_seed, 4), **rate_note},
            "multi_expected": multi, "single_need": int(ds["size"]) - multi,
            "have_good": sum(have.values()), "pending_pairs": sum(pending.values()),
            "unverified_candidates": sum(unver.values()), "judge_keep": round(judge_keep, 4), "inflight_seeds": inflight_n,
            "cells": cells, "seeds_remaining": rem, "fresh_pool": sum(fresh.values()),
            "pool_ok": all(x["pool_ok"] for x in cells.values()), "waves_remaining": math.ceil(rem / wave) if wave else None,
            "calls_per_seed": cps, "verify_calls_projected": calls, "ledger_now": led,
            "ledger_projected": (led + calls) if led is not None else None,
            "cap": int(tcfg["dataset_llm"]["max_calls_total"])}
    config.path(f"{OUT}/main_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1, sort_keys=True)
                                                    + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False))
    return 0 if plan["pool_ok"] and (plan["ledger_projected"] or 0) <= plan["cap"] else 1


_SKIP_LAW = __import__("re").compile(r"(?i)(no|not|never|neither)\b.{0,40}\b(law|draft|qonun|закон)|law[^.;]{0,30}not named|"
                                     r"discussed law|unnamed law|subject of the speech")
_SKIP_AMENDED = __import__("re").compile(r"(?i)amended (act|article|code|law)|names neither the amended|"
                                         r"not name[sd]? (the )?(amended|code)|which law it belongs")


def cmd_retry_packets(tcfg) -> int:
    """M3-R32 follow-up: single seeds whose earlier generator `skip` was an unnamed law in a speech or an amending
    item whose amended act is not named, and whose fragment meta NOW carries discussed_act / amended_act (graph meta,
    instructions v6). A skip is no generation attempt: each such seed gets exactly one fresh generation -- never a
    seed already retried. Split-pure packets of `packet_size` -> {OUT}/packets/retry-NNN.json. 0 API calls."""
    from collections import Counter
    from graphrag.dataset.opus import make_packets, single_entry
    from graphrag.dataset.seeds import Seed
    ds, o = tcfg["dataset"], _opus_cfg(tcfg)
    instr = _instr(tcfg, "generator")
    recs, chunks, index, bx, amended = _opus_context(tcfg)
    meta = _frag_meta(recs, index)
    gm = _graph_meta(recs, index)
    retried = set()
    for f in _packet_files():
        if f.stem.startswith("retry-"):
            retried |= {e["seed_id"] for e in json.loads(f.read_text(encoding="utf-8"))["seeds"]}
    picks, why = [], Counter()
    for f in _packet_files():
        g = config.path(f"{OUT}/gen/{f.stem}.jsonl")
        if f.stem.startswith("retry-") or not g.exists():
            continue
        pk = json.loads(f.read_text(encoding="utf-8"))
        ent = {e["seed_id"]: e for e in pk["seeds"] if "seed" in e}
        for line in g.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            e = ent.get(r.get("seed_id"))
            if not r.get("skip") or e is None or e["seed_id"] in retried:
                continue
            m = gm(e["fragments"][0]["chunk_id"])
            reason = r.get("reason") or ""
            if _SKIP_LAW.search(reason) and "discussed_act" in m:
                why["speech_law_now_known"] += 1
            elif _SKIP_AMENDED.search(reason) and "amended_act" in m:
                why["amended_act_now_known"] += 1
            else:
                why["not_eligible"] += 1
                continue
            picks.append(Seed.from_json(e["seed"]))
            retried.add(e["seed_id"])
    facts = _cached_facts(tcfg, recs, chunks, [c for x in picks for c in x.chunks])
    se = [single_entry(x, chunks, index, facts, amended, meta) for x in picks]
    groups = make_packets(se, int(o["packet_size"]), "retry-", ds["seed"])
    start = sum(1 for f in _packet_files() if f.stem.startswith("retry-"))
    groups = [(f"retry-{start + i + 1:03d}", es) for i, (_, es) in enumerate(groups)]
    ids = _write_packets(tcfg, groups, instr)
    _write_index(tcfg)
    rep = {"packets": ids, "seeds": len(se), "why": dict(why),
           "by_packet": {pid: {"split": es[0]["cell"]["split"], "n": len(es),
                               "langs": dict(Counter(e["cell"]["lang"] for e in es))} for pid, es in groups},
           "instructions": {k: instr[k] for k in ("path", "version", "digest")}}
    _stats_update("retry_packets", rep)
    print(json.dumps(rep, ensure_ascii=False))
    return 0

COMMANDS = {"seeds": cmd_seeds, "calibrate-copy": cmd_calibrate_copy, "pilot3-report": cmd_pilot3_report, "pilot3-seeds": cmd_pilot3_seeds,
            "bridge": cmd_bridge, "quality": cmd_quality, "quality-calibrate": cmd_quality_calibrate}
COMMANDS.update(facts=cmd_facts, generate=cmd_generate)
COMMANDS.update(verify=cmd_verify, finalize=cmd_finalize, trajectories=cmd_trajectories, project=cmd_project)
COMMANDS.update({"pilot-seeds": cmd_pilot_seeds, "pilot-report": cmd_pilot_report, "repilot-seeds": cmd_repilot_seeds,
                 "repilot-report": cmd_repilot_report})
COMMANDS.update({"retry-packets": cmd_retry_packets})
COMMANDS.update({"repair-packets": cmd_repair_packets, "ingest-repair": cmd_ingest_repair})
COMMANDS.update({"refresh-packets": cmd_refresh_packets, "wave-packets": cmd_wave_packets, "plan-main": cmd_plan_main})
COMMANDS.update({"calibrate-near-copy": cmd_calibrate_near_copy, "mini-pilot": cmd_mini_pilot,
                 "multi-packets": cmd_multi_packets, "pool-report": cmd_pool_report})
COMMANDS.update({"pilot-opus": cmd_pilot_opus, "packets": cmd_packets, "ingest-gen": cmd_ingest_gen,
                 "review-packets": cmd_review_packets, "ingest-review": cmd_ingest_review})


def main(argv=None) -> int:
    global OUT, ARGS, PILOT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seeds", "pilot-seeds", "facts", "generate", "verify", "finalize",
                                    "trajectories", "pilot-report", "project", "repilot-seeds", "repilot-report"] +
                    [c for c in COMMANDS if c not in ("seeds", "facts", "generate", "verify", "finalize", "trajectories",
                                                      "project", "pilot-seeds", "pilot-report", "repilot-seeds",
                                                      "repilot-report")])
    ap.add_argument("--dir", default="data/dataset")
    ap.add_argument("--forced")
    ap.add_argument("--rankings")
    ap.add_argument("--width", type=int)
    ap.add_argument("--no-select", action="store_true")
    ap.add_argument("--no-manual-gate", action="store_true")
    ap.add_argument("--pilot-dir")
    ap.add_argument("--agent-model")
    ap.add_argument("--pattern")              # verify: only this pattern (M3-R28 multi re-verify)
    ap.add_argument("--no-plants", action="store_true")
    ap.add_argument("--sample-frac", type=float)
    ap.add_argument("--packets")                  # refresh-packets: comma-separated packet ids
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--n-packets", type=int, default=5)
    ap.add_argument("--split")
    ap.add_argument("--splits")                   # wave-packets: "train:3,val:1,test:1" (packets per split)
    ap.add_argument("--repair-good", type=float, default=0.7347)  # plan-main: consensus GOOD share of repaired pairs
    ap.add_argument("--calls-per-seed", type=float, default=0.45)  # plan-main: ds_verify calls per generated seed (+ repair)
    ap.add_argument("--ledger", type=int)
    ap.add_argument("--rate-wave")                # plan-main: measure pairs per seed on this wave (e.g. w3)
    ap.add_argument("--multi-expected", type=int, default=10)   # review-packets: Opus second-reader sample (M3-R29, e.g. 0.2)   # verify: no plants (sensitivity measured by the full run)          # ingest-gen: the generator agent's model (default config dataset.opus)
    ARGS = ap.parse_args(argv)
    OUT = ARGS.dir
    tcfg = config.load("train")
    opus = tcfg["dataset"].get("opus") or {}
    if ARGS.pilot_dir:
        PILOT_DIR = ARGS.pilot_dir
    elif opus and config.path(OUT) == config.path(opus["dir"]):    # M3-R23: the Opus run has its own pilot
        PILOT_DIR = opus["pilot_dir"]
    elif config.path(f"{OUT}/packets").is_dir():                    # any packet dir: its own pilot (per-dir plans)
        PILOT_DIR = f"{OUT}/pilot"
    else:
        PILOT_DIR = tcfg["dataset"].get("pilot_dir", PILOT_DIR)   # M3-R20: the re-pilot is the pilot
    return COMMANDS[ARGS.cmd](tcfg)


if __name__ == "__main__":
    sys.exit(main())
