"""M7 Task 5: the probe sets of hard questions -- Q1 (count) and Q2 (version) from grl.hard.templates, gold computed
from the pinned graph (docs/m7/2026-10-02-hard-questions-design.md §2). Light, Mac-safe: jsonl + records + pyg.pt.

    PYTHONPATH=. python scripts/hard_probe.py make --seed 20261002 --n-count 30 --n-version 30 --out data/hard/probe/

    PYTHONPATH=. python scripts/hard_probe.py pairs --n 40 --seed 20261002 --out data/hard/probe/compare_packets/
    PYTHONPATH=. python scripts/hard_probe.py ingest-compare --raw data/hard/probe/compare_raw.jsonl \
        --packets data/hard/probe/compare_packets/ --out data/hard/probe/ --max 30

`full-make` (part 2, Task 4): PYTHONPATH=. python scripts/hard_probe.py full-make --seed 20261003 --n-count 250 \
    --n-version 200 --exclude data/hard/probe/ids.json --out data/hard/full/  -> count_raw.jsonl, version_raw.jsonl
(HardItem + gold_docs + split; R32-dropped items are not written), stats.json.

`snapshot` (Task 7): PYTHONPATH=. python scripts/hard_probe.py snapshot --pairs data/hard/probe/*.jsonl --workers 2
-> data/m7/probe/<sha12>/ship/ (pairs.jsonl split probe, rankings/**, splits.json, manifest.json) + snapshot.json
(rank-row checks, gold-in-top-50 per type, old-set parity). Dev Postgres :5437 / qdrant :6343 only.

`pairs` (Task 6, Q3): doc pairs from graph links (grl.hard.compare.pick_doc_pairs) -> one generator packet per pair
<out>/cmp-NNN.json + <out>/index.json (pair stats by kind and supply, packet sizes). The Opus generator (dispatched by
the controller, instructions data/hard/probe/compare_instructions.md) writes compare_raw.jsonl; `ingest-compare` runs
the code checks and writes <out>/compare.jsonl (accepted, packet order, at most --max), compare_rejections.jsonl, adds
the qids to ids.json and a "compare" section to stats.json.

Writes <out>/count.jsonl and <out>/version.jsonl (one HardItem per line) and <out>/ids.json (every qid: plan 2 keeps
them out of every train / val / test). Prints the template / language / |gold| distributions, the R6 drop count,
the leak rejection rate (overall and per template) and the document components.

Inputs: STRUCT texts = grl.env.struct_data.build_struct_inputs (graph pin + sha256 checked, corpus tokenizer,
data.yaml max_tokens); μ⁻¹ = graphrag.env.data.load_env_data (pyg.pt CSR); fragment texts = the records' chunk
`text`; language shares = data/dataset/fable/stats.json ingest_gen.accepted_by_cell summed per language."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag import config  # noqa: E402
from grl.hard.checks import doc_components  # noqa: E402
from grl.hard.templates import count_items, lang_quotas, version_items  # noqa: E402

OLD_SET_STATS = "data/dataset/fable/stats.json"     # worktree symlink -> the main copy's data/dataset


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def lang_shares(path: str = OLD_SET_STATS) -> dict[str, int]:
    """Old-set language counts: ingest_gen.accepted_by_cell ("<split>|<kind>|<lang>": n) summed per language."""
    cells = json.loads(config.path(path).read_text(encoding="utf-8"))["ingest_gen"]["accepted_by_cell"]
    out = Counter()
    for cell, k in cells.items():
        out[cell.rsplit("|", 1)[1]] += int(k)
    return dict(sorted(out.items()))


def load_inputs():
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    from grl.env.struct_data import build_struct_inputs
    texts, _, _, version = build_struct_inputs()
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    d = config.path(latest["dir"])
    nodes, edges = _jsonl(d / "nodes.jsonl"), _jsonl(d / "edges.jsonl")
    data = load_env_data(version)
    mu = {}
    for v, nid in enumerate(data.node_ids):
        rows = data.mu_inv(v)
        if len(rows):
            mu[nid] = [data.chunk_ids[int(i)] for i in rows]
    chunk_texts = {c.chunk_id: c.text for r in read_records(config.path(config.load("data")["records_dir"]))
                   for c in r.chunks}
    return version, nodes, edges, texts, mu, chunk_texts


def _dist(items, f) -> dict:
    return dict(sorted(Counter(f(it) for it in items).items()))


def _rate(st: Counter, t: str | None = None) -> str:
    k = f":{t}" if t else ""
    n, r = st.get("leak_checked" + k, 0), st.get("leak_rejected" + k, 0)
    return f"{r}/{n} = {r / n:.1%}" if n else "0/0"


def report(name: str, items, st: Counter, templates) -> dict:
    print(f"== {name}: {len(items)} items")
    print("  templates:", _dist(items, lambda it: it.template))
    print("  languages:", _dist(items, lambda it: it.lang))
    print("  |gold|:", _dist(items, lambda it: len(it.gold)))
    print("  roots:", len({it.docs[0] for it in items}), "max per root:",
          max(Counter(it.docs[0] for it in items).values()) if items else 0)
    for rule in ("r16_title", "r16_twin"):
        print(f"  {rule} dropped:", st.get(f"{rule}_dropped", 0),
              {t: st[f"{rule}_dropped:{t}"] for t in templates if st.get(f"{rule}_dropped:{t}")})
    print("  candidates:", st.get("candidates", 0), "| R6 dropped:", st.get("r6_dropped", 0),
          {t: st[f"r6_dropped:{t}"] for t in templates if st.get(f"r6_dropped:{t}")})
    extra = {k: st[k] for k in ("no_chunks", "gold_out_of_range", "narrowed", "too_large_refused",
                                "evidence_invisible", "answer_in_question", "candidates_no_language", "singles",
                                "empty_bucket_refused")
             if st.get(k)}
    if extra:
        print("  other:", extra)
    print("  leak rejected (candidate x language):", _rate(st),
          {t: _rate(st, t) for t in templates if st.get(f"leak_checked:{t}")})
    return {"n": len(items), "templates": _dist(items, lambda it: it.template),
            "langs": _dist(items, lambda it: it.lang), "gold_sizes": _dist(items, lambda it: len(it.gold)),
            "stats": dict(sorted(st.items()))}


def cmd_make(a) -> int:
    from grl.hard.templates import COUNT_TEMPLATES, VERSION_TEMPLATES
    version, nodes, edges, texts, mu, chunk_texts = load_inputs()
    shares = lang_shares()
    print(f"graph {version}: {len(nodes)} nodes, {len(edges)} edges, {len(texts)} STRUCT texts, "
          f"{len(mu)} elements with fragments, {len(chunk_texts)} fragments")
    print("old-set language counts:", shares, f"({config.path(OLD_SET_STATS).resolve()})", "-> quotas count", lang_quotas(a.n_count, shares),
          "version", lang_quotas(a.n_version, shares))
    sc, sv = Counter(), Counter()
    count = count_items(nodes, edges, a.seed, a.n_count, struct_texts=texts, lang_shares=shares, stats=sc)
    ver = version_items(nodes, edges, mu, a.seed, a.n_version, chunk_texts=chunk_texts, lang_shares=shares, stats=sv)
    summary = {"graph_version": version, "seed": a.seed, "lang_shares": shares,
               "count": report("count", count, sc, COUNT_TEMPLATES),
               "version": report("version", ver, sv, VERSION_TEMPLATES)}
    items = count + ver
    ids = [it.qid for it in items]
    out = config.path(a.out)
    prev = out / "compare.jsonl"                # Q3 items (ingest-compare) stay excluded after a re-make
    if prev.exists():
        ids += [r["qid"] for r in _jsonl(prev)]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate qids")
    comp = doc_components(items, edges)
    print("document components:", len(set(comp.values())), "largest:", max(Counter(comp.values()).values()))
    out.mkdir(parents=True, exist_ok=True)
    for name, xs in (("count", count), ("version", ver)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(it.to_json(), ensure_ascii=False) + "\n" for it in xs),
                                           encoding="utf-8")
    (out / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    summary["components"] = {"n": len(set(comp.values())), "largest": max(Counter(comp.values()).values())}
    if (out / "stats.json").exists():
        old = json.loads((out / "stats.json").read_text(encoding="utf-8"))
        if "compare" in old:
            summary["compare"] = old["compare"]
    (out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


OLD_SPLITS = "data/gold/splits.json"           # worktree symlink -> the main copy's data/gold


LANG_SHARES_FILE = "lang_shares.json"          # R47: the frozen language counts of a full-make out dir


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_lang_shares(out: Path) -> tuple[dict[str, int], str]:
    """R47: the language counts full-make uses. <out>/lang_shares.json when it exists (frozen: the M3 waves keep
    growing data/dataset/fable/stats.json, and the shares fix every item's language, hence its qid); otherwise
    computed from the old-set stats and written there. Returns (shares, "frozen" | "computed")."""
    p = Path(out) / LANG_SHARES_FILE
    if p.is_file():
        shares = json.loads(p.read_text(encoding="utf-8"))
        if not shares or not all(isinstance(v, int) and v > 0 for v in shares.values()):
            raise SystemExit(f"{p}: malformed language counts")
        return dict(sorted(shares.items())), "frozen"
    shares = lang_shares()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(shares, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return shares, "computed"


def read_exclude(path) -> set[str]:
    """The probe qids to exclude (data/hard/probe/ids.json). Missing or empty is a loud error: full-make never runs
    without the exclusion list."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--exclude {p}: file missing -- the probe ids are required (plan M7 part 2, Task 4)")
    ids = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        raise SystemExit(f"--exclude {p}: no probe ids (expected a non-empty list of qids)")
    return set(ids)


def _old_docs_by_split(rows, old: dict) -> dict:
    """R32 effect: distinct old-set documents the rows touch (docs + gold_docs), by their old split."""
    seen = {d for r in rows for d in r["docs"] + r["gold_docs"] if d in old}
    return dict(sorted(Counter(old[d] for d in seen).items()))


def cmd_full_make(a) -> int:
    """Plan M7 part 2, Task 4: full Q1 / Q2 -- R33 gold (quoted fragments) for Q2, per-type per-root caps (R43: Q1 <= 4
    within Q1, Q2 <= 8 within Q2), the probe qids excluded in every language, R32 before selection (R44) and splits by
    document components with new components balanced to 60/20/20 per type (R45; grl.hard.splits.make_full); language
    shares frozen in <out>/lang_shares.json, input sha256 in stats.json (R47). Writes
    <out>/count_raw.jsonl, <out>/version_raw.jsonl (HardItem + gold_docs + split) and <out>/stats.json."""
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    from grl.env.struct_data import build_struct_inputs
    from grl.hard.splits import SPLITS, make_full
    from grl.hard.templates import COUNT_CAP, COUNT_TEMPLATES, VERSION_CAP, VERSION_TEMPLATES
    exclude = read_exclude(config.path(a.exclude))
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    texts, _, _, version = build_struct_inputs()
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
    out = config.path(a.out)
    shares, shares_src = frozen_lang_shares(out)
    print(f"graph {version}: {len(nodes)} nodes, {len(edges)} edges; {len(exclude)} probe ids excluded; "
          f"{len(old)} old-split docs; languages {shares}")
    rows, sc, sv, rounds = make_full(nodes, edges, mu, a.seed, a.n_count, a.n_version, old_splits=old,
                                     chunk_doc=chunk_doc, struct_texts=texts, chunk_texts=chunk_texts,
                                     chunk_spans=spans, lang_shares=shares, exclude=exclude)
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)):
        raise SystemExit("duplicate qids")
    if set(qids) & exclude:
        raise SystemExit(f"probe qids in the output: {sorted(set(qids) & exclude)[:3]}")
    for name, cap in (("count", COUNT_CAP), ("version", VERSION_CAP)):     # R43: each type within its own cap
        if max(Counter(r["docs"][0] for r in rows if r["qtype"] == name).values(), default=0) > cap:
            raise SystemExit("per-root cap broken")
    for r in rows:
        olds = {old[x] for x in r["docs"] + r["gold_docs"] if x in old}
        if r["split"] is None or (olds and olds != {r["split"]}):
            raise SystemExit(f"{r['qid']}: split {r['split']} vs old {olds}")      # R32 (1), checked again
    summary = {"graph_version": version, "seed": a.seed, "lang_shares": shares, "exclude": a.exclude,
               "n_exclude": len(exclude), "old_splits": OLD_SPLITS,
               "inputs_sha256": {OLD_SPLITS: _sha256(config.path(OLD_SPLITS)),
                                 OLD_SET_STATS: _sha256(config.path(OLD_SET_STATS)),
                                 LANG_SHARES_FILE: _sha256(out / LANG_SHARES_FILE)},
               "lang_shares_source": shares_src,
               "max_per_root": {"count": COUNT_CAP, "version": VERSION_CAP}, "cap_rule": "R43: per type, not shared",
               "r32": "R44: before selection (candidate filter + chain re-selection)", "r32_rounds": rounds,
               "split_rule": "R45: inherited old split; new components greedy by per-type deficit to 60/20/20"}
    targets = {"count": a.n_count, "version": a.n_version}
    for name, st, temps in (("count", sc, COUNT_TEMPLATES), ("version", sv, VERSION_TEMPLATES)):
        sel = [r for r in rows if r["qtype"] == name]
        n32 = st.get("r32_conflict", 0) + st.get("r32_chain", 0)
        print(f"== {name}: accepted {len(sel)} (target {targets[name]}); R32 candidates dropped before selection "
              f"{n32} (own conflict {st.get('r32_conflict', 0)}, chain {st.get('r32_chain', 0)})")
        rep = {"target": targets[name], "selected": len(sel), "accepted": len(sel),
               "shortfall": max(0, targets[name] - len(sel)),
               "accepted_by_template": _dist(sel, lambda r: r["template"]),
               "accepted_by_lang": _dist(sel, lambda r: r["lang"]),
               "accepted_by_split": _dist(sel, lambda r: r["split"]),
               "accepted_by_template_split": _dist(sel, lambda r: f"{r['template']}|{r['split']}"),
               "accepted_by_lang_split": _dist(sel, lambda r: f"{r['lang']}|{r['split']}"),
               "gold_sizes": _dist(sel, lambda r: len(r["gold"])),
               "roots": len({r["docs"][0] for r in sel}),
               "drops": {"r32_candidates": n32,
                         "r32_own_conflict": st.get("r32_conflict", 0), "r32_chain": st.get("r32_chain", 0),
                         "r32_by_template": {t: st.get(f"r32_conflict:{t}", 0) + st.get(f"r32_chain:{t}", 0)
                                             for t in temps if st.get(f"r32_conflict:{t}") or st.get(f"r32_chain:{t}")},
                         "r33_no_quote": st.get("r33_no_quote", 0),
                         "cap_blocked": st.get("cap_blocked", 0),
                         "leak_rejected_renderings": st.get("leak_rejected", 0),
                         "leak_checked_renderings": st.get("leak_checked", 0),
                         "candidates_no_language": st.get("candidates_no_language", 0),
                         "r6": st.get("r6_dropped", 0) + st.get("evidence_invisible", 0),
                         "r16_title": st.get("r16_title_dropped", 0), "r16_twin": st.get("r16_twin_dropped", 0),
                         "probe_excluded_candidates": st.get("probe_excluded", 0)},
               "stats": dict(sorted(st.items()))}
        for k, v in rep["drops"].items():
            print(f"  drop {k}: {v}")
        print("  accepted by template:", rep["accepted_by_template"], "lang:", rep["accepted_by_lang"],
              "split:", rep["accepted_by_split"])
        summary[name] = rep
    summary["r32_effect"] = {
        "old_docs_touched_by_old_split": _old_docs_by_split(rows, old),
        "old_docs_touched_by_type": {t: _old_docs_by_split([r for r in rows if r["qtype"] == t], old)
                                     for t in ("count", "version")},
        "items_inherit_old_split": _dist([r for r in rows if any(x in old for x in r["docs"] + r["gold_docs"])],
                                         lambda r: r["split"]),
        "items_all_docs_new": _dist([r for r in rows if not any(x in old for x in r["docs"] + r["gold_docs"])],
                                      lambda r: r["split"]),
        "accepted_by_split": {s: sum(r["split"] == s for r in rows) for s in SPLITS}}
    comp = doc_components([SimpleNamespace(qid=r["qid"], docs=r["docs"] + r["gold_docs"], gold=[]) for r in rows])
    members = defaultdict(list)
    for r in rows:
        members[comp[r["qid"]]].append(r)
    inherited = {c for c, rs in members.items() if any(x in old for r in rs for x in r["docs"] + r["gold_docs"])}
    summary["components"] = {
        "n": len(members), "with_old_split": len(inherited), "new": len(members) - len(inherited),
        "items_in_new_components": sum(len(members[c]) for c in members if c not in inherited),
        "largest": sorted(([len(rs), rs[0]["split"], dict(sorted(Counter(r["qtype"] for r in rs).items()))]
                           for rs in members.values()), key=lambda x: -x[0])[:5]}
    print("R32 effect:", summary["r32_effect"])
    print("components:", summary["components"])
    for name in ("count", "version"):
        (out / f"{name}_raw.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["qtype"] == name), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


AUG_FILE, AUG_STATS, P8_HARD = "q2_aug_train.jsonl", "q2_aug_stats.json", "p8_hard.jsonl"


def cmd_q2_augment(a) -> int:
    """M7 pilot 8, ruling R55 (grl.hard.augment): Q2 TRAIN rows with template wording under a per-act cap of 16 from
    the candidates Task 4 left unused. Writes <out>/q2_aug_train.jsonl (HardItem + gold_docs + split train +
    meta.augment), <out>/q2_aug_stats.json and <out>/p8_hard.jsonl = para_judged.jsonl rows + the added rows (the
    --hard input of full-snapshot for pilot 8). Val / test rows are copied from para_judged unchanged."""
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    from grl.env.struct_data import build_struct_inputs
    from grl.hard.augment import AUG_MARK, augment_q2
    from grl.hard.probe import probe_ids
    from grl.train.data import qreal_ids
    out = config.path(a.out)
    exclude = read_exclude(config.path(a.exclude))
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    shares_f = out / LANG_SHARES_FILE
    if not shares_f.is_file():
        raise SystemExit(f"{shares_f}: the frozen language counts of full-make are required (R47)")
    shares = json.loads(shares_f.read_text(encoding="utf-8"))
    texts, _, _, version = build_struct_inputs()
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
    q12 = _jsonl(out / "count_raw.jsonl") + _jsonl(out / "version_raw.jsonl")
    q3 = _jsonl(out / "compare_final.jsonl")
    judged = _jsonl(config.path(a.judged))
    rows, rep = augment_q2(nodes, edges, mu, a.seed, a.n, old_splits=old, chunk_doc=chunk_doc, q12_rows=q12,
                           q3_rows=q3, cap=a.cap, chunk_texts=chunk_texts, chunk_spans=spans, lang_shares=shares,
                           exclude=exclude)
    qids = {r["qid"] for r in rows}
    bad = qids & (probe_ids() | qreal_ids() | {r["qid"] for r in judged})
    if bad:
        raise SystemExit(f"added qids among probe / Q-real / judged rows: {sorted(bad)[:3]}")
    if any(r["split"] != "train" or r["meta"].get("augment") != AUG_MARK for r in rows):
        raise SystemExit("an added row is not a marked train row")
    (out / AUG_FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    comb = judged + rows
    (out / P8_HARD).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comb), encoding="utf-8")
    jv = Counter((r["qtype"], r["split"]) for r in judged)
    rep.update({"graph_version": version, "seed": a.seed, "exclude": a.exclude, "judged": a.judged,
                "judged_by_type_split": {f"{t}|{s}": k for (t, s), k in sorted(jv.items())},
                "p8_hard_by_type_split": {f"{t}|{s}": k for (t, s), k in
                                          sorted(Counter((r["qtype"], r["split"]) for r in comb).items())},
                "inputs_sha256": {n: _sha256(out / n) for n in ("count_raw.jsonl", "version_raw.jsonl",
                                                                 "compare_final.jsonl", LANG_SHARES_FILE)}
                | {a.judged: _sha256(config.path(a.judged))},
                "rule": "R55: Q2 train only, template wording (no paraphrase / judge / budget), per-act cap 16, "
                        "meta.augment = template_train_only"})
    (out / AUG_STATS).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_added", "by_template", "by_lang", "gold_sizes", "roots",
                                          "max_per_root_after", "drops", "p8_hard_by_type_split")},
                     ensure_ascii=False, indent=1))
    return 0


def load_pair_inputs():
    """nodes, edges (pinned graph), μ⁻¹ (pyg.pt CSR), and from the records: doc -> [(chunk_id, tokens)] in document
    order, chunk texts, chunk -> doc, doc -> meta."""
    from graphrag.data.ingest import read_records
    from graphrag.env.data import load_env_data
    data = load_env_data()
    d = config.path(json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))["dir"])
    nodes, edges = _jsonl(d / "nodes.jsonl"), _jsonl(d / "edges.jsonl")
    mu = {}
    for v, nid in enumerate(data.node_ids):
        rows = data.mu_inv(v)
        if len(rows):
            mu[nid] = [data.chunk_ids[int(i)] for i in rows]
    recs = read_records(config.path(config.load("data")["records_dir"]))
    doc_chunks = {r.doc_id: [(c.chunk_id, int(c.tokens)) for c in r.chunks] for r in recs}
    texts = {c.chunk_id: c.text for r in recs for c in r.chunks}
    chunk_doc = {c.chunk_id: r.doc_id for r in recs for c in r.chunks}
    meta = {r.doc_id: asdict(r.meta) for r in recs}
    return data.version, nodes, edges, mu, doc_chunks, texts, chunk_doc, meta


def _packet_summary(packets, st, extra: dict) -> dict:
    from grl.hard.compare import KINDS
    toks = sorted(p["tokens"] for p in packets) or [0]
    shown = sorted(sum(d["n_chunks_shown"] for d in p["docs"]) for p in packets) or [0]
    return {**extra,
            "supply": {k: st.get(f"supply:{k}", 0) for k in KINDS},
            "picked": {k: sum(1 for p in packets if p["kind"] == k) for k in KINDS},
            "langs": _dist(packets, lambda p: p["lang"]),
            "langs_by_kind": {k: _dist([p for p in packets if p["kind"] == k], lambda p: p["lang"]) for k in KINDS},
            "tokens": {"min": toks[0], "median": toks[len(toks) // 2], "max": toks[-1], "sum": sum(toks)},
            "chunks_shown": {"min": shown[0], "median": shown[len(shown) // 2], "max": shown[-1]},
            "truncated_docs": sum(1 for p in packets for d in p["docs"] if d["n_chunks_shown"] < d["n_chunks_total"]),
            "docs_without_linked_chunk": sum(1 for p in packets for d in p["docs"] if not d["linked_chunks"]),
            "stats": dict(sorted(st.items())),
            "packets": [{"packet_id": p["packet_id"], "qid": p["qid"], "kind": p["kind"], "lang": p["lang"],
                         "pair": list(p.get("pair") or [d["doc_id"] for d in p["docs"]] + [p["kind"]]),
                         "tokens": p["tokens"], "shown": [d["n_chunks_shown"] for d in p["docs"]],
                         "total": [d["n_chunks_total"] for d in p["docs"]]} for p in packets]}


def _write_new(path: Path, text: str) -> None:
    """R25: never overwrite a file the generators may be reading."""
    if path.exists():
        raise SystemExit(f"{path} exists: refusing to overwrite")
    path.write_text(text, encoding="utf-8")


def _existing_packets(out: Path) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out.glob("cmp-*.json"))]


def cmd_pairs(a) -> int:
    """Default: the first packets (refuses if any cmp-*.json exists). --valid-only (R25 dry mode): writes only
    compare_valid_pairs.json (every pair valid under R23 / R24 and the status of each existing packet).
    --topup (R25): new packets from cmp-<next> for valid pairs not among the existing ones, replacing the invalid
    ones per kind (filled from other kinds if a kind runs dry), total <= 60; never overwrites."""
    from grl.hard.compare import KINDS, build_packets, pair_candidates, packet_key, pick_doc_pairs
    version, nodes, edges, mu, doc_chunks, texts, _, meta = load_pair_inputs()
    shares = lang_shares()
    st = Counter()
    n_chunks = {d: len(cs) for d, cs in doc_chunks.items()}
    dates = {d: m.get("date") for d, m in meta.items()}
    out = config.path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = _existing_packets(out)
    if a.valid_only or a.topup:
        why = {}
        cands = pair_candidates(nodes, edges, a.seed, n_chunks=n_chunks, doc_dates=dates, stats=st, mu_inv=mu,
                                why=why)
        valid = {(x, y, k) for k in KINDS for x, y, _ in cands[k]}
        status = {}
        for p in old:
            key = packet_key(p)
            status[p["packet_id"]] = {"pair": list(key), "valid": key in valid,
                                      "reason": None if key in valid else why.get(tuple(sorted(key[:2])), "not_linked")}
        surv = [p for p in old if status[p["packet_id"]]["valid"]]
        by_kind = {k: {"packets": sum(1 for p in old if p["kind"] == k),
                       "valid": sum(1 for p in surv if p["kind"] == k)} for k in KINDS}
        reasons = _dist([v for v in status.values() if not v["valid"]], lambda v: v["reason"])
        print(f"existing packets: {len(old)}, valid under R23/R24: {len(surv)}", by_kind, "invalid reasons:", reasons)
        print("  supply:", {k: len(cands[k]) for k in KINDS})
        if a.valid_only:
            vp = config.path(a.valid_out)
            vp.write_text(json.dumps({"graph_version": version, "seed": a.seed, "rules": ["R23", "R24"],
                                      "supply": {k: len(cands[k]) for k in KINDS},
                                      "existing": {"n": len(old), "valid": len(surv), "by_kind": by_kind,
                                                   "invalid_reasons": reasons, "packets": status},
                                      "stats": dict(sorted(st.items())),
                                      "pairs": sorted([list(x) for x in valid])},
                                     ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print("wrote", vp)
            return 0
        room = a.cap - len(old)
        lost = {k: by_kind[k]["packets"] - by_kind[k]["valid"] for k in KINDS}
        need = min(room, sum(lost.values()))
        fill_quota = None
        if a.n_new is not None:                 # R26: a fixed number of new packets, some kinds excluded
            bad = set(a.exclude_kinds or ())
            if bad - set(KINDS):
                raise SystemExit(f"unknown kinds {sorted(bad - set(KINDS))}")
            need = min(room, a.n_new)
            lost = fill_quota = {k: (0 if k in bad else need) for k in KINDS}
        preset = [packet_key(p) for p in surv]
        exclude = [packet_key(p) for p in old]
        new = pick_doc_pairs(nodes, edges, need, a.seed, n_chunks=n_chunks, doc_dates=dates, mu_inv=mu,
                             preset=preset, exclude=exclude, quota=lost)
        if len(new) < need:                                    # a kind ran dry: fill from the others
            new += pick_doc_pairs(nodes, edges, need - len(new), a.seed, n_chunks=n_chunks, doc_dates=dates,
                                  mu_inv=mu, preset=preset + new, exclude=exclude + new, quota=fill_quota)
        start = max(int(p["packet_id"].split("-")[1]) for p in old) + 1 if old else 0
        packets = build_packets(new, nodes, edges, doc_chunks, mu, texts, meta, shares, start=start,
                                prior=[(p["kind"], p["lang"]) for p in surv])
        for p in packets:
            _write_new(out / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
        summary = _packet_summary(packets, st, {"graph_version": version, "seed": a.seed, "mode": "topup (R25)",
                                                "replaces": lost, "room": room, "new": len(packets)})
        name = f"index_topup_{start:03d}.json"
        _write_new(out / name, json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        print(f"new packets: {len(packets)} (cmp-{start:03d}..), by kind:", summary["picked"])
        print("  langs (new):", summary["langs"], "| valid set after top-up:",
              _dist(surv + packets, lambda p: p["lang"]), _dist(surv + packets, lambda p: p["kind"]))
        print("  tokens:", summary["tokens"], "chunks_shown:", summary["chunks_shown"],
              "docs_without_linked_chunk:", summary["docs_without_linked_chunk"])
        print("wrote", out / name)
        return 0
    if old:
        raise SystemExit(f"{out} already holds {len(old)} packets: use --topup / --valid-only (R25)")
    pairs = pick_doc_pairs(nodes, edges, a.n, a.seed, n_chunks=n_chunks, doc_dates=dates, stats=st, mu_inv=mu)
    packets = build_packets(pairs, nodes, edges, doc_chunks, mu, texts, meta, shares)
    for p in packets:
        _write_new(out / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    summary = _packet_summary(packets, st, {"graph_version": version, "seed": a.seed, "n": a.n,
                                            "lang_shares": shares})
    _write_new(out / "index.json", json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(f"graph {version}: {len(pairs)} pairs / {a.n} asked")
    for k in ("supply", "picked", "langs", "langs_by_kind", "tokens", "chunks_shown", "truncated_docs",
              "docs_without_linked_chunk"):
        print(f"  {k}:", summary[k])
    print("wrote", out)
    return 0


def _raw_lines(paths) -> list:
    """Raw generator lines from several files / globs, in the given order; a line that is not JSON is kept as its
    string (ingest rejects it as malformed)."""
    import glob as _glob
    files = []
    for x in paths:
        hits = sorted(_glob.glob(str(config.path(x))))
        if not hits:
            raise SystemExit(f"--raw {x}: no file matches ({config.path(x)})")
        files += [Path(h) for h in hits]
    out = []
    for f in dict.fromkeys(files):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append(line)
    return out


def load_valid(path) -> list:
    """R25 / R27: the validity list is mandatory -- a missing or unreadable file is a hard error."""
    vp = config.path(path)
    if not vp.exists():
        raise SystemExit(f"{vp} missing: run `hard_probe.py pairs --valid-only` first (R25 / R27)")
    try:
        return json.loads(vp.read_text(encoding="utf-8"))["pairs"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(f"{vp}: not a validity list ({e})")


def ingest_outputs(raws, packets, chunk_doc, texts, valid, out: Path, max_n: int | None):
    """Every ingest-compare output as {path: text}, computed in memory (nothing is written here)."""
    from grl.hard.checks import doc_components as comps
    from grl.hard.compare import ingest_compare, raw_packet_id
    items, st, rej, both = ingest_compare(raws, packets, chunk_doc, texts, valid=valid)
    kept = items[:max_n] if max_n else items
    st["kept"] = len(kept)
    ids_p = out / "ids.json"
    ids = json.loads(ids_p.read_text(encoding="utf-8")) if ids_p.exists() else []
    ids = [q for q in ids if not q.startswith("m7-compare-")] + [it.qid for it in kept]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate qids")
    comp = comps(kept)
    summary = {"packets": len(packets), "packets_answered": st["packets_answered"], "refusal_and_item": both,
               "kept": len(kept), "kinds": _dist(kept, lambda it: it.template), "langs": _dist(kept, lambda it: it.lang),
               "gold_sizes": _dist(kept, lambda it: len(it.gold)),
               "components": {"n": len(set(comp.values())),
                              "largest": max(Counter(comp.values()).values()) if comp else 0},
               "stats": dict(sorted(st.items()))}
    sp = out / "stats.json"
    allst = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    allst["compare"] = summary
    files = {
        out / "compare.jsonl": "".join(json.dumps(it.to_json(), ensure_ascii=False) + "\n" for it in kept),
        out / "compare_rejections.jsonl": "".join(
            json.dumps({"packet_id": raw_packet_id(r), "reasons": w, "raw": r}, ensure_ascii=False) + "\n"
            for r, w in rej),
        ids_p: json.dumps(ids, ensure_ascii=False, indent=0) + "\n",
        sp: json.dumps(allst, ensure_ascii=False, indent=1) + "\n"}
    return files, summary, kept, both


def atomic_write(files: dict) -> None:
    """Write every file to <name>.tmp first, then rename them all (no partial output set on a failure)."""
    import os
    tmps = []
    try:
        for path, text in files.items():
            t = path.with_name(path.name + ".tmp")
            t.write_text(text, encoding="utf-8")
            tmps.append((t, path))
    except BaseException:
        for t, _ in tmps:
            t.unlink(missing_ok=True)
        raise
    for t, path in tmps:
        os.replace(t, path)


def cmd_ingest_compare(a) -> int:
    from graphrag.data.ingest import read_records
    valid = load_valid(a.valid)
    raws = _raw_lines(a.raw)
    recs = read_records(config.path(config.load("data")["records_dir"]))
    chunk_doc = {c.chunk_id: r.doc_id for r in recs for c in r.chunks}
    texts = {c.chunk_id: c.text for r in recs for c in r.chunks}
    packets = {}
    for f in sorted(config.path(a.packets).glob("cmp-*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        packets[p["packet_id"]] = p
    out = config.path(a.out)
    files, summary, kept, both = ingest_outputs(raws, packets, chunk_doc, texts, valid, out, a.max)
    atomic_write(files)
    if both:
        print("FLAG: packets with both a refusal and an item:", both)
    for k, v in summary.items():
        print(f"  {k}:", v)
    if len(kept) < (a.max or 0):
        print(f"SHORT: {len(kept)} accepted < {a.max}; packets so far {len(packets)} (cap 60)")
    print("wrote", out / "compare.jsonl")
    return 0


PROBE_OUT_ROOT = "data/m7/probe"                   # <sha12>/ (Mac) + <sha12>/ship (the only part sent to Spark)
PARITY_SNAPSHOT = "data/m5/pilot/083e116775b3"     # the old set's pilot 5 / 6 snapshot (U9 rankings, Mac-built)
RANKINGS_STRAY = ("data/m3/rankings_report_probe.json", "data/m3/rankings_report_dataset.json",
                  "data/m3/rankings_report_hard.json")


def _run_rankings(*args) -> None:
    """scripts/build_rankings.py in a child (the old set's path: m5_pilot_data._rankings), its side report removed
    (a report that existed before is restored byte for byte)."""
    import os
    import subprocess
    reps = [config.path(r) for r in RANKINGS_STRAY]
    before = {r: (r.read_bytes() if r.exists() else None) for r in reps}
    had_dir = reps[0].parent.exists()
    try:
        r = subprocess.run([sys.executable, "scripts/build_rankings.py", *args], cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": str(ROOT)})
        if r.returncode:
            raise SystemExit(f"build_rankings.py exited {r.returncode}")
    finally:
        for rep, b in before.items():
            if b is not None:
                rep.write_bytes(b)
            else:
                rep.unlink(missing_ok=True)
        d = reps[0].parent
        if not had_dir and d.exists() and not any(d.iterdir()):
            d.rmdir()


def build_probe_ship(d: Path, rows) -> dict:
    """<d>/ship/: pairs.jsonl (HardItem fields + split probe), splits.json, rankings/** (latest.json re-pointed),
    manifest.json; built in a temp dir, checked (check_probe_ship), renamed -- the pilot 5-6 build_ship layout."""
    import shutil
    from grl.hard.probe import PROBE_SPLIT, check_probe_ship
    from grl.train.pilot import SHIP_DIR, _repoint, _sha
    ship, tmp = d / SHIP_DIR, d / f".tmp-{SHIP_DIR}"
    for p in (tmp, ship):
        if p.exists():
            shutil.rmtree(p)
    tmp.mkdir()
    shutil.copyfile(d / "pairs.jsonl", tmp / "pairs.jsonl")
    (tmp / "splits.json").write_text(json.dumps({PROBE_SPLIT: len(rows)}, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copytree(d / "rankings", tmp / "rankings")
    _repoint(tmp / "rankings" / "latest.json", ship / "rankings")
    check_probe_ship(tmp, {r["qid"] for r in rows})
    files = sorted(p for p in tmp.rglob("*") if p.is_file())
    man = {"sha256": {str(p.relative_to(tmp)): _sha(p) for p in files}}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    import os
    os.rename(tmp, ship)
    return man


def parity(d: Path, n: int) -> dict:
    """Rebuild the U9 rankings of the first n old-set TRAIN queries of the pilot 5 / 6 snapshot through the same
    build_rankings path (Mac, CPU) into <d>/parity/ (never shipped) and compare them with the stored arrays."""
    import numpy as np
    from graphrag.retrieval.rankings import load_rankings
    snap = config.path(PARITY_SNAPSHOT)
    rows = [r for r in _jsonl(snap / "candidates.jsonl") if r["split"] == "train"]
    rows = sorted(rows, key=lambda r: r["id"])[:n]
    pd = d / "parity"
    pd.mkdir(exist_ok=True)
    (pd / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                 for r in rows), encoding="utf-8")
    _run_rankings("--family", "dataset", "--candidates", str(pd / "candidates.jsonl"), "--out", str(pd / "rankings"))
    old = load_rankings(snap / "ship" / "rankings" / "latest.json")
    new = load_rankings(pd / "rankings" / "latest.json")
    per = {}
    for k in new.keys():
        a, b = old.get(k), new.get(k)
        per[k] = {"lex_equal": bool(np.array_equal(a.lex, b.lex)), "vec_equal": bool(np.array_equal(a.vec, b.vec)),
                  "title_equal": bool(np.array_equal(a.title, b.title)),
                  "vec_top50_equal": bool(np.array_equal(a.vec[:50], b.vec[:50])),
                  "lex_score_maxdiff": float(np.max(np.abs(a.lex_score - b.lex_score))) if len(a.lex) == len(b.lex) else None,
                  "vec_score_maxdiff": float(np.max(np.abs(a.vec_score - b.vec_score))) if len(a.vec) == len(b.vec) else None,
                  "qvec_maxdiff": float(np.max(np.abs(a.qvec - b.qvec)))}
    skip = ("family", "candidates")
    oi, ni = old.meta["inputs"], new.meta["inputs"]
    return {"old": PARITY_SNAPSHOT, "old_version": old.meta["version"], "n": len(per), "per_query": per,
            "settings_equal_except_family_candidates": {k: oi.get(k) == ni.get(k) for k in sorted(set(oi) | set(ni))
                                                        if k not in skip}}


def cmd_snapshot(a) -> int:
    """M7 Task 7 Step 1-2 (Mac, dev stores only, <= --workers threads): the probe rows -> <root>/<sha12>/pairs.jsonl
    -> U9 rankings (build_rankings.py --family probe, settings from config/) -> ship/ -> checks: non-empty q.lex /
    q.vec, no ranked row >= n_base_chunks, every gold id an env row (STRUCT past the base rows), gold-in-top-k rates
    per type; optional parity of n old-set queries. Writes <sha12>/snapshot.json."""
    import glob as _glob
    import hashlib
    import os
    from grl.threads import pin_blas
    pin_blas(a.workers, force=True)                     # the child inherits OMP / BLAS threads = --workers (<= 2)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from grl.hard.probe import (check_probe_ship, check_rank_rows, gold_in_top, load_probe_items, probe_ids,
                                probe_rows, STRUCT_PREFIX)
    from grl.train.pilot import require_dev_pg, verify_ship
    require_dev_pg(config.PG_DSN)                       # 5437 only (5436 is production)
    if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
        raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
    paths = []
    for x in a.pairs:
        hits = sorted(_glob.glob(str(config.path(x))))
        if not hits:
            raise SystemExit(f"--pairs {x}: no file matches")
        paths += [Path(h) for h in hits]
    from grl.hard.probe import PROBE_SOURCES
    paths = list(dict.fromkeys(paths))
    skipped = [p.name for p in paths if p.name not in PROBE_SOURCES]   # a data/hard/probe/*.jsonl glob also hits
    paths = [p for p in paths if p.name in PROBE_SOURCES]              # compare_raw_* / compare_rejections
    if sorted(p.name for p in paths) != sorted(PROBE_SOURCES):
        raise SystemExit(f"--pairs must name each of {PROBE_SOURCES} once (got {[p.name for p in paths]})")
    if skipped:
        print("not probe sets, ignored:", skipped)
    ids = sorted(probe_ids())
    rows = probe_rows(paths, ids)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    d = config.path(a.out_root) / tag
    d.mkdir(parents=True, exist_ok=True)
    if (d / "pairs.jsonl").exists() and (d / "pairs.jsonl").read_text(encoding="utf-8") != text:
        raise SystemExit(f"{d / 'pairs.jsonl'} differs from the rebuilt probe rows")
    (d / "pairs.jsonl").write_text(text, encoding="utf-8")
    if not (d / "ship" / "manifest.json").exists():
        if not (d / "rankings" / "latest.json").exists():
            _run_rankings("--family", "probe", "--candidates", str(d / "pairs.jsonl"), "--out", str(d / "rankings"))
        build_probe_ship(d, rows)
    man = verify_ship(d)
    check_probe_ship(d / "ship", set(ids))
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    data = load_struct_env_data(articles=getattr(a, "articles", False))   # Task 6c: article STRUCT gold
    ranks = load_rankings(d / "ship" / "rankings" / "latest.json", graph_version=data.version)
    keys = [f"{r['qid']}|q" for r in rows]
    rr = check_rank_rows(ranks, keys, data.n_base_chunks)
    items = load_probe_items(d / "ship" / "pairs.jsonl", data)
    struct_gold = sum(1 for it in items for g in it.gold if g >= data.n_base_chunks)
    struct_ids = sum(1 for r in rows for g in r["gold"] if g.startswith(STRUCT_PREFIX))
    if struct_gold != struct_ids:
        raise SystemExit(f"STRUCT gold rows {struct_gold} != STRUCT gold ids {struct_ids}")
    rep = {"dir": str(d.relative_to(config.ROOT)), "ship": str((d / "ship").relative_to(config.ROOT)),
           "n": len(rows), "by_type": _dist(rows, lambda r: r["qtype"]),
           "rankings": json.loads((d / "ship" / "rankings" / "latest.json").read_text(encoding="utf-8"))["version"],
           "ranking_inputs": ranks.meta["inputs"], "n_base_chunks": data.n_base_chunks, "n_chunks": data.n_chunks,
           "rank_rows": rr, "struct_gold_ids": struct_ids, "struct_gold_rows_ge_n_base": struct_gold,
           "gold_in_top50": gold_in_top(ranks, rows, data.chunk_ids, k=50),
           "manifest_sha256": hashlib.sha256((d / "ship" / "manifest.json").read_bytes()).hexdigest(),
           "ship_files": len(man["sha256"]), "workers": a.workers}
    if a.parity:
        rep["parity"] = parity(d, a.parity)
    (d / "snapshot.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "ranking_inputs"}, ensure_ascii=False, indent=1))
    return 0


FULL_OUT_ROOT = "data/m7/full"                     # <sha12>/ (Mac) + <sha12>/ship (the only part sent to Spark)
FULL_HARD = "data/hard/full/para_judged.jsonl"


def cmd_full_snapshot(a) -> int:
    """M7 pilot 7 (plan Task 7, Q1 / Q2 only; Mac, dev stores, <= --workers threads): the joint struct ship
    data/m7/full/<sha12>/ship = the pilot 5 ship's old train / val pairs (649 / 182, as pilot 6) + the M7 Q1 / Q2
    train / val rows of para_judged.jsonl as Pair rows (grl.hard.ship.hard_pairs; test rows never read in, Q3 out,
    probe ids refused). Rankings: the hard rows through build_rankings.py --family hard (the old set's path and
    settings), merged with the pilot 5 ship's rankings (grl.hard.ship.merge_rankings); trajectories.jsonl is empty
    (labels: train.py expert --variant struct). Checks: build_ship / check_ship with variant struct (STRUCT gold
    accepted), the STRUCT-free ranking check on every key, every gold id an env row, gold-in-top-50 per type, parity
    of --parity old-set queries. No test_eval/ (the tests stay sealed). Writes <sha12>/snapshot.json."""
    import os
    from grl.threads import pin_blas
    pin_blas(a.workers, force=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from grl.hard.probe import STRUCT_PREFIX, check_rank_rows, gold_in_top
    from grl.hard.ship import HARD_ORIGIN, STRUCT_VARIANT, hard_pairs, merge_rankings
    from grl.train.data import ALLOWED_SPLITS
    from grl.train.pilot import build_ship, check_ship, require_dev_pg, verify_ship
    require_dev_pg(config.PG_DSN)
    if config.QDRANT_URL.rstrip("/").rsplit(":", 1)[-1] != "6343":
        raise SystemExit("GRL_QDRANT_URL must be the dev qdrant :6343")
    old_snap = config.path(PARITY_SNAPSHOT)
    verify_ship(old_snap)
    old = _jsonl(old_snap / "ship" / "pairs.jsonl")
    if {r["split"] for r in old} - set(ALLOWED_SPLITS) or any(r.get("origin") == HARD_ORIGIN for r in old):
        raise SystemExit(f"{old_snap}/ship/pairs.jsonl: rows outside the old train / val")
    hard, hrep = hard_pairs(config.path(a.hard))
    clash = {r["id"] for r in old} & {r["id"] for r in hard}
    if clash:
        raise SystemExit(f"old and hard ids overlap: {sorted(clash)[:3]}")
    rows = sorted(old + hard, key=lambda r: r["id"])
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    d = config.path(a.out_root) / tag
    d.mkdir(parents=True, exist_ok=True)
    htext = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in hard)
    for name, t in (("candidates.jsonl", text), ("pairs.jsonl", text), ("hard_pairs.jsonl", htext),
                    ("trajectories.jsonl", "")):
        f = d / name
        if f.exists() and f.read_text(encoding="utf-8") != t:
            raise SystemExit(f"{f} differs from the rebuilt rows")
        f.write_text(t, encoding="utf-8")
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ALLOWED_SPLITS}
    reuse = getattr(a, "reuse_hard_rankings", None)
    if not (d / "ship" / "manifest.json").exists():
        if reuse:                                       # pilot 8: rank only the hard rows the reused artifact lacks
            from graphrag.retrieval.rankings import load_rankings as _lr
            have = set(_lr(config.path(reuse)).keys())
            new_rows = [r for r in hard if f"{r['id']}|q" not in have]
            if not (d / "rankings_hard" / "latest.json").exists():
                (d / "hard_pairs_new.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in new_rows),
                    encoding="utf-8")
                _run_rankings("--family", "hard", "--candidates", str(d / "hard_pairs_new.jsonl"),
                              "--out", str(d / "rankings_hard"))
            parts = [old_snap / "ship" / "rankings" / "latest.json", config.path(reuse),
                     d / "rankings_hard" / "latest.json"]
        else:
            if not (d / "rankings_hard" / "latest.json").exists():
                _run_rankings("--family", "hard", "--candidates", str(d / "hard_pairs.jsonl"),
                              "--out", str(d / "rankings_hard"))
            parts = [old_snap / "ship" / "rankings" / "latest.json", d / "rankings_hard" / "latest.json"]
        if not (d / "rankings" / "latest.json").exists():
            merge_rankings(parts, d / "rankings")
        build_ship(d, counts, test_eval=False, variant=STRUCT_VARIANT)
    man = verify_ship(d)
    allowed = {r["id"] for r in rows}
    check_ship(d / "ship", allowed, None, STRUCT_VARIANT)
    from graphrag.retrieval.rankings import load_rankings
    from grl.env.struct_data import load_struct_env_data
    data = load_struct_env_data(articles=getattr(a, "articles", False))   # Task 6c: article STRUCT gold
    ranks = load_rankings(d / "ship" / "rankings" / "latest.json", graph_version=data.version)
    keys = sorted(f"{r['id']}|q" for r in rows)
    if sorted(ranks.keys()) != keys:
        raise SystemExit("ship rankings keys != the ship's train / val ids")
    rr = check_rank_rows(ranks, keys, data.n_base_chunks)
    miss = [(r["id"], g) for r in rows for g in r["C_ref"] if g not in data.chunk_row]
    if miss:
        raise SystemExit(f"gold ids that are no struct env row: {miss[:3]}")
    struct_rows = sum(1 for r in rows for g in r["C_ref"] if int(data.chunk_row[g]) >= data.n_base_chunks)
    struct_ids = sum(1 for r in rows for g in r["C_ref"] if g.startswith(STRUCT_PREFIX))
    if struct_rows != struct_ids or any(g.startswith(STRUCT_PREFIX) for r in old for g in r["C_ref"]):
        raise SystemExit(f"STRUCT gold rows {struct_rows} != STRUCT gold ids {struct_ids} (or STRUCT in the old set)")
    old_lt = json.loads((old_snap / "ship" / "rankings" / "latest.json").read_text(encoding="utf-8"))
    import numpy as np
    oldr = load_rankings(old_snap / "ship" / "rankings" / "latest.json", graph_version=data.version)
    same_old = all(all(np.array_equal(getattr(ranks.get(k), f), getattr(oldr.get(k), f))
                       for f in ("lex", "vec", "title", "lex_score", "vec_score", "qvec"))
                   for k in (f"{r['id']}|q" for r in old))
    if not same_old:
        raise SystemExit("merged rankings differ from the pilot 5 ship's rankings on an old key")
    hrows = [{"qid": r["id"], "qtype": r["pattern"], "gold": r["C_ref"]} for r in hard]
    by_set = {f"{'hard' if r.get('origin') == HARD_ORIGIN else 'old'}|{r['pattern']}|{r['split']}": 0 for r in rows}
    for r in rows:
        by_set[f"{'hard' if r.get('origin') == HARD_ORIGIN else 'old'}|{r['pattern']}|{r['split']}"] += 1
    rep = {"dir": str(d.relative_to(config.ROOT)), "ship": str((d / "ship").relative_to(config.ROOT)),
           "variant": STRUCT_VARIANT, "n": len(rows), "splits": counts, "by_set": dict(sorted(by_set.items())),
           "hard": hrep, "old_snapshot": PARITY_SNAPSHOT, "old_rankings": old_lt["version"],
           "old_rankings_copied_unchanged": same_old,
           "rankings": json.loads((d / "ship" / "rankings" / "latest.json").read_text(encoding="utf-8"))["version"],
           "rankings_hard": json.loads((d / "rankings_hard" / "latest.json").read_text(encoding="utf-8"))["version"],
           "ranking_inputs": ranks.meta["inputs"], "n_base_chunks": data.n_base_chunks, "n_chunks": data.n_chunks,
           "rank_rows": rr, "struct_gold_ids": struct_ids, "struct_gold_rows_ge_n_base": struct_rows,
           "gold_in_top50_hard": gold_in_top(ranks, hrows, data.chunk_ids, k=50),
           "manifest_sha256": hashlib.sha256((d / "ship" / "manifest.json").read_bytes()).hexdigest(),
           "ship_files": len(man["sha256"]), "workers": a.workers, "test_eval": False}
    if a.parity:
        rep["parity"] = parity(d, a.parity)
    (d / "snapshot.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("ranking_inputs", "parity")} |
                     {"parity_ok": all(v["lex_equal"] and v["vec_equal"] and v["title_equal"]
                                       for v in rep.get("parity", {}).get("per_query", {}).values())},
                     ensure_ascii=False, indent=1))
    return 0


# --- Task 5 (part 2): Q1 / Q2 paraphrases + the ds_hard judge -------------------------------------------------------

PARA_DIR = "data/hard/full"
PARA_IN = (f"{PARA_DIR}/count_raw.jsonl", f"{PARA_DIR}/version_raw.jsonl")
PARA_PACKETS = f"{PARA_DIR}/para_packets"
PARA_GOLD = f"{PARA_DIR}/para_gold.json"
RETRY_PREFIX = "retry1"
RETRY_ROUNDS = (1, 2, 3)          # 1: failed round 0; 2, 3: forced (R51), the latest forced round decides
FORCED_ROUNDS = (2, 3)
HARD_API_ENV = "GRL_HARD_API"
Q1X_PREFIX = "para6c"             # Task 6c: the Q1 expansion packets (data/hard/full/para6c_packets)
BOOKKEEPING = {"duplicate", "not_retry", "unknown_qid", "malformed"}   # rejections of a line, not of an item


def _para_items(paths) -> list[dict]:
    rows = [r for p in paths for r in _jsonl(config.path(p))]
    if len({r["qid"] for r in rows}) != len(rows):
        raise SystemExit("duplicate qids in the inputs")
    return rows


def _gold_texts(items) -> tuple[dict, str]:
    """qid -> {texts: {fragment id: text}, exempt: {STRUCT id: its first line}} (R14b) from the pinned STRUCT texts and
    the records' chunk texts; a missing gold text is a hard error."""
    from graphrag.data.ingest import read_records
    from grl.env.struct_data import build_struct_inputs
    texts, _, _, version = build_struct_inputs()
    if any(g.startswith("STRUCT/") and g not in texts for it in items for g in it["gold"]):
        texts = {**texts, **build_struct_inputs(articles=True)[0]}        # Task 6c: STRUCT/<article> gold
    chunk_texts = {c.chunk_id: c.text for r in read_records(config.path(config.load("data")["records_dir"]))
                   for c in r.chunks}
    out = {}
    for it in items:
        t, ex = {}, {}
        for g in it["gold"]:
            src = texts if g.startswith("STRUCT/") else chunk_texts
            if g not in src:
                raise SystemExit(f"{it['qid']}: gold fragment {g} has no text")
            t[g] = src[g]
            if g.startswith("STRUCT/"):
                ex[g] = src[g].split("\n", 1)[0]
        out[it["qid"]] = {"texts": t, "exempt": ex}
    return out, version


def cmd_para_packets(a) -> int:
    """Paraphrase packets (one language each, ~--n items) + index.json, the frozen gold texts for the ingest
    (para_gold.json) and the paraphraser instructions. Refuses to overwrite (R25)."""
    from grl.hard.checks import leak_ok
    from grl.hard.paraphrase import anchor_problems, anchors, detect_lang, instructions, para_packets
    items = _para_items(a.inp)
    out = config.path(a.out)
    if out.exists() and any(out.glob(f"{a.prefix}-*.json")):
        raise SystemExit(f"{out} already holds packets: refusing to overwrite")
    gold, version = _gold_texts(items)
    for it in items:                       # every template passes its own checks (the anchors are well defined)
        g = gold[it["qid"]]
        bad = anchor_problems(it["question"], it, anchors(it))
        if bad or detect_lang(it["question"]) != it["lang"] or \
                not leak_ok(it["question"], g["texts"], ids=it["gold"], exempt=g["exempt"]):
            raise SystemExit(f"{it['qid']}: the template fails its own checks {bad}")
    packets = para_packets(items, a.n, a.seed, prefix=a.prefix)
    out.mkdir(parents=True, exist_ok=True)
    for p in packets:
        _write_new(out / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    index = {"graph_version": version, "seed": a.seed, "n_per_packet": a.n, "inputs": list(a.inp),
             "inputs_sha256": {p: _sha256(config.path(p)) for p in a.inp}, "n_items": len(items),
             "by_lang": _dist(items, lambda r: r["lang"]), "by_qtype": _dist(items, lambda r: r["qtype"]),
             "packets": [{"packet_id": p["packet_id"], "lang": p["lang"], "n": p["n"],
                          "qtypes": _dist(p["items"], lambda r: r["qtype"])} for p in packets]}
    _write_new(out / "index.json", json.dumps(index, ensure_ascii=False, indent=1) + "\n")
    _write_new(config.path(a.gold), json.dumps(gold, ensure_ascii=False, sort_keys=True) + "\n")
    _write_new(config.path(a.instructions),
               instructions(str(out), str(config.path(PARA_DIR)) + f"/{a.raw_stem}_<k>.jsonl", prefix=a.prefix,
                            notes=a.prefix == Q1X_PREFIX))
    for p in index["packets"]:
        print(f"  {p['packet_id']}: {p['lang']} n={p['n']} {p['qtypes']}")
    print(f"{len(items)} items -> {len(packets)} packets in {out}; gold {a.gold}; instructions {a.instructions}")
    return 0


def _round_lines(paths, rnd: int) -> list:
    import glob as _glob
    files = sorted({h for x in paths for h in _glob.glob(str(config.path(x)))})
    out = []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    r = line
                out.append({**r, "round": rnd, "file": Path(f).name} if isinstance(r, dict) else r)
    return out, [Path(f).name for f in files]


def _retry_packets(packets_dir: Path, k: int = 1) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(packets_dir.glob(f"retry{k}-*.json"))]


def cmd_para_ingest(a) -> int:
    """Round-0 lines (para_raw_<k>.jsonl) and round-1 lines (para_raw_r1_<k>.jsonl, qids of the retry packets) ->
    para_ok.jsonl, para_rejections.jsonl, para_ingest.json."""
    from grl.hard.paraphrase import para_ingest
    items = _para_items(a.inp)
    gold = json.loads(config.path(a.gold).read_text(encoding="utf-8"))
    r0, f0 = _round_lines(a.raw, 0)
    if not r0:
        raise SystemExit(f"no round-0 lines in {a.raw}")
    lines, files = {0: r0}, {0: f0}
    for k in RETRY_ROUNDS:
        lines[k], files[k] = _round_lines([f"{a.retry_dir}/para_raw_r{k}_*.jsonl"] if k > 1 or a.raw_retry is None
                                          else a.raw_retry, k)
    retry = {k: {x["qid"] for p in _retry_packets(config.path(a.packets), k) for x in p["items"]} for k in RETRY_ROUNDS}
    for k in RETRY_ROUNDS:
        if lines[k] and not retry[k]:
            raise SystemExit(f"round-{k} lines but no retry{k} packets")
    acc, rej = para_ingest([x for k in sorted(lines) for x in lines[k]], items, gold=gold, retry_qids=retry)
    out = config.path(a.out)
    st = {**{f"round{k}_files": files[k] for k in sorted(files)},
          "lines": {str(k): len(v) for k, v in sorted(lines.items())},
          "retry_qids": {str(k): len(v) for k, v in retry.items()}, "accepted": _dist(acc, lambda r: str(r["round"])),
          "rejected": _dist(rej, lambda r: str(r["round"])),
          "reasons": dict(sorted(Counter(w for r in rej for w in r["reasons"]).items())),
          "reasons_by_round": {str(k): dict(sorted(Counter(w for r in rej if r["round"] == k for w in r["reasons"])
                                                   .items())) for k in (0,) + RETRY_ROUNDS},
          "accepted_by_lang": _dist(acc, lambda r: r["lang"]), "accepted_by_template": _dist(acc, lambda r: r["template"]),
          "round0_missing": sorted({it["qid"] for it in items} - {r.get("qid") for r in r0 if isinstance(r, dict)})}
    atomic_write({out / "para_ok.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in acc),
                  out / "para_rejections.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rej),
                  out / "para_ingest.json": json.dumps(st, ensure_ascii=False, indent=1) + "\n"})
    print(json.dumps({k: v for k, v in st.items() if k != "round0_missing"}, ensure_ascii=False))
    print(f"round 0 items without a line: {len(st['round0_missing'])}")
    return 0


def _load_openai_key() -> None:
    """The OpenAI key from the main checkout's repo-root .env into this process only (never printed, never argv)."""
    import os
    if os.environ.get("OPENAI_API_KEY"):
        return
    from dotenv import dotenv_values
    from scripts.spark import repo_env
    v = dotenv_values(repo_env()).get("OPENAI_API_KEY")
    if not v:
        raise SystemExit("repo-root .env has no OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = v


def hard_llm(client=None):
    import os
    from graphrag.kg.llm import KgLLM
    tcfg = config.load("train")
    allow = os.environ.get(HARD_API_ENV) == "1"
    if allow:
        _load_openai_key()
    return KgLLM(cfg={"llm": tcfg["hard_llm"]}, cache_root=tcfg["hard_llm"]["cache_dir"], client=client,
                 allow_api=allow)


def _graph_titles() -> list[str]:
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    nodes = _jsonl(config.path(latest["dir"]) / "nodes.jsonl")
    return sorted({n["title"].strip() for n in nodes if n["kind"] == "document" and (n.get("title") or "").strip()
                   and (n["node_id"].startswith("EXT/") or (n.get("attrs") or {}).get("doc_type") == "law_text")})


def cmd_para_judge(a) -> int:
    """Round 0: pairs + plants -> the gate (STOP, exit 2, if it fails). Round 1: retry pairs, no plants. Items failing
    round 0 (ingest or judge) without a retry packet -> retry packets + instructions, exit 3 (the controller dispatches
    them). Final (nothing pending): para_judged.jsonl (question = the accepted paraphrase, meta.template_question,
    meta.para) and para_gate.json."""
    from grl.hard.judge import PLANT_KINDS, gate, judge_paraphrase, make_pair
    from grl.hard.paraphrase import anchors, asks_times, instructions, para_packets
    items = _para_items(a.inp)
    kinds = tuple(a.plant_kinds) if a.plant_kinds else PLANT_KINDS
    topup = []
    if a.topup:                                   # Task 6c: bases of an earlier judged pass (template <-> accepted)
        for r in _jsonl(config.path(a.topup)):
            t = {**r, "question": r["meta"]["template_question"]}
            topup.append(make_pair(t, r["question"], anchors(t)))
    by_qid = {it["qid"]: it for it in items}
    out = config.path(a.out)
    ok = _jsonl(out / "para_ok.jsonl")
    rej = _jsonl(out / "para_rejections.jsonl")
    ing = json.loads((out / "para_ingest.json").read_text(encoding="utf-8"))
    llm = hard_llm()
    titles = _graph_titles()
    pairs = {k: [make_pair(by_qid[r["qid"]], r["question"], anchors(by_qid[r["qid"]]))
                 for r in sorted(ok, key=lambda r: r["qid"]) if r["round"] == k] for k in (0,) + RETRY_ROUNDS}
    from graphrag.kg.llm import KgCacheMiss
    try:
        res0 = judge_paraphrase(pairs[0], llm, plant_share=a.plant_share, seed=a.seed, titles=titles, kinds=kinds,
                                topup=topup)
        g = gate(res0["plants"], kinds=kinds)
        res = {0: res0}
        for k in RETRY_ROUNDS:                  # retry rounds: no plants, each its own (cached) call plan
            res[k] = judge_paraphrase(pairs[k], llm, seed=a.seed, plants=False) if pairs[k] and g["ok"] else \
                {"verdicts": {}, "plants": [], "calls": 0}
        res1 = res[1]
    except KgCacheMiss as e:
        raise SystemExit(f"para-judge: a judge answer is not in {llm.cache_root} and the API is off -- set "
                         f"{HARD_API_ENV}=1 to call gpt-4.1 (cap {llm.max_total}, ledger {llm.cache_root}/ledger.json)"
                         f" [{e}]") from None
    led = llm.ledger()
    spend = {"ledger_total": sum(int(v) for p in led.values() for v in p.values()), "ledger": led,
             "cap": llm.max_total, "this_run": llm.stats_json(), "calls_round0": res0["calls"],
             **{f"calls_round{k}": res[k]["calls"] for k in RETRY_ROUNDS}}
    rounds = tuple(sorted(res.items()))
    verd = [{"round": k, "qid": q, **v} for k, res in rounds for q, v in sorted(res["verdicts"].items())]
    atomic_write({out / "para_verdicts.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verd),
                  out / "para_plants.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res0["plants"])})
    summary = {"seed": a.seed, "plant_share": a.plant_share, "plant_kinds": list(kinds), "topup": a.topup,
               "topup_plants": sum(1 for r in res0["plants"] if r.get("topup")), "gate": g, "spend": spend,
               "judged": {str(k): len(pairs[k]) for k in sorted(pairs)},
               "same": {str(k): sum(1 for v in res["verdicts"].values() if v["same"]) for k, res in rounds}}
    if not g["ok"]:
        summary["final"] = False
        (out / "para_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("STOP: ds_hard judge gate FAILED (threshold not lowered):", "; ".join(g["reasons"]), file=sys.stderr)
        return 2
    print("gate ok:", {k: (v["judged"], v["sensitivity"]) for k, v in g["by_kind"].items()})
    problem = {}
    ok0 = {r["qid"] for r in ok if r["round"] == 0}
    rej = [r for r in rej if not set(r["reasons"]) & BOOKKEEPING]     # I1: a duplicate / not_retry line says nothing
    for r in rej:                                                      # about the item's own (first) line
        if r["round"] == 0 and r["qid"] in by_qid and r["qid"] not in ok0:
            problem.setdefault(r["qid"], {"question": (r["raw"] or {}).get("question"),
                                          "problem": "code check: " + ", ".join(r["reasons"])})
    for q, v in res0["verdicts"].items():
        if not v["same"]:
            prev = next(p["paraphrase"] for p in pairs[0] if p["qid"] == q)
            problem[q] = {"question": prev, "problem": "judge: " + (v["why"] or "not the same question")}
    for q in by_qid:
        if q not in res0["verdicts"] and q not in problem:
            problem[q] = {"question": None, "problem": "no paraphrase written"}
    accepted = {q: (0, next(p["paraphrase"] for p in pairs[0] if p["qid"] == q), v["why"])
                for q, v in res0["verdicts"].items() if v["same"]}
    pdir = config.path(a.packets)
    retry_packets = _retry_packets(pdir)
    if problem and not retry_packets:
        fails = [by_qid[q] for q in sorted(problem)]
        packets = para_packets(fails, a.n, a.seed, prefix=RETRY_PREFIX, previous=problem)
        for p in packets:
            _write_new(pdir / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
        _write_new(pdir / f"index_{RETRY_PREFIX}.json", json.dumps(
            {"seed": a.seed, "n_items": len(fails), "reasons": _dist(list(problem.values()),
                                                                      lambda v: v["problem"].split(":")[0]),
             "packets": [{"packet_id": p["packet_id"], "lang": p["lang"], "n": p["n"]} for p in packets]},
            ensure_ascii=False, indent=1) + "\n")
        _write_new(config.path(a.retry_instructions),
                   instructions(str(pdir), str(out) + "/para_raw_r1_<k>.jsonl", prefix=RETRY_PREFIX,
                                index=f"index_{RETRY_PREFIX}.json", retry=True))
        summary["final"] = False
        summary["retry"] = {"n": len(fails), "packets": [p["packet_id"] for p in packets]}
        (out / "para_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"STOP for the controller: {len(fails)} items need a second paraphrase -> {len(packets)} retry packets "
              f"in {pdir} ({a.retry_instructions}); round 0: {len(accepted)} accepted. Calls: {spend['ledger_total']}"
              f" / {llm.max_total}")
        return 3
    retry_q = {x["qid"] for p in retry_packets for x in p["items"]}
    tried1 = {r["qid"] for r in ok if r["round"] == 1} | {r["qid"] for r in rej if r["round"] == 1}
    dropped, pending = {}, []
    for q in sorted(retry_q):
        v = res1["verdicts"].get(q)
        if v and v["same"]:
            accepted[q] = (1, next(p["paraphrase"] for p in pairs[1] if p["qid"] == q), v["why"])
        elif q in tried1:
            dropped[q] = "judge" if v else "code check: " + ", ".join(
                next(r["reasons"] for r in rej if r["round"] == 1 and r["qid"] == q))
        elif ing.get("round1_files"):
            dropped[q] = "no retry line"
        else:
            pending.append(q)
    uncovered = sorted(set(problem) - retry_q)            # a round-0 failure with no retry1 item: never silent
    pending += uncovered
    summary["retry"] = {"n": len(retry_q), "accepted": sum(1 for v in accepted.values() if v[0] == 1),
                        "dropped": dict(dropped), "pending": pending, "uncovered": uncovered}
    # R51: forced rounds 2, 3 -- a retryK item's earlier paraphrases are superseded; its latest forced round decides
    for k in FORCED_ROUNDS:
        rk = {x["qid"] for p in _retry_packets(pdir, k) for x in p["items"]}
        tried = {r["qid"] for r in ok if r["round"] == k} | {r["qid"] for r in rej if r["round"] == k}
        pend, drop = [], {}
        for q in sorted(rk):
            accepted.pop(q, None)
            dropped.pop(q, None)
            pending = [x for x in pending if x != q]
            v = res[k]["verdicts"].get(q)
            if v and v["same"]:
                accepted[q] = (k, next(p["paraphrase"] for p in pairs[k] if p["qid"] == q), v["why"])
            elif q in tried:
                drop[q] = "judge" if v else "code check: " + ", ".join(
                    next(r["reasons"] for r in rej if r["round"] == k and r["qid"] == q))
            elif ing.get(f"round{k}_files"):
                drop[q] = "no retry line"
            else:
                pend.append(q)
        dropped.update(drop)
        summary[f"retry{k}"] = {"n": len(rk), "rule": "R51",
                                "accepted": sum(1 for v in accepted.values() if v[0] == k),
                                "dropped": drop, "pending": pend}
        pending += pend
    if pending:
        summary["final"] = False
        (out / "para_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"STOP: {len(pending)} retry items are pending (no retry line yet, or a round-0 failure without a retry "
              f"item: {uncovered[:5]})", file=sys.stderr)
        return 3
    times = sorted(q for q, (_, t, _) in accepted.items() if by_qid[q]["qtype"] == "count" and asks_times(t))
    if times:
        raise SystemExit(f"R51: {len(times)} accepted count paraphrases ask «how many times» ({times[:3]}): run "
                         f"para-force-retry first")
    rows = []
    for it in items:
        if it["qid"] in accepted:
            rnd, q, why = accepted[it["qid"]]
            rows.append({**it, "question": q, "meta": {**it["meta"], "template_question": it["question"],
                                                       "para": {"round": rnd, "judge_why": why}}})
    summary["final"] = True
    summary["accepted"] = {"n": len(rows), "round0": sum(1 for v in accepted.values() if v[0] == 0),
                           "round1": sum(1 for v in accepted.values() if v[0] == 1),
                           **{f"round{k}": sum(1 for v in accepted.values() if v[0] == k) for k in FORCED_ROUNDS},
                           "dropped": len(dropped),
                           "by_qtype": _dist(rows, lambda r: r["qtype"]), "by_lang": _dist(rows, lambda r: r["lang"]),
                           "by_split": _dist(rows, lambda r: r["split"])}
    atomic_write({out / "para_judged.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                  out / "para_gate.json": json.dumps(summary, ensure_ascii=False, indent=1) + "\n"})
    print(f"final: {len(rows)} items accepted ({summary['accepted']}); calls {spend['ledger_total']} / {llm.max_total}")
    return 0



R51_PROBLEM = "question must ask for the number of changed elements (статьи/части/пункты), not 'how many times'"


def cmd_para_force_retry(a) -> int:
    """R51 (forced rounds 2, 3): accepted count paraphrases asking «how many times» (every TIMES pattern) -> retry<k>
    packets (k = --round, default the next forced round without packets; previous = the accepted paraphrase +
    R51_PROBLEM) + instructions; the paraphrasers write para_raw_r<k>_<n>.jsonl. Refuses to overwrite."""
    from grl.hard.paraphrase import asks_times, instructions, para_packets
    items = {it["qid"]: it for it in _para_items(a.inp)}
    out, pdir = config.path(a.out), config.path(a.packets)
    k = a.round or next((r for r in FORCED_ROUNDS if not _retry_packets(pdir, r)), None)
    if k is None or k not in FORCED_ROUNDS:
        raise SystemExit(f"no free forced round in {FORCED_ROUNDS} (or --round {a.round} is not one)")
    if _retry_packets(pdir, k):
        raise SystemExit(f"{pdir} already holds retry{k} packets: refusing to overwrite")
    rows = _jsonl(out / "para_judged.jsonl")
    sel = [r for r in rows if r["qtype"] == "count" and asks_times(r["question"])]
    if not sel:
        print("no accepted count paraphrase asks «how many times»")
        return 0
    prev = {r["qid"]: {"question": r["question"], "problem": R51_PROBLEM} for r in sel}
    packets = para_packets([items[r["qid"]] for r in sel], a.n, a.seed, prefix=f"retry{k}", previous=prev)
    for p in packets:
        _write_new(pdir / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    _write_new(pdir / f"index_retry{k}.json", json.dumps(
        {"rule": "R51", "round": k, "seed": a.seed, "n_items": len(sel), "problem": R51_PROBLEM,
         "qids": [r["qid"] for r in sel],
         "by_template": _dist(sel, lambda r: r["template"]), "by_lang": _dist(sel, lambda r: r["lang"]),
         "packets": [{"packet_id": p["packet_id"], "lang": p["lang"], "n": p["n"]} for p in packets]},
        ensure_ascii=False, indent=1) + "\n")
    instr = a.retry_instructions or f"{PARA_DIR}/para_instructions_retry{k}.md"
    _write_new(config.path(instr), instructions(str(pdir), str(out) + f"/para_raw_r{k}_<n>.jsonl",
                                               prefix=f"retry{k}", index=f"index_retry{k}.json", retry=True))
    for p in packets:
        print(f"  {pdir / (p['packet_id'] + '.json')}: {p['lang']} n={p['n']}")
    print(f"R51: {len(sel)} items -> {len(packets)} retry{k} packets; instructions {instr}")
    return 0


# --- plan M7 part 2, Task 6: the full Q3 set --------------------------------------------------------------------------

CMP_DIR = "data/hard/full"
CMP_PACKETS = f"{CMP_DIR}/compare_packets"
CMP_REVIEW = f"{CMP_DIR}/compare_review"
PROBE_VALID = "data/hard/probe/compare_valid_pairs.json"
PROBE_PACKETS = "data/hard/probe/compare_packets"
PROBE_INSTRUCTIONS = "data/hard/probe/compare_instructions.md"


def load_probe_pairs(valid_path, packets_dir) -> set:
    """The Q3 probe's document pairs (frozensets, any kind / order): every packet recorded in the validity list
    (existing.packets) plus every probe packet file (the top-ups after the list). Not the list's `pairs` -- those are
    all valid pairs, not the probe's. A missing file / directory or an empty result is a loud error."""
    from grl.hard.compare import packet_key
    vp, pd = Path(valid_path), Path(packets_dir)
    if not vp.is_file():
        raise SystemExit(f"--exclude-pairs {vp}: file missing -- the probe pairs are required (Task 6)")
    try:
        recorded = json.loads(vp.read_text(encoding="utf-8"))["existing"]["packets"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(f"--exclude-pairs {vp}: not a probe validity list ({e})")
    files = sorted(pd.glob("cmp-*.json")) if pd.is_dir() else []
    if not files:
        raise SystemExit(f"--exclude-packets {pd}: no probe packets")
    out = {frozenset(v["pair"][:2]) for v in recorded.values()}
    out |= {frozenset(packet_key(json.loads(f.read_text(encoding="utf-8")))[:2]) for f in files}
    if not out:
        raise SystemExit("no probe pairs")
    return out


def full_compare_instructions(probe_text: str, packets_dir: str, raw_pattern: str) -> str:
    """The probe generator instructions v2 adapted to the full set (v3): files, the R26 note, judge + readers."""
    for anchor in ("- Packets: ", "- Output: ", "  one JSON object per line"):
        if anchor not in probe_text:
            raise SystemExit(f"compare instructions: anchor {anchor!r} missing")
    edits = [
        ("# GENERATOR INSTRUCTIONS: M7 Q3 comparative questions, probe set (version 2, plan M7 part 1, Task 6; rulings "
         "R23 / R24 / R25)",
         "# GENERATOR INSTRUCTIONS: M7 Q3 comparative questions, full set (version 3, plan M7 part 2, Task 6; rulings "
         "R23 / R24 / R25 / R26)"),
        (probe_text[probe_text.index("- Packets: "):probe_text.index("- Output: ")],
         f"- Packets: `{packets_dir}/cmp-NNN.json`, listed in `index.json` (and any `index_topup_NNN.json`) in the\n"
         f"  same directory. Do only the packets the controller assigns you (batches of 10).\n"),
        (probe_text[probe_text.index("- Output: "):probe_text.index("  one JSON object per line")],
         f"- Output: `{raw_pattern}`, the file the controller names (one file per generator). Write\n"),
        ("- the conclusion or resolution only states that the law was considered or approved, and adds no content that "
         "relates",
         "- (R26) law_resolution packets were pre-filtered: every resolution has at least one point besides «approve» "
         "(e.g. a\n  task for the Cabinet of Ministers with a deadline, a committee's control, a rejection with a "
         "conciliation\n  commission). Build the comparison on such a point, never on the approval itself;\n"
         "- the conclusion or resolution only states that the law was considered or approved, and adds no content that "
         "relates"),
        ("There is no judge or reader in this probe, so these checks are all that stand between your line and the "
         "benchmark.",
         "After these checks, a gpt-4.1 judge (does the question need both documents; is the answer supported by the "
         "gold\nalone) and three blind readers check every line; one failed reader verdict rejects it."),
    ]
    out = probe_text
    for old, new in edits:
        if out.count(old) != 1:
            raise SystemExit(f"compare instructions: anchor not found once: {old[:60]!r}")
        out = out.replace(old, new)
    return out


def cmd_full_pairs(a) -> int:
    """Task 6: the full Q3 packets -- pick_doc_pairs (R23 / R24, <= --max-per-doc pairs per document, <=
    --max-per-doc-kind per kind) with every Q3 probe pair excluded and the R26 approval-only filter; 12k-token packets;
    languages from the frozen <full>/lang_shares.json (R47). Writes <out>/cmp-NNN.json + index.json and the generator
    instructions v3; never overwrites."""
    from grl.hard.compare import (KINDS, PACKET_TOKENS, approval_only, build_packets, packet_key, pair_candidates,
                                  pick_doc_pairs)
    out = config.path(a.out)
    old = _existing_packets(out) if out.exists() else []
    if old and not a.topup:
        raise SystemExit(f"{out} already holds {len(old)} packets: refusing to overwrite (--topup extends them)")
    if a.topup and not old:
        raise SystemExit(f"--topup: {out} holds no packets")
    probe = load_probe_pairs(config.path(a.exclude_pairs), config.path(a.exclude_packets))
    version, nodes, edges, mu, doc_chunks, texts, _, meta = load_pair_inputs()
    shares, shares_src = frozen_lang_shares(config.path(CMP_DIR))
    n_chunks = {d: len(cs) for d, cs in doc_chunks.items()}
    dates = {d: m.get("date") for d, m in meta.items()}
    res_texts = {d: [(c, texts[c]) for c, _ in cs] for d, cs in doc_chunks.items()}
    from grl.hard.splits import fixed_doc_splits
    fixed = fixed_doc_splits(json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8")),
                             [r for x in PARA_IN for r in _jsonl(config.path(x))])
    st = Counter()
    kept = [packet_key(p) for p in old]                 # --topup: the existing packets count toward the caps
    pairs = pick_doc_pairs(nodes, edges, a.n - len(old), a.seed, n_chunks=n_chunks, doc_dates=dates, stats=st,
                           mu_inv=mu, preset=kept, exclude=kept, exclude_pairs=probe, res_texts=res_texts,
                           max_per_doc=a.max_per_doc, max_per_doc_kind=a.max_per_doc_kind, fixed_splits=fixed)
    if any(frozenset(p[:2]) in probe for p in pairs):
        raise SystemExit("a probe pair in the output")
    # R26 audit: every resolution of a law_resolution candidate before the filter, and the verdict
    st0 = Counter()
    raw = pair_candidates(nodes, edges, a.seed, n_chunks=n_chunks, doc_dates=dates, stats=st0, mu_inv=mu)
    r26 = {y: approval_only(res_texts.get(y, ())) for _, y, _ in raw["law_resolution"]}
    start = max(int(p["packet_id"].split("-")[1]) for p in old) + 1 if old else 0
    packets = build_packets(pairs, nodes, edges, doc_chunks, mu, texts, meta, shares, budget=PACKET_TOKENS,
                            start=start, prior=[(p["kind"], p["lang"]) for p in old])
    probe_ids = set(json.loads(config.path("data/hard/probe/ids.json").read_text(encoding="utf-8")))
    qids = [p["qid"] for p in old + packets]
    if set(qids) & probe_ids or len(set(qids)) != len(qids):
        raise SystemExit("packet qids collide with the probe ids or each other")
    out.mkdir(parents=True, exist_ok=True)
    for p in packets:
        _write_new(out / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    summary = _packet_summary(packets, st, {
        "graph_version": version, "seed": a.seed, "n": a.n, "mode": "topup" if old else "first",
        "existing": len(old), "new": len(packets), "shortfall": max(0, a.n - len(old) - len(packets)),
        "lang_shares": shares, "lang_shares_source": shares_src,
        "caps": {"max_per_doc": a.max_per_doc, "max_per_doc_kind": a.max_per_doc_kind},
        "exclude": {"pairs": a.exclude_pairs, "packets": a.exclude_packets, "n_probe_pairs": len(probe),
                    "excluded_candidates": {k: st.get(f"probe_excluded:{k}", 0) for k in KINDS}},
        "r26": {"rule": "law_resolution only if the resolution has a point besides approve / in force / signature "
                        "(grl.hard.compare.substantive_points)",
                "resolutions": len(r26), "approval_only": sum(r26.values()),
                "dropped_pairs": st.get("approval_only", 0),
                "approval_only_docs": sorted(d for d, v in r26.items() if v)},
        "supply_before_r26_and_probe": {k: len(raw[k]) for k in KINDS},
        "r32_prescreen": {k: st.get(f"r32_prescreen:{k}", 0) for k in KINDS}})
    name = f"index_topup_{start:03d}.json" if old else "index.json"
    _write_new(out / name, json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    if not old:
        probe_text = config.path(PROBE_INSTRUCTIONS).read_text(encoding="utf-8")
        _write_new(config.path(a.instructions),
                   full_compare_instructions(probe_text, str(out), str(config.path(CMP_DIR)) + "/compare_raw_<k>.jsonl"))
    print(f"graph {version}: {len(pairs)} new pairs (+{len(old)} existing) / {a.n} asked (shortfall "
          f"{summary['shortfall']}); probe pairs "
          f"excluded {summary['exclude']['excluded_candidates']}; R26 {summary['r26']['approval_only']} / "
          f"{summary['r26']['resolutions']} resolutions approval-only, {summary['r26']['dropped_pairs']} pairs dropped")
    for k in ("supply_before_r26_and_probe", "supply", "picked", "langs", "langs_by_kind", "tokens", "chunks_shown",
              "truncated_docs", "docs_without_linked_chunk"):
        print(f"  {k}:", summary[k])
    print("  cap skips:", {k: st.get(f"cap_skipped:{k}", 0) for k in KINDS})
    print("wrote", out / name, "" if old else f"and {a.instructions}")
    return 0


def _cmp_packets(d) -> dict:
    out = {}
    for f in sorted(config.path(d).glob("cmp-*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        out[p["packet_id"]] = p
    if not out:
        raise SystemExit(f"{d}: no packets")
    return out


def cmd_full_ingest_compare(a) -> int:
    """Task 6: code checks of the generator lines (ingest_compare; every packet's pair is valid by construction) ->
    <out>/compare_ingested.jsonl (HardItem + gold_docs, packet order), compare_rejections.jsonl, compare_ingest.json.
    The probe's ids.json / stats.json are not touched."""
    from grl.hard.compare import ingest_compare, raw_packet_id
    from grl.hard.splits import gold_docs
    from graphrag.data.ingest import read_records
    packets = _cmp_packets(a.packets)
    raws = _raw_lines(a.raw)
    recs = read_records(config.path(config.load("data")["records_dir"]))
    chunk_doc = {c.chunk_id: r.doc_id for r in recs for c in r.chunks}
    texts = {c.chunk_id: c.text for r in recs for c in r.chunks}
    items, st, rej, both = ingest_compare(raws, packets, chunk_doc, texts, valid=None)
    probe_ids = set(json.loads(config.path("data/hard/probe/ids.json").read_text(encoding="utf-8")))
    if {it.qid for it in items} & probe_ids:
        raise SystemExit("probe qids among the ingested items")
    rows = [{**it.to_json(), "gold_docs": gold_docs(it.gold, chunk_doc)} for it in items]
    out = config.path(a.out)
    summary = {"packets": len(packets), "packets_answered": st["packets_answered"], "refusal_and_item": both,
               "accepted": len(rows), "kinds": _dist(rows, lambda r: r["template"]),
               "langs": _dist(rows, lambda r: r["lang"]), "gold_sizes": _dist(rows, lambda r: len(r["gold"])),
               "refusals_by_kind": {k[len("refusals:"):]: v for k, v in st.items() if k.startswith("refusals:")},
               "stats": dict(sorted(st.items()))}
    atomic_write({
        out / f"compare_ingested{a.tag}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        out / f"compare_rejections{a.tag}.jsonl": "".join(
            json.dumps({"packet_id": raw_packet_id(r), "reasons": w, "raw": r}, ensure_ascii=False) + "\n"
            for r, w in rej),
        out / f"compare_ingest{a.tag}.json": json.dumps(summary, ensure_ascii=False, indent=1) + "\n"})
    for k, v in summary.items():
        print(f"  {k}:", v)
    print("wrote", out / f"compare_ingested{a.tag}.jsonl")
    return 0


def q3_splits(items, seed: int = 20261003) -> tuple[list[dict], dict]:
    """R32 for Q3 items against the old splits AND the Q1 / Q2 splits (count_raw / version_raw): items with `split`
    (R32 conflicts dropped) + a summary {by_split, by_kind_split, r32_dropped: [{qid, docs, fixed splits}]}."""
    from grl.hard.splits import SPLITS, assign_q3_splits, fixed_doc_splits
    old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
    q12 = [r for p in PARA_IN for r in _jsonl(config.path(p))]
    split = assign_q3_splits(items, old, q12, seed)
    fixed = fixed_doc_splits(old, q12)
    kept = [{**it, "split": split[it["qid"]]} for it in items if split[it["qid"]] is not None]
    drops = [{"qid": it["qid"], "docs": it["docs"],
              "fixed": {d: fixed[d] for d in it["docs"] + it["gold_docs"] if d in fixed}}
             for it in items if split[it["qid"]] is None]
    inherited = sum(1 for it in kept if any(d in fixed for d in it["docs"] + it["gold_docs"]))
    return kept, {"by_split": {s: sum(r["split"] == s for r in kept) for s in SPLITS},
                  "by_kind_split": _dist(kept, lambda r: f"{r['template']}|{r['split']}"),
                  "inherited": inherited, "new_components_items": len(kept) - inherited, "r32_dropped": drops}


def doc_roots() -> dict:
    """R53 (c): doc -> its roots -- the acts its elements AMEND / REPEALS (the target node's doc_id, e.g. EXT/<code>)
    and its folded title ("title:<key>": translation / version twins)."""
    from grl.hard.templates import title_key
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    d = config.path(latest["dir"])
    nodes = {n["node_id"]: n for n in _jsonl(d / "nodes.jsonl")}
    out = defaultdict(set)
    for n in nodes.values():
        if n["kind"] == "document" and (n.get("title") or "").strip():
            out[n["node_id"]].add("title:" + " ".join(title_key(n["title"])))
    for e in _jsonl(d / "edges.jsonl"):
        if e["type"] in ("AMENDS", "REPEALS"):
            src, dst = nodes.get(e["src"]), nodes.get(e["dst"])
            if src and dst and src.get("doc_id") and dst.get("doc_id"):
                out[src["doc_id"]].add(dst["doc_id"])
    return dict(out)


def cmd_cmp_judge(a) -> int:
    """Task 6: the ds_hard compare judge with plants (doc_drop / answer_swap) and the Task 5 gate (>= 30 judged plants
    per kind, sensitivity >= 0.90; a failure is a STOP, exit 2). Writes compare_verdicts.jsonl, compare_plants.jsonl,
    compare_judged.jsonl (judge-accepted items) and compare_gate.json."""
    from graphrag.kg.llm import KgCacheMiss
    from grl.hard.judge import CMP_PLANT_KINDS, CMP_PROMPT_VERSION, gate, judge_compare
    out = config.path(a.out)
    items = _jsonl(out / f"compare_ingested{a.tag}.jsonl")
    packets = _cmp_packets(a.packets)
    llm = hard_llm()
    try:
        res = judge_compare(items, packets, llm, plant_share=a.plant_share, seed=a.seed, per_call=a.per_call,
                            roots=doc_roots(), plants=not a.no_plants)
    except KgCacheMiss as e:
        raise SystemExit(f"cmp-judge: a judge answer is not in {llm.cache_root} and the API is off -- set "
                         f"{HARD_API_ENV}=1 to call gpt-4.1 (cap {llm.max_total}) [{e}]") from None
    if a.no_plants:          # a repair round: the gate of round 0 (same prompt version and per-call) stands
        prev = json.loads((out / "compare_gate.json").read_text(encoding="utf-8"))
        if not prev["gate"]["ok"] or prev.get("prompt_version") != CMP_PROMPT_VERSION or prev.get("per_call") != a.per_call:
            raise SystemExit("--no-plants needs a passed round-0 gate with the same prompt version and per-call")
        g = {**prev["gate"], "from": "round 0 (compare_gate.json)"}
    else:
        g = gate(res["plants"], kinds=CMP_PLANT_KINDS)
    led = llm.ledger()
    spend = {"ledger_total": sum(int(v) for p in led.values() for v in p.values()), "cap": llm.max_total,
             "this_run": llm.stats_json(), "calls": res["calls"]}
    verd = [{"qid": q, **v} for q, v in sorted(res["verdicts"].items())]
    ok = [it for it in items if res["verdicts"][it["qid"]]["ok"]]
    summary = {"seed": a.seed, "plant_share": a.plant_share, "prompt_version": CMP_PROMPT_VERSION,
               "per_call": a.per_call, "donors": "R53: no shared document / root", "gate": g, "spend": spend,
               "judged": len(items),
               "accepted": len(ok), "not_needs_both": sum(v["needs_both"] is False for v in res["verdicts"].values()),
               "not_supported": sum(v["supported"] is False for v in res["verdicts"].values()),
               "no_verdict": sum(v["ok"] is None for v in res["verdicts"].values()),
               "accepted_by_kind": _dist(ok, lambda r: r["template"]), "accepted_by_lang": _dist(ok, lambda r: r["lang"])}
    files = {out / f"compare_verdicts{a.tag}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verd),
             out / f"compare_plants{a.tag}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                          for r in res["plants"]),
             out / f"compare_gate{a.tag}.json": json.dumps(summary, ensure_ascii=False, indent=1) + "\n"}
    if not g["ok"]:
        atomic_write(files)
        print("STOP: ds_hard compare judge gate FAILED (threshold not lowered):", "; ".join(g["reasons"]),
              file=sys.stderr)
        return 2
    ok, sp = q3_splits(ok)
    summary["splits"] = sp
    files[out / f"compare_gate{a.tag}.json"] = json.dumps(summary, ensure_ascii=False, indent=1) + "\n"
    files[out / f"compare_judged{a.tag}.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok)
    atomic_write(files)
    print("gate ok:", {k: (v["judged"], v["sensitivity"]) for k, v in g["by_kind"].items()})
    print("splits (R32 vs old + Q1/Q2):", sp)
    print(f"judge: {len(ok)} / {len(items)} accepted; calls {spend['ledger_total']} / {llm.max_total}")
    return 0


def cmd_reader_packets(a) -> int:
    """Task 6: review packets of 10 judge-accepted items (gold fragments only), the M7 Q3 rubric, the reader
    instructions and one codex prompt per packet. Never overwrites."""
    from grl.hard.readers import RUBRIC, RUBRIC_V2, codex_prompt, instructions, reader_packets
    out = config.path(a.out)
    items = _jsonl(out / f"compare_judged{a.tag}.jsonl")
    packets = _cmp_packets(a.packets)
    rdir = config.path(a.review)
    pdir, vdir, cdir = rdir / "packets", rdir / "verdicts", rdir / "codex"
    if pdir.exists() and any(pdir.glob("rev-*.json")):
        raise SystemExit(f"{pdir} already holds review packets: refusing to overwrite")
    rp = reader_packets(items, packets, n=a.n, seed=a.seed)
    for d in (pdir, vdir, cdir):
        d.mkdir(parents=True, exist_ok=True)
    rubric = rdir / "compare_reader_rubric.md"
    _write_new(rubric, RUBRIC_V2 if a.rubric_version == 2 else RUBRIC)
    for p in rp:
        _write_new(pdir / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
        _write_new(cdir / f"{p['packet_id']}.txt", codex_prompt(p["packet_id"], str(pdir), str(vdir), str(rubric)))
    _write_new(rdir / "compare_reader_instructions.md",
               instructions(str(pdir), str(vdir), str(cdir), str(rubric), [p["packet_id"] for p in rp]))
    _write_new(pdir / "index.json", json.dumps(
        {"seed": a.seed, "n_per_packet": a.n, "n_items": len(items),
         "packets": [{"packet_id": p["packet_id"], "n": len(p["items"]),
                      "kinds": _dist(p["items"], lambda x: x["kind"])} for p in rp]},
        ensure_ascii=False, indent=1) + "\n")
    print(f"{len(items)} items -> {len(rp)} review packets in {pdir}; rubric {rubric}; codex prompts {cdir}")
    return 0


def cmd_repair_packets(a) -> int:
    """R54: one repair round over every judged item -- repair packets (item, gold texts, both titles, every reader's
    verdict + reasons verbatim, the source packet path), the rubric v2 and the repair instructions. Never overwrites."""
    from grl.hard.readers import RUBRIC_V2, repair_instructions, repair_packets, reader_consensus
    out = config.path(a.out)
    items = _jsonl(out / "compare_judged.jsonl")
    packets = _cmp_packets(a.packets)
    vdir = config.path(a.review) / "verdicts"
    rows, st = reader_consensus(items, _verdict_lines(vdir, "A"), _verdict_lines(vdir, "B"),
                                _verdict_lines(vdir, "C"), reader_a="fable")
    by = {r["qid"]: r["readers"] for r in rows}
    rdir = config.path(a.repair)
    if rdir.exists() and any(rdir.glob("rep-*.json")):
        raise SystemExit(f"{rdir} already holds repair packets: refusing to overwrite")
    rdir.mkdir(parents=True, exist_ok=True)
    rp = repair_packets(items, packets, by, str(config.path(a.packets)), n=a.n)
    rubric = rdir / "compare_reader_rubric_v2.md"
    _write_new(rubric, RUBRIC_V2)
    for p in rp:
        _write_new(rdir / f"{p['packet_id']}.json", json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    _write_new(rdir / "compare_repair_instructions.md",
               repair_instructions(str(rdir), str(rubric), str(out) + "/compare_repair_raw_<k>.jsonl",
                                   [p["packet_id"] for p in rp]))
    _write_new(rdir / "index.json", json.dumps(
        {"rule": "R54", "n_items": len(items), "reader_stats": dict(sorted(st.items())),
         "packets": [{"packet_id": p["packet_id"], "n": len(p["items"]),
                      "kinds": _dist(p["items"], lambda x: x["kind"])} for p in rp]},
        ensure_ascii=False, indent=1) + "\n")
    print(f"{len(items)} items -> {len(rp)} repair packets in {rdir}; readers {dict(sorted(st.items()))}")
    return 0


def _verdict_lines(vdir: Path, who: str) -> list:
    out = []
    for f in sorted(vdir.glob(f"{who}-rev-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append(line)
    return out


def cmd_reader_ingest(a) -> int:
    """Task 6: A / B / C verdicts -> consensus (GOOD from A and B, C not BAD) -> R32 splits against the old set AND the
    Q1 / Q2 splits (count_raw / version_raw) -> compare_consensus.jsonl (accepted items + split + meta.readers),
    compare_reader_rows.jsonl (every item's verdicts) and compare_consensus.json (counts, R32 drops, pending)."""
    from grl.hard.readers import reader_consensus
    from grl.hard.splits import SPLITS, assign_q3_splits
    out = config.path(a.out)
    items = _jsonl(out / f"compare_judged{a.tag}.jsonl")
    vdir = config.path(a.review) / "verdicts"
    rows, st = reader_consensus(items, _verdict_lines(vdir, "A"), _verdict_lines(vdir, "B"),
                                _verdict_lines(vdir, "C"), reader_a=a.reader_a)
    status = {r["qid"]: r for r in rows}
    acc = [it for it in items if status[it["qid"]]["status"] == "accepted"]
    if all(it.get("split") in SPLITS for it in acc):        # fixed at cmp-judge (q3_splits): a subset stays consistent
        split = {it["qid"]: it["split"] for it in acc}
    else:
        old = json.loads(config.path(OLD_SPLITS).read_text(encoding="utf-8"))
        q12 = [r for p in PARA_IN for r in _jsonl(config.path(p))]
        split = assign_q3_splits(acc, old, q12, a.seed)
    dropped = [it["qid"] for it in acc if split[it["qid"]] is None]
    final = [{**it, "split": split[it["qid"]], "meta": {**it.get("meta", {}), "readers": status[it["qid"]]["readers"]}}
             for it in acc if split[it["qid"]] is not None]
    summary = {"judged_items": len(items), "accepted": len(final), "r32_dropped": dropped,
               "pending": sorted(q for q, r in status.items() if r["status"] == "pending"),
               "rejected": sum(r["status"] == "rejected" for r in rows),
               "reader_a": a.reader_a, "stats": dict(sorted(st.items())),
               "by_kind": _dist(final, lambda r: r["template"]), "by_lang": _dist(final, lambda r: r["lang"]),
               "by_split": {s: sum(r["split"] == s for r in final) for s in SPLITS},
               "target": a.target, "shortfall": max(0, a.target - len(final))}
    summary["acceptance_rule"] = ("R54: < 30 accepted -> Q3 train / val only, no separate test report (Q3 test as a "
                                  "diagnostic only if >= 10 test items)")
    summary["q3_test_reportable"] = len(final) >= 30
    summary["q3_test_diagnostic"] = summary["by_split"]["test"] >= 10
    atomic_write({out / f"compare_consensus{a.tag}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                                    for r in final),
                  out / f"compare_reader_rows{a.tag}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                                     for r in rows),
                  out / f"compare_consensus{a.tag}.json": json.dumps(summary, ensure_ascii=False, indent=1) + "\n"})
    print(f"consensus: {len(final)} accepted (target {a.target}), rejected {summary['rejected']}, pending "
          f"{len(summary['pending'])}, R32 dropped {len(dropped)}; by split {summary['by_split']}")
    if summary["pending"]:
        print("PENDING: items without both A and B verdicts:", summary["pending"][:10], file=sys.stderr)
        return 3
    return 0


def update_hashes(sha_file: Path, paths=(), root: Path = ROOT) -> dict:
    """`sha256  path` lines with each path exactly once: a listed path is re-hashed in place (first position kept), a
    new one appended, and a path that appeared more than once is re-hashed from the current file (dedupe)."""
    from grl.hashing import sha256_file
    order, cur, seen = [], {}, Counter()
    for line in (sha_file.read_text(encoding="utf-8").splitlines() if sha_file.exists() else []):
        if not line.strip():
            continue
        h, p = line.split(None, 1)
        p = p.strip().lstrip("*")
        seen[p] += 1
        if p not in cur:
            order.append(p)
        cur[p] = h
    redo = [p for p in order if seen[p] > 1] + [p for p in paths]
    for p in redo:
        cur[p] = sha256_file(root / p)
        if p not in order:
            order.append(p)
    sha_file.write_text("".join(f"{cur[p]}  {p}\n" for p in order), encoding="utf-8")
    return {"paths": len(order), "deduped": sorted(p for p in order if seen[p] > 1), "updated": list(paths)}


def cmd_hashes(a) -> int:
    rep = update_hashes(config.path(a.file), a.paths)
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--seed", type=int, default=20261002)
    m.add_argument("--n-count", type=int, default=30)
    m.add_argument("--n-version", type=int, default=30)
    m.add_argument("--out", default="data/hard/probe/")
    f = sub.add_parser("full-make", help="plan M7 part 2, Task 4: full Q1 / Q2 with R32 splits and R33 gold")
    f.add_argument("--seed", type=int, default=20261003)
    f.add_argument("--n-count", type=int, default=250)
    f.add_argument("--n-version", type=int, default=200)
    f.add_argument("--exclude", default="data/hard/probe/ids.json")
    f.add_argument("--out", default="data/hard/full/")
    qa = sub.add_parser("q2-augment", help="M7 pilot 8, R55: Q2 train rows, template wording, per-act cap 16")
    qa.add_argument("--seed", type=int, default=20261003)
    qa.add_argument("--n", type=int, default=400, help="selection slots (the supply binds far below)")
    qa.add_argument("--cap", type=int, default=16)
    qa.add_argument("--exclude", default="data/hard/probe/ids.json")
    qa.add_argument("--judged", default=FULL_HARD)
    qa.add_argument("--out", default="data/hard/full/")
    q = sub.add_parser("pairs")
    q.add_argument("--n", type=int, default=40)
    q.add_argument("--seed", type=int, default=20261002)
    q.add_argument("--out", default="data/hard/probe/compare_packets/")
    q.add_argument("--valid-only", action="store_true", help="R25 dry mode: write only the validity list")
    q.add_argument("--valid-out", default="data/hard/probe/compare_valid_pairs.json")
    q.add_argument("--topup", action="store_true", help="R25: new packets replacing the invalid ones")
    q.add_argument("--cap", type=int, default=60)
    q.add_argument("--n-new", type=int, default=None, help="--topup: this many new packets (R26)")
    q.add_argument("--exclude-kinds", nargs="*", default=None, help="--topup: kinds that get no new packet (R26)")
    g = sub.add_parser("ingest-compare")
    g.add_argument("--raw", nargs="+", default=["data/hard/probe/compare_raw*.jsonl"],
                   help="files or globs (R22: compare_raw_1..4.jsonl); lines are processed in packet order")
    g.add_argument("--valid", default="data/hard/probe/compare_valid_pairs.json")
    g.add_argument("--packets", default="data/hard/probe/compare_packets/")
    g.add_argument("--out", default="data/hard/probe/")
    g.add_argument("--max", type=int, default=30)
    s = sub.add_parser("snapshot", help="M7 Task 7: probe rankings (dev stores) + ship/ + checks")
    s.add_argument("--pairs", nargs="+", default=[f"data/hard/probe/{n}" for n in
                                                    ("count.jsonl", "version.jsonl", "compare.jsonl")])
    s.add_argument("--out-root", default=PROBE_OUT_ROOT)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--parity", type=int, default=3, help="old-set train queries rebuilt for the parity check (0: off)")
    fs = sub.add_parser("full-snapshot", help="M7 pilot 7: joint old + hard Q1 / Q2 struct ship (dev stores)")
    fs.add_argument("--hard", default=FULL_HARD)
    fs.add_argument("--articles", action="store_true", help="Task 6c: article STRUCT gold (article-mode EnvData)")
    fs.add_argument("--out-root", default=FULL_OUT_ROOT)
    fs.add_argument("--workers", type=int, default=2)
    fs.add_argument("--parity", type=int, default=3, help="old-set train queries rebuilt for the parity check (0: off)")
    fs.add_argument("--reuse-hard-rankings", default=None,
                    help="pilot 8: a hard rankings latest.json whose keys are reused; only the other hard rows are "
                         "ranked, and the three artifacts (old, reused, new) are merged")
    pp = sub.add_parser("para-packets", help="M7 part 2, Task 5: Q1 / Q2 paraphrase packets + instructions")
    pp.add_argument("--in", dest="inp", nargs="+", default=list(PARA_IN))
    pp.add_argument("--out", default=PARA_PACKETS)
    pp.add_argument("--n", type=int, default=40)
    pp.add_argument("--seed", type=int, default=20261003)
    pp.add_argument("--gold", default=PARA_GOLD)
    pp.add_argument("--instructions", default=f"{PARA_DIR}/para_instructions.md")
    pp.add_argument("--prefix", default="para", help=f"packet id prefix (Task 6c: {Q1X_PREFIX}, + its notes)")
    pp.add_argument("--raw-stem", default="para_raw", help="paraphraser output file stem named in the instructions")
    pi = sub.add_parser("para-ingest", help="Task 5: anchor-checked ingest of the paraphrases")
    pi.add_argument("--in", dest="inp", nargs="+", default=list(PARA_IN))
    pi.add_argument("--raw", nargs="+", default=[f"{PARA_DIR}/para_raw_[0-9]*.jsonl"])
    pi.add_argument("--raw-retry", nargs="+", default=None, help="round-1 files (default para_raw_r1_*.jsonl); "
                    "rounds 2, 3 read para_raw_r<k>_*.jsonl")
    pi.add_argument("--packets", default=PARA_PACKETS)
    pi.add_argument("--retry-dir", default=PARA_DIR, help="where retry-round lines para_raw_r<k>_*.jsonl live "
                    "(Task 6c: data/hard/full/q1x)")
    pi.add_argument("--gold", default=PARA_GOLD)
    pi.add_argument("--out", default=PARA_DIR)
    pj = sub.add_parser("para-judge", help=f"Task 5: ds_hard judge + plants + gate ({HARD_API_ENV}=1 for the API)")
    pj.add_argument("--in", dest="inp", nargs="+", default=list(PARA_IN))
    pj.add_argument("--out", default=PARA_DIR)
    pj.add_argument("--packets", default=PARA_PACKETS)
    pj.add_argument("--seed", type=int, default=20261003)
    pj.add_argument("--plant-share", type=float, default=0.15)
    pj.add_argument("--n", type=int, default=40)
    pj.add_argument("--retry-instructions", default=f"{PARA_DIR}/para_instructions_retry.md")
    pj.add_argument("--plant-kinds", nargs="+", default=None, help="Task 6c: + object_swap (default: Task 5's kinds)")
    pj.add_argument("--topup", default=None, help="Task 6c: a judged file (para_judged.jsonl) whose accepted pairs "
                    "top up plant kinds the items cannot fill (op_swap)")
    pf = sub.add_parser("para-force-retry", help="R51: forced round 2 for count paraphrases asking «how many times»")
    pf.add_argument("--in", dest="inp", nargs="+", default=list(PARA_IN))
    pf.add_argument("--out", default=PARA_DIR)
    pf.add_argument("--packets", default=PARA_PACKETS)
    pf.add_argument("--seed", type=int, default=20261003)
    pf.add_argument("--n", type=int, default=40)
    pf.add_argument("--round", type=int, default=None, help="forced round 2 or 3 (default: the next free one)")
    pf.add_argument("--retry-instructions", default=None, help="default para_instructions_retry<k>.md")
    fp = sub.add_parser("full-pairs", help="Task 6: full Q3 packets (probe pairs excluded, R26 approval filter)")
    fp.add_argument("--n", type=int, default=230)
    fp.add_argument("--seed", type=int, default=20261003)
    fp.add_argument("--exclude-pairs", default=PROBE_VALID)
    fp.add_argument("--exclude-packets", default=PROBE_PACKETS)
    fp.add_argument("--out", default=CMP_PACKETS + "/")
    fp.add_argument("--instructions", default=f"{CMP_DIR}/compare_instructions.md")
    fp.add_argument("--max-per-doc", type=int, default=2)
    fp.add_argument("--max-per-doc-kind", type=int, default=1)
    fp.add_argument("--topup", action="store_true", help="extend the existing packets up to --n in total (R25 style)")
    fi = sub.add_parser("full-ingest-compare", help="Task 6: code checks of the full Q3 generator lines")
    fi.add_argument("--raw", nargs="+", default=[f"{CMP_DIR}/compare_raw_*.jsonl"])
    fi.add_argument("--packets", default=CMP_PACKETS)
    fi.add_argument("--out", default=CMP_DIR)
    fi.add_argument("--tag", default="", help="output suffix of a round (R54 repair: _r1)")
    cj = sub.add_parser("cmp-judge", help=f"Task 6: ds_hard compare judge + plants + gate ({HARD_API_ENV}=1)")
    cj.add_argument("--packets", default=CMP_PACKETS)
    cj.add_argument("--out", default=CMP_DIR)
    cj.add_argument("--seed", type=int, default=20261003)
    cj.add_argument("--plant-share", type=float, default=0.15)
    cj.add_argument("--per-call", type=int, default=1, help="items per judge call (R53 (a): 1)")
    cj.add_argument("--tag", default="", help="round suffix (R54 repair: _r1)")
    cj.add_argument("--no-plants", action="store_true", help="repair round: round 0's passed gate stands")
    rpk = sub.add_parser("reader-packets", help="Task 6: A / B / C reader packets (10 items) + rubric + prompts")
    rpk.add_argument("--packets", default=CMP_PACKETS)
    rpk.add_argument("--out", default=CMP_DIR)
    rpk.add_argument("--review", default=CMP_REVIEW)
    rpk.add_argument("--n", type=int, default=10)
    rpk.add_argument("--seed", type=int, default=20261003)
    rpk.add_argument("--tag", default="")
    rpk.add_argument("--rubric-version", type=int, default=2, choices=(1, 2))
    ri = sub.add_parser("reader-ingest", help="Task 6: reader consensus + R32 splits -> compare_consensus.jsonl")
    ri.add_argument("--out", default=CMP_DIR)
    ri.add_argument("--review", default=CMP_REVIEW)
    ri.add_argument("--reader-a", default="fable", choices=("fable", "opus-a2"),
                    help="reader A model for lines without a \"reader\" field")
    ri.add_argument("--seed", type=int, default=20261003)
    ri.add_argument("--target", type=int, default=150)
    ri.add_argument("--tag", default="")
    rr = sub.add_parser("repair-packets", help="R54: one repair round over every judged Q3 item")
    rr.add_argument("--out", default=CMP_DIR)
    rr.add_argument("--packets", default=CMP_PACKETS)
    rr.add_argument("--review", default=CMP_REVIEW)
    rr.add_argument("--repair", default=f"{CMP_DIR}/compare_repair")
    rr.add_argument("--n", type=int, default=10)
    ph = sub.add_parser("hashes", help="update docs/m7/artifacts.sha256 in place (one line per path)")
    ph.add_argument("--file", default="docs/m7/artifacts.sha256")
    ph.add_argument("paths", nargs="*")
    a = p.parse_args(argv)
    if a.cmd == "para-force-retry":
        return cmd_para_force_retry(a)
    if a.cmd == "hashes":
        return cmd_hashes(a)
    cmds = {"full-pairs": cmd_full_pairs, "full-ingest-compare": cmd_full_ingest_compare, "cmp-judge": cmd_cmp_judge,
            "reader-packets": cmd_reader_packets, "reader-ingest": cmd_reader_ingest,
            "repair-packets": cmd_repair_packets}
    if a.cmd in cmds:
        return cmds[a.cmd](a)
    if a.cmd == "para-packets":
        return cmd_para_packets(a)
    if a.cmd == "para-ingest":
        return cmd_para_ingest(a)
    if a.cmd == "para-judge":
        return cmd_para_judge(a)
    if a.cmd == "full-snapshot":
        if not 1 <= a.workers <= 2:
            raise SystemExit("--workers: 1 or 2 on the Mac")
        return cmd_full_snapshot(a)
    if a.cmd == "snapshot":
        if not 1 <= a.workers <= 2:
            raise SystemExit("--workers: 1 or 2 on the Mac")
        return cmd_snapshot(a)
    if a.cmd == "pairs":
        if a.n > 60 or a.cap > 60:
            raise SystemExit("> 60: the Q3 probe cap is 60 packets")
        if a.exclude_kinds is not None and a.n_new is None:     # it is read only on the --n-new path
            raise SystemExit("--exclude-kinds needs --n-new (with --topup): without it no kind would be excluded")
        return cmd_pairs(a)
    if a.cmd == "ingest-compare":
        return cmd_ingest_compare(a)
    if a.cmd == "full-make":
        return cmd_full_make(a)
    if a.cmd == "q2-augment":
        return cmd_q2_augment(a)
    return cmd_make(a)


if __name__ == "__main__":
    sys.exit(main())
