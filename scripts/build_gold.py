"""M3 Task 10: Q-real -> data/gold/{qreal.jsonl, meta.json, review.jsonl} and the document split
data/gold/splits.json (П3, for the synthetic dataset). Flagged items (non-exact tier, several occurrences, > 2 core
chunks) need a decision in data/gold/review_decisions.json {qid: {"accept": bool, "note": str, "spans": [[s, e]]?,
"equiv_spans" / "mention_spans": [[s, e]]?, "gold_answer": str?}} -- dropped same-document occurrences: an equivalent
restatement -> GoldItem.alt_equiv (any-of eval credit), the quote without the answer -> alt_mention (leakage only);
gold_answer corrects the validation-set answer (original kept in gold_answer_orig). Split groups: M2 translation /
edition / near-duplicate pairs + M3-R4 title links + M3-R5 same-subject-act links; stop if a group > 10 % weight.
    python scripts/build_gold.py [--remap]"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace

from graphrag import config
from graphrag.data.ingest import read_records
from graphrag.eval.gold import (Anchor, anchors_of, assign_splits, build_items, chunks_overlapping, graph_act_link_pairs,
                                curated_link_pairs, graph_subject_act_pairs, linked_doc_pairs, read_gold, strict_doc_groups,
                                remap, text_sha, write_gold)


DECISION_KEYS = {"accept", "note", "spans", "equiv_spans", "mention_spans", "gold_answer"}


def _review_row(it, rec) -> dict:
    return {"qid": it.qid, "review": it.review, "tier": it.tier, "question": it.question_cyr, "quote": it.supporting_quote,
            "answer": it.gold_answer,
            "anchors": [[a.start, a.end, rec.text[max(0, a.start - 200):a.end + 200]] for a in it.anchors],
            "core": {c.chunk_id: c.text for c in rec.chunks if c.chunk_id in it.core}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remap", action="store_true")
    ap.add_argument("--out", default="data/gold")
    args = ap.parse_args(argv)
    out = config.path(args.out)
    records = read_records(config.path(config.load("data")["records_dir"]))
    recs = {r.doc_id: r for r in records}
    if args.remap:
        items, problems, notes = remap(read_gold(out / "qreal.jsonl"), records)
        print("\n".join(problems + notes))
        if problems:
            return 1
        write_gold(out / "qreal.jsonl", items)
        return 0
    items, stats = build_items(records)
    flagged = [i for i in items if i.review]
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.jsonl").write_text("".join(json.dumps(_review_row(i, recs[i.doc_id]), ensure_ascii=False) + "\n"
                                              for i in flagged), encoding="utf-8")
    dec_f = out / "review_decisions.json"
    dec = json.loads(dec_f.read_text(encoding="utf-8")) if dec_f.exists() else {}
    missing = [i.qid for i in flagged if i.qid not in dec]
    if missing:
        print(f"{len(missing)} flagged items without a decision in {dec_f}: {missing}")
        return 1
    kept, rejected = [], []
    for it in items:
        d = dec.get(it.qid)
        if d is not None and not d["accept"]:
            rejected.append(it.qid)
            continue
        if d is not None:
            rec = recs[it.doc_id]
            unknown = set(d) - DECISION_KEYS
            if unknown:
                raise SystemExit(f"{it.qid}: unknown review decision keys {sorted(unknown)}")
            for s, e in d.get("spans", []) + d.get("equiv_spans", []) + d.get("mention_spans", []):
                if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(rec.text)):
                    raise SystemExit(f"{it.qid}: bad review span [{s}, {e}] (document length {len(rec.text)})")
            if d.get("spans"):
                anchors = anchors_of(rec, [tuple(s) for s in d["spans"]])
                it = replace(it, anchors=anchors, tier=it.tier if it.tier == "exact" else "manual",
                             core=sorted({c for a in anchors for c in chunks_overlapping(rec, a.start, a.end)}))

            def _chunks(key):
                return {c for s, e in d.get(key, []) for c in chunks_overlapping(rec, s, e)} - set(it.core)
            # dropped same-document occurrences (Ruling M3-R5 (3)): an equivalent restatement of the answer ->
            # alt_equiv (any-of credit in eval); the quote without the answer -> alt_mention (leakage control only)
            equiv = _chunks("equiv_spans")
            mention = (_chunks("mention_spans") | set(it.alt_mention)) - equiv
            it = replace(it, alt_equiv=sorted(set(it.alt_equiv) | equiv), alt_mention=sorted(mention))
            if "gold_answer" in d:                      # corrected answer (M3-R5 (4)); the original is kept
                it = replace(it, gold_answer_orig=it.gold_answer, gold_answer=d["gold_answer"])
            it = replace(it, review=f"{it.review}; accepted: {d['note']}")
        kept.append(it)
    # Ruling M3-R7 (2): every item with alt_mention chunks is reviewed in alt_review.json {qid: {equiv: [chunk_id],
    # note}}; chunks of other documents that fully answer the question move to alt_equiv (any-of eval credit)
    ar_f = out / "alt_review.json"
    ar = json.loads(ar_f.read_text(encoding="utf-8")) if ar_f.exists() else {}
    missing = sorted(i.qid for i in kept if i.alt_mention and i.qid not in ar)
    if missing:
        print(f"{len(missing)} items with alt_mention but no entry in {ar_f}: {missing}")
        return 1
    for n, it in enumerate(kept):
        a = ar.get(it.qid)
        if a is None:
            continue
        eq = set(a["equiv"])
        if not eq <= set(it.alt_mention) or not str(a.get("note") or "").strip():
            raise SystemExit(f"{it.qid}: alt_review equiv must be alt_mention chunks and carry a note")
        kept[n] = replace(it, alt_equiv=sorted(set(it.alt_equiv) | eq), alt_mention=sorted(set(it.alt_mention) - eq))
    # Ruling M3-R8/R9: adjudicated support (data/gold/support_decisions.json {qid: {chunk_id: {keep, borderline,
    # evidence, answer_part, reason}}}). A keep becomes a support anchor on its evidence quote, which must be an exact
    # substring of that chunk of the item's own document; rejects never enter C*; borderline rejects (the chunk only
    # names the cited act / session) go to `context`.
    sd_f = out / "support_decisions.json"
    sd = json.loads(sd_f.read_text(encoding="utf-8")) if sd_f.exists() else None
    if sd is None and (out / "qreal.jsonl").exists() and any(i.support_anchors for i in read_gold(out / "qreal.jsonl")):
        raise SystemExit(f"{out / 'qreal.jsonl'} carries support but {sd_f} is missing: refusing to drop it")
    by_id = {c.chunk_id: (r, c) for r in records for c in r.chunks}
    n_sd = Counter()
    for n, it in enumerate(kept):
        dec_i = (sd or {}).get(it.qid, {})
        anchors, context = [], set()
        for cid, x in sorted(dec_i.items()):
            n_sd["decisions"] += 1
            if cid not in by_id:
                raise SystemExit(f"{it.qid}: support decision for unknown chunk {cid}")
            r, c = by_id[cid]
            if x.get("keep"):
                ev = x.get("evidence") or ""
                i = c.text.find(ev) if ev else -1
                if r.doc_id != it.doc_id or i < 0:
                    raise SystemExit(f"{it.qid}: keep {cid}: evidence is not an exact substring of that chunk of {it.doc_id}")
                anchors.append(Anchor(r.doc_id, c.start + i, c.start + i + len(ev), text_sha(ev)))
                n_sd["keep"] += 1
            elif x.get("borderline"):
                context.add(cid)
                n_sd["borderline"] += 1
            else:
                n_sd["reject"] += 1
        if sd is None:
            continue
        rec = recs[it.doc_id]
        sup = sorted({ch for a in anchors for ch in chunks_overlapping(rec, a.start, a.end)} - set(it.core))
        ctx = sorted(context - set(it.core) - set(sup) - set(it.alt_equiv))
        kept[n] = replace(it, support_anchors=sorted(anchors, key=lambda a: (a.start, a.end)), support=sup, context=ctx,
                          alt_equiv=sorted(set(it.alt_equiv) - set(sup)), alt_mention=sorted(set(it.alt_mention) - set(sup)))
    write_gold(out / "qreal.jsonl", kept)
    sp = config.load("train")["split"]
    sections = {r.doc_id: r.meta.section for r in records}
    weight = {r.doc_id: len(r.chunks) for r in records}
    act_pairs = graph_act_link_pairs(sections)                       # Ruling M3-R4: title links to corpus acts
    subj_pairs, subj_audit = graph_subject_act_pairs(records)         # Ruling M3-R5: same subject act (corpus / EXT)
    cur_pairs = curated_link_pairs(sp.get("links"), sections)       # Ruling M3-R7: curated families
    pairs = sorted(set(linked_doc_pairs()) | set(act_pairs) | set(subj_pairs) | set(cur_pairs))
    group = strict_doc_groups(sorted(sections), pairs)
    comp_w, comp_n = Counter(), Counter()
    for d, g in group.items():
        comp_w[g] += weight[d]
        comp_n[g] += 1
    big = max(comp_w, key=lambda g: (comp_w[g], g))
    most = max(comp_n, key=lambda g: (comp_n[g], g))
    groups_meta = {"pairs_linked_near_dup": len(linked_doc_pairs()), "pairs_act_links": len(act_pairs),
                   "pairs_subject_act": len(subj_pairs), "pairs_curated": len(cur_pairs), "pairs_total": len(pairs), "subject_act_audit": subj_audit,
                   "n_groups": len(comp_n), "group_size_hist": {str(k): v for k, v in sorted(Counter(comp_n.values()).items())},
                   "largest": {"group": big, "docs": comp_n[big], "weight": comp_w[big],
                               "weight_share": round(comp_w[big] / sum(weight.values()), 6)},
                   "most_docs": {"group": most, "docs": comp_n[most], "weight": comp_w[most],
                                 "weight_share": round(comp_w[most] / sum(weight.values()), 6)}}
    if comp_w[big] > 0.10 * sum(weight.values()):
        print(f"STOP (M3-R4/R5): component {big} holds {comp_w[big]} of {sum(weight.values())} fragments (> 10 %)")
        print(json.dumps(groups_meta, ensure_ascii=False))
        return 1
    splits = assign_splits(sections, weight, pairs, int(sp["seed"]), dict(sp["ratios"]))
    (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    manifest = config.path(config.load("data")["records_dir"]) / "manifest.jsonl"
    meta = {"tiers": dict(sorted(Counter(i.tier for i in kept).items())), "tiers_pre_review": stats["tiers"],
            "alt_equiv_items": sum(bool(i.alt_equiv) for i in kept), "alt_mention_items": sum(bool(i.alt_mention) for i in kept),
            "support_items": sum(bool(i.support) for i in kept), "context_items": sum(bool(i.context) for i in kept),
            "support_decisions": dict(sorted(n_sd.items())), "gold_answer_corrected": sorted(i.qid for i in kept if i.gold_answer_orig is not None), "n_items": len(kept), "rejected": sorted(rejected), "flagged": len(flagged),
            "records_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "groups": groups_meta,
            "split_docs": {s: sum(v == s for v in splits.values()) for s in sp["ratios"]},
            "split_chunks": {s: sum(len(recs[d].chunks) for d, v in splits.items() if v == s) for s in sp["ratios"]}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
