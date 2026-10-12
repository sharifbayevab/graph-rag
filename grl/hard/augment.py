"""M7 pilot 8, ruling R55: more Q2 (version) TRAIN rows from the candidates Task 4 left unused.

Task 4 selected at most 8 Q2 items per amended act (R43) and one item per (root, answer list) (R48); about 200
eligible candidates stayed unused. R55 adds, for the TRAIN split only, Q2 items under a per-act cap of 16 (the
existing Q2 rows of every split count toward it), worded by the template itself: no paraphrase, no judge, no LLM
budget. Every added row carries meta.augment = "template_train_only". Val / test keep the judged, paraphrased rows
only -- nothing here touches them.

Guards (all of Task 4's still apply -- the candidates come from grl.hard.templates.version_items):
- R33 gold (chunk_spans: the quoted fragments of the selected edges), R6 visibility, R16 naming, the leak check of
  every rendering (leak_ok against the gold texts), the probe qids excluded in every language, R48 one item per (root,
  answer list) -- also against the existing Q2 rows;
- R32 / R44: a candidate is kept only if every document it touches (question docs + gold docs) that already has a split
  -- old set (data/gold/splits.json), any Q1 / Q2 row of any split, any Q3 row -- has split train, and at least one
  does (so its component inherits train by R32 (1); a candidate whose documents are all new is dropped, never placed
  by R45). Afterwards fixed_doc_splits over old + Q1 / Q2 + Q3 + the added rows must hold (one split per document),
  and assign_splits over the Q1 / Q2 rows + the added rows must reproduce every existing split and give train to every
  added row;
- the existing candidate keys (template, root, meta) and qids are excluded: an added row never repeats a Task 4 item."""
from __future__ import annotations

import json
from collections import Counter

AUG_MARK = "template_train_only"
AUG_CAP = 16                     # R55: Q2 per amended act (Task 4 / R43: 8)
AUG_SPLIT = "train"


def _answer_key(root: str, answer) -> tuple:
    return root, json.dumps(answer, ensure_ascii=False)


def train_only_filter(fixed: dict, chunk_doc: dict, existing_keys: set, existing_answers: set):
    """pre_drop for version_items: a drop reason, or None for a train-only candidate (module docstring)."""
    from grl.hard.splits import gold_docs

    def pre_drop(c):
        if c.key in existing_keys:
            return "aug_existing_item"
        if _answer_key(c.root, c.answer) in existing_answers:
            return "aug_dup_existing_answer"
        docs = set(c.docs) | set(gold_docs(c.gold, chunk_doc))
        splits = {fixed[d] for d in docs if d in fixed}
        if not splits:
            return "aug_all_docs_new"
        if len(splits) > 1:
            return "aug_r32_conflict"                 # R32 (2): documents of two splits
        if splits != {AUG_SPLIT}:
            return "aug_val_or_test_component"        # a val / test question: R55 adds train rows only
        return None
    return pre_drop


def augment_q2(nodes, edges, mu_inv, seed: int, n: int, *, old_splits: dict, chunk_doc: dict, q12_rows: list,
               q3_rows: list, cap: int = AUG_CAP, **kw) -> tuple[list[dict], dict]:
    """(added rows -- HardItem dicts + gold_docs + split train + meta.augment --, stats). `q12_rows`: every Q1 / Q2 row
    of Task 4 (count_raw + version_raw, all splits, original meta); `q3_rows`: the Q3 rows (their documents' splits
    are fixed too). `kw`: chunk_texts, chunk_spans, lang_shares, exclude (passed to version_items)."""
    from grl.hard.splits import assign_splits, fixed_doc_splits, gold_docs
    from grl.hard.templates import item_key, version_items
    fixed = fixed_doc_splits(old_splits, list(q12_rows) + list(q3_rows))
    ver = [r for r in q12_rows if r["qtype"] == "version"]
    keys = {item_key(r) for r in ver}
    answers = {_answer_key(r["docs"][0], r["answer"]) for r in ver}
    roots = Counter(r["docs"][0] for r in ver)
    st = Counter()
    items = version_items(nodes, edges, mu_inv, seed, n, max_per_root=cap, stats=st, root_counts=roots,
                          pre_drop=train_only_filter(fixed, chunk_doc, keys, answers), dedup_answers=True, **kw)
    rows = []
    for it in items:
        r = {**it.to_json(), "gold_docs": gold_docs(it.gold, chunk_doc), "split": AUG_SPLIT}
        r["meta"] = {**r["meta"], "augment": AUG_MARK}
        rows.append(r)
    # --- R32 re-checks over everything ------------------------------------------------------------------------------
    old_q = {r["qid"] for r in q12_rows} | {r["qid"] for r in q3_rows}
    clash = sorted(old_q & {r["qid"] for r in rows})
    if clash:
        raise ValueError(f"added qids repeat existing rows: {clash[:3]}")
    if {item_key(r) for r in rows} & keys:
        raise ValueError("an added row repeats a Task 4 candidate key")
    fixed_doc_splits(old_splits, list(q12_rows) + list(q3_rows) + rows)          # ValueError on a 2-split document
    split = assign_splits(list(q12_rows) + rows, old_splits, seed)
    moved = [r["qid"] for r in q12_rows if split[r["qid"]] != r["split"]]
    if moved:
        raise ValueError(f"the added rows change existing splits: {moved[:3]}")
    bad = [r["qid"] for r in rows if split[r["qid"]] != AUG_SPLIT]
    if bad:
        raise ValueError(f"added rows outside train under R32: {bad[:3]}")
    per_root = Counter(r["docs"][0] for r in ver) + Counter(r["docs"][0] for r in rows)
    if max(per_root.values(), default=0) > cap:
        raise ValueError(f"per-act cap {cap} broken")
    rep = {"n_added": len(rows), "cap": cap, "target_slots": n,
           "by_template": dict(sorted(Counter(r["template"] for r in rows).items())),
           "by_lang": dict(sorted(Counter(r["lang"] for r in rows).items())),
           "gold_sizes": dict(sorted(Counter(len(r["gold"]) for r in rows).items())),
           "roots": len({r["docs"][0] for r in rows}),
           "max_per_root_after": max(per_root.values(), default=0),
           "drops": {k: v for k, v in sorted(st.items()) if k.startswith(("aug_", "probe_excluded", "dup_answer",
                                                                         "cap_blocked", "leak_", "r33_", "r6",
                                                                         "evidence_invisible", "r16_"))},
           "stats": dict(sorted(st.items()))}
    return rows, rep
