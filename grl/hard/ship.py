"""M7 part 2 (pilot 7 interim; plan Task 7 reduced): the joint old + hard ship of the struct variant.

- hard_pairs: the M7 Q1 / Q2 HardItems (data/hard/full/para_judged.jsonl: paraphrased `question`, `gold` = STRUCT/<root>
  for Q1, corpus chunk ids for Q2) of the train / val splits as §8.1 Pair rows (graphrag.dataset.finalize.Pair: id =
  qid, q = question, pattern = qtype, C_ref = gold, origin "m7_hard", provenance.set "hard"). Test rows are never
  read into a row (the test split stays sealed on the Mac); Q3 (compare) is not a type of this ship; a probe id
  (grl.hard.probe.probe_ids) is refused. Pair rows carry no "qid" key, so refuse_probe_file does not mistake them
  for probe rows -- they are read by the ordinary train / val loaders (load_items, expert_inputs).
- read_hard_pairs: a guarded reader of such a file (train / val only, hard origin only, no probe id, no Q-real) --
  scripts/build_rankings.py --family hard builds their U9 rankings with the old set's path and settings.
- merge_rankings: one U9 rankings artifact from several (same graph, encoder, BM25, top, lexicon, records, gold and
  code digests -- only `family` / `candidates` may differ; disjoint keys; equal array shapes), so the joint ship has
  one latest.json. The per-query arrays are copied unchanged.
- struct_gold_ok: STRUCT gold (a STRUCT/<root> id: in no index, an env row only in the struct EnvData) is accepted in a
  ship only when the ship's variant is "struct" (grl.train.pilot.check_ship)."""
from __future__ import annotations

import json
from pathlib import Path

HARD_ORIGIN = "m7_hard"
HARD_SET, OLD_SET = "hard", "old"
HARD_TYPES = ("count", "version")            # Q1 / Q2; Q3 (compare) is excluded from pilot 7 (still in repair)
STRUCT_VARIANT = "struct"
MERGE_SAME = ("graph_version", "top", "bm25", "records_manifest", "lexicon", "encoder", "gold", "code")
HARD_FIELDS = ("qid", "qtype", "template", "lang", "question", "answer", "gold", "docs", "split")


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _guard():
    from grl.train.data import SplitGuardError
    return SplitGuardError


def is_struct_id(g) -> bool:
    from grl.hard.probe import STRUCT_PREFIX
    return isinstance(g, str) and g.startswith(STRUCT_PREFIX)


def hard_pair(r: dict) -> dict:
    """One HardItem -> a §8.1 Pair dict (Pair.from_json-able)."""
    gold = sorted(r["gold"])
    if not gold or not all(isinstance(g, str) and g for g in gold):
        raise ValueError(f"{r.get('qid')}: empty / malformed gold")
    return {"id": r["qid"], "q": r["question"], "lang": r["lang"], "pattern": r["qtype"], "hop": "multi",
            "y_ref": {"text": str(r["answer"]), "sources": []}, "C_ref": gold, "C_ref_anchors": [], "facts": [],
            "docs": list(r.get("gold_docs") or r["docs"]), "split": r["split"], "section": "", "seed_id": r["qid"],
            "origin": HARD_ORIGIN, "trajectory": [], "actions": [], "action_equiv": [], "action_masks": [],
            "reward": None,
            "provenance": {"set": HARD_SET, "m7": {"qtype": r["qtype"], "template": r["template"],
                                                   "meta": r.get("meta") or {},
                                                   **({"alt_gold": list(r["alt_gold"]), "easy": bool(r.get("easy"))}
                                                      if r.get("alt_gold") else {})}}}


def hard_pairs(path, types=HARD_TYPES, splits=("train", "val"), _open_test: bool = False) -> tuple[list[dict], dict]:
    """(Pair dicts of the `types` x `splits` rows of a para_judged file sorted by id, report). A probe id or a
    duplicate qid is refused; rows of other splits / types are counted, never copied."""
    from grl.hard.probe import probe_ids
    from grl.train.data import ALLOWED_SPLITS, qreal_ids
    bad_splits = [s for s in splits if s not in ALLOWED_SPLITS]
    if bad_splits and not _open_test:
        raise _guard()(f"hard_pairs reads only {ALLOWED_SPLITS}; refused {bad_splits}")
    rows = _jsonl(path)
    miss = [(r.get("qid"), [k for k in HARD_FIELDS if k not in r]) for r in rows if any(k not in r for k in HARD_FIELDS)]
    if miss:
        raise ValueError(f"{path}: rows that are not split HardItems: {miss[:3]}")
    q = [r["qid"] for r in rows]
    if len(q) != len(set(q)):
        raise ValueError(f"{path}: duplicate qids")
    pids, qids = probe_ids(), qreal_ids()
    if set(q) & pids:
        raise _guard()(f"{path}: probe ids among the hard rows: {sorted(set(q) & pids)[:3]}")
    if set(q) & qids:
        raise _guard()(f"{path}: Q-real ids among the hard rows: {sorted(set(q) & qids)[:3]}")
    out, skipped = [], {}
    for r in rows:
        if r["qtype"] in types and r["split"] in splits:
            out.append(hard_pair(r))
        else:
            k = f"{r['qtype']}|{r['split']}"
            skipped[k] = skipped.get(k, 0) + 1
    out.sort(key=lambda p: p["id"])
    by = {}
    for p in out:
        k = f"{p['pattern']}|{p['split']}"
        by[k] = by.get(k, 0) + 1
    return out, {"n": len(out), "by": dict(sorted(by.items())), "skipped": dict(sorted(skipped.items())),
                 "types": list(types), "splits": list(splits)}


def hard_test_pairs(path, opened: dict) -> tuple[list[dict], dict]:
    """Task 9f / 10, the ONE strict test opening (scripts/m7_final_eval.py open, behind the once-token): the Q1 / Q2
    test rows of a para_judged-style file as Pair rows (split "test"; build_ship(test_eval=True) adds the test-eval
    mark). `opened` must be the once-token ledger record of that opening -- nothing else may call this."""
    if not (isinstance(opened, dict) and opened.get("stage") == "open" and opened.get("token_sha256")
            and opened.get("marker")):
        raise _guard()("hard_test_pairs: only scripts/m7_final_eval.py open (once-token) reads the hard test rows")
    m = Path(opened["marker"])                  # fix round 1 (M1): the O_EXCL marker on disk must be this record
    try:
        on_disk = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise _guard()(f"hard_test_pairs: no open marker at {m}") from None
    if on_disk != {k: v for k, v in opened.items() if k != "marker"} or \
            not m.name.startswith(str(opened["token_sha256"])[:16]):
        raise _guard()(f"hard_test_pairs: {m} is not the open record of this token")
    return hard_pairs(path, HARD_TYPES, ("test",), _open_test=True)


def read_hard_test_pairs(path) -> list:
    """The hard TEST Pair rows of the final opening (build_rankings.py --family hard_test): every row split test,
    origin m7_hard, no probe id, no Q-real."""
    from graphrag.dataset.finalize import read_pairs
    from grl.hard.probe import probe_ids
    from grl.train.data import QREAL_ORIGIN, guard_qkeys, qreal_ids
    ps = read_pairs(path)
    pids, qids = probe_ids(), qreal_ids()
    bad = [p.id for p in ps if p.split != "test" or p.origin != HARD_ORIGIN or p.id in pids or p.id in qids
           or p.origin == QREAL_ORIGIN]
    if bad:
        raise _guard()(f"{path}: rows that are not hard test rows: {bad[:3]}")
    guard_qkeys([f"{p.id}|q" for p in ps])
    return ps


def read_hard_pairs(path) -> list:
    """Hard Pair rows of a file (scripts/build_rankings.py --family hard): every row train / val, origin m7_hard, no
    probe id, no Q-real (id / origin), Q-real-shaped keys refused."""
    from graphrag.dataset.finalize import read_pairs
    from grl.hard.probe import probe_ids
    from grl.train.data import ALLOWED_SPLITS, QREAL_ORIGIN, guard_qkeys, qreal_ids
    ps = read_pairs(path)
    pids, qids = probe_ids(), qreal_ids()
    bad = [p.id for p in ps if p.split not in ALLOWED_SPLITS or p.origin != HARD_ORIGIN or p.id in pids
           or p.id in qids or p.origin == QREAL_ORIGIN]
    if bad:
        raise _guard()(f"{path}: rows that are not train / val hard rows: {bad[:3]}")
    guard_qkeys([f"{p.id}|q" for p in ps])
    return ps


def struct_gold_ok(rows, variant: str | None) -> None:
    """STRUCT gold ids (C_ref) only in a ship of variant struct (SplitGuardError otherwise)."""
    if variant == STRUCT_VARIANT:
        return
    bad = [r.get("id") for r in rows if any(is_struct_id(g) for g in (r.get("C_ref") or ()))]
    if bad:
        raise _guard()(f"STRUCT gold in a ship of variant {variant!r} (only struct reads it): {bad[:3]}")


def merge_rankings(latests, out_root) -> str:
    """One rankings artifact from several latest.json: same MERGE_SAME inputs, disjoint keys, arrays padded to the
    widest column count with -1 / 0 (load_rankings cuts idx < 0). Keys sorted. Returns the new version."""
    import numpy as np
    from graphrag.retrieval.rankings import ARRAYS, load_rankings, write_rankings
    parts = [load_rankings(x) for x in latests]
    base = parts[0].meta["inputs"]
    for p in parts[1:]:
        diff = [k for k in MERGE_SAME if p.meta["inputs"].get(k) != base.get(k)]
        if diff:
            raise ValueError(f"rankings {p.meta['version']} differ from {parts[0].meta['version']} in {diff}")
    keys = [k for p in parts for k in p.keys()]
    if len(keys) != len(set(keys)):
        raise ValueError("rankings to merge share query keys")
    src = []
    for p in parts:
        arr = {n: np.load(p.dir / f"{n}.npy") for n in ARRAYS}
        for i, k in enumerate(p.keys()):
            src.append((k, p._text[k], {n: arr[n][i] for n in ARRAYS}))
    src.sort(key=lambda t: t[0])
    arrays = {}
    for n in ARRAYS:
        rows = [s[2][n] for s in src]
        shapes = {(r.shape, r.dtype.str) for r in rows}
        if len(shapes) != 1:
            raise ValueError(f"rankings to merge differ in {n} shape / dtype: {sorted(shapes)}")
        arrays[n] = np.stack(rows)
    inputs = {**{k: base.get(k) for k in MERGE_SAME}, "family": "merged",
              "merged_from": {p.meta["version"]: {"family": p.meta["inputs"].get("family"),
                                                  "candidates": p.meta["inputs"].get("candidates"),
                                                  "n_queries": len(p.keys())} for p in parts}}
    return write_rankings(Path(out_root), [(k, t) for k, t, _ in src], arrays, inputs)
