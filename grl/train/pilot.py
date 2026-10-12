"""Pilot / final data snapshots for M5 (plan Task 8, 13, 14): a content-addressed copy of the consensus-GOOD pairs
(built in a temp dir and renamed; reuse re-verifies the recorded sha256s), the train / val filter, the shippable
subset ship/ (the only part Task 12 syncs to Spark), the dev-Postgres port check and a key subset of a U9 ranking
artifact. Pilot 5 (controller ruling): the verified source (verified_source: verified.jsonl minus reader-BAD /
reader-C-veto / superseded originals) and the optional ship/test_eval/ -- the strict consensus test, shipped for the
final report's evaluation only (train.py eval / baselines --split test); no training loader reads it."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

SNAPSHOT_FILES = ("verified.jsonl", "reviewed.jsonl", "stats.json")
PINNED_FILES = ("verified.jsonl", "reviewed.jsonl")   # stats.json is rewritten by the M3 commands (recorded apart)
SHIP_FILES = ("trajectories.jsonl", "pairs.jsonl", "splits.json")   # + rankings/**; pairs.jsonl only if present
SHIP_DIR = "ship"
DEV_PG_PORT = 5437
TEST_EVAL_DIR = "test_eval"            # ship/test_eval/: the strict-consensus test (pilot 5 final report only)
TEST_CANDIDATES = "test_candidates.jsonl"   # Mac-only: the finalized test rows set aside before keep_splits
TEST_RANKINGS = "test_rankings"        # Mac-only: U9 rankings of the test queries (copied into ship/test_eval/)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot(src, out_root) -> Path:
    src, out_root = Path(src), Path(out_root)
    tag = _sha(src / "reviewed.jsonl")[:12]
    d = out_root / tag
    if d.exists():
        meta = d / "snapshot.json"
        if not meta.exists():
            raise ValueError(f"{d}: partial snapshot (no snapshot.json) -- remove it and rerun")
        rec = json.loads(meta.read_text(encoding="utf-8"))["sha256"]
        bad = [n for n in PINNED_FILES if not (d / n).exists() or _sha(d / n) != rec.get(n)]
        if bad or rec.get("reviewed.jsonl", "")[:12] != tag:
            raise ValueError(f"{d}: snapshot files differ from snapshot.json: {bad or ['tag']}")
        return d
    tmp = out_root / f".tmp-{tag}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for n in SNAPSHOT_FILES:
        shutil.copyfile(src / n, tmp / n)
    (tmp / "snapshot.json").write_text(json.dumps({
        "source": str(src), "sha256": {n: _sha(tmp / n) for n in PINNED_FILES},
        "stats_json_at_snapshot": _sha(tmp / "stats.json")}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(tmp, d)
    return d


# ---- pilot 5: the verified source (controller ruling 2026-10-02) ---------------------------------------------------
SOURCE_READERS = (("A", "review/verdicts"), ("B", "review2/verdicts"), ("C", "review3/verdicts"))
EXCLUSIONS = ("superseded_by_repair", "reader_a_bad", "reader_b_bad", "reader_c_veto")   # primary-reason order
ADMITTED_BY = "m5-pilot5-verified-source"
SOURCE_JSON = "source.json"


def _reader_verdicts(d: Path) -> tuple[dict[str, list[str]], int]:
    """reader pair id -> every verdict line of the reader's verdict files; unparsable lines are counted, not read
    (ingest_verdicts rejects them too)."""
    out, bad = {}, 0
    for f in sorted(Path(d).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(r, dict) and r.get("pair_id"):
                out.setdefault(str(r["pair_id"]), []).append(str(r.get("verdict")))
            else:
                bad += 1
    return out, bad


def verified_source(src, splits=("train", "val")) -> tuple[list[dict], dict]:
    """Pilot 5 data (controller ruling): SOURCE = verified.jsonl minus every pair reader A (review/verdicts) or
    reader B (review2/verdicts) read BAD, minus every reader-C veto (review_rejections.jsonl reason reader_c_veto),
    minus every original superseded by its -r1 repair (the successor is kept); BORDERLINE and unread pairs stay.
    TEST = the strict consensus test of reviewed.jsonl. Returns (rows, report): rows = SOURCE rows of `splits` + the
    TEST rows, sorted by seed id, each with provenance.review.verdict GOOD so that `build_dataset.py finalize` (which
    finalizes only review-GOOD rows) takes them -- a consensus pair keeps its real reviewed.jsonl row, any other
    SOURCE pair gets {verdict: GOOD, admitted_by: ADMITTED_BY, readers: {A, B, C: the readers' real verdicts}}."""
    from graphrag.dataset.opus import REPAIR_SUFFIX, reader_id
    src = Path(src)
    ver = _jsonl(src / "verified.jsonl")
    rev = {r["seed"]["seed_id"]: r for r in _jsonl(src / "reviewed.jsonl")}
    readers, unparsable = {}, {}
    for k, sub in SOURCE_READERS:
        readers[k], unparsable[k] = _reader_verdicts(src / sub)
    veto = {r.get("seed_id") for r in _jsonl(src / "review_rejections.jsonl") if r.get("reason") == "reader_c_veto"}
    seeds = [r["seed"]["seed_id"] for r in ver]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{src / 'verified.jsonl'}: duplicate seed ids")
    have = set(seeds)
    any_reason, primary, kept_by, rows = Counter(), Counter(), Counter(), []
    for r in sorted(ver, key=lambda r: r["seed"]["seed_id"]):
        sid, split = r["seed"]["seed_id"], r["seed"]["split"]
        rid = reader_id(sid)
        v = {k: readers[k].get(rid, []) for k, _ in SOURCE_READERS}
        why = [w for w, hit in zip(EXCLUSIONS, (sid + REPAIR_SUFFIX in have, "BAD" in v["A"], "BAD" in v["B"],
                                                sid in veto)) if hit]
        any_reason.update(why)
        if why:
            primary[why[0]] += 1
            continue
        good = rev.get(sid)
        consensus = good is not None and (good["provenance"].get("review") or {}).get("consensus") is True
        kept_by[f"{split}|{'consensus' if consensus else 'non_consensus'}"] += 1
        if split not in splits:
            continue
        if consensus:
            if good["question"] != r["question"]:
                raise ValueError(f"{sid}: reviewed.jsonl question differs from verified.jsonl")
            rows.append(good)
            continue
        row = json.loads(json.dumps(r, ensure_ascii=False))
        row["provenance"]["review"] = {"verdict": "GOOD", "admitted_by": ADMITTED_BY,
                                       "readers": {k: (v[k][-1] if v[k] else None) for k, _ in SOURCE_READERS}}
        rows.append(row)
    test = [r for s, r in sorted(rev.items()) if r["seed"]["split"] == "test"]
    bad = [r["seed"]["seed_id"] for r in test if (r["provenance"].get("review") or {}).get("verdict") != "GOOD"
           or (r["provenance"].get("review") or {}).get("consensus") is not True or r["seed"]["seed_id"] in veto]
    if bad:
        raise ValueError(f"reviewed.jsonl test rows not strict-consensus GOOD: {bad[:3]}")
    rows = sorted(rows + test, key=lambda r: r["seed"]["seed_id"])
    if len({r["seed"]["seed_id"] for r in rows}) != len(rows):
        raise ValueError("a seed is both a SOURCE and a TEST row")
    rep = {"verified": len(ver), "excluded_primary": {k: primary.get(k, 0) for k in EXCLUSIONS},
           "excluded_any": {k: any_reason.get(k, 0) for k in EXCLUSIONS}, "excluded": sum(primary.values()),
           "source": sum(kept_by.values()), "source_by": dict(sorted(kept_by.items())),
           "source_in_splits": len(rows) - len(test), "splits": list(splits), "test_consensus": len(test),
           "reader_unread": {k: sum(1 for s in seeds if not readers[k].get(reader_id(s))) for k, _ in SOURCE_READERS},
           "reader_unparsable_lines": unparsable,
           "inputs_sha256": {n: _sha(src / n) for n in ("verified.jsonl", "reviewed.jsonl", "review_rejections.jsonl")}}
    return rows, rep


def build_source_dir(src, out_root, splits=("train", "val")) -> Path:
    """The staging dir snapshot() copies for pilot 5: verified.jsonl = reviewed.jsonl = verified_source rows (one
    canonical JSON line each), stats.json of the source, source.json = the exclusion report. Deterministic."""
    rows, rep = verified_source(src, splits)
    st = Path(out_root) / ".src-verified"
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    for n in ("verified.jsonl", "reviewed.jsonl"):
        (st / n).write_text(text, encoding="utf-8")
    shutil.copyfile(Path(src) / "stats.json", st / "stats.json")
    (st / SOURCE_JSON).write_text(json.dumps({"source": str(src), **rep}, indent=1, sort_keys=True) + "\n",
                                  encoding="utf-8")
    return st


def verified_snapshot(src, out_root, splits=("train", "val")) -> Path:
    """build_source_dir -> snapshot() (content-addressed by the staged reviewed.jsonl) + source.json beside it."""
    st = build_source_dir(src, out_root, splits)
    try:
        d = snapshot(st, out_root)
        if (d / SOURCE_JSON).exists() and (d / SOURCE_JSON).read_bytes() != (st / SOURCE_JSON).read_bytes():
            raise ValueError(f"{d / SOURCE_JSON}: differs from the rebuilt source report")
        shutil.copyfile(st / SOURCE_JSON, d / SOURCE_JSON)
    finally:
        shutil.rmtree(st)
    return d


def set_aside(path, split: str, out) -> int:
    """Copy the rows of `split` of a candidates file to `out` (before keep_splits drops them; Mac only)."""
    rows = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()
            and json.loads(l)["split"] == split]
    Path(out).write_text("".join(l + "\n" for l in rows), encoding="utf-8")
    return len(rows)


def keep_splits(path, splits) -> dict:
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r["split"] in splits]
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in kept), encoding="utf-8")
    c = Counter(r["split"] for r in kept)
    return {**{s: c.get(s, 0) for s in splits}, "dropped": len(rows) - len(kept)}


def dsn_port(dsn: str) -> int:
    """Port of a libpq DSN (URL or key=value form); 5432 when none is given. A multi-host URL raises ValueError."""
    if "://" in dsn:
        return urlsplit(dsn).port or 5432
    m = re.search(r"(?:^|\s)port\s*=\s*'?(\d+)", dsn)
    return int(m.group(1)) if m else 5432


def require_dev_pg(dsn: str, port: int = DEV_PG_PORT) -> None:
    try:
        got = dsn_port(dsn)
    except ValueError as e:
        raise RuntimeError(f"cannot read the Postgres port of GRL_PG_DSN ({e}); the pilot uses dev :{port} only") from e
    if got != port:
        raise RuntimeError(f"GRL_PG_DSN points at port {got}; the M5 pilot builds rankings on dev Postgres :{port} only"
                           " (5436 is production)")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_ship(ship, allowed_ids, test_ids=None, variant: str | None = None) -> None:
    """Every shipped row is train / val, never Q-real; every shipped ranking key is '<allowed id>|q' and not Q-real.
    ship/test_eval/ (pilot 5) may hold only test_eval.jsonl + rankings/**: rows of split test carrying the test-eval
    mark whose ids are `test_ids` (the finalized consensus test, disjoint from the train / val ids), keys '<id>|q'.
    M7: no id / row may be an M7 probe id (grl.hard.probe.probe_ids; the probe ships apart, check_probe_ship).
    M7 pilot 7: a row whose C_ref holds a STRUCT/<root> id (the hard Q1 gold) is accepted only when `variant` is
    "struct" (the ship's manifest variant; grl.hard.ship.struct_gold_ok)."""
    from grl.train.data import (ALLOWED_SPLITS, QREAL_ORIGIN, TEST_EVAL_MARK, TEST_EVAL_NAME, SplitGuardError,
                                guard_qkeys, qreal_ids)
    from grl.hard.probe import probe_ids
    ship, allowed, qids = Path(ship), set(allowed_ids), qreal_ids()
    pids = probe_ids()                          # M7: no probe item ever enters an old-set ship or its test-eval
    for ids_, what in ((allowed, "ids"), (set(test_ids or ()), "test ids")):
        if ids_ & pids:
            raise SplitGuardError(f"{ship}: probe ids among the {what}: {sorted(ids_ & pids)[:3]}")
    for n in ("trajectories.jsonl", "pairs.jsonl"):
        if (ship / n).exists():
            bad = [r.get("id") for r in _jsonl(ship / n) if r.get("split") not in ALLOWED_SPLITS or r.get("id") in pids
                   or r.get("origin") == QREAL_ORIGIN or r.get("id") in qids or r.get("id") not in allowed]
            if bad:
                raise SplitGuardError(f"{ship / n}: rows outside train / val or Q-real: {bad[:3]}")
            from grl.hard.ship import struct_gold_ok
            struct_gold_ok(_jsonl(ship / n), variant)
    for qf in sorted((ship / "rankings").rglob("queries.jsonl")):
        keys = [r["qkey"] for r in _jsonl(qf)]
        guard_qkeys(keys)
        bad = [k for k in keys if not k.endswith("|q") or k[:-2] not in allowed]
        if bad:
            raise SplitGuardError(f"{qf}: ranking keys outside the train / val candidates: {bad[:3]}")
    te = ship / TEST_EVAL_DIR
    if not te.exists():
        return
    if test_ids is None:
        raise SplitGuardError(f"{te}: a test-eval dir but no test ids to check it against")
    tids = set(test_ids)
    if tids & allowed:
        raise SplitGuardError(f"{te}: test ids overlap the train / val ids: {sorted(tids & allowed)[:3]}")
    names = sorted(p.relative_to(te).as_posix() for p in te.rglob("*") if p.is_file())
    extra = [n for n in names if n != TEST_EVAL_NAME and not n.startswith("rankings/")]
    if extra or TEST_EVAL_NAME not in names:
        raise SplitGuardError(f"{te}: holds only {TEST_EVAL_NAME} + rankings/** (got {extra[:3] or names[:3]})")
    bad = [r.get("id") for r in _jsonl(te / TEST_EVAL_NAME) if r.get("split") != "test" or r.get("id") in pids
           or r.get("origin") == QREAL_ORIGIN or r.get("id") in qids or r.get("id") not in tids
           or not (r.get("provenance") or {}).get(TEST_EVAL_MARK)]
    if bad:
        raise SplitGuardError(f"{te / TEST_EVAL_NAME}: rows that are not marked consensus-test rows: {bad[:3]}")
    from grl.hard.ship import struct_gold_ok
    struct_gold_ok(_jsonl(te / TEST_EVAL_NAME), variant)
    for qf in sorted((te / "rankings").rglob("queries.jsonl")):
        keys = [r["qkey"] for r in _jsonl(qf)]
        guard_qkeys(keys)
        bad = [k for k in keys if not k.endswith("|q") or k[:-2] not in tids]
        if bad:
            raise SplitGuardError(f"{qf}: test-eval ranking keys outside the test ids: {bad[:3]}")


def _repoint(latest: Path, final_dir: Path) -> None:
    from graphrag import config
    lt = json.loads(latest.read_text(encoding="utf-8"))
    final = final_dir / Path(lt["dir"]).name
    lt["dir"] = str(final.relative_to(config.ROOT)) if final.is_relative_to(config.ROOT) else str(final)
    latest.write_text(json.dumps(lt, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def build_ship(d, counts: dict, test_eval: bool = False, variant: str | None = None) -> dict:
    """<d>/ship/: trajectories.jsonl, pairs.jsonl (if present), rankings/** (latest.json re-pointed into ship/) and
    splits.json, plus manifest.json with their sha256s. test_eval=True (pilot 5) adds ship/test_eval/: the set-aside
    consensus test rows (TEST_CANDIDATES) marked provenance.test_eval and their rankings (TEST_RANKINGS) -- read only
    by train.py eval / baselines --split test; every training loader refuses the file. Built in a temp dir, checked
    (check_ship), then renamed."""
    from grl.train.data import TEST_EVAL_MARK, TEST_EVAL_NAME
    d = Path(d)
    allowed = {r["id"] for r in _jsonl(d / "candidates.jsonl") if r["split"] in counts}
    ship, tmp = d / SHIP_DIR, d / f".tmp-{SHIP_DIR}"
    for p in (tmp, ship):
        if p.exists():
            shutil.rmtree(p)
    tmp.mkdir()
    shutil.copyfile(d / "trajectories.jsonl", tmp / "trajectories.jsonl")
    if (d / "pairs.jsonl").exists():
        shutil.copyfile(d / "pairs.jsonl", tmp / "pairs.jsonl")
    (tmp / "splits.json").write_text(json.dumps(counts, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copytree(d / "rankings", tmp / "rankings")
    _repoint(tmp / "rankings" / "latest.json", ship / "rankings")
    test_ids = None
    if test_eval:
        rows = _jsonl(d / TEST_CANDIDATES)
        for r in rows:
            r.setdefault("provenance", {})[TEST_EVAL_MARK] = True
        test_ids = {r["id"] for r in rows}
        te = tmp / TEST_EVAL_DIR
        te.mkdir()
        (te / TEST_EVAL_NAME).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                 for r in sorted(rows, key=lambda r: r["id"])), encoding="utf-8")
        shutil.copytree(d / TEST_RANKINGS, te / "rankings")
        _repoint(te / "rankings" / "latest.json", ship / TEST_EVAL_DIR / "rankings")
    check_ship(tmp, allowed, test_ids, variant)
    files = sorted(p for p in tmp.rglob("*") if p.is_file())
    man = {"sha256": {str(p.relative_to(tmp)): _sha(p) for p in files},
           **({} if variant is None else {"variant": variant})}   # M7: check_extra passes it back to check_ship
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(tmp, ship)
    return man


def verify_ship(d) -> dict:
    ship = Path(d) / SHIP_DIR
    man = json.loads((ship / "manifest.json").read_text(encoding="utf-8"))
    files = {str(p.relative_to(ship)) for p in ship.rglob("*") if p.is_file()} - {"manifest.json"}
    bad = sorted(files ^ set(man["sha256"])) + [n for n, h in man["sha256"].items()
                                                  if (ship / n).exists() and _sha(ship / n) != h]
    if bad:
        raise ValueError(f"{ship}: differs from its manifest: {bad[:5]}")
    return man


def subset_rankings(latest, keys, out_root) -> str:
    from graphrag.retrieval.rankings import ARRAYS, load_rankings, write_rankings
    src = load_rankings(latest)
    keys = sorted(keys)
    missing = [k for k in keys if k not in set(src.keys())]
    if missing:
        raise KeyError(f"rankings have no {missing[:3]}")
    pos = [src.keys().index(k) for k in keys]
    arrays = {n: np.load(src.dir / f"{n}.npy")[pos] for n in ARRAYS}
    queries = [(k, src.get(k).text) for k in keys]
    inputs = {**src.meta["inputs"], "subset_of": src.meta["version"]}
    return write_rankings(Path(out_root), queries, arrays, inputs)
