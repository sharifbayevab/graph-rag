"""M7 Task 7: the probe snapshot (docs/m7 design §4). The 90 probe HardItems (count / version / compare) get
split "probe" -- a split no training / selection loader accepts (grl.train.data.guard_splits allows only train / val;
the loaders refuse a probe file outright, refuse_probe_file) -- and ship as <snap>/ship/: pairs.jsonl (HardItem
fields + split), splits.json, rankings/** (scripts/build_rankings.py --family probe: the U9 path and settings of the
old set) and manifest.json. Gold is STRUCT/<root> or corpus chunk ids; STRUCT ids are in no index, so a ranking never
holds a row >= n_base_chunks (check_rank_rows). The probe ids (data/hard/probe/ids.json, Mac only) are also refused
by the old-set check_ship, so no probe item enters an old-set ship or its test-eval file. A missing ids.json fails
hard on the Mac; on Spark (no ids.json) the guards fall back to split / qid checks (probe_ids)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROBE_SPLIT = "probe"
PROBE_IDS_PATH = "data/hard/probe/ids.json"       # Mac only (data/hard is never synced); see probe_ids
HOST_VAR, SPARK_HOST = "GRL_HOST", "spark"         # scripts/spark.py exports GRL_HOST=spark on every remote command
PROBE_SOURCES = ("count.jsonl", "version.jsonl", "compare.jsonl")
PROBE_SHIP_FILES = ("pairs.jsonl", "splits.json", "manifest.json")    # + rankings/**
STRUCT_PREFIX = "STRUCT/"
HARD_FIELDS = ("qid", "qtype", "template", "lang", "question", "answer", "gold", "docs")


def _jsonl(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


@lru_cache(maxsize=4)
def _probe_ids(path: str) -> frozenset:
    """Raises FileNotFoundError when absent; lru_cache does not cache a raise, so only a read file is cached."""
    from graphrag import config
    return frozenset(json.loads(config.path(path).read_text(encoding="utf-8")))


def on_spark() -> bool:
    return os.environ.get(HOST_VAR) == SPARK_HOST


def probe_ids() -> frozenset:
    """The M7 probe qids (data/hard/probe/ids.json). Mac: the file is required -- missing or empty is a
    SplitGuardError, never a silent empty set (R30), so every guard below really checks the ids. Spark (GRL_HOST=spark,
    exported by scripts/spark.py): ids.json is not shipped, the result is the empty set and the callers fall back to
    their split / qid guards (split "probe" rows, refuse_probe_file's split / qid test, check_probe_ship with the ship's
    own qids in scripts/hard_probe_eval.py) -- as before R30."""
    try:
        ids = _probe_ids(PROBE_IDS_PATH)
    except FileNotFoundError:
        if on_spark():
            return frozenset()
        raise _guard_error()(f"{PROBE_IDS_PATH} is missing: the probe ids are required off Spark (set "
                             f"{HOST_VAR}={SPARK_HOST} only on Spark, where scripts/spark.py does it)") from None
    if not ids:
        raise _guard_error()(f"{PROBE_IDS_PATH} holds no probe ids")
    return ids


def _guard_error():
    from grl.train.data import SplitGuardError
    return SplitGuardError


def probe_rows(paths, ids) -> list[dict]:
    """The probe HardItems of `paths` (count / version / compare jsonl) with split "probe", sorted by qid; their qids
    must be exactly `ids` (data/hard/probe/ids.json) -- no extra, none missing, no duplicate."""
    out = []
    for p in paths:
        for r in _jsonl(p):
            missing = [k for k in HARD_FIELDS if k not in r]
            if missing:
                raise ValueError(f"{p}: row {r.get('qid')} is not a HardItem (missing {missing})")
            if r.get("split", PROBE_SPLIT) != PROBE_SPLIT:
                raise ValueError(f"{p}: row {r['qid']} has split {r['split']!r}, not {PROBE_SPLIT!r}")
            out.append({**r, "split": PROBE_SPLIT})
    q = [r["qid"] for r in out]
    if len(q) != len(set(q)):
        raise ValueError("duplicate probe qids")
    if set(q) != set(ids):
        raise ValueError(f"probe rows != ids.json: extra {sorted(set(q) - set(ids))[:3]}, "
                         f"missing {sorted(set(ids) - set(q))[:3]}")
    return sorted(out, key=lambda r: r["qid"])


def probe_queries(rows) -> list[tuple[str, str]]:
    """Ranking queries (qkey "<qid>|q", question text), as graphrag.retrieval.rankings.queries_from_pairs."""
    return sorted((f"{r['qid']}|q", r["question"]) for r in rows)


def read_probe_rows(path) -> list[dict]:
    """A shipped probe pairs file: every row split "probe", never Q-real (id / origin), qid among the probe ids (when
    ids.json is present: Mac); Q-real-shaped keys refused as everywhere."""
    from grl.train import data as td
    err, qids, pids = _guard_error(), td.qreal_ids(), probe_ids()
    rows = _jsonl(path)
    bad = [r.get("qid") for r in rows if r.get("split") != PROBE_SPLIT or r.get("origin") == td.QREAL_ORIGIN
           or r.get("qid") in qids or (pids and r.get("qid") not in pids)]
    if bad:
        raise err(f"{path}: rows that are not probe rows (split {PROBE_SPLIT!r}, probe ids, no Q-real): {bad[:3]}")
    td.guard_qkeys([f"{r['qid']}|q" for r in rows])
    return sorted(rows, key=lambda r: r["qid"])


def refuse_probe_file(path) -> None:
    """Training / labelling / selection loaders never read a probe file: any row of split "probe" or with a HardItem
    qid (a probe row cannot parse as a §8.1 pair anyway; this makes the refusal a SplitGuardError, not a TypeError)."""
    p = Path(path)
    if not p.exists():
        return
    pids = probe_ids()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if isinstance(r, dict) and (r.get("split") == PROBE_SPLIT or "qid" in r or r.get("id") in pids):
            raise _guard_error()(f"{path}: probe rows (split {PROBE_SPLIT!r}) are read only by the probe eval "
                                 f"(grl.hard.probe.load_probe_items); never by training, labelling or selection")


def load_probe_items(path, data) -> list:
    """Probe rows -> TrainItems (qkey "<qid>|q", split "probe", pattern = qtype, gold = chunk rows of `data`, no
    trajectory). STRUCT gold needs the struct EnvData (grl.env.struct_data.load_struct_env_data)."""
    from grl.train.data import TrainItem
    out = []
    for r in read_probe_rows(path):
        miss = [g for g in r["gold"] if g not in data.chunk_row]
        if miss:
            hint = " (STRUCT gold: load the struct EnvData)" if any(g.startswith(STRUCT_PREFIX) for g in miss) else ""
            raise ValueError(f"{path}: {r['qid']} gold not in the env rows: {miss[:3]}{hint}")
        if not r["gold"]:
            raise ValueError(f"{path}: {r['qid']} has an empty gold")
        out.append(TrainItem(r["qid"], f"{r['qid']}|q", PROBE_SPLIT, r["qtype"], r["lang"],
                             frozenset(int(data.chunk_row[g]) for g in r["gold"]), (), (), ()))
    return out


def check_probe_ship(ship, allowed=None) -> None:
    """A probe ship/ holds only pairs.jsonl, splits.json, manifest.json and rankings/**; every row is a probe row with
    an allowed id (default: the probe ids) and a non-empty gold; every ranking key is '<allowed id>|q'. The rankings
    are not touched by the gold (STRUCT ids are in no index)."""
    from grl.train import data as td
    err, ship = _guard_error(), Path(ship)
    allowed = set(probe_ids() if allowed is None else allowed)
    if not allowed:
        raise err(f"{ship}: no probe ids to check against ({PROBE_IDS_PATH} missing)")
    names = sorted(p.relative_to(ship).as_posix() for p in ship.rglob("*") if p.is_file())
    extra = [n for n in names if n not in PROBE_SHIP_FILES and not n.startswith("rankings/")]
    if extra or "pairs.jsonl" not in names:
        raise err(f"{ship}: a probe ship holds only {PROBE_SHIP_FILES} + rankings/** (got {extra[:3] or names[:3]})")
    qids = td.qreal_ids()
    rows = _jsonl(ship / "pairs.jsonl")
    bad = [r.get("qid") for r in rows if r.get("split") != PROBE_SPLIT or r.get("origin") == td.QREAL_ORIGIN
           or r.get("qid") in qids or r.get("qid") not in allowed]
    if bad:
        raise err(f"{ship / 'pairs.jsonl'}: rows outside the probe: {bad[:3]}")
    bad = [r["qid"] for r in rows if not r.get("gold") or not all(isinstance(g, str) and g for g in r["gold"])]
    if bad:
        raise err(f"{ship / 'pairs.jsonl'}: rows with an empty / malformed gold: {bad[:3]}")
    for qf in sorted((ship / "rankings").rglob("queries.jsonl")):
        keys = [r["qkey"] for r in _jsonl(qf)]
        td.guard_qkeys(keys)
        bad = [k for k in keys if not k.endswith("|q") or k[:-2] not in allowed]
        if bad:
            raise err(f"{qf}: ranking keys outside the probe ids: {bad[:3]}")


def check_rank_rows(ranks, keys, n_base: int) -> dict:
    """Review Focus 1 on real data: every key has a non-empty lexical and dense ranking, and no ranked chunk row is
    >= n_base (a STRUCT row): STRUCT fragments are in no index."""
    missing = [k for k in keys if k not in set(ranks.keys())]
    if missing:
        raise ValueError(f"rankings missing keys {missing[:3]}")
    empty_lex = [k for k in keys if len(ranks.get(k).lex) == 0]
    empty_vec = [k for k in keys if len(ranks.get(k).vec) == 0]
    if empty_lex or empty_vec:
        raise ValueError(f"empty rankings: lex {empty_lex[:3]}, vec {empty_vec[:3]}")
    hi = [(k, int(r)) for k in keys for ch in ("lex", "vec") for r in getattr(ranks.get(k), ch) if int(r) >= n_base]
    if hi:
        raise ValueError(f"ranked rows >= n_base_chunks {n_base}: {hi[:3]}")
    mx = max(int(max(ranks.get(k).lex.max(), ranks.get(k).vec.max())) for k in keys) if keys else -1
    return {"n": len(keys), "empty_lex": 0, "empty_vec": 0, "rows_ge_n_base": 0, "max_row": mx}


def gold_in_top(ranks, rows, chunk_ids, k: int = 50) -> dict:
    """Per qtype: share of questions with any / all gold in the top-k of q.lex and q.vec, and the mean gold recall.
    A STRUCT gold id is never a ranked chunk id, so it always counts as missed."""
    by = {}
    for r in rows:
        q = ranks.get(f"{r['qid']}|q")
        g = set(r["gold"])
        rec = by.setdefault(r["qtype"], {"n": 0, **{f"{c}_{m}": 0.0 for m in ("any", "all", "recall")
                                                     for c in ("lex", "vec")}})
        rec["n"] += 1
        for ch in ("lex", "vec"):
            top = {chunk_ids[int(i)] for i in getattr(q, ch)[:k]}
            hit = len(g & top)
            rec[f"{ch}_any"] += float(hit > 0)
            rec[f"{ch}_all"] += float(hit == len(g))
            rec[f"{ch}_recall"] += hit / len(g)
    out = {}
    for t, rec in sorted(by.items()):
        n = rec["n"]
        out[t] = {"n": n, **{key: round(v / n, 4) for key, v in sorted(rec.items()) if key != "n"}}
    return out
