"""Shared helpers of the cached LLM judges (Ruling R31 (2)): the Task 7 edge audit (graphrag.kg.audit, pass
"audit") and the Task 8 RELATED judge (scripts/research_related.py, pass "audit_related").

The prompt, the schema and the batch input are what KgLLM hashes into the cache key, so these helpers build
byte-for-byte the same objects the two modules built before (no cached answer is invalidated):

    verdict_schema(fields)        {"verdicts": [{id, **fields}]}, required = ["id", *fields] in that order
    verdict_checker(ids, ok, exact_count)   verify_fn: every id exactly once (+ len check), per-verdict fields
    batch_input(batch)            {"items": [{"id": "e1", **view}, ...]}
    shuffled_batches(items, size, rng_key)  items sorted by uid, random.Random(rng_key).shuffle, cut in `size`
    make_judge(llm, pass_name, prompt, schema, checker, convert)   judge(batch) -> {uid: verdict | None}
    cap_middle(s, cap) / cap_tail(s, cap)   the two text caps of the judge views"""
from __future__ import annotations

import random
import re
from typing import Any, Callable


def verdict_schema(fields: dict) -> dict:
    v = {"type": "object", "additionalProperties": False, "required": ["id", *fields],
         "properties": {"id": {"type": "string"}, **fields}}
    return {"type": "object", "additionalProperties": False, "required": ["verdicts"],
            "properties": {"verdicts": {"type": "array", "items": v}}}


def verdict_checker(ids: list[str], fields_ok: Callable[[dict], bool],
                    exact_count: bool = False) -> Callable[[Any], tuple[bool, str | None]]:
    """verify_fn for KgLLM.cached_json: "shape" (no verdicts list), "ids" (the ids are not `ids` exactly; with
    exact_count also a different number of verdicts), "fields" (a verdict fails `fields_ok`)."""
    def check(ans):
        vs = ans.get("verdicts") if isinstance(ans, dict) else None
        if not isinstance(vs, list):
            return False, "shape"
        got = [v.get("id") for v in vs if isinstance(v, dict)]
        if sorted(got) != sorted(ids) or (exact_count and len(vs) != len(ids)):
            return False, "ids"
        if any(not fields_ok(v) for v in vs):
            return False, "fields"
        return True, None
    return check


def batch_input(batch: list[dict]) -> dict:
    return {"items": [{"id": f"e{k + 1}", **it["view"]} for k, it in enumerate(batch)]}


def shuffled_batches(items: list[dict], size: int, rng_key: str) -> list[list[dict]]:
    its = sorted(items, key=lambda x: x["uid"])
    random.Random(rng_key).shuffle(its)
    return [its[i:i + size] for i in range(0, len(its), size)]


def make_judge(llm, pass_name: str, prompt: str, schema: dict,
               checker: Callable[[list[str]], Callable], convert: Callable[[dict], dict]
               ) -> Callable[[list[dict]], dict]:
    """judge(batch) -> {uid: convert(verdict) | None} via KgLLM.cached_json (cache, gate, budget, ledger)."""
    def judge(batch):
        inp = batch_input(batch)
        ids = [x["id"] for x in inp["items"]]
        ans = llm.cached_json(pass_name, inp, schema, checker(ids), prompt=prompt)
        if ans is None:
            return {it["uid"]: None for it in batch}
        by_id = {v["id"]: v for v in ans["verdicts"]}
        return {it["uid"]: convert(by_id[i]) for i, it in zip(ids, batch)}
    return judge


def cap_middle(s: str, cap: int) -> str:
    """Head and tail of `s` around " … " (the edge-audit views)."""
    return s if len(s) <= cap else s[:cap // 2] + " … " + s[-cap // 2:]


def cap_tail(s: str, cap: int) -> str:
    """Whitespace runs collapsed, then the first `cap` characters + " …" (the RELATED judge views)."""
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s if len(s) <= cap else s[:cap].rstrip() + " …"
