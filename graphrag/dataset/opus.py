"""Opus generator and strict reader (user decision M3-R23). Opus agents generate the question-answer pairs. They use
the same seeds, facts, neighbours and code checks. gpt-4.1 verifies the pairs cross-model (blind judge + per-fact
ablation + plants, pilot first). An independent Opus strict reader, a different agent from the generator, reads every
verified pair, and only GOOD pairs go on to finalize / select.

- Generator WORK PACKETS (`make_packets` / `packet_doc`):
  - ~40 seeds of one split × pattern each (split-pure; order sha256(seed:opus|packet|id));
  - per seed: the cell (split, pattern, lang), the full fragment texts, act / where / kind / amended acts, and the
    cached ds_facts facts (snap-to-source values);
  - for a multi seed: an ANCHOR plus the <= 8 ds_bridge neighbour candidates (graphrag.dataset.bridge.shortlist) with
    texts and edge types;
  - the GENERATOR INSTRUCTIONS (config/dataset_opus_generator.md, versioned, sha256 digest recorded).
- `ingest_generated`: reads the generator's JSONL (the committed file, so the replay is offline and byte-identical)
  and runs EVERY code check of the gpt-4.1 path:
  - verbatim quotes, snapped to source (facts.snap_fact);
  - generate.check_item_ex: typed values, chain / redundant, answer leak, numbers, script purity, deixis, compound,
    article-without-act, foreign names, wrong act, copy / overlap at the calibrated thresholds;
  - for multi, the needed fact must be the neighbour's.

  It records provenance (packet id, instruction digest, role opus-generator, file, line, row sha256) and returns
  generate.Candidate objects for ds_verify.
- REVIEWER PACKETS (`make_review_packets`): the verified pair plus its fragments plus the READER RUBRIC
  (config/dataset_reader_rubric.md). They never carry generator notes, facts, sources or verify verdicts.
- `ingest_verdicts`: reads {pair_id, verdict GOOD | BORDERLINE | BAD, reasons}. Only GOOD pairs are kept.
  Rejections are logged per reason (criterion) and per cell."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from graphrag.dataset.facts import Fact, snap_fact, value_type
from graphrag.dataset.seeds import Seed, _h
from graphrag.eval.gold import wsn

ROLE_GEN = "opus-generator"
ROLE_READER = "strict-reader"             # M3-R31: reader packets never name a model (was "opus-reader")
FORMAT = 1
VERDICTS = ("GOOD", "BORDERLINE", "BAD")
CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")
KIND = {"QONUNLAR": "law", "QARORLAR": "Senate decision", "XULOSALAR": "committee conclusion",
        "STENOGRAMMA": "plenary transcript", "BOSHQA": "other document"}
LANG_NAME = {"uz-Cyrl": "Uzbek in Cyrillic script", "uz-Latn": "Uzbek in Latin script (oʻ gʻ with ʻ)", "ru": "Russian"}
OUTPUT_SCHEMA = ('{"seed_id", "question", "answer", "lang", "used_facts": [{"fact_id", "value"} | '
                 '{"fragment", "quote", "value"}], "neighbor_id": null | "N<k>", "sources": ["F1"] | ["A", "N<k>"], '
                 '"notes"}  or  {"seed_id", "skip": true, "reason"}')
VERDICT_SCHEMA = ('{"pair_id", "verdict": "GOOD" | "BORDERLINE" | "BAD", '
                  '"reasons": [{"criterion": "c1".."c7", "note"}]}')


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instructions(path, version: int, rel: str | None = None) -> dict:
    """{path, version, digest, text} of a versioned instruction file (digest = sha256 of its UTF-8 bytes)."""
    text = Path(path).read_text(encoding="utf-8")
    return {"path": rel or str(path), "version": int(version), "digest": sha256_text(text), "text": text}


def check_instructions(doc: dict) -> None:
    """A packet's embedded instructions must be intact (sha256(text) == digest)."""
    if sha256_text(doc["text"]) != doc["digest"]:
        raise ValueError(f"instructions {doc.get('path')}: embedded text does not match its digest")


def kind_of(chunk_id: str) -> str:
    return KIND.get(chunk_id.split("-", 1)[0], "other document")


def fragment(cid: str, label: str, chunks_by_id, index, amended=None, session=None) -> dict:
    """session (M3-R29): cid -> {"session": {date, title}, "speaker": ...} for plenary-transcript fragments, or None."""
    c = chunks_by_id[cid]
    out = {"id": label, "chunk_id": cid, "kind": kind_of(cid), "act": index.act[c.parent.split("/", 1)[0]],
           "where": index.format(c.parent), "amended": list(amended(cid)) if amended else [], "text": c.text}
    extra = session(cid) if session else None
    if extra:
        out.update(extra)
    return out


def session_info(records):
    """M3-R29: cid -> {session: {date, title}, speaker} for STENOGRAMMA fragments (record meta; the speech vertex's
    title is the speaker's name or role)."""
    title = {n.node_id: n.title for r in records for n in r.structure}
    meta = {r.doc_id: r.meta for r in records}

    def of(cid: str):
        doc = cid.split("/", 1)[0]
        m = meta.get(doc)
        if m is None or m.section != "STENOGRAMMA":
            return None
        parts = cid.split("#", 1)[0].split("/")
        spk = next((title.get("/".join(parts[:i + 1])) for i in range(len(parts) - 1, 0, -1)
                    if parts[i].startswith("ВЫСТ.") and title.get("/".join(parts[:i + 1]))), None)
        from graphrag.dataset.generate import session_iso
        out = {"session": {"date": m.date, "title": m.title}}
        days = session_iso(out["session"])                 # M3-R29 fix: full ISO dates (titles may say "28-29.05.25")
        if days:
            out["session"]["dates"] = days
            out["session"]["range"] = f"{days[0]}..{days[-1]}" if len(days) > 1 else days[0]
        if spk:
            out["speaker"] = spk
        return out
    return of


def fact_rows(facts, label: str) -> list[dict]:
    """Packet facts of one fragment: short labels "<fragment>.<k>" plus the ds_facts provenance (`src`)."""
    return [{"fact_id": f"{label}.{k}", "type": f.vtype or value_type(f.value), "statement": f.statement,
             "value": f.value, "quote": f.quote, "src": {"fact_id": f.fact_id, "block": f.block,
                                                         "cache_key": f.cache_key}}
            for k, f in enumerate(facts)]


def single_entry(seed: Seed, chunks_by_id, index, facts: dict, amended=None, session=None) -> dict:
    """facts: chunk_id -> [Fact] for the fragments whose ds_facts block is cached (absent = not cached)."""
    cid = seed.chunks[0]
    return {"seed_id": seed.seed_id, "cell": {"split": seed.split, "pattern": "single", "lang": seed.lang},
            "LANG": LANG_NAME[seed.lang], "section": seed.section,
            "fragments": [fragment(cid, "F1", chunks_by_id, index, amended, session)],
            "facts": fact_rows(facts.get(cid) or [], "F1"), "facts_cached": cid in facts,
            "seed": seed.to_json()}


def anchor_seed_id(seed, anchor: str) -> str:
    return f"anchor-{_h(seed, 'opus|' + anchor)[:10]}"


def multi_entry(pick: dict, chunks_by_id, index, facts: dict, seed, amended=None, dup=None, session=None) -> dict:
    """pick: bridge.sample_anchors row {anchor, stratum, lang, split, candidates}; neighbours = bridge.shortlist.
    dup (seeds.dup_content, M3-R25): a candidate with the anchor's own text (a duplicate document) is no neighbour."""
    from graphrag.dataset.bridge import RELATION, shortlist
    a = pick["anchor"]
    if dup and a in dup:
        same = set(dup[a]["twins"])
        pick = {**pick, "candidates": {s: [x for x in v if x[0] not in same] for s, v in pick["candidates"].items()}}
        pick["candidates"] = {s: v for s, v in pick["candidates"].items() if v}
    nbs, rows, cached = [], fact_rows(facts.get(a) or [], "A"), {"A": a in facts}
    for i, (src, c, hops) in enumerate(shortlist(pick["candidates"], pick["stratum"])):
        lab = f"N{i + 1}"
        nbs.append({**fragment(c, lab, chunks_by_id, index, amended, session), "source": src, "relation": RELATION[src],
                    "edges": [t for _, t, _ in hops], "hops": [list(h) for h in hops]})
        rows += fact_rows(facts.get(c) or [], lab)
        cached[lab] = c in facts
    return {"seed_id": anchor_seed_id(seed, a), "cell": {"split": pick["split"], "pattern": "multi", "lang": pick["lang"]},
            "LANG": LANG_NAME[pick["lang"]], "stratum": pick["stratum"],
            "anchor": fragment(a, "A", chunks_by_id, index, amended, session), "neighbors": nbs, "facts": rows,
            "facts_cached": cached}


def packet_order(seed, entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: _h(seed, "opus|packet|" + e["seed_id"]))


def make_packets(entries: list[dict], size: int, prefix: str, seed) -> list[tuple[str, list[dict]]]:
    """Split-pure, pattern-pure packets of <= size seeds: (packet_id, entries), entries in sha256 order; the last
    packets are balanced (ceil(n / size) packets of near-equal size)."""
    out = []
    groups = defaultdict(list)
    for e in entries:
        groups[(e["cell"]["pattern"], e["cell"]["split"])].append(e)
    for (pat, sp), es in sorted(groups.items()):
        es = packet_order(seed, es)
        k = math.ceil(len(es) / int(size))
        bounds = [round(i * len(es) / k) for i in range(k + 1)]
        for i in range(k):
            out.append((f"{prefix}{pat}-{sp}-{i + 1:03d}", es[bounds[i]:bounds[i + 1]]))
    return out


def packet_doc(packet_id: str, entries: list[dict], instr: dict, out_path: str) -> dict:
    cells = {e["cell"]["pattern"] for e in entries}, {e["cell"]["split"] for e in entries}
    if len(cells[0]) != 1 or len(cells[1]) != 1:
        raise ValueError(f"{packet_id}: a packet is split- and pattern-pure")
    return {"packet_id": packet_id, "format": FORMAT, "role": ROLE_GEN, "pattern": next(iter(cells[0])),
            "split": next(iter(cells[1])), "n_seeds": len(entries),
            "langs": dict(sorted(Counter(e["cell"]["lang"] for e in entries).items())),
            "instructions": instr, "output": {"path": out_path, "format": "jsonl", "schema": OUTPUT_SCHEMA},
            "seeds": entries}


def wave_order(packet_ids: list[str], group_of) -> list[str]:
    """Packets interleaved proportionally over their (pattern, split) groups: a prefix of the order covers every
    group in proportion (dispatch waves)."""
    g = defaultdict(list)
    for p in sorted(packet_ids):
        g[group_of(p)].append(p)
    return [p for _, _, p in sorted(((i + 0.5) / len(ps), k, p) for k, ps in g.items() for i, p in enumerate(ps))]


# ---- ingest-gen --------------------------------------------------------------------------------------------------------
def _frag_check(f: dict, label: str) -> dict:
    out = {"id": label, "act": f["act"], "where": f["where"], "text": f["text"], "amended": list(f.get("amended") or ())}
    for k in ("discussed_act", "amended_act", "amended_unit"):
        if f.get(k):
            out[k] = f[k]
    if f.get("session"):
        out["session"] = f["session"]
    if f.get("speaker"):
        out["speaker"] = f["speaker"]
    return out


def _parse_jsonl(text: str):
    for n, line in enumerate((text or "").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            yield n, None, line
            continue
        yield n, row, line


def ingest_generated(packet: dict, text: str | None, chunks_by_id, cfg: dict, doc_section: dict, gen_path: str,
                     agent_model: str = "claude-opus", generated_under: dict | None = None) -> tuple[list, list[dict], dict]:
    """One packet's generator output -> (Candidates, rejection rows, stats). `text` None = no output yet (pending).
    Rejection rows: {packet_id, seed_id, cell, reason, line, detail}; a skip is logged with reason `skip`.
    generated_under ({version, digest}; gen/<id>.meta.json): the instructions the output was written under when
    they differ from the packet's (an output kept across an instruction version, M3-R25) -- recorded as provenance.
    M3-R25: a value longer than the judge quote cap (cfg verify_quote_cap[0]) can never be recovered -> rejected."""
    from graphrag.dataset.generate import Candidate, cell_of, check_item_ex, exempt_words, lexical_overlap, max_copy_run
    from graphrag.dataset.walks import walk_class
    check_instructions(packet["instructions"])
    instr = {k: packet["instructions"][k] for k in ("path", "version", "digest")}
    packet_instr = dict(instr)
    if generated_under:
        instr = {"path": instr["path"], "version": int(generated_under["version"]), "digest": generated_under["digest"]}
    vcap = int((cfg.get("verify_quote_cap") or [10 ** 9])[0])
    pid = packet["packet_id"]
    entries = {e["seed_id"]: e for e in packet["seeds"]}
    for e in packet["seeds"]:                               # the packet must match the corpus it was built from
        for f in ([e["anchor"]] + e["neighbors"]) if "anchor" in e else e["fragments"]:
            if chunks_by_id[f["chunk_id"]].text != f["text"]:
                raise ValueError(f"{pid}: fragment {f['chunk_id']} text differs from the corpus (stale packet)")
    out, rej, seen = [], [], set()
    cellk = lambda e: f"{e['cell']['split']}|{e['cell']['pattern']}|{e['cell']['lang']}"

    def reject(e, reason, line=None, detail=""):
        rej.append({"packet_id": pid, "seed_id": e["seed_id"] if e else None, "cell": cellk(e) if e else None,
                    "reason": reason, "line": line, "detail": str(detail)[:300]})
    if text is None:
        return [], [], {"pending": True, "seeds": len(entries)}
    for n, row, raw in _parse_jsonl(text):
        if row is None or not isinstance(row, dict):
            reject(None, "bad_json", n, raw)
            continue
        e = entries.get(row.get("seed_id"))
        if e is None:
            reject(None, "unknown_seed", n, row.get("seed_id"))
            continue
        if e["seed_id"] in seen:
            reject(e, "duplicate_seed", n)
            continue
        seen.add(e["seed_id"])
        if row.get("skip") is True:
            reject(e, "skip", n, row.get("reason", ""))
            continue
        need = {"question": str, "answer": str, "lang": str, "used_facts": list, "sources": list}
        if any(not isinstance(row.get(k), t) for k, t in need.items()) or "neighbor_id" not in row \
                or not isinstance(row.get("notes", ""), str):
            reject(e, "schema", n)
            continue
        if row["lang"] != e["cell"]["lang"]:
            reject(e, "lang_mismatch", n, row["lang"])
            continue
        multi = e["cell"]["pattern"] == "multi"
        if multi:
            nb = {x["id"]: x for x in e["neighbors"]}
            nid = row["neighbor_id"]
            if nid not in nb:
                reject(e, "unknown_neighbor", n, nid)
                continue
            if sorted(row["sources"]) != sorted(["A", nid]):
                reject(e, "sources_mismatch", n, row["sources"])
                continue
            a, x = e["anchor"], nb[nid]
            allowed = {"A": (a, "F1"), nid: (x, "F2")}
            chunks = (a["chunk_id"], x["chunk_id"])
            docs = tuple(dict.fromkeys(c.split("/", 1)[0] for c in chunks))
            seed = Seed(f"multi-{_h(cfg['seed'], '|'.join(chunks))[:10]}", "multi", e["cell"]["lang"], chunks, docs,
                        e["cell"]["split"], doc_section[docs[0]], tuple(tuple(h) for h in x["hops"]))
        else:
            if row["neighbor_id"] is not None or row["sources"] != ["F1"]:
                reject(e, "sources_mismatch", n, [row["neighbor_id"], row["sources"]])
                continue
            f1 = e["fragments"][0]
            allowed = {"F1": (f1, "F1")}
            seed = Seed.from_json(e["seed"])
        row_sha = sha256_text(raw.strip())
        ffacts = {x["fact_id"]: x for x in e["facts"]}
        labels, bad = {}, None
        for k, u in enumerate(row["used_facts"]):
            if not isinstance(u, dict):
                bad = ("schema", u)
                break
            if u.get("fact_id") is not None:
                fr = str(u["fact_id"]).split(".", 1)[0]
                src = ffacts.get(u["fact_id"])
                if fr not in allowed or src is None:
                    bad = ("unknown_fact", u["fact_id"])
                    break
                if wsn(str(u.get("value", ""))) != wsn(src["value"]):
                    bad = ("fact_value_mismatch", u["fact_id"])
                    break
                f = Fact(src["src"]["fact_id"], allowed[fr][0]["chunk_id"], src["statement"], src["value"],
                         src["quote"], src["src"]["block"], src["src"]["cache_key"], src["type"])
                labels[f"{allowed[fr][1]}.q{k}"] = f
                continue
            q, v = u.get("quote"), u.get("value")
            if not isinstance(q, str) or not isinstance(v, str):
                bad = ("schema", u)
                break
            fr = u.get("fragment")
            if fr is None:                                  # the fragment whose text holds the quote
                fr = next((lab for lab, (g, _) in allowed.items() if wsn(q) in wsn(g["text"])), None)
            if fr not in allowed:
                bad = ("fact_wrong_fragment", fr)
                break
            g, flab = allowed[fr]
            cid = g["chunk_id"]
            snapped, r = snap_fact({"statement": "", "quote": q, "value": v}, chunks_by_id[cid].text,
                                   exempt_words([_frag_check(g, flab)]))
            if r:
                bad = (f"fact_{r}", q)
                break
            labels[f"{flab}.q{k}"] = Fact(f"{cid}@opus-{seed.seed_id}-{k}", cid, "", snapped["value"].strip(),
                                          snapped["quote"], "", row_sha, value_type(snapped["value"]))
        if not bad:
            long = next((f.value for f in labels.values() if len(wsn(f.value)) > vcap), None)
            if long is not None:
                bad = ("value_over_cap", long[:80])
        if bad:
            reject(e, bad[0], n, bad[1])
            continue
        if not labels:
            reject(e, "value_missing_in_answer", n)
            continue
        frags = [_frag_check(g, flab) for g, flab in sorted(allowed.values(), key=lambda t: t[1])]
        item = {"key": e["seed_id"], "question": row["question"], "answer": row["answer"], "used": list(labels)}
        r, info = check_item_ex(item, seed, labels, frags, cfg, path_chunks={seed.chunks[0]} if multi else None)
        if r:
            reject(e, r, n)
            continue
        if multi and not any(f.chunk_id == seed.chunks[1] for f in info["answer"]):
            reject(e, "needed_not_in_answer", n)
            continue
        used = list(dict.fromkeys(info["answer"] + info["path"]))
        lab_of = {id(f): lab for lab, f in labels.items()}
        q = row["question"].strip()
        ex = exempt_words(frags)
        ds = [f for f in used if f.block]
        keys = {f.block: f.cache_key for f in sorted(ds, key=lambda f: f.block)}
        if any(not f.block for f in used):
            keys[ROLE_GEN] = row_sha
        from graphrag.dataset.facts import DEFAULT_MAX_FACTS, SCHEMA_FACTS, facts_prompt
        from graphrag.kg.llm import prompt_digest
        fdig = prompt_digest(facts_prompt(int(cfg.get("max_facts_per_chunk", DEFAULT_MAX_FACTS))), SCHEMA_FACTS)
        prov = {"seed_id": seed.seed_id, "cell": cell_of(seed), "chunks": list(seed.chunks),
                "walk": [list(h) for h in seed.walk], "fact_ids": [f.fact_id for f in used],
                "fact_labels": [lab_of[id(f)] for f in used], "listed_labels": list(labels),
                "answer_fact_ids": [f.fact_id for f in info["answer"]], "path_fact_ids": [f.fact_id for f in info["path"]],
                "value_checks": dict(info["status"]), "walk_class": walk_class(seed.walk) if seed.walk else None,
                "facts": {"pass": "ds_facts" if len(ds) == len(used) else ROLE_GEN,
                          "prompt_digest": fdig if ds else instr["digest"], "cache_keys": keys},
                "generate": {"pass": ROLE_GEN, "role": ROLE_GEN, "model": agent_model,
                             "prompt_version": instr["version"], "prompt_digest": instr["digest"],
                             "cache_key": row_sha, "item": e["seed_id"], "packet_id": pid},
                "opus": {"role": ROLE_GEN, "packet_id": pid, "instructions": instr, "packet_instructions": packet_instr,
                         "gen_file": gen_path, "line": n,
                         "row_sha256": row_sha, "packet_seed_id": e["seed_id"], "neighbor_id": row["neighbor_id"],
                         "sources": list(row["sources"]), "notes": row.get("notes", ""),
                         "agent": row.get("agent")},
                "copy": {"max_run": max(max_copy_run(q, x["text"], ex) for x in frags),
                         "overlap": round(lexical_overlap(q, [x["text"] for x in frags],
                                                          " ".join(f"{x['act']} {x['where']}" for x in frags)), 4)}}
        if multi:
            prov.update(bridge_source=e["neighbors"][[x["id"] for x in e["neighbors"]].index(row["neighbor_id"])]
                        ["source"], stratum=e["stratum"], anchor=seed.chunks[0])
        out.append(Candidate(seed, q, row["answer"].strip(), used, prov))
    for sid, e in entries.items():
        if sid not in seen:
            reject(e, "missing_output")
    return out, rej, {"pending": False, "seeds": len(entries), "accepted": len(out),
                      "skipped": sum(r["reason"] == "skip" for r in rej)}


def summarize(rej: list[dict], cands: list, n_seeds: dict[str, int]) -> dict:
    """Rejection log per reason and per cell (M3-R3 п. 12) plus accepted per cell; n_seeds: cell -> packet seeds."""
    by_cell: dict[str, Counter] = defaultdict(Counter)
    for r in rej:
        by_cell[r["cell"] or "?"][r["reason"]] += 1
    acc = Counter(f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}" for c in cands)
    return {"rejected": dict(sorted(Counter(r["reason"] for r in rej).items())),
            "rejected_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
            "accepted_by_cell": dict(sorted(acc.items())),
            "accept_rate_by_cell": {k: round(acc[k] / n, 4) for k, n in sorted(n_seeds.items()) if n}}


def gen_provenance_problem(prov: dict, root) -> str | None:
    """M3-R3 п. 11 for an Opus pair: the committed generator file still holds the line the pair came from."""
    o = prov.get("opus") or {}
    f = Path(root) / o.get("gen_file", "") if o.get("gen_file") else None
    if f is None or not f.exists():
        return "generator file missing"
    lines = f.read_text(encoding="utf-8").splitlines()
    n = int(o.get("line") or 0)
    if not 1 <= n <= len(lines) or sha256_text(lines[n - 1].strip()) != o.get("row_sha256"):
        return f"generator line {n} of {o.get('gen_file')} changed"
    return None


# ---- reviewer packets ----------------------------------------------------------------------------------------------------
REPAIR_SUFFIX = "-r1"                      # M3-R32: one repair attempt per pair


def pair_id(seed_id: str) -> str:
    """The final §8.1 pair id (graphrag.dataset.finalize.to_pair); a repaired pair (seed id ...-r1) is the original's
    id + "-r1" (M3-R32)."""
    if seed_id.endswith(REPAIR_SUFFIX):
        return pair_id(seed_id[:-len(REPAIR_SUFFIX)]) + REPAIR_SUFFIX
    return f"syn-{hashlib.sha256(f'pair:{seed_id}'.encode()).hexdigest()[:10]}"


def reader_id(seed_id: str) -> str:
    """The id readers see: the pair id, but for a repaired pair an opaque id of the same shape -- a reader packet
    never reveals that a pair is a repair (M3-R32)."""
    if seed_id.endswith(REPAIR_SUFFIX):
        return f"syn-{hashlib.sha256(f'reader:{seed_id}'.encode()).hexdigest()[:10]}"
    return pair_id(seed_id)


def packet_meta(*fns):
    """Merge per-fragment meta providers (session_info, graph_meta, ...) into one cid -> dict callable."""
    def of(cid):
        out = {}
        for f in fns:
            out.update(f(cid) or {})
        return out or None
    return of


def reader_meta(records, index=None, graph=None):
    """M3-R31: cid -> the record META a reader needs to verify names and dates: doc_type, act_title, date and, for a
    plenary transcript, session {dates (every day of the sitting), title} and speaker."""
    from graphrag.dataset.generate import session_days
    sess = session_info(records)
    meta = {r.doc_id: r.meta for r in records}

    def of(cid: str) -> dict:
        doc = cid.split("/", 1)[0]
        m = meta.get(doc)
        out = {"doc_type": kind_of(cid)}
        if m is None:
            return out
        out["act_title"] = index.act[doc] if index is not None else (m.title or "")
        if m.date:
            out["date"] = m.date
        x = sess(cid)
        if x:
            sd = session_days(x["session"])
            days = sorted(f"{sd[0]}-{int(sd[1]):02d}-{int(d):02d}" for d in sd[2]) if sd else [m.date]
            out["session"] = {"dates": days, "title": x["session"].get("title")}
            if x.get("speaker"):
                out["speaker"] = x["speaker"]
        if graph is not None:
            out.update(graph(cid))                       # discussed_act / amended_act / amended_unit
        return out
    return of


def review_item(c, frag_of, meta_of=None) -> dict:
    """What the reader sees: the pair, its fragments and (M3-R31) their record meta -- never notes, facts, sources,
    cells, the generator's or the other reader's identity, or verify verdicts."""
    return {"pair_id": reader_id(c.seed.seed_id), "lang": c.seed.lang, "pattern": c.seed.pattern,
            "question": c.question, "answer": c.answer,
            "fragments": [{"id": f"F{i + 1}", "kind": kind_of(ch),
                           **{k: frag_of(ch, f"F{i + 1}")[k] for k in ("act", "where", "text")},
                           **({"meta": meta_of(ch)} if meta_of else {})}
                          for i, ch in enumerate(c.seed.chunks)]}


ROLE_SECOND = "strict-reader-2"           # M3-R31: the second, independent reader of every pair (was "opus-second-reader")


def second_sample(cands, frac: float, seed) -> list:
    """M3-R29: the Opus second reader's deterministic sample -- pairs with sha256(seed:second|pair_id) below `frac`."""
    return [c for c in cands if int(_h(seed, "second|" + reader_id(c.seed.seed_id))[:8], 16) / 16 ** 8 < float(frac)]


def make_review_packets(cands, frag_of, rubric: dict, size: int, seed, existing: dict[str, list[str]],
                        pilot_ids=(), out_dir: str = "", role: str = ROLE_READER, prefix: str = "",
                        meta_of=None) -> list[dict]:
    """Reviewer packets for the verified Opus pairs not yet in an existing packet: pilot pairs first (their own
    packets review-pilot-NNN), then the rest (review-NNN), each in sha256(seed:review|pair_id) order, <= size
    pairs per packet; numbering continues after the existing packets."""
    have = {p for ps in existing.values() for p in ps}
    pilot = set(pilot_ids)
    mine = [c for c in cands if (c.provenance.get("generate") or {}).get("role") == ROLE_GEN
            and reader_id(c.seed.seed_id) not in have]
    out = []
    for pre, part in ((prefix + "review-pilot-", [c for c in mine if c.seed.seed_id in pilot]),
                      (prefix + "review-", [c for c in mine if c.seed.seed_id not in pilot])):
        part = sorted(part, key=lambda c: _h(seed, "review|" + reader_id(c.seed.seed_id)))
        pilot_pre = prefix + "review-pilot-"
        start = sum(1 for p in existing if p.startswith(pre) and (pre == pilot_pre or not p.startswith(pilot_pre)))
        k = math.ceil(len(part) / int(size)) if part else 0
        bounds = [round(i * len(part) / k) for i in range(k + 1)] if k else []
        for i in range(k):
            pid = f"{pre}{start + i + 1:03d}"
            items = [review_item(c, frag_of, meta_of) for c in part[bounds[i]:bounds[i + 1]]]
            out.append({"packet_id": pid, "format": FORMAT, "role": role, "n_items": len(items),
                        "rubric": rubric, "output": {"path": f"{out_dir}/{pid}.jsonl", "format": "jsonl",
                                                     "schema": VERDICT_SCHEMA},
                        "items": items})
    return out


def ingest_verdicts(packets: list[dict], texts: dict[str, str | None], cands, reader_rubric_digest
                    ) -> tuple[list, list[dict], dict]:
    """-> (GOOD candidates with provenance["review"], rejection rows, stats). A pair is GOOD only with exactly one
    valid GOOD verdict line in its packet's file; a packet without a file is pending (its pairs unreviewed)."""
    by_pair = {reader_id(c.seed.seed_id): c for c in cands}
    rej, good, verdict_of, pending = [], [], {}, []
    cell = lambda c: f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}"
    in_packet = {}
    for p in packets:
        allowed = {reader_rubric_digest} if isinstance(reader_rubric_digest, str) else set(reader_rubric_digest)
        if p["rubric"]["digest"] not in allowed:
            raise ValueError(f"{p['packet_id']}: rubric digest {p['rubric']['digest'][:12]} is not the configured one")
        check_instructions(p["rubric"])
        for it in p["items"]:
            in_packet[it["pair_id"]] = p["packet_id"]

    def reject(pid, pair, reason, line=None, detail=""):
        c = by_pair.get(pair)
        rej.append({"packet_id": pid, "pair_id": pair, "seed_id": c.seed.seed_id if c else None,
                    "cell": cell(c) if c else None, "reason": reason, "line": line, "detail": str(detail)[:300]})
    for p in sorted(packets, key=lambda p: p["packet_id"]):
        pid, text = p["packet_id"], texts.get(p["packet_id"])
        ids = [it["pair_id"] for it in p["items"]]
        if text is None:
            pending.append(pid)
            continue
        seen = set()
        for n, row, raw in _parse_jsonl(text):
            if row is None or not isinstance(row, dict):
                reject(pid, None, "bad_json", n, raw)
                continue
            pr = row.get("pair_id")
            if pr not in ids:
                reject(pid, pr, "unknown_pair", n)
                continue
            if pr in seen:
                reject(pid, pr, "duplicate_verdict", n)
                verdict_of.pop(pr, None)
                continue
            seen.add(pr)
            v, reasons = row.get("verdict"), row.get("reasons", [])
            if v not in VERDICTS or not isinstance(reasons, list):
                reject(pid, pr, "bad_verdict", n, v)
                continue
            c = by_pair.get(pr)
            if c is None:
                reject(pid, pr, "not_verified", n)
                continue
            gen_agent = (c.provenance.get("opus") or {}).get("agent")
            if row.get("agent") and gen_agent and row["agent"] == gen_agent:
                reject(pid, pr, "same_agent", n, row["agent"])
                continue
            verdict_of[pr] = (v, reasons, pid, n, sha256_text(raw.strip()), row.get("agent"))
        for pr in ids:
            if pr not in seen:
                reject(pid, pr, "missing_verdict")
    dist, by_reason, by_cell = defaultdict(Counter), Counter(), defaultdict(Counter)
    dig = {p["packet_id"]: p["rubric"]["digest"] for p in packets}
    role = {p["packet_id"]: p.get("role") for p in packets}
    detail = {}
    for pr, (v, reasons, pid, n, sha, agent) in sorted(verdict_of.items()):
        c = by_pair[pr]
        dist[cell(c)][v] += 1
        crit0 = sorted({str((r or {}).get("criterion", "other")) if isinstance(r, dict) else "other" for r in reasons})
        detail[pr] = {"verdict": v, "criteria": crit0, "reasons": reasons, "packet_id": pid, "line": n, "row_sha256": sha,
                      "agent": agent, "rubric_digest": dig[pid], "role": role[pid]}
        if v == "GOOD":
            c.provenance["review"] = {"role": role[pid], "packet_id": pid, "rubric_digest": dig[pid],
                                      "verdict": v, "line": n, "row_sha256": sha, "agent": agent}
            good.append(c)
            continue
        crit = sorted({str((r or {}).get("criterion", "other")) if isinstance(r, dict) else "other" for r in reasons}
                      ) or ["unspecified"]
        for k in crit:
            k = k if k in CRITERIA else "other" if k != "unspecified" else k
            by_reason[f"{v}|{k}"] += 1
            by_cell[cell(c)][f"{v}|{k}"] += 1
        rej.append({"packet_id": pid, "pair_id": pr, "seed_id": c.seed.seed_id, "cell": cell(c),
                    "reason": v.lower(), "line": n, "detail": json.dumps(reasons, ensure_ascii=False)[:600]})
    unreviewed = sorted(reader_id(c.seed.seed_id) for c in cands if reader_id(c.seed.seed_id) not in in_packet)
    n_rev = sum(sum(v.values()) for v in dist.values())
    return sorted(good, key=lambda c: c.seed.seed_id), rej, {
        "packets": len(packets), "pending_packets": pending, "reviewed": n_rev, "good": len(good),
        "good_rate": round(len(good) / n_rev, 4) if n_rev else None,
        "verdicts_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(dist.items())},
        "rejected_by_reason": dict(sorted(by_reason.items())),
        "rejected_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
        "invalid": dict(sorted(Counter(r["reason"] for r in rej if r["reason"] not in ("bad", "borderline")).items())),
        "not_in_any_packet": len(unreviewed), "verdict_of": detail}


def consensus(cands, a: dict, b: dict) -> tuple[list, list[dict], dict]:
    """M3-R31 dual-reader consensus: a, b = verdict_of maps (ingest_verdicts stats) of the two independent readers
    (review/ and review2/). A pair is accepted only if BOTH read it GOOD; a pair missing either verdict is pending.
    Returns (accepted candidates with provenance["review"] = {verdict: GOOD, consensus, readers}, rejection rows,
    stats incl. agreement, disagreements per cell and per defect)."""
    cell = lambda c: f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}"
    good, rej, pending = [], [], []
    agree, dis_cell, dis_defect, pair_kind = Counter(), defaultdict(Counter), Counter(), Counter()
    for c in sorted(cands, key=lambda c: c.seed.seed_id):
        pr = reader_id(c.seed.seed_id)
        va, vb = a.get(pr), b.get(pr)
        if va is None or vb is None:
            pending.append({"pair_id": pr, "missing": [k for k, v in (("A", va), ("B", vb)) if v is None]})
            continue
        key = f"{va['verdict']}|{vb['verdict']}"
        agree[key] += 1
        if va["verdict"] != vb["verdict"]:
            dis_cell[cell(c)][key] += 1
            if "GOOD" in (va["verdict"], vb["verdict"]):     # one reader accepts, the other names the defect
                other = vb if va["verdict"] == "GOOD" else va
                for k in other["criteria"] or ["unspecified"]:
                    dis_defect[f"{'B' if other is vb else 'A'}|{k}"] += 1
        if va["verdict"] == "GOOD" and vb["verdict"] == "GOOD":
            c.provenance["review"] = {"verdict": "GOOD", "consensus": True, "readers": {"A": va, "B": vb}}
            good.append(c)
            continue
        rej.append({"pair_id": pr, "seed_id": c.seed.seed_id, "cell": cell(c), "reason": "no_consensus",
                    "verdicts": {"A": va["verdict"], "B": vb["verdict"]},
                    "criteria": {"A": va["criteria"], "B": vb["criteria"]}})
    both = sum(agree.values())
    return good, rej, {"read_by_both": both, "consensus_good": len(good), "pending": len(pending),
                       "pending_pairs": pending, "agreement": dict(sorted(agree.items())),
                       "exact_agreement": round(sum(v for k, v in agree.items() if k.split("|")[0] == k.split("|")[1])
                                                / both, 4) if both else None,
                       "disagreements_by_cell": {k: dict(sorted(v.items())) for k, v in sorted(dis_cell.items())},
                       "disagreements_by_defect": dict(sorted(dis_defect.items()))}


def reader_gate(good: int, n: int, min_good: float, min_wilson: float) -> dict:
    from graphrag.kg.audit import wilson
    lo, hi = wilson(good, n) if n else (0.0, 0.0)
    share = good / n if n else 0.0
    return {"n": n, "good": good, "share": round(share, 4), "wilson": [round(lo, 4), round(hi, 4)],
            "ok": bool(n) and share >= min_good and lo >= min_wilson}


def dedupe_content(cands, dup: dict, split_of: dict[str, str]) -> tuple[list, list[dict]]:
    """M3-R25: one candidate per duplicate content (seeds.content_signature; the smallest seed_id survives) and none
    whose C_ref text recurs in another split (seeds.cross_split_twin). Survivors record their twin fragments
    (provenance duplicate_twins: equivalent gold). Returns (kept, rejection rows)."""
    from graphrag.dataset.seeds import content_signature, cross_split_twin
    kept, rej, seen = [], [], {}
    for c in sorted(cands, key=lambda c: c.seed.seed_id):
        o = c.provenance.get("opus") or {}
        row = lambda reason, detail: {"packet_id": o.get("packet_id"), "seed_id": o.get("packet_seed_id", c.seed.seed_id),
                                      "cell": f"{c.seed.split}|{c.seed.pattern}|{c.seed.lang}", "reason": reason,
                                      "line": o.get("line"), "detail": detail}
        if cross_split_twin(c.seed.chunks, dup, split_of):
            rej.append(row("duplicate_cross_split", [t for x in c.seed.chunks for t in (dup.get(x) or {}).get("twins", ())]))
            continue
        sig = content_signature(c.seed.chunks, dup)
        if sig in seen:
            rej.append(row("duplicate_content", seen[sig]))
            continue
        seen[sig] = c.seed.seed_id
        tw = {x: dup[x]["twins"] for x in c.seed.chunks if x in dup}
        if tw:
            c.provenance["duplicate_twins"] = tw
        kept.append(c)
    return kept, rej


# ---- M3-R32: repair loop --------------------------------------------------------------------------------------------
ROLE_REPAIR = "repair-agent"


def repair_eligible(cands, va: dict, vb: dict, attempted: set) -> list[tuple]:
    """(candidate, verdict A, verdict B) for pairs BOTH readers read, at least one BORDERLINE, neither BAD, not
    consensus GOOD, not a repair itself and not repaired before (one attempt per pair)."""
    out = []
    for c in sorted(cands, key=lambda c: c.seed.seed_id):
        if c.seed.seed_id.endswith(REPAIR_SUFFIX) or c.seed.seed_id in attempted:
            continue
        a, b = va.get(reader_id(c.seed.seed_id)), vb.get(reader_id(c.seed.seed_id))
        if a is None or b is None:
            continue
        vs = {a["verdict"], b["verdict"]}
        if "BAD" in vs or "BORDERLINE" not in vs or vs == {"GOOD"}:
            continue
        out.append((c, a, b))
    return out


def _seed_entry_for(c, gen_packets: dict) -> dict:
    """The generator packet's seed entry of candidate c (multi: only the neighbour the pair uses)."""
    o = c.provenance["opus"]
    pk = gen_packets[o["packet_id"]]
    e = json.loads(json.dumps(next(x for x in pk["seeds"] if x["seed_id"] == o["packet_seed_id"])))
    if "anchor" in e:
        e["neighbors"] = [n for n in e["neighbors"] if n["id"] == o["neighbor_id"]]
        keep = {"A", o["neighbor_id"]}
        e["facts"] = [f for f in e["facts"] if f["fact_id"].split(".", 1)[0] in keep]
    return e


def make_repair_packets(items: list[tuple], gen_packets: dict, gen_row_of, repair_instr: dict, gen_instr: dict,
                        size: int, seed, existing: dict, out_dir: str, meta_fn=None) -> list[dict]:
    """items = repair_eligible(...). Packets repair-NNN of <= size items: the pair as generated (question, answer,
    used_facts, neighbor, sources), BOTH readers' reasons (criteria + notes), the seed entry, the REPAIR instructions
    and the current GENERATOR instructions. Numbering continues after `existing`."""
    items = sorted(items, key=lambda t: _h(seed, "repair|" + t[0].seed.seed_id))
    k = math.ceil(len(items) / int(size)) if items else 0
    bounds = [round(i * len(items) / k) for i in range(k + 1)] if k else []
    start = sum(1 for p in existing if p.startswith("repair-"))
    out = []
    for i in range(k):
        pid = f"repair-{start + i + 1:03d}"
        rows = []
        for c, a, b in items[bounds[i]:bounds[i + 1]]:
            g = gen_row_of(c)
            rows.append({"seed_id": c.provenance["opus"]["packet_seed_id"], "original_seed_id": c.seed.seed_id,
                         "pair": {"question": c.question, "answer": c.answer, "lang": c.seed.lang,
                                  "pattern": c.seed.pattern, "used_facts": g.get("used_facts"),
                                  "neighbor_id": g.get("neighbor_id"), "sources": g.get("sources")},
                         "reasons": {"reader_1": a.get("reasons") or [{"criterion": x} for x in a["criteria"]],
                                     "reader_2": b.get("reasons") or [{"criterion": x} for x in b["criteria"]]},
                         "verdicts": {"reader_1": a["verdict"], "reader_2": b["verdict"]},
                         "seed": enrich_entry(_seed_entry_for(c, gen_packets), meta_fn)})
        out.append({"packet_id": pid, "format": FORMAT, "role": ROLE_REPAIR, "n_items": len(rows),
                    "repair_instructions": repair_instr, "generator_instructions": gen_instr,
                    "output": {"path": f"{out_dir}/{pid}.jsonl", "format": "jsonl", "schema": OUTPUT_SCHEMA},
                    "items": rows})
    return out


def enrich_entry(e: dict, meta_fn) -> dict:
    """A copy of a packet seed entry whose fragments carry the CURRENT verified meta (session / speaker /
    discussed_act / amended_act / amended_unit) where they lack it -- repair packets built from older generator
    packets get the graph meta the readers see."""
    if meta_fn is None:
        return e
    e = json.loads(json.dumps(e))
    for f in e.get("fragments", []) + ([e["anchor"]] if "anchor" in e else []) + e.get("neighbors", []):
        for k, v in (meta_fn(f["chunk_id"]) or {}).items():
            f.setdefault(k, v)
    return e


def ingest_repairs(packet: dict, text: str | None, chunks_by_id, cfg: dict, doc_section: dict, gen_path: str,
                   agent_model: str, originals: dict, meta_fn=None) -> tuple[list, list[dict], dict]:
    """One repair packet's output -> NEW candidates (seed id <original>-r1, pair id syn-...-r1) through ALL code
    checks (ingest_generated over the packet's seed entries), provenance repaired_from {seed_id, pair_id, reasons,
    verdicts, repair packet, repair instructions}; a repair that changes the neighbour / sources is rejected.
    originals: seed_id -> original Candidate."""
    import dataclasses
    from graphrag.dataset.generate import Candidate
    check_instructions(packet["repair_instructions"])
    pseudo = {"packet_id": packet["packet_id"], "instructions": packet["generator_instructions"],
              "seeds": [enrich_entry(it["seed"], meta_fn) for it in packet["items"]]}
    cands, rej, st = ingest_generated(pseudo, text, chunks_by_id, cfg, doc_section, gen_path, agent_model)
    by_seed = {it["seed_id"]: it for it in packet["items"]}
    out = []
    ri = {k: packet["repair_instructions"][k] for k in ("path", "version", "digest")}
    for c in cands:
        it = by_seed[c.provenance["opus"]["packet_seed_id"]]
        orig = originals.get(it["original_seed_id"]) or _original_from_item(it, doc_section)
        if orig is None or tuple(c.seed.chunks) != tuple(orig.seed.chunks):
            rej.append({"packet_id": packet["packet_id"], "seed_id": it["seed_id"], "cell": None,
                        "reason": "repair_changed_fragments", "line": c.provenance["opus"]["line"], "detail": ""})
            continue
        new_id = orig.seed.seed_id + REPAIR_SUFFIX
        seed = dataclasses.replace(orig.seed, seed_id=new_id)
        prov = dict(c.provenance, seed_id=new_id)
        prov["generate"] = dict(prov["generate"], role=ROLE_GEN, repair_agent=ROLE_REPAIR)
        prov["repaired_from"] = {"seed_id": orig.seed.seed_id, "pair_id": pair_id(orig.seed.seed_id),
                                 "reasons": it["reasons"], "verdicts": it["verdicts"],
                                 "repair_packet": packet["packet_id"], "repair_instructions": ri,
                                 "original_generate": orig.provenance.get("generate")}
        out.append(Candidate(seed, c.question, c.answer, c.facts, prov))
    return out, rej, st


def _original_from_item(it: dict, doc_section: dict):
    """The original pair's seed as the repair packet recorded it (the packet is the stable record: the current
    candidates_raw / verified may no longer hold an original that later checks or the v6 judge dropped)."""
    from graphrag.dataset.generate import Candidate
    e, oid = it["seed"], it["original_seed_id"]
    if "seed" in e:
        seed = Seed.from_json(e["seed"])
    elif "anchor" in e and e.get("neighbors"):
        n = e["neighbors"][0]
        chunks = (e["anchor"]["chunk_id"], n["chunk_id"])
        docs = tuple(dict.fromkeys(c.split("/", 1)[0] for c in chunks))
        seed = Seed(oid, "multi", e["cell"]["lang"], chunks, docs, e["cell"]["split"], doc_section.get(docs[0], ""),
                    tuple(tuple(h) for h in n.get("hops", ())))
    else:
        return None
    if seed.seed_id != oid:
        import dataclasses
        seed = dataclasses.replace(seed, seed_id=oid)
    return Candidate(seed, it["pair"]["question"], it["pair"]["answer"], [], {})


def wave_of(c) -> str:
    """The generator wave of a candidate: its packet id's prefix (pilot / mini / w1 / w2 / multi / repair)."""
    pid = (c.provenance.get("opus") or {}).get("packet_id") or ""
    return pid.split("-", 1)[0] if pid else "?"


# ---- graph meta: the law a speech discusses, the act an amending item amends --------------------------------------
_QTITLE = re.compile(r"[«“\"„]([^«»“”\"„]{6,300})[»”\"“]")


def _clean_title(t: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", t or "").strip()


def _unit_text(path: str) -> str:
    from graphrag.dataset.sources import segment_text
    out = []
    for seg in [x for x in (path or "").split("/") if x]:
        try:
            out.append(segment_text(seg))
        except ValueError:
            return ""
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")                 # graph "СТ.11.1" = the inserted article 11¹
    return re.sub(r"(\d+)\.(\d+)", lambda m: m.group(1) + m.group(2).translate(sup), " ".join(out))


def _act_root(v: str) -> str:
    return "/".join(v.split("/")[:2]) if v.startswith("EXT/") else v.split("/", 1)[0]


def graph_meta(records, edges, index):
    """cid -> {discussed_act?, amended_act?, amended_unit?} from verified graph edges (M3-R32 follow-up):
    - discussed_act (plenary transcript fragment): the agenda item (ВОПР) the speech belongs to -- the corpus act of
      its REFERS_TO{agenda} edge (packet act title), else the agenda item's own title from the record structure;
    - amended_act / amended_unit (amending-law fragment): the AMENDS edges whose evidence lies in the fragment (else
      those of its node's ancestors / descendants) -- amended_act only when they all target ONE act (the corpus act
      title, or for an external act its verbatim quoted title from a REFERS_TO evidence, same document first);
      amended_unit only when there is exactly one target (its article / part path)."""
    from graphrag.dataset.walks import _chunk_at
    recs = {r.doc_id: r for r in records}
    title = {n.node_id: n.title for r in records for n in r.structure}
    agenda, amend_at, titles = {}, defaultdict(set), defaultdict(list)
    amend_src, amend_docs = defaultdict(set), set()
    for e in sorted(edges, key=lambda e: (e.src, e.type, e.dst)):
        a = e.attrs or {}
        if e.type == "REFERS_TO" and a.get("relation") == "agenda":
            agenda.setdefault(e.src, e.dst)
        elif e.type == "AMENDS":
            amend_src[e.src].add(e.dst)
            amend_docs.add(e.src.split("/", 1)[0])
            ev = a.get("evidence") or {}
            r = recs.get(ev.get("doc_id"))
            c = _chunk_at(r, ev["start"], ev["end"]) if r is not None and "start" in ev else None
            if c:
                amend_at[c].add(e.dst)
        if e.type == "REFERS_TO" and e.dst.startswith("EXT/") and e.dst.count("/") == 1:
            q = ((a.get("evidence") or {}).get("quote")) or ""
            m = _QTITLE.search(q)
            if m:
                titles[e.dst].append((e.src.split("/", 1)[0], m.group(1).strip()))

    def act_title(v: str, doc: str) -> str | None:
        root = "/".join(v.split("/")[:2]) if v.startswith("EXT/") else v.split("/", 1)[0]
        if not v.startswith("EXT/"):
            return _clean_title(index.act.get(root, "")) or None
        ts = titles.get(root) or []
        same = [t for d, t in ts if d == doc]
        return (same or [t for _, t in ts] or [None])[0]

    def of(cid: str) -> dict:
        out = {}
        doc = cid.split("/", 1)[0]
        parts = cid.split("#", 1)[0].split("/")
        vopr = next(("/".join(parts[:i + 1]) for i in range(len(parts)) if parts[i].startswith("ВОПР.")), None)
        if vopr:
            tgt = agenda.get(vopr)
            t = _clean_title(index.act.get(tgt, "")) if tgt and not tgt.startswith("EXT/") and tgt in index.act \
                else (title.get(vopr) or "").strip()
            if t:
                out["discussed_act"] = t
        tg = amend_at.get(cid) or set()
        if not tg:                  # the item's AMENDS evidence lies in another chunk of the same (ancestor) node
            node = cid.split("#", 1)[0]
            tg = {d for n, ds in amend_src.items() if node == n or node.startswith(n + "/") or n.startswith(node + "/")
                  for d in ds} if node.split("/", 1)[0] in amend_docs else set()
        roots = {_act_root(v) for v in tg}
        if len(roots) == 1:         # one amended act (every AMENDS target under it); the unit only if single
            v = next(iter(tg))
            t = act_title(v, doc)
            if t:
                out["amended_act"] = t
                if len(tg) == 1:
                    unit = _unit_text("/".join(v.split("/")[2:] if v.startswith("EXT/") else v.split("/")[1:]))
                    if unit:
                        out["amended_unit"] = unit
        return out
    return of
