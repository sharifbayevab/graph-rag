"""M7 part 2, Task 6: three blind readers of the Q3 (compare) items and their consensus (the M3 reader pipeline,
.superpowers/sdd/2026-09-28-m3-env-reward/codex/, rubric config/dataset_reader_rubric.md v3, adapted to Q3).

* `reader_packets`: the judge-accepted items in sha256(seed:rev|qid) order, cut into packets of n (default 10). An item
  shows its question, answer, the two documents (A / B: title, type, date) and its GOLD fragments only (id, document,
  text) -- the readers judge what the benchmark will score, not the generator's whole packet.
* Readers: A = Fable (if unavailable: a second Opus with the A2 instruction, every line marked "reader": "opus-a2"),
  B = Opus, C = Codex via `codex exec` (BAD veto, M3-R34). Verdict lines {pair_id, verdict: GOOD | BAD, reasons}.
* `consensus(a, b, c)`: accepted iff A and B say GOOD and C did not say BAD; a missing A / B verdict is pending
  unless the other one already says BAD (then rejected). `reader_consensus` parses the three verdict lists (malformed
  lines counted, never fatal; the first line per pair_id wins) and returns one row per item."""
from __future__ import annotations

import hashlib
from collections import Counter

VERDICTS = ("GOOD", "BAD")
READERS = ("A", "B", "C")
READER_A_MODELS = ("fable", "opus-a2")
PER_PACKET = 10
CRITERIA = ("q1", "q2", "q3", "q4", "q5", "q6")


def _u(seed, key: str) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def _pid(item: dict) -> str:
    return (item.get("meta") or {}).get("packet_id") or item["packet_id"]


def review_item(item: dict, packet: dict) -> dict:
    keys = {d["doc_id"]: k for k, d in zip("AB", packet["docs"])}
    text = {c["chunk_id"]: (d["doc_id"], c["text"]) for d in packet["docs"] for c in d["chunks"]}
    return {"pair_id": item["qid"], "lang": item["lang"], "kind": item["template"],
            "documents": [{"key": keys[d["doc_id"]], "title": d.get("full_title") or d["title"],
                           "doc_type": d["doc_type"], "date": d.get("date")} for d in packet["docs"]],
            "question": item["question"], "answer": item["answer"],
            "fragments": [{"id": g, "doc": keys[text[g][0]], "text": text[g][1]} for g in item["gold"]]}


def reader_packets(items, packets: dict, n: int = PER_PACKET, seed=20261003, prefix: str = "rev",
                   start: int = 0) -> list[dict]:
    order = sorted(items, key=lambda it: (_u(seed, f"rev|{it['qid']}"), it["qid"]))
    out = []
    for i in range(0, len(order), n):
        out.append({"packet_id": f"{prefix}-{start + i // n:03d}",
                    "items": [review_item(it, packets[_pid(it)]) for it in order[i:i + n]]})
    return out


def consensus(a: str | None, b: str | None, c: str | None) -> str:
    """accepted | rejected | pending (module doc)."""
    if a == "BAD" or b == "BAD" or c == "BAD":
        return "rejected"
    if a == "GOOD" and b == "GOOD":
        return "accepted"
    return "pending"


def parse_verdicts(lines, qids, who: str, st: Counter, default_reader: str | None = None) -> dict:
    """pair_id -> {verdict, reasons, reader}; malformed / unknown / duplicate lines counted in st."""
    out = {}
    for x in lines:
        ok = (isinstance(x, dict) and isinstance(x.get("pair_id"), str) and x.get("verdict") in VERDICTS
              and isinstance(x.get("reasons", []), list))
        if not ok:
            st[f"malformed:{who}"] += 1
            continue
        if x["pair_id"] not in qids:
            st[f"unknown:{who}"] += 1
            continue
        if x["pair_id"] in out:
            st[f"duplicate:{who}"] += 1
            continue
        reader = x.get("reader") or default_reader
        out[x["pair_id"]] = {"verdict": x["verdict"], "reasons": x.get("reasons", []), "reader": reader}
    return out


def reader_consensus(items, va, vb, vc, reader_a: str = "fable") -> tuple[list[dict], Counter]:
    """One row per item {qid, status, readers: {A, B, C: {verdict, reasons, reader} | None}} + stats (accepted /
    rejected / pending, per reader GOOD / BAD / missing, A-B agreement, C vetoes of A+B GOOD, A models)."""
    st = Counter()
    qids = {it["qid"] for it in items}
    got = {"A": parse_verdicts(va, qids, "A", st, reader_a), "B": parse_verdicts(vb, qids, "B", st, "opus"),
           "C": parse_verdicts(vc, qids, "C", st, "codex")}
    for r in got["A"].values():
        if r["reader"] not in READER_A_MODELS:
            raise ValueError(f"reader A model {r['reader']!r} not in {READER_A_MODELS}")
    rows = []
    for it in items:
        v = {k: got[k].get(it["qid"]) for k in READERS}
        a, b, c = (v[k]["verdict"] if v[k] else None for k in READERS)
        status = consensus(a, b, c)
        st[status] += 1
        for k in READERS:
            st[f"{k}:{v[k]['verdict'] if v[k] else 'missing'}"] += 1
        if v["A"]:
            st[f"A_model:{v['A']['reader']}"] += 1
        if a and b:
            st["ab_both"] += 1
            st["ab_agree"] += a == b
        if a == "GOOD" and b == "GOOD" and c == "BAD":
            st["c_veto"] += 1
        rows.append({"qid": it["qid"], "status": status, "readers": v})
    return rows, st


# --- rubric + instructions --------------------------------------------------------------------------------------

RUBRIC = """# READER RUBRIC: M7 Q3 comparative questions (version 1, plan M7 part 2, Task 6; after config/dataset_reader_rubric.md v3)

You are an INDEPENDENT STRICT READER of comparative questions for a legal retrieval benchmark over Uzbek parliamentary
documents (laws, Senate committee conclusions, Senate resolutions, plenary transcripts). You did not write these
items. A retrieval agent will be scored on finding the GOLD fragments, so the question must need exactly what the
gold gives. Most generated items have at least one defect; your job is to find it. When in doubt, the verdict is BAD.

## Input

A REVIEW PACKET (JSON): `items[]`, each with `pair_id`, `lang` (target language: uz-Cyrl / uz-Latn / ru), `kind`
(law_conclusion / law_resolution / laws_shared_entity / stenogram_law), `documents[2]` (`key` A / B, `title`,
`doc_type`, `date`), `question`, `answer` and `fragments[]` (`id`, `doc` = A / B, `text`): the GOLD, the only evidence.
Use only the packet: no other files, no earlier verdicts, no outside knowledge of the law.

## Criteria (check each one separately)

- q1 **needs both documents**: answering needs a fact from a fragment of A AND a fact from a fragment of B. It fails if
  the fragments of one document alone answer the question, if one document's fragment adds nothing the answer uses,
  or if the «comparison» is only that both documents share a title or a topic.
- q2 **supported by the gold only**: every name, number, date, body and claim of the question and the answer is in the
  gold fragments (or in the documents' `title` / `doc_type` / `date`). Nothing is reversed, narrowed, widened or added
  from memory; the answer fully answers the question (all listed items, or the one the question names).
- q3 **acts named unambiguously**: a reader who never saw the packet knows exactly which two documents are meant -- a
  law by its title (a short distinctive form is fine) and its date or number where the text gives them; a conclusion
  as the named committee's conclusion on that law; a resolution as the Senate's (or a committee's) resolution on that
  law; a plenary sitting by its date (a multi-day sitting by its full range). Each document is attributed correctly:
  a conclusion is not called a law, an adopted law not a draft, a speech belongs to the sitting. No pointer words
  («ушбу ҳужжат», «мазкур», «bu hujjat», «данный документ», «этот закон»); a generic title alone («айрим қонун
  ҳужжатларига ўзгартиришлар киритиш тўғрисида») identifies nothing.
- q4 **language**: fluent, grammatical `lang` with correct legal terms. uz-Cyrl: Cyrillic only (ў қ ғ ҳ), no Latin
  letters and no ʻ / ʼ; uz-Latn: Latin only (oʻ gʻ); ru: Russian, act titles rendered in Russian (at most one short
  Uzbek original in «»).
- q5 **no near-copy, no leak**: the question is not a fragment's sentence with one element blanked out (also in
  translation or transliteration) and copies no run of about 8 or more words from a fragment, long titles included.
  The question does not state or hint at what the answer gives.
- q6 **one question**: one information need with one point of comparison (one fact from each document is fine). Two
  unrelated questions glued with «and / ва / va / и», or a list of sub-questions, fail.

## Verdict

- **GOOD**: all six criteria hold without doubt.
- **BAD**: any criterion fails, or you are in doubt. There is no BORDERLINE in this rubric.

Rate each item on its own. There is no quota: a whole packet may be BAD, or a whole packet may be GOOD.

## Output (JSONL, one line per item, in packet order, nothing else)

```json
{"pair_id": "...", "verdict": "GOOD" | "BAD", "reasons": [{"criterion": "q1".."q6", "note": "a few words"}]}
```

`reasons` is empty for GOOD. For BAD list every failed criterion with a short note quoting the offending words.
Before you finish, check that the file is valid JSONL with exactly one line per `pair_id` of the packet.
"""

ROLE_NOTES = {
    "A": "Reader A (Fable). Add \"reader\": \"fable\" to every line.",
    "A2": ("Reader A2 (a second Opus, used ONLY when Fable is unavailable; the controller says so). Work adversarially: "
           "read the ANSWER first and try to prove it wrong -- for each claim find the fragment sentence that states "
           "it, then cover each document's fragments in turn and ask whether the other document alone still answers "
           "the question (q1). Only then read the question for q3-q6. Add \"reader\": \"opus-a2\" to every line."),
    "B": "Reader B (Opus). Read the question first, then the answer, then the fragments; check q1..q6 in order.",
}


def instructions(packets_dir: str, verdicts_dir: str, codex_dir: str, rubric_path: str, packet_ids) -> str:
    ids = list(packet_ids)
    first, last = (ids[0], ids[-1]) if ids else ("-", "-")
    return f"""# READER INSTRUCTIONS: M7 Q3 comparative questions (plan M7 part 2, Task 6)

Three blind readers judge every judge-accepted Q3 item: A = Fable (economically: at most two packets at a time), B =
Opus, C = Codex via `codex exec` (BAD veto). Consensus: GOOD from A and B, and C did not say BAD.

## Files

- Rubric (binding, read it in full first): `{rubric_path}`
- Review packets: `{packets_dir}/{first}.json` .. `{last}.json` ({len(ids)} packets, 10 items each, the last may be
  shorter). Do only the packets the controller assigns you.
- Output: `{verdicts_dir}/<R>-<packet_id>.jsonl` with <R> = A, B or C (A2 writes the A file), e.g.
  `{verdicts_dir}/B-{first}.jsonl`. JSONL in the rubric's schema, one line per item, packet order. Write each item's
  line as soon as you have judged it; if the file already has lines, skip those pair_ids and append.
- Use only the rubric and your packet. Do not read project code, other packets, other verdict files, generator outputs
  or reports. Make no other changes, no git commands, no network / API calls.

## Roles

- {ROLE_NOTES["A"]}
- {ROLE_NOTES["A2"]}
- {ROLE_NOTES["B"]}
- Reader C (Codex): the controller runs `codex exec` with the prompt file `{codex_dir}/<packet_id>.txt` (one per
  packet); it writes `{verdicts_dir}/C-<packet_id>.jsonl`.

Final message ONLY 2 lines: the GOOD / BAD counts and the main defects.
"""


def codex_prompt(packet_id: str, packets_dir: str, verdicts_dir: str, rubric_path: str) -> str:
    return (f"You are an independent, strict reviewer of comparative legal questions over the Uzbekistan parliament "
            f"corpus. The items are in Uzbek (Cyrillic and Latin script) and Russian. You don't know who wrote them.\n\n"
            f"First read the rubric in full: {rubric_path} (binding).\n\n"
            f"Then review every item in {packets_dir}/{packet_id}.json, reading the file in parts. Judge each item "
            f"GOOD only if it meets every criterion q1..q6, otherwise BAD.\n\n"
            f"Write the verdicts as JSONL in the rubric's schema (one line per item, packet order) to "
            f"{verdicts_dir}/C-{packet_id}.jsonl. Write each chunk of at most 5 records as soon as you have judged it. "
            f"If the file already has lines, skip those pair_ids and append.\n\n"
            f"Do not read project code, any other dataset files, other verdict files or reports. Make no other "
            f"changes, no git commands, no network/API calls.\n\n"
            f"Final message ONLY 2 lines: the GOOD/BAD counts and the main defects.\n")


# --- R54: rubric v2 and the repair round -------------------------------------------------------------------------

RUBRIC_VERSION = 2
RUBRIC_V2 = (RUBRIC
    .replace("(version 1, plan M7 part 2, Task 6; after config/dataset_reader_rubric.md v3)",
             "(version 2, ruling R54; plan M7 part 2, Task 6; after config/dataset_reader_rubric.md v3)")
    .replace("""- q2 **supported by the gold only**: every name, number, date, body and claim of the question and the answer is in the
  gold fragments (or in the documents' `title` / `doc_type` / `date`). Nothing is reversed, narrowed, widened or added
  from memory; the answer fully answers the question (all listed items, or the one the question names).""",
             """- q2 **supported by the gold only**: the ANSWER is entailed by the gold fragments alone: every name, number, body
  and claim it states is in a gold fragment's text. Nothing is reversed, narrowed, widened or added from memory (no
  «it kept / it used to», no completeness claim the fragments do not show); the answer fully answers the question.
  In the QUESTION, act titles, numbers and dates are allowed only as identifiers, and only when they appear verbatim
  in a gold fragment's text or in that fragment's document `title`. The adoption body or date, a committee name, or
  any description that is in no gold fragment or title is a q2 failure, even if it is true.""")
    .replace("""- q3 **acts named unambiguously**: a reader who never saw the packet knows exactly which two documents are meant -- a
  law by its title (a short distinctive form is fine) and its date or number where the text gives them; a conclusion
  as the named committee's conclusion on that law; a resolution as the Senate's (or a committee's) resolution on that
  law; a plenary sitting by its date (a multi-day sitting by its full range).""",
             """- q3 **acts named unambiguously**: a reader who never saw the packet knows exactly which two documents are meant,
  using only identifiers allowed by q2 -- a law by its title (a short distinctive form is fine); a law with a generic
  amending title («... айрим қонун ҳужжатларига ўзгартиш ... киритиш тўғрисида») by its subject or the article it
  amends as the gold states them, never by an invented description; a conclusion as a committee conclusion on that
  law (the committee named only if gold names it); a resolution as the Senate resolution on that law; a plenary
  sitting by its date only if the gold or its title gives it. A date is never required."""))
assert RUBRIC_V2.count("version 2") == 1 and "A date is never required" in RUBRIC_V2 and "entailed" in RUBRIC_V2


def repair_items(items, packets: dict, rows: dict, packets_dir: str) -> list[dict]:
    """One repair item per judged item: the item, its gold fragments, both documents' titles, every reader's verdict
    and reasons verbatim (rows: qid -> reader row), and the generator packet path (for re-picking gold)."""
    out = []
    for it in items:
        p = packets[_pid(it)]
        text = {c["chunk_id"]: (d["doc_id"], c["text"]) for d in p["docs"] for c in d["chunks"]}
        out.append({"qid": it["qid"], "packet_id": p["packet_id"], "kind": it["template"], "lang": it["lang"],
                    "source_packet": f"{packets_dir}/{p['packet_id']}.json",
                    "documents": [{"doc_id": d["doc_id"], "title": d["title"], "full_title": d.get("full_title"),
                                   "doc_type": d["doc_type"]} for d in p["docs"]],
                    "question": it["question"], "answer": it["answer"],
                    "gold": [{"chunk_id": g, "doc_id": text[g][0], "text": text[g][1]} for g in it["gold"]],
                    "reader_verdicts": {k: v for k, v in (rows.get(it["qid"]) or {}).items() if v}})
    return out


def repair_packets(items, packets: dict, rows: dict, packets_dir: str, n: int = PER_PACKET) -> list[dict]:
    rep = repair_items(sorted(items, key=lambda it: _pid(it)), packets, rows, packets_dir)
    return [{"packet_id": f"rep-{i // n:03d}", "items": rep[i:i + n]} for i in range(0, len(rep), n)]


def repair_instructions(repair_dir: str, rubric_path: str, raw_pattern: str, packet_ids) -> str:
    ids = list(packet_ids)
    return f"""# REPAIR INSTRUCTIONS: M7 Q3 comparative questions, one repair round (ruling R54)

The blind readers rejected almost every Q3 item, mostly because the question or answer used facts that are in no
gold fragment (an adoption body or date, a committee name, an invented description of a generic amending law).
Your job: for each item, either REWRITE it so that it satisfies the reader rubric v2, or DROP it with a reason.

## Files

- Rubric v2 (binding; read it in full first): `{rubric_path}`
- Repair packets: `{repair_dir}/{ids[0] if ids else '-'}.json` .. `{ids[-1] if ids else '-'}.json` (10 items each).
  Do only the packets the controller assigns you.
- Each item gives: `qid`, `packet_id`, `kind`, `lang`, both `documents` (doc_id, title, doc_type), the current
  `question`, `answer`, `gold` (chunk_id, doc_id, text) and `reader_verdicts` (each reader's verdict and reasons,
  verbatim). `source_packet` is the generator packet: its `docs[].chunks[]` are the only chunks you may use as gold.
- Output: `{raw_pattern}` (the file the controller names). JSONL, exactly ONE line per item, in packet order.
- Use only your repair packet and the source packets it names. Do not read code, other outputs or reports.
  No LLM API calls, no git.

## Output lines

Rewrite:
```json
{{"packet_id": "<packet_id>", "qid": "<qid>", "qtype": "compare", "template": "<kind>", "lang": "<lang>",
 "question": "...", "answer": "...", "gold": ["<chunk_id>", "..."], "docs": ["<doc_id A>", "<doc_id B>"],
 "meta": {{"packet_id": "<packet_id>", "repair": "one sentence: what you changed"}}}}
```
Drop:
```json
{{"packet_id": "<packet_id>", "qid": "<qid>", "refusal": "why no rubric-v2 item is possible from this pair"}}
```

## Rules (rubric v2 in short)

1. **Identifiers only from gold.** In the question, an act title, number or date may appear only if it is verbatim
   in a gold fragment's text or in that document's `title`. Never add the adoption body / date («Қонунчилик палатаси
   томонидан ... қабул қилинган»), a committee name, or a description that no gold fragment states. A date is never
   required. A law with a generic amending title is named by its subject or the article it amends, as gold states it.
2. **The answer is entailed by gold alone.** Every claim is in a gold fragment's text. No «it kept / previously /
   only one change» claims the fragments do not show.
3. **Needs both documents (strict).** A fact from a gold fragment of EACH document is required; if either document
   alone answers the question, the item is bad.
4. **Gold:** 2–5 chunk ids from the source packet's shown chunks, at least one from each document; minimal: every
   gold chunk contributes to the answer. You may re-pick gold.
5. One question, one point of comparison; no 8-word copy from any chunk (titles included); no leak of the answer;
   no pointer words; the language and script of `lang` (rubric q4).
6. Fix every reason the readers gave. If you cannot satisfy all of the above, DROP. A drop costs nothing; a forced
   item is rejected again.

Before you finish, check the file is valid JSONL with exactly one line per item of your packet.
"""


# --- R54: the final Q3 set -----------------------------------------------------------------------------------------

MIN_Q3_TOTAL, MIN_Q3_TEST = 30, 10


def q3_final(rows) -> tuple[list[dict], dict]:
    """R54 acceptance: >= MIN_Q3_TOTAL accepted and >= MIN_Q3_TEST test items -> every split kept and Q3 is reported on
    test. Otherwise Q3 is train / val only: its test items are EXCLUDED (not relabelled -- their documents are fixed to
    test by R32, moving them would put test documents into train / val), and Q3 is not in the hard test report.
    Returns (rows kept, stats)."""
    by = Counter(r["split"] for r in rows)
    ok = len(rows) >= MIN_Q3_TOTAL and by["test"] >= MIN_Q3_TEST
    kept = list(rows) if ok else [r for r in rows if r["split"] in ("train", "val")]
    return kept, {"rule": "R54", "consensus": len(rows), "consensus_by_split": dict(sorted(by.items())),
                  "min_total": MIN_Q3_TOTAL, "min_test": MIN_Q3_TEST, "test_reportable": ok,
                  "in_hard_test_report": ok, "test_diagnostic": by["test"] >= MIN_Q3_TEST,
                  "excluded_test": sorted(r["qid"] for r in rows if r not in kept),
                  "final": len(kept), "final_by_split": dict(sorted(Counter(r["split"] for r in kept).items())),
                  "final_by_kind": dict(sorted(Counter(r["template"] for r in kept).items())),
                  "final_by_lang": dict(sorted(Counter(r["lang"] for r in kept).items()))}
