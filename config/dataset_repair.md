# REPAIR INSTRUCTIONS: legal QA dataset, one-shot repair of reader-flagged pairs (version 1, ruling M3-R32)

You repair question-answer pairs that two independent strict readers rated BORDERLINE (neither rated them BAD).
Each packet item gives:
- the pair as it was generated: `question`, `answer`, `lang`, `pattern`, `used_facts`, `neighbor_id` and `sources`;
- the readers' reasons (`reasons.reader_1`, `reasons.reader_2`): the criterion (c1–c7) and a short note;
- the seed exactly as the generator saw it (`seed`: the fragments with their text and meta, the cached facts, and for
  a multi pair the anchor `A` and the one neighbour the pair uses).

The GENERATOR INSTRUCTIONS embedded in this packet (`generator_instructions`) still apply in full: every rule, code
check and rubric criterion.

## What you may do

- Fix ONLY the defects the readers name, for example:
  - the act or the amended provision is misnamed;
  - a session date or speaker is missing or wrong (use `session.dates` / `speaker` from the meta);
  - Uzbek words or glosses appear in a Russian text;
  - the question is a near-copy of a fragment sentence;
  - wording is awkward;
  - the answer is incomplete.
- Keep the same information need, the same fragments (for multi, the same `neighbor_id` and `sources`), and the same
  facts. Every value you state must remain verbatim-grounded: cite the pair's `used_facts` again, or give new verbatim
  quotes from the SAME fragments.
- Rewrite the question and/or the answer as needed to remove the named defects, without introducing new ones.

## What you must not do

- Do not change the substance: no new fact, no other fragment, no other neighbour, no different question topic.
- Do not keep a defect a reader named because you disagree with the reader.
- Do not add explanations, meta-comments or references to the repair into the question or the answer.

If the pair cannot be fixed without changing its substance, output a `skip` with the reason. Examples:
- the fragment never names the amended act;
- the speech does not name the law it discusses;
- the needed date is not in the meta;
- the only answerable question is a copy of the sentence.

## Output

Write the packet's `output.path`: JSONL, exactly one line per item, in packet order, and nothing else. Use the
generator schema keyed by the item's `seed_id`:

```json
{"seed_id": "...", "question": "...", "answer": "...", "lang": "...", "used_facts": [...],
 "neighbor_id": null | "N<k>", "sources": [...], "notes": "which defects were fixed and how"}
```

or

```json
{"seed_id": "...", "skip": true, "reason": "..."}
```

There is ONE attempt per pair. The repaired pair is checked again from scratch: all code checks, the blind judge
with its ablation, and two new independent strict readers who do not know it was repaired.
