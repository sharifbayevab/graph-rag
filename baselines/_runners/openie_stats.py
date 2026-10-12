"""Empty-extraction rates of a built index: python openie_stats.py <save_dir> -> prints JSON (run where the index is)."""
import json
import sys
from pathlib import Path

d = json.load(open(Path(sys.argv[1]) / "openie_results_ner_gpt-4.1-mini.json"))["docs"]
n = len(d)
out = {"passages": n, "empty_entities": sum(not x.get("extracted_entities") for x in d)}
if "extracted_triples" in d[0] and "propositions" not in d[0]:   # PropRAG keeps an unused, always-empty field
    out["empty_triples"] = sum(not x.get("extracted_triples") for x in d)
if "propositions" in d[0]:
    out["empty_propositions"] = sum(not x.get("propositions") for x in d)
for k in [k for k in out if k.startswith("empty")]:
    out[k + "_rate"] = round(out[k] / n, 4)
print(json.dumps(out))
