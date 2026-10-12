"""§7.3 reward-weight calibration on val (plan Task 11; live M_eval table from M6 by decision D5).
    PYTHONPATH=. python scripts/calibrate_reward.py --table F [--method spearman|pearson] [--out data/m5/calibration/weights.json]
Writes the result; train.yaml reward.weights changes only by a user decision."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402  -- stdlib only, before numpy

pin_blas()

from graphrag import config  # noqa: E402
from grl.train.calibrate import calibrate  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--out", default="data/m5/calibration/weights.json")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in config.path(a.table).read_text(encoding="utf-8").splitlines() if l.strip()]
    res = calibrate(rows, method=a.method)
    out = config.path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
