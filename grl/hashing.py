"""sha256 of a file, stdlib only (plan Task 7): shared by grl.train.log and scripts/spark.py, which must not import
torch."""
from __future__ import annotations

import hashlib


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
