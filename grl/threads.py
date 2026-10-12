"""BLAS / OpenMP thread pinning (IMPLEMENTATION.md §9.6; plan Task 5). Stdlib only: every entry point imports it
before numpy (scripts/{train,m5_pilot_data,check_spark}.py), and ProcVecEnv forces it before spawning workers."""
from __future__ import annotations

import os

BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def pin_blas(n: int = 1, force: bool = False) -> None:
    for v in BLAS_VARS:
        if force:
            os.environ[v] = str(n)
        else:
            os.environ.setdefault(v, str(n))
