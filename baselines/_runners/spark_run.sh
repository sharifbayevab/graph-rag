#!/usr/bin/env bash
# Spark launcher: spark_run.sh <method> <index|retrieve> [variant: base|struct|v2|kg] [EXT_ROOT]
# index: OPENAI_API_KEY is a dummy (extraction must replay from the cache). retrieve: the key is read from stdin.
set -euo pipefail
B=~/abdurakhmon/ext_baselines
M=$1; P=$2; V=${3:-base}; export EXT_ROOT=${4:-$B}
if [ "$V" != base ]; then export EXT_DATA=data_struct; fi
export HF_HOME=$B/hf TOKENIZERS_PARALLELISM=false PYTHONPATH=$B/scripts/shims
if [ "$P" = retrieve ]; then read -r OPENAI_API_KEY; else OPENAI_API_KEY=sk-dummy-index-phase; fi
export OPENAI_API_KEY
exec $B/venv/bin/python $B/scripts/run_method.py "$M" "$P" --variant "$V"
