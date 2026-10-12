#!/usr/bin/env bash
# DGX Spark environment for M4/M5 (plan Task 12): runs ON the Spark inside ~/abdurakhmon/grl/advanced-rag-v2.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$PWD" in "$HOME"/abdurakhmon/*) ;; *) echo "refusing to run outside ~/abdurakhmon ($PWD)"; exit 1;; esac
C="$HOME/abdurakhmon/.cache"                       # shared account: every cache / temp dir under ~/abdurakhmon
export XDG_CACHE_HOME="$C" PIP_CACHE_DIR="$C/pip" MPLCONFIGDIR="$C/matplotlib" CUDA_CACHE_PATH="$C/nv/ComputeCache" \
       TRITON_HOME="$C/triton" TRITON_CACHE_DIR="$C/triton/cache" HF_HOME="$C/huggingface" TORCH_HOME="$C/torch" \
       TORCHINDUCTOR_CACHE_DIR="$C/torchinductor" TMPDIR="$HOME/abdurakhmon/tmp" \
       ONEDNN_JIT_PROFILE=0 DNNL_JIT_PROFILE=0          # oneDNN perf maps go to /tmp regardless of TMPDIR
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
PY=${PY:-python3.12}
TORCH_VERSION=${TORCH_VERSION:-2.14.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip wheel
if ! .venv/bin/pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"; then
  echo "torch==${TORCH_VERSION} is not on ${TORCH_INDEX} for $(uname -m); available:"
  .venv/bin/pip index versions torch --index-url "$TORCH_INDEX" || true
  exit 2
fi
grep -v -E '^((torch|torchvision|torchaudio)==|nvidia-|triton)' requirements.lock > .spark-req.txt   # keeps torch-geometric
.venv/bin/pip install -r .spark-req.txt
.venv/bin/pip install "matplotlib>=3.9" "pytest>=8.3"
.venv/bin/pip install -e . --no-deps
.venv/bin/python - <<'PYEOF'
import platform
import torch, torch_geometric
assert platform.machine() == "aarch64", platform.machine()
assert torch.cuda.is_available(), "CUDA is not available to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "sm", torch.cuda.get_device_capability(0), "pyg", torch_geometric.__version__)
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok", float((x @ x).abs().mean()))
PYEOF
.venv/bin/python -c "from graphrag.env.data import load_env_data; d = load_env_data(); print('env', d.version, d.n_nodes)"
