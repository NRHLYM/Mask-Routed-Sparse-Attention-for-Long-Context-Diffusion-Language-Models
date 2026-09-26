#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext \
  --config configs/dream_dllm_hils/dolma3_8k_dual_gpu.json \
  "$@"
