#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m dream_dllm_hils.train_fulltext \
  --config configs/dream_dllm_hils/dolma3_8k_dual_gpu.json \
  --corpus_bin data/dolma3_dolmino_subset/dream-2k-smoke/train.bin \
  --corpus_meta data/dolma3_dolmino_subset/dream-2k-smoke/train.meta.json \
  --max_length 2048 \
  --max_steps 1 \
  --gradient_accumulation_steps 1 \
  --lora_r 4 \
  --lora_alpha 8 \
  --save_steps 0 \
  --log_steps 1 \
  --output_dir outputs/dream-hils-2k-7b-smoke \
  "$@"
