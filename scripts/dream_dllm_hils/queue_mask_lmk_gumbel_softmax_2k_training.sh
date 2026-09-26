#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"
CONFIG="configs/dream_dllm_hils/mixed_ruler05_hils_mask_lmk_hisa512_lr5e-5_500_gumbel_softmax_topk.json"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

config_value() {
  local key="$1"
  "$PYTHON" - "$CONFIG" "$key" <<PY
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
print(payload[sys.argv[2]])
PY
}

output_dir="$(config_value output_dir)"
checkpoint="$output_dir/step-500"
eval_dir="$output_dir/longbench_mfen_exact"
mkdir -p "$output_dir"

if [[ -f "$checkpoint/checkpoint_manifest.json" ]] \
  && [[ -f "$checkpoint/trainable_state.pt" ]]; then
  echo "[mask-gumbel-2k-queue] train skip complete: $checkpoint"
else
  echo "[mask-gumbel-2k-queue] train start: $CONFIG"
  scripts/dream_dllm_hils/train_2k_dual_gpu.sh "$CONFIG" \
    2>&1 | tee "$output_dir/train.log"
  echo "[mask-gumbel-2k-queue] train done: $CONFIG"
fi

if [[ -f "$eval_dir/metrics.json" ]]; then
  echo "[mask-gumbel-2k-queue] eval skip complete: $eval_dir"
else
  echo "[mask-gumbel-2k-queue] eval start: $checkpoint"
  PHYSICAL_LENGTH=2048 CHUNK_SIZE=64 ANSWER_TOKENS=64 STEPS=64 LIMIT="${MFEN_LIMIT:-0}" \
    scripts/dream_dllm_hils/run_mfen_exact_dual_gpu.sh \
      "$CONFIG" "$checkpoint" "$eval_dir" \
    2>&1 | tee "$output_dir/eval_mfen_2k.log"
  echo "[mask-gumbel-2k-queue] eval done: $eval_dir"
fi

echo "[mask-gumbel-2k-queue] all done"
