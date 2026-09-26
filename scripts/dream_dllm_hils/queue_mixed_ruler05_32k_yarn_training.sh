#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONFIGS=(
  "configs/dream_dllm_hils/mixed_ruler05_32k_yarn_hils_mask_lmk_hisa512_lr2e-4_500.json"
  "configs/dream_dllm_hils/mixed_ruler05_32k_yarn_hils_external_lmk_hisa512_lr2e-4_500.json"
  "configs/dream_dllm_hils/mixed_ruler05_32k_yarn_dense_lr2e-4_500.json"
)

config_value() {
  local config="$1"
  local key="$2"
  "$PYTHON" - "$config" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
print(payload[sys.argv[2]])
PY
}

train_one() {
  local config="$1"
  local output_dir
  output_dir="$(config_value "$config" output_dir)"
  mkdir -p "$output_dir"
  if [[ -f "$output_dir/step-500/checkpoint_manifest.json" ]] \
    && [[ -f "$output_dir/step-500/trainable_state.pt" ]]; then
    echo "[queue] train skip complete: $config"
    return
  fi
  echo "[queue] train start: $config"
  scripts/dream_dllm_hils/train_2k_dual_gpu.sh "$config" \
    2>&1 | tee "$output_dir/train.log"
  echo "[queue] train done: $config"
}

eval_one() {
  local config="$1"
  local output_dir checkpoint eval_dir
  output_dir="$(config_value "$config" output_dir)"
  checkpoint="$output_dir/step-500"
  eval_dir="$output_dir/longbench_mfen_exact_32k"
  if [[ ! -f "$checkpoint/checkpoint_manifest.json" ]] \
    || [[ ! -f "$checkpoint/trainable_state.pt" ]]; then
    echo "[queue] eval missing checkpoint: $checkpoint" >&2
    return 1
  fi
  if [[ -f "$eval_dir/metrics.json" ]]; then
    echo "[queue] eval skip complete: $config"
    return
  fi
  echo "[queue] eval start: $config"
  PHYSICAL_LENGTH=32768 CHUNK_SIZE=64 ANSWER_TOKENS=64 STEPS=64 LIMIT="${MFEN_LIMIT:-0}" \
    scripts/dream_dllm_hils/run_mfen_exact_dual_gpu.sh \
      "$config" "$checkpoint" "$eval_dir" \
    2>&1 | tee "$output_dir/eval_mfen_32k.log"
  echo "[queue] eval done: $config"
}

for config in "${CONFIGS[@]}"; do
  train_one "$config"
done

for config in "${CONFIGS[@]}"; do
  eval_one "$config"
done

echo "[queue] all done"
