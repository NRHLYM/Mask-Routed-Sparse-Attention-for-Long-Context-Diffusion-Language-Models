#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT"

PHYSICAL_LENGTH="${PHYSICAL_LENGTH:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
ANSWER_TOKENS="${ANSWER_TOKENS:-64}"
STEPS="${STEPS:-64}"
LIMIT="${LIMIT:-50}"
TASKS="${TASKS:-qasper hotpotqa 2wikimqa musique triviaqa}"

MASK_CONFIG="configs/dream_dllm_hils/mixed_ruler05_hils_mask_lmk_hisa512_lr5e-5_500.json"
MASK_CHECKPOINT="outputs/dream-mixed-ruler05-hils-mask-lmk-hisa512-lr5e-5-dense21-dolma3-2k/step-500"
GUMBEL_CONFIG="configs/dream_dllm_hils/mixed_ruler05_hils_mask_lmk_hisa512_lr5e-5_500_gumbel_softmax_topk.json"
GUMBEL_CHECKPOINT="outputs/dream-mixed-ruler05-hils-mask-lmk-hisa512-lr5e-5-gumbel-softmax-topk-dense21-dolma3-2k/step-500"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/longbench_multi_task_compare_2k_limit${LIMIT}}"

run_eval() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local task="$4"
  local out="$OUTPUT_ROOT/$label/$task"

  if [[ -f "$out/metrics.json" ]]; then
    echo "[longbench-compare] skip $label $task: $out/metrics.json"
    return
  fi
  echo "[longbench-compare] start $label $task"
  TASK="$task" \
  PHYSICAL_LENGTH="$PHYSICAL_LENGTH" \
  CHUNK_SIZE="$CHUNK_SIZE" \
  ANSWER_TOKENS="$ANSWER_TOKENS" \
  STEPS="$STEPS" \
  LIMIT="$LIMIT" \
    scripts/dream_dllm_hils/run_mfen_exact_dual_gpu.sh \
      "$config" "$checkpoint" "$out"
  echo "[longbench-compare] done $label $task"
}

for task in $TASKS; do
  run_eval mask "$MASK_CONFIG" "$MASK_CHECKPOINT" "$task"
  run_eval gumbel "$GUMBEL_CONFIG" "$GUMBEL_CHECKPOINT" "$task"
done

echo "[longbench-compare] all done: $OUTPUT_ROOT"
