#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"
SPARSE_CONFIG="configs/dream_dllm_hils/dolma3_2k_hils_dense21_dual_gpu.json"
DENSE_CONFIG="configs/dream_dllm_hils/dolma3_2k_dense_dual_gpu.json"
SPARSE_OUTPUT="outputs/dream-hils-dense21-dolma3-2k"
DENSE_OUTPUT="outputs/dream-dense-dolma3-2k"
SPARSE_FINAL="$SPARSE_OUTPUT/step-500"
DENSE_FINAL="$DENSE_OUTPUT/step-500"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" ]]
}

latest_checkpoint() {
  if [[ ! -d "$1" ]]; then
    return 0
  fi
  find "$1" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null \
    | sort -V \
    | tail -n 1
}

echo "waiting for sparse step-500 checkpoint" >&2
while ! checkpoint_is_complete "$SPARSE_FINAL"; do
  if ! pgrep -f "dream_dllm_hils.train_fulltext.*dolma3_2k_hils_dense21_dual_gpu.json" >/dev/null; then
    echo "sparse trainer exited before producing $SPARSE_FINAL" >&2
    exit 1
  fi
  sleep 60
done

while pgrep -f "dream_dllm_hils.train_fulltext.*dolma3_2k_hils_dense21_dual_gpu.json" >/dev/null; do
  sleep 10
done

if ! checkpoint_is_complete "$DENSE_FINAL"; then
  dense_resume="$(latest_checkpoint "$DENSE_OUTPUT")"
  dense_args=()
  if [[ -n "$dense_resume" ]]; then
    dense_args+=(--resume_from "$dense_resume")
  fi
  NCCL_DEBUG=WARN bash scripts/dream_dllm_hils/train_2k_dual_gpu.sh \
    "$DENSE_CONFIG" "${dense_args[@]}" \
    >> "$DENSE_OUTPUT.log" 2>&1
fi

run_mfen() {
  local config="$1"
  local checkpoint="$2"
  local output_dir="$3"

  mkdir -p "$output_dir"
  "$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
    --training_config "$config" \
    --checkpoint "$checkpoint" \
    --output_dir "$output_dir" \
    --physical_length 2048 \
    --chunk_size 64 \
    --answer_tokens 64 \
    --steps 64 \
    --rank 0 --world_size 2 --device cuda:0 \
    >> "$output_dir/rank-0.log" 2>&1 &
  rank0_pid=$!
  "$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
    --training_config "$config" \
    --checkpoint "$checkpoint" \
    --output_dir "$output_dir" \
    --physical_length 2048 \
    --chunk_size 64 \
    --answer_tokens 64 \
    --steps 64 \
    --rank 1 --world_size 2 --device cuda:1 \
    >> "$output_dir/rank-1.log" 2>&1 &
  rank1_pid=$!
  wait "$rank0_pid"
  wait "$rank1_pid"

  "$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
    --training_config "$config" \
    --checkpoint "$checkpoint" \
    --output_dir "$output_dir" \
    --physical_length 2048 \
    --chunk_size 64 \
    --answer_tokens 64 \
    --steps 64 \
    --merge \
    > "$output_dir/merge.log" 2>&1
}

run_mfen \
  "$SPARSE_CONFIG" \
  "$SPARSE_FINAL" \
  "$SPARSE_OUTPUT/longbench_mfen_exact"
run_mfen \
  "$DENSE_CONFIG" \
  "$DENSE_FINAL" \
  "$DENSE_OUTPUT/longbench_mfen_exact"

echo "dense and sparse MFEN evaluations completed" >&2
