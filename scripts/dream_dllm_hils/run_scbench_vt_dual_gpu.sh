#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TRAINING_CONFIG CHECKPOINT OUTPUT_DIR" >&2
  echo "optional env: MODE=oracle_2k|long_context PHYSICAL_LENGTH=2048 LIMIT=0" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"
CONFIG="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"

MODE="${MODE:-oracle_2k}"
ORACLE_SCOPE="${ORACLE_SCOPE:-all_assignments}"
LONG_TRUNCATION="${LONG_TRUNCATION:-assignment_windows}"
PHYSICAL_LENGTH="${PHYSICAL_LENGTH:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
ANSWER_TOKENS="${ANSWER_TOKENS:-64}"
STEPS="${STEPS:-64}"
LIMIT="${LIMIT:-0}"
DATA="${DATA:-/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/data_scbench/scbench_vt.jsonl}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$CHECKPOINT/checkpoint_manifest.json" ]] \
  || [[ ! -f "$CHECKPOINT/trainable_state.pt" ]]; then
  echo "checkpoint is incomplete: $CHECKPOINT" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  --training_config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --data "$DATA"
  --output_dir "$OUTPUT_DIR"
  --mode "$MODE"
  --oracle_scope "$ORACLE_SCOPE"
  --long_truncation "$LONG_TRUNCATION"
  --physical_length "$PHYSICAL_LENGTH"
  --chunk_size "$CHUNK_SIZE"
  --answer_tokens "$ANSWER_TOKENS"
  --steps "$STEPS"
  --limit "$LIMIT"
)

"$PYTHON" scripts/dream_dllm_hils/eval_scbench_vt.py \
  "${COMMON_ARGS[@]}" \
  --rank 0 --world_size 2 --device cuda:0 \
  >> "$OUTPUT_DIR/rank-0.log" 2>&1 &
rank0_pid=$!

"$PYTHON" scripts/dream_dllm_hils/eval_scbench_vt.py \
  "${COMMON_ARGS[@]}" \
  --rank 1 --world_size 2 --device cuda:1 \
  >> "$OUTPUT_DIR/rank-1.log" 2>&1 &
rank1_pid=$!

wait "$rank0_pid"
wait "$rank1_pid"

"$PYTHON" scripts/dream_dllm_hils/eval_scbench_vt.py \
  "${COMMON_ARGS[@]}" \
  --merge \
  > "$OUTPUT_DIR/merge.log" 2>&1

test -f "$OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
