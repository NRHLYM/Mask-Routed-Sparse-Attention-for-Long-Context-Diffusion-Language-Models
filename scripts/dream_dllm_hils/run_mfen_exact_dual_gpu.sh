#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TRAINING_CONFIG CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/ma-user/work/venvs/d2f/bin/python}"
CONFIG="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
PHYSICAL_LENGTH="${PHYSICAL_LENGTH:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
ANSWER_TOKENS="${ANSWER_TOKENS:-64}"
STEPS="${STEPS:-64}"
LIMIT="${LIMIT:-0}"
TASK="${TASK:-multifieldqa_en}"
DATA="${DATA:-}"
HILS_ROUTE_QUERY_SOURCE="${HILS_ROUTE_QUERY_SOURCE:-per_position}"
POSITION_ENCODING="${POSITION_ENCODING:-trained}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT_CONFIG="${PROMPT_CONFIG:-}"

DATA_ARGS=()
if [[ -n "$DATA" ]]; then
  DATA_ARGS=(--data "$DATA")
fi
MODEL_ARGS=()
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_ARGS=(--model_path "$MODEL_PATH")
fi
PROMPT_ARGS=()
if [[ -n "$PROMPT_CONFIG" ]]; then
  PROMPT_ARGS=(--prompt_config "$PROMPT_CONFIG")
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$CHECKPOINT/checkpoint_manifest.json" ]] \
  || [[ ! -f "$CHECKPOINT/trainable_state.pt" ]]; then
  echo "checkpoint is incomplete: $CHECKPOINT" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
  --training_config "$CONFIG" \
  "${MODEL_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --task "$TASK" \
  "${DATA_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --physical_length "$PHYSICAL_LENGTH" \
  --chunk_size "$CHUNK_SIZE" \
  --answer_tokens "$ANSWER_TOKENS" \
  --steps "$STEPS" \
  --hils_route_query_source "$HILS_ROUTE_QUERY_SOURCE" \
  --position_encoding "$POSITION_ENCODING" \
  --limit "$LIMIT" \
  --rank 0 --world_size 2 --device cuda:0 \
  >> "$OUTPUT_DIR/rank-0.log" 2>&1 &
rank0_pid=$!

"$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
  --training_config "$CONFIG" \
  "${MODEL_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --task "$TASK" \
  "${DATA_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --physical_length "$PHYSICAL_LENGTH" \
  --chunk_size "$CHUNK_SIZE" \
  --answer_tokens "$ANSWER_TOKENS" \
  --steps "$STEPS" \
  --hils_route_query_source "$HILS_ROUTE_QUERY_SOURCE" \
  --position_encoding "$POSITION_ENCODING" \
  --limit "$LIMIT" \
  --rank 1 --world_size 2 --device cuda:1 \
  >> "$OUTPUT_DIR/rank-1.log" 2>&1 &
rank1_pid=$!

wait "$rank0_pid"
wait "$rank1_pid"

"$PYTHON" scripts/dream_dllm_hils/eval_longbench_mfen.py \
  --training_config "$CONFIG" \
  "${MODEL_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --task "$TASK" \
  "${DATA_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --physical_length "$PHYSICAL_LENGTH" \
  --chunk_size "$CHUNK_SIZE" \
  --answer_tokens "$ANSWER_TOKENS" \
  --steps "$STEPS" \
  --hils_route_query_source "$HILS_ROUTE_QUERY_SOURCE" \
  --position_encoding "$POSITION_ENCODING" \
  --limit "$LIMIT" \
  --merge \
  > "$OUTPUT_DIR/merge.log" 2>&1

test -f "$OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
