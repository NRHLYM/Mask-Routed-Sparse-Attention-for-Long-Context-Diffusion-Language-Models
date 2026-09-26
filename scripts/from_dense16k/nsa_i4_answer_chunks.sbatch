#!/usr/bin/env bash
#SBATCH --job-name=nsa-i4-chunks
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CONFIG=configs/from_dense16k/nsa-yarn8-16k-b32-w512.json
CHECKPOINT=outputs/nsa-yarn8-16k-b32-w512-i4-fromdense/step-500
OUTPUT=outputs/nsa-yarn8-16k-b32-w512-i4-fromdense/answer-chunks.json
PROBE=scripts/probe_answer_chunks.py

test -s "$CHECKPOINT/trainable_state.pt"
test -f "$PROBE"
test -s "$DATA_ROOT/multifieldqa_en.jsonl"
mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON" "$PROBE" \
  --training_config "$CONFIG" \
  --model_path "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT/multifieldqa_en.jsonl" \
  --prompt_config "$PROMPTS" \
  --output "$OUTPUT" \
  --physical_length 16384 \
  --answer_tokens 64 \
  --device cuda:0
echo NSA_I4_CHUNKS_DONE
