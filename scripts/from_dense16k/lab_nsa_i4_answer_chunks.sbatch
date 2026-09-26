#!/usr/bin/env bash
#SBATCH --job-name=nsa-i4-chunks
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/Data/xiongjing/nsa-probe-20260918/logs/%j-nsa-i4-chunks.out
#SBATCH --error=/mnt/Data/xiongjing/nsa-probe-20260918/logs/%j-nsa-i4-chunks.err

set -euo pipefail
GUMBEL=/mnt/Data/xiongjing/hils-gumbel-all-20260917
NSA=/mnt/Data/xiongjing/nsa-probe-20260918
PYTHON="$GUMBEL/venv/bin/python"
export PYTHONPATH="$NSA"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="$NSA/tmp"
export TILELANG_CACHE_DIR="$NSA/tilelang-cache"
export TVM_FFI_CACHE_DIR="$NSA/tvm-ffi-cache"
export TRITON_CACHE_DIR="$NSA/triton-cache"
mkdir -p "$TMPDIR" "$TILELANG_CACHE_DIR" "$TVM_FFI_CACHE_DIR" "$TRITON_CACHE_DIR" "$NSA/logs"

CUDA_VISIBLE_DEVICES="$("$PYTHON" "$GUMBEL/scripts/select_allocated_gpus.py" 1)"
export CUDA_VISIBLE_DEVICES
cd "$NSA"

CONFIG="$NSA/configs/from_dense16k/nsa-yarn8-16k-b32-w512-lab.json"
CHECKPOINT="$NSA/checkpoints/nsa-yarn8-16k-b32-w512-i4-fromdense/step-500"
OUTPUT="$NSA/outputs/nsa-yarn8-16k-b32-w512-i4-fromdense/answer-chunks.json"
DATA="$GUMBEL/multifieldqa_en.jsonl"
PROMPTS="$GUMBEL/dataset2prompt_raw.json"

test -s "$CHECKPOINT/trainable_state.pt"
test -f scripts/probe_answer_chunks.py
test -s "$DATA"
mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON" scripts/probe_answer_chunks.py \
  --training_config "$CONFIG" \
  --model_path "$GUMBEL/model-overlay" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --prompt_config "$PROMPTS" \
  --output "$OUTPUT" \
  --physical_length 16384 \
  --answer_tokens 64 \
  --device cuda:0
echo NSA_I4_CHUNKS_DONE
