#!/usr/bin/env bash
#SBATCH --job-name=dream_dsa_32k_eval
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_eval_%j.out
#SBATCH --error=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_eval_%j.err

set -euo pipefail

ROOT="${ROOT:-/home/guests/zhen/dsa_baselines/HiLS-Attention}"
PYTHON="${PYTHON:-/home/guests/zhen/dsa_baselines/venvs/dream_dsa/bin/python}"
MODEL="${MODEL:-/home/guests/zhen/jing/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B}"
CONFIG="${CONFIG:-configs/dream_dllm_hils/mixed_ruler05_32k_yarn_dsa_topk1024_lr2e-4_500.json}"
CHECKPOINT="${CHECKPOINT:-outputs/dream-mixed-ruler05-32k-yarn-dsa-topk1024-lr2e-4-dolma3/step-500}"
OUTPUT="${OUTPUT:-outputs/dream-mixed-ruler05-32k-yarn-dsa-topk1024-lr2e-4-dolma3/longbench_mfen_exact_32k}"
DATA="${DATA:-/home/guests/zhen/jing/ParallelComp_official/datasets/LongBench/multifieldqa_en.jsonl}"
PROMPTS="${PROMPTS:-/home/guests/zhen/jing/ParallelComp_official/longbench_config/dataset2prompt_raw.json}"
LIMIT="${LIMIT:-0}"

cd "$ROOT"
mkdir -p logs

for required in \
  "$CHECKPOINT/checkpoint_manifest.json" \
  "$CHECKPOINT/trainable_state.pt" \
  "$DATA" \
  "$PROMPTS"; do
  if [[ ! -f "$required" ]]; then
    echo "required input is missing: $required" >&2
    exit 2
  fi
done

PYTHON="$PYTHON" \
MODEL_PATH="$MODEL" \
DATA="$DATA" \
PROMPT_CONFIG="$PROMPTS" \
PHYSICAL_LENGTH=32768 \
CHUNK_SIZE=64 \
ANSWER_TOKENS=64 \
STEPS=64 \
LIMIT="$LIMIT" \
  scripts/dream_dllm_hils/run_mfen_exact_dual_gpu.sh \
    "$CONFIG" "$CHECKPOINT" "$OUTPUT"
