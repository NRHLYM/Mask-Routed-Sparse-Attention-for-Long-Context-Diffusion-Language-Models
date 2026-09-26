#!/usr/bin/env bash
#SBATCH --job-name=dream_dsa_32k_gate
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_from_warmup_profile_%j.out
#SBATCH --error=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_from_warmup_profile_%j.err

set -euo pipefail

ROOT=/home/guests/zhen/dsa_baselines/HiLS-Attention
PYTHON=/home/guests/zhen/dsa_baselines/venvs/dream_dsa/bin/python
MODEL=/home/guests/zhen/jing/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
CONFIG=configs/dream_dllm_hils/mixed_ruler05_32k_yarn_dsa_topk1024_lr2e-4_500.json
INITIALIZER=outputs/dream-dsa-2k-topk2048-indexer-warmup-34/step-34
OUTPUT=outputs/dream-dsa-32k-topk1024-from-warmup-selected-profile-tilde

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG=WARN

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
start_epoch=$(date +%s)
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext \
  --config "$CONFIG" \
  --model_path "$MODEL" \
  --initialize_from "$INITIALIZER" \
  --output_dir "$OUTPUT" \
  --stop_after_steps 1 \
  --validation_packs 0 \
  --validation_batches 0 \
  --eval_steps 0 \
  --save_steps 1 \
  --log_steps 1
end_epoch=$(date +%s)
echo "profile_wall_seconds=$((end_epoch - start_epoch))"
