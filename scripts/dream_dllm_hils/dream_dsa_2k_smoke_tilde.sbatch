#!/usr/bin/env bash
#SBATCH --job-name=dream_dsa_2k_smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_2k_smoke_%j.out
#SBATCH --error=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_2k_smoke_%j.err

set -euo pipefail

ROOT=/home/guests/zhen/dsa_baselines/HiLS-Attention
PYTHON=/home/guests/zhen/dsa_baselines/venvs/dream_dsa/bin/python
CONFIG=configs/dream_dllm_hils/dream_dsa_2k_topk1024_smoke.json

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext \
  --config "$CONFIG"
