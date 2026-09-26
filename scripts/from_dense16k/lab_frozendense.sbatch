#!/bin/bash
#SBATCH --job-name=hils-fdens-lab
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/Data/xiongjing/hils-evidence-recall-20260919/logs/%j-fdens.out
#SBATCH --error=/mnt/Data/xiongjing/hils-evidence-recall-20260919/logs/%j-fdens.err

set -euo pipefail
GUMBEL=/mnt/Data/xiongjing/hils-gumbel-all-20260917
PROBE=/mnt/Data/xiongjing/hils-evidence-recall-20260919
source "$GUMBEL/scripts/lab_env.sh"
export PYTHONPATH="$PROBE:$GUMBEL"
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "cuda_visible=${CUDA_VISIBLE_DEVICES}"
"$GUMBEL/venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m dream_dllm_hils.train_fulltext \
  --config "$PROBE/configs/hils-t4-lab-frozendense-lora-qcal.json"
