#!/usr/bin/env bash
#SBATCH --job-name=hils-gate-probe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/hils-gate-bce-ceforce-s100-%j.out
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/hils-gate-bce-ceforce-s100-%j.err

set -euo pipefail
ROOT=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917
source "${ROOT}/scripts/fullteacher_env.sh"
cd "$ROOT"
mkdir -p "$ROOT/logs"
CONFIG="${1:-configs/from_dense16k/hils-gate-bce-ceforce-tilde.json}"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG"
