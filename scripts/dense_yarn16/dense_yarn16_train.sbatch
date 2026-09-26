#!/usr/bin/env bash
#SBATCH --job-name=dense-y16-32k-train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-y16-32k-train.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-y16-32k-train.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
cd "$ROOT"
CONFIG=configs/dense_yarn16/dense-yarn16-32k.json
OUTPUT=outputs/dense-yarn16-32k-step500
FINAL="$OUTPUT/step-500"

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}
if checkpoint_is_complete "$FINAL"; then
  echo "completed checkpoint already exists: $FINAL"
  exit 0
fi
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$resume_from" ]] && checkpoint_is_complete "$resume_from"; then
  start_args=(--resume_from "$resume_from")
else
  start_args=()
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}"
