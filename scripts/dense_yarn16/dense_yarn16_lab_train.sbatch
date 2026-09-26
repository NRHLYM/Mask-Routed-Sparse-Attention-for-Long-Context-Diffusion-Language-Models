#!/usr/bin/env bash
#SBATCH --job-name=dense-y16-32k-lab
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=360G
#SBATCH --gres=gpu:5
#SBATCH --time=24:00:00
set -euo pipefail
source scripts/dense_yarn16_lab_env.sh
CUDA_VISIBLE_DEVICES="$("$PYTHON" scripts/select_allocated_gpus.py 5)"
export CUDA_VISIBLE_DEVICES
printf 'job=%s CUDA_VISIBLE_DEVICES=%s hostname=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES" "$(hostname)"

CONFIG=configs/dense-yarn16-32k.json
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

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=5 \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}"
