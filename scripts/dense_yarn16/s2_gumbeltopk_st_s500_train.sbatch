#!/usr/bin/env bash
#SBATCH --job-name=s2-gumbel-st
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s2-gumbeltopk-st-s500.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s2-gumbeltopk-st-s500.err

# s2-sync 1:1, YaRN 8, dense-500 init. Top-32 Gumbel-softmax is STE only:
# forward fusion is the noiseless 33-way softmax; backward uses Gumbel-softmax.
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/outputs" "$ROOT/configs/from_dense16k"

CONFIG=configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-gumbeltopk-st-s500-tilde.json
OUTPUT=outputs/hils-s2-cefusion-dolma-ruler-sync-gumbeltopk-st-s500
DENSE=outputs/dense-yarn8-16k-step500/step-500
FINAL="$OUTPUT/step-500"
NPROC=4

[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }

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

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}"
