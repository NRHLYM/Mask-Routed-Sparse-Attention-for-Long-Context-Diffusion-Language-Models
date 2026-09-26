#!/usr/bin/env bash
#SBATCH --job-name=s2-from2k-hope
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s2-from2k-hope-s1000.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s2-from2k-hope-s1000.err

# Dream 2k + HoPE (no YaRN) + HiLS s2 1:1, cosine 1000, lr 1e-4, 4 GPU.
# Does not initialize from dense-500.
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/outputs" "$ROOT/configs/from_dense16k"

CONFIG=configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-from2k-hope-s1000-tilde.json
OUTPUT=outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-hope-s1000
FINAL="$OUTPUT/step-1000"
NPROC=4

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
