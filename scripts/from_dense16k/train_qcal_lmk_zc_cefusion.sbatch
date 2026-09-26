#!/usr/bin/env bash
#SBATCH --job-name=hils-zc-lmk
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
cd "$ROOT"
CONFIG=configs/from_dense16k/hils-t4-yarn8-16k-k32-w512-hardroute-qcal-lmk-frozendense-attnmass.json
DENSE="$ROOT/outputs/dense-yarn8-16k-step500/step-500"
OUTPUT="$(python3 - <<PY
import json
print(json.load(open("$CONFIG"))["output_dir"])
PY
)"
FINAL="$OUTPUT/step-500"
checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}
if ! checkpoint_is_complete "$DENSE"; then
  echo "missing completed dense 16k checkpoint: $DENSE" >&2
  exit 1
fi
if checkpoint_is_complete "$FINAL"; then
  echo "completed checkpoint already exists: $FINAL"
  exit 0
fi
start_args=()
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$resume_from" ]] && checkpoint_is_complete "$resume_from"; then
  start_args=(--resume_from "$resume_from")
fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}"
