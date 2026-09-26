#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="configs/dream_dllm_hils/dolma3_2k_hils_hisa256_dense21_dual_gpu.json"
TRAINER="scripts/dream_dllm_hils/train_2k_dual_gpu.sh"
SMOKE_DIR="outputs/dream-hils-hisa256-dense21-dolma3-2k-smoke"
FINAL_DIR="outputs/dream-hils-hisa256-dense21-dolma3-2k"
QUEUE_LOG="outputs/dream-hils-hisa256-dense21-dolma3-2k.queue.log"
EVAL_DIR="${FINAL_DIR}/longbench_mfen_exact"
MAX_USED_MIB=5000

cd "$ROOT"
mkdir -p outputs

exec 9>/tmp/dream-hils-hisa256-2k.lock
if ! flock -n 9; then
  echo "another HISA-256 queue or training process owns the lock"
  exit 3
fi

echo "[$(date -Is)] waiting for both GPUs to use < ${MAX_USED_MIB} MiB"
idle_checks=0
while (( idle_checks < 3 )); do
  mapfile -t used < <(
    nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits
  )
  if [[ ${#used[@]} -eq 2 ]] \
    && (( used[0] < MAX_USED_MIB )) \
    && (( used[1] < MAX_USED_MIB )); then
    ((idle_checks += 1))
    echo "[$(date -Is)] idle check ${idle_checks}/3: ${used[*]} MiB"
  else
    idle_checks=0
    echo "[$(date -Is)] GPUs busy: ${used[*]:-unavailable} MiB"
  fi
  if (( idle_checks < 3 )); then
    sleep 60
  fi
done

if [[ -d "${FINAL_DIR}/step-500" ]]; then
  echo "final step-500 already exists; refusing to train twice"
  exit 4
fi
if [[ -e "$SMOKE_DIR" ]]; then
  echo "smoke output already exists; refusing to overwrite $SMOKE_DIR"
  exit 5
fi

echo "[$(date -Is)] starting two-GPU one-step smoke"
"$TRAINER" "$CONFIG" \
  --stop_after_steps 1 \
  --output_dir "$SMOKE_DIR"

if [[ ! -f "${SMOKE_DIR}/step-1/checkpoint_manifest.json" ]]; then
  echo "one-step smoke did not produce a complete checkpoint"
  exit 6
fi

echo "[$(date -Is)] smoke passed; starting 500-step HISA-256 training"
"$TRAINER" "$CONFIG"
echo "[$(date -Is)] training completed; starting MFEN exact evaluation"
scripts/dream_dllm_hils/run_mfen_exact_dual_gpu.sh \
  "$CONFIG" \
  "${FINAL_DIR}/step-500" \
  "$EVAL_DIR"
echo "[$(date -Is)] training and MFEN evaluation completed"
