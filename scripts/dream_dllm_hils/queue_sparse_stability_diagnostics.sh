#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINER="scripts/dream_dllm_hils/train_2k_dual_gpu.sh"
SCHEME2_METRICS="outputs/dream-hils-entropy256-dense21-dolma3-2k/longbench_mfen_exact/metrics.json"
MAX_USED_MIB=5000

cd "$ROOT"
mkdir -p outputs

exec 9>/tmp/dream-hils-stability-diagnostics.lock
if ! flock -n 9; then
  echo "another stability-diagnostic queue owns the lock"
  exit 3
fi

echo "[$(date -Is)] waiting for scheme-2 MFEN metrics"
while [[ ! -f "$SCHEME2_METRICS" ]]; do
  sleep 60
done

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

run_diagnostic() {
  local config="$1"
  local output_dir="$2"
  local log="$3"
  if [[ -d "${output_dir}/step-100" ]]; then
    echo "[$(date -Is)] diagnostic already complete: $output_dir"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    echo "incomplete diagnostic output exists: $output_dir" >&2
    exit 4
  fi
  echo "[$(date -Is)] starting diagnostic: $config"
  "$TRAINER" "$config" > "$log" 2>&1
  test -f "${output_dir}/step-100/checkpoint_manifest.json"
}

run_diagnostic \
  configs/dream_dllm_hils/diagnostic_hils_vo_only_100.json \
  outputs/diagnostic-hils-vo-only-100 \
  outputs/diagnostic-hils-vo-only-100.log

run_diagnostic \
  configs/dream_dllm_hils/diagnostic_hils_low_lr_100.json \
  outputs/diagnostic-hils-low-lr-100 \
  outputs/diagnostic-hils-low-lr-100.log

/home/ma-user/work/venvs/d2f/bin/python \
  scripts/dream_dllm_hils/analyze_training_stability.py \
  outputs/dream-dense-dolma3-2k.log \
  outputs/dream-hils-dense21-dolma3-2k.log \
  outputs/diagnostic-hils-vo-only-100.log \
  outputs/diagnostic-hils-low-lr-100.log \
  --output outputs/sparse-stability-diagnosis.json

/home/ma-user/work/venvs/d2f/bin/python \
  scripts/dream_dllm_hils/compare_token_policy_runs.py \
  > outputs/chunk-token-policy-comparison.log

echo "[$(date -Is)] stability diagnostics completed"
