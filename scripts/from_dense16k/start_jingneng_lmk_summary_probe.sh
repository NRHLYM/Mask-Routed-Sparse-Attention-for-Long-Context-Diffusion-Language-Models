#!/usr/bin/env bash
# Jingneng GPU: 16k S-N LMK-summary probes on MASK / vocab / EOS.
# Conditions: clean, shuffle_lmk, shuffle_interior, surgery.
# 4 GPUs, one job per GPU; then plot bars + heatmaps.
# Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_lmk_summary_probe.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true

PYTHON="${PYTHON:-python}"
PROBE="$NSA_ROOT/scripts/from_dense16k/probe_lmk_summary.py"
PLOT="$NSA_ROOT/scripts/from_dense16k/plot_lmk_summary.py"
OUT_ROOT="/Data/xiongjing/outputs/lmk_summary_probe_sn16k"
LOG="$ROOT/logs/lmk-summary-probe-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUT_ROOT" "$ROOT/logs"

stage_tree() {
  local tree="/Data/xiongjing/src/eval-trees/lmk-summary-probe"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local name
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  rm -f "$tree/dream_dllm_hils/attention.py" \
    "$tree/dream_dllm_hils/qcal.py" \
    "$tree/dream_dllm_hils/train_fulltext.py"
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/qcal.py" "$tree/dream_dllm_hils/qcal.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  grep -q "_lmk_summary_hook" "$tree/dream_dllm_hils/attention.py" \
    || { echo "attention.py missing lmk summary hook" >&2; exit 1; }
  grep -q "residual-random-lowrank-rmsnorm-v1" "$tree/dream_dllm_hils/qcal.py" \
    || { echo "qcal missing RMSNorm" >&2; exit 1; }
  echo "$tree"
}

TREE="$(stage_tree)"
export PYTHONPATH="$TREE"

declare -A CFG=(
  [mask]="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json"
  [vocab]="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-learned-route-s1000-jingneng.json"
  [eos]="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-eos-route-s1000-jingneng.json"
)
declare -A CKPT=(
  [mask]="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000"
  [vocab]="/Data/xiongjing/outputs/hils-s2-qcal-ablate-learned-route-s1000/step-1000"
  [eos]="/Data/xiongjing/outputs/hils-s2-qcal-ablate-eos-route-s1000/step-1000"
)

for arm in mask vocab eos; do
  [[ -s "${CKPT[$arm]}/trainable_state.pt" ]] || { echo "missing ${CKPT[$arm]}" >&2; exit 1; }
done
[[ -f "$PROBE" && -f "$PLOT" ]] || { echo "missing probe/plot" >&2; exit 1; }

jobs=()
for arm in mask vocab eos; do
  for cond in clean shuffle_lmk shuffle_interior surgery; do
    jobs+=("${arm}:${cond}")
  done
done

run_one() {
  local arm="$1" cond="$2" gpu="$3"
  local dest="$OUT_ROOT/${arm}/${cond}"
  if [[ -s "$dest/metrics.json" ]]; then
    echo "skip $arm $cond"
    return 0
  fi
  mkdir -p "$dest"
  echo "RUN arm=$arm cond=$cond gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TREE" \
    "$PYTHON" "$PROBE" \
      --arm "$arm" \
      --condition "$cond" \
      --training_config "${CFG[$arm]}" \
      --checkpoint "${CKPT[$arm]}" \
      --output_dir "$dest"
}

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "===== LMK summary probe 16k S-N MASK/vocab/EOS ====="
  echo "devices=${GPUS[*]} out=$OUT_ROOT tree=$TREE pending=${#jobs[@]}"
  pids=()
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    (
      for ((j=i; j<${#jobs[@]}; j+=${#GPUS[@]})); do
        IFS=':' read -r arm cond <<< "${jobs[$j]}"
        run_one "$arm" "$cond" "$gpu"
      done
    ) > "$OUT_ROOT/worker-gpu${i}.log" 2>&1 &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if (( fail != 0 )); then
    echo "worker failed; see $OUT_ROOT/worker-gpu*.log" >&2
    exit 1
  fi
  "$PYTHON" "$PLOT" --root "$OUT_ROOT" --out_dir "$OUT_ROOT"
  echo "LMK_SUMMARY_PROBE_DONE"
} 2>&1 | tee "$LOG"
