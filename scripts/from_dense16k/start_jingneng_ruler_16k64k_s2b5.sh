#!/usr/bin/env bash
# Official goldspan RULER for HiLS S2 B5 (Q-Cal 0.5x) step-1000.
# SN / MK-MQ / VT. 16k YaRN 8 on GPUs 0,1; 64k factor=32 on GPUs 2,3.
# Do not overlap value-fusion (4 GPU) on the same devices.
#
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2b5.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="/Data/xiongjing/outputs/hils-s2-ablate-b5-qcal-lr0p5-s1000/step-1000"
[[ -s "$CKPT/trainable_state.pt" ]] || { echo "missing $CKPT" >&2; exit 1; }
[[ -s "$CKPT/checkpoint_manifest.json" ]] || { echo "incomplete $CKPT" >&2; exit 1; }

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2b5 RULER split 16k=${GPUS[0]},${GPUS[1]} 64k=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" s2b5 &
  p16=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k.sh" s2b5 &
  p64=$!
  fail=0
  wait "$p16" || fail=1
  wait "$p64" || fail=1
  (( fail == 0 )) || { echo "s2b5 RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  bash "$HERE/start_jingneng_ruler_16k.sh" s2b5
  bash "$HERE/start_jingneng_ruler_64k.sh" s2b5
fi
echo "s2b5_RULER_16K64K_DONE"
