#!/usr/bin/env bash
# Official RULER for s2-alltasks (7+21, 4-task 0.25 means, step-500).
# 16k goldspan (YaRN 8) on GPUs 0,1; 64k factor=32 on GPUs 2,3.
# Two-GPU fallback: 16k then 64k on the same pair.
#
# GPU job terminal (from2k / alltasks machine; do not steal all28-s2000 GPUs):
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2alltasks.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-alltasks-s500/step-500"
[[ -s "$CKPT/trainable_state.pt" ]] || { echo "missing $CKPT" >&2; exit 1; }
[[ -s "$CKPT/checkpoint_manifest.json" ]] || { echo "incomplete $CKPT" >&2; exit 1; }

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2alltasks RULER split 16k=${GPUS[0]},${GPUS[1]} 64k=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" s2alltasks &
  p16=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k.sh" s2alltasks &
  p64=$!
  fail=0
  wait "$p16" || fail=1
  wait "$p64" || fail=1
  (( fail == 0 )) || { echo "s2alltasks RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  bash "$HERE/start_jingneng_ruler_16k.sh" s2alltasks
  bash "$HERE/start_jingneng_ruler_64k.sh" s2alltasks
fi
echo "s2alltasks_RULER_16K64K_DONE"
