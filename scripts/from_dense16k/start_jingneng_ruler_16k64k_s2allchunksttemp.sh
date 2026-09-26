#!/usr/bin/env bash
# Official goldspan RULER for s2-sync + all-chunk ST with Jacobian τ cosine 1.0→0.1.
# 16k YaRN 8 on first pair; 64k factor=32 on second pair. Eval is hard top-k (ST off).
#
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2allchunksttemp.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500/step-500"
until [[ -s "$CKPT/trainable_state.pt" && -s "$CKPT/checkpoint_manifest.json" ]]; do
  echo "waiting for $CKPT"
  sleep 30
done

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2allchunksttemp RULER split 16k=${GPUS[0]},${GPUS[1]} 64k=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" s2allchunksttemp &
  p16=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k.sh" s2allchunksttemp &
  p64=$!
  fail=0
  wait "$p16" || fail=1
  wait "$p64" || fail=1
  (( fail == 0 )) || { echo "s2allchunksttemp RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  bash "$HERE/start_jingneng_ruler_16k.sh" s2allchunksttemp
  bash "$HERE/start_jingneng_ruler_64k.sh" s2allchunksttemp
fi
echo "s2allchunksttemp_RULER_16K64K_DONE"
