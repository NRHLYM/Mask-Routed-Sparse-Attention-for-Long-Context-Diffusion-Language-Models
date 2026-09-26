#!/usr/bin/env bash
# 64k goldspan S-N needle-in-topk for s2-sync vs all-chunk ST. 1 GPU each.
# Keep 0,1 for dense-hope; use 2,3.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_needle_in_topk_s2_allchunk_64k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LIMIT="${LIMIT:-100}"
export MAX_SEQ_LEN=65536
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-2,3}"
if (( ${#GPUS[@]} < 2 )); then
  echo "need 2 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 1
fi
echo "needle-in-topk 64k SN limit=$LIMIT  s2sync=${GPUS[0]} allchunk=${GPUS[1]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$HERE/start_jingneng_needle_in_topk.sh" s2sync &
p0=$!
CUDA_VISIBLE_DEVICES="${GPUS[1]}" bash "$HERE/start_jingneng_needle_in_topk.sh" s2allchunk &
p1=$!
fail=0
wait "$p0" || fail=1
wait "$p1" || fail=1
(( fail == 0 )) || { echo "needle-in-topk worker failed" >&2; exit 1; }
echo "NEEDLE_IN_TOPK_S2_ALLCHUNK_64K_DONE"
