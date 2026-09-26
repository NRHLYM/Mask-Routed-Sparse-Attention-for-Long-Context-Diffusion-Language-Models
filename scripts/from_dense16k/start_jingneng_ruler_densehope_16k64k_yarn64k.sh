#!/usr/bin/env bash
# Official goldspan RULER:
#   dense HoPE 16k + 64k (keep HoPE, do not convert to YaRN)
#   mix=0.05 YaRN dense 64k with training factor=8 (not 32)
#
# GPU job (4 cards; do not overlap allchunk-sttemp):
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_densehope_16k64k_yarn64k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} < 4 )); then
  echo "need 4 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 1
fi
echo "densehope 16k+64k on ${GPUS[0]},${GPUS[1]}  yarn-dense 64k factor=8 on ${GPUS[2]},${GPUS[3]}"
(
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" densehope
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_64k.sh" densehope
) &
p0=$!
CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k_yarn8.sh" dense &
p1=$!
fail=0
wait "$p0" || fail=1
wait "$p1" || fail=1
(( fail == 0 )) || { echo "dense/HoPE RULER worker failed" >&2; exit 1; }
echo "DENSEHOPE_16K64K_YARN_64K_FACTOR8_DONE"
