#!/usr/bin/env bash
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
LAUNCH="$NSA_ROOT/scripts/from_dense16k/start_jingneng_fusion_gate.sh"
CUDA_VISIBLE_DEVICES="${HILS_GPU:-0}" bash "$LAUNCH" hils1000 &
p1=$!
CUDA_VISIBLE_DEVICES="${S2_GPU:-1}" bash "$LAUNCH" s2 &
p2=$!
fail=0
wait "$p1" || fail=1
wait "$p2" || fail=1
exit "$fail"
