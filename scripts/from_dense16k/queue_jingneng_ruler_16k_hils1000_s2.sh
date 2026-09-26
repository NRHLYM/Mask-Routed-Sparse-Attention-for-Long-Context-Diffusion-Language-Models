#!/usr/bin/env bash
# Machine 1: 16k goldspan RULER for HiLS-1000 and S2-1000, 2 GPUs each.
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
export RULER_LENGTHS="${RULER_LENGTHS:-16384}"
export RULER_TASKS="${RULER_TASKS:-hils_sn hils_mkmq hils_vt}"
export RULER_OUTPUT_SUBDIR="${RULER_OUTPUT_SUBDIR:-ruler_probes_goldspan}"
LAUNCH="$NSA_ROOT/scripts/from_dense16k/start_jingneng_ruler_16k.sh"
CUDA_VISIBLE_DEVICES="${HILS_GPUS:-0,1}" bash "$LAUNCH" hils1000 &
p1=$!
CUDA_VISIBLE_DEVICES="${S2_GPUS:-2,3}" bash "$LAUNCH" s2 &
p2=$!
fail=0
wait "$p1" || fail=1
wait "$p2" || fail=1
exit "$fail"
