#!/usr/bin/env bash
# 4-GPU server B: S2 (live fusion, no teacher) then S4 (all-chunk ST).
set -euo pipefail
source /Data/xiongjing/env.sh
export NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LAUNCH="$NSA_ROOT/scripts/from_dense16k/start_jingneng_hils_route_ablation.sh"
SETTING=s2 bash "$LAUNCH"
SETTING=s4 bash "$LAUNCH"
echo "server B queue done: s2 then s4"
