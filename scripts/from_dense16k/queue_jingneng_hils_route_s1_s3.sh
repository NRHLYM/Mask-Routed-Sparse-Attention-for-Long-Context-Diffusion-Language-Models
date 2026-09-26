#!/usr/bin/env bash
# 4-GPU server A: S1 (live fusion + teacher 0.01) then S3 (mean-V CE STE).
set -euo pipefail
source /Data/xiongjing/env.sh
export NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LAUNCH="$NSA_ROOT/scripts/from_dense16k/start_jingneng_hils_route_ablation.sh"
SETTING=s1 bash "$LAUNCH"
SETTING=s3 bash "$LAUNCH"
echo "server A queue done: s1 then s3"
