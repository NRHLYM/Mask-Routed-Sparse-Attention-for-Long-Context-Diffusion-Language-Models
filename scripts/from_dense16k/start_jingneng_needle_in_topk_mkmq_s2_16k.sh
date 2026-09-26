#!/usr/bin/env bash
# 16k goldspan MK-MQ: how many of the two evidence chunks are in last-layer top-32.
# 1 GPU. Do not steal dense-hope 0,1 or allchunk-sttemp if those still occupy the job.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=2 bash scripts/from_dense16k/start_jingneng_needle_in_topk_mkmq_s2_16k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
export TASK=mkmq
export LIMIT="${LIMIT:-100}"
export MAX_SEQ_LEN=16384
bash "$(cd "$(dirname "$0")" && pwd)/start_jingneng_needle_in_topk.sh" s2sync
echo "NEEDLE_IN_TOPK_MKMQ_S2_16K_DONE"
