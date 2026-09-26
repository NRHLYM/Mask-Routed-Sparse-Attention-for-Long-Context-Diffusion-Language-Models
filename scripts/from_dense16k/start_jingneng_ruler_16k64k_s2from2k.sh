#!/usr/bin/env bash
# Official RULER 16k goldspan then 64k for HiLS sync trained from Dream 2k.
# GPU job terminal:
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2from2k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
bash "$HERE/start_jingneng_ruler_16k.sh" s2from2k
bash "$HERE/start_jingneng_ruler_64k.sh" s2from2k
