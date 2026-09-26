#!/usr/bin/env bash
# Jingneng: NSA 1:1 s1000 goldspan RULER at 128k (2-GPU layer-parallel).
# Delegates to the 4-GPU NSA/SWA/s2 pipeline so a free pair starts the next
# task immediately (NSA VT can overlap SWA if you run the full pipeline).
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_nsa_s1000_ruler_128k.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/start_jingneng_ruler_128k_nsa_swa_s2_pipeline.sh" nsa
