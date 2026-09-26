#!/usr/bin/env bash
# Jingneng: s2-qcal RMSNorm s1000 goldspan RULER at 128k (2-GPU layer-parallel).
# Delegates to the 4-GPU pipeline. Prefer the combined nsa,swa,s2 launcher
# so NSA/SWA leftover pair time is not idle.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_128k_s2qcal.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/start_jingneng_ruler_128k_nsa_swa_s2_pipeline.sh" s2
