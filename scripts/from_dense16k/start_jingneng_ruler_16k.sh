#!/usr/bin/env bash
# 16k RULER probes (train length): hils_sn / hils_mkmq / hils_vt, 100 each.
# YaRN factor = 16384/2048 = 8. Fast-dLLM mask span = gold answer tokens
# (no pad to 32). Writes ruler_probes_goldspan, leaves the 32-slot table alone.
#
# GPU job terminal:
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh hils
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2sync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2b5
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2qcal
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2vf
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2fromhope
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2all28
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2from2k
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2from2ks1500
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2alltasks
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2allchunk
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh s2allchunksttemp
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh nsa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh nsasync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh swa
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh dsa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh dense
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh densehope
# Dense + Fast-dLLM KV cache (full-window FA, separate output dir):
#   RULER_OUTPUT_SUBDIR=ruler_probes_goldspan_facache \
#     CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_16k.sh dense
set -euo pipefail
export RULER_LENGTHS="${RULER_LENGTHS:-16384}"
export RULER_TASKS="${RULER_TASKS:-hils_sn hils_mkmq hils_vt}"
export RULER_OUTPUT_SUBDIR="${RULER_OUTPUT_SUBDIR:-ruler_probes_goldspan}"
exec bash "$(dirname "$0")/start_jingneng_ruler_2gpu.sh" "$@"
