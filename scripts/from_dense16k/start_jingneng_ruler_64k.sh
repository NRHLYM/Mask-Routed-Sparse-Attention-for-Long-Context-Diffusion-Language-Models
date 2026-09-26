#!/usr/bin/env bash
# 64k RULER probes for one baseline: hils_sn / hils_mkmq / hils_vt, 100 each.
# YaRN factor = 65536/2048 = 32. Same 16k-trained adapters.
#
# GPU job terminal:
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh hils
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2sync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2b5
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2qcal
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2vf
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2all28
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2from2k
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2from2ks1500
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2alltasks
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2allchunk
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh s2allchunksttemp
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh nsa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh nsasync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh swa
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_ruler_64k.sh dsa
set -euo pipefail
export RULER_LENGTHS="${RULER_LENGTHS:-65536}"
export RULER_TASKS="${RULER_TASKS:-hils_sn hils_mkmq hils_vt}"
export INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-${1:-ruler}-64k}"
exec bash "$(dirname "$0")/start_jingneng_ruler_2gpu.sh" "$@"
