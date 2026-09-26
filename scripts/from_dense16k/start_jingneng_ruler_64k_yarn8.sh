#!/usr/bin/env bash
# 64k official RULER, keep training YaRN factor=8 (do not scale to 32).
# Physical length 65536; sparse windows unchanged. Independent of factor-32 dirs.
#
# GPU job terminal:
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k_yarn8.sh s2sync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k_yarn8.sh s2all28
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/from_dense16k/start_jingneng_ruler_64k_yarn8.sh nsasync
set -euo pipefail
export RULER_KEEP_TRAIN_YARN=1
export RULER_YARN_FACTOR="${RULER_YARN_FACTOR:-8}"
export RULER_LENGTHS="${RULER_LENGTHS:-65536}"
export RULER_TASKS="${RULER_TASKS:-hils_sn hils_mkmq hils_vt}"
export RULER_OUTPUT_SUBDIR="${RULER_OUTPUT_SUBDIR:-ruler_probes_yarn8}"
export INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-${1:-ruler}-64k-yarn8}"
exec bash "$(dirname "$0")/start_jingneng_ruler_2gpu.sh" "$@"
