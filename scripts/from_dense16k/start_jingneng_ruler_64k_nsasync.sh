#!/usr/bin/env bash
# Official RULER 64k for NSA Dolma+RULER sync step-500 (not old fromdense `nsa`).
# YaRN factor = 65536/2048 = 32. Tasks: hils_sn / hils_mkmq / hils_vt, 100 each.
# Writes .../nsa-...-dolma-ruler-sync-s500/ruler_probes/len65536/
#
# GPU job terminal (do not start until 16k goldspan is DONE):
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_64k_nsasync.sh
set -euo pipefail
export RULER_LENGTHS=65536
export RULER_TASKS="hils_sn hils_mkmq hils_vt"
export INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsasync-64k}"
exec bash "$(dirname "$0")/start_jingneng_ruler_64k.sh" nsasync
