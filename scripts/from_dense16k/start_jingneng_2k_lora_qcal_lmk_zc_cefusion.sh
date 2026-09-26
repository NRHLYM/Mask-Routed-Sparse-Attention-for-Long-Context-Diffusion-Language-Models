#!/usr/bin/env bash
# Jingneng GPU job. Native 2k HiLS: LoRA + Q-Cal + mask_type LMK, Z_c, CE attached.
# Default 2 cards on 2,3 so 0,1 stay free for the 16k qcal_lmk job.
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
NPROC="${NPROC:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-2k-frozenbase-lora-qcal-lmk-mass.json"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-2k-ab64
OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-2k-lora-qcal-lmk-frozenbase-attnmass-cefusion"
LOG="$ROOT/logs/hils-2k-lora-qcal-lmk-zc-cefusion-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 2k packs: missing $PACK/$f" >&2; exit 1; }
done
python3 - <<PY
from pathlib import Path
p = Path("$PACK/train.bin")
# 64 packs * 2016 uint32 tokens
if p.stat().st_size < 516096:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 516096")
PY
start_args=()
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${resume_from}" && -f "${resume_from}/trainable_state.pt" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
