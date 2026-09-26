#!/usr/bin/env bash
# 1 GPU. Train-distribution dense-teacher audit on s2-sync.
# Dolma 16k packs + every-step RULER view (same collator as training).
# Snapshot dense-500 Q/K, then load s2 LoRA. Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0 bash scripts/from_dense16k/start_jingneng_teacher_traindist_audit.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-teacher-traindist}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-500"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
PROBE="$NSA_ROOT/scripts/from_dense16k/probe_jingneng_teacher_traindist.py"
OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/teacher_traindist_audit.json"
LOG="$ROOT/logs/teacher-traindist-audit-$(date +%Y%m%d-%H%M%S).log"
LIMIT="${LIMIT:-16}"

[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" ]] || { echo "missing s2 ckpt $CHECKPOINT" >&2; exit 1; }
[[ -s "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt $DENSE" >&2; exit 1; }
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete packs: $PACK/$f" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/logs"

{
  echo "===== teacher traindist audit devices=$CUDA_VISIBLE_DEVICES n=$LIMIT ====="
  echo "student=$CHECKPOINT dense=$DENSE packs=$PACK"
  python "$PROBE" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --num-samples "$LIMIT" \
    --device cuda:0
} 2>&1 | tee "$LOG"
echo "log=$LOG out=$OUTPUT"
