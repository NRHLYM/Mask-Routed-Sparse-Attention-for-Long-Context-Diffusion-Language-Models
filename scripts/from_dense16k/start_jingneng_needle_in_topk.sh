#!/usr/bin/env bash
# Machine 1: 16k goldspan S-N needle-in-topk at quiz tokens. 1 GPU.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/from_dense16k/start_jingneng_needle_in_topk.sh hils1000
#   CUDA_VISIBLE_DEVICES=1 bash scripts/from_dense16k/start_jingneng_needle_in_topk.sh s2
set -euo pipefail
source /Data/xiongjing/env.sh
MODEL="${1:?hils1000|s2|hils500|s2sync|s2allchunk|s2allchunksttemp}"
TASK="${TASK:-sn}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-needle-topk-${MODEL}-${TASK}-${MAX_SEQ_LEN}}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"

case "$MODEL" in
  hils1000)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/step-1000"
    OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/needle_in_topk_sn16k.json"
    ;;
  s2)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-noteacher-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/step-1000"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/needle_in_topk_sn16k.json"
    ;;
  hils500)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/step-500"
    OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/needle_in_topk_sn${MAX_SEQ_LEN}.json"
    ;;
  s2sync)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-500"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/${PROBE_TAG:-needle_fusion}_${TASK}${MAX_SEQ_LEN}.json"
    ;;
  s2allchunk)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500/step-500"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500/${PROBE_TAG:-needle_fusion}_${TASK}${MAX_SEQ_LEN}.json"
    ;;
  s2allchunksttemp)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500/step-500"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500/${PROBE_TAG:-needle_fusion}_${TASK}${MAX_SEQ_LEN}.json"
    ;;
  *)
    echo "unknown MODEL=$MODEL (hils1000|s2|hils500|s2sync|s2allchunk|s2allchunksttemp)" >&2
    exit 1
    ;;
esac

PROBE="$NSA_ROOT/scripts/from_dense16k/probe_jingneng_needle_in_topk.py"
LOG="$ROOT/logs/${MODEL}-${TASK}-needle-in-topk-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/logs"
[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" ]] || { echo "missing ckpt $CHECKPOINT" >&2; exit 1; }

{
  echo "===== $MODEL needle-in-topk ${TASK} ${MAX_SEQ_LEN} devices=$CUDA_VISIBLE_DEVICES limit=$LIMIT ====="
  echo "ckpt=$CHECKPOINT out=$OUTPUT"
  python "$PROBE" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --task "$TASK" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --limit "$LIMIT" \
    --device cuda:0
} 2>&1 | tee "$LOG"
