#!/usr/bin/env bash
# Machine 1: answer-slot local_weight vs remote-only needle mass.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/from_dense16k/start_jingneng_fusion_gate.sh hils1000
#   STEP=600 CUDA_VISIBLE_DEVICES=0 bash scripts/from_dense16k/start_jingneng_fusion_gate.sh supportkl
#   STEP=950 CUDA_VISIBLE_DEVICES=0 bash scripts/from_dense16k/start_jingneng_fusion_gate.sh wdetach
set -euo pipefail
source /Data/xiongjing/env.sh
MODEL="${1:?hils1000|s2|hils500|supportkl|wdetach|niahjoint|s2s2000|s2ruler1|s2sync|s2vf}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${VALUE_FUSION_CODE_ROOT:-${PYTHONPATH:-$FULLTEACHER_ROOT}}"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-32}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-fusion-gate-${MODEL}}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$PYTHONPATH"

case "$MODEL" in
  hils1000)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/step-1000"
    OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/fusion_gate_sn16k.json"
    ;;
  s2)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-noteacher-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/step-1000"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/fusion_gate_sn16k.json"
    ;;
  hils500)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/step-500"
    OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/fusion_gate_sn16k.json"
    ;;
  supportkl)
    STEP="${STEP:-600}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-support-attn-kl-lora-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-support-attn-kl-lora-s1000/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-support-attn-kl-lora-s1000/fusion_gate_sn16k_step${STEP}.json"
    ;;
  wdetach)
    STEP="${STEP:-1000}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-support-attn-kl-wdetach-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-support-attn-kl-wdetach-s1000/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-support-attn-kl-wdetach-s1000/fusion_gate_sn16k_step${STEP}.json"
    ;;
  niahjoint)
    STEP="${STEP:-250}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-niah-joint-ce-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-niah-joint-ce-s1000/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-niah-joint-ce-s1000/fusion_gate_sn16k_step${STEP}.json"
    ;;
  s2s2000)
    STEP="${STEP:-2000}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-noteacher-s2000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s2000/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s2000/fusion_gate_sn16k_step${STEP}.json"
    ;;
  s2ruler1)
    STEP="${STEP:-500}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-rulermix1-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-rulermix1-s500/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-rulermix1-s500/fusion_gate_sn16k_step${STEP}.json"
    ;;
  s2sync)
    STEP="${STEP:-500}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/fusion_wremote_sn16k_step${STEP}.json"
    ;;
  s2vf)
    STEP="${STEP:-250}"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-value-fusion-beta0p3-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/step-${STEP}"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/fusion_wremote_sn16k_step${STEP}.json"
    ;;
  *)
    echo "unknown MODEL=$MODEL (hils1000|s2|hils500|supportkl|wdetach|niahjoint|s2s2000|s2ruler1|s2sync|s2vf)" >&2
    exit 1
    ;;
esac

PROBE="$NSA_ROOT/scripts/from_dense16k/probe_jingneng_fusion_gate.py"
LOG="$ROOT/logs/${MODEL}-fusion-gate-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/logs"
[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" ]] || { echo "missing ckpt $CHECKPOINT" >&2; exit 1; }

{
  echo "===== $MODEL fusion-gate SN 16k devices=$CUDA_VISIBLE_DEVICES limit=$LIMIT ====="
  echo "ckpt=$CHECKPOINT"
  python "$PROBE" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --max_seq_len 16384 \
    --limit "$LIMIT" \
    --device cuda:0
} 2>&1 | tee "$LOG"
