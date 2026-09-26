#!/usr/bin/env bash
# Jingneng GPU: mf-en answer-chunk recall for split-old (LoRA CE, Q-Cal KL, LMK frozen).
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
cd "$FULLTEACHER_ROOT"
PYTHON="${PYTHON:-python}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach.json"
CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach/step-500"
OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach/answer-chunks.json"
LOG="$ROOT/logs/hils-chunks-klqcal-cedetach-$(date +%Y%m%d-%H%M%S).log"
PROBE=scripts/probe_answer_chunks.py
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/logs"
[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" ]] || { echo "missing ckpt $CHECKPOINT" >&2; exit 1; }
[[ -s "$DATA_ROOT/multifieldqa_en.jsonl" ]] || { echo "missing mf-en" >&2; exit 1; }

{
  echo "===== answer-chunk probe split-old devices=$CUDA_VISIBLE_DEVICES ====="
  "$PYTHON" "$PROBE" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_ROOT/multifieldqa_en.jsonl" \
    --prompt_config "$PROMPTS" \
    --output "$OUTPUT" \
    --physical_length 16384 \
    --answer_tokens 64 \
    --device cuda:0
  echo HILS_SPLIT_OLD_CHUNKS_DONE
} 2>&1 | tee "$LOG"
