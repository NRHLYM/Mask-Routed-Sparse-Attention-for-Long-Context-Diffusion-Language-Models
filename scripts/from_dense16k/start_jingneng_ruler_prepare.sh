#!/usr/bin/env bash
# CPU-ok: NVIDIA RULER packer + probe YAML (S-N / MK-MQ / VT) at 16k/32k/64k.
# Noise haystack; matches training task 0/1/2. No GPU.
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export NLTK_DATA="${NLTK_DATA:-/Data/xiongjing/nltk_data}"
mkdir -p "$NLTK_DATA"
PYTHON="${PYTHON:-python}"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
MODEL="${MODEL_PATH:-/Data/xiongjing/models/Dream-v0-Base-7B}"
RULER_DIR="${RULER_DIR:-/Data/xiongjing/src/RULER}"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes}"
LOG="$ROOT/logs/ruler-probes-prepare-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/logs" "$DATA_DIR"
{
  echo "===== RULER probes prepare (hils_sn / hils_mkmq / hils_vt) ====="
  echo "model=$MODEL ruler=$RULER_DIR data=$DATA_DIR"
  "$PYTHON" "$EVAL" \
    --mode prepare \
    --ruler_dir "$RULER_DIR" \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --num_samples "${NUM_SAMPLES:-100}" \
    --lengths ${RULER_LENGTHS:-16384 32768 65536} \
    --tasks ${RULER_TASKS:-hils_sn hils_mkmq hils_vt}
  echo RULER_PREPARE_DONE
} 2>&1 | tee "$LOG"
