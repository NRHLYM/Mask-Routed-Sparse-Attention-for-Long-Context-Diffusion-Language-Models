#!/usr/bin/env bash
# Jingneng GPU: Fast-dLLM mf-en for the split run (LoRA<-CE, Q-Cal<-KL, LMK frozen).
# Default GPUs 2,3 so it can overlap STE training on 0,1.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
cd "$FULLTEACHER_ROOT"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${NPROC:-${#GPUS[@]}}"
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach.json"
CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach/step-500"
OUTPUT_ROOT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach/longbench_fastdllm_mfen"
PHYS=16384
LOG="$ROOT/logs/hils-mfen-klqcal-cedetach-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$ROOT/logs"

[[ -f "$EVAL_SCRIPT" ]] || { echo "missing $EVAL_SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$DATA_ROOT/multifieldqa_en.jsonl" ]] || { echo "missing mf-en jsonl" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "missing prompts $PROMPTS" >&2; exit 1; }
until [[ -s "$CHECKPOINT/trainable_state.pt" ]]; do
  echo "waiting for $CHECKPOINT"
  sleep 30
done

run_eval() {
  local rank="$1"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --task multifieldqa_en \
    --data "$DATA_ROOT/multifieldqa_en.jsonl" \
    --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/multifieldqa_en" \
    --physical_length "$PHYS" \
    --chunk_size 64 \
    --local_window 512 \
    --topk 32 \
    --answer_tokens 64 \
    --block_length 32 \
    --threshold 0.9 \
    --rank "$rank" --world_size "$NPROC" --device cuda:0 \
    --limit 0
}

{
  echo "===== multifieldqa_en ${NPROC}-way split-old klqcal-cedetach ====="
  echo "devices=$CUDA_VISIBLE_DEVICES ckpt=$CHECKPOINT"
  mkdir -p "$OUTPUT_ROOT/multifieldqa_en"
  pids=()
  rank=0
  for gpu in "${GPUS[@]}"; do
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      run_eval "$rank"
    ) > "$OUTPUT_ROOT/mfen-rank-$rank.log" 2>&1 &
    pids+=("$!")
    rank=$((rank + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed != 0 )); then
    echo "rank eval failed; see $OUTPUT_ROOT/mfen-rank-*.log" >&2
    exit 1
  fi
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --task multifieldqa_en \
    --data "$DATA_ROOT/multifieldqa_en.jsonl" \
    --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/multifieldqa_en" \
    --physical_length "$PHYS" \
    --chunk_size 64 \
    --local_window 512 \
    --topk 32 \
    --answer_tokens 64 \
    --block_length 32 \
    --threshold 0.9 \
    --merge
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$OUTPUT_ROOT/multifieldqa_en/metrics.json")
d = json.loads(p.read_text())
print("MFEN_DONE", d.get("score_avg", d.get("qa_f1")))
print(json.dumps({k: d.get(k) for k in ["task", "examples", "qa_f1", "score_avg", "metric"]}, indent=2))
PY
  echo HILS_SPLIT_OLD_MFEN_DONE
} 2>&1 | tee "$LOG"
