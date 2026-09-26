#!/usr/bin/env bash
# English-only LongBench-v1 (16 tasks) on ci-h20.
# Sequential: 2 GPUs on DSA, then 2 GPUs on hybrid (swa3-dense1).
set -euo pipefail
ROOT="${ROOT:-/home/sgli/work/hils-lb-ci-20260926}"
PYTHON="${PYTHON:-/home/sgli/work/code-smell-detection-runtime/venv/bin/python}"
MODEL="${MODEL:-/home/sgli/work/models/Dream-v0-Base-7B}"
DATA_ROOT="${DATA_ROOT:-/home/sgli/work/LongBench/data}"
PROMPTS="${PROMPTS:-/home/sgli/work/prefilling-dllm-reference/longbench_config/dataset2prompt_raw.json}"
PHYS=16384
NGPU=2
cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True

declare -A ANSWER_TOKENS=(
  [narrativeqa]=128 [qasper]=128 [multifieldqa_en]=64
  [hotpotqa]=32 [2wikimqa]=32 [musique]=32
  [gov_report]=512 [qmsum]=512 [multi_news]=512
  [trec]=64 [triviaqa]=32 [samsum]=128
  [passage_count]=32 [passage_retrieval_en]=32
  [lcc]=64 [repobench-p]=64
)
EN_TASKS=(
  multifieldqa_en gov_report narrativeqa qasper musique qmsum samsum
  hotpotqa triviaqa multi_news repobench-p trec passage_retrieval_en passage_count
  lcc 2wikimqa
)

arm_paths() {
  local ARM="$1"
  case "$ARM" in
    swa3dense1)
      TREE="$ROOT/eval-trees/swa3dense1"
      EVAL_SCRIPT="$TREE/scripts/dream_dllm_hils/eval_longbench_fastdllm_dense_ckpt.py"
      CONFIG="$ROOT/configs/swa3-dense1-ci.json"
      CHECKPOINT="$ROOT/ckpts/swa3d1/step-1000"
      OUTPUT_ROOT="$ROOT/outputs/swa3-dense1/longbench_fastdllm_en16"
      extra_eval=(--local_window "$PHYS" --allow_empty_predictions)
      extra_merge=(--local_window "$PHYS" --allow_empty_predictions)
      ;;
    dsa)
      TREE="$ROOT/eval-trees/dsa"
      EVAL_SCRIPT="$TREE/scripts/dream_dllm_hils/eval_longbench_fastdllm_dsa.py"
      CONFIG="$ROOT/configs/dsa-official-ci.json"
      CHECKPOINT="$ROOT/ckpts/dsa/step-1000"
      OUTPUT_ROOT="$ROOT/outputs/dsa-official/longbench_fastdllm_en16"
      extra_eval=(--model_path "$MODEL" --cache_mode dual_block)
      extra_merge=(--model_path "$MODEL" --cache_mode dual_block)
      ;;
    *) echo "bad arm $ARM" >&2; exit 1 ;;
  esac
  export PYTHONPATH="$TREE:$ROOT"
}

run_one_task() {
  local ARM="$1" GPU="$2" task="$3"
  arm_paths "$ARM"
  local LOG="$ROOT/logs/${ARM}.gpu${GPU}.log"
  mkdir -p "$OUTPUT_ROOT/$task" "$ROOT/logs"
  echo "gpu=$GPU task=$task $(date -Is)" | tee -a "$LOG"
  export CUDA_VISIBLE_DEVICES="$GPU"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --rank 0 --world_size 1 --device cuda:0 --limit 0 \
    "${extra_eval[@]}" >>"$LOG" 2>&1
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --merge \
    "${extra_merge[@]}" >>"$LOG" 2>&1
}

run_arm_2gpu() {
  local ARM="$1"
  arm_paths "$ARM"
  local LOG="$ROOT/logs/${ARM}.log"
  mkdir -p "$OUTPUT_ROOT" "$ROOT/logs"
  echo "===== $ARM 2gpu start $(date -Is) =====" | tee -a "$LOG"
  remaining=()
  for task in "${EN_TASKS[@]}"; do
    [[ -s "$DATA_ROOT/$task.jsonl" ]] || { echo "missing $DATA_ROOT/$task.jsonl" >&2; exit 1; }
    if [[ -s "$OUTPUT_ROOT/$task/metrics.json" ]]; then
      echo "skip $task" | tee -a "$LOG"
    else
      remaining+=("$task")
    fi
  done
  echo "REMAINING ${#remaining[@]}: ${remaining[*]:-none}" | tee -a "$LOG"
  if (( ${#remaining[@]} == 0 )); then
    echo "===== $ARM already complete $(date -Is) =====" | tee -a "$LOG"
    return 0
  fi
  local gpu
  pids=()
  for gpu in $(seq 0 $((NGPU - 1))); do
    (
      for i in "${!remaining[@]}"; do
        if (( i % NGPU == gpu )); then
          run_one_task "$ARM" "$gpu" "${remaining[$i]}"
        fi
      done
    ) &
    pids+=($!)
  done
  local p rc=0
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  echo "===== $ARM 2gpu done $(date -Is) rc=$rc =====" | tee -a "$LOG"
  return 0
}

MODE="${1:-seq}"
case "$MODE" in
  dsa) run_arm_2gpu dsa ;;
  swa3dense1|hybrid) run_arm_2gpu swa3dense1 ;;
  seq|dsa-then-hybrid)
    run_arm_2gpu dsa
    run_arm_2gpu swa3dense1
    ;;
  gpu1-dsa)
    arm_paths dsa
    for task in gov_report qasper qmsum hotpotqa multi_news trec passage_count 2wikimqa; do
      if [[ -s "$OUTPUT_ROOT/$task/metrics.json" ]]; then
        echo "skip $task"
        continue
      fi
      run_one_task dsa 1 "$task"
    done
    ;;
  *)
    echo "usage: $0 seq|dsa|hybrid|gpu1-dsa" >&2
    exit 1
    ;;
esac
