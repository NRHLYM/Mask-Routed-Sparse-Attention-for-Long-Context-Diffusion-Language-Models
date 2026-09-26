#!/usr/bin/env bash
#SBATCH --job-name=dream-y8-lb
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
cd "$ROOT"

EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_dense.py
FASTDLLM=/home/guests/zhen/jing/d3LLM/baseline/Fast_dLLM_v1/dream
OUTPUT_ROOT=outputs/dream-v0-base-7b-untuned-yarn8-16k/longbench_fastdllm_all21
PHYS=16384

test -d "$MODEL"
test -f "$FASTDLLM/model/generation_utils_block.py"
test -f "$EVAL_SCRIPT"
mkdir -p "$OUTPUT_ROOT"

declare -A ANSWER_TOKENS=(
  [narrativeqa]=128 [qasper]=128 [multifieldqa_en]=64 [multifieldqa_zh]=64
  [hotpotqa]=32 [2wikimqa]=32 [musique]=32 [dureader]=128
  [gov_report]=512 [qmsum]=512 [multi_news]=512 [vcsum]=512
  [trec]=64 [triviaqa]=32 [samsum]=128 [lsht]=64
  [passage_count]=32 [passage_retrieval_en]=32 [passage_retrieval_zh]=32
  [lcc]=64 [repobench-p]=64
)

task_done() {
  [[ -s "$OUTPUT_ROOT/$1/metrics.json" ]]
}

reset_incomplete() {
  local task="$1"
  local output_dir="$OUTPUT_ROOT/$task"
  if [[ -d "$output_dir" ]] && ! task_done "$task"; then
    echo "reset incomplete $task"
    rm -rf "$output_dir"
  fi
}

run_eval() {
  local task="$1" rank="$2" world="$3"
  local output_dir="$OUTPUT_ROOT/$task"
  mkdir -p "$output_dir"
  "$PYTHON" "$EVAL_SCRIPT" \
    --model_path "$MODEL" --fastdllm_dream_dir "$FASTDLLM" \
    --task "$task" --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$output_dir" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --rank "$rank" --world_size "$world" --device cuda:0 --limit 0 \
    --allow_empty_predictions
}

merge_task() {
  local task="$1"
  local output_dir="$OUTPUT_ROOT/$task"
  "$PYTHON" "$EVAL_SCRIPT" \
    --model_path "$MODEL" --fastdllm_dream_dir "$FASTDLLM" \
    --task "$task" --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$output_dir" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --merge --allow_empty_predictions
}

ALL_TASKS=(
  multifieldqa_en gov_report narrativeqa qasper musique qmsum dureader samsum
  hotpotqa triviaqa multi_news repobench-p trec passage_retrieval_en passage_count
  vcsum lcc multifieldqa_zh lsht 2wikimqa passage_retrieval_zh
)

remaining=()
for task in "${ALL_TASKS[@]}"; do
  if task_done "$task"; then
    echo "skip $task"
  else
    reset_incomplete "$task"
    remaining+=("$task")
  fi
done
echo "REMAINING ${#remaining[@]}: ${remaining[*]:-none}"

if (( ${#remaining[@]} == 0 )); then
  "$PYTHON" scripts/summarize_longbench_suite.py --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
  echo DREAM_YARN8_16K_LONGBENCH_DONE
  exit 0
fi

run_group() {
  local gpu="$1"
  shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  for task in "$@"; do
    echo "gpu=$gpu task=$task"
    run_eval "$task" 0 1
    merge_task "$task"
  done
}

pids=()
for gpu in 0 1; do
  tasks=()
  for i in "${!remaining[@]}"; do
    if (( i % 2 == gpu )); then
      tasks+=("${remaining[$i]}")
    fi
  done
  if (( ${#tasks[@]} == 0 )); then
    continue
  fi
  (
    run_group "$gpu" "${tasks[@]}"
  ) > "$OUTPUT_ROOT/worker-$gpu.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))
"$PYTHON" scripts/summarize_longbench_suite.py --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
echo DREAM_YARN8_16K_LONGBENCH_DONE
