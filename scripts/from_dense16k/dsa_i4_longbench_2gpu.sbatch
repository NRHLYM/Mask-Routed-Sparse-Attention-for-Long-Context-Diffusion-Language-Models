#!/usr/bin/env bash
#SBATCH --job-name=dsa-i4-lb
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

EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_dsa.py
CONFIG=configs/from_dense16k/dsa-yarn8-16k-topk2560.json
CHECKPOINT=outputs/dsa-yarn8-16k-topk2560-i4-fromdense/step-500
OUTPUT_ROOT=outputs/dsa-yarn8-16k-topk2560-i4-fromdense/longbench_fastdllm_all21_v2
PHYS=16384

test -s "$CHECKPOINT/checkpoint_manifest.json"
test -s "$CHECKPOINT/trainable_state.pt"
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

run_eval() {
  local task="$1" rank="$2" world="$3"
  local output_dir="$OUTPUT_ROOT/$task"
  mkdir -p "$output_dir"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --model_path "$MODEL" --checkpoint "$CHECKPOINT" \
    --task "$task" --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$output_dir" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --cache_mode dual_block --rank "$rank" --world_size "$world" --device cuda:0 \
    --limit 0
}

merge_task() {
  local task="$1"
  local output_dir="$OUTPUT_ROOT/$task"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --model_path "$MODEL" --checkpoint "$CHECKPOINT" \
    --task "$task" --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$output_dir" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --cache_mode dual_block --merge
}

echo "===== first task: multifieldqa_en (2-way) ====="
mfen_pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_eval multifieldqa_en "$gpu" 2
  ) > "$OUTPUT_ROOT/mfen-rank-$gpu.log" 2>&1 &
  mfen_pids+=("$!")
done
failed=0
for pid in "${mfen_pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))
merge_task multifieldqa_en > "$OUTPUT_ROOT/multifieldqa_en/merge.log" 2>&1
echo "MFEN_DONE $(python3 -c "import json; print(json.load(open('$OUTPUT_ROOT/multifieldqa_en/metrics.json'))['score_avg'])")"

TASK_GROUPS=(
  "gov_report narrativeqa qasper musique qmsum dureader samsum hotpotqa triviaqa multi_news"
  "repobench-p trec passage_retrieval_en passage_count vcsum lcc multifieldqa_zh lsht 2wikimqa passage_retrieval_zh"
)

pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for task in ${TASK_GROUPS[$gpu]}; do
      echo "gpu=$gpu task=$task"
      run_eval "$task" 0 1
      merge_task "$task"
    done
  ) > "$OUTPUT_ROOT/worker-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))
"$PYTHON" scripts/summarize_longbench_suite.py --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
echo DSA_I4_LONGBENCH_DONE
