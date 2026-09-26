#!/usr/bin/env bash
# Jingneng 2-GPU Fast-dLLM LongBench-v1 all21 for NSA w128 + 21-layer SWA 1280.
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsa-lb}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$HILS_FT_ROOT"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#GPUS[@]}"
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_nsa.py
CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-yarn8-16k-b32-w128-c64s64-swa1280-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-swa1280-i4-fromdense/step-500"
OUTPUT_ROOT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-swa1280-i4-fromdense/longbench_fastdllm_all21"
PHYS=16384
LIMIT="${LIMIT:-0}"
LOG="$ROOT/logs/nsa-longbench-w128-swa1280-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$ROOT/logs"

[[ -f "$EVAL_SCRIPT" ]] || { echo "missing $EVAL_SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "missing prompts $PROMPTS" >&2; exit 1; }
python3 - <<PY
import json
from inspect import signature
from dream_dllm_hils.nsa_attention import install_dream_nsa_attention
cfg=json.loads(open("$CONFIG").read())
if cfg.get("attention_mode") != "nsa":
    raise SystemExit("NSA eval config must be attention_mode=nsa")
if int(cfg.get("local_window", 0)) != 128 or int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("NSA must keep sparse W=128 and swa_local_window=1280")
if "swa_local_window" not in signature(install_dream_nsa_attention).parameters:
    raise SystemExit("install_dream_nsa_attention missing swa_local_window")
print("preflight_ok nsa_w128_swa1280_eval")
PY
until [[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]]; do
  echo "waiting for $CHECKPOINT"
  sleep 30
done

declare -A ANSWER_TOKENS=(
  [narrativeqa]=128 [qasper]=128 [multifieldqa_en]=64 [multifieldqa_zh]=64
  [hotpotqa]=32 [2wikimqa]=32 [musique]=32 [dureader]=128
  [gov_report]=512 [qmsum]=512 [multi_news]=512 [vcsum]=512
  [trec]=64 [triviaqa]=32 [samsum]=128 [lsht]=64
  [passage_count]=32 [passage_retrieval_en]=32 [passage_retrieval_zh]=32
  [lcc]=64 [repobench-p]=64
)
ALL_TASKS=(
  multifieldqa_en gov_report narrativeqa qasper musique qmsum dureader samsum
  hotpotqa triviaqa multi_news repobench-p trec passage_retrieval_en passage_count
  vcsum lcc multifieldqa_zh lsht 2wikimqa passage_retrieval_zh
)

task_done() { [[ -s "$OUTPUT_ROOT/$1/metrics.json" ]]; }
reset_incomplete() {
  local output_dir="$OUTPUT_ROOT/$1"
  if [[ -d "$output_dir" ]] && ! task_done "$1"; then
    echo "reset incomplete $1"
    rm -rf "$output_dir"
  fi
}

run_eval() {
  local task="$1"
  mkdir -p "$OUTPUT_ROOT/$task"
  local limit_args=()
  [[ "$LIMIT" != "0" ]] && limit_args=(--limit "$LIMIT")
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --cache_mode dual_block --rank 0 --world_size 1 --device cuda:0 "${limit_args[@]}"
}
merge_task() {
  local task="$1"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --cache_mode dual_block --merge
}

remaining=()
for task in "${ALL_TASKS[@]}"; do
  [[ -s "$DATA_ROOT/$task.jsonl" ]] || { echo "missing $DATA_ROOT/$task.jsonl" >&2; exit 1; }
  if task_done "$task"; then echo "skip $task"
  else reset_incomplete "$task"; remaining+=("$task"); fi
done

{
  echo "===== nsa w128 swa1280 longbench all21 ${NGPU}-gpu ====="
  echo "devices=$CUDA_VISIBLE_DEVICES ckpt=$CHECKPOINT code_root=$HILS_FT_ROOT"
  echo "REMAINING ${#remaining[@]}: ${remaining[*]:-none}"
  if (( ${#remaining[@]} > 0 )); then
    pids=()
    for gi in "${!GPUS[@]}"; do
      gpu="${GPUS[$gi]}"
      tasks=()
      for i in "${!remaining[@]}"; do
        if (( i % NGPU == gi )); then tasks+=("${remaining[$i]}"); fi
      done
      (( ${#tasks[@]} == 0 )) && continue
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        for task in "${tasks[@]}"; do
          echo "gpu=$gpu task=$task"
          run_eval "$task"
          merge_task "$task"
        done
      ) > "$OUTPUT_ROOT/worker-$gi.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if (( failed != 0 )); then
      echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
      exit 1
    fi
  fi
  if [[ "$LIMIT" == "0" ]]; then
    if [[ -f "$HILS_FT_ROOT/scripts/summarize_longbench_suite.py" ]]; then
      "$PYTHON" "$HILS_FT_ROOT/scripts/summarize_longbench_suite.py" \
        --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
    else
      "$PYTHON" "$FULLTEACHER_ROOT/scripts/summarize_longbench_suite.py" \
        --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
    fi
  fi
  echo NSA_W128_SWA1280_LONGBENCH_DONE
} 2>&1 | tee "$LOG"
