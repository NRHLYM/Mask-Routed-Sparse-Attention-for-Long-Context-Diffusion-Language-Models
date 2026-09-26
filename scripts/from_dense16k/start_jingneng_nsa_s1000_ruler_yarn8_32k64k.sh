#!/usr/bin/env bash
# Jingneng GPU: NSA 1:1 s1000 goldspan RULER at 32k/64k with YaRN fixed at 8.
# 16k is the same as L/2048 (already done). Do not write into the L/2048 dir.
# Goldspan Fast-dLLM 32/0.9. Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_nsa_s1000_ruler_yarn8_32k64k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RULER_KEEP_TRAIN_YARN=1
export RULER_YARN_FACTOR=8

stage_nsa_eval_tree() {
  local tree="/Data/xiongjing/src/eval-trees/nsasync1000-ruler-yarn8"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local overlay="$NSA_ROOT/dream_dllm_hils"
  local name
  rm -rf "$tree"
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  for name in nsa_attention.py train_fulltext.py attention.py \
      fastdllm_v1.py fastdllm_attention.py fastdllm_cache.py \
      dsa_attention.py local_attention.py; do
    [[ -f "$overlay/$name" ]] || continue
    ln -sfn "$overlay/$name" "$tree/dream_dllm_hils/$name"
  done
  echo "$tree"
}

CODE_ROOT="$(stage_nsa_eval_tree)"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"

CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000/step-1000"
OUTPUT_ROOT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000/ruler_probes_goldspan_yarn8fixed"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
TASKS=(hils_sn hils_mkmq hils_vt)
LENGTHS=(65536 32768)
SUMMARY_LENGTHS=(32768 65536)
NAME=nsasync1000_yarn8

INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsa-s1000-ruler-yarn8}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/nsa-s1000-ruler-yarn8-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    dest="$DATA_DIR/len${length}/${task}/validation.jsonl"
    mkdir -p "$(dirname "$dest")"
    if [[ ! -s "$dest" ]]; then
      python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
    fi
  done
done

run_job() {
  local length="$1"
  local task="$2"
  python "$EVAL" \
    --mode eval \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length" \
    --rank 0 --world_size 1 --device cuda:0 \
    --block_length 32 --threshold 0.9
  python "$EVAL" \
    --mode merge \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length"
}

job_done() {
  [[ -s "$OUTPUT_ROOT/len${1}/${2}/metrics.json" ]]
}

jsonl_ready() {
  local f="$OUTPUT_ROOT/len${1}/${2}/rank-0.jsonl"
  [[ -s "$f" ]] || return 1
  python3 - "$f" <<'PY'
import json, sys
path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if line:
        rows.append(json.loads(line))
by = {int(r["index"]): r for r in rows}
ok = all(i in by and by[i].get("outputs") for i in range(100))
sys.exit(0 if ok else 1)
PY
}

pairs=()
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    if job_done "$length" "$task"; then
      echo "skip nsa len${length} $task"
      continue
    fi
    if jsonl_ready "$length" "$task"; then
      echo "merge existing nsa len${length} $task"
      python "$EVAL" \
        --mode merge \
        --output_dir "$OUTPUT_ROOT" \
        --data_dir "$DATA_DIR" \
        --task "$task" \
        --max_seq_len "$length"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
      echo "reset incomplete nsa len${length} $task"
      rm -rf "$OUTPUT_ROOT/len${length}/${task}"
    fi
    pairs+=("${length}:${task}")
  done
done

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "===== nsa official RULER goldspan 32k/64k yarn8fixed ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT keep_train_yarn=1 yarn_factor=8 pending=${#pairs[@]}"
  if (( ${#pairs[@]} == 0 )); then
    echo "all metrics present"
  else
    pids=()
    for i in "${!GPUS[@]}"; do
      gpu="${GPUS[$i]}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        for ((j=i; j<${#pairs[@]}; j+=${#GPUS[@]})); do
          IFS=':' read -r length task <<< "${pairs[$j]}"
          echo "gpu=$gpu job=len${length}:${task}"
          run_job "$length" "$task"
        done
      ) > "$OUTPUT_ROOT/worker-gpu${i}.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if (( fail != 0 )); then
      echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
      exit 1
    fi
  fi
  python "$EVAL" \
    --mode summary \
    --output_dir "$OUTPUT_ROOT" \
    --model_name "$NAME" \
    --lengths "${SUMMARY_LENGTHS[@]}"
  echo "nsa_S1000_RULER_32K64K_YARN8FIXED_DONE"
} 2>&1 | tee "$LOG"
