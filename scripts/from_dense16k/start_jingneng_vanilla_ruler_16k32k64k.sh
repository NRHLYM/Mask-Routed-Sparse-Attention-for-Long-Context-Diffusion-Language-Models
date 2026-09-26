#!/usr/bin/env bash
# Jingneng GPU: Vanilla Dream-v0-Base-7B goldspan RULER (no LoRA, no continued
# training). SN / MK-MQ / VT. Goldspan Fast-dLLM 32/0.9. 4 GPUs, 1 GPU/task.
# 16k Fixed == Unfixed (factor 8). Do not launch from the notebook.
# Do not overlap the live DSA Unfixed 128k job.
#
# Unfixed (YaRN = L/2048): 16k/32k/64k  -> fills 16k both rows + Unfixed 32/64
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_vanilla_ruler_16k32k64k.sh unfixed
# Fixed (keep-train factor=8): 32k/64k only
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_vanilla_ruler_16k32k64k.sh yarn8fixed
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODE="${1:?usage: $0 unfixed|yarn8fixed}"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
CONFIG="$NSA_ROOT/configs/from_dense16k/vanilla-dream-v0-base-7b-jingneng.json"
CHECKPOINT=none
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
PYTHON="${PYTHON:-python}"
TASKS=(hils_sn hils_mkmq hils_vt)
MODEL_DIR="${MODEL_DIR:-/Data/xiongjing/models/Dream-v0-Base-7B}"

case "$MODE" in
  unfixed)
    unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
    LENGTHS=(65536 32768 16384)
    OUTPUT_ROOT="/Data/xiongjing/outputs/vanilla-dream-v0-base-7b-ruler/ruler_probes_goldspan"
    NAME=vanilla_unfixed
    DONE_FLAG=VANILLA_RULER_16K32K64K_L2048_DONE
    ;;
  yarn8fixed)
    export RULER_KEEP_TRAIN_YARN=1
    export RULER_YARN_FACTOR=8
    LENGTHS=(65536 32768)
    OUTPUT_ROOT="/Data/xiongjing/outputs/vanilla-dream-v0-base-7b-ruler/ruler_probes_goldspan_yarn8fixed"
    NAME=vanilla_yarn8
    DONE_FLAG=VANILLA_RULER_32K64K_YARN8FIXED_DONE
    ;;
  *)
    echo "usage: $0 unfixed|yarn8fixed" >&2
    exit 1
    ;;
esac

cd "$FULLTEACHER_ROOT"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-vanilla-ruler-${MODE}}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/vanilla-ruler-${MODE}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" && -f "$CONFIG" ]] || { echo "missing eval/config" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "missing $MODEL_DIR" >&2; exit 1; }
grep -q "if int(getattr(args, \"lora_r\", 0) or 0) <= 0:" \
  "$FULLTEACHER_ROOT/dream_dllm_hils/train_fulltext.py" || {
  echo "FULLTEACHER apply_lora still rejects lora_r=0; scp train_fulltext.py" >&2
  exit 1
}

"$PYTHON" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
if cfg.get("attention_mode") != "dense":
    raise SystemExit("vanilla must be attention_mode=dense")
if int(cfg.get("lora_r", -1)) != 0:
    raise SystemExit("vanilla must set lora_r=0")
if cfg.get("initialize_from"):
    raise SystemExit("vanilla must not initialize_from a continued-training ckpt")
print("preflight_ok vanilla", "$MODE", "lora_r=0", "ckpt=none")
PY

for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    dest="$DATA_DIR/len${length}/${task}/validation.jsonl"
    mkdir -p "$(dirname "$dest")"
    if [[ ! -s "$dest" ]]; then
      python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
    fi
  done
done

job_done() { [[ -s "$OUTPUT_ROOT/len${1}/${2}/metrics.json" ]]; }

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

run_job() {
  local length="$1" task="$2"
  "$PYTHON" "$EVAL" \
    --mode eval \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length" \
    --rank 0 --world_size 1 --device cuda:0 \
    --block_length 32 --threshold 0.9
  "$PYTHON" "$EVAL" \
    --mode merge \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length"
}

pairs=()
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    if job_done "$length" "$task"; then
      echo "skip vanilla $MODE len${length} $task"
      continue
    fi
    if jsonl_ready "$length" "$task"; then
      echo "merge existing vanilla $MODE len${length} $task"
      "$PYTHON" "$EVAL" --mode merge --output_dir "$OUTPUT_ROOT" \
        --data_dir "$DATA_DIR" --task "$task" --max_seq_len "$length"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
      echo "reset incomplete vanilla $MODE len${length} $task"
      rm -rf "$OUTPUT_ROOT/len${length}/${task}"
    fi
    pairs+=("${length}:${task}")
  done
done

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "===== vanilla Dream-base goldspan RULER $MODE ====="
  echo "devices=${GPUS[*]} model=$MODEL_DIR output=$OUTPUT_ROOT pending=${#pairs[@]}"
  if (( ${#pairs[@]} > 0 )); then
    pids=()
    for i in "${!GPUS[@]}"; do
      gpu="${GPUS[$i]}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$FULLTEACHER_ROOT"
        if [[ "$MODE" == "yarn8fixed" ]]; then
          export RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8
        else
          unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
        fi
        cd "$FULLTEACHER_ROOT"
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
  "$PYTHON" "$EVAL" \
    --mode summary \
    --output_dir "$OUTPUT_ROOT" \
    --model_name "$NAME" \
    --lengths "${LENGTHS[@]}"
  echo "$DONE_FLAG"
} 2>&1 | tee "$LOG"
