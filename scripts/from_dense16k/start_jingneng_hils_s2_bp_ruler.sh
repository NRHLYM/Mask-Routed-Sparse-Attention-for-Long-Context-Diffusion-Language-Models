#!/usr/bin/env bash
# Jingneng GPU: official goldspan RULER for HiLS S2 B′ step-1000.
# SN / MK-MQ / VT at 16k/32k/64k. YaRN factor = L/2048 (8/16/32).
# Goldspan Fast-dLLM 32/0.9. PYTHONPATH=$FULLTEACHER_ROOT.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_bp_ruler.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
cd "$FULLTEACHER_ROOT"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-bp-qcal-lr0p1-s1000-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/hils-s2-ablate-bp-qcal-lr0p1-s1000/step-1000"
OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-ablate-bp-qcal-lr0p1-s1000/ruler_probes_goldspan"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
TASKS=(hils_sn hils_mkmq hils_vt)
LENGTHS=(65536 32768 16384)
SUMMARY_LENGTHS=(16384 32768 65536)
NAME=s2bp1000

INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-bp-ruler}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/hils-s2-bp-ruler-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
    raise SystemExit("B' must stay lora_qcal_lmk")
if str(cfg.get("lmk_token_mode", "")) != "mask_type":
    raise SystemExit("B' must keep mask_type")
if bool(cfg.get("hils_freeze_qcal", False)):
    raise SystemExit("B' trains Q-Cal")
print("preflight_ok s2-bp-ruler")
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
      echo "skip bp len${length} $task"
      continue
    fi
    if jsonl_ready "$length" "$task"; then
      echo "merge existing bp len${length} $task"
      python "$EVAL" \
        --mode merge \
        --output_dir "$OUTPUT_ROOT" \
        --data_dir "$DATA_DIR" \
        --task "$task" \
        --max_seq_len "$length"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
      echo "reset incomplete bp len${length} $task"
      rm -rf "$OUTPUT_ROOT/len${length}/${task}"
    fi
    pairs+=("${length}:${task}")
  done
done

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "===== s2-bp official RULER goldspan 16k/32k/64k ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT pending=${#pairs[@]}"
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
  echo "s2_BP_S1000_RULER_16K32K64K_DONE"
} 2>&1 | tee "$LOG"
