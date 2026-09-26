#!/usr/bin/env bash
# Jingneng GPU: official goldspan RULER for dense 16k 1:1 s1000.
# SN / MK-MQ / VT at 16k/32k/64k/128k. YaRN = L/2048 (8/16/32/64), not
# keep-train factor=8. Goldspan Fast-dLLM 32/0.9.
# 16/32/64: one GPU per task (4-way). 128k: two GPUs per task, layer-parallel
# (same as NSA 128k), two pair workers on a 4-GPU job.
# Do not launch from the notebook. Do not set RULER_KEEP_TRAIN_YARN.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_dense_s1000_ruler_16k32k64k128k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true

HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="$HERE/run_official_ruler_task_2gpu.sh"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
CONFIG="$NSA_ROOT/configs/from_dense16k/dense-yarn8-16k-dolmaruler-sync-s1000-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/step-1000"
OUTPUT_ROOT="/Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/ruler_probes_goldspan"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
PYTHON="${PYTHON:-python}"
export EVAL CONFIG CHECKPOINT OUTPUT_ROOT DATA_DIR PYTHON
TASKS=(hils_sn hils_mkmq hils_vt)
SHORT_LENGTHS=(65536 32768 16384)
ALL_LENGTHS=(16384 32768 65536 131072)
NAME=densesync1000

cd "$FULLTEACHER_ROOT"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dense-s1000-ruler}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/dense-s1000-ruler-L2048-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" && -f "$PAIR" && -f "$CONFIG" ]] || {
  echo "missing eval/pair/config" >&2
  exit 1
}
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
if cfg.get("attention_mode") != "dense":
    raise SystemExit("dense 1:1 ruler needs attention_mode=dense")
if abs(float(cfg.get("ruler_mix_ratio", 1) or 0)) > 1e-12:
    raise SystemExit("this ckpt is 1:1 (ruler_mix_ratio=0)")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
if int(man.get("step", 0)) != 1000:
    raise SystemExit(f"need step-1000, manifest step={man.get('step')}")
print("preflight_ok dense-s1000-ruler L/2048 16/32/64/128k")
PY

for length in "${ALL_LENGTHS[@]}"; do
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

run_short() {
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

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPUS[@]} < 4 )); then
  echo "need 4 GPUs for 16/32/64 plus 128k pairs, got $CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

{
  echo "===== dense 1:1 s1000 goldspan RULER 16/32/64/128k YaRN=L/2048 ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT"
  echo "yarn=unfixed (16k=8, 32k=16, 64k=32, 128k=64); 128k layer_parallel=2"

  short_pairs=()
  for length in "${SHORT_LENGTHS[@]}"; do
    for task in "${TASKS[@]}"; do
      if job_done "$length" "$task"; then
        echo "skip dense len${length} $task"
        continue
      fi
      if jsonl_ready "$length" "$task"; then
        echo "merge existing dense len${length} $task"
        "$PYTHON" "$EVAL" --mode merge --output_dir "$OUTPUT_ROOT" \
          --data_dir "$DATA_DIR" --task "$task" --max_seq_len "$length"
        continue
      fi
      if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
        echo "reset incomplete dense len${length} $task"
        rm -rf "$OUTPUT_ROOT/len${length}/${task}"
      fi
      short_pairs+=("${length}:${task}")
    done
  done
  echo "16/32/64 pending=${#short_pairs[@]}"

  if (( ${#short_pairs[@]} > 0 )); then
    pids=()
    for i in "${!GPUS[@]}"; do
      gpu="${GPUS[$i]}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
        for ((j=i; j<${#short_pairs[@]}; j+=${#GPUS[@]})); do
          IFS=':' read -r length task <<< "${short_pairs[$j]}"
          echo "gpu=$gpu job=len${length}:${task}"
          run_short "$length" "$task"
        done
      ) > "$OUTPUT_ROOT/worker-short-gpu${i}.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if (( fail != 0 )); then
      echo "16/32/64 worker failed; see $OUTPUT_ROOT/worker-short-*.log" >&2
      exit 1
    fi
  fi

  echo "===== 128k two-GPU layer-parallel (NSA-style pairs) ====="
  WORKDIR="$(mktemp -d /tmp/xiongjing-dense128k-pipe.XXXXXX)"
  QUEUE="$WORKDIR/queue.txt"
  LOCK="$WORKDIR/queue.lock"
  FAILS="$WORKDIR/fails.txt"
  : > "$QUEUE"
  : > "$FAILS"
  for task in "${TASKS[@]}"; do
    if job_done 131072 "$task"; then
      echo "skip dense len131072 $task"
      continue
    fi
    printf '%s\n' "$task" >> "$QUEUE"
  done
  echo "128k queue ($(wc -l < "$QUEUE" | tr -d ' ')) tasks"
  cat "$QUEUE"

  pop_job() {
    python3 - "$QUEUE" "$LOCK" <<'PY'
import fcntl, pathlib, sys
queue, lock_path = map(pathlib.Path, sys.argv[1:3])
lock_path.touch()
with lock_path.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    lines = queue.read_text().splitlines() if queue.exists() else []
    if not lines:
        sys.exit(2)
    print(lines[0])
    rest = "\n".join(lines[1:])
    queue.write_text(rest + ("\n" if rest else ""))
PY
  }

  pair_worker() {
    local slot="$1" g0="$2" g1="$3"
    local task inductor
    while true; do
      if ! task="$(pop_job)"; then
        echo "slot=$slot idle, 128k queue empty"
        break
      fi
      inductor="/tmp/xiongjing-inductor-dense128k-${slot}-${task}"
      mkdir -p "$inductor/triton" "$inductor/torchinductor" "$inductor/torch_extensions"
      echo "START 128k slot=$slot gpus=$g0,$g1 $task"
      if (
        export PYTHONPATH="$FULLTEACHER_ROOT"
        export CONFIG CHECKPOINT OUTPUT_ROOT EVAL DATA_DIR PYTHON
        export TMPDIR="$inductor"
        export TRITON_CACHE_DIR="$inductor/triton"
        export TORCHINDUCTOR_CACHE_DIR="$inductor/torchinductor"
        export TORCH_EXTENSIONS_DIR="$inductor/torch_extensions"
        unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
        cd "$FULLTEACHER_ROOT"
        CUDA_VISIBLE_DEVICES="$g0,$g1" bash "$PAIR" 131072 "$task"
      ); then
        echo "DONE 128k $task slot=$slot"
      else
        echo "FAIL 128k $task slot=$slot" | tee -a "$FAILS"
      fi
    done
  }

  if [[ -s "$QUEUE" ]]; then
    pair_worker 0 "${GPUS[0]}" "${GPUS[1]}" &
    w0=$!
    pair_worker 1 "${GPUS[2]}" "${GPUS[3]}" &
    w1=$!
    wait "$w0"
    wait "$w1"
  fi
  if [[ -s "$FAILS" ]]; then
    echo "128k FAILS"
    cat "$FAILS"
    exit 1
  fi

  "$PYTHON" "$EVAL" \
    --mode summary \
    --output_dir "$OUTPUT_ROOT" \
    --model_name "$NAME" \
    --lengths "${ALL_LENGTHS[@]}"
  echo "DENSE_S1000_RULER_16K32K64K128K_L2048_DONE"
} 2>&1 | tee "$LOG"
