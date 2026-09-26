#!/usr/bin/env bash
# Jingneng GPU: official goldspan RULER for 3 SWA + 1 dense s1000.
# SN / MK-MQ / VT at 16k/32k/64k/128k. YaRN = L/2048 (not keep-train 8).
# 21 SWA stay radius 1280; only the 7 dense slots get full-window Fast-dLLM
# cache wrap. 16/32/64: 1 GPU/task. 128k: 2 GPU/task layer-parallel.
# PYTHONPATH = staged tree (hybrid train_fulltext.py). Do not launch from
# the notebook. Do not set RULER_KEEP_TRAIN_YARN.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_swa3_dense1_s1000_ruler_16k32k64k128k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true

HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="$HERE/run_official_ruler_task_2gpu.sh"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
CONFIG="$NSA_ROOT/configs/from_dense16k/swa3-dense1-i4-w1280-dolmaruler-sync-s1000-jingneng.json"
CHECKPOINT="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/step-1000"
OUTPUT_ROOT="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/ruler_probes_goldspan"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
PATCH_DIR="$NSA_ROOT/dream_dllm_hils"
TREE="/Data/xiongjing/src/eval-trees/swa3-dense1-i4-s1000"
PYTHON="${PYTHON:-python}"
export EVAL CONFIG CHECKPOINT OUTPUT_ROOT DATA_DIR PYTHON TREE
TASKS=(hils_sn hils_mkmq hils_vt)
SHORT_LENGTHS=(65536 32768 16384)
ALL_LENGTHS=(16384 32768 65536 131072)
NAME=swa3dense1s1000
# HILS_FT cache forbids multi-device KV (128k layer-parallel). Copy NSA
# Fast-dLLM files that keep per-layer cache on that layer's GPU.
STAGE_REAL=(train_fulltext.py fastdllm_cache.py fastdllm_v1.py)

stage_tree() {
  local src="$HILS_FT_ROOT/dream_dllm_hils"
  local name base
  mkdir -p "$TREE/dream_dllm_hils"
  ln -sfn "$HILS_FT_ROOT/ops" "$TREE/ops"
  ln -sfn "$HILS_FT_ROOT/scripts" "$TREE/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    base="$(basename "$name")"
    skip=0
    for real in "${STAGE_REAL[@]}"; do
      [[ "$base" == "$real" ]] && skip=1 && break
    done
    if (( skip )); then
      continue
    fi
    ln -sfn "$name" "$TREE/dream_dllm_hils/$base"
  done
  for real in "${STAGE_REAL[@]}"; do
    rm -f "$TREE/dream_dllm_hils/$real"
    cp -a "$PATCH_DIR/$real" "$TREE/dream_dllm_hils/$real"
  done
  [[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
  if ! grep -q "layer_indices" "$EVAL"; then
    echo "eval_jingneng_official_ruler.py must wrap only dense slots" >&2
    exit 1
  fi
  "$PYTHON" - <<PY
from pathlib import Path
tree = Path("$TREE/dream_dllm_hils")
for name in ("train_fulltext.py", "fastdllm_cache.py", "fastdllm_v1.py"):
    p = tree / name
    if p.is_symlink():
        raise SystemExit(f"{name} must be a real file: {p}")
    text = p.read_text(encoding="utf-8")
    if name == "train_fulltext.py" and "def interleaved_swa_dense" not in text:
        raise SystemExit("staged train_fulltext missing interleaved_swa_dense")
    if name == "fastdllm_cache.py" and "all cache layers must share a device" in text:
        raise SystemExit("staged fastdllm_cache still forbids multi-device KV")
    if name == "fastdllm_v1.py" and "layer_dev" not in text:
        raise SystemExit("staged fastdllm_v1 missing per-layer device moves")
    print("staged", p)
PY
}
stage_tree
export PYTHONPATH="$TREE"
cd "$TREE"

INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-swa3dense1-ruler}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/swa3-dense1-ruler-L2048-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" && -f "$PAIR" && -f "$CONFIG" ]] || {
  echo "missing eval/pair/config" >&2
  exit 1
}
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

"$PYTHON" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("$TREE").resolve()))
from dream_dllm_hils.train_fulltext import interleaved_swa_dense
import argparse
cfg = json.loads(Path("$CONFIG").read_text())
ns = argparse.Namespace(**cfg)
if not interleaved_swa_dense(ns):
    raise SystemExit("need attention_mode=dense and non_hils_attention=sliding")
if int(cfg.get("hils_interleave", 0)) != 4:
    raise SystemExit("hils_interleave=4")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("swa_local_window=1280")
if abs(float(cfg.get("ruler_mix_ratio", 1) or 0)) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0")
man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
if int(man.get("step", 0)) != 1000:
    raise SystemExit(f"need step-1000, manifest step={man.get('step')}")
print("preflight_ok swa3-dense1-s1000-ruler L/2048 16/32/64/128k")
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
  echo "===== swa3+dense1 s1000 goldspan RULER 16/32/64/128k YaRN=L/2048 ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT tree=$TREE"
  echo "yarn=unfixed; 21 SWA@1280 kept; 7 dense slots Fast-dLLM cache; 128k layer_parallel=2"

  short_pairs=()
  for length in "${SHORT_LENGTHS[@]}"; do
    for task in "${TASKS[@]}"; do
      if job_done "$length" "$task"; then
        echo "skip swa3dense1 len${length} $task"
        continue
      fi
      if jsonl_ready "$length" "$task"; then
        echo "merge existing swa3dense1 len${length} $task"
        "$PYTHON" "$EVAL" --mode merge --output_dir "$OUTPUT_ROOT" \
          --data_dir "$DATA_DIR" --task "$task" --max_seq_len "$length"
        continue
      fi
      if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
        echo "reset incomplete swa3dense1 len${length} $task"
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
        export PYTHONPATH="$TREE"
        unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
        cd "$TREE"
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
  WORKDIR="$(mktemp -d /tmp/xiongjing-swa3dense1-128k-pipe.XXXXXX)"
  QUEUE="$WORKDIR/queue.txt"
  LOCK="$WORKDIR/queue.lock"
  FAILS="$WORKDIR/fails.txt"
  : > "$QUEUE"
  : > "$FAILS"
  for task in "${TASKS[@]}"; do
    if job_done 131072 "$task"; then
      echo "skip swa3dense1 len131072 $task"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len131072/${task}" ]]; then
      echo "reset incomplete swa3dense1 len131072 $task"
      rm -rf "$OUTPUT_ROOT/len131072/${task}"
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
      inductor="/tmp/xiongjing-inductor-swa3dense1-128k-${slot}-${task}"
      mkdir -p "$inductor/triton" "$inductor/torchinductor" "$inductor/torch_extensions"
      echo "START 128k slot=$slot gpus=$g0,$g1 $task"
      if (
        export PYTHONPATH="$TREE"
        export CONFIG CHECKPOINT OUTPUT_ROOT EVAL DATA_DIR PYTHON
        export TMPDIR="$inductor"
        export TRITON_CACHE_DIR="$inductor/triton"
        export TORCHINDUCTOR_CACHE_DIR="$inductor/torchinductor"
        export TORCH_EXTENSIONS_DIR="$inductor/torch_extensions"
        unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
        cd "$TREE"
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
  echo "SWA3_DENSE1_S1000_RULER_16K32K64K128K_L2048_DONE"
} 2>&1 | tee "$LOG"
