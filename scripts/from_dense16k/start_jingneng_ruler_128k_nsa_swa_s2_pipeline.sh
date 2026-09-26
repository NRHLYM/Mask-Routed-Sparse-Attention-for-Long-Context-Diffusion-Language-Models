#!/usr/bin/env bash
# Jingneng 4-GPU goldspan RULER 128k pipeline.
# Each task: 1 process, 2 GPUs, layer-parallel (not 50/50 item shard).
# Two pair workers pull a shared queue: as soon as a pair frees, the next
# job starts. Queue: NSA → SWA → s2-qcal, each SN → MK-MQ → VT at L/2048
# only. yarn8fixed (keep-train factor=8) is omitted for all three.
#
# Do not launch from the notebook. Do not overlap the from-dense VF train.
#
#   source /Data/xiongjing/env.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     "$NSA_ROOT/scripts/from_dense16k/start_jingneng_ruler_128k_nsa_swa_s2_pipeline.sh"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../start_jingneng_ruler_128k_nsa_swa_s2_pipeline.sh nsa
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="$HERE/run_official_ruler_task_2gpu.sh"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
export EVAL
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
export DATA_DIR
PYTHON="${PYTHON:-python}"
export PYTHON
MODELS="${1:-nsa,swa,s2}"
IFS=',' read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#G[@]} < 4 )); then
  echo "need 4 GPUs in CUDA_VISIBLE_DEVICES, got ${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 1
fi
[[ -f "$EVAL" && -f "$PAIR" ]] || { echo "missing $EVAL or $PAIR" >&2; exit 1; }

NSA_CKPT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000/step-1000"
S2_CKPT="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000"
SWA_SYNC="/Data/xiongjing/outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/step-1000"
SWA_STOP="/Data/xiongjing/outputs/swa-yarn8-16k-w1280-noskipinert-fromdense-s1000stop800/step-1000"
NSA_CFG="$NSA_ROOT/configs/from_dense16k/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000-jingneng.json"
S2_CFG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json"
SWA_SYNC_CFG="$NSA_ROOT/configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000-jingneng.json"
SWA_STOP_CFG="$NSA_ROOT/configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-jingneng.json"

want_model() {
  [[ ",$MODELS," == *",$1,"* ]]
}

stage_nsa_swa_tree() {
  local tree="/Data/xiongjing/src/eval-trees/nsa-swa-128k-pipe"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local overlay="$NSA_ROOT/dream_dllm_hils"
  local name
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
  printf '%s\n' "$tree"
}

stage_qcal_tree() {
  local tree="/Data/xiongjing/src/eval-trees/hils-s2-qcal-rmsnorm-128k-pipe"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local name
  [[ -f "$src/qcal.py" ]] || { echo "missing $src/qcal.py" >&2; exit 1; }
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  rm -f "$tree/dream_dllm_hils/qcal.py" \
    "$tree/dream_dllm_hils/attention.py" \
    "$tree/dream_dllm_hils/train_fulltext.py"
  cp -f "$src/qcal.py" "$tree/dream_dllm_hils/qcal.py"
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  python3 - "$tree/dream_dllm_hils/qcal.py" "$tree/dream_dllm_hils/train_fulltext.py" "$tree/dream_dllm_hils/attention.py" <<'PY'
from pathlib import Path
import sys
want = "residual-random-lowrank-rmsnorm-v1"
old = "residual-zero-up-native-scale-v1"
qcal, tft, attn = map(Path, sys.argv[1:4])
if want not in qcal.read_text():
    raise SystemExit(f"{qcal} missing {want}")
ttext = tft.read_text()
if want in ttext and old not in ttext:
    print("qcal_version already pinned", want, file=sys.stderr)
elif old in ttext:
    tft.write_text(ttext.replace(old, want))
else:
    raise SystemExit(f"{tft} missing {old} and {want}")
if "qcal_norm" not in attn.read_text():
    raise SystemExit(f"{attn} missing qcal_norm")
print("preflight_ok s2qcal 128k RMSNorm pipe tree", file=sys.stderr)
PY
  printf '%s\n' "$tree"
}

has_ckpt() {
  [[ -s "$1/trainable_state.pt" && -s "$1/checkpoint_manifest.json" ]]
}

NSA_TREE=""
S2_TREE=""
SWA_CKPT=""
SWA_CFG=""
SWA_OUT_BASE=""
if want_model nsa; then
  has_ckpt "$NSA_CKPT" || { echo "missing $NSA_CKPT" >&2; exit 1; }
  [[ -f "$NSA_CFG" ]] || { echo "missing $NSA_CFG" >&2; exit 1; }
  NSA_TREE="$(stage_nsa_swa_tree)"
fi
if want_model swa; then
  if [[ -n "${SWA_CHECKPOINT:-}" ]]; then
    SWA_CKPT="$SWA_CHECKPOINT"
    SWA_CFG="${SWA_CONFIG:-$SWA_SYNC_CFG}"
    SWA_OUT_BASE="$(dirname "$SWA_CKPT")"
  elif has_ckpt "$SWA_SYNC"; then
    SWA_CKPT="$SWA_SYNC"
    SWA_CFG="$SWA_SYNC_CFG"
    SWA_OUT_BASE="$(dirname "$SWA_CKPT")"
  elif has_ckpt "$SWA_STOP"; then
    echo "SWA sync s1000 missing; using $SWA_STOP" >&2
    SWA_CKPT="$SWA_STOP"
    SWA_CFG="$SWA_STOP_CFG"
    SWA_OUT_BASE="$(dirname "$SWA_CKPT")"
  else
    echo "missing SWA ckpt ($SWA_SYNC or $SWA_STOP). Copy tilde s1000 or set SWA_CHECKPOINT." >&2
    exit 1
  fi
  [[ -f "$SWA_CFG" ]] || { echo "missing $SWA_CFG" >&2; exit 1; }
  has_ckpt "$SWA_CKPT" || { echo "missing $SWA_CKPT" >&2; exit 1; }
  [[ -n "$NSA_TREE" ]] || NSA_TREE="$(stage_nsa_swa_tree)"
fi
if want_model s2; then
  has_ckpt "$S2_CKPT" || { echo "missing $S2_CKPT" >&2; exit 1; }
  [[ -f "$S2_CFG" ]] || { echo "missing $S2_CFG" >&2; exit 1; }
  S2_TREE="$(stage_qcal_tree)"
fi

WORKDIR="$(mktemp -d /tmp/xiongjing-ruler128k-pipe.XXXXXX)"
QUEUE="$WORKDIR/queue.txt"
LOCK="$WORKDIR/queue.lock"
FAILS="$WORKDIR/fails.txt"
: > "$QUEUE"
: > "$FAILS"
mkdir -p "$ROOT/logs"

enqueue() {
  local model="$1"
  want_model "$model" || return 0
  local task
  for task in hils_sn hils_mkmq hils_vt; do
    printf '%s %s %s\n' "$model" "L2048" "$task" >> "$QUEUE"
  done
}

enqueue nsa
enqueue swa
enqueue s2
echo "128k pipeline queue ($(wc -l < "$QUEUE") jobs) models=$MODELS pairs=${G[0]},${G[1]} ${G[2]},${G[3]}"
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

output_root_for() {
  local model="$1" yarn="$2"
  local base=""
  case "$model" in
    nsa) base="$(dirname "$NSA_CKPT")" ;;
    swa) base="$SWA_OUT_BASE" ;;
    s2) base="$(dirname "$S2_CKPT")" ;;
    *) echo "bad model $model" >&2; return 1 ;;
  esac
  if [[ "$yarn" == yarn8 ]]; then
    if [[ "$model" == s2 ]]; then
      printf '%s\n' "$base/ruler_probes_goldspan_yarn8"
    else
      printf '%s\n' "$base/ruler_probes_goldspan_yarn8fixed"
    fi
  else
    printf '%s\n' "$base/ruler_probes_goldspan"
  fi
}

run_job() {
  local model="$1" yarn="$2" task="$3" g0="$4" g1="$5" slot="$6"
  local pypath cfg ckpt out inductor
  case "$model" in
    nsa)
      pypath="$NSA_TREE"
      cfg="$NSA_CFG"
      ckpt="$NSA_CKPT"
      ;;
    swa)
      pypath="$NSA_TREE"
      cfg="$SWA_CFG"
      ckpt="$SWA_CKPT"
      ;;
    s2)
      pypath="$S2_TREE"
      cfg="$S2_CFG"
      ckpt="$S2_CKPT"
      ;;
    *)
      echo "bad model $model" >&2
      return 1
      ;;
  esac
  out="$(output_root_for "$model" "$yarn")"
  inductor="/tmp/xiongjing-inductor-128kpipe-${slot}-${model}-${yarn}-${task}"
  mkdir -p "$out" "$inductor/triton" "$inductor/torchinductor" "$inductor/torch_extensions"
  echo "START slot=$slot gpus=$g0,$g1 $model $yarn $task out=$out"
  (
    export PYTHONPATH="$pypath"
    export CONFIG="$cfg"
    export CHECKPOINT="$ckpt"
    export OUTPUT_ROOT="$out"
    export EVAL
    export DATA_DIR
    export PYTHON
    export TMPDIR="$inductor"
    export TRITON_CACHE_DIR="$inductor/triton"
    export TORCHINDUCTOR_CACHE_DIR="$inductor/torchinductor"
    export TORCH_EXTENSIONS_DIR="$inductor/torch_extensions"
    if [[ "$yarn" == yarn8 ]]; then
      export RULER_KEEP_TRAIN_YARN=1
      export RULER_YARN_FACTOR=8
    else
      unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true
    fi
    cd "$pypath"
    CUDA_VISIBLE_DEVICES="$g0,$g1" bash "$PAIR" 131072 "$task"
  )
}

pair_worker() {
  local slot="$1" g0="$2" g1="$3"
  local line model yarn task
  while true; do
    if ! line="$(pop_job)"; then
      echo "slot=$slot idle, queue empty"
      break
    fi
    read -r model yarn task <<< "$line"
    if ! run_job "$model" "$yarn" "$task" "$g0" "$g1" "$slot"; then
      echo "FAIL $model $yarn $task slot=$slot" | tee -a "$FAILS"
    else
      echo "DONE $model $yarn $task slot=$slot"
    fi
  done
}

summarize_all() {
  local model yarn out name
  for model in nsa swa s2; do
    want_model "$model" || continue
    for yarn in L2048; do
      out="$(output_root_for "$model" "$yarn")"
      [[ -d "$out" ]] || continue
      name="${model}128k_${yarn}"
      PYTHONPATH="${S2_TREE:-$NSA_TREE}" "$PYTHON" "$EVAL" \
        --mode summary --output_dir "$out" --model_name "$name" --lengths 131072 \
        || true
    done
  done
}

LOG="$ROOT/logs/ruler-128k-nsa-swa-s2-pipe-$(date +%Y%m%d-%H%M%S).log"
{
  pair_worker 0 "${G[0]}" "${G[1]}" &
  w0=$!
  pair_worker 1 "${G[2]}" "${G[3]}" &
  w1=$!
  wait "$w0"
  wait "$w1"
  summarize_all
  if [[ -s "$FAILS" ]]; then
    echo "PIPELINE_FAILS"
    cat "$FAILS"
    exit 1
  fi
  echo "NSA_SWA_S2_RULER_128K_PIPELINE_DONE models=$MODELS"
} 2>&1 | tee "$LOG"
