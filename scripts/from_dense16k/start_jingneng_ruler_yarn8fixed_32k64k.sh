#!/usr/bin/env bash
# Deprecated 32k/64k-only Fixed-YaRN launcher.
# Use start_jingneng_ruler_yarn8fixed_16k128k.sh (dsa -> hybrid -> dense).
# Jingneng GPU: fill table Fixed-YaRN (factor=8) RULER cells at 32k/64k.
# 16k Fixed == Unfixed (both factor 8); do not re-run 16k.
# Goldspan Fast-dLLM 32/0.9. SN / MK-MQ / VT. 4 GPUs, 1 GPU/task.
# Do not launch from the notebook. Do not mix with the Unfixed 128k job
# (that still holds the four cards until DSA 128k finishes).
#
# Missing 16/32/64 table cells this script covers:
#   Full dense  Fixed 32k/64k
#   Hybrid dense Fixed 32k/64k
#   DSA         Fixed 32k/64k
# Vanilla is not covered here (no LoRA-free eval ckpt).
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_32k64k.sh dense
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_32k64k.sh hybrid
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_32k64k.sh dsa
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RULER_KEEP_TRAIN_YARN=1
export RULER_YARN_FACTOR=8

ARM="${1:?usage: $0 dense|hybrid|dsa}"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
PYTHON="${PYTHON:-python}"
TASKS=(hils_sn hils_mkmq hils_vt)
LENGTHS=(65536 32768)
PATCH_DIR="$NSA_ROOT/dream_dllm_hils"

case "$ARM" in
  dense)
    CONFIG="$NSA_ROOT/configs/from_dense16k/dense-yarn8-16k-dolmaruler-sync-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/ruler_probes_goldspan_yarn8fixed"
    NAME=densesync1000_yarn8
    CODE_ROOT="$FULLTEACHER_ROOT"
    export PYTHONPATH="$FULLTEACHER_ROOT"
    ;;
  hybrid)
    CONFIG="$NSA_ROOT/configs/from_dense16k/swa3-dense1-i4-w1280-dolmaruler-sync-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/ruler_probes_goldspan_yarn8fixed"
    NAME=swa3dense1s1000_yarn8
    TREE="/Data/xiongjing/src/eval-trees/swa3-dense1-i4-s1000-yarn8fixed"
    STAGE_REAL=(train_fulltext.py fastdllm_cache.py fastdllm_v1.py)
    CODE_ROOT="$TREE"
    ;;
  dsa)
    CONFIG="$NSA_ROOT/configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000/ruler_probes_goldspan_yarn8fixed"
    NAME=dsaoff1000_yarn8
    TREE="/Data/xiongjing/src/eval-trees/dsa-official-warmup-s1000-yarn8fixed"
    STAGE_REAL=(fastdllm_cache.py fastdllm_v1.py dsa_attention.py)
    CODE_ROOT="$TREE"
    ;;
  *)
    echo "usage: $0 dense|hybrid|dsa" >&2
    exit 1
    ;;
esac

stage_tree() {
  local src="$HILS_FT_ROOT/dream_dllm_hils"
  local name base skip real
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
}

if [[ "$ARM" == "hybrid" || "$ARM" == "dsa" ]]; then
  stage_tree
  export PYTHONPATH="$TREE"
fi
cd "$CODE_ROOT"

INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-${ARM}-ruler-yarn8fixed}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/${ARM}-s1000-ruler-yarn8fixed-32k64k-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"

[[ -f "$EVAL" && -f "$CONFIG" ]] || { echo "missing eval/config" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

"$PYTHON" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
arm = "$ARM"
mode = str(cfg.get("attention_mode", ""))
man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
if int(man.get("step", 0)) != 1000:
    raise SystemExit(f"need step-1000, got {man.get('step')}")
if arm == "dense":
    if mode != "dense" or str(cfg.get("non_hils_attention", "dense")) != "dense":
        raise SystemExit("dense arm needs full dense")
elif arm == "hybrid":
    if mode != "dense" or str(cfg.get("non_hils_attention", "dense")) != "sliding":
        raise SystemExit("hybrid arm needs interleaved SWA+dense")
    if int(cfg.get("hils_interleave", 0)) != 4 or int(cfg.get("swa_local_window", 0)) != 1280:
        raise SystemExit("hybrid must keep i4 swa=1280")
elif arm == "dsa":
    if mode != "dsa":
        raise SystemExit("dsa arm needs attention_mode=dsa")
    if int(cfg.get("dsa_topk", 0)) != 2560 or int(cfg.get("swa_local_window", 0)) != 1280:
        raise SystemExit("dsa must keep topk=2560 swa=1280")
print("preflight_ok", arm, "ruler yarn8fixed 32k/64k")
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
  export RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8
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
      echo "skip $ARM len${length} $task"
      continue
    fi
    if jsonl_ready "$length" "$task"; then
      echo "merge existing $ARM len${length} $task"
      "$PYTHON" "$EVAL" --mode merge --output_dir "$OUTPUT_ROOT" \
        --data_dir "$DATA_DIR" --task "$task" --max_seq_len "$length"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
      echo "reset incomplete $ARM len${length} $task"
      rm -rf "$OUTPUT_ROOT/len${length}/${task}"
    fi
    pairs+=("${length}:${task}")
  done
done

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
{
  echo "===== $ARM goldspan RULER 32k/64k yarn8fixed ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT keep_train_yarn=1 yarn_factor=8 pending=${#pairs[@]}"
  if (( ${#pairs[@]} > 0 )); then
    pids=()
    for i in "${!GPUS[@]}"; do
      gpu="${GPUS[$i]}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="${PYTHONPATH:-}"
        export RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8
        cd "$CODE_ROOT"
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
  echo "${ARM}_S1000_RULER_32K64K_YARN8FIXED_DONE"
} 2>&1 | tee "$LOG"
