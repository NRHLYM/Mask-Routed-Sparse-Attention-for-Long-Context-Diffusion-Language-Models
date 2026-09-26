#!/usr/bin/env bash
# Jingneng GPU: Fixed-YaRN (keep-train factor=8) goldspan RULER for s2-qcal
# mechanism ablations. 16/32/64/128k. SN / MK-MQ / VT. Fast-dLLM 32/0.9.
# 16k Fixed == Unfixed numerically; writes yarn8fixed dirs.
# 16/32/64: 1 GPU/task. 128k: 2 GPU layer-parallel.
# Sequential queue. Multi-arm (all/rest/QUEUE_ARMS): every arm's 16/32/64 first,
# then every arm's 128k. Single-arm still does 16/32/64 then 128k unless
# RULER_SKIP_128K=1 or RULER_SKIP_SHORT=1. Do not launch from the notebook.
#   all  = learned_route no_qcal no_role
#   rest = no_entropy eos_route mean_pool  (mean_pool last; needs step-1000)
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh all
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh rest
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh learned_route
#   QUEUE_ARMS="no_entropy eos_route" CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RULER_KEEP_TRAIN_YARN=1
export RULER_YARN_FACTOR=8

HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh"
PAIR="$HERE/run_official_ruler_task_2gpu.sh"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
PYTHON="${PYTHON:-python}"
TASKS=(hils_sn hils_mkmq hils_vt)
SHORT_LENGTHS=(65536 32768 16384)
ALL_LENGTHS=(16384 32768 65536 131072)
DEFAULT_ARMS=(learned_route no_qcal no_role)
REST_ARMS=(no_entropy eos_route mean_pool)
QUEUE_ARMS="${QUEUE_ARMS:-}"

arm_meta() {
  case "$1" in
    learned_route)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-learned-route-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-learned-route-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-learned-route-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_learned_route_yarn8
      ;;
    no_qcal)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-qcal-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-qcal-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-qcal-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_no_qcal_yarn8
      ;;
    no_role)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-role-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-role-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-role-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_no_role_yarn8
      ;;
    mean_pool)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-mean-pool-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-mean-pool-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-mean-pool-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_mean_pool_yarn8
      ;;
    no_entropy)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-entropy-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-entropy-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-entropy-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_no_entropy_yarn8
      ;;
    eos_route)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-eos-route-s1000-jingneng.json"
      CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-eos-route-s1000/step-1000"
      OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-eos-route-s1000/ruler_probes_goldspan_yarn8fixed"
      NAME=s2qcal_eos_route_yarn8
      ;;
    *)
      echo "usage: $0 all|rest|learned_route|no_qcal|no_role|mean_pool|no_entropy|eos_route" >&2
      return 1
      ;;
  esac
}

run_arms_short_then_128k() {
  local next
  for next in "$@"; do
    RULER_SKIP_128K=1 bash "$SELF" "$next"
  done
  for next in "$@"; do
    RULER_SKIP_SHORT=1 bash "$SELF" "$next"
  done
}

if [[ "${1:-}" == "all" || ( "${1:-}" == "" && -z "$QUEUE_ARMS" ) ]]; then
  run_arms_short_then_128k "${DEFAULT_ARMS[@]}"
  echo "MECH_ABLATE_RULER_16K128K_YARN8FIXED_DONE"
  exit 0
elif [[ "${1:-}" == "rest" ]]; then
  run_arms_short_then_128k "${REST_ARMS[@]}"
  echo "MECH_ABLATE_RULER_16K128K_YARN8FIXED_REST_DONE"
  exit 0
elif [[ "${1:-}" == "" && -n "$QUEUE_ARMS" ]]; then
  # shellcheck disable=SC2206
  run_arms_short_then_128k $QUEUE_ARMS
  echo "MECH_ABLATE_RULER_16K128K_YARN8FIXED_DONE"
  exit 0
fi

ARM="$1"
arm_meta "$ARM"

stage_tree() {
  local tree="/Data/xiongjing/src/eval-trees/hils-s2-qcal-mech-ablate-yarn8fixed"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local name
  [[ -f "$src/qcal.py" ]] || { echo "missing $src/qcal.py; scp first" >&2; exit 1; }
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
  python3 - "$tree/dream_dllm_hils/qcal.py" \
    "$tree/dream_dllm_hils/train_fulltext.py" \
    "$tree/dream_dllm_hils/attention.py" <<'PY'
from pathlib import Path
import sys
want = "residual-random-lowrank-rmsnorm-v1"
old = "residual-zero-up-native-scale-v1"
qcal, tft, attn = map(Path, sys.argv[1:4])
qtext = qcal.read_text()
if want not in qtext:
    raise SystemExit(f"{qcal} missing {want}")
ttext = tft.read_text()
if want in ttext and old not in ttext:
    print("qcal_version already pinned", want, file=sys.stderr)
elif old in ttext:
    tft.write_text(ttext.replace(old, want))
    print("pinned qcal_version", want, file=sys.stderr)
else:
    raise SystemExit(f"{tft} missing {old} and {want}")
if "qcal_norm" not in attn.read_text():
    raise SystemExit(f"{attn} missing qcal_norm")
if "install_vocab_lmk_embedding" not in ttext:
    raise SystemExit(f"{tft} missing vocab route token; scp train_fulltext.py")
if "lmk_token_mode" not in ttext:
    raise SystemExit(f"{tft} missing lmk_token_mode")
atext = attn.read_text()
if "mean_k = (pooled / denom).to(dtype=k_chunked.dtype)" not in atext:
    raise SystemExit(f"{attn} missing mean-pool bf16 cast; scp attention.py")
print("preflight_ok mech-ablate ruler RMSNorm tree", file=sys.stderr)
PY
  printf '%s\n' "$tree"
}

CODE_ROOT="$(stage_tree)"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"

export EVAL CONFIG CHECKPOINT OUTPUT_ROOT DATA_DIR PYTHON
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-${ARM}-ruler-yarn8fixed-16k128k}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

LOG="$ROOT/logs/${ARM}-s1000-ruler-yarn8fixed-16k128k-$(date +%Y%m%d-%H%M%S).log"
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
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
arm = "$ARM"
man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
if int(man.get("step", 0)) != 1000:
    raise SystemExit(f"need step-1000, got {man.get('step')}")
if str(cfg.get("attention_mode", "")) != "hils":
    raise SystemExit("mech ablate arms are HiLS")
if int(cfg.get("hils_interleave", 0)) != 4 or int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("must keep i4 swa=1280")
if int(cfg.get("hils_topk", 0)) != 32 or int(cfg.get("chunk_size", 0)) != 64:
    raise SystemExit("query budget: hils_topk=32 chunk_size=64")
mode = str(cfg.get("lmk_token_mode", ""))
scope = str(cfg.get("hils_trainable_scope", ""))
rank = int(cfg.get("hils_qcal_rank", -1))
if arm == "learned_route":
    if mode != "vocab" or scope != "lora_qcal_lmk" or rank != 64:
        raise SystemExit("learned_route must be vocab lora_qcal_lmk rank64")
    if str(cfg.get("hils_route_token", "")) != "<|hils_route|>":
        raise SystemExit("learned_route must add <|hils_route|>")
elif arm == "no_qcal":
    if rank != 0 or scope != "lora_lmk" or mode != "mask_type":
        raise SystemExit("no_qcal must be rank0 lora_lmk mask_type")
elif arm == "no_role":
    if rank != 64 or scope != "lora_qcal" or mode != "mask":
        raise SystemExit("no_role must be lora_qcal mask rank64")
elif arm == "mean_pool":
    if str(cfg.get("hils_chunk_summary", "")) != "mean":
        raise SystemExit("mean_pool must be hils_chunk_summary=mean")
    if rank != 64 or scope != "lora_qcal_lmk" or mode != "mask_type":
        raise SystemExit("mean_pool must be lora_qcal_lmk mask_type rank64")
elif arm == "no_entropy":
    if cfg.get("hils_entropy_prior", True) is not False:
        raise SystemExit("no_entropy must set hils_entropy_prior=false")
    if rank != 64 or scope != "lora_qcal_lmk" or mode != "mask_type":
        raise SystemExit("no_entropy must be lora_qcal_lmk mask_type rank64")
elif arm == "eos_route":
    if mode != "eos" or scope != "lora_qcal" or rank != 64:
        raise SystemExit("eos_route must be eos lora_qcal rank64")
print("preflight_ok", arm, "ruler yarn8fixed 16/32/64/128k")
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

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPUS[@]} < 4 )); then
  echo "need 4 GPUs, got $CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

{
  echo "===== $ARM goldspan RULER 16/32/64/128k yarn8fixed ====="
  echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT keep_train_yarn=1 yarn_factor=8"
  echo "skip_short=${RULER_SKIP_SHORT:-0} skip_128k=${RULER_SKIP_128K:-0}"
  echo "order note: multi-arm = all 16/32/64 then all 128k; rest = no_entropy -> eos_route -> mean_pool"

  if [[ "${RULER_SKIP_SHORT:-0}" != "1" ]]; then
  short_pairs=()
  for length in "${SHORT_LENGTHS[@]}"; do
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
        export PYTHONPATH="${PYTHONPATH:-}"
        export RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8
        cd "$CODE_ROOT"
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
  else
    echo "skip 16/32/64 (RULER_SKIP_SHORT=1)"
  fi

  if [[ "${RULER_SKIP_128K:-0}" == "1" ]]; then
    echo "skip 128k (RULER_SKIP_128K=1)"
    echo "${ARM}_S1000_RULER_16K64K_YARN8FIXED_DONE"
    exit 0
  fi

  echo "===== 128k two-GPU layer-parallel yarn8fixed ====="
  WORKDIR="$(mktemp -d /tmp/xiongjing-${ARM}-yarn8-128k-pipe.XXXXXX)"
  QUEUE="$WORKDIR/queue.txt"
  LOCK="$WORKDIR/queue.lock"
  FAILS="$WORKDIR/fails.txt"
  : > "$QUEUE"
  : > "$FAILS"
  for task in "${TASKS[@]}"; do
    if job_done 131072 "$task"; then
      echo "skip $ARM len131072 $task"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len131072/${task}" ]]; then
      echo "reset incomplete $ARM len131072 $task"
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
      inductor="/tmp/xiongjing-inductor-${ARM}-yarn8-128k-${slot}-${task}"
      mkdir -p "$inductor/triton" "$inductor/torchinductor" "$inductor/torch_extensions"
      echo "START 128k slot=$slot gpus=$g0,$g1 $task"
      if (
        export PYTHONPATH="${PYTHONPATH:-}"
        export CONFIG CHECKPOINT OUTPUT_ROOT EVAL DATA_DIR PYTHON
        export TMPDIR="$inductor"
        export TRITON_CACHE_DIR="$inductor/triton"
        export TORCHINDUCTOR_CACHE_DIR="$inductor/torchinductor"
        export TORCH_EXTENSIONS_DIR="$inductor/torch_extensions"
        export RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8
        cd "$CODE_ROOT"
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
  echo "${ARM}_S1000_RULER_16K128K_YARN8FIXED_DONE"
} 2>&1 | tee "$LOG"
