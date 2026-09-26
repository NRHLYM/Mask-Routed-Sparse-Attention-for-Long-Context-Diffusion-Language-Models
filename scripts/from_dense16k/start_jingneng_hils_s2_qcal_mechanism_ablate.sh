#!/usr/bin/env bash
# Jingneng GPU: six s2-qcal mechanism ablations, 1000-step mainline recipe.
# Sequential queue + watchdog: if an arm dies or the log/ckpt goes stale, this
# process kills it, resumes the latest complete step, or advances to the next arm.
#
# Differences vs hils-s2-qcal-rmsnorm-rand-s1000 (everything else identical):
#   learned_route  add <|hils_route|> to the tokenizer/embedding table and train that row
#   no_qcal        drop query residual (rank 0, scope lora_lmk)
#   no_role        drop role offset (mask slots, scope lora_qcal)
#   mean_pool      chunk summary = mean pool
#   no_entropy     do not add H(alpha_c) to chunk scores
#   eos_route      pretrained EOS embedding as the route slot
#
# Mask-token / freeze-Q-Cal / B/C/Bp/B5 group is NOT in this queue.
# SCP FULLTEACHER train_fulltext.py + attention.py before launching.
# Do not launch from the notebook. This script is a blocking queue+watchdog;
# wrap it in nohup so a hang can be taken over by a new invocation:
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     bash scripts/from_dense16k/start_jingneng_hils_s2_qcal_mechanism_ablate.sh \
#     >"$ROOT/logs/hils-s2-qcal-mech-ablate-queue-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_qcal_mechanism_ablate.sh learned_route
#   QUEUE_ARMS="no_qcal no_role" CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../start_jingneng_hils_s2_qcal_mechanism_ablate.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
NPROC="${NPROC:-4}"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
STALE_SECS="${STALE_SECS:-1200}"
POLL_SECS="${POLL_SECS:-60}"
START_GRACE_SECS="${START_GRACE_SECS:-180}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
DEFAULT_ARMS=(learned_route no_qcal no_role mean_pool no_entropy eos_route)
QUEUE_ARMS="${QUEUE_ARMS:-}"

arm_meta() {
  case "$1" in
    learned_route)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-learned-route-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-learned-route-s1000"
      TAG="hils-s2-qcal-ablate-learned-route-s1000"
      ;;
    no_qcal)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-qcal-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-qcal-s1000"
      TAG="hils-s2-qcal-ablate-no-qcal-s1000"
      ;;
    no_role)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-role-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-role-s1000"
      TAG="hils-s2-qcal-ablate-no-role-s1000"
      ;;
    mean_pool)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-mean-pool-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-mean-pool-s1000"
      TAG="hils-s2-qcal-ablate-mean-pool-s1000"
      ;;
    no_entropy)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-no-entropy-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-no-entropy-s1000"
      TAG="hils-s2-qcal-ablate-no-entropy-s1000"
      ;;
    eos_route)
      CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-ablate-eos-route-s1000-jingneng.json"
      OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-ablate-eos-route-s1000"
      TAG="hils-s2-qcal-ablate-eos-route-s1000"
      ;;
    *)
      echo "unknown arm: $1 (learned_route|no_qcal|no_role|mean_pool|no_entropy|eos_route|all)" >&2
      return 1
      ;;
  esac
}

ARMS=()
if [[ "${1:-}" == "all" || ( "${1:-}" == "" && -z "$QUEUE_ARMS" ) ]]; then
  ARMS=("${DEFAULT_ARMS[@]}")
elif [[ "${1:-}" == "" && -n "$QUEUE_ARMS" ]]; then
  # shellcheck disable=SC2206
  ARMS=($QUEUE_ARMS)
else
  ARMS=("$@")
fi

file_mtime() {
  if [[ -e "$1" ]]; then
    stat -c %Y "$1" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

latest_activity_epoch() {
  local output="$1" log="$2"
  local latest=0 m
  m=$(file_mtime "$log")
  if (( m > latest )); then latest=$m; fi
  local cand
  while IFS= read -r cand; do
    [[ -n "$cand" ]] || continue
    m=$(file_mtime "$cand")
    if (( m > latest )); then latest=$m; fi
    m=$(file_mtime "$cand/trainer_state.pt")
    if (( m > latest )); then latest=$m; fi
  done < <(find "$output" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null)
  printf '%s\n' "$latest"
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_train() {
  local pidfile="$1"
  local pid=""
  [[ -f "$pidfile" ]] || return 0
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 0
  if pid_alive "$pid"; then
    echo "stopping train pid=$pid" >&2
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
      pid_alive "$pid" || break
      sleep 1
    done
    if pid_alive "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    pkill -9 -P "$pid" 2>/dev/null || true
  fi
}

claim_watchdog() {
  local lock="$1"
  mkdir -p "$(dirname "$lock")"
  if [[ -f "$lock" ]]; then
    local old
    old="$(cat "$lock" 2>/dev/null || true)"
    if [[ -n "$old" ]] && pid_alive "$old"; then
      echo "watchdog already running pid=$old lock=$lock" >&2
      return 1
    fi
  fi
  echo $$ >"$lock"
  return 0
}

preflight_arm() {
  local arm="$1"
  python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import dream_dllm_hils.train_fulltext as tft
import json

arm = "$arm"
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
tft_src = Path("$FULLTEACHER_ROOT/dream_dllm_hils/train_fulltext.py").read_text()
attn_src = Path("$FULLTEACHER_ROOT/dream_dllm_hils/attention.py").read_text()
if "lora_lmk" not in tft_src:
    raise SystemExit("FULLTEACHER train_fulltext.py missing lora_lmk; scp first")
if "hils_chunk_summary" not in tft.DEFAULTS:
    raise SystemExit("FULLTEACHER train_fulltext.py missing hils_chunk_summary; scp first")
if "hils_entropy_prior" not in tft.DEFAULTS:
    raise SystemExit("FULLTEACHER train_fulltext.py missing hils_entropy_prior; scp first")
if "chunk_summary" not in signature(install_dream_sparse_attention).parameters:
    raise SystemExit("install_dream_sparse_attention missing chunk_summary; scp attention.py")
if "_gqa_mean_chunk_keys" not in attn_src:
    raise SystemExit("attention.py missing mean-pool helper; scp first")
if 'mean_k = (pooled / denom).to(dtype=k_chunked.dtype)' not in attn_src:
    raise SystemExit("attention.py mean-pool must cast landmark keys back to K dtype; scp first")
if "install_vocab_lmk_embedding" not in tft_src:
    raise SystemExit("FULLTEACHER missing vocab route token; scp train_fulltext.py")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type") or "") != "yarn" or abs(float(rope.get("factor") or 0) - 8.0) > 1e-12:
    raise SystemExit("must stay YaRN factor=8")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-yarn8-16k-step500")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate must be 1e-4")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("lr_schedule must be cosine over 1000 steps")
if int(cfg.get("warmup_steps", 0)) != 50:
    raise SystemExit("warmup_steps=50")
if int(cfg.get("hils_topk", 0)) != 32 or int(cfg.get("chunk_size", 0)) != 64:
    raise SystemExit("query budget: hils_topk=32 chunk_size=64")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers must set swa_local_window=1280")
if int(cfg.get("hils_interleave", 0)) != 4:
    raise SystemExit("hils_interleave=4")
if abs(float(cfg.get("ruler_mix_ratio", 0))) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("keep live fusion")
if int(cfg.get("hils_allchunk_st_queries", -1)) != 0:
    raise SystemExit("do not enable S4")
summary = str(cfg.get("hils_chunk_summary", "attn"))
entropy = bool(cfg.get("hils_entropy_prior", True))
scope = str(cfg.get("hils_trainable_scope", ""))
mode = str(cfg.get("lmk_token_mode", ""))
rank = int(cfg.get("hils_qcal_rank", -1))
if arm == "no_qcal":
    if rank != 0 or scope != "lora_lmk" or mode != "mask_type":
        raise SystemExit("no_qcal must be rank0 lora_lmk mask_type")
    if summary != "attn" or not entropy:
        raise SystemExit("no_qcal must keep attn summary and entropy prior")
elif arm == "no_role":
    if rank != 64 or scope != "lora_qcal" or mode != "mask":
        raise SystemExit("no_role must be lora_qcal mask rank64")
    if summary != "attn" or not entropy:
        raise SystemExit("no_role must keep attn summary and entropy prior")
elif arm == "mean_pool":
    if summary != "mean" or scope != "lora_qcal_lmk" or mode != "mask_type" or rank != 64:
        raise SystemExit("mean_pool must only change chunk summary")
    if not entropy:
        raise SystemExit("mean_pool keeps entropy prior")
elif arm == "no_entropy":
    if entropy or summary != "attn" or scope != "lora_qcal_lmk" or mode != "mask_type":
        raise SystemExit("no_entropy must only drop H(alpha_c)")
elif arm == "learned_route":
    if mode != "vocab" or scope != "lora_qcal_lmk" or rank != 64:
        raise SystemExit("learned_route must be vocab lora_qcal_lmk")
    if str(cfg.get("hils_route_token", "")) != "<|hils_route|>":
        raise SystemExit("learned_route must add <|hils_route|>")
    if summary != "attn" or not entropy:
        raise SystemExit("learned_route keeps attn summary and entropy prior")
elif arm == "eos_route":
    if mode != "eos" or scope != "lora_qcal" or rank != 64:
        raise SystemExit("eos_route must be eos lora_qcal rank64")
    if summary != "attn" or not entropy:
        raise SystemExit("eos_route keeps attn summary and entropy prior")
print(f"preflight_ok {arm}")
PY
}

launch_train() {
  local log="$1"
  local resume_from=""
  local start_args=()
  resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
  if [[ -n "$resume_from" ]]; then
    start_args=(--resume_from "$resume_from")
  fi
  echo "setting=$TAG nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$log ${start_args[*]:-}"
  setsid python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
    >"$log" 2>&1 </dev/null &
  echo $! >"$OUTPUT/train.pid"
  echo "started pid=$(cat "$OUTPUT/train.pid") log=$log"
}

run_arm() {
  local arm="$1"
  arm_meta "$arm"
  local inductor="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-$TAG}"
  export TMPDIR="$inductor"
  export TRITON_CACHE_DIR="$inductor/triton"
  export TORCHINDUCTOR_CACHE_DIR="$inductor/torchinductor"
  export TORCH_EXTENSIONS_DIR="$inductor/torch_extensions"
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
  mkdir -p "$OUTPUT" "$ROOT/logs"
  local lock="$OUTPUT/watchdog.lock"
  if ! claim_watchdog "$lock"; then
    echo "skip $arm: another watchdog holds the lock" >&2
    return 0
  fi
  trap 'rm -f "$lock"' RETURN
  if checkpoint_is_complete "$OUTPUT/step-1000"; then
    echo "already complete: $OUTPUT/step-1000" >&2
    stop_train "$OUTPUT/train.pid"
    return 0
  fi
  preflight_arm "$arm"

  local restarts=0
  while (( restarts <= MAX_RESTARTS )); do
    if checkpoint_is_complete "$OUTPUT/step-1000"; then
      echo "complete: $OUTPUT/step-1000" >&2
      stop_train "$OUTPUT/train.pid"
      return 0
    fi
    local log="$ROOT/logs/${TAG}-$(date +%Y%m%d-%H%M%S).log"
    stop_train "$OUTPUT/train.pid"
    launch_train "$log"
    local pid
    pid="$(cat "$OUTPUT/train.pid")"
    local started_at
    started_at=$(date +%s)
    echo "$log" >"$OUTPUT/train.logpath"
    while pid_alive "$pid"; do
      if checkpoint_is_complete "$OUTPUT/step-1000"; then
        echo "complete while running: $OUTPUT/step-1000" >&2
        stop_train "$OUTPUT/train.pid"
        return 0
      fi
      sleep "$POLL_SECS"
      local now activity age
      now=$(date +%s)
      if (( now - started_at < START_GRACE_SECS )); then
        continue
      fi
      activity=$(latest_activity_epoch "$OUTPUT" "$log")
      age=$(( now - activity ))
      if (( activity > 0 && age > STALE_SECS )); then
        echo "stale ${age}s on $arm (log=$log); kill and resume" >&2
        stop_train "$OUTPUT/train.pid"
        break
      fi
    done
    sleep 2
    if checkpoint_is_complete "$OUTPUT/step-1000"; then
      echo "complete: $OUTPUT/step-1000" >&2
      stop_train "$OUTPUT/train.pid"
      return 0
    fi
    restarts=$((restarts + 1))
    echo "arm $arm exited without step-1000; restart $restarts/$MAX_RESTARTS" >&2
  done
  echo "arm $arm failed after $MAX_RESTARTS restarts; next arm will take the GPUs" >&2
  return 0
}

cd "$FULLTEACHER_ROOT"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#_gpus[@]} != NPROC )); then
  echo "need NPROC=$NPROC GPUs, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/qcal.py" ]] || {
  echo "missing qcal.py on FULLTEACHER_ROOT" >&2
  exit 1
}

echo "queue=${ARMS[*]} stale=${STALE_SECS}s grace=${START_GRACE_SECS}s max_restarts=$MAX_RESTARTS"
for arm in "${ARMS[@]}"; do
  run_arm "$arm"
done
echo "queue finished"
