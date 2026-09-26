#!/usr/bin/env bash
# Stage-2 of all-28 HiLS: load s500 weights only, new 500-step cosine at 1e-4.
# Not a stretched cosine-1000 (that would jump lr from 0 to ~5e-5).
# Fresh Adam (moments were from vanishing lr). New dir; s500 untouched.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-dolmaruler-sync-all28-s1000}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-all28-s1000-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s1000"
SRC="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-s2-dolmaruler-sync-all28-s1000-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$SRC/trainable_state.pt" ]] || { echo "missing all28-500: $SRC" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#_gpus[@]} != NPROC )); then
  echo "need NPROC=$NPROC GPUs, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done

python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import hils_layer_indices
from dream_dllm_hils.train_fulltext import validate_training_config
import json

p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if hils_layer_indices(28, 1) != list(range(28)):
    raise SystemExit("hils_interleave=1 must select all 28 layers")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir must be all28-s1000")
if str(cfg.get("initialize_from", "")) != "$SRC":
    raise SystemExit("stage-2 must initialize_from all28 step-500 weights")
if int(cfg.get("hils_interleave", 0)) != 1:
    raise SystemExit("hils_interleave=1")
if int(cfg.get("max_steps", 0)) != 500:
    raise SystemExit("stage-2 is a new 500-step cosine, not a stretched 1000")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate=1e-4")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("new cosine over these 500 steps")
if int(cfg.get("warmup_steps", 0)) != 25:
    raise SystemExit("warmup_steps=25")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
    raise SystemExit("scope must stay lora_qcal_lmk")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("live fusion")
print("preflight_ok s2-dolma-ruler-sync-all28-s1000-stage2")
PY

source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
start_args=()
if checkpoint_is_complete "$OUTPUT/step-500"; then
  echo "already complete: $OUTPUT/step-500" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -n "$resume_from" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "setting=s2-dolma-ruler-sync-all28-s1000-stage2 nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
