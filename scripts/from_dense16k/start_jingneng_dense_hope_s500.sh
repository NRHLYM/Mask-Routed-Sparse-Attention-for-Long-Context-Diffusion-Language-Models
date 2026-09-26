#!/usr/bin/env bash
# Jingneng GPU: Dream-v0 2k + HoPE CPT to 16k dense. Matches dense-yarn8-16k-step500
# except RoPE: no YaRN, in-range HoPE (period_multiplier=1, orig=2048).
# mix=0.05, lr 2e-4, cosine 500, warmup 25, seed 7, world_size=2 (do not use 4 GPU).
# From base weights, not from dense-yarn-500. Do not steal allchunk GPUs.
#
#   source /Data/xiongjing/env.sh
#   cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_dense_hope_s500.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dense-hope-16k}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-2}"

CONFIG="$NSA_ROOT/configs/from_dense16k/dense-hope-16k-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/dense-hope-16k-step500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/dense-hope-16k-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/hope.py" ]] || {
  echo "missing hope.py in $FULLTEACHER_ROOT" >&2
  exit 1
}
IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#_gpus[@]} != NPROC )); then
  echo "need NPROC=$NPROC GPUs to match dense-yarn world_size=2, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
if (( NPROC != 2 )); then
  echo "dense-yarn8-16k-step500 used world_size=2; refusing NPROC=$NPROC" >&2
  exit 1
fi
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done

python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.train_fulltext import validate_training_config
from dream_dllm_hils.hope import apply_hope_inrange
import json

p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if cfg.get("initialize_from"):
    raise SystemExit("must start from Dream 2k, not a dense checkpoint")
if str(cfg.get("attention_mode", "")) != "dense":
    raise SystemExit("attention_mode must be dense")
if str(cfg.get("non_hils_attention", "")) != "dense":
    raise SystemExit("non_hils_attention must be dense")
scaling = cfg.get("model_rope_scaling") or {}
if str(scaling.get("rope_type", "")) != "hope":
    raise SystemExit("rope_type must be hope")
if int(scaling.get("original_max_position_embeddings") or 0) != 2048:
    raise SystemExit("HoPE original_max_position_embeddings must be 2048")
if abs(float(scaling.get("period_multiplier") or 0) - 1.0) > 1e-12:
    raise SystemExit("period_multiplier must be 1.0")
if "factor" in scaling:
    raise SystemExit("HoPE must not set YaRN factor")
if int(cfg.get("max_steps", 0)) != 500:
    raise SystemExit("max_steps=500")
if abs(float(cfg.get("learning_rate", 0)) - 2e-4) > 1e-12:
    raise SystemExit("learning_rate must be 2e-4 like dense-yarn")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("lr_schedule must be cosine")
if int(cfg.get("warmup_steps", 0)) != 25:
    raise SystemExit("warmup_steps=25")
if abs(float(cfg.get("ruler_mix_ratio", -1)) - 0.05) > 1e-12:
    raise SystemExit("ruler_mix_ratio must stay 0.05")
if bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("dense-yarn used remix mix=0.05, not sync 1:1")
if int(cfg.get("seed", -1)) != 7:
    raise SystemExit("seed must be 7")
print("preflight_ok dense-hope-16k-s500", "apply_hope_inrange", apply_hope_inrange.__name__)
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
echo "setting=dense-hope-16k nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
