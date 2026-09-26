#!/usr/bin/env bash
# From-scratch 4-GPU HiLS: Dream-v0 2k + YaRN8, Dolma+RULER 1:1 CE, 1500 cosine steps.
# LR 1e-4 to match s2-sync / nsa-sync. Does not resume the 2e-4 blow-up or 2-GPU 500.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-dolmaruler-sync-from2k-s1500-lr1e4}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-from2k-s1500-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-s1500-lr1e4"
BASE="/Data/xiongjing/models/Dream-v0-Base-7B"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-s2-dolmaruler-sync-from2k-s1500-lr1e4-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$BASE/config.json" ]] || { echo "missing Dream 2k base: $BASE" >&2; exit 1; }
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
from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import json

p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "swa_local_window" not in signature(install_dream_sparse_attention).parameters:
    raise SystemExit("install_dream_sparse_attention missing swa_local_window")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir must be the from2k s1500 lr1e4 dir")
if str(cfg.get("model_path", "")) != "$BASE":
    raise SystemExit("must load Dream-v0-Base-7B")
if cfg.get("initialize_from"):
    raise SystemExit("from-scratch 1500 must not initialize_from dense or step-500")
if str(cfg.get("attention_mode", "")) != "hils":
    raise SystemExit("attention_mode=hils")
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type", "")) != "yarn" or abs(float(rope.get("factor", 0)) - 8.0) > 1e-12:
    raise SystemExit("YaRN factor must be 8 (16k / 2048)")
if int(rope.get("original_max_position_embeddings", 0)) != 2048:
    raise SystemExit("original_max_position_embeddings must be 2048")
if int(cfg.get("max_length", 0)) != 16384:
    raise SystemExit("max_length=16384")
if int(cfg.get("max_steps", 0)) != 1500:
    raise SystemExit("max_steps=1500")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate=1e-4 to match s2-sync / nsa-sync")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("lr_schedule=cosine")
if int(cfg.get("warmup_steps", 0)) != 25:
    raise SystemExit("warmup_steps=25")
if abs(float(cfg.get("ruler_mix_ratio", 0))) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
    raise SystemExit("scope must stay lora_qcal_lmk")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("live fusion")
print("preflight_ok s2-dolma-ruler-sync-from2k-s1500-lr1e4")
PY

source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
start_args=()
if checkpoint_is_complete "$OUTPUT/step-1500"; then
  echo "already complete: $OUTPUT/step-1500" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -n "$resume_from" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "setting=s2-from2k-s1500-lr1e4 nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
