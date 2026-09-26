#!/usr/bin/env bash
# Jingneng GPU: NSA i4 budget-aligned (W=128, 32x64 select, compress 256).
# Code: tilde snapshot $HILS_FT_ROOT (not fullteacher, not nsa-dream overlay).
# Default GPUs 2,3 so it can share a 4-GPU job with HiLS w256 on 0,1.
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
cd "$HILS_FT_ROOT"
NPROC="${NPROC:-2}"
CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-yarn8-16k-b32-w128-c64s64-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-i4-fromdense"
LOG="$ROOT/logs/nsa-i4-w128-c64s64-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -d "$HILS_FT_ROOT/dream_dllm_hils" && -d "$HILS_FT_ROOT/ops" ]] || {
  echo "missing tilde snapshot $HILS_FT_ROOT" >&2
  exit 1
}
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done
python - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.nsa_attention import install_dream_nsa_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import dream_dllm_hils
import json, ops

root = Path("$HILS_FT_ROOT").resolve()
pkg = Path(dream_dllm_hils.__file__).resolve().parent
if pkg != root / "dream_dllm_hils":
    raise SystemExit(f"NSA must import tilde snapshot, got {pkg}")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if cfg.get("attention_mode") != "nsa":
    raise SystemExit("config must be attention_mode=nsa")
if int(cfg.get("local_window", 0)) != 128:
    raise SystemExit("aligned NSA must set local_window=128")
if int(cfg.get("nsa_block_count", 0)) != 32:
    raise SystemExit("aligned NSA must set nsa_block_count=32")
if int(cfg.get("nsa_compress_block", 0)) != 64 or int(cfg.get("nsa_compress_stride", 0)) != 64:
    raise SystemExit("aligned NSA must set compress_block=stride=64")
if "compress_block" not in signature(install_dream_nsa_attention).parameters:
    raise SystemExit("nsa_attention.install_dream_nsa_attention missing compress_block")
import ops.dsa_selected_attention_tilelang  # noqa: F401
print("preflight_ok nsa_w128_c64s64", "code_root", root)
PY

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}

start_args=()
if checkpoint_is_complete "$OUTPUT/step-500"; then
  echo "already complete: $OUTPUT/step-500" >&2
  exit 0
fi
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${resume_from}" ]] && checkpoint_is_complete "$resume_from"; then
  start_args=(--resume_from "$resume_from")
fi
echo "nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
