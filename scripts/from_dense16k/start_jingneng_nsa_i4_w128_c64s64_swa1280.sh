#!/usr/bin/env bash
# Jingneng GPU: NSA sparse W=128 + 21-layer SWA radius 1280 (match all-SWA baseline).
# Cosine 1000, run through step-1000. Fresh output dir. Do not resume the 500-step ckpt.
# Code: $HILS_FT_ROOT (hils-fullteacher snapshot with nsa/swa). Do not use FULLTEACHER_ROOT.
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsa}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$HILS_FT_ROOT"
NPROC="${NPROC:-2}"
CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-yarn8-16k-b32-w128-c64s64-swa1280-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-swa1280-i4-fromdense-s1000stop800"
LOG="$ROOT/logs/nsa-i4-w128-c64s64-swa1280-$(date +%Y%m%d-%H%M%S).log"
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
if pkg.parent.name == "fullteacher-20260917":
    raise SystemExit("NSA PYTHONPATH must not be FULLTEACHER_ROOT")
if pkg != root / "dream_dllm_hils":
    raise SystemExit(f"NSA must import HILS_FT_ROOT, got {pkg}")
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
    raise SystemExit("aligned NSA sparse branch must set local_window=128")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers must set swa_local_window=1280")
if int(cfg.get("nsa_block_count", 0)) != 32:
    raise SystemExit("aligned NSA must set nsa_block_count=32")
if int(cfg.get("nsa_compress_block", 0)) != 64 or int(cfg.get("nsa_compress_stride", 0)) != 64:
    raise SystemExit("aligned NSA must set compress_block=stride=64")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("second-stage must set max_steps=1000")
if cfg.get("stop_after_steps") not in (None, 0):
    raise SystemExit("do not set stop_after_steps; train through max_steps")
if int(cfg.get("warmup_steps", 0)) != 50 or int(cfg.get("ruler_val_batches", 0)) != 8:
    raise SystemExit("second-stage must set warmup_steps=50 ruler_val_batches=8")
if "swa_local_window" not in signature(install_dream_nsa_attention).parameters:
    raise SystemExit("nsa_attention.install_dream_nsa_attention missing swa_local_window")
import ops.dsa_selected_attention_tilelang  # noqa: F401
print("preflight_ok nsa_w128_swa1280_c64s64", "code_root", root)
PY

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}

start_args=()
if checkpoint_is_complete "$OUTPUT/step-1000"; then
  echo "already complete: $OUTPUT/step-1000" >&2
  exit 0
fi
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${resume_from}" ]] && checkpoint_is_complete "$resume_from"; then
  start_args=(--resume_from "$resume_from")
fi
echo "nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH tmpdir=$TMPDIR log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
