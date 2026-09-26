#!/usr/bin/env bash
# Jingneng GPU: DSA token topk=2560 + 21-layer SWA radius 1280 (match all-SWA baseline).
# Cosine 1000, run through step-1000. Code: $HILS_FT_ROOT. Fresh output dir. Default GPUs 2,3 when SWA holds 0,1.
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dsa}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$HILS_FT_ROOT"
NPROC="${NPROC:-2}"
CONFIG="$NSA_ROOT/configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-fromdense-s1000stop800"
LOG="$ROOT/logs/dsa-i4-k2560-swa1280-$(date +%Y%m%d-%H%M%S).log"
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
from inspect import signature
from pathlib import Path
from dream_dllm_hils.attention import KernelDreamSlidingWindowAttention
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.dsa_attention import install_dream_dsa_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import dream_dllm_hils
import json, ops

root = Path("$HILS_FT_ROOT").resolve()
pkg = Path(dream_dllm_hils.__file__).resolve().parent
if pkg != root / "dream_dllm_hils":
    raise SystemExit(f"DSA must import tilde snapshot, got {pkg}")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "skip_inert_slots" not in signature(KernelDreamSlidingWindowAttention.__init__).parameters:
    raise SystemExit("KernelDreamSlidingWindowAttention missing skip_inert_slots")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if cfg.get("attention_mode") != "dsa":
    raise SystemExit("config must be attention_mode=dsa")
if int(cfg.get("dsa_topk", 0)) != 2560:
    raise SystemExit("aligned DSA must set dsa_topk=2560")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers must set swa_local_window=1280")
if str(cfg.get("dsa_backend")) != "tilelang":
    raise SystemExit("DSA must use dsa_backend=tilelang")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("second-stage must set max_steps=1000")
if cfg.get("stop_after_steps") not in (None, 0):
    raise SystemExit("do not set stop_after_steps; train through max_steps")
if int(cfg.get("warmup_steps", 0)) != 50 or int(cfg.get("ruler_val_batches", 0)) != 8:
    raise SystemExit("second-stage must set warmup_steps=50 ruler_val_batches=8")
src = Path(install_dream_dsa_attention.__code__.co_filename).read_text()
if "skip_inert_slots" not in src:
    raise SystemExit("DSA sliding install must pass skip_inert_slots=False")
import ops.dsa_selected_attention_tilelang  # noqa: F401
print("preflight_ok dsa_k2560_swa1280", "code_root", root)
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
