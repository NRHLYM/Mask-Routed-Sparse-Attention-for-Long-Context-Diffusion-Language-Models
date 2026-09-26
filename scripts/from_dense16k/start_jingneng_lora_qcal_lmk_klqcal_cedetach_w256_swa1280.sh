#!/usr/bin/env bash
# Jingneng GPU: HiLS sparse W=256 + 21-layer SWA radius 1280 (match all-SWA baseline).
# Cosine 1000, run through step-1000. New output dir; do not resume the 500-step ckpt. PYTHONPATH=$FULLTEACHER_ROOT.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-2}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800"
LOG="$ROOT/logs/hils-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done
python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.routing import route_topk_g7
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
if int(cfg.get("local_window", 0)) != 256:
    raise SystemExit("HiLS sparse branch must set local_window=256")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers must set swa_local_window=1280")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("second-stage must set max_steps=1000")
if cfg.get("stop_after_steps") not in (None, 0):
    raise SystemExit("do not set stop_after_steps; train through max_steps")
if int(cfg.get("warmup_steps", 0)) != 50 or int(cfg.get("ruler_val_batches", 0)) != 8:
    raise SystemExit("second-stage must set warmup_steps=50 ruler_val_batches=8")
if cfg.get("hils_lmk_kl_ste", True) or cfg.get("hils_lmk_ce_ste", False):
    raise SystemExit("w256 split-old must disable LMK KL-STE and CE-STE")
print("preflight_ok", route_topk_g7.__name__, "klqcal_cedetach_w256_swa1280")
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
