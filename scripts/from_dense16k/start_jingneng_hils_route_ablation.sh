#!/usr/bin/env bash
# Jingneng GPU: HiLS routing ablations S1-S4. 4 GPUs. PYTHONPATH=$FULLTEACHER_ROOT.
# SETTING=s1|s2|s3|s4
# Fresh dirs from dense-500. Resume only the same setting's incomplete ckpt.
set -euo pipefail
SETTING="${SETTING:?set SETTING=s1|s2|s3|s4}"
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-${SETTING}}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"

case "$SETTING" in
  s1)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s1-cefusion-teacher001-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s1-cefusion-teacher001-s1000"
    WANT_TEACHER=0.01 WANT_CE_STE=0 WANT_ALLCHUNK=0
    ;;
  s2)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-noteacher-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000"
    WANT_TEACHER=0 WANT_CE_STE=0 WANT_ALLCHUNK=0
    ;;
  s3)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s3-cefusion-lmkceste-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s3-cefusion-lmkceste-s1000"
    WANT_TEACHER=0 WANT_CE_STE=1 WANT_ALLCHUNK=0
    ;;
  s4)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s4-cefusion-allchunkst-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s4-cefusion-allchunkst-s1000"
    WANT_TEACHER=0 WANT_CE_STE=0 WANT_ALLCHUNK=16
    ;;
  *)
    echo "unknown SETTING=$SETTING" >&2
    exit 1
    ;;
esac

DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-route-${SETTING}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/allchunk_gumbel.py" ]] || {
  echo "missing allchunk_gumbel.py in FULLTEACHER_ROOT" >&2
  exit 1
}
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
from dream_dllm_hils.allchunk_gumbel import attach_allchunk_st
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
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("S1-S4 must set hils_detach_fusion_weights=false")
if abs(float(cfg.get("hils_dense_teacher_weight", -1)) - float("$WANT_TEACHER")) > 1e-12:
    raise SystemExit("teacher weight mismatch for $SETTING")
if bool(cfg.get("hils_lmk_ce_ste", False)) != bool(int("$WANT_CE_STE")):
    raise SystemExit("lmk_ce_ste mismatch for $SETTING")
if int(cfg.get("hils_allchunk_st_queries", -1)) != int("$WANT_ALLCHUNK"):
    raise SystemExit("allchunk queries mismatch for $SETTING")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("exploration arms use max_steps=1000")
if int(cfg.get("warmup_steps", 0)) != 50 or int(cfg.get("ruler_val_batches", 0)) != 8:
    raise SystemExit("warmup_steps=50 ruler_val_batches=8 required")
print("preflight_ok", "$SETTING", route_topk_g7.__name__, attach_allchunk_st.__name__)
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
echo "setting=$SETTING nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH tmpdir=$TMPDIR log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
