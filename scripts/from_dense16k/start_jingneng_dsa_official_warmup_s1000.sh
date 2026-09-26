#!/usr/bin/env bash
# Jingneng GPU: DeepSeek-style DSA warm-up 100 + sparse 900 = 1000.
# PYTHONPATH=$HILS_FT_ROOT. 4 GPU. New output dir.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_dsa_official_warmup_s1000.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dsa-official-warmup-s1000}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$HILS_FT_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/dsa-official-warmup-s1000-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -d "$HILS_FT_ROOT/dream_dllm_hils" && -d "$HILS_FT_ROOT/ops" ]] || {
  echo "missing HILS_FT_ROOT $HILS_FT_ROOT" >&2
  exit 1
}
grep -q "warmup_dense" "$HILS_FT_ROOT/dream_dllm_hils/dsa_attention.py" || {
  echo "scp fullteacher dsa_attention.py into HILS_FT_ROOT first" >&2
  exit 1
}
grep -q 'Sparse (and post-warmup resume)' "$HILS_FT_ROOT/dream_dllm_hils/dsa_attention.py" || {
  echo "HILS_FT_ROOT dsa_attention.py still allows full-key KL; scp selected-KL copy" >&2
  exit 1
}
grep -q 'Selected KL must not nest' "$HILS_FT_ROOT/dream_dllm_hils/dsa_attention.py" || {
  echo "HILS_FT_ROOT dsa_attention.py still checkpoints selected KL tiles; scp again" >&2
  exit 1
}
grep -q "_unused_indexer_anchor" "$HILS_FT_ROOT/dream_dllm_hils/dsa_attention.py" || {
  echo "HILS_FT_ROOT dsa_attention.py missing DDP indexer anchor; scp again" >&2
  exit 1
}
grep -q "dsa_warmup_steps" "$HILS_FT_ROOT/dream_dllm_hils/train_fulltext.py" || {
  echo "scp fullteacher train_fulltext.py into HILS_FT_ROOT first" >&2
  exit 1
}
grep -q "dsa_sparse_aux_queries" "$HILS_FT_ROOT/dream_dllm_hils/train_fulltext.py" || {
  echo "HILS_FT_ROOT trainer missing dsa_sparse_aux_queries; scp nsa overlay train_fulltext.py" >&2
  exit 1
}
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
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
from dream_dllm_hils.dsa_attention import set_dsa_official_stage
from dream_dllm_hils.train_fulltext import validate_training_config
import json
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if int(cfg.get("dsa_warmup_steps", 0)) != 100:
    raise SystemExit("dsa_warmup_steps=100")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000")
if int(cfg.get("dsa_aux_queries", -1)) != 0:
    raise SystemExit("dsa_aux_queries=0")
if str(cfg.get("dsa_aux_loss_scope", "")) != "selected":
    raise SystemExit("dsa_aux_loss_scope=selected")
if str(cfg.get("dsa_sparse_aux_scope", "")) != "selected":
    raise SystemExit("dsa_sparse_aux_scope=selected")
if int(cfg.get("dsa_sparse_aux_queries", 0) or 0) != 64:
    raise SystemExit("dsa_sparse_aux_queries=64")
print("preflight_ok dsa-official-warmup", "stage_helper", set_dsa_official_stage.__name__)
print("skip_inert", "skip_inert_slots" in signature(__import__("dream_dllm_hils.attention", fromlist=["KernelDreamSlidingWindowAttention"]).KernelDreamSlidingWindowAttention.__init__).parameters)
PY

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}
FINAL="$OUTPUT/step-1000"
if checkpoint_is_complete "$FINAL"; then
  echo "already complete: $FINAL" >&2
  exit 0
fi
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$resume_from" ]] && checkpoint_is_complete "$resume_from"; then
  # Config still has initialize_from=dense-500; CLI must clear it.
  start_args=(--resume_from "$resume_from" --initialize_from "")
else
  start_args=()
fi
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
