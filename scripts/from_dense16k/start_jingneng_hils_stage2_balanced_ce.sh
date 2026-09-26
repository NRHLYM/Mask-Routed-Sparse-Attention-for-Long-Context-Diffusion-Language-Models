#!/usr/bin/env bash
# Jingneng GPU: stage-2 HiLS from dense-500. Does not retrain dense 16k.
# Every step: Dolma local denoise + long distant span-copy (256-512).
# Live fusion on both. Loss = 0.5 mean(local CE) + 0.5 mean(remote CE).
# ruler_mix=0. Independent of S2 / paper-1000 / support-KL / asymmetric arm.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-stage2-bal}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-stage2-balanced-ce-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-stage2-balanced-ce-s1000"
LOG="$ROOT/logs/hils-stage2-balanced-ce-$(date +%Y%m%d-%H%M%S).log"
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
from dream_dllm_hils.train_fulltext import validate_training_config
from dream_dllm_hils.data import synthesize_distant_span_copy
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
if bool(cfg.get("hils_detach_fusion_weights", True)):
    raise SystemExit("stage-2 requires live fusion")
if bool(cfg.get("hils_asymmetric_gate_ce", False)):
    raise SystemExit("stage-2 is balanced live CE, not asymmetric detach")
if not bool(cfg.get("hils_balanced_view_ce", False)):
    raise SystemExit("hils_balanced_view_ce must be true")
if float(cfg.get("ruler_mix_ratio", 1)) != 0.0:
    raise SystemExit("ruler_mix_ratio must be 0")
if int(cfg.get("hils_distant_span_min", 0)) < 256:
    raise SystemExit("stage-2 distant span must be >= 256")
if int(cfg.get("hils_distant_needles_min", 0)) != 1:
    raise SystemExit("stage-2 must keep single-needle examples")
if int(cfg.get("hils_distant_needles_max", 0)) < 2:
    raise SystemExit("stage-2 must include multi-needle copies")
if int(cfg.get("hils_distant_cue_len", 0)) < 8:
    raise SystemExit("stage-2 distant copies need a visible retrieval cue")
if abs(float(cfg.get("hils_dense_teacher_weight", -1))) > 1e-12:
    raise SystemExit("chunk teacher must be 0")
if cfg.get("initialize_from") != "$DENSE":
    raise SystemExit("stage-2 must initialize from dense-500")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000 required")
print("preflight_ok", "stage2_balanced_ce", synthesize_distant_span_copy.__name__)
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
