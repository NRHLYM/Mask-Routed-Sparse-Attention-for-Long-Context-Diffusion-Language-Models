#!/usr/bin/env bash
# Jingneng GPU: stage-2 short-copy HiLS from dense-500. Does not retrain dense 16k.
# Every step: Dolma local denoise + distant copy 32-64 with 16-token cue.
# Live fusion. Loss = token-sum / global_count (S2 scaling), not 0.5 mean.
# Single needle. ruler_mix=0. Independent of the diverged long-copy arm.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-stage2-short}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-stage2-shortcopy-ce-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-stage2-shortcopy-ce-s1000"
LOG="$ROOT/logs/hils-stage2-shortcopy-ce-$(date +%Y%m%d-%H%M%S).log"
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
    raise SystemExit("short-copy arm requires live fusion")
if bool(cfg.get("hils_asymmetric_gate_ce", False)):
    raise SystemExit("short-copy is balanced live CE, not asymmetric detach")
if not bool(cfg.get("hils_balanced_view_ce", False)):
    raise SystemExit("hils_balanced_view_ce must be true")
if float(cfg.get("ruler_mix_ratio", 1)) != 0.0:
    raise SystemExit("ruler_mix_ratio must be 0")
if int(cfg.get("hils_distant_span_min", 0)) != 32:
    raise SystemExit("short-copy span_min must be 32")
if int(cfg.get("hils_distant_span_max", 0)) != 64:
    raise SystemExit("short-copy span_max must be 64")
if int(cfg.get("hils_distant_needles_min", 0)) != 1:
    raise SystemExit("short-copy is single-needle")
if int(cfg.get("hils_distant_needles_max", 0)) != 1:
    raise SystemExit("short-copy is single-needle")
if int(cfg.get("hils_distant_cue_len", 0)) != 16:
    raise SystemExit("short-copy cue_len must be 16")
if abs(float(cfg.get("hils_dense_teacher_weight", -1))) > 1e-12:
    raise SystemExit("chunk teacher must be 0")
if cfg.get("initialize_from") != "$DENSE":
    raise SystemExit("short-copy must initialize from dense-500")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000 required")
if cfg.get("output_dir") != "$OUTPUT":
    raise SystemExit("do not reuse the long-copy output dir")
print("preflight_ok", "stage2_shortcopy_ce", synthesize_distant_span_copy.__name__)
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
