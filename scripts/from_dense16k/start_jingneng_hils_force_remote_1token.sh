#!/usr/bin/env bash
# Jingneng GPU: 1-token KEY/VALUE force-remote unit from dense-500.
# Freeze shared LoRA (qcal_lmk). Force fusion remote=1 on the VALUE query.
# No Dolma complementary views. clip=1. Independent of copy/NIAH arms.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-force-remote}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-force-remote-1token-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-force-remote-1token-s200"
LOG="$ROOT/logs/hils-force-remote-1token-$(date +%Y%m%d-%H%M%S).log"
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
from dream_dllm_hils.data import synthesize_one_token_key_value
from dream_dllm_hils.force_remote import apply_forced_remote_gate
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
if not bool(cfg.get("hils_force_remote_unit", False)):
    raise SystemExit("hils_force_remote_unit must be true")
if cfg.get("hils_trainable_scope") != "qcal_lmk":
    raise SystemExit("must freeze LoRA with qcal_lmk")
if bool(cfg.get("hils_detach_fusion_weights", True)):
    raise SystemExit("force-remote keeps live remote mix")
if bool(cfg.get("hils_balanced_view_ce") or cfg.get("hils_asymmetric_gate_ce")):
    raise SystemExit("do not stack copy views")
if float(cfg.get("ruler_mix_ratio", 1)) != 0.0:
    raise SystemExit("ruler_mix_ratio must be 0")
if int(cfg.get("hils_distant_span_min", 0)) != 2:
    raise SystemExit("span_min must be 2")
if int(cfg.get("hils_distant_cue_len", 0)) != 1:
    raise SystemExit("cue_len must be 1")
if abs(float(cfg.get("hils_dense_teacher_weight", -1))) > 1e-12:
    raise SystemExit("chunk teacher must be 0")
if cfg.get("initialize_from") != "$DENSE":
    raise SystemExit("must initialize from dense-500")
if int(cfg.get("max_steps", 0)) != 200:
    raise SystemExit("max_steps=200 required")
if cfg.get("output_dir") != "$OUTPUT":
    raise SystemExit("do not reuse another arm's output dir")
print("preflight_ok", "force_remote_1token", synthesize_one_token_key_value.__name__, apply_forced_remote_gate.__name__)
PY

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}

start_args=()
if checkpoint_is_complete "$OUTPUT/step-200"; then
  echo "already complete: $OUTPUT/step-200" >&2
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
