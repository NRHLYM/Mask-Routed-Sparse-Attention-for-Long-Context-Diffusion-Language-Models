#!/usr/bin/env bash
# KEY= VALUE=[MASK] gate BCE: oracle support + native live gate.
# Train never force_remote=1. force_on is eval-only positive control.
# Freeze LoRA. Fresh from dense-500. Do not reuse oracle/keyvalue/1token dirs.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-gate-bce}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-gate-bce-oracle-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-gate-bce-oracle-s200"
STALE_A="/Data/xiongjing/outputs/hils-force-remote-1token-s200"
STALE_B="/Data/xiongjing/outputs/hils-force-remote-keyvalue-s200"
STALE_C="/Data/xiongjing/outputs/hils-force-remote-oracle-s200"
LOG="$ROOT/logs/hils-gate-bce-oracle-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ "$OUTPUT" != "$STALE_A" && "$OUTPUT" != "$STALE_B" && "$OUTPUT" != "$STALE_C" ]] || { echo "refusing stale output dir" >&2; exit 1; }
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done

python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.train_fulltext import validate_training_config
from dream_dllm_hils.data import (
    FullTextComplementaryCollator,
    synthesize_one_token_key_value,
)
from dream_dllm_hils.force_remote import fusion_gate_bce
import json

p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "swa_local_window" not in signature(install_dream_sparse_attention).parameters:
    raise SystemExit("install_dream_sparse_attention missing swa_local_window")
if "encode_fn" not in signature(synthesize_one_token_key_value).parameters:
    raise SystemExit("synthesize_one_token_key_value missing encode_fn (KEY= VALUE= template)")
fields = getattr(FullTextComplementaryCollator, "__dataclass_fields__", {})
if "label_on_mask" not in fields:
    raise SystemExit("collator missing label_on_mask")
if FullTextComplementaryCollator(mask_token_id=1, pad_token_id=0, eos_token_id=2, lmk_token_id=3, distant_infill=True, distant_only=True).label_on_mask:
    raise SystemExit("distant_only must keep predictor-1 labels")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if not bool(cfg.get("hils_force_remote_unit", False)):
    raise SystemExit("hils_force_remote_unit must be true")
if not bool(cfg.get("hils_force_remote_oracle_route", False)):
    raise SystemExit("hils_force_remote_oracle_route must be true")
if float(cfg.get("hils_gate_bce_weight", 0)) <= 0:
    raise SystemExit("hils_gate_bce_weight must be positive")
if int(cfg.get("hils_distant_min_gap", 0)) < int(cfg.get("swa_local_window", 0)):
    raise SystemExit("gap must be >= SWA window")
if cfg.get("hils_trainable_scope") != "qcal_lmk":
    raise SystemExit("must freeze LoRA with qcal_lmk")
if bool(cfg.get("hils_detach_fusion_weights", True)):
    raise SystemExit("gate BCE keeps live fusion")
if bool(cfg.get("hils_balanced_view_ce") or cfg.get("hils_asymmetric_gate_ce")):
    raise SystemExit("do not stack copy views")
if float(cfg.get("ruler_mix_ratio", 1)) != 0.0:
    raise SystemExit("ruler_mix_ratio must be 0")
if abs(float(cfg.get("hils_dense_teacher_weight", -1))) > 1e-12:
    raise SystemExit("chunk teacher must be 0")
if cfg.get("initialize_from") != "$DENSE":
    raise SystemExit("must initialize from dense-500")
if int(cfg.get("max_steps", 0)) != 200:
    raise SystemExit("max_steps=200 required")
if cfg.get("output_dir") != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
print("preflight_ok", "gate_bce_oracle", fusion_gate_bce.__name__)
PY

if [[ "${FRESH:-1}" == "1" ]]; then
  echo "fresh start from dense-500, no resume. output=$OUTPUT"
  start_args=()
else
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
fi
echo "nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH tmpdir=$TMPDIR log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
