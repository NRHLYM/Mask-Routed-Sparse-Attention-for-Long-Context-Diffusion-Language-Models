#!/usr/bin/env bash
# Jingneng GPU: S2-like HiLS with asymmetric gate CE.
# Local Dolma complementary CE detaches fusion; every-step Dolma span-copy
# CE trains the gate. Independent of S2 / paper-1000 / support-KL.
# 4 GPUs. From dense-500. ruler_mix=0.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-asymmetric-gate}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-asymmetric-gate-ce-jingneng.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-asymmetric-gate-ce-s1000"
LOG="$ROOT/logs/hils-asymmetric-gate-ce-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/data.py" ]] || {
  echo "missing data.py in FULLTEACHER_ROOT" >&2
  exit 1
}
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done

python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import install_dream_sparse_attention, set_hils_fusion_detach
from dream_dllm_hils.routing import route_topk_g7
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
    raise SystemExit("asymmetric arm installs live fusion; detach only on local CE")
if not bool(cfg.get("hils_asymmetric_gate_ce", False)):
    raise SystemExit("hils_asymmetric_gate_ce must be true")
if float(cfg.get("ruler_mix_ratio", 1)) != 0.0:
    raise SystemExit("ruler_mix_ratio must be 0; distant view is every step")
if abs(float(cfg.get("hils_dense_teacher_weight", -1))) > 1e-12:
    raise SystemExit("chunk teacher must be 0")
if abs(float(cfg.get("hils_support_attn_kl_weight", 0) or 0)) > 1e-12:
    raise SystemExit("do not stack support-KL on this arm")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000 required")
if int(cfg.get("warmup_steps", 0)) != 50 or int(cfg.get("ruler_val_batches", 0)) != 8:
    raise SystemExit("warmup_steps=50 ruler_val_batches=8 required")
if cfg.get("hils_trainable_scope") != "lora_qcal_lmk":
    raise SystemExit("trainable_scope must be lora_qcal_lmk")
print(
    "preflight_ok",
    "asymmetric_gate_ce",
    route_topk_g7.__name__,
    set_hils_fusion_detach.__name__,
    synthesize_distant_span_copy.__name__,
)
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
