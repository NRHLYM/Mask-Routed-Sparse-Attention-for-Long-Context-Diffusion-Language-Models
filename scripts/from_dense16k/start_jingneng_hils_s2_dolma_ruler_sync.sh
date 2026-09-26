#!/usr/bin/env bash
# Jingneng GPU: S2 diagnostic, every step Dolma complementary + RULER answer CE.
# Loss = 0.5 mean(Dolma CE) + 0.5 mean(RULER CE). Live fusion. Not paper mainline.
# Log goes to a file only; do not tee to a web terminal.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-dolmaruler-sync}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-s2-dolmaruler-sync-$(date +%Y%m%d-%H%M%S).log"
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
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir must be the dolma-ruler-sync dir")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-500")
if int(cfg.get("max_steps", 0)) != 500:
    raise SystemExit("max_steps=500")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate must be 1e-4")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("lr_schedule must stay cosine over 500 steps")
if int(cfg.get("warmup_steps", 0)) != 25:
    raise SystemExit("warmup_steps=25")
if abs(float(cfg.get("ruler_mix_ratio", 0))) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0; RULER is a fixed every-step view")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
if abs(float(cfg.get("ruler_answer_ce_weight", 1)) - 1.0) > 1e-12:
    raise SystemExit("ruler_answer_ce_weight must stay 1")
if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
    raise SystemExit("scope must stay lora_qcal_lmk")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("this diagnostic must keep live fusion")
if abs(float(cfg.get("hils_evidence_token_attn_weight", 0) or 0)) > 1e-12:
    raise SystemExit("token attn must be 0")
if bool(cfg.get("hils_force_remote_unit", False)):
    raise SystemExit("force remote must be off")
if bool(cfg.get("hils_lmk_ce_ste", False)):
    raise SystemExit("do not enable S3")
if int(cfg.get("hils_allchunk_st_queries", -1)) != 0:
    raise SystemExit("do not enable S4")
if bool(cfg.get("hils_asymmetric_gate_ce", False)) or bool(cfg.get("hils_balanced_view_ce", False)):
    raise SystemExit("do not stack copy-view CE")
print("preflight_ok s2-dolma-ruler-sync")
PY

# shellcheck source=select_latest_complete_checkpoint.sh
source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"

start_args=()
if checkpoint_is_complete "$OUTPUT/step-500"; then
  echo "already complete: $OUTPUT/step-500" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -n "$resume_from" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "setting=s2-dolma-ruler-sync nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH tmpdir=$TMPDIR log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
