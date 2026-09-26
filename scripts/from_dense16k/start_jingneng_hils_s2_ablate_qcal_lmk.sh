#!/usr/bin/env bash
# Jingneng GPU: S2 paired ablations vs existing official s1000 (arm A, do not rerun).
# Same cosine as A: max_steps=1000, warmup=50, lr=1e-4, live fusion, 1:1.
# B/C/B5 stop at 400. Bp continues the same 1000 cosine from step-400 after the diagnostic window.
#
#   B  = freeze Q-Cal (identity residual, still installed); keep mask_type LMK
#   C  = keep Q-Cal; landmark slots stay, no mask_type embed (scope lora_qcal)
#   Bp = Q-Cal trainable; 0.1x Q-Cal LR (1e-5) + Q-Cal clip 1.0; keep type embed
#   B5 = Q-Cal trainable; 0.5x Q-Cal LR (5e-5) + Q-Cal clip 1.0; keep type embed
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_ablate_qcal_lmk.sh B
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_ablate_qcal_lmk.sh C
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_ablate_qcal_lmk.sh Bp
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_ablate_qcal_lmk.sh B5
set -euo pipefail
ARM="${1:?usage: $0 B|C|Bp|B5}"
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc

case "$ARM" in
  B|b)
    ARM=B
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-b-freeze-qcal-s1000-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-ablate-b-freeze-qcal-s1000"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-ablate-b}"
    LOG="$ROOT/logs/hils-s2-ablate-b-freeze-qcal-$(date +%Y%m%d-%H%M%S).log"
    ;;
  C|c)
    ARM=C
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-c-nomasktype-s1000-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-ablate-c-nomasktype-s1000"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-ablate-c}"
    LOG="$ROOT/logs/hils-s2-ablate-c-nomasktype-$(date +%Y%m%d-%H%M%S).log"
    ;;
  Bp|BP|bp)
    ARM=Bp
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-bp-qcal-lr0p1-s1000-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-ablate-bp-qcal-lr0p1-s1000"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-ablate-bp}"
    LOG="$ROOT/logs/hils-s2-ablate-bp-qcal-lr0p1-$(date +%Y%m%d-%H%M%S).log"
    ;;
  B5|b5)
    ARM=B5
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-b5-qcal-lr0p5-s1000-jingneng.json"
    OUTPUT="/Data/xiongjing/outputs/hils-s2-ablate-b5-qcal-lr0p5-s1000"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-ablate-b5}"
    LOG="$ROOT/logs/hils-s2-ablate-b5-qcal-lr0p5-$(date +%Y%m%d-%H%M%S).log"
    ;;
  *)
    echo "usage: $0 B|C|Bp|B5" >&2
    exit 1
    ;;
esac

export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
mkdir -p "$OUTPUT" "$ROOT/logs"
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
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import json

arm = "$ARM"
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
import dream_dllm_hils.train_fulltext as tft
if "hils_freeze_qcal" not in tft.DEFAULTS:
    raise SystemExit("PVC train_fulltext.py is stale; scp freeze_qcal support first")
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type") or "") != "yarn" or abs(float(rope.get("factor") or 0) - 8.0) > 1e-12:
    raise SystemExit("must stay YaRN factor=8")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-yarn8-16k-step500")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000 (cosine length; do not shrink)")
if arm in ("Bp", "B5"):
    if int(cfg.get("stop_after_steps") or 0) not in (0, 1000):
        raise SystemExit(f"arm {arm} continues through max_steps=1000")
else:
    if int(cfg.get("stop_after_steps") or 0) != 400:
        raise SystemExit("stop_after_steps=400")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate must be 1e-4")
if str(cfg.get("lr_schedule", "cosine")) != "cosine":
    raise SystemExit("lr_schedule must be cosine over 1000 steps")
if int(cfg.get("warmup_steps", 0)) != 50:
    raise SystemExit("warmup_steps=50")
if int(cfg.get("hils_topk", 0)) != 32 or int(cfg.get("chunk_size", 0)) != 64:
    raise SystemExit("query budget: hils_topk=32 chunk_size=64")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers must set swa_local_window=1280")
if int(cfg.get("hils_interleave", 0)) != 4:
    raise SystemExit("hils_interleave=4")
if abs(float(cfg.get("ruler_mix_ratio", 0))) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("keep live fusion")
if int(cfg.get("hils_qcal_rank", 0)) != 64:
    raise SystemExit("keep qcal rank 64 installed")
if arm == "B":
    if not bool(cfg.get("hils_freeze_qcal", False)):
        raise SystemExit("arm B must freeze Q-Cal")
    if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
        raise SystemExit("arm B keeps lora_qcal_lmk")
    if str(cfg.get("lmk_token_mode", "")) != "mask_type":
        raise SystemExit("arm B keeps mask_type")
elif arm == "C":
    if bool(cfg.get("hils_freeze_qcal", False)):
        raise SystemExit("arm C trains Q-Cal")
    if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal":
        raise SystemExit("arm C is lora_qcal (no type embed)")
    if str(cfg.get("lmk_token_mode", "")) != "mask":
        raise SystemExit("arm C drops mask_type; landmark MASK slots stay")
elif arm == "Bp":
    if bool(cfg.get("hils_freeze_qcal", False)):
        raise SystemExit("arm Bp must train Q-Cal")
    if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
        raise SystemExit("arm Bp keeps lora_qcal_lmk")
    if str(cfg.get("lmk_token_mode", "")) != "mask_type":
        raise SystemExit("arm Bp keeps mask_type")
    if abs(float(cfg.get("hils_qcal_lr", 0) or 0) - 1e-5) > 1e-12:
        raise SystemExit("arm Bp Q-Cal lr must be 1e-5 (0.1x)")
    if abs(float(cfg.get("hils_qcal_max_grad_norm", 0) or 0) - 1.0) > 1e-12:
        raise SystemExit("arm Bp Q-Cal clip must be 1.0")
elif arm == "B5":
    if bool(cfg.get("hils_freeze_qcal", False)):
        raise SystemExit("arm B5 must train Q-Cal")
    if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
        raise SystemExit("arm B5 keeps lora_qcal_lmk")
    if str(cfg.get("lmk_token_mode", "")) != "mask_type":
        raise SystemExit("arm B5 keeps mask_type")
    if abs(float(cfg.get("hils_qcal_lr", 0) or 0) - 5e-5) > 1e-12:
        raise SystemExit("arm B5 Q-Cal lr must be 5e-5 (0.5x)")
    if abs(float(cfg.get("hils_qcal_max_grad_norm", 0) or 0) - 1.0) > 1e-12:
        raise SystemExit("arm B5 Q-Cal clip must be 1.0")
print(f"preflight_ok s2-ablate-{arm}")
PY

source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
start_args=()
DONE_STEP=1000
if [[ "$ARM" != "Bp" && "$ARM" != "B5" ]]; then
  DONE_STEP=400
fi
if checkpoint_is_complete "$OUTPUT/step-$DONE_STEP"; then
  echo "already complete: $OUTPUT/step-$DONE_STEP" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -n "$resume_from" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "setting=s2-ablate-$ARM nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
