#!/usr/bin/env bash
# Jingneng GPU: s2-sync 1000-step recipe + residual-random-lowrank-rmsnorm-v1 Q-Cal.
# Same 1:1 Dolma+RULER, live fusion, lr 1e-4 (Q-Cal included), warmup 50, cosine 1000.
# Init dense-yarn8-16k-step500. Do not resume B5 / old Q-Cal ckpts.
# Extra vs s2-sync: Q-Cal group clip 1.0 (not bound to LMK/LoRA global clip).
# Do not launch from the notebook. Do not overlap value-fusion on the same GPUs.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_qcal_rmsnorm_rand_s1000.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-qcal-rmsnorm-rand-s1000}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-s2-qcal-rmsnorm-rand-s1000-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/qcal.py" ]] || {
  echo "missing qcal.py on FULLTEACHER_ROOT" >&2
  exit 1
}
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
import dream_dllm_hils.qcal as qcal
import json

qcal_src = Path(qcal.__file__).read_text()
tft_src = Path("$FULLTEACHER_ROOT/dream_dllm_hils/train_fulltext.py").read_text()
attn_src = Path("$FULLTEACHER_ROOT/dream_dllm_hils/attention.py").read_text()
if "QCalRMSNorm" not in qcal_src:
    raise SystemExit("FULLTEACHER qcal.py missing QCalRMSNorm")
if 'hils_qcal_version = "residual-random-lowrank-rmsnorm-v1"' not in qcal_src:
    raise SystemExit("FULLTEACHER qcal.py must set residual-random-lowrank-rmsnorm-v1")
if "Pinned for value-fusion resume" in qcal_src or "nn.init.zeros_" in qcal_src:
    raise SystemExit("FULLTEACHER qcal.py is the old VF overlay; restore RMSNorm qcal.py")
if "residual-zero-up-native-scale-v1" in qcal_src:
    raise SystemExit("FULLTEACHER qcal.py still mentions old Q-Cal")
if "residual-zero-up-native-scale-v1" in tft_src:
    raise SystemExit("FULLTEACHER train_fulltext.py still pins old Q-Cal")
if '"hils_qcal_version": "residual-random-lowrank-rmsnorm-v1"' not in tft_src:
    raise SystemExit("FULLTEACHER train_fulltext.py must pin residual-random-lowrank-rmsnorm-v1")
if "qcal_norm" not in attn_src:
    raise SystemExit("attention.py missing score-query RMSNorm")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "swa_local_window" not in signature(install_dream_sparse_attention).parameters:
    raise SystemExit("install_dream_sparse_attention missing swa_local_window")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type") or "") != "yarn" or abs(float(rope.get("factor") or 0) - 8.0) > 1e-12:
    raise SystemExit("must stay YaRN factor=8")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-yarn8-16k-step500 (do not resume B5)")
if "ablate-b5" in str(cfg.get("initialize_from", "")) or "ablate-bp" in str(cfg.get("initialize_from", "")):
    raise SystemExit("do not init from old Q-Cal ablations")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate must be 1e-4")
qcal_lr = float(cfg.get("hils_qcal_lr", 0) or 0) or float(cfg.get("learning_rate", 0))
if abs(qcal_lr - 1e-4) > 1e-12:
    raise SystemExit("Q-Cal lr must be 1e-4 (1x s2-sync)")
if abs(float(cfg.get("hils_qcal_max_grad_norm", 0) or 0) - 1.0) > 1e-12:
    raise SystemExit("hils_qcal_max_grad_norm must be 1.0")
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
if str(cfg.get("hils_trainable_scope", "")) != "lora_qcal_lmk":
    raise SystemExit("scope must stay lora_qcal_lmk")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("live fusion")
if int(cfg.get("hils_allchunk_st_queries", -1)) != 0:
    raise SystemExit("do not enable S4")
print("preflight_ok s2-qcal-rmsnorm-rand-s1000")
PY

source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
start_args=()
if checkpoint_is_complete "$OUTPUT/step-1000"; then
  echo "already complete: $OUTPUT/step-1000" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -n "$resume_from" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "setting=s2-qcal-rmsnorm-rand-s1000 nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
