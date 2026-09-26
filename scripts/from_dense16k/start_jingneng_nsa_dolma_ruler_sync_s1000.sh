#!/usr/bin/env bash
# Jingneng GPU: official YaRN NSA 1:1, cosine 1000 from dense-yarn8-16k-step500.
# Budget: NSA 32 blocks / compress 64 + 21 SWA radius 1280, skip_inert=False.
# PYTHONPATH=$HILS_FT_ROOT. 4 GPU.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_nsa_dolma_ruler_sync_s1000.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HILS_FT_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsa-dolmaruler-sync-s1000}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$HILS_FT_ROOT"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/nsa-dolmaruler-sync-s1000-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -d "$HILS_FT_ROOT/dream_dllm_hils" && -d "$HILS_FT_ROOT/ops" ]] || {
  echo "missing HILS_FT_ROOT $HILS_FT_ROOT" >&2
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
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.nsa_attention import install_dream_nsa_attention
from dream_dllm_hils.train_fulltext import validate_training_config
from dream_dllm_hils.data import FullTextComplementaryCollator
import dream_dllm_hils
import json, ops

root = Path("$HILS_FT_ROOT").resolve()
pkg = Path(dream_dllm_hils.__file__).resolve().parent
if pkg.parent.name == "fullteacher-20260917":
    raise SystemExit("NSA PYTHONPATH must not be FULLTEACHER_ROOT")
if pkg != root / "dream_dllm_hils":
    raise SystemExit(f"NSA must import HILS_FT_ROOT, got {pkg}")
if "ruler_every_step" not in FullTextComplementaryCollator.__dataclass_fields__:
    raise SystemExit("HILS_FT_ROOT collator missing ruler_every_step")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "swa_local_window" not in signature(install_dream_nsa_attention).parameters:
    raise SystemExit("install_dream_nsa_attention missing swa_local_window")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type") or "") != "yarn" or abs(float(rope.get("factor") or 0) - 8.0) > 1e-12:
    raise SystemExit("official 1000-step run must stay YaRN factor=8")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-500")
# HILS_FT_ROOT parser has no lr_schedule / ruler_answer_ce_weight keys.
if "lr_schedule" in cfg or "ruler_answer_ce_weight" in cfg:
    raise SystemExit("NSA HILS_FT_ROOT rejects lr_schedule and ruler_answer_ce_weight; cosine+weight=1 are hardcoded")
if cfg.get("attention_mode") != "nsa":
    raise SystemExit("attention_mode=nsa")
if int(cfg.get("local_window", 0)) != 128:
    raise SystemExit("NSA sparse branch local_window=128")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers swa_local_window=1280")
if int(cfg.get("nsa_block_count", 0)) != 32:
    raise SystemExit("nsa_block_count=32")
if int(cfg.get("nsa_compress_block", 0)) != 64 or int(cfg.get("nsa_compress_stride", 0)) != 64:
    raise SystemExit("compress_block=stride=64")
if int(cfg.get("max_steps", 0)) != 1000:
    raise SystemExit("max_steps=1000")
if abs(float(cfg.get("learning_rate", 0)) - 1e-4) > 1e-12:
    raise SystemExit("learning_rate=1e-4")
if int(cfg.get("warmup_steps", 0)) != 50:
    raise SystemExit("warmup_steps=50")
if abs(float(cfg.get("ruler_mix_ratio", 0))) > 1e-12:
    raise SystemExit("ruler_mix_ratio must be 0")
if not bool(cfg.get("hils_sync_ruler_ce", False)):
    raise SystemExit("hils_sync_ruler_ce must be true")
import ops.dsa_selected_attention_tilelang  # noqa: F401
print("preflight_ok nsa-dolma-ruler-sync-s1000", "code_root", root)
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
echo "setting=nsa-dolma-ruler-sync-s1000 nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
