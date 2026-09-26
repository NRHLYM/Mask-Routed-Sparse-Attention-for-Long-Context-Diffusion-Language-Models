#!/usr/bin/env bash
# Jingneng GPU: 3 SWA + 1 dense, interleave=4, 1:1 Dolma+RULER, 1000 steps.
# Same 21 SWA @ radius 1280 (skip_inert=False) as NSA/DSA; sparse slots are
# full dense instead of NSA. Isolates the 21 SWA layers vs the sparse branch.
# Init dense-yarn8-16k-step500. PYTHONPATH = staged HILS_FT_ROOT + this
# train_fulltext.py. Do not launch from the notebook. Do not overwrite live
# HILS_FT_ROOT.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_swa3_dense1_dolmaruler_sync_s1000.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-swa3-dense1-s1000}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
NPROC="${NPROC:-4}"

CONFIG="$NSA_ROOT/configs/from_dense16k/swa3-dense1-i4-w1280-dolmaruler-sync-s1000-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
PATCH="$NSA_ROOT/dream_dllm_hils/train_fulltext.py"
TREE="/Data/xiongjing/src/eval-trees/swa3-dense1-i4-s1000"
LOG="$ROOT/logs/swa3-dense1-dolmaruler-sync-s1000-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -d "$HILS_FT_ROOT/dream_dllm_hils" && -d "$HILS_FT_ROOT/ops" ]] || {
  echo "missing HILS_FT_ROOT $HILS_FT_ROOT" >&2
  exit 1
}
[[ -f "$PATCH" ]] || { echo "missing $PATCH" >&2; exit 1; }
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

stage_tree() {
  local src="$HILS_FT_ROOT/dream_dllm_hils"
  local name
  rm -rf "$TREE"
  mkdir -p "$TREE/dream_dllm_hils"
  ln -sfn "$HILS_FT_ROOT/ops" "$TREE/ops"
  ln -sfn "$HILS_FT_ROOT/scripts" "$TREE/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    [[ "$(basename "$name")" == train_fulltext.py ]] && continue
    ln -sfn "$name" "$TREE/dream_dllm_hils/$(basename "$name")"
  done
  # cp -f onto an existing symlink follows it and overwrites HILS_FT_ROOT.
  rm -f "$TREE/dream_dllm_hils/train_fulltext.py"
  cp -a "$PATCH" "$TREE/dream_dllm_hils/train_fulltext.py"
  python - <<PY
from pathlib import Path
p = Path("$TREE/dream_dllm_hils/train_fulltext.py")
text = p.read_text(encoding="utf-8")
if "def interleaved_swa_dense" not in text:
    raise SystemExit(f"staged train_fulltext missing interleaved_swa_dense: {p}")
if p.is_symlink():
    raise SystemExit(f"train_fulltext.py must be a real file, not symlink: {p}")
print("staged", p)
PY
}
stage_tree
export PYTHONPATH="$TREE"
cd "$TREE"

python - <<PY
from pathlib import Path
from inspect import signature
import sys
sys.path.insert(0, str(Path("$TREE").resolve()))
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.train_fulltext import (
    interleaved_swa_dense,
    validate_training_config,
)
from dream_dllm_hils.data import FullTextComplementaryCollator
import dream_dllm_hils.train_fulltext as train_mod
import json
import argparse

root = Path("$TREE").resolve()
train_path = Path(train_mod.__file__).resolve()
if train_path != (root / "dream_dllm_hils" / "train_fulltext.py").resolve():
    raise SystemExit(f"must import staged train_fulltext, got {train_path}")
if "interleaved_swa_dense" not in train_path.read_text(encoding="utf-8"):
    raise SystemExit(f"staged train_fulltext missing hybrid wrap: {train_path}")
if "ruler_every_step" not in FullTextComplementaryCollator.__dataclass_fields__:
    raise SystemExit("collator missing ruler_every_step")
p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
ns = argparse.Namespace(**cfg)
if not interleaved_swa_dense(ns):
    raise SystemExit("need attention_mode=dense and non_hils_attention=sliding")
rope = cfg.get("model_rope_scaling") or {}
if str(rope.get("rope_type") or "") != "yarn" or abs(float(rope.get("factor") or 0) - 8.0) > 1e-12:
    raise SystemExit("must stay YaRN factor=8")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
if str(cfg.get("initialize_from", "")) != "$DENSE":
    raise SystemExit("must initialize from dense-yarn8-16k-step500")
if int(cfg.get("hils_interleave", 0)) != 4:
    raise SystemExit("hils_interleave=4 (3 SWA + 1 dense)")
if int(cfg.get("swa_local_window", 0)) != 1280:
    raise SystemExit("21 SWA layers swa_local_window=1280")
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
print("preflight_ok swa3-dense1-i4-dolmaruler-sync-s1000", "code_root", root)
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
echo "setting=swa3-dense1-dolmaruler-sync-s1000 nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES log=$LOG ${start_args[*]:-}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
