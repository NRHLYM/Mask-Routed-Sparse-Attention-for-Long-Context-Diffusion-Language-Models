#!/usr/bin/env bash
# Jingneng GPU: resume s2-value-fusion (beta=0.3) from step-250 with the OLD
# Q-Cal contract residual-zero-up-native-scale-v1.
# Uses a private code tree so FULLTEACHER_ROOT (new RMSNorm Q-Cal 1000-step) is
# not modified. Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_value_fusion.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-s2-value-fusion}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
NPROC="${NPROC:-4}"

stage_value_fusion_old_qcal_tree() {
  local tree="/Data/xiongjing/src/eval-trees/hils-s2-value-fusion-old-qcal"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local overlay="$NSA_ROOT/overlays/value-fusion-old-qcal/qcal.py"
  local name
  [[ -f "$overlay" ]] || { echo "missing old Q-Cal overlay: $overlay" >&2; exit 1; }
  rm -rf "$tree"
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  # Freeze copies so the live FULLTEACHER_ROOT Q-Cal rewrite cannot change this job.
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  cp -f "$overlay" "$tree/dream_dllm_hils/qcal.py"
  python3 - "$tree/dream_dllm_hils/train_fulltext.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
old = "residual-random-lowrank-rmsnorm-v1"
new = "residual-zero-up-native-scale-v1"
if old not in text:
    raise SystemExit(f"{path} missing {old}; cannot pin old Q-Cal resume contract")
path.write_text(text.replace(old, new))
PY
  echo "$tree"
}

CODE_ROOT="$(stage_value_fusion_old_qcal_tree)"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-value-fusion-beta0p3-s500-jingneng.json"
OUTPUT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500"
S2="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
LOG="$ROOT/logs/hils-s2-value-fusion-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$S2/trainable_state.pt" ]] || { echo "missing s2 ckpt: $S2" >&2; exit 1; }
[[ -f "$CODE_ROOT/dream_dllm_hils/value_aware_fusion.py" ]] || {
  echo "missing value_aware_fusion.py on value-fusion tree" >&2
  exit 1
}
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete packs: $PACK/$f" >&2; exit 1; }
done

python3 - <<PY
import json
from pathlib import Path
import dream_dllm_hils.qcal as qcal
qcal_src = Path(qcal.__file__).read_text()
if "residual-zero-up-native-scale-v1" not in qcal_src:
    raise SystemExit("value-fusion tree qcal.py must be residual-zero-up-native-scale-v1")
if "nn.init.zeros_" not in qcal_src:
    raise SystemExit("value-fusion tree qcal.py must zero-up")
if "residual-random-lowrank-rmsnorm-v1" in qcal_src:
    raise SystemExit("value-fusion tree still has new Q-Cal")
tf = Path("$CODE_ROOT/dream_dllm_hils/train_fulltext.py").read_text()
if "residual-zero-up-native-scale-v1" not in tf:
    raise SystemExit("train_fulltext resume contract not pinned to old Q-Cal")
if "residual-random-lowrank-rmsnorm-v1" in tf:
    raise SystemExit("train_fulltext still expects new Q-Cal")
cfg = json.loads(Path("$CONFIG").read_text())
if abs(float(cfg.get("hils_value_fusion_beta", 0)) - 0.3) > 1e-12:
    raise SystemExit("hils_value_fusion_beta must be 0.3")
qcal_lr = float(cfg.get("hils_qcal_lr", 0) or 0) or float(cfg.get("learning_rate", 0))
if abs(qcal_lr - 1e-4) > 1e-12:
    raise SystemExit("Q-Cal lr must match s2 1e-4 for this run")
if str(cfg.get("initialize_from", "")) != "$S2":
    raise SystemExit("must initialize from s2-sync step-500")
if cfg.get("hils_detach_fusion_weights", True):
    raise SystemExit("live fusion required")
if str(cfg.get("output_dir", "")) != "$OUTPUT":
    raise SystemExit("output_dir mismatch")
print("preflight_ok s2-value-fusion-beta0p3 old-qcal-tree")
PY

source "$NSA_ROOT/scripts/from_dense16k/select_latest_complete_checkpoint.sh"
start_args=()
if checkpoint_is_complete "$OUTPUT/step-500"; then
  echo "already complete: $OUTPUT/step-500" >&2
  exit 0
fi
resume_from="$(select_latest_complete_checkpoint "$OUTPUT")"
if [[ -z "$resume_from" ]]; then
  echo "expected complete ckpt under $OUTPUT (step-250)" >&2
  exit 1
fi
start_args=(--resume_from "$resume_from")
echo "setting=s2-value-fusion nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES code_root=$CODE_ROOT log=$LOG ${start_args[*]}"
nohup python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  >"$LOG" 2>&1 </dev/null &
echo $! > "$OUTPUT/train.pid"
disown || true
echo "started pid=$(cat "$OUTPUT/train.pid") log=$LOG"
echo "follow with: tail -f $LOG"
