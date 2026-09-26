#!/usr/bin/env bash
# Official goldspan RULER for HiLS s2 value-fusion beta=0.3 step-250.
# SN / MK-MQ / VT. 16k YaRN 8 on GPUs 0,1; 64k factor=32 on GPUs 2,3.
# Uses a private old-Q-Cal tree. Do not overlap the RMSNorm Q-Cal 1000-step job.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2vf.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/step-250"
[[ -s "$CKPT/trainable_state.pt" ]] || { echo "missing $CKPT" >&2; exit 1; }
[[ -s "$CKPT/checkpoint_manifest.json" ]] || { echo "incomplete $CKPT" >&2; exit 1; }

stage_old_qcal_tree() {
  local tree="/Data/xiongjing/src/eval-trees/hils-s2-value-fusion-old-qcal"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local overlay="$NSA_ROOT/overlays/value-fusion-old-qcal/qcal.py"
  local name
  [[ -f "$overlay" ]] || { echo "missing $overlay" >&2; exit 1; }
  rm -rf "$tree"
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  rm -f "$tree/dream_dllm_hils/qcal.py" \
    "$tree/dream_dllm_hils/attention.py" \
    "$tree/dream_dllm_hils/train_fulltext.py"
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  cp -f "$overlay" "$tree/dream_dllm_hils/qcal.py"
  python3 - "$tree/dream_dllm_hils/train_fulltext.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
path.write_text(
    path.read_text().replace(
        "residual-random-lowrank-rmsnorm-v1",
        "residual-zero-up-native-scale-v1",
    )
)
PY
  echo "$tree"
}

export VALUE_FUSION_CODE_ROOT="$(stage_old_qcal_tree)"
export PYTHONPATH="$VALUE_FUSION_CODE_ROOT"
export RULER_OUTPUT_SUBDIR="${RULER_OUTPUT_SUBDIR:-ruler_probes_goldspan}"
echo "s2vf RULER code_root=$VALUE_FUSION_CODE_ROOT"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2vf RULER split 16k=${GPUS[0]},${GPUS[1]} 64k=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" s2vf &
  p16=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k.sh" s2vf &
  p64=$!
  fail=0
  wait "$p16" || fail=1
  wait "$p64" || fail=1
  (( fail == 0 )) || { echo "s2vf RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  bash "$HERE/start_jingneng_ruler_16k.sh" s2vf
  bash "$HERE/start_jingneng_ruler_64k.sh" s2vf
fi
echo "s2vf_RULER_16K64K_DONE"
