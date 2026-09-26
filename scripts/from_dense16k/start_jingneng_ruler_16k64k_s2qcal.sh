#!/usr/bin/env bash
# Official goldspan RULER for HiLS s2 Q-Cal RMSNorm+random step-1000.
# SN / MK-MQ / VT. 16k YaRN 8 on GPUs 0,1; 64k factor=32 on GPUs 2,3.
# Live FULLTEACHER is RMSNorm. Old zero-up Q-Cal stays only in
# overlays/value-fusion-old-qcal for VF-250 resume.
# Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_16k64k_s2qcal.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000"
[[ -s "$CKPT/trainable_state.pt" ]] || { echo "missing $CKPT" >&2; exit 1; }
[[ -s "$CKPT/checkpoint_manifest.json" ]] || { echo "incomplete $CKPT" >&2; exit 1; }

stage_qcal_rmsnorm_tree() {
  local tree="/Data/xiongjing/src/eval-trees/hils-s2-qcal-rmsnorm"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local name
  [[ -f "$src/qcal.py" ]] || { echo "missing $src/qcal.py" >&2; exit 1; }
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
  cp -f "$src/qcal.py" "$tree/dream_dllm_hils/qcal.py"
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  python3 - "$tree/dream_dllm_hils/qcal.py" "$tree/dream_dllm_hils/train_fulltext.py" "$tree/dream_dllm_hils/attention.py" <<'PY'
from pathlib import Path
import sys
want = "residual-random-lowrank-rmsnorm-v1"
old = "residual-zero-up-native-scale-v1"
qcal, tft, attn = map(Path, sys.argv[1:4])
qtext = qcal.read_text()
if want not in qtext:
    raise SystemExit(f"{qcal} missing {want}")
ttext = tft.read_text()
if want in ttext and old not in ttext:
    print("qcal_version already pinned", want, file=sys.stderr)
elif old in ttext:
    tft.write_text(ttext.replace(old, want))
    print("pinned qcal_version", want, file=sys.stderr)
else:
    raise SystemExit(f"{tft} missing {old} and {want}")
if "qcal_norm" not in attn.read_text():
    raise SystemExit(f"{attn} missing qcal_norm")
print("preflight_ok s2qcal ruler RMSNorm tree", file=sys.stderr)
PY
  printf '%s\n' "$tree"
}

export QCAL_RMSNORM_CODE_ROOT="$(stage_qcal_rmsnorm_tree)"
[[ -d "$QCAL_RMSNORM_CODE_ROOT" ]] || {
  echo "bad QCAL_RMSNORM_CODE_ROOT=$QCAL_RMSNORM_CODE_ROOT" >&2
  exit 1
}
export PYTHONPATH="$QCAL_RMSNORM_CODE_ROOT"
export RULER_OUTPUT_SUBDIR="${RULER_OUTPUT_SUBDIR:-ruler_probes_goldspan}"
echo "s2qcal RULER code_root=$QCAL_RMSNORM_CODE_ROOT"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2qcal RULER split 16k=${GPUS[0]},${GPUS[1]} 64k=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" bash "$HERE/start_jingneng_ruler_16k.sh" s2qcal &
  p16=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" bash "$HERE/start_jingneng_ruler_64k.sh" s2qcal &
  p64=$!
  fail=0
  wait "$p16" || fail=1
  wait "$p64" || fail=1
  (( fail == 0 )) || { echo "s2qcal RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  bash "$HERE/start_jingneng_ruler_16k.sh" s2qcal
  bash "$HERE/start_jingneng_ruler_64k.sh" s2qcal
fi
echo "s2qcal_RULER_16K64K_DONE"
