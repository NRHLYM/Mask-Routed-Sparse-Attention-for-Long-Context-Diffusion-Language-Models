#!/usr/bin/env bash
# Official goldspan RULER for s2-qcal RMSNorm s1000 at 32k, both YaRN modes:
#   L/2048 (factor=16) -> ruler_probes_goldspan
#   keep-train factor=8 -> ruler_probes_goldspan_yarn8
# 4 GPUs: 0,1 unfixed; 2,3 fixed. Do not launch from the notebook.
# Do not overlap the from-dense VF 1000-step train on the same GPUs.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_32k_s2qcal.sh
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
export RULER_LENGTHS=32768
export RULER_TASKS="hils_sn hils_mkmq hils_vt"
export RULER_DATA="${RULER_DATA:-/Data/xiongjing/data/ruler-probes-goldspan}"
echo "s2qcal 32k RULER code_root=$QCAL_RMSNORM_CODE_ROOT"

run_unfixed() {
  unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR
  export RULER_OUTPUT_SUBDIR=ruler_probes_goldspan
  export INDUCTOR_LOCAL=/tmp/xiongjing-inductor-s2qcal-32k-L2048
  bash "$HERE/start_jingneng_ruler_2gpu.sh" s2qcal
}

run_fixed() {
  export RULER_KEEP_TRAIN_YARN=1
  export RULER_YARN_FACTOR=8
  export RULER_OUTPUT_SUBDIR=ruler_probes_goldspan_yarn8
  export INDUCTOR_LOCAL=/tmp/xiongjing-inductor-s2qcal-32k-yarn8
  bash "$HERE/start_jingneng_ruler_2gpu.sh" s2qcal
}

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} >= 4 )); then
  echo "s2qcal 32k split L/2048=${GPUS[0]},${GPUS[1]} yarn8=${GPUS[2]},${GPUS[3]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" run_unfixed &
  p_l=$!
  CUDA_VISIBLE_DEVICES="${GPUS[2]},${GPUS[3]}" run_fixed &
  p_8=$!
  fail=0
  wait "$p_l" || fail=1
  wait "$p_8" || fail=1
  (( fail == 0 )) || { echo "s2qcal 32k RULER worker failed" >&2; exit 1; }
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export CUDA_VISIBLE_DEVICES
  run_unfixed
  run_fixed
fi
echo "s2qcal_RULER_32K_L2048_AND_YARN8_DONE"
