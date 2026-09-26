#!/usr/bin/env bash
# Compare 33-way remote mass: s2-sync step-500 vs value-fusion step-250.
# SN 16k goldspan, answer/predictor slots. Old Q-Cal tree only (do not
# touch FULLTEACHER_ROOT used by the RMSNorm Q-Cal 1000-step job).
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_value_fusion_wremote.sh
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LIMIT="${LIMIT:-32}"

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
text = path.read_text()
path.write_text(text.replace("residual-random-lowrank-rmsnorm-v1", "residual-zero-up-native-scale-v1"))
PY
  echo "$tree"
}

CODE_ROOT="$(stage_old_qcal_tree)"
export PYTHONPATH="$CODE_ROOT"
export VALUE_FUSION_CODE_ROOT="$CODE_ROOT"
cd "$CODE_ROOT"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if (( ${#GPUS[@]} < 1 )); then
  echo "need a GPU" >&2
  exit 1
fi

echo "wremote probe tree=$CODE_ROOT limit=$LIMIT gpus=${GPUS[*]}"
fail=0
if (( ${#GPUS[@]} >= 2 )); then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$HERE/start_jingneng_fusion_gate.sh" s2sync &
  p0=$!
  CUDA_VISIBLE_DEVICES="${GPUS[1]}" bash "$HERE/start_jingneng_fusion_gate.sh" s2vf &
  p1=$!
  wait "$p0" || fail=1
  wait "$p1" || fail=1
else
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$HERE/start_jingneng_fusion_gate.sh" s2sync || fail=1
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" bash "$HERE/start_jingneng_fusion_gate.sh" s2vf || fail=1
fi
(( fail == 0 )) || { echo "wremote probe failed" >&2; exit 1; }
python3 - <<'PY'
import json
from pathlib import Path
pairs = [
    ("s2-sync-500", Path("/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/fusion_wremote_sn16k_step500.json")),
    ("value-fusion-250", Path("/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/fusion_wremote_sn16k_step250.json")),
]
print("===== 33-way remote weight (SN 16k, 1-local_weight) =====")
for name, path in pairs:
    d = json.loads(path.read_text())["summary"]
    layers = d["per_layer"]
    print(name, "mean_w_local", round(d["mean_local_weight"], 4),
          "mean_w_remote", round(1.0 - d["mean_local_weight"], 4),
          "mean_w_needle_chunk", round(d["mean_needle_chunk_remote_weight"], 4))
    for lid in d["hils_layers"]:
        loc = layers[str(lid)]["local_weight"]
        print(f"  L{lid} w_local={loc:.4f} w_remote={1-loc:.4f} w_needle={layers[str(lid)]['needle_chunk_remote_weight']:.4f}")
PY
echo "VALUE_FUSION_WREMOTE_PROBE_DONE"
