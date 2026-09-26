#!/usr/bin/env bash
# Run this on the Jingneng GPU job (not the notebook login). Start/stop yourself.
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"

CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-frozendense-qcal-mass.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-qcalonly-frozendense-attnmass-cedetach"
LOG="$ROOT/logs/hils-qcal-attnmass-$(date +%Y%m%d-%H%M%S).log"
NPROC="${NPROC:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f /Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc/train.bin ]] || {
  echo "missing 16k packs" >&2
  exit 1
}

start_args=()
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${resume_from}" && -f "${resume_from}/trainable_state.pt" ]]; then
  start_args=(--resume_from "$resume_from")
fi

echo "nproc=$NPROC log=$LOG config=$CONFIG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
