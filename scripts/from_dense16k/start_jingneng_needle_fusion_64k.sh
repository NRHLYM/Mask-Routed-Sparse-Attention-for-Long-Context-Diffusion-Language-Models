#!/usr/bin/env bash
# 64k goldspan fusion-mass probe: last-layer needle rank + w_needle vs local/other.
# 3 ckpts x SN + MKMQ. Prefill only (no generate). 1 GPU per job.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_needle_fusion_64k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
cd "$NSA_ROOT"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LIMIT="${LIMIT:-100}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
export PROBE_TAG="${PROBE_TAG:-needle_fusion}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} < 1 )); then
  echo "need at least 1 GPU" >&2
  exit 1
fi

jobs=(
  "s2sync:sn"
  "s2allchunk:sn"
  "s2allchunksttemp:sn"
  "s2sync:mkmq"
  "s2allchunk:mkmq"
  "s2allchunksttemp:mkmq"
)
echo "needle-fusion L=$MAX_SEQ_LEN limit=$LIMIT gpus=${GPUS[*]} jobs=${#jobs[@]}"

fail=0
idx=0
while (( idx < ${#jobs[@]} )); do
  pids=()
  for gpu in "${GPUS[@]}"; do
    if (( idx >= ${#jobs[@]} )); then
      break
    fi
    model="${jobs[$idx]%%:*}"
    task="${jobs[$idx]#*:}"
    echo "launch gpu=$gpu $model $task"
    CUDA_VISIBLE_DEVICES="$gpu" TASK="$task" \
      bash "$HERE/start_jingneng_needle_in_topk.sh" "$model" &
    pids+=($!)
    idx=$((idx + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  (( fail == 0 )) || { echo "needle-fusion worker failed" >&2; exit 1; }
done
echo "NEEDLE_FUSION_64K_DONE"
