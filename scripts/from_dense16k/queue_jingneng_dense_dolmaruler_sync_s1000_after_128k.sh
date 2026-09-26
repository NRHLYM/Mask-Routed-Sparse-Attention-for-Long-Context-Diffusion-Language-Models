#!/usr/bin/env bash
# Wait for the current 128k NSA/SWA/s2 RULER pipeline to leave the 4 GPUs,
# then start dense 16k 1:1 1000-step training (no sparse).
# Safe to run now in the GPU job: this process only polls until eval exits.
# Do not launch from the notebook.
#
#   source /Data/xiongjing/env.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     "$NSA_ROOT/scripts/from_dense16k/queue_jingneng_dense_dolmaruler_sync_s1000_after_128k.sh"
set -euo pipefail
source /Data/xiongjing/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="$HERE/start_jingneng_dense_dolmaruler_sync_s1000.sh"
[[ -f "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 1; }

wait_128k() {
  echo "waiting for 128k RULER pipeline to finish (poll 60s)"
  while true; do
    local latest evals
    latest="$(ls -t "$ROOT/logs"/ruler-128k-nsa-swa-s2-pipe-*.log 2>/dev/null | head -1 || true)"
    evals="$(pgrep -f eval_jingneng_official_ruler.py || true)"
    if [[ -n "$latest" ]] && grep -qE 'NSA_SWA_S2_RULER_128K_PIPELINE_DONE|PIPELINE_FAILS' "$latest"; then
      if [[ -z "$evals" ]]; then
        echo "128k eval gone: $latest"
        grep -E 'NSA_SWA_S2_RULER_128K_PIPELINE_DONE|PIPELINE_FAILS' "$latest" | tail -3
        return 0
      fi
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) still waiting latest=${latest:-none} eval_pids=${evals:-none}"
    sleep 60
  done
}

wait_128k
exec bash "$TRAIN"
