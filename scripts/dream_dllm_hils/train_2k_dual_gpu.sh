#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG [TRAINER_ARGS...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/ma-user/work/venvs/d2f/bin/python"
CONFIG="$1"
shift

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext \
  --config "$CONFIG" \
  "$@"
