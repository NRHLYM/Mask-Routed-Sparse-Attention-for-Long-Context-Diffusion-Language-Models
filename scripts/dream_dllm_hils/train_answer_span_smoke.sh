#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/venvs/sparsed-py310/bin/python}"

"$PYTHON_BIN" -m dream_dllm_hils.smoke_test

# A real 7B smoke can be launched after PEFT is installed in the environment:
# "$PYTHON_BIN" -m dream_dllm_hils.train_answer_span \
#   --config configs/dream_dllm_hils/first_v1.json \
#   --max_steps 2 \
#   --synthetic_size 8 \
#   --max_length 256 \
#   --gradient_accumulation_steps 1 \
#   --save_steps 0
