#!/usr/bin/env bash
ROOT=/mnt/Data/xiongjing/dense-yarn16-32k-20260919
PYTHON=/mnt/Data/xiongjing/hils-gumbel-all-20260917/venv/bin/python
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/hf-cache"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
mkdir -p "$TMPDIR" "$HF_HOME" "$ROOT/logs" "$ROOT/outputs"
cd "$ROOT"
