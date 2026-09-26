#!/usr/bin/env bash

ROOT=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917
TILELANG_ROOT=/home/guests/zhen/dsa_baselines/tilelang-portable
PYTHON=/home/guests/zhen/dsa_baselines/venvs/dream_dsa/bin/python
MODEL=/home/guests/zhen/jing/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
DATA_ROOT=/home/guests/zhen/jing/ParallelComp_official/datasets/LongBench
PROMPTS=/home/guests/zhen/jing/ParallelComp_official/longbench_config/dataset2prompt_raw.json

export PYTHONPATH="$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
export LD_LIBRARY_PATH="$TILELANG_ROOT/tilelang-runtime-py312/z3/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export TVM_FFI_CACHE_DIR="$TILELANG_ROOT/tvm-ffi-cache-torch29-h100"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG=WARN
