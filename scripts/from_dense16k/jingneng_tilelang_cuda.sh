#!/usr/bin/env bash
# TileLang CUDA toolchain for Jingneng GPU jobs.
# PyTorch can run without nvcc; HiLS kernel JIT cannot.
# Prefer a system CUDA toolkit on the GPU node, then a pip nvidia-cuda-nvcc tree.

_jingneng_have_nvcc() {
  local home="${1:-}"
  [[ -n "$home" && -x "${home}/bin/nvcc" ]]
}

if ! _jingneng_have_nvcc "${CUDA_HOME:-}"; then
  unset CUDA_HOME
  for _cuda_home in \
    /usr/local/cuda \
    /usr/local/cuda-12.9 \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12 \
    /usr/lib/cuda \
    "${ROOT:-/Data/xiongjing}/cuda-nvcc-12.9"
  do
    if _jingneng_have_nvcc "$_cuda_home"; then
      export CUDA_HOME="$_cuda_home"
      break
    fi
  done
  unset _cuda_home
fi

if ! _jingneng_have_nvcc "${CUDA_HOME:-}"; then
  _nvcc="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$_nvcc" ]]; then
    export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
  fi
  unset _nvcc
fi

if ! _jingneng_have_nvcc "${CUDA_HOME:-}"; then
  _pip_nvcc="$(
    python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
try:
    files = metadata.files("nvidia-cuda-nvcc") or []
except metadata.PackageNotFoundError:
    raise SystemExit(0)
for file in files:
    if file.name in {"nvcc", "nvcc.exe"}:
        path = Path(file.locate()).resolve()
        print(path.parent.parent)
        break
PY
  )"
  if _jingneng_have_nvcc "${_pip_nvcc:-}"; then
    export CUDA_HOME="$_pip_nvcc"
  fi
  unset _pip_nvcc
fi

if ! _jingneng_have_nvcc "${CUDA_HOME:-}"; then
  echo "TileLang needs nvcc. On this GPU job CUDA_HOME is unset and no nvcc was found." >&2
  echo "Checked /usr/local/cuda* and pip nvidia-cuda-nvcc. Run: ls -d /usr/local/cuda*; which nvcc" >&2
  exit 1
fi

export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:${PATH:-}"
if [[ -d "$CUDA_HOME/lib64" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -d "$CUDA_HOME/lib" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
python - <<'PY'
from pathlib import Path
import os
from tilelang.contrib import nvcc
home = nvcc.find_cuda_path()
nvcc_bin = Path(home) / "bin" / "nvcc"
print(f"tilelang_cuda_home={home} nvcc={nvcc_bin} exists={nvcc_bin.exists()}")
if not nvcc_bin.exists():
    raise SystemExit("CUDA_HOME does not contain bin/nvcc")
PY
unset -f _jingneng_have_nvcc
