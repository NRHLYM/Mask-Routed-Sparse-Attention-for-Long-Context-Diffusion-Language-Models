#!/usr/bin/env bash
# Jingneng GPU: frozen LMK vs token-LSE diagnosis. 2 GPUs, one_shot prefill.
# Does not train. Default is s2-sync, same ckpt as ruler_2gpu.sh s2sync.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_lmk_vs_token_lse.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_lmk_vs_token_lse.sh s2sync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_lmk_vs_token_lse.sh s2vf
# Optional: LIMIT=16 LENGTH=16384
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC="${#DEVICES[@]}"
LIMIT="${LIMIT:-16}"
LENGTH="${LENGTH:-16384}"
MODEL="${1:-s2sync}"
INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-lmk-token-lse-${MODEL}}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

case "$MODEL" in
  s2sync)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-500"
    OUTDIR="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/lmk_vs_token_lse"
    ;;
  s2sync1000)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s1000/step-1000"
    OUTDIR="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s1000/lmk_vs_token_lse"
    ;;
  s2vf)
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-value-fusion-beta0p3-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/step-250"
    OUTDIR="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/lmk_vs_token_lse"
    ;;
  *)
    echo "unknown MODEL=$MODEL (s2sync|s2sync1000|s2vf)" >&2
    exit 1
    ;;
esac
PROBE="$NSA_ROOT/scripts/from_dense16k/probe_lmk_vs_token_lse.py"
SUMMARIZE="$NSA_ROOT/scripts/from_dense16k/summarize_lmk_vs_token_lse.py"
LOG="$ROOT/logs/lmk-vs-token-lse-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTDIR" "$ROOT/logs"
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" ]] || { echo "missing ckpt $CHECKPOINT" >&2; exit 1; }
[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }

if [[ "$MODEL" == "s2vf" ]]; then
  overlay="$NSA_ROOT/overlays/value-fusion-old-qcal/qcal.py"
  tree="/Data/xiongjing/src/eval-trees/hils-s2-value-fusion-old-qcal-lmkprobe"
  src="$FULLTEACHER_ROOT/dream_dllm_hils"
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
want = "residual-zero-up-native-scale-v1"
old = "residual-random-lowrank-rmsnorm-v1"
text = path.read_text()
if want in text and old not in text:
    print("qcal_version already pinned", want)
else:
    if old not in text:
        raise SystemExit(f"{path} missing {old} and {want}")
    path.write_text(text.replace(old, want))
    print("pinned qcal_version", want)
PY
  export PYTHONPATH="$tree"
  [[ -f "$tree/dream_dllm_hils/value_aware_fusion.py" ]] || {
    echo "missing value_aware_fusion.py on VF probe tree" >&2
    exit 1
  }
  echo "s2vf probe code_root=$tree"
fi
[[ -f "$FULLTEACHER_ROOT/dream_dllm_hils/lmk_token_lse_stats.py" ]] || {
  echo "missing lmk_token_lse_stats.py; rsync fullteacher" >&2
  exit 1
}

{
  echo "===== lmk vs token LSE model=$MODEL devices=$CUDA_VISIBLE_DEVICES limit=$LIMIT length=$LENGTH ====="
  echo "ckpt=$CHECKPOINT out=$OUTDIR"
  rm -f "$OUTDIR"/rank*.jsonl
  pids=()
  for rank in $(seq 0 $((NPROC - 1))); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$rank]}" python "$PROBE" \
      --training_config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output "$OUTDIR/rank${rank}.jsonl" \
      --length "$LENGTH" \
      --limit "$LIMIT" \
      --rank "$rank" \
      --world_size "$NPROC" \
      --device cuda:0 &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  [[ "$fail" -eq 0 ]] || { echo "probe rank failed" >&2; exit 1; }
  python "$SUMMARIZE" "$OUTDIR"/rank*.jsonl --out "$OUTDIR/summary.json"
} 2>&1 | tee "$LOG"
echo "LMK_TOKEN_LSE_DONE out=$OUTDIR/summary.json log=$LOG"
