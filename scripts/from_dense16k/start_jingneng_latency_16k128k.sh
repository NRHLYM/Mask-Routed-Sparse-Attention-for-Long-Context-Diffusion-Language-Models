#!/usr/bin/env bash
# Jingneng GPU: Fast-dLLM prefill + decode latency, 16k/32k/64k/128k.
# Arms: MRSA (S2-Q-Cal), Full dense, Hybrid dense. YaRN = L/2048.
# 16/32/64: 1 GPU. 128k: 2 GPU layer-parallel. Warmup 3, 10 timed trials.
# Do not launch from the notebook. Wait until DSA 128k RULER has released GPUs.
#
#   source /Data/xiongjing/env.sh && cd "$NSA_ROOT"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
#     scripts/from_dense16k/start_jingneng_latency_16k128k.sh
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
export PYTHONUNBUFFERED=1
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR || true

PYTHON="${PYTHON:-python}"
BENCH="$NSA_ROOT/scripts/from_dense16k/bench_jingneng_fastdllm_latency.py"
OUT_ROOT="/Data/xiongjing/outputs/latency_fastdllm_16k128k"
PATCH_DIR="$NSA_ROOT/dream_dllm_hils"
LOG="$ROOT/logs/latency-16k128k-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUT_ROOT" "$ROOT/logs"
export NSA_ROOT

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPUS[@]} < 2 )); then
  echo "need at least 2 GPUs (128k layer-parallel), got $CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

stage_mrsa() {
  local tree="/Data/xiongjing/src/eval-trees/latency-s2qcal"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local name
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
  cp -f "$src/qcal.py" "$tree/dream_dllm_hils/qcal.py"
  cp -f "$src/attention.py" "$tree/dream_dllm_hils/attention.py"
  cp -f "$src/train_fulltext.py" "$tree/dream_dllm_hils/train_fulltext.py"
  grep -q "residual-random-lowrank-rmsnorm-v1" "$tree/dream_dllm_hils/qcal.py" \
    || { echo "qcal missing RMSNorm" >&2; exit 1; }
  echo "$tree"
}

stage_hybrid() {
  local tree="/Data/xiongjing/src/eval-trees/latency-hybrid-dense"
  local src="$HILS_FT_ROOT/dream_dllm_hils"
  local name base skip real
  local STAGE_REAL=(train_fulltext.py fastdllm_cache.py fastdllm_v1.py)
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$HILS_FT_ROOT/ops" "$tree/ops"
  ln -sfn "$HILS_FT_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    base="$(basename "$name")"
    skip=0
    for real in "${STAGE_REAL[@]}"; do
      [[ "$base" == "$real" ]] && skip=1 && break
    done
    if (( skip )); then continue; fi
    ln -sfn "$name" "$tree/dream_dllm_hils/$base"
  done
  for real in "${STAGE_REAL[@]}"; do
    rm -f "$tree/dream_dllm_hils/$real"
    cp -a "$PATCH_DIR/$real" "$tree/dream_dllm_hils/$real"
  done
  grep -q "def interleaved_swa_dense" "$tree/dream_dllm_hils/train_fulltext.py" \
    || { echo "hybrid train_fulltext missing interleaved_swa_dense" >&2; exit 1; }
  echo "$tree"
}

MRSA_TREE="$(stage_mrsa)"
HYBRID_TREE="$(stage_hybrid)"
DENSE_TREE="$FULLTEACHER_ROOT"

run_one() {
  local arm="$1" tree="$2" config="$3" ckpt="$4" length="$5" gpus="$6" npar="$7"
  local out="$OUT_ROOT/${arm}_len${length}.json"
  if [[ -s "$out" ]]; then
    echo "skip $arm len${length}"
    return 0
  fi
  echo "RUN arm=$arm len=$length gpus=$gpus parallel=$npar"
  CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$tree" \
    "$PYTHON" "$BENCH" \
      --arm "$arm" \
      --training_config "$config" \
      --checkpoint "$ckpt" \
      --output_json "$out" \
      --max_seq_len "$length" \
      --layer_parallel_gpus "$npar"
}

{
  echo "===== Fast-dLLM latency 16k-128k MRSA / dense / hybrid ====="
  echo "devices=${GPUS[*]} out=$OUT_ROOT"

  [[ -s /Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000/trainable_state.pt ]] \
    || { echo "missing MRSA ckpt" >&2; exit 1; }
  [[ -s /Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/step-1000/trainable_state.pt ]] \
    || { echo "missing dense ckpt" >&2; exit 1; }
  [[ -s /Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/step-1000/trainable_state.pt ]] \
    || { echo "missing hybrid ckpt" >&2; exit 1; }
  [[ -f "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }

  declare -A TREE=([mrsa]="$MRSA_TREE" [dense]="$DENSE_TREE" [hybrid]="$HYBRID_TREE")
  declare -A CFG=(
    [mrsa]="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json"
    [dense]="$NSA_ROOT/configs/from_dense16k/dense-yarn8-16k-dolmaruler-sync-s1000-jingneng.json"
    [hybrid]="$NSA_ROOT/configs/from_dense16k/swa3-dense1-i4-w1280-dolmaruler-sync-s1000-jingneng.json"
  )
  declare -A CKPT=(
    [mrsa]="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000"
    [dense]="/Data/xiongjing/outputs/dense-yarn8-16k-dolmaruler-sync-s1000/step-1000"
    [hybrid]="/Data/xiongjing/outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/step-1000"
  )

  SHORT=(16384 32768 65536)
  for arm in mrsa dense hybrid; do
    echo "===== $arm 16/32/64 ====="
    pids=()
    ng=${#GPUS[@]}
    (( ng > 3 )) && ng=3
    for i in "${!SHORT[@]}"; do
      length="${SHORT[$i]}"
      gpu="${GPUS[$((i % ng))]}"
      (
        run_one "$arm" "${TREE[$arm]}" "${CFG[$arm]}" "${CKPT[$arm]}" \
          "$length" "$gpu" 1
      ) > "$OUT_ROOT/worker-${arm}-len${length}.log" 2>&1 &
      pids+=("$!")
    done
    fail=0
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
    if (( fail != 0 )); then
      echo "$arm 16/32/64 failed; see $OUT_ROOT/worker-${arm}-len*.log" >&2
      exit 1
    fi
    echo "===== $arm 128k ====="
    run_one "$arm" "${TREE[$arm]}" "${CFG[$arm]}" "${CKPT[$arm]}" \
      131072 "${GPUS[0]},${GPUS[1]}" 2 \
      > "$OUT_ROOT/worker-${arm}-len131072.log" 2>&1
  done

  "$PYTHON" - <<PY
import json
from pathlib import Path
root = Path("$OUT_ROOT")
rows = []
for p in sorted(root.glob("*_len*.json")):
    rows.append(json.loads(p.read_text()))
by = {}
for r in rows:
    by.setdefault(r["max_seq_len"], {})[r["arm"]] = r
summary = {"suite": "fastdllm_latency_16k128k", "lengths": {}}
for length in (16384, 32768, 65536, 131072):
    cell = {}
    dense = by.get(length, {}).get("dense")
    for arm in ("dense", "hybrid", "mrsa"):
        rec = by.get(length, {}).get(arm)
        if not rec:
            continue
        item = {
            "prefill_ms": rec["prefill_ms_median"],
            "decode_ms_per_token": rec["decode_ms_per_token_median"],
            "cached_forwards": rec["cached_forwards_median"],
            "peak_gib": rec["peak_memory_bytes"] / (1024 ** 3),
        }
        if dense and arm != "dense":
            item["prefill_speedup_vs_dense"] = dense["prefill_ms_median"] / rec["prefill_ms_median"]
            item["decode_speedup_vs_dense"] = (
                dense["decode_ms_per_token_median"] / rec["decode_ms_per_token_median"]
            )
        cell[arm] = item
    summary["lengths"][str(length)] = cell
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY
  echo "LATENCY_16K128K_DONE"
} 2>&1 | tee "$LOG"
