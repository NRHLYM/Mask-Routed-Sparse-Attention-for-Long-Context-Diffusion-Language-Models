#!/usr/bin/env bash
# Jingneng 2-GPU official RULER for one 16k-trained baseline.
# YaRN factor = L/2048 at 16k / 32k / 64k. Sparse budgets stay as trained.
#
# Usage (from $NSA_ROOT after env.sh):
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh hils
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh s2sync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh s2all28
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh s2from2k
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh s2from2ks1500
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh s2alltasks
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh nsa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh nsasync
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh dsa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh swa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/from_dense16k/start_jingneng_ruler_2gpu.sh dense
set -euo pipefail
source /Data/xiongjing/env.sh
source "$NSA_ROOT/scripts/from_dense16k/jingneng_baselines.sh"
MODEL="${1:?hils|s2sync|s2b5|s2qcal|s2vf|s2fromhope|s2all28|s2all28s1000|s2all28s2000|s2from2k|s2from2ks1500|s2alltasks|s2gumbeltopk|s2allchunk|s2allchunksttemp|nsa|nsasync|dsa|swa|dense|densehope}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"

# NSA/SWA live in a thin overlay. Importing $NSA_ROOT alone misses
# longbench_eval / data / ops; mixing PYTHONPATH also fails because
# dream_dllm_hils is a real package. Stage fullteacher + overlay files.
stage_nsa_swa_eval_tree() {
  local tree="/Data/xiongjing/src/eval-trees/${MODEL}"
  local src="$FULLTEACHER_ROOT/dream_dllm_hils"
  local overlay="$NSA_ROOT/dream_dllm_hils"
  local name
  rm -rf "$tree"
  mkdir -p "$tree/dream_dllm_hils"
  ln -sfn "$FULLTEACHER_ROOT/ops" "$tree/ops"
  ln -sfn "$FULLTEACHER_ROOT/scripts" "$tree/scripts"
  for name in "$src"/*; do
    [[ -e "$name" ]] || continue
    ln -sfn "$name" "$tree/dream_dllm_hils/$(basename "$name")"
  done
  for name in nsa_attention.py train_fulltext.py attention.py \
      fastdllm_v1.py fastdllm_attention.py fastdllm_cache.py \
      dsa_attention.py local_attention.py; do
    [[ -f "$overlay/$name" ]] || continue
    ln -sfn "$overlay/$name" "$tree/dream_dllm_hils/$name"
  done
  echo "$tree"
}

case "$MODEL" in
  hils)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils-ruler}"
    ;;
  hils1000)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-w256-swa1280.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-w256-swa1280-s1000stop800/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-hils1000-ruler}"
    ;;
  s2)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-noteacher-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-noteacher-s1000/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2-ruler}"
    ;;
  s2sync)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2sync-ruler}"
    ;;
  s2b5)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-ablate-b5-qcal-lr0p5-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-ablate-b5-qcal-lr0p5-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-ablate-b5-qcal-lr0p5-s1000/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2b5-ruler}"
    ;;
  s2qcal)
    if [[ -z "${QCAL_RMSNORM_CODE_ROOT:-}" ]]; then
      echo "s2qcal needs QCAL_RMSNORM_CODE_ROOT (use start_jingneng_ruler_16k64k_s2qcal.sh)" >&2
      exit 1
    fi
    CODE_ROOT="$QCAL_RMSNORM_CODE_ROOT"
    export PYTHONPATH="$CODE_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-qcal-rmsnorm-rand-s1000/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2qcal-ruler}"
    ;;
  s2vf)
    CODE_ROOT="${VALUE_FUSION_CODE_ROOT:?s2vf needs VALUE_FUSION_CODE_ROOT (old Q-Cal tree)}"
    export PYTHONPATH="$CODE_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-value-fusion-beta0p3-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/step-250"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-value-fusion-beta0p3-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2vf-ruler}"
    ;;
  s2fromhope)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-fromhope-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-fromhope-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-fromhope-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2fromhope-ruler}"
    ;;
  s2all28)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-all28-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2all28-ruler}"
    ;;
  s2all28s1000)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-all28-s1000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s1000/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s1000/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2all28s1000-ruler}"
    ;;
  s2all28s2000)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-all28-s2000-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s2000/step-2000"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-all28-s2000/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2all28s2000-ruler}"
    ;;
  s2from2k)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-from2k-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2from2k-ruler}"
    ;;
  s2from2ks1500)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-from2k-s1500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-s1500-lr1e4/step-1500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-from2k-s1500-lr1e4/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2from2ks1500-ruler}"
    ;;
  s2alltasks)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-alltasks-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-alltasks-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-alltasks-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2alltasks-ruler}"
    ;;
  s2gumbeltopk)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-gumbeltopk-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-gumbeltopk-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-gumbeltopk-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2gumbeltopk-ruler}"
    ;;
  s2allchunk)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2allchunk-ruler}"
    ;;
  s2allchunksttemp)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/hils-s2-cefusion-dolma-ruler-sync-allchunk-sttemp-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-s2allchunksttemp-ruler}"
    ;;
  nsa)
    CODE_ROOT="$(stage_nsa_swa_eval_tree)"
    export PYTHONPATH="$CODE_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-yarn8-16k-b32-w128-c64s64-swa1280-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-swa1280-i4-fromdense/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/nsa-yarn8-16k-b32-w128-c64s64-swa1280-i4-fromdense/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsa-ruler}"
    ;;
  nsasync)
    CODE_ROOT="$(stage_nsa_swa_eval_tree)"
    export PYTHONPATH="$CODE_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s500-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-nsasync-ruler}"
    ;;
  dsa)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-fromdense/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/dsa-yarn8-16k-topk2560-swa1280-i4-fromdense/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dsa-ruler}"
    ;;
  swa)
    CODE_ROOT="$(stage_nsa_swa_eval_tree)"
    export PYTHONPATH="$CODE_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/swa-yarn8-16k-w1280-noskipinert-fromdense/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/swa-yarn8-16k-w1280-noskipinert-fromdense/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-swa-ruler}"
    ;;
  dense)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/dense-yarn8-16k-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/dense-yarn8-16k-step500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-dense-ruler}"
    ;;
  densehope)
    export PYTHONPATH="$FULLTEACHER_ROOT"
    CODE_ROOT="$FULLTEACHER_ROOT"
    CONFIG="$NSA_ROOT/configs/from_dense16k/dense-hope-16k-jingneng.json"
    CHECKPOINT="/Data/xiongjing/outputs/dense-hope-16k-step500/step-500"
    OUTPUT_ROOT="/Data/xiongjing/outputs/dense-hope-16k-step500/ruler_probes"
    INDUCTOR_LOCAL="${INDUCTOR_LOCAL:-/tmp/xiongjing-inductor-densehope-ruler}"
    ;;
  *)
    echo "unknown MODEL=$MODEL (hils|hils1000|s2|s2sync|s2b5|s2qcal|s2vf|s2fromhope|s2all28|s2all28s1000|s2all28s2000|s2from2k|s2from2ks1500|s2alltasks|s2gumbeltopk|s2allchunk|s2allchunksttemp|nsa|nsasync|dsa|swa|dense|densehope)" >&2
    exit 1
    ;;
esac
if [[ -n "${RULER_OUTPUT_SUBDIR:-}" ]]; then
  OUTPUT_ROOT="$(dirname "$OUTPUT_ROOT")/$RULER_OUTPUT_SUBDIR"
fi

export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$CODE_ROOT"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU="${#GPUS[@]}"
PYTHON="${PYTHON:-python}"
EVAL="$NSA_ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="${RULER_DATA:-/Data/xiongjing/data/ruler-probes}"
LIMIT="${LIMIT:-0}"
LOG="$ROOT/logs/${MODEL}-ruler-probes-L$(IFS=-; echo "${LENGTHS[*]}")-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$ROOT/logs"

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
until [[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]]; do
  echo "waiting for $CHECKPOINT"
  sleep 30
done

# First slice default in this launcher: S-N at 64k.
# For all three 64k probes use start_jingneng_ruler_64k.sh instead.
TASKS=(hils_sn)
if [[ -n "${RULER_TASKS:-}" ]]; then
  read -r -a TASKS <<< "$RULER_TASKS"
fi
LENGTHS=(65536)
if [[ -n "${RULER_LENGTHS:-}" ]]; then
  read -r -a LENGTHS <<< "$RULER_LENGTHS"
fi
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    [[ -s "$DATA_DIR/len${length}/${task}/validation.jsonl" ]] || {
      echo "missing $DATA_DIR/len${length}/${task}/validation.jsonl (run start_jingneng_ruler_prepare.sh)" >&2
      exit 1
    }
  done
done

jobs=()
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    jobs+=("$length:$task")
  done
done

job_done() {
  local length="${1%%:*}"
  local task="${1##*:}"
  [[ -s "$OUTPUT_ROOT/len${length}/${task}/metrics.json" ]]
}

reset_incomplete() {
  local spec="$1"
  local length="${spec%%:*}"
  local task="${spec##*:}"
  local output_dir="$OUTPUT_ROOT/len${length}/${task}"
  local shard
  if [[ -d "$output_dir" ]] && ! job_done "$spec"; then
    for shard in "$output_dir"/rank-*.jsonl; do
      [[ -s "$shard" ]] || continue
      echo "resume incomplete $spec from $shard"
      return 0
    done
    echo "reset empty incomplete $spec"
    rm -rf "$output_dir"
  fi
}

run_job() {
  local spec="$1"
  local length="${spec%%:*}"
  local task="${spec##*:}"
  local limit_args=()
  [[ "$LIMIT" != "0" ]] && limit_args=(--limit "$LIMIT")
  "$PYTHON" "$EVAL" \
    --mode eval \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length" \
    --rank 0 --world_size 1 --device cuda:0 \
    --block_length 32 --threshold 0.9 \
    "${limit_args[@]}"
  "$PYTHON" "$EVAL" \
    --mode merge \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length" \
    "${limit_args[@]}"
}

remaining=()
for spec in "${jobs[@]}"; do
  if job_done "$spec"; then
    echo "skip $spec"
  else
    reset_incomplete "$spec"
    remaining+=("$spec")
  fi
done

{
  echo "===== $MODEL official RULER ${NGPU}-gpu ====="
  echo "devices=$CUDA_VISIBLE_DEVICES ckpt=$CHECKPOINT code_root=$CODE_ROOT"
  echo "output=$OUTPUT_ROOT keep_train_yarn=${RULER_KEEP_TRAIN_YARN:-0} yarn_factor=${RULER_YARN_FACTOR:-L/2048} lengths=${LENGTHS[*]} remaining=${#remaining[@]}"
  if (( ${#remaining[@]} > 0 )); then
    pids=()
    for gi in "${!GPUS[@]}"; do
      gpu="${GPUS[$gi]}"
      specs=()
      for i in "${!remaining[@]}"; do
        if (( i % NGPU == gi )); then specs+=("${remaining[$i]}"); fi
      done
      (( ${#specs[@]} == 0 )) && continue
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        for spec in "${specs[@]}"; do
          echo "gpu=$gpu job=$spec"
          run_job "$spec"
        done
      ) > "$OUTPUT_ROOT/worker-${LENGTHS[*]}-$gi.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if (( failed != 0 )); then
      echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
      exit 1
    fi
  fi
  "$PYTHON" "$EVAL" \
    --mode summary \
    --output_dir "$OUTPUT_ROOT" \
    --model_name "$MODEL" \
    --lengths "${LENGTHS[@]}"
  echo "${MODEL}_RULER_PROBES_DONE"
} 2>&1 | tee "$LOG"
