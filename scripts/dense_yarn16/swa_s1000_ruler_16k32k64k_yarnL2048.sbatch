#!/usr/bin/env bash
#SBATCH --job-name=swa-ruler-L
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-swa-s1000-ruler-yarnL.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-swa-s1000-ruler-yarnL.err

# Finish official-table SWA s1000 goldspan RULER with YaRN = L/2048
# (not keep-train factor 8). Skips cells that already have metrics.json.
# Remaining: 32k MK-MQ/VT and 64k SN/MK-MQ/VT. 16k already done (= factor 8).
#
#   cd $ROOT
#   sbatch scripts/dense_yarn16/swa_s1000_ruler_16k32k64k_yarnL2048.sbatch
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR

CONFIG=configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000-tilde.json
CHECKPOINT=outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/step-1000
OUTPUT_ROOT=outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/ruler_probes_goldspan
NAME=swasync1000
EVAL="$ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="$ROOT/data/ruler-probes-goldspan"
TASKS=(hils_sn hils_mkmq hils_vt)
LENGTHS=(16384 32768 65536)

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

mkdir -p "$OUTPUT_ROOT" "$DATA_DIR" "$ROOT/logs"
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    dest="$DATA_DIR/len${length}/${task}/validation.jsonl"
    mkdir -p "$(dirname "$dest")"
    if [[ ! -s "$dest" ]]; then
      python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
    fi
  done
done

INDUCTOR_LOCAL="${TMPDIR:-/tmp}/swa-s1000-ruler-yarnL-${SLURM_JOB_ID:-$$}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

run_job() {
  local length="$1"
  local task="$2"
  "$PYTHON" "$EVAL" \
    --mode eval \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length" \
    --rank 0 --world_size 1 --device cuda:0 \
    --block_length 32 --threshold 0.9
  "$PYTHON" "$EVAL" \
    --mode merge \
    --output_dir "$OUTPUT_ROOT" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$length"
}

job_done() {
  [[ -s "$OUTPUT_ROOT/len${1}/${2}/metrics.json" ]]
}

pairs=()
for length in "${LENGTHS[@]}"; do
  for task in "${TASKS[@]}"; do
    if job_done "$length" "$task"; then
      echo "skip swa len${length} $task"
      continue
    fi
    if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
      echo "reset incomplete swa len${length} $task"
      rm -rf "$OUTPUT_ROOT/len${length}/${task}"
    fi
    pairs+=("${length}:${task}")
  done
done

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "===== swa official RULER goldspan 16k/32k/64k YaRN=L/2048 ====="
echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT keep_train_yarn=0 yarn_factor=L/2048 pending=${#pairs[@]}"
if (( ${#pairs[@]} == 0 )); then
  echo "all metrics present"
else
  pids=()
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      for ((j=i; j<${#pairs[@]}; j+=${#GPUS[@]})); do
        IFS=':' read -r length task <<< "${pairs[$j]}"
        echo "gpu=$gpu job=len${length}:${task}"
        run_job "$length" "$task"
      done
    ) > "$OUTPUT_ROOT/worker-gpu${i}.log" 2>&1 &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if (( fail != 0 )); then
    echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
    exit 1
  fi
fi

"$PYTHON" "$EVAL" \
  --mode summary \
  --output_dir "$OUTPUT_ROOT" \
  --model_name "$NAME" \
  --lengths 16384 32768 65536
echo "swa_S1000_RULER_16K32K64K_YARN_L2048_DONE"
