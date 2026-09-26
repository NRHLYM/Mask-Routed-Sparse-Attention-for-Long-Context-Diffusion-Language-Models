#!/usr/bin/env bash
#SBATCH --job-name=dense-sync-ruler
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-y8-16k-dolmaruler-sync-ruler.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-y8-16k-dolmaruler-sync-ruler.err

# Official goldspan RULER for dense 1:1 step-500.
# 16k (YaRN 8) on GPUs 0,1; 64k (factor=32) on GPUs 2,3.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1

CONFIG=configs/dense_yarn16/dense-yarn8-16k-dolmaruler-sync-s500.json
CHECKPOINT=outputs/dense-yarn8-16k-dolmaruler-sync-s500/step-500
OUTPUT_ROOT=outputs/dense-yarn8-16k-dolmaruler-sync-s500/ruler_probes_goldspan
EVAL="$ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="$ROOT/data/ruler-probes-goldspan"
TASKS=(hils_sn hils_mkmq hils_vt)

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}

mkdir -p "$OUTPUT_ROOT" "$DATA_DIR"
for length in 16384 65536; do
  for task in "${TASKS[@]}"; do
    dest="$DATA_DIR/len${length}/${task}/validation.jsonl"
    mkdir -p "$(dirname "$dest")"
    if [[ ! -s "$dest" ]]; then
      python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
    fi
  done
done

INDUCTOR_LOCAL="${TMPDIR:-/tmp}/dense-sync-ruler-${SLURM_JOB_ID:-$$}"
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

run_length() {
  local length="$1"
  shift
  local gpus=("$@")
  local pending=()
  local task
  for task in "${TASKS[@]}"; do
    if job_done "$length" "$task"; then
      echo "skip len${length} $task"
    else
      if [[ -d "$OUTPUT_ROOT/len${length}/${task}" ]] && ! job_done "$length" "$task"; then
        echo "reset incomplete len${length} $task"
        rm -rf "$OUTPUT_ROOT/len${length}/${task}"
      fi
      pending+=("$task")
    fi
  done
  (( ${#pending[@]} == 0 )) && return 0
  local pids=()
  local i gpu
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      for ((j=i; j<${#pending[@]}; j+=${#gpus[@]})); do
        echo "gpu=$gpu job=len${length}:${pending[$j]}"
        run_job "$length" "${pending[$j]}"
      done
    ) > "$OUTPUT_ROOT/worker-len${length}-gpu${i}.log" 2>&1 &
    pids+=("$!")
  done
  local fail=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  return "$fail"
}

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "===== dense-sync official RULER goldspan ====="
echo "devices=${GPUS[*]} ckpt=$CHECKPOINT output=$OUTPUT_ROOT"
fail=0
if (( ${#GPUS[@]} >= 4 )); then
  run_length 16384 "${GPUS[0]}" "${GPUS[1]}" &
  p16=$!
  run_length 65536 "${GPUS[2]}" "${GPUS[3]}" &
  p64=$!
  wait "$p16" || fail=1
  wait "$p64" || fail=1
else
  run_length 16384 "${GPUS[@]}" || fail=1
  run_length 65536 "${GPUS[@]}" || fail=1
fi
if (( fail != 0 )); then
  echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
  exit 1
fi
"$PYTHON" "$EVAL" \
  --mode summary \
  --output_dir "$OUTPUT_ROOT" \
  --model_name denssync \
  --lengths 16384 65536
echo DENSE_SYNC_RULER_16K64K_DONE
