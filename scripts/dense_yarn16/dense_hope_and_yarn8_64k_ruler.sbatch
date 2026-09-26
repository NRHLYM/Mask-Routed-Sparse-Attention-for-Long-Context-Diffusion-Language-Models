#!/usr/bin/env bash
#SBATCH --job-name=dense-64k-hope-yarn8
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-hope-yarn8-64k-ruler.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dense-hope-yarn8-64k-ruler.err

# Mix=0.05 dense 64k probes moved off Jingneng:
#   GPUs 0,1: HoPE dense-16k (keep hope, no YaRN rewrite)
#   GPUs 2,3: YaRN dense-16k with training factor=8 (not 32)
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1

EVAL="$ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
DATA_DIR="$ROOT/data/ruler-probes-goldspan"
TASKS=(hils_sn hils_mkmq hils_vt)
LENGTH=65536

HOPE_CONFIG=configs/from_dense16k/dense-hope-16k-tilde.json
HOPE_CKPT=outputs/dense-hope-16k-step500/step-500
HOPE_OUT=outputs/dense-hope-16k-step500/ruler_probes
YARN_CONFIG=configs/dense_yarn16/dense-yarn8-16k.json
YARN_CKPT=outputs/dense-yarn8-16k-step500/step-500
YARN_OUT=outputs/dense-yarn8-16k-step500/ruler_probes_yarn8

[[ -f "$EVAL" ]] || { echo "missing $EVAL" >&2; exit 1; }
[[ -f "$HOPE_CONFIG" ]] || { echo "missing $HOPE_CONFIG" >&2; exit 1; }
[[ -f "$YARN_CONFIG" ]] || { echo "missing $YARN_CONFIG" >&2; exit 1; }
for ckpt in "$HOPE_CKPT" "$YARN_CKPT"; do
  [[ -s "$ckpt/trainable_state.pt" && -s "$ckpt/checkpoint_manifest.json" ]] || {
    echo "missing complete $ckpt" >&2
    exit 1
  }
done

mkdir -p "$HOPE_OUT" "$YARN_OUT" "$DATA_DIR" "$ROOT/logs"
for task in "${TASKS[@]}"; do
  dest="$DATA_DIR/len${LENGTH}/${task}/validation.jsonl"
  mkdir -p "$(dirname "$dest")"
  if [[ ! -s "$dest" ]]; then
    python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
  fi
done

INDUCTOR_LOCAL="${TMPDIR:-/tmp}/dense-64k-hope-yarn8-${SLURM_JOB_ID:-$$}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

run_job() {
  local config="$1" ckpt="$2" out="$3" task="$4"
  "$PYTHON" "$EVAL" \
    --mode eval \
    --training_config "$config" \
    --checkpoint "$ckpt" \
    --output_dir "$out" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$LENGTH" \
    --rank 0 --world_size 1 --device cuda:0 \
    --block_length 32 --threshold 0.9
  "$PYTHON" "$EVAL" \
    --mode merge \
    --output_dir "$out" \
    --data_dir "$DATA_DIR" \
    --task "$task" \
    --max_seq_len "$LENGTH"
}

job_done() {
  [[ -s "$1/len${LENGTH}/${2}/metrics.json" ]]
}

run_pair() {
  local config="$1" ckpt="$2" out="$3"
  shift 3
  local gpus=("$@")
  local pending=()
  local task
  for task in "${TASKS[@]}"; do
    if job_done "$out" "$task"; then
      echo "skip $out len${LENGTH} $task"
    else
      if [[ -d "$out/len${LENGTH}/${task}" ]] && ! job_done "$out" "$task"; then
        echo "reset incomplete $out $task"
        rm -rf "$out/len${LENGTH}/${task}"
      fi
      pending+=("$task")
    fi
  done
  (( ${#pending[@]} == 0 )) && return 0
  local pids=() i gpu
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      for ((j=i; j<${#pending[@]}; j+=${#gpus[@]})); do
        echo "gpu=$gpu job=len${LENGTH}:${pending[$j]} out=$out"
        run_job "$config" "$ckpt" "$out" "${pending[$j]}"
      done
    ) > "$out/worker-len${LENGTH}-gpu${i}.log" 2>&1 &
    pids+=("$!")
  done
  local fail=0 pid
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  return "$fail"
}

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPUS[@]} < 4 )); then
  echo "need 4 GPUs" >&2
  exit 1
fi
echo "===== dense 64k HoPE + YaRN factor=8 ====="
echo "hope ckpt=$HOPE_CKPT yarn ckpt=$YARN_CKPT"
fail=0
(
  unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR
  run_pair "$HOPE_CONFIG" "$HOPE_CKPT" "$HOPE_OUT" "${GPUS[0]}" "${GPUS[1]}"
) &
p0=$!
(
  export RULER_KEEP_TRAIN_YARN=1
  export RULER_YARN_FACTOR=8
  run_pair "$YARN_CONFIG" "$YARN_CKPT" "$YARN_OUT" "${GPUS[2]}" "${GPUS[3]}"
) &
p1=$!
wait "$p0" || fail=1
wait "$p1" || fail=1
if (( fail != 0 )); then
  echo "worker failed; see $HOPE_OUT/worker-*.log $YARN_OUT/worker-*.log" >&2
  exit 1
fi
"$PYTHON" "$EVAL" --mode summary --output_dir "$HOPE_OUT" --model_name densehope --lengths "$LENGTH"
"$PYTHON" "$EVAL" --mode summary --output_dir "$YARN_OUT" --model_name dense --lengths "$LENGTH"
echo DENSE_HOPE_AND_YARN8_64K_RULER_DONE
