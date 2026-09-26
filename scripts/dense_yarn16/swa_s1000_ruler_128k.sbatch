#!/usr/bin/env bash
#SBATCH --job-name=swa-ruler-128k
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-swa-s1000-ruler-128k.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-swa-s1000-ruler-128k.err

# Official SWA s1000 goldspan RULER at 128k. Two GPUs per task (layer-parallel).
# Both YaRN modes: L/2048 (factor=64) and keep-train factor=8.
#
#   cd $ROOT
#   sbatch scripts/dense_yarn16/swa_s1000_ruler_128k.sbatch
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PAIR="$ROOT/scripts/from_dense16k/run_official_ruler_task_2gpu.sh"
export PYTHONPATH="$ROOT"
export EVAL="$ROOT/scripts/from_dense16k/eval_jingneng_official_ruler.py"
export CONFIG=configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000-tilde.json
export CHECKPOINT=outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/step-1000
export DATA_DIR="$ROOT/data/ruler-probes-goldspan"
export PYTHON="${PYTHON}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

[[ -f "$EVAL" && -f "$PAIR" && -f "$CONFIG" ]] || { echo "missing eval/pair/config" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing $CHECKPOINT" >&2
  exit 1
}

INDUCTOR_LOCAL="${TMPDIR:-/tmp}/swa-s1000-ruler-128k-${SLURM_JOB_ID:-$$}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$DATA_DIR"

run_three() {
  local g0="$1" g1="$2" g2="$3" g3="$4"
  CUDA_VISIBLE_DEVICES="$g0,$g1" bash "$PAIR" 131072 hils_sn &
  local p_sn=$!
  CUDA_VISIBLE_DEVICES="$g2,$g3" bash "$PAIR" 131072 hils_mkmq &
  local p_mk=$!
  wait "$p_sn"
  wait "$p_mk"
  CUDA_VISIBLE_DEVICES="$g0,$g1" bash "$PAIR" 131072 hils_vt
}

run_mode() {
  export OUTPUT_ROOT="$1"
  mkdir -p "$OUTPUT_ROOT"
  IFS=',' read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
  echo "swa 128k out=$OUTPUT_ROOT keep=${RULER_KEEP_TRAIN_YARN:-0} factor=${RULER_YARN_FACTOR:-L/2048} devices=${G[*]}"
  if (( ${#G[@]} >= 4 )); then
    run_three "${G[0]}" "${G[1]}" "${G[2]}" "${G[3]}"
  elif (( ${#G[@]} == 2 )); then
    CUDA_VISIBLE_DEVICES="${G[0]},${G[1]}" bash "$PAIR" 131072 hils_sn
    CUDA_VISIBLE_DEVICES="${G[0]},${G[1]}" bash "$PAIR" 131072 hils_mkmq
    CUDA_VISIBLE_DEVICES="${G[0]},${G[1]}" bash "$PAIR" 131072 hils_vt
  else
    echo "need 2 or 4 GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
  "$PYTHON" "$EVAL" --mode summary --output_dir "$OUTPUT_ROOT" --model_name swasync1000_128k --lengths 131072
}

unset RULER_KEEP_TRAIN_YARN RULER_YARN_FACTOR
run_mode outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/ruler_probes_goldspan
export RULER_KEEP_TRAIN_YARN=1
export RULER_YARN_FACTOR=8
run_mode outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/ruler_probes_goldspan_yarn8fixed
echo "swa_S1000_RULER_128K_L2048_AND_YARN8_DONE"
