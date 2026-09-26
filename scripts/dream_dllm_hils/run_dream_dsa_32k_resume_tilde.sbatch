#!/usr/bin/env bash
#SBATCH --job-name=dream_dsa_32k
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_train_%j.out
#SBATCH --error=/home/guests/zhen/dsa_baselines/HiLS-Attention/logs/dream_dsa_32k_train_%j.err

set -euo pipefail

ROOT=/home/guests/zhen/dsa_baselines/HiLS-Attention
PYTHON=/home/guests/zhen/dsa_baselines/venvs/dream_dsa/bin/python
MODEL=/home/guests/zhen/jing/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
CONFIG=configs/dream_dllm_hils/mixed_ruler05_32k_yarn_dsa_topk1024_lr2e-4_500.json
OUTPUT=outputs/dream-mixed-ruler05-32k-yarn-dsa-topk1024-lr2e-4-dolma3
FINAL="$OUTPUT/step-500"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG=WARN

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
    && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" \
    && -f "$1/trainer_state.pt" ]]
}

latest_checkpoint() {
  local candidate
  while IFS= read -r candidate; do
    if checkpoint_is_complete "$candidate"; then
      printf '%s\n' "$candidate"
    fi
  done < <(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V)
}

if checkpoint_is_complete "$FINAL"; then
  echo "[dsa-32k] complete checkpoint already exists: $FINAL"
  exit 0
fi

resume_from="$(latest_checkpoint | tail -n 1)"
if [[ -z "$resume_from" ]]; then
  echo "[dsa-32k] no complete checkpoint available for exact resume" >&2
  exit 1
fi

echo "[dsa-32k] resuming on tilde from $resume_from"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  -m dream_dllm_hils.train_fulltext \
  --config "$CONFIG" \
  --model_path "$MODEL" \
  --resume_from "$resume_from"
