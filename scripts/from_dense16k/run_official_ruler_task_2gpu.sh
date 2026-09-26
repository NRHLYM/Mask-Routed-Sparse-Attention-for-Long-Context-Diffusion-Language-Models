#!/usr/bin/env bash
# One goldspan RULER task on exactly two GPUs.
# One process, world_size=1, 100 items. Layers+KV split across cuda:0/cuda:1
# so a single 128k sequence fits. CUDA_VISIBLE_DEVICES must be two ids.
# Caller sets PYTHONPATH, CONFIG, CHECKPOINT, OUTPUT_ROOT, DATA_DIR, EVAL, PYTHON.
set -euo pipefail
length="${1:?max_seq_len}"
task="${2:?task}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?need CUDA_VISIBLE_DEVICES}"
if (( ${#GPUS[@]} != 2 )); then
  echo "need exactly 2 GPUs in CUDA_VISIBLE_DEVICES, got ${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 1
fi
: "${EVAL:?}" "${CONFIG:?}" "${CHECKPOINT:?}" "${OUTPUT_ROOT:?}" "${DATA_DIR:?}"
PYTHON="${PYTHON:-python}"
out_dir="$OUTPUT_ROOT/len${length}/${task}"
if [[ -s "$out_dir/metrics.json" ]]; then
  echo "skip ${length}:${task}"
  exit 0
fi
mkdir -p "$out_dir" "$DATA_DIR/len${length}/${task}"
dest="$DATA_DIR/len${length}/${task}/validation.jsonl"
if [[ ! -s "$dest" ]]; then
  python3 -c 'import json,sys; [sys.stdout.write(json.dumps({"index": i})+"\n") for i in range(100)]' > "$dest"
fi

echo "pair_gpus=${GPUS[0]},${GPUS[1]} job=len${length}:${task} layer_parallel=2 world_size=1"
CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]}" "$PYTHON" "$EVAL" \
  --mode eval \
  --training_config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_ROOT" \
  --data_dir "$DATA_DIR" \
  --task "$task" \
  --max_seq_len "$length" \
  --rank 0 --world_size 1 --device cuda:0 \
  --layer_parallel_gpus 2 \
  --num_samples 100 \
  --block_length 32 --threshold 0.9 \
  > "$out_dir/rank-0.worker.log" 2>&1

"$PYTHON" "$EVAL" \
  --mode merge \
  --output_dir "$OUTPUT_ROOT" \
  --data_dir "$DATA_DIR" \
  --task "$task" \
  --max_seq_len "$length" \
  --num_samples 100
