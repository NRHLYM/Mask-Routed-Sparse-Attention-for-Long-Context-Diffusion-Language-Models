#!/usr/bin/env bash
#SBATCH --job-name=hils-fdens-mfen
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
cd "$ROOT"

EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py
CONFIG=configs/from_dense16k/hils-t4-yarn8-16k-k32-w512-hardroute-frozendense.json
CHECKPOINT=outputs/hils-t4-yarn8-16k-k32-w512-i4-hardroute-frozendense-fromdense/step-500
OUTPUT_ROOT=outputs/hils-t4-yarn8-16k-k32-w512-i4-hardroute-frozendense-fromdense/longbench_fastdllm_mfen
PHYS=16384

test -s "$CHECKPOINT/checkpoint_manifest.json"
test -s "$CHECKPOINT/trainable_state.pt"
test -f "$EVAL_SCRIPT"
test -f "$DATA_ROOT/multifieldqa_en.jsonl"
mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local rank="$1" world="$2"
  local output_dir="$OUTPUT_ROOT/multifieldqa_en"
  mkdir -p "$output_dir"
  "$PYTHON" "$EVAL_SCRIPT" \
    --training_config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --task multifieldqa_en \
    --data "$DATA_ROOT/multifieldqa_en.jsonl" \
    --prompt_config "$PROMPTS" \
    --output_dir "$output_dir" \
    --physical_length "$PHYS" \
    --chunk_size 64 \
    --local_window 512 \
    --topk 32 \
    --answer_tokens 64 \
    --block_length 32 \
    --threshold 0.9 \
    --rank "$rank" --world_size "$world" --device cuda:0 \
    --limit 0
}

echo "===== multifieldqa_en (2-way) frozen-dense ====="
mfen_pids=()
for gpu in 0 1; do
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_eval "$gpu" 2
  ) > "$OUTPUT_ROOT/mfen-rank-$gpu.log" 2>&1 &
  mfen_pids+=("$!")
done
failed=0
for pid in "${mfen_pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 ))

"$PYTHON" "$EVAL_SCRIPT" \
  --training_config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --task multifieldqa_en \
  --data "$DATA_ROOT/multifieldqa_en.jsonl" \
  --prompt_config "$PROMPTS" \
  --output_dir "$OUTPUT_ROOT/multifieldqa_en" \
  --physical_length "$PHYS" \
  --chunk_size 64 \
  --local_window 512 \
  --topk 32 \
  --answer_tokens 64 \
  --block_length 32 \
  --threshold 0.9 \
  --merge > "$OUTPUT_ROOT/multifieldqa_en/merge.log" 2>&1

python3 - << PY
import json
from pathlib import Path
p = Path("$OUTPUT_ROOT/multifieldqa_en/metrics.json")
d = json.loads(p.read_text())
print("MFEN_DONE", d.get("score_avg", d.get("qa_f1")))
print(json.dumps({k: d.get(k) for k in ["task", "examples", "qa_f1", "score_avg", "metric"]}, indent=2))
PY
echo HILS_FDENS_MFEN_DONE
