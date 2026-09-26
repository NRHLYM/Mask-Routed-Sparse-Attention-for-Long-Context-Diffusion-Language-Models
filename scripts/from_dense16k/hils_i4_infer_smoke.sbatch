#!/usr/bin/env bash
#SBATCH --job-name=hils-i4-infer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
cd "$ROOT"

CONFIG=configs/from_dense16k/hils-t4-yarn8-16k-k32-w512.json
CHECKPOINT=outputs/hils-t4-yarn8-16k-k32-w512-i4-fromdense/step-500
OUT=outputs/hils-t4-yarn8-16k-k32-w512-i4-fromdense/infer-smoke-mfen
mkdir -p "$OUT" logs

test -f "$CHECKPOINT/trainable_state.pt"
test -f "$DATA_ROOT/multifieldqa_en.jsonl"
test -f scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py

"$PYTHON" scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py \
  --training_config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --task multifieldqa_en \
  --data "$DATA_ROOT/multifieldqa_en.jsonl" \
  --prompt_config "$PROMPTS" \
  --output_dir "$OUT" \
  --physical_length 16384 \
  --chunk_size 64 \
  --local_window 512 \
  --topk 32 \
  --answer_tokens 64 \
  --block_length 32 \
  --threshold 0.9 \
  --limit 2 \
  --rank 0 --world_size 1 --device cuda:0 \
  --allow_empty_predictions

"$PYTHON" scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py \
  --training_config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --task multifieldqa_en \
  --data "$DATA_ROOT/multifieldqa_en.jsonl" \
  --prompt_config "$PROMPTS" \
  --output_dir "$OUT" \
  --physical_length 16384 \
  --chunk_size 64 \
  --local_window 512 \
  --topk 32 \
  --merge --allow_empty_predictions

python3 - << PY
import json
from pathlib import Path
root = Path("$OUT")
print("===== metrics =====")
print((root / "metrics.json").read_text())
print("===== predictions =====")
for line in (root / "merged.jsonl").read_text().splitlines():
    rec = json.loads(line)
    pred = str(rec.get("prediction", rec.get("pred", "")))[:400]
    gold = rec.get("answers") or rec.get("answer")
    print("index", rec.get("index"), "score", rec.get("score") or rec.get("qa_f1"))
    print("pred:", repr(pred))
    print("gold:", gold)
    print("---")
print("HILS_I4_INFER_SMOKE_DONE")
PY
