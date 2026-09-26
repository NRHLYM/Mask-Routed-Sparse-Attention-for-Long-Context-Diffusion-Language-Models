#!/usr/bin/env bash
#SBATCH --job-name=dsa-sync-s1000
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dsa-k2560-sync-s1000.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-dsa-k2560-sync-s1000.err

# Official YaRN DSA: token topk=2560 + 21 SWA radius 1280, skip_inert=False,
# 1:1 Dolma+RULER, cosine 1000, lr 1e-4, 4 GPU, dense-yarn8-16k-step500 init.
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/outputs"

CONFIG=configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-i4-dolma-ruler-sync-s1000-tilde.json
OUTPUT=outputs/dsa-yarn8-16k-topk2560-swa1280-i4-dolma-ruler-sync-s1000
DENSE=outputs/dense-yarn8-16k-step500/step-500
FINAL="$OUTPUT/step-1000"
NPROC=4

[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}
if checkpoint_is_complete "$FINAL"; then
  echo "completed checkpoint already exists: $FINAL"
  exit 0
fi
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$resume_from" ]] && checkpoint_is_complete "$resume_from"; then
  start_args=(--resume_from "$resume_from")
else
  start_args=()
fi

"$PYTHON" - <<'PY'
from inspect import signature
from pathlib import Path
from dream_dllm_hils.attention import KernelDreamSlidingWindowAttention
from dream_dllm_hils.dsa_attention import install_dream_dsa_attention
from dream_dllm_hils.train_fulltext import validate_training_config
import json
cfg = json.loads(Path("configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-i4-dolma-ruler-sync-s1000-tilde.json").read_text())
validate_training_config(cfg)
if cfg.get("attention_mode") != "dsa":
    raise SystemExit("attention_mode=dsa")
if int(cfg["dsa_topk"]) != 2560:
    raise SystemExit("dsa_topk=2560")
if int(cfg["swa_local_window"]) != 1280:
    raise SystemExit("21 SWA radius 1280")
if int(cfg["max_steps"]) != 1000 or int(cfg["warmup_steps"]) != 50:
    raise SystemExit("cosine 1000 warmup 50")
if abs(float(cfg["learning_rate"]) - 1e-4) > 1e-12:
    raise SystemExit("lr 1e-4")
if not bool(cfg.get("hils_sync_ruler_ce")):
    raise SystemExit("1:1 sync")
if "skip_inert_slots" not in signature(KernelDreamSlidingWindowAttention.__init__).parameters:
    raise SystemExit("KernelDreamSlidingWindowAttention missing skip_inert_slots")
src = Path(install_dream_dsa_attention.__code__.co_filename).read_text()
if "skip_inert_slots" not in src:
    raise SystemExit("DSA SWA install must pass skip_inert_slots")
print("preflight_ok dsa-sync-s1000")
PY

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}"
