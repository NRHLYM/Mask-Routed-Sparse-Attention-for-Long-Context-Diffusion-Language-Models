#!/usr/bin/env bash
# Jingneng GPU: LoRA+LMK-type <- CE, Q-Cal <- dense Z_c KL (no KL STE on type embed).
# Default GPUs 0,1. Fresh output dir.
set -euo pipefail
source /Data/xiongjing/env.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH="$FULLTEACHER_ROOT"
source "$NSA_ROOT/scripts/from_dense16k/jingneng_tilelang_cuda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
cd "$FULLTEACHER_ROOT"
NPROC="${NPROC:-2}"
CONFIG="$NSA_ROOT/configs/from_dense16k/hils-t4-jingneng-lora-qcal-lmk-klqcal-cedetach-lmkce.json"
DENSE="/Data/xiongjing/outputs/dense-yarn8-16k-step500/step-500"
PACK=/Data/xiongjing/data/dolma3_dolmino_subset/dream-16k-onedoc
OUTPUT="/Data/xiongjing/outputs/hils-t4-i4-hardroute-lora-qcal-lmk-klqcal-cedetach-lmkce"
LOG="$ROOT/logs/hils-lora-qcal-lmk-klqcal-cedetach-lmkce-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT" "$ROOT/logs"
[[ -f "$DENSE/trainable_state.pt" ]] || { echo "missing dense ckpt: $DENSE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }
for f in train.bin train.meta.json train.segments.bin train.valid.bin; do
  [[ -s "$PACK/$f" ]] || { echo "incomplete 16k packs: missing $PACK/$f" >&2; exit 1; }
done
python3 - <<PY
from pathlib import Path
from inspect import signature
from dream_dllm_hils.checkpointing import load_trainable_checkpoint
from dream_dllm_hils.data import FullTextComplementaryCollator
from dream_dllm_hils.routing import route_topk_g7
from dream_dllm_hils.train_fulltext import validate_training_config
import json

p = Path("$PACK/train.bin")
if p.stat().st_size < 206438400:
    raise SystemExit(f"truncated train.bin: {p.stat().st_size} bytes, need 206438400")
if "allow_partial" not in signature(load_trainable_checkpoint).parameters:
    raise SystemExit("checkpointing.load_trainable_checkpoint missing allow_partial")
if "insert_landmarks" not in FullTextComplementaryCollator.__dataclass_fields__:
    raise SystemExit("FullTextComplementaryCollator missing insert_landmarks")
cfg = json.loads(Path("$CONFIG").read_text())
validate_training_config(cfg)
if cfg.get("hils_lmk_kl_ste", True) or not cfg.get("hils_lmk_ce_ste", False):
    raise SystemExit("lmkce config must set hils_lmk_kl_ste=false and hils_lmk_ce_ste=true")
from dream_dllm_hils.full_dense_teacher import ste_type_embed_hidden
attn_src = Path("dream_dllm_hils/attention.py").read_text()
if "_lmk_ce_ste_residual" not in attn_src or "ste_type_embed_hidden" not in Path("dream_dllm_hils/full_dense_teacher.py").read_text():
    raise SystemExit("missing CE STE type-embed path")
print("preflight_ok", route_topk_g7.__name__, "klqcal_cedetach_lmkce")
PY
start_args=()
resume_from="$(find "$OUTPUT" -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${resume_from}" && -f "${resume_from}/trainable_state.pt" ]]; then
  start_args=(--resume_from "$resume_from")
fi
echo "nproc=$NPROC devices=$CUDA_VISIBLE_DEVICES pythonpath=$PYTHONPATH log=$LOG ${start_args[*]:-}"
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  -m dream_dllm_hils.train_fulltext --config "$CONFIG" "${start_args[@]}" \
  2>&1 | tee "$LOG"
