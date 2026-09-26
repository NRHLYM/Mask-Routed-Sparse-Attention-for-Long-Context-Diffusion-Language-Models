#!/usr/bin/env bash
#SBATCH --job-name=s1000-lb8
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1100G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --output=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s1000-longbench.log
#SBATCH --error=/home/guests/zhen/dsa_baselines/runs/hils-fullteacher-20260917/logs/%j-s1000-longbench.err

# Official Fast-dLLM LongBench-v1 all21. Take a whole 8-GPU node
# (exclusive + 96 CPU + 1100G) so a 4-GPU/78-CPU job cannot colocate
# and requeue us. Completed metrics.json tasks are skipped on resume.
#   cd $ROOT
#   sbatch scripts/from_dense16k/s1000_longbench_8gpu.sbatch nsa
#   sbatch scripts/from_dense16k/s1000_longbench_8gpu.sbatch swa
#   sbatch scripts/from_dense16k/s1000_longbench_8gpu.sbatch s2qcal
#   sbatch scripts/from_dense16k/s1000_longbench_8gpu.sbatch dense
#   sbatch --job-name=swa3d1-lb8 scripts/from_dense16k/s1000_longbench_8gpu.sbatch swa3dense1
# 3SWA+1dense: Jingneng 1:1 s1000, English 16 only (skip 5 Chinese).
#   sbatch --job-name=dsa-lb8 scripts/from_dense16k/s1000_longbench_8gpu.sbatch dsa
# DSA: Jingneng official-warmup s1000, English 16 only (skip 5 Chinese).
set -euo pipefail
_ENV="${SLURM_SUBMIT_DIR}/dense_yarn16_env.sh"
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/scripts/dense_yarn16/dense_yarn16_env.sh"
fi
if [[ ! -f "$_ENV" ]]; then
  _ENV="${SLURM_SUBMIT_DIR}/../dense_yarn16/dense_yarn16_env.sh"
fi
source "$_ENV"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PHYS=16384
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
  NGPU=${#_gpus[@]}
else
  NGPU="${SLURM_GPUS_ON_NODE:-8}"
fi

ARM="${1:?usage: sbatch ...sbatch nsa|swa|s2qcal|dense|dsa|swa3dense1}"
case "$ARM" in
  dsa)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_dsa.py
    CONFIG=configs/from_dense16k/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000-tilde.json
    CHECKPOINT=outputs/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000/step-1000
    OUTPUT_ROOT=outputs/dsa-yarn8-16k-topk2560-swa1280-i4-official-warmup-s1000/longbench_fastdllm_en16
    DONE_FLAG=DSA_OFFICIAL_WARMUP_S1000_LONGBENCH_EN16_DONE
    TREE="$ROOT/eval-trees/dsa-official-s1000-lb"
    OVER="$TREE/overlay"
    mkdir -p "$TREE/dream_dllm_hils" "$TREE/scripts/dream_dllm_hils"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$ROOT/dream_dllm_hils/" "$TREE/dream_dllm_hils/"
      rsync -a "$ROOT/scripts/" "$TREE/scripts/"
    else
      cp -a "$ROOT/dream_dllm_hils/." "$TREE/dream_dllm_hils/"
      cp -a "$ROOT/scripts/." "$TREE/scripts/"
    fi
    DSA_OVERLAY=(
      train_fulltext.py dsa_attention.py attention.py
      fastdllm_attention.py fastdllm_v1.py fastdllm_cache.py
    )
    for f in "${DSA_OVERLAY[@]}"; do
      [[ -f "$OVER/$f" ]] || { echo "missing DSA overlay $OVER/$f" >&2; exit 1; }
      cp -f "$OVER/$f" "$TREE/dream_dllm_hils/"
    done
    if [[ -f "$OVER/eval_longbench_mfen.py" ]]; then
      cp -f "$OVER/eval_longbench_mfen.py" "$TREE/scripts/dream_dllm_hils/"
    fi
    if [[ -f "$OVER/eval_longbench_fastdllm_dsa.py" ]]; then
      cp -f "$OVER/eval_longbench_fastdllm_dsa.py" "$TREE/scripts/dream_dllm_hils/"
    fi
    EVAL_SCRIPT="$TREE/scripts/dream_dllm_hils/eval_longbench_fastdllm_dsa.py"
    export PYTHONPATH="$TREE:$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
    export PYTHONSAFEPATH=1
    ;;
  nsa)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_nsa.py
    CONFIG=configs/from_dense16k/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000-tilde.json
    CHECKPOINT=outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000/step-1000
    OUTPUT_ROOT=outputs/nsa-i4-w128-c64s64-swa1280-dolma-ruler-sync-s1000/longbench_fastdllm_all21
    DONE_FLAG=NSA_S1000_LONGBENCH_DONE
    TREE="$ROOT/eval-trees/nsa-s1000-lb"
    OVER="$TREE/overlay"
    mkdir -p "$TREE/dream_dllm_hils" "$TREE/scripts/dream_dllm_hils"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$ROOT/dream_dllm_hils/" "$TREE/dream_dllm_hils/"
      rsync -a "$ROOT/scripts/" "$TREE/scripts/"
    else
      cp -a "$ROOT/dream_dllm_hils/." "$TREE/dream_dllm_hils/"
      cp -a "$ROOT/scripts/." "$TREE/scripts/"
    fi
    NSA_OVERLAY=(
      train_fulltext.py nsa_attention.py
      fastdllm_attention.py fastdllm_v1.py fastdllm_cache.py
    )
    for f in "${NSA_OVERLAY[@]}"; do
      [[ -f "$OVER/$f" ]] || { echo "missing NSA overlay $OVER/$f" >&2; exit 1; }
      cp -f "$OVER/$f" "$TREE/dream_dllm_hils/"
    done
    if [[ -f "$OVER/eval_longbench_mfen.py" ]]; then
      cp -f "$OVER/eval_longbench_mfen.py" "$TREE/scripts/dream_dllm_hils/"
    fi
    export PYTHONPATH="$TREE:$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
    export PYTHONSAFEPATH=1
    ;;
  swa)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_swa.py
    CONFIG=configs/from_dense16k/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000-tilde.json
    CHECKPOINT=outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/step-1000
    OUTPUT_ROOT=outputs/swa-yarn8-16k-w1280-noskipinert-dolma-ruler-sync-s1000/longbench_fastdllm_all21
    DONE_FLAG=SWA_S1000_LONGBENCH_DONE
    ;;
  s2qcal)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_hils.py
    CONFIG=configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-tilde.json
    CHECKPOINT=outputs/hils-s2-qcal-rmsnorm-rand-s1000/step-1000
    OUTPUT_ROOT=outputs/hils-s2-qcal-rmsnorm-rand-s1000/longbench_fastdllm_all21
    DONE_FLAG=S2QCAL_S1000_LONGBENCH_DONE
    TREE="$ROOT/eval-trees/hils-s2-qcal-rmsnorm"
    OVER="$TREE/overlay"
    mkdir -p "$TREE/dream_dllm_hils" "$TREE/scripts/dream_dllm_hils"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$ROOT/dream_dllm_hils/" "$TREE/dream_dllm_hils/"
      rsync -a "$ROOT/scripts/" "$TREE/scripts/"
    else
      cp -a "$ROOT/dream_dllm_hils/." "$TREE/dream_dllm_hils/"
      cp -a "$ROOT/scripts/." "$TREE/scripts/"
    fi
    S2_OVERLAY=(qcal.py attention.py train_fulltext.py)
    for f in "${S2_OVERLAY[@]}"; do
      [[ -f "$OVER/$f" ]] || { echo "missing s2 overlay $OVER/$f" >&2; exit 1; }
      cp -f "$OVER/$f" "$TREE/dream_dllm_hils/"
    done
    if [[ -f "$OVER/eval_longbench_mfen.py" ]]; then
      cp -f "$OVER/eval_longbench_mfen.py" "$TREE/scripts/dream_dllm_hils/"
    fi
    export PYTHONPATH="$TREE:$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
    export PYTHONSAFEPATH=1
    ;;
  dense)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_dense_ckpt.py
    CONFIG=configs/from_dense16k/dense-yarn8-16k-dolmaruler-sync-s1000-tilde.json
    CHECKPOINT=outputs/dense-yarn8-16k-dolmaruler-sync-s1000/step-1000
    OUTPUT_ROOT=outputs/dense-yarn8-16k-dolmaruler-sync-s1000/longbench_fastdllm_all21
    DONE_FLAG=DENSE_DOLMARULER_SYNC_S1000_LONGBENCH_DONE
    export PYTHONPATH="$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
    export PYTHONSAFEPATH=1
    ;;
  swa3dense1)
    EVAL_SCRIPT=scripts/dream_dllm_hils/eval_longbench_fastdllm_dense_ckpt.py
    CONFIG=configs/from_dense16k/swa3-dense1-i4-w1280-dolmaruler-sync-s1000-tilde.json
    CHECKPOINT=outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/step-1000
    OUTPUT_ROOT=outputs/swa3-dense1-i4-w1280-dolma-ruler-sync-s1000/longbench_fastdllm_en16
    DONE_FLAG=SWA3DENSE1_S1000_LONGBENCH_EN16_DONE
    TREE="$ROOT/eval-trees/swa3-dense1-s1000-lb"
    OVER="$TREE/overlay"
    mkdir -p "$TREE/dream_dllm_hils" "$TREE/scripts/dream_dllm_hils"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$ROOT/dream_dllm_hils/" "$TREE/dream_dllm_hils/"
      rsync -a "$ROOT/scripts/" "$TREE/scripts/"
    else
      cp -a "$ROOT/dream_dllm_hils/." "$TREE/dream_dllm_hils/"
      cp -a "$ROOT/scripts/." "$TREE/scripts/"
    fi
    [[ -f "$OVER/train_fulltext.py" ]] || { echo "missing overlay $OVER/train_fulltext.py" >&2; exit 1; }
    cp -f "$OVER/train_fulltext.py" "$TREE/dream_dllm_hils/"
    if [[ -f "$OVER/eval_longbench_fastdllm_dense_ckpt.py" ]]; then
      cp -f "$OVER/eval_longbench_fastdllm_dense_ckpt.py" "$TREE/scripts/dream_dllm_hils/"
    fi
    export PYTHONPATH="$TREE:$ROOT:$TILELANG_ROOT/tilelang-src-cu125-gpu-full:$TILELANG_ROOT/tilelang-runtime-py312"
    export PYTHONSAFEPATH=1
    ;;
  *)
    echo "usage: sbatch ...sbatch nsa|swa|s2qcal|dense|dsa|swa3dense1" >&2
    exit 1
    ;;
esac

INDUCTOR_LOCAL="${TMPDIR:-/tmp}/${ARM}-s1000-lb-${SLURM_JOB_ID:-$$}"
export TMPDIR="$INDUCTOR_LOCAL"
export TRITON_CACHE_DIR="$INDUCTOR_LOCAL/triton"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_LOCAL/torchinductor"
export TORCH_EXTENSIONS_DIR="$INDUCTOR_LOCAL/torch_extensions"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$OUTPUT_ROOT" "$ROOT/logs"

[[ -f "$EVAL_SCRIPT" ]] || { echo "missing $EVAL_SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing $CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT/trainable_state.pt" && -s "$CHECKPOINT/checkpoint_manifest.json" ]] || {
  echo "missing complete $CHECKPOINT" >&2
  exit 1
}
[[ -s "$PROMPTS" ]] || { echo "missing $PROMPTS" >&2; exit 1; }

"$PYTHON" -P - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
arm = "$ARM"
mode = str(cfg.get("attention_mode", ""))
if arm == "nsa" and mode != "nsa":
    raise SystemExit("NSA config must be attention_mode=nsa")
if arm == "swa" and mode != "swa":
    raise SystemExit("SWA config must be attention_mode=swa")
if arm == "s2qcal" and mode != "hils":
    raise SystemExit("s2qcal config must be attention_mode=hils")
if arm == "nsa" and (int(cfg.get("local_window", 0)) != 128 or int(cfg.get("swa_local_window", 0)) != 1280):
    raise SystemExit("NSA must keep W=128 and swa_local_window=1280")
if arm == "nsa":
    import dream_dllm_hils.train_fulltext as tft
    from inspect import signature
    from dream_dllm_hils.nsa_attention import install_dream_nsa_attention
    if "nsa_backend" not in tft.DEFAULTS:
        raise SystemExit(f"NSA PYTHONPATH missing nsa_backend: {tft.__file__}")
    if "swa_local_window" not in signature(install_dream_nsa_attention).parameters:
        raise SystemExit("install_dream_nsa_attention missing swa_local_window")
    import dream_dllm_hils.fastdllm_attention as fa
    fa_text = Path(fa.__file__).read_text()
    if "DreamNsaAttention" not in fa_text:
        raise SystemExit(f"NSA eval-tree fastdllm_attention missing DreamNsaAttention: {fa.__file__}")
if arm == "swa" and int(cfg.get("local_window", 0)) != 1280:
    raise SystemExit("SWA must keep local_window=1280")
if arm == "s2qcal":
    import dream_dllm_hils.qcal as qcal
    text = Path(qcal.__file__).read_text()
    if "QCalRMSNorm" not in text or "residual-random-lowrank-rmsnorm-v1" not in text:
        raise SystemExit(f"s2qcal PYTHONPATH not RMSNorm Q-Cal: {qcal.__file__}")
    if int(cfg.get("hils_topk", 0)) != 32 or int(cfg.get("swa_local_window", 0)) != 1280:
        raise SystemExit("s2qcal must keep topk=32 swa=1280")
if arm == "dense":
    if mode != "dense":
        raise SystemExit("dense config must be attention_mode=dense")
    if abs(float(cfg.get("ruler_mix_ratio", 1) or 0)) > 1e-12:
        raise SystemExit("dense 1:1 LongBench needs ruler_mix_ratio=0")
    if not bool(cfg.get("hils_sync_ruler_ce", False)):
        raise SystemExit("dense 1:1 LongBench needs hils_sync_ruler_ce")
    man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
    if int(man.get("step", 0)) != 1000:
        raise SystemExit(f"dense LongBench needs step-1000, got {man.get('step')}")
if arm == "dsa":
    if mode != "dsa":
        raise SystemExit("DSA config must be attention_mode=dsa")
    if int(cfg.get("dsa_topk", 0)) != 2560 or int(cfg.get("swa_local_window", 0)) != 1280:
        raise SystemExit("DSA must keep dsa_topk=2560 and swa_local_window=1280")
    if int(cfg.get("hils_interleave", 0)) != 4:
        raise SystemExit("DSA must keep hils_interleave=4")
    man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
    if int(man.get("step", 0)) != 1000:
        raise SystemExit(f"DSA LongBench needs step-1000, got {man.get('step')}")
    import dream_dllm_hils.dsa_attention as da
    da_text = Path(da.__file__).read_text()
    if "warmup_dense" not in da_text:
        raise SystemExit(f"DSA eval-tree missing warmup_dense: {da.__file__}")
    import dream_dllm_hils.fastdllm_attention as fa
    fa_text = Path(fa.__file__).read_text()
    if "DreamDsaAttention" not in fa_text:
        raise SystemExit(f"DSA eval-tree fastdllm_attention missing DreamDsaAttention: {fa.__file__}")
if arm == "swa3dense1":
    if mode != "dense" or str(cfg.get("non_hils_attention", "dense")) != "sliding":
        raise SystemExit("swa3dense1 needs attention_mode=dense and non_hils_attention=sliding")
    if int(cfg.get("hils_interleave", 0)) != 4:
        raise SystemExit("swa3dense1 must keep hils_interleave=4")
    if int(cfg.get("local_window", 0)) != 1280 or int(cfg.get("swa_local_window", 0)) != 1280:
        raise SystemExit("swa3dense1 must keep SWA radius 1280")
    if abs(float(cfg.get("ruler_mix_ratio", 1) or 0)) > 1e-12:
        raise SystemExit("swa3dense1 1:1 LongBench needs ruler_mix_ratio=0")
    if not bool(cfg.get("hils_sync_ruler_ce", False)):
        raise SystemExit("swa3dense1 needs hils_sync_ruler_ce")
    man = json.loads(Path("$CHECKPOINT/checkpoint_manifest.json").read_text())
    if int(man.get("step", 0)) != 1000:
        raise SystemExit(f"swa3dense1 LongBench needs step-1000, got {man.get('step')}")
    import dream_dllm_hils.train_fulltext as tft
    if "def interleaved_swa_dense" not in Path(tft.__file__).read_text():
        raise SystemExit(f"swa3dense1 PYTHONPATH missing interleaved_swa_dense: {tft.__file__}")
print("preflight_ok", arm, "longbench8")
PY

declare -A ANSWER_TOKENS=(
  [narrativeqa]=128 [qasper]=128 [multifieldqa_en]=64 [multifieldqa_zh]=64
  [hotpotqa]=32 [2wikimqa]=32 [musique]=32 [dureader]=128
  [gov_report]=512 [qmsum]=512 [multi_news]=512 [vcsum]=512
  [trec]=64 [triviaqa]=32 [samsum]=128 [lsht]=64
  [passage_count]=32 [passage_retrieval_en]=32 [passage_retrieval_zh]=32
  [lcc]=64 [repobench-p]=64
)
ALL_TASKS=(
  multifieldqa_en gov_report narrativeqa qasper musique qmsum dureader samsum
  hotpotqa triviaqa multi_news repobench-p trec passage_retrieval_en passage_count
  vcsum lcc multifieldqa_zh lsht 2wikimqa passage_retrieval_zh
)
# Official 5 Chinese tasks: multifieldqa_zh dureader vcsum lsht passage_retrieval_zh
if [[ "$ARM" == "dsa" || "$ARM" == "swa3dense1" ]]; then
  ALL_TASKS=(
    multifieldqa_en gov_report narrativeqa qasper musique qmsum samsum
    hotpotqa triviaqa multi_news repobench-p trec passage_retrieval_en passage_count
    lcc 2wikimqa
  )
fi

task_done() { [[ -s "$OUTPUT_ROOT/$1/metrics.json" ]]; }
reset_incomplete() {
  local output_dir="$OUTPUT_ROOT/$1"
  if [[ -d "$output_dir" ]] && ! task_done "$1"; then
    echo "reset incomplete $1"
    rm -rf "$output_dir"
  fi
}

extra_eval=()
extra_merge=()
case "$ARM" in
  nsa)
    extra_eval=(--model_path "$MODEL" --cache_mode dual_block)
    extra_merge=(--model_path "$MODEL" --cache_mode dual_block)
    ;;
  swa)
    extra_eval=(--local_window 1280 --allow_empty_predictions)
    extra_merge=(--local_window 1280 --allow_empty_predictions)
    ;;
  s2qcal)
    extra_eval=(--local_window 256 --topk 32 --qcal_scale 1.0 --allow_empty_predictions)
    extra_merge=(--local_window 256 --topk 32 --qcal_scale 1.0 --allow_empty_predictions)
    ;;
  dense)
    extra_eval=(--local_window "$PHYS" --allow_empty_predictions)
    extra_merge=(--local_window "$PHYS" --allow_empty_predictions)
    ;;
  dsa)
    extra_eval=(--model_path "$MODEL" --cache_mode dual_block)
    extra_merge=(--model_path "$MODEL" --cache_mode dual_block)
    ;;
  swa3dense1)
    extra_eval=(--local_window "$PHYS" --allow_empty_predictions)
    extra_merge=(--local_window "$PHYS" --allow_empty_predictions)
    ;;
esac

run_eval() {
  local task="$1"
  mkdir -p "$OUTPUT_ROOT/$task"
  "$PYTHON" -P "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --rank 0 --world_size 1 --device cuda:0 --limit 0 \
    "${extra_eval[@]}"
}
merge_task() {
  local task="$1"
  "$PYTHON" -P "$EVAL_SCRIPT" \
    --training_config "$CONFIG" --checkpoint "$CHECKPOINT" --task "$task" \
    --data "$DATA_ROOT/$task.jsonl" --prompt_config "$PROMPTS" \
    --output_dir "$OUTPUT_ROOT/$task" --physical_length "$PHYS" --chunk_size 64 \
    --answer_tokens "${ANSWER_TOKENS[$task]}" --block_length 32 --threshold 0.9 \
    --merge \
    "${extra_merge[@]}"
}

remaining=()
for task in "${ALL_TASKS[@]}"; do
  [[ -s "$DATA_ROOT/$task.jsonl" ]] || { echo "missing $DATA_ROOT/$task.jsonl" >&2; exit 1; }
  if task_done "$task"; then echo "skip $task"
  else reset_incomplete "$task"; remaining+=("$task"); fi
done
echo "===== $ARM longbench ${#ALL_TASKS[@]}tasks ${NGPU}gpu ====="
echo "ckpt=$CHECKPOINT output=$OUTPUT_ROOT pythonpath_head=${PYTHONPATH%%:*}"
echo "REMAINING ${#remaining[@]}: ${remaining[*]:-none}"

summarize_suite() {
  if [[ "$ARM" == "dsa" || "$ARM" == "swa3dense1" ]]; then
    "$PYTHON" -P - <<PY
import json, os
from pathlib import Path
from dream_dllm_hils.longbench_eval import LONGBENCH_EN_TASKS
output_root = Path("$OUTPUT_ROOT")
data_root = Path("$DATA_ROOT")
task_metrics = {}
for task in LONGBENCH_EN_TASKS:
    metrics_path = output_root / task / "metrics.json"
    data_path = data_root / f"{task}.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"missing metrics for {task}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = sum(1 for line in data_path.open("rb") if line.strip())
    n = int(metrics.get("examples", metrics.get("dataset_examples", -1)))
    if n != expected:
        raise SystemExit(f"{task}: expected {expected} examples, got {n}")
    if int(metrics.get("fallback_max", -1)) != 0:
        raise SystemExit(f"{task}: nonzero kernel fallback")
    task_metrics[task] = {
        "examples": expected,
        "metric": str(metrics["metric"]),
        "score": float(metrics["score_avg"]),
        "mean_seconds": float(metrics.get("mean_seconds", 0)),
        "p50_seconds": float(metrics.get("p50_seconds", 0)),
        "fallback_max": int(metrics.get("fallback_max", 0)),
    }
scores = [task_metrics[t]["score"] for t in LONGBENCH_EN_TASKS]
payload = {
    "suite": "LongBench-v1-en16",
    "english_task_count": len(LONGBENCH_EN_TASKS),
    "all_task_count": len(LONGBENCH_EN_TASKS),
    "english_macro_average": sum(scores) / len(scores),
    "all_macro_average": sum(scores) / len(scores),
    "skipped_zh": [
        "multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh"
    ],
    "total_examples": sum(int(task_metrics[t]["examples"]) for t in LONGBENCH_EN_TASKS),
    "tasks": task_metrics,
}
out = output_root / "suite_metrics.json"
tmp = out.with_name(out.name + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
PY
  else
    "$PYTHON" scripts/summarize_longbench_suite.py --output_root "$OUTPUT_ROOT" --data_root "$DATA_ROOT"
  fi
}

if (( ${#remaining[@]} == 0 )); then
  summarize_suite
  echo "$DONE_FLAG"
  exit 0
fi

pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
  tasks=()
  for i in "${!remaining[@]}"; do
    if (( i % NGPU == gpu )); then tasks+=("${remaining[$i]}"); fi
  done
  (( ${#tasks[@]} == 0 )) && continue
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for task in "${tasks[@]}"; do
      echo "gpu=$gpu task=$task"
      run_eval "$task"
      merge_task "$task"
    done
  ) > "$OUTPUT_ROOT/worker-$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then
  echo "worker failed; see $OUTPUT_ROOT/worker-*.log" >&2
  exit 1
fi
summarize_suite
echo "$DONE_FLAG"
