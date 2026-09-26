# Mask-Routed Sparse Attention (MRSA)

Continued-training sparse attention for long-context diffusion language models (Dream-v0-Base-7B).
A mask token at the end of each chunk summarizes that chunk and routes queries to remote chunks.

The paper name is **MRSA**. The training code still uses the internal names HiLS / S2 / Q-Cal.

This tree is the current **main recipe**: 16k YaRN-8, 1:1 Dolma + goldspan RULER, LoRA + Q-Cal, 1000 cosine steps.

## Layout

```
dream_dllm_hils/          attention, Q-Cal, training loop, Fast-dLLM wrap
ops/                      TileLang / Triton kernels
configs/from_dense16k/    mainline and ablation JSON configs
scripts/from_dense16k/    train / RULER / LongBench launchers
scripts/dense_yarn16/     sbatch variants of the same recipe
scripts/dream_dllm_hils/  corpus pack + LongBench helpers
unittests/
```

Set `PYTHONPATH` to the repository root.

## Main method (S2 + Q-Cal)

- Interleave 4: route layers `{3,7,11,15,19,23,27}`, other 21 layers bidirectional SWA radius 1280
- Route layer: local radius 256 + top-32 chunks of size 64 (`hils_topk=32`, `chunk_size=64`)
- Landmark / route slot: Dream mask token with a role offset (`lmk_token_mode=mask_type`)
- Query residual: Q-Cal `residual-random-lowrank-rmsnorm-v1`, rank 64
- Chunk summary: attention pool (`hils_chunk_summary=attn`) + entropy prior
- Trainable scope: `lora_qcal_lmk`
- Init: dense 16k YaRN-8 step-500 remix; sparse run uses `ruler_mix_ratio=0` and `hils_sync_ruler_ce=true`
- Loss: `0.5 mean(Dolma) + 0.5 mean(RULER)` live fusion
- Optimizer: lr `1e-4`, cosine 1000, warmup 50, 4 GPUs

Canonical config and launcher:

- `configs/from_dense16k/hils-s2-qcal-rmsnorm-rand-s1000-jingneng.json`
- `scripts/from_dense16k/start_jingneng_hils_s2_qcal_rmsnorm_rand_s1000.sh`

Matched-budget baselines (same 1:1 recipe): dense, all-SWA radius 1280, NSA, DSA, 3 SWA + 1 dense.

Mechanism ablations (same recipe, one change each): `learned_route`, `no_qcal`, `no_role`, `mean_pool`, `no_entropy`, `eos_route`.

```bash
# MRSA S2+Q-Cal
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_qcal_rmsnorm_rand_s1000.sh

# mechanism ablations (sequential queue)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_hils_s2_qcal_mechanism_ablate.sh

# goldspan RULER 16/32/64/128k, keep-train YaRN factor=8
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/from_dense16k/start_jingneng_ruler_yarn8fixed_16k128k_mech_ablate.sh rest
```

Configs currently hard-code cluster paths (`model_path`, Dolma packs, `output_dir`). Point those fields at your own Dream checkpoint, packed corpus, and output directory before launching. Launchers expect `env.sh` to export `NSA_ROOT` (this repo) and `FULLTEACHER_ROOT` (same tree, or a copy used as `PYTHONPATH`).

## Evaluation

Official probes: goldspan RULER S-N / MK-MQ / VT. Decode with Fast-dLLM `block_length=32`, `threshold=0.9`.
16k uses YaRN factor 8 (same as train). 32k/64k/128k in the yarn8fixed scripts keep factor 8; L/2048 scripts scale the factor with length.

## Data (not in this repository)

- Base: `Dream-v0-Base-7B`
- Language packs: 16k one-document Dolma subset (`train.bin` / `train.meta.json`)
- RULER goldspan jsonl under `ruler-probes-goldspan/len{16384,32768,65536,131072}/{hils_sn,hils_mkmq,hils_vt}/`

Checkpoints and metric dumps are kept out of git.
