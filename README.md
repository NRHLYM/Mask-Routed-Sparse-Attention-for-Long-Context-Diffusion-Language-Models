# Mask-Routed Sparse Attention for Long-Context Diffusion Language Models

This branch is README-only and is intended as an anonymous code-release placeholder.

## Method

Mask-Routed Sparse Attention (MRSA) is a continued-training sparse attention method for diffusion language models.
Diffusion models unmask tokens under bidirectional attention, so each denoising step is quadratic in context length.
Autoregressive sparse attention (sliding windows, token-level routers, or compressed-plus-selected blocks) is designed for causal decoding.
MRSA instead reuses a fact already present in masked training: a mask token is trained to absorb the context needed to predict an unknown token.
Placed at the end of a chunk, that token summarizes the chunk and scores which remote chunks a query should read.

Route layers interleave with bidirectional sliding-window layers.
On a route layer a query sees a local window plus a small number of selected chunks.
On the remaining layers every query sees a sliding window whose radius is chosen so that the number of keys per layer stays matched across methods (about 2560 keys).

## Recipe (high level)

- Base checkpoint: Dream-v0-Base-7B, original context 2048, YaRN to 16K
- Frozen base weights; LoRA on attention projections
- MRSA also trains a role offset on the route slot and a low-rank residual on the scoring query
- Continued training: 1000 cosine steps, 1:1 mixture of a language-model corpus and three RULER families (single-needle, multi-key multi-query, variable tracking)
- Comparisons use the same token budget, the same mixture, and the same step count: dense attention, all-window attention, Native Sparse Attention, DeepSeek Sparse Attention, and a hybrid of windows plus dense slots

## Status

Source code, training configs, and evaluation scripts will be released after review.
This branch contains no author names, affiliations, emails, or host-specific paths.
