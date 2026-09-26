#!/usr/bin/env python3
"""Frozen-ckpt probe: LMK score vs token LSE on selected remote chunks.

one_shot prefill only. Does not train. Writes one jsonl row per (task, index).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

NSA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NSA / "scripts" / "from_dense16k"))
sys.path.insert(0, str(NSA / "scripts"))

from eval_jingneng_official_ruler import (  # noqa: E402
    PROBE_TASKS,
    TASK_SYNTH_ID,
    apply_yarn,
    gold_output_list,
    haystack_token_ids,
    text_slot_budget,
)
from dream_dllm_hils.data import RulerDenoisingSynthesizer
from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM
from dream_dllm_hils.lmk_token_lse_stats import item_stats
from dream_dllm_hils.longbench_eval import build_generation_layout
from dream_dllm_hils.train_fulltext import (
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _set_seed,
    parse_args as parse_training_args,
)
from scripts.dream_dllm_hils.eval_longbench_mfen import (
    configure_eval_trainables,
    decode_answer,
    load_trainables,
)


class RecordingSynthesizer(RulerDenoisingSynthesizer):
    def _compose(self, base_ids, needles, needle_evidence, question_ids, answer_ids, rng, **kwargs):
        self.components = {
            "needles": [np.array(x, copy=True) for x in needles],
            "needle_evidence": [np.array(x, copy=True) for x in needle_evidence],
        }
        return super()._compose(
            base_ids, needles, needle_evidence, question_ids, answer_ids, rng, **kwargs
        )


def spans_of(clean: torch.Tensor, facts: list[np.ndarray]) -> list[tuple[int, int]]:
    array = clean.numpy()
    spans = []
    for fact in facts:
        hits = [
            int(i)
            for i in np.flatnonzero(array == fact[0])
            if np.array_equal(array[i : i + len(fact)], fact)
        ]
        if len(hits) != 1:
            raise RuntimeError(f"expected one fact span, got {hits}")
        spans.append((hits[0], hits[0] + len(fact)))
    return spans


def text_to_physical(index: int) -> int:
    return index // 63 * 64 + index % 63


def output_token_groups(tokenizer, gold_ids: list[int], outputs: list[str], task: str) -> list[list[int]]:
    text = (", " if task == "hils_vt" else " ").join(outputs)
    encoded = tokenizer(text, add_special_tokens=False).input_ids
    if encoded != gold_ids:
        raise RuntimeError("gold tokenisation drifted from synthesizer")
    offsets, previous = [], ""
    for stop in range(1, len(encoded) + 1):
        current = tokenizer.decode(encoded[:stop], clean_up_tokenization_spaces=False)
        offsets.append((len(previous), len(current)))
        previous = current
    groups, start = [], 0
    for value in outputs:
        lo = text.index(value, start)
        hi = lo + len(value)
        groups.append([i for i, (a, b) in enumerate(offsets) if a < hi and b > lo])
        start = hi
    return groups


def chunk_token_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    key_valid: torch.Tensor,
    indices: torch.Tensor,
    predictor: torch.Tensor,
    chunk_size: int,
    evidence_physical: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """z_token / z_evidence with shape [P, Hq, K]; LMK slots excluded."""
    p_idx = predictor
    batch_q = q[0, p_idx].float()
    p, h_q, dim = batch_q.shape
    h_kv = k.shape[2]
    group = h_q // h_kv
    selected = indices.shape[-1]
    idx = indices[0, p_idx].long()
    idx_h = idx.repeat_interleave(group, dim=1)
    scale = dim ** -0.5
    starts = idx_h.clamp_min(0) * chunk_size
    offs = torch.arange(chunk_size - 1, device=q.device)
    tok = starts.unsqueeze(-1) + offs
    head_kv = (torch.arange(h_q, device=q.device) // group).view(1, h_q, 1, 1).expand(
        p, h_q, selected, chunk_size - 1
    )
    k_sel = k[0][tok, head_kv].float()
    logits = (batch_q[:, :, None, None, :] * k_sel).sum(-1) * scale
    real = key_valid[0, tok].bool() & ((tok + 1) % chunk_size != 0)
    selected_ok = idx_h.unsqueeze(-1) >= 0
    logits = logits.masked_fill(~(real & selected_ok), float("-inf"))
    z_token = torch.logsumexp(logits, dim=-1)
    z_token = z_token.masked_fill(~torch.isfinite(z_token), float("nan"))
    if evidence_physical is None or evidence_physical.numel() == 0:
        z_ev = torch.full_like(z_token, float("nan"))
    else:
        ev = torch.zeros(k.shape[1], dtype=torch.bool, device=q.device)
        ev[evidence_physical] = True
        z_ev = torch.logsumexp(logits.masked_fill(~ev[tok], float("-inf")), dim=-1)
        z_ev = z_ev.masked_fill(~torch.isfinite(z_ev), float("nan"))
    z_token = z_token.masked_fill(idx_h < 0, float("nan"))
    z_ev = z_ev.masked_fill(idx_h < 0, float("nan"))
    return z_token, z_ev


def all_chunk_lmk(
    q: torch.Tensor,
    lmks: torch.Tensor,
    drop_mask: torch.Tensor,
    predictor: torch.Tensor,
    fact_chunks: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw scaled q·lmk over all chunks, matching route scores (no bias)."""
    pred = predictor
    batch_q = q[0, pred].float()
    p, h_q, dim = batch_q.shape
    chunks = lmks.shape[1]
    logical = lmks[0].reshape(chunks, h_q, dim).float()
    logits = torch.einsum("phd,chd->phc", batch_q, logical) * (dim ** -0.5)
    dropped = drop_mask[0, pred].bool()
    logits = logits.masked_fill(dropped.unsqueeze(1), float("-inf"))
    correct_all = torch.zeros(p, h_q, chunks, dtype=torch.bool, device=q.device)
    for slot in fact_chunks:
        if 0 <= int(slot) < chunks:
            correct_all[:, :, int(slot)] = True
    return logits, correct_all


def make_case(tokenizer, synth, task: str, length: int, index: int, chunk_size: int) -> dict:
    slots = text_slot_budget(length, hils=True, chunk_size=chunk_size)
    base = haystack_token_ids(tokenizer, None, slots)
    vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
    base[0] = (int(base[0]) + int(index) * 1315423911) % vocab
    clean, mask, _evidence = synth.synthesize_with_evidence(
        torch.as_tensor(base, dtype=torch.long),
        task_id=TASK_SYNTH_ID[task],
    )
    start = int(torch.where(mask)[0][0])
    gold = clean[start:].clone()
    text = decode_answer(tokenizer, gold.tolist())
    outputs = gold_output_list(task, text)
    facts = [
        x
        for x, ev in zip(synth.components["needles"], synth.components["needle_evidence"])
        if bool(np.asarray(ev).any())
    ]
    return {
        "clean": clean,
        "gold": gold,
        "answer_start": start,
        "outputs": outputs,
        "facts": facts,
        "input_sha256": hashlib.sha256(clean.numpy().tobytes()).hexdigest(),
    }


def install_hooks(captured, route_pack, state):
    import dream_dllm_hils.fastdllm_attention as fastdllm_attention
    import dream_dllm_hils.routing as routing

    original_route = routing.route_topk_g7
    original_selected = routing.selected_attention_g7
    original_cached = routing.selected_attention_g7_cached
    original_fa_route = fastdllm_attention.route_topk_g7
    original_fa_selected = fastdllm_attention.selected_attention_g7_cached

    def route_topk_g7(*route_args, **route_kwargs):
        result = original_route(*route_args, **route_kwargs)
        route_pack["indices"] = result[0]
        route_pack["scores"] = result[1]
        route_pack["q"] = route_args[0]
        route_pack["lmks"] = route_args[1]
        route_pack["drop_mask"] = route_args[4]
        return result

    def capture_selected(q, k, v, weights, indices, key_valid, chunk_size, *rest, **kwargs):
        scores = route_pack["scores"]
        predictor = state["predictor"]
        groups = state["groups"]
        item_fact_chunks = state["item_fact_chunks"]
        item_fact_physical = state["item_fact_physical"]
        layer_rows = []
        for item, (token_idx, fact_chunks, fact_pos) in enumerate(
            zip(groups, item_fact_chunks, item_fact_physical)
        ):
            if not token_idx:
                continue
            pred = predictor[token_idx]
            a = scores[0, pred].float()
            w = weights[0, pred].float()
            idx = indices[0, pred].long()
            group = q.shape[2] // k.shape[2]
            idx_h = idx.repeat_interleave(group, dim=1)
            z_tok, z_ev = chunk_token_lse(
                q, k, key_valid, indices, pred, chunk_size, fact_pos.to(q.device)
            )
            selected = idx_h >= 0
            correct = torch.zeros_like(selected)
            for slot in fact_chunks:
                correct = correct | (idx_h == int(slot))
            a_all, correct_all = all_chunk_lmk(
                route_pack["q"],
                route_pack["lmks"],
                route_pack["drop_mask"],
                pred,
                fact_chunks,
            )
            stats = item_stats(
                a_lmk=a,
                z_token=z_tok,
                z_evidence=z_ev,
                weight=w,
                correct=correct,
                selected=selected,
                a_lmk_all=a_all,
                correct_all=correct_all,
            )
            stats["item"] = item
            layer_rows.append(stats)
        captured.append({"items": layer_rows})

    def selected_attention_g7(q, k, v, weights, indices, key_valid, chunk_size, training, **kwargs):
        capture_selected(q, k, v, weights, indices, key_valid, chunk_size, training, **kwargs)
        return original_selected(
            q, k, v, weights, indices, key_valid, chunk_size, training, **kwargs
        )

    def selected_attention_g7_cached(q, k, v, weights, indices, key_valid, chunk_size, **kwargs):
        capture_selected(q, k, v, weights, indices, key_valid, chunk_size, **kwargs)
        return original_cached(
            q, k, v, weights, indices, key_valid, chunk_size, **kwargs
        )

    routing.route_topk_g7 = route_topk_g7
    routing.selected_attention_g7 = selected_attention_g7
    routing.selected_attention_g7_cached = selected_attention_g7_cached
    fastdllm_attention.route_topk_g7 = route_topk_g7
    fastdllm_attention.selected_attention_g7_cached = selected_attention_g7_cached

    def restore():
        routing.route_topk_g7 = original_route
        routing.selected_attention_g7 = original_selected
        routing.selected_attention_g7_cached = original_cached
        fastdllm_attention.route_topk_g7 = original_fa_route
        fastdllm_attention.selected_attention_g7_cached = original_fa_selected

    return restore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = parse_training_args(["--config", args.training_config, "--no_gradient_checkpointing"])
    apply_yarn(cfg, args.length)
    _set_seed(7)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(cfg, device)
    configure_eval_trainables(model, cfg)
    load_trainables(model, Path(args.checkpoint))
    model.eval()

    captured: list[dict] = []
    route_pack: dict = {}
    state: dict = {
        "predictor": None,
        "groups": [],
        "item_fact_chunks": [],
        "item_fact_physical": [],
    }
    restore = install_hooks(captured, route_pack, state)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    synth = RecordingSynthesizer(tokenizer)
    decoder = DreamHiLSFastDLLM(
        model=model,
        mask_token_id=tokenizer.mask_token_id,
        threshold=0.9,
        use_cache=False,
        bootstrap="confidence",
    )
    try:
        for task in PROBE_TASKS:
            for index in range(args.limit):
                if index % args.world_size != args.rank:
                    continue
                case = make_case(
                    tokenizer, synth, task, args.length, index, int(cfg.chunk_size)
                )
                spans = spans_of(case["clean"], case["facts"])
                phys = [
                    [text_to_physical(i) for i in range(lo, hi)] for lo, hi in spans
                ]
                state["item_fact_physical"] = [
                    torch.tensor(rows, dtype=torch.long) for rows in phys
                ]
                state["item_fact_chunks"] = [set(p // 64 for p in rows) for rows in phys]
                gold_ids = case["gold"].tolist()
                eos = tokenizer.eos_token_id
                if eos in gold_ids:
                    gold_ids = gold_ids[: gold_ids.index(eos)]
                groups = output_token_groups(
                    tokenizer, gold_ids, case["outputs"], task
                )
                layout = build_generation_layout(
                    prompt_ids=case["clean"][: case["answer_start"]].tolist(),
                    answer_tokens=int(case["gold"].numel()),
                    physical_length=args.length,
                    chunk_size=int(cfg.chunk_size),
                    mask_token_id=int(tokenizer.mask_token_id),
                    pad_token_id=int(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    landmark_token_id=int(tokenizer.mask_token_id),
                )
                state["predictor"] = layout.predictor_positions.to(device)
                state["groups"] = groups
                captured.clear()
                input_ids = layout.input_ids.unsqueeze(0).to(device)
                attention_mask = layout.attention_mask.unsqueeze(0).to(device)
                position_ids = layout.position_ids.unsqueeze(0).to(device)
                landmarks = layout.landmark_positions.to(device)
                with torch.inference_mode():
                    out = decoder.prefill(
                        input_ids,
                        attention_mask,
                        position_ids,
                        landmark_positions=landmarks,
                    )
                    logits = out.logits[0, state["predictor"]].float()
                    pred = logits.argmax(dim=-1)
                    del out
                if _kernel_fallback_count(model):
                    raise RuntimeError("kernel fallback is disallowed")
                if len(captured) != len(plan.hils_layers):
                    raise RuntimeError(
                        f"expected {len(plan.hils_layers)} HiLS captures, got {len(captured)}"
                    )
                decoded = decode_answer(tokenizer, pred.tolist())
                row = {
                    "task": task,
                    "index": index,
                    "outputs": case["outputs"],
                    "score": float(
                        sum(int(g.lower() in decoded.lower()) for g in case["outputs"])
                        / max(len(case["outputs"]), 1)
                    ),
                    "sha256": case["input_sha256"],
                    "layers": dict(zip((str(i) for i in plan.hils_layers), captured)),
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(
                    json.dumps({"task": task, "index": index, "score": row["score"]}),
                    flush=True,
                )
    finally:
        restore()
    print("LMK_TOKEN_LSE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
