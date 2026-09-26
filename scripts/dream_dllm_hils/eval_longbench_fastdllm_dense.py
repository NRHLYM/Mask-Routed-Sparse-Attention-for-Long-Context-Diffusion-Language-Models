#!/usr/bin/env python3
"""Untuned Dream-v0-Base-7B + YaRN 2k→16k + official Fast-dLLM DualCache."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from dream_dllm_hils.longbench_eval import (
    append_jsonl_fsync,
    load_resumable_jsonl,
    merge_evaluation_shards,
    shard_indices,
)
from dream_dllm_hils.train_fulltext import _set_seed
from scripts.dream_dllm_hils.eval_longbench_fastdllm_hils import _read_jsonl, score_prediction
from scripts.dream_dllm_hils.eval_longbench_mfen import decode_answer, tokenize_prompt


DEFAULT_FASTDLLM = "/home/guests/zhen/jing/d3LLM/baseline/Fast_dLLM_v1/dream"
DEFAULT_MODEL = "/home/guests/zhen/jing/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B"
YARN = {
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 2048,
}


@dataclass(frozen=True)
class OriginalForwardStats:
    full_prefills: int
    cached_forwards: int
    recomputed_tokens: int


class _ForwardCounter:
    def __init__(self, model: torch.nn.Module, block_length: int) -> None:
        self.block_length = int(block_length)
        self.full_prefills = 0
        self.cached_forwards = 0
        self.recomputed_tokens = 0
        self._handle = model.register_forward_pre_hook(self._record, with_kwargs=True)

    def _record(self, module, args, kwargs) -> None:
        del module
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return
        self.recomputed_tokens += int(input_ids.numel())
        if input_ids.shape[1] > self.block_length:
            self.full_prefills += 1
        else:
            self.cached_forwards += 1

    def close(self) -> OriginalForwardStats:
        self._handle.remove()
        return OriginalForwardStats(
            self.full_prefills, self.cached_forwards, self.recomputed_tokens
        )


def load_official_dream(*, model_path: str, fastdllm_dir: str, device: torch.device, answer_tokens: int, block_length: int, threshold: float, dual_cache: bool):
    dream_dir = os.path.abspath(fastdllm_dir)
    if dream_dir not in sys.path:
        sys.path.insert(0, dream_dir)
    from model.configuration_dream import DreamConfig
    from model.generation_utils_block import DreamGenerationMixin
    from model.modeling_dream import DreamModel, DreamRotaryEmbedding
    import transformers

    config = DreamConfig.from_pretrained(model_path)
    config.max_position_embeddings = 16384
    config.rope_theta = 1000000.0
    config.rope_scaling = dict(YARN)
    model = (
        DreamModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        .to(dtype=torch.bfloat16, device=device)
        .eval()
    )
    for module in model.modules():
        if isinstance(module, DreamRotaryEmbedding):
            module.__init__(config=model.config)
            module.to(device=device, dtype=torch.bfloat16)
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    return model, tokenizer


@torch.inference_mode()
def generate_hils_right_padded(
    *,
    model,
    prompt_ids: list[int],
    physical_length: int,
    answer_tokens: int,
    block_length: int,
    threshold: float,
    pad_id: int,
    mask_id: int,
    device: torch.device,
):
    """Match HiLS: [prompt][MASK answers][PAD to 32k], pads masked out of attention.

    Dream's pad id is EOS, same as HiLS. DualCache still runs on the 32k tensor.
    """
    from model.generation_utils_block import sample_tokens

    if answer_tokens % block_length:
        raise ValueError("answer_tokens must be divisible by block_length")
    prompt_len = len(prompt_ids)
    if prompt_len + answer_tokens > physical_length:
        raise ValueError("prompt plus answer exceeds physical_length")
    x = torch.full((1, physical_length), int(pad_id), dtype=torch.long, device=device)
    valid = torch.zeros((1, physical_length), dtype=torch.bool, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    x[0, prompt_len : prompt_len + answer_tokens] = int(mask_id)
    valid[0, : prompt_len + answer_tokens] = True
    # Broadcast key mask: every query attends only to prompt+answer, not EOS pads.
    key_mask = valid[:, None, None, :]
    num_blocks = answer_tokens // block_length
    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length
        model_output = model(x, attention_mask=key_mask, use_cache=True)
        past_key_values = model_output.past_key_values
        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        _, x0 = sample_tokens(logits, temperature=0.0)
        x[:, block_start] = x0[:, block_start]
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True
        while True:
            mask_index = x[:, block_start:block_end] == mask_id
            if not bool(mask_index.any()):
                break
            block_mask = key_mask.expand(1, 1, block_length, physical_length)
            model_output = model(
                x[:, block_start:block_end],
                attention_mask=block_mask,
                past_key_values=past_key_values,
                use_cache=True,
                dual_cache=True,
                replace_position=replace_position,
            )
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]
            confidence, x0 = sample_tokens(mask_logits, temperature=0.0)
            x_ = torch.full(
                (1, block_length), mask_id, dtype=torch.long, device=device
            )
            full_confidence = torch.full(
                (1, block_length), -float("inf"), device=device, dtype=logits.dtype
            )
            x_[mask_index] = x0.clone()
            full_confidence[mask_index] = confidence
            current_transfer_tokens = int(mask_index.sum().item())
            selected_confidence, select_index = torch.topk(
                full_confidence, current_transfer_tokens
            )
            transfer_index = torch.zeros_like(x_, dtype=torch.bool, device=device)
            transfer_index[0, select_index[0, 0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] >= threshold:
                    transfer_index[0, select_index[0, k]] = True
            x[:, block_start:block_end][transfer_index] = x_[transfer_index]
    return x[0, prompt_len : prompt_len + answer_tokens].tolist()


def eos_summary(records: list[dict]) -> dict:
    empty = 0
    eos0 = 0
    for record in records:
        pred = str(record.get("prediction") or "").strip()
        ids = record.get("raw_answer_ids") or []
        if not pred:
            empty += 1
        if record.get("eos_offset") == 0 or (ids and int(ids[0]) == 151643):
            eos0 += 1
    n = len(records)
    nonempty = [r for r in records if str(r.get("prediction") or "").strip()]
    return {
        "n": n,
        "empty": empty,
        "eos_offset_zero": eos0,
        "f1": (sum(float(r["score"]) for r in records) / n) if n else None,
        "nonempty_n": len(nonempty),
        "nonempty_f1": (
            sum(float(r["score"]) for r in nonempty) / len(nonempty) if nonempty else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--fastdllm_dream_dir", default=DEFAULT_FASTDLLM)
    parser.add_argument("--task", default="multifieldqa_en")
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_length", type=int, default=16384)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow_empty_predictions", action="store_true")
    parser.add_argument(
        "--pad_to_physical",
        action="store_true",
        help="Right-pad to physical_length like HiLS, masking pad/EOS slots.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")
    examples = _read_jsonl(args.data)
    expected_examples = len(examples)
    if args.limit > 0:
        expected_examples = min(expected_examples, args.limit * args.world_size)
    args.model_variant = "dream_v0_base_7b_untuned_fastdllm_official_dualcache_yarn8_16k"
    if args.pad_to_physical:
        args.model_variant += "_pad32k"
    if args.merge:
        metrics = merge_evaluation_shards(
            args.output_dir,
            total_examples=expected_examples,
            expected_variant=None,
            require_nonempty_predictions=not args.allow_empty_predictions,
        )
        records = load_resumable_jsonl(Path(args.output_dir) / "merged.jsonl")
        extras = eos_summary(records)
        metrics.update(extras)
        metrics["task"] = args.task
        metrics["score_avg"] = metrics.get("qa_f1", extras["f1"])
        Path(args.output_dir, "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{args.rank}.jsonl"
    completed = {int(record["index"]) for record in load_resumable_jsonl(output_path)}
    assigned = shard_indices(len(examples), args.rank, args.world_size)
    if args.limit > 0:
        assigned = assigned[: args.limit]
    pending = [index for index in assigned if index not in completed]
    templates = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))
    template = templates[args.task]
    _set_seed(args.seed + args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer = load_official_dream(
        model_path=args.model_path,
        fastdllm_dir=args.fastdllm_dream_dir,
        device=device,
        answer_tokens=args.answer_tokens,
        block_length=args.block_length,
        threshold=args.threshold,
        dual_cache=True,
    )
    max_prompt_tokens = args.physical_length - args.answer_tokens
    steps = args.answer_tokens // args.block_length
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(assigned),
                "pending": len(pending),
                "model_variant": args.model_variant,
                "model_path": args.model_path,
                "fastdllm_dream_dir": args.fastdllm_dream_dir,
                "device": str(device),
                "rope_scaling": YARN,
                "prompt_capacity": max_prompt_tokens,
                "dual_cache": True,
                "alg": "confidence_threshold",
                "pad_to_physical": bool(args.pad_to_physical),
                "pad_token_id": int(pad_id),
                "pad_is_eos": int(pad_id) == int(tokenizer.eos_token_id),
            }
        ),
        flush=True,
    )
    for ordinal, index in enumerate(pending, start=1):
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_prompt(
            tokenizer, template, example, max_prompt_tokens
        )
        torch.cuda.reset_peak_memory_stats(device)
        counter = _ForwardCounter(model, args.block_length)
        started = time.perf_counter()
        try:
            if args.pad_to_physical:
                answer_ids = generate_hils_right_padded(
                    model=model,
                    prompt_ids=prompt_ids,
                    physical_length=args.physical_length,
                    answer_tokens=args.answer_tokens,
                    block_length=args.block_length,
                    threshold=args.threshold,
                    pad_id=int(pad_id),
                    mask_id=int(tokenizer.mask_token_id),
                    device=device,
                )
            else:
                prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
                output = model.diffusion_generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=args.answer_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=0.0,
                    alg="confidence_threshold",
                    alg_temp=0.0,
                    threshold=args.threshold,
                    block_length=args.block_length,
                    dual_cache=True,
                )
                if isinstance(output, tuple):
                    output = output[0]
                sequences = output.sequences if hasattr(output, "sequences") else output
                answer_ids = sequences[0, prompt.shape[1] :].tolist()
        finally:
            forward_stats = counter.close()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        prediction = decode_answer(tokenizer, answer_ids)
        answers = [str(a) for a in example.get("answers", [])]
        score, metric = score_prediction(args.task, prediction, answers)
        record = {
            "task": args.task,
            "rank": int(args.rank),
            "index": int(index),
            "example_id": example.get("_id", index),
            "question": example.get("input", ""),
            "prediction": prediction,
            "answers": answers,
            "score": score,
            "metric": metric,
            "length": example.get("length"),
            "prompt_metadata": prompt_metadata,
            "seconds": float(seconds),
            "model_variant": args.model_variant,
            "cache_mode": (
                "official_dualcache_rightpad32k"
                if args.pad_to_physical
                else "official_dualcache"
            ),
            "physical_length": int(args.physical_length),
            "actual_sequence_length": int(
                args.physical_length
                if args.pad_to_physical
                else len(prompt_ids) + args.answer_tokens
            ),
            "prompt_tokens": len(prompt_ids),
            "pad_token_id": int(pad_id),
            "pad_is_eos": int(pad_id) == int(tokenizer.eos_token_id),
            "answer_tokens": int(args.answer_tokens),
            "block_length": int(args.block_length),
            "threshold": float(args.threshold),
            "bootstrap": "official_confidence_threshold",
            "dual_cache": True,
            "full_prefills": forward_stats.full_prefills,
            "cached_forwards": forward_stats.cached_forwards,
            "routing_calls": 0,
            "recomputed_tokens": forward_stats.recomputed_tokens,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "fallback_count": 0,
            "raw_answer_ids": answer_ids,
            "raw_decoded": tokenizer.decode(answer_ids, skip_special_tokens=False),
            "eos_offset": (
                answer_ids.index(tokenizer.eos_token_id)
                if tokenizer.eos_token_id in answer_ids
                else None
            ),
        }
        append_jsonl_fsync(output_path, record)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "score": score,
                    "seconds": seconds,
                    "seq_len": record["actual_sequence_length"],
                    "empty": not str(prediction).strip(),
                    "eos_offset": record["eos_offset"],
                    "full_prefills": forward_stats.full_prefills,
                    "cached_forwards": forward_stats.cached_forwards,
                    "prediction": prediction[:120],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    shard_records = [
        record
        for record in load_resumable_jsonl(output_path)
        if int(record["index"]) in set(assigned)
    ]
    summary = {
        "rank": args.rank,
        "assigned": len(assigned),
        "completed": len(shard_records),
        "model_variant": args.model_variant,
        **eos_summary(shard_records),
    }
    (output_dir / f"rank-{args.rank}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
