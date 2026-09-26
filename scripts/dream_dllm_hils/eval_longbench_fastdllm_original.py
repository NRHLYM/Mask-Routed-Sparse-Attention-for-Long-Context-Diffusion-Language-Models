#!/usr/bin/env python3
"""Evaluate original Dream with the official Fast-dLLM v1 DualCache runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from dream_dllm_hils.longbench_eval import (
    append_jsonl_fsync,
    load_resumable_jsonl,
    merge_evaluation_shards,
    qa_f1_score,
    shard_indices,
    truncate_prompt_parts,
)


DEFAULT_DATA = (
    "/home/ma-user/work/ParallelComp_official/datasets/LongBench/"
    "multifieldqa_en.jsonl"
)
DEFAULT_PROMPTS = (
    "/home/ma-user/work/ParallelComp_official/longbench_config/"
    "dataset2prompt_raw.json"
)
DEFAULT_MODEL = "/home/ma-user/work/models/Dream-v0-Base-7B"
DEFAULT_FASTDLLM = "/home/ma-user/work/Fast-dLLM/v1/dream"
DEFAULT_WRAPPER_DIR = "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval"
DEFAULT_OUTPUT = (
    "outputs/dream-hils-dolma3-8k-pilot/"
    "longbench_mfen_fastdllm_original_2k"
)
MODEL_VARIANT = "dream_fastdllm_v1_original_2k"


@dataclass(frozen=True)
class OriginalForwardStats:
    full_prefills: int
    cached_forwards: int
    recomputed_tokens: int


def is_full_prefill_length(query_length: int, *, block_length: int) -> bool:
    if query_length <= 0 or block_length <= 0:
        raise ValueError("query_length and block_length must be positive")
    return query_length > block_length


class _ForwardCounter:
    def __init__(self, model: torch.nn.Module, block_length: int) -> None:
        self.block_length = int(block_length)
        self.full_prefills = 0
        self.cached_forwards = 0
        self.recomputed_tokens = 0
        self._handle = model.register_forward_pre_hook(
            self._record,
            with_kwargs=True,
        )

    def _record(self, module, args, kwargs) -> None:
        del module
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return
        self.recomputed_tokens += input_ids.numel()
        if is_full_prefill_length(
            input_ids.shape[1],
            block_length=self.block_length,
        ):
            self.full_prefills += 1
        else:
            self.cached_forwards += 1

    def close(self) -> OriginalForwardStats:
        self._handle.remove()
        return OriginalForwardStats(
            self.full_prefills,
            self.cached_forwards,
            self.recomputed_tokens,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--fastdllm_dream_dir", default=DEFAULT_FASTDLLM)
    parser.add_argument("--wrapper_dir", default=DEFAULT_WRAPPER_DIR)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--prompt_config", default=DEFAULT_PROMPTS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--dual_cache", dest="dual_cache", action="store_true")
    parser.add_argument("--no_dual_cache", dest="dual_cache", action="store_false")
    parser.set_defaults(dual_cache=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    return parser


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prompt_parts(
    template: str,
    example: dict[str, object],
) -> tuple[str, str, str]:
    sentinel = "__LONGBENCH_CONTEXT_SENTINEL__"
    rendered = template.format(
        context=sentinel,
        input=example.get("input", ""),
    )
    if rendered.count(sentinel) != 1:
        raise ValueError("multifieldqa_en template must contain one context slot")
    prefix, query = rendered.split(sentinel)
    return prefix, str(example.get("context", "")), query


def tokenize_prompt(
    tokenizer,
    template: str,
    example: dict[str, object],
    max_prompt_tokens: int,
) -> tuple[list[int], dict[str, int | str]]:
    prefix, context, query = _prompt_parts(template, example)
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids
    return truncate_prompt_parts(
        prefix_ids=encode(prefix),
        context_ids=encode(context),
        query_ids=encode(query),
        max_prompt_tokens=max_prompt_tokens,
        bos_token_id=tokenizer.bos_token_id,
    )


def make_record(
    *,
    args: argparse.Namespace,
    rank: int,
    index: int,
    example: dict[str, object],
    prediction: str,
    prompt_metadata: dict[str, int | str],
    seconds: float,
    forward_stats: OriginalForwardStats,
    peak_memory_bytes: int,
) -> dict[str, object]:
    answers = [str(answer) for answer in example.get("answers", [])]
    return {
        "task": "multifieldqa_en",
        "rank": int(rank),
        "index": int(index),
        "example_id": example.get("_id", index),
        "question": example.get("input", ""),
        "prediction": prediction,
        "answers": answers,
        "score": qa_f1_score(prediction, answers),
        "length": example.get("length"),
        "prompt_metadata": prompt_metadata,
        "seconds": float(seconds),
        "model_variant": MODEL_VARIANT,
        "cache_mode": "official_dualcache",
        "max_length": int(args.max_length),
        "total_length": int(prompt_metadata["prompt_tokens_after_truncation"])
        + int(args.answer_tokens),
        "answer_tokens": int(args.answer_tokens),
        "block_length": int(args.block_length),
        "threshold": float(args.threshold),
        "dual_cache": bool(args.dual_cache),
        "rope_scale_factor": float(args.rope_scale_factor),
        "full_prefills": forward_stats.full_prefills,
        "cached_forwards": forward_stats.cached_forwards,
        "routing_calls": 0,
        "recomputed_tokens": forward_stats.recomputed_tokens,
        "peak_memory_bytes": int(peak_memory_bytes),
        "fallback_count": 0,
    }


def _load_fastdllm(args: argparse.Namespace):
    wrapper_dir = str(Path(args.wrapper_dir).resolve())
    if wrapper_dir not in sys.path:
        sys.path.insert(0, wrapper_dir)
    from fastdllm_v1_model import FastDLLMv1Config, FastDLLMv1Dream

    return FastDLLMv1Dream(
        FastDLLMv1Config(
            fastdllm_dream_dir=args.fastdllm_dream_dir,
            pretrained=args.model_path,
            device=args.device,
            dtype="bfloat16",
            max_new_tokens=args.answer_tokens,
            max_length=args.max_length,
            block_length=args.block_length,
            threshold=args.threshold,
            dual_cache=args.dual_cache,
            add_bos_token=False,
            truncation_strategy="head_tail",
            rope_scale_factor=args.rope_scale_factor,
        )
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_length != 2048 or args.answer_tokens != 64:
        raise ValueError("original evaluation is fixed to 2048 total / 64 answer tokens")
    if args.block_length != 32 or args.threshold != 0.9:
        raise ValueError("original evaluation is fixed to block=32, threshold=0.9")
    if not args.dual_cache or args.rope_scale_factor != 1.0:
        raise ValueError("original evaluation requires DualCache and no RoPE scaling")
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    examples = _read_jsonl(args.data)
    if args.merge:
        metrics = merge_evaluation_shards(
            args.output_dir,
            total_examples=len(examples),
            expected_variant=MODEL_VARIANT,
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{args.rank}.jsonl"
    completed = {
        int(record["index"]) for record in load_resumable_jsonl(output_path)
    }
    assigned = shard_indices(len(examples), args.rank, args.world_size)
    if args.limit > 0:
        assigned = assigned[: args.limit]
    pending = [index for index in assigned if index not in completed]
    templates = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))
    template = templates["multifieldqa_en"]
    runtime = _load_fastdllm(args)
    device = runtime.device

    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(assigned),
                "pending": len(pending),
                "model_variant": MODEL_VARIANT,
                "device": str(device),
            }
        ),
        flush=True,
    )
    for ordinal, index in enumerate(pending, start=1):
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_prompt(
            runtime.tokenizer,
            template,
            example,
            args.max_length - args.answer_tokens,
        )
        prompt_tensor = torch.tensor(
            prompt_ids,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        torch.cuda.reset_peak_memory_stats(device)
        counter = _ForwardCounter(runtime.model, args.block_length)
        started = time.perf_counter()
        try:
            prediction = runtime.generate_one_ids(prompt_tensor).strip()
        finally:
            forward_stats = counter.close()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        record = make_record(
            args=args,
            rank=args.rank,
            index=index,
            example=example,
            prediction=prediction,
            prompt_metadata=prompt_metadata,
            seconds=seconds,
            forward_stats=forward_stats,
            peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        )
        append_jsonl_fsync(output_path, record)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "score": record["score"],
                    "seconds": seconds,
                    "full_prefills": forward_stats.full_prefills,
                    "cached_forwards": forward_stats.cached_forwards,
                    "prediction": prediction,
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
        "model_variant": MODEL_VARIANT,
        "qa_f1": (
            100.0 * sum(float(record["score"]) for record in shard_records) / len(shard_records)
            if shard_records
            else 0.0
        ),
    }
    (output_dir / f"rank-{args.rank}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
