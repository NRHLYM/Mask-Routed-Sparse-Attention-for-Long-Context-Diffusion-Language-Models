#!/usr/bin/env python3
"""Run resumable Dream HiLS evaluation on SCBench-VT."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch

from dream_dllm_hils.longbench_eval import (
    GenerationLayout,
    append_jsonl_fsync,
    build_generation_layout,
    load_resumable_jsonl,
    shard_indices,
)
from dream_dllm_hils.scbench_vt_eval import (
    candidate_vars_from_context,
    extract_answer_vars,
    score_prediction_text,
    tokenize_scbench_prompt,
)
from dream_dllm_hils.train_fulltext import (
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _landmark_inputs_embeds,
    _set_seed,
    parse_args as parse_training_args,
    resolve_landmark_token_id,
)


DEFAULT_DATA = (
    "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/"
    "data_scbench/scbench_vt.jsonl"
)
DEFAULT_CHECKPOINT = "outputs/dream-hils-hisa512-dense21-dolma3-2k/step-500"
DEFAULT_OUTPUT = "outputs/dream-hils-hisa512-dense21-dolma3-2k/scbench_vt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_config",
        default=(
            "configs/dream_dllm_hils/"
            "dolma3_2k_hils_hisa512_dense21_dual_gpu.json"
        ),
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("oracle_2k", "long_context"),
        default="oracle_2k",
        help=(
            "oracle_2k keeps compact VAR assignments; long_context uses the "
            "raw SCBench context without head-tail truncation."
        ),
    )
    parser.add_argument(
        "--oracle_scope",
        choices=("all_assignments", "gold_chain"),
        default="all_assignments",
        help="Which assignment statements to keep in oracle_2k mode.",
    )
    parser.add_argument(
        "--long_truncation",
        choices=("assignment_windows", "error", "head", "tail"),
        default="assignment_windows",
        help=(
            "Policy when raw long_context does not fit physical_length. "
            "assignment_windows keeps windows around every VAR assignment."
        ),
    )
    parser.add_argument(
        "--assignment_window_chars",
        type=int,
        default=256,
        help=(
            "Initial characters kept before and after each VAR assignment "
            "when long_truncation=assignment_windows. The adapter shrinks "
            "this value if needed to fit the token budget."
        ),
    )
    parser.add_argument(
        "--require_full_evidence",
        action="store_true",
        help="Fail an example if not every gold assignment survives the prompt.",
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_length", type=int, default=2048)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum examples globally; zero evaluates all examples.",
    )
    parser.add_argument(
        "--dry_run_prompts",
        action="store_true",
        help="Tokenize and score evidence retention without loading the model.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Validate and merge complete rank shards instead of evaluating.",
    )
    return parser


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty list")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_trainables(model: torch.nn.Module, checkpoint: Path) -> int:
    saved = torch.load(
        checkpoint / "trainable_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise ValueError(
            f"checkpoint trainables mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(parameter.device, parameter.dtype))
    return len(current)


def model_variant_from_training_args(args: argparse.Namespace, length: int) -> str:
    if args.attention_mode == "hils" and args.hils_token_budget > 0:
        return (
            f"dream_hils_{args.hils_token_policy}_"
            f"{args.hils_token_budget}_{length}"
        )
    return f"dream_{args.attention_mode}_{length}"


def decode_answer(tokenizer, token_ids: list[int]) -> str:
    if tokenizer.eos_token_id in token_ids:
        token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    for stop in ("\n\n\n", "\n\nQuestion:", "\n\nMemorize"):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text


@torch.inference_mode()
def denoise_answer(
    *,
    model: torch.nn.Module,
    tokenizer,
    layout: GenerationLayout,
    landmark_token_id: int,
    steps: int,
) -> tuple[str, int, float]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    device = next(model.parameters()).device
    input_ids = layout.input_ids.unsqueeze(0).to(device)
    attention_mask = layout.attention_mask.unsqueeze(0).to(device)
    position_ids = layout.position_ids.unsqueeze(0).to(device)
    answer_positions = layout.answer_positions.to(device)
    predictor_positions = layout.predictor_positions.to(device)
    landmark_positions = layout.landmark_positions.to(device)
    remaining = torch.ones(
        answer_positions.numel(), device=device, dtype=torch.bool
    )

    started = time.perf_counter()
    completed_steps = 0
    for step in range(steps):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            model_input_ids, inputs_embeds = _landmark_inputs_embeds(
                model,
                input_ids,
                input_ids.eq(int(landmark_token_id)),
            )
            outputs = model(
                input_ids=model_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        candidate_logits = outputs.logits[0, predictor_positions].float()
        candidate_logits[:, int(tokenizer.mask_token_id)] = float("-inf")
        active = torch.where(remaining)[0]
        probabilities = torch.softmax(candidate_logits[active], dim=-1)
        confidence, candidate_ids = probabilities.max(dim=-1)
        transfer_count = math.ceil(active.numel() / (steps - step))
        selected_active = torch.topk(confidence, k=transfer_count).indices
        selected = active[selected_active]
        input_ids[0, answer_positions[selected]] = candidate_ids[selected_active]
        remaining[selected] = False
        completed_steps = step + 1

        if not torch.all(
            input_ids[0, landmark_positions] == int(landmark_token_id)
        ):
            raise RuntimeError("a fixed landmark token was modified")
        if not remaining.any():
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if remaining.any():
        raise RuntimeError(f"{int(remaining.sum())} answer masks remain")
    generated = input_ids[0, answer_positions].tolist()
    return decode_answer(tokenizer, generated), completed_steps, elapsed


def prepare_examples(args: argparse.Namespace) -> list[dict[str, object]]:
    examples = read_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("no SCBench-VT examples selected")
    return examples


def merge_scbench_shards(
    output_dir: str | Path,
    *,
    total_examples: int,
    expected_variant: str | None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    shard_paths = sorted(output_dir.glob("rank-*.jsonl"))
    if not shard_paths:
        raise ValueError(f"no rank shards found in {output_dir}")

    records: list[dict[str, object]] = []
    for shard_path in shard_paths:
        records.extend(load_resumable_jsonl(shard_path))

    indices = [int(record["index"]) for record in records]
    duplicates = sorted(
        index for index, count in Counter(indices).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate evaluation indices: {duplicates[:8]}")
    expected_indices = set(range(total_examples))
    actual_indices = set(indices)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"incomplete evaluation: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    records.sort(key=lambda record: int(record["index"]))
    variants = {str(record.get("model_variant", "")) for record in records}
    cache_modes = {str(record.get("cache_mode", "")) for record in records}
    if len(variants) != 1 or len(cache_modes) != 1:
        raise ValueError("all merged records must share model_variant/cache_mode")
    variant = next(iter(variants))
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"expected model_variant={expected_variant}, got {variant}"
        )

    fallback_max = max(int(record.get("fallback_count", 0)) for record in records)
    if fallback_max:
        raise ValueError(f"kernel fallback observed: maximum={fallback_max}")
    seconds = [float(record["seconds"]) for record in records]
    rank_seconds: dict[int, float] = {}
    for record in records:
        rank = int(record.get("rank", 0))
        rank_seconds[rank] = rank_seconds.get(rank, 0.0) + float(record["seconds"])

    truncation_modes = Counter(
        str(record.get("prompt_metadata", {}).get("truncation_mode", "unknown"))
        for record in records
    )
    evidence_full = [
        bool(record.get("prompt_metadata", {}).get("evidence_full", False))
        for record in records
    ]
    metrics: dict[str, object] = {
        "examples": len(records),
        "model_variant": variant,
        "cache_mode": next(iter(cache_modes)),
        "accuracy": 100.0
        * sum(float(record["accuracy"]) for record in records)
        / len(records),
        "precision": 100.0
        * sum(float(record["precision"]) for record in records)
        / len(records),
        "recall": 100.0
        * sum(float(record["recall"]) for record in records)
        / len(records),
        "set_f1": 100.0
        * sum(float(record["set_f1"]) for record in records)
        / len(records),
        "evidence_full_rate": 100.0 * sum(evidence_full) / len(evidence_full),
        "mean_seconds": sum(seconds) / len(seconds),
        "p50_seconds": percentile(seconds, 0.50),
        "p95_seconds": percentile(seconds, 0.95),
        "wall_seconds_estimate": max(rank_seconds.values()),
        "mean_full_prefills": sum(
            int(record.get("full_prefills", 0)) for record in records
        )
        / len(records),
        "mean_routing_calls": sum(
            int(record.get("routing_calls", 0)) for record in records
        )
        / len(records),
        "mean_recomputed_tokens": sum(
            int(record.get("recomputed_tokens", 0)) for record in records
        )
        / len(records),
        "peak_memory_bytes_max": max(
            int(record.get("peak_memory_bytes", 0)) for record in records
        ),
        "fallback_max": fallback_max,
        "truncation_modes": dict(sorted(truncation_modes.items())),
    }
    write_text_atomic(
        output_dir / "merged.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    write_text_atomic(
        output_dir / "metrics.json",
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    return metrics


def run_dry_prompt_check(
    *,
    args: argparse.Namespace,
    tokenizer,
    examples: list[dict[str, object]],
    assigned: list[int],
    max_prompt_tokens: int,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dry-run-rank-{args.rank}.jsonl"
    if output_path.exists():
        output_path.unlink()

    records: list[dict[str, object]] = []
    for index in assigned:
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_scbench_prompt(
            tokenizer,
            example,
            mode=args.mode,
            max_prompt_tokens=max_prompt_tokens,
            bos_token_id=tokenizer.bos_token_id,
            oracle_scope=args.oracle_scope,
            long_truncation=args.long_truncation,
            assignment_window_chars=args.assignment_window_chars,
        )
        record = {
            "dry_run": True,
            "index": index,
            "example_id": example.get("id", example.get("_id", index)),
            "row_id": example.get("row_id"),
            "turn_id": example.get("turn_id"),
            "answers": extract_answer_vars(example),
            "prompt_tokens": len(prompt_ids),
            "prompt_metadata": prompt_metadata,
        }
        append_jsonl_fsync(output_path, record)
        records.append(record)

    evidence_full = [
        bool(record["prompt_metadata"].get("evidence_full", False))
        for record in records
    ]
    summary = {
        "dry_run": True,
        "rank": args.rank,
        "world_size": args.world_size,
        "examples": len(records),
        "mode": args.mode,
        "physical_length": args.physical_length,
        "max_prompt_tokens": max_prompt_tokens,
        "mean_prompt_tokens": sum(int(record["prompt_tokens"]) for record in records)
        / len(records),
        "evidence_full_rate": 100.0 * sum(evidence_full) / len(evidence_full),
    }
    write_text_atomic(
        output_dir / f"dry-run-rank-{args.rank}.summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run_prompts and args.rank is None:
        args.rank = 0
        args.world_size = 1
    if args.physical_length % args.chunk_size:
        raise ValueError("physical_length must be divisible by chunk_size")
    if args.answer_tokens <= 0 or args.steps <= 0:
        raise ValueError("answer_tokens and steps must be positive")
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")

    examples = prepare_examples(args)
    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    if training_args.max_length != args.physical_length:
        raise ValueError(
            "training max_length does not match evaluation physical_length"
        )
    if training_args.chunk_size != args.chunk_size:
        raise ValueError("training and evaluation chunk sizes differ")
    model_variant = model_variant_from_training_args(
        training_args,
        args.physical_length,
    )

    if args.merge:
        metrics = merge_scbench_shards(
            args.output_dir,
            total_examples=len(examples),
            expected_variant=model_variant,
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return

    real_slots = (args.physical_length // args.chunk_size) * (
        args.chunk_size - 1
    )
    max_prompt_tokens = real_slots - args.answer_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("answer_tokens leave no room for the prompt")

    assigned = shard_indices(len(examples), args.rank, args.world_size)

    if args.dry_run_prompts:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            training_args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        run_dry_prompt_check(
            args=args,
            tokenizer=tokenizer,
            examples=examples,
            assigned=assigned,
            max_prompt_tokens=max_prompt_tokens,
        )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{args.rank}.jsonl"
    summary_path = output_dir / f"rank-{args.rank}.summary.json"
    existing = load_resumable_jsonl(output_path)
    completed = {int(record["index"]) for record in existing}
    pending = [index for index in assigned if index not in completed]

    _set_seed(args.seed + args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    trainable_tensors = load_trainables(model, Path(args.checkpoint))
    model.eval()
    landmark_token_id = resolve_landmark_token_id(training_args, tokenizer)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(assigned),
                "completed": len(completed & set(assigned)),
                "pending": len(pending),
                "device": str(device),
                "mode": args.mode,
                "steps": args.steps,
                "answer_tokens": args.answer_tokens,
                "landmark_token_id": landmark_token_id,
                "lmk_token_mode": training_args.lmk_token_mode,
                "fast_dllm": False,
                "use_cache": False,
                "model_variant": model_variant,
                "cache_mode": "no_cache_exact",
                "hils_layers": plan.hils_layers,
                "trainable_tensors": trainable_tensors,
            }
        ),
        flush=True,
    )

    for ordinal, index in enumerate(pending, start=1):
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_scbench_prompt(
            tokenizer,
            example,
            mode=args.mode,
            max_prompt_tokens=max_prompt_tokens,
            bos_token_id=tokenizer.bos_token_id,
            oracle_scope=args.oracle_scope,
            long_truncation=args.long_truncation,
            assignment_window_chars=args.assignment_window_chars,
        )
        missing_evidence = list(
            prompt_metadata.get("evidence_missing_vars_after_prompt", [])
        )
        if missing_evidence and (
            args.require_full_evidence or args.mode == "oracle_2k"
        ):
            raise ValueError(
                f"example {index} is missing oracle evidence: "
                f"{missing_evidence}"
            )

        layout = build_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.physical_length,
            chunk_size=args.chunk_size,
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
            landmark_token_id=int(landmark_token_id),
        )
        prediction, nfe, seconds = denoise_answer(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            landmark_token_id=int(landmark_token_id),
            steps=args.steps,
        )
        gold_vars = extract_answer_vars(example)
        candidates = candidate_vars_from_context(str(example.get("context", "")))
        scores = score_prediction_text(
            prediction,
            gold_vars,
            candidate_vars=candidates,
        )
        fallback_count = _kernel_fallback_count(model)
        if fallback_count:
            raise RuntimeError(f"kernel fallback count became {fallback_count}")

        record: dict[str, object] = {
            "task": "scbench_vt",
            "mode": args.mode,
            "rank": args.rank,
            "index": index,
            "example_id": example.get("id", example.get("_id", index)),
            "row_id": example.get("row_id"),
            "turn_id": example.get("turn_id"),
            "question": example.get("input", ""),
            "prediction": prediction,
            "prediction_vars": scores["prediction_vars"],
            "answers": gold_vars,
            "score": scores["set_f1"],
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "set_f1": scores["set_f1"],
            "true_positive": scores["true_positive"],
            "false_positive": scores["false_positive"],
            "false_negative": scores["false_negative"],
            "predicted_count": scores["predicted_count"],
            "gold_count": scores["gold_count"],
            "length": example.get("length"),
            "prompt_metadata": prompt_metadata,
            "physical_length": args.physical_length,
            "answer_tokens": args.answer_tokens,
            "nfe": nfe,
            "seconds": seconds,
            "fallback_count": fallback_count,
            "fast_dllm": False,
            "use_cache": False,
            "model_variant": model_variant,
            "cache_mode": "no_cache_exact",
            "full_prefills": nfe,
            "cached_forwards": 0,
            "routing_calls": nfe if plan.hils_layers else 0,
            "recomputed_tokens": args.physical_length * nfe,
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
        }
        append_jsonl_fsync(output_path, record)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "set_f1": record["set_f1"],
                    "accuracy": record["accuracy"],
                    "precision": record["precision"],
                    "recall": record["recall"],
                    "prediction_vars": record["prediction_vars"],
                    "answers": gold_vars,
                    "seconds": seconds,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    records = load_resumable_jsonl(output_path)
    assigned_set = set(assigned)
    shard_records = [
        record for record in records if int(record["index"]) in assigned_set
    ]
    if not shard_records:
        raise ValueError("no completed records for this shard")
    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned": len(assigned),
        "completed": len(shard_records),
        "accuracy": 100
        * sum(float(record["accuracy"]) for record in shard_records)
        / len(shard_records),
        "precision": 100
        * sum(float(record["precision"]) for record in shard_records)
        / len(shard_records),
        "recall": 100
        * sum(float(record["recall"]) for record in shard_records)
        / len(shard_records),
        "set_f1": 100
        * sum(float(record["set_f1"]) for record in shard_records)
        / len(shard_records),
        "mean_seconds": sum(float(record["seconds"]) for record in shard_records)
        / len(shard_records),
        "fallback_max": max(
            int(record["fallback_count"]) for record in shard_records
        ),
        "fast_dllm": False,
        "use_cache": False,
        "mode": args.mode,
        "steps": args.steps,
        "answer_tokens": args.answer_tokens,
        "model_variant": model_variant,
        "cache_mode": "no_cache_exact",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
