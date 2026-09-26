"""Train Dream + naive HiLS on answer-span denoising.

This is the first-version training entrypoint discussed for dLLM + HiLS:
Dream-style shifted prediction, answer-only complementary masks, 3 SWA + 1 HiLS
layer pattern, and top-k retrieval per query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from dream_dllm_hils.attention import install_dream_sparse_attention
from dream_dllm_hils.data import (
    AnswerSpanDataset,
    AnswerSpanDenoisingCollator,
    build_synthetic_records,
    load_jsonl_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/ma-user/work/models/Dream-v0-Base-7B")
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/dream-dllm-hils-answer-span")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hils_interleave", type=int, default=4)
    parser.add_argument("--local_window", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--hils_topk", type=int, default=16)
    parser.add_argument(
        "--hils_backend",
        type=str,
        default="kernel_bidir",
        choices=["torch_bidir", "chunk_kernel_bidir", "kernel_bidir"],
    )
    parser.add_argument("--no_kernel_fallback", action="store_true")
    parser.add_argument("--t_min", type=float, default=0.2)
    parser.add_argument("--t_max", type=float, default=0.8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_records", type=int, default=None)
    parser.add_argument("--synthetic_size", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()

    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
    return args


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.lora_r <= 0:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is required for LoRA training. Install it in the active env with: pip install peft"
        ) from exc

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "entropy_bias_scale" in name:
            param.requires_grad = True
    return model


def build_dataset(args: argparse.Namespace, tokenizer) -> AnswerSpanDataset:
    if args.train_jsonl:
        records = load_jsonl_records(args.train_jsonl, limit=args.limit_records)
    else:
        records = build_synthetic_records(size=args.synthetic_size)
    if not records:
        raise ValueError("No training records were loaded")
    return AnswerSpanDataset(records, tokenizer=tokenizer, max_length=args.max_length)


def infinite_loader(loader: Iterable[Dict[str, torch.Tensor]]):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(model, tokenizer, output_dir: str, step: int) -> None:
    path = Path(output_dir) / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    extra_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and "entropy_bias_scale" in name
    }
    if extra_state:
        torch.save(extra_state, path / "hils_extra_trainable.pt")
    tokenizer.save_pretrained(path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype)

    mask_token_id = getattr(tokenizer, "mask_token_id", None) or getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("Could not find Dream mask_token_id")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = int(model.config.eos_token_id)

    plan = install_dream_sparse_attention(
        model,
        interleave=args.hils_interleave,
        local_window=args.local_window,
        chunk_size=args.chunk_size,
        topk=args.hils_topk,
        backend=args.hils_backend,
        allow_kernel_fallback=not args.no_kernel_fallback,
    )
    model = apply_lora(model, args)
    model.to(device)
    model.train()
    model.config.use_cache = False

    dataset = build_dataset(args, tokenizer)
    collator = AnswerSpanDenoisingCollator(
        mask_token_id=int(mask_token_id),
        pad_token_id=int(pad_token_id),
        lmk_token_id=int(mask_token_id),
        chunk_size=args.chunk_size,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = infinite_loader(loader)

    trainable, total = count_trainable(model)
    print(f"HiLS layers: {plan.hils_layers}")
    print(
        f"SWA layers: {len(plan.sliding_window_layers)}; topk={args.hils_topk}; "
        f"chunk_size={args.chunk_size}; backend={args.hils_backend}; "
        f"kernel_fallback={not args.no_kernel_fallback}"
    )
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.4%})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for step in range(1, args.max_steps + 1):
        for _ in range(args.gradient_accumulation_steps):
            batch = move_to_device(next(batches), device)
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    use_cache=False,
                )
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running += float(loss.detach().cpu())

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={running:.4f} lr={scheduler.get_last_lr()[0]:.6g}")
            running = 0.0
        if args.save_steps > 0 and step % args.save_steps == 0:
            save_checkpoint(model, tokenizer, args.output_dir, step)

    save_checkpoint(model, tokenizer, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
