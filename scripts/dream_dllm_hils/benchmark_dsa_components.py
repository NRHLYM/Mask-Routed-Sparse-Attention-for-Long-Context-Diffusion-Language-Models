"""Microbenchmark the correctness-first Dream DSA PyTorch components."""

from __future__ import annotations

import argparse
import math
import time

import torch

from dream_dllm_hils.dsa_attention import selected_token_attention_reference


def _bench_cuda(fn, *, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=32768)
    parser.add_argument("--query_block", type=int, default=64)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--query_heads", type=int, default=28)
    parser.add_argument("--kv_heads", type=int, default=4)
    parser.add_argument("--index_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    blocks = math.ceil(args.sequence_length / args.query_block)

    index_q = torch.randn(
        batch,
        args.query_block,
        args.index_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    index_k = torch.randn(
        batch,
        args.sequence_length,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    index_w = torch.randn(
        batch,
        args.query_block,
        args.index_heads,
        device=device,
        dtype=torch.float32,
    )
    allowed = torch.ones(
        batch,
        args.query_block,
        args.sequence_length,
        device=device,
        dtype=torch.bool,
    )
    allowed[..., 63::64] = False

    def score_only() -> torch.Tensor:
        per_head = torch.einsum("bqhd,bkd->bqhk", index_q, index_k)
        return (
            torch.relu(per_head.float()) * index_w.unsqueeze(-1)
        ).sum(dim=2) * (args.head_dim**-0.5)

    def score_and_topk() -> torch.Tensor:
        scores = score_only()
        scores.masked_fill_(~allowed, float("-inf"))
        return torch.topk(
            scores, k=args.topk, dim=-1, sorted=False
        ).indices

    with torch.no_grad():
        score_ms = _bench_cuda(score_only, warmup=3, repetitions=10)
        route_ms = _bench_cuda(score_and_topk, warmup=3, repetitions=10)

    q = torch.randn(
        batch,
        args.query_block,
        args.query_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch,
        args.sequence_length,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    indices = torch.randint(
        0,
        args.sequence_length,
        (batch, args.query_block, args.topk),
        device=device,
    )
    valid = torch.ones_like(indices, dtype=torch.bool)

    def attention_forward() -> torch.Tensor:
        return selected_token_attention_reference(q, k, v, indices, valid)

    with torch.no_grad():
        attention_fwd_ms = _bench_cuda(
            attention_forward, warmup=3, repetitions=10
        )

    q_grad = q.detach().requires_grad_(True)
    k_grad = k.detach().requires_grad_(True)
    v_grad = v.detach().requires_grad_(True)
    grad_output = torch.randn_like(q_grad)

    def attention_forward_backward() -> None:
        q_grad.grad = None
        k_grad.grad = None
        v_grad.grad = None
        output = selected_token_attention_reference(
            q_grad, k_grad, v_grad, indices, valid
        )
        output.backward(grad_output)

    attention_fwd_bwd_ms = _bench_cuda(
        attention_forward_backward, warmup=1, repetitions=3
    )

    print(
        {
            "device": torch.cuda.get_device_name(),
            "sequence_length": args.sequence_length,
            "query_block": args.query_block,
            "topk": args.topk,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "index_heads": args.index_heads,
            "score_ms_per_query_block": score_ms,
            "topk_incremental_ms_per_query_block": route_ms - score_ms,
            "route_ms_per_query_block": route_ms,
            "attention_fwd_ms_per_query_block": attention_fwd_ms,
            "attention_fwd_bwd_ms_per_query_block": attention_fwd_bwd_ms,
            "estimated_route_seconds_per_layer": route_ms * blocks / 1000.0,
            "estimated_attention_fwd_seconds_per_layer": (
                attention_fwd_ms * blocks / 1000.0
            ),
            "estimated_attention_fwd_bwd_seconds_per_layer": (
                attention_fwd_bwd_ms * blocks / 1000.0
            ),
        }
    )


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print({"wall_seconds": time.perf_counter() - started})
