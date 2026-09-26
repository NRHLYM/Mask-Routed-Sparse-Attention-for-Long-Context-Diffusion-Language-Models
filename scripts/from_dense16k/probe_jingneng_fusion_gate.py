#!/usr/bin/env python3
"""Answer-slot HiLS fusion: local_weight vs remote-only needle QK mass."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_jingneng_official_ruler as ruler_eval  # noqa: E402
from probe_jingneng_needle_in_topk import (  # noqa: E402
    forward_with_lmk,
    pack_sn,
    selected_chunks,
    text_to_physical,
)

import dream_dllm_hils.attention as hils_attention  # noqa: E402
import dream_dllm_hils.routing as routing  # noqa: E402
from dream_dllm_hils.attention import KernelDreamFullHiLSAttention  # noqa: E402
from dream_dllm_hils.longbench_eval import build_generation_layout  # noqa: E402
from dream_dllm_hils.train_fulltext import (  # noqa: E402
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _set_seed,
    parse_args as parse_training_args,
)
from dream_dllm_hils.data import RulerDenoisingSynthesizer  # noqa: E402
from scripts.dream_dllm_hils.eval_longbench_mfen import (  # noqa: E402
    configure_eval_trainables,
    load_trainables,
)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def remote_chunk_positions(chunks: set[int], chunk_size: int, seq_len: int) -> list[int]:
    keep: list[int] = []
    for chunk in sorted(chunks):
        start = int(chunk) * chunk_size
        keep.extend(range(start, min(seq_len, start + chunk_size)))
    return keep


def remote_needle_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    query_positions: list[int],
    needle_phys: list[int],
    *,
    chunk_size: int,
) -> float:
    seq_len = int(q.shape[1])
    scale = float(q.shape[-1]) ** -0.5
    h_q = int(q.shape[2])
    h_kv = int(k.shape[2])
    groups = h_q // h_kv
    needle = set(needle_phys)
    masses: list[float] = []
    for qp in query_positions:
        keep = remote_chunk_positions(selected_chunks(indices, qp), chunk_size, seq_len)
        if not keep:
            masses.append(0.0)
            continue
        keep_t = torch.tensor(keep, device=q.device, dtype=torch.long)
        qq = q[0, qp].float()
        kk = k[0, keep_t].float().repeat_interleave(groups, dim=1)
        logits = torch.einsum("hd,thd->ht", qq, kk) * scale
        prob = torch.softmax(logits, dim=-1)
        needle_mask = torch.tensor([p in needle for p in keep], device=q.device)
        if bool(needle_mask.any()):
            masses.append(float(prob[:, needle_mask].sum(-1).mean()))
        else:
            masses.append(0.0)
    return sum(masses) / max(len(masses), 1)


def needle_chunk_remote_weight(
    remote_w: torch.Tensor,
    indices: torch.Tensor,
    query_positions: list[int],
    needle_phys: list[int],
    *,
    chunk_size: int,
) -> float:
    """Mean fusion weight on selected chunks that contain needle tokens.

    This uses the live per-chunk `remote_weights`, not a re-softmax over 2048 keys.
    """
    needle_chunks = {int(p) // int(chunk_size) for p in needle_phys}
    if indices.ndim != 4 or remote_w.ndim != 4:
        raise ValueError("indices and remote_w must be 4D")
    h_q = int(remote_w.shape[2])
    h_kv = int(indices.shape[2])
    groups = h_q // h_kv
    idx_hq = indices.to(torch.long).repeat_interleave(groups, dim=2)
    weights: list[float] = []
    for qp in query_positions:
        chosen = idx_hq[0, qp]
        mass = remote_w[0, qp].float()
        hit = torch.zeros((), dtype=mass.dtype)
        for head in range(h_q):
            match = torch.zeros((), dtype=mass.dtype)
            for slot, chunk in enumerate(chosen[head].tolist()):
                if int(chunk) in needle_chunks:
                    match = match + mass[head, slot]
            hit = hit + match
        weights.append(float(hit / h_q))
    return sum(weights) / max(len(weights), 1)


@torch.inference_mode()
def main() -> None:
    args = parse_cli()
    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    ruler_eval.apply_yarn(training_args, args.max_seq_len)
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    configure_eval_trainables(model, training_args)
    n_loaded = load_trainables(model, Path(args.checkpoint))
    model.eval()
    chunk_size = int(training_args.chunk_size)
    hils_layers = list(plan.hils_layers)
    slots = ruler_eval.text_slot_budget(args.max_seq_len, hils=True, chunk_size=chunk_size)
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    original_route = routing.route_topk_g7
    original_proj = KernelDreamFullHiLSAttention._project_qkv_blhd
    original_rw = hils_attention._route_weights_torch
    last_qk: dict[str, torch.Tensor] = {}
    route_caps: list[dict] = []
    weight_caps: list[torch.Tensor] = []

    def wrapped_proj(self, *pargs, **pkwargs):
        q, k, v = original_proj(self, *pargs, **pkwargs)
        last_qk["q"] = q
        last_qk["k"] = k
        return q, k, v

    def wrapped_route(*pargs, **pkwargs):
        result = original_route(*pargs, **pkwargs)
        route_caps.append(
            {
                "indices": result[0].detach().to("cpu"),
                "q": last_qk["q"],
                "k": last_qk["k"],
            }
        )
        return result

    def wrapped_rw(*pargs, **pkwargs):
        routed = original_rw(*pargs, **pkwargs)
        remote_w, local_w = routed[0], routed[1]
        weight_caps.append((remote_w.detach(), local_w.detach()))
        return routed

    KernelDreamFullHiLSAttention._project_qkv_blhd = wrapped_proj
    routing.route_topk_g7 = wrapped_route
    hils_attention._route_weights_torch = wrapped_rw

    layer_local = defaultdict(float)
    layer_remote_mass = defaultdict(float)
    layer_gated_mass = defaultdict(float)
    layer_needle_w = defaultdict(float)
    samples = []
    try:
        for index in range(int(args.limit)):
            route_caps.clear()
            weight_caps.clear()
            prompt_ids, needle_text, _quiz, answer_tokens, _gold = pack_sn(
                tokenizer, synthesizer, slots, index
            )
            layout = build_generation_layout(
                prompt_ids=prompt_ids,
                answer_tokens=answer_tokens,
                physical_length=args.max_seq_len,
                chunk_size=chunk_size,
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(pad_token_id),
                landmark_token_id=int(tokenizer.mask_token_id),
            )
            needle_phys = text_to_physical(needle_text, slots, chunk_size)
            query_phys = [int(x) for x in layout.predictor_positions.tolist()]
            forward_with_lmk(model, layout, device)
            fallback = _kernel_fallback_count(model)
            if fallback:
                raise RuntimeError(f"kernel fallback count became {fallback}")
            if len(route_caps) != len(hils_layers) or len(weight_caps) != len(hils_layers):
                raise RuntimeError(
                    f"layer mismatch route={len(route_caps)} weights={len(weight_caps)} "
                    f"expected={len(hils_layers)}"
                )
            per_layer = {}
            for layer_id, cap, (remote_w, local_w) in zip(
                hils_layers, route_caps, weight_caps
            ):
                pred = torch.tensor(query_phys, device=local_w.device)
                local_mean = float(local_w[0, pred].float().mean())
                remote_mass = remote_needle_mass(
                    cap["q"],
                    cap["k"],
                    cap["indices"],
                    query_phys,
                    needle_phys,
                    chunk_size=chunk_size,
                )
                needle_w = needle_chunk_remote_weight(
                    remote_w,
                    cap["indices"].to(remote_w.device),
                    query_phys,
                    needle_phys,
                    chunk_size=chunk_size,
                )
                proxy_gated = (1.0 - local_mean) * remote_mass
                layer_local[layer_id] += local_mean
                layer_remote_mass[layer_id] += remote_mass
                layer_gated_mass[layer_id] += proxy_gated
                layer_needle_w[layer_id] += needle_w
                per_layer[str(layer_id)] = {
                    "local_weight": local_mean,
                    "needle_chunk_remote_weight": needle_w,
                    "proxy_flat_needle_qk_mass": remote_mass,
                    "proxy_gated_flat_mass": proxy_gated,
                    "remote_needle_qk_mass": remote_mass,
                    "gated_needle_mass": proxy_gated,
                }
                del cap["q"], cap["k"]
            samples.append({"index": index, "layers": per_layer})
            print(json.dumps(samples[-1]), flush=True)
            torch.cuda.empty_cache()
    finally:
        routing.route_topk_g7 = original_route
        KernelDreamFullHiLSAttention._project_qkv_blhd = original_proj
        hils_attention._route_weights_torch = original_rw

    n = max(int(args.limit), 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "n": int(args.limit),
        "trainable_tensors": n_loaded,
        "hils_layers": hils_layers,
        "detach_fusion_weights": bool(getattr(training_args, "hils_detach_fusion_weights", False)),
        "query_positions": "predictor_positions",
        "lmk_input": "mask_type_type_embed",
        "mass_note": (
            "proxy_flat_needle_qk_mass resoftmaxes QK over selected-chunk tokens; "
            "proxy_gated_flat_mass multiplies mean (1-local) by that proxy; "
            "needle_chunk_remote_weight sums live per-chunk fusion weights on "
            "chunks that contain needle tokens."
        ),
        "per_layer": {
            str(layer): {
                "local_weight": layer_local[layer] / n,
                "needle_chunk_remote_weight": layer_needle_w[layer] / n,
                "proxy_flat_needle_qk_mass": layer_remote_mass[layer] / n,
                "proxy_gated_flat_mass": layer_gated_mass[layer] / n,
                "remote_needle_qk_mass": layer_remote_mass[layer] / n,
                "gated_needle_mass": layer_gated_mass[layer] / n,
            }
            for layer in hils_layers
        },
        "mean_local_weight": sum(layer_local[layer] for layer in hils_layers)
        / (n * len(hils_layers)),
        "mean_needle_chunk_remote_weight": sum(
            layer_needle_w[layer] for layer in hils_layers
        )
        / (n * len(hils_layers)),
        "mean_remote_needle_qk_mass": sum(layer_remote_mass[layer] for layer in hils_layers)
        / (n * len(hils_layers)),
        "mean_gated_needle_mass": sum(layer_gated_mass[layer] for layer in hils_layers)
        / (n * len(hils_layers)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    print("FUSION_GATE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
