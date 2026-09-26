#!/usr/bin/env python3
"""16k goldspan S-N: answer-slot routing, needle tokens in support, QK mass, gold logit rank.

This is Fast-dLLM's first prefill (answer spans are MASK). token_budget=0 means
the whole selected chunk is in KV; min_tokens_per_chunk never drops digits.
"""
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


def support_positions(
    query_pos: int,
    chunks: set[int],
    *,
    chunk_size: int,
    local_window: int,
    seq_len: int,
) -> set[int]:
    keep = set(range(max(0, query_pos - local_window), min(seq_len, query_pos + local_window + 1)))
    for chunk in chunks:
        start = int(chunk) * chunk_size
        keep.update(range(start, min(seq_len, start + chunk_size)))
    return keep


def needle_in_support(needle_phys: list[int], keep: set[int]) -> float:
    if not needle_phys:
        return 0.0
    return sum(int(p in keep) for p in needle_phys) / len(needle_phys)


def qk_needle_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
    query_positions: list[int],
    needle_phys: list[int],
    *,
    chunk_size: int,
    local_window: int,
) -> float:
    """Mean head/query softmax mass on needle tokens, among local+selected support."""

    seq_len = int(q.shape[1])
    scale = float(q.shape[-1]) ** -0.5
    h_q = int(q.shape[2])
    h_kv = int(k.shape[2])
    groups = h_q // h_kv
    needle = set(needle_phys)
    masses: list[float] = []
    for qp in query_positions:
        chunks = selected_chunks(indices, qp)
        keep = sorted(
            support_positions(
                qp,
                chunks,
                chunk_size=chunk_size,
                local_window=local_window,
                seq_len=seq_len,
            )
        )
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


def gold_first_token(gold_ids: list[int], eos_id: int | None) -> int | None:
    for tok in gold_ids:
        if eos_id is not None and int(tok) == int(eos_id):
            continue
        return int(tok)
    return None


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
    local_window = int(training_args.local_window)
    token_budget = int(getattr(training_args, "hils_token_budget", 0) or 0)
    hils_layers = list(plan.hils_layers)
    slots = ruler_eval.text_slot_budget(args.max_seq_len, hils=True, chunk_size=chunk_size)
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    import dream_dllm_hils.routing as routing

    original_route = routing.route_topk_g7
    original_proj = KernelDreamFullHiLSAttention._project_qkv_blhd
    last_qk: dict[str, torch.Tensor] = {}
    captures: list[dict] = []

    def wrapped_proj(self, *pargs, **pkwargs):
        q, k, v = original_proj(self, *pargs, **pkwargs)
        last_qk["q"] = q
        last_qk["k"] = k
        return q, k, v

    def wrapped_route(*pargs, **pkwargs):
        result = original_route(*pargs, **pkwargs)
        indices = result[0]
        captures.append(
            {
                "indices": indices.detach().to("cpu"),
                "q": last_qk["q"],
                "k": last_qk["k"],
            }
        )
        return result

    KernelDreamFullHiLSAttention._project_qkv_blhd = wrapped_proj
    routing.route_topk_g7 = wrapped_route

    layer_answer_hits = defaultdict(int)
    layer_support = defaultdict(float)
    layer_mass = defaultdict(float)
    union_answer_hits = 0
    all7_answer_hits = 0
    last_answer_hits = 0
    gold_top1 = 0
    gold_rank_sum = 0.0
    gold_n = 0
    samples = []

    try:
        for index in range(int(args.limit)):
            captures.clear()
            prompt_ids, needle_text, quiz_text, answer_tokens, _gold = pack_sn(
                tokenizer, synthesizer, slots, index
            )
            # Recover gold ids from the same pack: prompt + gold = synthesizer clean.
            base = ruler_eval.haystack_token_ids(tokenizer, None, slots)
            vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
            base[0] = (int(base[0]) + int(index) * 1315423911) % vocab
            clean, target_mask, _ = synthesizer.synthesize_with_evidence(
                torch.as_tensor(base, dtype=torch.long), task_id=0
            )
            answer_start = int(target_mask.nonzero(as_tuple=False).flatten()[0])
            gold_ids = clean[answer_start:].tolist()
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
            outputs = forward_with_lmk(model, layout, device)
            fallback = _kernel_fallback_count(model)
            if fallback:
                raise RuntimeError(f"kernel fallback count became {fallback}")
            if len(captures) != len(hils_layers):
                raise RuntimeError(
                    f"expected {len(hils_layers)} HiLS route calls, got {len(captures)}"
                )

            per_layer = {}
            layer_hit_flags = []
            for layer_id, cap in zip(hils_layers, captures):
                indices = cap["indices"]
                token_hits = []
                support_fracs = []
                for qp in query_phys:
                    chunks = selected_chunks(indices, qp)
                    hit = bool(chunks & {p // chunk_size for p in needle_phys})
                    token_hits.append(hit)
                    keep = support_positions(
                        qp,
                        chunks,
                        chunk_size=chunk_size,
                        local_window=local_window,
                        seq_len=args.max_seq_len,
                    )
                    support_fracs.append(needle_in_support(needle_phys, keep))
                sample_hit = any(token_hits)
                layer_hit_flags.append(sample_hit)
                layer_answer_hits[layer_id] += int(sample_hit)
                mean_support = sum(support_fracs) / max(len(support_fracs), 1)
                layer_support[layer_id] += mean_support
                mass = qk_needle_mass(
                    cap["q"],
                    cap["k"],
                    indices,
                    query_phys,
                    needle_phys,
                    chunk_size=chunk_size,
                    local_window=local_window,
                )
                layer_mass[layer_id] += mass
                per_layer[str(layer_id)] = {
                    "answer_any_hit": sample_hit,
                    "answer_token_hit_rate": sum(token_hits) / max(len(token_hits), 1),
                    "needle_token_in_support": mean_support,
                    "needle_qk_mass": mass,
                }
                del cap["q"], cap["k"]

            union_hit = any(layer_hit_flags)
            all7 = all(layer_hit_flags)
            last_hit = bool(layer_hit_flags[-1])
            union_answer_hits += int(union_hit)
            all7_answer_hits += int(all7)
            last_answer_hits += int(last_hit)

            first_gold = gold_first_token(gold_ids, eos_id)
            first_pos = int(layout.predictor_positions[0])
            logits = outputs.logits[0, first_pos].float()
            rank = None
            top1 = None
            if first_gold is not None:
                gold_n += 1
                order = torch.argsort(logits, descending=True)
                rank = int((order == first_gold).nonzero(as_tuple=False)[0]) + 1
                top1 = int(order[0].item()) == int(first_gold)
                gold_top1 += int(top1)
                gold_rank_sum += rank
            row = {
                "index": index,
                "union_answer_any_hit": union_hit,
                "all7_answer_hit": all7,
                "last_layer_answer_hit": last_hit,
                "gold_first_token": first_gold,
                "gold_first_top1": top1,
                "gold_first_rank": rank,
                "layers": per_layer,
            }
            samples.append(row)
            print(json.dumps(row), flush=True)
            del outputs
            torch.cuda.empty_cache()
    finally:
        routing.route_topk_g7 = original_route
        KernelDreamFullHiLSAttention._project_qkv_blhd = original_proj

    n = max(int(args.limit), 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "n": int(args.limit),
        "trainable_tensors": n_loaded,
        "hils_layers": hils_layers,
        "token_budget": token_budget,
        "token_keep": "full_selected_chunk" if token_budget <= 0 else "refined",
        "union_answer_any_hit": union_answer_hits / n,
        "all7_answer_hit": all7_answer_hits / n,
        "last_layer_answer_hit": last_answer_hits / n,
        "gold_first_top1": (gold_top1 / gold_n) if gold_n else None,
        "gold_first_mean_rank": (gold_rank_sum / gold_n) if gold_n else None,
        "per_layer": {
            str(layer): {
                "answer_any_hit": layer_answer_hits[layer] / n,
                "needle_token_in_support": layer_support[layer] / n,
                "needle_qk_mass": layer_mass[layer] / n,
            }
            for layer in hils_layers
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    print("ANSWER_READ_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
