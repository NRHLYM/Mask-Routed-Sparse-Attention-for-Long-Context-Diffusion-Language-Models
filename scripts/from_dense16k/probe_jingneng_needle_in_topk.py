#!/usr/bin/env python3
"""Prefill HiLS routing on 16k goldspan S-N: does quiz top-k contain the needle chunk?"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_jingneng_official_ruler as ruler_eval  # noqa: E402

from dream_dllm_hils.longbench_eval import (  # noqa: E402
    build_generation_layout,
    physical_text_positions,
)
from dream_dllm_hils.train_fulltext import (  # noqa: E402
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _landmark_inputs_embeds,
    _set_seed,
    parse_args as parse_training_args,
)
from dream_dllm_hils.data import RulerDenoisingSynthesizer  # noqa: E402
from scripts.dream_dllm_hils.eval_longbench_mfen import (  # noqa: E402
    configure_eval_trainables,
    load_trainables,
)

SN_QUESTION = (
    " What is the special magic number for long-context mentioned "
    "in the provided text? Answer: "
)
MKMQ_QUESTION_PREFIX = " What are all the special magic numbers for "
TASK_IDS = {"sn": 0, "mkmq": 1}


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=("sn", "mkmq"), default="sn")
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    n = len(needle)
    if n <= 0 or n > len(haystack):
        return -1
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start : start + n] == needle:
            return start
    return -1


def evidence_runs(positions: list[int]) -> list[list[int]]:
    if not positions:
        return []
    ordered = sorted(positions)
    runs = [[ordered[0]]]
    for pos in ordered[1:]:
        if pos == runs[-1][-1] + 1:
            runs[-1].append(pos)
        else:
            runs.append([pos])
    return runs


def pack_ruler(tokenizer, synthesizer, text_slots: int, sample_index: int, task: str):
    task_id = TASK_IDS[task]
    base = ruler_eval.haystack_token_ids(tokenizer, None, text_slots)
    vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
    base[0] = (int(base[0]) + int(sample_index) * 1315423911) % vocab
    clean, target_mask, evidence = synthesizer.synthesize_with_evidence(
        torch.as_tensor(base, dtype=torch.long),
        task_id=task_id,
    )
    answer_idx = target_mask.nonzero(as_tuple=False).flatten()
    answer_start = int(answer_idx[0].item())
    prompt_ids = clean[:answer_start].tolist()
    gold_ids = clean[answer_start:].tolist()
    if task == "sn":
        question_ids = synthesizer._encode(SN_QUESTION).tolist()
    else:
        rng = synthesizer._rng(np.asarray(base, dtype=np.int64), salt=1)
        names = synthesizer._unique_names(rng, 6)
        queried = " and ".join(names[:2])
        question_ids = synthesizer._encode(
            f" What are all the special magic numbers for {queried} "
            "mentioned in the provided text?. Answer: "
        ).tolist()
    quiz_start = len(prompt_ids) - len(question_ids)
    if quiz_start < 0 or prompt_ids[quiz_start:] != question_ids:
        raise RuntimeError(f"{task} quiz suffix mismatch; packing drifted from synthesizer")
    needle_text = [
        int(i) for i in evidence[:answer_start].nonzero(as_tuple=False).flatten().tolist()
    ]
    if not needle_text:
        raise RuntimeError(f"{task} pack has no needle evidence tokens")
    quiz_text = list(range(quiz_start, len(prompt_ids)))
    answer_tokens = ruler_eval.padded_answer_tokens(len(gold_ids), block_length=32)
    eos_id = tokenizer.eos_token_id
    gold_decode = list(gold_ids)
    if eos_id in gold_decode:
        gold_decode = gold_decode[: gold_decode.index(eos_id)]
    gold_text = tokenizer.decode(gold_decode, skip_special_tokens=True).strip()
    return prompt_ids, needle_text, quiz_text, answer_tokens, gold_text


def pack_sn(tokenizer, synthesizer, text_slots: int, sample_index: int):
    return pack_ruler(tokenizer, synthesizer, text_slots, sample_index, "sn")


def text_to_physical(text_positions: list[int], real_slots: int, chunk_size: int) -> list[int]:
    mapping = physical_text_positions(real_slots, chunk_size)
    return [int(mapping[int(t)]) for t in text_positions]


def landmark_mask_from_layout(layout) -> torch.Tensor:
    mask = torch.zeros_like(layout.input_ids, dtype=torch.bool)
    if layout.landmark_positions.numel():
        mask[layout.landmark_positions.to(dtype=torch.long)] = True
    return mask


def forward_with_lmk(model, layout, device: torch.device):
    """Match training: mask_type LMK adds dream_hils_lmk_type_embed on landmarks."""
    input_ids = layout.input_ids.unsqueeze(0).to(device)
    landmark_mask = landmark_mask_from_layout(layout).unsqueeze(0).to(device)
    ids, inputs_embeds = _landmark_inputs_embeds(model, input_ids, landmark_mask)
    return model(
        input_ids=ids,
        inputs_embeds=inputs_embeds,
        attention_mask=layout.attention_mask.unsqueeze(0).to(device),
        position_ids=layout.position_ids.unsqueeze(0).to(device),
        use_cache=False,
    )


def selected_chunks(indices: torch.Tensor, physical_pos: int) -> set[int]:
    row = indices[0, physical_pos]
    return {int(x) for x in row.reshape(-1).tolist() if x >= 0}


def official_records(checkpoint: str, task: str, max_seq_len: int) -> dict[int, dict]:
    root = Path(checkpoint).resolve().parent
    task_dir = "hils_sn" if task == "sn" else "hils_mkmq"
    sub = "ruler_probes_goldspan" if int(max_seq_len) <= 16384 else "ruler_probes"
    path = root / sub / f"len{max_seq_len}" / task_dir / "rank-0.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["index"])] = rec
    return out


def quiz_fusion_stats(indices, remote_weights, local_weight, quiz_index, needle_chunks):
    """Mean over quiz tokens / query heads. Rank 1 = heaviest selected remote slot.

    indices / weights are already sliced to quiz tokens on dim=1.
    """
    groups = remote_weights.shape[2] // indices.shape[2]
    idx_hq = indices.to(torch.long).repeat_interleave(groups, dim=2)
    needle = {int(c) for c in needle_chunks}
    w_n, w_l, w_o, ranks, hits = [], [], [], [], []
    for qp in quiz_index:
        idx = idx_hq[0, qp]
        rw = remote_weights[0, qp].float()
        lw = local_weight[0, qp].float()
        valid = idx >= 0
        is_needle = torch.zeros_like(idx, dtype=torch.bool)
        for chunk in needle:
            is_needle |= idx == chunk
        w_n.append(float((rw * is_needle).sum(-1).mean()))
        w_l.append(float(lw.mean()))
        other = rw.masked_fill(is_needle | ~valid, float("-inf"))
        finite = torch.isfinite(other)
        w_o.append(float(other.masked_fill(~finite, 0.0).amax(-1).mean()) if finite.any() else 0.0)
        hit = bool(is_needle.any())
        hits.append(hit)
        if not hit:
            continue
        head_ranks = []
        for head in range(idx.shape[0]):
            nmask = is_needle[head]
            if not bool(nmask.any()):
                continue
            needle_max = rw[head][nmask].max()
            better = (rw[head][valid[head]] > needle_max).sum()
            head_ranks.append(int(better.item()) + 1)
        if head_ranks:
            ranks.append(min(head_ranks))
    return {
        "w_needle": sum(w_n) / max(len(w_n), 1),
        "w_local": sum(w_l) / max(len(w_l), 1),
        "w_max_other": sum(w_o) / max(len(w_o), 1),
        "hit_frac": sum(hits) / max(len(hits), 1),
        "min_rank": min(ranks) if ranks else None,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
    }


def patch_route_and_fusion(route_store: list, fusion_store: list, keep):
    import dream_dllm_hils.attention as attention
    import dream_dllm_hils.routing as routing

    original_route = routing.route_topk_g7
    original_fuse = attention._route_weights_torch

    def _quiz(tensor):
        pos = keep["pos"]
        if pos is None:
            raise RuntimeError("quiz gather positions were not set")
        # Stay on GPU. .cpu() here device-syncs mid-Tilelang forward and
        # deadlocks after a few 64k samples.
        return tensor.index_select(1, pos.to(device=tensor.device)).detach()

    def wrapped_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        route_store.append(_quiz(result[0]))
        return result

    def wrapped_fuse(*args, **kwargs):
        result = original_fuse(*args, **kwargs)
        fusion_store.append((_quiz(result[0]), _quiz(result[1])))
        return result

    routing.route_topk_g7 = wrapped_route
    attention._route_weights_torch = wrapped_fuse
    return original_route, original_fuse


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
    hils_layers = list(plan.hils_layers)
    slots = ruler_eval.text_slot_budget(
        args.max_seq_len, hils=True, chunk_size=chunk_size
    )
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(TASK_IDS[args.task],))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    layer_token_hits = defaultdict(int)
    layer_sample_hits = defaultdict(int)
    layer_token_n = defaultdict(int)
    union_sample_hits = 0
    needle_local_samples = 0
    n_quiz_tokens = 0
    n_needle_chunks = 0
    samples = []

    official = official_records(args.checkpoint, args.task, args.max_seq_len)
    keep = {"pos": None}
    original_route, original_fuse = patch_route_and_fusion(store := [], fusion := [], keep)
    try:
        for index in range(int(args.limit)):
            store.clear()
            fusion.clear()
            prompt_ids, needle_text, quiz_text, answer_tokens, gold_text = pack_ruler(
                tokenizer, synthesizer, slots, index, args.task
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
            quiz_phys = text_to_physical(quiz_text, slots, chunk_size)
            evidence_chunk_sets = [
                {p // chunk_size for p in text_to_physical(run, slots, chunk_size)}
                for run in evidence_runs(needle_text)
            ]
            needle_chunks = set().union(*evidence_chunk_sets) if evidence_chunk_sets else set()
            n_needle_chunks += len(needle_chunks)
            n_evidence = len(evidence_chunk_sets)
            n_quiz_tokens += len(quiz_phys)
            needle_in_local = False
            for qp in quiz_phys:
                left = max(0, qp - local_window)
                right = min(args.max_seq_len - 1, qp + local_window)
                local_chunks = set(range(left // chunk_size, right // chunk_size + 1))
                if needle_chunks & local_chunks:
                    needle_in_local = True
                    break
            needle_local_samples += int(needle_in_local)
            quiz_index = list(range(len(quiz_phys)))
            keep["pos"] = torch.tensor(quiz_phys, device=device, dtype=torch.long)
            print(
                json.dumps({"progress": "begin", "index": index, "n_quiz": len(quiz_phys)}),
                flush=True,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                forward_with_lmk(model, layout, device)
            keep["pos"] = None
            store[:] = [t.detach().to("cpu") for t in store]
            fusion[:] = [
                (remote.detach().to("cpu"), local.detach().to("cpu"))
                for remote, local in fusion
            ]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            fallback = _kernel_fallback_count(model)
            if fallback:
                raise RuntimeError(f"kernel fallback count became {fallback}")
            if len(store) != len(hils_layers):
                raise RuntimeError(
                    f"expected {len(hils_layers)} HiLS route calls, got {len(store)}"
                )
            if len(fusion) != len(hils_layers):
                raise RuntimeError(
                    f"expected {len(hils_layers)} fusion calls, got {len(fusion)}"
                )

            def coverage(selected: set[int]) -> int:
                return sum(bool(selected & chunks) for chunks in evidence_chunk_sets)

            sample_union = False
            per_layer = {}
            for layer_id, indices, (remote_w, local_w) in zip(hils_layers, store, fusion):
                token_hits = [
                    bool(selected_chunks(indices, qp) & needle_chunks) for qp in quiz_index
                ]
                token_cov = [
                    coverage(selected_chunks(indices, qp)) for qp in quiz_index
                ]
                union_selected = set()
                for qp in quiz_index:
                    union_selected |= selected_chunks(indices, qp)
                n_hit = sum(token_hits)
                layer_token_hits[layer_id] += n_hit
                layer_token_n[layer_id] += len(token_hits)
                sample_hit = n_hit > 0
                layer_sample_hits[layer_id] += int(sample_hit)
                sample_union = sample_union or sample_hit
                fusion_stats = quiz_fusion_stats(
                    indices, remote_w, local_w, quiz_index, needle_chunks
                )
                group_fusion = [
                    quiz_fusion_stats(indices, remote_w, local_w, quiz_index, group)
                    for group in evidence_chunk_sets
                ]
                per_layer[str(layer_id)] = {
                    "quiz_token_hit_rate": n_hit / max(len(token_hits), 1),
                    "sample_any_hit": sample_hit,
                    "k_union": coverage(union_selected),
                    "k_max_token": max(token_cov) if token_cov else 0,
                    "k_mean_token": (
                        sum(token_cov) / max(len(token_cov), 1) if token_cov else 0.0
                    ),
                    **fusion_stats,
                    "per_evidence": group_fusion,
                }
            union_sample_hits += int(sample_union)
            layer_hits = [v["sample_any_hit"] for v in per_layer.values()]
            token_rates = [v["quiz_token_hit_rate"] for v in per_layer.values()]
            last_id = str(hils_layers[-1])
            last = per_layer[last_id]
            pred_rec = official.get(index)
            official_em = None
            if pred_rec is not None:
                refs = pred_rec.get("outputs") or []
                pred = str(pred_rec.get("prediction") or "")
                if refs:
                    official_em = sum(
                        1.0 if str(r).lower() in pred.lower() else 0.0 for r in refs
                    ) / len(refs)
            row = {
                "index": index,
                "task": args.task,
                "gold_text": gold_text,
                "needle_chunks": sorted(needle_chunks),
                "n_needle_chunks": len(needle_chunks),
                "n_evidence_groups": n_evidence,
                "evidence_chunks": [sorted(g) for g in evidence_chunk_sets],
                "needle_in_local_window": needle_in_local,
                "union_sample_any_hit": sample_union,
                "all_layers_hit": all(layer_hits),
                "last_layer_hit": bool(last["sample_any_hit"]),
                "last_layer_k_union": int(last["k_union"]),
                "last_layer_k_max_token": int(last["k_max_token"]),
                "last_layer_k_mean_token": float(last["k_mean_token"]),
                "last_layer_w_needle": float(last["w_needle"]),
                "last_layer_w_local": float(last["w_local"]),
                "last_layer_w_max_other": float(last["w_max_other"]),
                "last_layer_min_rank": last["min_rank"],
                "last_layer_mean_rank": last["mean_rank"],
                "official_prediction": None if pred_rec is None else pred_rec.get("prediction"),
                "official_em": official_em,
                "mean_quiz_token_hit_rate": sum(token_rates) / max(len(token_rates), 1),
                "layers": per_layer,
            }
            samples.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "layers"}),
                flush=True,
            )
    finally:
        import dream_dllm_hils.attention as attention
        import dream_dllm_hils.routing as routing

        routing.route_topk_g7 = original_route
        attention._route_weights_torch = original_fuse

    n = max(int(args.limit), 1)
    def k_frac(key, value):
        return sum(int(s[key]) == value for s in samples) / n

    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "n": int(args.limit),
        "trainable_tensors": n_loaded,
        "hils_layers": hils_layers,
        "mean_needle_chunks": n_needle_chunks / n,
        "mean_quiz_tokens": n_quiz_tokens / n,
        "needle_in_local_window_frac": needle_local_samples / n,
        "union_sample_any_hit": union_sample_hits / n,
        "all_layers_hit": sum(1 for s in samples if s["all_layers_hit"]) / n,
        "last_layer_hit": sum(1 for s in samples if s["last_layer_hit"]) / n,
        "last_layer_k_union_0": k_frac("last_layer_k_union", 0),
        "last_layer_k_union_1": k_frac("last_layer_k_union", 1),
        "last_layer_k_union_2": k_frac("last_layer_k_union", 2),
        "last_layer_k_max_token_0": k_frac("last_layer_k_max_token", 0),
        "last_layer_k_max_token_1": k_frac("last_layer_k_max_token", 1),
        "last_layer_k_max_token_2": k_frac("last_layer_k_max_token", 2),
        "last_layer_w_needle": sum(s["last_layer_w_needle"] for s in samples) / n,
        "last_layer_w_local": sum(s["last_layer_w_local"] for s in samples) / n,
        "last_layer_w_max_other": sum(s["last_layer_w_max_other"] for s in samples) / n,
        "last_layer_mean_rank_hits": (
            sum(s["last_layer_mean_rank"] for s in samples if s["last_layer_mean_rank"] is not None)
            / max(sum(s["last_layer_mean_rank"] is not None for s in samples), 1)
        ),
        "official_em_n": sum(s["official_em"] is not None for s in samples),
        "official_em": (
            sum(s["official_em"] for s in samples if s["official_em"] is not None)
            / max(sum(s["official_em"] is not None for s in samples), 1)
        ),
        "w_needle_on_official_ok": (
            sum(s["last_layer_w_needle"] for s in samples if s["official_em"] == 1.0)
            / max(sum(s["official_em"] == 1.0 for s in samples), 1)
        ),
        "w_needle_on_official_bad": (
            sum(
                s["last_layer_w_needle"]
                for s in samples
                if s["official_em"] is not None and s["official_em"] < 1.0
            )
            / max(
                sum(s["official_em"] is not None and s["official_em"] < 1.0 for s in samples),
                1,
            )
        ),
        "per_layer": {
            str(layer): {
                "quiz_token_hit_rate": layer_token_hits[layer]
                / max(layer_token_n[layer], 1),
                "sample_any_hit": layer_sample_hits[layer] / n,
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
    print("NEEDLE_IN_TOPK_DONE", flush=True)


if __name__ == "__main__":
    main()
