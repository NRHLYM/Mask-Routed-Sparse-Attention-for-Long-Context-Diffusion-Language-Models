#!/usr/bin/env python3
"""Train-distribution teacher audit: Dolma packs + s2-sync RULER view.

Snapshot dense-500 Q/K first, then load the student LoRA.  Compare frozen
dense branch recall against the live hard top-32 router on labeled answer
rows.  Do not use random-vocab haystacks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dream_dllm_hils.data import (  # noqa: E402
    RULER_VIEW_ID_BASE,
    FullTextComplementaryCollator,
    RulerDenoisingSynthesizer,
)
from dream_dllm_hils.full_dense_teacher import snapshot_frozen_dense_qk  # noqa: E402
from dream_dllm_hils.packed_corpus import DreamPackedCorpus  # noqa: E402
from dream_dllm_hils.train_fulltext import (  # noqa: E402
    _build_model_and_tokenizer,
    _landmark_inputs_embeds,
    _set_seed,
    resolve_landmark_token_id,
    parse_args as parse_training_args,
)
from scripts.dream_dllm_hils.eval_longbench_mfen import (  # noqa: E402
    configure_eval_trainables,
    load_trainables,
)

RECALL_GAP = 0.10
UNIFORM_RATIO = 4.0
LOCAL_GAP = 0.10


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.bool()
    safe = logits.float().masked_fill(~valid, float("-inf"))
    any_valid = valid.any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, safe, torch.zeros_like(safe))
    out = torch.softmax(safe, dim=-1)
    return torch.where(any_valid, out, torch.zeros_like(out)) * valid


def _slice_ruler(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    view_ids = batch["view_ids"]
    keep = (view_ids >= int(RULER_VIEW_ID_BASE)).nonzero(as_tuple=False).flatten()
    if keep.numel() != 1:
        raise RuntimeError(f"expected one RULER view, got {int(keep.numel())}")
    index = int(keep[0])
    out = {}
    for key, value in batch.items():
        out[key] = value[index : index + 1] if torch.is_tensor(value) else value
    return out


def _install_capture(model, store: dict) -> tuple:
    import dream_dllm_hils.attention as attention
    import dream_dllm_hils.routing as routing

    orig_route = routing.route_topk_g7
    orig_fuse = attention._route_weights_torch
    orig_proj = {}

    def wrapped_route(*args, **kwargs):
        result = orig_route(*args, **kwargs)
        store["indices"].append(result[0].detach())
        return result

    def wrapped_fuse(*args, **kwargs):
        result = orig_fuse(*args, **kwargs)
        store["remote"].append(result[0].detach())
        store["local"].append(result[1].detach())
        return result

    routing.route_topk_g7 = wrapped_route
    attention._route_weights_torch = wrapped_fuse
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    for layer in core.model.layers:
        attn = layer.self_attn
        if attn.__class__.__name__ != "KernelDreamFullHiLSAttention":
            continue
        orig_proj[id(attn)] = attn._project_qkv_blhd

        def make_proj(module, original):
            def wrapped(hidden_states, position_ids, position_embeddings):
                q, k, v = original(hidden_states, position_ids, position_embeddings)
                module._audit_live_q = q.detach()
                module._audit_live_k = k.detach()
                if hasattr(module, "frozen_dense_q_weight"):
                    tq, tk = module._project_qk_blhd_frozen_dense(
                        hidden_states, position_ids, position_embeddings
                    )
                    module._audit_teacher_q = tq.detach()
                    module._audit_teacher_k = tk.detach()
                return q, k, v

            return wrapped

        attn._project_qkv_blhd = make_proj(attn, attn._project_qkv_blhd)
    return orig_route, orig_fuse, orig_proj


def _restore_capture(model, orig_route, orig_fuse, orig_proj) -> None:
    import dream_dllm_hils.attention as attention
    import dream_dllm_hils.routing as routing

    routing.route_topk_g7 = orig_route
    attention._route_weights_torch = orig_fuse
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    for layer in core.model.layers:
        attn = layer.self_attn
        if id(attn) in orig_proj:
            attn._project_qkv_blhd = orig_proj[id(attn)]


def _collator(args, tokenizer) -> FullTextComplementaryCollator:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return FullTextComplementaryCollator(
        mask_token_id=int(tokenizer.mask_token_id),
        pad_token_id=int(pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        lmk_token_id=int(resolve_landmark_token_id(args, tokenizer)),
        chunk_size=int(args.chunk_size),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        seed=int(args.seed),
        ruler_mix_ratio=0.0,
        ruler_synthesizer=RulerDenoisingSynthesizer(
            tokenizer, task_ids=tuple(int(t) for t in args.ruler_task_ids)
        ),
        ruler_every_step=True,
        insert_landmarks=True,
    )


def _query_rows(batch: dict[str, torch.Tensor], chunk_size: int) -> torch.Tensor:
    valid = batch["attention_mask"].bool().clone()
    valid[:, chunk_size - 1 :: chunk_size] = False
    labeled = valid & batch["labels"].ne(-100)
    rows = torch.where(labeled[0])[0]
    if rows.numel() == 0:
        raise RuntimeError("RULER view has no labeled query positions")
    return rows.unsqueeze(0)


def _layer_metrics(layer, batch, positions, chunk_size, local_window, indices, remote_w, local_w) -> dict[str, float]:
    live_q = layer._audit_live_q
    live_k = layer._audit_live_k
    teacher_q = layer._audit_teacher_q
    teacher_k = layer._audit_teacher_k
    bsz, length, h_q, dim = live_q.shape
    h_kv = live_k.shape[2]
    groups = h_q // h_kv
    rows = positions.shape[1]
    bi = torch.arange(bsz, device=live_q.device)[:, None]
    live_k_hq = live_k.repeat_interleave(groups, dim=2)
    teacher_k_hq = teacher_k.repeat_interleave(groups, dim=2)
    q_rows = live_q[bi, positions]
    tq_rows = teacher_q[bi, positions]
    scale = 1.0 / math.sqrt(dim)
    with torch.autocast(live_q.device.type, enabled=False):
        student_scores = torch.einsum(
            "brhd,bnhd->brhn", q_rows.float(), live_k_hq.float()
        ) * scale
        teacher_scores = torch.einsum(
            "brhd,bnhd->brhn", tq_rows.float(), teacher_k_hq.float()
        ) * scale

    key_ids = torch.arange(length, device=live_q.device)
    real_key = ((key_ids + 1).remainder(int(chunk_size)) != 0)[None, :]
    key_valid = batch["attention_mask"].bool()
    segments = batch["segment_ids"]
    query_segments = segments[bi, positions]
    valid = key_valid[:, None, :] & real_key[:, None, :]
    valid = valid & (segments[:, None, :] == query_segments[:, :, None])
    teacher_prob = _masked_softmax(teacher_scores, valid[:, :, None, :])
    n_valid = valid.float().sum(-1).clamp_min(1.0)

    gold_chunks = batch["route_evidence_chunks"].bool()[0]
    chunk_ids = key_ids // int(chunk_size)
    gold_token = gold_chunks[chunk_ids][None, :] & real_key
    gold = gold_token[:, None, :] & valid
    n_gold = gold.float().sum(-1).clamp_min(1.0).unsqueeze(-1)
    teacher_gold = (teacher_prob * gold[:, :, None, :]).sum(-1)
    per_token = teacher_gold / n_gold
    uniform = (1.0 / n_valid).unsqueeze(-1)
    teacher_nll = -(teacher_gold.clamp_min(1e-12).log())

    n_chunks = length // int(chunk_size)
    chunk_mass = teacher_prob.reshape(bsz, rows, h_q, n_chunks, int(chunk_size)).sum(-1)
    gold_chunk = gold_chunks[None, None, None, :].expand(bsz, rows, h_q, n_chunks)
    teacher_top32 = chunk_mass.topk(min(32, n_chunks), dim=-1).indices
    teacher_top1 = chunk_mass.argmax(-1)
    teacher_top32_hit = gold_chunk.gather(-1, teacher_top32).any(-1)
    teacher_top1_hit = gold_chunk.gather(-1, teacher_top1.unsqueeze(-1)).squeeze(-1)

    selected = indices[bi, positions].long()
    student_hit = torch.zeros(bsz, rows, h_kv, device=live_q.device, dtype=torch.bool)
    for kv in range(h_kv):
        idx = selected[:, :, kv]
        ok = idx >= 0
        safe = idx.clamp(min=0, max=n_chunks - 1)
        student_hit[:, :, kv] = (gold_chunks[safe] & ok).any(-1)
    student_hit_hq = student_hit.repeat_interleave(groups, dim=2)

    left = ((positions - int(local_window)).clamp_min(0) // int(chunk_size)) * int(
        chunk_size
    )
    right = (
        ((positions + int(local_window)).clamp_max(length - 1) // int(chunk_size)) + 1
    ) * int(chunk_size)
    right = right.clamp_max(length)
    local_struct = (key_ids[None, None, :] >= left[:, :, None]) & (
        key_ids[None, None, :] < right[:, :, None]
    )
    local_valid = valid & local_struct
    support_chunks = torch.zeros(
        bsz, rows, n_chunks, device=live_q.device, dtype=torch.bool
    )
    flat_idx = selected.reshape(bsz, rows, -1)
    flat_ok = selected.reshape(bsz, rows, -1) >= 0
    support_chunks.scatter_(2, flat_idx.clamp(min=0), flat_ok)
    support = valid & (
        local_struct | support_chunks[:, :, chunk_ids]
    )
    coverage = (teacher_prob * support[:, :, None, :]).sum(-1)
    teacher_local = (teacher_prob * local_valid[:, :, None, :]).sum(-1)
    student_local = local_w[bi, positions].float()
    idx_hq = selected.repeat_interleave(groups, dim=2)
    rw = remote_w[bi, positions].float()
    needle_w = torch.zeros(bsz, rows, h_q, device=live_q.device)
    for head in range(h_q):
        hit = gold_chunks[idx_hq[:, :, head].clamp(min=0)] & (idx_hq[:, :, head] >= 0)
        needle_w[:, :, head] = (rw[:, :, head] * hit).sum(-1)

    gold_in_local = (local_valid & gold).any(-1)
    return {
        "n_rows": float(rows * h_q),
        "n_gold_tokens": float(n_gold.mean()),
        "n_valid_keys": float(n_valid.mean()),
        "teacher_top1_hit": float(teacher_top1_hit.float().mean()),
        "teacher_top32_hit": float(teacher_top32_hit.float().mean()),
        "student_top32_hit": float(student_hit_hq.float().mean()),
        "teacher_gold_mass": float(teacher_gold.mean()),
        "teacher_gold_nll": float(teacher_nll.mean()),
        "gold_over_uniform": float((per_token / uniform).mean()),
        "teacher_support_coverage": float(coverage.mean()),
        "teacher_local_mass": float(teacher_local.mean()),
        "student_local_mass": float(student_local.mean()),
        "student_w_needle": float(needle_w.mean()),
        "local_evidence_frac": float(gold_in_local.float().mean()),
        "task_id": float(int(batch["view_ids"].reshape(-1)[0]) - int(RULER_VIEW_ID_BASE)),
    }


def _mean(store: dict[str, list[float]]) -> dict[str, float]:
    out = {}
    for key, values in store.items():
        if key in {"n_rows", "task_id"}:
            continue
        out[key] = float(sum(values) / max(len(values), 1))
    out["n_rows"] = float(sum(store.get("n_rows", [0.0])))
    return out


def _verdict(summary: dict[str, float]) -> dict[str, object]:
    teacher = summary["teacher_top32_hit"]
    student = summary["student_top32_hit"]
    peaky = summary["gold_over_uniform"] >= UNIFORM_RATIO
    recall_clear = teacher >= student + RECALL_GAP
    more_local = summary["student_local_mass"] >= summary["teacher_local_mass"] + LOCAL_GAP
    if (not recall_clear) or (not peaky):
        decision = "abandon_dense_needle_teacher"
        reason = (
            "teacher gold-chunk recall is not clearly above student top-32"
            if not recall_clear
            else "teacher gold token mass is still near uniform"
        )
        next_step = "gold_chunk_ce"
    elif more_local:
        decision = "try_branch_kl"
        reason = "teacher retrieves the needle chunk more than the router; fusion is more local"
        next_step = "branch_kl_on_s2sync"
    else:
        decision = "inconclusive_fusion"
        reason = "teacher chunk recall is higher and student local is not clearly worse"
        next_step = "inspect_last_layer_w"
    return {
        "decision": decision,
        "reason": reason,
        "next_step": next_step,
        "recall_gap": teacher - student,
        "teacher_peaky": peaky,
        "thresholds": {
            "recall_gap": RECALL_GAP,
            "uniform_ratio": UNIFORM_RATIO,
            "local_gap": LOCAL_GAP,
        },
    }


@torch.inference_mode()
def main() -> int:
    cli = parse_cli()
    args = parse_training_args(
        ["--config", cli.training_config, "--no_gradient_checkpointing"]
    )
    _set_seed(cli.seed)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, tokenizer, plan = _build_model_and_tokenizer(args, device)
    configure_eval_trainables(model, args)
    n_teacher = snapshot_frozen_dense_qk(model)
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    for layer in core.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "frozen_dense_q_weight"):
            continue
        attn.frozen_dense_q_weight.data = attn.frozen_dense_q_weight.data.to(
            attn.q_proj.weight.dtype
        )
        attn.frozen_dense_k_weight.data = attn.frozen_dense_k_weight.data.to(
            attn.k_proj.weight.dtype
        )
    n_loaded = load_trainables(model, Path(cli.checkpoint))
    model.eval()
    print(
        f"dense_teacher_layers={n_teacher} student_tensors={n_loaded} "
        f"hils={list(plan.hils_layers)}",
        flush=True,
    )

    expected_slots = (int(args.max_length) // int(args.chunk_size)) * (
        int(args.chunk_size) - 1
    )
    corpus = DreamPackedCorpus(
        args.corpus_bin,
        args.corpus_meta,
        expected_real_slots=expected_slots,
    )
    collator = _collator(args, tokenizer)
    store = {"indices": [], "remote": [], "local": []}
    orig = _install_capture(model, store)
    aggregate = defaultdict(list)
    per_layer = []
    try:
        for sample_id in range(int(cli.num_samples)):
            store["indices"].clear()
            store["remote"].clear()
            store["local"].clear()
            example = corpus[sample_id]
            example["sample_id"] = sample_id
            batch = _slice_ruler(collator([example]))
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            positions = _query_rows(batch, int(args.chunk_size))
            ids, inputs_embeds = _landmark_inputs_embeds(
                model, batch["input_ids"], batch["landmark_mask"]
            )
            model(
                input_ids=ids,
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                labels=batch["labels"],
                use_cache=False,
            )
            hils_attns = [
                (index, layer.self_attn)
                for index, layer in enumerate(core.model.layers)
                if layer.self_attn.__class__.__name__ == "KernelDreamFullHiLSAttention"
            ]
            if len(store["indices"]) != len(hils_attns) or len(store["remote"]) != len(hils_attns):
                raise RuntimeError(
                    f"capture mismatch: route={len(store['indices'])} "
                    f"fuse={len(store['remote'])} hils={len(hils_attns)}"
                )
            for (index, attn), indices, remote_w, local_w in zip(
                hils_attns, store["indices"], store["remote"], store["local"]
            ):
                metrics = _layer_metrics(
                    attn,
                    batch,
                    positions,
                    int(args.chunk_size),
                    int(args.local_window),
                    indices,
                    remote_w,
                    local_w,
                )
                for key, value in metrics.items():
                    aggregate[key].append(value)
                row = {"sample_id": sample_id, "layer": index, **metrics}
                per_layer.append(row)
                print(
                    f"sample={sample_id} layer={index} task={int(metrics['task_id'])} "
                    f"T32={metrics['teacher_top32_hit']:.3f} "
                    f"S32={metrics['student_top32_hit']:.3f} "
                    f"gold={metrics['teacher_gold_mass']:.4f} "
                    f"peak={metrics['gold_over_uniform']:.2f} "
                    f"cov={metrics['teacher_support_coverage']:.3f} "
                    f"Tw={metrics['teacher_local_mass']:.3f} "
                    f"Sw={metrics['student_local_mass']:.3f}",
                    flush=True,
                )
            torch.cuda.empty_cache()
    finally:
        _restore_capture(model, *orig)

    summary = _mean(aggregate)
    last_rows = [row for row in per_layer if row["layer"] == max(plan.hils_layers)]
    metric_keys = [
        key
        for key in last_rows[0]
        if key not in {"sample_id", "layer", "task_id"}
    ]
    last = _mean({key: [row[key] for row in last_rows] for key in metric_keys})
    verdict = _verdict(last)
    payload = {
        "checkpoint": str(cli.checkpoint),
        "dense_initialize_from": str(args.initialize_from),
        "corpus_bin": str(args.corpus_bin),
        "num_samples": int(cli.num_samples),
        "hils_layers": list(plan.hils_layers),
        "local_window": int(args.local_window),
        "summary_all_hils": summary,
        "summary_last_hils": last,
        "verdict": verdict,
        "layers": per_layer,
    }
    out = Path(cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary_last_hils": last, "verdict": verdict}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
