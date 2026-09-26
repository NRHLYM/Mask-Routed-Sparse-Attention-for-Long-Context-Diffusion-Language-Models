#!/usr/bin/env python3
"""Count valid EOS keys in DSA top-k (pad excluded via attention_mask).

MFQA: first-answer predictor (last prompt token) at prefill.
Training packs: predictors of document-final EOS, plus mean over all valid queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dream_dllm_hils.data import FullTextComplementaryCollator
from dream_dllm_hils.dsa_attention import DreamDsaAttention
from dream_dllm_hils.longbench_eval import (
    build_generation_layout,
    physical_text_positions,
)
from dream_dllm_hils.packed_corpus import DreamPackedCorpus
from dream_dllm_hils.train_fulltext import (
    _build_model_and_tokenizer,
    _set_seed,
    parse_args as parse_training_args,
    resolve_landmark_token_id,
)
from scripts.dream_dllm_hils.eval_longbench_mfen import (
    load_trainables,
    tokenize_prompt,
)


def _summarize(
    ids: torch.Tensor, selected_valid: torch.Tensor, groups: dict[str, torch.Tensor]
) -> dict:
    kept = ids[selected_valid]
    n = int(kept.numel())
    out = {"n": n}
    if n == 0:
        for name in groups:
            out[name] = 0.0
            out[f"{name}_count"] = 0
        return out
    for name, mask in groups.items():
        hit = mask.index_select(0, kept)
        out[name] = float(hit.float().mean().item())
        out[f"{name}_count"] = int(hit.sum().item())
    return out


def _mean_stats(stats: list[dict], key: str) -> float | None:
    if not stats:
        return None
    return sum(float(row[key]) for row in stats) / len(stats)


def _install_capture(model) -> tuple[list[tuple[int, DreamDsaAttention]], dict]:
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    dsa_layers = [
        (idx, layer.self_attn)
        for idx, layer in enumerate(core.model.layers)
        if isinstance(layer.self_attn, DreamDsaAttention)
    ]
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _install(layer_idx: int, attn: DreamDsaAttention) -> None:
        original = attn.dsa_indexer.select

        def wrapped(*call_args, **kwargs):
            indices, valid = original(*call_args, **kwargs)
            captured[layer_idx] = (indices.detach(), valid.detach())
            return indices, valid

        attn.dsa_indexer.select = wrapped  # type: ignore[method-assign]

    for layer_idx, attn in dsa_layers:
        _install(layer_idx, attn)
    return dsa_layers, captured


def _groups(input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int, mask_id: int):
    valid = attention_mask.bool()
    return {
        "eos_valid": input_ids.eq(eos_id) & valid,
        "mask_id": input_ids.eq(mask_id) & valid,
        "pad_invalid": ~valid,
    }


def _forward(model, input_ids, attention_mask, position_ids):
    with torch.inference_mode(), torch.autocast(
        device_type=input_ids.device.type, dtype=torch.bfloat16
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )


def run_mfen(
    *,
    model,
    tokenizer,
    training_args,
    dsa_layers,
    captured,
    args,
    device,
) -> list[dict]:
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    landmark_id = resolve_landmark_token_id(training_args, tokenizer)
    examples = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    template = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))[args.task]
    real_slots = (args.physical_length // args.chunk_size) * (args.chunk_size - 1)
    max_prompt = real_slots - args.answer_tokens
    rows = []
    for index, example in enumerate(examples):
        captured.clear()
        prompt_ids, _ = tokenize_prompt(tokenizer, template, example, max_prompt)
        layout = build_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.physical_length,
            chunk_size=args.chunk_size,
            mask_token_id=mask_id,
            pad_token_id=int(pad_id),
            landmark_token_id=int(landmark_id),
        )
        input_ids = layout.input_ids.unsqueeze(0).to(device)
        attention_mask = layout.attention_mask.unsqueeze(0).to(device)
        groups = _groups(input_ids[0], attention_mask[0], eos_id, mask_id)
        n_eos = int(groups["eos_valid"].sum().item())
        predictor0 = int(layout.predictor_positions[0])
        _forward(
            model,
            input_ids,
            attention_mask,
            layout.position_ids.unsqueeze(0).to(device),
        )
        layer_rows = []
        for layer_idx, _attn in dsa_layers:
            indices, valid = captured[layer_idx]
            pred = _summarize(indices[0, predictor0], valid[0, predictor0], groups)
            layer_rows.append({"layer": layer_idx, "predictor0": pred})
        rows.append(
            {
                "index": index,
                "prompt_tokens": len(prompt_ids),
                "valid_tokens": int(attention_mask.sum().item()),
                "valid_eos_keys": n_eos,
                "layers": layer_rows,
            }
        )
        print("MFEN " + json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return rows


def run_pack(
    *,
    model,
    tokenizer,
    training_args,
    dsa_layers,
    captured,
    args,
    device,
) -> list[dict]:
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    landmark_id = resolve_landmark_token_id(training_args, tokenizer)
    expected_slots = (args.physical_length // args.chunk_size) * (args.chunk_size - 1)
    corpus = DreamPackedCorpus(
        training_args.corpus_bin,
        training_args.corpus_meta,
        expected_real_slots=expected_slots,
    )
    collator = FullTextComplementaryCollator(
        mask_token_id=mask_id,
        pad_token_id=int(pad_id),
        eos_token_id=eos_id,
        lmk_token_id=int(landmark_id),
        chunk_size=args.chunk_size,
        t_min=float(training_args.t_min),
        t_max=float(training_args.t_max),
        seed=int(args.seed),
        ruler_mix_ratio=0.0,
    )
    mapping = physical_text_positions(expected_slots, args.chunk_size)
    rows = []
    for pack_id in range(min(args.limit, len(corpus))):
        example = corpus[pack_id]
        batch = collator([example])
        # Complementary collator emits two views; use the first training view.
        input_ids = batch["input_ids"][:1].to(device)
        attention_mask = batch["attention_mask"][:1].to(device)
        position_ids = batch["position_ids"][:1].to(device)
        captured.clear()
        groups = _groups(input_ids[0], attention_mask[0], eos_id, mask_id)
        clean = example["clean_ids"]
        valid_text = example["valid_tokens"].bool()
        eos_text = torch.where(clean.eq(eos_id) & valid_text)[0]
        pre_eos = eos_text[(eos_text > 0) & valid_text[eos_text - 1]]
        query_positions = mapping[pre_eos - 1].tolist() if pre_eos.numel() else []
        valid_queries = torch.where(attention_mask[0].bool())[0].tolist()
        _forward(model, input_ids, attention_mask, position_ids)
        layer_rows = []
        for layer_idx, _attn in dsa_layers:
            indices, valid = captured[layer_idx]
            pre_stats = [
                _summarize(indices[0, int(q)], valid[0, int(q)], groups)
                for q in query_positions
            ]
            all_stats = [
                _summarize(indices[0, int(q)], valid[0, int(q)], groups)
                for q in valid_queries[:: max(1, len(valid_queries) // 64)]
            ]
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "pre_eos_n": len(pre_stats),
                    "pre_eos_mean_eos_valid": _mean_stats(pre_stats, "eos_valid"),
                    "pre_eos_mean_eos_count": _mean_stats(pre_stats, "eos_valid_count"),
                    "pre_eos_mean_mask": _mean_stats(pre_stats, "mask_id"),
                    "all_query_mean_eos_valid": _mean_stats(all_stats, "eos_valid"),
                    "all_query_mean_eos_count": _mean_stats(all_stats, "eos_valid_count"),
                }
            )
        rows.append(
            {
                "pack_id": pack_id,
                "valid_tokens": int(attention_mask.sum().item()),
                "valid_eos_keys": int(groups["eos_valid"].sum().item()),
                "clean_eos_tokens": int(eos_text.numel()),
                "pre_eos_queries": len(query_positions),
                "layers": layer_rows,
            }
        )
        print("PACK " + json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt_config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="multifieldqa_en")
    parser.add_argument("--physical_length", type=int, default=32768)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    training_args = parse_training_args(["--config", args.training_config])
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    load_trainables(model, Path(args.checkpoint))
    model.eval()
    dsa_layers, captured = _install_capture(model)
    mfen = run_mfen(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        dsa_layers=dsa_layers,
        captured=captured,
        args=args,
        device=device,
    )
    pack = run_pack(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        dsa_layers=dsa_layers,
        captured=captured,
        args=args,
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dsa_layers": plan.dsa_layers,
        "eos_token_id": int(tokenizer.eos_token_id),
        "mfen": mfen,
        "pack": pack,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print("EOS_TOPK_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
