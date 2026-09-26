#!/usr/bin/env python3
"""Frozen same-forward LSE audit; all oracle quantities are observational."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dream_dllm_hils.longbench_eval import append_jsonl_fsync, build_fastdllm_block_layouts
from dream_dllm_hils.lse_calibration import LSECalibrationProbe
from scripts.dream_dllm_hils.diagnose_retrieval import make_layout, sha256, compare


def restore_model(model, training, checkpoint):
    from dream_dllm_hils.train_fulltext import initialize_and_configure_trainables
    from dream_dllm_hils.checkpointing import load_trainable_checkpoint
    source = Path(training.initialize_from) if training.initialize_from else None
    initialize_and_configure_trainables(model, training)
    target = Path(checkpoint)
    if source is None or target.resolve() != source.resolve():
        load_trainable_checkpoint(checkpoint_dir=target, model=model)
    expected = {}
    provenance = []
    for path in ([source, target] if source and source.resolve() != target.resolve() else [target]):
        saved = torch.load(path / "trainable_state.pt", map_location="cpu", weights_only=True)
        expected.update(saved)
        provenance.append(dict(path=str(path), sha256=sha256(path / "trainable_state.pt"),
                               tensors=len(saved), qcal_tensors=sum(".qcal." in k for k in saved)))
    params = dict(model.named_parameters())
    for name, tensor in expected.items():
        if name not in params or not torch.equal(params[name].detach().cpu(), tensor.to(params[name].dtype)):
            raise AssertionError("checkpoint reconstruction mismatch: " + name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return dict(chain=provenance, verified_tensors=len(expected),
                verified_frozen_non_qcal_tensors=sum(".qcal." not in k for k in expected))


def save_phase(output, identity, row, phase, probe, elapsed):
    if len(probe.records) != 7:
        raise AssertionError(f"expected 7 HiLS layers, got {len(probe.records)}")
    if [r["layer"] for r in probe.records] != [3, 7, 11, 15, 19, 23, 27]:
        raise AssertionError("unexpected layer IDs")
    for record in probe.records:
        if "gate_reconstruction_abs_error" not in record:
            raise AssertionError("fusion capture missing")
        if record["native_selector_reconstruction_overlap"] < .99:
            raise AssertionError(f"native selector mismatch: layer={record['layer']}, "
                                 f"overlap={record['native_selector_reconstruction_overlap']}")
        if record["gate_reconstruction_abs_error"]["mean"] > .005:
            raise AssertionError("gate reconstruction does not match native implementation")
    tensor_path = output.parent / "tensors" / (row["case_id"] + "-" + phase + ".npz")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for record, arrays in zip(probe.records, probe.arrays):
        payload.update({f"layer{record['layer']}_{key}": value for key, value in arrays.items()})
    np.savez_compressed(tensor_path, **payload)
    record = dict(identity=identity, case_id=row["case_id"], task=row["task"], depth=row["depth"],
                  phase=phase, seconds=elapsed, tensors=str(tensor_path), observations=probe.records)
    append_jsonl_fsync(output, record)
    print(json.dumps(dict(case_id=row["case_id"], phase=phase, layers=len(probe.records),
                          seconds=round(elapsed, 2))), flush=True)


@torch.inference_mode()
def run(args):
    from dream_dllm_hils.train_fulltext import _build_model_and_tokenizer, parse_args, _kernel_fallback_count
    from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    training = parse_args(["--config", args.config, "--no_gradient_checkpointing"])
    if (training.hils_topk != 16 or training.hils_token_budget != 0 or training.chunk_size != 64
        or training.hils_route_selection_mode != "post_softmax" or training.hils_route_temperature != 1):
        raise ValueError("first audit requires native post-softmax top16/noevict/temperature1")
    model, tok, plan = _build_model_and_tokenizer(training, torch.device("cuda"))
    restoration = restore_model(model, training, args.checkpoint)
    decoder = DreamHiLSFastDLLM(model=model, mask_token_id=int(tok.mask_token_id), threshold=.9, bootstrap="confidence")
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    rows = [row for row in rows if row["physical_length"] == 32768]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.rank::args.world_size]
    if not rows:
        raise ValueError("empty shard")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    code_paths = ["dream_dllm_hils/attention.py", "dream_dllm_hils/routing.py",
                  "dream_dllm_hils/fastdllm_attention.py", "dream_dllm_hils/lse_calibration.py",
                  "scripts/dream_dllm_hils/diagnose_lse_calibration.py"]
    identity = dict(version="lse-calibration-v1", config=json.loads(Path(args.config).read_text()),
                    config_sha256=sha256(args.config), manifest_sha256=sha256(args.manifest),
                    checkpoint=restoration, code_sha256={p: sha256(root / p) for p in code_paths},
                    max_queries=args.max_queries, background_mask_ratio=0,
                    snapshot="initial all-mask prefill and same-input cached refresh; no decoding transition",
                    exact_dtype="float32, autocast/TF32 disabled", torch=torch.__version__,
                    device=torch.cuda.get_device_name(),
                    ranking="per-query-head Spearman; 4096 seeded ordered chunk pairs per row",
                    denominator="native late-drop; early-drop selector reported separately")
    existing = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    if any(x["identity"] != identity for x in existing):
        raise ValueError("output provenance mismatch")
    done = {(x["case_id"], x["phase"]) for x in existing}
    print(json.dumps(dict(event="loaded", restoration=restoration, cases=len(rows), rank=args.rank,
                          rope=identity["config"]["model_rope_scaling"])), flush=True)
    validated = False
    with LSECalibrationProbe(decoder, args.max_queries) as probe:
        for row in rows:
            if all((row["case_id"], phase) in done for phase in ("prefill", "cached_refresh")):
                continue
            case, layout = make_layout(row, tok, training, 0)
            ids, valid, positions = [x[None].cuda() for x in (layout.input_ids, layout.attention_mask, layout.position_ids)]
            landmarks = layout.landmark_positions.cuda()
            expected = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if not validated:
                    probe.enabled = False
                    warm = decoder.prefill(ids, valid, positions, landmarks)
                    expected = warm.logits[:, layout.predictor_positions].float().clone()
                    del warm
                    torch.cuda.synchronize()
                probe.begin(layout, "prefill")
                start = time.perf_counter()
                pref = decoder.prefill(ids, valid, positions, landmarks)
                torch.cuda.synchronize()
                prefill_seconds = time.perf_counter() - start
                cache = pref.cache
                logits = pref.logits[:, layout.predictor_positions].float().clone()
                del pref
                if expected is not None:
                    passthrough = compare(logits, expected)
                    if passthrough["relative_l2"] > .005 or passthrough["cosine"] < .999:
                        raise AssertionError(f"observer perturbs output: {passthrough}")
                if (row["case_id"], "prefill") not in done:
                    save_phase(output, identity, row, "prefill", probe, prefill_seconds)
                block = build_fastdllm_block_layouts(layout, 32, 64)[0]
                native_cached_logits = None
                if not validated:
                    probe.enabled = False
                    native = decoder.cached_forward(ids, block, cache, landmarks)
                    active_rows = torch.searchsorted(native.query_positions, layout.predictor_positions.cuda())
                    native_cached_logits = native.logits[:, active_rows].float().clone()
                    del native
                probe.begin(layout, "cached_refresh")
                start = time.perf_counter()
                cached = decoder.cached_forward(ids, block, cache, landmarks)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                active_rows = torch.searchsorted(cached.query_positions, layout.predictor_positions.cuda())
                cached_logits = cached.logits[:, active_rows].float()
                full_cached = compare(cached_logits, logits)
                if not validated:
                    cached_passthrough = compare(cached_logits, native_cached_logits)
                    if cached_passthrough["relative_l2"] > .005 or cached_passthrough["cosine"] < .999:
                        raise AssertionError(f"cached observer mismatch: {cached_passthrough}")
                    validation = dict(passed=True, prefill_passthrough=passthrough,
                                      cached_passthrough=cached_passthrough, full_vs_cached=full_cached)
                    output.with_suffix(".validation.json").write_text(json.dumps(validation, indent=2) + "\n")
                    print(json.dumps(dict(event="validation", **validation)), flush=True)
                    validated = True
                if (row["case_id"], "cached_refresh") not in done:
                    save_phase(output, identity, row, "cached_refresh", probe, elapsed)
                del cache, cached, cached_logits, logits, expected, native_cached_logits
            if _kernel_fallback_count(model):
                raise AssertionError("unexpected kernel fallback")
    print(json.dumps(dict(event="complete", cases=len(rows), fallback_count=0)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_queries", type=int, default=16)
    run(p.parse_args())
