#!/usr/bin/env python3
"""Execute isolated selection-only/fusion-only active-query oracle forwards."""
import argparse
import json
import time
from pathlib import Path

import torch

from dream_dllm_hils.lse_oracle import LSEOracleProbe
from dream_dllm_hils.longbench_eval import append_jsonl_fsync, build_fastdllm_block_layouts
from scripts.dream_dllm_hils.diagnose_retrieval import make_layout, sha256, compare
from scripts.dream_dllm_hils.diagnose_lse_calibration import restore_model


def save_record(path, identity, row, variant, phase, observations, logits, baseline_logits, gold, elapsed):
    if [r["layer"] for r in observations] != [3, 7, 11, 15, 19, 23, 27]:
        raise AssertionError("missing HiLS observations")
    if any(r["native_selector_reconstruction_overlap"] < .99 for r in observations):
        raise AssertionError("native reconstruction failed")
    if variant in {"baseline", "fusion_exact"} and not all(r["support_matches_frozen_baseline"] for r in observations):
        raise AssertionError("fixed support intervention changed baseline support")
    nll = float(torch.nn.functional.cross_entropy(logits[0, :len(gold)].float(), gold))
    if not torch.isfinite(torch.tensor(nll)):
        raise AssertionError("nonfinite gold NLL")
    result = dict(identity=identity, case_id=row["case_id"], task=row["task"], depth=row["depth"],
                  variant=variant, phase=phase, observations=observations, gold_answer_nll=nll,
                  logits_vs_baseline=compare(logits, baseline_logits), elapsed_seconds=elapsed)
    append_jsonl_fsync(path, result)
    hits = sum(r["actual_value"]["hits"] for r in observations)
    units = sum(r["actual_value"]["units"] for r in observations)
    print(json.dumps(dict(case_id=row["case_id"], variant=variant, phase=phase,
                          evidence_recall=hits / units if units else None,
                          gold_nll=nll, seconds=round(elapsed, 2))), flush=True)


@torch.inference_mode()
def run(args):
    from dream_dllm_hils.train_fulltext import _build_model_and_tokenizer, parse_args, _kernel_fallback_count
    from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    training = parse_args(["--config", args.config, "--no_gradient_checkpointing"])
    if (training.hils_topk != 16 or training.hils_token_budget != 0 or training.chunk_size != 64
        or training.hils_route_selection_mode != "post_softmax" or training.hils_route_temperature != 1):
        raise ValueError("requires native top16/noevict/post-softmax/temperature1 config")
    model, tok, plan = _build_model_and_tokenizer(training, torch.device("cuda"))
    restoration = restore_model(model, training, args.checkpoint)
    decoder = DreamHiLSFastDLLM(model=model, mask_token_id=int(tok.mask_token_id), threshold=.9, bootstrap="confidence")
    all_rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines()]
    rows = [r for r in all_rows if r["physical_length"] == 32768]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.rank::args.world_size]
    root = Path(__file__).resolve().parents[2]
    code = ["dream_dllm_hils/lse_oracle.py", "dream_dllm_hils/lse_calibration.py",
            "dream_dllm_hils/attention.py", "dream_dllm_hils/fastdllm_attention.py",
            "dream_dllm_hils/routing.py", "scripts/dream_dllm_hils/diagnose_lse_oracle.py"]
    identity = dict(version="lse-oracle-active-query-v2-frozen-fusion-support", config=json.loads(Path(args.config).read_text()),
        config_sha256=sha256(args.config), checkpoint=restoration, manifest_sha256=sha256(args.manifest),
        code_sha256={name: sha256(root / name) for name in code},
        intervention_scope="all 32 answer predictor queries at each of seven HiLS layers; other queries native",
        observation_scope="16 evenly spaced predictor queries, matched to earlier frozen tensor audit",
        scoring="FP32 exact token-QK chunk LSE; native late-drop denominator then GQA max; no gold labels in selection",
        selection_only="exact selector, original surrogate logits for fusion on new support",
        fusion_only="active-query support frozen to baseline at all seven layers for each phase; exact token LSE for remote weights",
        snapshots="initial all-mask prefill and same-input cached refresh, no decoding transition",
        torch=torch.__version__, device=torch.cuda.get_device_name())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    if any(r["identity"] != identity for r in existing):
        raise ValueError("resume provenance mismatch")
    done = {(r["case_id"], r["variant"], r["phase"]) for r in existing}
    print(json.dumps(dict(event="loaded", cases=len(rows), rank=args.rank, restoration=restoration)), flush=True)
    validated = False
    with LSEOracleProbe(decoder) as probe:
        for row in rows:
            if all((row["case_id"], v, p) in done for v in ("baseline", "select_exact", "fusion_exact")
                   for p in ("prefill", "cached_refresh")):
                continue
            case, layout = make_layout(row, tok, training, 0)
            ids, valid, positions = [x[None].cuda() for x in (layout.input_ids, layout.attention_mask, layout.position_ids)]
            landmarks = layout.landmark_positions.cuda()
            block = build_fastdllm_block_layouts(layout, 32, 64)[0]
            gold = torch.tensor(tok(" " + " ".join(case.answers), add_special_tokens=False).input_ids, device="cuda")
            if not 0 < gold.numel() <= 32:
                raise ValueError("gold span does not fit predictor layout")
            baseline_logits = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                warm_logits = None
                if not validated:
                    probe.enabled = False
                    warm = decoder.prefill(ids, valid, positions, landmarks)
                    warm_logits = warm.logits[:, layout.predictor_positions].float().clone()
                    del warm
                for variant in ("baseline", "select_exact", "fusion_exact"):
                    probe.begin(case, layout, variant, "prefill")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    pref = decoder.prefill(ids, valid, positions, landmarks)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    logits = pref.logits[:, layout.predictor_positions].float().clone()
                    cache = pref.cache
                    del pref
                    if variant == "baseline":
                        baseline_logits["prefill"] = logits.clone()
                        if warm_logits is not None:
                            error = compare(logits, warm_logits)
                            if error["relative_l2"] > 1e-6:
                                raise AssertionError(f"baseline observer is not passthrough: {error}")
                    if (row["case_id"], variant, "prefill") not in done:
                        save_record(output, identity, row, variant, "prefill", probe.records, logits,
                                    baseline_logits["prefill"], gold, elapsed)
                    native_cached_logits = None
                    if variant == "baseline" and not validated:
                        probe.enabled = False
                        native = decoder.cached_forward(ids, block, cache, landmarks)
                        active = torch.searchsorted(native.query_positions, layout.predictor_positions.cuda())
                        native_cached_logits = native.logits[:, active].float().clone()
                        del native
                    probe.begin(case, layout, variant, "cached_refresh")
                    start = time.perf_counter()
                    cached = decoder.cached_forward(ids, block, cache, landmarks)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    active = torch.searchsorted(cached.query_positions, layout.predictor_positions.cuda())
                    cached_logits = cached.logits[:, active].float().clone()
                    if variant == "baseline":
                        baseline_logits["cached_refresh"] = cached_logits.clone()
                        if native_cached_logits is not None:
                            cached_error = compare(cached_logits, native_cached_logits)
                            if cached_error["relative_l2"] > 1e-6:
                                raise AssertionError(f"cached baseline observer mismatch: {cached_error}")
                            check = dict(prefill_passthrough=error, cached_passthrough=cached_error,
                                         full_vs_cached=compare(cached_logits, logits), passed=True)
                            output.with_suffix(".validation.json").write_text(json.dumps(check, indent=2) + "\n")
                            print(json.dumps(dict(event="validation", **check)), flush=True)
                            validated = True
                    if (row["case_id"], variant, "cached_refresh") not in done:
                        save_record(output, identity, row, variant, "cached_refresh", probe.records, cached_logits,
                                    baseline_logits["cached_refresh"], gold, elapsed)
                    del cache, cached, cached_logits, logits, native_cached_logits
            if _kernel_fallback_count(model):
                raise AssertionError("unexpected kernel fallback")
    print(json.dumps(dict(event="complete", cases=len(rows), variants=3, fallback_count=0)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    run(p.parse_args())
