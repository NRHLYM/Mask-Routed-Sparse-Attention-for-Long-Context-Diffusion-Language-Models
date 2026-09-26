#!/usr/bin/env python3
"""Compare pristine and fine-tuned all-dense Dream on the frozen 32k audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dream_dllm_hils.dense_reference import (
    LAYERS, DenseEvidenceProbe, body_key_mask, dense_forward, validate_native,
)
from dream_dllm_hils.diagnostic_helpers import physical_positions
from dream_dllm_hils.fastdllm_v1 import _unwrap_dream_model
from dream_dllm_hils.longbench_eval import append_jsonl_fsync
from scripts.dream_dllm_hils.diagnose_lse_calibration import restore_model
from scripts.dream_dllm_hils.diagnose_retrieval import make_layout, sha256


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def load_model(training, arm, checkpoint):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from dream_dllm_hils.train_fulltext import _build_model_and_tokenizer, _install_legacy_dream_rope_compat
    if arm == "finetuned_dense":
        model, tokenizer, _ = _build_model_and_tokenizer(training, torch.device("cuda"))
        provenance = restore_model(model, training, checkpoint)
        provenance["loaded_hils_parameters"] = "verified but never used in all-dense computation"
    else:
        _install_legacy_dream_rope_compat()
        tokenizer = AutoTokenizer.from_pretrained(training.model_path, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(training.model_path, trust_remote_code=True, local_files_only=True)
        config.max_position_embeddings = training.model_max_position_embeddings
        config.rope_theta = training.model_rope_theta
        config.rope_scaling = dict(training.model_rope_scaling)
        model = AutoModel.from_pretrained(training.model_path, config=config, trust_remote_code=True,
                                          local_files_only=True, torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True).cuda()
        forbidden = [n for n, _ in model.named_parameters() if any(
            s in n for s in ("lora_", ".qcal.", "lmk_embed", "entropy_bias"))]
        if forbidden:
            raise AssertionError(f"base reference is contaminated: {forbidden}")
        provenance = dict(model_path=training.model_path, adapter_parameters=0,
                          checkpoint_loaded=False, forbidden_parameters=forbidden)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, provenance


@torch.inference_mode()
def run(args):
    from dream_dllm_hils.train_fulltext import parse_args
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    training = parse_args(["--config", args.config, "--no_gradient_checkpointing"])
    if training.chunk_size != 64 or training.hils_topk != 16 or training.hils_token_budget != 0:
        raise ValueError("requires the frozen top16/noevict audit config")
    model, tokenizer, restoration = load_model(training, args.arm, args.checkpoint)
    core = _unwrap_dream_model(model)
    if len(core.model.layers) != 28:
        raise AssertionError("expected Dream with 28 layers")
    rows = [r for r in map(json.loads, Path(args.manifest).read_text().splitlines())
            if r["physical_length"] == 32768]
    if len(rows) != 14:
        raise AssertionError("frozen manifest must contain 14 32k cases")
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.rank::args.world_size]
    source = {r["case_id"]: r for p in sorted(Path(args.source).glob("rank-*.jsonl"))
              for r in map(json.loads, p.read_text().splitlines()) if r["phase"] == "prefill"}
    root = Path(__file__).resolve().parents[2]
    code_paths = ["dream_dllm_hils/dense_reference.py", "scripts/dream_dllm_hils/diagnose_dense_reference.py",
                  "dream_dllm_hils/lse_calibration.py", "dream_dllm_hils/lse_oracle.py"]
    identity = dict(version="dense-reference-v1", arm=args.arm, checkpoint=restoration,
                    config_sha256=sha256(args.config), manifest_sha256=sha256(args.manifest),
                    config=json.loads(Path(args.config).read_text()),
                    source_identity=source[rows[0]["case_id"]]["identity"],
                    code_sha256={p: sha256(root / p) for p in code_paths},
                    base_index_sha256=sha256(Path(training.model_path) / "model.safetensors.index.json"),
                    torch=torch.__version__, device=torch.cuda.get_device_name(),
                    attention="all 28 layers noncausal full-body dense via forced Flash SDPA",
                    input="same physical positions and valid body tokens; LMK placeholders use native MASK; keys excluded",
                    snapshot="initial all-mask prefill only; no denoising/cache transition; no generated F1",
                    ranking="same late-drop GQA max post-softmax top16; per-head and early-drop sensitivity",
                    yarn_caveat="factor4/original8192 held fixed to prior oracle, not native2k-to32k factor16")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = list(map(json.loads, output.read_text().splitlines())) if output.exists() else []
    if any(r["identity"] != identity for r in existing):
        raise AssertionError("output provenance mismatch")
    done = {r["case_id"] for r in existing}
    print(json.dumps(dict(event="loaded", arm=args.arm, cases=len(rows), restoration=restoration)), flush=True)
    validated = False
    for row in rows:
        if row["case_id"] in done:
            continue
        old = source[row["case_id"]]
        if old["identity"]["config_sha256"] != identity["config_sha256"] or old["identity"]["manifest_sha256"] != identity["manifest_sha256"]:
            raise AssertionError("config or manifest differs from sparse audit")
        case, layout = make_layout(row, tokenizer, training, 0)
        ids = layout.input_ids[None].cuda().clone()
        positions = layout.position_ids[None].cuda()
        valid = body_key_mask(layout.attention_mask[None].cuda())
        ids[:, layout.landmark_positions] = int(tokenizer.mask_token_id)
        if int(ids.max()) >= core.model.embed_tokens.num_embeddings:
            raise AssertionError("external token ID reached base embedding lookup")
        values = torch.unique(physical_positions(torch.tensor(sum(case.evidence_values, [])), 64) // 64)
        facts = torch.unique(physical_positions(torch.tensor(sum(case.evidence_facts, [])), 64) // 64)
        if not validated:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                validation = validate_native(core, ids[:, -512:], valid[:, -512:], positions[:, -512:])
            output.with_suffix(".validation.json").write_text(json.dumps(dict(
                passed=True, native_short_sequence=validation, positions="original final512 positions",
                all_layers=len(core.model.layers)), indent=2) + "\n")
            print(json.dumps(dict(event="native_validation", **validation)), flush=True)
            validated = True
        with np.load(old["tensors"]) as old_arrays:
            probe = DenseEvidenceProbe(layout, values, facts, old, old_arrays)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = dense_forward(core, ids, valid, positions, layout.predictor_positions.cuda(), probe)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if probe.visited != list(range(28)) or [o["layer"] for o in probe.records] != list(LAYERS):
            raise AssertionError("not all dense layers or observations executed")
        if not torch.isfinite(logits).all():
            raise AssertionError("non-finite logits")
        gold = torch.tensor(tokenizer(" " + " ".join(case.answers), add_special_tokens=False).input_ids, device="cuda")
        if len(gold) > logits.shape[1]:
            raise AssertionError("gold answer exceeds predictor budget")
        nll = torch.nn.functional.cross_entropy(logits[0, :len(gold)].float(), gold)
        tensors = output.parent / "tensors" / (args.arm + "-" + row["case_id"] + ".npz")
        tensors.parent.mkdir(exist_ok=True)
        np.savez_compressed(tensors, **probe.arrays)
        record = dict(identity=identity, arm=args.arm, case_id=row["case_id"], task=row["task"],
                      depth=row["depth"], phase="prefill", seconds=elapsed, gold_nll=float(nll),
                      peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      valid_body_tokens=int(valid.sum()), input_sha256=tensor_hash(ids),
                      positions_sha256=tensor_hash(positions), key_mask_sha256=tensor_hash(valid),
                      layers_executed=probe.visited, tensors=str(tensors), observations=probe.records)
        append_jsonl_fsync(output, record)
        recall = np.mean([o["value"]["recall"] for o in probe.records])
        print(json.dumps(dict(event="case", arm=args.arm, case_id=row["case_id"],
                              recall=recall, seconds=round(elapsed, 2), peak_gib=record["peak_memory_gib"])), flush=True)
        del probe, logits
    print(json.dumps(dict(event="complete", arm=args.arm, cases=len(rows))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--arm", choices=("base_dense", "finetuned_dense"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    run(parser.parse_args())
