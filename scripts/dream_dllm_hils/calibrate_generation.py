#!/usr/bin/env python3
"""Small positive controls for generation; never changes the v1 diagnostic scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from dream_dllm_hils import fastdllm_v1
from dream_dllm_hils.diagnostic_helpers import score_codes
from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM, PrefillOutput
from dream_dllm_hils.longbench_eval import (
    FastDLLMBlockLayout, GenerationLayout, append_jsonl_fsync,
    build_fastdllm_block_layouts, build_generation_layout,
)
from dream_dllm_hils.train_fulltext import (
    _build_model_and_tokenizer, _install_legacy_dream_rope_compat,
    _kernel_fallback_count, parse_args, resolve_landmark_token_id,
)
from scripts.dream_dllm_hils.eval_longbench_mfen import decode_answer, load_trainables


class NativeDenseDecoder(DreamHiLSFastDLLM):
    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask, position_ids, landmark_positions=None):
        # Native Dream expects a 4D bidirectional key mask, not sparse's 2D mask.
        allowed = attention_mask[:, None, None, :].bool()
        output = self.model(input_ids=input_ids, attention_mask=allowed,
                            position_ids=position_ids, use_cache=False)
        return PrefillOutput(output.logits, None)


def plain_layout(prompt_ids, mask_id, answer_tokens=32):
    ids = torch.tensor(prompt_ids + [mask_id] * answer_tokens, dtype=torch.long)
    answer = torch.arange(len(prompt_ids), len(ids))
    empty = torch.empty(0, dtype=torch.long)
    layout = GenerationLayout(ids, torch.ones_like(ids, dtype=torch.bool),
                             torch.arange(len(ids)), answer, answer - 1, empty)
    block = FastDLLMBlockLayout(answer, answer - 1, empty, empty,
                               torch.unique(torch.cat((answer - 1, answer))), answer)
    return layout, (block,)


def controls(tok, manifest, suite, *, origin=None, training=None):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if suite == "matched":
        from dream_dllm_hils.data import RulerDenoisingSynthesizer
        from dream_dllm_hils.packed_corpus import DreamPackedCorpus
        if origin is None:
            raise ValueError("matched controls require the original corpus root")
        ds = DreamPackedCorpus(Path(origin) / training.corpus_bin, Path(origin) / training.corpus_meta)
        synth = RulerDenoisingSynthesizer(tok, task_ids=(0,))
        cases = []
        for code in ("94179140", "99647194"):
            prompt = (f" |One of the special magic numbers for long-context is: {code}|"
                      " What is the special magic number for long-context mentioned in the provided text? Answer: ")
            cases.append(dict(case_id=f"matched-copy-{code}", prompt_ids=tok(prompt, add_special_tokens=False).input_ids,
                              answers=[code], task="matched_single", physical_length=2048))
        for row in rows:
            if row["physical_length"] not in {2048, 32768} or not row["case_id"].endswith(("case00", "case03")):
                continue
            pack = ds[row["validation_pack_id"]]
            base = pack["clean_ids"][pack["valid_tokens"]]
            # Leave room for the fixed 32-token decoder, without trimming any facts.
            clean, target = synth.synthesize(base[:row["physical_length"] // 64 * 63 - 64], task_id=0)
            start = int(torch.where(target)[0][0])
            gold = tok.decode(clean[target].tolist(), skip_special_tokens=True).strip()
            assert len(gold) == 8 and gold.isdigit()
            prompt_ids = clean[:start].tolist()
            assert gold in tok.decode(prompt_ids, skip_special_tokens=False)
            cases.append(dict(case_id="matched-" + row["case_id"], prompt_ids=prompt_ids, answers=[gold],
                              task="matched_single", physical_length=row["physical_length"],
                              validation_pack_id=row["validation_pack_id"]))
            if row["physical_length"] == 32768:
                cases.append(dict(row, task="original_single"))
        return cases
    cases = []
    for seed, code in enumerate(("94179140", "99647194")):
        prompt = f"The password is {code}.\nQuestion: What is the password?\nAnswer:"
        ids = tok(prompt, add_special_tokens=False).input_ids
        for bos in (False, True):
            if bos and tok.bos_token_id is None:
                continue
            cases.append(dict(case_id=f"copy8-{seed}-bos{int(bos)}", prompt_ids=([int(tok.bos_token_id)] if bos else []) + ids,
                              answers=[code], task="copy8", physical_length=2048))
    ids = tok("The code word is violet.\nQuestion: What is the code word?\nAnswer:", add_special_tokens=False).input_ids
    cases.append(dict(case_id="copyword", prompt_ids=ids, answers=["violet"], task="copyword", physical_length=2048))
    lengths = {2048} if suite == "screen" else {2048, 32768}
    cases.extend(dict(row, task="original_" + row["task"]) for row in rows
                 if row["physical_length"] in lengths and row["case_id"].endswith(("case00", "case03")))
    return cases


def native_model(path, training):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    _install_legacy_dream_rope_compat()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    for name, value in (("max_position_embeddings", training.model_max_position_embeddings),
                        ("rope_theta", training.model_rope_theta), ("rope_scaling", training.model_rope_scaling)):
        if value is not None:
            setattr(config, name, value)
    model = AutoModel.from_pretrained(path, config=config, trust_remote_code=True, local_files_only=True,
                                     torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return model, tok


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--training_config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--manifest", required=True)
    p.add_argument("--origin")
    p.add_argument("--output", required=True)
    p.add_argument("--model_kind", choices=("hils", "dsa", "base_dense"), required=True)
    p.add_argument("--suite", choices=("screen", "long", "matched"), default="screen")
    args = p.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError("calibration output already exists; use a new run path")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    training = parse_args(["--config", args.training_config, "--model_path", args.model_path, "--no_gradient_checkpointing"])
    if args.model_kind == "base_dense":
        model, tok = native_model(args.model_path, training)
        checkpoint_sha = None
    else:
        if training.attention_mode != args.model_kind:
            raise ValueError("config/model mismatch")
        model, tok, _ = _build_model_and_tokenizer(training, torch.device("cuda"))
        load_trainables(model, Path(args.checkpoint))
        model.eval()
        checkpoint_sha = hashlib.sha256((Path(args.checkpoint) / "trainable_state.pt").read_bytes()).hexdigest()
    original_select = fastdllm_v1.select_confidence_transfers
    for case in controls(tok, args.manifest, args.suite, origin=args.origin, training=training):
        if args.model_kind == "base_dense":
            layout, blocks = plain_layout(case["prompt_ids"], int(tok.mask_token_id))
        else:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            layout = build_generation_layout(prompt_ids=case["prompt_ids"], answer_tokens=32,
                physical_length=case["physical_length"], chunk_size=64, mask_token_id=int(tok.mask_token_id),
                pad_token_id=int(pad_id), landmark_token_id=resolve_landmark_token_id(training, tok))
            blocks = build_fastdllm_block_layouts(layout, 32, 64)
        for cached in ((False,) if args.model_kind == "base_dense" else (True, False)):
            thresholds = (0.9, 1.0) if case["case_id"] in {"copy8-0-bos0", "L2048-case00"} else (0.9,)
            for threshold in thresholds:
                trace = []
                def select(logits, active_rows, **kwargs):
                    chosen, ids, confidence = original_select(logits, active_rows, **kwargs)
                    trace.append(dict(active=active_rows.tolist(), chosen=chosen.tolist(),
                                      ids=ids.tolist(), confidence=confidence.tolist()))
                    return chosen, ids, confidence
                decoder_type = NativeDenseDecoder if args.model_kind == "base_dense" else DreamHiLSFastDLLM
                decoder = decoder_type(model=model, mask_token_id=int(tok.mask_token_id), threshold=threshold, use_cache=cached)
                torch.manual_seed(7)
                started = time.perf_counter()
                with patch.object(fastdllm_v1, "select_confidence_transfers", select), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated, stats = decoder.generate(input_ids=layout.input_ids[None].cuda(),
                        attention_mask=layout.attention_mask[None].cuda(), position_ids=layout.position_ids[None].cuda(),
                        landmark_positions=layout.landmark_positions.cuda(), blocks=blocks)
                ids = generated[0, layout.answer_positions.cuda()].tolist()
                prediction = decode_answer(tok, ids)
                scores = score_codes(prediction, case["answers"]) if case["task"] != "copyword" else {}
                result = dict(case_id=case["case_id"], task=case["task"], model=args.model_kind,
                    checkpoint_sha256=checkpoint_sha, prompt_sha256=hashlib.sha256(json.dumps(case["prompt_ids"]).encode()).hexdigest(),
                    cache=cached, threshold=threshold, physical_length=layout.input_ids.numel(),
                    prompt_text=tok.decode(case["prompt_ids"], skip_special_tokens=False),
                    answers=case["answers"], prediction=prediction, raw_answer_ids=ids,
                    raw_decoded=tok.decode(ids, skip_special_tokens=False),
                    eos_offset=ids.index(tok.eos_token_id) if tok.eos_token_id in ids else None,
                    answer_substring_present=all(answer in prediction for answer in case["answers"]),
                    **scores, **asdict(stats), fallback_count=_kernel_fallback_count(model), trace=trace,
                    seconds=time.perf_counter() - started)
                assert result["fallback_count"] == 0
                assert cached == (stats.cached_forwards > 0) or (cached and len(trace) == 1)
                append_jsonl_fsync(path, result)
                print(json.dumps({k:result[k] for k in ("case_id", "model", "cache", "threshold", "prediction", "answers", "eos_offset", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
