#!/usr/bin/env python3
"""Predeclared decoder bridge and positive controls; no checkpoint or score edits."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dream_dllm_hils.data import RulerDenoisingSynthesizer
from dream_dllm_hils.diagnostic_helpers import RetrievalCase, score_codes
from dream_dllm_hils.longbench_eval import append_jsonl_fsync, build_generation_layout, build_fastdllm_block_layouts, longbench_score
from dream_dllm_hils.packed_corpus import DreamPackedCorpus
from dream_dllm_hils import fastdllm_v1
from dream_dllm_hils.fastdllm_v1 import DreamHiLSFastDLLM
from dream_dllm_hils.train_fulltext import _build_model_and_tokenizer, _kernel_fallback_count, parse_args, resolve_landmark_token_id
from scripts.dream_dllm_hils import eval_longbench_mfen as exact


MODES = ("legacy_exact", "first_token_no_cache", "first_token_cached", "confidence_no_cache", "confidence_cached")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(value)


def find_span(tokens, needle):
    found = [i for i in range(len(tokens) - len(needle) + 1) if tokens[i:i + len(needle)] == needle]
    if len(found) != 1:
        raise ValueError(f"expected one exact evidence span, found {len(found)}")
    return list(range(found[0], found[0] + len(needle)))


class PairedSynthesizer(RulerDenoisingSynthesizer):
    """Training's literal task-0 template, but fixed code/depth across lengths."""
    def __init__(self, tokenizer, seed, depth):
        super().__init__(tokenizer, task_ids=(0,))
        self.seed, self.depth = seed, depth

    def _rng(self, input_ids, salt):
        return np.random.RandomState(self.seed)

    def _compose(self, base_ids, needles, question_ids, answer_ids, rng):
        if len(needles) != 1:
            raise ValueError("paired controls require one needle")
        budget = len(base_ids) - len(needles[0]) - len(question_ids) - len(answer_ids)
        if budget <= 0:
            raise ValueError("insufficient filler")
        position = int(budget * self.depth)
        context = np.concatenate((base_ids[:position], needles[0], base_ids[position:budget], question_ids))
        ids = torch.tensor(np.concatenate((context, answer_ids)), dtype=torch.long)
        target = torch.arange(len(ids)) >= len(context)
        self.fact = list(range(position, position + len(needles[0])))
        self.question_start = len(context) - len(question_ids)
        return ids, target


def prepare(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    config = json.loads(Path(args.training_config).read_text())
    corpus = Path(args.origin) / config["corpus_bin"]
    meta = Path(args.origin) / config["corpus_meta"]
    ds = DreamPackedCorpus(corpus, meta)
    old = read_rows(args.original_manifest)
    packs = [r["validation_pack_id"] for r in old if r["physical_length"] == 32768][:12]
    if len(set(packs)) != 12:
        raise ValueError("need 12 distinct held-out packs")
    rows = []
    for i, pack_id in enumerate(packs):
        pack = ds[pack_id]
        filler = pack["clean_ids"][pack["valid_tokens"]]
        for length in (2048, 8192, 16384, 32768):
            seed, depth = 9302609 + i, (0.1, 0.5, 0.9)[i % 3]
            synth = PairedSynthesizer(tok, seed, depth)
            clean, target = synth.synthesize(filler[:length // 64 * 63 - 32], task_id=0)
            start = int(torch.where(target)[0][0])
            gold = tok.decode(clean[target].tolist(), skip_special_tokens=True).strip()
            assert len(gold) == 8 and gold.isdigit()
            prompt = clean[:start].tolist()
            fact_tokens = [prompt[p] for p in synth.fact]
            relative = find_span(fact_tokens, tok(gold, add_special_tokens=False).input_ids)
            value = [synth.fact[0] + p for p in relative]
            case = RetrievalCase(prompt, [value], [synth.fact], synth.question_start, [gold], "single", seed, depth)
            rows.append(dict(asdict(case), case_id=f"paired-L{length}-case{i:02}", physical_length=length,
                             validation_pack_id=pack_id, kind="paired", answer_tokens=32, pair_id=i))
    for row in old:
        if row["physical_length"] in (2048, 32768) and row["case_id"].endswith(("case00", "case03")):
            rows.append(dict(row, kind="original", answer_tokens=32))
    prompts = json.loads(Path(args.prompts).read_text())
    for task, answer_tokens in (("hotpotqa", 32), ("triviaqa", 32), ("multifieldqa_en", 64)):
        for i, example in enumerate(read_rows(Path(args.data_root) / f"{task}.jsonl")[:4]):
            prompt_ids, metadata = exact.tokenize_prompt(tok, prompts[task], example, 32768 // 64 * 63 - answer_tokens)
            rows.append(dict(case_id=f"lb-{task}-{i}", kind="longbench", task=task, index=i,
                             prompt_ids=prompt_ids, prompt_metadata=metadata, answers=example["answers"],
                             all_classes=example.get("all_classes"), physical_length=32768, answer_tokens=answer_tokens))
    write_new(args.manifest, "".join(json.dumps(r) + "\n" for r in rows))
    write_new(Path(args.manifest).with_name("diagnostic-cases.jsonl"), "".join(json.dumps(r) + "\n" for r in rows if r["kind"] != "longbench"))
    write_new(Path(args.manifest).with_suffix(".provenance.json"), json.dumps(dict(
        original_manifest_sha256=digest(args.original_manifest), corpus_sha256=digest(corpus),
        corpus_meta_sha256=digest(meta), manifest_sha256=digest(args.manifest), held_out_pack_ids=packs,
        selection="first 12 existing eligible held-out packs; first 4 examples per named LongBench task; no output-based selection",
        paired_control="training task-0 wording, fixed code/relative depth/pack per length pair; filler length changes",
        counts={k: sum(r["kind"] == k for r in rows) for k in ("paired", "original", "longbench")},
        longbench_source_sha256={t:digest(Path(args.data_root) / f"{t}.jsonl") for t in ("hotpotqa", "triviaqa", "multifieldqa_en")},
        prompts_sha256=digest(args.prompts)), indent=2) + "\n")
    print(json.dumps({"prepared": len(rows), "manifest_sha256": digest(args.manifest)}), flush=True)


def selected_cases(rows, stage):
    if stage == "controls":
        return [r for r in rows if r["kind"] == "paired"]
    return [r for r in rows if r["kind"] != "paired" or (r["pair_id"] < 2 and r["physical_length"] in (2048, 32768))]


@torch.inference_mode()
def run(args):
    rows = selected_cases(read_rows(args.manifest), args.stage)
    modes = MODES if args.stage == "bridge" else ("first_token_cached", "confidence_cached")
    if Path(args.output).exists():
        raise ValueError("output exists; use a fresh run")
    torch.manual_seed(7)
    training = parse_args(["--config", args.training_config, "--model_path", args.model_path, "--no_gradient_checkpointing"])
    model, tok, _ = _build_model_and_tokenizer(training, torch.device("cuda"))
    exact.load_trainables(model, Path(args.checkpoint))
    model.eval()
    lmk = resolve_landmark_token_id(training, tok)
    provenance = dict(manifest_sha256=digest(args.manifest), checkpoint_sha256=digest(Path(args.checkpoint) / "trainable_state.pt"),
                      config_sha256=digest(args.training_config), model=training.attention_mode)
    for row in rows:
        layout = build_generation_layout(prompt_ids=row["prompt_ids"], answer_tokens=row["answer_tokens"],
            physical_length=row["physical_length"], chunk_size=64, mask_token_id=int(tok.mask_token_id),
            pad_token_id=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id), landmark_token_id=lmk)
        for mode in modes:
            torch.manual_seed(7)
            raw, trace = [], []
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            if mode == "legacy_exact":
                original_decode = exact.decode_answer
                def capture(tokenizer, ids):
                    raw.extend(ids)
                    return original_decode(tokenizer, ids)
                with patch.object(exact, "decode_answer", capture):
                    prediction, nfe, _ = exact.denoise_answer(model=model, tokenizer=tok, layout=layout,
                        landmark_token_id=lmk, steps=64)
                stats = dict(full_prefills=nfe, cached_forwards=0)
            else:
                bootstrap = "confidence" if mode.startswith("confidence") else "first_token"
                decoder = DreamHiLSFastDLLM(model=model, mask_token_id=int(tok.mask_token_id), threshold=0.9,
                    use_cache=mode.endswith("_cached"), bootstrap=bootstrap)
                original_select = fastdllm_v1.select_confidence_transfers
                def select(logits, active_rows, **kwargs):
                    chosen, ids, confidence = original_select(logits, active_rows, **kwargs)
                    trace.append(dict(active=active_rows.tolist(), chosen=chosen.tolist(), ids=ids.tolist(), confidence=confidence.tolist()))
                    return chosen, ids, confidence
                with patch.object(fastdllm_v1, "select_confidence_transfers", select), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated, generation_stats = decoder.generate(input_ids=layout.input_ids[None].cuda(),
                        attention_mask=layout.attention_mask[None].cuda(), position_ids=layout.position_ids[None].cuda(),
                        landmark_positions=layout.landmark_positions.cuda(), blocks=build_fastdllm_block_layouts(layout, 32, 64))
                raw = generated[0, layout.answer_positions.cuda()].tolist()
                prediction = exact.decode_answer(tok, raw)
                stats = asdict(generation_stats)
                del generated, decoder
            torch.cuda.synchronize()
            if row["kind"] == "longbench":
                score, metric = longbench_score(row["task"], prediction, row["answers"], all_classes=row.get("all_classes"))
                scores = dict(score=score, metric=metric)
            else:
                scores = score_codes(prediction, row["answers"])
            fallback = _kernel_fallback_count(model)
            if fallback or len(raw) != row["answer_tokens"]:
                raise AssertionError("invalid forward or raw answer coverage")
            result = dict(provenance, case_id=row["case_id"], kind=row["kind"], task=row["task"], mode=mode,
                physical_length=row["physical_length"], answers=row["answers"], prediction=prediction,
                prompt_sha256=hashlib.sha256(json.dumps(row["prompt_ids"]).encode()).hexdigest(),
                raw_answer_ids=raw, raw_decoded=tok.decode(raw, skip_special_tokens=False),
                eos_offset=raw.index(tok.eos_token_id) if tok.eos_token_id in raw else None,
                trace=trace, fallback_count=fallback, seconds=time.perf_counter() - started, **scores, **stats)
            append_jsonl_fsync(args.output, result)
            print(json.dumps({k:result[k] for k in ("case_id", "model", "mode", "prediction", "seconds")}), flush=True)


def gate(args):
    from dream_dllm_hils.bridge_gate import assess_controls
    root = Path(args.output)
    rows = read_rows(args.manifest)
    bridge = read_rows(root / "bridge-dsa.jsonl") + read_rows(root / "bridge-hils.jsonl")
    controls = read_rows(root / "controls-dsa.jsonl") + read_rows(root / "controls-hils.jsonl")
    result = assess_controls(rows, bridge, controls, manifest_sha256=digest(args.manifest))
    write_new(root / "gate.json", json.dumps(result, indent=2) + "\n")
    eligible = [r for r in rows if r["kind"] == "paired" and r["physical_length"] in result["eligible_lengths"]]
    write_new(root / "oracles" / "cases.jsonl", "".join(json.dumps(r) + "\n" for r in eligible))
    # Probes do not need a successful generation positive control.
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "bridge", "controls", "gate"), required=True)
    for name in ("origin", "model_path", "training_config", "checkpoint", "original_manifest", "manifest", "prompts", "data_root", "output"):
        p.add_argument("--" + name)
    args = p.parse_args()
    {"prepare": prepare, "bridge": run, "controls": run, "gate": gate}[args.stage](args)
