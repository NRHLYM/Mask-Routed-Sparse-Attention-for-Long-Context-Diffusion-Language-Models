#!/usr/bin/env python3
"""Create a deterministic held-out candidate pool for DSA-conditioned diagnostics."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dream_dllm_hils.diagnostic_helpers import RetrievalCase
from dream_dllm_hils.packed_corpus import DreamPackedCorpus
from scripts.dream_dllm_hils.bridge_generation import PairedSynthesizer, find_span


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--source_manifest", required=True,
                        help="Existing bridge manifest that fixes the known 32k-capable held-out packs")
    parser.add_argument("--candidate_count", type=int, default=64)
    parser.add_argument("--lengths", nargs="+", type=int, default=(2048, 8192, 16384, 32768))
    args = parser.parse_args()
    if args.candidate_count <= 0 or any(length <= 0 or length % 64 for length in args.lengths):
        raise ValueError("candidate count must be positive and lengths must be positive multiples of 64")

    from transformers import AutoTokenizer
    config = json.loads(Path(args.training_config).read_text())
    corpus, meta = Path(args.origin) / config["corpus_bin"], Path(args.origin) / config["corpus_meta"]
    dataset = DreamPackedCorpus(corpus, meta)
    source_rows = [json.loads(line) for line in Path(args.source_manifest).read_text().splitlines() if line.strip()]
    validation_ids = []
    for row in source_rows:
        if row.get("kind") == "paired" and row.get("physical_length") == 32768:
            pack_id = int(row["validation_pack_id"])
            if pack_id not in validation_ids:
                validation_ids.append(pack_id)
    if len(validation_ids) < 1:
        raise ValueError("source manifest has no known 32k-capable paired pack")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    eligible = []
    for pack_id in validation_ids:
        pack = dataset[pack_id]
        filler = pack["clean_ids"][pack["valid_tokens"]]
        eligible.append((pack_id, filler))
    if not eligible:
        raise ValueError("no held-out pack has required context capacity")
    # Long-context packs are scarce in this fixed validation split. Reusing a
    # pack with a new seed creates a new task-0 code/depth instance while
    # retaining the same held-out source distribution across every length.
    selected = [eligible[index % len(eligible)] for index in range(args.candidate_count)]

    rows = []
    for index, (pack_id, filler) in enumerate(selected):
        seed, depth = 9402609 + index, (0.1, 0.5, 0.9)[index % 3]
        expected_answer = None
        for length in args.lengths:
            synth = PairedSynthesizer(tokenizer, seed, depth)
            clean, target = synth.synthesize(filler[:length // 64 * 63 - 32], task_id=0)
            start = int(np.flatnonzero(target.numpy())[0])
            gold = tokenizer.decode(clean[target].tolist(), skip_special_tokens=True).strip()
            if len(gold) != 8 or not gold.isdigit():
                raise ValueError("task-0 template did not produce an eight-digit code")
            prompt = clean[:start].tolist()
            fact_tokens = [prompt[position] for position in synth.fact]
            relative = find_span(fact_tokens, tokenizer(gold, add_special_tokens=False).input_ids)
            value = [synth.fact[0] + position for position in relative]
            case = RetrievalCase(prompt, [value], [synth.fact], synth.question_start, [gold], "single", seed, depth)
            if expected_answer is None:
                expected_answer = gold
            elif gold != expected_answer:
                raise AssertionError("paired lengths changed the target code")
            row = asdict(case)
            row.update(
                case_id=f"conditioned-L{length}-candidate{index:03}",
                candidate_id=index,
                physical_length=length,
                validation_pack_id=pack_id,
                kind="dsa_conditioned_candidate",
                answer_tokens=32,
            )
            rows.append(row)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    if manifest.exists() and manifest.read_text() != payload:
        raise ValueError("refusing to replace an existing candidate manifest")
    manifest.write_text(payload)
    provenance = {
        "candidate_count_per_length": args.candidate_count,
        "lengths": args.lengths,
        "manifest_sha256": digest(manifest),
        "corpus_sha256": digest(corpus),
        "corpus_meta_sha256": digest(meta),
        "source_manifest_sha256": digest(args.source_manifest),
        "eligible_validation_pack_ids": [pack_id for pack_id, _ in eligible],
        "source_pack_valid_text_tokens": {str(pack_id): int(len(filler)) for pack_id, filler in eligible},
        "candidate_pack_assignment": [pack_id for pack_id, _ in selected],
        "selection": "round-robin over 32k-capable pack IDs fixed by the prior bridge manifest; distinct fixed code seeds; no DSA or HiLS output used",
        "task": "literal training task-0 template; fixed code seed and relative depth per candidate across lengths",
        "held_out_scope": "32k adaptation split only; warm-start overlap is not established",
        "synthetic": True,
        "benchmark": "DSA-conditioned routing diagnostic, not official RULER or a main-table benchmark",
    }
    Path(args.provenance).write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "manifest_sha256": provenance["manifest_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
